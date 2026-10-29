//! Remote storage in the browser: files on a same-origin WebDAV server appear under mounted
//! virtual paths (`./effectcraft-config.json`, see [`crate::remote_config`]) and are read with
//! `fetch` (PROPFIND listings, whole-file GETs). The pure parts live in [`crate::remote_config`]
//! and [`crate::remote_dav`]; this module is the browser side: config load, listing and stat
//! caches, fetching files into the file table, and opening a project together with its media.
//!
//! A remote file is fetched whole into the file table (not persisted) when a project or an import
//! needs it, up to [`FILE_MAX`] bytes: there is no ranged reader yet. The Media Browser lists
//! synchronously, so a listing starts on the first look and answers [`RemoteInfo::LOADING`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use effectcraft_engine::media_browser::{Entry, kind_of};
use effectcraft_ui_egui::{EffectcraftApp, RemoteInfo};
use serde_json::json;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::remote_config::{self, Config};
use crate::remote_dav::{self, Child};

/// The largest remote file fetched (bytes): it is held in memory, twice while it is copied in.
pub const FILE_MAX: u64 = 1 << 30;
/// A cached listing is shown for this long before it is refreshed (seconds).
const LIST_TTL_S: f64 = 15.0;
/// A failed listing is not retried for this long (seconds).
const ERROR_RETRY_S: f64 = 5.0;
/// Entries kept in the stat cache.
const STAT_MAX: usize = 200_000;
/// Media paths of one project fetched when it opens.
const MEDIA_MAX: usize = 10_000;
/// Largest config file read.
pub(crate) const CONFIG_MAX: usize = 64 * 1024;

/// What the server said about one path.
#[derive(Clone, Debug)]
pub struct Stat {
    pub size: u64,
    pub is_dir: bool,
    pub etag: Option<String>,
}

#[derive(Default)]
struct DirState {
    entries: Option<Vec<Entry>>,
    fetched: f64,
    loading: bool,
    error: Option<(String, f64)>,
}

#[derive(Default)]
struct State {
    cfg: Option<Config>,
    origin: String,
    stats: HashMap<String, Stat>,
    dirs: HashMap<String, DirState>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
    static T0: web_time::Instant = web_time::Instant::now();
}

fn now_s() -> f64 {
    T0.with(|t| t.elapsed().as_secs_f64())
}

/// Whether remote storage is configured.
pub fn enabled() -> bool {
    STATE.with(|s| s.borrow().cfg.is_some())
}

/// Whether `path` is on a remote mount.
pub fn is_remote(path: &str) -> bool {
    STATE.with(|s| s.borrow().cfg.as_ref().is_some_and(|c| c.mount_for(path).is_some()))
}

/// The mounts as (name, path).
pub fn mounts() -> Vec<(String, String)> {
    STATE.with(|s| s.borrow().cfg.iter().flat_map(|c| c.mounts.iter()).map(|m| (m.name.clone(), m.path.clone())).collect())
}

/// The cached stat of a path (from a listing or a HEAD).
pub fn stat(path: &str) -> Option<Stat> {
    STATE.with(|s| s.borrow().stats.get(path.strip_suffix('/').unwrap_or(path)).cloned())
}

pub(crate) fn js_err(e: wasm_bindgen::JsValue) -> String {
    format!("{e:?}")
}

/// A `fetch` that stays on this origin and always asks the server (the service worker leaves
/// `no-cache` requests alone).
pub(crate) async fn fetch(method: &str, url: &str, headers: &[(&str, &str)], body: Option<&[u8]>) -> Result<web_sys::Response, String> {
    let init = web_sys::RequestInit::new();
    init.set_method(method);
    // Same origin only: a redirect elsewhere fails instead of leaking the request.
    init.set_mode(web_sys::RequestMode::SameOrigin);
    init.set_cache(web_sys::RequestCache::NoCache);
    let h = web_sys::Headers::new().map_err(js_err)?;
    for (k, v) in headers {
        h.set(k, v).map_err(js_err)?;
    }
    init.set_headers(&h);
    if let Some(b) = body {
        init.set_body(&js_sys::Uint8Array::from(b));
    }
    let req = web_sys::Request::new_with_str_and_init(url, &init).map_err(js_err)?;
    let win = web_sys::window().ok_or("no window")?;
    let resp = JsFuture::from(win.fetch_with_request(&req)).await.map_err(js_err)?;
    resp.dyn_into::<web_sys::Response>().map_err(|_| "fetch did not return a Response".to_string())
}

pub(crate) fn header(r: &web_sys::Response, name: &str) -> Option<String> {
    r.headers().get(name).ok().flatten()
}

/// The body, refused before reading when `Content-Length` exceeds `cap` and after when it does.
pub(crate) async fn read_body(r: &web_sys::Response, cap: usize) -> Result<Vec<u8>, String> {
    if header(r, "content-length").and_then(|h| remote_dav::parse_content_length(&h)).is_some_and(|n| n > cap as u64) {
        return Err("response too large".into());
    }
    let buf = JsFuture::from(r.array_buffer().map_err(js_err)?).await.map_err(js_err)?;
    let a = js_sys::Uint8Array::new(&buf);
    if a.length() as usize > cap {
        return Err("response too large".into());
    }
    Ok(a.to_vec())
}

pub(crate) fn cfg_and_origin() -> Result<(Config, String), String> {
    STATE.with(|s| {
        let s = s.borrow();
        s.cfg.clone().map(|c| (c, s.origin.clone())).ok_or_else(|| "remote storage is not configured".to_string())
    })
}

/// Read `./effectcraft-config.json`; remote storage stays off when it is missing or invalid.
pub async fn load_config() {
    let Some(loc) = web_sys::window().map(|w| w.location()) else { return };
    let (origin, path) = (loc.origin().unwrap_or_default(), loc.pathname().unwrap_or_default());
    let base_dir = path.get(..=path.rfind('/').unwrap_or(0)).unwrap_or("/").to_string();
    let bytes = match fetch("GET", "effectcraft-config.json", &[], None).await {
        Ok(r) if r.status() == 200 => read_body(&r, CONFIG_MAX).await,
        Ok(r) => Err(format!("status {}", r.status())),
        Err(e) => Err(e),
    };
    match bytes.and_then(|b| remote_config::parse(&b, &origin, &base_dir)) {
        Ok(cfg) => {
            log::info!("remote storage: {} mount(s)", cfg.mounts.len());
            STATE.with(|s| {
                let mut s = s.borrow_mut();
                s.cfg = Some(cfg);
                s.origin = origin;
            });
        }
        Err(e) => log::info!("remote storage off: {e}"),
    }
}

/// PROPFIND `dir` (Depth 1) and map the answer to virtual paths.
async fn propfind(dir: &str) -> Result<Vec<Child>, String> {
    let (cfg, origin) = cfg_and_origin()?;
    let mount = cfg.mount_for(dir).ok_or_else(|| format!("{dir}: not on a remote mount"))?.clone();
    let url = cfg.url_for(&format!("{}/", dir.trim_end_matches('/')))?;
    let r = fetch("PROPFIND", &url, &[("Depth", "1"), ("Content-Type", "application/xml")], Some(remote_dav::PROPFIND_BODY.as_bytes())).await?;
    if r.status() != 207 {
        return Err(format!("{dir}: the server answered {} to a listing", r.status()));
    }
    let entries = remote_dav::parse_multistatus(&read_body(&r, remote_dav::MAX_XML_BYTES).await?)?;
    Ok(remote_dav::children_of(&cfg, &mount, dir, &origin, entries))
}

fn entry_of(c: &Child) -> Entry {
    let is_dir = c.entry.is_dir;
    Entry {
        name: c.name.clone(),
        path: c.path.clone(),
        is_dir,
        size: c.entry.size.unwrap_or(0),
        modified: c.entry.modified,
        kind: if is_dir { None } else { kind_of(&c.path) },
    }
}

fn record_dir(dir: &str, res: Result<Vec<Child>, String>) {
    let now = now_s();
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let st = s.dirs.entry(dir.to_string()).or_default();
        st.loading = false;
        match res {
            Ok(children) => {
                st.error = None;
                st.fetched = now;
                st.entries = Some(children.iter().map(entry_of).collect());
                if s.stats.len() > STAT_MAX {
                    s.stats.clear();
                }
                for c in children {
                    let stat = Stat { size: c.entry.size.unwrap_or(0), is_dir: c.entry.is_dir, etag: c.entry.etag };
                    s.stats.insert(c.path, stat);
                }
            }
            Err(e) => st.error = Some((e, now)),
        }
    });
}

fn start_listing(dir: String) {
    wasm_bindgen_futures::spawn_local(async move {
        let res = propfind(&dir).await;
        record_dir(&dir, res);
        // The Media Browser keeps a listing for a second: look again after that too.
        crate::repaint();
        crate::repaint_after(Duration::from_millis(1100));
    });
}

/// The Media Browser's listing of a directory under a mount; `None` for other paths. A listing
/// that is not here yet starts and answers [`RemoteInfo::LOADING`]; an expired one is shown while
/// it refreshes.
pub fn list_entries(dir: &str, importable_only: bool) -> Option<Result<Vec<Entry>, String>> {
    let dir = dir.trim_end_matches('/').to_string();
    if !is_remote(&dir) {
        return None;
    }
    let now = now_s();
    let (result, start) = STATE.with(|s| {
        let mut s = s.borrow_mut();
        let st = s.dirs.entry(dir.clone()).or_default();
        let fresh = st.entries.is_some() && now - st.fetched < LIST_TTL_S;
        let blocked = st.loading || st.error.as_ref().is_some_and(|(_, at)| now - at < ERROR_RETRY_S);
        let start = !fresh && !blocked;
        if start {
            st.loading = true;
        }
        let result = match (&st.entries, &st.error) {
            (Some(e), _) => Ok(e.clone()),
            (None, Some((msg, at))) if now - at < ERROR_RETRY_S => Err(msg.clone()),
            _ => Err(RemoteInfo::LOADING.to_string()),
        };
        (result, start)
    });
    if start {
        start_listing(dir);
    }
    Some(result.map(|mut v| {
        // Folders first, then files, by name; "media only" hides files of no known kind.
        v.retain(|e| e.is_dir || !importable_only || e.kind.is_some());
        v.sort_by_cached_key(|e| (!e.is_dir, e.name.to_lowercase()));
        v
    }))
}

/// A file's size: cached, else from its parent's listing, else `HEAD`.
pub async fn ensure_stat(path: &str) -> Result<Stat, String> {
    if let Some(s) = stat(path) {
        return Ok(s);
    }
    if let Some(parent) = remote_config::parent(path) {
        let res = propfind(parent).await;
        if let Err(e) = &res {
            log::info!("listing {parent}: {e}");
        }
        record_dir(parent, res);
        if let Some(s) = stat(path) {
            return Ok(s);
        }
    }
    let (cfg, _) = cfg_and_origin()?;
    let r = fetch("HEAD", &cfg.url_for(path)?, &[], None).await?;
    if r.status() != 200 {
        return Err(format!("{path}: the server answered {}", r.status()));
    }
    let size = header(&r, "content-length").and_then(|h| remote_dav::parse_content_length(&h)).ok_or_else(|| format!("{path}: the server gave no size"))?;
    let s = Stat { size, is_dir: false, etag: header(&r, "etag") };
    STATE.with(|st| st.borrow_mut().stats.insert(path.to_string(), s.clone()));
    Ok(s)
}

/// Fetch the file at remote `path` whole into the file table (not persisted), unless the table
/// already holds a file of that size. Refuses files over [`FILE_MAX`].
pub async fn fetch_file(path: &str) -> Result<(), String> {
    let st = ensure_stat(path).await?;
    if st.is_dir {
        return Err(format!("{path}: is a folder"));
    }
    if st.size > FILE_MAX {
        return Err(format!("{path}: {} MiB is over the {} MiB limit for files from remote storage", st.size >> 20, FILE_MAX >> 20));
    }
    if crate::files::get(path).is_some_and(|b| b.len() as u64 == st.size) {
        return Ok(());
    }
    let (cfg, _) = cfg_and_origin()?;
    let r = fetch("GET", &cfg.url_for(path)?, &[], None).await?;
    if r.status() != 200 {
        return Err(format!("{path}: the server answered {}", r.status()));
    }
    let bytes = read_body(&r, FILE_MAX as usize).await?;
    crate::files::put_remote(path, Arc::from(bytes));
    Ok(())
}

/// The media (and proxy) file paths a project file refers to, relative ones made absolute next to
/// the project.
fn media_paths(project: &str, bytes: &[u8]) -> Vec<String> {
    let Ok(text) = std::str::from_utf8(bytes) else { return Vec::new() };
    let Ok((proj, _)) = effectcraft_engine::project::Project::from_json_repaired(text) else { return Vec::new() };
    let dir = remote_config::parent(project).unwrap_or("/").trim_end_matches('/').to_string();
    let mut out = Vec::new();
    for it in proj.items.values() {
        let effectcraft_engine::project::ItemKind::Footage(f) = &it.kind else { continue };
        let proxy = it.proxy.as_ref().map(|p| &p.footage);
        for f in std::iter::once(f).chain(proxy) {
            for p in std::iter::once(&f.path).chain(&f.sequence).filter(|p| !p.is_empty()) {
                out.push(if p.starts_with('/') { p.clone() } else { format!("{dir}/{p}") });
            }
        }
    }
    out
}

/// Drop what is cached about `path` and its folder (after it was written): the next look asks again.
pub fn forget(path: &str) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.stats.remove(path);
        if let Some(p) = remote_config::parent(path) {
            s.dirs.remove(p);
        }
    });
}

/// Fetch every remote file of `paths`; what failed is named in the status line.
pub async fn fetch_all(paths: &[String]) -> Vec<String> {
    let mut failed = Vec::new();
    for p in paths.iter().filter(|p| is_remote(p)).take(MEDIA_MAX) {
        if let Err(e) = fetch_file(p).await {
            log::info!("remote file {p}: {e}");
            failed.push(e);
        }
    }
    failed
}

/// Open the project at a remote path: fetch it and its media, then `file.open`. Media that cannot
/// be fetched stay missing (the engine reports them offline) and are named in the status line.
pub async fn open_project(path: &str) -> Result<(), String> {
    let st = ensure_stat(path).await?;
    if st.is_dir || st.size > remote_dav::MAX_PROJECT_BYTES {
        return Err(format!("{path}: not a project file this size can open"));
    }
    let (cfg, _) = cfg_and_origin()?;
    let r = fetch("GET", &cfg.url_for(path)?, &[], None).await?;
    if r.status() != 200 {
        return Err(format!("{path}: the server answered {}", r.status()));
    }
    crate::remote_upload::note_opened(path, header(&r, "etag"));
    let bytes = read_body(&r, remote_dav::MAX_PROJECT_BYTES as usize).await?;
    let failed = fetch_all(&media_paths(path, &bytes)).await;
    crate::files::put_remote(path, Arc::from(bytes));
    let path = path.to_string();
    crate::post(move |app| match app.session.execute("file.open", json!({"path": path})) {
        Ok(_) if !failed.is_empty() => {
            app.ui.status = format!("{} media file(s) could not be loaded: {}", failed.len(), failed.iter().take(3).cloned().collect::<Vec<_>>().join("; "))
        }
        Ok(_) => {}
        Err(e) => app.ui.status = e.to_string(),
    });
    Ok(())
}

/// Route "open project by path" through the remote fetch and offer the remote menu entries.
pub fn install_hooks(app: &mut EffectcraftApp) {
    if !enabled() {
        return;
    }
    app.hooks.remote = Some(RemoteInfo { mounts: mounts() });
    app.hooks.open_project = Some(Box::new(|path: &str| {
        let path = path.to_string();
        if !is_remote(&path) {
            crate::post(move |app| {
                if let Err(e) = app.session.execute("file.open", json!({"path": path})) {
                    app.ui.status = e.to_string();
                }
            });
            return;
        }
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = open_project(&path).await {
                crate::post(move |app| app.ui.status = e);
            }
        });
    }));
}
