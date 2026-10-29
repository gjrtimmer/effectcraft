//! `./effectcraft-config.json`, fetched once per page load and shared by whoever reads a key.

use std::cell::RefCell;

use crate::startup::{self, Startup};
use crate::{remote_http, remote_util};

thread_local! {
    /// `None`: not fetched yet; `Some(None)`: no usable file.
    static FILE: RefCell<Option<Option<Vec<u8>>>> = const { RefCell::new(None) };
}

async fn fetch() -> Option<Vec<u8>> {
    // Always asked from the server (the file is the operator's switch, never a cached copy), and
    // time-limited: a server that does not answer counts as no file, the app still starts.
    let resp = remote_http::fetch_within("GET", "effectcraft-config.json", &[], None, remote_util::CONFIG_TIMEOUT)
        .await
        .inspect_err(|e| log::info!("config file: {e}"))
        .ok()?;
    if resp.status() != 200 {
        return None;
    }
    remote_http::read_body(&resp, startup::CONFIG_MAX).await.ok()
}

/// The config file's bytes (`None` when there is none), fetched on the first call.
pub async fn bytes() -> Option<Vec<u8>> {
    if let Some(done) = FILE.with(|f| f.borrow().clone()) {
        return done;
    }
    let got = fetch().await;
    FILE.with(|f| *f.borrow_mut() = Some(got.clone()));
    got
}

/// The configured startup behaviour.
pub async fn startup() -> Startup {
    let s = bytes().await.map(|b| startup::parse(&b)).unwrap_or_default();
    log::info!("startup: {s:?}");
    s
}
