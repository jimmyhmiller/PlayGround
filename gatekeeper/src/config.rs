//! Configuration loading and validation.
//!
//! The safety posture lives here as much as in routing: a route is **private
//! unless it explicitly sets `public = true`**. `public` defaults to `false`
//! (via serde default), so forgetting the field fails safe — the route is
//! private, not exposed.
//!
//! The shared auth token is deliberately **not** part of this file. It comes
//! from the `GATEKEEPER_TOKEN` environment variable or a `--token-file`, so a
//! plaintext secret never lives in the committed config. Boot fails closed if a
//! private route exists and no token is configured.

use std::path::PathBuf;

use serde::Deserialize;

/// Top-level config, parsed from a TOML file.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Address to bind, e.g. `0.0.0.0:443` (HTTPS) or `127.0.0.1:8080` (plain).
    #[serde(default = "default_bind")]
    pub bind: String,
    /// PEM certificate chain file. Set both `tls_cert` and `tls_key` to serve
    /// HTTPS; omit both to serve plain HTTP (use only behind a TLS terminator
    /// or on localhost). Certs are provisioned out-of-band (e.g. certbot).
    #[serde(default)]
    pub tls_cert: Option<PathBuf>,
    /// PEM private key file. See `tls_cert`.
    #[serde(default)]
    pub tls_key: Option<PathBuf>,
    /// What to return when a request matches no route. `404` (default) hides
    /// the existence of services; `403` says "forbidden". Either way: deny.
    #[serde(default = "default_unmatched_status")]
    pub unmatched_status: u16,
    /// The routes. Order does not matter; matching is longest-prefix.
    #[serde(default)]
    pub route: Vec<Route>,
}

/// One route: a path prefix that maps to either a static folder or a proxied
/// upstream. Private unless `public = true`.
#[derive(Debug, Clone, Deserialize)]
pub struct Route {
    /// Path prefix, e.g. `/blog`. Matched on `/` boundaries (see `route.rs`).
    pub path: String,
    /// Serve this filesystem directory. Mutually exclusive with `proxy`.
    #[serde(default, rename = "static")]
    pub static_dir: Option<PathBuf>,
    /// Reverse-proxy to this `host:port` (typically `127.0.0.1:PORT`).
    /// Mutually exclusive with `static`.
    #[serde(default)]
    pub proxy: Option<String>,
    /// Public routes skip auth. **Defaults to false** — fail safe.
    #[serde(default)]
    pub public: bool,
}

/// The resolved target of a route after validation.
#[derive(Debug, Clone)]
pub enum Target {
    Static(PathBuf),
    Proxy(String),
}

impl Route {
    /// The validated target. Exactly one of `static`/`proxy` must be set; this
    /// is enforced by [`Config::validate`], so here we can assume it holds.
    pub fn target(&self) -> Target {
        match (&self.static_dir, &self.proxy) {
            (Some(dir), None) => Target::Static(dir.clone()),
            (None, Some(up)) => Target::Proxy(up.clone()),
            // validate() rejects the other two cases before we ever get here.
            _ => unreachable!("route target validated at load time"),
        }
    }
}

fn default_bind() -> String {
    "0.0.0.0:443".to_string()
}
fn default_unmatched_status() -> u16 {
    404
}

#[derive(Debug)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "config error: {}", self.0)
    }
}
impl std::error::Error for ConfigError {}

impl Config {
    /// Parse a TOML string into a `Config` and validate it. Validation is where
    /// we reject ambiguous routes before they can ever serve a request.
    pub fn from_toml(s: &str) -> Result<Config, ConfigError> {
        let cfg: Config = toml::from_str(s).map_err(|e| ConfigError(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load and validate a config from a file path.
    pub fn load(path: &std::path::Path) -> Result<Config, ConfigError> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| ConfigError(format!("reading {}: {}", path.display(), e)))?;
        Config::from_toml(&s)
    }

    /// Boot-time validation. Every check here exists to make accidental
    /// exposure or an outright broken config impossible to start.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.route.is_empty() {
            return Err(ConfigError("no routes defined".into()));
        }
        if self.unmatched_status != 403 && self.unmatched_status != 404 {
            return Err(ConfigError(format!(
                "unmatched_status must be 403 or 404, got {}",
                self.unmatched_status
            )));
        }
        match (&self.tls_cert, &self.tls_key) {
            (Some(_), None) | (None, Some(_)) => {
                return Err(ConfigError(
                    "tls_cert and tls_key must be set together (or both omitted)".into(),
                ));
            }
            _ => {}
        }
        for r in &self.route {
            if !r.path.starts_with('/') {
                return Err(ConfigError(format!(
                    "route path must start with '/': {:?}",
                    r.path
                )));
            }
            // A trailing slash on a prefix would break our `/`-boundary match
            // (every request would need the slash). Reject it at the door so
            // the operator writes `/blog`, not `/blog/`.
            if r.path.len() > 1 && r.path.ends_with('/') {
                return Err(ConfigError(format!(
                    "route path must not end with '/': {:?} (use {:?})",
                    r.path,
                    r.path.trim_end_matches('/')
                )));
            }
            match (&r.static_dir, &r.proxy) {
                (Some(_), Some(_)) => {
                    return Err(ConfigError(format!(
                        "route {:?} sets both 'static' and 'proxy' — pick one",
                        r.path
                    )));
                }
                (None, None) => {
                    return Err(ConfigError(format!(
                        "route {:?} sets neither 'static' nor 'proxy'",
                        r.path
                    )));
                }
                _ => {}
            }
        }
        // Duplicate exact paths are almost certainly a mistake (which one wins
        // is non-obvious), so reject them.
        let mut paths: Vec<&str> = self.route.iter().map(|r| r.path.as_str()).collect();
        paths.sort_unstable();
        for w in paths.windows(2) {
            if w[0] == w[1] {
                return Err(ConfigError(format!("duplicate route path {:?}", w[0])));
            }
        }
        Ok(())
    }

    /// True if any route requires auth — i.e. a token must be configured.
    pub fn has_private_route(&self) -> bool {
        self.route.iter().any(|r| !r.public)
    }

    /// True if TLS is configured (both cert and key present).
    pub fn tls_enabled(&self) -> bool {
        self.tls_cert.is_some() && self.tls_key.is_some()
    }
}

/// Source the shared auth token. Precedence: explicit `--token-file`, then the
/// `GATEKEEPER_TOKEN` env var. Returns `None` if neither is set (the caller
/// decides whether that is fatal — it is, if any private route exists).
///
/// The token is read here and never logged. A whitespace-only token is treated
/// as absent so an empty file or `GATEKEEPER_TOKEN=` doesn't silently disable
/// auth.
pub fn load_token(token_file: Option<&std::path::Path>) -> Result<Option<String>, ConfigError> {
    if let Some(path) = token_file {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| ConfigError(format!("reading token file {}: {}", path.display(), e)))?;
        let tok = raw.trim().to_string();
        return Ok(if tok.is_empty() { None } else { Some(tok) });
    }
    match std::env::var("GATEKEEPER_TOKEN") {
        Ok(v) => {
            let tok = v.trim().to_string();
            Ok(if tok.is_empty() { None } else { Some(tok) })
        }
        Err(_) => Ok(None),
    }
}
