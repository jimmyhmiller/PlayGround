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
    /// Scheduled jobs: commands gatekeeper runs on a fixed interval. One place
    /// to put the periodic things you want to manage alongside the gate.
    #[serde(default)]
    pub job: Vec<Job>,
}

/// One scheduled job: a command run on a fixed interval by the gate's scheduler
/// thread. Output is logged (with exit status + duration). Jobs are independent
/// of routing — a gate with only jobs and no routes is valid.
#[derive(Debug, Clone, Deserialize)]
pub struct Job {
    /// A short name for logs, e.g. `axiom-reload`.
    pub name: String,
    /// The command + args to run. First element is the executable. Run directly
    /// (no shell), so no shell quoting/globbing — pass explicit args.
    pub command: Vec<String>,
    /// How often to run it, e.g. `"15m"`, `"1h"`, `"30s"`, `"90"` (bare = secs).
    /// Parsed by [`parse_duration`]. The interval is measured between the *start*
    /// of consecutive runs is avoided: we sleep `every` AFTER each run finishes,
    /// so a long run never overlaps itself.
    pub every: String,
    /// Run the job once immediately at startup (and on reload), in addition to
    /// every interval. Defaults to false (first run is after one interval).
    #[serde(default)]
    pub run_at_start: bool,
    /// Environment variables to set for the job (e.g. an API key). Merged onto
    /// the gate's own environment. Optional.
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,
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
    /// Mutually exclusive with `static` and `function`.
    #[serde(default)]
    pub proxy: Option<String>,
    /// Invoke this Rust function dylib (`.so`/`.dylib`/`.dll`, built against
    /// `gatekeeper-fn`) in process. Serverless: loaded on first request, cached.
    /// Mutually exclusive with `static` and `proxy`.
    #[serde(default)]
    pub function: Option<PathBuf>,
    /// Public routes skip auth. **Defaults to false** — fail safe.
    #[serde(default)]
    pub public: bool,
}

/// The resolved target of a route after validation.
#[derive(Debug, Clone)]
pub enum Target {
    Static(PathBuf),
    Proxy(String),
    Function(PathBuf),
}

impl Route {
    /// The validated target. Exactly one of `static`/`proxy`/`function` must be
    /// set; this is enforced by [`Config::validate`], so here we assume it holds.
    pub fn target(&self) -> Target {
        match (&self.static_dir, &self.proxy, &self.function) {
            (Some(dir), None, None) => Target::Static(dir.clone()),
            (None, Some(up), None) => Target::Proxy(up.clone()),
            (None, None, Some(lib)) => Target::Function(lib.clone()),
            // validate() rejects every other combination before we get here.
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
        if self.route.is_empty() && self.job.is_empty() {
            return Err(ConfigError("no routes or jobs defined".into()));
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
            // Exactly one target kind per route. Count how many are set so we
            // can reject both "none" and "more than one" with a clear message.
            let set = [
                r.static_dir.is_some(),
                r.proxy.is_some(),
                r.function.is_some(),
            ]
            .iter()
            .filter(|b| **b)
            .count();
            match set {
                0 => {
                    return Err(ConfigError(format!(
                        "route {:?} sets none of 'static', 'proxy', 'function' — pick one",
                        r.path
                    )));
                }
                1 => {}
                _ => {
                    return Err(ConfigError(format!(
                        "route {:?} sets more than one of 'static'/'proxy'/'function' — pick one",
                        r.path
                    )));
                }
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

        // Validate jobs: non-empty name + command, a parseable interval, and no
        // duplicate names (so logs are unambiguous).
        let mut job_names: Vec<&str> = Vec::with_capacity(self.job.len());
        for j in &self.job {
            if j.name.trim().is_empty() {
                return Err(ConfigError("job has an empty name".into()));
            }
            if j.command.is_empty() {
                return Err(ConfigError(format!("job {:?} has an empty command", j.name)));
            }
            parse_duration(&j.every).map_err(|e| {
                ConfigError(format!("job {:?} has invalid 'every' {:?}: {e}", j.name, j.every))
            })?;
            job_names.push(j.name.as_str());
        }
        job_names.sort_unstable();
        for w in job_names.windows(2) {
            if w[0] == w[1] {
                return Err(ConfigError(format!("duplicate job name {:?}", w[0])));
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

/// Parse a human interval like `"15m"`, `"1h"`, `"30s"`, `"2d"`, or a bare
/// number (interpreted as seconds). Returns the duration. Rejects zero and
/// unparseable input so a misconfigured job can't spin in a tight loop.
pub fn parse_duration(s: &str) -> Result<std::time::Duration, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty duration".into());
    }
    let (num_str, mult) = match s.chars().last().unwrap() {
        's' => (&s[..s.len() - 1], 1u64),
        'm' => (&s[..s.len() - 1], 60),
        'h' => (&s[..s.len() - 1], 3600),
        'd' => (&s[..s.len() - 1], 86_400),
        c if c.is_ascii_digit() => (s, 1), // bare number = seconds
        c => return Err(format!("unknown duration unit '{c}' (use s/m/h/d)")),
    };
    let n: u64 = num_str
        .trim()
        .parse()
        .map_err(|_| format!("not a number: {num_str:?}"))?;
    let secs = n.checked_mul(mult).ok_or("duration overflow")?;
    if secs == 0 {
        return Err("duration must be greater than zero".into());
    }
    Ok(std::time::Duration::from_secs(secs))
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
