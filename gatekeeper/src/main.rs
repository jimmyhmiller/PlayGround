//! gatekeeper — a fail-closed HTTP front gate.
//!
//! Sits in front of your services and checks a shared-secret token on every
//! request. A route is **private unless it sets `public = true`** in the config,
//! so you can't accidentally expose something: forgetting the flag fails safe.
//!
//! Synchronous, thread-per-request (`tiny_http`). Optional TLS via rustls when
//! `tls_cert`/`tls_key` are configured; otherwise plain HTTP (for use behind a
//! TLS terminator or on localhost).

use std::path::PathBuf;
use std::sync::Arc;

use gatekeeper::auth::Authenticator;
use gatekeeper::config::{self, Config, Target};
use gatekeeper::function::FunctionRegistry;
use gatekeeper::proxy;
use gatekeeper::reply::Reply;
use gatekeeper::route::{Match, Router};
use gatekeeper::schedule::Scheduler;
use gatekeeper::serve;

/// Reserved built-in meta route: a JSON catalog of every route and each
/// function's self-description. Private (token required). Lives under a
/// `/_gatekeeper/` namespace so it can't collide with a real service route.
const DESCRIBE_PATH: &str = "/_gatekeeper/describe";

struct Args {
    config: PathBuf,
    token_file: Option<PathBuf>,
    check: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut config = PathBuf::from("gatekeeper.toml");
    let mut token_file = None;
    let mut check = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--config" | "-c" => {
                config = it.next().ok_or("--config needs a path")?.into();
            }
            "--token-file" => {
                token_file = Some(it.next().ok_or("--token-file needs a path")?.into());
            }
            "--check" => check = true,
            "-h" | "--help" => {
                println!(
                    "gatekeeper --config <file> [--token-file <file>] [--check]\n\
                     \n  --config       config TOML (default ./gatekeeper.toml)\
                     \n  --token-file   read shared token from a file (else $GATEKEEPER_TOKEN)\
                     \n  --check        validate config + print exposure report, then exit"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Args { config, token_file, check })
}

/// The hot-swappable part of a worker's view: routing table, auth, and the
/// unmatched status. Rebuilt from the config on SIGHUP and swapped in atomically
/// so route/token changes take effect without a restart. The function dylib
/// cache is deliberately NOT here — it lives in [`Gate`] and persists across
/// reloads so already-loaded dylibs are not re-`dlopen`ed.
struct Routing {
    router: Router,
    auth: Option<Authenticator>,
    unmatched_status: u16,
}

/// Everything a worker needs to handle a request, shared across threads. The
/// `routing` is swappable on reload; `functions` persists.
struct Gate {
    /// Current routing/auth, swapped wholesale on SIGHUP (config reload).
    routing: std::sync::Mutex<Arc<Routing>>,
    /// Lazily-loaded cache of function dylibs (the serverless backend). Shared
    /// across reloads: reloading the config does not drop loaded functions.
    functions: FunctionRegistry,
    /// Runs scheduled `[[job]]`s on their intervals. Persists across reloads;
    /// its `reload` is called with the new job set on each SIGHUP.
    scheduler: Scheduler,
}

impl Gate {
    /// Snapshot the current routing for a request. Cheap `Arc` clone so a
    /// concurrent reload swapping in a new `Routing` never tears a request.
    fn routing(&self) -> Arc<Routing> {
        Arc::clone(&self.routing.lock().unwrap())
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gatekeeper: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let cfg = Config::load(&args.config).map_err(|e| e.to_string())?;

    // Source the token. Fail closed: if any route is private, a token is
    // mandatory. (A config with only public routes can run without one.)
    let token = config::load_token(args.token_file.as_deref()).map_err(|e| e.to_string())?;
    if cfg.has_private_route() && token.is_none() {
        return Err(
            "config has private routes but no token configured \
             (set GATEKEEPER_TOKEN or --token-file)"
                .into(),
        );
    }

    print_exposure_report(&cfg, token.is_some());

    if args.check {
        println!("\n--check: config valid. Not binding.");
        return Ok(());
    }

    let gate = Arc::new(Gate {
        routing: std::sync::Mutex::new(Arc::new(build_routing(&cfg, token.as_deref()))),
        functions: FunctionRegistry::new(),
        scheduler: Scheduler::new(),
    });
    // Start the scheduled jobs (if any). Re-applied on every config reload.
    gate.scheduler.reload(&cfg.job);

    // Bind the listening socket ONCE. We keep our own handle so we can rebuild
    // the tiny_http Server (with a freshly-loaded cert) on SIGHUP without ever
    // closing the socket — that's what makes cert reload zero-downtime.
    let listener = std::net::TcpListener::bind(&cfg.bind)
        .map_err(|e| format!("binding {}: {e}", cfg.bind))?;

    let server = build_server(&listener, &cfg)?;
    // The current Server lives behind a Mutex so the SIGHUP handler can swap it.
    let current: Arc<std::sync::Mutex<Arc<tiny_http::Server>>> =
        Arc::new(std::sync::Mutex::new(Arc::new(server)));

    println!(
        "\ngatekeeper listening on {} ({})",
        cfg.bind,
        if cfg.tls_enabled() { "HTTPS" } else { "HTTP" }
    );
    println!(
        "  reload (cert + routes): send SIGHUP (kill -HUP {}) or `systemctl reload gatekeeper`",
        std::process::id()
    );

    // SIGHUP -> reload the config: rebuild the Server with the (possibly renewed)
    // cert AND rebuild the routing table + auth from the config file, swapping
    // both in. The old Server is unblocked so its workers release it; in-flight
    // requests finish. Loaded function dylibs persist across the reload.
    install_reload_handler(
        Arc::clone(&current),
        Arc::clone(&gate),
        listener,
        args.config.clone(),
        args.token_file.clone(),
    );

    // A small fixed pool of workers. Each re-reads the current Server every
    // iteration (cheap Arc clone) and uses recv_timeout so a swap is picked up
    // promptly even on an idle connection.
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(2, 16);
    let mut handles = Vec::new();
    for _ in 0..workers {
        let current = Arc::clone(&current);
        let gate = Arc::clone(&gate);
        handles.push(std::thread::spawn(move || loop {
            let server = { Arc::clone(&*current.lock().unwrap()) };
            match server.recv_timeout(std::time::Duration::from_millis(500)) {
                Ok(Some(req)) => handle(&gate, req),
                Ok(None) => {} // timeout or unblocked: loop, re-read current server
                Err(e) => {
                    eprintln!("gatekeeper: recv error: {e}");
                    break;
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

/// Build the hot-swappable [`Routing`] from a config and optional token. Used at
/// boot and on every reload, so route/auth construction is identical both times.
fn build_routing(cfg: &Config, token: Option<&str>) -> Routing {
    Routing {
        router: Router::new(cfg.route.clone()),
        auth: token.map(Authenticator::new),
        unmatched_status: cfg.unmatched_status,
    }
}

/// Spawn a thread that watches for SIGHUP and, on each, **reloads the config**:
/// it re-reads the config file and token, then rebuilds (a) the TLS Server from
/// the same socket with the current certificate and (b) the routing table + auth,
/// swapping both in atomically. This is what makes adding/changing routes (and
/// rotating the token, and renewing the cert) take effect with no restart.
///
/// Fail-safe at every step: if the config is invalid, the token is now missing
/// for a private route, or the cert can't be read, we log and keep serving the
/// *current* state rather than going down. A botched edit can't take the gate
/// offline or accidentally drop auth.
///
/// Loaded function dylibs are NOT touched here — they live in the `Gate` and stay
/// resident across reloads, so a reload never re-`dlopen`s a warm function.
fn install_reload_handler(
    current: Arc<std::sync::Mutex<Arc<tiny_http::Server>>>,
    gate: Arc<Gate>,
    listener: std::net::TcpListener,
    config_path: PathBuf,
    token_file: Option<PathBuf>,
) {
    use std::sync::atomic::{AtomicBool, Ordering};
    let flag = Arc::new(AtomicBool::new(false));
    // signal_hook flips the flag from the real signal handler; we poll it.
    if signal_hook::flag::register(signal_hook::consts::SIGHUP, Arc::clone(&flag)).is_err() {
        eprintln!("gatekeeper: warning: could not install SIGHUP handler; reload disabled");
        return;
    }
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if !flag.swap(false, Ordering::SeqCst) {
            continue;
        }
        reload_once(&current, &gate, &listener, &config_path, token_file.as_deref());
    });
}

/// Perform one reload cycle. Separated out so the logic is linear and each
/// failure mode logs + bails without partially applying a reload.
fn reload_once(
    current: &Arc<std::sync::Mutex<Arc<tiny_http::Server>>>,
    gate: &Arc<Gate>,
    listener: &std::net::TcpListener,
    config_path: &std::path::Path,
    token_file: Option<&std::path::Path>,
) {
    // 1. Re-read + validate the config. Invalid -> keep current, don't apply.
    let cfg = match Config::load(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("gatekeeper: reload: config invalid, keeping current config: {e}");
            return;
        }
    };

    // 2. Re-source the token and re-check the fail-closed invariant. If the new
    //    config has a private route but no token is available, refuse the reload
    //    rather than swap in routing that would 401 everything (or worse).
    let token = match config::load_token(token_file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("gatekeeper: reload: reading token failed, keeping current config: {e}");
            return;
        }
    };
    if cfg.has_private_route() && token.is_none() {
        eprintln!(
            "gatekeeper: reload: new config has private routes but no token configured; \
             keeping current config (set GATEKEEPER_TOKEN or --token-file)"
        );
        return;
    }

    // 3. Rebuild the TLS server (picks up a renewed cert). Bad cert -> keep
    //    current cert AND skip the routing swap, so a half-applied reload can't
    //    happen. We rebuild the server even for plain HTTP (cheap) so a cert
    //    *added* to the config takes effect.
    let new_server = match build_server(listener, &cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("gatekeeper: reload: building server failed, keeping current: {e}");
            return;
        }
    };

    // 4. Apply: swap routing first, then the server. Both are independent atomic
    //    swaps; workers snapshot each per-request, so the worst interleaving is a
    //    single request seeing new routing with the old server (or vice versa) —
    //    both valid states, never a torn one.
    let new_routing = Arc::new(build_routing(&cfg, token.as_deref()));
    {
        let mut guard = gate.routing.lock().unwrap();
        *guard = new_routing;
    }
    {
        let mut guard = current.lock().unwrap();
        let old = std::mem::replace(&mut *guard, Arc::new(new_server));
        drop(guard);
        old.unblock(); // release workers blocked on the old Server
    }

    // Re-apply scheduled jobs: stale job threads stop, the new set starts.
    gate.scheduler.reload(&cfg.job);

    print_exposure_report(&cfg, token.is_some());
    println!("gatekeeper: reloaded config (SIGHUP) — routes + cert applied");
}

/// Build a tiny_http Server bound to a *clone* of `listener`'s socket, with TLS
/// if the config has a cert/key. Cloning the fd lets us build a fresh Server on
/// the same bound socket without closing it.
fn build_server(
    listener: &std::net::TcpListener,
    cfg: &Config,
) -> Result<tiny_http::Server, String> {
    let sock = listener
        .try_clone()
        .map_err(|e| format!("cloning listener socket: {e}"))?;
    let ssl = match (&cfg.tls_cert, &cfg.tls_key) {
        (Some(cert), Some(key)) => {
            let certificate = std::fs::read(cert)
                .map_err(|e| format!("reading tls_cert {}: {e}", cert.display()))?;
            let private_key = std::fs::read(key)
                .map_err(|e| format!("reading tls_key {}: {e}", key.display()))?;
            Some(tiny_http::SslConfig {
                certificate,
                private_key,
            })
        }
        _ => None,
    };
    tiny_http::Server::from_listener(sock, ssl).map_err(|e| format!("starting server: {e}"))
}

/// Handle a single request: match route, enforce auth on private routes, then
/// serve static or proxy. Anything unexpected fails closed.
fn handle(gate: &Gate, mut request: tiny_http::Request) {
    // tiny_http's url() is the path+query. Split the query off for routing;
    // keep the full thing for proxying.
    let raw_url = request.url().to_string();
    let (path, query) = raw_url.split_once('?').unwrap_or((raw_url.as_str(), ""));
    let query = query.to_string();

    // Snapshot the current routing for the whole request. A concurrent reload
    // swaps in a fresh Arc<Routing>; we hold our snapshot so the decision is
    // consistent even if a SIGHUP lands mid-request.
    let routing = gate.routing();

    // Built-in meta route: a self-describing API catalog. Reserved path, always
    // available, and PRIVATE — it requires the same token as any private route,
    // because it enumerates every route (including private ones) and their
    // functions' endpoints. Normalize first so `/_gatekeeper/describe/` etc. and
    // any encoding match the same way the router would.
    if let Some(norm) = Router::normalize(path) {
        if norm == DESCRIBE_PATH {
            let authed = routing
                .auth
                .as_ref()
                .map(|a| a.check_headers(request.headers()))
                .unwrap_or(false);
            let reply = if authed {
                describe_catalog(gate, &routing)
            } else {
                Reply::status(401, "Unauthorized").with_header("WWW-Authenticate", "Bearer")
            };
            let _ = reply.respond(request);
            return;
        }
    }

    let reply = match routing.router.resolve(path) {
        Match::BadPath => Reply::status(400, "Bad Request"),
        Match::NoRoute => Reply::status(routing.unmatched_status, "Not Found"),
        Match::Route { route, rest, .. } => {
            // The safety gate: private routes require a valid token.
            if !route.public {
                let ok = routing
                    .auth
                    .as_ref()
                    .map(|a| a.check_headers(request.headers()))
                    .unwrap_or(false);
                if !ok {
                    let r = Reply::status(401, "Unauthorized")
                        .with_header("WWW-Authenticate", "Bearer");
                    let _ = r.respond(request);
                    return;
                }
            }
            match route.target() {
                Target::Static(dir) => serve::serve(&dir, &rest),
                Target::Proxy(upstream) => {
                    // Read the request body to forward it (bounded by tiny_http).
                    let mut body = Vec::new();
                    let _ = request.as_reader().read_to_end(&mut body);
                    let method = request.method().as_str().to_string();
                    // Proxy the full URL (path + query) so upstreams see queries.
                    proxy::forward(&upstream, &method, &raw_url, request.headers(), &body)
                }
                Target::Function(lib) => {
                    // Read the body, then invoke the dylib in process. `rest` is
                    // the path after the route prefix (already normalized); the
                    // function sees that plus the query separately.
                    let mut body = Vec::new();
                    let _ = request.as_reader().read_to_end(&mut body);
                    let method = request.method().as_str().to_string();
                    gate.functions.invoke(
                        &lib,
                        &method,
                        &rest,
                        &query,
                        request.headers(),
                        &body,
                    )
                }
            }
        }
    };
    let _ = reply.respond(request);
}

/// Print, at every boot, exactly what is and isn't exposed — so a human can
/// eyeball "did I mean to make these public?".
fn print_exposure_report(cfg: &Config, token_configured: bool) {
    let line = "=".repeat(60);
    println!("\n{line}\nGATEKEEPER EXPOSURE REPORT\n{line}");
    if !cfg.tls_enabled() {
        println!("  TLS: OFF (plain HTTP — use only behind a terminator or on localhost)");
    } else {
        println!("  TLS: on");
    }
    println!(
        "  Auth token: {}",
        if token_configured { "configured" } else { "NONE" }
    );
    println!("  Unmatched requests -> {}", cfg.unmatched_status);

    let public: Vec<_> = cfg.route.iter().filter(|r| r.public).collect();
    let private: Vec<_> = cfg.route.iter().filter(|r| !r.public).collect();

    println!("\n  PUBLIC routes (no auth):");
    if public.is_empty() {
        println!("    (none)");
    }
    for r in &public {
        println!("    {}  ->  {}", r.path, target_desc(r));
    }

    println!("\n  PRIVATE routes (token required):");
    if private.is_empty() {
        println!("    (none)");
    }
    for r in &private {
        println!("    {}  ->  {}", r.path, target_desc(r));
    }

    println!("\n  SCHEDULED jobs:");
    if cfg.job.is_empty() {
        println!("    (none)");
    }
    for j in &cfg.job {
        let when = if j.run_at_start {
            format!("every {} (and at start)", j.every)
        } else {
            format!("every {}", j.every)
        };
        println!("    {}  ->  `{}`  [{}]", j.name, j.command.join(" "), when);
    }
    println!("{line}");
}

fn target_desc(r: &config::Route) -> String {
    match (&r.static_dir, &r.proxy, &r.function) {
        (Some(d), _, _) => format!("static {}", d.display()),
        (_, Some(u), _) => format!("proxy {u}"),
        (_, _, Some(l)) => format!("function {}", l.display()),
        _ => "(invalid)".into(),
    }
}

/// Build the `/_gatekeeper/describe` catalog: a JSON object listing every route
/// (path, access, target) and, for function routes, the function's own
/// self-description (endpoints/params/examples) fetched via `gk_describe`.
///
/// This is the one place the gate's knowledge (routes, public/private, from the
/// toml) is joined with each function's knowledge (its endpoints, from the
/// dylib). Function descriptions are embedded under their route so a caller sees
/// the full path: route prefix + the function's sub-paths.
fn describe_catalog(gate: &Gate, routing: &Routing) -> Reply {
    use serde_json::{json, Value};

    let mut routes = Vec::new();
    for r in routing.router.routes() {
        let access = if r.public { "public" } else { "private" };
        let mut entry = json!({
            "path": r.path,
            "access": access,
            "target": target_desc(r),
        });

        // For a function route, fetch and embed its self-description. The
        // function's endpoint paths are RELATIVE to this route's prefix, so we
        // also surface the prefix to make the full path obvious.
        if let Some(lib) = &r.function {
            entry["kind"] = json!("function");
            match gate.functions.describe(lib) {
                Ok(desc_json) => {
                    // The function returned JSON text; embed it parsed if valid,
                    // else surface the raw string so nothing is silently lost.
                    match serde_json::from_str::<Value>(&desc_json) {
                        Ok(v) => entry["description"] = v,
                        Err(_) => entry["description_raw"] = json!(desc_json),
                    }
                }
                Err(e) => entry["description_error"] = json!(e),
            }
        } else if r.static_dir.is_some() {
            entry["kind"] = json!("static");
        } else if r.proxy.is_some() {
            entry["kind"] = json!("proxy");
        }
        routes.push(entry);
    }

    let catalog = json!({
        "gatekeeper": {
            "describe_path": DESCRIBE_PATH,
            "abi_version": gatekeeper_abi::GK_ABI_VERSION,
        },
        "routes": routes,
    });

    let body = serde_json::to_vec_pretty(&catalog).unwrap_or_else(|_| b"{}".to_vec());
    Reply::new(200, body).with_header("Content-Type", "application/json")
}
