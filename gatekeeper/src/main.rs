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
use gatekeeper::proxy;
use gatekeeper::reply::Reply;
use gatekeeper::route::{Match, Router};
use gatekeeper::serve;

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

/// Everything a worker needs to handle a request, shared across threads.
struct Gate {
    router: Router,
    auth: Option<Authenticator>,
    unmatched_status: u16,
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
        router: Router::new(cfg.route.clone()),
        auth: token.as_deref().map(Authenticator::new),
        unmatched_status: cfg.unmatched_status,
    });

    // Build the server (TLS or plain).
    let server = build_server(&cfg)?;
    let server = Arc::new(server);
    println!(
        "\ngatekeeper listening on {} ({})",
        cfg.bind,
        if cfg.tls_enabled() { "HTTPS" } else { "HTTP" }
    );

    // A small fixed pool of workers, each pulling requests off the server.
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(2, 16);
    let mut handles = Vec::new();
    for _ in 0..workers {
        let server = Arc::clone(&server);
        let gate = Arc::clone(&gate);
        handles.push(std::thread::spawn(move || loop {
            match server.recv() {
                Ok(req) => handle(&gate, req),
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

fn build_server(cfg: &Config) -> Result<tiny_http::Server, String> {
    let addr = tiny_http::ConfigListenAddr::from_socket_addrs(cfg.bind.as_str())
        .map_err(|e| format!("bad bind address {:?}: {e}", cfg.bind))?;
    let mut server_config = tiny_http::ServerConfig { addr, ssl: None };
    if let (Some(cert), Some(key)) = (&cfg.tls_cert, &cfg.tls_key) {
        let certificate = std::fs::read(cert)
            .map_err(|e| format!("reading tls_cert {}: {e}", cert.display()))?;
        let private_key =
            std::fs::read(key).map_err(|e| format!("reading tls_key {}: {e}", key.display()))?;
        server_config.ssl = Some(tiny_http::SslConfig {
            certificate,
            private_key,
        });
    }
    tiny_http::Server::new(server_config).map_err(|e| format!("starting server: {e}"))
}

/// Handle a single request: match route, enforce auth on private routes, then
/// serve static or proxy. Anything unexpected fails closed.
fn handle(gate: &Gate, mut request: tiny_http::Request) {
    // tiny_http's url() is the path+query. Split the query off for routing;
    // keep the full thing for proxying.
    let raw_url = request.url().to_string();
    let (path, _query) = raw_url.split_once('?').unwrap_or((raw_url.as_str(), ""));

    let reply = match gate.router.resolve(path) {
        Match::BadPath => Reply::status(400, "Bad Request"),
        Match::NoRoute => Reply::status(gate.unmatched_status, "Not Found"),
        Match::Route { route, rest, .. } => {
            // The safety gate: private routes require a valid token.
            if !route.public {
                let ok = gate
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
    println!("{line}");
}

fn target_desc(r: &config::Route) -> String {
    match (&r.static_dir, &r.proxy) {
        (Some(d), _) => format!("static {}", d.display()),
        (_, Some(u)) => format!("proxy {u}"),
        _ => "(invalid)".into(),
    }
}
