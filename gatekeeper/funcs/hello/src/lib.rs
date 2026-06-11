//! Example gatekeeper serverless function.
//!
//! This is the *entire* app. No ports, no ABI, no unsafe — just a handler. Build
//! it with `cargo build -p hello-fn` and point a gatekeeper route at the
//! resulting `target/debug/libhello_fn.so` via `function = "..."`.

use gatekeeper_fn::{handler, Request, Response};

#[handler]
fn app(req: Request) -> Response {
    match req.path() {
        "/health" | "/health/" => Response::text("ok"),
        "/echo" => Response::json(format!(
            r#"{{"method":"{}","query":"{}","body":"{}"}}"#,
            req.method(),
            req.query(),
            req.text().replace('"', "\\\"")
        )),
        "/panic" => panic!("deliberate panic to prove the gate survives it"),
        p => Response::html(format!(
            "<h1>hello from a gatekeeper function</h1><p>you asked for <code>{p}</code></p>"
        )),
    }
}
