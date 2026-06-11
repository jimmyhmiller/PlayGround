//! Test fixture: a function that has a #[handler] but NO #[describe]. The gate
//! must refuse to load it, because self-description is required (ABI v2).
use gatekeeper_fn::{handler, Request, Response};

#[handler]
fn app(_req: Request) -> Response {
    Response::text("should never be reachable — this dylib must fail to load")
}
