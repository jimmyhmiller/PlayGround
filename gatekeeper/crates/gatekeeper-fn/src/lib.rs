//! Write a serverless Rust function for gatekeeper.
//!
//! Your function is a normal Rust `cdylib`. You write one handler:
//!
//! ```ignore
//! use gatekeeper_fn::{handler, Request, Response};
//!
//! #[handler]
//! fn app(req: Request) -> Response {
//!     match req.path() {
//!         "/health" => Response::text("ok"),
//!         _ => Response::json(r#"{"hello":"world"}"#),
//!     }
//! }
//! ```
//!
//! and in `Cargo.toml`:
//!
//! ```toml
//! [lib]
//! crate-type = ["cdylib"]
//! ```
//!
//! That's the whole contract. The `#[handler]` macro generates the C-ABI symbols
//! the gate loads (see [`gatekeeper_abi`]); you never touch raw pointers, ports,
//! versions, or unsafe. A panic in your handler is caught and turned into a 500
//! by the gate-facing glue, so one bad request can't take down the gate.
//!
//! The ergonomic [`Request`]/[`Response`] types here own normal Rust data; the
//! marshalling to/from the `#[repr(C)]` ABI structs happens in [`__rt`].

pub use gatekeeper_fn_macro::handler;

/// An incoming HTTP request, as seen by your handler. All data is owned (copied
/// out of the borrowed ABI request before your code runs), so you can keep it,
/// move it, and return after the call without lifetime worries.
#[derive(Debug, Clone)]
pub struct Request {
    method: String,
    path: String,
    query: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    /// HTTP method, uppercase (`GET`, `POST`, …).
    pub fn method(&self) -> &str {
        &self.method
    }
    /// Request path after the route prefix (e.g. `/users/3`); `""` if the
    /// request hit the route root exactly.
    pub fn path(&self) -> &str {
        &self.path
    }
    /// Raw query string without the leading `?` (empty if none).
    pub fn query(&self) -> &str {
        &self.query
    }
    /// All headers as (name, value) pairs, in arrival order.
    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }
    /// First header value matching `name` (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    /// Raw request body bytes.
    pub fn body(&self) -> &[u8] {
        &self.body
    }
    /// Request body as UTF-8 text, lossily (invalid bytes become `�`).
    pub fn text(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.body)
    }
}

/// The response your handler returns.
#[derive(Debug, Clone)]
pub struct Response {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Response {
    /// A response with an explicit status and raw body, no headers.
    pub fn new(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Response {
            status,
            headers: Vec::new(),
            body: body.into(),
        }
    }
    /// `200 OK` with a `text/plain` body.
    pub fn text(body: impl Into<String>) -> Self {
        Response::new(200, body.into().into_bytes())
            .header("Content-Type", "text/plain; charset=utf-8")
    }
    /// `200 OK` with an `application/json` body. You supply the JSON string.
    pub fn json(body: impl Into<String>) -> Self {
        Response::new(200, body.into().into_bytes()).header("Content-Type", "application/json")
    }
    /// `200 OK` with a `text/html` body.
    pub fn html(body: impl Into<String>) -> Self {
        Response::new(200, body.into().into_bytes())
            .header("Content-Type", "text/html; charset=utf-8")
    }
    /// A bare status with a short text body (e.g. `Response::status(404, "nope")`).
    pub fn status(status: u16, msg: impl Into<String>) -> Self {
        Response::new(status, msg.into().into_bytes())
    }
    /// Add a response header (builder style).
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
    /// Override the status code (builder style).
    pub fn with_status(mut self, status: u16) -> Self {
        self.status = status;
        self
    }
}

pub mod describe;
pub use describe::{Description, Endpoint, Param};

pub use gatekeeper_fn_macro::describe;

#[doc(hidden)]
pub mod __rt;
