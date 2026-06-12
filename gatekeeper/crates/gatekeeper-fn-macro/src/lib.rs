//! The `#[handler]` attribute macro for `gatekeeper-fn`.
//!
//! Applied to a `fn(Request) -> Response`, it leaves that function untouched and
//! *also* emits the three C-ABI symbols the gate looks for (`gk_abi_version`,
//! `gk_handle`, `gk_free`), wiring the unsafe marshalling + panic catching to the
//! user's function. The app author writes only the clean handler.
//!
//! Implemented without `syn`/`quote` to keep the dependency tree tiny (in the
//! spirit of the rest of gatekeeper): we scan the token stream for the `fn`
//! keyword and take the following identifier as the handler name, then splice it
//! into a generated glue block built from raw token parsing.

use proc_macro::{TokenStream, TokenTree};

/// Mark a `fn(Request) -> Response` as the dylib's request handler.
///
/// ```ignore
/// use gatekeeper_fn::{handler, Request, Response};
///
/// #[handler]
/// fn app(req: Request) -> Response {
///     Response::text(format!("you asked for {}", req.path()))
/// }
/// ```
///
/// The macro re-emits `app` unchanged and adds the exported `gk_*` symbols that
/// forward to it. Exactly one `#[handler]` per cdylib.
#[proc_macro_attribute]
pub fn handler(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let name = match handler_name(&item) {
        Some(n) => n,
        None => panic!("#[handler] must be applied to a `fn name(req: Request) -> Response`"),
    };

    // The generated glue, with `{NAME}` replaced by the user's function name.
    // We build it as a string and parse it back into tokens — simplest reliable
    // path without quote!.
    let glue = GLUE.replace("{NAME}", &name);
    let generated: TokenStream = glue
        .parse()
        .expect("gatekeeper-fn-macro: internal error generating glue (please report)");

    // Emit the original function first, then the glue.
    let mut out = item;
    out.extend(generated);
    out
}

/// Mark a `fn() -> Description` as the dylib's self-description.
///
/// ```ignore
/// use gatekeeper_fn::{describe, Description, Endpoint, Param};
///
/// #[describe]
/// fn describe() -> Description {
///     Description::new("analytics", "Website-visit analytics")
///         .endpoint(Endpoint::get("/summary", "headline numbers"))
/// }
/// ```
///
/// Emits the `gk_describe` ABI symbol forwarding to it, so the gate can build its
/// `/describe` catalog. **Required**: every function must have exactly
/// one `#[describe]` — the gate refuses to load a dylib without `gk_describe`, so
/// the catalog is always complete.
#[proc_macro_attribute]
pub fn describe(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let name = match handler_name(&item) {
        Some(n) => n,
        None => panic!("#[describe] must be applied to a `fn name() -> Description`"),
    };
    let glue = DESCRIBE_GLUE.replace("{NAME}", &name);
    let generated: TokenStream = glue
        .parse()
        .expect("gatekeeper-fn-macro: internal error generating describe glue (please report)");
    let mut out = item;
    out.extend(generated);
    out
}

/// Find the identifier that immediately follows the `fn` keyword.
fn handler_name(item: &TokenStream) -> Option<String> {
    let mut saw_fn = false;
    for tt in item.clone() {
        match tt {
            TokenTree::Ident(id) => {
                if saw_fn {
                    return Some(id.to_string());
                }
                if id.to_string() == "fn" {
                    saw_fn = true;
                }
            }
            _ => {}
        }
    }
    None
}

/// The glue, as a template. Every exported symbol forwards into the SDK runtime
/// in `gatekeeper_fn::__rt`, which owns all the unsafe pointer work and panic
/// catching so it is written once and tested, not regenerated per handler.
const GLUE: &str = r#"
#[no_mangle]
pub extern "C" fn gk_abi_version() -> u32 {
    ::gatekeeper_fn::__rt::abi_version()
}

#[no_mangle]
pub unsafe extern "C" fn gk_handle(
    req: *const ::gatekeeper_fn::__rt::GkRequest,
) -> *mut ::gatekeeper_fn::__rt::GkResponse {
    ::gatekeeper_fn::__rt::dispatch(req, {NAME})
}

#[no_mangle]
pub unsafe extern "C" fn gk_free(resp: *mut ::gatekeeper_fn::__rt::GkResponse) {
    ::gatekeeper_fn::__rt::free(resp)
}
"#;

/// Glue for `#[describe]`: emit `gk_describe`, forwarding to the user's function
/// (which returns a `Description`) through the SDK runtime, which serializes it to
/// a JSON `GkResponse` and catches panics.
const DESCRIBE_GLUE: &str = r#"
#[no_mangle]
pub unsafe extern "C" fn gk_describe() -> *mut ::gatekeeper_fn::__rt::GkResponse {
    ::gatekeeper_fn::__rt::describe({NAME})
}
"#;
