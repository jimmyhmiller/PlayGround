//! The C-ABI contract between the gatekeeper gate and a function dylib.
//!
//! This crate is the **entire** surface across the `dlopen` boundary. Both the
//! gate (which loads dylibs) and every function (which is a `cdylib`) depend on
//! it, so they agree on memory layout. Everything here is `#[repr(C)]` and uses
//! only raw pointers + lengths — no Rust types (`String`, `Vec`, enums with
//! payloads, …) ever cross the boundary, because Rust has no stable ABI for
//! those.
//!
//! ## Ownership rule (the one thing that matters for soundness)
//!
//! Each side frees what it allocated. The gate builds the [`GkRequest`] and owns
//! that memory; the function only *borrows* it for the duration of the call. The
//! function builds the [`GkResponse`] and the gate must hand it back to the
//! function's [`GK_FREE_SYMBOL`] to release it — the gate never frees response
//! memory itself, because the dylib may use a different allocator.
//!
//! ## Symbols a function dylib must export
//!
//! - [`GK_ABI_VERSION_SYMBOL`] — `extern "C" fn() -> u32` returning
//!   [`GK_ABI_VERSION`]. The gate refuses to load a mismatched dylib.
//! - [`GK_HANDLE_SYMBOL`] — `extern "C" fn(*const GkRequest) -> *mut GkResponse`.
//! - [`GK_FREE_SYMBOL`] — `extern "C" fn(*mut GkResponse)` to free a response.
//!
//! The `gatekeeper-fn` SDK generates all three for you from a single handler;
//! you should not hand-write them.

#![allow(clippy::missing_safety_doc)]

use std::os::raw::c_char;

/// ABI version. Bump on ANY layout change to the structs below or the symbol
/// contract. The gate compares this against the dylib's reported version at load
/// time and refuses to call a mismatched function (fail closed).
pub const GK_ABI_VERSION: u32 = 1;

/// Name of the exported version function: `extern "C" fn() -> u32`.
pub const GK_ABI_VERSION_SYMBOL: &[u8] = b"gk_abi_version";
/// Name of the exported handler: `extern "C" fn(*const GkRequest) -> *mut GkResponse`.
pub const GK_HANDLE_SYMBOL: &[u8] = b"gk_handle";
/// Name of the exported deallocator: `extern "C" fn(*mut GkResponse)`.
pub const GK_FREE_SYMBOL: &[u8] = b"gk_free";

/// One HTTP header as a borrowed name/value pair. Pointers are valid only for
/// the duration of the [`GK_HANDLE_SYMBOL`] call. Not NUL-terminated — use the
/// length fields. Bytes are raw (the gate does not validate UTF-8 for you; the
/// SDK does when it presents them as `&str`).
#[repr(C)]
pub struct GkHeader {
    pub name_ptr: *const c_char,
    pub name_len: usize,
    pub value_ptr: *const c_char,
    pub value_len: usize,
}

/// The request handed to a function. All pointers are **borrowed** from the gate
/// and valid only until [`GK_HANDLE_SYMBOL`] returns. The function must not free
/// them or retain them past the call.
#[repr(C)]
pub struct GkRequest {
    /// HTTP method, e.g. `GET`. Borrowed bytes, length-delimited.
    pub method_ptr: *const c_char,
    pub method_len: usize,
    /// Request path after the matched route prefix (always `""` or starts with
    /// `/`), already normalized + traversal-checked by the gate.
    pub path_ptr: *const c_char,
    pub path_len: usize,
    /// Raw query string without the leading `?` (empty if none).
    pub query_ptr: *const c_char,
    pub query_len: usize,
    /// Borrowed array of `header_count` headers.
    pub headers_ptr: *const GkHeader,
    pub header_count: usize,
    /// Request body bytes (may be empty).
    pub body_ptr: *const u8,
    pub body_len: usize,
}

/// The response a function returns. The function **owns** this allocation; the
/// gate copies the data out and then returns the pointer to [`GK_FREE_SYMBOL`].
/// A null return from the handler is treated by the gate as an internal error
/// (500).
#[repr(C)]
pub struct GkResponse {
    pub status: u16,
    /// Owned array of `header_count` headers (pointers owned by the function's
    /// allocator). May be null iff `header_count == 0`.
    pub headers_ptr: *mut GkHeader,
    pub header_count: usize,
    /// Owned body bytes. May be null iff `body_len == 0`.
    pub body_ptr: *mut u8,
    pub body_len: usize,
}
