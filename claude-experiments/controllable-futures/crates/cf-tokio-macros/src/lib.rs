//! Proc-macro for the `#[cf_tokio::main]` attribute. Wraps an async `fn main`
//! into a sync `fn main` that constructs a multi-threaded cf-runtime and
//! `block_on`s the body. Mirrors `#[tokio::main]` minimally — no flavor or
//! worker_threads attributes yet.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let f = parse_macro_input!(item as ItemFn);
    let attrs = &f.attrs;
    let vis = &f.vis;
    let sig = &f.sig;
    let body = &f.block;
    if sig.asyncness.is_none() {
        return syn::Error::new_spanned(sig.fn_token, "cf_tokio::main requires an async fn")
            .to_compile_error()
            .into();
    }
    let mut sync_sig = sig.clone();
    sync_sig.asyncness = None;
    let output = quote! {
        #(#attrs)*
        #vis #sync_sig {
            let __cf_rt = ::cf_tokio::runtime::Runtime::new().expect("build runtime");
            __cf_rt.block_on(async move #body)
        }
    };
    output.into()
}
