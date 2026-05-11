//! `InlineBody` — a synthesized helper fn with positional captured
//! locals, called once at its synthesis site via `fb.invoke` or
//! `fb.call`. Stack-only, never escapes, never goes on the heap.
//!
//! This is the right shape for `try`/`catch` wrappers, `(catch …)` arm
//! bodies, and any other lowering pattern that wants to "lift this
//! block into a helper fn so the outer can use `fb.invoke` to catch
//! exceptions thrown from inside it."
//!
//! ## Why this is a separate primitive from [`ClosureKit`]
//!
//! Closures and inline-bodies look alike from a distance — both are
//! "free-variable analysis → spill captures → synthesize body fn" — but
//! their IR shapes differ:
//!
//! | | Closure ([`crate::closure::ClosureKit`]) | InlineBody (this module) |
//! |---|---|---|
//! | Storage | Heap GC object (lives past the call site) | Nothing — captures stay in the outer SSA |
//! | Captures arrive in body via | Varlen-tail loads through `self_fn` | Positional block params |
//! | Invocation | `call_via_func_ref` through the heap object | `fb.invoke` (or `fb.call`) |
//! | Called | Many times, from anywhere | Exactly once, at the synthesis site |
//! | Body signature | `(self_fn, args_list)` or `(self_fn, …)` | `(cap0, cap1, …, capN, extra0, …, extraM)` |
//!
//! Trying to unify these as `CallConv` variants of a single kit drags
//! every method through a switch and dilutes the closure's "first-class
//! value" semantics. So we keep them as two small, clean primitives
//! that share `crate::clojure::freevars`-style frontend analysis (which
//! is the only piece that genuinely transfers).
//!
//! ## Lifecycle
//!
//! ```ignore
//! // 1. Declare the body fn. Returns a handle; no IR emitted yet.
//! let body = InlineBody::declare(mb, "__try_body_7", n_captures, n_extras);
//!
//! // 2. Open it for code emission. Returns a FunctionBuilder switched
//! //    to the entry block + the bound capture/extra Values.
//! let (mut inner_fb, captures, extras) = body.open(mb);
//!
//! // 3. Lower the body using the FunctionBuilder. The caller is free
//! //    to use `self` (and `self.module_builder`) recursively here — no borrow is
//! //    held by InlineBody during this phase.
//! let result = self.lower_do(&mut inner_fb, &mut inner_env, body_forms);
//! inner_fb.ret(result);
//!
//! // 4. Finish.
//! body.finish(mb, inner_fb);
//!
//! // 5. At the use site, invoke (or call) the body.
//! body.invoke(fb, &cap_values, &[], normal_bb, exception_bb, &live);
//! ```

use crate::DynModule;
use dynir::builder::{FunctionBuilder, ModuleBuilder};
use dynir::ir::{BlockId, FuncRef, Value};
use dynir::types::{Signature, Type};

/// Handle to a synthesized inline body fn.
///
/// All fields are `pub` so the caller can use the raw FuncRef directly
/// if needed (e.g., to attach debug info or query the module).
pub struct InlineBody {
    pub fref: FuncRef,
    pub n_captures: usize,
    pub n_extras: usize,
    /// Cached name for `import_module_func` — outer builders that were
    /// created before this body was declared need to be taught about
    /// the FuncRef before they can `invoke` it.
    name: String,
    /// Cached signature for the same reason.
    sig: Signature,
}

impl InlineBody {
    /// Declare an inline body with `n_captures` capture params followed
    /// by `n_extras` extra user params (e.g. a `thrown` slot for a
    /// catch-arm body). All params and the return are `I64`.
    ///
    /// Returns a handle. Open the body with [`InlineBody::open`].
    pub fn declare(
        module_builder: &mut ModuleBuilder,
        name: &str,
        n_captures: usize,
        n_extras: usize,
    ) -> Self {
        let total = n_captures + n_extras;
        let params: Vec<Type> = vec![Type::I64; total];
        let sig = Signature { params: params.clone(), ret: Some(Type::I64) };
        let fref = module_builder.declare_func(name, &params, Some(Type::I64));
        InlineBody {
            fref,
            n_captures,
            n_extras,
            name: name.to_string(),
            sig,
        }
    }

    /// Open the body for code emission. Returns:
    /// - a `FunctionBuilder` already switched to the entry block,
    /// - the captured `Value`s (in declaration order),
    /// - the extra `Value`s (in declaration order, after captures).
    ///
    /// After the caller has emitted the body and terminated with
    /// `fb.ret(...)` (or `fb.unreachable()`), call [`InlineBody::finish`]
    /// to install it back into the module.
    pub fn open(&self, module_builder: &ModuleBuilder) -> (FunctionBuilder, Vec<Value>, Vec<Value>) {
        let fb = module_builder.define_func(self.fref);
        let entry = fb.entry_block();
        let captures: Vec<Value> = (0..self.n_captures)
            .map(|i| fb.block_param(entry, i))
            .collect();
        let extras: Vec<Value> = (0..self.n_extras)
            .map(|i| fb.block_param(entry, self.n_captures + i))
            .collect();
        (fb, captures, extras)
    }

    /// Install a body that was opened via [`InlineBody::open`]. The
    /// caller is responsible for having terminated every block in
    /// `fb` (e.g. with `fb.ret`).
    pub fn finish(&self, module_builder: &mut ModuleBuilder, fb: FunctionBuilder) {
        module_builder.finish_func(self.fref, fb);
    }

    /// Emit `fb.invoke` of this body at `fb`'s current insertion point.
    /// Imports the body's signature into `fb` if necessary — idempotent
    /// even if `fb` was created before this body was declared.
    ///
    /// - `captures` and `extras` must match the lengths declared in
    ///   [`InlineBody::declare`].
    /// - `normal_bb` must take a single `Type::I64` block param (the
    ///   body's return value).
    /// - `exception_bb` must take a single `Type::I64` block param (the
    ///   thrown value).
    /// - `live_roots` is the live-value set for the pre-call safepoint.
    pub fn invoke(
        &self,
        fb: &mut FunctionBuilder,
        captures: &[Value],
        extras: &[Value],
        normal_bb: BlockId,
        exception_bb: BlockId,
        live_roots: &[Value],
    ) {
        assert_eq!(
            captures.len(),
            self.n_captures,
            "InlineBody::invoke: expected {} captures, got {}",
            self.n_captures,
            captures.len()
        );
        assert_eq!(
            extras.len(),
            self.n_extras,
            "InlineBody::invoke: expected {} extras, got {}",
            self.n_extras,
            extras.len()
        );
        fb.import_module_func(self.fref, &self.name, self.sig.clone());
        let mut args: Vec<Value> = Vec::with_capacity(self.n_captures + self.n_extras);
        args.extend_from_slice(captures);
        args.extend_from_slice(extras);
        fb.safepoint(live_roots);
        fb.invoke(self.fref, &args, normal_bb, &[], exception_bb, &[]);
    }

    /// Emit `fb.call` of this body at the current point. Same shape as
    /// [`InlineBody::invoke`] but without exception propagation — the
    /// returned value is the body's I64 result. Use this when you've
    /// lifted code into a helper for structural reasons (e.g.
    /// scope-isolating a temporary computation) but don't need the
    /// exception block.
    pub fn call(
        &self,
        fb: &mut FunctionBuilder,
        captures: &[Value],
        extras: &[Value],
        live_roots: &[Value],
    ) -> Value {
        assert_eq!(
            captures.len(),
            self.n_captures,
            "InlineBody::call: expected {} captures, got {}",
            self.n_captures,
            captures.len()
        );
        assert_eq!(
            extras.len(),
            self.n_extras,
            "InlineBody::call: expected {} extras, got {}",
            self.n_extras,
            extras.len()
        );
        fb.import_module_func(self.fref, &self.name, self.sig.clone());
        let mut args: Vec<Value> = Vec::with_capacity(self.n_captures + self.n_extras);
        args.extend_from_slice(captures);
        args.extend_from_slice(extras);
        fb.safepoint(live_roots);
        fb.call(self.fref, &args)
            .expect("InlineBody returns a value")
    }
}

impl DynModule {
    /// Convenience: declare an inline body on this module's builder.
    /// See [`InlineBody::declare`] for semantics.
    pub fn inline_body(
        &mut self,
        name: &str,
        n_captures: usize,
        n_extras: usize,
    ) -> InlineBody {
        InlineBody::declare(&mut self.module_builder, name, n_captures, n_extras)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GcConfig, NanBoxTags};

    #[test]
    fn declare_then_open_returns_block_params() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let body = dyn_module.inline_body("helper", 2, 1);
        assert_eq!(body.n_captures, 2);
        assert_eq!(body.n_extras, 1);

        let (mut fb, captures, extras) = body.open(&dyn_module.module_builder);
        assert_eq!(captures.len(), 2);
        assert_eq!(extras.len(), 1);
        // Use them to keep things real — sum captures + extra and ret.
        let sum1 = fb.add(captures[0], captures[1]);
        let sum2 = fb.add(sum1, extras[0]);
        fb.ret(sum2);
        body.finish(&mut dyn_module.module_builder, fb);
    }

    #[test]
    fn zero_captures_zero_extras_is_just_a_thunk() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let body = dyn_module.inline_body("thunk", 0, 0);
        let (mut fb, captures, extras) = body.open(&dyn_module.module_builder);
        assert!(captures.is_empty());
        assert!(extras.is_empty());
        let nil = fb.iconst(Type::I64, 0);
        fb.ret(nil);
        body.finish(&mut dyn_module.module_builder, fb);
    }

    #[test]
    #[should_panic(expected = "expected 2 captures, got 1")]
    fn invoke_with_wrong_capture_count_panics() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let body = dyn_module.inline_body("helper", 2, 0);
        let (mut bfb, _, _) = body.open(&dyn_module.module_builder);
        let nil = bfb.iconst(Type::I64, 0);
        bfb.ret(nil);
        body.finish(&mut dyn_module.module_builder, bfb);

        // Outer fn that invokes it with wrong arity:
        let outer = dyn_module.declare_func("outer", 0);
        let mut fb = dyn_module.start_func(outer);
        let v = fb.fb.iconst(Type::I64, 0);
        let normal_bb = fb.fb.create_block(&[Type::I64]);
        let exc_bb = fb.fb.create_block(&[Type::I64]);
        body.invoke(&mut fb.fb, &[v], &[], normal_bb, exc_bb, &[]);
    }

    #[test]
    fn invoke_emits_safepoint_then_invoke() {
        // We can't easily inspect the emitted IR without going through
        // the verifier, but we can at least exercise the API end-to-end
        // and trust the builder's own validation.
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let body = dyn_module.inline_body("helper", 1, 0);
        let (mut bfb, caps, _) = body.open(&dyn_module.module_builder);
        bfb.ret(caps[0]);
        body.finish(&mut dyn_module.module_builder, bfb);

        let outer = dyn_module.declare_func("outer", 1);
        let mut fb = dyn_module.start_func(outer);
        let arg0 = fb.fb.block_param(fb.fb.entry_block(), 0);
        let normal_bb = fb.fb.create_block(&[Type::I64]);
        let exc_bb = fb.fb.create_block(&[Type::I64]);
        body.invoke(&mut fb.fb, &[arg0], &[], normal_bb, exc_bb, &[]);

        // Terminate the normal/exc blocks so the function verifies.
        fb.fb.switch_to_block(normal_bb);
        let n = fb.fb.block_param(normal_bb, 0);
        fb.fb.ret(n);
        fb.fb.switch_to_block(exc_bb);
        fb.fb.unreachable();
        dyn_module.finish_func(fb);
    }
}
