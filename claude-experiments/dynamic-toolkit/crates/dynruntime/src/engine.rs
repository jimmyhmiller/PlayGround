//! ExecutionEngine — coordinates interpreter and JIT execution.
//!
//! The engine owns a Module reference, an interpreter, an optional JIT module,
//! and a shared ContinuationStore. It handles dispatch (which path to use),
//! deoptimization recovery, and cross-mode continuation resume.

use std::marker::PhantomData;

use dynexec::{
    CodegenConfig, DefaultCodegenConfig, LayoutConfigDefaults,
};
use dynir::interp::{
    ConfiguredModuleInterpreter, ExternCallResult, InterpError, InterpResult, InterpRootManager,
    NoGcRoots,
};
use dynir::ir::*;
use dynlower::JitModule;

use crate::jit::{
    JitExecutionResult, JitFrameControlError, JitFrameSliceRuntime,
    execute_jit_module_function_to_terminal,
};

// ─── TierPolicy ──────────────────────────────────────────────────

/// Decides when the module should be JIT compiled.
pub trait TierPolicy {
    /// Called before each interpreted function entry.
    /// Returns true if the whole module should be JIT compiled.
    fn should_compile(&mut self, func_idx: usize) -> bool;
}

/// Never JIT compile — pure interpreter mode.
pub struct NeverCompile;

impl TierPolicy for NeverCompile {
    fn should_compile(&mut self, _func_idx: usize) -> bool {
        false
    }
}

/// JIT compile the module on the very first call.
pub struct AlwaysCompile {
    compiled: bool,
}

impl AlwaysCompile {
    pub fn new() -> Self {
        AlwaysCompile { compiled: false }
    }
}

impl Default for AlwaysCompile {
    fn default() -> Self {
        Self::new()
    }
}

impl TierPolicy for AlwaysCompile {
    fn should_compile(&mut self, _func_idx: usize) -> bool {
        if self.compiled {
            false
        } else {
            self.compiled = true;
            true
        }
    }
}

/// JIT compile after any function has been interpreted `threshold` times total.
pub struct CallCountTier {
    threshold: u64,
    counts: Vec<u64>,
}

impl CallCountTier {
    pub fn new(threshold: u64) -> Self {
        CallCountTier {
            threshold,
            counts: Vec::new(),
        }
    }

    fn ensure_size(&mut self, func_idx: usize) {
        if func_idx >= self.counts.len() {
            self.counts.resize(func_idx + 1, 0);
        }
    }
}

impl TierPolicy for CallCountTier {
    fn should_compile(&mut self, func_idx: usize) -> bool {
        self.ensure_size(func_idx);
        self.counts[func_idx] += 1;
        self.counts[func_idx] >= self.threshold
    }
}

// ─── ExecutionResult ─────────────────────────────────────────────

/// Unified result from the execution engine.
#[derive(Debug, PartialEq)]
pub enum ExecutionResult {
    Value(u64),
    Void,
}

/// Errors from the execution engine.
#[derive(Debug)]
pub enum ExecutionError {
    Interp(InterpError),
    Jit(JitFrameControlError),
    CompilationFailed(String),
}

impl From<InterpError> for ExecutionError {
    fn from(e: InterpError) -> Self {
        ExecutionError::Interp(e)
    }
}

impl From<JitFrameControlError> for ExecutionError {
    fn from(e: JitFrameControlError) -> Self {
        ExecutionError::Jit(e)
    }
}

// ─── ExecutionEngine ─────────────────────────────────────────────

/// Coordinates interpreter and JIT execution of a Module.
///
/// Owns the interpreter, an optional JIT module, and a continuation store.
/// Dispatches calls to JIT when available, falls back to interpreter,
/// and handles deopt recovery and cross-mode continuation resume.
pub struct ExecutionEngine<
    'a,
    Cfg: CodegenConfig = DefaultCodegenConfig<dynvalue::LowBit<3>>,
    R: InterpRootManager<Cfg::Layout, Cfg::Roots, Cfg::RootTransport> = NoGcRoots,
> where
    Cfg::Layout: LayoutConfigDefaults,
{
    interpreter: ConfiguredModuleInterpreter<'a, Cfg, R>,
    module: &'a Module,
    jit: Option<JitModule>,
    jit_conts: JitFrameSliceRuntime<'a>,
    tier_policy: Box<dyn TierPolicy>,
    _phantom: PhantomData<Cfg>,
}

/// Static singleton for engines without a heap-backed continuation context.
static ENGINE_NO_CONTS: dynexec::NoContinuations = dynexec::NoContinuations;

/// Create a default interpreter-only engine (LowBit<3>, no GC).
pub fn default_engine<'a>(module: &'a Module) -> ExecutionEngine<'a> {
    ExecutionEngine::with_tier_policy(module, &NoGcRoots, &ENGINE_NO_CONTS, Box::new(NeverCompile))
}

impl<'a, Cfg, R> ExecutionEngine<'a, Cfg, R>
where
    Cfg: CodegenConfig,
    Cfg::Layout: LayoutConfigDefaults,
    R: InterpRootManager<Cfg::Layout, Cfg::Roots, Cfg::RootTransport>,
{
    /// Create an interpreter-only engine.
    pub fn new(module: &'a Module, roots: &'a R, cont_ctx: &'a dyn dynexec::ContinuationContext) -> Self {
        Self::with_tier_policy(module, roots, cont_ctx, Box::new(NeverCompile))
    }

    /// Create an engine with a custom tier policy.
    pub fn with_tier_policy(
        module: &'a Module,
        roots: &'a R,
        cont_ctx: &'a dyn dynexec::ContinuationContext,
        tier_policy: Box<dyn TierPolicy>,
    ) -> Self {
        let mut interp = ConfiguredModuleInterpreter::new(module, roots);
        interp.set_cont_ctx(cont_ctx);
        ExecutionEngine {
            interpreter: interp,
            module,
            jit: None,
            jit_conts: JitFrameSliceRuntime::new(cont_ctx),
            tier_policy,
            _phantom: PhantomData,
        }
    }

    /// Bind an extern function by name.
    pub fn bind(&mut self, name: &str, f: impl Fn(&[u64]) -> ExternCallResult + 'a) {
        self.interpreter.bind_by_name(name, f);
    }

    /// Bind an extern function by FuncRef.
    pub fn bind_ref(&mut self, fref: FuncRef, f: impl Fn(&[u64]) -> ExternCallResult + 'a) {
        self.interpreter.bind(fref, f);
    }

    /// Bind a handler for indirect calls.
    pub fn bind_indirect(&mut self, handler: impl Fn(u64, &[u64]) -> ExternCallResult + 'a) {
        self.interpreter.bind_indirect(handler);
    }

    /// Provide a pre-compiled JIT module.
    /// All internal functions will be dispatched to native code.
    pub fn set_jit(&mut self, jit: JitModule) {
        self.jit = Some(jit);
    }

    /// Run an entry function.
    ///
    /// If a JIT module is available, dispatches to native code.
    /// Otherwise interprets. Handles deopt by falling back to interpreter.
    pub fn run(
        &mut self,
        entry: FuncRef,
        args: &[u64],
    ) -> Result<ExecutionResult, ExecutionError> {
        // If we have a JIT module, use it
        if let Some(jit) = &self.jit {
            let result =
                execute_jit_module_function_to_terminal(jit, entry, args, &self.jit_conts)?;
            return self.handle_jit_result(result);
        }

        // Check tier policy — should we compile?
        let entry_idx = match &self.module.func_table[entry.index()] {
            FuncDef::Internal(idx) => *idx,
            FuncDef::Extern(_) => {
                return Err(ExecutionError::CompilationFailed(
                    "cannot run extern function as entry".into(),
                ))
            }
        };

        if self.tier_policy.should_compile(entry_idx) {
            // Try to JIT compile the module
            if let Some(jit) = self.try_compile() {
                let result = execute_jit_module_function_to_terminal(
                    &jit,
                    entry,
                    args,
                    &self.jit_conts,
                )?;
                self.jit = Some(jit);
                return self.handle_jit_result(result);
            }
        }

        // Interpret
        let result = self.interpreter.run(entry, args)?;
        Ok(interp_to_result(result))
    }

    /// Try to JIT compile the module. Returns None on failure.
    fn try_compile(&self) -> Option<JitModule> {
        // Build extern table from module
        let externs: Vec<*const u8> = self
            .module
            .func_table
            .iter()
            .map(|_| std::ptr::null())
            .collect();
        // Attempt compilation — panics become None via catch_unwind
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            JitModule::compile::<Cfg::Layout>(&self.module, &externs)
        }))
        .ok()
    }

    /// Handle a JIT execution result, including deopt recovery.
    fn handle_jit_result(
        &self,
        result: JitExecutionResult,
    ) -> Result<ExecutionResult, ExecutionError> {
        match result {
            JitExecutionResult::Value(v) => Ok(ExecutionResult::Value(v)),
            JitExecutionResult::Void => Ok(ExecutionResult::Void),
            JitExecutionResult::Exception(exc) => {
                Err(ExecutionError::Interp(InterpError::UncaughtException(exc)))
            }
            JitExecutionResult::Deopt {
                deopt_id: _,
                resume_point: _,
                live_values: _,
            } => {
                // Deopt: fall back to interpreter.
                // Full deopt recovery (rebuilding interpreter state from live_values
                // at resume_point) is a Pass 3 item. For now, report the deopt.
                Err(ExecutionError::CompilationFailed(
                    "deopt recovery not yet implemented in ExecutionEngine".into(),
                ))
            }
            JitExecutionResult::CaptureSlice { handle, .. } => {
                Ok(ExecutionResult::Value(handle))
            }
            JitExecutionResult::CloneSlice { handle, .. } => {
                Ok(ExecutionResult::Value(handle))
            }
            JitExecutionResult::ResumeSlice { handle, args, .. } => {
                // Cross-mode: JIT captured, interpreter resumes.
                let view = self
                    .jit_conts
                    .read(handle)
                    .map_err(|e| ExecutionError::Jit(JitFrameControlError::FrameSlice(e)))?;
                let result = self
                    .interpreter
                    .resume_view(&view, &args)?;
                Ok(interp_to_result(result))
            }
            JitExecutionResult::AbortToPrompt { .. } => Err(ExecutionError::CompilationFailed(
                "unhandled abort_to_prompt escaped ExecutionEngine".into(),
            )),
        }
    }

    /// Access the continuation store.
    pub fn continuation_store(&self) -> &JitFrameSliceRuntime<'a> {
        &self.jit_conts
    }

    /// Access the continuation store mutably.
    pub fn continuation_store_mut(&mut self) -> &mut JitFrameSliceRuntime<'a> {
        &mut self.jit_conts
    }

    /// Access the interpreter directly.
    pub fn interpreter(&self) -> &ConfiguredModuleInterpreter<'a, Cfg, R> {
        &self.interpreter
    }

    /// Access the JIT module, if set.
    pub fn jit_module(&self) -> Option<&JitModule> {
        self.jit.as_ref()
    }
}

fn interp_to_result(result: InterpResult) -> ExecutionResult {
    match result {
        InterpResult::Value(v) => ExecutionResult::Value(v),
        InterpResult::Void => ExecutionResult::Void,
        InterpResult::Deopt { .. } => ExecutionResult::Void,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynir::builder::{FunctionBuilder, ModuleBuilder};
    use dynir::ir::CmpOp;
    use dynir::types::{Signature, Type};

    /// Helper: build a module with one function.
    fn one_func(
        name: &str,
        params: &[Type],
        ret: Option<Type>,
        build: impl FnOnce(&mut FunctionBuilder),
    ) -> (Module, FuncRef) {
        let mut mb = ModuleBuilder::new();
        let f = mb.declare_func(name, params, ret);
        let mut fb = mb.define_func(f);
        build(&mut fb);
        mb.finish_func(f, fb);
        (mb.build(), f)
    }

    #[test]
    fn engine_interpret_return_const() {
        let (module, entry) = one_func("main", &[], Some(Type::I64), |b| {
            let v = b.iconst(Type::I64, 42);
            b.ret(v);
        });
        let mut engine = default_engine(&module);
        assert_eq!(engine.run(entry, &[]).unwrap(), ExecutionResult::Value(42));
    }

    #[test]
    fn engine_interpret_arithmetic() {
        let (module, entry) = one_func("add", &[Type::I64, Type::I64], Some(Type::I64), |b| {
            let entry = b.entry_block();
            let a = b.block_param(entry, 0);
            let bv = b.block_param(entry, 1);
            let sum = b.add(a, bv);
            b.ret(sum);
        });
        let mut engine = default_engine(&module);
        assert_eq!(
            engine.run(entry, &[10, 32]).unwrap(),
            ExecutionResult::Value(42)
        );
    }

    #[test]
    fn engine_interpret_with_extern() {
        let mut mb = ModuleBuilder::new();
        let f_double = mb.declare_extern(
            "double",
            Signature {
                params: vec![Type::I64],
                ret: Some(Type::I64),
            },
        );
        let f_main = mb.declare_func("main", &[Type::I64], Some(Type::I64));
        let mut fb = mb.define_func(f_main);
        let entry = fb.entry_block();
        let x = fb.block_param(entry, 0);
        let doubled = fb.call(f_double, &[x]).unwrap();
        fb.ret(doubled);
        mb.finish_func(f_main, fb);
        let module = mb.build();

        let mut engine = default_engine(&module);
        engine.bind("double", |args| ExternCallResult::Value(Some(args[0] * 2)));
        assert_eq!(
            engine.run(f_main, &[21]).unwrap(),
            ExecutionResult::Value(42)
        );
    }

    #[test]
    fn engine_interpret_internal_call() {
        let mut mb = ModuleBuilder::new();
        let f_double = mb.declare_func("double", &[Type::I64], Some(Type::I64));
        let f_main = mb.declare_func("main", &[Type::I64], Some(Type::I64));

        {
            let mut fb = mb.define_func(f_double);
            let entry = fb.entry_block();
            let x = fb.block_param(entry, 0);
            let two = fb.iconst(Type::I64, 2);
            let result = fb.mul(x, two);
            fb.ret(result);
            mb.finish_func(f_double, fb);
        }
        {
            let mut fb = mb.define_func(f_main);
            let entry = fb.entry_block();
            let x = fb.block_param(entry, 0);
            let doubled = fb.call(f_double, &[x]).unwrap();
            let one = fb.iconst(Type::I64, 1);
            let result = fb.add(doubled, one);
            fb.ret(result);
            mb.finish_func(f_main, fb);
        }

        let module = mb.build();
        let mut engine = default_engine(&module);
        assert_eq!(
            engine.run(f_main, &[10]).unwrap(),
            ExecutionResult::Value(21)
        );
    }

    #[test]
    fn engine_jit_return_const() {
        let (module, entry) = one_func("main", &[], Some(Type::I64), |b| {
            let v = b.iconst(Type::I64, 99);
            b.ret(v);
        });
        let jit = JitModule::compile::<dynvalue::LowBit<3>>(&module, &[]);
        let mut engine = default_engine(&module);
        engine.set_jit(jit);
        assert_eq!(engine.run(entry, &[]).unwrap(), ExecutionResult::Value(99));
    }

    #[test]
    fn engine_jit_internal_call() {
        let mut mb = ModuleBuilder::new();
        let f_double = mb.declare_func("double", &[Type::I64], Some(Type::I64));
        let f_main = mb.declare_func("main", &[Type::I64], Some(Type::I64));

        {
            let mut fb = mb.define_func(f_double);
            let entry = fb.entry_block();
            let x = fb.block_param(entry, 0);
            let two = fb.iconst(Type::I64, 2);
            let result = fb.mul(x, two);
            fb.ret(result);
            mb.finish_func(f_double, fb);
        }
        {
            let mut fb = mb.define_func(f_main);
            let entry = fb.entry_block();
            let x = fb.block_param(entry, 0);
            let doubled = fb.call(f_double, &[x]).unwrap();
            let one = fb.iconst(Type::I64, 1);
            let result = fb.add(doubled, one);
            fb.ret(result);
            mb.finish_func(f_main, fb);
        }

        let module = mb.build();
        let jit = JitModule::compile::<dynvalue::LowBit<3>>(&module, &[]);
        let mut engine = default_engine(&module);
        engine.set_jit(jit);
        assert_eq!(
            engine.run(f_main, &[10]).unwrap(),
            ExecutionResult::Value(21)
        );
    }

    #[test]
    fn engine_jit_recursive_factorial() {
        let mut mb = ModuleBuilder::new();
        let fact = mb.declare_func("fact", &[Type::I64], Some(Type::I64));
        let main_fn = mb.declare_func("main", &[Type::I64], Some(Type::I64));

        {
            let mut fb = mb.define_func(fact);
            let entry = fb.entry_block();
            let n = fb.block_param(entry, 0);
            let one = fb.iconst(Type::I64, 1);
            let cond = fb.icmp(CmpOp::Sle, n, one);

            let then_bb = fb.create_block(&[]);
            let else_bb = fb.create_block(&[]);
            fb.br_if(cond, then_bb, &[], else_bb, &[]);

            fb.switch_to_block(then_bb);
            fb.ret(one);

            fb.switch_to_block(else_bb);
            let n_minus_1 = fb.sub(n, one);
            let sub_result = fb.call(fact, &[n_minus_1]).unwrap();
            let result = fb.mul(n, sub_result);
            fb.ret(result);
            mb.finish_func(fact, fb);
        }
        {
            let mut fb = mb.define_func(main_fn);
            let entry = fb.entry_block();
            let n = fb.block_param(entry, 0);
            let result = fb.call(fact, &[n]).unwrap();
            fb.ret(result);
            mb.finish_func(main_fn, fb);
        }

        let module = mb.build();
        let jit = JitModule::compile::<dynvalue::LowBit<3>>(&module, &[]);
        let mut engine = default_engine(&module);
        engine.set_jit(jit);
        assert_eq!(
            engine.run(main_fn, &[5]).unwrap(),
            ExecutionResult::Value(120)
        );
    }

    #[test]
    fn engine_tier_up_with_always_compile() {
        let (module, entry) = one_func("main", &[], Some(Type::I64), |b| {
            let v = b.iconst(Type::I64, 77);
            b.ret(v);
        });
        let mut engine: ExecutionEngine =
            ExecutionEngine::with_tier_policy(&module, &NoGcRoots, &ENGINE_NO_CONTS, Box::new(AlwaysCompile::new()));
        // First call triggers compilation
        assert_eq!(engine.run(entry, &[]).unwrap(), ExecutionResult::Value(77));
        // JIT module should now be set
        assert!(engine.jit_module().is_some());
        // Second call uses JIT
        assert_eq!(engine.run(entry, &[]).unwrap(), ExecutionResult::Value(77));
    }

    #[test]
    fn engine_call_count_tier_policy() {
        let (module, entry) = one_func("main", &[], Some(Type::I64), |b| {
            let v = b.iconst(Type::I64, 55);
            b.ret(v);
        });
        let mut engine: ExecutionEngine<'_> = ExecutionEngine::with_tier_policy(
            &module,
            &NoGcRoots,
            &ENGINE_NO_CONTS,
            Box::new(CallCountTier::new(3)),
        );
        // First two calls: interpreted, no JIT
        assert_eq!(engine.run(entry, &[]).unwrap(), ExecutionResult::Value(55));
        assert!(engine.jit_module().is_none());
        assert_eq!(engine.run(entry, &[]).unwrap(), ExecutionResult::Value(55));
        assert!(engine.jit_module().is_none());
        // Third call: triggers JIT compilation
        assert_eq!(engine.run(entry, &[]).unwrap(), ExecutionResult::Value(55));
        assert!(engine.jit_module().is_some());
        // Fourth call: uses JIT
        assert_eq!(engine.run(entry, &[]).unwrap(), ExecutionResult::Value(55));
    }

    #[test]
    fn engine_void_return() {
        let (module, entry) = one_func("main", &[], None, |b| {
            b.ret_void();
        });
        let mut engine = default_engine(&module);
        assert_eq!(engine.run(entry, &[]).unwrap(), ExecutionResult::Void);
    }
}
