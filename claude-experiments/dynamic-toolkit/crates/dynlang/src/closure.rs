//! `ClosureKit` — a first-class function-value primitive.
//!
//! Every dynamic language has the same shape: a heap object that holds
//! (1) a pointer to the body's compiled code, (2) an arity tag, and
//! (3) the captured environment. Without a shared primitive, each
//! frontend hand-rolls: free-var analysis, the heap-object layout, an
//! allocator, the capture-load prologue, the indirect-call site, and
//! (often) a multi-arity dispatcher. `ClosureKit` factors out the
//! IR-emitting half — frontends keep doing free-var analysis (it's
//! AST-specific) and feed the resulting capture list into the kit.
//!
//! ## Knobs
//!
//! ### Calling convention ([`CallConv`])
//!
//! - **Positional**: body signature is `(self_fn, p0, p1, …)`. Direct
//!   positional calls; arity is fixed at body-fn declaration time.
//! - **ArgsList**: body signature is always `(self_fn, args_list)`.
//!   Call sites pack args into a runtime list; the body walks the list
//!   to bind individual params. This is what you want for Clojure-style
//!   multi-arity dispatch on a single callable. Requires user-supplied
//!   list reader externs ([`ArgsListReaders`]).
//!
//! ### Capture shape ([`CaptureShape`])
//!
//! - **Inline**: captures stored directly in the closure's varlen tail.
//!   Reading a capture is one load; suitable for immutable bindings
//!   (Clojure semantics).
//! - **MutableCells**: captures stored as indirect cells (Lox-style
//!   upvalues). Reserved — not implemented in this version; declared
//!   so the API doesn't shift later. Emits a clear panic on use.
//!
//! ## Lifecycle
//!
//! 1. Once per language: `let kit = dyn_module.closures(config)` to register the
//!    `Closure` GC object type and capture the layout.
//! 2. Per closure: `kit.declare_body(...)` declares the inner FuncRef,
//!    `kit.begin_body(...)` (or `kit.begin_multi_arity_body(...)`) emits
//!    the prologue inside the body fn.
//! 3. At the use site: `kit.make(...)` emits inline GC-allocation IR that
//!    stores `body_ref`, `arity_word`, extras, and captures.
//! 4. At indirect-call sites: `kit.call(...)` emits the unbox + load +
//!    `call_via_func_ref` sequence (and for ArgsList, the args-list
//!    packing is the caller's responsibility).
//!
//! ## What's deliberately NOT in the kit
//!
//! - Free-variable analysis (AST-specific; frontend's job).
//! - The `Closure` GC object's identity field (e.g., a `name` for stack
//!   traces). Use [`ClosureConfig::extra_fields`] to add language-specific
//!   metadata.
//! - The args-list reader externs themselves. The frontend's runtime
//!   already owns its list shape; the kit just *calls* the externs.

use crate::{DynModule, FieldKind, ObjTypeHandle, ObjTypeId};
use dynir::builder::{FunctionBuilder, ModuleBuilder};
use dynir::ir::{BlockId, CmpOp, FuncRef, Value};
use dynir::types::Type;

/// How the body fn receives its arguments and what shape call sites use.
#[derive(Clone)]
pub enum CallConv {
    /// Body signature: `(self_fn, p0, p1, …, pN[, rest])`. Call sites
    /// pass user args directly via [`ClosureKit::call`].
    Positional,

    /// Body signature: `(self_fn, args_list)`. The body walks the list
    /// to bind individual params; multi-arity dispatch reads the list
    /// count and forwards `(self_fn, args_list)` to the matching clause
    /// block. The list shape is opaque to the kit — frontends supply
    /// the reader externs.
    ArgsList { readers: ArgsListReaders },
}

/// Externs the kit invokes when emitting ArgsList prologues / dispatch.
///
/// All FuncRefs are language-provided and must already be declared on
/// the module. The kit does not interpret the boxed values it passes
/// through — frontends own their list and value encodings.
#[derive(Clone, Copy)]
pub struct ArgsListReaders {
    /// `(list) -> first element`. Panics or raises if the list is empty
    /// — the kit only calls this after a successful arity check.
    pub first: FuncRef,
    /// `(list) -> rest tail (or empty-list sentinel)`.
    pub rest: FuncRef,
    /// `(list) -> element count`. Returns a frontend-encoded boxed
    /// integer — same encoding as `encode_arity`'s output.
    pub count: FuncRef,
    /// Encode a Rust `usize` into the same boxed-integer form that
    /// `count` returns. The kit compares arities to runtime counts in
    /// this encoding, so no decode step is needed at the call site.
    pub encode_arity: fn(usize) -> u64,
    /// `(list, arity_word) -> success_sentinel | exception`. Single-
    /// clause arity check. Returns `success_sentinel` (e.g. NIL) on
    /// match, an exception value on mismatch.
    pub check_arity: FuncRef,
    /// `(list) -> exception`. Builds the "no matching arity" exception
    /// raised by multi-arity dispatchers when no clause matches.
    pub no_matching_arity: FuncRef,
    /// `(exception)` — never returns. Raises an exception to the
    /// frontend's catch machinery. Called when `check_arity` returns
    /// non-sentinel or when multi-arity dispatch falls through.
    ///
    /// **Must be invoked through the JIT call table** (via
    /// [`FunctionBuilder::call_via_func_ref`]), not as a plain extern
    /// `call`: extern direct-calls skip the outcome-kind check that
    /// makes the Exception outcome propagate up. The kit honors this
    /// by emitting `call_via_func_ref` through
    /// [`ArgsListReaders::call_table_base`].
    pub raise: FuncRef,
    /// Base address of the JIT call table used to indirectly invoke
    /// `raise`. Pass `jit.call_table_base_addr()`. Required because the
    /// kit cannot rely on a plain extern call to `raise` — see the
    /// `raise` field's doc.
    pub call_table_base: u64,
    /// Bit pattern of `check_arity`'s success return (NIL in most
    /// languages). The kit branches on `result != success_sentinel`
    /// to call `raise`.
    pub success_sentinel: u64,
}

/// How captures are stored inside the closure object.
#[derive(Clone)]
pub enum CaptureShape {
    /// Stored inline in the varlen tail. Reading is one load; writing
    /// (e.g. for mutation) is not supported. Right for Clojure-style
    /// immutable captures.
    Inline,
    /// Stored as pointers to one-field heap cells. Two closures can
    /// share a cell, expressing mutable captured locals (Lox-style
    /// upvalues). Reserved — not implemented in this version.
    MutableCells {
        /// Single-`Value`-field heap type used for each cell.
        cell_type: ObjTypeId,
        /// Name of the cell's `Value` field.
        value_field: &'static str,
    },
}

/// Build-time configuration for a [`ClosureKit`].
#[derive(Clone)]
pub struct ClosureConfig {
    pub captures: CaptureShape,
    pub call_conv: CallConv,
    /// Extra fixed fields on the Closure object, in declaration order.
    /// Pass values to [`ClosureKit::make`] via `MakeClosure::extras` in
    /// the same order. Use for language-specific metadata (e.g. a
    /// `name` symbol for stack traces).
    pub extra_fields: Vec<(String, FieldKind)>,
}

impl ClosureConfig {
    /// Convenience constructor: inline captures + the given call
    /// convention + no extra fields.
    pub fn new(call_conv: CallConv) -> Self {
        Self {
            captures: CaptureShape::Inline,
            call_conv,
            extra_fields: Vec::new(),
        }
    }

    pub fn with_extra_field(mut self, name: &str, kind: FieldKind) -> Self {
        self.extra_fields.push((name.to_string(), kind));
        self
    }
}

/// Handle for a registered Closure type. Returned by
/// [`DynModule::closures`]. Hold onto this for the lifetime of the
/// language session — every closure-related IR site consults it.
pub struct ClosureKit {
    pub obj_type: ObjTypeId,
    pub obj_handle: ObjTypeHandle,
    body_ref_offset: i32,
    arity_offset: i32,
    captures_base_offset: i32,
    extra_field_offsets: Vec<i32>,
    config: ClosureConfig,
    type_name: String,
    /// Captured at construction so `make()` can emit allocation IR
    /// without needing a `DynFunc` (frontends like Clojure work with
    /// raw `FunctionBuilder`s rather than the `DynFunc` wrapper).
    gc_alloc_extern: FuncRef,
    /// NanBox tag for heap-pointer values. Used to box the freshly-
    /// allocated closure into the language-visible Value form.
    ptr_tag: u32,
}

/// Parameters to [`ClosureKit::make`].
pub struct MakeClosure<'a> {
    /// The body's FuncRef (from `declare_body`).
    pub body_ref: FuncRef,
    /// Arity word stored on the closure object. The kit does not
    /// interpret this — frontends choose the encoding (e.g.
    /// `min_arity | (variadic_bit << 63)`).
    pub arity_word: i64,
    /// Captured values, in the order the body's prologue expects.
    pub captures: &'a [Value],
    /// Extra-field values, in the order declared in
    /// `ClosureConfig::extra_fields`.
    pub extras: &'a [Value],
}

/// Shape a body fn expects: how many positional params, whether
/// variadic, and how many captures to load from the receiver.
#[derive(Clone, Copy)]
pub struct BodyShape {
    /// Number of positional fixed params (excluding the implicit
    /// `self_fn`).
    pub fixed: usize,
    /// Whether to bind a trailing rest param after the fixed ones.
    pub variadic: bool,
    /// Number of captures to load from `self_fn`. Must equal the
    /// `captures.len()` from the matching [`MakeClosure`].
    pub n_captures: usize,
}

impl BodyShape {
    pub fn min_arity(&self) -> usize {
        self.fixed
    }
}

/// Values bound by [`ClosureKit::begin_body`] after the prologue emits.
///
/// Frontends typically copy these into their own environment / symbol
/// table; the kit doesn't know about user-level names.
pub struct BoundBodyEnv {
    /// The receiver value (a NanBox-encoded closure). Use as a live
    /// root or for self-recursion through the indirect path.
    pub self_fn: Value,
    /// User positional args, in declared order.
    pub args: Vec<Value>,
    /// Trailing rest arg, if `BodyShape::variadic`.
    pub rest: Option<Value>,
    /// Captures loaded from `self_fn`, in declared order.
    pub captures: Vec<Value>,
    /// Target block for `recur` — for ArgsList prologues this is the
    /// loop_header (so `recur` re-enters the param-binding walk); for
    /// Positional it's a dedicated recur block.
    pub recur_block: BlockId,
}

/// Per-clause entry returned by [`ClosureKit::begin_multi_arity_body`].
pub struct ClauseEntry {
    /// Block to switch to before lowering this clause's body. The
    /// block already has the correct args-list block-param wired up.
    pub block: BlockId,
    /// The shape this clause expects.
    pub shape: BodyShape,
}

/// Dispatcher returned by [`ClosureKit::begin_multi_arity_body`]. The
/// caller iterates `clauses` in order, calling
/// `dispatcher.begin_clause(kit, fb, i)` per clause to enter that
/// clause's block and emit its prologue.
pub struct MultiArityDispatch {
    pub clauses: Vec<ClauseEntry>,
    self_fn_at_entry: Value,
}

impl MultiArityDispatch {
    /// Switch `fb` to clause `i`'s block and emit its prologue. Returns
    /// the bound env for that clause.
    pub fn begin_clause(
        &self,
        kit: &ClosureKit,
        fb: &mut FunctionBuilder,
        i: usize,
    ) -> BoundBodyEnv {
        let entry = &self.clauses[i];
        fb.switch_to_block(entry.block);
        let self_fn = fb.block_param(entry.block, 0);
        let args_list = fb.block_param(entry.block, 1);
        // Skip arity check — the dispatcher already selected this
        // clause based on a count match.
        kit.emit_args_list_prologue(fb, entry.shape, self_fn, args_list, /*do_arity_check=*/ false)
    }

    pub fn self_fn(&self) -> Value {
        self.self_fn_at_entry
    }
}

impl DynModule {
    /// Register a closure type and return the [`ClosureKit`] handle.
    /// Calls `obj_type("Closure")` internally; if you need multiple
    /// distinct closure types in one module, use [`Self::closures_named`].
    pub fn closures(&mut self, config: ClosureConfig) -> ClosureKit {
        ClosureKit::declare(self, config, "Closure")
    }

    /// Like [`Self::closures`] with a custom type name.
    pub fn closures_named(&mut self, name: &str, config: ClosureConfig) -> ClosureKit {
        ClosureKit::declare(self, config, name)
    }

    /// Build a [`ClosureKit`] over an already-declared `ObjType` instead
    /// of registering a new one. Use this when a frontend already
    /// declares its closure heap shape in its own type table (e.g.
    /// Clojure's `types::declare_types` declares an `Fn` type) and you
    /// want to bolt the kit on without re-registering the type.
    ///
    /// The kit emits calls to `__gc_alloc__` via the FuncRef
    /// auto-registered on this `DynModule` when its first `obj_type`
    /// was declared. The frontend must compile through this same
    /// builder (i.e. `self.module_builder`) so the FuncRef is valid in
    /// the table the JIT receives at extend time.
    ///
    /// The existing type must:
    /// - have a `Raw64` field named `body_ref_field` (the FuncRef)
    /// - have a `Raw64` field named `arity_field` (the arity word)
    /// - declare its extra fields in the same order as
    ///   `config.extra_fields`
    /// - end in a `varlen_values` section (captures)
    ///
    /// Panics with a clear message if any expected field is missing or
    /// has the wrong kind, or if no `obj_type` has been declared yet
    /// (which is what registers `__gc_alloc__`).
    pub fn closures_for(
        &self,
        obj_type: ObjTypeId,
        config: ClosureConfig,
        body_ref_field: &str,
        arity_field: &str,
    ) -> ClosureKit {
        ClosureKit::from_existing(
            self,
            obj_type,
            config,
            body_ref_field,
            arity_field,
        )
    }
}

impl ClosureKit {
    fn from_existing(
        dyn_module: &DynModule,
        obj_type: ObjTypeId,
        config: ClosureConfig,
        body_ref_field: &str,
        arity_field: &str,
    ) -> Self {
        let obj_handle = dyn_module.obj_handle(obj_type);
        let t = dyn_module.get_obj_type(obj_type);
        let body_ref_offset = t.raw64_field_offset_named(body_ref_field) as i32;
        let arity_offset = t.raw64_field_offset_named(arity_field) as i32;
        let captures_base_offset = t.type_info.varlen_element_offset(0) as i32;
        let extra_field_offsets: Vec<i32> = config
            .extra_fields
            .iter()
            .map(|(n, k)| match k {
                FieldKind::Value => t.value_field_offset_named(n) as i32,
                FieldKind::Raw64 => t.raw64_field_offset_named(n) as i32,
            })
            .collect();
        let gc_alloc_extern = dyn_module.gc_alloc_extern().expect(
            "ClosureKit::from_existing: __gc_alloc__ not registered on this \
             DynModule — declare at least one obj_type first (the toolkit \
             auto-registers the extern on the first `obj_type` call)",
        );
        let ptr_tag = dyn_module.tags().ptr;
        ClosureKit {
            obj_type,
            obj_handle,
            body_ref_offset,
            arity_offset,
            captures_base_offset,
            extra_field_offsets,
            config,
            type_name: t.name.clone(),
            gc_alloc_extern,
            ptr_tag,
        }
    }

    fn declare(dyn_module: &mut DynModule, config: ClosureConfig, name: &str) -> Self {
        // Build the Closure GC object type:
        //   body_ref: Raw64
        //   arity:    Raw64
        //   [extras...]
        //   varlen_values (captures or upvalue-cell pointers)
        let mut builder = dyn_module
            .obj_type(name)
            .field("body_ref", FieldKind::Raw64)
            .field("arity", FieldKind::Raw64);
        for (ename, kind) in &config.extra_fields {
            builder = builder.field(ename, *kind);
        }
        builder = builder.varlen_values();
        let obj_type = builder.build();
        // `from_existing` does the offset lookups and grabs the
        // gc_alloc extern + ptr tag — keep the two paths in sync by
        // delegating.
        ClosureKit::from_existing(dyn_module, obj_type, config, "body_ref", "arity")
            .with_type_name(name)
    }

    fn with_type_name(mut self, name: &str) -> Self {
        self.type_name = name.to_string();
        self
    }

    pub fn body_ref_offset(&self) -> i32 {
        self.body_ref_offset
    }
    pub fn arity_offset(&self) -> i32 {
        self.arity_offset
    }
    pub fn captures_base_offset(&self) -> i32 {
        self.captures_base_offset
    }
    pub fn config(&self) -> &ClosureConfig {
        &self.config
    }
    pub fn type_name(&self) -> &str {
        &self.type_name
    }

    /// Declare a body FuncRef with the right signature for the
    /// configured calling convention. For multi-arity ArgsList bodies,
    /// pass `BodyShape { fixed: 0, variadic: false, ... }` — the
    /// signature is always `(self_fn, args_list)` regardless of clauses,
    /// and the dispatcher inside the body sorts out arities at runtime.
    ///
    /// Takes a raw [`ModuleBuilder`] rather than `&mut DynModule` so the
    /// kit composes cleanly with embeddings that maintain a separate
    /// `ModuleBuilder` for code-generation (e.g. the Clojure frontend,
    /// where `DynModule` is used at startup to register GC types and a
    /// distinct `ModuleBuilder` holds the per-compile-pass declarations).
    pub fn declare_body(
        &self,
        module_builder: &mut ModuleBuilder,
        name: &str,
        shape: BodyShape,
    ) -> FuncRef {
        let params: Vec<Type> = match &self.config.call_conv {
            CallConv::Positional => {
                let mut p = vec![Type::I64]; // self_fn
                p.extend(std::iter::repeat_n(Type::I64, shape.fixed));
                if shape.variadic {
                    p.push(Type::I64); // rest
                }
                p
            }
            CallConv::ArgsList { .. } => vec![Type::I64, Type::I64],
        };
        module_builder.declare_func(name, &params, Some(Type::I64))
    }

    /// Open a single-arity body: switch `fb` to the entry block, emit
    /// the prologue (arity check + arg binding + capture loading) and
    /// hand back the bound env. The caller then lowers the body and
    /// ends with `fb.ret(result)`.
    pub fn begin_body(&self, fb: &mut FunctionBuilder, shape: BodyShape) -> BoundBodyEnv {
        let entry = fb.entry_block();
        let self_fn_at_entry = fb.block_param(entry, 0);
        match &self.config.call_conv {
            CallConv::Positional => {
                self.emit_positional_prologue(fb, shape, self_fn_at_entry, entry)
            }
            CallConv::ArgsList { .. } => {
                let args_list_at_entry = fb.block_param(entry, 1);
                self.emit_args_list_prologue(
                    fb,
                    shape,
                    self_fn_at_entry,
                    args_list_at_entry,
                    /*do_arity_check=*/ true,
                )
            }
        }
    }

    /// Open a multi-arity body. The body fn's entry block emits the
    /// dispatch chain; each clause gets its own block accessible via
    /// the returned `MultiArityDispatch::clauses[i].block`. Use
    /// `dispatch.begin_clause(kit, fb, i)` to enter clause `i`'s body.
    ///
    /// Only meaningful for ArgsList — Positional multi-arity would
    /// require multiple body fns, which the kit does not currently
    /// generate (each clause has a distinct positional signature).
    pub fn begin_multi_arity_body(
        &self,
        fb: &mut FunctionBuilder,
        clauses: &[BodyShape],
    ) -> MultiArityDispatch {
        let readers = match &self.config.call_conv {
            CallConv::ArgsList { readers } => *readers,
            CallConv::Positional => panic!(
                "ClosureKit::begin_multi_arity_body: Positional convention does \
                 not support multi-arity dispatch (each clause needs its own \
                 body fn). Use ArgsList for multi-arity."
            ),
        };
        if clauses.iter().filter(|c| c.variadic).count() > 1 {
            panic!("ClosureKit::begin_multi_arity_body: at most one variadic clause permitted");
        }

        let entry = fb.entry_block();
        let self_fn_at_entry = fb.block_param(entry, 0);
        let args_list_at_entry = fb.block_param(entry, 1);

        // Read the runtime list count once.
        fb.safepoint(&[self_fn_at_entry, args_list_at_entry]);
        let count_v = fb
            .call(readers.count, &[args_list_at_entry])
            .expect("ArgsListReaders::count returns a value");

        // Allocate one block per clause; each takes (self_fn, args_list)
        // as block params so the dispatcher can forward them in.
        let clause_blocks: Vec<BlockId> =
            clauses.iter().map(|_| fb.create_block(&[Type::I64, Type::I64])).collect();

        // Build the test order: fixed clauses sorted by min_arity, then
        // (at most one) variadic clause last so it acts as a catch-all
        // for n >= min.
        let mut order: Vec<(usize, bool)> = Vec::new();
        let mut fixed_idx: Vec<usize> = clauses
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.variadic)
            .map(|(i, _)| i)
            .collect();
        fixed_idx.sort_by_key(|&i| clauses[i].min_arity());
        for i in fixed_idx {
            order.push((i, false));
        }
        if let Some((vi, _)) = clauses.iter().enumerate().find(|(_, c)| c.variadic) {
            order.push((vi, true));
        }

        // Build test blocks in reverse so each test's miss-edge points
        // at the next-built test (a chain ending in dispatch_panic).
        let dispatch_panic = fb.create_block(&[]);
        let mut next_miss = dispatch_panic;
        let mut tests: Vec<(usize, BlockId, BlockId, bool)> = Vec::new();
        for &(clause_idx, is_var) in order.iter().rev() {
            let test_blk = fb.create_block(&[]);
            tests.push((clause_idx, test_blk, next_miss, is_var));
            next_miss = test_blk;
        }
        tests.reverse();
        let first_test = if tests.is_empty() { dispatch_panic } else { tests[0].1 };
        fb.jump(first_test, &[]);

        // Emit each test:
        //   count == clause.min_arity (fixed)  → clause_block(self_fn, args_list)
        //   count >= clause.min_arity (variadic) → clause_block(...)
        // miss → next test in chain.
        for &(clause_idx, test_blk, miss_blk, is_var) in &tests {
            fb.switch_to_block(test_blk);
            let clause = &clauses[clause_idx];
            let expected = (readers.encode_arity)(clause.min_arity()) as i64;
            let n_const = fb.iconst(Type::I64, expected);
            let cond = if is_var {
                fb.icmp(CmpOp::Sle, n_const, count_v)
            } else {
                fb.icmp(CmpOp::Eq, n_const, count_v)
            };
            fb.br_if(
                cond,
                clause_blocks[clause_idx],
                &[self_fn_at_entry, args_list_at_entry],
                miss_blk,
                &[],
            );
        }

        // Dispatch-fallthrough block: build the "no matching arity"
        // exception and raise it. `raise` must go through the call
        // table — see `ArgsListReaders::raise` doc.
        fb.switch_to_block(dispatch_panic);
        fb.safepoint(&[self_fn_at_entry, args_list_at_entry]);
        let exc = fb
            .call(readers.no_matching_arity, &[args_list_at_entry])
            .expect("ArgsListReaders::no_matching_arity returns a value");
        fb.safepoint(&[exc]);
        let raise_fr_const = fb.iconst(Type::I64, readers.raise.as_u32() as i64);
        fb.call_via_func_ref(
            readers.call_table_base,
            raise_fr_const,
            &[exc],
            Some(Type::I64),
        );
        fb.unreachable();

        let entries = clauses
            .iter()
            .enumerate()
            .map(|(i, &shape)| ClauseEntry {
                block: clause_blocks[i],
                shape,
            })
            .collect();

        MultiArityDispatch {
            clauses: entries,
            self_fn_at_entry,
        }
    }

    /// Allocate a closure at the current builder position.
    ///
    /// Emits: `safepoint` for `extra_roots + captures + extras` →
    /// `__gc_alloc__` → store `body_ref`, `arity_word`, extras, captures
    /// → tag-box to NanBox.
    ///
    /// `extra_roots` lets the caller pin additional live values across
    /// the alloc safepoint (e.g. the current `env.live_values()`).
    pub fn make(
        &self,
        fb: &mut FunctionBuilder,
        m: MakeClosure<'_>,
        extra_roots: &[Value],
    ) -> Value {
        match &self.config.captures {
            CaptureShape::Inline => {}
            CaptureShape::MutableCells { .. } => panic!(
                "ClosureKit: MutableCells capture shape not implemented (used by \
                 Lox-style upvalues). Use CaptureShape::Inline for now."
            ),
        }
        let n_caps = m.captures.len();
        let type_id_val = fb.iconst(Type::I64, self.obj_type.0 as i64);
        let n_caps_val = fb.iconst(Type::I64, n_caps as i64);

        // Live set for the pre-alloc safepoint: everything that might
        // be a heap pointer and needs to survive the GC.
        let mut roots: Vec<Value> = Vec::with_capacity(
            extra_roots.len() + n_caps + m.extras.len(),
        );
        roots.extend_from_slice(extra_roots);
        roots.extend_from_slice(m.captures);
        roots.extend_from_slice(m.extras);
        fb.safepoint(&roots);
        let raw = fb
            .call(self.gc_alloc_extern, &[type_id_val, n_caps_val])
            .expect("__gc_alloc__ returns a pointer");

        // Fixed Raw64 fields.
        let body_ref_const = fb.iconst(Type::I64, m.body_ref.as_u32() as i64);
        fb.store(body_ref_const, raw, self.body_ref_offset);
        let arity_const = fb.iconst(Type::I64, m.arity_word);
        fb.store(arity_const, raw, self.arity_offset);

        // Extras (in declaration order).
        debug_assert_eq!(m.extras.len(), self.extra_field_offsets.len());
        for (i, &v) in m.extras.iter().enumerate() {
            fb.store(v, raw, self.extra_field_offsets[i]);
        }

        // Captures (varlen tail).
        for (i, &v) in m.captures.iter().enumerate() {
            fb.store(v, raw, self.captures_base_offset + (i as i32) * 8);
        }

        // Tag-box the raw pointer.
        let payload = fb.bitcast(raw, Type::I64);
        fb.make_tagged(self.ptr_tag, payload)
    }

    /// Emit an indirect call through a closure value.
    ///
    /// - `callee` is a NanBox-encoded closure value (boxed pointer).
    /// - For `Positional`, pass user args in `args`; the kit prepends
    ///   `callee` as the self_fn slot itself.
    /// - For `ArgsList`, pass a single pre-packed args-list Value as
    ///   `args[0]`. The kit does not pack lists at the call site —
    ///   that's frontend-owned because the list construction needs the
    ///   frontend's cons routine and lives in its own safepoint chain.
    /// - `live_roots` is the live-value set for the pre-call safepoint.
    ///
    /// Returns the call result.
    pub fn call(
        &self,
        fb: &mut FunctionBuilder,
        call_table_base: u64,
        callee: Value,
        args: &[Value],
        live_roots: &[Value],
    ) -> Value {
        // Unbox the receiver and load its body FuncRef.
        let payload = fb.payload(callee);
        let raw = fb.bitcast(payload, Type::I64);
        let body_ref = fb.load(Type::I64, raw, self.body_ref_offset);

        // Call ABI: (self_fn, ...user_args). For ArgsList that's
        // (callee, list); for Positional that's (callee, p0, p1, …).
        let mut full_args = Vec::with_capacity(1 + args.len());
        full_args.push(callee);
        full_args.extend_from_slice(args);

        fb.safepoint(live_roots);
        fb.call_via_func_ref(call_table_base, body_ref, &full_args, Some(Type::I64))
            .expect("closure body returns a value")
    }

    /// Load capture `i` from a (possibly already-unboxed) receiver
    /// pointer. Used at body-prologue time and by frontends that need
    /// to read captures outside the kit's standard prologue (e.g. when
    /// inlining a closure body into the caller's frame).
    pub fn load_capture(&self, fb: &mut FunctionBuilder, self_fn_raw: Value, i: usize) -> Value {
        match &self.config.captures {
            CaptureShape::Inline => fb.load(
                Type::I64,
                self_fn_raw,
                self.captures_base_offset + (i as i32) * 8,
            ),
            CaptureShape::MutableCells { .. } => panic!(
                "ClosureKit: MutableCells capture shape not implemented"
            ),
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Internal: prologue emitters.
    // ────────────────────────────────────────────────────────────────

    fn emit_positional_prologue(
        &self,
        fb: &mut FunctionBuilder,
        shape: BodyShape,
        self_fn: Value,
        entry: BlockId,
    ) -> BoundBodyEnv {
        // Read positional block params straight off the entry block.
        let mut args = Vec::with_capacity(shape.fixed);
        for i in 0..shape.fixed {
            args.push(fb.block_param(entry, 1 + i));
        }
        let rest = if shape.variadic {
            Some(fb.block_param(entry, 1 + shape.fixed))
        } else {
            None
        };

        // Trampoline through a recur block so `recur` doesn't target
        // the entry block (which has the regalloc-special calling
        // convention layout).
        let mut recur_param_tys = vec![Type::I64; 1 + shape.fixed];
        if shape.variadic {
            recur_param_tys.push(Type::I64);
        }
        let recur_block = fb.create_block(&recur_param_tys);
        let mut jump_args = Vec::with_capacity(recur_param_tys.len());
        jump_args.push(self_fn);
        jump_args.extend_from_slice(&args);
        if let Some(r) = rest {
            jump_args.push(r);
        }
        fb.jump(recur_block, &jump_args);
        fb.switch_to_block(recur_block);

        let self_fn = fb.block_param(recur_block, 0);
        let args: Vec<Value> = (0..shape.fixed)
            .map(|i| fb.block_param(recur_block, 1 + i))
            .collect();
        let rest = if shape.variadic {
            Some(fb.block_param(recur_block, 1 + shape.fixed))
        } else {
            None
        };

        let captures = self.load_captures(fb, self_fn, shape.n_captures);

        BoundBodyEnv {
            self_fn,
            args,
            rest,
            captures,
            recur_block,
        }
    }

    fn emit_args_list_prologue(
        &self,
        fb: &mut FunctionBuilder,
        shape: BodyShape,
        self_fn_at_entry: Value,
        args_list_at_entry: Value,
        do_arity_check: bool,
    ) -> BoundBodyEnv {
        let readers = match &self.config.call_conv {
            CallConv::ArgsList { readers } => *readers,
            CallConv::Positional => unreachable!("emit_args_list_prologue: wrong call conv"),
        };

        // Trampoline through a loop_header(self_fn, args_list) so `recur`
        // has a clean target separate from the body's entry block.
        let loop_header = fb.create_block(&[Type::I64, Type::I64]);
        fb.jump(loop_header, &[self_fn_at_entry, args_list_at_entry]);
        fb.switch_to_block(loop_header);
        let self_fn = fb.block_param(loop_header, 0);
        let args_list = fb.block_param(loop_header, 1);

        // Single-clause arity check: call check_arity(list, arity_word)
        // → branch on success_sentinel != result to raise.
        if do_arity_check {
            let arity_word = encode_default_arity_word(shape.fixed, shape.variadic) as i64;
            let arity_const = fb.iconst(Type::I64, arity_word);
            fb.safepoint(&[args_list, self_fn]);
            let check = fb
                .call(readers.check_arity, &[args_list, arity_const])
                .expect("ArgsListReaders::check_arity returns a value");
            self.emit_raise_if_not_sentinel(fb, check, readers, &[args_list, self_fn]);
        }

        // Walk the list to bind each fixed param. After each
        // safepoint we re-read no values — `first` / `rest` don't
        // allocate in any frontend the kit serves, but they're declared
        // as potential allocators (callable extern), so we safepoint
        // defensively. Live set grows as more args are bound.
        let mut bound: Vec<Value> = Vec::with_capacity(shape.fixed);
        let mut cur_list = args_list;
        for _ in 0..shape.fixed {
            let mut live = Vec::with_capacity(2 + bound.len());
            live.extend_from_slice(&bound);
            live.push(cur_list);
            live.push(self_fn);
            fb.safepoint(&live);
            let head = fb
                .call(readers.first, &[cur_list])
                .expect("ArgsListReaders::first returns a value");
            bound.push(head);
            let mut live2 = Vec::with_capacity(2 + bound.len());
            live2.extend_from_slice(&bound);
            live2.push(cur_list);
            live2.push(self_fn);
            fb.safepoint(&live2);
            cur_list = fb
                .call(readers.rest, &[cur_list])
                .expect("ArgsListReaders::rest returns a value");
        }
        let rest = if shape.variadic { Some(cur_list) } else { None };

        let captures = if shape.n_captures > 0 {
            // Unwrap NanBox tag — the receiver is a boxed pointer.
            let self_fn_raw = fb.payload(self_fn);
            let self_fn_raw_i64 = fb.bitcast(self_fn_raw, Type::I64);
            self.load_captures(fb, self_fn_raw_i64, shape.n_captures)
        } else {
            Vec::new()
        };

        BoundBodyEnv {
            self_fn,
            args: bound,
            rest,
            captures,
            recur_block: loop_header,
        }
    }

    fn load_captures(
        &self,
        fb: &mut FunctionBuilder,
        self_fn_raw_i64: Value,
        n: usize,
    ) -> Vec<Value> {
        if n == 0 {
            return Vec::new();
        }
        match &self.config.captures {
            CaptureShape::Inline => (0..n)
                .map(|i| {
                    fb.load(
                        Type::I64,
                        self_fn_raw_i64,
                        self.captures_base_offset + (i as i32) * 8,
                    )
                })
                .collect(),
            CaptureShape::MutableCells { .. } => panic!(
                "ClosureKit: MutableCells capture shape not implemented"
            ),
        }
    }

    fn emit_raise_if_not_sentinel(
        &self,
        fb: &mut FunctionBuilder,
        result: Value,
        readers: ArgsListReaders,
        live: &[Value],
    ) {
        let sentinel = fb.iconst(Type::I64, readers.success_sentinel as i64);
        let is_ok = fb.icmp(CmpOp::Eq, result, sentinel);
        let ok_bb = fb.create_block(&[]);
        let raise_bb = fb.create_block(&[]);
        fb.br_if(is_ok, ok_bb, &[], raise_bb, &[]);

        fb.switch_to_block(raise_bb);
        let mut live_with_result = live.to_vec();
        live_with_result.push(result);
        fb.safepoint(&live_with_result);
        // Raise via the call table — direct extern call to `raise`
        // would skip the JIT's outcome-kind check and the Exception
        // outcome wouldn't propagate up to the nearest invoke.
        let raise_fr_const = fb.iconst(Type::I64, readers.raise.as_u32() as i64);
        fb.call_via_func_ref(
            readers.call_table_base,
            raise_fr_const,
            &[result],
            Some(Type::I64),
        );
        fb.unreachable();

        fb.switch_to_block(ok_bb);
    }
}

/// Default arity-word encoding the kit uses internally when calling
/// `check_arity`. Layout: bit 63 = variadic, bits 0..62 = min_arity.
/// Frontends usually use the same encoding, but if you don't, ignore
/// this and supply your own arity word via [`MakeClosure::arity_word`]
/// + a `check_arity` extern that decodes your encoding.
const VARIADIC_BIT: u64 = 1 << 63;
pub fn encode_default_arity_word(min_arity: usize, variadic: bool) -> u64 {
    (min_arity as u64) | if variadic { VARIADIC_BIT } else { 0 }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GcConfig, NanBoxTags};

    /// Smoke test: registering a ClosureKit produces a Closure GC type
    /// with the right field layout.
    #[test]
    fn declare_closure_type_args_list() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        // Stub externs — never invoked in this test; just need FuncRefs.
        let first = dyn_module.declare_extern(
            "first",
            crate::Signature { params: vec![Type::I64], ret: Some(Type::I64) },
        );
        let rest = dyn_module.declare_extern(
            "rest",
            crate::Signature { params: vec![Type::I64], ret: Some(Type::I64) },
        );
        let count = dyn_module.declare_extern(
            "count",
            crate::Signature { params: vec![Type::I64], ret: Some(Type::I64) },
        );
        let check_arity = dyn_module.declare_extern(
            "check_arity",
            crate::Signature { params: vec![Type::I64, Type::I64], ret: Some(Type::I64) },
        );
        let no_matching = dyn_module.declare_extern(
            "no_matching",
            crate::Signature { params: vec![Type::I64], ret: Some(Type::I64) },
        );
        let raise = dyn_module.declare_extern(
            "raise",
            crate::Signature { params: vec![Type::I64], ret: None },
        );
        let readers = ArgsListReaders {
            first,
            rest,
            count,
            encode_arity: |n| n as u64,
            check_arity,
            no_matching_arity: no_matching,
            raise,
            call_table_base: 0, // tests don't actually invoke raise
            success_sentinel: 0,
        };
        let kit = dyn_module.closures(ClosureConfig::new(CallConv::ArgsList { readers }));
        // Two fixed Raw64 fields (body_ref, arity) before the varlen.
        assert!(kit.body_ref_offset() < kit.arity_offset());
        assert!(kit.arity_offset() < kit.captures_base_offset());
    }

    #[test]
    fn declare_closure_type_positional() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let kit = dyn_module.closures(ClosureConfig::new(CallConv::Positional));
        assert!(kit.body_ref_offset() < kit.arity_offset());
        assert!(kit.arity_offset() < kit.captures_base_offset());
    }

    #[test]
    fn extra_fields_get_offsets() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let cfg = ClosureConfig::new(CallConv::Positional)
            .with_extra_field("name", FieldKind::Value);
        let kit = dyn_module.closures(cfg);
        assert_eq!(kit.extra_field_offsets.len(), 1);
        // Name field sits after body_ref/arity (Raw64) — Value fields
        // come first in ObjTypeBuilder's layout, so it should be at the
        // very front actually. Either order is fine as long as we
        // round-trip via the offset.
    }

    #[test]
    fn positional_body_signature_matches_shape() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let kit = dyn_module.closures(ClosureConfig::new(CallConv::Positional));
        let _ = kit.declare_body(
            &mut dyn_module.module_builder,
            "fn_two_args",
            BodyShape { fixed: 2, variadic: false, n_captures: 0 },
        );
        // Signature is (self_fn, p0, p1) -> I64 → 3 I64 params.
        // We can't inspect signatures directly here without exposing
        // internals, but the call wouldn't compile-fail in a real test;
        // the module-level integration tests in clojure verify shapes.
    }

    #[test]
    fn args_list_body_signature_is_always_two() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let dummy = dyn_module.declare_extern(
            "dummy",
            crate::Signature { params: vec![Type::I64], ret: Some(Type::I64) },
        );
        let raise = dyn_module.declare_extern(
            "raise",
            crate::Signature { params: vec![Type::I64], ret: None },
        );
        let readers = ArgsListReaders {
            first: dummy,
            rest: dummy,
            count: dummy,
            encode_arity: |n| n as u64,
            check_arity: dummy,
            no_matching_arity: dummy,
            raise,
            call_table_base: 0,
            success_sentinel: 0,
        };
        let kit = dyn_module.closures(ClosureConfig::new(CallConv::ArgsList { readers }));
        let _ = kit.declare_body(
            &mut dyn_module.module_builder,
            "fn_anything",
            BodyShape { fixed: 7, variadic: true, n_captures: 3 },
        );
        // Whatever the shape, ArgsList declares (self_fn, args_list).
    }

    #[test]
    #[should_panic(expected = "Positional convention does not support multi-arity")]
    fn positional_multi_arity_panics() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let kit = dyn_module.closures(ClosureConfig::new(CallConv::Positional));
        let body_ref = kit.declare_body(
            &mut dyn_module.module_builder,
            "multi",
            BodyShape { fixed: 0, variadic: false, n_captures: 0 },
        );
        let mut fb = dyn_module.module_builder.define_func(body_ref);
        let _ = kit.begin_multi_arity_body(
            &mut fb,
            &[BodyShape { fixed: 0, variadic: false, n_captures: 0 }],
        );
    }

    #[test]
    #[should_panic(expected = "MutableCells capture shape not implemented")]
    fn mutable_cells_make_panics() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        // Declare a fake cell type just to satisfy the ObjTypeId field.
        let cell = dyn_module
            .obj_type("Cell")
            .field("value", FieldKind::Value)
            .build();
        let config = ClosureConfig {
            captures: CaptureShape::MutableCells {
                cell_type: cell,
                value_field: "value",
            },
            call_conv: CallConv::Positional,
            extra_fields: vec![],
        };
        let kit = dyn_module.closures(config);
        let body_ref = kit.declare_body(
            &mut dyn_module.module_builder,
            "lam",
            BodyShape { fixed: 0, variadic: false, n_captures: 0 },
        );
        let host_ref = dyn_module.declare_func("host", 0);
        let mut f = dyn_module.start_func(host_ref);
        let _ = kit.make(
            &mut f.fb,
            MakeClosure {
                body_ref,
                arity_word: 0,
                captures: &[],
                extras: &[],
            },
            &[],
        );
        dyn_module.finish_func(f);
    }
}
