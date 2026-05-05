//! AST → dynir lowering for the beagle-on-toolkit port.
//!
//! Scope grew from the binary_trees subset to also cover the ray_cast_bench
//! benchmark: top-level `let` constants, `let mut` + assignment, `while`,
//! short-circuit `&&`/`||`, array literals, indexing, `push`, and a handful
//! of math/time externs (`cos`, `sin`, `core/time-now`).
//!
//! Anything outside this surface still panics explicitly.

use std::collections::HashMap;

use dynir::{FuncRef, Module, Signature, Type, Value};
use dynlang::{
    DynFunc, DynModule, FieldKind, GcConfig, NanBoxTags, ObjTypeId,
    gc::DynGcRuntime,
    ic::{PropertyIc, PropertyIcRuntime},
    stdlib::IndexedSeq,
};

use crate::ast::{Ast, Condition, Pattern};
use crate::types::{Ty, TypeEnv, analyze_types};

/// Growable pool of string literals. We intern every `Ast::String` and
/// materialise a NanBox with `STRING_TAG` + the intern ID. At runtime,
/// the print extern looks up the ID here.
#[derive(Default)]
pub struct StringPool {
    strings: Vec<String>,
    by_text: HashMap<String, u32>,
}

impl StringPool {
    pub fn add(&mut self, s: String) -> u32 {
        if let Some(&id) = self.by_text.get(&s) {
            return id;
        }
        let id = self.strings.len() as u32;
        self.by_text.insert(s.clone(), id);
        self.strings.push(s);
        id
    }

    pub fn get(&self, id: u32) -> Option<&str> {
        self.strings.get(id as usize).map(|s| s.as_str())
    }
}

pub struct Lowered {
    pub module: Module,
    pub main: FuncRef,
    /// Number of params declared on `fn main`. The runner uses this to
    /// decide whether to pass the synthetic `args` value or not.
    pub main_arity: usize,
    pub strings: StringPool,
    /// Preconfigured GC runtime — already knows the module's tag scheme
    /// and type table. Use it via `DynGcRuntime::compile_jit` / `run_jit`
    /// or `interp_gc_alloc`. Language embedders should never roll their
    /// own `Heap` / `PtrPolicy` / `JitSafepointSession`.
    pub gc: DynGcRuntime,
    /// Property-access inline-cache runtime. Must outlive the JIT —
    /// IR holds the address of its indirection cell. Host calls
    /// `ic.install_thread()` before `run_jit`.
    pub ic: PropertyIcRuntime,
    /// Handle to the synthetic Array obj-type. Carries everything the
    /// host needs to recognize array NanBoxes (`view`) and the lowerer
    /// needs to emit literal/push/get IR. `Copy` — pass by value.
    pub array_seq: IndexedSeq,
    /// Allocator FuncRefs (currently just `__gc_alloc__`). Pass to
    /// `Module::validate_safepoints` at JIT compile time.
    pub allocator_frefs: Vec<FuncRef>,
}

/// Field layout we extract from dynlang up front, so we can do field
/// loads/stores without re-borrowing the DynModule while we build IR.
#[derive(Clone)]
struct StructInfo {
    type_id: ObjTypeId,
    field_offsets: HashMap<String, i32>,
}

/// A function that's safe to inline at every call site.
#[derive(Clone)]
struct InlineableFn {
    params: Vec<String>,
    /// The function body, lowered as a block at each call site. May be
    /// a multi-statement chain (`let` chains followed by a final
    /// expression — last value is the call's result).
    body: Vec<Ast>,
}

/// Tag used by beagle to carry string-pool IDs inline. Tags 0..2 are
/// reserved by NanBoxTags::default (nil/bool/ptr); tag 3 is ours.
pub const STRING_TAG: u32 = 3;

pub fn lower_program(program: &Ast) -> Lowered {
    let elements = match program {
        Ast::Program { elements, .. } => elements,
        _ => panic!("lower_program: expected Program, got {:?}", program),
    };

    let gc_config = GcConfig::generational(2 * 1024 * 1024 * 1024);
    let tags = NanBoxTags::default();
    let mut dm = DynModule::new(gc_config.clone(), tags.clone());

    // Slow paths default to panic stubs from `dynlang::slow_paths`.
    // The binary_trees / ray_cast_bench subset never hits them; future
    // beagle features can override individual ops via `dm.override_extern`.
    dm.register_slow_paths_with_defaults("beagle");

    let print_ref = dm.declare_extern(
        "beagle_print",
        Signature { params: vec![Type::I64], ret: None },
    );
    let println_ref = dm.declare_extern(
        "beagle_println",
        Signature { params: vec![Type::I64], ret: None },
    );

    // Stub stdlib externs for binary_trees: length(v), get(v,i), to-number(v).
    // All return I64 (NanBox), take I64 args. `length` is also used by the
    // ray_cast_bench, where the host extern recognises array NanBoxes and
    // reads their `len` field directly.
    let length_ref = dm.declare_extern(
        "beagle_length",
        Signature { params: vec![Type::I64], ret: Some(Type::I64) },
    );
    let get_ref = dm.declare_extern(
        "beagle_get",
        Signature { params: vec![Type::I64, Type::I64], ret: Some(Type::I64) },
    );
    let to_number_ref = dm.declare_extern(
        "beagle_to_number",
        Signature { params: vec![Type::I64], ret: Some(Type::I64) },
    );

    // Math + clock externs used by ray_cast_bench.
    let cos_ref = dm.declare_extern(
        "beagle_cos",
        Signature { params: vec![Type::I64], ret: Some(Type::I64) },
    );
    let sin_ref = dm.declare_extern(
        "beagle_sin",
        Signature { params: vec![Type::I64], ret: Some(Type::I64) },
    );
    let time_now_ref = dm.declare_extern(
        "beagle_time_now",
        Signature { params: vec![], ret: Some(Type::I64) },
    );

    // Property-access inline cache. dynlang owns the slow-path extern,
    // the IC array, and the class-key encoding — see crates/dynlang/src/ic.rs.
    let mut ic = PropertyIc::new(&mut dm);

    // ── Phase 1a: register the synthetic Array type. ──────────────
    // Single Raw64 field `len` (current logical length, in elements) +
    // a varlen-values backing region whose capacity is fixed at alloc time.
    // `Ast::Array` literals allocate with capacity = literal length;
    // `push` allocates a fresh array of capacity old_len + 1 and copies.
    let array_seq = IndexedSeq::register(&mut dm, "__Array__");

    // Phase 1b: register object types for all `struct` declarations.
    // Field names are allowed to collide across structs — polymorphic
    // property access goes through the inline cache.
    let mut structs: HashMap<String, StructInfo> = HashMap::new();

    for el in elements {
        if let Ast::Struct { name, fields, .. } = el {
            let mut builder = dm.obj_type(name);
            for f in fields {
                let fname = match f {
                    Ast::StructField { name, .. } => name.clone(),
                    _ => panic!("struct body: expected StructField, got {:?}", f),
                };
                builder = builder.field(&fname, FieldKind::Value);
            }
            let id = builder.build();
            let ty = dm.get_obj_type(id);
            let offsets: HashMap<String, i32> = ty
                .field_offsets
                .iter()
                .map(|(k, (off, _kind))| (k.clone(), *off))
                .collect();
            ic.register_type(ty);

            structs.insert(
                name.clone(),
                StructInfo { type_id: id, field_offsets: offsets },
            );
        }
    }

    // ── Phase 2a: collect top-level `let` constants. ─────────────
    // The bench uses these for `SHADOW_FAR` and `PI`. Restricted to literal
    // RHS — that's all we need, and a richer scheme would force us to run
    // a module-init function which the runner doesn't have today.
    let mut globals: HashMap<String, Ast> = HashMap::new();
    for el in elements {
        if let Ast::Let { pattern, value, .. } = el {
            let name = match pattern {
                Pattern::Identifier { name, .. } => name.clone(),
                _ => panic!("top-level let must bind a single identifier, got {:?}", pattern),
            };
            assert!(
                is_const_literal(value),
                "top-level `let {name} = ...` must be a literal (number / bool / null), got {:?}",
                value
            );
            globals.insert(name, (**value).clone());
        }
    }

    // ── Phase 2b: declare every function (enables mutual recursion). ──
    let mut func_refs: HashMap<String, FuncRef> = HashMap::new();
    let mut func_arities: HashMap<String, usize> = HashMap::new();
    for el in elements {
        if let Ast::Function { name, args, .. } = el {
            let fname = name.as_ref().expect("top-level fn must be named").clone();
            let fref = dm.declare_func(&fname, args.len());
            func_refs.insert(fname.clone(), fref);
            func_arities.insert(fname, args.len());
        }
    }

    let main_ref = *func_refs.get("main").expect("program must declare fn main");
    let main_arity = *func_arities.get("main").unwrap();

    // ── Phase 2c: identify inlinable leaf functions. ────────────────
    // Criteria (deliberately strict — soundness > coverage):
    //   • body is a single expression (no top-level `let` chain)
    //   • every param is a plain identifier pattern
    //   • body never references the function itself (no recursion —
    //     inlining a recursive callee would diverge during lowering)
    // The body Ast is cloned into the table so call sites can splice it
    // without re-borrowing `elements`. Cheap — these are tiny.
    // Size budget: total AST node count across the body, summed. Keeps
    // huge functions (think `main`, `parameterizedTree`) from blowing up
    // call-site code size when inlined into hot loops. The threshold is
    // tuned by hand — high enough to swallow `ray_segment_hit`-shaped
    // helpers, low enough to leave the orchestration-style top-level
    // functions out.
    // Set to 0 to fully disable inlining (useful for isolating other
    // optimizations during measurement). Otherwise tuned by hand.
    let inline_budget: usize = std::env::var("BEAGLE_INLINE_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let mut inlinable: HashMap<String, InlineableFn> = HashMap::new();
    for el in elements {
        if let Ast::Function { name, args, body, .. } = el {
            let fname = match name {
                Some(n) => n.clone(),
                None => continue,
            };
            if body.is_empty() {
                continue;
            }
            let params: Option<Vec<String>> = args
                .iter()
                .map(|p| match p {
                    Pattern::Identifier { name, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect();
            let Some(params) = params else { continue };
            // Recursion check: if any sub-expression calls back into this
            // function (directly or via Recurse/TailRecurse), we can't
            // safely inline — expansion would loop forever.
            if body.iter().any(|s| calls_function(s, &fname)) {
                continue;
            }
            // Size guard.
            let size: usize = body.iter().map(node_count).sum();
            if size > inline_budget {
                continue;
            }
            inlinable.insert(
                fname,
                InlineableFn { params, body: body.clone() },
            );
        }
    }
    if std::env::var("BEAGLE_DUMP_INLINE").is_ok() {
        let mut names: Vec<&String> = inlinable.keys().collect();
        names.sort();
        eprintln!("[beagle] inlinable functions: {names:?}");
    }

    let mut strings = StringPool::default();

    // ── Phase 2d: type analysis. ────────────────────────────────────
    // Per-function forward analysis producing let-mut narrowed types.
    // Sound under open-world: each function's facts depend only on its
    // own body. Used by lowering to skip the `dyn_*` tag-test branch
    // when both operands of an arithmetic / comparison op are provably
    // `Number` from local context (literals, locals, arithmetic on
    // them, array literal/push chains).
    let type_info = analyze_types(elements, &globals);
    if std::env::var("BEAGLE_DUMP_TYPES").is_ok() {
        let mut fns: Vec<(&String, &HashMap<String, Ty>)> =
            type_info.let_mut_types.iter().collect();
        fns.sort_by_key(|(n, _)| (*n).clone());
        eprintln!("[beagle] let_mut_types:");
        for (n, m) in fns {
            let mut entries: Vec<(&String, &Ty)> = m.iter().collect();
            entries.sort_by_key(|(k, _)| (*k).clone());
            for (var, t) in entries {
                eprintln!("  {n}.{var:24} -> {t:?}");
            }
        }
    }

    // ── Phase 3: define bodies. ──
    // PropertyIc mints cache slot ids during emit_load; finalize allocates
    // the array once we know the total count. No pre-walk needed.
    for el in elements {
        if let Ast::Function { name, args, body, .. } = el {
            let fname = name.as_ref().unwrap();
            let fref = *func_refs.get(fname).unwrap();
            let mut f = dm.start_func(fref);
            {
                let mut lw = Lowerer {
                    structs: &structs,
                    func_refs: &func_refs,
                    globals: &globals,
                    inlinable: &inlinable,
                    types: TypeEnv::new(&type_info, &globals, fname.clone()),
                    array_seq,
                    print_ref,
                    println_ref,
                    length_ref,
                    get_ref,
                    to_number_ref,
                    cos_ref,
                    sin_ref,
                    time_now_ref,
                    strings: &mut strings,
                    ic: &mut ic,
                    hoisted_len_for: HashMap::new(),
                };
                let entry = f.fb.entry_block();
                for (i, pat) in args.iter().enumerate() {
                    let pname = match pat {
                        Pattern::Identifier { name, .. } => name.clone(),
                        _ => panic!("only identifier params supported, got {:?}", pat),
                    };
                    let v = f.fb.block_param(entry, i);
                    f.def_var(&pname, v);
                }
                let result = lw.lower_block(&mut f, body);
                f.fb.ret(result);
            }
            dm.finish_func(f);
        }
    }

    // Build the GC runtime from the module's obj_types *before* we
    // consume `dm` via `build()`. The runtime carries everything needed
    // to compile and run the module: tag scheme, type table, safepoint
    // handler, __gc_alloc__ thunk.
    let mut gc = DynGcRuntime::new(&gc_config, &tags, &dm.obj_types);
    gc.set_auto_externs(dm.auto_externs.clone());

    let built = dm.build();
    // `built.strings` is dynlang's pool, not ours. Discard it.
    let _ = built.strings;

    // Include the IC slow-path extern in allocator_frefs — it can box a
    // method into a bound closure, which allocates.
    let mut allocator_frefs = built.allocator_frefs;
    allocator_frefs.push(ic.slow_ref());

    Lowered {
        module: built.module,
        main: main_ref,
        main_arity,
        strings,
        gc,
        ic: ic.finalize(),
        array_seq,
        allocator_frefs,
    }
}

struct Lowerer<'a> {
    structs: &'a HashMap<String, StructInfo>,
    func_refs: &'a HashMap<String, FuncRef>,
    globals: &'a HashMap<String, Ast>,
    inlinable: &'a HashMap<String, InlineableFn>,
    /// Per-function type-tracking scope. Pushed/popped in lockstep with
    /// `DynFunc::push_scope` so inlined function frames don't bleed
    /// types into the caller. Also carries a borrow of the
    /// whole-program `TypeInfo` and the `current_fn` name.
    types: TypeEnv<'a>,
    array_seq: IndexedSeq,
    print_ref: FuncRef,
    println_ref: FuncRef,
    length_ref: FuncRef,
    get_ref: FuncRef,
    to_number_ref: FuncRef,
    cos_ref: FuncRef,
    sin_ref: FuncRef,
    time_now_ref: FuncRef,
    strings: &'a mut StringPool,
    /// Property-access inline cache builder. Mints cache slot ids and
    /// emits the guard / fast-load / slow-call IR shape per site.
    ic: &'a mut PropertyIc,
    /// LICM for `length(x)`: when entering a `while` whose body never
    /// reassigns or rebinds `x`, the call is hoisted into a let above
    /// the loop and `x → hoisted_slot_name` is recorded here. While
    /// lowering, `Call("length", [Identifier(x)])` consults this map
    /// and emits a stack-slot load instead of an extern call. The map
    /// is a stack of overrides — when leaving the loop we restore the
    /// previous mapping (saved in `hoist_save`). Slot names are minted
    /// via `DynFunc::fresh_slot_name`.
    hoisted_len_for: HashMap<String, String>,
}

impl<'a> Lowerer<'a> {
    fn lower_block(&mut self, f: &mut DynFunc, body: &[Ast]) -> Value {
        if body.is_empty() {
            return f.nil();
        }
        let mut last = f.nil();
        for expr in body {
            last = self.lower_expr(f, expr);
        }
        last
    }

    fn lower_expr(&mut self, f: &mut DynFunc, ast: &Ast) -> Value {
        match ast {
            // ── Literals ────────────────────────────────────────────
            Ast::IntegerLiteral(n, _) => f.number(*n as f64),
            Ast::FloatLiteral(s, _) => {
                let n: f64 = s.parse().expect("invalid float literal");
                f.number(n)
            }
            Ast::Null(_) => f.nil(),
            Ast::True(_) => f.bool_val(true),
            Ast::False(_) => f.bool_val(false),
            Ast::String(s, _) => {
                // beagle's parser keeps the surrounding quotes in the
                // literal value; strip them.
                let cleaned: String = s.trim_matches('"').to_string();
                let id = self.strings.add(cleaned);
                f.tagged_const(STRING_TAG, id as u64)
            }

            // ── Variables ───────────────────────────────────────────
            Ast::Identifier(name, _) => {
                if let Some(global) = self.globals.get(name).cloned() {
                    self.lower_expr(f, &global)
                } else {
                    f.get_var(name)
                }
            }

            Ast::Let { pattern, value, .. } => {
                let vname = match pattern {
                    Pattern::Identifier { name, .. } => name.clone(),
                    _ => panic!("only simple `let <name> = ...` supported, got {:?}", pattern),
                };
                let ty = self.types.type_of(value);
                let v = self.lower_expr(f, value);
                f.def_var(&vname, v);
                self.types.bind(&vname, ty);
                v
            }
            Ast::LetMut { pattern, value, .. } => {
                let vname = match pattern {
                    Pattern::Identifier { name, .. } => name.clone(),
                    _ => panic!("only simple `let mut <name> = ...` supported, got {:?}", pattern),
                };
                // The pre-pass walked the entire enclosing function body
                // and computed the LUB of (init ∪ every assignment) for
                // each let-mut name. Use that here so reads of the
                // variable downstream see the conservative type — even
                // before the assignments have been lowered.
                let ty = self.types.let_mut_type(&vname, value);
                let v = self.lower_expr(f, value);
                f.def_var(&vname, v);
                self.types.bind(&vname, ty);
                v
            }

            Ast::Assignment { name, value, .. } => {
                let target = match name.as_ref() {
                    Ast::Identifier(n, _) => n.clone(),
                    other => panic!(
                        "assignment LHS must be an identifier; field/index assignment \
                         not supported. Got {:?}",
                        other
                    ),
                };
                // No type update — the let-mut binding's pre-computed
                // LUB already accounts for this assignment's RHS.
                let v = self.lower_expr(f, value);
                f.set_var(&target, v);
                v
            }

            // ── Arithmetic ──────────────────────────────────────────
            // Sound specialization: when forward type-flow proves both
            // operands are `Number`, emit the bitcast→fop→bitcast form
            // (`num_*`) which skips the tag-check branch. Otherwise the
            // full `dyn_*` fast/slow dispatch remains in place.
            Ast::Add { left, right, .. } => {
                let (lh, rh) = (self.types.type_of(left), self.types.type_of(right));
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                f.add(l, (&lh).into(), r, (&rh).into())
            }
            Ast::Sub { left, right, .. } => {
                let (lh, rh) = (self.types.type_of(left), self.types.type_of(right));
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                f.sub(l, (&lh).into(), r, (&rh).into())
            }
            Ast::Mul { left, right, .. } => {
                let (lh, rh) = (self.types.type_of(left), self.types.type_of(right));
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                f.mul(l, (&lh).into(), r, (&rh).into())
            }
            Ast::Div { left, right, .. } => {
                let (lh, rh) = (self.types.type_of(left), self.types.type_of(right));
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                f.div(l, (&lh).into(), r, (&rh).into())
            }

            // ── Comparison ──────────────────────────────────────────
            Ast::Condition { operator, left, right, .. } => {
                let (lh, rh) = (self.types.type_of(left), self.types.type_of(right));
                let (lhint, rhint) = ((&lh).into(), (&rh).into());
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                match operator {
                    Condition::LessThan => f.lt(l, lhint, r, rhint),
                    Condition::GreaterThan => f.gt(l, lhint, r, rhint),
                    Condition::LessThanOrEqual => f.le(l, lhint, r, rhint),
                    Condition::GreaterThanOrEqual => f.ge(l, lhint, r, rhint),
                    Condition::Equal => f.bit_eq(l, r),
                    Condition::NotEqual => {
                        let eq = f.bit_eq(l, r);
                        f.bool_not(eq)
                    }
                }
            }

            // ── Short-circuit boolean ───────────────────────────────
            // `a && b` evaluates b only if a is truthy. We branch on a's
            // truthiness and pipe both candidates into a phi-style merge
            // block. Result is whichever value was selected (raw, not
            // canonicalised to bool — matches typical dynamic-language
            // semantics where `&&` returns the operand).
            Ast::And { left, right, .. } => {
                let l = self.lower_expr(f, left);
                let then_bb = f.fb.create_block(&[]);
                let merge_bb = f.fb.create_block(&[Type::I64]);
                f.br_if_truthy(l, then_bb, &[], merge_bb, &[l]);
                f.fb.switch_to_block(then_bb);
                let r = self.lower_expr(f, right);
                f.fb.jump(merge_bb, &[r]);
                f.fb.switch_to_block(merge_bb);
                f.fb.block_param(merge_bb, 0)
            }
            Ast::Or { left, right, .. } => {
                let l = self.lower_expr(f, left);
                let else_bb = f.fb.create_block(&[]);
                let merge_bb = f.fb.create_block(&[Type::I64]);
                f.br_if_truthy(l, merge_bb, &[l], else_bb, &[]);
                f.fb.switch_to_block(else_bb);
                let r = self.lower_expr(f, right);
                f.fb.jump(merge_bb, &[r]);
                f.fb.switch_to_block(merge_bb);
                f.fb.block_param(merge_bb, 0)
            }
            Ast::Not { expr, .. } => {
                let v = self.lower_expr(f, expr);
                f.bool_not(v)
            }

            // ── Control flow ────────────────────────────────────────
            Ast::If { condition, then, else_, .. } => {
                let c = self.lower_expr(f, condition);
                let then_bb = f.fb.create_block(&[]);
                let else_bb = f.fb.create_block(&[]);
                let merge_bb = f.fb.create_block(&[Type::I64]);

                f.br_if_truthy(c, then_bb, &[], else_bb, &[]);

                f.fb.switch_to_block(then_bb);
                let tv = self.lower_block(f, then);
                f.fb.jump(merge_bb, &[tv]);

                f.fb.switch_to_block(else_bb);
                let ev = if else_.is_empty() {
                    f.nil()
                } else {
                    self.lower_block(f, else_)
                };
                f.fb.jump(merge_bb, &[ev]);

                f.fb.switch_to_block(merge_bb);
                f.fb.block_param(merge_bb, 0)
            }

            Ast::While { condition, body, .. } => {
                // ── LICM for `length(x)` ────────────────────────────
                // Find every Identifier `x` such that the loop has at
                // least one `length(x)` call AND `x` is not reassigned
                // or rebound anywhere inside condition+body. For each,
                // emit one extern call before the loop and stash the
                // result in a fresh local; rewrite later `length(x)`
                // hits inside the loop to a stack-slot load.
                let mut mutated: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                collect_mutated_in(condition, &mut mutated);
                for s in body {
                    collect_mutated_in(s, &mut mutated);
                }

                let mut candidates: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                find_length_of_ident(condition, &mut candidates);
                for s in body {
                    find_length_of_ident(s, &mut candidates);
                }
                // Don't try to hoist a global (we have no var slot for
                // its identifier — `lower_expr(Identifier)` re-emits the
                // constant). Also drop anything that's mutated in the
                // loop, and anything that's already been hoisted by an
                // outer loop (to avoid double-hoisting the same name).
                candidates.retain(|x| {
                    !mutated.contains(x)
                        && !self.globals.contains_key(x)
                        && !self.hoisted_len_for.contains_key(x)
                });

                // Save shadowed entries so we can restore on exit.
                let mut hoist_save: Vec<(String, Option<String>)> = Vec::new();
                for x in &candidates {
                    let arr_v = self.lower_expr(f, &Ast::Identifier(x.clone(), 0));
                    let len_v = f.fb.call(self.length_ref, &[arr_v]).unwrap();
                    let hoist_name = f.fresh_slot_name();
                    f.def_var(&hoist_name, len_v);
                    let prev = self.hoisted_len_for.insert(x.clone(), hoist_name);
                    hoist_save.push((x.clone(), prev));
                }

                let header_bb = f.fb.create_block(&[]);
                let body_bb = f.fb.create_block(&[]);
                let exit_bb = f.fb.create_block(&[]);

                f.fb.jump(header_bb, &[]);
                f.fb.switch_to_block(header_bb);
                let c = self.lower_expr(f, condition);
                f.br_if_truthy(c, body_bb, &[], exit_bb, &[]);

                f.fb.switch_to_block(body_bb);
                let _ = self.lower_block(f, body);
                f.fb.jump(header_bb, &[]);

                f.fb.switch_to_block(exit_bb);

                // Restore the prior shadowing state.
                for (x, prev) in hoist_save {
                    match prev {
                        Some(p) => {
                            self.hoisted_len_for.insert(x, p);
                        }
                        None => {
                            self.hoisted_len_for.remove(&x);
                        }
                    }
                }

                f.nil()
            }

            // ── Function calls ──────────────────────────────────────
            Ast::Call { name, args, .. } => self.lower_call(f, name, args),

            // ── Struct construction ─────────────────────────────────
            Ast::StructCreation { name, fields, spread: None, .. } => {
                let info = self
                    .structs
                    .get(name)
                    .unwrap_or_else(|| panic!("unknown struct `{name}`"))
                    .clone();

                // Evaluate field values first; pin them across the alloc
                // via with_rooted. Per-field offsets are looked up before
                // the closure so the inner store loop has only the data
                // it needs.
                let field_offsets: Vec<i32> = fields
                    .iter()
                    .map(|(fname, _)| {
                        *info
                            .field_offsets
                            .get(fname)
                            .unwrap_or_else(|| panic!("unknown field `{fname}` on `{name}`"))
                    })
                    .collect();
                let field_vals: Vec<Value> = fields
                    .iter()
                    .map(|(_, fexpr)| self.lower_expr(f, fexpr))
                    .collect();

                let type_id = info.type_id;
                f.with_rooted(&field_vals, |f, slots| {
                    let zero = f.fb.iconst(Type::I64, 0);
                    // Empty safepoint before the allocation: dynlower's
                    // batch_lower records ALL stack/spill/callee-save
                    // slots here, and the NanBox PtrPolicy filters out
                    // non-pointer payloads at scan time.
                    f.fb.safepoint(&[]);
                    let raw = f.gc_alloc(type_id, zero);

                    for (slot, offset) in slots.iter().zip(field_offsets.iter()) {
                        let v = slot.get(f);
                        f.fb.store(v, raw, *offset);
                    }

                    f.obj_wrap(raw)
                })
            }

            // ── Array literal ───────────────────────────────────────
            Ast::Array { array, .. } => self.lower_array_literal(f, array),

            // ── Indexing ────────────────────────────────────────────
            Ast::IndexOperator { array, index, .. } => {
                let arr = self.lower_expr(f, array);
                let idx_box = self.lower_expr(f, index);
                self.array_seq.emit_get(f, arr, idx_box)
            }

            // ── Property access ─────────────────────────────────────
            Ast::PropertyAccess { object, property, .. } => {
                let obj_val = self.lower_expr(f, object);
                let fname = match property.as_ref() {
                    Ast::Identifier(n, _) => n.as_str(),
                    other => panic!("property access expects ident, got {:?}", other),
                };
                self.ic.emit_load(f, obj_val, fname)
            }

            // ── Inert at lowering ──────────────────────────────────
            Ast::Namespace { .. }
            | Ast::Struct { .. }
            | Ast::Use { .. }
            | Ast::StructField { .. } => f.nil(),

            other => panic!(
                "beagle lowering: unsupported AST node in `{}`: {:?}",
                self.types.current_fn, other
            ),
        }
    }

    /// Lower a function call, handling builtins and user functions.
    fn lower_call(&mut self, f: &mut DynFunc, name: &str, args: &[Ast]) -> Value {
        // Identity / inline first — these don't take the generic arg-eval path
        // because they have shapes our IR can express directly.
        if name == "push" {
            assert_eq!(args.len(), 2, "push() takes 2 args");
            return self.lower_push(f, &args[0], &args[1]);
        }
        if name == "to-float" {
            assert_eq!(args.len(), 1, "to-float() takes 1 arg");
            // All our numbers are NanBox floats already; conversion is a no-op.
            return self.lower_expr(f, &args[0]);
        }
        // LICM: `length(x)` where x is an Identifier hoisted by an
        // enclosing while loop becomes a slot load instead of an extern.
        if name == "length" && args.len() == 1 {
            if let Ast::Identifier(x, _) = &args[0] {
                if let Some(hoist_name) = self.hoisted_len_for.get(x).cloned() {
                    return f.get_var(&hoist_name);
                }
            }
        }

        let arg_vals: Vec<Value> = args.iter().map(|a| self.lower_expr(f, a)).collect();

        match name {
            "print" => {
                assert_eq!(arg_vals.len(), 1, "print() takes exactly 1 arg");
                f.fb.call(self.print_ref, &arg_vals);
                f.nil()
            }
            "println" => {
                assert_eq!(arg_vals.len(), 1, "println() takes exactly 1 arg");
                f.fb.call(self.println_ref, &arg_vals);
                f.nil()
            }
            "length" => f.fb.call(self.length_ref, &arg_vals).unwrap(),
            "get" => f.fb.call(self.get_ref, &arg_vals).unwrap(),
            "to-number" => f.fb.call(self.to_number_ref, &arg_vals).unwrap(),
            "cos" => f.fb.call(self.cos_ref, &arg_vals).unwrap(),
            "sin" => f.fb.call(self.sin_ref, &arg_vals).unwrap(),
            "core/time-now" => {
                assert!(arg_vals.is_empty(), "core/time-now() takes no args");
                f.fb.call(self.time_now_ref, &[]).unwrap()
            }
            // The beagle parser lowers `==` to a call to `beagle.core/equal`.
            // For our MVP that's identity (bit) equality — correct for
            // NanBox-encoded numbers, nils, and pointer-tagged objects.
            "beagle.core/equal" => {
                assert_eq!(arg_vals.len(), 2, "beagle.core/equal takes 2 args");
                f.bit_eq(arg_vals[0], arg_vals[1])
            }
            other => {
                // Static inlining: if the callee is a single-expression
                // leaf we identified during phase 2c, splice its body in
                // place of the call. Soundly substitutes args for params
                // via a pushed scope; nested inlining works naturally
                // because `lower_expr` recurses.
                if let Some(inlinee) = self.inlinable.get(other).cloned() {
                    assert_eq!(
                        inlinee.params.len(),
                        arg_vals.len(),
                        "arity mismatch inlining `{other}`: expected {} args, got {}",
                        inlinee.params.len(),
                        arg_vals.len(),
                    );
                    // Compute arg types BEFORE pushing the scope — they
                    // resolve in the caller's env. Inside the inlinee
                    // they bind to the params' names.
                    let arg_types: Vec<Ty> = args.iter().map(|a| self.types.type_of(a)).collect();
                    f.push_scope();
                    self.types.push_scope();
                    for ((p, v), t) in inlinee
                        .params
                        .iter()
                        .zip(arg_vals.iter())
                        .zip(arg_types.iter())
                    {
                        f.def_var(p, *v);
                        self.types.bind(p, t.clone());
                    }
                    let result = self.lower_block(f, &inlinee.body);
                    self.types.pop_scope();
                    f.pop_scope();
                    return result;
                }

                let fref = *self.func_refs.get(other).unwrap_or_else(|| {
                    panic!("unknown function `{other}` called from `{}`", self.types.current_fn)
                });
                f.fb.call(fref, &arg_vals).unwrap()
            }
        }
    }

    /// Lower `[a, b, c, ...]`. Allocates an `__Array__` of capacity = N,
    /// stores `len = N`, and writes each element. Each element value is
    /// pinned into a stack slot before the allocation so a GC firing
    /// during alloc preserves it (the slots are GC roots and the moving
    /// collector updates them in place).
    fn lower_array_literal(&mut self, f: &mut DynFunc, elems: &[Ast]) -> Value {
        let elem_vals: Vec<Value> =
            elems.iter().map(|e| self.lower_expr(f, e)).collect();
        self.array_seq.emit_literal(f, &elem_vals)
    }

    fn lower_push(&mut self, f: &mut DynFunc, src_ast: &Ast, val_ast: &Ast) -> Value {
        let src_arr = self.lower_expr(f, src_ast);
        let val = self.lower_expr(f, val_ast);
        self.array_seq.emit_push(f, src_arr, val)
    }
}

/// Collect every identifier name that the loop body could rebind or
/// reassign — the conservative "don't hoist" set for length-LICM.
/// Includes both `x = ...` (Assignment) and `let [mut] x = ...`
/// (Let / LetMut) because the latter shadows for the rest of the scope,
/// which is observable by re-evaluations of the loop condition.
fn collect_mutated_in(ast: &Ast, out: &mut std::collections::HashSet<String>) {
    match ast {
        Ast::Assignment { name, value, .. } => {
            if let Ast::Identifier(n, _) = name.as_ref() {
                out.insert(n.clone());
            }
            collect_mutated_in(value, out);
        }
        Ast::Let { pattern, value, .. } | Ast::LetMut { pattern, value, .. } => {
            if let Pattern::Identifier { name, .. } = pattern {
                out.insert(name.clone());
            }
            collect_mutated_in(value, out);
        }

        // Recurse into every form that holds child expressions.
        Ast::Program { elements, .. } => {
            for x in elements {
                collect_mutated_in(x, out);
            }
        }
        Ast::Function { body, .. }
        | Ast::Loop { body, .. }
        | Ast::Reset { body, .. }
        | Ast::Shift { body, .. }
        | Ast::Test { body, .. } => {
            for x in body {
                collect_mutated_in(x, out);
            }
        }
        Ast::If { condition, then, else_, .. } => {
            collect_mutated_in(condition, out);
            for x in then {
                collect_mutated_in(x, out);
            }
            for x in else_ {
                collect_mutated_in(x, out);
            }
        }
        Ast::While { condition, body, .. } => {
            collect_mutated_in(condition, out);
            for x in body {
                collect_mutated_in(x, out);
            }
        }
        Ast::For { collection, body, .. } => {
            collect_mutated_in(collection, out);
            for x in body {
                collect_mutated_in(x, out);
            }
        }
        Ast::Condition { left, right, .. }
        | Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. }
        | Ast::ShiftLeft { left, right, .. }
        | Ast::ShiftRight { left, right, .. }
        | Ast::ShiftRightZero { left, right, .. }
        | Ast::BitWiseAnd { left, right, .. }
        | Ast::BitWiseOr { left, right, .. }
        | Ast::BitWiseXor { left, right, .. }
        | Ast::And { left, right, .. }
        | Ast::Or { left, right, .. } => {
            collect_mutated_in(left, out);
            collect_mutated_in(right, out);
        }
        Ast::Call { args, .. } | Ast::Recurse { args, .. } | Ast::TailRecurse { args, .. } => {
            for a in args {
                collect_mutated_in(a, out);
            }
        }
        Ast::CallExpr { callee, args, .. } => {
            collect_mutated_in(callee, out);
            for a in args {
                collect_mutated_in(a, out);
            }
        }
        Ast::LetDynamic { value, .. } => collect_mutated_in(value, out),
        Ast::Binding { value_expr, body, .. } => {
            collect_mutated_in(value_expr, out);
            for x in body {
                collect_mutated_in(x, out);
            }
        }
        Ast::Not { expr, .. } => collect_mutated_in(expr, out),
        Ast::PropertyAccess { object, property, .. } => {
            collect_mutated_in(object, out);
            collect_mutated_in(property, out);
        }
        Ast::IndexOperator { array, index, .. } => {
            collect_mutated_in(array, out);
            collect_mutated_in(index, out);
        }
        Ast::Array { array, .. } => {
            for x in array {
                collect_mutated_in(x, out);
            }
        }
        Ast::StructCreation { fields, spread, .. } => {
            for (_, v) in fields {
                collect_mutated_in(v, out);
            }
            if let Some(s) = spread {
                collect_mutated_in(s, out);
            }
        }
        Ast::EnumCreation { fields, .. } => {
            for (_, v) in fields {
                collect_mutated_in(v, out);
            }
        }
        Ast::MapLiteral { pairs, .. } => {
            for (k, v) in pairs {
                collect_mutated_in(k, out);
                collect_mutated_in(v, out);
            }
        }
        Ast::SetLiteral { elements, .. } => {
            for x in elements {
                collect_mutated_in(x, out);
            }
        }
        Ast::Break { value, .. } | Ast::Return { value, .. } | Ast::Throw { value, .. } => {
            collect_mutated_in(value, out)
        }
        Ast::StringInterpolation { parts, .. } => {
            for p in parts {
                if let crate::ast::StringInterpolationPart::Expression(e) = p {
                    collect_mutated_in(e, out);
                }
            }
        }
        Ast::Try { body, catch_body, .. } => {
            for x in body {
                collect_mutated_in(x, out);
            }
            for x in catch_body {
                collect_mutated_in(x, out);
            }
        }
        Ast::Match { value, arms, .. } => {
            collect_mutated_in(value, out);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    collect_mutated_in(g, out);
                }
                for x in &arm.body {
                    collect_mutated_in(x, out);
                }
            }
        }
        Ast::MultiArityFunction { cases, .. } => {
            for c in cases {
                for x in &c.body {
                    collect_mutated_in(x, out);
                }
            }
        }
        Ast::Perform { value, .. } => collect_mutated_in(value, out),
        Ast::Handle { handler_instance, body, .. } => {
            collect_mutated_in(handler_instance, out);
            for x in body {
                collect_mutated_in(x, out);
            }
        }
        Ast::Future { body, .. } => collect_mutated_in(body, out),
        Ast::Use { alias, .. } => collect_mutated_in(alias, out),

        // Leaves and pure declarations — nothing to assign or rebind.
        Ast::Struct { .. }
        | Ast::StructField { .. }
        | Ast::Enum { .. }
        | Ast::EnumVariant { .. }
        | Ast::EnumStaticVariant { .. }
        | Ast::Protocol { .. }
        | Ast::Extend { .. }
        | Ast::FunctionStub { .. }
        | Ast::ProtocolDispatch { .. }
        | Ast::IntegerLiteral(..)
        | Ast::FloatLiteral(..)
        | Ast::Identifier(..)
        | Ast::String(..)
        | Ast::Keyword(..)
        | Ast::True(..)
        | Ast::False(..)
        | Ast::Null(..)
        | Ast::Namespace { .. }
        | Ast::Continue { .. } => {}
    }
}

/// Walk `ast` and add every identifier `x` that appears as the sole
/// argument of `length(x)` to `out`. Other shapes (`length(arr[0])`,
/// `length(s.field)`) are skipped — we only hoist the simplest case.
fn find_length_of_ident(ast: &Ast, out: &mut std::collections::HashSet<String>) {
    match ast {
        Ast::Call { name, args, .. } if name == "length" && args.len() == 1 => {
            if let Ast::Identifier(x, _) = &args[0] {
                out.insert(x.clone());
            }
            // The arg might still contain nested length() calls — fall
            // through to the generic recursion below.
            find_length_of_ident(&args[0], out);
        }
        Ast::Call { args, .. } | Ast::Recurse { args, .. } | Ast::TailRecurse { args, .. } => {
            for a in args {
                find_length_of_ident(a, out);
            }
        }
        Ast::CallExpr { callee, args, .. } => {
            find_length_of_ident(callee, out);
            for a in args {
                find_length_of_ident(a, out);
            }
        }
        Ast::Program { elements, .. } => {
            for x in elements {
                find_length_of_ident(x, out);
            }
        }
        Ast::Function { body, .. }
        | Ast::Loop { body, .. }
        | Ast::Reset { body, .. }
        | Ast::Shift { body, .. }
        | Ast::Test { body, .. } => {
            for x in body {
                find_length_of_ident(x, out);
            }
        }
        Ast::If { condition, then, else_, .. } => {
            find_length_of_ident(condition, out);
            for x in then {
                find_length_of_ident(x, out);
            }
            for x in else_ {
                find_length_of_ident(x, out);
            }
        }
        Ast::While { condition, body, .. } => {
            find_length_of_ident(condition, out);
            for x in body {
                find_length_of_ident(x, out);
            }
        }
        Ast::For { collection, body, .. } => {
            find_length_of_ident(collection, out);
            for x in body {
                find_length_of_ident(x, out);
            }
        }
        Ast::Condition { left, right, .. }
        | Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. }
        | Ast::ShiftLeft { left, right, .. }
        | Ast::ShiftRight { left, right, .. }
        | Ast::ShiftRightZero { left, right, .. }
        | Ast::BitWiseAnd { left, right, .. }
        | Ast::BitWiseOr { left, right, .. }
        | Ast::BitWiseXor { left, right, .. }
        | Ast::And { left, right, .. }
        | Ast::Or { left, right, .. } => {
            find_length_of_ident(left, out);
            find_length_of_ident(right, out);
        }
        Ast::Let { value, .. }
        | Ast::LetMut { value, .. }
        | Ast::LetDynamic { value, .. } => find_length_of_ident(value, out),
        Ast::Binding { value_expr, body, .. } => {
            find_length_of_ident(value_expr, out);
            for x in body {
                find_length_of_ident(x, out);
            }
        }
        Ast::Assignment { name, value, .. } => {
            find_length_of_ident(name, out);
            find_length_of_ident(value, out);
        }
        Ast::Not { expr, .. } => find_length_of_ident(expr, out),
        Ast::PropertyAccess { object, property, .. } => {
            find_length_of_ident(object, out);
            find_length_of_ident(property, out);
        }
        Ast::IndexOperator { array, index, .. } => {
            find_length_of_ident(array, out);
            find_length_of_ident(index, out);
        }
        Ast::Array { array, .. } => {
            for x in array {
                find_length_of_ident(x, out);
            }
        }
        Ast::StructCreation { fields, spread, .. } => {
            for (_, v) in fields {
                find_length_of_ident(v, out);
            }
            if let Some(s) = spread {
                find_length_of_ident(s, out);
            }
        }
        Ast::EnumCreation { fields, .. } => {
            for (_, v) in fields {
                find_length_of_ident(v, out);
            }
        }
        Ast::MapLiteral { pairs, .. } => {
            for (k, v) in pairs {
                find_length_of_ident(k, out);
                find_length_of_ident(v, out);
            }
        }
        Ast::SetLiteral { elements, .. } => {
            for x in elements {
                find_length_of_ident(x, out);
            }
        }
        Ast::Break { value, .. } | Ast::Return { value, .. } | Ast::Throw { value, .. } => {
            find_length_of_ident(value, out)
        }
        Ast::StringInterpolation { parts, .. } => {
            for p in parts {
                if let crate::ast::StringInterpolationPart::Expression(e) = p {
                    find_length_of_ident(e, out);
                }
            }
        }
        Ast::Try { body, catch_body, .. } => {
            for x in body {
                find_length_of_ident(x, out);
            }
            for x in catch_body {
                find_length_of_ident(x, out);
            }
        }
        Ast::Match { value, arms, .. } => {
            find_length_of_ident(value, out);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    find_length_of_ident(g, out);
                }
                for x in &arm.body {
                    find_length_of_ident(x, out);
                }
            }
        }
        Ast::MultiArityFunction { cases, .. } => {
            for c in cases {
                for x in &c.body {
                    find_length_of_ident(x, out);
                }
            }
        }
        Ast::Perform { value, .. } => find_length_of_ident(value, out),
        Ast::Handle { handler_instance, body, .. } => {
            find_length_of_ident(handler_instance, out);
            for x in body {
                find_length_of_ident(x, out);
            }
        }
        Ast::Future { body, .. } => find_length_of_ident(body, out),
        Ast::Use { alias, .. } => find_length_of_ident(alias, out),

        Ast::Struct { .. }
        | Ast::StructField { .. }
        | Ast::Enum { .. }
        | Ast::EnumVariant { .. }
        | Ast::EnumStaticVariant { .. }
        | Ast::Protocol { .. }
        | Ast::Extend { .. }
        | Ast::FunctionStub { .. }
        | Ast::ProtocolDispatch { .. }
        | Ast::IntegerLiteral(..)
        | Ast::FloatLiteral(..)
        | Ast::Identifier(..)
        | Ast::String(..)
        | Ast::Keyword(..)
        | Ast::True(..)
        | Ast::False(..)
        | Ast::Null(..)
        | Ast::Namespace { .. }
        | Ast::Continue { .. } => {}
    }
}

/// Crude AST node count for the inliner's size budget. Counts every
/// reachable `Ast` node (including this one) by 1. Approximate — doesn't
/// distinguish a 5-instruction `Call` from a leaf `Identifier` — but
/// good enough to keep huge functions out of the inline set.
fn node_count(ast: &Ast) -> usize {
    let mut n = 0;
    nc(ast, &mut n);
    n
}

fn nc(ast: &Ast, n: &mut usize) {
    *n += 1;
    match ast {
        Ast::Program { elements, .. } => {
            for x in elements {
                nc(x, n);
            }
        }
        Ast::Function { body, .. }
        | Ast::Loop { body, .. }
        | Ast::Reset { body, .. }
        | Ast::Shift { body, .. }
        | Ast::Test { body, .. } => {
            for x in body {
                nc(x, n);
            }
        }
        Ast::Struct { fields, .. } | Ast::Enum { variants: fields, .. } => {
            for x in fields {
                nc(x, n);
            }
        }
        Ast::Protocol { body, .. } | Ast::Extend { body, .. } => {
            for x in body {
                nc(x, n);
            }
        }
        Ast::EnumVariant { fields, .. } => {
            for x in fields {
                nc(x, n);
            }
        }
        Ast::StructField { default_value, .. } => {
            if let Some(v) = default_value {
                nc(v, n);
            }
        }
        Ast::If { condition, then, else_, .. } => {
            nc(condition, n);
            for x in then {
                nc(x, n);
            }
            for x in else_ {
                nc(x, n);
            }
        }
        Ast::While { condition, body, .. } => {
            nc(condition, n);
            for x in body {
                nc(x, n);
            }
        }
        Ast::For { collection, body, .. } => {
            nc(collection, n);
            for x in body {
                nc(x, n);
            }
        }
        Ast::Condition { left, right, .. }
        | Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. }
        | Ast::ShiftLeft { left, right, .. }
        | Ast::ShiftRight { left, right, .. }
        | Ast::ShiftRightZero { left, right, .. }
        | Ast::BitWiseAnd { left, right, .. }
        | Ast::BitWiseOr { left, right, .. }
        | Ast::BitWiseXor { left, right, .. }
        | Ast::And { left, right, .. }
        | Ast::Or { left, right, .. } => {
            nc(left, n);
            nc(right, n);
        }
        Ast::Call { args, .. } | Ast::Recurse { args, .. } | Ast::TailRecurse { args, .. } => {
            for a in args {
                nc(a, n);
            }
        }
        Ast::CallExpr { callee, args, .. } => {
            nc(callee, n);
            for a in args {
                nc(a, n);
            }
        }
        Ast::Let { value, .. }
        | Ast::LetMut { value, .. }
        | Ast::LetDynamic { value, .. } => nc(value, n),
        Ast::Binding { value_expr, body, .. } => {
            nc(value_expr, n);
            for x in body {
                nc(x, n);
            }
        }
        Ast::Assignment { name, value, .. } => {
            nc(name, n);
            nc(value, n);
        }
        Ast::Not { expr, .. } => nc(expr, n),
        Ast::PropertyAccess { object, property, .. } => {
            nc(object, n);
            nc(property, n);
        }
        Ast::IndexOperator { array, index, .. } => {
            nc(array, n);
            nc(index, n);
        }
        Ast::Array { array, .. } => {
            for x in array {
                nc(x, n);
            }
        }
        Ast::StructCreation { fields, spread, .. } => {
            for (_, v) in fields {
                nc(v, n);
            }
            if let Some(s) = spread {
                nc(s, n);
            }
        }
        Ast::EnumCreation { fields, .. } => {
            for (_, v) in fields {
                nc(v, n);
            }
        }
        Ast::Use { alias, .. } => nc(alias, n),
        Ast::MapLiteral { pairs, .. } => {
            for (k, v) in pairs {
                nc(k, n);
                nc(v, n);
            }
        }
        Ast::SetLiteral { elements, .. } => {
            for x in elements {
                nc(x, n);
            }
        }
        Ast::Break { value, .. } | Ast::Return { value, .. } | Ast::Throw { value, .. } => {
            nc(value, n)
        }
        Ast::StringInterpolation { parts, .. } => {
            for p in parts {
                if let crate::ast::StringInterpolationPart::Expression(e) = p {
                    nc(e, n);
                }
            }
        }
        Ast::Try { body, catch_body, .. } => {
            for x in body {
                nc(x, n);
            }
            for x in catch_body {
                nc(x, n);
            }
        }
        Ast::Match { value, arms, .. } => {
            nc(value, n);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    nc(g, n);
                }
                for x in &arm.body {
                    nc(x, n);
                }
            }
        }
        Ast::MultiArityFunction { cases, .. } => {
            for c in cases {
                for x in &c.body {
                    nc(x, n);
                }
            }
        }
        Ast::Perform { value, .. } => nc(value, n),
        Ast::Handle { handler_instance, body, .. } => {
            nc(handler_instance, n);
            for x in body {
                nc(x, n);
            }
        }
        Ast::Future { body, .. } => nc(body, n),

        // Leaves: just the +1 above.
        Ast::FunctionStub { .. }
        | Ast::EnumStaticVariant { .. }
        | Ast::IntegerLiteral(..)
        | Ast::FloatLiteral(..)
        | Ast::Identifier(..)
        | Ast::String(..)
        | Ast::Keyword(..)
        | Ast::True(..)
        | Ast::False(..)
        | Ast::Null(..)
        | Ast::Namespace { .. }
        | Ast::Continue { .. }
        | Ast::ProtocolDispatch { .. } => {}
    }
}

/// True iff any subtree of `ast` is a `Call` (or `TailRecurse`/`Recurse`)
/// that targets `name`. Used to bail out of inlining recursive functions —
/// inlining one would loop forever during lowering.
fn calls_function(ast: &Ast, name: &str) -> bool {
    let mut hit = false;
    calls_in(ast, name, &mut hit);
    hit
}

fn calls_in(ast: &Ast, name: &str, hit: &mut bool) {
    if *hit {
        return;
    }
    match ast {
        Ast::Call { name: n, args, .. } => {
            if n == name {
                *hit = true;
                return;
            }
            for a in args {
                calls_in(a, name, hit);
            }
        }
        Ast::CallExpr { callee, args, .. } => {
            calls_in(callee, name, hit);
            for a in args {
                calls_in(a, name, hit);
            }
        }
        Ast::Recurse { .. } | Ast::TailRecurse { .. } => {
            // These are forms of self-call by definition.
            *hit = true;
        }
        Ast::If { condition, then, else_, .. } => {
            calls_in(condition, name, hit);
            for s in then {
                calls_in(s, name, hit);
            }
            for s in else_ {
                calls_in(s, name, hit);
            }
        }
        Ast::While { condition, body, .. } => {
            calls_in(condition, name, hit);
            for s in body {
                calls_in(s, name, hit);
            }
        }
        Ast::Loop { body, .. } => {
            for s in body {
                calls_in(s, name, hit);
            }
        }
        Ast::For { collection, body, .. } => {
            calls_in(collection, name, hit);
            for s in body {
                calls_in(s, name, hit);
            }
        }
        Ast::Let { value, .. }
        | Ast::LetMut { value, .. }
        | Ast::LetDynamic { value, .. } => calls_in(value, name, hit),
        Ast::Binding { value_expr, body, .. } => {
            calls_in(value_expr, name, hit);
            for s in body {
                calls_in(s, name, hit);
            }
        }
        Ast::Assignment { name: lhs, value, .. } => {
            calls_in(lhs, name, hit);
            calls_in(value, name, hit);
        }
        Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. }
        | Ast::ShiftLeft { left, right, .. }
        | Ast::ShiftRight { left, right, .. }
        | Ast::ShiftRightZero { left, right, .. }
        | Ast::BitWiseAnd { left, right, .. }
        | Ast::BitWiseOr { left, right, .. }
        | Ast::BitWiseXor { left, right, .. }
        | Ast::And { left, right, .. }
        | Ast::Or { left, right, .. }
        | Ast::Condition { left, right, .. } => {
            calls_in(left, name, hit);
            calls_in(right, name, hit);
        }
        Ast::Not { expr, .. } => calls_in(expr, name, hit),
        Ast::PropertyAccess { object, property, .. } => {
            calls_in(object, name, hit);
            calls_in(property, name, hit);
        }
        Ast::IndexOperator { array, index, .. } => {
            calls_in(array, name, hit);
            calls_in(index, name, hit);
        }
        Ast::Array { array, .. } => {
            for x in array {
                calls_in(x, name, hit);
            }
        }
        Ast::StructCreation { fields, spread, .. } => {
            for (_, v) in fields {
                calls_in(v, name, hit);
            }
            if let Some(s) = spread {
                calls_in(s, name, hit);
            }
        }
        Ast::EnumCreation { fields, .. } => {
            for (_, v) in fields {
                calls_in(v, name, hit);
            }
        }
        Ast::Return { value, .. } | Ast::Break { value, .. } | Ast::Throw { value, .. } => {
            calls_in(value, name, hit)
        }
        Ast::MapLiteral { pairs, .. } => {
            for (k, v) in pairs {
                calls_in(k, name, hit);
                calls_in(v, name, hit);
            }
        }
        Ast::SetLiteral { elements, .. } => {
            for x in elements {
                calls_in(x, name, hit);
            }
        }
        Ast::StringInterpolation { parts, .. } => {
            for p in parts {
                if let crate::ast::StringInterpolationPart::Expression(e) = p {
                    calls_in(e, name, hit);
                }
            }
        }
        Ast::Try { body, catch_body, .. } => {
            for s in body {
                calls_in(s, name, hit);
            }
            for s in catch_body {
                calls_in(s, name, hit);
            }
        }
        Ast::Match { value, arms, .. } => {
            calls_in(value, name, hit);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    calls_in(g, name, hit);
                }
                for s in &arm.body {
                    calls_in(s, name, hit);
                }
            }
        }
        // Leaves and forms we don't recurse into for this purpose.
        Ast::IntegerLiteral(..)
        | Ast::FloatLiteral(..)
        | Ast::Identifier(..)
        | Ast::String(..)
        | Ast::Keyword(..)
        | Ast::True(..)
        | Ast::False(..)
        | Ast::Null(..)
        | Ast::Continue { .. }
        | Ast::Namespace { .. }
        | Ast::Use { .. }
        | Ast::Struct { .. }
        | Ast::StructField { .. }
        | Ast::Enum { .. }
        | Ast::EnumVariant { .. }
        | Ast::EnumStaticVariant { .. }
        | Ast::Protocol { .. }
        | Ast::Extend { .. }
        | Ast::FunctionStub { .. }
        | Ast::Function { .. }
        | Ast::MultiArityFunction { .. }
        | Ast::Test { .. }
        | Ast::Reset { .. }
        | Ast::Shift { .. }
        | Ast::Perform { .. }
        | Ast::Handle { .. }
        | Ast::Future { .. }
        | Ast::ProtocolDispatch { .. }
        | Ast::Program { .. } => {}
    }
}

/// True iff `ast` is a constant we can re-emit at every reference (so a
/// top-level `let X = ...` with this RHS doesn't need a module-init pass).
fn is_const_literal(ast: &Ast) -> bool {
    matches!(
        ast,
        Ast::IntegerLiteral(..)
            | Ast::FloatLiteral(..)
            | Ast::True(..)
            | Ast::False(..)
            | Ast::Null(..)
    )
}

