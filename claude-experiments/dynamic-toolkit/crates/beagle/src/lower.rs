//! AST → dynir lowering for the beagle-on-toolkit port.
//!
//! Scope grew from the binary_trees subset to also cover the ray_cast_bench
//! benchmark: top-level `let` constants, `let mut` + assignment, `while`,
//! short-circuit `&&`/`||`, array literals, indexing, `push`, and a handful
//! of math/time externs (`cos`, `sin`, `core/time-now`).
//!
//! Anything outside this surface still panics explicitly.

use std::collections::HashMap;

use dynir::{CmpOp, FuncRef, Module, Signature, Type, Value};
use dynlang::{
    DynFunc, DynModule, FieldKind, GcConfig, NanBoxTags, ObjTypeId, gc::DynGcRuntime,
};
use dynsym::{DispatchTable, InlineCacheArray, InlineCacheEntry, Symbol, SymbolTable};

use crate::ast::{Ast, Condition, Pattern};

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
    /// Inline-cache state for dynamic property access. Must outlive the JIT
    /// module — the JIT holds a raw pointer into `ic.array`. Host binary
    /// installs this into a thread_local and reads it from the slow path.
    pub ic: IcContext,
    /// `u16` type id for the synthetic `__Array__` GC type. The host needs
    /// this so `ext_length` can recognise array NanBoxes and read their
    /// `len` field directly.
    pub array_type_id_u16: u16,
    /// Byte offset of the array's `len` Raw64 field, relative to the
    /// object's raw header pointer. Same value the lowerer baked into IR.
    pub array_len_offset: i32,
}

/// Inline-cache state: interned field names, per-struct offset tables
/// keyed by class key (u16 type_id + 1, stored as u64), and the array of
/// call-site cache slots whose stable base pointer is embedded in JIT code.
///
/// Why the +1: the object header stores a u16 `type_id` at offset 0
/// (`dynobj::Compact`). We load it as u64 and shift by 1 so that
/// `class_key = 0` can be used as the empty-cache sentinel — the first
/// struct declared in a program would otherwise collide with the
/// `InlineCacheEntry::EMPTY` marker and never take the fast path.
pub struct IcContext {
    pub symbols: SymbolTable,
    /// class_key (type_id as u64, +1) → (symbol → field byte offset as u64).
    pub per_type: HashMap<u64, DispatchTable>,
    pub array: InlineCacheArray,
}

/// Field layout we extract from dynlang up front, so we can do field
/// loads/stores without re-borrowing the DynModule while we build IR.
#[derive(Clone)]
struct StructInfo {
    type_id: ObjTypeId,
    field_offsets: HashMap<String, i32>,
}

/// Static type lattice used by the `num_*` specialization analysis.
/// Deliberately minimal — just enough to recognize "definitely a NaN-boxed
/// f64" and pass that knowledge through `let` bindings, struct fields,
/// array elements, and function returns. Anything we can't prove falls
/// back to `Unknown` and the call site emits the safe `dyn_*` form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Ty {
    /// Bottom: no value of this type exists yet. Used for empty array
    /// literals (`[]` is `Array<Bottom>`) and for fresh let-mut bindings
    /// before any assignment has been observed. `Bottom.lub(T) = T` for
    /// any `T`, which lets a `let mut xs = []` accumulate into
    /// `Array<T>` once `push(xs, x: T)` is observed.
    Bottom,
    Number,
    Object(String),
    /// Array with statically-known element type. Only assigned to array
    /// literals whose elements all share a type, and to chains of `push`
    /// that preserve it.
    Array(Box<Ty>),
    Unknown,
}

impl Ty {
    fn is_number(&self) -> bool {
        matches!(self, Ty::Number)
    }

    /// Pointwise least-upper-bound. Bottom is the identity; same-type
    /// joins (incl. matching object/array element) preserve type;
    /// mismatches collapse to `Unknown`.
    fn lub(&self, other: &Ty) -> Ty {
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Ty::Bottom, t) | (t, Ty::Bottom) => t.clone(),
            (Ty::Array(a), Ty::Array(b)) => Ty::Array(Box::new(a.lub(b))),
            _ => Ty::Unknown,
        }
    }
}

/// Whole-program type information produced by `analyze_types` and
/// consumed by `Lowerer` while choosing between `num_*` and `dyn_*`.
#[derive(Default)]
struct TypeInfo {
    /// Top-level function name → inferred return type. Functions in a
    /// recursive cycle (or that we couldn't otherwise analyse) are
    /// absent or `Unknown`.
    fn_returns: HashMap<String, Ty>,
    /// Inferred type of each top-level function's parameters, by index.
    /// Used to seed the per-function var_types env when lowering the
    /// standalone body. Each entry is keyed by function name.
    fn_param_types: HashMap<String, Vec<Ty>>,
    /// Per-struct, per-field LUB of every value ever assigned to that
    /// field at a `StructCreation` site. `(struct_name, field_name)`.
    struct_fields: HashMap<(String, String), Ty>,
    /// Per top-level function: for each let-mut-bound variable in that
    /// function's body, the LUB of (its initializer ∪ every reachable
    /// `Assignment`). This is the type we bind when lowering the
    /// `let mut`. Reads of the variable use this type uniformly so a
    /// later assignment's type can't surprise an earlier read.
    let_mut_types: HashMap<String, HashMap<String, Ty>>,
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

/// Layout for the synthetic `__Array__` GC type. One Raw64 `len` field +
/// `varlen_values` storage. `elem_base` is the byte offset of element 0
/// from the object's raw pointer.
#[derive(Clone, Copy)]
struct ArrayLayout {
    type_id: ObjTypeId,
    len_offset: i32,
    elem_base: i64,
}

/// Tag used by beagle to carry string-pool IDs inline. Tags 0..2 are
/// reserved by NanBoxTags::default (nil/bool/ptr); tag 3 is ours.
pub const STRING_TAG: u32 = 3;

/// Internal name of the synthetic array obj-type. Not user-visible —
/// `Ast::Array` lowers to allocations of this type.
const ARRAY_TYPE_NAME: &str = "__Array__";

pub fn lower_program(program: &Ast) -> Lowered {
    let elements = match program {
        Ast::Program { elements, .. } => elements,
        _ => panic!("lower_program: expected Program, got {:?}", program),
    };

    let gc_config = GcConfig::generational(2 * 1024 * 1024 * 1024);
    let tags = NanBoxTags::default();
    let mut dm = DynModule::new(gc_config.clone(), tags.clone());

    dm.register_slow_paths("beagle");

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

    // Inline-cache slow path for property access. The host binary
    // registers this in `jit_extern_for`. Takes (obj, sym_id, cache_id),
    // fills the IC entry, returns the loaded field.
    let prop_slow_ref = dm.declare_extern(
        "beagle_prop_slow",
        Signature {
            params: vec![Type::I64, Type::I64, Type::I64],
            ret: Some(Type::I64),
        },
    );

    // ── Phase 1a: register the synthetic Array type. ──────────────
    // Single Raw64 field `len` (current logical length, in elements) +
    // a varlen-values backing region whose capacity is fixed at alloc time.
    // `Ast::Array` literals allocate with capacity = literal length;
    // `push` allocates a fresh array of capacity old_len + 1 and copies.
    let array_type_id = dm
        .obj_type(ARRAY_TYPE_NAME)
        .field("len", FieldKind::Raw64)
        .varlen_values()
        .build();

    // Phase 1b: register object types for all `struct` declarations. ──
    // Also build the symbol table + per-type dispatch tables that the IC
    // slow path will consult. Field names are allowed to collide across
    // structs — polymorphic access goes through the inline cache.
    let mut structs: HashMap<String, StructInfo> = HashMap::new();
    let mut symbols = SymbolTable::new();
    let mut per_type: HashMap<u64, DispatchTable> = HashMap::new();

    for el in elements {
        if let Ast::Struct { name, fields, .. } = el {
            let mut builder = dm.obj_type(name);
            let mut field_names: Vec<String> = Vec::new();
            for f in fields {
                let fname = match f {
                    Ast::StructField { name, .. } => name.clone(),
                    _ => panic!("struct body: expected StructField, got {:?}", f),
                };
                builder = builder.field(&fname, FieldKind::Value);
                field_names.push(fname);
            }
            let id = builder.build();
            let ty = dm.get_obj_type(id);
            let offsets: HashMap<String, i32> = ty
                .field_offsets
                .iter()
                .map(|(k, (off, _kind))| (k.clone(), *off))
                .collect();
            // Class key = u16 type_id + 1, so no valid key is 0 (which
            // InlineCacheEntry reserves as "empty"). type_ids are
            // assigned sequentially from 0 by ObjTypeBuilder::build.
            let class_key = (ty.type_info.type_id as u64) + 1;

            let mut table = DispatchTable::with_capacity(0);
            for fname in &field_names {
                let sym = symbols.intern(fname);
                let off = offsets[fname] as u64;
                table.set(sym, off);
            }
            per_type.insert(class_key, table);

            structs.insert(
                name.clone(),
                StructInfo { type_id: id, field_offsets: offsets },
            );
        }
    }

    // Pull the array layout out *now*, while we still have `&dm`.
    let array_layout = {
        let ty = dm.get_obj_type(array_type_id);
        let len_offset = ty
            .field_offsets
            .get("len")
            .map(|(o, _)| *o)
            .expect("__Array__ must have len field");
        let elem_base = ty.type_info.varlen_element_offset(0) as i64;
        ArrayLayout { type_id: array_type_id, len_offset, elem_base }
    };
    let array_type_id_u16 = dm.get_obj_type(array_type_id).type_info.type_id;

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
    // Whole-program forward analysis producing function return types,
    // struct-field types, and per-function let-mut narrowed types. Used
    // by lowering to skip the `dyn_*` tag-test branch when both operands
    // of an arithmetic / comparison op are provably `Number`.
    let type_info = analyze_types(elements, &globals, &inlinable);
    if std::env::var("BEAGLE_DUMP_TYPES").is_ok() {
        let mut fns: Vec<(&String, &Ty)> = type_info.fn_returns.iter().collect();
        fns.sort_by_key(|(n, _)| (*n).clone());
        eprintln!("[beagle] fn_returns:");
        for (n, t) in fns {
            eprintln!("  {n:24} -> {t:?}");
        }
        let mut fields: Vec<(&(String, String), &Ty)> = type_info.struct_fields.iter().collect();
        fields.sort_by_key(|(k, _)| (*k).clone());
        eprintln!("[beagle] struct_fields:");
        for ((s, f), t) in fields {
            eprintln!("  {s}.{f:18} -> {t:?}");
        }
        let mut params: Vec<(&String, &Vec<Ty>)> = type_info.fn_param_types.iter().collect();
        params.sort_by_key(|(n, _)| (*n).clone());
        eprintln!("[beagle] fn_param_types:");
        for (n, ps) in params {
            eprintln!("  {n:24} -> {ps:?}");
        }
    }

    // Pre-walk the program to count property-access sites so we can
    // allocate a fixed-size `InlineCacheArray` whose base pointer we can
    // embed in IR as a constant. Growing the array later would invalidate
    // that pointer.
    //
    // Inlining duplicates property accesses: each call site to an inlinable
    // callee expands into a fresh copy of the callee's body, which gets its
    // own IC slots. The count below walks the program with inline expansion
    // simulated, so the IC array is sized correctly.
    let num_ic_sites = count_property_accesses_with_inlining(program, &inlinable);
    let ic_array = InlineCacheArray::new(num_ic_sites);
    let ic_base_addr = ic_array.as_ptr() as u64;
    let mut next_cache_id: u32 = 0;

    // ── Phase 3: define bodies. ──
    for el in elements {
        if let Ast::Function { name, args, body, .. } = el {
            let fname = name.as_ref().unwrap();
            let fref = *func_refs.get(fname).unwrap();
            let mut f = dm.start_func(fref);
            {
                // Seed the function's var_types with the analyzed
                // parameter types so reads of params during lowering see
                // them as Number where applicable.
                let mut entry_scope: HashMap<String, Ty> = HashMap::new();
                if let Some(param_tys) = type_info.fn_param_types.get(fname) {
                    for (pat, ty) in args.iter().zip(param_tys.iter()) {
                        if let Pattern::Identifier { name, .. } = pat {
                            entry_scope.insert(name.clone(), ty.clone());
                        }
                    }
                }

                let mut lw = Lowerer {
                    structs: &structs,
                    func_refs: &func_refs,
                    globals: &globals,
                    inlinable: &inlinable,
                    types: &type_info,
                    var_types: vec![entry_scope],
                    array: array_layout,
                    print_ref,
                    println_ref,
                    length_ref,
                    get_ref,
                    to_number_ref,
                    cos_ref,
                    sin_ref,
                    time_now_ref,
                    prop_slow_ref,
                    current_fn: fname.clone(),
                    strings: &mut strings,
                    symbols: &mut symbols,
                    ic_base_addr,
                    next_cache_id: &mut next_cache_id,
                    push_counter: 0,
                    arr_lit_counter: 0,
                    hoisted_len_for: HashMap::new(),
                    hoist_counter: 0,
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
    let gc = DynGcRuntime::new(&gc_config, &tags, &dm.obj_types);

    let built = dm.build();
    // `built.strings` is dynlang's pool, not ours. Discard it.
    let _ = built.strings;

    assert_eq!(
        next_cache_id as usize, num_ic_sites,
        "IC site count mismatch: pre-count said {num_ic_sites} but lowering consumed {next_cache_id}",
    );

    Lowered {
        module: built.module,
        main: main_ref,
        main_arity,
        strings,
        gc,
        ic: IcContext {
            symbols,
            per_type,
            array: ic_array,
        },
        array_type_id_u16,
        array_len_offset: array_layout.len_offset,
    }
}

struct Lowerer<'a> {
    structs: &'a HashMap<String, StructInfo>,
    func_refs: &'a HashMap<String, FuncRef>,
    globals: &'a HashMap<String, Ast>,
    inlinable: &'a HashMap<String, InlineableFn>,
    types: &'a TypeInfo,
    /// Parallel scope stack tracking the static type of each in-scope
    /// variable. Pushed/popped in lockstep with `DynFunc::push_scope` so
    /// inlined function frames don't bleed types into the caller. Each
    /// `let` / `let mut` / inlined-arg binding records a type here at
    /// the same point its `def_var` happens.
    var_types: Vec<HashMap<String, Ty>>,
    array: ArrayLayout,
    print_ref: FuncRef,
    println_ref: FuncRef,
    length_ref: FuncRef,
    get_ref: FuncRef,
    to_number_ref: FuncRef,
    cos_ref: FuncRef,
    sin_ref: FuncRef,
    time_now_ref: FuncRef,
    prop_slow_ref: FuncRef,
    current_fn: String,
    strings: &'a mut StringPool,
    symbols: &'a mut SymbolTable,
    /// Base address of the InlineCacheArray, embedded as an IR constant.
    ic_base_addr: u64,
    next_cache_id: &'a mut u32,
    /// Per-function counter used to mint unique stack-slot var names for
    /// nested `push` calls (the slots are GC roots — must not collide).
    push_counter: u32,
    /// Per-function counter used to mint unique stack-slot var names for
    /// each array literal. Without this, a nested literal like
    /// `[[1.0, 2.0], [3.0, 4.0]]` would have its inner literals reuse the
    /// outer's `__beagle_arr_elem_N__` names, silently clobbering the
    /// outer bindings before the outer alloc could read them back.
    arr_lit_counter: u32,
    /// LICM for `length(x)`: when entering a `while` whose body never
    /// reassigns or rebinds `x`, the call is hoisted into a let above
    /// the loop and `x → hoisted_var_name` is recorded here. While
    /// lowering, `Call("length", [Identifier(x)])` consults this map
    /// and emits a stack-slot load instead of an extern call. The map
    /// is a stack of overrides — when leaving the loop we restore the
    /// previous mapping (saved in `hoist_save`).
    hoisted_len_for: HashMap<String, String>,
    hoist_counter: u32,
}

impl<'a> Lowerer<'a> {
    // ── Type tracking helpers ────────────────────────────────────────
    //
    // Maintained in lockstep with `DynFunc`'s scope/var stacks so the
    // type env always matches the variable env.

    fn push_type_scope(&mut self) {
        self.var_types.push(HashMap::new());
    }

    fn pop_type_scope(&mut self) {
        self.var_types.pop();
    }

    fn bind_type(&mut self, name: &str, ty: Ty) {
        if let Some(scope) = self.var_types.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    /// Resolve a let-mut variable's type. Tries the analysis's
    /// per-function table first (which reflects the LUB of init +
    /// every assignment in scope). Falls back to typing the initializer
    /// directly when no analysis entry exists — useful for synthetic
    /// helper let-muts inserted by lowering itself.
    fn let_mut_type(&self, name: &str, init: &Ast) -> Ty {
        if let Some(per_fn) = self.types.let_mut_types.get(&self.current_fn) {
            if let Some(t) = per_fn.get(name) {
                return t.clone();
            }
        }
        self.type_of(init)
    }

    fn lookup_type(&self, name: &str) -> Ty {
        for scope in self.var_types.iter().rev() {
            if let Some(t) = scope.get(name) {
                return t.clone();
            }
        }
        Ty::Unknown
    }

    /// Static type of an arbitrary expression, evaluated against the
    /// current var-types env + the whole-program `TypeInfo`. Used at
    /// arithmetic and comparison call sites to decide between `num_*`
    /// and `dyn_*`. Conservative: when in doubt, returns `Unknown` and
    /// the call site falls back to `dyn_*`.
    fn type_of(&self, ast: &Ast) -> Ty {
        match ast {
            Ast::IntegerLiteral(..) | Ast::FloatLiteral(..) => Ty::Number,
            // Bool / null / strings aren't `Number`; we don't bother
            // distinguishing them further — the consumer only checks
            // `is_number()`.
            Ast::True(..) | Ast::False(..) | Ast::Null(..) | Ast::String(..) => Ty::Unknown,

            Ast::Identifier(name, _) => {
                if let Some(global) = self.globals.get(name) {
                    return self.type_of(global);
                }
                self.lookup_type(name)
            }

            Ast::Add { left, right, .. }
            | Ast::Sub { left, right, .. }
            | Ast::Mul { left, right, .. }
            | Ast::Div { left, right, .. }
            | Ast::Modulo { left, right, .. } => {
                if self.type_of(left).is_number() && self.type_of(right).is_number() {
                    Ty::Number
                } else {
                    Ty::Unknown
                }
            }

            // Comparisons and boolean ops aren't `Number`. We don't need
            // a `Bool` slot in the lattice yet.
            Ast::Condition { .. }
            | Ast::And { .. }
            | Ast::Or { .. }
            | Ast::Not { .. } => Ty::Unknown,

            Ast::If { then, else_, .. } => {
                let tt = then
                    .last()
                    .map(|e| self.type_of(e))
                    .unwrap_or(Ty::Unknown);
                let et = else_
                    .last()
                    .map(|e| self.type_of(e))
                    .unwrap_or(Ty::Unknown);
                tt.lub(&et)
            }

            Ast::Let { value, .. } | Ast::LetMut { value, .. } => self.type_of(value),

            Ast::Assignment { value, .. } => self.type_of(value),

            Ast::StructCreation { name, .. } => Ty::Object(name.clone()),

            Ast::Array { array, .. } => {
                if array.is_empty() {
                    return Ty::Array(Box::new(Ty::Bottom));
                }
                let mut acc = self.type_of(&array[0]);
                for x in &array[1..] {
                    acc = acc.lub(&self.type_of(x));
                }
                Ty::Array(Box::new(acc))
            }

            Ast::IndexOperator { array, .. } => {
                if let Ty::Array(elem) = self.type_of(array) {
                    *elem
                } else {
                    Ty::Unknown
                }
            }

            Ast::PropertyAccess { object, property, .. } => {
                let obj_ty = self.type_of(object);
                if let Ty::Object(struct_name) = obj_ty {
                    if let Ast::Identifier(field, _) = property.as_ref() {
                        if let Some(t) =
                            self.types.struct_fields.get(&(struct_name, field.clone()))
                        {
                            return t.clone();
                        }
                    }
                }
                Ty::Unknown
            }

            Ast::Call { name, args, .. } => {
                // Builtins with hard-coded return types.
                match name.as_str() {
                    "cos" | "sin" | "to-float" | "to-number" | "length"
                    | "core/time-now" => return Ty::Number,
                    "push" => {
                        // `push(arr, x)` returns Array<LUB(prior_elem, type_of(x))>.
                        if args.len() == 2 {
                            let arr_ty = self.type_of(&args[0]);
                            let val_ty = self.type_of(&args[1]);
                            if let Ty::Array(prior) = arr_ty {
                                return Ty::Array(Box::new(prior.lub(&val_ty)));
                            }
                            return Ty::Array(Box::new(val_ty));
                        }
                    }
                    _ => {}
                }
                // User function: consult analysis.
                self.types
                    .fn_returns
                    .get(name)
                    .cloned()
                    .unwrap_or(Ty::Unknown)
            }

            Ast::CallExpr { .. } => Ty::Unknown,

            // Statement-ish or unsupported forms: not a value we care
            // about for arithmetic dispatch.
            _ => Ty::Unknown,
        }
    }

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
                let ty = self.type_of(value);
                let v = self.lower_expr(f, value);
                f.def_var(&vname, v);
                self.bind_type(&vname, ty);
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
                let ty = self.let_mut_type(&vname, value);
                let v = self.lower_expr(f, value);
                f.def_var(&vname, v);
                self.bind_type(&vname, ty);
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
                let lt = self.type_of(left);
                let rt = self.type_of(right);
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                if lt.is_number() && rt.is_number() {
                    f.num_add(l, r)
                } else {
                    f.dyn_add(l, r)
                }
            }
            Ast::Sub { left, right, .. } => {
                let lt = self.type_of(left);
                let rt = self.type_of(right);
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                if lt.is_number() && rt.is_number() {
                    f.num_sub(l, r)
                } else {
                    f.dyn_sub(l, r)
                }
            }
            Ast::Mul { left, right, .. } => {
                let lt = self.type_of(left);
                let rt = self.type_of(right);
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                if lt.is_number() && rt.is_number() {
                    f.num_mul(l, r)
                } else {
                    f.dyn_mul(l, r)
                }
            }
            Ast::Div { left, right, .. } => {
                let lt = self.type_of(left);
                let rt = self.type_of(right);
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                if lt.is_number() && rt.is_number() {
                    f.num_div(l, r)
                } else {
                    f.dyn_div(l, r)
                }
            }

            // ── Comparison ──────────────────────────────────────────
            Ast::Condition { operator, left, right, .. } => {
                let lt = self.type_of(left);
                let rt = self.type_of(right);
                let both_num = lt.is_number() && rt.is_number();
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                match operator {
                    Condition::LessThan => {
                        if both_num { f.num_lt(l, r) } else { f.dyn_lt(l, r) }
                    }
                    Condition::GreaterThan => {
                        if both_num { f.num_gt(l, r) } else { f.dyn_gt(l, r) }
                    }
                    Condition::Equal => self.bit_eq(f, l, r),
                    Condition::NotEqual => {
                        let eq = self.bit_eq(f, l, r);
                        self.bool_not(f, eq)
                    }
                    Condition::LessThanOrEqual => {
                        if both_num {
                            f.num_le(l, r)
                        } else {
                            let gt = f.dyn_gt(l, r);
                            self.bool_not(f, gt)
                        }
                    }
                    Condition::GreaterThanOrEqual => {
                        if both_num {
                            f.num_ge(l, r)
                        } else {
                            let lt_v = f.dyn_lt(l, r);
                            self.bool_not(f, lt_v)
                        }
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
                self.bool_not(f, v)
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
                    let hoist_name =
                        format!("__hlen_{}_{}__", x, self.hoist_counter);
                    self.hoist_counter += 1;
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

                // Evaluate all field values first. We bind each into a
                // fresh `let` so it lives in a stack slot — the
                // interpreter's root manager scans stack slots (via the
                // NanBox PtrPolicy) so pointers survive collection.
                // This sidesteps the safepoint-live-values type
                // restriction (Safepoint requires GcPtr-typed values,
                // but our field values are NanBox I64s).
                let field_vals: Vec<(String, Value)> = fields
                    .iter()
                    .map(|(fname, fexpr)| {
                        let v = self.lower_expr(f, fexpr);
                        // Stash in a stack slot by name-shadowing so it
                        // is visible to frame-scan GC roots.
                        let slot_name = format!("__beagle_tmp_{}__", fname);
                        f.def_var(&slot_name, v);
                        (fname.clone(), f.get_var(&slot_name))
                    })
                    .collect();

                let zero = f.fb.iconst(Type::I64, 0);
                // Emit an empty safepoint right before the allocation. The
                // JIT's batch_lower records ALL stack/spill/callee-save
                // slots here regardless of the `live` list, and the
                // PtrPolicy filters non-pointer NanBox words out during
                // GC. This gives the collector a place to safely run.
                f.fb.safepoint(&[]);
                let raw = f.gc_alloc(info.type_id, zero);

                for (fname, val) in &field_vals {
                    let offset = *info
                        .field_offsets
                        .get(fname)
                        .unwrap_or_else(|| panic!("unknown field `{fname}` on `{name}`"));
                    f.fb.store(*val, raw, offset);
                }

                f.obj_wrap(raw)
            }

            // ── Array literal ───────────────────────────────────────
            Ast::Array { array, .. } => self.lower_array_literal(f, array),

            // ── Indexing ────────────────────────────────────────────
            Ast::IndexOperator { array, index, .. } => {
                let arr = self.lower_expr(f, array);
                let idx_box = self.lower_expr(f, index);
                self.array_load_at(f, arr, idx_box)
            }

            // ── Property access ─────────────────────────────────────
            Ast::PropertyAccess { object, property, .. } => {
                let obj_val = self.lower_expr(f, object);
                let fname = match property.as_ref() {
                    Ast::Identifier(n, _) => n.clone(),
                    other => panic!("property access expects ident, got {:?}", other),
                };
                let sym = self.symbols.intern(&fname);
                let cache_id = *self.next_cache_id;
                *self.next_cache_id += 1;
                self.emit_ic_property_load(f, obj_val, sym, cache_id)
            }

            // ── Inert at lowering ──────────────────────────────────
            Ast::Namespace { .. }
            | Ast::Struct { .. }
            | Ast::Use { .. }
            | Ast::StructField { .. } => f.nil(),

            other => panic!(
                "beagle lowering: unsupported AST node in `{}`: {:?}",
                self.current_fn, other
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
                self.bit_eq(f, arg_vals[0], arg_vals[1])
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
                    let arg_types: Vec<Ty> = args.iter().map(|a| self.type_of(a)).collect();
                    f.push_scope();
                    self.push_type_scope();
                    for ((p, v), t) in inlinee
                        .params
                        .iter()
                        .zip(arg_vals.iter())
                        .zip(arg_types.iter())
                    {
                        f.def_var(p, *v);
                        self.bind_type(p, t.clone());
                    }
                    let result = self.lower_block(f, &inlinee.body);
                    self.pop_type_scope();
                    f.pop_scope();
                    return result;
                }

                let fref = *self.func_refs.get(other).unwrap_or_else(|| {
                    panic!("unknown function `{other}` called from `{}`", self.current_fn)
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
        // Evaluate elements into rooted slots first. Reusing the
        // struct-creation pattern: bind each into a uniquely-named
        // stack slot so its NanBox bits live across the gc_alloc call.
        // The id is per-literal so a nested literal like
        // `[[1.0, 2.0], [3.0, 4.0]]` doesn't have its inner element
        // bindings clobber the outer's by name.
        let id = self.arr_lit_counter;
        self.arr_lit_counter += 1;
        let slot_names: Vec<String> = (0..elems.len())
            .map(|i| format!("__beagle_arr_{id}_elem_{i}__"))
            .collect();
        for (i, elem) in elems.iter().enumerate() {
            let v = self.lower_expr(f, elem);
            f.def_var(&slot_names[i], v);
        }

        let n = elems.len() as i64;
        let n_const = f.fb.iconst(Type::I64, n);
        f.fb.safepoint(&[]);
        let raw = f.gc_alloc(self.array.type_id, n_const);

        // len is stored as a raw i64 in the Raw64 field.
        f.fb.store(n_const, raw, self.array.len_offset);

        // Write each element. After alloc, reload from the slots so the
        // values reflect any forwarding the GC may have done.
        for (i, slot_name) in slot_names.iter().enumerate() {
            let v = f.get_var(slot_name);
            let idx = f.fb.iconst(Type::I64, i as i64);
            self.array_store_elem(f, raw, idx, v);
        }

        f.obj_wrap(raw)
    }

    /// Lower `push(arr, x)`. Allocates a new array of capacity old_len+1,
    /// copies the existing elements over, then appends `x`. The source
    /// array NanBox and the new value are both pinned into stack slots
    /// across the alloc. Element copying happens via a tight `while` loop
    /// in IR.
    fn lower_push(&mut self, f: &mut DynFunc, src_ast: &Ast, val_ast: &Ast) -> Value {
        let id = self.push_counter;
        self.push_counter += 1;
        let src_slot = format!("__beagle_push_src_{id}__");
        let val_slot = format!("__beagle_push_val_{id}__");

        let src_arr = self.lower_expr(f, src_ast);
        let val = self.lower_expr(f, val_ast);
        f.def_var(&src_slot, src_arr);
        f.def_var(&val_slot, val);

        // Read old length from the source array (no alloc has happened yet).
        let src0 = f.get_var(&src_slot);
        let src_raw0 = f.obj_unwrap(src0);
        let old_len = f.fb.load(Type::I64, src_raw0, self.array.len_offset);
        let one = f.fb.iconst(Type::I64, 1);
        let new_len = f.fb.add(old_len, one);

        // Allocate the new array. After this point the source pointer
        // we read above may be stale — reload from the slot.
        f.fb.safepoint(&[]);
        let new_raw = f.gc_alloc(self.array.type_id, new_len);
        f.fb.store(new_len, new_raw, self.array.len_offset);

        // Reload the source after the alloc so we copy from the live
        // (post-forwarding) location.
        let src1 = f.get_var(&src_slot);
        let src_raw1 = f.obj_unwrap(src1);

        // Copy loop: for i in 0..old_len, new[i] = src[i].
        let i_slot = f.fb.create_stack_slot(8, false);
        let zero = f.fb.iconst(Type::I64, 0);
        let i_addr0 = f.fb.stack_addr(i_slot);
        f.fb.store(zero, i_addr0, 0);

        let header_bb = f.fb.create_block(&[]);
        let body_bb = f.fb.create_block(&[]);
        let exit_bb = f.fb.create_block(&[]);
        f.fb.jump(header_bb, &[]);

        f.fb.switch_to_block(header_bb);
        let i_addr_h = f.fb.stack_addr(i_slot);
        let i_h = f.fb.load(Type::I64, i_addr_h, 0);
        let cond = f.fb.icmp(CmpOp::Slt, i_h, old_len);
        f.fb.br_if(cond, body_bb, &[], exit_bb, &[]);

        f.fb.switch_to_block(body_bb);
        let i_addr_b = f.fb.stack_addr(i_slot);
        let i_b = f.fb.load(Type::I64, i_addr_b, 0);
        let elem = self.array_load_elem_raw(f, src_raw1, i_b);
        self.array_store_elem(f, new_raw, i_b, elem);
        let i_inc = f.fb.add(i_b, one);
        f.fb.store(i_inc, i_addr_b, 0);
        f.fb.jump(header_bb, &[]);

        f.fb.switch_to_block(exit_bb);
        // Append the new value at index old_len.
        let val_v = f.get_var(&val_slot);
        self.array_store_elem(f, new_raw, old_len, val_v);

        f.obj_wrap(new_raw)
    }

    /// Index a NanBox-tagged array by a NanBox-float index. Caller is
    /// responsible for ensuring `arr` actually points at an `__Array__` —
    /// we don't type-check here (no out-of-bounds either). Lowering for
    /// a benchmark, not a safe runtime.
    fn array_load_at(&mut self, f: &mut DynFunc, arr: Value, idx_box: Value) -> Value {
        let raw = f.obj_unwrap(arr);
        let idx_int = nanbox_to_int(f, idx_box);
        self.array_load_elem_raw(f, raw, idx_int)
    }

    fn array_load_elem_raw(&self, f: &mut DynFunc, raw: Value, idx_int: Value) -> Value {
        let base = f.fb.iconst(Type::I64, self.array.elem_base);
        let eight = f.fb.iconst(Type::I64, 8);
        let byte_off = f.fb.mul(idx_int, eight);
        let off = f.fb.add(base, byte_off);
        let addr = f.fb.add(raw, off);
        f.fb.load(Type::I64, addr, 0)
    }

    fn array_store_elem(&self, f: &mut DynFunc, raw: Value, idx_int: Value, val: Value) {
        let base = f.fb.iconst(Type::I64, self.array.elem_base);
        let eight = f.fb.iconst(Type::I64, 8);
        let byte_off = f.fb.mul(idx_int, eight);
        let off = f.fb.add(base, byte_off);
        let addr = f.fb.add(raw, off);
        f.fb.store(val, addr, 0);
    }

    /// Emit inline-cache dispatch for a property load. The shape is
    /// monomorphic-with-slow-path: compare the object header's TypeInfo*
    /// against the cached class id, take the fast load on hit, call the
    /// slow-path extern on miss (which fills the entry and does the load).
    fn emit_ic_property_load(
        &mut self,
        f: &mut DynFunc,
        obj: Value,
        sym: Symbol,
        cache_id: u32,
    ) -> Value {
        // Raw heap pointer. `dynobj::Compact` stores a u16 `type_id` at
        // offset 0 with zeroed padding in the remaining 6 bytes, so a
        // full I64 load gives `type_id as u64`. We add 1 to produce the
        // class key (keeps 0 free as the IC empty-sentinel).
        let raw = f.obj_unwrap(obj);
        let type_id = f.fb.load(Type::I64, raw, 0);
        let one = f.fb.iconst(Type::I64, 1);
        let ti = f.fb.add(type_id, one);

        // IC entry address: ic_base + cache_id * sizeof(InlineCacheEntry).
        // The base is a compile-time-known u64; the array is heap-allocated
        // once with fixed capacity, so the pointer is stable.
        let entry_size = std::mem::size_of::<InlineCacheEntry>() as i64;
        let entry_addr_const =
            self.ic_base_addr as i64 + (cache_id as i64) * entry_size;
        let entry_addr = f.fb.iconst(Type::I64, entry_addr_const);

        // cached_class_id is the first u64 in InlineCacheEntry.
        let cached_class = f.fb.load(Type::I64, entry_addr, 0);
        let hit = f.fb.icmp(CmpOp::Eq, cached_class, ti);

        let hit_bb = f.fb.create_block(&[]);
        let miss_bb = f.fb.create_block(&[]);
        let merge_bb = f.fb.create_block(&[Type::I64]);
        f.fb.br_if(hit, hit_bb, &[], miss_bb, &[]);

        // ── Fast path: load(raw + cached_offset) ───────────────────
        f.fb.switch_to_block(hit_bb);
        let cached_off = f.fb.load(Type::I64, entry_addr, 8); // cached_value field
        let addr = f.fb.add(raw, cached_off);
        let fast_val = f.fb.load(Type::I64, addr, 0);
        f.fb.jump(merge_bb, &[fast_val]);

        // ── Slow path: extern call, fills the IC entry and returns value ─
        f.fb.switch_to_block(miss_bb);
        let sym_val = f.fb.iconst(Type::I64, sym.as_u32() as i64);
        let cid_val = f.fb.iconst(Type::I64, cache_id as i64);
        let slow_val = f
            .fb
            .call(self.prop_slow_ref, &[obj, sym_val, cid_val])
            .unwrap();
        f.fb.jump(merge_bb, &[slow_val]);

        f.fb.switch_to_block(merge_bb);
        f.fb.block_param(merge_bb, 0)
    }

    /// Bit-equal two NanBox values. For beagle's `==`, this gives the
    /// correct answer for `null` checks against pointers (different bit
    /// patterns), integer-valued floats (stored as canonical bits), and
    /// pointer identity. Not IEEE eq — NaN == NaN here.
    fn bit_eq(&mut self, f: &mut DynFunc, a: Value, b: Value) -> Value {
        let raw_eq = f.fb.icmp(CmpOp::Eq, a, b);
        let t = f.bool_val(true);
        let fal = f.bool_val(false);
        f.fb.select(raw_eq, t, fal)
    }

    fn bool_not(&mut self, f: &mut DynFunc, v: Value) -> Value {
        let falsey = f.is_falsey(v);
        let t = f.bool_val(true);
        let fal = f.bool_val(false);
        f.fb.select(falsey, t, fal)
    }
}

/// Whole-program type analysis. Produces a `TypeInfo` consumed by the
/// lowerer to decide between `num_*` and `dyn_*` ops.
///
/// Iterates a forward dataflow analysis to a fixed point (or up to a
/// small cap — anything still oscillating after that gets `Unknown`).
/// The lattice is the simple `Ty` type; LUB is monotonic so the fixpoint
/// terminates. Three things settle together:
///
///   - **Function return types.** Each top-level fn's return is the
///     type of its body's last expression, evaluated against the
///     current env. Recursive cycles converge to whatever the LUB of
///     each pass produces (often `Unknown`).
///
///   - **Struct field types.** For each `(struct, field)` pair, take
///     the LUB of every value passed at every `StructCreation` site.
///     A struct's field type can stabilize as `Number` only if every
///     constructor passes a known-`Number` value.
///
///   - **Let-mut narrowed types.** For each top-level fn body, walk
///     every `let mut x = init` and join with the type of every
///     `Assignment(Identifier(x), v)` reachable in the same body. We
///     don't track shadowing here — the analysis assumes a let-mut
///     name is unique within its function (true for the bench).
fn analyze_types(
    elements: &[Ast],
    globals: &HashMap<String, Ast>,
    inlinable: &HashMap<String, InlineableFn>,
) -> TypeInfo {
    let mut info = TypeInfo::default();

    // Pre-seed param types as `Bottom` for every declared function so
    // Pass C's LUB across call sites accumulates properly: `Bottom.lub(T) = T`,
    // letting the first observed argument actually contribute. Seeding
    // with `Unknown` would lock the lattice at the top forever, since
    // `Unknown.lub(T) = Unknown`.
    for el in elements {
        if let Ast::Function { name, args, .. } = el {
            if let Some(n) = name {
                info.fn_param_types
                    .insert(n.clone(), vec![Ty::Bottom; args.len()]);
            }
        }
    }

    // Iterate to fixpoint. The lattice is finite, but our "recompute
    // fresh" passes (Pass B for struct_fields, Pass C for fn_param_types)
    // can take several rounds to settle as fn_returns and the field
    // table feed each other. 15 iterations is overkill for the
    // benchmark (settles in ~6) and cheap.
    for _ in 0..15 {
        let mut changed = false;
        analyze_pass(elements, globals, inlinable, &mut info, &mut changed);
        if !changed {
            break;
        }
    }

    info
}

fn analyze_pass(
    elements: &[Ast],
    globals: &HashMap<String, Ast>,
    inlinable: &HashMap<String, InlineableFn>,
    info: &mut TypeInfo,
    changed: &mut bool,
) {
    // ── Pass A: function return types + per-fn let-mut LUBs. ────────
    for el in elements {
        if let Ast::Function { name, args, body, .. } = el {
            let Some(fname) = name.as_ref() else { continue };
            // Build a per-function env from current param-type guesses.
            let mut env: Vec<HashMap<String, Ty>> = vec![HashMap::new()];
            if let Some(param_tys) = info.fn_param_types.get(fname) {
                for (pat, ty) in args.iter().zip(param_tys.iter()) {
                    if let Pattern::Identifier { name, .. } = pat {
                        env[0].insert(name.clone(), ty.clone());
                    }
                }
            }

            // Let-mut LUBs: walk body, collecting (name → init type)
            // and joining over every assignment to that name.
            let mut let_mut: HashMap<String, Ty> = HashMap::new();
            collect_let_mut_types(body, &env, info, globals, &mut let_mut);

            // Stage entries into env so subsequent reads in this
            // function's analysis pick them up.
            for (k, v) in &let_mut {
                env[0].insert(k.clone(), v.clone());
            }

            // Update the stored let-mut table; mark `changed` if it
            // actually moved.
            let prev = info.let_mut_types.get(fname).cloned().unwrap_or_default();
            if prev != let_mut {
                info.let_mut_types.insert(fname.clone(), let_mut);
                *changed = true;
            }

            // Now compute the function's return type by typing every
            // statement of its body in order, updating `env` for `let`s.
            let ret_ty = type_block(body, &mut env, info, globals);
            let prev_ret = info.fn_returns.get(fname).cloned();
            if prev_ret.as_ref() != Some(&ret_ty) {
                info.fn_returns.insert(fname.clone(), ret_ty);
                *changed = true;
            }
        }
    }

    // ── Pass B: struct field LUBs. ─────────────────────────────────
    // Recompute fresh each iteration. We do NOT carry forward a prior
    // pass's `struct_fields` value: doing so would let an early
    // iteration's pessimistic `Unknown` (from before fn_param_types
    // converged) permanently poison the field. Instead, drop the table
    // and rebuild from a clean start using the latest env / param
    // types — once those stabilize, the field types do too.
    let prev_fields = std::mem::take(&mut info.struct_fields);
    for el in elements {
        if let Ast::Function { args, body, name, .. } = el {
            let mut env: Vec<HashMap<String, Ty>> = vec![HashMap::new()];
            if let Some(fname) = name {
                if let Some(param_tys) = info.fn_param_types.get(fname) {
                    for (pat, ty) in args.iter().zip(param_tys.iter()) {
                        if let Pattern::Identifier { name, .. } = pat {
                            env[0].insert(name.clone(), ty.clone());
                        }
                    }
                }
            }
            for s in body {
                walk_struct_creations(s, &mut env, info, globals, changed);
            }
        }
    }
    if prev_fields != info.struct_fields {
        *changed = true;
    }

    // ── Pass C: function param types from call sites. ──────────────
    // For each Call/inline site, take LUB of arg types into the
    // callee's params. Like Pass B, this restarts from a `Bottom` seed
    // so an early iteration's `Unknown` (computed before fn_returns
    // converged) doesn't permanently poison the param.
    let mut new_params: HashMap<String, Vec<Ty>> = info
        .fn_param_types
        .iter()
        .map(|(k, v)| (k.clone(), vec![Ty::Bottom; v.len()]))
        .collect();
    for el in elements {
        if let Ast::Function { body, args: caller_args, name, .. } = el {
            let mut env: Vec<HashMap<String, Ty>> = vec![HashMap::new()];
            if let Some(fname) = name {
                if let Some(param_tys) = info.fn_param_types.get(fname) {
                    for (pat, ty) in caller_args.iter().zip(param_tys.iter()) {
                        if let Pattern::Identifier { name, .. } = pat {
                            env[0].insert(name.clone(), ty.clone());
                        }
                    }
                }
            }
            for s in body {
                walk_calls_for_params(s, &mut env, info, globals, &mut new_params);
            }
        }
    }
    if new_params != info.fn_param_types {
        info.fn_param_types = new_params;
        *changed = true;
    }

    // Avoid `unused` warnings — `inlinable` is here for symmetry with
    // call-site logic that may want to recognise inline-only types
    // later, but the analysis above doesn't actually need to walk
    // inlinable bodies separately (they're top-level functions too).
    let _ = inlinable;
}

/// Type the full block in order, updating `env` as `let`s are
/// encountered. Returns the type of the final expression.
fn type_block(
    body: &[Ast],
    env: &mut Vec<HashMap<String, Ty>>,
    info: &TypeInfo,
    globals: &HashMap<String, Ast>,
) -> Ty {
    let mut last = Ty::Unknown;
    for stmt in body {
        match stmt {
            Ast::Let { pattern, value, .. } => {
                let ty = expr_type(value, env, info, globals);
                if let Pattern::Identifier { name, .. } = pattern {
                    if let Some(scope) = env.last_mut() {
                        scope.insert(name.clone(), ty.clone());
                    }
                }
                last = ty;
            }
            Ast::LetMut { value, .. } => {
                // The pre-pass already staged the LUB type into env;
                // don't overwrite it with the (narrower) initializer's
                // type — later assignments may widen it.
                last = expr_type(value, env, info, globals);
            }
            other => last = expr_type(other, env, info, globals),
        }
    }
    last
}

/// Type a sequence of statements in a fresh nested scope. Used for if /
/// while / for branches whose `let` bindings shouldn't leak past them.
/// Returns the type of the last statement.
fn type_block_local(
    body: &[Ast],
    env: &Vec<HashMap<String, Ty>>,
    info: &TypeInfo,
    globals: &HashMap<String, Ast>,
) -> Ty {
    let mut local: Vec<HashMap<String, Ty>> = env.clone();
    local.push(HashMap::new());
    let mut last = Ty::Unknown;
    for stmt in body {
        match stmt {
            Ast::Let { pattern, value, .. } => {
                let ty = expr_type(value, &local, info, globals);
                if let Pattern::Identifier { name, .. } = pattern {
                    if let Some(scope) = local.last_mut() {
                        scope.insert(name.clone(), ty.clone());
                    }
                }
                last = ty;
            }
            Ast::LetMut { pattern, value, .. } => {
                let ty = expr_type(value, &local, info, globals);
                if let Pattern::Identifier { name, .. } = pattern {
                    if let Some(scope) = local.last_mut() {
                        scope.insert(name.clone(), ty.clone());
                    }
                }
                last = ty;
            }
            other => last = expr_type(other, &local, info, globals),
        }
    }
    last
}

/// Mirror of `Lowerer::type_of`, free-standing so the analysis pass can
/// invoke it before any `Lowerer` exists. Should stay in sync — they
/// implement the same lattice rules.
fn expr_type(
    ast: &Ast,
    env: &Vec<HashMap<String, Ty>>,
    info: &TypeInfo,
    globals: &HashMap<String, Ast>,
) -> Ty {
    match ast {
        Ast::IntegerLiteral(..) | Ast::FloatLiteral(..) => Ty::Number,
        Ast::True(..) | Ast::False(..) | Ast::Null(..) | Ast::String(..) => Ty::Unknown,

        Ast::Identifier(name, _) => {
            if let Some(global) = globals.get(name) {
                return expr_type(global, env, info, globals);
            }
            for scope in env.iter().rev() {
                if let Some(t) = scope.get(name) {
                    return t.clone();
                }
            }
            Ty::Unknown
        }

        Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. } => {
            if expr_type(left, env, info, globals).is_number()
                && expr_type(right, env, info, globals).is_number()
            {
                Ty::Number
            } else {
                Ty::Unknown
            }
        }

        Ast::Condition { .. } | Ast::And { .. } | Ast::Or { .. } | Ast::Not { .. } => {
            Ty::Unknown
        }

        Ast::If { then, else_, .. } => {
            // Each branch is a Vec<Ast> with its own `let` chain. We
            // can't just type the last expression — `let` bindings made
            // earlier in the branch must be visible to it. Run a
            // statement-by-statement walk in a fresh scope for each.
            let tt = type_block_local(then, env, info, globals);
            let et = type_block_local(else_, env, info, globals);
            tt.lub(&et)
        }

        Ast::Let { value, .. } | Ast::LetMut { value, .. } => {
            expr_type(value, env, info, globals)
        }
        Ast::Assignment { value, .. } => expr_type(value, env, info, globals),

        Ast::StructCreation { name, .. } => Ty::Object(name.clone()),

        Ast::Array { array, .. } => {
            if array.is_empty() {
                return Ty::Array(Box::new(Ty::Bottom));
            }
            let mut acc = expr_type(&array[0], env, info, globals);
            for x in &array[1..] {
                acc = acc.lub(&expr_type(x, env, info, globals));
            }
            Ty::Array(Box::new(acc))
        }

        Ast::IndexOperator { array, .. } => {
            if let Ty::Array(elem) = expr_type(array, env, info, globals) {
                *elem
            } else {
                Ty::Unknown
            }
        }

        Ast::PropertyAccess { object, property, .. } => {
            let obj_ty = expr_type(object, env, info, globals);
            if let Ty::Object(struct_name) = obj_ty {
                if let Ast::Identifier(field, _) = property.as_ref() {
                    if let Some(t) =
                        info.struct_fields.get(&(struct_name, field.clone()))
                    {
                        return t.clone();
                    }
                }
            }
            Ty::Unknown
        }

        Ast::Call { name, args, .. } => {
            match name.as_str() {
                "cos" | "sin" | "to-float" | "to-number" | "length" | "core/time-now" => {
                    return Ty::Number;
                }
                "push" => {
                    if args.len() == 2 {
                        let arr_ty = expr_type(&args[0], env, info, globals);
                        let val_ty = expr_type(&args[1], env, info, globals);
                        if let Ty::Array(prior) = arr_ty {
                            return Ty::Array(Box::new(prior.lub(&val_ty)));
                        }
                        return Ty::Array(Box::new(val_ty));
                    }
                }
                _ => {}
            }
            info.fn_returns.get(name).cloned().unwrap_or(Ty::Unknown)
        }

        Ast::CallExpr { .. } => Ty::Unknown,

        Ast::While { .. } | Ast::Loop { .. } | Ast::For { .. } => Ty::Unknown,

        _ => Ty::Unknown,
    }
}

/// Collect let-mut narrowed types via a forward walk over the function
/// body. Threads a mutable env through so `let` bindings encountered
/// mid-body (including ones inside loop bodies) are visible when we
/// type subsequent expressions — including assignments to outer
/// let-muts whose RHS reads those inner lets.
fn collect_let_mut_types(
    body: &[Ast],
    seed_env: &Vec<HashMap<String, Ty>>,
    info: &TypeInfo,
    globals: &HashMap<String, Ast>,
    out: &mut HashMap<String, Ty>,
) {
    // Working env, seeded from the function's parameter scope.
    let mut env = seed_env.clone();

    for s in body {
        forward_let_mut(s, &mut env, info, globals, out);
    }
}

/// Walk the AST in lexical order. Maintains `env` as encountered
/// `let`/`let mut` bindings extend it. For each `let mut x = init`,
/// record/init `out[x] = init_type`. For each `Assignment(Identifier(x),
/// v)`, update `out[x] = out[x].lub(value_type)` AND `env[x] = out[x]`
/// so later reads of `x` see the running LUB.
fn forward_let_mut(
    ast: &Ast,
    env: &mut Vec<HashMap<String, Ty>>,
    info: &TypeInfo,
    globals: &HashMap<String, Ast>,
    out: &mut HashMap<String, Ty>,
) {
    match ast {
        Ast::Let { pattern, value, .. } => {
            forward_let_mut(value, env, info, globals, out);
            let ty = expr_type(value, env, info, globals);
            if let Pattern::Identifier { name, .. } = pattern {
                if let Some(scope) = env.last_mut() {
                    scope.insert(name.clone(), ty);
                }
            }
        }
        Ast::LetMut { pattern, value, .. } => {
            forward_let_mut(value, env, info, globals, out);
            let init_ty = expr_type(value, env, info, globals);
            if let Pattern::Identifier { name, .. } = pattern {
                let merged = match out.get(name) {
                    Some(prev) => prev.lub(&init_ty),
                    None => init_ty.clone(),
                };
                out.insert(name.clone(), merged.clone());
                if let Some(scope) = env.last_mut() {
                    scope.insert(name.clone(), merged);
                }
            }
        }
        Ast::Assignment { name, value, .. } => {
            forward_let_mut(value, env, info, globals, out);
            if let Ast::Identifier(n, _) = name.as_ref() {
                let v_ty = expr_type(value, env, info, globals);
                let merged = match out.get(n) {
                    Some(prev) => prev.lub(&v_ty),
                    None => v_ty,
                };
                out.insert(n.clone(), merged.clone());
                // Push the running LUB into env so subsequent reads of
                // the variable in this same pass see the wider type.
                for scope in env.iter_mut().rev() {
                    if scope.contains_key(n) {
                        scope.insert(n.clone(), merged);
                        break;
                    }
                }
            }
        }
        Ast::While { condition, body, .. } => {
            forward_let_mut(condition, env, info, globals, out);
            // Loops can iterate, so a let-mut written inside the body
            // can be read back at the top of the body. Two passes
            // suffice for the simple lattice we use (a third pass would
            // make no further progress).
            for _ in 0..2 {
                env.push(HashMap::new());
                for s in body {
                    forward_let_mut(s, env, info, globals, out);
                }
                env.pop();
            }
        }
        Ast::For { collection, body, .. } => {
            forward_let_mut(collection, env, info, globals, out);
            for _ in 0..2 {
                env.push(HashMap::new());
                for s in body {
                    forward_let_mut(s, env, info, globals, out);
                }
                env.pop();
            }
        }
        Ast::Loop { body, .. } => {
            for _ in 0..2 {
                env.push(HashMap::new());
                for s in body {
                    forward_let_mut(s, env, info, globals, out);
                }
                env.pop();
            }
        }
        Ast::If { condition, then, else_, .. } => {
            forward_let_mut(condition, env, info, globals, out);
            env.push(HashMap::new());
            for s in then {
                forward_let_mut(s, env, info, globals, out);
            }
            env.pop();
            env.push(HashMap::new());
            for s in else_ {
                forward_let_mut(s, env, info, globals, out);
            }
            env.pop();
        }
        _ => {
            walk_children(ast, |c| forward_let_mut(c, env, info, globals, out));
        }
    }
}

/// Walk every `StructCreation` reachable from `ast`. For each one, type
/// each field value in the current env and LUB into
/// `info.struct_fields[(struct_name, field_name)]`. `let` bindings
/// inside the walk extend `env` so subsequent expressions see them.
fn walk_struct_creations(
    ast: &Ast,
    env: &mut Vec<HashMap<String, Ty>>,
    info: &mut TypeInfo,
    globals: &HashMap<String, Ast>,
    changed: &mut bool,
) {
    match ast {
        Ast::Let { pattern, value, .. } | Ast::LetMut { pattern, value, .. } => {
            walk_struct_creations(value, env, info, globals, changed);
            let ty = expr_type(value, env, info, globals);
            if let Pattern::Identifier { name, .. } = pattern {
                if let Some(scope) = env.last_mut() {
                    scope.insert(name.clone(), ty);
                }
            }
        }
        Ast::StructCreation { name, fields, .. } => {
            for (fname, fval) in fields {
                walk_struct_creations(fval, env, info, globals, changed);
                let ty = expr_type(fval, env, info, globals);
                let key = (name.clone(), fname.clone());
                let merged = match info.struct_fields.get(&key) {
                    Some(prev) => prev.lub(&ty),
                    None => ty.clone(),
                };
                if info.struct_fields.get(&key) != Some(&merged) {
                    info.struct_fields.insert(key, merged);
                    *changed = true;
                }
            }
        }
        _ => {
            walk_children(ast, |c| walk_struct_creations(c, env, info, globals, changed));
        }
    }
}

/// Walk every call site and propagate caller's arg types into the
/// callee's param types via LUB.
fn walk_calls_for_params(
    ast: &Ast,
    env: &mut Vec<HashMap<String, Ty>>,
    info: &TypeInfo,
    globals: &HashMap<String, Ast>,
    new_params: &mut HashMap<String, Vec<Ty>>,
) {
    match ast {
        Ast::Let { pattern, value, .. } | Ast::LetMut { pattern, value, .. } => {
            walk_calls_for_params(value, env, info, globals, new_params);
            let ty = expr_type(value, env, info, globals);
            if let Pattern::Identifier { name, .. } = pattern {
                if let Some(scope) = env.last_mut() {
                    scope.insert(name.clone(), ty);
                }
            }
        }
        Ast::Call { name, args, .. } => {
            for a in args {
                walk_calls_for_params(a, env, info, globals, new_params);
            }
            // Skip builtins — we don't track those in fn_param_types.
            if !matches!(
                name.as_str(),
                "cos" | "sin" | "to-float" | "to-number" | "length" | "core/time-now"
                    | "push" | "print" | "println" | "get" | "beagle.core/equal"
            ) {
                if let Some(existing) = new_params.get(name).cloned() {
                    let mut updated = existing.clone();
                    for (i, a) in args.iter().enumerate() {
                        if i >= updated.len() {
                            break;
                        }
                        let arg_ty = expr_type(a, env, info, globals);
                        updated[i] = updated[i].lub(&arg_ty);
                    }
                    if updated != existing {
                        new_params.insert(name.clone(), updated);
                    }
                }
            }
        }
        _ => walk_children(ast, |c| {
            walk_calls_for_params(c, env, info, globals, new_params)
        }),
    }
}

/// Generic AST child-walker: invokes `f` on every immediate child
/// expression. Used by the small AST passes above to avoid duplicating
/// the structural recursion. Doesn't recurse into nested function
/// definitions or other top-level forms — those are walked at the
/// program level.
fn walk_children(ast: &Ast, mut f: impl FnMut(&Ast)) {
    match ast {
        Ast::If { condition, then, else_, .. } => {
            f(condition);
            for x in then {
                f(x);
            }
            for x in else_ {
                f(x);
            }
        }
        Ast::While { condition, body, .. } => {
            f(condition);
            for x in body {
                f(x);
            }
        }
        Ast::For { collection, body, .. } => {
            f(collection);
            for x in body {
                f(x);
            }
        }
        Ast::Loop { body, .. }
        | Ast::Reset { body, .. }
        | Ast::Shift { body, .. }
        | Ast::Test { body, .. } => {
            for x in body {
                f(x);
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
            f(left);
            f(right);
        }
        Ast::Call { args, .. } | Ast::Recurse { args, .. } | Ast::TailRecurse { args, .. } => {
            for a in args {
                f(a);
            }
        }
        Ast::CallExpr { callee, args, .. } => {
            f(callee);
            for a in args {
                f(a);
            }
        }
        Ast::Let { value, .. } | Ast::LetMut { value, .. } | Ast::LetDynamic { value, .. } => {
            f(value);
        }
        Ast::Binding { value_expr, body, .. } => {
            f(value_expr);
            for s in body {
                f(s);
            }
        }
        Ast::Assignment { name, value, .. } => {
            f(name);
            f(value);
        }
        Ast::Not { expr, .. } => f(expr),
        Ast::PropertyAccess { object, property, .. } => {
            f(object);
            f(property);
        }
        Ast::IndexOperator { array, index, .. } => {
            f(array);
            f(index);
        }
        Ast::Array { array, .. } => {
            for x in array {
                f(x);
            }
        }
        Ast::StructCreation { fields, spread, .. } => {
            for (_, v) in fields {
                f(v);
            }
            if let Some(s) = spread {
                f(s);
            }
        }
        Ast::EnumCreation { fields, .. } => {
            for (_, v) in fields {
                f(v);
            }
        }
        Ast::MapLiteral { pairs, .. } => {
            for (k, v) in pairs {
                f(k);
                f(v);
            }
        }
        Ast::SetLiteral { elements, .. } => {
            for x in elements {
                f(x);
            }
        }
        Ast::Break { value, .. } | Ast::Return { value, .. } | Ast::Throw { value, .. } => {
            f(value);
        }
        Ast::StringInterpolation { parts, .. } => {
            for p in parts {
                if let crate::ast::StringInterpolationPart::Expression(e) = p {
                    f(e);
                }
            }
        }
        Ast::Try { body, catch_body, .. } => {
            for x in body {
                f(x);
            }
            for x in catch_body {
                f(x);
            }
        }
        Ast::Match { value, arms, .. } => {
            f(value);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    f(g);
                }
                for x in &arm.body {
                    f(x);
                }
            }
        }
        Ast::MultiArityFunction { cases, .. } => {
            for c in cases {
                for x in &c.body {
                    f(x);
                }
            }
        }
        Ast::Perform { value, .. } => f(value),
        Ast::Handle { handler_instance, body, .. } => {
            f(handler_instance);
            for x in body {
                f(x);
            }
        }
        Ast::Future { body, .. } => f(body),
        Ast::Use { alias, .. } => f(alias),

        // Leaves and top-level decls — no children we walk in this
        // helper.
        Ast::Program { .. }
        | Ast::Function { .. }
        | Ast::FunctionStub { .. }
        | Ast::Struct { .. }
        | Ast::StructField { .. }
        | Ast::Enum { .. }
        | Ast::EnumVariant { .. }
        | Ast::EnumStaticVariant { .. }
        | Ast::Protocol { .. }
        | Ast::Extend { .. }
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

/// Convert a NanBox-encoded float (stored as I64 bits) into a signed
/// 64-bit integer. Used for array indices.
fn nanbox_to_int(f: &mut DynFunc, v: Value) -> Value {
    let as_f64 = f.unbox_number(v);
    f.fb.float_to_int(as_f64)
}

/// Count every `Ast::PropertyAccess` node reachable from `ast`,
/// **simulating static inlining** of any `Call` whose target is in
/// `inlinable`. Each inlined call site expands into a fresh copy of the
/// callee's body, so its property accesses count again per call site.
///
/// Recursion guard: a cycle in the inlinable set (mutual recursion that
/// my self-call check missed) would cause unbounded counting; the
/// `visited` set bails out the second time we'd recurse into the same
/// callee. Lowering uses the same guard implicitly via Rust's stack
/// limit, but better to fail loudly here than to crash later.
fn count_property_accesses_with_inlining(
    ast: &Ast,
    inlinable: &HashMap<String, InlineableFn>,
) -> usize {
    let mut n = 0;
    let mut visited = std::collections::HashSet::new();
    count_in(ast, inlinable, &mut visited, &mut n);
    n
}

fn count_in(
    ast: &Ast,
    inlinable: &HashMap<String, InlineableFn>,
    visited: &mut std::collections::HashSet<String>,
    n: &mut usize,
) {
    macro_rules! recur {
        ($e:expr) => { count_in($e, inlinable, visited, n) };
    }
    macro_rules! recur_vec {
        ($xs:expr) => { for x in $xs { recur!(x) } };
    }
    macro_rules! recur_opt {
        ($x:expr) => { if let Some(b) = $x { recur!(b) } };
    }
    match ast {
        Ast::PropertyAccess { object, property, .. } => {
            *n += 1;
            recur!(object);
            recur!(property);
        }

        Ast::Program { elements, .. } => recur_vec!(elements),
        Ast::Function { body, .. } => recur_vec!(body),
        Ast::Struct { fields, .. } => recur_vec!(fields),
        Ast::StructField { default_value, .. } => recur_opt!(default_value),
        Ast::Enum { variants, .. } => recur_vec!(variants),
        Ast::EnumVariant { fields, .. } => recur_vec!(fields),
        Ast::Protocol { body, .. } => recur_vec!(body),
        Ast::Extend { body, .. } => recur_vec!(body),
        Ast::If { condition, then, else_, .. } => {
            recur!(condition);
            recur_vec!(then);
            recur_vec!(else_);
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
            recur!(left);
            recur!(right);
        }
        Ast::Recurse { args, .. } | Ast::TailRecurse { args, .. } => recur_vec!(args),
        Ast::Call { name, args, .. } => {
            for a in args {
                recur!(a);
            }
            // Inline expansion: each call site to an inlinable callee
            // contributes another copy of the callee body's PA count.
            // The `visited` set is a cycle guard — should only fire if
            // mutual recursion sneaks past the per-function self-call
            // check.
            if let Some(inlinee) = inlinable.get(name) {
                if visited.insert(name.clone()) {
                    for stmt in &inlinee.body {
                        count_in(stmt, inlinable, visited, n);
                    }
                    visited.remove(name);
                }
            }
        }
        Ast::CallExpr { callee, args, .. } => {
            recur!(callee);
            recur_vec!(args);
        }
        Ast::Let { value, .. }
        | Ast::LetMut { value, .. }
        | Ast::LetDynamic { value, .. } => recur!(value),
        Ast::Binding { value_expr, body, .. } => {
            recur!(value_expr);
            recur_vec!(body);
        }
        Ast::StringInterpolation { parts, .. } => {
            for p in parts {
                if let crate::ast::StringInterpolationPart::Expression(e) = p {
                    recur!(e);
                }
            }
        }
        Ast::StructCreation { fields, spread, .. } => {
            for (_, v) in fields {
                recur!(v);
            }
            recur_opt!(spread);
        }
        Ast::EnumCreation { fields, .. } => {
            for (_, v) in fields {
                recur!(v);
            }
        }
        Ast::Use { alias, .. } => recur!(alias),
        Ast::Not { expr, .. } => recur!(expr),
        Ast::Array { array, .. } => recur_vec!(array),
        Ast::MapLiteral { pairs, .. } => {
            for (k, v) in pairs {
                recur!(k);
                recur!(v);
            }
        }
        Ast::SetLiteral { elements, .. } => recur_vec!(elements),
        Ast::IndexOperator { array, index, .. } => {
            recur!(array);
            recur!(index);
        }
        Ast::Loop { body, .. } => recur_vec!(body),
        Ast::While { condition, body, .. } => {
            recur!(condition);
            recur_vec!(body);
        }
        Ast::Break { value, .. } | Ast::Return { value, .. } | Ast::Throw { value, .. } => {
            recur!(value);
        }
        Ast::For { collection, body, .. } => {
            recur!(collection);
            recur_vec!(body);
        }
        Ast::Assignment { name, value, .. } => {
            recur!(name);
            recur!(value);
        }
        Ast::Try { body, catch_body, .. } => {
            recur_vec!(body);
            recur_vec!(catch_body);
        }
        Ast::Match { value, arms, .. } => {
            recur!(value);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    recur!(g);
                }
                recur_vec!(&arm.body);
            }
        }
        Ast::MultiArityFunction { cases, .. } => {
            for c in cases {
                recur_vec!(&c.body);
            }
        }
        Ast::Reset { body, .. } | Ast::Shift { body, .. } | Ast::Test { body, .. } => {
            recur_vec!(body);
        }
        Ast::Perform { value, .. } => recur!(value),
        Ast::Handle { handler_instance, body, .. } => {
            recur!(handler_instance);
            recur_vec!(body);
        }
        Ast::Future { body, .. } => recur!(body),

        // Leaves and inert declarations — no child expressions.
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
