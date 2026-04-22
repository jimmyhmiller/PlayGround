//! AST → dynir lowering for the beagle-on-toolkit port.
//!
//! Scope: whatever binary_trees.bg needs. Everything else panics explicitly.

use std::collections::HashMap;

use dynir::{CmpOp, FuncRef, Module, Signature, Type, Value};
use dynlang::{DynFunc, DynModule, FieldKind, GcConfig, NanBoxTags, ObjTypeId, gc::DynGcRuntime};
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

/// Entry: lower a `Program` AST into a dynir Module ready to interpret.
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
    // All return I64 (NanBox), take I64 args.
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

    // ── Phase 1: register object types for all `struct` declarations. ──
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

    // ── Phase 2: declare every function (enables mutual recursion). ──
    let mut func_refs: HashMap<String, FuncRef> = HashMap::new();
    for el in elements {
        if let Ast::Function { name, args, .. } = el {
            let fname = name.as_ref().expect("top-level fn must be named").clone();
            let fref = dm.declare_func(&fname, args.len());
            func_refs.insert(fname, fref);
        }
    }

    let main_ref = *func_refs.get("main").expect("program must declare fn main");

    let mut strings = StringPool::default();

    // Pre-walk the program to count property-access sites so we can
    // allocate a fixed-size `InlineCacheArray` whose base pointer we can
    // embed in IR as a constant. Growing the array later would invalidate
    // that pointer.
    let num_ic_sites = count_property_accesses(program);
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
                let mut lw = Lowerer {
                    structs: &structs,
                    func_refs: &func_refs,
                    print_ref,
                    println_ref,
                    length_ref,
                    get_ref,
                    to_number_ref,
                    prop_slow_ref,
                    current_fn: fname.clone(),
                    strings: &mut strings,
                    symbols: &mut symbols,
                    ic_base_addr,
                    next_cache_id: &mut next_cache_id,
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
        strings,
        gc,
        ic: IcContext {
            symbols,
            per_type,
            array: ic_array,
        },
    }
}

struct Lowerer<'a> {
    structs: &'a HashMap<String, StructInfo>,
    func_refs: &'a HashMap<String, FuncRef>,
    print_ref: FuncRef,
    println_ref: FuncRef,
    length_ref: FuncRef,
    get_ref: FuncRef,
    to_number_ref: FuncRef,
    prop_slow_ref: FuncRef,
    current_fn: String,
    strings: &'a mut StringPool,
    symbols: &'a mut SymbolTable,
    /// Base address of the InlineCacheArray, embedded as an IR constant.
    ic_base_addr: u64,
    next_cache_id: &'a mut u32,
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
            Ast::Identifier(name, _) => f.get_var(name),

            Ast::Let { pattern, value, .. } => {
                let vname = match pattern {
                    Pattern::Identifier { name, .. } => name.clone(),
                    _ => panic!("only simple `let <name> = ...` supported, got {:?}", pattern),
                };
                let v = self.lower_expr(f, value);
                f.def_var(&vname, v);
                v
            }

            // ── Arithmetic ──────────────────────────────────────────
            Ast::Add { left, right, .. } => {
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                f.dyn_add(l, r)
            }
            Ast::Sub { left, right, .. } => {
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                f.dyn_sub(l, r)
            }
            Ast::Mul { left, right, .. } => {
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                f.dyn_mul(l, r)
            }
            Ast::Div { left, right, .. } => {
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                f.dyn_div(l, r)
            }

            // ── Comparison ──────────────────────────────────────────
            Ast::Condition { operator, left, right, .. } => {
                let l = self.lower_expr(f, left);
                let r = self.lower_expr(f, right);
                match operator {
                    Condition::LessThan => f.dyn_lt(l, r),
                    Condition::GreaterThan => f.dyn_gt(l, r),
                    Condition::Equal => self.bit_eq(f, l, r),
                    Condition::NotEqual => {
                        let eq = self.bit_eq(f, l, r);
                        self.bool_not(f, eq)
                    }
                    Condition::LessThanOrEqual => {
                        let gt = f.dyn_gt(l, r);
                        self.bool_not(f, gt)
                    }
                    Condition::GreaterThanOrEqual => {
                        let lt = f.dyn_lt(l, r);
                        self.bool_not(f, lt)
                    }
                }
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

            // ── Function calls ──────────────────────────────────────
            Ast::Call { name, args, .. } => {
                let arg_vals: Vec<Value> =
                    args.iter().map(|a| self.lower_expr(f, a)).collect();

                if name == "print" {
                    assert_eq!(args.len(), 1, "print() takes exactly 1 arg");
                    f.fb.call(self.print_ref, &arg_vals);
                    return f.nil();
                }
                if name == "println" {
                    assert_eq!(args.len(), 1, "println() takes exactly 1 arg");
                    f.fb.call(self.println_ref, &arg_vals);
                    return f.nil();
                }
                if name == "length" {
                    return f.fb.call(self.length_ref, &arg_vals).unwrap();
                }
                if name == "get" {
                    return f.fb.call(self.get_ref, &arg_vals).unwrap();
                }
                if name == "to-number" {
                    return f.fb.call(self.to_number_ref, &arg_vals).unwrap();
                }
                // The beagle parser lowers `==` to a call to `beagle.core/equal`.
                // For our MVP that's identity (bit) equality — correct for
                // NanBox-encoded numbers, nils, and pointer-tagged objects.
                if name == "beagle.core/equal" {
                    assert_eq!(arg_vals.len(), 2, "beagle.core/equal takes 2 args");
                    return self.bit_eq(f, arg_vals[0], arg_vals[1]);
                }

                let fref = *self.func_refs.get(name).unwrap_or_else(|| {
                    panic!("unknown function `{name}` called from `{}`", self.current_fn)
                });
                f.fb.call(fref, &arg_vals).unwrap()
            }

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

/// Snapshot every variable currently in scope. Used as a conservative
/// live-root set at allocation sites.
fn snapshot_vars(f: &mut DynFunc) -> Vec<Value> {
    let names: Vec<String> = f
        .vars
        .iter()
        .flat_map(|scope| scope.keys().cloned())
        .collect();
    names.into_iter().map(|n| f.get_var(&n)).collect()
}

/// Count every `Ast::PropertyAccess` node reachable from `ast`. Must
/// match (or exceed) the number of IC sites lowering will emit — we
/// embed the IC array's base pointer as a compile-time constant, so
/// the array cannot grow once lowering starts. The assertion in
/// `lower_program` catches mismatches.
///
/// Exhaustive: every `Ast` variant is enumerated so adding a new
/// variant to the AST triggers a compile error here.
pub fn count_property_accesses(ast: &Ast) -> usize {
    let mut n = 0;
    count_in(ast, &mut n);
    n
}

fn count_in(ast: &Ast, n: &mut usize) {
    match ast {
        Ast::PropertyAccess { object, property, .. } => {
            *n += 1;
            count_in(object, n);
            count_in(property, n);
        }

        Ast::Program { elements, .. } => count_vec(elements, n),
        Ast::Function { body, .. } => count_vec(body, n),
        Ast::Struct { fields, .. } => count_vec(fields, n),
        Ast::StructField { default_value, .. } => count_opt_box(default_value, n),
        Ast::Enum { variants, .. } => count_vec(variants, n),
        Ast::EnumVariant { fields, .. } => count_vec(fields, n),
        Ast::Protocol { body, .. } => count_vec(body, n),
        Ast::Extend { body, .. } => count_vec(body, n),
        Ast::If { condition, then, else_, .. } => {
            count_in(condition, n);
            count_vec(then, n);
            count_vec(else_, n);
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
            count_in(left, n);
            count_in(right, n);
        }
        Ast::Recurse { args, .. } | Ast::TailRecurse { args, .. } => count_vec(args, n),
        Ast::Call { args, .. } => count_vec(args, n),
        Ast::CallExpr { callee, args, .. } => {
            count_in(callee, n);
            count_vec(args, n);
        }
        Ast::Let { value, .. }
        | Ast::LetMut { value, .. }
        | Ast::LetDynamic { value, .. } => count_in(value, n),
        Ast::Binding { value_expr, body, .. } => {
            count_in(value_expr, n);
            count_vec(body, n);
        }
        Ast::StringInterpolation { parts, .. } => {
            for p in parts {
                if let crate::ast::StringInterpolationPart::Expression(e) = p {
                    count_in(e, n);
                }
            }
        }
        Ast::StructCreation { fields, spread, .. } => {
            for (_, v) in fields {
                count_in(v, n);
            }
            count_opt_box(spread, n);
        }
        Ast::EnumCreation { fields, .. } => {
            for (_, v) in fields {
                count_in(v, n);
            }
        }
        Ast::Use { alias, .. } => count_in(alias, n),
        Ast::Not { expr, .. } => count_in(expr, n),
        Ast::Array { array, .. } => count_vec(array, n),
        Ast::MapLiteral { pairs, .. } => {
            for (k, v) in pairs {
                count_in(k, n);
                count_in(v, n);
            }
        }
        Ast::SetLiteral { elements, .. } => count_vec(elements, n),
        Ast::IndexOperator { array, index, .. } => {
            count_in(array, n);
            count_in(index, n);
        }
        Ast::Loop { body, .. } => count_vec(body, n),
        Ast::While { condition, body, .. } => {
            count_in(condition, n);
            count_vec(body, n);
        }
        Ast::Break { value, .. } | Ast::Return { value, .. } | Ast::Throw { value, .. } => {
            count_in(value, n);
        }
        Ast::For { collection, body, .. } => {
            count_in(collection, n);
            count_vec(body, n);
        }
        Ast::Assignment { name, value, .. } => {
            count_in(name, n);
            count_in(value, n);
        }
        Ast::Try { body, catch_body, .. } => {
            count_vec(body, n);
            count_vec(catch_body, n);
        }
        Ast::Match { value, arms, .. } => {
            count_in(value, n);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    count_in(g, n);
                }
                count_vec(&arm.body, n);
            }
        }
        Ast::MultiArityFunction { cases, .. } => {
            for c in cases {
                count_vec(&c.body, n);
            }
        }
        Ast::Reset { body, .. } | Ast::Shift { body, .. } | Ast::Test { body, .. } => {
            count_vec(body, n);
        }
        Ast::Perform { value, .. } => count_in(value, n),
        Ast::Handle { handler_instance, body, .. } => {
            count_in(handler_instance, n);
            count_vec(body, n);
        }
        Ast::Future { body, .. } => count_in(body, n),

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

fn count_vec(xs: &[Ast], n: &mut usize) {
    for x in xs {
        count_in(x, n);
    }
}

fn count_opt_box(x: &Option<Box<Ast>>, n: &mut usize) {
    if let Some(b) = x {
        count_in(b, n);
    }
}
