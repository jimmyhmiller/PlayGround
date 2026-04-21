//! AST → dynir lowering for the beagle-on-toolkit port.
//!
//! Scope: whatever binary_trees.bg needs. Everything else panics explicitly.

use std::collections::HashMap;

use dynir::{CmpOp, FuncRef, Module, Signature, Type, Value};
use dynlang::{DynFunc, DynModule, FieldKind, GcConfig, NanBoxTags, ObjTypeId};
use dynobj::TypeInfo;

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
    /// TypeInfos for every registered object type, in ObjTypeId order.
    /// Used at runtime to set up the GC's type table.
    pub type_infos: Vec<TypeInfo>,
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

    let mut dm = DynModule::new(
        GcConfig::semi_space(2 * 1024 * 1024 * 1024),
        NanBoxTags::default(),
    );

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

    // ── Phase 1: register object types for all `struct` declarations. ──
    let mut structs: HashMap<String, StructInfo> = HashMap::new();
    let mut field_to_struct: HashMap<String, String> = HashMap::new();

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
            // Grab the offsets now, before we start mutating dm with
            // function bodies.
            let ty = dm.get_obj_type(id);
            let offsets: HashMap<String, i32> = ty
                .field_offsets
                .iter()
                .map(|(k, (off, _kind))| (k.clone(), *off))
                .collect();
            structs.insert(
                name.clone(),
                StructInfo { type_id: id, field_offsets: offsets },
            );
            for fname in field_names {
                if let Some(prev) = field_to_struct.insert(fname.clone(), name.clone()) {
                    if prev != *name {
                        panic!(
                            "field name `{fname}` appears on both `{prev}` and `{name}`; \
                             MVP requires statically resolvable field access",
                        );
                    }
                }
            }
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

    // ── Phase 3: define bodies. ──
    for el in elements {
        if let Ast::Function { name, args, body, .. } = el {
            let fname = name.as_ref().unwrap();
            let fref = *func_refs.get(fname).unwrap();
            let mut f = dm.start_func(fref);
            {
                let mut lw = Lowerer {
                    structs: &structs,
                    field_to_struct: &field_to_struct,
                    func_refs: &func_refs,
                    print_ref,
                    println_ref,
                    length_ref,
                    get_ref,
                    to_number_ref,
                    current_fn: fname.clone(),
                    strings: &mut strings,
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

    // Collect TypeInfos (in ObjTypeId order) before consuming dm.
    let mut type_infos: Vec<TypeInfo> = Vec::with_capacity(dm.obj_types.len());
    for obj_ty in &dm.obj_types {
        type_infos.push(*obj_ty.type_info);
    }

    let built = dm.build();
    // `built.strings` is dynlang's pool, not ours. Discard it.
    let _ = built.strings;
    Lowered {
        module: built.module,
        main: main_ref,
        strings,
        type_infos,
    }
}

struct Lowerer<'a> {
    structs: &'a HashMap<String, StructInfo>,
    field_to_struct: &'a HashMap<String, String>,
    func_refs: &'a HashMap<String, FuncRef>,
    print_ref: FuncRef,
    println_ref: FuncRef,
    length_ref: FuncRef,
    get_ref: FuncRef,
    to_number_ref: FuncRef,
    current_fn: String,
    strings: &'a mut StringPool,
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
                let raw = f.obj_unwrap(obj_val);
                let fname = match property.as_ref() {
                    Ast::Identifier(n, _) => n.clone(),
                    other => panic!("property access expects ident, got {:?}", other),
                };
                let struct_name = self
                    .field_to_struct
                    .get(&fname)
                    .unwrap_or_else(|| panic!("unknown field `{fname}`"));
                let info = self.structs.get(struct_name).unwrap();
                let offset = *info.field_offsets.get(&fname).unwrap();
                f.fb.load(Type::I64, raw, offset)
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
