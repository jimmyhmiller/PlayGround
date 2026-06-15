//! Layout registry: maps concrete semantic types ([`Ty`]) to core-IR
//! representations ([`Repr`]) and heap/value layouts, building each shape on
//! demand and assigning stable ids.
//!
//! This is the bridge from the type system to the GC: a reference type becomes
//! a [`Layout`] with **pointer fields first** (so `gc::scan_object` traces
//! exactly the right slots), and a `value` type becomes an inline
//! [`ValueLayout`]. The registry is monomorphic-only — every `Ty` it sees must
//! be ground (no `Ty::Var`/`Ty::Infer`); the monomorphizer guarantees this.

use crate::core::*;
use crate::types::{Prim, Ty, TyCtx};
use crate::ast::{StructBody, VariantPayload};
use std::collections::HashMap;

pub struct LayoutRegistry<'a> {
    ctx: &'a TyCtx,
    /// Interning: a canonical key for a `Ty` → its assigned id.
    ref_ids: HashMap<String, LayoutId>,
    value_ids: HashMap<String, ValueId>,
    pub layouts: Vec<Layout>,
    pub values: Vec<ValueLayout>,
}

#[derive(Debug)]
pub struct LayoutError(pub String);
type R<T> = Result<T, LayoutError>;

impl<'a> LayoutRegistry<'a> {
    pub fn new(ctx: &'a TyCtx) -> Self {
        LayoutRegistry { ctx, ref_ids: HashMap::new(), value_ids: HashMap::new(), layouts: vec![], values: vec![] }
    }

    /// The [`Repr`] of a ground type, registering any needed layouts.
    pub fn repr(&mut self, ty: &Ty) -> R<Repr> {
        match ty {
            Ty::Prim(Prim::Unit) => Ok(Repr::Unit),
            Ty::Prim(Prim::Str) => {
                // String is a built-in reference type (varlen bytes).
                Ok(Repr::Ref(self.string_layout()))
            }
            Ty::Prim(p) => Ok(Repr::Scalar(scalar_of(*p).expect("non-unit prim scalar"))),
            Ty::Named { name, args } => self.named_repr(name, args),
            Ty::Array(elem, n) => {
                // Fixed array → a reference object holding `n` inline elements.
                let er = self.repr(elem)?;
                Ok(Repr::Ref(self.array_layout(&er, *n)))
            }
            Ty::Tuple(elems) => {
                // Tuples are value aggregates.
                let reprs: Vec<Repr> = elems.iter().map(|e| self.repr(e)).collect::<R<_>>()?;
                Ok(Repr::Value(self.tuple_value(&reprs)))
            }
            Ty::Fn { .. } => {
                // A closure value is a reference to its environment object; the
                // closure repr is the env layout (code ptr lives in the layout).
                Ok(Repr::Ref(self.closure_placeholder()))
            }
            Ty::Var(v) => Err(LayoutError(format!("non-ground type variable `{}` reached layout", v))),
            Ty::Infer(_) => Err(LayoutError("inference hole reached layout".into())),
        }
    }

    fn named_repr(&mut self, name: &str, args: &[Ty]) -> R<Repr> {
        // Builtin generic containers.
        match name {
            "Array" => {
                let er = self.repr(&args[0])?;
                return Ok(Repr::Ref(self.array_for(&er)));
            }
            "Vec" => {
                let er = self.repr(&args[0])?;
                return Ok(Repr::Ref(self.vec_layout(&er)));
            }
            "Option" | "Result" => {
                // Built-in enums — treat structurally as reference enums for now.
                return Ok(Repr::Ref(self.builtin_enum_layout(name, args)?));
            }
            _ => {}
        }
        // User struct or enum.
        if let Some(s) = self.ctx.structs.get(name).cloned() {
            let is_value = s.is_value;
            if is_value {
                Ok(Repr::Value(self.value_struct(name, &s, args)?))
            } else {
                Ok(Repr::Ref(self.ref_struct(name, &s, args)?))
            }
        } else if let Some(e) = self.ctx.enums.get(name).cloned() {
            if e.is_value {
                Ok(Repr::Value(self.value_enum(name, &e, args)?))
            } else {
                Ok(Repr::Ref(self.ref_enum(name, &e, args)?))
            }
        } else {
            Err(LayoutError(format!("unknown named type `{}`", name)))
        }
    }

    // ---- key construction (monomorphic interning) -------------------------
    fn key(name: &str, args: &[Ty]) -> String {
        if args.is_empty() {
            name.to_string()
        } else {
            let a: Vec<String> = args.iter().map(ty_key).collect();
            format!("{}<{}>", name, a.join(","))
        }
    }

    // ---- reference structs -------------------------------------------------
    fn ref_struct(&mut self, name: &str, s: &crate::ast::StructDef, args: &[Ty]) -> R<LayoutId> {
        let key = Self::key(name, args);
        if let Some(id) = self.ref_ids.get(&key) {
            return Ok(*id);
        }
        // Reserve id up-front for recursive types.
        let id = self.layouts.len() as LayoutId;
        self.ref_ids.insert(key.clone(), id);
        self.layouts.push(placeholder_layout(&key));

        let field_tys = struct_field_tys(s, args)?;
        let layout = self.build_ref_layout(&key, &field_tys)?;
        self.layouts[id as usize] = layout;
        Ok(id)
    }

    fn ref_enum(&mut self, name: &str, e: &crate::ast::EnumDef, args: &[Ty]) -> R<LayoutId> {
        let key = Self::key(name, args);
        if let Some(id) = self.ref_ids.get(&key) {
            return Ok(*id);
        }
        let id = self.layouts.len() as LayoutId;
        self.ref_ids.insert(key.clone(), id);
        self.layouts.push(placeholder_layout(&key));

        // A reference enum is one heap object: [tag: u32 raw][union of variant
        // payloads]. We give it the max over variants of (ptr_fields, raw_bytes)
        // so any variant fits. The tag is a raw scalar field. Pointer payload
        // fields go in the pointer region (traced); a variant with fewer ptr
        // fields than the max simply leaves trailing ptr slots null (safe to
        // trace — null is skipped).
        let mut max_ptrs = 0u16;
        let mut max_raw = 0u16;
        for v in &e.variants {
            let payload_tys = variant_payload_tys(v, &e.generics, args)?;
            let (ptrs, raw) = self.count_fields(&payload_tys)?;
            max_ptrs = max_ptrs.max(ptrs);
            max_raw = max_raw.max(raw);
        }
        // tag occupies 4 bytes within the raw region; reserve 8 for alignment.
        let raw_bytes = max_raw + 8;
        let layout = Layout {
            ptr_fields: max_ptrs,
            raw_bytes,
            varlen: VarLen::None,
            // field_map for an enum is per-variant; we store the tag at raw
            // offset 0 and let codegen compute payload locations per variant.
            field_map: vec![FieldLoc::Raw { offset: 0, repr: ScalarRepr::U32 }],
            name: key,
            elem_stride: 0,
        };
        self.layouts[id as usize] = layout;
        Ok(id)
    }

    fn builtin_enum_layout(&mut self, name: &str, args: &[Ty]) -> R<LayoutId> {
        // Option<T> = { None, Some(T) }; Result<T,E> = { Ok(T), Err(E) }.
        let key = Self::key(name, args);
        if let Some(id) = self.ref_ids.get(&key) {
            return Ok(*id);
        }
        let id = self.layouts.len() as LayoutId;
        self.ref_ids.insert(key.clone(), id);
        self.layouts.push(placeholder_layout(&key));

        let variants: Vec<Vec<Ty>> = match name {
            "Option" => vec![vec![], vec![args[0].clone()]],
            "Result" => vec![vec![args[0].clone()], vec![args[1].clone()]],
            _ => unreachable!(),
        };
        let mut max_ptrs = 0u16;
        let mut max_raw = 0u16;
        for vtys in &variants {
            let (ptrs, raw) = self.count_fields(vtys)?;
            max_ptrs = max_ptrs.max(ptrs);
            max_raw = max_raw.max(raw);
        }
        let layout = Layout {
            ptr_fields: max_ptrs,
            raw_bytes: max_raw + 8,
            varlen: VarLen::None,
            field_map: vec![FieldLoc::Raw { offset: 0, repr: ScalarRepr::U32 }],
            name: key,
            elem_stride: 0,
        };
        self.layouts[id as usize] = layout;
        Ok(id)
    }

    // ---- value aggregates --------------------------------------------------
    fn value_struct(&mut self, name: &str, s: &crate::ast::StructDef, args: &[Ty]) -> R<ValueId> {
        let key = Self::key(name, args);
        if let Some(id) = self.value_ids.get(&key) {
            return Ok(*id);
        }
        let field_tys = struct_field_tys(s, args)?;
        let fields: Vec<Repr> = field_tys.iter().map(|t| self.repr(t)).collect::<R<_>>()?;
        let (size, align) = value_size_align(&fields, &self.values);
        let id = self.values.len() as ValueId;
        self.values.push(ValueLayout { name: key.clone(), variants: None, fields, size, align });
        self.value_ids.insert(key, id);
        Ok(id)
    }

    fn value_enum(&mut self, name: &str, e: &crate::ast::EnumDef, args: &[Ty]) -> R<ValueId> {
        let key = Self::key(name, args);
        if let Some(id) = self.value_ids.get(&key) {
            return Ok(*id);
        }
        let mut variants = Vec::new();
        for v in &e.variants {
            let ptys = variant_payload_tys(v, &e.generics, args)?;
            let fields: Vec<Repr> = ptys.iter().map(|t| self.repr(t)).collect::<R<_>>()?;
            variants.push(ValueVariant { name: v.name.clone(), fields });
        }
        // size = tag(4) + max variant payload size; align computed conservatively.
        let mut max = 0u32;
        let mut align = 4u32;
        for vv in &variants {
            let (s, a) = value_size_align(&vv.fields, &self.values);
            max = max.max(s);
            align = align.max(a);
        }
        let size = align_up(4 + max, align);
        let id = self.values.len() as ValueId;
        self.values.push(ValueLayout { name: key.clone(), variants: Some(variants), fields: vec![], size, align });
        self.value_ids.insert(key, id);
        Ok(id)
    }

    fn tuple_value(&mut self, reprs: &[Repr]) -> ValueId {
        let key = format!("({})", reprs.iter().map(repr_key).collect::<Vec<_>>().join(","));
        if let Some(id) = self.value_ids.get(&key) {
            return *id;
        }
        let (size, align) = value_size_align(reprs, &self.values);
        let id = self.values.len() as ValueId;
        self.values.push(ValueLayout { name: key.clone(), variants: None, fields: reprs.to_vec(), size, align });
        self.value_ids.insert(key, id);
        id
    }

    // ---- built-in reference layouts ---------------------------------------
    fn string_layout(&mut self) -> LayoutId {
        self.intern_ref("String", Layout {
            ptr_fields: 0, raw_bytes: 0, varlen: VarLen::Bytes,
            field_map: vec![], name: "String".into(), elem_stride: 0,
        })
    }
    fn vec_layout(&mut self, elem: &Repr) -> LayoutId {
        // A Vec is a small header object {len, cap, ptr-to-backing}; for v0 we
        // model it as a reference with a pointer to a varlen backing array.
        let key = format!("Vec<{}>", repr_key(elem));
        let traced = matches!(elem, Repr::Ref(_));
        self.intern_ref(&key, Layout {
            ptr_fields: 1, // backing array pointer
            raw_bytes: 16, // len + cap
            varlen: VarLen::None,
            field_map: vec![
                FieldLoc::Ptr { idx: 0 },
                FieldLoc::Raw { offset: 0, repr: ScalarRepr::U64 },
                FieldLoc::Raw { offset: 8, repr: ScalarRepr::U64 },
            ],
            name: if traced { format!("{key}#traced") } else { key.clone() },
            elem_stride: 0,
        })
    }
    /// A varlen array layout for the given element repr. Pointer elements use a
    /// traced `Values` tail; scalar elements use an untraced `Bytes` tail whose
    /// element stride is the scalar's byte size (encoded via raw_bytes=stride).
    pub fn array_for(&mut self, elem: &Repr) -> LayoutId {
        let traced = matches!(elem, Repr::Ref(_));
        let stride = match elem {
            Repr::Ref(_) => 8u16,
            Repr::Scalar(s) => (s.bits().max(8) / 8) as u16,
            Repr::Value(vid) => self.values[*vid as usize].size as u16,
            Repr::Unit => 8,
        };
        let key = format!("Array<{}>", repr_key(elem));
        if let Some(id) = self.ref_ids.get(&key) {
            return *id;
        }
        let id = self.layouts.len() as LayoutId;
        self.layouts.push(Layout {
            ptr_fields: 0,
            raw_bytes: 0,
            // Reference elements → traced 8-byte `Values` tail (count = n).
            // Scalar elements → untraced `Bytes` tail; codegen addresses by the
            // element stride, count word = n * stride bytes.
            varlen: if traced { VarLen::Values } else { VarLen::Bytes },
            field_map: vec![],
            name: key.clone(),
            elem_stride: stride,
        });
        self.ref_ids.insert(key, id);
        id
    }

    fn array_layout(&mut self, elem: &Repr, n: u64) -> LayoutId {
        let key = format!("[{};{}]", repr_key(elem), n);
        let traced = matches!(elem, Repr::Ref(_));
        let stride = match elem { Repr::Ref(_) => 8u16, Repr::Scalar(s) => (s.bits().max(8) / 8) as u16, Repr::Value(vid) => self.values[*vid as usize].size as u16, Repr::Unit => 8 };
        self.intern_ref(&key, Layout {
            ptr_fields: 0,
            raw_bytes: 0,
            varlen: if traced { VarLen::Values } else { VarLen::Bytes },
            field_map: vec![],
            name: key.clone(),
            elem_stride: stride,
        })
    }
    fn closure_placeholder(&mut self) -> LayoutId {
        // Filled per-closure during closure lowering; a generic env placeholder.
        self.intern_ref("<closure-env>", placeholder_layout("<closure-env>"))
    }

    /// Build a closure-environment layout: pointer captures first (traced),
    /// then a raw section holding [code_ptr: u64][scalar captures...]. The
    /// returned layout's `field_map` is empty (codegen addresses by section
    /// offset using `capture_kinds`). `key` makes distinct closures distinct.
    pub fn closure_env(&mut self, key: &str, captures: &[Repr]) -> LayoutId {
        let mut ptr_fields = 0u16;
        let mut raw_bytes = 8u16; // code pointer
        for c in captures {
            match c {
                Repr::Ref(_) => ptr_fields += 1,
                Repr::Scalar(s) => raw_bytes += (s.bits().max(8) / 8) as u16,
                Repr::Value(vid) => raw_bytes += self.values[*vid as usize].size as u16,
                Repr::Unit => {}
            }
        }
        let id = self.layouts.len() as LayoutId;
        self.layouts.push(Layout {
            ptr_fields,
            raw_bytes: align_up(raw_bytes as u32, 8) as u16,
            varlen: VarLen::None,
            field_map: vec![],
            name: key.to_string(),
            elem_stride: 0,
        });
        id
    }

    fn intern_ref(&mut self, key: &str, layout: Layout) -> LayoutId {
        if let Some(id) = self.ref_ids.get(key) {
            return *id;
        }
        let id = self.layouts.len() as LayoutId;
        self.layouts.push(layout);
        self.ref_ids.insert(key.to_string(), id);
        id
    }

    // ---- field counting + layout building ---------------------------------
    fn count_fields(&mut self, tys: &[Ty]) -> R<(u16, u16)> {
        let mut ptrs = 0u16;
        let mut raw = 0u16;
        for t in tys {
            match self.repr(t)? {
                Repr::Ref(_) => ptrs += 1,
                Repr::Scalar(s) => raw += (s.bits().max(8) / 8) as u16,
                Repr::Value(vid) => raw += self.values[vid as usize].size as u16,
                Repr::Unit => {}
            }
        }
        Ok((ptrs, align_up(raw as u32, 8) as u16))
    }

    fn build_ref_layout(&mut self, name: &str, field_tys: &[Ty]) -> R<Layout> {
        // Pointer fields first (traced), then raw scalar/value bytes.
        let mut ptr_idx = 0u16;
        let mut raw_off = 0u16;
        let mut field_map = Vec::with_capacity(field_tys.len());
        // First pass: place pointers.
        let reprs: Vec<Repr> = field_tys.iter().map(|t| self.repr(t)).collect::<R<_>>()?;
        for r in &reprs {
            if let Repr::Ref(_) = r {
                field_map.push(FieldLoc::Ptr { idx: ptr_idx });
                ptr_idx += 1;
            } else {
                field_map.push(FieldLoc::Raw { offset: 0, repr: ScalarRepr::I64 }); // patched below
            }
        }
        // Second pass: place raw fields, patching the placeholders.
        for (i, r) in reprs.iter().enumerate() {
            match r {
                Repr::Ref(_) | Repr::Unit => {}
                Repr::Scalar(s) => {
                    let sz = (s.bits().max(8) / 8) as u16;
                    raw_off = align_up(raw_off as u32, sz as u32) as u16;
                    field_map[i] = FieldLoc::Raw { offset: raw_off, repr: *s };
                    raw_off += sz;
                }
                Repr::Value(vid) => {
                    let sz = self.values[*vid as usize].size as u16;
                    raw_off = align_up(raw_off as u32, 8) as u16;
                    // Value-typed field stored as raw bytes; codegen knows the ValueId.
                    field_map[i] = FieldLoc::Raw { offset: raw_off, repr: ScalarRepr::I64 };
                    raw_off += sz;
                }
            }
        }
        Ok(Layout {
            ptr_fields: ptr_idx,
            raw_bytes: align_up(raw_off as u32, 8) as u16,
            varlen: VarLen::None,
            field_map,
            name: name.to_string(),
            elem_stride: 0,
        })
    }
}

// ---- free helpers ----------------------------------------------------------

fn placeholder_layout(name: &str) -> Layout {
    Layout { ptr_fields: 0, raw_bytes: 0, varlen: VarLen::None, field_map: vec![], name: name.to_string(), elem_stride: 0 }
}

fn struct_field_tys(s: &crate::ast::StructDef, args: &[Ty]) -> R<Vec<Ty>> {
    let gparams: Vec<String> = s.generics.params.iter().map(|p| p.name.clone()).collect();
    let subst = build_subst(&gparams, args);
    match &s.body {
        StructBody::Named(fields) => fields.iter().map(|f| surface_to_ground(&f.ty, &subst)).collect(),
        StructBody::Tuple(tys) => tys.iter().map(|t| surface_to_ground(t, &subst)).collect(),
        StructBody::Unit => Ok(vec![]),
    }
}

fn variant_payload_tys(v: &crate::ast::VariantDef, generics: &crate::ast::Generics, args: &[Ty]) -> R<Vec<Ty>> {
    let gparams: Vec<String> = generics.params.iter().map(|p| p.name.clone()).collect();
    let subst = build_subst(&gparams, args);
    match &v.payload {
        VariantPayload::None => Ok(vec![]),
        VariantPayload::Tuple(tys) => tys.iter().map(|t| surface_to_ground(t, &subst)).collect(),
        VariantPayload::Named(fields) => fields.iter().map(|f| surface_to_ground(&f.ty, &subst)).collect(),
    }
}

fn build_subst(params: &[String], args: &[Ty]) -> HashMap<String, Ty> {
    params.iter().cloned().zip(args.iter().cloned()).collect()
}

/// Convert a surface type to a ground `Ty`, substituting generic params.
/// (A light reimplementation of `types::lower_type` that also applies a subst —
/// the monomorphizer will supersede this with a proper instantiation pass.)
fn surface_to_ground(t: &crate::ast::Type, subst: &HashMap<String, Ty>) -> R<Ty> {
    use crate::ast::TypeKind;
    match &t.kind {
        TypeKind::Path(path, targs) => {
            let name = path.last();
            if let Some(sub) = subst.get(name) {
                return Ok(sub.clone());
            }
            if let Some(p) = prim_of(name) {
                return Ok(Ty::Prim(p));
            }
            let gargs: Vec<Ty> = targs.iter().map(|a| surface_to_ground(a, subst)).collect::<R<_>>()?;
            Ok(Ty::Named { name: name.to_string(), args: gargs })
        }
        TypeKind::Tuple(tys) => {
            if tys.is_empty() {
                Ok(Ty::Prim(Prim::Unit))
            } else {
                Ok(Ty::Tuple(tys.iter().map(|t| surface_to_ground(t, subst)).collect::<R<_>>()?))
            }
        }
        TypeKind::Array(elem, count) => {
            let n = match &*count.kind {
                crate::ast::ExprKind::Int(n, _) => *n,
                _ => return Err(LayoutError("array length must be a literal".into())),
            };
            Ok(Ty::Array(Box::new(surface_to_ground(elem, subst)?), n))
        }
        TypeKind::Fn(params, ret) => {
            let ps = params.iter().map(|p| surface_to_ground(p, subst)).collect::<R<_>>()?;
            let r = match ret {
                Some(r) => surface_to_ground(r, subst)?,
                None => Ty::Prim(Prim::Unit),
            };
            Ok(Ty::Fn { params: ps, ret: Box::new(r) })
        }
        TypeKind::SelfType => subst.get("Self").cloned().ok_or_else(|| LayoutError("Self outside impl".into())),
    }
}

fn prim_of(name: &str) -> Option<Prim> {
    Some(match name {
        "i8" => Prim::I8, "i16" => Prim::I16, "i32" => Prim::I32, "i64" => Prim::I64,
        "u8" => Prim::U8, "u16" => Prim::U16, "u32" => Prim::U32, "u64" => Prim::U64,
        "f32" => Prim::F32, "f64" => Prim::F64,
        "bool" => Prim::Bool, "char" => Prim::Char, "String" => Prim::Str,
        _ => return None,
    })
}

fn scalar_of(p: Prim) -> Option<ScalarRepr> {
    Some(match p {
        Prim::I8 => ScalarRepr::I8, Prim::I16 => ScalarRepr::I16,
        Prim::I32 => ScalarRepr::I32, Prim::I64 => ScalarRepr::I64,
        Prim::U8 => ScalarRepr::U8, Prim::U16 => ScalarRepr::U16,
        Prim::U32 => ScalarRepr::U32, Prim::U64 => ScalarRepr::U64,
        Prim::F32 => ScalarRepr::F32, Prim::F64 => ScalarRepr::F64,
        Prim::Bool => ScalarRepr::Bool, Prim::Char => ScalarRepr::Char,
        Prim::Str | Prim::Unit => return None,
    })
}

fn value_size_align(fields: &[Repr], values: &[ValueLayout]) -> (u32, u32) {
    let mut off = 0u32;
    let mut align = 1u32;
    for f in fields {
        let (sz, a) = match f {
            Repr::Unit => (0, 1),
            Repr::Scalar(s) => { let b = (s.bits().max(8) / 8).max(1); (b, b) }
            Repr::Ref(_) => (8, 8),
            Repr::Value(vid) => (values[*vid as usize].size, values[*vid as usize].align),
        };
        off = align_up(off, a) + sz;
        align = align.max(a);
    }
    (align_up(off, align.max(1)), align.max(1))
}

fn align_up(n: u32, a: u32) -> u32 {
    if a == 0 { n } else { (n + a - 1) & !(a - 1) }
}

fn ty_key(t: &Ty) -> String {
    match t {
        Ty::Prim(p) => format!("{:?}", p),
        Ty::Named { name, args } => if args.is_empty() { name.clone() } else {
            format!("{}<{}>", name, args.iter().map(ty_key).collect::<Vec<_>>().join(","))
        },
        Ty::Var(v) => format!("'{}", v),
        Ty::Array(e, n) => format!("[{};{}]", ty_key(e), n),
        Ty::Tuple(es) => format!("({})", es.iter().map(ty_key).collect::<Vec<_>>().join(",")),
        Ty::Fn { params, ret } => format!("fn({})->{}", params.iter().map(ty_key).collect::<Vec<_>>().join(","), ty_key(ret)),
        Ty::Infer(n) => format!("?{}", n),
    }
}

fn repr_key(r: &Repr) -> String {
    match r {
        Repr::Unit => "()".into(),
        Repr::Scalar(s) => format!("{:?}", s),
        Repr::Ref(id) => format!("ref{}", id),
        Repr::Value(id) => format!("val{}", id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;

    fn ctx(src: &str) -> TyCtx {
        let m = parse_module(&lex(src).unwrap()).unwrap();
        let r = resolve_module(m).unwrap();
        TyCtx::from_globals(&r.globals)
    }

    #[test]
    fn value_struct_is_inline() {
        let c = ctx("value struct Vec3 { x: f64, y: f64, z: f64 }");
        let mut reg = LayoutRegistry::new(&c);
        let r = reg.repr(&Ty::Named { name: "Vec3".into(), args: vec![] }).unwrap();
        let Repr::Value(id) = r else { panic!("Vec3 should be a value type") };
        let v = &reg.values[id as usize];
        assert_eq!(v.fields.len(), 3);
        assert_eq!(v.size, 24);
    }

    #[test]
    fn ref_struct_ptr_fields_first() {
        // a struct with a scalar and a reference field: ptr must be field 0 slot.
        let c = ctx("struct Node { val: i64, next: Node }");
        let mut reg = LayoutRegistry::new(&c);
        let r = reg.repr(&Ty::Named { name: "Node".into(), args: vec![] }).unwrap();
        let Repr::Ref(id) = r else { panic!("Node is a reference type") };
        let l = &reg.layouts[id as usize];
        assert_eq!(l.ptr_fields, 1, "next is a traced pointer");
        // val:i64 lives in raw region.
        assert!(matches!(l.field_map[0], FieldLoc::Raw { repr: ScalarRepr::I64, .. }));
        assert!(matches!(l.field_map[1], FieldLoc::Ptr { idx: 0 }));
    }

    #[test]
    fn generic_struct_monomorphizes_layout() {
        let c = ctx("struct Pair<A, B> { a: A, b: B }");
        let mut reg = LayoutRegistry::new(&c);
        let r = reg.repr(&Ty::Named {
            name: "Pair".into(),
            args: vec![Ty::Prim(Prim::I64), Ty::Prim(Prim::F64)],
        }).unwrap();
        let Repr::Ref(id) = r else { panic!() };
        let l = &reg.layouts[id as usize];
        assert_eq!(l.ptr_fields, 0);
        assert_eq!(l.raw_bytes, 16); // i64 + f64
    }

    #[test]
    fn option_layout() {
        let c = ctx("");
        let mut reg = LayoutRegistry::new(&c);
        let r = reg.repr(&Ty::Named { name: "Option".into(), args: vec![Ty::Prim(Prim::I64)] }).unwrap();
        assert!(matches!(r, Repr::Ref(_)));
    }
}
