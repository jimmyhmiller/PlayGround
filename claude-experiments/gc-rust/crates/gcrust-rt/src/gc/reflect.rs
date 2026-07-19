//! Runtime reflection metadata — the *nominal* type information that the GC
//! `TypeInfo` table deliberately omits.
//!
//! `TypeInfo` (see [`super::type_info`]) describes an object's **shape**: how
//! many pointer slots, how many raw bytes, whether it has a varlen tail. That's
//! all the collector needs to trace. It carries no names: two structs with the
//! same shape are indistinguishable to the GC.
//!
//! [`TypeMeta`] is the parallel, *cold* table — one entry per `type_id`, holding
//! the source type name, field names, field byte offsets, field types, and (for
//! enums) per-variant payloads. It is what heap-exploration tooling and
//! in-language reflection read to render an object as `Point { x: 3, y: 4 }`
//! instead of `<type 7: 2 fields>`.
//!
//! The table is built by the compiler (`src/layout.rs`) alongside the layouts,
//! and reaches the runtime by two routes that must agree:
//!   - **JIT**: handed directly to [`super::heap::Heap::set_type_meta`].
//!   - **AOT**: [`encode`]d into a byte blob baked into the executable, then
//!     [`decode`]d at startup by `gcr_runtime_main`.
//!
//! Field byte offsets are **absolute from the start of the object** (i.e. they
//! include the 16-byte `Full` header), so a tool with only an object pointer and
//! a `FieldMeta` can read the field directly.

/// Reflection metadata for one heap layout. Indexed by `type_id` (its position
/// in the table equals the `TypeInfo.type_id` it describes).
#[derive(Clone, Debug, PartialEq)]
pub struct TypeMeta {
    pub type_id: u16,
    /// Source-level (monomorphized) name, e.g. `Point` or `List<i64>`.
    pub name: String,
    pub kind: TypeKind,
}

/// Compile-time descriptor for one allocation site (Target-1b allocation-site
/// profiling). Baked by the compiler (one entry per unique
/// `(function, allocated-type)` pair — the honest v1 granularity, since Core IR
/// carries no source span to distinguish two allocations of the same type in
/// the same function) and installed at startup via
/// [`super::heap::Heap::set_alloc_sites`]. The runtime's per-site counters are
/// indexed by this entry's position in the table (the `site_id` passed to
/// `ai_gc_alloc_*`).
#[derive(Clone, Debug, PartialEq)]
pub struct AllocSite {
    /// The function containing the allocation site (its compiled symbol name).
    pub function: String,
    /// The `type_id` (= `LayoutId`) of the object allocated here. Pairs with
    /// [`TypeMeta`] to recover the source type name in a profile.
    pub type_id: u16,
    /// Source location `file:line:col` (debugger P1 span-threading), or empty if
    /// no span/source was available. Two same-(function,type) allocations at
    /// different lines are distinct sites once this is populated.
    pub location: String,
}

/// The shape category of a type, with the nominal detail the GC shape lacks.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    /// A heap struct: a fixed, named (or positional) set of fields.
    Struct { fields: Vec<FieldMeta> },
    /// A heap enum: a u32 tag at `tag_offset` followed by the active variant's
    /// payload. Every variant's fields are laid out in the *same* object.
    Enum {
        tag_offset: u16,
        variants: Vec<VariantMeta>,
    },
    /// A builtin / opaque / varlen object (String, Array, closure env, …) whose
    /// contents the renderer treats as a blob rather than named fields.
    Opaque,
}

/// One enum variant: its name, discriminant, and payload fields (which share
/// the enclosing object's storage with the other variants).
#[derive(Clone, Debug, PartialEq)]
pub struct VariantMeta {
    pub name: String,
    pub tag: u32,
    pub fields: Vec<FieldMeta>,
}

/// One field: where it lives and what it holds.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldMeta {
    /// `x` for a named field; `"0"`, `"1"`, … for tuple / positional payloads.
    pub name: String,
    /// Byte offset of the field. In a [`TypeMeta`] this is absolute from the
    /// object pointer (header included); in a [`ValueMeta`] it is relative to the
    /// start of that value aggregate. The renderer composes the two when it
    /// descends into a flattened value field.
    pub offset: u16,
    pub ty: FieldTy,
}

/// What a field holds — enough to decode its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldTy {
    /// A GC pointer to a heap object of the given `type_id`.
    Ref(u16),
    /// An unboxed scalar.
    Scalar(ScalarKind),
    /// An inline `value struct`/`value enum` aggregate stored flat, described by
    /// the [`ValueMeta`] at the given `value_id`. The renderer recurses into it,
    /// adding this field's offset to the value's own (value-relative) offsets.
    Value(u16),
}

/// Reflection metadata for an inline `#[value]` aggregate (a value struct or
/// value enum). Indexed by `value_id`. Unlike [`TypeMeta`], a value is never a
/// heap object on its own — it only appears flattened inside a heap object's
/// fields — so its `FieldMeta` offsets are value-relative (see [`FieldMeta`]).
#[derive(Clone, Debug, PartialEq)]
pub struct ValueMeta {
    pub value_id: u16,
    pub name: String,
    /// `Struct`/`Enum` only (never `Opaque`); reuses [`TypeKind`] for its field
    /// and variant shapes.
    pub kind: TypeKind,
}

/// Scalar field kinds — a runtime mirror of the compiler's `core::ScalarRepr`
/// (kept here so gcrust-rt stays LLVM-/compiler-free).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarKind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    Char,
    /// FFI raw pointer (`RawPtr`).
    Ptr,
}

impl ScalarKind {
    /// Size in bytes of this scalar in an object's raw region.
    pub const fn size(self) -> usize {
        match self {
            ScalarKind::I8 | ScalarKind::U8 | ScalarKind::Bool => 1,
            ScalarKind::I16 | ScalarKind::U16 => 2,
            ScalarKind::I32 | ScalarKind::U32 | ScalarKind::F32 | ScalarKind::Char => 4,
            ScalarKind::I64
            | ScalarKind::U64
            | ScalarKind::F64
            | ScalarKind::Ptr => 8,
        }
    }

    /// Stable wire tag used by [`encode`]/[`decode`].
    const fn to_tag(self) -> u8 {
        match self {
            ScalarKind::I8 => 0,
            ScalarKind::I16 => 1,
            ScalarKind::I32 => 2,
            ScalarKind::I64 => 3,
            ScalarKind::U8 => 4,
            ScalarKind::U16 => 5,
            ScalarKind::U32 => 6,
            ScalarKind::U64 => 7,
            ScalarKind::F32 => 8,
            ScalarKind::F64 => 9,
            ScalarKind::Bool => 10,
            ScalarKind::Char => 11,
            ScalarKind::Ptr => 12,
        }
    }

    fn from_tag(t: u8) -> ScalarKind {
        match t {
            0 => ScalarKind::I8,
            1 => ScalarKind::I16,
            2 => ScalarKind::I32,
            3 => ScalarKind::I64,
            4 => ScalarKind::U8,
            5 => ScalarKind::U16,
            6 => ScalarKind::U32,
            7 => ScalarKind::U64,
            8 => ScalarKind::F32,
            9 => ScalarKind::F64,
            10 => ScalarKind::Bool,
            11 => ScalarKind::Char,
            12 => ScalarKind::Ptr,
            other => panic!("reflect::decode: invalid ScalarKind tag {other}"),
        }
    }

    /// Human name as it appears in a type rendering (`i64`, `f64`, `bool`, …).
    pub const fn as_str(self) -> &'static str {
        match self {
            ScalarKind::I8 => "i8",
            ScalarKind::I16 => "i16",
            ScalarKind::I32 => "i32",
            ScalarKind::I64 => "i64",
            ScalarKind::U8 => "u8",
            ScalarKind::U16 => "u16",
            ScalarKind::U32 => "u32",
            ScalarKind::U64 => "u64",
            ScalarKind::F32 => "f32",
            ScalarKind::F64 => "f64",
            ScalarKind::Bool => "bool",
            ScalarKind::Char => "char",
            ScalarKind::Ptr => "ptr",
        }
    }
}

// ============================================================================
// Binary serialization (for AOT: baked into the executable, decoded at startup)
// ============================================================================
//
// A self-describing little-endian format. The compiler's `encode` and the
// runtime's `decode` are the same code, so the two compilation paths can never
// drift. Layout:
//
//   u32  type_count
//   repeat type_count:
//     u16  type_id
//     str  name                       (u16 len + UTF-8 bytes)
//     u8   kind tag (0=Struct 1=Enum 2=Opaque)
//     Struct: field_list
//     Enum:   u16 tag_offset
//             u16 variant_count
//             repeat: str name; u32 tag; field_list
//     Opaque: (nothing)
//
// After the type table comes the value table (inline `#[value]` aggregates),
// in the same per-entry shape but keyed by `value_id`:
//
//   u32  value_count
//   repeat value_count: u16 value_id; str name; u8 kind tag (0=Struct 1=Enum);
//                       Struct/Enum body as above
//
//   field_list = u16 count; repeat: str name; u16 offset; field_ty
//   field_ty   = u8 tag (0=Ref 1=Scalar 2=Value)
//                Ref:    u16 target type_id
//                Scalar: u8 scalar tag
//                Value:  u16 value_id

fn put_str(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    assert!(bytes.len() <= u16::MAX as usize, "reflect: name too long");
    out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn put_field(out: &mut Vec<u8>, f: &FieldMeta) {
    put_str(out, &f.name);
    out.extend_from_slice(&f.offset.to_le_bytes());
    match f.ty {
        FieldTy::Ref(tid) => {
            out.push(0);
            out.extend_from_slice(&tid.to_le_bytes());
        }
        FieldTy::Scalar(s) => {
            out.push(1);
            out.push(s.to_tag());
        }
        FieldTy::Value(vid) => {
            out.push(2);
            out.extend_from_slice(&vid.to_le_bytes());
        }
    }
}

/// Encode one type/value entry body (the `kind` after `id` + `name`).
fn put_kind(out: &mut Vec<u8>, kind: &TypeKind) {
    match kind {
        TypeKind::Struct { fields } => {
            out.push(0);
            put_field_list(out, fields);
        }
        TypeKind::Enum { tag_offset, variants } => {
            out.push(1);
            out.extend_from_slice(&tag_offset.to_le_bytes());
            assert!(variants.len() <= u16::MAX as usize, "reflect: too many variants");
            out.extend_from_slice(&(variants.len() as u16).to_le_bytes());
            for v in variants {
                put_str(out, &v.name);
                out.extend_from_slice(&v.tag.to_le_bytes());
                put_field_list(out, &v.fields);
            }
        }
        TypeKind::Opaque => out.push(2),
    }
}

fn put_field_list(out: &mut Vec<u8>, fields: &[FieldMeta]) {
    assert!(fields.len() <= u16::MAX as usize, "reflect: too many fields");
    out.extend_from_slice(&(fields.len() as u16).to_le_bytes());
    for f in fields {
        put_field(out, f);
    }
}

/// Encode the type table, value table, per-type interior-pointer offsets, and
/// the allocation-site table into the self-describing byte format above.
/// Decoded by [`decode`] (the same code, so the JIT/AOT paths can't drift).
/// `interior[i]` lists the absolute byte offsets of GC refs embedded in
/// flattened value fields of type `i` (used to set
/// `gc::TypeInfo::interior_ptrs` for the AOT type table). `alloc_sites[k]`
/// describes the allocation site whose compile-time id is `k`.
pub fn encode(
    types: &[TypeMeta],
    values: &[ValueMeta],
    interior: &[Vec<u16>],
    alloc_sites: &[AllocSite],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(types.len() as u32).to_le_bytes());
    for m in types {
        out.extend_from_slice(&m.type_id.to_le_bytes());
        put_str(&mut out, &m.name);
        put_kind(&mut out, &m.kind);
    }
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    for v in values {
        out.extend_from_slice(&v.value_id.to_le_bytes());
        put_str(&mut out, &v.name);
        put_kind(&mut out, &v.kind);
    }
    out.extend_from_slice(&(interior.len() as u32).to_le_bytes());
    for offs in interior {
        assert!(offs.len() <= u16::MAX as usize, "reflect: too many interior ptrs");
        out.extend_from_slice(&(offs.len() as u16).to_le_bytes());
        for &o in offs {
            out.extend_from_slice(&o.to_le_bytes());
        }
    }
    out.extend_from_slice(&(alloc_sites.len() as u32).to_le_bytes());
    for s in alloc_sites {
        put_str(&mut out, &s.function);
        out.extend_from_slice(&s.type_id.to_le_bytes());
        put_str(&mut out, &s.location);
    }
    out
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.buf[self.pos];
        self.pos += 1;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
        self.pos += 2;
        v
    }
    fn u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        u32::from_le_bytes(b)
    }
    fn string(&mut self) -> String {
        let len = self.u16() as usize;
        let s = core::str::from_utf8(&self.buf[self.pos..self.pos + len])
            .expect("reflect::decode: invalid UTF-8 in name")
            .to_string();
        self.pos += len;
        s
    }
    fn field(&mut self) -> FieldMeta {
        let name = self.string();
        let offset = self.u16();
        let ty = match self.u8() {
            0 => FieldTy::Ref(self.u16()),
            1 => FieldTy::Scalar(ScalarKind::from_tag(self.u8())),
            2 => FieldTy::Value(self.u16()),
            other => panic!("reflect::decode: invalid FieldTy tag {other}"),
        };
        FieldMeta { name, offset, ty }
    }
    fn field_list(&mut self) -> Vec<FieldMeta> {
        let n = self.u16() as usize;
        (0..n).map(|_| self.field()).collect()
    }
    fn kind(&mut self) -> TypeKind {
        match self.u8() {
            0 => TypeKind::Struct { fields: self.field_list() },
            1 => {
                let tag_offset = self.u16();
                let vcount = self.u16() as usize;
                let variants = (0..vcount)
                    .map(|_| {
                        let name = self.string();
                        let tag = self.u32();
                        let fields = self.field_list();
                        VariantMeta { name, tag, fields }
                    })
                    .collect();
                TypeKind::Enum { tag_offset, variants }
            }
            2 => TypeKind::Opaque,
            other => panic!("reflect::decode: invalid TypeKind tag {other}"),
        }
    }
}

/// Decode the type table, value table, per-type interior-pointer offsets, and
/// allocation-site table previously produced by [`encode`]. Panics (rather than
/// silently truncating) on a malformed blob — a corrupt table means a build/link
/// bug, not a recoverable condition.
pub fn decode(bytes: &[u8]) -> (Vec<TypeMeta>, Vec<ValueMeta>, Vec<Vec<u16>>, Vec<AllocSite>) {
    let mut r = Reader { buf: bytes, pos: 0 };
    let tcount = r.u32() as usize;
    let mut types = Vec::with_capacity(tcount);
    for _ in 0..tcount {
        let type_id = r.u16();
        let name = r.string();
        let kind = r.kind();
        types.push(TypeMeta { type_id, name, kind });
    }
    let vcount = r.u32() as usize;
    let mut values = Vec::with_capacity(vcount);
    for _ in 0..vcount {
        let value_id = r.u16();
        let name = r.string();
        let kind = r.kind();
        values.push(ValueMeta { value_id, name, kind });
    }
    let icount = r.u32() as usize;
    let mut interior = Vec::with_capacity(icount);
    for _ in 0..icount {
        let n = r.u16() as usize;
        interior.push((0..n).map(|_| r.u16()).collect());
    }
    let scount = r.u32() as usize;
    let mut alloc_sites = Vec::with_capacity(scount);
    for _ in 0..scount {
        let function = r.string();
        let type_id = r.u16();
        let location = r.string();
        alloc_sites.push(AllocSite { function, type_id, location });
    }
    (types, values, interior, alloc_sites)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<TypeMeta> {
        vec![
            TypeMeta {
                type_id: 0,
                name: "String".into(),
                kind: TypeKind::Opaque,
            },
            TypeMeta {
                type_id: 1,
                name: "Point".into(),
                kind: TypeKind::Struct {
                    fields: vec![
                        FieldMeta {
                            name: "x".into(),
                            offset: 16,
                            ty: FieldTy::Scalar(ScalarKind::I64),
                        },
                        FieldMeta {
                            name: "y".into(),
                            offset: 24,
                            ty: FieldTy::Scalar(ScalarKind::I64),
                        },
                    ],
                },
            },
            TypeMeta {
                type_id: 2,
                name: "List<i64>".into(),
                kind: TypeKind::Enum {
                    tag_offset: 24,
                    variants: vec![
                        VariantMeta {
                            name: "Nil".into(),
                            tag: 0,
                            fields: vec![],
                        },
                        VariantMeta {
                            name: "Cons".into(),
                            tag: 1,
                            fields: vec![
                                FieldMeta {
                                    name: "0".into(),
                                    offset: 16,
                                    ty: FieldTy::Scalar(ScalarKind::I64),
                                },
                                FieldMeta {
                                    name: "1".into(),
                                    offset: 16,
                                    ty: FieldTy::Ref(2),
                                },
                            ],
                        },
                    ],
                },
            },
        ]
    }

    fn sample_values() -> Vec<ValueMeta> {
        vec![ValueMeta {
            value_id: 0,
            name: "Vec3".into(),
            kind: TypeKind::Struct {
                fields: vec![
                    FieldMeta { name: "x".into(), offset: 0, ty: FieldTy::Scalar(ScalarKind::F64) },
                    FieldMeta { name: "y".into(), offset: 8, ty: FieldTy::Scalar(ScalarKind::F64) },
                    // A nested value field, to exercise FieldTy::Value(value_id).
                    FieldMeta { name: "inner".into(), offset: 16, ty: FieldTy::Value(0) },
                ],
            },
        }]
    }

    fn sample_alloc_sites() -> Vec<AllocSite> {
        vec![
            AllocSite { function: "main".into(), type_id: 1, location: "a.gcr:3:9".into() },
            AllocSite { function: "build_list".into(), type_id: 2, location: "a.gcr:7:5".into() },
            // Same type allocated in a different function => distinct site.
            AllocSite { function: "helper".into(), type_id: 1, location: String::new() },
        ]
    }

    #[test]
    fn encode_decode_roundtrip() {
        let types = sample();
        let values = sample_values();
        let interior = vec![vec![], vec![16u16], vec![16u16, 24u16]];
        let alloc_sites = sample_alloc_sites();
        let bytes = encode(&types, &values, &interior, &alloc_sites);
        let (t, v, i, s) = decode(&bytes);
        assert_eq!(types, t);
        assert_eq!(values, v);
        assert_eq!(interior, i);
        assert_eq!(alloc_sites, s);
    }

    #[test]
    fn empty_table_roundtrips() {
        let (t, v, i, s) = decode(&encode(&[], &[], &[], &[]));
        assert_eq!(t, Vec::<TypeMeta>::new());
        assert_eq!(v, Vec::<ValueMeta>::new());
        assert_eq!(i, Vec::<Vec<u16>>::new());
        assert_eq!(s, Vec::<AllocSite>::new());
    }
}