//! C-ABI struct-by-value classification.
//!
//! When a struct crosses the C boundary by value (an `extern`/`c`-convention
//! parameter or return), the System V AMD64 and AArch64 AAPCS64 ABIs do *not*
//! pass a pointer to it: small structs are coerced into registers and only large
//! ones are passed indirectly (`byval`/`sret`). This module computes, for a given
//! struct on a given architecture, exactly the LLVM-level coercion clang emits —
//! the coerced register type(s) for small structs, or the indirect marker for
//! large ones — so codegen can match the C ABI byte-for-byte.
//!
//! `clang -arch <a> -S -emit-llvm` over the equivalent C is the oracle these
//! rules were derived against.

use crate::codegen::int_width;
use inkwell::context::Context;
use inkwell::targets::TargetData;
use inkwell::types::{BasicType, BasicTypeEnum, StructType};

use crate::ast::Type;

/// The target architecture, selecting which ABI classifier to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    AArch64,
}

/// How a single struct argument or return value is realized at the LLVM level.
#[derive(Debug, Clone)]
pub enum Class<'ctx> {
    /// Passed/returned directly in registers, coerced to these LLVM types. One
    /// entry per ABI register slot:
    ///  * a SysV argument expands to one parameter per eightbyte;
    ///  * a SysV return with two eightbytes is wrapped in a `{T0, T1}` literal
    ///    struct (handled by the caller via `Class::direct_return_type`);
    ///  * AArch64 always coerces to a single type (an `iN`, `[N x i64]`,
    ///    `[N x fT]`, or — for an HFA return — the struct type itself).
    Direct(Vec<BasicTypeEnum<'ctx>>),
    /// Passed indirectly: the argument is a pointer to a caller-allocated copy
    /// (`byval` on x86-64, a plain pointer on AArch64), and a return uses a
    /// hidden `sret` pointer. `align` is the struct's ABI alignment.
    Indirect { align: u32 },
}

impl<'ctx> Class<'ctx> {
    pub fn is_indirect(&self) -> bool {
        matches!(self, Class::Indirect { .. })
    }
}

/// The classification of a struct in both argument and return position. The two
/// can differ (e.g. an AArch64 HFA returns as its struct type but passes as an
/// `[N x fT]` array; a SysV two-eightbyte struct passes as two params but returns
/// as one `{T0,T1}` value).
pub struct StructAbi<'ctx> {
    /// LLVM coercion for an argument.
    pub arg: Class<'ctx>,
    /// LLVM coercion for a return value.
    pub ret: Class<'ctx>,
    /// The struct's LLVM type (used for `byval(T)`/`sret(T)` and copies).
    pub llvm_ty: StructType<'ctx>,
    /// ABI size in bytes.
    pub size: u64,
    /// ABI alignment in bytes.
    pub align: u32,
}

impl<'ctx> StructAbi<'ctx> {
    /// The single LLVM type a `Direct` return value coerces to. A one-slot direct
    /// return is that slot's type; a two-slot SysV return is a `{T0, T1}` literal
    /// struct (matching clang). Indirect returns have no direct type.
    pub fn direct_return_type(&self, ctx: &'ctx Context) -> Option<BasicTypeEnum<'ctx>> {
        match &self.ret {
            Class::Direct(slots) => match slots.as_slice() {
                [t] => Some(*t),
                [t0, t1] => Some(ctx.struct_type(&[*t0, *t1], false).into()),
                _ => None,
            },
            Class::Indirect { .. } => None,
        }
    }
}

/// An ordered struct field list: `(field-name, field-type)` pairs.
pub type Fields = Vec<(String, Type)>;

/// Everything the classifier needs from codegen to walk a struct's layout: the
/// LLVM type of a Coil type, and (for nested aggregates) the field list of a
/// named struct. `resolve_struct` returns `None` for a non-struct name.
pub struct AbiCtx<'a, 'ctx> {
    pub field_llvm: &'a dyn Fn(&Type) -> BasicTypeEnum<'ctx>,
    pub resolve_struct: &'a dyn Fn(&str) -> Option<Fields>,
}

/// Classify `sty` (an already-built LLVM struct type, with `fields` its Coil
/// field list) for the given architecture.
pub fn classify<'ctx>(
    arch: Arch,
    ctx: &'ctx Context,
    td: &TargetData,
    fields: &[(String, Type)],
    sty: StructType<'ctx>,
    abi: &AbiCtx<'_, 'ctx>,
) -> Result<StructAbi<'ctx>, String> {
    let size = td.get_abi_size(&BasicTypeEnum::StructType(sty));
    let align = td.get_abi_alignment(&BasicTypeEnum::StructType(sty));
    match arch {
        Arch::X86_64 => classify_sysv(ctx, td, fields, sty, size, align, abi),
        Arch::AArch64 => classify_aapcs64(ctx, fields, sty, size, align, abi),
    }
}

// ---------------------------------------------------------------------------
// System V AMD64
// ---------------------------------------------------------------------------

/// The SysV class of one byte of the struct. `Sse` byte are floating-point;
/// `Integer` covers everything else that holds data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteClass {
    None,
    Integer,
    Sse,
}

impl ByteClass {
    /// The SysV "merge" rule for two classes covering the same byte: INTEGER
    /// dominates SSE, SSE dominates NONE.
    fn merge(self, other: ByteClass) -> ByteClass {
        match (self, other) {
            (ByteClass::Integer, _) | (_, ByteClass::Integer) => ByteClass::Integer,
            (ByteClass::Sse, _) | (_, ByteClass::Sse) => ByteClass::Sse,
            _ => ByteClass::None,
        }
    }
}

/// One scalar leaf of a (possibly nested) struct: its byte offset, byte size,
/// and SysV class. Produced by flattening the field tree.
#[derive(Debug, Clone, Copy)]
struct Leaf {
    offset: u64,
    size: u64,
    class: ByteClass,
    is_f32: bool,
}

fn classify_sysv<'ctx>(
    ctx: &'ctx Context,
    td: &TargetData,
    fields: &[(String, Type)],
    sty: StructType<'ctx>,
    size: u64,
    align: u32,
    abi: &AbiCtx<'_, 'ctx>,
) -> Result<StructAbi<'ctx>, String> {
    // Structs larger than two eightbytes (16 bytes) are passed in memory.
    if size > 16 {
        return Ok(StructAbi {
            arg: Class::Indirect { align },
            ret: Class::Indirect { align },
            llvm_ty: sty,
            size,
            align,
        });
    }

    // Flatten to scalar leaves, then per-byte-classify and collapse into
    // eightbytes (the SysV field-walk + merge rules).
    let mut leaves: Vec<Leaf> = Vec::new();
    flatten(td, fields, 0, &mut leaves, abi)?;
    let nbytes = size as usize;
    let mut bytes = vec![ByteClass::None; nbytes];
    for lf in &leaves {
        let lo = lf.offset as usize;
        let hi = (lf.offset + lf.size) as usize;
        if hi > nbytes {
            return Err(format!(
                "SysV ABI: scalar at byte {lo} of size {} runs past struct size {nbytes}",
                lf.size
            ));
        }
        for b in &mut bytes[lo..hi] {
            *b = b.merge(lf.class);
        }
    }

    let neb = nbytes.div_ceil(8);
    let mut slots: Vec<BasicTypeEnum<'ctx>> = Vec::with_capacity(neb);
    for eb in 0..neb {
        let lo = eb * 8;
        let hi = (lo + 8).min(nbytes);
        let mut cls = ByteClass::None;
        for b in &bytes[lo..hi] {
            cls = cls.merge(*b);
        }
        let used = (hi - lo) as u32; // 1..=8 data bytes in this eightbyte
        let slot: BasicTypeEnum = match cls {
            ByteClass::Sse => {
                // An SSE eightbyte fully occupied by two distinct 4-byte floats
                // coerces to `<2 x float>`; one double (or 5..8 SSE bytes) to
                // `double`; a single low-half 4-byte float to `float`.
                let lo64 = lo as u64;
                let two_floats = leaves.iter().any(|l| l.is_f32 && l.offset == lo64)
                    && leaves.iter().any(|l| l.is_f32 && l.offset == lo64 + 4);
                if used == 8 && two_floats {
                    ctx.f32_type().vec_type(2).into()
                } else if used > 4 {
                    ctx.f64_type().into()
                } else {
                    ctx.f32_type().into()
                }
            }
            // An INTEGER (or padding-only) eightbyte coerces to the smallest
            // power-of-two integer covering its data bytes.
            ByteClass::Integer | ByteClass::None => {
                int_width(ctx, int_bits_for_bytes(used)).into()
            }
        };
        slots.push(slot);
    }

    Ok(StructAbi {
        arg: Class::Direct(slots.clone()),
        ret: Class::Direct(slots),
        llvm_ty: sty,
        size,
        align,
    })
}

/// Flatten `fields` into scalar leaves at absolute byte offsets, recursing into
/// nested structs and arrays. Field offsets mirror the C/LLVM struct layout
/// (natural alignment). An unclassifiable field is a hard error.
fn flatten<'ctx>(
    td: &TargetData,
    fields: &[(String, Type)],
    base: u64,
    out: &mut Vec<Leaf>,
    abi: &AbiCtx<'_, 'ctx>,
) -> Result<(), String> {
    let mut acc = base;
    for (fname, fty) in fields {
        let lty = (abi.field_llvm)(fty);
        let falign = td.get_abi_alignment(&lty) as u64;
        acc = align_up(acc, falign);
        flatten_type(td, fty, acc, out, abi).map_err(|e| format!("field '{fname}': {e}"))?;
        acc += td.get_abi_size(&lty);
    }
    Ok(())
}

fn flatten_type<'ctx>(
    td: &TargetData,
    ty: &Type,
    offset: u64,
    out: &mut Vec<Leaf>,
    abi: &AbiCtx<'_, 'ctx>,
) -> Result<(), String> {
    match ty {
        Type::Never => unreachable!("Never type has no ABI representation"),
        Type::Void => unreachable!("void is a return type only; never a field/by-value param"),
        Type::Float(bits) => {
            out.push(Leaf {
                offset,
                size: (*bits as u64) / 8,
                class: ByteClass::Sse,
                is_f32: *bits == 32,
            });
            Ok(())
        }
        Type::Int(..) | Type::Bool | Type::Ptr(..) | Type::Ref(..) | Type::Fn(..) => {
            out.push(Leaf {
                offset,
                size: td.get_abi_size(&(abi.field_llvm)(ty)),
                class: ByteClass::Integer,
                is_f32: false,
            });
            Ok(())
        }
        Type::Array(elem, n) => {
            let esz = td.get_abi_size(&(abi.field_llvm)(elem));
            for i in 0..*n as u64 {
                flatten_type(td, elem, offset + i * esz, out, abi)?;
            }
            Ok(())
        }
        Type::Struct(name) => {
            let nested = (abi.resolve_struct)(name).ok_or_else(|| {
                format!("SysV ABI: nested type '{name}' is not a classifiable struct")
            })?;
            flatten(td, &nested, offset, out, abi)
        }
        Type::Vec(..) => Err(
            "SysV ABI: passing a vec by value across the C boundary is not supported yet \
             (keep vectors inside Coil, or pass a pointer)".to_string(),
        ),
        Type::Slice(..) => Err(
            "SysV ABI: passing a slice by value across the C boundary is not supported \
             (slices are a Coil view type; for FFI pass a c\"…\"/(ptr i8) and a length)".to_string(),
        ),
        Type::App(..) => Err(format!(
            "SysV ABI: generic type {ty:?} survived to ABI classification (compiler bug)"
        )),
    }
}

/// The smallest power-of-two-byte integer width (in bits) covering `bytes` data
/// bytes: 1->i8, 2->i16, 3..4->i32, 5..8->i64.
fn int_bits_for_bytes(bytes: u32) -> u32 {
    match bytes {
        0 | 1 => 8,
        2 => 16,
        3 | 4 => 32,
        _ => 64,
    }
}

// ---------------------------------------------------------------------------
// AArch64 AAPCS64
// ---------------------------------------------------------------------------

fn classify_aapcs64<'ctx>(
    ctx: &'ctx Context,
    fields: &[(String, Type)],
    sty: StructType<'ctx>,
    size: u64,
    align: u32,
    abi: &AbiCtx<'_, 'ctx>,
) -> Result<StructAbi<'ctx>, String> {
    // Homogeneous Floating-point/Vector Aggregate: 1..=4 members all the same FP
    // type. Passed in v0..v7 as `[N x fT]`; returned as the struct type itself.
    if let Some((fty, n)) = hfa(fields, abi) {
        let elem = (abi.field_llvm)(&fty);
        let arr = elem.array_type(n);
        return Ok(StructAbi {
            arg: Class::Direct(vec![arr.into()]),
            // clang returns an HFA as the struct type directly.
            ret: Class::Direct(vec![sty.into()]),
            llvm_ty: sty,
            size,
            align,
        });
    }

    // Other composites > 16 bytes are passed indirectly (pointer to a copy; the
    // return uses x8/sret).
    if size > 16 {
        return Ok(StructAbi {
            arg: Class::Indirect { align },
            ret: Class::Indirect { align },
            llvm_ty: sty,
            size,
            align,
        });
    }

    // <=16 bytes: pack into x-registers. An argument always rounds each 8-byte
    // chunk up to i64 (`i64` for <=8 bytes, `[2 x i64]` for 9..16). A return
    // narrows a single sub-8-byte chunk to its natural width (i8/i16/i32/i64).
    let arg = if size <= 8 {
        Class::Direct(vec![ctx.i64_type().into()])
    } else {
        Class::Direct(vec![ctx.i64_type().array_type(2).into()])
    };
    let ret = if size <= 8 {
        Class::Direct(vec![int_width(ctx, int_bits_for_bytes(size as u32)).into()])
    } else {
        Class::Direct(vec![ctx.i64_type().array_type(2).into()])
    };
    Ok(StructAbi {
        arg,
        ret,
        llvm_ty: sty,
        size,
        align,
    })
}

/// If `fields` form a Homogeneous Floating-point Aggregate — 1..=4 members all
/// the *same* floating-point type (recursing through nested structs/arrays of FP)
/// — return that FP type and the member count.
fn hfa(fields: &[(String, Type)], abi: &AbiCtx<'_, '_>) -> Option<(Type, u32)> {
    let mut found: Option<Type> = None;
    let mut count: u32 = 0;
    if !hfa_walk(fields, &mut found, &mut count, abi) {
        return None;
    }
    match found {
        Some(fty) if (1..=4).contains(&count) => Some((fty, count)),
        _ => None,
    }
}

fn hfa_walk(
    fields: &[(String, Type)],
    found: &mut Option<Type>,
    count: &mut u32,
    abi: &AbiCtx<'_, '_>,
) -> bool {
    for (_, fty) in fields {
        if !hfa_walk_ty(fty, found, count, abi) {
            return false;
        }
    }
    true
}

fn hfa_walk_ty(
    ty: &Type,
    found: &mut Option<Type>,
    count: &mut u32,
    abi: &AbiCtx<'_, '_>,
) -> bool {
    match ty {
        Type::Float(_) => {
            match found {
                None => *found = Some(ty.clone()),
                Some(prev) if prev == ty => {}
                Some(_) => return false, // mixed FP types -> not an HFA
            }
            *count += 1;
            *count <= 4
        }
        Type::Array(elem, n) => {
            for _ in 0..*n {
                if !hfa_walk_ty(elem, found, count, abi) {
                    return false;
                }
            }
            true
        }
        // Recurse into a nested struct: an aggregate of FP members is still an HFA.
        Type::Struct(name) => match (abi.resolve_struct)(name) {
            Some(nested) => hfa_walk(&nested, found, count, abi),
            None => false,
        },
        // Any non-FP member disqualifies the aggregate from being an HFA.
        _ => false,
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    if a == 0 {
        x
    } else {
        x.div_ceil(a) * a
    }
}
