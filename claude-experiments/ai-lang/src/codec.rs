//! Handrolled binary canonical encoding.
//!
//! This format is the input to the content hasher. It must be byte-for-byte
//! deterministic and stable across runtime versions and host languages —
//! every node on the network must agree on the encoding of every AST.
//!
//! ## Format rules (forever)
//!
//! - All integers are big-endian, fixed-width.
//! - Lengths and counts are `u32` big-endian.
//! - Integer literals are `i64` big-endian.
//! - Strings are length-prefixed (`u32` BE) UTF-8 bytes.
//! - Bytes blobs are length-prefixed (`u32` BE) raw bytes.
//! - Sums (Rust enums on the wire) are tagged by *name* (length-prefixed
//!   ASCII string), not by index. This costs a few bytes per node and buys
//!   us the freedom to add new variants later without invalidating existing
//!   hashes. Variant payloads follow the tag, encoded recursively.
//! - Sequences (lists, captures, params) are `u32` count followed by
//!   `count` encoded elements back-to-back.
//! - Optionals: `u8` 0 (absent) or 1 (present), and if present the payload.
//! - There is no padding, no alignment, no version header. The format IS
//!   the version. Any future change to it produces a new format that will
//!   be selected via runtime negotiation, not silent reinterpretation.

use crate::ast::{Def, Expr, MatchArm, Pattern, Type};
use crate::hash::Hash;

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    UnexpectedEof,
    BadUtf8,
    UnknownTag { kind: &'static str, tag: String },
    BadBool(u8),
    BadOption(u8),
    TrailingBytes(usize),
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::UnexpectedEof => write!(f, "unexpected end of input"),
            DecodeError::BadUtf8 => write!(f, "invalid utf-8 in string field"),
            DecodeError::UnknownTag { kind, tag } => {
                write!(f, "unknown {} variant tag: {:?}", kind, tag)
            }
            DecodeError::BadBool(b) => write!(f, "bool must be 0 or 1, got {}", b),
            DecodeError::BadOption(b) => write!(f, "option must be 0 or 1, got {}", b),
            DecodeError::TrailingBytes(n) => write!(f, "{} trailing bytes after decode", n),
        }
    }
}

impl std::error::Error for DecodeError {}

// =============================================================================
// Public encode/decode entry points
// =============================================================================

pub fn encode_def(def: &Def) -> Vec<u8> {
    let mut buf = Vec::new();
    write_def(&mut buf, def);
    buf
}

pub fn decode_def(bytes: &[u8]) -> Result<Def, DecodeError> {
    let mut r = Reader::new(bytes);
    let d = read_def(&mut r)?;
    r.finish()?;
    Ok(d)
}

pub fn encode_expr(expr: &Expr) -> Vec<u8> {
    let mut buf = Vec::new();
    write_expr(&mut buf, expr);
    buf
}

/// Encode a `Type` to canonical bytes. Used by the typecheck cache
/// codec (`TypeScheme` contains `Type`s by value).
pub fn encode_type(t: &Type) -> Vec<u8> {
    let mut buf = Vec::new();
    write_type(&mut buf, t);
    buf
}

/// Decode a `Type` from canonical bytes — counterpart to
/// [`encode_type`]. Rejects trailing bytes.
pub fn decode_type(bytes: &[u8]) -> Result<Type, DecodeError> {
    let mut r = Reader::new(bytes);
    let t = read_type(&mut r)?;
    r.finish()?;
    Ok(t)
}

/// Encode an extern's wire-shippable signature: `name`, parameter
/// types, return type, and optional library. Externs are NOT
/// content-addressed (they bind to a host/C symbol resolved at install
/// time), so when shipping code that calls an extern, the requirement
/// must travel alongside the code so the receiver can declare and
/// resolve it (or fail clearly when the library/symbol is unavailable).
pub fn encode_extern(
    name: &str,
    params: &[Type],
    ret: &Type,
    library: Option<&str>,
    variadic: bool,
) -> Vec<u8> {
    let mut buf = Vec::new();
    write_str(&mut buf, name);
    write_u32(&mut buf, params.len() as u32);
    for p in params {
        write_type(&mut buf, p);
    }
    write_type(&mut buf, ret);
    match library {
        None => buf.push(0),
        Some(lib) => {
            buf.push(1);
            write_str(&mut buf, lib);
        }
    }
    buf.push(if variadic { 1 } else { 0 });
    buf
}

/// Decode an extern signature produced by [`encode_extern`]. Returns
/// `(name, params, ret, library, variadic)`.
pub fn decode_extern(
    bytes: &[u8],
) -> Result<(String, Vec<Type>, Type, Option<String>, bool), DecodeError> {
    let mut r = Reader::new(bytes);
    let name = r.read_str()?;
    let n = r.read_u32()? as usize;
    let mut params = Vec::with_capacity(n);
    for _ in 0..n {
        params.push(read_type(&mut r)?);
    }
    let ret = read_type(&mut r)?;
    let library = match r.read_u8()? {
        0 => None,
        1 => Some(r.read_str()?),
        b => {
            return Err(DecodeError::UnknownTag {
                kind: "extern library option",
                tag: format!("{}", b),
            });
        }
    };
    let variadic = r.read_bool()?;
    r.finish()?;
    Ok((name, params, ret, library, variadic))
}

pub fn decode_expr(bytes: &[u8]) -> Result<Expr, DecodeError> {
    let mut r = Reader::new(bytes);
    let e = read_expr(&mut r)?;
    r.finish()?;
    Ok(e)
}

// =============================================================================
// Low-level writers
// =============================================================================

fn write_u32(buf: &mut Vec<u8>, n: u32) {
    buf.extend_from_slice(&n.to_be_bytes());
}

fn write_i64(buf: &mut Vec<u8>, n: i64) {
    buf.extend_from_slice(&n.to_be_bytes());
}

fn write_bool(buf: &mut Vec<u8>, b: bool) {
    buf.push(if b { 1 } else { 0 });
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len: u32 = bytes
        .len()
        .try_into()
        .expect("string longer than u32::MAX bytes (4 GiB); refusing to encode");
    write_u32(buf, len);
    buf.extend_from_slice(bytes);
}

fn write_hash(buf: &mut Vec<u8>, h: &Hash) {
    buf.extend_from_slice(h.as_bytes());
}

fn write_tag(buf: &mut Vec<u8>, tag: &str) {
    write_str(buf, tag);
}

// =============================================================================
// Reader
// =============================================================================

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.pos + n > self.bytes.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32, DecodeError> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn read_i64(&mut self) -> Result<i64, DecodeError> {
        let s = self.take(8)?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(s);
        Ok(i64::from_be_bytes(buf))
    }

    fn read_u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    fn read_bool(&mut self) -> Result<bool, DecodeError> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            b => Err(DecodeError::BadBool(b)),
        }
    }

    fn read_str(&mut self) -> Result<String, DecodeError> {
        let len = self.read_u32()? as usize;
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| DecodeError::BadUtf8)
    }

    fn read_hash(&mut self) -> Result<Hash, DecodeError> {
        let s = self.take(Hash::SIZE)?;
        let mut buf = [0u8; Hash::SIZE];
        buf.copy_from_slice(s);
        Ok(Hash(buf))
    }

    fn finish(self) -> Result<(), DecodeError> {
        let trailing = self.bytes.len() - self.pos;
        if trailing == 0 {
            Ok(())
        } else {
            Err(DecodeError::TrailingBytes(trailing))
        }
    }
}

// =============================================================================
// Expr
// =============================================================================

fn write_expr(buf: &mut Vec<u8>, e: &Expr) {
    match e {
        Expr::IntLit(n) => {
            write_tag(buf, "IntLit");
            write_i64(buf, *n);
        }
        Expr::FloatLit(x) => {
            write_tag(buf, "FloatLit");
            // Serialize the exact f64 bit-pattern (deterministic; no
            // rounding) so the content hash is stable.
            write_i64(buf, x.to_bits() as i64);
        }
        Expr::BoolLit(b) => {
            write_tag(buf, "BoolLit");
            write_bool(buf, *b);
        }
        Expr::StringLit(s) => {
            write_tag(buf, "StringLit");
            write_str(buf, s);
        }
        Expr::LocalVar(idx) => {
            write_tag(buf, "LocalVar");
            write_u32(buf, *idx);
        }
        Expr::TopRef(h) => {
            write_tag(buf, "TopRef");
            write_hash(buf, h);
        }
        Expr::SelfRef(idx) => {
            write_tag(buf, "SelfRef");
            write_u32(buf, *idx);
        }
        Expr::BuiltinRef(name) => {
            write_tag(buf, "BuiltinRef");
            write_str(buf, name);
        }
        Expr::Call(callee, args) => {
            write_tag(buf, "Call");
            write_expr(buf, callee);
            let n: u32 = args.len().try_into().expect("too many args");
            write_u32(buf, n);
            for a in args {
                write_expr(buf, a);
            }
        }
        Expr::Lambda { params, body } => {
            write_tag(buf, "Lambda");
            let n: u32 = params.len().try_into().expect("too many lambda params");
            write_u32(buf, n);
            for p in params {
                write_type(buf, p);
            }
            write_expr(buf, body);
        }
        Expr::Let { value, body } => {
            write_tag(buf, "Let");
            write_expr(buf, value);
            write_expr(buf, body);
        }
        Expr::StructNew { struct_ref, fields } => {
            write_tag(buf, "StructNew");
            write_hash(buf, struct_ref);
            let n: u32 = fields.len().try_into().expect("too many struct fields");
            write_u32(buf, n);
            for f in fields {
                write_expr(buf, f);
            }
        }
        Expr::Field { base, struct_ref, index } => {
            write_tag(buf, "Field");
            write_expr(buf, base);
            write_hash(buf, struct_ref);
            write_u32(buf, *index);
        }
        Expr::EnumNew {
            enum_ref,
            variant_index,
            payload,
        } => {
            write_tag(buf, "EnumNew");
            write_hash(buf, enum_ref);
            write_u32(buf, *variant_index);
            match payload {
                None => write_bool(buf, false),
                Some(e) => {
                    write_bool(buf, true);
                    write_expr(buf, e);
                }
            }
        }
        Expr::Match { scrutinee, arms } => {
            write_tag(buf, "Match");
            write_expr(buf, scrutinee);
            let n: u32 = arms.len().try_into().expect("too many match arms");
            write_u32(buf, n);
            for arm in arms {
                write_pattern(buf, &arm.pattern);
                write_expr(buf, &arm.body);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            write_tag(buf, "If");
            write_expr(buf, cond);
            write_expr(buf, then_branch);
            write_expr(buf, else_branch);
        }
        Expr::Try {
            expr,
            enum_ref,
            ok_index,
            err_index,
        } => {
            write_tag(buf, "Try");
            write_expr(buf, expr);
            write_hash(buf, enum_ref);
            write_u32(buf, *ok_index);
            write_u32(buf, *err_index);
        }
        Expr::Defer { cleanup, body } => {
            write_tag(buf, "Defer");
            write_expr(buf, cleanup);
            write_expr(buf, body);
        }
    }
    // INVARIANT: every variant of Expr must have an arm here. Adding a new
    // variant without updating this match will fail compilation (Rust enforces
    // exhaustive matches with no wildcard). That is intentional — every node
    // type that affects identity must have a stable, explicit encoding.
}

fn read_expr(r: &mut Reader) -> Result<Expr, DecodeError> {
    let tag = r.read_str()?;
    Ok(match tag.as_str() {
        "IntLit" => Expr::IntLit(r.read_i64()?),
        "FloatLit" => Expr::FloatLit(f64::from_bits(r.read_i64()? as u64)),
        "BoolLit" => Expr::BoolLit(r.read_bool()?),
        "StringLit" => Expr::StringLit(r.read_str()?),
        "LocalVar" => Expr::LocalVar(r.read_u32()?),
        "TopRef" => Expr::TopRef(r.read_hash()?),
        "SelfRef" => Expr::SelfRef(r.read_u32()?),
        "BuiltinRef" => Expr::BuiltinRef(r.read_str()?),
        "Call" => {
            let callee = Box::new(read_expr(r)?);
            let n = r.read_u32()?;
            let mut args = Vec::with_capacity(n as usize);
            for _ in 0..n {
                args.push(read_expr(r)?);
            }
            Expr::Call(callee, args)
        }
        "Lambda" => {
            let n = r.read_u32()?;
            let mut params = Vec::with_capacity(n as usize);
            for _ in 0..n {
                params.push(read_type(r)?);
            }
            let body = Box::new(read_expr(r)?);
            Expr::Lambda { params, body }
        }
        "Let" => {
            let value = Box::new(read_expr(r)?);
            let body = Box::new(read_expr(r)?);
            Expr::Let { value, body }
        }
        "StructNew" => {
            let struct_ref = r.read_hash()?;
            let n = r.read_u32()?;
            let mut fields = Vec::with_capacity(n as usize);
            for _ in 0..n {
                fields.push(read_expr(r)?);
            }
            Expr::StructNew { struct_ref, fields }
        }
        "Field" => {
            let base = Box::new(read_expr(r)?);
            let struct_ref = r.read_hash()?;
            let index = r.read_u32()?;
            Expr::Field {
                base,
                struct_ref,
                index,
            }
        }
        "EnumNew" => {
            let enum_ref = r.read_hash()?;
            let variant_index = r.read_u32()?;
            let has_payload = r.read_bool()?;
            let payload = if has_payload {
                Some(Box::new(read_expr(r)?))
            } else {
                None
            };
            Expr::EnumNew {
                enum_ref,
                variant_index,
                payload,
            }
        }
        "Match" => {
            let scrutinee = Box::new(read_expr(r)?);
            let n = r.read_u32()?;
            let mut arms = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let pattern = read_pattern(r)?;
                let body = read_expr(r)?;
                arms.push(MatchArm { pattern, body });
            }
            Expr::Match { scrutinee, arms }
        }
        "If" => {
            let cond = Box::new(read_expr(r)?);
            let then_branch = Box::new(read_expr(r)?);
            let else_branch = Box::new(read_expr(r)?);
            Expr::If {
                cond,
                then_branch,
                else_branch,
            }
        }
        "Try" => {
            let expr = Box::new(read_expr(r)?);
            let enum_ref = r.read_hash()?;
            let ok_index = r.read_u32()?;
            let err_index = r.read_u32()?;
            Expr::Try {
                expr,
                enum_ref,
                ok_index,
                err_index,
            }
        }
        "Defer" => {
            let cleanup = Box::new(read_expr(r)?);
            let body = Box::new(read_expr(r)?);
            Expr::Defer { cleanup, body }
        }
        _ => {
            return Err(DecodeError::UnknownTag {
                kind: "Expr",
                tag,
            });
        }
    })
}

// =============================================================================
// Pattern
// =============================================================================

fn write_pattern(buf: &mut Vec<u8>, p: &Pattern) {
    match p {
        Pattern::Wildcard => write_tag(buf, "Wildcard"),
        Pattern::Var => write_tag(buf, "Var"),
        Pattern::Enum {
            enum_ref,
            variant_index,
            payload,
        } => {
            write_tag(buf, "EnumPat");
            write_hash(buf, enum_ref);
            write_u32(buf, *variant_index);
            match payload {
                None => write_bool(buf, false),
                Some(p) => {
                    write_bool(buf, true);
                    write_pattern(buf, p);
                }
            }
        }
    }
}

fn read_pattern(r: &mut Reader) -> Result<Pattern, DecodeError> {
    let tag = r.read_str()?;
    Ok(match tag.as_str() {
        "Wildcard" => Pattern::Wildcard,
        "Var" => Pattern::Var,
        "EnumPat" => {
            let enum_ref = r.read_hash()?;
            let variant_index = r.read_u32()?;
            let has_payload = r.read_bool()?;
            let payload = if has_payload {
                Some(Box::new(read_pattern(r)?))
            } else {
                None
            };
            Pattern::Enum {
                enum_ref,
                variant_index,
                payload,
            }
        }
        _ => {
            return Err(DecodeError::UnknownTag {
                kind: "Pattern",
                tag,
            });
        }
    })
}

// =============================================================================
// Type
// =============================================================================

fn write_type(buf: &mut Vec<u8>, t: &Type) {
    match t {
        Type::Builtin(name) => {
            write_tag(buf, "Builtin");
            write_str(buf, name);
        }
        Type::TypeRef(h) => {
            write_tag(buf, "TypeRef");
            write_hash(buf, h);
        }
        Type::TypeVar(idx) => {
            write_tag(buf, "TypeVar");
            write_u32(buf, *idx);
        }
        Type::Apply(head, args) => {
            write_tag(buf, "Apply");
            write_type(buf, head);
            let n: u32 = args.len().try_into().expect("too many type args");
            write_u32(buf, n);
            for a in args {
                write_type(buf, a);
            }
        }
        Type::FnType { params, ret } => {
            write_tag(buf, "FnType");
            let n: u32 = params.len().try_into().expect("too many params");
            write_u32(buf, n);
            for p in params {
                write_type(buf, p);
            }
            write_type(buf, ret);
        }
        Type::SelfRef(idx) => {
            write_tag(buf, "TypeSelfRef");
            write_u32(buf, *idx);
        }
    }
}

fn read_type(r: &mut Reader<'_>) -> Result<Type, DecodeError> {
    let tag = r.read_str()?;
    Ok(match tag.as_str() {
        "Builtin" => Type::Builtin(r.read_str()?),
        "TypeRef" => Type::TypeRef(r.read_hash()?),
        "TypeVar" => Type::TypeVar(r.read_u32()?),
        "Apply" => {
            let head = Box::new(read_type(r)?);
            let n = r.read_u32()?;
            let mut args = Vec::with_capacity(n as usize);
            for _ in 0..n {
                args.push(read_type(r)?);
            }
            Type::Apply(head, args)
        }
        "FnType" => {
            let n = r.read_u32()?;
            let mut params = Vec::with_capacity(n as usize);
            for _ in 0..n {
                params.push(read_type(r)?);
            }
            let ret = Box::new(read_type(r)?);
            Type::FnType { params, ret }
        }
        "TypeSelfRef" => Type::SelfRef(r.read_u32()?),
        _ => {
            return Err(DecodeError::UnknownTag {
                kind: "Type",
                tag,
            });
        }
    })
}

// =============================================================================
// Def
// =============================================================================

fn write_def(buf: &mut Vec<u8>, d: &Def) {
    match d {
        Def::Fn {
            is_local,
            type_params,
            params,
            ret,
            body,
        } => {
            write_tag(buf, "Fn");
            write_bool(buf, *is_local);
            write_u32(buf, *type_params);
            let n: u32 = params.len().try_into().expect("too many params");
            write_u32(buf, n);
            for p in params {
                write_type(buf, p);
            }
            write_type(buf, ret);
            write_expr(buf, body);
        }
        Def::Struct {
            type_params,
            fields,
        } => {
            write_tag(buf, "Struct");
            write_u32(buf, *type_params);
            let n: u32 = fields.len().try_into().expect("too many fields");
            write_u32(buf, n);
            for (name, ty) in fields {
                write_str(buf, name);
                write_type(buf, ty);
            }
        }
        Def::Enum {
            type_params,
            variants,
        } => {
            write_tag(buf, "Enum");
            write_u32(buf, *type_params);
            let n: u32 = variants.len().try_into().expect("too many variants");
            write_u32(buf, n);
            for (name, payload) in variants {
                write_str(buf, name);
                match payload {
                    None => write_bool(buf, false),
                    Some(t) => {
                        write_bool(buf, true);
                        write_type(buf, t);
                    }
                }
            }
        }
    }
}

fn read_def(r: &mut Reader) -> Result<Def, DecodeError> {
    let tag = r.read_str()?;
    Ok(match tag.as_str() {
        "Fn" => {
            let is_local = r.read_bool()?;
            let type_params = r.read_u32()?;
            let n = r.read_u32()?;
            let mut params = Vec::with_capacity(n as usize);
            for _ in 0..n {
                params.push(read_type(r)?);
            }
            let ret = read_type(r)?;
            let body = read_expr(r)?;
            Def::Fn {
                is_local,
                type_params,
                params,
                ret,
                body,
            }
        }
        "Struct" => {
            let type_params = r.read_u32()?;
            let n = r.read_u32()?;
            let mut fields = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let name = r.read_str()?;
                let ty = read_type(r)?;
                fields.push((name, ty));
            }
            Def::Struct {
                type_params,
                fields,
            }
        }
        "Enum" => {
            let type_params = r.read_u32()?;
            let n = r.read_u32()?;
            let mut variants = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let name = r.read_str()?;
                let has_payload = r.read_bool()?;
                let payload = if has_payload {
                    Some(read_type(r)?)
                } else {
                    None
                };
                variants.push((name, payload));
            }
            Def::Enum {
                type_params,
                variants,
            }
        }
        _ => return Err(DecodeError::UnknownTag { kind: "Def", tag }),
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Roundtrip ----

    #[test]
    fn roundtrip_int_lit() {
        let e = Expr::IntLit(42);
        let bytes = encode_expr(&e);
        assert_eq!(decode_expr(&bytes).unwrap(), e);
    }

    #[test]
    fn roundtrip_nested_call() {
        let e = Expr::Call(
            Box::new(Expr::BuiltinRef("core/i64.mul".to_owned())),
            vec![Expr::LocalVar(0), Expr::IntLit(2)],
        );
        let bytes = encode_expr(&e);
        assert_eq!(decode_expr(&bytes).unwrap(), e);
    }

    #[test]
    fn roundtrip_def_fn() {
        let d = Def::Fn {
            is_local: false,
            type_params: 0,
            params: vec![Type::Builtin("Int".to_owned())],
            ret: Type::Builtin("Int".to_owned()),
            body: Expr::Call(
                Box::new(Expr::BuiltinRef("core/i64.mul".to_owned())),
                vec![Expr::LocalVar(0), Expr::IntLit(2)],
            ),
        };
        let bytes = encode_def(&d);
        assert_eq!(decode_def(&bytes).unwrap(), d);
    }

    // ---- Determinism ----

    #[test]
    fn encoding_is_deterministic() {
        let d = Def::Fn {
            is_local: false,
            type_params: 0,
            params: vec![Type::Builtin("Int".to_owned())],
            ret: Type::Builtin("Int".to_owned()),
            body: Expr::Call(
                Box::new(Expr::BuiltinRef("core/i64.mul".to_owned())),
                vec![Expr::LocalVar(0), Expr::IntLit(2)],
            ),
        };
        let a = encode_def(&d);
        let b = encode_def(&d);
        assert_eq!(a, b, "encoding the same def twice must produce identical bytes");
    }

    // ---- Errors ----

    #[test]
    fn trailing_bytes_rejected() {
        let mut bytes = encode_expr(&Expr::IntLit(1));
        bytes.push(0xff);
        assert_eq!(decode_expr(&bytes), Err(DecodeError::TrailingBytes(1)));
    }

    #[test]
    fn unknown_tag_rejected() {
        // Fabricate: tag = "Bogus", no payload.
        let mut bytes = Vec::new();
        write_str(&mut bytes, "Bogus");
        let err = decode_expr(&bytes).unwrap_err();
        assert_eq!(
            err,
            DecodeError::UnknownTag {
                kind: "Expr",
                tag: "Bogus".to_owned()
            }
        );
    }

    #[test]
    fn truncated_input_rejected() {
        let bytes = encode_expr(&Expr::IntLit(1));
        let truncated = &bytes[..bytes.len() - 1];
        assert_eq!(decode_expr(truncated), Err(DecodeError::UnexpectedEof));
    }
}
