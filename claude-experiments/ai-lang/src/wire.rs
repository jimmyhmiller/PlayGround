//! Wire format for runtime values.
//!
//! Values produced by `ai-lang` code (the result of a `def fn(...) -> T`
//! evaluation, or a value passed to a future `at(node, …)` primitive)
//! are encoded to bytes here and decoded back into a target [`Runtime`]'s
//! heap. The format is the foundation for shipping closures across the
//! network — once a value roundtrips between two heaps, TCP is just
//! byte transport.
//!
//! ## Format (binary, big-endian, all u32 lengths)
//!
//! ```text
//! Value:
//!   kind: u8
//!     0 = Int        → i64 payload
//!     1 = Closure    → code_hash (32) + n_captures (u32) + captures…
//!     2 = Struct     → struct_ref (32) + n_fields (u32)  + fields…
//!     3 = Enum       → enum_ref (32)  + variant_index (u32) + has_payload (u8)
//!                       + optional payload Value
//! ```
//!
//! ## Cycle handling
//!
//! ai-lang values are immutable (no mutation primitives) and allocated
//! in strict tail-order, so the heap is acyclic by construction. The
//! encoder uses simple recursion; it cannot run into a cycle.
//!
//! ## What this module does NOT do (yet)
//!
//! - **Code fetching.** If the receiver doesn't have a shape's hash in
//!   its registry, decoding errors with `MissingShape`. The next step
//!   (the `at` protocol) shuttles canonical def bytes between peers so
//!   the receiver can register the missing shape before retrying.
//! - **Network transport.** Bytes in / bytes out — the caller plugs in
//!   whatever transport they want (a Vec, a TcpStream, etc.).

use crate::codegen::{FieldMeta, ShapeMeta};
use crate::gc::{Full, ObjHeader};
use crate::hash::Hash;
use crate::runtime::{Runtime, ai_array_new, ai_array_set, ai_gc_alloc_closure, ai_str_new};

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug)]
pub enum WireError {
    /// Reached the end of the input mid-decode.
    UnexpectedEof,
    /// Encoded `kind` byte wasn't 0/1/2/3.
    BadKind(u8),
    /// Encoded enum had `has_payload` byte ≠ 0 or 1.
    BadOption(u8),
    /// A heap pointer's `type_id` wasn't in the encoder's
    /// `shape_by_type_id` map. Means the codegen failed to register
    /// the shape, or we're encoding a value from a different runtime.
    UnknownTypeId(u16),
    /// The decoder hit a shape hash that the target runtime hasn't
    /// registered. Resolution requires fetching the canonical def
    /// bytes for the listed hashes (the future `at` code-fetch
    /// protocol).
    MissingShape(Hash),
    /// Internal inconsistency — e.g. shape_meta claims a closure with
    /// N captures but the encoder ran out of bytes after K<N.
    Corrupt(&'static str),
}

impl core::fmt::Display for WireError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WireError::UnexpectedEof => f.write_str("unexpected end of input"),
            WireError::BadKind(b) => write!(f, "bad value kind tag {}", b),
            WireError::BadOption(b) => write!(f, "bad option byte {}", b),
            WireError::UnknownTypeId(id) => write!(f, "unknown type_id {} in heap header", id),
            WireError::MissingShape(h) => write!(f, "shape {} not registered in target runtime", h),
            WireError::Corrupt(msg) => write!(f, "corrupt wire input: {}", msg),
        }
    }
}

impl std::error::Error for WireError {}

// =============================================================================
// Public encode/decode entry points
// =============================================================================

/// A runtime value as the encoder sees it: either a raw i64 or a heap
/// pointer. Callers usually have one of these from a JIT call result.
#[derive(Copy, Clone, Debug)]
pub enum WireValue {
    Int(i64),
    Heap(*const u8),
}

/// Encode a `WireValue` (and everything it transitively references) to
/// `out`. Reads from `rt`'s shape metadata to walk pointer fields.
///
/// # Safety
/// Any `WireValue::Heap` pointer must be a live heap object allocated
/// in `rt`'s heap. The encoder reads object headers and field bytes
/// through this pointer.
pub unsafe fn encode_value(
    rt: &Runtime,
    value: WireValue,
    out: &mut Vec<u8>,
) -> Result<(), WireError> {
    match value {
        WireValue::Int(n) => {
            out.push(0); // kind = Int
            out.extend_from_slice(&n.to_be_bytes());
            Ok(())
        }
        WireValue::Heap(ptr) => unsafe { encode_heap(rt, ptr, out) },
    }
}

/// Decode bytes into a value in `rt`'s heap. Returns the decoded
/// `WireValue` (Int or a freshly-allocated heap pointer) and the
/// number of bytes consumed.
///
/// Mutates `rt`'s thread state during allocation (uses
/// `ai_gc_alloc_closure`).
///
/// # Safety
/// `rt.thread` and `rt.heap` must be valid (they are by construction).
pub unsafe fn decode_value(rt: &Runtime, bytes: &[u8]) -> Result<(WireValue, usize), WireError> {
    let mut r = Reader::new(bytes);
    let v = unsafe { decode_one(rt, &mut r)? };
    Ok((v, r.pos))
}

// =============================================================================
// Encoding
// =============================================================================

unsafe fn encode_heap(rt: &Runtime, ptr: *const u8, out: &mut Vec<u8>) -> Result<(), WireError> {
    // Read the type_id from the object header.
    let type_id = unsafe { read_type_id_from_header(ptr) };
    let shape_hash = rt
        .shape_by_type_id
        .get(type_id as usize)
        .copied()
        .flatten()
        .ok_or(WireError::UnknownTypeId(type_id))?;
    // String and Array are varlen shapes with no `shape_meta` entry;
    // recognize them by their canonical hash and read the heap layout
    // directly. This is what makes immutable container values (e.g. a
    // persistent HAMT, which holds Array<HNode> and String keys)
    // wire-portable.
    if shape_hash == crate::runtime::array_shape_hash() {
        return unsafe { encode_array(rt, ptr, out) };
    }
    if shape_hash == crate::runtime::string_shape_hash() {
        return unsafe { encode_string(ptr, out) };
    }
    let meta = rt
        .shape_meta
        .get(&shape_hash)
        .ok_or(WireError::MissingShape(shape_hash))?
        .clone();

    match meta {
        ShapeMeta::Closure {
            code_hash,
            captures,
        } => unsafe {
            out.push(1); // kind = Closure
            out.extend_from_slice(code_hash.as_bytes());
            let n_captures: u32 = captures.len().try_into().map_err(|_| {
                WireError::Corrupt("closure has more captures than fit in u32")
            })?;
            out.extend_from_slice(&n_captures.to_be_bytes());
            // Walk captures in source order. Each is either a pointer
            // (encoded via the heap-encoder, which lands as
            // `kind = Closure/Struct/Enum`) or a raw i64 (kind = Int).
            // The receiver replays the same source order.
            for cap in &captures {
                let slot = ptr.add(cap.offset as usize);
                if cap.is_pointer {
                    let p = *(slot as *const *const u8);
                    encode_heap(rt, p, out)?;
                } else {
                    let v = *(slot as *const u64);
                    out.push(0); // kind = Int
                    out.extend_from_slice(&(v as i64).to_be_bytes());
                }
            }
            Ok(())
        },
        ShapeMeta::Struct { struct_ref, fields } => unsafe {
            out.push(2); // kind = Struct
            out.extend_from_slice(struct_ref.as_bytes());
            let n: u32 = fields.len().try_into().map_err(|_| {
                WireError::Corrupt("struct has more fields than fit in u32")
            })?;
            out.extend_from_slice(&n.to_be_bytes());
            for field in &fields {
                encode_field(rt, ptr, field, out)?;
            }
            Ok(())
        },
        ShapeMeta::EnumVariant {
            enum_ref,
            variant_index,
            tag_offset: _,
            payload,
        } => unsafe {
            out.push(3); // kind = Enum
            out.extend_from_slice(enum_ref.as_bytes());
            out.extend_from_slice(&variant_index.to_be_bytes());
            match payload {
                None => out.push(0),
                Some(field) => {
                    out.push(1);
                    encode_field(rt, ptr, &field, out)?;
                }
            }
            Ok(())
        },
    }
}

unsafe fn encode_field(
    rt: &Runtime,
    base: *const u8,
    field: &FieldMeta,
    out: &mut Vec<u8>,
) -> Result<(), WireError> {
    let slot = unsafe { base.add(field.offset as usize) };
    if field.is_pointer {
        let p = unsafe { *(slot as *const *const u8) };
        unsafe { encode_heap(rt, p, out) }
    } else {
        let v = unsafe { *(slot as *const u64) };
        out.push(0); // kind = Int
        out.extend_from_slice(&(v as i64).to_be_bytes());
        Ok(())
    }
}

unsafe fn read_type_id_from_header(ptr: *const u8) -> u16 {
    let off = <Full as ObjHeader>::TYPE_ID_OFFSET;
    unsafe { *(ptr.add(off) as *const u16) }
}

/// Encode an `Array` heap object (kind = 4). Layout: GC header, then an
/// 8-byte element count, then `count` pointer slots. Every slot is a GC
/// pointer (Int elements are boxed), so each is encoded via `encode_heap`.
unsafe fn encode_array(rt: &Runtime, ptr: *const u8, out: &mut Vec<u8>) -> Result<(), WireError> {
    let header = <Full as ObjHeader>::SIZE;
    let count = unsafe { *(ptr.add(header) as *const u64) };
    let n: u32 = count
        .try_into()
        .map_err(|_| WireError::Corrupt("array longer than u32"))?;
    out.push(4); // kind = Array
    out.extend_from_slice(&n.to_be_bytes());
    for i in 0..count as usize {
        let slot = unsafe { ptr.add(header + 8 + i * 8) as *const *const u8 };
        let elem = unsafe { *slot };
        unsafe { encode_heap(rt, elem, out)? };
    }
    Ok(())
}

/// Encode a `String`/`Bytes` heap object (kind = 5). Layout: GC header,
/// then an 8-byte byte count, then the raw bytes.
unsafe fn encode_string(ptr: *const u8, out: &mut Vec<u8>) -> Result<(), WireError> {
    let header = <Full as ObjHeader>::SIZE;
    let len = unsafe { *(ptr.add(header) as *const u64) };
    let n: u32 = len
        .try_into()
        .map_err(|_| WireError::Corrupt("string longer than u32"))?;
    let bytes = unsafe { std::slice::from_raw_parts(ptr.add(header + 8), len as usize) };
    out.push(5); // kind = String
    out.extend_from_slice(&n.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

// =============================================================================
// Decoding
// =============================================================================

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        if self.pos + n > self.bytes.len() {
            return Err(WireError::UnexpectedEof);
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, WireError> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn read_i64(&mut self) -> Result<i64, WireError> {
        let s = self.take(8)?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(s);
        Ok(i64::from_be_bytes(buf))
    }

    fn read_hash(&mut self) -> Result<Hash, WireError> {
        let s = self.take(Hash::SIZE)?;
        let mut buf = [0u8; Hash::SIZE];
        buf.copy_from_slice(s);
        Ok(Hash(buf))
    }
}

unsafe fn decode_one(rt: &Runtime, r: &mut Reader) -> Result<WireValue, WireError> {
    let kind = r.read_u8()?;
    match kind {
        0 => Ok(WireValue::Int(r.read_i64()?)),
        1 => unsafe { decode_closure(rt, r) },
        2 => unsafe { decode_struct(rt, r) },
        3 => unsafe { decode_enum(rt, r) },
        4 => unsafe { decode_array(rt, r) },
        5 => unsafe { decode_string(rt, r) },
        other => Err(WireError::BadKind(other)),
    }
}

/// Decode an `Array` (kind 4): element count, then each pointer slot.
/// Mirrors the encoder's layout; rebuilds via the runtime allocator.
unsafe fn decode_array(rt: &Runtime, r: &mut Reader) -> Result<WireValue, WireError> {
    let n = r.read_u32()? as i64;
    let arr = unsafe { ai_array_new(rt.thread_ptr(), n) };
    for i in 0..n {
        let elem = unsafe { decode_one(rt, r)? };
        let p = match elem {
            WireValue::Heap(p) => p,
            WireValue::Int(_) => {
                return Err(WireError::Corrupt(
                    "array element decoded as Int; array slots are GC pointers",
                ));
            }
        };
        unsafe { ai_array_set(arr, i, p as *mut u8) };
    }
    Ok(WireValue::Heap(arr as *const u8))
}

/// Decode a `String`/`Bytes` (kind 5): byte count, then raw bytes copied
/// into a fresh heap String.
unsafe fn decode_string(rt: &Runtime, r: &mut Reader) -> Result<WireValue, WireError> {
    let n = r.read_u32()? as usize;
    let bytes = r.take(n)?;
    let s = unsafe { ai_str_new(rt.thread_ptr(), bytes.as_ptr(), n as i64) };
    Ok(WireValue::Heap(s as *const u8))
}

unsafe fn decode_closure(rt: &Runtime, r: &mut Reader) -> Result<WireValue, WireError> {
    let code_hash = r.read_hash()?;
    let n_captures = r.read_u32()?;
    let ti = rt
        .type_info_for(&code_hash)
        .ok_or(WireError::MissingShape(code_hash))?;

    // Pull per-capture layout BEFORE allocating the closure so we
    // can validate the count + compute the header offsets, which
    // depend on the number of pointer captures.
    let meta = rt
        .shape_meta
        .get(&code_hash)
        .ok_or(WireError::MissingShape(code_hash))?
        .clone();
    let captures = match meta {
        ShapeMeta::Closure { captures, .. } => captures,
        _ => return Err(WireError::Corrupt("registered closure hash isn't a Closure shape")),
    };
    if n_captures as usize != captures.len() {
        return Err(WireError::Corrupt(
            "closure wire payload captures-count disagrees with registered shape",
        ));
    }

    // Closure header (code_hash + n_captures + pad) sits AFTER the
    // value_field slots that hold pointer captures. Compute its
    // absolute offset.
    let ptr_count = captures.iter().filter(|c| c.is_pointer).count();
    let header_base = Full::SIZE + ptr_count * 8;

    // Allocate a fresh closure object via the runtime allocator.
    let ptr = unsafe { ai_gc_alloc_closure(rt.thread_ptr(), ti) };

    unsafe {
        let dst_hash = ptr.add(header_base) as *mut u8;
        core::ptr::copy_nonoverlapping(code_hash.as_bytes().as_ptr(), dst_hash, Hash::SIZE);
        let dst_n = ptr.add(header_base + 32) as *mut u32;
        *dst_n = n_captures;
    }

    // Decode each capture in source order. Pointer slots get a
    // heap-pointer (the decoder allocates sub-objects); Int slots
    // get a raw i64.
    for cap in &captures {
        let v = unsafe { decode_one(rt, r)? };
        let slot_addr = unsafe { ptr.add(cap.offset as usize) };
        if cap.is_pointer {
            let heap_ptr = match v {
                WireValue::Heap(p) => p,
                WireValue::Int(_) => {
                    return Err(WireError::Corrupt(
                        "pointer-typed capture slot received an Int wire value",
                    ));
                }
            };
            unsafe {
                let slot = slot_addr as *mut *const u8;
                *slot = heap_ptr;
            }
        } else {
            let int = match v {
                WireValue::Int(n) => n,
                _ => {
                    return Err(WireError::Corrupt(
                        "Int-typed capture slot received a heap wire value",
                    ));
                }
            };
            unsafe {
                let slot = slot_addr as *mut i64;
                *slot = int;
            }
        }
    }

    Ok(WireValue::Heap(ptr as *const u8))
}

unsafe fn decode_struct(rt: &Runtime, r: &mut Reader) -> Result<WireValue, WireError> {
    let struct_ref = r.read_hash()?;
    let n_fields = r.read_u32()?;
    let ti = rt
        .type_info_for(&struct_ref)
        .ok_or(WireError::MissingShape(struct_ref))?;
    let ptr = unsafe { ai_gc_alloc_closure(rt.thread_ptr(), ti) };

    let meta = rt
        .shape_meta
        .get(&struct_ref)
        .ok_or(WireError::MissingShape(struct_ref))?
        .clone();
    let fields = match meta {
        ShapeMeta::Struct { fields, .. } => fields,
        _ => return Err(WireError::Corrupt("registered struct hash isn't a Struct shape")),
    };
    if fields.len() as u32 != n_fields {
        return Err(WireError::Corrupt(
            "struct wire payload field-count disagrees with registered shape",
        ));
    }

    for field in &fields {
        let v = unsafe { decode_one(rt, r)? };
        unsafe { store_field(ptr, field, v)? };
    }
    Ok(WireValue::Heap(ptr as *const u8))
}

unsafe fn decode_enum(rt: &Runtime, r: &mut Reader) -> Result<WireValue, WireError> {
    let enum_ref = r.read_hash()?;
    let variant_index = r.read_u32()?;
    let has_payload = r.read_u8()?;

    // Look up the variant's shape via derive_variant_hash. We need the
    // variant *name* but only have its index — pull from shape_meta by
    // scanning entries (acceptable for small enums; can be indexed
    // later).
    let variant_hash = find_variant_hash(rt, enum_ref, variant_index)?;
    let ti = rt
        .type_info_for(&variant_hash)
        .ok_or(WireError::MissingShape(variant_hash))?;
    let ptr = unsafe { ai_gc_alloc_closure(rt.thread_ptr(), ti) };

    let meta = rt
        .shape_meta
        .get(&variant_hash)
        .ok_or(WireError::MissingShape(variant_hash))?
        .clone();
    let (tag_offset, payload_field) = match meta {
        ShapeMeta::EnumVariant {
            tag_offset,
            payload,
            ..
        } => (tag_offset, payload),
        _ => return Err(WireError::Corrupt("variant hash isn't an EnumVariant shape")),
    };

    // Store tag.
    unsafe {
        let slot = ptr.add(tag_offset as usize) as *mut u32;
        *slot = variant_index;
    }

    match (has_payload, payload_field) {
        (0, None) => {}
        (1, Some(field)) => {
            let v = unsafe { decode_one(rt, r)? };
            unsafe { store_field(ptr, &field, v)? };
        }
        (0, Some(_)) | (1, None) => {
            return Err(WireError::Corrupt(
                "enum wire payload presence disagrees with registered variant",
            ));
        }
        (other, _) => return Err(WireError::BadOption(other)),
    }

    Ok(WireValue::Heap(ptr as *const u8))
}

fn find_variant_hash(rt: &Runtime, enum_ref: Hash, variant_index: u32) -> Result<Hash, WireError> {
    for (h, m) in rt.shape_meta.iter() {
        if let ShapeMeta::EnumVariant {
            enum_ref: er,
            variant_index: vi,
            ..
        } = m
        {
            if *er == enum_ref && *vi == variant_index {
                return Ok(*h);
            }
        }
    }
    Err(WireError::MissingShape(enum_ref))
}

unsafe fn store_field(
    obj_ptr: *mut u8,
    field: &FieldMeta,
    v: WireValue,
) -> Result<(), WireError> {
    let slot = unsafe { obj_ptr.add(field.offset as usize) };
    if field.is_pointer {
        let ptr = match v {
            WireValue::Heap(p) => p,
            WireValue::Int(_) => {
                return Err(WireError::Corrupt(
                    "expected heap pointer for pointer-typed field; received Int",
                ));
            }
        };
        unsafe {
            *(slot as *mut *const u8) = ptr;
        }
    } else {
        let int = match v {
            WireValue::Int(n) => n,
            WireValue::Heap(_) => {
                return Err(WireError::Corrupt(
                    "expected Int for non-pointer field; received heap pointer",
                ));
            }
        };
        unsafe {
            *(slot as *mut i64) = int;
        }
    }
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{CompiledModule, Jit, init_native_target};
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;
    use inkwell::context::Context;
    use std::sync::Once;

    static INIT: Once = Once::new();
    fn init() {
        INIT.call_once(|| {
            init_native_target().expect("init native target");
        });
    }

    fn make_runtime<'ctx>(
        ctx: &'ctx Context,
        src: &str,
    ) -> (Runtime, Jit<'ctx>, std::collections::HashMap<String, Hash>) {
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let names: std::collections::HashMap<String, Hash> =
            r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
        let cm = CompiledModule::build(ctx, &r).unwrap();
        let rt = Runtime::new_with_metadata(
            cm.closure_type_infos.clone(),
            cm.shape_registry.clone(),
            cm.shape_meta.clone(),
            cm.shape_by_type_id.clone(),
        );
        let jit = Jit::new(cm, &rt).unwrap();
        (rt, jit, names)
    }

    #[test]
    fn int_roundtrip() {
        init();
        let ctx = Context::create();
        let (rt, _jit, _names) = make_runtime(&ctx, "def x() -> Int = 0");
        let mut buf = Vec::new();
        unsafe { encode_value(&rt, WireValue::Int(42), &mut buf).unwrap() };
        let (v, n) = unsafe { decode_value(&rt, &buf).unwrap() };
        assert_eq!(n, buf.len());
        match v {
            WireValue::Int(x) => assert_eq!(x, 42),
            _ => panic!("expected Int"),
        }
    }

    #[test]
    fn closure_with_int_capture_roundtrips_and_runs() {
        init();
        let ctx_a = Context::create();
        let ctx_b = Context::create();
        let src = "
            def make_adder(n: Int) -> fn(Int) -> Int = |x: Int| x + n
            def run(closure_n: Int, x: Int) -> Int = {
                let f = make_adder(closure_n);
                f(x)
            }
        ";
        let (rt_a, jit_a, names_a) = make_runtime(&ctx_a, src);
        let (rt_b, jit_b, names_b) = make_runtime(&ctx_b, src);

        // On A: build a closure capturing 10 via make_adder(10).
        // We can't easily call make_adder directly from Rust (it
        // returns a pointer), but we can call run(10, x) and inspect:
        // instead, allocate directly by encoding "Some(7)" via the
        // closure path through the heap. We'll go via jit calls.

        // Build the closure on A: call make_adder(10), get a closure
        // pointer back.
        let h_make = names_a["make_adder"];
        let make_fn = unsafe {
            jit_a
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut crate::runtime::Thread,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&h_make))
                .unwrap()
        };
        let closure_a = unsafe { make_fn.call(rt_a.thread_ptr(), 10) };

        // Encode on A.
        let mut buf = Vec::new();
        unsafe {
            encode_value(&rt_a, WireValue::Heap(closure_a as *const u8), &mut buf).unwrap()
        };

        // Decode on B. B has its own heap but the SAME module compiled,
        // so its shape registry contains the same code_hash.
        let (decoded, _) = unsafe { decode_value(&rt_b, &buf).unwrap() };
        let closure_b = match decoded {
            WireValue::Heap(p) => p,
            _ => panic!("expected heap closure"),
        };

        // Invoke the closure on B by JIT-looking up the lifted lambda
        // via the code table and calling it directly. With the uniform
        // closure ABI (introduced for generic HOFs), lambda signatures
        // are always (*Thread, *Closure, *u8 × N) -> *u8: callers must
        // box Int args via `ai_gc_box_int` and unbox Int returns via
        // `ai_gc_unbox_int`.
        let code_hash = {
            let mut h = [0u8; 32];
            unsafe {
                core::ptr::copy_nonoverlapping(
                    (closure_b as *const u8).add(Full::SIZE),
                    h.as_mut_ptr(),
                    32,
                )
            };
            Hash(h)
        };
        let fn_ptr = rt_b.code_table.lookup(&code_hash).unwrap();
        let lambda: unsafe extern "C" fn(
            *mut crate::runtime::Thread,
            *const u8,
            *const u8,
        ) -> *mut u8 =
            unsafe { core::mem::transmute(fn_ptr) };
        let boxed_5 = unsafe { crate::runtime::ai_gc_box_int(rt_b.thread_ptr(), 5) };
        let ret_ptr = unsafe { lambda(rt_b.thread_ptr(), closure_b, boxed_5) };
        let result = unsafe { crate::runtime::ai_gc_unbox_int(ret_ptr) };
        assert_eq!(result, 15);

        // Both names_b and the closure on B share the same code_hash
        // because both runtimes JIT'd the same canonical module.
        let _ = names_b;
        let _ = jit_b;
    }

    #[test]
    fn struct_roundtrips_and_field_access_works() {
        init();
        let ctx_a = Context::create();
        let ctx_b = Context::create();
        let src = "
            struct Point { x: Int, y: Int }
            def make(a: Int, b: Int) -> Point = Point { x: a, y: b }
            def get_x(p: Point) -> Int = p.x
            def get_y(p: Point) -> Int = p.y
        ";
        let (rt_a, jit_a, names_a) = make_runtime(&ctx_a, src);
        let (rt_b, jit_b, names_b) = make_runtime(&ctx_b, src);

        let make_fn = unsafe {
            jit_a
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut crate::runtime::Thread,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names_a["make"]))
                .unwrap()
        };
        let p_a = unsafe { make_fn.call(rt_a.thread_ptr(), 7, 99) };

        // Encode on A.
        let mut buf = Vec::new();
        unsafe { encode_value(&rt_a, WireValue::Heap(p_a as *const u8), &mut buf).unwrap() };

        // Decode on B.
        let (decoded, _) = unsafe { decode_value(&rt_b, &buf).unwrap() };
        let p_b = match decoded {
            WireValue::Heap(p) => p,
            _ => panic!("expected heap struct"),
        };

        // Call get_x and get_y on B with the decoded struct.
        let get_x = unsafe {
            jit_b
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut crate::runtime::Thread,
                    *const u8,
                ) -> i64>(&crate::codegen::def_symbol(&names_b["get_x"]))
                .unwrap()
        };
        let get_y = unsafe {
            jit_b
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut crate::runtime::Thread,
                    *const u8,
                ) -> i64>(&crate::codegen::def_symbol(&names_b["get_y"]))
                .unwrap()
        };
        assert_eq!(unsafe { get_x.call(rt_b.thread_ptr(), p_b) }, 7);
        assert_eq!(unsafe { get_y.call(rt_b.thread_ptr(), p_b) }, 99);
    }

    #[test]
    fn enum_with_int_payload_roundtrips() {
        init();
        let ctx_a = Context::create();
        let ctx_b = Context::create();
        let src = "
            enum IntOpt { Some(Int), None }
            def make_some(v: Int) -> IntOpt = IntOpt::Some(v)
            def get_or(o: IntOpt, default: Int) -> Int = match o {
                IntOpt::Some(x) => x,
                IntOpt::None => default,
            }
        ";
        let (rt_a, jit_a, names_a) = make_runtime(&ctx_a, src);
        let (rt_b, jit_b, names_b) = make_runtime(&ctx_b, src);

        let make_some = unsafe {
            jit_a
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut crate::runtime::Thread,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names_a["make_some"]))
                .unwrap()
        };
        let some_a = unsafe { make_some.call(rt_a.thread_ptr(), 123) };

        let mut buf = Vec::new();
        unsafe { encode_value(&rt_a, WireValue::Heap(some_a as *const u8), &mut buf).unwrap() };

        let (decoded, _) = unsafe { decode_value(&rt_b, &buf).unwrap() };
        let some_b = match decoded {
            WireValue::Heap(p) => p,
            _ => panic!("expected heap enum"),
        };

        let get_or = unsafe {
            jit_b
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut crate::runtime::Thread,
                    *const u8,
                    i64,
                ) -> i64>(&crate::codegen::def_symbol(&names_b["get_or"]))
                .unwrap()
        };
        assert_eq!(unsafe { get_or.call(rt_b.thread_ptr(), some_b, 0) }, 123);
    }
}
