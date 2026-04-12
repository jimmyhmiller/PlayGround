//! Heap-backed delimited continuations.
//!
//! A captured continuation is represented as two GC heap objects:
//!
//! ```text
//!   ContObj  (registered TypeInfo):
//!     [header]
//!     field 0 : tagged ptr → ContMeta     (GC-traced)
//!     varlen Values tail : all captured frame values, flattened
//!                          bottom-to-top. Each slot is a tagged u64;
//!                          heap pointers are traced/forwarded, fixnums
//!                          pass through untouched.
//!
//!   ContMeta (registered TypeInfo):
//!     [header]
//!     varlen Bytes tail  : packed per-frame metadata (see METADATA
//!                          LAYOUT below). No GC-traced fields.
//! ```
//!
//! Why two objects? `TypeInfo` supports at most one varlen tail per
//! object. Flattened values go in ContObj's varlen Values tail; the
//! per-frame metadata (func/block/inst indices, caller_resume, active
//! prompts, root-index lists) goes in ContMeta's varlen Bytes tail.
//! The pointer in ContObj.field[0] keeps ContMeta alive transitively.
//!
//! Both allocations happen on a caller-provided heap (`&dyn Alloc`).
//! Nothing in this module owns an allocator.
//!
//! # Metadata layout (ContMeta's varlen Bytes, little-endian)
//!
//! Fixed 16-byte header:
//!   u32  frame_count
//!   u32  total_value_count   (= varlen length of the ContObj tail)
//!   u32  prompt_id
//!   u32  flags               (reserved, currently 0)
//!
//! Followed by `frame_count` variable-length frame entries. Each entry
//! starts with a fixed 40-byte preamble:
//!   u32  func_idx
//!   u32  block_idx
//!   u32  inst_idx
//!   u32  val_count
//!   u32  val_offset          (offset into ContObj's values tail)
//!   u32  resume_arg_slot     (u32::MAX = None; top frame only)
//!   u32  resume_kind         (0=TopLevel, 1=FromCall, 2=FromResume)
//!   u32  resume_extra_a      (FromCall::return_dest | FromResume::return_block; u32::MAX=None)
//!   u32  resume_extra_b      (FromResume::return_param_dest; u32::MAX=None)
//!   u32  array_word          (bit[0..16]=active_prompt_count, bit[16..32]=root_index_count)
//!
//! Then two inline arrays, each padded to u32 alignment:
//!   u32  active_prompts[active_prompt_count]
//!   u16  root_indices[root_index_count]        (padded tail to u32 boundary)
//!
//! This is a one-shot encode at capture and one linear-scan decode at
//! read. We cache per-frame byte offsets in the `ContinuationView` so
//! random access to `frame(i)` is O(1) after an O(frame_count) decode.

use core::marker::PhantomData;

use dynalloc::{Alloc, PtrPolicy};
use dynobj::{
    ObjHeader, TypeInfo,
    init_header, read_varlen_bytes, read_varlen_count, write_value_field,
    write_varlen_count,
};

// ─── Public types ──────────────────────────────────────────────────

/// Runtime type descriptors for captured-continuation heap objects.
///
/// Constructed via [`ContinuationTypes::register_into`], which appends
/// the ContObj and ContMeta TypeInfos to a type table under construction
/// and returns the descriptors with their type_ids baked in.
#[derive(Debug, Clone, Copy)]
pub struct ContinuationTypes {
    pub cont_obj: TypeInfo,
    pub cont_meta: TypeInfo,
}

impl ContinuationTypes {
    /// Append the ContObj and ContMeta TypeInfos to `type_table`, using
    /// the given object header type `H`. Returns a descriptor holding
    /// the TypeInfos with their type_ids assigned to match their
    /// positions in `type_table`.
    ///
    /// Call this before building the heap. Pass the augmented
    /// `type_table` to your heap constructor.
    pub fn register_into<H: ObjHeader>(type_table: &mut Vec<TypeInfo>) -> Self {
        let cont_meta_id = type_table.len() as u16;
        let cont_meta = TypeInfo::for_header(H::SIZE)
            .with_varlen_bytes(0)
            .with_type_id(cont_meta_id);
        type_table.push(cont_meta);

        let cont_obj_id = type_table.len() as u16;
        let cont_obj = TypeInfo::for_header(H::SIZE)
            .with_varlen_values(1) // field 0 = pointer to ContMeta
            .with_type_id(cont_obj_id);
        type_table.push(cont_obj);

        ContinuationTypes { cont_obj, cont_meta }
    }
}

/// Resume information for a captured frame. A restricted subset of
/// `FrameResume` — only the variants that appear in correctly-formed
/// captures. Capturing a frame that sits under a `try/catch` (FromInvoke)
/// is not supported and panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapturedResume {
    TopLevel,
    FromCall { return_dest: Option<u32> },
    FromResume { return_block: u32, return_param_dest: Option<u32> },
    FromInvoke {
        normal_block: u32,
        normal_args_vals: Vec<u64>,
        exception_block: u32,
        exception_args_vals: Vec<u64>,
        has_ret_param: bool,
    },
}

/// Describes a single frame to be captured. Construction is bottom-to-top
/// (innermost frame last) so `builder.frames[0]` is the frame that owns
/// the prompt and `builder.frames.last()` is the frame where `shift`
/// was invoked.
#[derive(Debug, Clone)]
pub struct BuilderFrame {
    pub func_idx: u32,
    pub block_idx: u32,
    pub inst_idx: u32,
    pub values: Vec<u64>,
    pub active_prompts: Vec<u32>,
    pub root_indices: Vec<u16>,
    pub resume_arg_slot: Option<u32>,
    pub caller_resume: CapturedResume,
}

/// Describes a full captured stack slice.
#[derive(Debug, Clone)]
pub struct CapturedStackBuilder {
    pub prompt_id: u32,
    pub frames: Vec<BuilderFrame>,
}

/// Read-only view of a previously-captured continuation. The lifetime
/// is tied to the heap borrow passed to [`read_continuation`] — the
/// borrow checker will reject holding a view across a call that could
/// trigger a collection on the same heap (provided the heap enforces
/// exclusive `&mut` for collection, which the standard `RefCell`
/// wrapper pattern provides).
pub struct ContinuationView<'h> {
    /// Pointer to the ContObj. Used to reach the values tail.
    cont_obj: *const u8,
    /// Copy of the TypeInfo (so we don't keep another borrow alive).
    cont_obj_info: TypeInfo,
    /// Pointer to the ContMeta bytes section. Used for metadata reads.
    meta_bytes: *const u8,
    /// Length of the meta bytes.
    meta_len: usize,
    /// Header: prompt_id.
    prompt_id: u32,
    /// Cached per-frame metadata byte offsets, decoded once on read.
    frame_byte_offsets: Vec<usize>,
    /// Total value count in the ContObj varlen tail.
    total_value_count: usize,
    _heap_borrow: PhantomData<&'h ()>,
}

impl<'h> ContinuationView<'h> {
    pub fn frame_count(&self) -> usize {
        self.frame_byte_offsets.len()
    }

    pub fn prompt_id(&self) -> u32 {
        self.prompt_id
    }

    pub fn total_value_count(&self) -> usize {
        self.total_value_count
    }

    /// Decode frame `i`. Returns a `FrameView` that borrows the
    /// ContObj's values tail and the ContMeta's metadata bytes.
    pub fn frame(&self, i: usize) -> FrameView<'_> {
        let off = self.frame_byte_offsets[i];
        let bytes = unsafe { core::slice::from_raw_parts(self.meta_bytes, self.meta_len) };
        let (preamble, rest) = decode_frame_preamble(bytes, off);

        let active_prompts_off = off + FRAME_PREAMBLE_SIZE;
        let active_prompts_end = active_prompts_off + 4 * preamble.active_prompt_count as usize;
        let active_prompts = unsafe {
            core::slice::from_raw_parts(
                self.meta_bytes.add(active_prompts_off) as *const u32,
                preamble.active_prompt_count as usize,
            )
        };

        let root_indices_off = active_prompts_end;
        let root_indices = unsafe {
            core::slice::from_raw_parts(
                self.meta_bytes.add(root_indices_off) as *const u16,
                preamble.root_index_count as usize,
            )
        };

        // Values slice out of ContObj's varlen Values tail.
        let base_offset = self.cont_obj_info.varlen_count_offset() + 8;
        let vals_ptr = unsafe {
            self.cont_obj.add(base_offset + preamble.val_offset as usize * 8) as *const u64
        };
        let values = unsafe {
            core::slice::from_raw_parts(vals_ptr, preamble.val_count as usize)
        };

        let mut caller_resume = decode_caller_resume(
            preamble.resume_kind,
            preamble.resume_extra_a,
            preamble.resume_extra_b,
        );

        // For FromInvoke, decode the inline args that follow root_indices.
        if preamble.resume_kind == RESUME_KIND_FROM_INVOKE {
            let root_idx_bytes = 2 * preamble.root_index_count as usize;
            let invoke_off = root_indices_off + ((root_idx_bytes + 3) & !3);
            let normal_count = unsafe {
                u32::from_le_bytes(core::ptr::read(
                    self.meta_bytes.add(invoke_off) as *const [u8; 4],
                )) as usize
            };
            let exception_count = unsafe {
                u32::from_le_bytes(core::ptr::read(
                    self.meta_bytes.add(invoke_off + 4) as *const [u8; 4],
                )) as usize
            };
            let data_off = invoke_off + 8;
            let normal_args: Vec<u64> = (0..normal_count)
                .map(|j| unsafe {
                    core::ptr::read(self.meta_bytes.add(data_off + j * 8) as *const u64)
                })
                .collect();
            let exception_args: Vec<u64> = (0..exception_count)
                .map(|j| unsafe {
                    core::ptr::read(
                        self.meta_bytes
                            .add(data_off + normal_count * 8 + j * 8)
                            as *const u64,
                    )
                })
                .collect();
            if let CapturedResume::FromInvoke {
                normal_args_vals,
                exception_args_vals,
                ..
            } = &mut caller_resume
            {
                *normal_args_vals = normal_args;
                *exception_args_vals = exception_args;
            }
        }

        FrameView {
            func_idx: preamble.func_idx,
            block_idx: preamble.block_idx,
            inst_idx: preamble.inst_idx,
            values,
            active_prompts,
            root_indices,
            resume_arg_slot: if preamble.resume_arg_slot == u32::MAX {
                None
            } else {
                Some(preamble.resume_arg_slot)
            },
            caller_resume,
            _rest: PhantomData,
            _rest2: rest.len(), // keep the decoded size for debugging
        }
    }
}

/// Read-only view of one captured frame. `values`, `active_prompts`,
/// and `root_indices` borrow slices directly from the heap object —
/// zero-copy but borrow-bound to the enclosing view.
pub struct FrameView<'a> {
    pub func_idx: u32,
    pub block_idx: u32,
    pub inst_idx: u32,
    pub values: &'a [u64],
    pub active_prompts: &'a [u32],
    pub root_indices: &'a [u16],
    pub resume_arg_slot: Option<u32>,
    pub caller_resume: CapturedResume,
    _rest: PhantomData<&'a ()>,
    _rest2: usize,
}

// ─── Metadata encoding constants ───────────────────────────────────

const META_HEADER_SIZE: usize = 16;
const FRAME_PREAMBLE_SIZE: usize = 40;

const RESUME_KIND_TOP_LEVEL: u32 = 0;
const RESUME_KIND_FROM_CALL: u32 = 1;
const RESUME_KIND_FROM_RESUME: u32 = 2;
const RESUME_KIND_FROM_INVOKE: u32 = 3;

// ─── Metadata encoding helpers ─────────────────────────────────────

/// Encode the fixed part of a CapturedResume. For simple variants
/// this is all that's needed. For FromInvoke the variable-length
/// args are written separately by `encode_meta`.
fn encode_caller_resume(r: &CapturedResume) -> (u32, u32, u32) {
    match r {
        CapturedResume::TopLevel => (RESUME_KIND_TOP_LEVEL, u32::MAX, u32::MAX),
        CapturedResume::FromCall { return_dest } => (
            RESUME_KIND_FROM_CALL,
            return_dest.unwrap_or(u32::MAX),
            u32::MAX,
        ),
        CapturedResume::FromResume { return_block, return_param_dest } => (
            RESUME_KIND_FROM_RESUME,
            *return_block,
            return_param_dest.unwrap_or(u32::MAX),
        ),
        CapturedResume::FromInvoke {
            normal_block,
            exception_block,
            has_ret_param,
            ..
        } => (
            RESUME_KIND_FROM_INVOKE,
            *normal_block,
            (*exception_block & 0x7FFF_FFFF) | ((*has_ret_param as u32) << 31),
        ),
    }
}

fn decode_caller_resume(kind: u32, a: u32, b: u32) -> CapturedResume {
    match kind {
        RESUME_KIND_TOP_LEVEL => CapturedResume::TopLevel,
        RESUME_KIND_FROM_CALL => CapturedResume::FromCall {
            return_dest: if a == u32::MAX { None } else { Some(a) },
        },
        RESUME_KIND_FROM_RESUME => CapturedResume::FromResume {
            return_block: a,
            return_param_dest: if b == u32::MAX { None } else { Some(b) },
        },
        RESUME_KIND_FROM_INVOKE => CapturedResume::FromInvoke {
            normal_block: a,
            exception_block: b & 0x7FFF_FFFF,
            has_ret_param: (b >> 31) != 0,
            // args are decoded separately from the variable-length tail
            normal_args_vals: Vec::new(),
            exception_args_vals: Vec::new(),
        },
        other => panic!("cont_heap: unknown resume_kind {other}"),
    }
}

/// Decoded preamble fields for one frame.
#[derive(Debug, Clone, Copy)]
struct FramePreamble {
    func_idx: u32,
    block_idx: u32,
    inst_idx: u32,
    val_count: u32,
    val_offset: u32,
    resume_arg_slot: u32,
    resume_kind: u32,
    resume_extra_a: u32,
    resume_extra_b: u32,
    active_prompt_count: u16,
    root_index_count: u16,
}

fn decode_frame_preamble(bytes: &[u8], off: usize) -> (FramePreamble, &[u8]) {
    let p = &bytes[off..off + FRAME_PREAMBLE_SIZE];
    let read_u32 = |i: usize| -> u32 {
        u32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]])
    };
    let array_word = read_u32(36);
    let preamble = FramePreamble {
        func_idx: read_u32(0),
        block_idx: read_u32(4),
        inst_idx: read_u32(8),
        val_count: read_u32(12),
        val_offset: read_u32(16),
        resume_arg_slot: read_u32(20),
        resume_kind: read_u32(24),
        resume_extra_a: read_u32(28),
        resume_extra_b: read_u32(32),
        active_prompt_count: (array_word & 0xFFFF) as u16,
        root_index_count: (array_word >> 16) as u16,
    };
    (preamble, &bytes[off + FRAME_PREAMBLE_SIZE..])
}

/// Compute the total encoded length of the metadata bytes for a builder.
fn encoded_meta_len(builder: &CapturedStackBuilder) -> usize {
    let mut len = META_HEADER_SIZE;
    for f in &builder.frames {
        len += FRAME_PREAMBLE_SIZE;
        len += 4 * f.active_prompts.len();
        let root_idx_bytes = 2 * f.root_indices.len();
        len += (root_idx_bytes + 3) & !3;
        // FromInvoke: store normal_args_vals and exception_args_vals
        // as u64 arrays preceded by two u32 counts.
        if let CapturedResume::FromInvoke {
            normal_args_vals,
            exception_args_vals,
            ..
        } = &f.caller_resume
        {
            len += 4; // u32 normal_args_count
            len += 4; // u32 exception_args_count
            len += 8 * normal_args_vals.len();
            len += 8 * exception_args_vals.len();
        }
    }
    len
}

/// Encode metadata into `out`, returning per-frame byte offsets into
/// `out` (for caching later in `ContinuationView`).
fn encode_meta(builder: &CapturedStackBuilder, out: &mut [u8]) -> Vec<usize> {
    let total_value_count: u32 = builder
        .frames
        .iter()
        .map(|f| f.values.len() as u32)
        .sum();

    // Header.
    out[0..4].copy_from_slice(&(builder.frames.len() as u32).to_le_bytes());
    out[4..8].copy_from_slice(&total_value_count.to_le_bytes());
    out[8..12].copy_from_slice(&builder.prompt_id.to_le_bytes());
    out[12..16].copy_from_slice(&0u32.to_le_bytes());

    let mut cursor = META_HEADER_SIZE;
    let mut offsets = Vec::with_capacity(builder.frames.len());
    let mut running_val_offset: u32 = 0;

    for f in &builder.frames {
        offsets.push(cursor);

        let (resume_kind, ra, rb) = encode_caller_resume(&f.caller_resume);
        let resume_arg_slot = f.resume_arg_slot.unwrap_or(u32::MAX);
        let array_word = (f.active_prompts.len() as u32 & 0xFFFF)
            | ((f.root_indices.len() as u32 & 0xFFFF) << 16);

        let p = &mut out[cursor..cursor + FRAME_PREAMBLE_SIZE];
        p[0..4].copy_from_slice(&f.func_idx.to_le_bytes());
        p[4..8].copy_from_slice(&f.block_idx.to_le_bytes());
        p[8..12].copy_from_slice(&f.inst_idx.to_le_bytes());
        p[12..16].copy_from_slice(&(f.values.len() as u32).to_le_bytes());
        p[16..20].copy_from_slice(&running_val_offset.to_le_bytes());
        p[20..24].copy_from_slice(&resume_arg_slot.to_le_bytes());
        p[24..28].copy_from_slice(&resume_kind.to_le_bytes());
        p[28..32].copy_from_slice(&ra.to_le_bytes());
        p[32..36].copy_from_slice(&rb.to_le_bytes());
        p[36..40].copy_from_slice(&array_word.to_le_bytes());

        cursor += FRAME_PREAMBLE_SIZE;

        for &prompt in &f.active_prompts {
            out[cursor..cursor + 4].copy_from_slice(&prompt.to_le_bytes());
            cursor += 4;
        }
        for &idx in &f.root_indices {
            out[cursor..cursor + 2].copy_from_slice(&idx.to_le_bytes());
            cursor += 2;
        }
        // Pad root_indices tail up to u32 alignment.
        while cursor & 3 != 0 {
            out[cursor] = 0;
            cursor += 1;
        }

        // FromInvoke: write invoke args inline.
        if let CapturedResume::FromInvoke {
            normal_args_vals,
            exception_args_vals,
            ..
        } = &f.caller_resume
        {
            out[cursor..cursor + 4]
                .copy_from_slice(&(normal_args_vals.len() as u32).to_le_bytes());
            cursor += 4;
            out[cursor..cursor + 4]
                .copy_from_slice(&(exception_args_vals.len() as u32).to_le_bytes());
            cursor += 4;
            for &val in normal_args_vals {
                out[cursor..cursor + 8].copy_from_slice(&val.to_le_bytes());
                cursor += 8;
            }
            for &val in exception_args_vals {
                out[cursor..cursor + 8].copy_from_slice(&val.to_le_bytes());
                cursor += 8;
            }
        }

        running_val_offset += f.values.len() as u32;
    }

    debug_assert_eq!(cursor, out.len());
    offsets
}

// ─── Capture / read ────────────────────────────────────────────────

/// Allocate a heap-backed captured continuation.
///
/// Returns a tagged handle (the ContObj pointer encoded via `P::encode_ptr`),
/// or `None` if the heap is out of memory. The caller is responsible for
/// storing the handle in a GC root slot (e.g., an IR value typed
/// `Type::FrameSlice`) so the continuation is kept alive and its
/// internal pointers get forwarded on subsequent collections.
///
/// Panics if any captured frame's `caller_resume` is a variant this
/// module does not support (currently only `FromInvoke` is
/// unsupported — see module docs).
///
/// # Safety contract
///
/// `H` must be the same header type used to construct `heap`. The
/// TypeInfos in `types` must have been registered in `heap`'s type
/// table (so the GC can find their layout during collection).
pub fn capture_continuation<H: ObjHeader, A: Alloc, P: PtrPolicy>(
    heap: &A,
    types: &ContinuationTypes,
    builder: &CapturedStackBuilder,
) -> Option<u64> {
    // Step 1: encode the metadata into a temporary buffer.
    let meta_len = encoded_meta_len(builder);
    let mut meta_bytes = vec![0u8; meta_len];
    let _offsets = encode_meta(builder, &mut meta_bytes);

    // Step 2: allocate the ContMeta object and copy the bytes.
    let meta_ptr = heap.alloc(&types.cont_meta, meta_len);
    if meta_ptr.is_null() {
        return None;
    }
    unsafe {
        init_header::<H>(meta_ptr, types.cont_meta.type_id);
        write_varlen_count(meta_ptr, &types.cont_meta, meta_len);
        let base = types.cont_meta.varlen_count_offset() + 8;
        core::ptr::copy_nonoverlapping(
            meta_bytes.as_ptr(),
            meta_ptr.add(base),
            meta_len,
        );
    }

    // Step 3: allocate the ContObj with a varlen Values tail large
    // enough for all frames' values, flattened.
    let total_values: usize = builder
        .frames
        .iter()
        .map(|f| f.values.len())
        .sum();
    let obj_ptr = heap.alloc(&types.cont_obj, total_values);
    if obj_ptr.is_null() {
        // We've leaked the meta allocation into from-space, but it's
        // fine — next GC will reclaim it (nothing references it).
        return None;
    }
    unsafe {
        init_header::<H>(obj_ptr, types.cont_obj.type_id);
        write_varlen_count(obj_ptr, &types.cont_obj, total_values);

        // field[0] = tagged pointer to ContMeta.
        let tagged_meta = P::encode_ptr(meta_ptr);
        write_value_field::<dynvalue::LowBit<3>>(
            obj_ptr,
            &types.cont_obj,
            0,
            dynvalue::Value::from_bits(tagged_meta),
        );
        // NB: write_value_field takes a specific TagScheme param; we
        // pass LowBit<3> but only its to_bits/from_bits are used and
        // those don't actually depend on the scheme — they just round-
        // trip the u64. PtrPolicy is what decides what's a pointer at
        // trace time.

        // Copy values into the varlen tail at their flat offsets.
        let base_offset = types.cont_obj.varlen_count_offset() + 8;
        let mut running = 0usize;
        for f in &builder.frames {
            let dst = obj_ptr.add(base_offset + running * 8) as *mut u64;
            core::ptr::copy_nonoverlapping(f.values.as_ptr(), dst, f.values.len());
            running += f.values.len();
        }
    }

    // Return the tagged handle.
    Some(P::encode_ptr(obj_ptr))
}

/// Read a previously-captured continuation.
///
/// Returns `None` if `handle` does not decode as a heap pointer under
/// `P`. The returned view borrows memory owned by `heap`; its lifetime
/// is tied to the heap borrow so holding a view across a call that
/// could trigger a collection on the same heap is statically rejected
/// (given the usual `&mut`-on-collect discipline).
///
/// The `_heap` parameter is not computationally required (we don't
/// call methods on it), but the borrow is essential for GC safety —
/// see module docs.
pub fn read_continuation<'h, A: Alloc, P: PtrPolicy>(
    _heap: &'h A,
    types: &ContinuationTypes,
    handle: u64,
) -> Option<ContinuationView<'h>> {
    let cont_obj = P::try_decode_ptr(handle)?;
    let cont_obj: *const u8 = cont_obj;

    // Read field[0] to find the ContMeta.
    let meta_field_offset = types.cont_obj.value_field_offset(0);
    let meta_tagged = unsafe { core::ptr::read(cont_obj.add(meta_field_offset) as *const u64) };
    let meta_ptr = P::try_decode_ptr(meta_tagged)?;

    // Grab the raw metadata bytes from the ContMeta.
    let meta_bytes_slice = unsafe { read_varlen_bytes(meta_ptr, &types.cont_meta) };
    let meta_bytes_ptr = meta_bytes_slice.as_ptr();
    let meta_len = meta_bytes_slice.len();

    // Decode the 16-byte header.
    if meta_len < META_HEADER_SIZE {
        return None;
    }
    let hdr = meta_bytes_slice;
    let frame_count = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
    let total_value_count =
        u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    let prompt_id = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]);

    // Linear-scan to compute per-frame byte offsets.
    let mut frame_byte_offsets = Vec::with_capacity(frame_count);
    let mut cursor = META_HEADER_SIZE;
    for _ in 0..frame_count {
        frame_byte_offsets.push(cursor);
        if cursor + FRAME_PREAMBLE_SIZE > meta_len {
            return None;
        }
        let (preamble, _) = decode_frame_preamble(meta_bytes_slice, cursor);
        cursor += FRAME_PREAMBLE_SIZE;
        cursor += 4 * preamble.active_prompt_count as usize;
        let root_idx_bytes = 2 * preamble.root_index_count as usize;
        cursor += (root_idx_bytes + 3) & !3;
        // FromInvoke: skip the inline invoke args (two u32 counts + u64 arrays).
        if preamble.resume_kind == RESUME_KIND_FROM_INVOKE {
            let normal_count = u32::from_le_bytes([
                meta_bytes_slice[cursor],
                meta_bytes_slice[cursor + 1],
                meta_bytes_slice[cursor + 2],
                meta_bytes_slice[cursor + 3],
            ]) as usize;
            let exception_count = u32::from_le_bytes([
                meta_bytes_slice[cursor + 4],
                meta_bytes_slice[cursor + 5],
                meta_bytes_slice[cursor + 6],
                meta_bytes_slice[cursor + 7],
            ]) as usize;
            cursor += 8; // two u32 counts
            cursor += 8 * normal_count;
            cursor += 8 * exception_count;
        }
    }

    // Sanity-check ContObj's varlen count.
    let obj_val_count = unsafe { read_varlen_count(cont_obj, &types.cont_obj) };
    if obj_val_count != total_value_count {
        return None;
    }

    Some(ContinuationView {
        cont_obj,
        cont_obj_info: types.cont_obj,
        meta_bytes: meta_bytes_ptr,
        meta_len,
        prompt_id,
        frame_byte_offsets,
        total_value_count,
        _heap_borrow: PhantomData,
    })
}

// ─── Conversion utilities ─────────────────────────────────────────
//
// Bridge between the old `FrameSliceSnapshot`-based API and the new
// `CapturedStackBuilder`/`ContinuationView`-based API.

use crate::{
    CapturedCallerResume, CapturedFrame, FrameResume, FrameResumePoint,
    FrameSliceSnapshot,
};

/// Convert a legacy `FrameSliceSnapshot` into a `CapturedStackBuilder`
/// suitable for `capture_continuation`.
pub fn snapshot_to_builder(snapshot: &FrameSliceSnapshot) -> CapturedStackBuilder {
    let frames = snapshot
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let is_top = i + 1 == snapshot.frames.len();
            BuilderFrame {
                func_idx: f.resume.func_idx as u32,
                block_idx: f.resume.block_idx as u32,
                inst_idx: f.resume.inst_idx as u32,
                values: f.values.clone(),
                active_prompts: f.active_prompts.clone(),
                root_indices: f.root_value_indices.iter().map(|&i| i as u16).collect(),
                resume_arg_slot: if is_top {
                    f.resume_arg_value_indices.first().map(|&i| i as u32)
                } else {
                    None
                },
                caller_resume: match &f.caller_resume {
                    CapturedCallerResume::TopLevel => CapturedResume::TopLevel,
                    CapturedCallerResume::FromCall { return_dest } => {
                        CapturedResume::FromCall {
                            return_dest: return_dest.map(|d| d as u32),
                        }
                    }
                    CapturedCallerResume::FromResume {
                        return_block,
                        return_param_dest,
                    } => CapturedResume::FromResume {
                        return_block: *return_block as u32,
                        return_param_dest: return_param_dest.map(|d| d as u32),
                    },
                    CapturedCallerResume::FromInvoke {
                        normal_block,
                        normal_args_vals,
                        exception_block,
                        exception_args_vals,
                        has_ret_param,
                    } => CapturedResume::FromInvoke {
                        normal_block: *normal_block as u32,
                        normal_args_vals: normal_args_vals.clone(),
                        exception_block: *exception_block as u32,
                        exception_args_vals: exception_args_vals.clone(),
                        has_ret_param: *has_ret_param,
                    },
                },
            }
        })
        .collect();
    CapturedStackBuilder {
        prompt_id: snapshot.prompt_id,
        frames,
    }
}

/// Materialize a `FrameSliceSnapshot` from a `ContinuationView`.
/// Used when the old API shape is needed (e.g., passing to
/// `interpreter.resume_snapshot`).
pub fn view_to_snapshot(view: &ContinuationView<'_>) -> FrameSliceSnapshot {
    let mut frames = Vec::with_capacity(view.frame_count());
    for i in 0..view.frame_count() {
        let f = view.frame(i);
        let caller_resume = match &f.caller_resume {
            CapturedResume::TopLevel => FrameResume::TopLevel,
            CapturedResume::FromCall { return_dest } => FrameResume::FromCall {
                return_dest: return_dest.map(|d| d as usize),
            },
            CapturedResume::FromResume {
                return_block,
                return_param_dest,
            } => FrameResume::FromResume {
                return_block: *return_block as usize,
                return_param_dest: return_param_dest.map(|d| d as usize),
            },
            CapturedResume::FromInvoke {
                normal_block,
                normal_args_vals,
                exception_block,
                exception_args_vals,
                has_ret_param,
            } => FrameResume::FromInvoke {
                normal_block: *normal_block as usize,
                normal_args_vals: normal_args_vals.clone(),
                exception_block: *exception_block as usize,
                exception_args_vals: exception_args_vals.clone(),
                has_ret_param: *has_ret_param,
            },
        };
        frames.push(CapturedFrame {
            resume: FrameResumePoint {
                func_idx: f.func_idx as usize,
                block_idx: f.block_idx as usize,
                inst_idx: f.inst_idx as usize,
            },
            values: f.values.to_vec(),
            root_value_indices: f.root_indices.iter().map(|&i| i as usize).collect(),
            resume_arg_value_indices: f
                .resume_arg_slot
                .map(|s| vec![s as usize])
                .unwrap_or_default(),
            active_prompts: f.active_prompts.to_vec(),
            caller_resume,
        });
    }
    FrameSliceSnapshot {
        prompt_id: view.prompt_id(),
        frames,
    }
}

// ─── ContinuationContext trait ─────────────────────────────────────

/// The contract the interpreter uses to create and read heap-backed
/// continuations. The interpreter's capture/resume instruction
/// handlers call through this trait, so the concrete heap + pointer
/// scheme are hidden behind one plug-in point.
///
/// Implemented by:
///   - `NoGcContContext` (panics on all ops — use when the interpreter
///     is running a program that never captures a continuation)
///   - `dynir::gc_runtime::GcInterpCtx` (real heap-backed impl)
///
/// The `'h` lifetime on `read` ties the returned view to the self
/// borrow, so the view cannot outlive the call that produced it.
/// In the reference interpreter, the dispatch loop reads a view,
/// copies out the frame data it needs, drops the view, and only then
/// pushes live frames (which may allocate).
pub trait ContinuationContext {
    /// Allocate a captured continuation. Returns `None` on OOM.
    fn capture(&self, builder: &CapturedStackBuilder) -> Option<u64>;

    /// Read a previously-captured continuation. Returns `None` if the
    /// handle does not decode as a valid continuation pointer.
    fn read<'h>(&'h self, handle: u64) -> Option<ContinuationView<'h>>;
}

/// A no-op `ContinuationContext` for programs that never capture or
/// resume. Any call panics with a clear message.
pub struct NoContinuations;

impl ContinuationContext for NoContinuations {
    fn capture(&self, _builder: &CapturedStackBuilder) -> Option<u64> {
        panic!(
            "NoContinuations::capture called — this interpreter was \
             not configured with a heap-backed continuation context"
        );
    }
    fn read<'h>(&'h self, _handle: u64) -> Option<ContinuationView<'h>> {
        panic!(
            "NoContinuations::read called — this interpreter was \
             not configured with a heap-backed continuation context"
        );
    }
}

// ─── Unit tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use dynalloc::{PtrPolicy, SemiSpace};
    use dynobj::Compact;

    /// A trivial PtrPolicy for tests: treat any non-zero, non-fixnum-
    /// tagged value as a raw pointer. We use the LowBit<3> scheme
    /// where tag 0 = pointer, tag 1 = fixnum.
    struct TestPolicy;
    impl PtrPolicy for TestPolicy {
        fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
            if bits == 0 { return None; }
            // tag 0 = pointer in LowBit<3>
            if bits & 0b111 == 0 {
                Some(bits as *mut u8)
            } else {
                None
            }
        }
        fn encode_ptr(ptr: *mut u8) -> u64 {
            // ptr must already be 8-byte aligned (bump allocator ensures this)
            debug_assert_eq!((ptr as u64) & 0b111, 0);
            ptr as u64
        }
    }

    /// An `Alloc` impl for `RefCell<SemiSpace>` so tests can share a
    /// heap through a plain reference without threading `&mut` around.
    struct RcSemi(RefCell<SemiSpace>);
    impl Alloc for RcSemi {
        fn alloc(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
            self.0.borrow().alloc(info, varlen_len)
        }
    }

    fn build_test_heap() -> (RcSemi, ContinuationTypes, Vec<TypeInfo>) {
        let mut type_table: Vec<TypeInfo> = Vec::new();
        let types = ContinuationTypes::register_into::<Compact>(&mut type_table);
        let ss = SemiSpace::new::<Compact>(64 * 1024);
        (RcSemi(RefCell::new(ss)), types, type_table)
    }

    #[test]
    fn roundtrip_single_frame_no_prompts() {
        let (heap, types, _table) = build_test_heap();
        let builder = CapturedStackBuilder {
            prompt_id: 7,
            frames: vec![BuilderFrame {
                func_idx: 3,
                block_idx: 1,
                inst_idx: 5,
                values: vec![10, 20, 30, 40],
                active_prompts: vec![7],
                root_indices: vec![],
                resume_arg_slot: Some(2),
                caller_resume: CapturedResume::FromCall { return_dest: Some(9) },
            }],
        };

        let handle = capture_continuation::<Compact, _, TestPolicy>(&heap, &types, &builder)
            .expect("capture OOM");
        let view = read_continuation::<_, TestPolicy>(&heap, &types, handle)
            .expect("read failed");

        assert_eq!(view.frame_count(), 1);
        assert_eq!(view.prompt_id(), 7);
        assert_eq!(view.total_value_count(), 4);

        let f = view.frame(0);
        assert_eq!(f.func_idx, 3);
        assert_eq!(f.block_idx, 1);
        assert_eq!(f.inst_idx, 5);
        assert_eq!(f.values, &[10u64, 20, 30, 40]);
        assert_eq!(f.active_prompts, &[7u32]);
        assert_eq!(f.root_indices.len(), 0);
        assert_eq!(f.resume_arg_slot, Some(2));
        assert_eq!(f.caller_resume, CapturedResume::FromCall { return_dest: Some(9) });
    }

    #[test]
    fn roundtrip_multi_frame_mixed_prompts() {
        let (heap, types, _table) = build_test_heap();
        let builder = CapturedStackBuilder {
            prompt_id: 12,
            frames: vec![
                BuilderFrame {
                    func_idx: 0,
                    block_idx: 0,
                    inst_idx: 0,
                    values: vec![100, 200],
                    active_prompts: vec![12],
                    root_indices: vec![0],
                    resume_arg_slot: None,
                    caller_resume: CapturedResume::FromCall { return_dest: None },
                },
                BuilderFrame {
                    func_idx: 1,
                    block_idx: 2,
                    inst_idx: 3,
                    values: vec![1, 2, 3],
                    active_prompts: vec![12, 99],
                    root_indices: vec![1, 2],
                    resume_arg_slot: None,
                    caller_resume: CapturedResume::FromCall { return_dest: Some(5) },
                },
                BuilderFrame {
                    func_idx: 2,
                    block_idx: 0,
                    inst_idx: 7,
                    values: vec![42],
                    active_prompts: vec![12, 99],
                    root_indices: vec![],
                    resume_arg_slot: Some(0),
                    caller_resume: CapturedResume::FromResume {
                        return_block: 4,
                        return_param_dest: Some(6),
                    },
                },
            ],
        };

        let handle = capture_continuation::<Compact, _, TestPolicy>(&heap, &types, &builder)
            .expect("capture OOM");
        let view = read_continuation::<_, TestPolicy>(&heap, &types, handle)
            .expect("read failed");

        assert_eq!(view.frame_count(), 3);
        assert_eq!(view.prompt_id(), 12);
        assert_eq!(view.total_value_count(), 2 + 3 + 1);

        let f0 = view.frame(0);
        assert_eq!(f0.values, &[100u64, 200]);
        assert_eq!(f0.root_indices, &[0u16]);
        assert_eq!(f0.active_prompts, &[12u32]);
        assert_eq!(f0.caller_resume, CapturedResume::FromCall { return_dest: None });

        let f1 = view.frame(1);
        assert_eq!(f1.func_idx, 1);
        assert_eq!(f1.block_idx, 2);
        assert_eq!(f1.inst_idx, 3);
        assert_eq!(f1.values, &[1u64, 2, 3]);
        assert_eq!(f1.active_prompts, &[12u32, 99]);
        assert_eq!(f1.root_indices, &[1u16, 2]);
        assert_eq!(f1.caller_resume, CapturedResume::FromCall { return_dest: Some(5) });

        let f2 = view.frame(2);
        assert_eq!(f2.func_idx, 2);
        assert_eq!(f2.values, &[42u64]);
        assert_eq!(f2.resume_arg_slot, Some(0));
        assert_eq!(
            f2.caller_resume,
            CapturedResume::FromResume { return_block: 4, return_param_dest: Some(6) }
        );
    }

    /// The critical test: a captured continuation holds a tagged heap
    /// pointer in one of its value slots. We trigger a GC that moves
    /// the referenced object to to-space. After GC, the captured slot
    /// must point at the NEW location, and the object's contents must
    /// be intact.
    ///
    /// This is the whole reason continuations are heap objects.
    #[test]
    fn gc_forwards_pointers_captured_inside_continuation() {
        use core::cell::Cell;
        use dynobj::{init_header as dyn_init_header, write_varlen_count as dyn_write_varlen_count, RootSource, VarLenKind};

        // ── Set up heap + type table ──────────────────────────────
        // We need a user type to allocate (so the continuation has
        // something real to capture). Use a tiny byte-array type.
        let user_type_id: u16 = 0;
        let user_type = TypeInfo::for_header(Compact::SIZE)
            .with_varlen_bytes(0)
            .with_type_id(user_type_id);

        let mut type_table: Vec<TypeInfo> = vec![user_type];
        // ContMeta goes in as type_id 1, ContObj as type_id 2.
        let types = ContinuationTypes::register_into::<Compact>(&mut type_table);
        assert_eq!(types.cont_meta.type_id, 1);
        assert_eq!(types.cont_obj.type_id, 2);

        // Single SemiSpace, no RefCell this time — we need &mut for
        // collect and we're doing everything inline.
        let mut ss = SemiSpace::new::<Compact>(64 * 1024);

        // ── Allocate a user object and fill it with magic bytes ───
        const MAGIC: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
        let user_ptr = ss.alloc(&user_type, MAGIC.len());
        assert!(!user_ptr.is_null());
        unsafe {
            dyn_init_header::<Compact>(user_ptr, user_type_id);
            dyn_write_varlen_count(user_ptr, &user_type, MAGIC.len());
            let base = user_type.varlen_count_offset() + 8;
            core::ptr::copy_nonoverlapping(MAGIC.as_ptr(), user_ptr.add(base), MAGIC.len());
        }
        let user_tagged_before = TestPolicy::encode_ptr(user_ptr);

        // ── Capture a continuation that holds the user pointer ────
        let builder = CapturedStackBuilder {
            prompt_id: 1,
            frames: vec![BuilderFrame {
                func_idx: 0,
                block_idx: 0,
                inst_idx: 0,
                // First slot holds the tagged user pointer; other
                // slots are fixnums (LowBit<3> tag 1 is non-pointer).
                values: vec![user_tagged_before, 0b001, 0b001],
                active_prompts: vec![1],
                root_indices: vec![0], // slot 0 is a root
                resume_arg_slot: None,
                caller_resume: CapturedResume::TopLevel,
            }],
        };

        // Need a separate scope because alloc() on SemiSpace takes &self.
        let handle_before = capture_continuation::<Compact, _, TestPolicy>(
            &ss, &types, &builder,
        )
        .expect("capture OOM");

        // ── Set up a root source for the handle and collect ──────
        struct HandleRoot(Cell<u64>);
        impl RootSource for HandleRoot {
            fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
                visitor(self.0.as_ptr());
            }
        }
        let root = HandleRoot(Cell::new(handle_before));

        unsafe {
            ss.collect::<TestPolicy>(&type_table, &mut [&root]);
        }

        let handle_after = root.0.get();
        assert_ne!(handle_after, handle_before, "handle should have been forwarded");

        // ── Read the continuation via the new handle ──────────────
        let view = read_continuation::<_, TestPolicy>(&ss, &types, handle_after)
            .expect("read after GC failed");
        assert_eq!(view.frame_count(), 1);
        let f = view.frame(0);
        let captured_user = f.values[0];
        assert_ne!(captured_user, user_tagged_before,
            "captured user pointer should have been forwarded by GC");
        assert_eq!(f.values[1], 0b001);
        assert_eq!(f.values[2], 0b001);

        // ── Verify the user object contents are intact ────────────
        let new_user_ptr = TestPolicy::try_decode_ptr(captured_user).unwrap();
        let base = user_type.varlen_count_offset() + 8;
        let bytes = unsafe {
            core::slice::from_raw_parts(new_user_ptr.add(base) as *const u8, MAGIC.len())
        };
        assert_eq!(bytes, &MAGIC, "magic bytes must survive the move");

        // ── Sanity: the user type varlen count is also intact ────
        let varlen = unsafe { read_varlen_count(new_user_ptr, &user_type) };
        assert_eq!(varlen, MAGIC.len());

        // Silence unused warning.
        let _ = VarLenKind::Bytes;
    }

    /// Reclamation test: allocate several continuations, root only one,
    /// force a GC, and verify that from-space usage drops to just the
    /// reachable continuation's size. The unreachable continuations'
    /// memory (ContObj + ContMeta pairs) must be reclaimed.
    #[test]
    fn unreferenced_continuations_are_reclaimed() {
        use core::cell::Cell;
        use dynobj::RootSource;

        let mut type_table: Vec<TypeInfo> = Vec::new();
        let types = ContinuationTypes::register_into::<Compact>(&mut type_table);
        let mut ss = SemiSpace::new::<Compact>(64 * 1024);

        // Allocate 5 tiny continuations. Note that alloc_continuation
        // takes an `&dyn Alloc`, so we pass the SemiSpace directly.
        let mut handles = Vec::new();
        for i in 0..5u64 {
            let builder = CapturedStackBuilder {
                prompt_id: 1,
                frames: vec![BuilderFrame {
                    func_idx: 0,
                    block_idx: 0,
                    inst_idx: 0,
                    values: vec![i, i + 100],
                    active_prompts: vec![1],
                    root_indices: vec![],
                    resume_arg_slot: None,
                    caller_resume: CapturedResume::TopLevel,
                }],
            };
            let h = capture_continuation::<Compact, _, TestPolicy>(&ss, &types, &builder)
                .expect("capture OOM");
            handles.push(h);
        }

        let used_with_all = ss.from_used();
        assert!(used_with_all > 0);

        // Keep only handle[2] alive. Everything else is dropped from
        // the root set.
        struct Root(Cell<u64>);
        impl RootSource for Root {
            fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
                visitor(self.0.as_ptr());
            }
        }
        let keep = Root(Cell::new(handles[2]));

        unsafe {
            ss.collect::<TestPolicy>(&type_table, &mut [&keep]);
        }

        let used_after = ss.from_used();
        assert!(
            used_after < used_with_all,
            "reclamation must free the four unreferenced continuations \
             ({} bytes used before, {} after)",
            used_with_all,
            used_after,
        );

        // The surviving continuation should still be readable via the
        // (possibly forwarded) handle.
        let handle_after = keep.0.get();
        let view = read_continuation::<_, TestPolicy>(&ss, &types, handle_after)
            .expect("surviving continuation should be readable");
        assert_eq!(view.frame_count(), 1);
        let f = view.frame(0);
        // Values [2, 102] correspond to handles[2] (i = 2).
        assert_eq!(f.values, &[2u64, 102]);

        // A second GC with an EMPTY root set should reclaim everything.
        let empty_roots: [&dyn RootSource; 0] = [];
        let mut empty_roots_mut: [&dyn RootSource; 0] = empty_roots;
        unsafe {
            ss.collect::<TestPolicy>(&type_table, &mut empty_roots_mut);
        }
        assert_eq!(
            ss.from_used(),
            0,
            "second GC with empty roots must fully reclaim the heap"
        );
    }

    #[test]
    fn read_is_multishot() {
        // The same handle can be read any number of times; the view is
        // read-only and does not mutate the heap objects. Prove this
        // by reading twice and checking both reads produce equal views.
        let (heap, types, _table) = build_test_heap();
        let builder = CapturedStackBuilder {
            prompt_id: 1,
            frames: vec![BuilderFrame {
                func_idx: 11,
                block_idx: 22,
                inst_idx: 33,
                values: vec![7, 8, 9],
                active_prompts: vec![1],
                root_indices: vec![],
                resume_arg_slot: None,
                caller_resume: CapturedResume::TopLevel,
            }],
        };

        let handle = capture_continuation::<Compact, _, TestPolicy>(&heap, &types, &builder)
            .expect("capture OOM");

        for _ in 0..3 {
            let view = read_continuation::<_, TestPolicy>(&heap, &types, handle)
                .expect("read failed");
            assert_eq!(view.frame_count(), 1);
            let f = view.frame(0);
            assert_eq!(f.func_idx, 11);
            assert_eq!(f.values, &[7u64, 8, 9]);
            assert_eq!(f.caller_resume, CapturedResume::TopLevel);
        }
    }
}
