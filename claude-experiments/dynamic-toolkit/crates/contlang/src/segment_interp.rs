//! GC-segment backend for the generic interpreter.
//!
//! Implements `StackBackend` with GC-allocated segments (SemiSpace collector).
//! Frame values are tagged: tag 0 = heap pointer, tag 1 = fixnum.
//! Continuations are GC objects. The collector traces segments and
//! continuations, reclaims unreferenced ones, and forwards moved objects.

use std::cell::Cell;

use dynalloc::{PtrPolicy, SemiSpace};
use dynexec::{CallerResume, FrameMeta, InterpFrame, StackBackend};
use dynir::ir::PromptId;
use dynobj::{Compact, ObjHeader, RootSource, TypeInfo};
use dynvalue::LowBit;

use crate::generic_interp::GenericInterpreter;

type TV = dynvalue::Value<LowBit<3>>;

// ─── Tagged Value Encoding ─────────────────────────────────────────

fn tag_fixnum(n: i64) -> u64 { TV::tagged(1, n as u64).to_bits() }

fn untag_fixnum(bits: u64) -> i64 {
    let payload = TV::from_bits(bits).payload();
    ((payload as i64) << 3) >> 3
}

fn tag_ptr(ptr: *mut u8) -> u64 {
    if ptr.is_null() { return 0; }
    TV::tagged(0, (ptr as u64) >> 3).to_bits()
}

fn untag_ptr(bits: u64) -> *mut u8 {
    (TV::from_bits(bits).payload() << 3) as *mut u8
}

fn is_tagged_ptr(bits: u64) -> bool {
    TV::from_bits(bits).has_tag(0) && bits != 0
}

pub struct SegPtrPolicy;
impl PtrPolicy for SegPtrPolicy {
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        if is_tagged_ptr(bits) { Some(untag_ptr(bits)) } else { None }
    }
    fn encode_ptr(ptr: *mut u8) -> u64 { tag_ptr(ptr) }
}

// ─── GC Object TypeInfos ──────────────────────────────────────────

static SEGMENT_INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(0);
static CONT_INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(1);
const CONT_SEG_FIELD: u16 = 0;  // field[0] = tagged ptr to top segment

const FRAME_HEADER_WORDS: usize = 9;

// ─── Segment access (raw, for inside the backend) ──────────────────

fn raw_get(seg: *mut u8, idx: usize) -> u64 {
    unsafe { *(seg.add(SEGMENT_INFO.varlen_element_offset(idx)) as *const u64) }
}
fn raw_set(seg: *mut u8, idx: usize, val: u64) {
    unsafe { *(seg.add(SEGMENT_INFO.varlen_element_offset(idx)) as *mut u64) = val; }
}

fn cont_raw_get(cont: *mut u8, idx: usize) -> u64 {
    unsafe { *(cont.add(CONT_INFO.varlen_element_offset(idx)) as *const u64) }
}
fn cont_raw_set(cont: *mut u8, idx: usize, val: u64) {
    unsafe { *(cont.add(CONT_INFO.varlen_element_offset(idx)) as *mut u64) = val; }
}

// ─── Root tracking ─────────────────────────────────────────────────

struct SegRoots {
    slots: Vec<Cell<u64>>,
}

impl SegRoots {
    fn new() -> Self { SegRoots { slots: Vec::new() } }
    fn clear(&mut self) { self.slots.clear(); }
    fn push_tagged(&mut self, ptr: *mut u8) -> usize {
        let idx = self.slots.len();
        self.slots.push(Cell::new(tag_ptr(ptr)));
        idx
    }
    fn get_ptr(&self, idx: usize) -> *mut u8 { untag_ptr(self.slots[idx].get()) }
}

impl RootSource for SegRoots {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for cell in &self.slots { visitor(cell.as_ptr()); }
    }
}

// ─── Segment and Continuation types ────────────────────────────────

#[derive(Clone)]
pub struct GCSegment {
    pub ptr: *mut u8,
    pub val_count: usize,
}

#[derive(Clone)]
pub struct GCContinuation {
    pub ptr: *mut u8,
}

// ─── GCSegmentBackend ──────────────────────────────────────────────

pub struct GCSegmentBackend {
    gc: SemiSpace,
    alloc_count: usize,
    gc_count: usize,
    gc_stress: Option<usize>,
}

impl GCSegmentBackend {
    pub fn new(heap_size: usize) -> Self {
        GCSegmentBackend {
            gc: SemiSpace::new::<Compact>(heap_size),
            alloc_count: 0,
            gc_count: 0,
            gc_stress: None,
        }
    }

    pub fn gc(&self) -> &SemiSpace { &self.gc }
    pub fn from_used(&self) -> usize { self.gc.from_used() }
    pub fn gc_count(&self) -> usize { self.gc_count }
    pub fn set_gc_stress(&mut self, n: usize) { self.gc_stress = Some(n); }

    pub fn collect_empty(&mut self) {
        let roots = SegRoots::new();
        unsafe { self.gc.collect::<SegPtrPolicy>(&mut [&roots]); }
        self.gc_count += 1;
    }

    fn do_safepoint(&mut self, frames: &mut [InterpFrame<Self>]) {
        let mut roots = SegRoots::new();
        // Root only the stack segments. Continuation pointers stored IN
        // segment slots are traced transitively by the GC via PtrPolicy.
        // Unreferenced continuations (not in any live segment) get collected.
        for frame in frames.iter() {
            roots.push_tagged(frame.segment.ptr);
        }

        unsafe { self.gc.collect::<SegPtrPolicy>(&mut [&roots]); }
        self.gc_count += 1;

        for (i, frame) in frames.iter_mut().enumerate() {
            frame.segment.ptr = roots.get_ptr(i);
        }
    }

    fn write_cont_frames(&self, cont: *mut u8, frames: &[InterpFrame<Self>], resume_dest: usize) {
        let frame_count = frames.len();
        cont_raw_set(cont, 0, tag_fixnum(frame_count as i64));

        let mut idx = 1;
        for (i, frame) in frames.iter().enumerate() {
            let is_top = i + 1 == frame_count;
            let m = &frame.meta;
            cont_raw_set(cont, idx, tag_ptr(frame.segment.ptr));          idx += 1;
            cont_raw_set(cont, idx, tag_fixnum(m.func_idx as i64));       idx += 1;
            cont_raw_set(cont, idx, tag_fixnum(m.block_idx as i64));      idx += 1;
            cont_raw_set(cont, idx, tag_fixnum(m.inst_idx as i64));       idx += 1;
            cont_raw_set(cont, idx, tag_fixnum(m.val_count as i64));      idx += 1;
            let (kind, dest) = match &m.resume {
                CallerResume::TopLevel => (0i64, -1i64),
                CallerResume::FromCall { return_dest } => (1, return_dest.map(|d| d as i64).unwrap_or(-1)),
            };
            cont_raw_set(cont, idx, tag_fixnum(kind));                    idx += 1;
            cont_raw_set(cont, idx, tag_fixnum(dest));                    idx += 1;
            let ra = if is_top { resume_dest as i64 } else { -1 };
            cont_raw_set(cont, idx, tag_fixnum(ra));                      idx += 1;
            cont_raw_set(cont, idx, tag_fixnum(m.active_prompts.len() as i64)); idx += 1;
            for &p in &m.active_prompts {
                cont_raw_set(cont, idx, tag_fixnum(p as i64));            idx += 1;
            }
        }
    }

    fn read_cont_frames(&self, cont: *mut u8) -> Vec<(FrameMeta, *mut u8, Option<usize>)> {
        let frame_count = untag_fixnum(cont_raw_get(cont, 0)) as usize;
        let mut result = Vec::with_capacity(frame_count);
        let mut idx = 1;
        for _ in 0..frame_count {
            let seg_ptr = untag_ptr(cont_raw_get(cont, idx));        idx += 1;
            let func_idx = untag_fixnum(cont_raw_get(cont, idx)) as usize; idx += 1;
            let block_idx = untag_fixnum(cont_raw_get(cont, idx)) as usize; idx += 1;
            let inst_idx = untag_fixnum(cont_raw_get(cont, idx)) as usize; idx += 1;
            let val_count = untag_fixnum(cont_raw_get(cont, idx)) as usize; idx += 1;
            let kind = untag_fixnum(cont_raw_get(cont, idx));        idx += 1;
            let dest_raw = untag_fixnum(cont_raw_get(cont, idx));    idx += 1;
            let resume = if kind == 0 {
                CallerResume::TopLevel
            } else {
                CallerResume::FromCall {
                    return_dest: if dest_raw < 0 { None } else { Some(dest_raw as usize) },
                }
            };
            let ra_raw = untag_fixnum(cont_raw_get(cont, idx));      idx += 1;
            let resume_arg = if ra_raw < 0 { None } else { Some(ra_raw as usize) };
            let prompt_count = untag_fixnum(cont_raw_get(cont, idx)) as usize; idx += 1;
            let mut active_prompts = Vec::with_capacity(prompt_count);
            for _ in 0..prompt_count {
                active_prompts.push(untag_fixnum(cont_raw_get(cont, idx)) as u32); idx += 1;
            }
            result.push((
                FrameMeta { func_idx, val_count, block_idx, inst_idx, resume, active_prompts },
                seg_ptr,
                resume_arg,
            ));
        }
        result
    }

    fn cont_varlen_size(frames: &[InterpFrame<Self>]) -> usize {
        1 + frames.iter().map(|f| FRAME_HEADER_WORDS + f.meta.active_prompts.len()).sum::<usize>()
    }
}

impl StackBackend for GCSegmentBackend {
    type Segment = GCSegment;
    type Continuation = GCContinuation;

    fn alloc_segment(&mut self, val_count: usize) -> GCSegment {
        let ptr = self.gc.alloc_obj::<Compact>(&SEGMENT_INFO, val_count);
        assert!(!ptr.is_null(), "segment allocation failed");
        self.alloc_count += 1;
        GCSegment { ptr, val_count }
    }

    fn needs_safepoint(&self) -> bool {
        if let Some(n) = self.gc_stress {
            self.alloc_count > 0 && self.alloc_count % n == 0
        } else {
            false
        }
    }

    fn get(seg: &GCSegment, idx: usize) -> u64 {
        let bits = raw_get(seg.ptr, idx);
        if is_tagged_ptr(bits) {
            bits // tagged pointer — pass through (for continuation handles)
        } else {
            untag_fixnum(bits) as u64 // untag fixnum → raw integer
        }
    }

    fn set(seg: &mut GCSegment, idx: usize, val: u64) {
        if is_tagged_ptr(val) {
            raw_set(seg.ptr, idx, val); // already a tagged pointer
        } else {
            raw_set(seg.ptr, idx, tag_fixnum(val as i64)); // tag as fixnum
        }
    }

    fn capture(
        &mut self,
        frames: &[InterpFrame<Self>],
        start_idx: usize,
        resume_dest: usize,
    ) -> GCContinuation {
        let captured = &frames[start_idx..];
        let varlen = Self::cont_varlen_size(captured);
        let cont = self.gc.alloc_obj::<Compact>(&CONT_INFO, varlen);
        assert!(!cont.is_null(), "cont allocation failed");

        // field[0] = tagged ptr to top segment
        let top_seg = frames.last().unwrap().segment.ptr;
        unsafe {
            let offset = CONT_INFO.value_field_offset(CONT_SEG_FIELD);
            *(cont.add(offset) as *mut u64) = tag_ptr(top_seg);
        }

        self.write_cont_frames(cont, captured, resume_dest);
        GCContinuation { ptr: cont }
    }

    fn restore(
        &mut self,
        cont: &GCContinuation,
        args: &[u64],
    ) -> Vec<InterpFrame<Self>> {
        let frame_data = self.read_cont_frames(cont.ptr);
        let frame_count = frame_data.len();
        let mut result = Vec::with_capacity(frame_count);

        for (i, (meta, seg_ptr, resume_arg)) in frame_data.into_iter().enumerate() {
            let mut segment = GCSegment { ptr: seg_ptr, val_count: meta.val_count };
            if i + 1 == frame_count {
                if let Some(idx) = resume_arg {
                    if let Some(&arg) = args.first() {
                        // Store the resume arg — it's a raw u64 from the interpreter
                        Self::set(&mut segment, idx, arg);
                    }
                }
            }
            result.push(InterpFrame { meta, segment });
        }
        result
    }

    fn clone_cont(&mut self, cont: &GCContinuation) -> GCContinuation {
        let varlen_count = unsafe { dynobj::read_varlen_count(cont.ptr, &CONT_INFO) };
        let cont_size = CONT_INFO.allocation_size(varlen_count);

        // Allocate new continuation, memcpy metadata
        let new_cont = self.gc.alloc_obj::<Compact>(&CONT_INFO, varlen_count);
        assert!(!new_cont.is_null(), "clone cont alloc failed");
        unsafe {
            std::ptr::copy_nonoverlapping(cont.ptr, new_cont, cont_size);
            dynobj::init_header::<Compact>(new_cont, &CONT_INFO as *const TypeInfo);
            dynobj::write_varlen_count(new_cont, &CONT_INFO, varlen_count);
        }

        // Clone each frame's segment via memcpy
        let frame_data = self.read_cont_frames(cont.ptr);
        let mut idx = 1; // skip frame_count word
        for (meta, seg_ptr, _) in &frame_data {
            let seg_size = SEGMENT_INFO.allocation_size(meta.val_count);
            let new_seg = self.gc.alloc_obj::<Compact>(&SEGMENT_INFO, meta.val_count);
            assert!(!new_seg.is_null(), "clone segment alloc failed");
            unsafe {
                std::ptr::copy_nonoverlapping(*seg_ptr, new_seg, seg_size);
                dynobj::init_header::<Compact>(new_seg, &SEGMENT_INFO as *const TypeInfo);
                dynobj::write_varlen_count(new_seg, &SEGMENT_INFO, meta.val_count);
            }
            // Update the segment pointer in the new continuation
            cont_raw_set(new_cont, idx, tag_ptr(new_seg));
            idx += FRAME_HEADER_WORDS + meta.active_prompts.len();
        }

        // Update field[0] to point to new top segment
        if let Some((meta, _, _)) = frame_data.last() {
            let _ = meta;
            // Find the last frame's segment slot
            let mut last_idx = 1;
            for (i, (m, _, _)) in frame_data.iter().enumerate() {
                if i + 1 == frame_data.len() { break; }
                last_idx += FRAME_HEADER_WORDS + m.active_prompts.len();
            }
            let top_seg_bits = cont_raw_get(new_cont, last_idx);
            unsafe {
                let offset = CONT_INFO.value_field_offset(CONT_SEG_FIELD);
                *(new_cont.add(offset) as *mut u64) = top_seg_bits;
            }
        }

        GCContinuation { ptr: new_cont }
    }

    fn store_cont(&mut self, cont: GCContinuation) -> u64 {
        // The handle IS the tagged pointer. No side storage needed.
        // The GC traces it transitively from the segment slot where it's stored.
        tag_ptr(cont.ptr)
    }

    fn load_cont(&self, handle: u64) -> GCContinuation {
        // The handle IS a tagged pointer to the GC continuation object.
        GCContinuation { ptr: untag_ptr(handle) }
    }

    fn safepoint(&mut self, frames: &mut [InterpFrame<Self>]) {
        self.do_safepoint(frames);
    }
}

/// Convenience: create a generic interpreter with GC segment backend.
pub fn gc_interpreter<'a>(module: &'a dynir::ir::Module, heap_size: usize) -> GenericInterpreter<'a, GCSegmentBackend> {
    GenericInterpreter::new(module, GCSegmentBackend::new(heap_size))
}
