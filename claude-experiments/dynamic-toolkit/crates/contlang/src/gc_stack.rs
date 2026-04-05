//! GC-segment stack strategy for the unified interpreter.
//!
//! Implements `UnifiedStackStrategy` + `InterpFrameStore` with GC-allocated
//! frame segments. Frame values are tagged: tag 0 = heap pointer, tag 1 = fixnum.
//! Continuations are stored as cloned frame snapshots with GC segment pointers.

use std::cell::Cell;

use dynalloc::{PtrPolicy, SemiSpace};
use dynexec::{
    CapturedFrame, FrameResume, FrameResumePoint,
    FrameSliceMode, FrameSliceSnapshot, InterpFrameStore, StackConfig, UnifiedStackStrategy,
};
use dynobj::{Compact, ObjHeader, RootSource, TypeInfo};
use dynvalue::LowBit;

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

// ─── Segment access ───────────────────────────────────────────────

fn raw_get(seg: *mut u8, idx: usize) -> u64 {
    unsafe { *(seg.add(SEGMENT_INFO.varlen_element_offset(idx)) as *const u64) }
}
fn raw_set(seg: *mut u8, idx: usize, val: u64) {
    unsafe { *(seg.add(SEGMENT_INFO.varlen_element_offset(idx)) as *mut u64) = val; }
}

fn seg_get(seg: *mut u8, idx: usize) -> u64 {
    let bits = raw_get(seg, idx);
    if is_tagged_ptr(bits) { bits } else { untag_fixnum(bits) as u64 }
}

fn seg_set(seg: *mut u8, idx: usize, val: u64) {
    if is_tagged_ptr(val) {
        raw_set(seg, idx, val);
    } else {
        raw_set(seg, idx, tag_fixnum(val as i64));
    }
}

// ─── Root tracking ─────────────────────────────────────────────────

struct SegRoots {
    slots: Vec<Cell<u64>>,
}

impl SegRoots {
    fn new() -> Self { SegRoots { slots: Vec::new() } }
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

// ─── Frame & Continuation types ───────────────────────────────────

struct GCFrame {
    seg_ptr: *mut u8,
    val_count: usize,
    func_idx: usize,
    block_idx: usize,
    inst_idx: usize,
    resume: FrameResume,
    active_prompts: Vec<u32>,
}

// ─── GCSegmentRuntime ─────────────────────────────────────────────

pub struct GCSegmentRuntime {
    stack: Vec<GCFrame>,
    gc: SemiSpace,
    alloc_count: usize,
    gc_count: usize,
    gc_stress: Option<usize>,
}

impl GCSegmentRuntime {
    pub fn new(heap_size: usize) -> Self {
        GCSegmentRuntime {
            stack: Vec::new(),
            gc: SemiSpace::new::<Compact>(heap_size),
            alloc_count: 0,
            gc_count: 0,
            gc_stress: None,
        }
    }

    pub fn gc_count(&self) -> usize { self.gc_count }
    pub fn from_used(&self) -> usize { self.gc.from_used() }
    pub fn set_gc_stress(&mut self, n: usize) { self.gc_stress = Some(n); }

    pub fn collect_empty(&mut self) {
        let roots = SegRoots::new();
        unsafe { self.gc.collect::<SegPtrPolicy>(&mut [&roots]); }
        self.gc_count += 1;
    }

    fn alloc_segment(&mut self, val_count: usize) -> *mut u8 {
        let ptr = self.gc.alloc_obj::<Compact>(&SEGMENT_INFO, val_count);
        assert!(!ptr.is_null(), "segment allocation failed");
        self.alloc_count += 1;
        ptr
    }

    fn do_gc(&mut self) {
        let mut roots = SegRoots::new();
        for frame in &self.stack {
            roots.push_tagged(frame.seg_ptr);
        }

        unsafe { self.gc.collect::<SegPtrPolicy>(&mut [&roots]); }
        self.gc_count += 1;

        // Update stack frame pointers
        for (idx, frame) in self.stack.iter_mut().enumerate() {
            frame.seg_ptr = roots.get_ptr(idx);
        }
    }

}

impl InterpFrameStore for GCSegmentRuntime {
    fn push_frame(&mut self, func_idx: usize, val_count: usize, _slot_count: usize, args: &[(usize, u64)], resume: FrameResume) {
        let seg_ptr = self.alloc_segment(val_count);
        for &(idx, val) in args {
            seg_set(seg_ptr, idx, val);
        }
        self.stack.push(GCFrame {
            seg_ptr, val_count, func_idx,
            block_idx: 0, inst_idx: 0,
            resume, active_prompts: Vec::new(),
        });
    }

    fn pop_frame(&mut self) -> FrameResume {
        self.stack.pop().expect("pop_frame on empty stack").resume
    }

    fn is_empty(&self) -> bool { self.stack.is_empty() }

    fn get(&self, idx: usize) -> u64 {
        seg_get(self.stack.last().unwrap().seg_ptr, idx)
    }

    fn set(&mut self, idx: usize, val: u64) {
        seg_set(self.stack.last_mut().unwrap().seg_ptr, idx, val);
    }

    fn func_idx(&self) -> usize { self.stack.last().unwrap().func_idx }
    fn block_idx(&self) -> usize { self.stack.last().unwrap().block_idx }
    fn set_block(&mut self, block: usize) { self.stack.last_mut().unwrap().block_idx = block; }
    fn inst_idx(&self) -> usize { self.stack.last().unwrap().inst_idx }
    fn set_inst(&mut self, inst: usize) { self.stack.last_mut().unwrap().inst_idx = inst; }
    fn advance_inst(&mut self) { self.stack.last_mut().unwrap().inst_idx += 1; }

    fn push_prompt(&mut self, prompt: u32) {
        self.stack.last_mut().unwrap().active_prompts.push(prompt);
    }
    fn pop_prompt(&mut self, prompt: u32) {
        let popped = self.stack.last_mut().unwrap().active_prompts.pop();
        assert_eq!(popped, Some(prompt));
    }
    fn find_prompt_depth(&self, prompt: u32) -> Option<usize> {
        self.stack.iter().rposition(|f| f.active_prompts.contains(&prompt))
    }
    fn pop_frames_above(&mut self, depth: usize) {
        while self.stack.len() > depth + 1 {
            self.stack.pop();
        }
    }

    fn capture_snapshot(&mut self, prompt: u32, resume_dest: usize) -> FrameSliceSnapshot {
        let start = self.stack.iter().rposition(|f| f.active_prompts.contains(&prompt))
            .expect("capture: prompt not found");
        let frame_count = self.stack.len() - start;
        let mut frames = Vec::with_capacity(frame_count);
        for (i, f) in self.stack[start..].iter().enumerate() {
            let is_top = i + 1 == frame_count;
            // Read values out of GC segment into a plain Vec
            let values: Vec<u64> = (0..f.val_count).map(|idx| seg_get(f.seg_ptr, idx)).collect();
            frames.push(CapturedFrame {
                resume: FrameResumePoint {
                    func_idx: f.func_idx,
                    block_idx: f.block_idx,
                    inst_idx: f.inst_idx,
                },
                values,
                root_value_indices: Vec::new(),
                resume_arg_value_indices: if is_top { vec![resume_dest] } else { Vec::new() },
                active_prompts: f.active_prompts.clone(),
                caller_resume: f.resume.clone(),
            });
        }
        FrameSliceSnapshot {
            prompt_id: prompt,
            mode: FrameSliceMode::OneShot,
            frames,
            consumed: false,
        }
    }

    fn resume_snapshot(&mut self, snapshot: &FrameSliceSnapshot, args: &[u64]) {
        self.stack.clear();
        let frame_count = snapshot.frames.len();
        for (i, captured) in snapshot.frames.iter().enumerate() {
            let val_count = captured.values.len();
            let seg_ptr = self.alloc_segment(val_count);
            // Write values into new GC segment
            for (idx, &val) in captured.values.iter().enumerate() {
                seg_set(seg_ptr, idx, val);
            }
            if i + 1 == frame_count {
                for (idx, &value_idx) in captured.resume_arg_value_indices.iter().enumerate() {
                    if let Some(&arg) = args.get(idx) {
                        seg_set(seg_ptr, value_idx, arg);
                    }
                }
            }
            self.stack.push(GCFrame {
                seg_ptr,
                val_count,
                func_idx: captured.resume.func_idx,
                block_idx: captured.resume.block_idx,
                inst_idx: captured.resume.inst_idx,
                resume: captured.caller_resume.clone(),
                active_prompts: captured.active_prompts.clone(),
            });
        }
    }

    fn needs_gc(&self) -> bool {
        if let Some(n) = self.gc_stress {
            self.alloc_count > 0 && self.alloc_count % n == 0
        } else {
            false
        }
    }

    fn collect_gc(&mut self) {
        self.do_gc();
    }
}

// ─── Strategy ─────────────────────────────────────────────────────

pub struct GCSegmentStack;

impl UnifiedStackStrategy for GCSegmentStack {
    const NAME: &'static str = "gc-segment";
    fn supports_continuations() -> bool { true }
    type Runtime = GCSegmentRuntime;
    fn create_runtime(config: StackConfig) -> GCSegmentRuntime {
        GCSegmentRuntime::new(config.heap_size)
    }
}
