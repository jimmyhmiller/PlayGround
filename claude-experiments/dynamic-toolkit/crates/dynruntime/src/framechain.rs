use dynalloc::Heap;
use dynir::{Function, InterpError, InterpResult, Interpreter};
use dynvalue::NanBox;
use dynobj::{Compact, DynRootFrame, FrameChain, ObjHeader, TypeInfo};

use crate::ptr_policy::NanBoxPtrPolicy;

// ─── FrameChainGcInterp ─────────────────────────────────────────────

/// GC-integrated interpreter using NaN-boxing + linked frame chain roots.
///
/// Uses NaN-boxing where tag 0 = heap pointer (48-bit payload).
/// GC roots are tracked via a linked list of frames (the Fil-C model):
/// each interpreted "call" pushes a [`DynRootFrame`] containing its live
/// GC pointers onto a per-thread [`FrameChain`]. The GC walks the chain
/// to find all roots.
///
/// Uses [`Heap`] for allocation and collection, and [`DynRootFrame`] +
/// [`FrameChain`] from dynobj for root management.
///
/// At each safepoint:
/// 1. Live GcPtr values are mirrored from the SSA value array into frame slots
/// 2. GC runs, walking the frame chain to find roots and updating slots in-place
/// 3. Updated values are read back from frame slots into the SSA value array
pub struct FrameChainGcInterp {
    heap: Heap,
}

impl FrameChainGcInterp {
    /// Create a new frame-chain interpreter with a heap-backed GC.
    pub fn new(space_size: usize) -> Self {
        FrameChainGcInterp {
            heap: Heap::new::<Compact>(space_size),
        }
    }

    /// Create with a custom header type.
    pub fn new_with_header<H: ObjHeader>(space_size: usize) -> Self {
        FrameChainGcInterp {
            heap: Heap::new::<H>(space_size),
        }
    }

    /// Allocate a heap object. Returns the raw pointer (to be NaN-box encoded
    /// as tag 0 by the caller).
    pub fn alloc(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        self.heap.alloc_obj::<Compact>(info, varlen_len)
    }

    /// Number of collections so far.
    pub fn collections(&self) -> usize {
        self.heap.collections()
    }

    /// Bytes used in from-space.
    pub fn from_used(&self) -> usize {
        self.heap.from_used()
    }

    /// Run a function with GC integration via frame chain roots.
    ///
    /// Builds a mapping from GcPtr SSA values to frame slots, pushes a
    /// [`DynRootFrame`] onto a [`FrameChain`], and at each safepoint mirrors
    /// values into the frame, triggers collection, then reads back.
    pub fn run(
        &self,
        interp: &Interpreter<'_, NanBox>,
        func: &Function,
        args: &[u64],
    ) -> Result<InterpResult, InterpError> {
        // Build mapping: SSA value index → frame slot index (only for GcPtr values).
        let mut val_to_slot: Vec<Option<usize>> = vec![None; func.value_types.len()];
        let mut slot_count = 0;
        for (i, ty) in func.value_types.iter().enumerate() {
            if ty.is_gc() {
                val_to_slot[i] = Some(slot_count);
                slot_count += 1;
            }
        }

        // Create the frame chain and push a dynamic frame.
        let chain = FrameChain::new();
        let frame = DynRootFrame::new(slot_count);
        let _guard = frame.push_onto(&chain);

        let frame_ref = &frame;
        let val_to_slot_ref = &val_to_slot;

        interp.run_with_safepoint(args, |vals, live_values| {
            // Clear all frame slots first (don't trace stale pointers).
            frame_ref.clear_all();

            // Mirror live GcPtr values into frame slots.
            for v in live_values {
                if let Some(slot_idx) = val_to_slot_ref[v.index()] {
                    frame_ref.set(slot_idx, vals[v.index()]);
                }
            }

            // Trigger GC — the frame chain is the root source.
            unsafe {
                self.heap.collect::<NanBoxPtrPolicy>(&[&chain]);
            }

            // Read back (possibly forwarded) values from frame slots.
            for v in live_values {
                if let Some(slot_idx) = val_to_slot_ref[v.index()] {
                    vals[v.index()] = frame_ref.get(slot_idx);
                }
            }
        })
    }

    /// Run with GC triggered only above a threshold.
    pub fn run_with_threshold(
        &self,
        interp: &Interpreter<'_, NanBox>,
        func: &Function,
        args: &[u64],
        threshold: f64,
    ) -> Result<InterpResult, InterpError> {
        let mut val_to_slot: Vec<Option<usize>> = vec![None; func.value_types.len()];
        let mut slot_count = 0;
        for (i, ty) in func.value_types.iter().enumerate() {
            if ty.is_gc() {
                val_to_slot[i] = Some(slot_count);
                slot_count += 1;
            }
        }

        let chain = FrameChain::new();
        let frame = DynRootFrame::new(slot_count);
        let _guard = frame.push_onto(&chain);

        let frame_ref = &frame;
        let val_to_slot_ref = &val_to_slot;

        interp.run_with_safepoint(args, |vals, live_values| {
            let usage = self.heap.from_used() as f64 / self.heap.space_size() as f64;

            if usage >= threshold {
                frame_ref.clear_all();
                for v in live_values {
                    if let Some(slot_idx) = val_to_slot_ref[v.index()] {
                        frame_ref.set(slot_idx, vals[v.index()]);
                    }
                }

                unsafe {
                    self.heap.collect::<NanBoxPtrPolicy>(&[&chain]);
                }

                for v in live_values {
                    if let Some(slot_idx) = val_to_slot_ref[v.index()] {
                        vals[v.index()] = frame_ref.get(slot_idx);
                    }
                }
            }
        })
    }
}
