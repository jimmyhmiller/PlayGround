use dynalloc::{Heap, Mutator, Root};
use dynir::{Function, InterpError, InterpResult, Interpreter};
use dynvalue::LowBit;
use dynobj::{Compact, ObjHeader, TypeInfo};

use crate::ptr_policy::LowBitPtrPolicy;

// ─── StackmapGcInterp ───────────────────────────────────────────────

/// GC-integrated interpreter using tagged pointers + stackmap-based root finding.
///
/// Uses `LowBit<N>` tagging where tag 0 = heap pointer. At each `Safepoint`
/// instruction, the live GcPtr values from the SSA value array are mirrored
/// into a [`Mutator`]'s root slots, the GC updates them in-place, and the
/// updated values are read back.
///
/// This models how a JIT compiler would work: the compiler emits stack maps
/// at each safepoint recording which registers/stack slots hold GC pointers.
/// The interpreter's `vals` array is the conceptual "register file".
///
/// Uses [`Heap`] for thread-safe allocation and collection, and [`Mutator`]
/// from dynalloc for root management.
pub struct StackmapGcInterp<const N: u32> {
    heap: Heap,
}

impl<const N: u32> StackmapGcInterp<N> {
    /// Create a new stackmap interpreter with a heap-backed GC.
    ///
    /// `space_size` is the size of each semi-space in bytes.
    pub fn new(space_size: usize) -> Self {
        StackmapGcInterp {
            heap: Heap::new::<Compact>(space_size),
        }
    }

    /// Create with a custom header type.
    pub fn new_with_header<H: ObjHeader>(space_size: usize) -> Self {
        StackmapGcInterp {
            heap: Heap::new::<H>(space_size),
        }
    }

    /// Allocate a heap object, initialize its header, and return the raw pointer.
    ///
    /// The pointer has tag 0 (aligned, low bits zero) so it can be used
    /// directly as a LowBit-tagged value.
    ///
    /// Returns null if space is exhausted.
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

    /// Run a function with GC integration.
    ///
    /// Pre-allocates [`Root`] handles in a [`Mutator`] for all GcPtr-typed SSA
    /// values. At each `Safepoint`, live values are mirrored into the Mutator,
    /// the Heap collects with the Mutator as a root source, and updated values
    /// are read back.
    pub fn run(
        &self,
        interp: &Interpreter<'_, LowBit<N>>,
        func: &Function,
        args: &[u64],
    ) -> Result<InterpResult, InterpError> {
        // Build mapping: SSA value index → Root handle (only for GcPtr values).
        let mut mutator = Mutator::new();
        let mut val_to_root: Vec<Option<Root>> = vec![None; func.value_types.len()];
        for (i, ty) in func.value_types.iter().enumerate() {
            if ty.is_gc() {
                val_to_root[i] = Some(mutator.root(0));
            }
        }

        let val_to_root_ref = &val_to_root;

        interp.run_with_safepoint(args, |vals, live_values| {
            // Clear all root slots (don't trace stale pointers).
            for root_opt in val_to_root_ref.iter().flatten() {
                mutator.set(root_opt, 0);
            }

            // Mirror live GcPtr values into Mutator root slots.
            for v in live_values {
                if let Some(root) = &val_to_root_ref[v.index()] {
                    mutator.set(root, vals[v.index()]);
                }
            }

            // Trigger GC — the Mutator is the root source.
            unsafe {
                self.heap.collect::<LowBitPtrPolicy<N>>(&[&mutator]);
            }

            // Read back (possibly forwarded) values from Mutator.
            for v in live_values {
                if let Some(root) = &val_to_root_ref[v.index()] {
                    vals[v.index()] = mutator.get(root).bits();
                }
            }
        })
    }

    /// Run a function, triggering GC only when from-space usage exceeds
    /// the given threshold (as a fraction, 0.0 to 1.0).
    pub fn run_with_threshold(
        &self,
        interp: &Interpreter<'_, LowBit<N>>,
        func: &Function,
        args: &[u64],
        threshold: f64,
    ) -> Result<InterpResult, InterpError> {
        let mut mutator = Mutator::new();
        let mut val_to_root: Vec<Option<Root>> = vec![None; func.value_types.len()];
        for (i, ty) in func.value_types.iter().enumerate() {
            if ty.is_gc() {
                val_to_root[i] = Some(mutator.root(0));
            }
        }

        let val_to_root_ref = &val_to_root;

        interp.run_with_safepoint(args, |vals, live_values| {
            let usage = self.heap.from_used() as f64 / self.heap.space_size() as f64;

            if usage >= threshold {
                for root_opt in val_to_root_ref.iter().flatten() {
                    mutator.set(root_opt, 0);
                }
                for v in live_values {
                    if let Some(root) = &val_to_root_ref[v.index()] {
                        mutator.set(root, vals[v.index()]);
                    }
                }

                unsafe {
                    self.heap.collect::<LowBitPtrPolicy<N>>(&[&mutator]);
                }

                for v in live_values {
                    if let Some(root) = &val_to_root_ref[v.index()] {
                        vals[v.index()] = mutator.get(root).bits();
                    }
                }
            }
        })
    }
}
