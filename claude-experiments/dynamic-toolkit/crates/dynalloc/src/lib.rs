mod alloc;
mod barrier;
mod mutator;
mod semi_space;

mod heap;
mod thread;

pub use alloc::{Alloc, HeapWalker, alloc_obj, BumpAllocator, AtomicBumpAllocator};
pub use barrier::{SATBBuffer, SATBQueue, read_barrier, read_barrier_atomic};
pub use mutator::{Root, GcRef, RootScope, Mutator};
pub use semi_space::{PtrPolicy, SemiSpace};
pub use heap::Heap;
pub use thread::{ThreadState, MutatorThread};

#[cfg(test)]
mod tests;
