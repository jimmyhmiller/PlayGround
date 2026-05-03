mod alloc;
mod barrier;
pub mod card_table;
mod mutator;
mod semi_space;

mod heap;
pub mod statemap;
mod thread;

pub use alloc::{Alloc, AtomicBumpAllocator, BumpAllocator, HeapWalker, alloc_obj};
pub use barrier::{SATBBuffer, SATBQueue, read_barrier, read_barrier_atomic};
pub use card_table::CardTable;
pub use heap::Heap;
pub use mutator::{GcRef, Mutator, Root, RootScope};
pub use semi_space::{FORWARDING_BIT, PtrPolicy, SemiSpace, follow_forwarding};
pub use statemap::StatemapTracer;
pub use thread::{MutatorThread, ThreadState};

mod gc_regression;

#[cfg(test)]
mod tests;
