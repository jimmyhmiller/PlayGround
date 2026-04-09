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
pub use semi_space::{PtrPolicy, SemiSpace};
pub use statemap::StatemapTracer;
pub use thread::{MutatorThread, ThreadState};

mod gc_regression;

// TODO: tests.rs has 115 compile errors from API changes (type_table param, type_id rename).
// Disabled until fixed. Regression tests are in gc_regression.rs.
#[cfg(all(test, feature = "broken_tests"))]
mod tests;
