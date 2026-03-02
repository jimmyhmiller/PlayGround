mod alloc;
mod mutator;

pub use alloc::{Alloc, HeapWalker, alloc_obj, BumpAllocator};
pub use mutator::{Root, GcRef, RootScope, Mutator};

#[cfg(test)]
mod tests;
