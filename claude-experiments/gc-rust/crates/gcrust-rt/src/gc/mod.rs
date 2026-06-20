mod alloc;
mod barrier;
pub mod card_table;
pub mod dump;
mod field;
mod header;
mod heap;
mod mutator;
mod ptr_policy;
pub mod reflect;
pub mod roots;
mod scan;
mod semi_space;
pub mod statemap;
mod thread;
mod type_info;

pub use alloc::{Alloc, AllocWindow, AtomicBumpAllocator, BumpAllocator, HeapWalker, alloc_obj};
pub use barrier::{SATBBuffer, SATBQueue, read_barrier, read_barrier_atomic};
pub use card_table::CardTable;
pub use dump::{dump_heap_json, dump_heap_text};
pub use field::{
    init_header, lookup_type_info, raw_data_mut, read_raw_bytes, read_type_id,
    read_varlen_bytes, read_varlen_count, write_varlen_count,
};
pub use header::{Compact, Full, ObjHeader};
pub use heap::{AllocSiteStat, Heap};
pub use mutator::{GcRef, Mutator, Root, RootScope};
pub use ptr_policy::IdentityPtrPolicy;
pub use reflect::{
    AllocSite, FieldMeta, FieldTy, ScalarKind, TypeKind, TypeMeta, ValueMeta, VariantMeta,
};
pub use roots::{
    AtomicRootSet, DynRootFrame, FrameChain, FrameGuard, FrameHeader, RootFrame, RootSet,
    RootSource,
};
pub use scan::scan_object;
pub use semi_space::{FORWARDING_BIT, PtrPolicy, SemiSpace, follow_forwarding};
pub use statemap::StatemapTracer;
pub use thread::{MutatorThread, SiteCounter, ThreadState};
pub use type_info::{TypeInfo, VarLenKind};

#[cfg(test)]
mod tests;
