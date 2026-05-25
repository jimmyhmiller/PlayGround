mod field;
mod header;
pub mod roots;
mod scan;
mod type_info;
mod typed_ptr;

#[doc(hidden)]
pub use paste::paste as __paste;

pub use field::{
    init_header, lookup_type_info, raw_data_mut, read_raw_bytes, read_type_id,
    read_value_field, read_varlen_bytes, read_varlen_count, read_varlen_value,
    write_value_field, write_varlen_count, write_varlen_value,
};
pub use header::{Compact, Full, ObjHeader};
pub use roots::{
    AtomicRootSet, DynRootFrame, FrameChain, FrameGuard, FrameHeader, RootFrame, RootSet,
    RootSource,
};
pub use scan::scan_object;
pub use type_info::{TypeInfo, VarLenKind};
pub use typed_ptr::TypedPtr;

#[cfg(test)]
mod tests;
