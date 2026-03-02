mod header;
mod type_info;
mod field;
mod scan;
pub mod roots;

#[doc(hidden)]
pub use paste::paste as __paste;

pub use header::{ObjHeader, Compact, Full};
pub use type_info::{TypeInfo, VarLenKind};
pub use field::{
    init_header, read_type_info, read_value_field, write_value_field,
    read_raw_bytes, raw_data_mut,
    read_varlen_count, read_varlen_value, write_varlen_value,
    read_varlen_bytes, write_varlen_count,
};
pub use scan::scan_object;
pub use roots::{RootSource, ShadowStack, ShadowFrame, RootStack, PinnedRoots};

#[cfg(test)]
mod tests;
