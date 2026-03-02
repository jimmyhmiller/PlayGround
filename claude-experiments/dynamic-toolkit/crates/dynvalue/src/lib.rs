mod scheme;
mod value;
mod low_bit;
mod nan_box;
mod layout;

pub use scheme::{Decoded, HasUnboxedFloat, TagScheme};
pub use value::Value;
pub use low_bit::LowBit;
pub use nan_box::NanBox;
pub use layout::{Payload, TaggedValue};

#[cfg(test)]
mod tests;
