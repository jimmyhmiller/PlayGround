pub mod types;
pub mod ir;
pub mod target;
pub mod allocator;
pub mod cost;
pub mod verify;
pub mod testing;
pub mod liveness;
pub mod linear_scan;
pub mod aarch64;
pub mod flat;

#[cfg(test)]
mod tests;
