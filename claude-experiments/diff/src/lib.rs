pub mod bundle_benchmark;
pub mod bundler;
mod frontend_profile;
pub mod memory;
pub mod parser;
pub mod resource_id;
pub mod transform;
pub mod visualizer;

/// Track every allocation so the guard suite can assert on peak/retained memory
/// deterministically. Relaxed atomics keep the overhead negligible and uniform,
/// so speed measurements stay representative.
#[global_allocator]
static GLOBAL_ALLOCATOR: memory::TrackingAllocator = memory::TrackingAllocator;
