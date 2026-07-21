pub mod bundle_benchmark;
pub mod bundler;
pub mod config;
pub mod css;
pub mod dead_branch;
pub mod dev_server;
pub mod env_file;
pub mod html_entry;
mod frontend_profile;
pub mod hmr;
pub mod import_meta_env;
pub mod import_meta_glob;
pub mod js_reachability;
pub mod manifest;
pub mod memory;
pub mod parser;
pub mod resource_id;
pub mod route_split;
pub mod route_tree;
pub mod server_fn;
pub mod side_effects;
pub mod tailwind;
pub mod transform;
pub mod visualizer;
pub mod vite_config;
pub mod vite_define;

/// Track every allocation so the guard suite can assert on peak/retained memory
/// deterministically. Relaxed atomics keep the overhead negligible and uniform,
/// so speed measurements stay representative.
#[global_allocator]
static GLOBAL_ALLOCATOR: memory::TrackingAllocator = memory::TrackingAllocator;
