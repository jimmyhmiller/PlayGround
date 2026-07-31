pub mod build_profile;
pub mod bundle_benchmark;
pub mod bundler;
pub mod config;
pub mod css;
pub mod dead_branch;
pub mod dev_server;
pub mod dynamic_import_context;
pub mod env_file;
pub mod font_file;
pub mod html_entry;
mod frontend_profile;
pub mod hmr;
pub mod import_meta_env;
pub mod import_meta_glob;
pub mod js_reachability;
pub mod jsx_project_config;
pub mod less_stylus;
pub mod manifest;
pub mod mdx;
#[cfg(feature = "memory-accounting")]
pub mod memory;
pub mod next_adapter;
pub mod next_pages;
pub mod parser;
pub mod next_font;
pub mod styled_jsx;
pub mod postcss;
pub mod project_graph;
pub mod resource_id;
pub mod rsc;
pub mod rsc_runtime_resolve;
pub mod route_split;
pub mod runtime_helpers;
pub mod sass;
pub mod sfc;
pub mod route_tree;
pub mod server_fn;
pub mod side_effects;
pub mod source_map;
pub mod tailwind;
pub mod tailwind_delegate;
pub mod transform;
pub mod visualizer;
pub mod vite_config;
pub mod vite_define;
pub mod vite_manifest;

/// Track every allocation so the guard suite can assert on peak/retained memory
/// deterministically. Relaxed atomics keep the overhead negligible and uniform,
/// so speed measurements stay representative.
// The accounting allocator exists only in `memory-accounting` builds (the
// memory benchmark and its guards). A default build overrides nothing: the
// system allocator, no wrapper, no measurement layer.
#[cfg(feature = "memory-accounting")]
#[global_allocator]
static GLOBAL_ALLOCATOR: memory::TrackingAllocator = memory::TrackingAllocator;
