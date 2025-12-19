pub mod api;
pub mod classes;
pub mod core;
pub mod compilers;
pub mod graph;
pub mod graph_layout;
pub mod html_layout_provider;
pub mod html_templates;
pub mod iongraph;
pub mod javascript_generator;
pub mod layout_provider;
pub mod pure_svg_text_layout_provider;
pub mod utils;
pub mod wasm_html_generator;

// JSON modules
#[cfg(not(feature = "serde"))]
pub mod json;
pub mod json_compat;

// WASM module (conditionally compiled for wasm32 target)
#[cfg(target_arch = "wasm32")]
pub mod wasm;

// WebGL renderer module (conditionally compiled with webgl feature)
#[cfg(feature = "webgl")]
pub mod webgl;

// WebGL WASM bindings (requires both webgl feature and wasm32 target)
#[cfg(all(feature = "webgl", target_arch = "wasm32"))]
pub mod wasm_webgl;

// Re-export core traits for easy access
pub use core::{CompilerIR, IRInstruction, IRBlock, SemanticAttribute};

// Re-export Ion implementation for backward compatibility
pub use compilers::IonIR;

// Re-export high-level API for easy access
pub use api::{render_svg, render_svg_from_json, render_ion_pass, GraphBuilder};
