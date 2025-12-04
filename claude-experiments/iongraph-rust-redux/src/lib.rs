pub mod classes;
pub mod config;
pub mod core;
pub mod compilers;
pub mod graph;
pub mod graph_layout;
pub mod iongraph;
pub mod layout_provider;
pub mod pure_svg_text_layout_provider;
pub mod utils;

// Re-export core traits for easy access
pub use core::{CompilerIR, IRInstruction, IRBlock, SemanticAttribute};

// Re-export Ion implementation for backward compatibility
pub use compilers::IonIR;

// Re-export config types
pub use config::{Theme, ThemeConfig, LayoutConfig};
