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

// Re-export core traits for easy access
pub use core::{CompilerIR, IRInstruction, IRBlock, SemanticAttribute};

// Re-export Ion implementation for backward compatibility
pub use compilers::IonIR;
