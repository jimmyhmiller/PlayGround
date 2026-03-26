use tensor_lang_graph::Graph;

pub mod assemblyscript;
pub mod loop_ir;

/// A backend takes a compiled graph and emits code as a string.
pub trait Backend {
    fn emit(&self, graph: &Graph) -> String;
}
