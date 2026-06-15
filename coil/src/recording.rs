//! A `Backend` that builds nothing real — it hands out sequential fake handles
//! and records a human-readable log of every call. This lets us test the
//! lisp→MLIR mapping (`emit`) with no MLIR/LLVM present: a test feeds core forms
//! and asserts on the recorded sequence of builder calls.

use crate::backend::{Backend, BackendError, Handle, NamedAttr, ResultTypes};
use std::collections::HashMap;

#[derive(Default)]
pub struct RecordingBackend {
    next: u64,
    /// One line per builder call, in order.
    pub log: Vec<String>,
    /// Block → its argument value handles (so `block_arg` is stable).
    block_args: HashMap<u64, Vec<Handle>>,
}

impl RecordingBackend {
    pub fn new() -> Self {
        Self::default()
    }

    fn fresh(&mut self) -> Handle {
        self.next += 1;
        Handle(self.next)
    }

    /// The full log joined by newlines (handy in test assertions).
    pub fn log_text(&self) -> String {
        self.log.join("\n")
    }
}

fn ids(hs: &[Handle]) -> String {
    let parts: Vec<String> = hs.iter().map(|h| format!("v{}", h.0)).collect();
    format!("[{}]", parts.join(" "))
}

impl Backend for RecordingBackend {
    fn parse_type(&mut self, text: &str) -> Result<Handle, BackendError> {
        let h = self.fresh();
        self.log.push(format!("type v{} = !{}", h.0, text));
        Ok(h)
    }

    fn integer_type(&mut self, width: u32, signed: bool) -> Result<Handle, BackendError> {
        let h = self.fresh();
        let s = if signed { "i" } else { "u" };
        self.log.push(format!("type v{} = {}{}", h.0, s, width));
        Ok(h)
    }

    fn create_module(&mut self) -> Result<Handle, BackendError> {
        let h = self.fresh();
        self.log.push(format!("module v{}", h.0));
        Ok(h)
    }

    fn module_body(&mut self, module: Handle) -> Result<Handle, BackendError> {
        let h = self.fresh();
        self.log.push(format!("module-body v{} of v{}", h.0, module.0));
        self.block_args.insert(h.0, vec![]);
        Ok(h)
    }

    fn create_region(&mut self) -> Result<Handle, BackendError> {
        let h = self.fresh();
        self.log.push(format!("region v{}", h.0));
        Ok(h)
    }

    fn create_block(&mut self, region: Handle, arg_types: &[Handle]) -> Result<Handle, BackendError> {
        let block = self.fresh();
        // pre-mint the block-argument value handles
        let mut args = Vec::with_capacity(arg_types.len());
        for _ in arg_types {
            args.push(self.fresh());
        }
        self.log.push(format!(
            "block v{} in r{} argtypes={} args={}",
            block.0,
            region.0,
            ids(arg_types),
            ids(&args)
        ));
        self.block_args.insert(block.0, args);
        Ok(block)
    }

    fn block_arg(&mut self, block: Handle, i: usize) -> Result<Handle, BackendError> {
        self.block_args
            .get(&block.0)
            .and_then(|a| a.get(i).copied())
            .ok_or_else(|| BackendError(format!("no arg {i} on block v{}", block.0)))
    }

    fn set_insertion_end(&mut self, block: Handle) -> Result<(), BackendError> {
        self.log.push(format!("insert-into v{}", block.0));
        Ok(())
    }

    fn build_op(
        &mut self,
        name: &str,
        operands: &[Handle],
        results: ResultTypes,
        attrs: &[NamedAttr],
        regions: &[Handle],
        successors: &[Handle],
    ) -> Result<Vec<Handle>, BackendError> {
        let result_handles = match &results {
            // Infer: a recording backend can't really infer, so assume one result.
            ResultTypes::Infer => vec![self.fresh()],
            ResultTypes::Explicit(ts) => ts.iter().map(|_| self.fresh()).collect(),
        };
        let res_desc = match &results {
            ResultTypes::Infer => "infer".to_string(),
            ResultTypes::Explicit(ts) => format!("types={}", ids(ts)),
        };
        let attr_desc: Vec<String> = attrs
            .iter()
            .map(|a| format!("{}={}", a.name, a.value))
            .collect();
        self.log.push(format!(
            "op {} operands={} results({}) attrs={{{}}} regions={} succs={} -> {}",
            name,
            ids(operands),
            res_desc,
            attr_desc.join(" "),
            ids(regions),
            ids(successors),
            ids(&result_handles),
        ));
        Ok(result_handles)
    }
}
