use crate::{Memory, BuiltInTypes};

/// Manages memory inspection and analysis
pub struct MemoryInspector {
    pub stack: Vec<Memory>,
    pub heap: Vec<Memory>,
    pub heap_pointers: Vec<usize>,
}

impl MemoryInspector {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            heap: Vec::new(),
            heap_pointers: Vec::new(),
        }
    }

    pub fn update_stack(&mut self, stack_root: u64, stack_data: &[u64]) {
        self.stack = stack_data
            .iter()
            .enumerate()
            .map(|(i, value)| Memory::new(stack_root + (i as u64 * 8), *value))
            .collect();
    }

    pub fn update_heap(&mut self, heap_root: u64, heap_data: &[u64]) {
        self.heap = heap_data
            .iter()
            .enumerate()
            .map(|(i, value)| Memory::new(heap_root + (i as u64 * 8), *value))
            .collect();
    }

    pub fn add_heap_pointer(&mut self, pointer: usize) {
        if !self.heap_pointers.contains(&pointer) {
            self.heap_pointers.push(pointer);
        }
    }

    pub fn format_stack_with_pointers(&self, sp: u64, fp: u64) -> Vec<String> {
        self.stack
            .iter()
            .map(|memory| {
                if memory.address == sp && memory.address == fp {
                    format!("fsp> {}", memory.to_string())
                } else if memory.address == fp {
                    format!("fp>  {}", memory.to_string())
                } else if memory.address == sp {
                    format!("sp>  {}", memory.to_string())
                } else {
                    format!("     {}", memory.to_string())
                }
            })
            .collect()
    }

    pub fn get_heap_objects(&self) -> Vec<&Memory> {
        self.heap
            .iter()
            .filter(|mem| BuiltInTypes::is_heap_pointer(mem.value as usize))
            .collect()
    }
}

impl Default for MemoryInspector {
    fn default() -> Self {
        Self::new()
    }
}