// Beagle modules - reused from the Beagle project
pub mod ir;
pub mod types;
pub mod arm;
pub mod common;
pub mod code_memory;
pub mod machine_code;
pub mod register_allocation;

// We need to provide a stub for ast since IR references it
pub mod ast {
    #[derive(Debug, Clone, Copy)]
    pub struct IRRange {
        pub start: usize,
        pub end: usize,
    }
}
