use std::sync::Mutex;

use crate::compiler::Allocator;


pub struct MutexAllocator<Alloc: Allocator> {
    alloc: Alloc,
    mutex: Mutex<()>,
}


impl<Alloc: Allocator> Allocator for MutexAllocator<Alloc> {

    fn new() -> Self {
        MutexAllocator {
            alloc: Alloc::new(),
            mutex: Mutex::new(()),
        }
    }
    fn allocate(
        &mut self,
        bytes: usize,
        kind: crate::ir::BuiltInTypes,
        options: crate::compiler::AllocatorOptions,
    ) -> Result<crate::compiler::AllocateAction, Box<dyn std::error::Error>> {
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.allocate(bytes, kind, options);
        drop(lock);
        result
    }

    fn gc(
        &mut self,
        stack_map: &crate::compiler::StackMap,
        stack_pointers: &Vec<(usize, usize)>,
        options: crate::compiler::AllocatorOptions,
    ) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc(stack_map, stack_pointers, options);
        drop(lock)
    }

    fn grow(&mut self, options: crate::compiler::AllocatorOptions) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.grow(options);
        drop(lock)
    }

    fn gc_add_root(&mut self, old: usize, young: usize) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc_add_root(old, young);
        drop(lock)
    }
}