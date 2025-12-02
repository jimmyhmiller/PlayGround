use std::{error::Error, sync::Mutex};

use crate::types::BuiltInTypes;

use super::{AllocateAction, Allocator, AllocatorOptions, StackMap};

pub struct MutexAllocator<Alloc: Allocator> {
    alloc: Alloc,
    mutex: Mutex<()>,
    options: AllocatorOptions,
    registered_threads: usize,
}

impl<Alloc: Allocator> Allocator for MutexAllocator<Alloc> {
    fn new(options: AllocatorOptions) -> Self {
        MutexAllocator {
            alloc: Alloc::new(options),
            mutex: Mutex::new(()),
            options,
            registered_threads: 0,
        }
    }
    fn try_allocate(
        &mut self,
        bytes: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        if self.registered_threads == 0 {
            return self.alloc.try_allocate(bytes, kind);
        }

        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.try_allocate(bytes, kind);
        drop(lock);
        result
    }

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        if self.registered_threads == 0 {
            return self.alloc.gc(stack_map, stack_pointers);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc(stack_map, stack_pointers);
        drop(lock)
    }

    fn grow(&mut self) {
        if self.registered_threads == 0 {
            return self.alloc.grow();
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.grow();
        drop(lock)
    }

    fn gc_add_root(&mut self, old: usize) {
        if self.registered_threads == 0 {
            return self.alloc.gc_add_root(old);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc_add_root(old);
        drop(lock)
    }

    fn register_temporary_root(&mut self, root: usize) -> usize {
        if self.registered_threads == 0 {
            return self.alloc.register_temporary_root(root);
        }
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.register_temporary_root(root);
        drop(lock);
        result
    }

    fn unregister_temporary_root(&mut self, id: usize) -> usize {
        if self.registered_threads == 0 {
            return self.alloc.unregister_temporary_root(id);
        }
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.unregister_temporary_root(id);
        drop(lock);
        result
    }

    fn add_namespace_root(&mut self, namespace_id: usize, root: usize) {
        if self.registered_threads == 0 {
            return self.alloc.add_namespace_root(namespace_id, root);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.add_namespace_root(namespace_id, root);
        drop(lock)
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        if self.registered_threads == 0 {
            return self.alloc.get_namespace_relocations();
        }
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.get_namespace_relocations();
        drop(lock);
        result
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }

    fn register_thread(&mut self, _thread_id: std::thread::ThreadId) {
        self.registered_threads += 1;
    }

    fn remove_thread(&mut self, _thread_id: std::thread::ThreadId) {
        self.registered_threads -= 1;
    }
}
