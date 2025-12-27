// Compacting GC - Cheney's two-space copying collector
//
// This GC uses two semi-spaces and copies live objects between them,
// which eliminates fragmentation and improves cache locality.

use std::error::Error;

use super::space::{DEFAULT_PAGE_COUNT, Space};
use super::stack_walker::StackWalker;
use super::types::{BuiltInTypes, Header, HeapObject};
use super::{
    AllocateAction, Allocator, AllocatorOptions, DetailedHeapStats, HeapInspector, StackMap,
    type_id_to_name,
};

/// Compacting heap using Cheney's algorithm
pub struct CompactingHeap {
    to_space: Space,
    from_space: Space,
    namespace_roots: Vec<(usize, usize)>,
    temporary_roots: Vec<Option<usize>>,
    namespace_relocations: Vec<(usize, Vec<(usize, usize)>)>,
    options: AllocatorOptions,
}

impl CompactingHeap {
    fn copy_using_cheneys_algorithm(&mut self, heap_object: HeapObject) -> usize {
        let untagged = heap_object.get_pointer() as usize;

        debug_assert!(
            self.to_space.contains(untagged as *const u8)
                || self.from_space.contains(untagged as *const u8),
            "Pointer is not in to space"
        );

        // If already in to_space, it's been copied
        if self.to_space.contains(untagged as *const u8) {
            debug_assert!(untagged % 8 == 0, "Pointer is not aligned");
            return heap_object.tagged_pointer();
        }

        // If marked, object has been forwarded - get forwarding pointer from header
        if heap_object.marked() {
            // The header contains the forwarding pointer with marked bit preserved
            let untagged = heap_object.untagged();
            let pointer = untagged as *mut usize;
            let header_data = unsafe { *pointer };
            // Clear the marked bit to get the clean forwarding pointer
            return Header::clear_marked_bit(header_data);
        }

        // Copy the object to to_space
        let data = heap_object.get_full_object_data();
        let new_pointer = self.to_space.copy_data_to_offset(data);
        debug_assert!(new_pointer % 8 == 0, "Pointer is not aligned");

        // Store forwarding pointer in header for all objects
        let tagged_new = heap_object.get_object_type().unwrap().tag(new_pointer) as usize;
        let untagged = heap_object.untagged();
        let pointer = untagged as *mut usize;
        // Set the forwarding pointer with marked bit preserved
        unsafe { *pointer = Header::set_marked_bit(tagged_new) };

        tagged_new
    }

    unsafe fn copy_all(&mut self, roots: Vec<usize>) -> Vec<usize> {
        let start_offset = self.to_space.allocation_offset;
        let mut new_roots = vec![];
        for root in roots.iter() {
            let heap_object = HeapObject::from_tagged(*root);
            new_roots.push(self.copy_using_cheneys_algorithm(heap_object));
        }

        unsafe { self.copy_remaining(start_offset) };

        new_roots
    }

    unsafe fn copy_remaining(&mut self, start_offset: usize) {
        // Collect objects first to avoid borrow conflict
        let mut offset = start_offset;
        loop {
            if offset >= self.to_space.allocation_offset {
                break;
            }
            if self.to_space.allocation_offset == 0 {
                break;
            }

            let pointer = unsafe { self.to_space.start.add(offset) };
            let mut object = HeapObject::from_untagged(pointer);
            let size = object.full_size();

            if object.marked() {
                panic!("We are copying to this space, nothing should be marked");
            }
            if !object.is_zero_size() {
                for datum in object.get_fields_mut() {
                    if BuiltInTypes::is_heap_pointer(*datum) {
                        let heap_object = HeapObject::from_tagged(*datum);
                        *datum = self.copy_using_cheneys_algorithm(heap_object);
                    }
                }
            }

            offset += size;
            if offset % 8 != 0 {
                panic!("Heap offset is not aligned");
            }
        }
    }

    pub fn gather_roots(
        &mut self,
        stack_base: usize,
        stack_map: &StackMap,
        stack_pointer: usize,
    ) -> Vec<(usize, usize)> {
        StackWalker::collect_stack_roots(stack_base, stack_pointer, stack_map)
    }
}

impl Allocator for CompactingHeap {
    fn new(options: AllocatorOptions) -> Self {
        let to_space = Space::new(DEFAULT_PAGE_COUNT / 2);
        let from_space = Space::new(DEFAULT_PAGE_COUNT / 2);

        Self {
            to_space,
            from_space,
            namespace_roots: vec![],
            temporary_roots: vec![],
            namespace_relocations: vec![],
            options,
        }
    }

    fn try_allocate(
        &mut self,
        words: usize,
        _kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        if words > self.from_space.word_count() {
            self.grow();
        }

        if self.from_space.allocation_offset + words * 8 >= self.from_space.byte_count() {
            return Ok(AllocateAction::Gc);
        }

        let pointer = self.from_space.allocate(words);

        Ok(AllocateAction::Allocated(pointer))
    }

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        if !self.options.gc {
            return;
        }

        #[cfg(debug_assertions)]
        {
            self.to_space.unprotect();
        }

        let start_offset = self.to_space.allocation_offset;
        let mut temporary_roots_to_update: Vec<(usize, usize)> = vec![];
        for (i, root) in self.temporary_roots.clone().iter().enumerate() {
            if let Some(root) = root {
                if BuiltInTypes::is_heap_pointer(*root) {
                    let heap_object = HeapObject::from_tagged(*root);
                    debug_assert!(self.from_space.contains(heap_object.get_pointer()));
                    let new_root = self.copy_using_cheneys_algorithm(heap_object);
                    temporary_roots_to_update.push((i, new_root));
                }
            }
        }

        unsafe { self.copy_remaining(start_offset) };

        for (i, new_root) in temporary_roots_to_update.iter() {
            self.temporary_roots[*i] = Some(*new_root);
        }

        for (stack_base, stack_pointer) in stack_pointers.iter() {
            let roots = self.gather_roots(*stack_base, stack_map, *stack_pointer);
            let new_roots = unsafe { self.copy_all(roots.iter().map(|x| x.1).collect()) };

            let stack_buffer = StackWalker::get_live_stack_mut(*stack_base, *stack_pointer);
            for (i, (stack_offset, _)) in roots.iter().enumerate() {
                debug_assert!(
                    BuiltInTypes::untag(new_roots[i]) % 8 == 0,
                    "Pointer is not aligned"
                );
                stack_buffer[*stack_offset] = new_roots[i];
            }
        }

        let start_offset = self.to_space.allocation_offset;
        let namespace_roots = std::mem::take(&mut self.namespace_roots);
        // There has to be a better answer than this. But it does seem to work.
        for (namespace_id, root) in namespace_roots.into_iter() {
            if BuiltInTypes::is_heap_pointer(root) {
                let heap_object = HeapObject::from_tagged(root);
                let new_pointer = self.copy_using_cheneys_algorithm(heap_object);
                self.namespace_relocations
                    .push((namespace_id, vec![(root, new_pointer)]));
                self.namespace_roots.push((namespace_id, new_pointer));
            }
        }
        unsafe { self.copy_remaining(start_offset) };

        std::mem::swap(&mut self.from_space, &mut self.to_space);

        self.to_space.clear();
        // Only do this when debug mode
        #[cfg(debug_assertions)]
        {
            self.to_space.protect();
        }
    }

    fn grow(&mut self) {
        // From space is never protected
        self.from_space.double_committed_memory();

        #[cfg(debug_assertions)]
        {
            let currently_protect = self.to_space.protected;
            if currently_protect {
                self.to_space.unprotect();
            }
            self.to_space.double_committed_memory();
            if currently_protect {
                self.to_space.protect();
            }
        }

        #[cfg(not(debug_assertions))]
        {
            self.to_space.double_committed_memory();
        }
    }

    fn gc_add_root(&mut self, _old: usize) {
        // Don't need this because this is a write barrier for generational
    }

    fn register_temporary_root(&mut self, root: usize) -> usize {
        debug_assert!(self.temporary_roots.len() < 10, "Too many temporary roots");
        for (i, temp_root) in self.temporary_roots.iter_mut().enumerate() {
            if temp_root.is_none() {
                *temp_root = Some(root);
                return i;
            }
        }
        self.temporary_roots.push(Some(root));
        self.temporary_roots.len() - 1
    }

    fn unregister_temporary_root(&mut self, id: usize) -> usize {
        let value = self.temporary_roots[id];
        self.temporary_roots[id] = None;
        value.unwrap()
    }

    fn add_namespace_root(&mut self, namespace_id: usize, root: usize) {
        self.namespace_roots.push((namespace_id, root));
    }

    fn remove_namespace_root(&mut self, namespace_id: usize, root: usize) -> bool {
        if let Some(pos) = self
            .namespace_roots
            .iter()
            .position(|(ns, r)| *ns == namespace_id && *r == root)
        {
            self.namespace_roots.swap_remove(pos);
            true
        } else {
            false
        }
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        self.namespace_relocations.drain(0..).collect()
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }
}

// ========== Heap Inspection ==========

use std::collections::HashMap;

impl HeapInspector for CompactingHeap {
    fn iter_objects(&self) -> Box<dyn Iterator<Item = HeapObject> + '_> {
        // Objects are in from_space (contiguous allocation, no gaps)
        Box::new(self.from_space.object_iter_from_position(0))
    }

    fn detailed_stats(&self) -> DetailedHeapStats {
        let mut object_count = 0;
        let mut used_bytes = 0;
        let mut type_counts: HashMap<u8, (usize, usize)> = HashMap::new();

        for obj in self.iter_objects() {
            object_count += 1;
            let size = obj.full_size();
            used_bytes += size;

            let type_id = obj.get_type_id() as u8;
            let entry = type_counts.entry(type_id).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += size;
        }

        // Convert type_counts to vector with names
        let mut objects_by_type: Vec<(u8, String, usize, usize)> = type_counts
            .into_iter()
            .map(|(id, (count, bytes))| (id, type_id_to_name(id).to_string(), count, bytes))
            .collect();
        objects_by_type.sort_by_key(|(id, _, _, _)| *id);

        DetailedHeapStats {
            gc_algorithm: "compacting",
            total_bytes: self.from_space.byte_count() + self.to_space.byte_count(),
            used_bytes,
            object_count,
            objects_by_type,
            free_list_entries: None,
            free_bytes: None,
            largest_free_block: None,
        }
    }

    fn contains_address(&self, addr: usize) -> bool {
        self.from_space.contains(addr as *const u8) || self.to_space.contains(addr as *const u8)
    }

    fn get_roots(&self) -> &[(usize, usize)] {
        &self.namespace_roots
    }
}
