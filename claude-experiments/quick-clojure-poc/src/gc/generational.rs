// Generational GC
//
// Two-generation garbage collector with:
// - Young generation: bump-pointer allocation with copying collection
// - Old generation: mark-and-sweep

use std::error::Error;

use super::mark_and_sweep::MarkAndSweep;
use super::space::{DEFAULT_PAGE_COUNT, Space};
use super::stack_walker::StackWalker;
use super::types::{BuiltInTypes, HeapObject, Word};
use super::{
    AllocateAction, Allocator, AllocatorOptions, DetailedHeapStats, HeapInspector, StackMap,
    type_id_to_name,
};

/// Generational garbage collector
pub struct GenerationalGC {
    young: Space,
    old: MarkAndSweep,
    copied: Vec<HeapObject>,
    gc_count: usize,
    full_gc_frequency: usize,
    // Additional roots for write barrier
    additional_roots: Vec<usize>,
    namespace_roots: Vec<(usize, usize)>,
    relocated_namespace_roots: Vec<(usize, Vec<(usize, usize)>)>,
    temporary_roots: Vec<Option<usize>>,
    atomic_pause: [u8; 8],
    options: AllocatorOptions,
}

impl Allocator for GenerationalGC {
    fn new(options: AllocatorOptions) -> Self {
        let young = Space::new(DEFAULT_PAGE_COUNT * 10);
        let old = MarkAndSweep::new_with_page_count(DEFAULT_PAGE_COUNT * 100, options);
        Self {
            young,
            old,
            copied: vec![],
            gc_count: 0,
            full_gc_frequency: 100,
            additional_roots: vec![],
            namespace_roots: vec![],
            relocated_namespace_roots: vec![],
            temporary_roots: vec![],
            atomic_pause: [0; 8],
            options,
        }
    }

    fn try_allocate(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let pointer = self.allocate_inner(words, kind)?;
        Ok(pointer)
    }

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        if !self.options.gc {
            return;
        }
        if self.gc_count % self.full_gc_frequency == 0 {
            self.gc_count = 0;
            self.full_gc(stack_map, stack_pointers);
        } else {
            self.minor_gc(stack_map, stack_pointers);
        }
        self.gc_count += 1;
    }

    fn grow(&mut self) {
        self.old.grow();
    }

    fn gc_add_root(&mut self, old: usize) {
        self.additional_roots.push(old);
    }

    #[allow(unused)]
    fn get_pause_pointer(&self) -> usize {
        self.atomic_pause.as_ptr() as usize
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
        std::mem::take(&mut self.relocated_namespace_roots)
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
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
}

impl GenerationalGC {
    fn allocate_inner(
        &mut self,
        words: usize,
        _kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let size = Word::from_word(words);
        if self.young.can_allocate(size) {
            Ok(AllocateAction::Allocated(self.young.allocate_word(size)))
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn minor_gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        let start = std::time::Instant::now();

        self.process_temporary_roots();
        self.process_additional_roots();
        self.process_namespace_roots();
        self.update_old_generation_namespace_roots();
        self.process_stack_roots(stack_map, stack_pointers);

        self.young.clear();

        if self.options.print_stats {
            println!("Minor GC took {:?}", start.elapsed());
        }
    }

    fn process_temporary_roots(&mut self) {
        let roots_to_copy: Vec<(usize, usize)> = self
            .temporary_roots
            .iter()
            .enumerate()
            .filter_map(|(i, root)| root.map(|r| (i, r)))
            .collect();

        for (index, root) in roots_to_copy {
            let new_root = unsafe { self.copy(root) };
            self.temporary_roots[index] = Some(new_root);
        }
        self.copy_remaining();
    }

    fn process_additional_roots(&mut self) {
        let additional_roots = std::mem::take(&mut self.additional_roots);
        for old in additional_roots.into_iter() {
            self.move_objects_referenced_from_old_to_old(&mut HeapObject::from_tagged(old));
        }
    }

    fn process_namespace_roots(&mut self) {
        let namespace_roots = std::mem::take(&mut self.namespace_roots);
        for (namespace_id, root) in namespace_roots.into_iter() {
            if !BuiltInTypes::is_heap_pointer(root) {
                continue;
            }
            let mut heap_object = HeapObject::from_tagged(root);
            if self.young.contains(heap_object.get_pointer()) && heap_object.marked() {
                // We have already copied this object, so the first field points to the new location
                let new_pointer = heap_object.get_field(0);
                self.namespace_roots.push((namespace_id, new_pointer));
                self.relocated_namespace_roots
                    .push((namespace_id, vec![(root, new_pointer)]));
            } else if self.young.contains(heap_object.get_pointer()) {
                let new_pointer = unsafe { self.copy(root) };
                self.relocated_namespace_roots
                    .push((namespace_id, vec![(root, new_pointer)]));
                self.namespace_roots.push((namespace_id, new_pointer));
                self.move_objects_referenced_from_old_to_old(&mut HeapObject::from_tagged(
                    new_pointer,
                ));
            } else {
                self.namespace_roots.push((namespace_id, root));
                self.move_objects_referenced_from_old_to_old(&mut heap_object);
            }
        }
    }

    fn update_old_generation_namespace_roots(&mut self) {
        self.old.clear_namespace_roots();
        for (namespace_id, root) in self.namespace_roots.iter() {
            self.old.add_namespace_root(*namespace_id, *root);
        }
    }

    fn process_stack_roots(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        for (stack_base, stack_pointer) in stack_pointers.iter() {
            let roots = self.gather_roots(*stack_base, stack_map, *stack_pointer);
            let new_roots: Vec<usize> = roots.iter().map(|x| x.1).collect();
            let new_roots = unsafe { self.copy_all(new_roots) };

            self.copy_remaining();

            let stack_buffer = StackWalker::get_live_stack_mut(*stack_base, *stack_pointer);
            for (i, (stack_offset, _)) in roots.iter().enumerate() {
                debug_assert!(
                    BuiltInTypes::untag(new_roots[i]) % 8 == 0,
                    "Pointer is not aligned"
                );
                stack_buffer[*stack_offset] = new_roots[i];
            }
        }
    }

    fn full_gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        self.minor_gc(stack_map, stack_pointers);
        self.old.gc(stack_map, stack_pointers);
    }

    unsafe fn copy_all(&mut self, roots: Vec<usize>) -> Vec<usize> {
        let mut new_roots = vec![];
        for root in roots.iter() {
            new_roots.push(unsafe { self.copy(*root) });
        }

        self.copy_remaining();

        new_roots
    }

    fn copy_remaining(&mut self) {
        while let Some(mut object) = self.copied.pop() {
            if object.marked() {
                panic!("We are copying to this space, nothing should be marked");
            }

            for datum in object.get_fields_mut() {
                if BuiltInTypes::is_heap_pointer(*datum) {
                    *datum = unsafe { self.copy(*datum) };
                }
            }
        }
    }

    unsafe fn copy(&mut self, root: usize) -> usize {
        let heap_object = HeapObject::from_tagged(root);

        if !self.young.contains(heap_object.get_pointer()) {
            return root;
        }

        // if it is marked we have already copied it
        // We now know that the first field is a pointer
        if heap_object.marked() {
            let first_field = heap_object.get_field(0);
            assert!(BuiltInTypes::is_heap_pointer(first_field));
            assert!(
                !self
                    .young
                    .contains(BuiltInTypes::untag(first_field) as *const u8)
            );
            return first_field;
        }

        let data = heap_object.get_full_object_data();
        let new_pointer = self.old.copy_data_to_offset(data);
        debug_assert!(new_pointer as usize % 8 == 0, "Pointer is not aligned");
        // update header of original object to now be the forwarding pointer
        let tagged_new = BuiltInTypes::get_kind(root).tag(new_pointer as isize) as usize;

        if heap_object.is_zero_size() {
            // Zero-size objects don't have space for forwarding pointer
            return tagged_new;
        }

        // For opaque objects (strings, keywords), we still need to mark them
        // and write the forwarding pointer to prevent duplicate copies
        if !heap_object.is_opaque_object() {
            let first_field = heap_object.get_field(0);
            if let Some(heap_object) = HeapObject::try_from_tagged(first_field) {
                if self.young.contains(heap_object.get_pointer()) {
                    unsafe { self.copy(first_field) };
                }
            }
        }

        // Write forwarding pointer to first word (safe even for opaque objects
        // since we've already copied the data to the new location)
        heap_object.write_field(0, tagged_new);
        heap_object.mark();
        self.copied.push(HeapObject::from_untagged(new_pointer));
        tagged_new
    }

    fn move_objects_referenced_from_old_to_old(&mut self, old_object: &mut HeapObject) {
        if self.young.contains(old_object.get_pointer()) {
            return;
        }
        let data = old_object.get_fields_mut();
        for datum in data.iter_mut() {
            if BuiltInTypes::is_heap_pointer(*datum) {
                let untagged = BuiltInTypes::untag(*datum);
                if !self.young.contains(untagged as *const u8) {
                    continue;
                }
                let new_pointer = unsafe { self.copy(*datum) };
                *datum = new_pointer;
            }
        }
    }

    pub fn gather_roots(
        &mut self,
        stack_base: usize,
        stack_map: &StackMap,
        stack_pointer: usize,
    ) -> Vec<(usize, usize)> {
        let mut roots: Vec<(usize, usize)> = Vec::with_capacity(36);

        StackWalker::walk_stack_roots(stack_base, stack_pointer, stack_map, |offset, pointer| {
            let untagged = BuiltInTypes::untag(pointer);
            if self.young.contains(untagged as *const u8) {
                roots.push((offset, pointer));
            }
        });

        roots
    }
}

// ========== Heap Inspection ==========

use std::collections::HashMap;

impl HeapInspector for GenerationalGC {
    fn iter_objects(&self) -> Box<dyn Iterator<Item = HeapObject> + '_> {
        // Chain young generation (contiguous) with old generation (mark-and-sweep)
        let young_iter = self.young.object_iter_from_position(0);
        let old_iter = self.old.iter_objects();
        Box::new(young_iter.chain(old_iter))
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

        // Get old gen stats for free list info
        let old_stats = self.old.detailed_stats();

        DetailedHeapStats {
            gc_algorithm: "generational",
            total_bytes: self.young.byte_count() + old_stats.total_bytes,
            used_bytes,
            object_count,
            objects_by_type,
            free_list_entries: old_stats.free_list_entries,
            free_bytes: old_stats.free_bytes,
            largest_free_block: old_stats.largest_free_block,
        }
    }

    fn contains_address(&self, addr: usize) -> bool {
        self.young.contains(addr as *const u8) || self.old.contains_address(addr)
    }

    fn get_roots(&self) -> &[(usize, usize)] {
        &self.namespace_roots
    }
}
