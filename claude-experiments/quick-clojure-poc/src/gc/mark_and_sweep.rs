// Mark and Sweep GC
//
// Classic mark-and-sweep garbage collector with free list allocation.

use std::error::Error;

use super::space::{DEFAULT_PAGE_COUNT, Space};
use super::stack_walker::StackWalker;
use super::types::{BuiltInTypes, HeapObject, Word};
use super::{
    AllocateAction, Allocator, AllocatorOptions, DetailedHeapStats, HeapInspector, StackMap,
    type_id_to_name,
};

/// Free list entry
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct FreeListEntry {
    offset: usize,
    size: usize,
}

impl FreeListEntry {
    pub fn end(&self) -> usize {
        self.offset + self.size
    }

    pub fn can_hold(&self, size: usize) -> bool {
        self.size >= size
    }

    pub fn contains(&self, offset: usize) -> bool {
        self.offset <= offset && offset < self.end()
    }
}

/// Free list for mark-and-sweep allocation
pub struct FreeList {
    ranges: Vec<FreeListEntry>, // always sorted by start
}

impl FreeList {
    fn new(starting_range: FreeListEntry) -> Self {
        FreeList {
            ranges: vec![starting_range],
        }
    }

    fn insert(&mut self, range: FreeListEntry) {
        let mut i = match self
            .ranges
            .binary_search_by_key(&range.offset, |r| r.offset)
        {
            Ok(i) | Err(i) => i,
        };

        // Coalesce with previous if adjacent
        if i > 0 && self.ranges[i - 1].end() == range.offset {
            i -= 1;
            self.ranges[i].size += range.size;
        } else {
            self.ranges.insert(i, range);
        }

        // Coalesce with next if adjacent
        if i + 1 < self.ranges.len() && self.ranges[i].end() == self.ranges[i + 1].offset {
            self.ranges[i].size += self.ranges[i + 1].size;
            self.ranges.remove(i + 1);
        }
    }

    fn allocate(&mut self, size: usize) -> Option<usize> {
        for (i, r) in self.ranges.iter_mut().enumerate() {
            if r.can_hold(size) {
                let addr = r.offset;
                if addr % 8 != 0 {
                    panic!("Heap offset is not aligned");
                }
                r.offset += size;
                r.size -= size;

                if r.size == 0 {
                    self.ranges.remove(i);
                }

                return Some(addr);
            }
        }
        None
    }

    fn iter(&self) -> impl Iterator<Item = &FreeListEntry> {
        self.ranges.iter()
    }

    fn find_entry_contains(&self, offset: usize) -> Option<&FreeListEntry> {
        self.ranges.iter().find(|&entry| entry.contains(offset))
    }
}

/// Mark-and-sweep garbage collector
pub struct MarkAndSweep {
    space: Space,
    free_list: FreeList,
    namespace_roots: Vec<(usize, usize)>,
    options: AllocatorOptions,
    temporary_roots: Vec<Option<usize>>,
}

impl MarkAndSweep {
    fn can_allocate(&self, words: usize) -> bool {
        let words = Word::from_word(words);
        let size = words.to_bytes() + HeapObject::header_size();
        let spot = self
            .free_list
            .iter()
            .enumerate()
            .find(|(_, x)| x.size >= size);
        spot.is_some()
    }

    fn allocate_inner(
        &mut self,
        words: Word,
        data: Option<&[u8]>,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let size_bytes = words.to_bytes() + HeapObject::header_size();

        let offset = self.free_list.allocate(size_bytes);
        if let Some(offset) = offset {
            self.space.update_highmark(offset);
            let pointer = self.space.write_object(offset, words);
            if let Some(data) = data {
                self.space.copy_data_to_specific_offset(offset, data);
            }
            return Ok(AllocateAction::Allocated(pointer));
        }

        Ok(AllocateAction::Gc)
    }

    #[allow(unused)]
    pub fn copy_data_to_offset(&mut self, data: &[u8]) -> *const u8 {
        let pointer = self
            .allocate_inner(Word::from_bytes(data.len() - 8), Some(data))
            .unwrap();

        if let AllocateAction::Allocated(pointer) = pointer {
            pointer
        } else {
            self.grow();
            self.copy_data_to_offset(data)
        }
    }

    fn mark_and_sweep(&mut self, stack_map: &StackMap, stack_info: &[(usize, usize, usize)]) {
        let start = std::time::Instant::now();
        // Mark from namespace roots (always done)
        // Then optionally scan stack if stack_info provided
        self.mark_from_roots();
        for (stack_base, frame_pointer, gc_return_addr) in stack_info {
            self.mark_stack(*stack_base, stack_map, *frame_pointer, *gc_return_addr);
        }
        self.sweep();
        if self.options.print_stats {
            println!("Mark and sweep took {:?}", start.elapsed());
        }
    }

    fn mark_from_roots(&self) {
        let mut to_mark: Vec<HeapObject> = Vec::with_capacity(128);

        // Mark namespace roots
        for (_, root) in self.namespace_roots.iter() {
            if !BuiltInTypes::is_heap_pointer(*root) {
                continue;
            }
            to_mark.push(HeapObject::from_tagged(*root));
        }

        // Mark temporary roots (registered during allocations to protect intermediate values)
        for root in self.temporary_roots.iter().flatten() {
            if BuiltInTypes::is_heap_pointer(*root) {
                to_mark.push(HeapObject::from_tagged(*root));
            }
        }

        self.trace_objects(to_mark);
    }

    fn mark_stack(&self, stack_base: usize, stack_map: &StackMap, frame_pointer: usize, gc_return_addr: usize) {
        let mut to_mark: Vec<HeapObject> = Vec::with_capacity(128);

        // Use the stack walker to find heap pointers using frame pointer chain traversal
        // The walker will naturally stop when FP is outside the JIT stack range
        StackWalker::walk_stack_roots_with_return_addr(
            stack_base,
            frame_pointer,
            gc_return_addr,
            stack_map,
            |_, pointer| {
                let untagged = BuiltInTypes::untag(pointer);
                // Only add if the pointer is within our heap bounds
                if self.space.contains(untagged as *const u8) {
                    to_mark.push(HeapObject::from_tagged(pointer));
                }
            },
        );

        self.trace_objects(to_mark);
    }

    fn trace_objects(&self, mut to_mark: Vec<HeapObject>) {
        while let Some(object) = to_mark.pop() {
            if object.marked() {
                continue;
            }

            object.mark();
            for child in object.get_heap_references() {
                to_mark.push(child);
            }
        }
    }

    fn sweep(&mut self) {
        let mut offset = 0;

        loop {
            if offset > self.space.highmark {
                break;
            }
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }
            let heap_object = HeapObject::from_untagged(unsafe { self.space.start.add(offset) });

            if heap_object.marked() {
                heap_object.unmark();
                offset += heap_object.full_size();
                offset = (offset + 7) & !7;
                continue;
            }
            let size = heap_object.full_size();
            let entry = FreeListEntry { offset, size };
            self.free_list.insert(entry);
            offset += size;
            offset = (offset + 7) & !7;
            if offset % 8 != 0 {
                panic!("Heap offset is not aligned");
            }

            if offset > self.space.byte_count() {
                panic!("Heap offset is out of bounds");
            }
        }
    }

    #[allow(unused)]
    pub fn clear_namespace_roots(&mut self) {
        self.namespace_roots.clear();
    }

    pub fn new_with_page_count(page_count: usize, options: AllocatorOptions) -> Self {
        let space = Space::new(page_count);
        let size = space.byte_count();
        Self {
            space,
            free_list: FreeList::new(FreeListEntry { offset: 0, size }),
            namespace_roots: vec![],
            options,
            temporary_roots: vec![],
        }
    }
}

impl Allocator for MarkAndSweep {
    fn new(options: AllocatorOptions) -> Self {
        let page_count = DEFAULT_PAGE_COUNT;
        Self::new_with_page_count(page_count, options)
    }

    fn try_allocate(
        &mut self,
        words: usize,
        _kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        if self.can_allocate(words) {
            self.allocate_inner(Word::from_word(words), None)
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn gc(&mut self, stack_map: &StackMap, stack_info: &[(usize, usize, usize)]) {
        self.mark_and_sweep(stack_map, stack_info);
    }

    fn grow(&mut self) {
        let current_max_offset = self.space.byte_count();
        self.space.double_committed_memory();
        let after_max_offset = self.space.byte_count();
        self.free_list.insert(FreeListEntry {
            offset: current_max_offset,
            size: after_max_offset - current_max_offset,
        });
    }

    fn gc_add_root(&mut self, _old: usize) {}

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

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        // This mark and sweep doesn't relocate
        // so we don't have any relocations
        vec![]
    }
}

// ========== Heap Inspection ==========

use std::collections::HashMap;

/// Iterator over live objects in mark-and-sweep heap (skips free list regions)
pub struct LiveObjectIterator<'a> {
    space: &'a Space,
    free_list: &'a FreeList,
    offset: usize,
    highmark: usize,
}

impl<'a> LiveObjectIterator<'a> {
    fn new(space: &'a Space, free_list: &'a FreeList, highmark: usize) -> Self {
        Self {
            space,
            free_list,
            offset: 0,
            highmark,
        }
    }
}

impl<'a> Iterator for LiveObjectIterator<'a> {
    type Item = HeapObject;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.offset > self.highmark {
                return None;
            }

            // Skip free list regions
            if let Some(entry) = self.free_list.find_entry_contains(self.offset) {
                self.offset = entry.end();
                continue;
            }

            let pointer = unsafe { self.space.start.add(self.offset) };
            let object = HeapObject::from_untagged(pointer);
            let size = object.full_size();

            self.offset += size;
            // Align to 8 bytes
            self.offset = (self.offset + 7) & !7;

            return Some(object);
        }
    }
}

impl HeapInspector for MarkAndSweep {
    fn iter_objects(&self) -> Box<dyn Iterator<Item = HeapObject> + '_> {
        Box::new(LiveObjectIterator::new(
            &self.space,
            &self.free_list,
            self.space.highmark,
        ))
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

        // Free list stats
        let free_list_entries = self.free_list.ranges.len();
        let free_bytes: usize = self.free_list.ranges.iter().map(|e| e.size).sum();
        let largest_free_block = self
            .free_list
            .ranges
            .iter()
            .map(|e| e.size)
            .max()
            .unwrap_or(0);

        DetailedHeapStats {
            gc_algorithm: "mark-and-sweep",
            total_bytes: self.space.byte_count(),
            used_bytes,
            object_count,
            objects_by_type,
            free_list_entries: Some(free_list_entries),
            free_bytes: Some(free_bytes),
            largest_free_block: Some(largest_free_block),
        }
    }

    fn contains_address(&self, addr: usize) -> bool {
        self.space.contains(addr as *const u8)
    }
}
