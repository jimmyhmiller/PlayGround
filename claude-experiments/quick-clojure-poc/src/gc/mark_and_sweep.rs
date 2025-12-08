// Mark and Sweep GC
//
// Classic mark-and-sweep garbage collector with free list allocation.

use std::error::Error;

use super::space::{Space, DEFAULT_PAGE_COUNT};
use super::stack_walker::StackWalker;
use super::types::{BuiltInTypes, HeapObject, Word};
use super::{AllocateAction, Allocator, AllocatorOptions, StackMap};

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

    fn mark_and_sweep(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        let start = std::time::Instant::now();
        for (stack_base, stack_pointer) in stack_pointers {
            self.mark(*stack_base, stack_map, *stack_pointer);
        }
        self.sweep();
        if self.options.print_stats {
            println!("Mark and sweep took {:?}", start.elapsed());
        }
    }

    fn mark(&self, stack_base: usize, stack_map: &StackMap, stack_pointer: usize) {
        let mut to_mark: Vec<HeapObject> = Vec::with_capacity(128);

        for (_, root) in self.namespace_roots.iter() {
            if !BuiltInTypes::is_heap_pointer(*root) {
                continue;
            }
            to_mark.push(HeapObject::from_tagged(*root));
        }

        // Use the stack walker to find heap pointers
        StackWalker::walk_stack_roots(stack_base, stack_pointer, stack_map, |_, pointer| {
            to_mark.push(HeapObject::from_tagged(pointer));
        });

        while let Some(object) = to_mark.pop() {
            if object.marked() {
                continue;
            }

            object.mark();
            for object in object.get_heap_references() {
                to_mark.push(object);
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

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        self.mark_and_sweep(stack_map, stack_pointers);
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

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }
}
