//! Mark-and-sweep garbage collector.
//!
//! This is a simple non-moving collector that uses a free list for allocation.
//! Objects are marked during tracing, then unmarked objects are added to the free list.

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::traits::{GcObject, GcTypes, RootProvider, TaggedPointer, Word};

use super::{AllocError, AllocateAction, Allocator, AllocatorOptions, MemoryProvider};

const DEFAULT_PAGE_COUNT: usize = 1024;
const MAX_PAGE_COUNT: usize = 1000000;

// =============================================================================
// Memory Space
// =============================================================================

struct Space<M: MemoryProvider> {
    start: *const u8,
    page_count: usize,
    highmark: usize,
    memory: M,
}

unsafe impl<M: MemoryProvider + Send> Send for Space<M> {}
unsafe impl<M: MemoryProvider + Sync> Sync for Space<M> {}

impl<M: MemoryProvider> Space<M> {
    fn byte_count(&self) -> usize {
        self.page_count * self.memory.page_size()
    }

    fn contains(&self, pointer: *const u8) -> bool {
        let start = self.start as usize;
        let end = start + self.byte_count();
        let pointer = pointer as usize;
        pointer >= start && pointer < end
    }

    fn copy_data_to_offset(&mut self, offset: usize, data: &[u8]) -> isize {
        unsafe {
            let start = self.start.add(offset);
            let new_pointer = start as isize;
            core::ptr::copy_nonoverlapping(data.as_ptr(), start as *mut u8, data.len());
            new_pointer
        }
    }

    fn write_object<T: GcTypes>(&mut self, offset: usize, size_bytes: usize) -> *const u8 {
        let ptr = unsafe { self.start.add(offset) };
        let mut heap_object = T::ObjectHandle::from_untagged(ptr);

        assert!(self.contains(heap_object.get_pointer()));
        heap_object.write_header(size_bytes);

        heap_object.get_pointer()
    }

    fn new(mut memory: M, default_page_count: usize) -> Result<Self, AllocError> {
        let page_size = memory.page_size();
        let total_size = page_size * MAX_PAGE_COUNT;

        let start = memory
            .allocate_region(total_size)
            .ok_or(AllocError::ProviderFailed)?;

        let initial_size = default_page_count * page_size;
        if !memory.commit(start, initial_size) {
            return Err(AllocError::ProviderFailed);
        }

        Ok(Self {
            start: start as *const u8,
            page_count: default_page_count,
            highmark: 0,
            memory,
        })
    }

    fn double_committed_memory(&mut self) -> bool {
        let page_size = self.memory.page_size();
        let new_page_count = self.page_count * 2;
        let new_size = new_page_count * page_size;

        if self.memory.commit(self.start as *mut u8, new_size) {
            self.page_count = new_page_count;
            true
        } else {
            false
        }
    }

    fn update_highmark(&mut self, highmark: usize) {
        if highmark > self.highmark {
            self.highmark = highmark;
        }
    }
}

// =============================================================================
// Free List
// =============================================================================

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

pub struct FreeList {
    ranges: Vec<FreeListEntry>,
}

impl FreeList {
    fn new(starting_range: FreeListEntry) -> Self {
        FreeList {
            ranges: alloc::vec![starting_range],
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

// =============================================================================
// Mark and Sweep Collector
// =============================================================================

/// Mark-and-sweep garbage collector.
///
/// Uses a free list for allocation and marks live objects during collection.
///
/// # Type Parameters
/// - `T`: The runtime's type system implementing [`GcTypes`]
/// - `M`: The memory provider implementing [`MemoryProvider`]
pub struct MarkAndSweep<T: GcTypes, M: MemoryProvider> {
    space: Space<M>,
    free_list: FreeList,
    options: AllocatorOptions,
    _phantom: PhantomData<T>,
}

impl<T: GcTypes, M: MemoryProvider> MarkAndSweep<T, M> {
    /// Check if a pointer is within this allocator's space.
    pub fn contains(&self, pointer: *const u8) -> bool {
        self.space.contains(pointer)
    }

    /// Get the start address of this heap space.
    pub fn heap_start(&self) -> usize {
        self.space.start as usize
    }

    /// Get the size of this heap space in bytes.
    pub fn heap_size(&self) -> usize {
        self.space.byte_count()
    }

    fn can_allocate(&self, words: usize) -> bool {
        let word = Word::from_words(words);
        let header_size = 8;
        let size = word.to_bytes() + header_size;
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
    ) -> Result<AllocateAction, AllocError> {
        let header_size = 8;
        let size_bytes = words.to_bytes() + header_size;

        let offset = self.free_list.allocate(size_bytes);
        if let Some(offset) = offset {
            self.space.update_highmark(offset);
            let pointer = self.space.write_object::<T>(offset, words.to_bytes());
            if let Some(data) = data {
                self.space.copy_data_to_offset(offset, data);
            }
            return Ok(AllocateAction::Allocated(pointer));
        }

        Ok(AllocateAction::Gc)
    }

    #[allow(unused)]
    pub fn copy_data_to_offset(&mut self, data: &[u8]) -> *const u8 {
        let header_size = 8;

        let pointer = self
            .allocate_inner(Word::from_bytes(data.len() - header_size), Some(data))
            .unwrap();

        if let AllocateAction::Allocated(pointer) = pointer {
            pointer
        } else {
            self.grow();
            self.copy_data_to_offset(data)
        }
    }

    fn mark(&self, roots: &dyn RootProvider<T::TaggedValue>) {
        let mut to_mark: Vec<T::ObjectHandle> = Vec::with_capacity(128);

        roots.enumerate_roots(&mut |_slot_addr, tagged| {
            if tagged.is_heap_pointer() {
                to_mark.push(T::ObjectHandle::from_tagged(tagged));
            }
        });

        while let Some(object) = to_mark.pop() {
            if object.marked() {
                continue;
            }

            object.mark();

            if !object.is_opaque() {
                for &field_value in object.get_fields() {
                    let tagged = T::TaggedValue::from_usize(field_value);
                    if tagged.is_heap_pointer() {
                        to_mark.push(T::ObjectHandle::from_tagged(tagged));
                    }
                }
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
            let heap_object =
                T::ObjectHandle::from_untagged(unsafe { self.space.start.add(offset) });

            let full_size = heap_object.full_size();

            if heap_object.marked() {
                heap_object.unmark();
                offset += full_size;
                offset = (offset + 7) & !7;
                continue;
            }
            let size = full_size;
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

    /// Create a new allocator with a specific page count.
    pub fn new_with_page_count(
        memory: M,
        page_count: usize,
        options: AllocatorOptions,
    ) -> Result<Self, AllocError> {
        let space = Space::new(memory, page_count)?;
        let size = space.byte_count();
        Ok(Self {
            space,
            free_list: FreeList::new(FreeListEntry { offset: 0, size }),
            options,
            _phantom: PhantomData,
        })
    }

    /// Walk all live objects in the heap, calling the provided function for each one.
    pub fn walk_objects_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut T::ObjectHandle),
    {
        let mut offset = 0;
        loop {
            if offset > self.space.highmark {
                break;
            }
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }
            let ptr = unsafe { self.space.start.add(offset) };
            let mut heap_object = T::ObjectHandle::from_untagged(ptr);
            f(ptr as usize, &mut heap_object);
            offset += heap_object.full_size();
            offset = (offset + 7) & !7;
        }
    }
}

impl<T: GcTypes, M: MemoryProvider> Allocator<T, M> for MarkAndSweep<T, M> {
    fn new(options: AllocatorOptions, memory: M) -> Self {
        Self::new_with_page_count(memory, DEFAULT_PAGE_COUNT, options)
            .expect("Failed to create MarkAndSweep allocator")
    }

    fn try_allocate(
        &mut self,
        words: usize,
        _kind: T::ObjectKind,
    ) -> Result<AllocateAction, AllocError> {
        if self.can_allocate(words) {
            self.allocate_inner(Word::from_words(words), None)
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn gc(&mut self, roots: &dyn RootProvider<T::TaggedValue>) {
        if !self.options.gc {
            return;
        }

        #[cfg(feature = "std")]
        let start = std::time::Instant::now();

        self.mark(roots);
        self.sweep();

        #[cfg(feature = "std")]
        if self.options.print_stats {
            println!("Mark and sweep took {:?}", start.elapsed());
        }
    }

    fn grow(&mut self) {
        let current_max_offset = self.space.byte_count();
        if self.space.double_committed_memory() {
            let after_max_offset = self.space.byte_count();
            self.free_list.insert(FreeListEntry {
                offset: current_max_offset,
                size: after_max_offset - current_max_offset,
            });
        }
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }
}
