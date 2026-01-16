//! Compacting garbage collector using Cheney's algorithm.
//!
//! This is a copying/semi-space collector that compacts live objects,
//! eliminating fragmentation and providing fast bump-pointer allocation.

use std::{ffi::c_void, io::Error, marker::PhantomData, mem};

use libc::mprotect;

use super::get_page_size;
use crate::traits::{ForwardingSupport, GcObject, GcTypes, RootProvider, TaggedPointer};

use super::{AllocateAction, Allocator, AllocatorOptions};

const DEFAULT_PAGE_COUNT: usize = 1024;
const MAX_PAGE_COUNT: usize = 1000000;

// =============================================================================
// Memory Space
// =============================================================================

struct Space {
    start: *const u8,
    page_count: usize,
    allocation_offset: usize,
    #[cfg(debug_assertions)]
    protected: bool,
}

unsafe impl Send for Space {}
unsafe impl Sync for Space {}

impl Space {
    fn word_count(&self) -> usize {
        (self.page_count * get_page_size()) / 8
    }

    fn byte_count(&self) -> usize {
        self.page_count * get_page_size()
    }

    fn contains(&self, pointer: *const u8) -> bool {
        let start = self.start as usize;
        let end = start + self.byte_count();
        let pointer = pointer as usize;
        pointer >= start && pointer < end
    }

    fn copy_data_to_offset(&mut self, data: &[u8]) -> isize {
        unsafe {
            let start = self.start.add(self.allocation_offset);
            let new_pointer = start as isize;
            self.allocation_offset += data.len();
            if self.allocation_offset % 8 != 0 {
                panic!("Heap offset is not aligned");
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), start as *mut u8, data.len());
            new_pointer
        }
    }

    fn write_object<T: GcTypes>(&mut self, offset: usize, size_bytes: usize) -> *const u8 {
        let ptr = unsafe { self.start.add(offset) };
        let mut heap_object = T::ObjectHandle::from_untagged(ptr);

        assert!(self.contains(heap_object.get_pointer()));

        // Zero the full object memory (header + fields) to prevent stale pointers
        let header_size = 8; // Assume simple 8-byte header
        let full_size = size_bytes + header_size;
        unsafe {
            std::ptr::write_bytes(self.start.add(offset) as *mut u8, 0, full_size);
        }

        heap_object.write_header(size_bytes);

        heap_object.get_pointer()
    }

    fn allocate<T: GcTypes>(&mut self, words: usize) -> *const u8 {
        let offset = self.allocation_offset;
        let size_bytes = words * 8;
        let header_size = 8; // Assume simple 8-byte header
        let full_size = size_bytes + header_size;
        let pointer = self.write_object::<T>(offset, size_bytes);
        self.increment_current_offset(full_size);
        pointer
    }

    fn increment_current_offset(&mut self, size: usize) {
        self.allocation_offset += size;
    }

    #[cfg(debug_assertions)]
    fn protect(&mut self) {
        unsafe {
            mprotect(
                self.start as *mut _,
                self.byte_count() - 1024,
                libc::PROT_NONE,
            )
        };
        self.protected = true;
    }

    #[cfg(debug_assertions)]
    fn unprotect(&mut self) {
        unsafe {
            mprotect(
                self.start as *mut _,
                self.byte_count() - 1024,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };
        self.protected = false;
    }

    fn clear(&mut self) {
        self.allocation_offset = 0;
    }

    fn new(default_page_count: usize) -> Self {
        let pre_allocated_space = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                get_page_size() * MAX_PAGE_COUNT,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        Self::commit_memory(pre_allocated_space, default_page_count * get_page_size()).unwrap();
        Self {
            start: pre_allocated_space as *const u8,
            page_count: default_page_count,
            allocation_offset: 0,
            #[cfg(debug_assertions)]
            protected: false,
        }
    }

    fn commit_memory(addr: *mut c_void, size: usize) -> Result<(), Error> {
        unsafe {
            if mprotect(addr, size, libc::PROT_READ | libc::PROT_WRITE) != 0 {
                Err(Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }

    fn double_committed_memory(&mut self) {
        let new_page_count = self.page_count * 2;
        Self::commit_memory(self.start as *mut c_void, new_page_count * get_page_size()).unwrap();
        self.page_count = new_page_count;
    }
}

// =============================================================================
// Object Iterator
// =============================================================================

struct ObjectIterator<'a, T: GcTypes> {
    space: &'a Space,
    offset: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: GcTypes> Iterator for ObjectIterator<'a, T> {
    type Item = T::ObjectHandle;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.space.allocation_offset {
            return None;
        }

        if self.space.allocation_offset == 0 {
            return None;
        }

        let pointer = unsafe { self.space.start.add(self.offset) };
        let object = T::ObjectHandle::from_untagged(pointer);
        let size = object.full_size();

        self.offset += size;
        if self.offset % 8 != 0 {
            panic!("Heap offset is not aligned");
        }
        Some(object)
    }
}

// =============================================================================
// Compacting Heap
// =============================================================================

/// Compacting garbage collector using Cheney's semi-space algorithm.
///
/// This collector maintains two memory spaces and copies live objects
/// from one to the other during collection, compacting them in the process.
///
/// # Type Parameters
/// - `T`: The runtime's type system implementing [`GcTypes`]
///
/// # Requirements
/// - `T::ObjectHandle` must implement [`ForwardingSupport`] for object relocation
pub struct CompactingHeap<T: GcTypes>
where
    T::ObjectHandle: ForwardingSupport,
{
    to_space: Space,
    from_space: Space,
    options: AllocatorOptions,
    _phantom: PhantomData<T>,
}

impl<T: GcTypes> CompactingHeap<T>
where
    T::ObjectHandle: ForwardingSupport,
{
    fn object_iter_from_position(&self, offset: usize) -> ObjectIterator<'_, T> {
        ObjectIterator {
            space: &self.to_space,
            offset,
            _phantom: PhantomData,
        }
    }

    fn copy_using_cheneys_algorithm(&mut self, mut heap_object: T::ObjectHandle) -> T::TaggedValue {
        let untagged = heap_object.get_pointer() as usize;

        debug_assert!(
            self.to_space.contains(untagged as *const u8)
                || self.from_space.contains(untagged as *const u8),
            "Pointer is not in either space"
        );

        // If already in to_space, it's been copied
        if self.to_space.contains(untagged as *const u8) {
            debug_assert!(untagged % 8 == 0, "Pointer is not aligned");
            return heap_object.tagged_pointer();
        }

        // If forwarded, object has been moved - get forwarding pointer
        if heap_object.is_forwarded() {
            return heap_object.get_forwarding_pointer();
        }

        // Copy the object to to_space
        let data = heap_object.get_full_object_data();
        let new_pointer = self.to_space.copy_data_to_offset(data);
        debug_assert!(new_pointer % 8 == 0, "Pointer is not aligned");

        // Get the kind for tagging the new pointer
        let kind = heap_object.get_object_kind().expect("object must have a kind");
        let tagged_new = T::TaggedValue::tag(new_pointer as *const u8, kind);

        // Set forwarding pointer in the old object
        heap_object.set_forwarding_pointer(tagged_new);

        tagged_new
    }

    unsafe fn copy_all(&mut self, roots: Vec<T::TaggedValue>) -> Vec<T::TaggedValue> {
        let start_offset = self.to_space.allocation_offset;
        let mut new_roots = vec![];

        for root in roots.iter() {
            let heap_object = T::ObjectHandle::from_tagged(*root);
            new_roots.push(self.copy_using_cheneys_algorithm(heap_object));
        }

        self.copy_remaining(start_offset);

        new_roots
    }

    fn copy_remaining(&mut self, start_offset: usize) {
        // Iterate over objects in to_space starting from start_offset
        let mut offset = start_offset;
        while offset < self.to_space.allocation_offset {
            let ptr = unsafe { self.to_space.start.add(offset) };
            let mut object = T::ObjectHandle::from_untagged(ptr);

            if object.marked() {
                panic!("Objects in to_space should not be marked");
            }
            if object.is_zero_size() {
                offset += object.full_size();
                offset = (offset + 7) & !7;
                continue;
            }

            // Update fields that point to from_space objects
            for field in object.get_fields_mut() {
                let tagged = T::TaggedValue::from_usize(*field);
                if tagged.is_heap_pointer() {
                    let field_object = T::ObjectHandle::from_tagged(tagged);
                    *field = self.copy_using_cheneys_algorithm(field_object).as_usize();
                }
            }

            offset += object.full_size();
            offset = (offset + 7) & !7;
        }
    }
}

impl<T: GcTypes> Allocator<T> for CompactingHeap<T>
where
    T::ObjectHandle: ForwardingSupport,
{
    fn new(options: AllocatorOptions) -> Self {
        let to_space = Space::new(DEFAULT_PAGE_COUNT / 2);
        let from_space = Space::new(DEFAULT_PAGE_COUNT / 2);

        Self {
            to_space,
            from_space,
            options,
            _phantom: PhantomData,
        }
    }

    fn try_allocate(
        &mut self,
        words: usize,
        _kind: T::ObjectKind,
    ) -> Result<AllocateAction, Box<dyn std::error::Error>> {
        if words > self.from_space.word_count() {
            self.grow();
        }

        if self.from_space.allocation_offset + words * 8 >= self.from_space.byte_count() {
            return Ok(AllocateAction::Gc);
        }

        let pointer = self.from_space.allocate::<T>(words);

        Ok(AllocateAction::Allocated(pointer))
    }

    fn gc(&mut self, roots: &dyn RootProvider<T::TaggedValue>) {
        if !self.options.gc {
            return;
        }

        #[cfg(debug_assertions)]
        {
            self.to_space.unprotect();
        }

        // Collect roots with their slot addresses
        let mut root_slots: Vec<(usize, T::TaggedValue)> = Vec::new();
        roots.enumerate_roots(&mut |slot_addr, tagged| {
            if tagged.is_heap_pointer() {
                root_slots.push((slot_addr, tagged));
            }
        });

        // Copy all root objects
        let new_roots = unsafe {
            self.copy_all(root_slots.iter().map(|(_, tagged)| *tagged).collect())
        };

        // Update root slots with new locations
        for (i, (slot_addr, _)) in root_slots.iter().enumerate() {
            debug_assert!(
                new_roots[i].untag() as usize % 8 == 0,
                "Pointer is not aligned"
            );
            unsafe {
                *(*slot_addr as *mut usize) = new_roots[i].as_usize();
            }
        }

        let start_offset = self.to_space.allocation_offset;
        self.copy_remaining(start_offset);

        mem::swap(&mut self.from_space, &mut self.to_space);

        self.to_space.clear();

        #[cfg(debug_assertions)]
        {
            self.to_space.protect();
        }
    }

    fn grow(&mut self) {
        self.from_space.double_committed_memory();

        #[cfg(debug_assertions)]
        {
            let currently_protected = self.to_space.protected;
            if currently_protected {
                self.to_space.unprotect();
            }
            self.to_space.double_committed_memory();
            if currently_protected {
                self.to_space.protect();
            }
        }

        #[cfg(not(debug_assertions))]
        {
            self.to_space.double_committed_memory();
        }
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }
}
