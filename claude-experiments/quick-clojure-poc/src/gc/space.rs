// Space - mmap-backed memory management for GC
//
// This provides the memory space abstraction used by all GC algorithms.

use std::{ffi::c_void, io};

use libc::{mprotect, vm_page_size};

use super::types::{HeapObject, Word};

pub const DEFAULT_PAGE_COUNT: usize = 1024;
// Arbitrary number that should be changed when I have better options for gc
pub const MAX_PAGE_COUNT: usize = 1000000;

/// A memory space backed by mmap
pub struct Space {
    pub start: *const u8,
    pub page_count: usize,
    pub allocation_offset: usize,
    pub highmark: usize,
    pub protected: bool,
}

unsafe impl Send for Space {}
unsafe impl Sync for Space {}

impl Space {
    pub fn new(default_page_count: usize) -> Self {
        let pre_allocated_space = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                vm_page_size * MAX_PAGE_COUNT,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        Self::commit_memory(
            pre_allocated_space,
            default_page_count * unsafe { vm_page_size },
        )
        .unwrap();

        Self {
            start: pre_allocated_space as *const u8,
            page_count: default_page_count,
            allocation_offset: 0,
            highmark: 0,
            protected: false,
        }
    }

    pub fn commit_memory(addr: *mut c_void, size: usize) -> Result<(), io::Error> {
        unsafe {
            if mprotect(addr, size, libc::PROT_READ | libc::PROT_WRITE) != 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }

    pub fn word_count(&self) -> usize {
        (self.page_count * unsafe { vm_page_size }) / 8
    }

    pub fn byte_count(&self) -> usize {
        self.page_count * unsafe { vm_page_size }
    }

    pub fn contains(&self, pointer: *const u8) -> bool {
        let start = self.start as usize;
        let end = start + self.byte_count();
        let pointer = pointer as usize;
        pointer >= start && pointer < end
    }

    pub fn copy_data_to_offset(&mut self, data: &[u8]) -> isize {
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

    pub fn copy_data_to_specific_offset(&mut self, offset: usize, data: &[u8]) -> isize {
        unsafe {
            let start = self.start.add(offset);
            let new_pointer = start as isize;
            std::ptr::copy_nonoverlapping(data.as_ptr(), start as *mut u8, data.len());
            new_pointer
        }
    }

    pub fn write_object(&mut self, offset: usize, size: Word) -> *const u8 {
        let mut heap_object = HeapObject::from_untagged(unsafe { self.start.add(offset) });

        assert!(self.contains(heap_object.get_pointer()));
        heap_object.write_header(size);

        heap_object.get_pointer()
    }

    pub fn allocate(&mut self, words: usize) -> *const u8 {
        let offset = self.allocation_offset;
        let size = Word::from_word(words);
        let full_size = size.to_bytes() + HeapObject::header_size();
        let pointer = self.write_object(offset, size);
        self.increment_current_offset(full_size);
        pointer
    }

    pub fn allocate_word(&mut self, size: Word) -> *const u8 {
        let offset = self.allocation_offset;
        let full_size = size.to_bytes() + HeapObject::header_size();
        let pointer = self.write_object(offset, size);
        self.increment_current_offset(full_size);
        pointer
    }

    pub fn increment_current_offset(&mut self, size: usize) {
        self.allocation_offset += size;
    }

    pub fn update_highmark(&mut self, highmark: usize) {
        if highmark > self.highmark {
            self.highmark = highmark;
        }
    }

    pub fn clear(&mut self) {
        self.allocation_offset = 0;
    }

    pub fn protect(&mut self) {
        unsafe {
            mprotect(
                self.start as *mut _,
                self.byte_count() - 1024,
                libc::PROT_NONE,
            )
        };
        self.protected = true;
    }

    pub fn unprotect(&mut self) {
        unsafe {
            mprotect(
                self.start as *mut _,
                self.byte_count() - 1024,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };
        self.protected = false;
    }

    pub fn double_committed_memory(&mut self) {
        let new_page_count = self.page_count * 2;
        Self::commit_memory(
            self.start as *mut c_void,
            new_page_count * unsafe { vm_page_size },
        )
        .unwrap();
        self.page_count = new_page_count;
    }

    pub fn can_allocate(&self, size: Word) -> bool {
        let size = size.to_bytes() + HeapObject::header_size();
        let new_offset = self.allocation_offset + size;
        if new_offset > self.byte_count() {
            return false;
        }
        true
    }

    pub fn object_iter_from_position(&self, offset: usize) -> ObjectIterator<'_> {
        ObjectIterator {
            space: self,
            offset,
        }
    }
}

/// Iterator over objects in a space
pub struct ObjectIterator<'a> {
    space: &'a Space,
    offset: usize,
}

impl<'a> Iterator for ObjectIterator<'a> {
    type Item = HeapObject;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.space.allocation_offset {
            return None;
        }

        if self.space.allocation_offset == 0 {
            return None;
        }

        let pointer = unsafe { self.space.start.add(self.offset) };
        let object = HeapObject::from_untagged(pointer);
        let size = object.full_size();

        self.offset += size;
        if self.offset % 8 != 0 {
            panic!("Heap offset is not aligned");
        }
        Some(object)
    }
}
