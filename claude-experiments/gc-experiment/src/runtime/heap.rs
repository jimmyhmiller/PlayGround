use std::alloc::{alloc, dealloc, Layout};
use std::ffi::c_void;
use crate::runtime::object::{ObjectHeader, ObjectMeta, Object};

/// A simple heap that tracks all allocated objects.
/// This is intentionally simple - you can replace it with your own GC.
pub struct Heap {
    /// All allocated objects (for iteration during GC)
    objects: Vec<Object>,
    /// Total bytes allocated
    bytes_allocated: usize,
}

impl Heap {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            objects: Vec::new(),
            bytes_allocated: 0,
        })
    }

    /// Allocate a new object with the given metadata and payload size.
    /// Returns a pointer to the ObjectHeader.
    pub fn allocate(&mut self, meta: *const ObjectMeta, payload_size: usize) -> Object {
        let total_size = std::mem::size_of::<ObjectHeader>() + payload_size;
        let layout = Layout::from_size_align(total_size, 8).unwrap();

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            panic!("Out of memory");
        }

        // Initialize the header
        let header = ptr as *mut ObjectHeader;
        unsafe {
            (*header).meta = meta;
            (*header).gc_flags = 0;
            (*header).aux = 0;

            // Zero the payload
            std::ptr::write_bytes(ptr.add(std::mem::size_of::<ObjectHeader>()), 0, payload_size);
        }

        self.objects.push(header);
        self.bytes_allocated += total_size;

        header
    }

    /// Allocate an array object
    pub fn allocate_array(&mut self, meta: *const ObjectMeta, length: u32) -> Object {
        let payload_size = (length as usize) * std::mem::size_of::<*mut c_void>();
        let obj = self.allocate(meta, payload_size);
        unsafe {
            (*obj).aux = length;
        }
        obj
    }

    /// Get all objects (for GC iteration)
    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    /// Get total bytes allocated
    pub fn bytes_allocated(&self) -> usize {
        self.bytes_allocated
    }

    /// Clear mark bits on all objects
    pub fn clear_marks(&mut self) {
        for &obj in &self.objects {
            unsafe {
                (*obj).set_marked(false);
            }
        }
    }

    /// Sweep unmarked objects and free their memory
    pub fn sweep(&mut self) {
        let mut live_objects = Vec::new();
        let mut freed_bytes = 0;

        for &obj in &self.objects {
            let header = unsafe { &*obj };
            if header.is_marked() {
                live_objects.push(obj);
            } else {
                // Free this object
                let meta = header.meta;
                let payload_size = if meta.is_null() {
                    0
                } else {
                    let meta = unsafe { &*meta };
                    match meta.object_type {
                        crate::runtime::object::ObjectType::Struct => {
                            // Would need to store size in meta or object
                            // For now, we don't free struct memory (leak)
                            0
                        }
                        crate::runtime::object::ObjectType::Array => {
                            (header.aux as usize) * std::mem::size_of::<*mut c_void>()
                        }
                    }
                };

                let total_size = std::mem::size_of::<ObjectHeader>() + payload_size;
                let layout = Layout::from_size_align(total_size, 8).unwrap();
                unsafe {
                    dealloc(obj as *mut u8, layout);
                }
                freed_bytes += total_size;
            }
        }

        self.objects = live_objects;
        self.bytes_allocated -= freed_bytes;
    }

    /// Simple mark-sweep GC (for testing)
    /// You would replace this with your own GC
    pub fn collect(&mut self, roots: &[*mut c_void]) {
        // Clear marks
        self.clear_marks();

        // Mark phase
        let mut worklist: Vec<Object> = roots
            .iter()
            .filter_map(|&root| {
                if root.is_null() {
                    None
                } else {
                    Some(root as Object)
                }
            })
            .collect();

        while let Some(obj) = worklist.pop() {
            let header = unsafe { &mut *obj };
            if header.is_marked() {
                continue;
            }
            header.set_marked(true);

            // Trace pointers in this object
            for ptr_field in header.pointer_fields() {
                let child = unsafe { *ptr_field };
                if !child.is_null() {
                    worklist.push(child as Object);
                }
            }
        }

        // Sweep phase
        self.sweep();
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self {
            objects: Vec::new(),
            bytes_allocated: 0,
        }
    }
}
