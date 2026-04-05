use std::collections::HashMap;
use std::ptr;

use crate::value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjType {
    String,
    Function,
    Native,
    Closure,
    Upvalue,
    Class,
    Instance,
    BoundMethod,
}

#[repr(C)]
pub struct Obj {
    pub obj_type: ObjType,
    pub is_marked: bool,
    pub next: *mut Obj,
}

#[repr(C)]
pub struct ObjString {
    pub obj: Obj,
    pub hash: u32,
    pub chars: String,
}

pub type NativeFn = fn(arg_count: u8, args: *const Value) -> Value;

#[repr(C)]
pub struct ObjNative {
    pub obj: Obj,
    pub function: NativeFn,
}

#[repr(C)]
pub struct ObjClass {
    pub obj: Obj,
    pub name: *mut ObjString,
    pub methods: HashMap<*mut ObjString, Value>,
}

#[repr(C)]
pub struct ObjInstance {
    pub obj: Obj,
    pub class: *mut ObjClass,
    pub fields: HashMap<*mut ObjString, Value>,
}

#[repr(C)]
pub struct ObjBoundMethod {
    pub obj: Obj,
    pub receiver: Value,
    pub method: Value,
}

// GC heap
pub struct GcHeap {
    pub objects: *mut Obj,
    pub strings: HashMap<u64, *mut ObjString>,
    pub bytes_allocated: usize,
    pub next_gc: usize,
    pub gray_stack: Vec<*mut Obj>,
}

impl GcHeap {
    pub fn new() -> Self {
        GcHeap {
            objects: ptr::null_mut(),
            strings: HashMap::new(),
            bytes_allocated: 0,
            next_gc: 1024 * 1024,
            gray_stack: Vec::new(),
        }
    }

    fn alloc_obj<T>(&mut self, obj_type: ObjType) -> *mut T {
        let size = std::mem::size_of::<T>();
        self.bytes_allocated += size;
        let layout = std::alloc::Layout::new::<T>();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut T };
        let obj_ptr = ptr as *mut Obj;
        unsafe {
            (*obj_ptr).obj_type = obj_type;
            (*obj_ptr).is_marked = false;
            (*obj_ptr).next = self.objects;
        }
        self.objects = obj_ptr;
        ptr
    }

    pub fn alloc_string(&mut self, chars: String) -> *mut ObjString {
        let hash = hash_string(&chars);
        if let Some(&interned) = self.strings.get(&Self::intern_key(hash, &chars)) {
            return interned;
        }
        let ptr = self.alloc_obj::<ObjString>(ObjType::String);
        unsafe {
            ptr::write(&mut (*ptr).hash, hash);
            ptr::write(&mut (*ptr).chars, chars.clone());
        }
        self.strings.insert(Self::intern_key(hash, &chars), ptr);
        ptr
    }

    fn intern_key(hash: u32, chars: &str) -> u64 {
        let mut key = hash as u64;
        key = key.wrapping_mul(0x517cc1b727220a95);
        key ^= chars.len() as u64;
        for (i, b) in chars.bytes().take(8).enumerate() {
            key ^= (b as u64) << (i * 8);
        }
        key
    }

    pub fn alloc_native(&mut self, function: NativeFn) -> *mut ObjNative {
        let ptr = self.alloc_obj::<ObjNative>(ObjType::Native);
        unsafe { (*ptr).function = function; }
        ptr
    }

    pub fn alloc_class(&mut self, name: *mut ObjString) -> *mut ObjClass {
        let ptr = self.alloc_obj::<ObjClass>(ObjType::Class);
        unsafe {
            (*ptr).name = name;
            ptr::write(&mut (*ptr).methods, HashMap::new());
        }
        ptr
    }

    pub fn alloc_instance(&mut self, class: *mut ObjClass) -> *mut ObjInstance {
        let ptr = self.alloc_obj::<ObjInstance>(ObjType::Instance);
        unsafe {
            (*ptr).class = class;
            ptr::write(&mut (*ptr).fields, HashMap::new());
        }
        ptr
    }

    pub fn free_objects(&mut self) {
        let mut obj = self.objects;
        while !obj.is_null() {
            let next = unsafe { (*obj).next };
            self.free_object(obj);
            obj = next;
        }
        self.objects = ptr::null_mut();
    }

    fn free_object(&mut self, obj: *mut Obj) {
        unsafe {
            let size = match (*obj).obj_type {
                ObjType::String => {
                    let s = obj as *mut ObjString;
                    ptr::drop_in_place(&mut (*s).chars);
                    std::mem::size_of::<ObjString>()
                }
                ObjType::Class => {
                    let c = obj as *mut ObjClass;
                    ptr::drop_in_place(&mut (*c).methods);
                    std::mem::size_of::<ObjClass>()
                }
                ObjType::Instance => {
                    let i = obj as *mut ObjInstance;
                    ptr::drop_in_place(&mut (*i).fields);
                    std::mem::size_of::<ObjInstance>()
                }
                ObjType::Native => std::mem::size_of::<ObjNative>(),
                ObjType::BoundMethod => std::mem::size_of::<ObjBoundMethod>(),
                _ => 8,
            };
            let layout = std::alloc::Layout::from_size_align_unchecked(size, 8);
            std::alloc::dealloc(obj as *mut u8, layout);
            self.bytes_allocated = self.bytes_allocated.saturating_sub(size);
        }
    }
}

impl Drop for GcHeap {
    fn drop(&mut self) {
        self.free_objects();
    }
}

pub fn hash_string(s: &str) -> u32 {
    let mut hash: u32 = 2166136261;
    for byte in s.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}
