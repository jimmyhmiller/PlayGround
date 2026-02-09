//! Runtime declarations for C functions in runtime/runtime.c and runtime/gc_bridge.c.
//! All implementations are in C; this module only provides extern declarations
//! and the Thread struct for Rust-side usage.

#[repr(C)]
pub struct FrameHeader {
    pub parent: *mut FrameHeader,
    pub origin: *const FrameOrigin,
}

#[repr(C)]
pub struct FrameOrigin {
    pub num_roots: i32,
    pub name: *const u8,
}

#[repr(C)]
pub struct Thread {
    pub top_frame: *mut FrameHeader,
    pub state: i32,
    pub _pad: i32,
}

impl Thread {
    pub fn new() -> Self {
        Self {
            top_frame: std::ptr::null_mut(),
            state: 0,
            _pad: 0,
        }
    }
}

// GC functions (gc_bridge.c)
extern "C" {
    pub fn gc_init();
    pub fn gc_set_thread(thread: *mut u8);
    pub fn gc_alloc(total_fields: i64, ptr_fields: i64, type_id: i64) -> *mut u8;
    pub fn gc_read_field(obj: *mut u8, index: i64) -> *mut u8;
    pub fn gc_write_field(obj: *mut u8, index: i64, val: *mut u8) -> i64;
    pub fn gc_read_field_i64(obj: *mut u8, index: i64) -> i64;
    pub fn gc_write_field_i64(obj: *mut u8, index: i64, val: i64) -> i64;
    pub fn gc_write_barrier(thread: *mut u8, obj: *mut u8, val: *mut u8);
    pub fn gc_pollcheck_slow(thread: *mut u8, origin: *mut u8);
}

// Runtime functions (runtime.c)
extern "C" {
    // Pointer intrinsics
    pub fn ptr_load_ptr(p: *mut u8, off: i64) -> *mut u8;
    pub fn ptr_store_ptr(p: *mut u8, off: i64, val: *mut u8) -> i64;
    pub fn ptr_load_i64(p: *mut u8, off: i64) -> i64;
    pub fn ptr_store_i64(p: *mut u8, off: i64, val: i64) -> i64;
    pub fn ptr_load_i32(p: *mut u8, off: i64) -> i64;
    pub fn ptr_store_i32(p: *mut u8, off: i64, val: i64) -> i64;
    pub fn ptr_load_i8(p: *mut u8, off: i64) -> i64;
    pub fn ptr_store_i8(p: *mut u8, off: i64, val: i64) -> i64;
    pub fn ptr_offset(p: *mut u8, off: i64) -> *mut u8;

    // String operations
    pub fn string_slice(ptr: *const u8, start: i64, end_pos: i64) -> *const u8;

    // I/O and misc
    pub fn write_file(path: *const u8, data: *const u8, len: i64) -> i64;
    pub fn system_cmd(cmd: *const u8) -> i64;
    pub fn file_exists(path: *const u8) -> i64;
    pub fn get_stdlib_path() -> *const u8;
}
