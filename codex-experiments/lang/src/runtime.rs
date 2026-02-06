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

static mut ARGC: i32 = 0;
static mut ARGV: *mut *mut u8 = std::ptr::null_mut();

#[no_mangle]
pub extern "C" fn rt_init_args(argc: i32, argv: *mut *mut u8) {
    unsafe {
        ARGC = argc;
        ARGV = argv;
    }
}

#[no_mangle]
pub extern "C" fn arg_i64(index: i64) -> i64 {
    if index < 0 {
        return 0;
    }
    unsafe {
        if ARGV.is_null() {
            return 0;
        }
        let arg_index = index + 1;
        if arg_index >= ARGC as i64 {
            return 0;
        }
        let ptr = *ARGV.add(arg_index as usize) as *const i8;
        if ptr.is_null() {
            return 0;
        }
        let cstr = std::ffi::CStr::from_ptr(ptr);
        match cstr.to_str() {
            Ok(s) => s.parse::<i64>().unwrap_or(0),
            Err(_) => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn arg_str(index: i64) -> *const u8 {
    if index < 0 {
        return std::ptr::null();
    }
    unsafe {
        if ARGV.is_null() {
            return std::ptr::null();
        }
        let arg_index = index + 1;
        if arg_index >= ARGC as i64 {
            return std::ptr::null();
        }
        *ARGV.add(arg_index as usize) as *const u8
    }
}

#[no_mangle]
pub extern "C" fn print_int(v: i64) -> i64 {
    println!("{v}");
    0
}

#[no_mangle]
pub extern "C" fn print_str(ptr: *const u8) -> i64 {
    if ptr.is_null() {
        println!();
        return 0;
    }
    unsafe {
        let mut len = 0usize;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let slice = std::slice::from_raw_parts(ptr, len);
        if let Ok(s) = std::str::from_utf8(slice) {
            println!("{s}");
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn add_i64(a: i64, b: i64) -> i64 {
    a + b
}

#[no_mangle]
pub extern "C" fn null_ptr() -> *mut u8 {
    std::ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn ptr_is_null(p: *mut u8) -> i64 {
    if p.is_null() { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn string_len(ptr: *const u8) -> i64 {
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        let mut len = 0usize;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        len as i64
    }
}

#[no_mangle]
pub extern "C" fn string_eq(a: *const u8, b: *const u8) -> i64 {
    if a.is_null() || b.is_null() {
        return if a == b { 1 } else { 0 };
    }
    unsafe {
        let mut i = 0usize;
        loop {
            let ca = *a.add(i);
            let cb = *b.add(i);
            if ca != cb {
                return 0;
            }
            if ca == 0 {
                return 1;
            }
            i += 1;
        }
    }
}

#[no_mangle]
pub extern "C" fn string_concat(a: *const u8, b: *const u8) -> *const u8 {
    if a.is_null() && b.is_null() {
        return std::ptr::null();
    }
    unsafe {
        let mut bytes: Vec<u8> = Vec::new();
        if !a.is_null() {
            let mut i = 0usize;
            while *a.add(i) != 0 {
                bytes.push(*a.add(i));
                i += 1;
            }
        }
        if !b.is_null() {
            let mut i = 0usize;
            while *b.add(i) != 0 {
                bytes.push(*b.add(i));
                i += 1;
            }
        }
        bytes.push(0);
        let boxed = bytes.into_boxed_slice();
        Box::into_raw(boxed) as *const u8
    }
}

#[no_mangle]
pub extern "C" fn string_slice(ptr: *const u8, start: i64, end_pos: i64) -> *const u8 {
    if ptr.is_null() {
        return std::ptr::null();
    }
    if start < 0 || end_pos < start {
        return std::ptr::null();
    }
    unsafe {
        let mut len = 0usize;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let start_usize = start as usize;
        let end_usize = end_pos as usize;
        if start_usize > len || end_usize > len {
            return std::ptr::null();
        }
        let slice = std::slice::from_raw_parts(ptr.add(start_usize), end_usize - start_usize);
        let mut bytes = Vec::with_capacity(slice.len() + 1);
        bytes.extend_from_slice(slice);
        bytes.push(0);
        let boxed = bytes.into_boxed_slice();
        Box::into_raw(boxed) as *const u8
    }
}

#[no_mangle]
pub extern "C" fn string_byte_at(ptr: *const u8, index: i64) -> i64 {
    if ptr.is_null() || index < 0 {
        return 0;
    }
    unsafe {
        let mut i = 0i64;
        while *ptr.add(i as usize) != 0 {
            if i == index {
                return *ptr.add(i as usize) as i64;
            }
            i += 1;
        }
        0
    }
}

#[no_mangle]
pub extern "C" fn read_file(path: *const u8) -> *const u8 {
    if path.is_null() {
        return std::ptr::null();
    }
    unsafe {
        let mut len = 0usize;
        while *path.add(len) != 0 {
            len += 1;
        }
        let slice = std::slice::from_raw_parts(path, len);
        let path_str = match std::str::from_utf8(slice) {
            Ok(s) => s,
            Err(_) => return std::ptr::null(),
        };
        match std::fs::read(path_str) {
            Ok(mut bytes) => {
                bytes.push(0);
                let boxed = bytes.into_boxed_slice();
                Box::into_raw(boxed) as *const u8
            }
            Err(_) => std::ptr::null(),
        }
    }
}

#[repr(C)]
struct VecPtr {
    len: i64,
    cap: i64,
    data: *mut *mut u8,
}

#[no_mangle]
pub extern "C" fn vec_new() -> *mut u8 {
    let v = VecPtr {
        len: 0,
        cap: 0,
        data: std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(v)) as *mut u8
}

#[no_mangle]
pub extern "C" fn vec_len(vec: *mut u8) -> i64 {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*(vec as *mut VecPtr)).len }
}

#[no_mangle]
pub extern "C" fn vec_get(vec: *mut u8, index: i64) -> *mut u8 {
    if vec.is_null() || index < 0 {
        return std::ptr::null_mut();
    }
    unsafe {
        let v = &mut *(vec as *mut VecPtr);
        if index >= v.len {
            return std::ptr::null_mut();
        }
        if v.data.is_null() {
            return std::ptr::null_mut();
        }
        *v.data.add(index as usize)
    }
}

#[no_mangle]
pub extern "C" fn vec_push(vec: *mut u8, item: *mut u8) -> i64 {
    if vec.is_null() {
        return 0;
    }
    unsafe {
        let v = &mut *(vec as *mut VecPtr);
        let len = v.len as usize;
        let cap = v.cap as usize;
        let mut buf: Vec<*mut u8> = if v.data.is_null() || cap == 0 {
            Vec::new()
        } else {
            Vec::from_raw_parts(v.data, len, cap)
        };
        buf.push(item);
        v.len = buf.len() as i64;
        v.cap = buf.capacity() as i64;
        v.data = buf.as_mut_ptr();
        std::mem::forget(buf);
        v.len
    }
}

#[no_mangle]
pub extern "C" fn print_stretch(depth: i64, check: i64) {
    println!("stretch tree of depth {depth}\t check: {check}");
}

#[no_mangle]
pub extern "C" fn print_trees(iterations: i64, depth: i64, check: i64) {
    println!("{iterations}\t trees of depth {depth}\t check: {check}");
}

#[no_mangle]
pub extern "C" fn print_long_lived(depth: i64, check: i64) {
    println!("long lived tree of depth {depth}\t check: {check}");
}

#[no_mangle]
pub extern "C" fn gc_pollcheck_slow(thread: *mut Thread, _origin: *const FrameOrigin) {
    unsafe {
        if let Some(t) = thread.as_mut() {
            t.state = 0;
            let mut frame = t.top_frame;
            while !frame.is_null() {
                let origin = (*frame).origin;
                let num_roots = if origin.is_null() { 0 } else { (*origin).num_roots as isize };
                let roots_base = (frame as *mut u8).add(std::mem::size_of::<FrameHeader>()) as *mut *mut u8;
                for i in 0..num_roots {
                    let _slot = roots_base.offset(i);
                    // Placeholder for GC root processing.
                }
                frame = (*frame).parent;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn gc_allocate(_thread: *mut Thread, _meta: *mut u8, size: i64) -> *mut u8 {
    unsafe {
        let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
        std::alloc::alloc_zeroed(layout)
    }
}

#[no_mangle]
pub extern "C" fn gc_allocate_array(_thread: *mut Thread, _meta: *mut u8, length: i64) -> *mut u8 {
    unsafe {
        let bytes = length as usize;
        let layout = std::alloc::Layout::from_size_align(bytes, 8).unwrap();
        std::alloc::alloc_zeroed(layout)
    }
}

#[no_mangle]
pub extern "C" fn gc_write_barrier(_thread: *mut Thread, _obj: *mut u8, _value: *mut u8) {}
