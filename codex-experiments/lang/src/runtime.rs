use gc_library::gc::generational::GenerationalGC;
use gc_library::gc::{AllocateAction, Allocator, AllocatorOptions, LibcMemoryProvider};
use gc_library::traits::{ForwardingSupport, GcObject, GcTypes, ObjectKind, RootProvider, TaggedPointer};

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

// =============================================================================
// GC Library Trait Implementations
// =============================================================================

/// TypeInfo layout as emitted by codegen's create_type_info:
///   offset 0:  kind (i32)
///   offset 4:  pad (i32)
///   offset 8:  size (i64)      <- total object size including 16-byte header
///   offset 16: num_ptrs (i32)  <- number of pointer fields
///   offset 20: pad (i32)
///   offset 24: name (ptr)
///   offset 32: ptr_offsets (ptr)
#[repr(C)]
struct TypeInfo {
    kind: i32,
    _pad0: i32,
    size: i64,
    num_ptrs: i32,
    _pad1: i32,
    name: *const u8,
    ptr_offsets: *const i64,
}

/// Object header layout (16 bytes, matches codegen's object_header_ty):
///   offset 0:  meta (ptr)      <- pointer to TypeInfo
///   offset 8:  gc_flags (u32)  <- bit 0 = mark bit
///   offset 12: aux (u32)       <- reserved
#[repr(C)]
struct ObjHeader {
    meta: *const TypeInfo,
    gc_flags: u32,
    aux: u32,
}

const MARK_BIT: u32 = 1;
const FORWARDING_BIT: u32 = 1 << 2;
const HEADER_SIZE: usize = 16;
/// MarkAndSweep internally uses 8-byte headers. Our objects use 16-byte headers.
/// This constant is the MarkAndSweep internal header size.
const MS_HEADER_SIZE: usize = 8;

/// Compute the number of words to request from MarkAndSweep for a given total_size.
/// Also returns the actual allocation size (which must match full_size).
fn alloc_words_for_size(total_size: usize) -> (usize, usize) {
    let payload = total_size.saturating_sub(HEADER_SIZE);
    let words = (payload + 7) / 8 + 1;
    let actual = words * 8 + MS_HEADER_SIZE;
    (words, actual)
}

// -- ObjectKind --

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LangTypeTag {
    HeapObject = 0,
    Null = 7,
}

impl ObjectKind for LangTypeTag {
    fn is_heap_type(self) -> bool {
        matches!(self, LangTypeTag::HeapObject)
    }
}

// -- TaggedPointer (untagged raw pointers) --

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LangTaggedPtr(usize);

impl TaggedPointer for LangTaggedPtr {
    type Kind = LangTypeTag;

    fn tag(raw_ptr: *const u8, _kind: LangTypeTag) -> Self {
        LangTaggedPtr(raw_ptr as usize)
    }

    fn untag(self) -> *const u8 {
        self.0 as *const u8
    }

    fn get_kind(self) -> LangTypeTag {
        if self.0 == 0 {
            LangTypeTag::Null
        } else {
            LangTypeTag::HeapObject
        }
    }

    fn is_heap_pointer(self) -> bool {
        self.0 != 0
    }

    fn as_usize(self) -> usize {
        self.0
    }

    fn from_usize(value: usize) -> Self {
        LangTaggedPtr(value)
    }
}

// -- GcObject --

pub struct LangObject {
    ptr: *const u8,
}

impl LangObject {
    fn header(&self) -> &ObjHeader {
        unsafe { &*(self.ptr as *const ObjHeader) }
    }

    fn header_mut(&self) -> &mut ObjHeader {
        unsafe { &mut *(self.ptr as *mut ObjHeader) }
    }

    fn fields_ptr(&self) -> *mut usize {
        unsafe { (self.ptr as *mut u8).add(HEADER_SIZE) as *mut usize }
    }

    /// Read num_ptrs from the TypeInfo metadata.
    /// Returns 0 if meta is null (object not yet initialized by codegen).
    fn num_ptr_fields(&self) -> usize {
        let hdr = self.header();
        if hdr.meta.is_null() {
            return 0;
        }
        unsafe { (*hdr.meta).num_ptrs as usize }
    }

    /// Read total object size from TypeInfo metadata.
    /// Falls back to just header size if meta is null.
    fn total_size_from_meta(&self) -> usize {
        let hdr = self.header();
        if hdr.meta.is_null() {
            return HEADER_SIZE;
        }
        unsafe { (*hdr.meta).size as usize }
    }
}

impl GcObject for LangObject {
    type TaggedValue = LangTaggedPtr;

    fn from_tagged(tagged: LangTaggedPtr) -> Self {
        LangObject {
            ptr: tagged.untag(),
        }
    }

    fn from_untagged(ptr: *const u8) -> Self {
        LangObject { ptr }
    }

    fn get_pointer(&self) -> *const u8 {
        self.ptr
    }

    fn tagged_pointer(&self) -> LangTaggedPtr {
        LangTaggedPtr::tag(self.ptr, LangTypeTag::HeapObject)
    }

    fn mark(&self) {
        let hdr = self.header_mut();
        hdr.gc_flags |= MARK_BIT;
    }

    fn unmark(&self) {
        let hdr = self.header_mut();
        hdr.gc_flags &= !MARK_BIT;
    }

    fn marked(&self) -> bool {
        (self.header().gc_flags & MARK_BIT) != 0
    }

    fn get_fields(&self) -> &[usize] {
        let n = self.num_ptr_fields();
        if n == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.fields_ptr(), n) }
    }

    fn get_fields_mut(&mut self) -> &mut [usize] {
        let n = self.num_ptr_fields();
        if n == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.fields_ptr(), n) }
    }

    fn is_opaque(&self) -> bool {
        self.num_ptr_fields() == 0
    }

    fn is_zero_size(&self) -> bool {
        self.total_size_from_meta() <= HEADER_SIZE
    }

    fn get_object_kind(&self) -> Option<LangTypeTag> {
        Some(LangTypeTag::HeapObject)
    }

    fn full_size(&self) -> usize {
        let type_size = self.total_size_from_meta();
        let (_words, actual) = alloc_words_for_size(type_size);
        actual
    }

    fn header_size(&self) -> usize {
        HEADER_SIZE
    }

    fn get_full_object_data(&self) -> &[u8] {
        let size = self.full_size();
        unsafe { std::slice::from_raw_parts(self.ptr, size) }
    }

    fn write_header(&mut self, _field_size_bytes: usize) {
        // Zero the 16-byte header. Codegen's init_header will overwrite
        // with the real meta pointer and zeroed gc_flags/aux.
        let hdr = self.header_mut();
        hdr.meta = std::ptr::null();
        hdr.gc_flags = 0;
        hdr.aux = 0;
    }
}

// -- ForwardingSupport (required for generational GC) --

impl ForwardingSupport for LangObject {
    fn is_forwarded(&self) -> bool {
        (self.header().gc_flags & FORWARDING_BIT) != 0
    }

    fn get_forwarding_pointer(&self) -> LangTaggedPtr {
        debug_assert!(self.is_forwarded(), "object is not forwarded");
        // When forwarded, the meta field stores the new location
        LangTaggedPtr(self.header().meta as usize)
    }

    fn set_forwarding_pointer(&mut self, new_location: LangTaggedPtr) {
        let hdr = self.header_mut();
        hdr.meta = new_location.0 as *const TypeInfo;
        hdr.gc_flags |= FORWARDING_BIT;
    }
}

// -- GcTypes bundle --

pub struct LangRuntime;

impl GcTypes for LangRuntime {
    type TaggedValue = LangTaggedPtr;
    type ObjectHandle = LangObject;
    type ObjectKind = LangTypeTag;
}

// -- RootProvider: walk the Thread's frame chain --

struct FrameChainRoots {
    thread: *const Thread,
}

impl RootProvider<LangTaggedPtr> for FrameChainRoots {
    fn enumerate_roots(&self, callback: &mut dyn FnMut(usize, LangTaggedPtr)) {
        unsafe {
            let thread = &*self.thread;
            let mut frame = thread.top_frame;
            while !frame.is_null() {
                let origin = (*frame).origin;
                let num_roots = if origin.is_null() {
                    0
                } else {
                    (*origin).num_roots as usize
                };
                let roots_base = (frame as *mut u8).add(std::mem::size_of::<FrameHeader>())
                    as *mut *mut u8;
                for i in 0..num_roots {
                    let slot = roots_base.add(i);
                    let val = *slot;
                    if !val.is_null() {
                        let tagged = LangTaggedPtr(val as usize);
                        callback(slot as usize, tagged);
                    }
                }
                frame = (*frame).parent;
            }
        }
    }
}

// =============================================================================
// Global GC State
// =============================================================================

type LangGc = GenerationalGC<LangRuntime, LibcMemoryProvider>;

static mut GC_PTR: *mut LangGc = std::ptr::null_mut();

#[no_mangle]
pub extern "C" fn gc_init() {
    let options = AllocatorOptions {
        gc: true,
        print_stats: false,
        gc_always: false,
    };
    let memory = LibcMemoryProvider::new();
    let gc = LangGc::new(options, memory);
    let boxed = Box::new(gc);
    unsafe {
        GC_PTR = Box::into_raw(boxed);
    }
}

#[inline(always)]
fn with_gc<F, R>(f: F) -> R
where
    F: FnOnce(&mut LangGc) -> R,
{
    unsafe {
        debug_assert!(!GC_PTR.is_null(), "gc_init() must be called before allocation");
        f(&mut *GC_PTR)
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
pub extern "C" fn arg_is_i64(index: i64) -> i64 {
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
            Ok(s) => if s.parse::<i64>().is_ok() { 1 } else { 0 },
            Err(_) => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn arg_len() -> i64 {
    unsafe {
        if ARGV.is_null() {
            return 0;
        }
        if ARGC <= 1 {
            return 0;
        }
        (ARGC - 1) as i64
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
        }
    }
    let roots = FrameChainRoots { thread };
    with_gc(|gc| gc.gc(&roots));
}

#[no_mangle]
pub extern "C" fn gc_allocate(thread: *mut Thread, _meta: *mut u8, size: i64) -> *mut u8 {
    // MarkAndSweep internally adds an 8-byte header. Our objects use 16-byte
    // headers, so we request one extra word (8 bytes) to make the total
    // allocation = size bytes.
    //
    // `size` is the total object size including our 16-byte header.
    // payload = size - 16 (the data after our header)
    // We need: words = payload/8 + 1
    //   => allocator carves words*8 + 8 = payload + 16 = size bytes  ✓
    let (words, actual_size) = alloc_words_for_size(size as usize);

    loop {
        let result = with_gc(|gc| gc.try_allocate(words, LangTypeTag::HeapObject));
        match result {
            Ok(AllocateAction::Allocated(ptr)) => {
                // Zero the entire allocation (actual_size bytes)
                unsafe {
                    std::ptr::write_bytes(ptr as *mut u8, 0, actual_size);
                }
                return ptr as *mut u8;
            }
            Ok(AllocateAction::Gc) => {
                // Run GC and retry
                let roots = FrameChainRoots { thread };
                with_gc(|gc| gc.gc(&roots));
                // After GC, try growing if still no space
                let retry = with_gc(|gc| gc.try_allocate(words, LangTypeTag::HeapObject));
                match retry {
                    Ok(AllocateAction::Allocated(ptr)) => {
                        unsafe {
                            std::ptr::write_bytes(ptr as *mut u8, 0, actual_size);
                        }
                        return ptr as *mut u8;
                    }
                    Ok(AllocateAction::Gc) => {
                        // Grow and retry once more
                        with_gc(|gc| gc.grow());
                        let final_try =
                            with_gc(|gc| gc.try_allocate(words, LangTypeTag::HeapObject));
                        match final_try {
                            Ok(AllocateAction::Allocated(ptr)) => {
                                unsafe {
                                    std::ptr::write_bytes(ptr as *mut u8, 0, actual_size);
                                }
                                return ptr as *mut u8;
                            }
                            _ => panic!("gc_allocate: out of memory after GC and grow"),
                        }
                    }
                    Err(e) => panic!("gc_allocate: allocation error: {:?}", e),
                }
            }
            Err(e) => panic!("gc_allocate: allocation error: {:?}", e),
        }
    }
}

#[no_mangle]
pub extern "C" fn gc_allocate_array(thread: *mut Thread, meta: *mut u8, length: i64) -> *mut u8 {
    // Arrays use the same allocation path — length is total byte size
    gc_allocate(thread, meta, length)
}

#[no_mangle]
pub extern "C" fn gc_write_barrier(_thread: *mut Thread, obj: *mut u8, value: *mut u8) {
    with_gc(|gc| gc.write_barrier(obj as usize, value as usize));
}

#[no_mangle]
pub extern "C" fn exit_process(code: i64) -> i64 {
    std::process::exit(code as i32);
}

#[no_mangle]
pub extern "C" fn string_from_i64(val: i64) -> *const u8 {
    let s = format!("{val}");
    let mut bytes = s.into_bytes();
    bytes.push(0);
    let boxed = bytes.into_boxed_slice();
    Box::into_raw(boxed) as *const u8
}

#[no_mangle]
pub extern "C" fn print_str_stderr(ptr: *const u8) -> i64 {
    if ptr.is_null() {
        eprintln!();
        return 0;
    }
    unsafe {
        let mut len = 0usize;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let slice = std::slice::from_raw_parts(ptr, len);
        if let Ok(s) = std::str::from_utf8(slice) {
            eprintln!("{s}");
        }
    }
    0
}

// Vec aliases — typed wrappers that forward to vec_push/vec_get.
// Each extern fn must have one signature in the lang type system,
// so we need named aliases for different element types.

#[no_mangle]
pub extern "C" fn vec_push_str(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_str(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_item(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_item(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_param(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_param(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_field(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_field(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_variant(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_variant(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_type(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_type(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_expr(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_expr(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_arm(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_arm(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_stmt(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_stmt(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_pfield(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_pfield(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_slfield(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_slfield(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_get_tok(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_ientry(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_ientry(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_uentry(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_uentry(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_ventry(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_ventry(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_vec(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_vec(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

#[no_mangle]
pub extern "C" fn vec_push_module(vec: *mut u8, item: *mut u8) -> i64 { vec_push(vec, item) }
#[no_mangle]
pub extern "C" fn vec_get_module(vec: *mut u8, index: i64) -> *mut u8 { vec_get(vec, index) }

// Codegen pass vec aliases — the .lang type checker needs distinct extern fn
// names for each element type, but at the ABI level all struct types are
// pointers so these all forward to the same vec_push/vec_get.
#[no_mangle] pub extern "C" fn vec_push_cgfn(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_cgfn(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_cgsl(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_cgsl(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_cgel(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_cgel(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_cgfl(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_cgfl(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_cgvfl(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_cgvfl(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_cgvl(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_cgvl(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_cgloc(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_cgloc(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_loop(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_loop(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }

// Typecheck pass vec aliases
#[no_mangle] pub extern "C" fn vec_push_ty(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_ty(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_smentry(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_smentry(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_ementry(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_ementry(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_fnentry(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_fnentry(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_vdentry(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_vdentry(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_subst(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_subst(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_local(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_local(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }
#[no_mangle] pub extern "C" fn vec_push_tfield(v: *mut u8, i: *mut u8) -> i64 { vec_push(v, i) }
#[no_mangle] pub extern "C" fn vec_get_tfield(v: *mut u8, i: i64) -> *mut u8 { vec_get(v, i) }

/// Returns the raw data pointer of a vec (contiguous array of *mut u8).
/// Used to pass vec contents to LLVM C API functions that expect arrays.
#[no_mangle]
pub extern "C" fn vec_data(vec: *mut u8) -> *mut u8 {
    if vec.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*(vec as *mut VecPtr)).data as *mut u8 }
}

#[no_mangle]
pub extern "C" fn vec_clear(vec: *mut u8) -> i64 {
    if vec.is_null() {
        return 0;
    }
    unsafe {
        let v = &mut *(vec as *mut VecPtr);
        v.len = 0;
    }
    0
}

#[no_mangle]
pub extern "C" fn vec_set_len(vec: *mut u8, new_len: i64) -> i64 {
    if vec.is_null() {
        return 0;
    }
    unsafe {
        let v = &mut *(vec as *mut VecPtr);
        if new_len < 0 {
            v.len = 0;
        } else if new_len < v.len {
            v.len = new_len;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn string_parse_i64(ptr: *const u8) -> i64 {
    if ptr.is_null() {
        return -1;
    }
    unsafe {
        let mut len = 0usize;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let slice = std::slice::from_raw_parts(ptr, len);
        match std::str::from_utf8(slice) {
            Ok(s) => s.parse::<i64>().unwrap_or(-1),
            Err(_) => -1,
        }
    }
}

#[no_mangle]
pub extern "C" fn write_file(path: *const u8, data: *const u8, len: i64) -> i64 {
    if path.is_null() || data.is_null() {
        return 1;
    }
    unsafe {
        let mut path_len = 0usize;
        while *path.add(path_len) != 0 {
            path_len += 1;
        }
        let path_slice = std::slice::from_raw_parts(path, path_len);
        let path_str = match std::str::from_utf8(path_slice) {
            Ok(s) => s,
            Err(_) => return 1,
        };
        let data_slice = std::slice::from_raw_parts(data, len as usize);
        match std::fs::write(path_str, data_slice) {
            Ok(_) => 0,
            Err(_) => 1,
        }
    }
}

#[no_mangle]
pub extern "C" fn system_cmd(cmd: *const u8) -> i64 {
    if cmd.is_null() {
        return 1;
    }
    unsafe {
        let mut len = 0usize;
        while *cmd.add(len) != 0 {
            len += 1;
        }
        let slice = std::slice::from_raw_parts(cmd, len);
        let cmd_str = match std::str::from_utf8(slice) {
            Ok(s) => s,
            Err(_) => return 1,
        };
        match std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd_str)
            .status()
        {
            Ok(status) => {
                if status.success() { 0 } else { status.code().unwrap_or(1) as i64 }
            }
            Err(_) => 1,
        }
    }
}

/// Read the i32 enum tag from a GC-allocated enum object.
/// `raw_base` is the byte offset from the object start to the raw data area
/// (where the tag is stored as the first i32).
#[no_mangle]
pub extern "C" fn enum_tag(obj: *mut u8, raw_base: i64) -> i64 {
    if obj.is_null() {
        return -1;
    }
    unsafe {
        let tag_ptr = obj.add(raw_base as usize) as *const i32;
        *tag_ptr as i64
    }
}
