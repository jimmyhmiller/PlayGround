mod ast;
mod codegen;
mod runtime;

use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{InitializationConfig, Target, TargetMachine, RelocMode, CodeModel};
use inkwell::OptimizationLevel;
use std::ffi::c_void;
use std::cell::RefCell;

use ast::{Expr, FunctionDef, Parameter, Program, Stmt, StructDef, StructField, Type};
use codegen::Compiler;
use runtime::Thread;

// Import gc-library
use gc_library::gc::{Allocator, AllocatorOptions, AllocateAction};
use gc_library::gc::generational::GenerationalGC;
use gc_library::traits::{GcTypes, GcObject, TaggedPointer, ObjectKind, RootProvider, ForwardingSupport};

// =============================================================================
// GC Integration Types
// =============================================================================

/// Our type tag - we use untagged aligned pointers for heap objects
/// Since raw pointers are 8-byte aligned (tag bits = 0), we use:
/// - Tag 0 + non-zero value = HeapObject (raw aligned pointer)
/// - Value 0 = Null
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum OurTypeTag {
    /// Heap object (raw 8-byte aligned pointer, tag bits = 0)
    HeapObject = 0,
    /// Null value (the entire value is 0)
    Null = 7,
}

impl OurTypeTag {
    fn from_value(value: usize) -> Self {
        // Null is represented as value 0
        // Everything else with aligned pointer (tag bits = 0) is a heap object
        if value == 0 {
            OurTypeTag::Null
        } else {
            OurTypeTag::HeapObject
        }
    }
}

impl ObjectKind for OurTypeTag {
    fn is_heap_type(self) -> bool {
        matches!(self, OurTypeTag::HeapObject)
    }
}

/// Tagged pointer - but we actually use UNTAGGED pointers since our codegen
/// stores raw pointers. Null is represented as value 0.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OurTaggedPtr(usize);

impl OurTaggedPtr {
    pub fn null() -> Self {
        OurTaggedPtr(0)
    }

    pub fn from_raw_ptr(ptr: *const u8) -> Self {
        // Raw pointers are already in the right format (untagged)
        OurTaggedPtr(ptr as usize)
    }
}

impl TaggedPointer for OurTaggedPtr {
    type Kind = OurTypeTag;

    fn tag(raw_ptr: *const u8, _kind: OurTypeTag) -> Self {
        // We don't actually tag pointers - store raw
        OurTaggedPtr(raw_ptr as usize)
    }

    fn untag(self) -> *const u8 {
        // No tag bits to clear for our scheme
        self.0 as *const u8
    }

    fn get_kind(self) -> OurTypeTag {
        OurTypeTag::from_value(self.0)
    }

    fn is_heap_pointer(self) -> bool {
        // Non-null values are heap pointers
        self.0 != 0
    }

    fn as_usize(self) -> usize {
        self.0
    }

    fn from_usize(value: usize) -> Self {
        OurTaggedPtr(value)
    }
}

/// Header format (16 bytes) - matches the codegen's ObjectHeader:
/// { meta: ptr (8 bytes), gc_flags: i32 (4 bytes), aux: i32 (4 bytes) }
///
/// We pack mark/opaque/forwarding bits and size into gc_flags
/// Bits 0:     Marked bit
/// Bits 1:     Opaque bit (no pointer fields)
/// Bits 2:     Forwarding bit
/// Bits 16-31: Size in words (u16) - number of pointer fields
#[derive(Debug, Copy, Clone)]
pub struct OurHeader {
    pub marked: bool,
    pub opaque: bool,
    pub forwarding: bool,
    pub size_words: u16,
}

/// Raw header layout in memory (16 bytes total)
#[repr(C)]
struct RawHeader {
    meta: usize,     // 8 bytes - pointer to type metadata
    gc_flags: u32,   // 4 bytes - mark bit, etc.
    aux: u32,        // 4 bytes - reserved
}

impl OurHeader {
    const MARKED_BIT: u32 = 1 << 0;
    const OPAQUE_BIT: u32 = 1 << 1;
    const FORWARDING_BIT: u32 = 1 << 2;
    const SIZE_SHIFT: u32 = 16;
    const SIZE_MASK: u32 = 0xFFFF;

    pub fn from_raw(raw: &RawHeader) -> Self {
        OurHeader {
            marked: (raw.gc_flags & Self::MARKED_BIT) != 0,
            opaque: (raw.gc_flags & Self::OPAQUE_BIT) != 0,
            forwarding: (raw.gc_flags & Self::FORWARDING_BIT) != 0,
            size_words: ((raw.gc_flags >> Self::SIZE_SHIFT) & Self::SIZE_MASK) as u16,
        }
    }

    pub fn to_gc_flags(self) -> u32 {
        let mut flags = 0u32;
        if self.marked {
            flags |= Self::MARKED_BIT;
        }
        if self.opaque {
            flags |= Self::OPAQUE_BIT;
        }
        if self.forwarding {
            flags |= Self::FORWARDING_BIT;
        }
        flags |= (self.size_words as u32) << Self::SIZE_SHIFT;
        flags
    }

    pub fn new(size_words: u16) -> Self {
        OurHeader {
            marked: false,
            opaque: false,
            forwarding: false,
            size_words,
        }
    }
}

/// Heap object handle
pub struct OurObject {
    ptr: *const u8,
}

impl OurObject {
    /// Get a reference to the raw header at the start of the object
    fn raw_header(&self) -> &RawHeader {
        unsafe { &*(self.ptr as *const RawHeader) }
    }

    /// Get a mutable reference to the raw header
    fn raw_header_mut(&self) -> &mut RawHeader {
        unsafe { &mut *(self.ptr as *mut RawHeader) }
    }

    fn read_header(&self) -> OurHeader {
        OurHeader::from_raw(self.raw_header())
    }

    fn write_header_raw(&self, header: OurHeader) {
        let raw = self.raw_header_mut();
        raw.gc_flags = header.to_gc_flags();
    }

    fn fields_ptr(&self) -> *mut usize {
        // Fields start after the 16-byte header
        unsafe { (self.ptr as *mut u8).add(16) as *mut usize }
    }
}

impl GcObject for OurObject {
    type TaggedValue = OurTaggedPtr;

    fn from_tagged(tagged: OurTaggedPtr) -> Self {
        debug_assert!(
            tagged.is_heap_pointer(),
            "expected heap pointer, got {:?}",
            tagged.get_kind()
        );
        OurObject {
            ptr: tagged.untag(),
        }
    }

    fn from_untagged(ptr: *const u8) -> Self {
        OurObject { ptr }
    }

    fn get_pointer(&self) -> *const u8 {
        self.ptr
    }

    fn tagged_pointer(&self) -> OurTaggedPtr {
        OurTaggedPtr::tag(self.ptr, OurTypeTag::HeapObject)
    }

    fn mark(&self) {
        let mut header = self.read_header();
        header.marked = true;
        self.write_header_raw(header);
    }

    fn unmark(&self) {
        let mut header = self.read_header();
        header.marked = false;
        self.write_header_raw(header);
    }

    fn marked(&self) -> bool {
        self.read_header().marked
    }

    fn get_fields(&self) -> &[usize] {
        let header = self.read_header();
        if header.opaque {
            return &[];
        }
        let count = header.size_words as usize;
        unsafe { std::slice::from_raw_parts(self.fields_ptr(), count) }
    }

    fn get_fields_mut(&mut self) -> &mut [usize] {
        let header = self.read_header();
        if header.opaque {
            return &mut [];
        }
        let count = header.size_words as usize;
        unsafe { std::slice::from_raw_parts_mut(self.fields_ptr(), count) }
    }

    fn is_opaque(&self) -> bool {
        self.read_header().opaque
    }

    fn is_zero_size(&self) -> bool {
        self.read_header().size_words == 0
    }

    fn get_object_kind(&self) -> Option<OurTypeTag> {
        Some(OurTypeTag::HeapObject)
    }

    fn full_size(&self) -> usize {
        self.header_size() + (self.read_header().size_words as usize * 8)
    }

    fn header_size(&self) -> usize {
        16 // 16 bytes: { meta: ptr, gc_flags: i32, aux: i32 }
    }

    fn get_full_object_data(&self) -> &[u8] {
        let size = self.full_size();
        unsafe { std::slice::from_raw_parts(self.ptr, size) }
    }

    fn write_header(&mut self, field_size_bytes: usize) {
        let size_words = field_size_bytes / 8;
        let header = OurHeader::new(size_words as u16);
        self.write_header_raw(header);
    }
}

impl ForwardingSupport for OurObject {
    fn is_forwarded(&self) -> bool {
        // Check the forwarding bit in gc_flags
        let raw = self.raw_header();
        (raw.gc_flags & OurHeader::FORWARDING_BIT) != 0
    }

    fn get_forwarding_pointer(&self) -> OurTaggedPtr {
        debug_assert!(self.is_forwarded(), "object is not forwarded");
        // When forwarded, the meta field contains the new location
        let raw = self.raw_header();
        OurTaggedPtr::from_usize(raw.meta)
    }

    fn set_forwarding_pointer(&mut self, new_location: OurTaggedPtr) {
        let raw = self.raw_header_mut();
        // Store new location in meta field and set forwarding bit
        raw.meta = new_location.as_usize();
        raw.gc_flags |= OurHeader::FORWARDING_BIT;
    }
}

/// Runtime type bundle
pub struct OurRuntime;

impl GcTypes for OurRuntime {
    type TaggedValue = OurTaggedPtr;
    type ObjectHandle = OurObject;
    type ObjectKind = OurTypeTag;
}

/// Root provider that walks our frame chain
pub struct FrameChainRoots<'a> {
    thread: &'a Thread,
}

impl<'a> FrameChainRoots<'a> {
    pub fn new(thread: &'a Thread) -> Self {
        Self { thread }
    }
}

impl<'a> RootProvider<OurTaggedPtr> for FrameChainRoots<'a> {
    fn enumerate_roots(&self, callback: &mut dyn FnMut(usize, OurTaggedPtr)) {
        // Walk the frame chain
        for frame in self.thread.frames() {
            for (slot_addr, value) in frame.root_slots() {
                if !value.is_null() {
                    // value is a raw pointer to a heap object
                    let tagged = OurTaggedPtr::from_raw_ptr(value as *const u8);
                    if tagged.is_heap_pointer() {
                        // Pass the actual slot address so GC can update it after copying
                        callback(slot_addr as usize, tagged);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Global GC State
// =============================================================================

// Global GC pointer for fast access (single-threaded, like beagle)
// This is much faster than thread_local! + RefCell
static mut GC_PTR: *mut GenerationalGC<OurRuntime> = std::ptr::null_mut();

#[inline(always)]
fn with_gc<F, R>(f: F) -> R
where
    F: FnOnce(&mut GenerationalGC<OurRuntime>) -> R,
{
    unsafe { f(&mut *GC_PTR) }
}

// =============================================================================
// Binary Trees AST
// =============================================================================

fn build_binary_trees_program() -> Program {
    let tree_node = StructDef::new(
        "TreeNode",
        vec![
            StructField {
                name: "left".to_string(),
                typ: Type::Struct("TreeNode".to_string()),
            },
            StructField {
                name: "right".to_string(),
                typ: Type::Struct("TreeNode".to_string()),
            },
        ],
    );

    // Restructured to evaluate children BEFORE allocation (like beagle).
    // This ensures no GC safepoints between allocation and field stores,
    // so the object is guaranteed to be in young gen when stores happen.
    // No write barriers needed for construction!
    let bottom_up_tree = FunctionDef::new(
        "bottomUpTree",
        vec![Parameter::new("depth", Type::I64)],
        Type::Struct("TreeNode".to_string()),
        vec![
            // For interior nodes (depth > 0): evaluate children first, then allocate
            Stmt::if_stmt(
                Expr::gt(Expr::var("depth"), Expr::int(0)),
                vec![
                    // Evaluate children FIRST (these calls can trigger GC)
                    Stmt::let_decl(
                        "left",
                        Type::Struct("TreeNode".to_string()),
                        Some(Expr::call("bottomUpTree", vec![Expr::sub(Expr::var("depth"), Expr::int(1))])),
                    ),
                    Stmt::let_decl(
                        "right",
                        Type::Struct("TreeNode".to_string()),
                        Some(Expr::call("bottomUpTree", vec![Expr::sub(Expr::var("depth"), Expr::int(1))])),
                    ),
                    // NOW allocate - no GC can happen between here and the stores
                    Stmt::let_decl("node", Type::Struct("TreeNode".to_string()), Some(Expr::new_struct("TreeNode"))),
                    // Store children immediately - object is definitely in young gen
                    Stmt::field_set(
                        Expr::var("node"),
                        "TreeNode",
                        "left",
                        Expr::var("left"),
                    ),
                    Stmt::field_set(
                        Expr::var("node"),
                        "TreeNode",
                        "right",
                        Expr::var("right"),
                    ),
                    Stmt::ret(Some(Expr::var("node"))),
                ],
                vec![
                    // Leaf node (depth == 0): just allocate, children stay null
                    Stmt::let_decl("node", Type::Struct("TreeNode".to_string()), Some(Expr::new_struct("TreeNode"))),
                    Stmt::ret(Some(Expr::var("node"))),
                ],
            ),
        ],
    );

    let item_check = FunctionDef::new(
        "itemCheck",
        vec![Parameter::new("node", Type::Struct("TreeNode".to_string()))],
        Type::I64,
        vec![
            Stmt::let_decl(
                "left",
                Type::Struct("TreeNode".to_string()),
                Some(Expr::field_get(Expr::var("node"), "TreeNode", "left")),
            ),
            Stmt::if_stmt(
                Expr::eq(Expr::var("left"), Expr::Null),
                vec![Stmt::ret(Some(Expr::int(1)))],
                vec![],
            ),
            Stmt::ret(Some(Expr::add(
                Expr::int(1),
                Expr::add(
                    Expr::call("itemCheck", vec![Expr::var("left")]),
                    Expr::call(
                        "itemCheck",
                        vec![Expr::field_get(Expr::var("node"), "TreeNode", "right")],
                    ),
                ),
            ))),
        ],
    );

    let work = FunctionDef::new(
        "work",
        vec![
            Parameter::new("iterations", Type::I64),
            Parameter::new("depth", Type::I64),
        ],
        Type::I64,
        vec![
            Stmt::let_decl("check", Type::I64, Some(Expr::int(0))),
            Stmt::let_decl("i", Type::I64, Some(Expr::int(0))),
            Stmt::while_loop(
                Expr::lt(Expr::var("i"), Expr::var("iterations")),
                vec![
                    Stmt::let_decl(
                        "tree",
                        Type::Struct("TreeNode".to_string()),
                        Some(Expr::call("bottomUpTree", vec![Expr::var("depth")])),
                    ),
                    Stmt::assign(
                        "check",
                        Expr::add(
                            Expr::var("check"),
                            Expr::call("itemCheck", vec![Expr::var("tree")]),
                        ),
                    ),
                    Stmt::assign("i", Expr::add(Expr::var("i"), Expr::int(1))),
                ],
            ),
            Stmt::ret(Some(Expr::var("check"))),
        ],
    );

    // Match the official benchmark: iterations = 1 << (maxDepth - depth + minDepth)
    // where minDepth = 4
    let main_fn = FunctionDef::new(
        "main",
        vec![],
        Type::I64,
        vec![
            Stmt::let_decl("maxDepth", Type::I64, Some(Expr::int(21))), // N=21 for verification
            Stmt::let_decl("minDepth", Type::I64, Some(Expr::int(4))),
            Stmt::let_decl(
                "stretchDepth",
                Type::I64,
                Some(Expr::add(Expr::var("maxDepth"), Expr::int(1))),
            ),
            Stmt::let_decl(
                "stretchTree",
                Type::Struct("TreeNode".to_string()),
                Some(Expr::call("bottomUpTree", vec![Expr::var("stretchDepth")])),
            ),
            Stmt::let_decl(
                "stretchCheck",
                Type::I64,
                Some(Expr::call("itemCheck", vec![Expr::var("stretchTree")])),
            ),
            Stmt::let_decl(
                "longLivedTree",
                Type::Struct("TreeNode".to_string()),
                Some(Expr::call("bottomUpTree", vec![Expr::var("maxDepth")])),
            ),
            Stmt::let_decl("depth", Type::I64, Some(Expr::int(4))),
            Stmt::let_decl("total", Type::I64, Some(Expr::int(0))),
            Stmt::while_loop(
                Expr::le(Expr::var("depth"), Expr::var("maxDepth")),
                vec![
                    // iterations = 1 << (maxDepth - depth + minDepth)
                    Stmt::let_decl(
                        "iterations",
                        Type::I64,
                        Some(Expr::shl(
                            Expr::int(1),
                            Expr::add(
                                Expr::sub(Expr::var("maxDepth"), Expr::var("depth")),
                                Expr::var("minDepth"),
                            ),
                        )),
                    ),
                    Stmt::let_decl(
                        "checksum",
                        Type::I64,
                        Some(Expr::call(
                            "work",
                            vec![Expr::var("iterations"), Expr::var("depth")],
                        )),
                    ),
                    Stmt::assign(
                        "total",
                        Expr::add(Expr::var("total"), Expr::var("checksum")),
                    ),
                    Stmt::assign("depth", Expr::add(Expr::var("depth"), Expr::int(2))),
                ],
            ),
            Stmt::let_decl(
                "longLivedCheck",
                Type::I64,
                Some(Expr::call("itemCheck", vec![Expr::var("longLivedTree")])),
            ),
            Stmt::ret(Some(Expr::add(
                Expr::var("total"),
                Expr::add(Expr::var("longLivedCheck"), Expr::var("stretchCheck")),
            ))),
        ],
    );

    Program::new(
        vec![tree_node],
        vec![bottom_up_tree, item_check, work, main_fn],
    )
}

// =============================================================================
// FFI Functions
// =============================================================================

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static GC_COUNT: AtomicUsize = AtomicUsize::new(0);
static GC_TIME_NS: AtomicU64 = AtomicU64::new(0);  // Total GC time in nanoseconds
static BARRIER_CALLS: AtomicUsize = AtomicUsize::new(0);
static BARRIER_MARKS: AtomicUsize = AtomicUsize::new(0);

// Young generation bounds for fast write barrier check
// Objects in young gen don't need write barriers
// These are exposed as external symbols for the JIT-generated code
#[no_mangle]
static mut YOUNG_GEN_START: usize = 0;
#[no_mangle]
static mut YOUNG_GEN_END: usize = 0;

#[no_mangle]
pub extern "C" fn gc_pollcheck_slow(thread: *mut Thread, _origin: *const c_void) {
    let thread = unsafe { &mut *thread };

    // Clear the check flag
    thread.state = 0;

    // Create roots from frame chain and run GC
    let roots = FrameChainRoots::new(thread);
    GC_COUNT.fetch_add(1, Ordering::SeqCst);

    let start = Instant::now();
    with_gc(|gc| {
        gc.gc(&roots);
    });
    GC_TIME_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

#[no_mangle]
pub extern "C" fn gc_allocate(
    thread: *mut Thread,
    _meta: *const c_void,
    payload_size: u64,
) -> *mut u8 {
    let thread_ref = unsafe { &mut *thread };
    let count = ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);

    // Number of words (8-byte fields)
    // We need extra space: gc-library assumes 8-byte headers, but we use 16-byte.
    // So we request payload + 1 extra word to get the additional 8 bytes.
    let words = payload_size as usize / 8 + 1;

    // Allocation loop
    loop {
        let result = with_gc(|gc| {
            gc.try_allocate(words, OurTypeTag::HeapObject)
        });

        match result {
            Ok(AllocateAction::Allocated(ptr)) => {
                // Initialize header
                let mut obj = OurObject::from_untagged(ptr);
                obj.write_header(payload_size as usize);

                // Zero the fields
                let fields = obj.get_fields_mut();
                for field in fields {
                    *field = 0;
                }

                return ptr as *mut u8;
            }
            Ok(AllocateAction::Gc) => {
                // Need to run GC
                let roots = FrameChainRoots::new(thread_ref);
                GC_COUNT.fetch_add(1, Ordering::SeqCst);

                let start = Instant::now();
                with_gc(|gc| {
                    gc.gc(&roots);
                });
                GC_TIME_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                // Loop continues to retry allocation
            }
            Err(e) => {
                panic!("Allocation failed: {}", e);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn gc_allocate_array(
    thread: *mut Thread,
    _meta: *const c_void,
    length: u64,
) -> *mut u8 {
    // Arrays are just objects with `length` pointer fields
    gc_allocate(thread, _meta, length * 8)
}

#[no_mangle]
#[inline(always)]
pub extern "C" fn gc_write_barrier(object_ptr: *const u8) {
    // BARRIER_CALLS.fetch_add(1, Ordering::Relaxed);  // disabled for perf

    // Note: This function is only called when JIT inline check fails
    // (object is NOT in young gen). So we just mark the card directly.
    BARRIER_MARKS.fetch_add(1, Ordering::Relaxed);
    with_gc(|gc| {
        gc.mark_card_unconditional(object_ptr as usize);
    });
}

#[no_mangle]
pub extern "C" fn print_int(value: i64) -> i64 {
    println!("{}", value);
    value
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("Building binary trees program...");
    let program = build_binary_trees_program();

    println!("Compiling to LLVM IR...");
    let context = Context::create();
    let compiler = Compiler::new(&context, program);
    compiler.compile();

    // Print IR if PRINT_IR is set
    if std::env::var("PRINT_IR").is_ok() {
        println!("\n=== Generated LLVM IR (excerpt) ===\n");
        compiler.print_ir();
    }

    println!("\n=== Setting up GC ===\n");

    // Initialize GC with gc-library (generational collector)
    let options = AllocatorOptions::new();
    let gc: GenerationalGC<OurRuntime> = GenerationalGC::new(options);

    // Cache young gen bounds for fast write barrier checks
    let (young_start, young_end) = gc.get_young_gen_bounds();
    unsafe {
        YOUNG_GEN_START = young_start;
        YOUNG_GEN_END = young_end;
    }

    // Store GC in a Box and leak it to get a stable pointer
    let gc_box = Box::new(gc);
    unsafe {
        GC_PTR = Box::into_raw(gc_box);
    }

    println!("GC initialized (generational collector)");

    println!("\n=== Running with JIT ===\n");

    // Test if fixing the order of frame escape vs origin store helps with globalopt
    Target::initialize_native(&InitializationConfig::default()).unwrap();
    let target_triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&target_triple).unwrap();
    let host_cpu = TargetMachine::get_host_cpu_name();
    let host_features = TargetMachine::get_host_cpu_features();
    let target_machine = target
        .create_target_machine(
            &target_triple,
            host_cpu.to_str().unwrap(),
            host_features.to_str().unwrap(),
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::Default,
        )
        .unwrap();

    let pass_options = PassBuilderOptions::create();

    // Optional: run specific LLVM passes via TEST_PASS env var for testing.
    // Examples:
    //   TEST_PASS="default<O2>" - full O2 pipeline
    //   TEST_PASS="module(globalopt,instcombine)" - specific passes
    //
    // Note: JIT Aggressive already provides good optimization (~9.5s baseline).
    // Additional module passes don't significantly improve performance for this workload.
    let test_pass = std::env::var("TEST_PASS").unwrap_or_default();
    if !test_pass.is_empty() {
        println!("Testing pass: {}", test_pass);
        compiler
            .module
            .run_passes(&test_pass, &target_machine, pass_options)
            .unwrap();

        // Print IR after optimization if requested
        if std::env::var("PRINT_IR_AFTER").is_ok() {
            println!("\n=== IR After Optimization ===\n");
            compiler.print_ir();
        }
    }

    // Remove llvm.compiler.used before JIT - it's not supported by the execution engine.
    // llvm.compiler.used was created to prevent optimization passes from eliminating our
    // metadata globals (frame origins, object metas), but the JIT doesn't understand it.
    if let Some(used_global) = compiler.module.get_global("llvm.compiler.used") {
        unsafe {
            used_global.delete();
        }
    }

    // Always use aggressive JIT optimization - module passes + JIT Aggressive can work together
    let jit_opt_level = OptimizationLevel::Aggressive;
    let execution_engine = compiler
        .module
        .create_jit_execution_engine(jit_opt_level)
        .unwrap();

    // Add symbol mappings for runtime functions that survive optimization.
    // GlobalOpt may eliminate unused declarations, so we only map functions that still exist.
    let map_if_exists = |name: &str, addr: usize| {
        if let Some(func) = compiler.module.get_function(name) {
            execution_engine.add_global_mapping(&func, addr);
        }
    };
    map_if_exists("gc_pollcheck_slow", gc_pollcheck_slow as usize);
    map_if_exists("gc_allocate", gc_allocate as usize);
    map_if_exists("gc_allocate_array", gc_allocate_array as usize);
    map_if_exists("gc_write_barrier", gc_write_barrier as usize);
    map_if_exists("print_int", print_int as usize);

    // Map global variables for young gen bounds
    unsafe {
        if let Some(g) = compiler.module.get_global("YOUNG_GEN_START") {
            execution_engine.add_global_mapping(&g, std::ptr::addr_of!(YOUNG_GEN_START) as usize);
        }
        if let Some(g) = compiler.module.get_global("YOUNG_GEN_END") {
            execution_engine.add_global_mapping(&g, std::ptr::addr_of!(YOUNG_GEN_END) as usize);
        }
    }

    // Set up thread (no longer needs heap pointer)
    let mut thread = Thread::new(std::ptr::null_mut());

    // Get the main function
    type MainFn = unsafe extern "C" fn(*mut Thread) -> i64;
    let main_fn: JitFunction<MainFn> = unsafe {
        execution_engine.get_function("main").unwrap()
    };

    // Run it!
    println!("Calling main()...\n");
    let exec_start = Instant::now();
    let result = unsafe { main_fn.call(&mut thread) };
    let exec_time = exec_start.elapsed();

    println!("\n=== Results ===");
    println!("Result: {}", result);
    println!("Total execution time: {:.2} s", exec_time.as_secs_f64());
    println!("Total allocations: {}", ALLOCATION_COUNT.load(Ordering::SeqCst));
    println!("Total GC cycles: {}", GC_COUNT.load(Ordering::SeqCst));
    let gc_time_ns = GC_TIME_NS.load(Ordering::Relaxed);
    let gc_time_ms = gc_time_ns as f64 / 1_000_000.0;
    let gc_percentage = (gc_time_ms / 1000.0) / exec_time.as_secs_f64() * 100.0;
    println!("Total GC time: {:.2} ms ({:.2} s, {:.1}% of execution)", gc_time_ms, gc_time_ms / 1000.0, gc_percentage);
    println!("Write barrier calls: {}", BARRIER_CALLS.load(Ordering::Relaxed));
    println!("Write barrier marks (old gen): {}", BARRIER_MARKS.load(Ordering::Relaxed));

    // Final stack state (should be empty after main returns)
    println!("\n=== Final stack state ===");
    thread.dump_stack();
}
