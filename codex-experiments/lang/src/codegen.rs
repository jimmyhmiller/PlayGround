use crate::ast::*;
use crate::runtime;
use crate::{lexer::Lexer, parser::Parser};
use crate::{qualify, resolve, typecheck};
use inkwell::builder::Builder;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module;
use inkwell::types::{BasicType, BasicTypeEnum, StructType};
use inkwell::values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, PointerValue};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

#[derive(Debug)]
pub struct CodegenError {
    pub message: String,
}

pub struct Codegen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    thread_struct_ty: StructType<'ctx>,
    thread_ty: inkwell::types::PointerType<'ctx>,
    mode: CodegenMode,
    functions: HashMap<String, FunctionValue<'ctx>>,
    externs: HashMap<String, FunctionValue<'ctx>>,
    gc_pollcheck: FunctionValue<'ctx>,
    gc_allocate: FunctionValue<'ctx>,
    gc_allocate_array: FunctionValue<'ctx>,
    gc_write_barrier: FunctionValue<'ctx>,
    object_header_ty: StructType<'ctx>,
    structs: HashMap<String, StructLayout<'ctx>>,
    enums: HashMap<String, EnumLayout<'ctx>>,
    tuples: HashMap<String, StructLayout<'ctx>>,
    str_lit_id: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodegenMode {
    Jit,
    Aot,
}

#[derive(Clone)]
struct StructLayout<'ctx> {
    size: u64,
    fields: Vec<FieldLayout<'ctx>>,
    meta: PointerValue<'ctx>,
}

#[derive(Clone)]
struct FieldLayout<'ctx> {
    name: String,
    ty: Type,
    llvm_ty: BasicTypeEnum<'ctx>,
    offset: u64,
    is_ptr: bool,
}

#[derive(Clone)]
struct EnumLayout<'ctx> {
    size: u64,
    ptr_count: usize,
    raw_size: u64,
    raw_base: u64,
    meta: PointerValue<'ctx>,
    variants: HashMap<String, VariantLayout<'ctx>>,
}

#[derive(Clone)]
struct VariantLayout<'ctx> {
    tag: u32,
    fields: Vec<VariantFieldLayout<'ctx>>,
}

#[derive(Clone)]
struct VariantFieldLayout<'ctx> {
    name: Option<String>,
    ty: Type,
    llvm_ty: BasicTypeEnum<'ctx>,
    is_ptr: bool,
    ptr_index: usize,
    raw_offset: u64,
}

/// Default memory manager that delegates to system allocator.
/// Used to create MCJIT engine with FastISel disabled.
#[derive(Debug)]
struct DefaultMemoryManager;

impl inkwell::memory_manager::McjitMemoryManager for DefaultMemoryManager {
    fn allocate_code_section(
        &mut self,
        size: libc::uintptr_t,
        alignment: libc::c_uint,
        _section_id: libc::c_uint,
        _section_name: &str,
    ) -> *mut u8 {
        unsafe {
            let layout = std::alloc::Layout::from_size_align(size, alignment as usize)
                .unwrap_or(std::alloc::Layout::from_size_align(size, 16).unwrap());
            std::alloc::alloc(layout)
        }
    }

    fn allocate_data_section(
        &mut self,
        size: libc::uintptr_t,
        alignment: libc::c_uint,
        _section_id: libc::c_uint,
        _section_name: &str,
        _is_read_only: bool,
    ) -> *mut u8 {
        unsafe {
            let layout = std::alloc::Layout::from_size_align(size, alignment as usize)
                .unwrap_or(std::alloc::Layout::from_size_align(size, 16).unwrap());
            std::alloc::alloc(layout)
        }
    }

    fn finalize_memory(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn destroy(&mut self) {}
}

pub fn compile_and_run(modules: &[crate::ast::Module]) -> Result<i64, CodegenError> {
    use inkwell::targets::{InitializationConfig, Target, TargetMachine, RelocMode, CodeModel};
    let context = Context::create();
    let mut gen = Codegen::new(&context, "lang_module", CodegenMode::Jit);
    gen.declare_structs(modules)?;
    gen.declare_enums(modules)?;
    gen.declare_tuples(modules)?;
    gen.declare_functions(modules)?;
    gen.define_functions(modules)?;
    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| CodegenError { message: format!("target init failed: {e}") })?;
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple)
        .map_err(|e| CodegenError { message: format!("target lookup failed: {e}") })?;
    let cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();
    let cpu_str = cpu.to_str().unwrap_or("generic");
    let feat_str = features.to_str().unwrap_or("");
    if let Some(tm) = target.create_target_machine(
        &triple,
        cpu_str,
        feat_str,
        OptimizationLevel::None,
        RelocMode::Default,
        CodeModel::Default,
    ) {
        gen.module.set_triple(&triple);
        let data_layout = tm.get_target_data().get_data_layout();
        gen.module.set_data_layout(&data_layout);
    }
    let engine = gen
        .module
        .create_mcjit_execution_engine_with_memory_manager(
            DefaultMemoryManager,
            OptimizationLevel::None,
            inkwell::targets::CodeModel::Default,
            false, // no_frame_pointer_elim
            false, // enable_fast_isel = false (DISABLED)
        )
        .map_err(|e| CodegenError {
            message: format!("jit create failed: {e}"),
        })?;
    add_runtime_mappings(
        &engine,
        &gen.externs,
        gen.gc_pollcheck,
        gen.gc_allocate,
        gen.gc_allocate_array,
        gen.gc_write_barrier,
    );
    runtime::gc_init();
    let mut thread = runtime::Thread::new();
    unsafe {
        let main_name = gen.llvm_fn_name("main");
        let main = engine
            .get_function::<unsafe extern "C" fn(*mut u8) -> i64>(&main_name)
            .map_err(|e| CodegenError {
                message: format!("missing main: {e}"),
            })?;
        Ok(main.call(&mut thread as *mut _ as *mut u8))
    }
}

pub fn compile_to_object(modules: &[crate::ast::Module], output: &std::path::Path) -> Result<(), CodegenError> {
    use inkwell::targets::{InitializationConfig, RelocMode, Target, TargetMachine, CodeModel, FileType};

    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| CodegenError { message: format!("target init failed: {e}") })?;
    let context = Context::create();
    let mut gen = Codegen::new(&context, "lang_module", CodegenMode::Aot);
    gen.declare_structs(modules)?;
    gen.declare_enums(modules)?;
    gen.declare_tuples(modules)?;
    gen.declare_functions(modules)?;
    gen.define_functions(modules)?;
    gen.emit_aot_wrapper_main()?;
    if std::env::var("LANGC_DUMP_IR").is_ok() {
        eprintln!("{}", gen.module.print_to_string().to_string());
    }

    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple)
        .map_err(|e| CodegenError { message: format!("target lookup failed: {e}") })?;
    let cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();
    let cpu_str = cpu.to_str().unwrap_or("generic");
    let feat_str = features.to_str().unwrap_or("");
    let target_machine = target
        .create_target_machine(
            &triple,
            cpu_str,
            feat_str,
            OptimizationLevel::None,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or_else(|| CodegenError { message: "failed to create target machine".to_string() })?;

    gen.module.set_triple(&triple);
    let data_layout = target_machine.get_target_data().get_data_layout();
    gen.module.set_data_layout(&data_layout);

    target_machine
        .write_to_file(&gen.module, FileType::Object, output)
        .map_err(|e| CodegenError { message: format!("write object failed: {e}") })?;
    Ok(())
}

extern "C" fn rt_print_int(value: i64) -> i64 {
    println!("{value}");
    0
}

extern "C" fn rt_print_str(ptr: *const u8) -> i64 {
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

extern "C" fn rt_add_i64(a: i64, b: i64) -> i64 {
    a + b
}

extern "C" fn rt_null_ptr() -> *mut u8 {
    std::ptr::null_mut()
}

extern "C" fn rt_ptr_is_null(p: *mut u8) -> i64 {
    if p.is_null() { 1 } else { 0 }
}

extern "C" fn rt_arg_i64(index: i64) -> i64 {
    if index < 0 {
        return 0;
    }
    let args: Vec<String> = std::env::args().collect();
    let mut base = 1usize;
    if let Some(pos) = args.iter().position(|a| a == "--") {
        base = pos + 1;
    } else if args.len() >= 3 && args[1] == "run" {
        base = 3;
    }
    let pos = base as i64 + index;
    if pos < 0 || pos as usize >= args.len() {
        return 0;
    }
    args[pos as usize].parse::<i64>().unwrap_or(0)
}

extern "C" fn rt_arg_str(index: i64) -> *const u8 {
    if index < 0 {
        return std::ptr::null();
    }
    let args: Vec<String> = std::env::args().collect();
    let mut base = 1usize;
    if let Some(pos) = args.iter().position(|a| a == "--") {
        base = pos + 1;
    } else if args.len() >= 3 && args[1] == "run" {
        base = 3;
    }
    let pos = base as i64 + index;
    if pos < 0 || pos as usize >= args.len() {
        return std::ptr::null();
    }
    let mut bytes = args[pos as usize].as_bytes().to_vec();
    bytes.push(0);
    let boxed = bytes.into_boxed_slice();
    Box::into_raw(boxed) as *const u8
}

extern "C" fn rt_arg_is_i64(index: i64) -> i64 {
    if index < 0 {
        return 0;
    }
    let args: Vec<String> = std::env::args().collect();
    let mut base = 1usize;
    if let Some(pos) = args.iter().position(|a| a == "--") {
        base = pos + 1;
    } else if args.len() >= 3 && args[1] == "run" {
        base = 3;
    }
    let pos = base as i64 + index;
    if pos < 0 || pos as usize >= args.len() {
        return 0;
    }
    if args[pos as usize].parse::<i64>().is_ok() { 1 } else { 0 }
}

extern "C" fn rt_arg_len() -> i64 {
    let args: Vec<String> = std::env::args().collect();
    let mut base = 1usize;
    if let Some(pos) = args.iter().position(|a| a == "--") {
        base = pos + 1;
    } else if args.len() >= 3 && args[1] == "run" {
        base = 3;
    }
    if args.len() < base {
        0
    } else {
        (args.len() - base) as i64
    }
}

extern "C" fn rt_host_build() -> i64 {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn collect_lang_files(dir: &Path, out: &mut Vec<String>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    collect_lang_files(&path, out);
                } else if path.extension().and_then(|s| s.to_str()) == Some("lang") {
                    out.push(path.to_string_lossy().to_string());
                }
            }
        }
    }

    fn build_roots(paths: &[String]) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        for p in paths {
            let path = Path::new(p);
            if path.is_dir() {
                roots.push(path.to_path_buf());
            } else if let Some(parent) = path.parent() {
                roots.push(parent.to_path_buf());
            }
        }
        if roots.is_empty() {
            roots.push(Path::new(".").to_path_buf());
        }
        roots
    }

    fn derive_module_path(path: &Path, roots: &[PathBuf], root_file: Option<&Path>) -> Option<Vec<String>> {
        let mut best_root: Option<&PathBuf> = None;
        let mut best_len = 0usize;
        for root in roots {
            if path.starts_with(root) {
                let len = root.components().count();
                if len >= best_len {
                    best_len = len;
                    best_root = Some(root);
                }
            }
        }
        let root = best_root?;
        let rel = path.strip_prefix(root).ok()?;
        let rel_dir = rel.parent().unwrap_or_else(|| Path::new(""));
        let mut parts = Vec::new();
        for comp in rel_dir.components() {
            if let std::path::Component::Normal(s) = comp {
                if let Some(text) = s.to_str() {
                    parts.push(text.to_string());
                }
            }
        }
        let skip_stem = root_file.map_or(false, |root| root == path);
        if !skip_stem {
            let stem = path.file_stem().and_then(|s| s.to_str());
            if let Some(stem) = stem {
                let is_root = matches!(stem, "main" | "lib" | "mod");
                if !is_root {
                    parts.push(stem.to_string());
                }
            }
        }
        Some(parts)
    }

    let args: Vec<String> = std::env::args().collect();
    let mut forwarded = Vec::new();
    if let Some(pos) = args.iter().position(|a| a == "--") {
        forwarded.extend_from_slice(&args[pos + 1..]);
    }
    if forwarded.is_empty() {
        eprintln!("bootstrap build: no target files");
        return 1;
    }
    if forwarded[0].parse::<i64>().is_ok() {
        forwarded.remove(0);
    }
    if forwarded.is_empty() {
        eprintln!("bootstrap build: no target files");
        return 1;
    }

    let mut all_paths: Vec<String> = Vec::new();
    for p in &forwarded {
        let path = Path::new(p);
        if path.is_dir() {
            collect_lang_files(path, &mut all_paths);
        } else {
            all_paths.push(p.clone());
        }
    }
    if all_paths.is_empty() {
        eprintln!("bootstrap build: no .lang files found");
        return 1;
    }

    let roots = build_roots(&forwarded);
    let root_file = if forwarded.len() == 1 && Path::new(&forwarded[0]).is_file() {
        Some(Path::new(&forwarded[0]).to_path_buf())
    } else {
        None
    };
    let mut modules: Vec<crate::ast::Module> = Vec::new();
    for path in &all_paths {
        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("failed to read {}: {}", path, err);
                return 1;
            }
        };
        let tokens = match Lexer::new(&src).lex_all() {
            Ok(toks) => toks,
            Err(errors) => {
                for err in errors {
                    eprintln!("lex error: {} at {}..{}", err.message, err.span.start, err.span.end);
                }
                return 1;
            }
        };
        let mut module = match Parser::new(tokens).parse_module() {
            Ok(module) => module,
            Err(errors) => {
                for err in errors {
                    eprintln!("parse error: {} at {}..{}", err.message, err.span.start, err.span.end);
                }
                return 1;
            }
        };
        if module.path.is_none() {
            if let Some(p) = derive_module_path(Path::new(path), &roots, root_file.as_deref()) {
                if !p.is_empty() {
                    module.path = Some(p);
                }
            }
        }
        modules.push(module);
    }

    if let Err(errors) = qualify::qualify_modules(&mut modules) {
        for err in errors {
            eprintln!("qualify error: {} at {}..{}", err.message, err.span.start, err.span.end);
        }
        return 1;
    }
    if let Err(errors) = resolve::resolve_modules(&modules) {
        for err in errors {
            eprintln!("resolve error: {} at {}..{}", err.message, err.span.start, err.span.end);
        }
        return 1;
    }
    if let Err(errors) = typecheck::typecheck_modules(&modules) {
        for err in errors {
            eprintln!("type error: {} at {}..{}", err.message, err.span.start, err.span.end);
        }
        return 1;
    }

    let input = Path::new(&all_paths[0]);
    let obj_path = input.with_extension("o");
    let mut exe_path = input.with_extension("");
    if input.extension().is_none() {
        exe_path = input.with_extension("out");
    }
    if let Err(err) = compile_to_object(&modules, &obj_path) {
        eprintln!("codegen error: {}", err.message);
        return 1;
    }

    let status = Command::new("cargo")
        .arg("build")
        .arg("--quiet")
        .arg("--lib")
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("failed to build runtime lib: {}", s);
            return 1;
        }
        Err(err) => {
            eprintln!("failed to invoke cargo: {err}");
            return 1;
        }
    }

    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let lib_path = target_dir.join(profile).join("liblang_runtime.a");

    let status = Command::new("cc")
        .arg(&obj_path)
        .arg(&lib_path)
        .arg("-O2")
        .arg("-o")
        .arg(&exe_path)
        .status();
    match status {
        Ok(s) if s.success() => {
            println!("built {}", exe_path.display());
            0
        }
        Ok(s) => {
            eprintln!("link failed: {}", s);
            1
        }
        Err(err) => {
            eprintln!("failed to invoke cc: {err}");
            1
        }
    }
}
extern "C" fn rt_string_len(ptr: *const u8) -> i64 {
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

extern "C" fn rt_string_eq(a: *const u8, b: *const u8) -> i64 {
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

extern "C" fn rt_string_concat(a: *const u8, b: *const u8) -> *const u8 {
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

extern "C" fn rt_string_slice(ptr: *const u8, start: i64, end_pos: i64) -> *const u8 {
    runtime::string_slice(ptr, start, end_pos)
}

extern "C" fn rt_read_file(path: *const u8) -> *const u8 {
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

extern "C" fn rt_vec_new() -> *mut u8 {
    let v = VecPtr { len: 0, cap: 0, data: std::ptr::null_mut() };
    Box::into_raw(Box::new(v)) as *mut u8
}

extern "C" fn rt_vec_len(vec: *mut u8) -> i64 {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*(vec as *mut VecPtr)).len }
}

extern "C" fn rt_vec_get(vec: *mut u8, index: i64) -> *mut u8 {
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

extern "C" fn rt_vec_push(vec: *mut u8, item: *mut u8) -> i64 {
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

extern "C" fn rt_vec_clear(vec: *mut u8) -> i64 {
    if vec.is_null() {
        return 0;
    }
    unsafe {
        let v = &mut *(vec as *mut VecPtr);
        v.len = 0;
    }
    0
}

extern "C" fn rt_vec_set_len(vec: *mut u8, new_len: i64) -> i64 {
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

extern "C" fn rt_enum_tag(obj: *mut u8, raw_base: i64) -> i64 {
    if obj.is_null() {
        return -1;
    }
    unsafe {
        let tag_ptr = obj.add(raw_base as usize) as *const i32;
        *tag_ptr as i64
    }
}

extern "C" fn rt_string_byte_at(ptr: *const u8, index: i64) -> i64 {
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

extern "C" fn rt_exit_process(code: i64) -> i64 {
    std::process::exit(code as i32);
}

extern "C" fn rt_string_from_i64(val: i64) -> *const u8 {
    let s = format!("{val}");
    let mut bytes = s.into_bytes();
    bytes.push(0);
    let boxed = bytes.into_boxed_slice();
    Box::into_raw(boxed) as *const u8
}

extern "C" fn rt_string_parse_i64(ptr: *const u8) -> i64 {
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

extern "C" fn rt_print_str_stderr(ptr: *const u8) -> i64 {
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

extern "C" fn rt_print_stretch(depth: i64, check: i64) {
    println!("stretch tree of depth {depth}\t check: {check}");
}

extern "C" fn rt_print_trees(iterations: i64, depth: i64, check: i64) {
    println!("{iterations}\t trees of depth {depth}\t check: {check}");
}

extern "C" fn rt_print_long_lived(depth: i64, check: i64) {
    println!("long lived tree of depth {depth}\t check: {check}");
}

fn add_runtime_mappings<'ctx>(
    engine: &ExecutionEngine<'ctx>,
    externs: &HashMap<String, FunctionValue<'ctx>>,
    gc_pollcheck: FunctionValue<'ctx>,
    gc_allocate: FunctionValue<'ctx>,
    gc_allocate_array: FunctionValue<'ctx>,
    gc_write_barrier: FunctionValue<'ctx>,
) {
    let map_all = |name: &str, addr: usize| {
        for (k, v) in externs.iter() {
            if last_segment(k) == name {
                engine.add_global_mapping(v, addr);
            }
        }
    };
    map_all("print_int", rt_print_int as usize);
    map_all("print_str", rt_print_str as usize);
    map_all("add_i64", rt_add_i64 as usize);
    map_all("null_ptr", rt_null_ptr as usize);
    map_all("ptr_is_null", rt_ptr_is_null as usize);
    map_all("arg_i64", rt_arg_i64 as usize);
    map_all("arg_str", rt_arg_str as usize);
    map_all("arg_is_i64", rt_arg_is_i64 as usize);
    map_all("arg_len", rt_arg_len as usize);
    map_all("host_build", rt_host_build as usize);
    map_all("string_len", rt_string_len as usize);
    map_all("string_eq", rt_string_eq as usize);
    map_all("string_concat", rt_string_concat as usize);
    map_all("string_slice", rt_string_slice as usize);
    map_all("read_file", rt_read_file as usize);
    map_all("vec_new", rt_vec_new as usize);
    map_all("vec_len", rt_vec_len as usize);
    map_all("vec_get", rt_vec_get as usize);
    map_all("vec_push", rt_vec_push as usize);
    map_all("string_byte_at", rt_string_byte_at as usize);
    map_all("print_stretch", rt_print_stretch as usize);
    map_all("print_trees", rt_print_trees as usize);
    map_all("print_long_lived", rt_print_long_lived as usize);
    map_all("exit_process", rt_exit_process as usize);
    map_all("string_from_i64", rt_string_from_i64 as usize);
    map_all("print_str_stderr", rt_print_str_stderr as usize);
    map_all("vec_clear", rt_vec_clear as usize);
    map_all("vec_set_len", rt_vec_set_len as usize);
    map_all("enum_tag", rt_enum_tag as usize);
    map_all("string_parse_i64", rt_string_parse_i64 as usize);
    // Vec aliases — all forward to the same rt_vec_push / rt_vec_get
    for alias in &[
        "vec_push_str", "vec_push_item", "vec_push_param", "vec_push_field",
        "vec_push_variant", "vec_push_type", "vec_push_expr", "vec_push_arm",
        "vec_push_stmt", "vec_push_pfield", "vec_push_slfield",
        "vec_push_ientry", "vec_push_uentry", "vec_push_vec", "vec_push_module",
        "vec_push_ventry",
    ] {
        map_all(alias, rt_vec_push as usize);
    }
    for alias in &[
        "vec_get_str", "vec_get_item", "vec_get_param", "vec_get_field",
        "vec_get_variant", "vec_get_type", "vec_get_expr", "vec_get_arm",
        "vec_get_stmt", "vec_get_pfield", "vec_get_slfield", "vec_get_tok",
        "vec_get_ientry", "vec_get_uentry", "vec_get_vec", "vec_get_module",
        "vec_get_ventry",
    ] {
        map_all(alias, rt_vec_get as usize);
    }
    engine.add_global_mapping(&gc_pollcheck, runtime::gc_pollcheck_slow as usize);
    engine.add_global_mapping(&gc_allocate, runtime::gc_allocate as usize);
    engine.add_global_mapping(&gc_allocate_array, runtime::gc_allocate_array as usize);
    engine.add_global_mapping(&gc_write_barrier, runtime::gc_write_barrier as usize);
    map_all("gc_init", runtime::gc_init as usize);

    // LLVM wrapper mappings
    use crate::llvm_wrappers;
    map_all("llvm_context_create", llvm_wrappers::llvm_context_create as usize);
    map_all("llvm_module_create", llvm_wrappers::llvm_module_create as usize);
    map_all("llvm_create_builder", llvm_wrappers::llvm_create_builder as usize);
    map_all("llvm_dispose_builder", llvm_wrappers::llvm_dispose_builder as usize);
    map_all("llvm_dispose_module", llvm_wrappers::llvm_dispose_module as usize);
    map_all("llvm_context_dispose", llvm_wrappers::llvm_context_dispose as usize);
    map_all("llvm_int1_type", llvm_wrappers::llvm_int1_type as usize);
    map_all("llvm_int8_type", llvm_wrappers::llvm_int8_type as usize);
    map_all("llvm_int16_type", llvm_wrappers::llvm_int16_type as usize);
    map_all("llvm_int32_type", llvm_wrappers::llvm_int32_type as usize);
    map_all("llvm_int64_type", llvm_wrappers::llvm_int64_type as usize);
    map_all("llvm_float_type", llvm_wrappers::llvm_float_type as usize);
    map_all("llvm_double_type", llvm_wrappers::llvm_double_type as usize);
    map_all("llvm_ptr_type", llvm_wrappers::llvm_ptr_type as usize);
    map_all("llvm_void_type", llvm_wrappers::llvm_void_type as usize);
    map_all("llvm_function_type", llvm_wrappers::llvm_function_type as usize);
    map_all("llvm_struct_type", llvm_wrappers::llvm_struct_type as usize);
    map_all("llvm_array_type", llvm_wrappers::llvm_array_type as usize);
    map_all("llvm_const_int", llvm_wrappers::llvm_const_int as usize);
    map_all("llvm_const_real", llvm_wrappers::llvm_const_real as usize);
    map_all("llvm_const_null", llvm_wrappers::llvm_const_null as usize);
    map_all("llvm_const_string", llvm_wrappers::llvm_const_string as usize);
    map_all("llvm_const_named_struct", llvm_wrappers::llvm_const_named_struct as usize);
    map_all("llvm_const_array", llvm_wrappers::llvm_const_array as usize);
    map_all("llvm_const_gep2", llvm_wrappers::llvm_const_gep2 as usize);
    map_all("llvm_add_function", llvm_wrappers::llvm_add_function as usize);
    map_all("llvm_get_named_function", llvm_wrappers::llvm_get_named_function as usize);
    map_all("llvm_add_global", llvm_wrappers::llvm_add_global as usize);
    map_all("llvm_set_initializer", llvm_wrappers::llvm_set_initializer as usize);
    map_all("llvm_get_value_type", llvm_wrappers::llvm_get_value_type as usize);
    map_all("llvm_print_module_to_string", llvm_wrappers::llvm_print_module_to_string as usize);
    map_all("llvm_get_param", llvm_wrappers::llvm_get_param as usize);
    map_all("llvm_count_params", llvm_wrappers::llvm_count_params as usize);
    map_all("llvm_append_basic_block", llvm_wrappers::llvm_append_basic_block as usize);
    map_all("llvm_get_first_basic_block", llvm_wrappers::llvm_get_first_basic_block as usize);
    map_all("llvm_get_bb_terminator", llvm_wrappers::llvm_get_bb_terminator as usize);
    map_all("llvm_get_insert_block", llvm_wrappers::llvm_get_insert_block as usize);
    map_all("llvm_position_at_end", llvm_wrappers::llvm_position_at_end as usize);
    map_all("llvm_position_before", llvm_wrappers::llvm_position_before as usize);
    map_all("llvm_build_alloca", llvm_wrappers::llvm_build_alloca as usize);
    map_all("llvm_build_load", llvm_wrappers::llvm_build_load as usize);
    map_all("llvm_build_store", llvm_wrappers::llvm_build_store as usize);
    map_all("llvm_build_call", llvm_wrappers::llvm_build_call as usize);
    map_all("llvm_build_ret", llvm_wrappers::llvm_build_ret as usize);
    map_all("llvm_build_ret_void", llvm_wrappers::llvm_build_ret_void as usize);
    map_all("llvm_build_br", llvm_wrappers::llvm_build_br as usize);
    map_all("llvm_build_cond_br", llvm_wrappers::llvm_build_cond_br as usize);
    map_all("llvm_build_switch", llvm_wrappers::llvm_build_switch as usize);
    map_all("llvm_build_phi", llvm_wrappers::llvm_build_phi as usize);
    map_all("llvm_build_gep2", llvm_wrappers::llvm_build_gep2 as usize);
    map_all("llvm_build_inbounds_gep2", llvm_wrappers::llvm_build_inbounds_gep2 as usize);
    map_all("llvm_build_bitcast", llvm_wrappers::llvm_build_bitcast as usize);
    map_all("llvm_build_struct_gep2", llvm_wrappers::llvm_build_struct_gep2 as usize);
    map_all("llvm_build_memset", llvm_wrappers::llvm_build_memset as usize);
    map_all("llvm_build_global_string_ptr", llvm_wrappers::llvm_build_global_string_ptr as usize);
    map_all("llvm_build_add", llvm_wrappers::llvm_build_add as usize);
    map_all("llvm_build_sub", llvm_wrappers::llvm_build_sub as usize);
    map_all("llvm_build_mul", llvm_wrappers::llvm_build_mul as usize);
    map_all("llvm_build_sdiv", llvm_wrappers::llvm_build_sdiv as usize);
    map_all("llvm_build_srem", llvm_wrappers::llvm_build_srem as usize);
    map_all("llvm_build_icmp", llvm_wrappers::llvm_build_icmp as usize);
    map_all("llvm_build_and", llvm_wrappers::llvm_build_and as usize);
    map_all("llvm_build_or", llvm_wrappers::llvm_build_or as usize);
    map_all("llvm_build_xor", llvm_wrappers::llvm_build_xor as usize);
    map_all("llvm_build_zext", llvm_wrappers::llvm_build_zext as usize);
    map_all("llvm_build_trunc", llvm_wrappers::llvm_build_trunc as usize);
    map_all("llvm_build_fadd", llvm_wrappers::llvm_build_fadd as usize);
    map_all("llvm_build_fsub", llvm_wrappers::llvm_build_fsub as usize);
    map_all("llvm_build_fmul", llvm_wrappers::llvm_build_fmul as usize);
    map_all("llvm_build_fdiv", llvm_wrappers::llvm_build_fdiv as usize);
    map_all("llvm_build_frem", llvm_wrappers::llvm_build_frem as usize);
    map_all("llvm_build_fcmp", llvm_wrappers::llvm_build_fcmp as usize);
    map_all("llvm_build_fneg", llvm_wrappers::llvm_build_fneg as usize);
    map_all("llvm_add_incoming", llvm_wrappers::llvm_add_incoming as usize);
    map_all("llvm_add_case", llvm_wrappers::llvm_add_case as usize);
    map_all("llvm_set_volatile", llvm_wrappers::llvm_set_volatile as usize);
    map_all("llvm_init_native_target", llvm_wrappers::llvm_init_native_target as usize);
    map_all("llvm_init_native_asm_printer", llvm_wrappers::llvm_init_native_asm_printer as usize);
    map_all("llvm_get_default_triple", llvm_wrappers::llvm_get_default_triple as usize);
    map_all("llvm_get_target_from_triple", llvm_wrappers::llvm_get_target_from_triple as usize);
    map_all("llvm_get_host_cpu_name", llvm_wrappers::llvm_get_host_cpu_name as usize);
    map_all("llvm_get_host_cpu_features", llvm_wrappers::llvm_get_host_cpu_features as usize);
    map_all("llvm_create_target_machine", llvm_wrappers::llvm_create_target_machine as usize);
    map_all("llvm_set_module_target", llvm_wrappers::llvm_set_module_target as usize);
    map_all("llvm_emit_to_file", llvm_wrappers::llvm_emit_to_file as usize);
    map_all("llvm_get_first_instruction", llvm_wrappers::llvm_get_first_instruction as usize);
    map_all("llvm_is_null", llvm_wrappers::llvm_is_null as usize);
    map_all("llvm_get_global_value_type", llvm_wrappers::llvm_get_global_value_type as usize);
    map_all("llvm_get_element_type", llvm_wrappers::llvm_get_element_type as usize);
    map_all("llvm_pointer_type", llvm_wrappers::llvm_pointer_type as usize);
    map_all("llvm_get_type_kind", llvm_wrappers::llvm_get_type_kind as usize);

    // Runtime helpers
    map_all("write_file", runtime::write_file as usize);
    map_all("system_cmd", runtime::system_cmd as usize);
    map_all("vec_data", runtime::vec_data as usize);

    // Codegen pass vec aliases
    for alias in &["vec_push_cgfn","vec_push_cgsl","vec_push_cgel","vec_push_cgfl",
                    "vec_push_cgvfl","vec_push_cgvl","vec_push_cgloc","vec_push_loop"] {
        map_all(alias, rt_vec_push as usize);
    }
    for alias in &["vec_get_cgfn","vec_get_cgsl","vec_get_cgel","vec_get_cgfl",
                    "vec_get_cgvfl","vec_get_cgvl","vec_get_cgloc","vec_get_loop"] {
        map_all(alias, rt_vec_get as usize);
    }
    // Typecheck pass vec aliases
    for alias in &["vec_push_ty","vec_push_smentry","vec_push_ementry","vec_push_fnentry",
                    "vec_push_vdentry","vec_push_subst","vec_push_local","vec_push_tfield"] {
        map_all(alias, rt_vec_push as usize);
    }
    for alias in &["vec_get_ty","vec_get_smentry","vec_get_ementry","vec_get_fnentry",
                    "vec_get_vdentry","vec_get_subst","vec_get_local","vec_get_tfield"] {
        map_all(alias, rt_vec_get as usize);
    }
}

impl<'ctx> Codegen<'ctx> {
    fn declare_structs(&mut self, modules: &[crate::ast::Module]) -> Result<(), CodegenError> {
        for module in modules {
            let module_path = module.path.clone().unwrap_or_default();
            for item in &module.items {
                if let Item::Struct(s) = item {
                    let full_name = full_item_name(&module_path, &s.name);
                    let layout = self.compute_struct_layout(&full_name, s)?;
                    self.structs.insert(full_name, layout);
                }
            }
        }
        Ok(())
    }

    fn declare_enums(&mut self, modules: &[crate::ast::Module]) -> Result<(), CodegenError> {
        for module in modules {
            let module_path = module.path.clone().unwrap_or_default();
            for item in &module.items {
                if let Item::Enum(e) = item {
                    let full_name = full_item_name(&module_path, &e.name);
                    let layout = self.compute_enum_layout(&full_name, e)?;
                    self.enums.insert(full_name, layout);
                }
            }
        }
        Ok(())
    }

    fn declare_tuples(&mut self, modules: &[crate::ast::Module]) -> Result<(), CodegenError> {
        for module in modules {
            for item in &module.items {
                match item {
                    Item::Struct(s) => {
                        for field in &s.fields {
                            self.collect_tuple_types(&field.ty)?;
                        }
                    }
                    Item::Enum(e) => {
                        for variant in &e.variants {
                            match &variant.kind {
                                EnumVariantKind::Unit => {}
                                EnumVariantKind::Tuple(types) => {
                                    for ty in types {
                                        self.collect_tuple_types(ty)?;
                                    }
                                }
                                EnumVariantKind::Struct(fields) => {
                                    for field in fields {
                                        self.collect_tuple_types(&field.ty)?;
                                    }
                                }
                            }
                        }
                    }
                    Item::Fn(f) => {
                        for param in &f.params {
                            self.collect_tuple_types(&param.ty)?;
                        }
                        self.collect_tuple_types(&f.ret_type)?;
                        self.collect_tuple_types_in_block(&f.body)?;
                    }
                    Item::ExternFn(f) => {
                        for param in &f.params {
                            self.collect_tuple_types(&param.ty)?;
                        }
                        self.collect_tuple_types(&f.ret_type)?;
                    }
                    Item::Use(_) => {}
                }
            }
        }
        Ok(())
    }

    fn compute_struct_layout(&mut self, full_name: &str, s: &StructDecl) -> Result<StructLayout<'ctx>, CodegenError> {
        let ptr_size = std::mem::size_of::<usize>() as u64;
        let mut ptr_fields = Vec::new();
        let mut raw_fields = Vec::new();

        for field in &s.fields {
            let is_ptr = self.is_gc_ref_type(&field.ty);
            let llvm_ty = self.field_llvm_type(&field.ty)?;
            let entry = FieldLayout {
                name: field.name.clone(),
                ty: field.ty.clone(),
                llvm_ty,
                offset: 0,
                is_ptr,
            };
            if is_ptr {
                ptr_fields.push(entry);
            } else {
                raw_fields.push(entry);
            }
        }

        let mut fields = Vec::new();
        let mut offset = 0u64;

        for mut field in ptr_fields {
            field.offset = offset;
            offset += ptr_size;
            fields.push(field);
        }

        for mut field in raw_fields {
            let align = match field.llvm_ty {
                BasicTypeEnum::IntType(t) if t.get_bit_width() == 1 => 1,
                BasicTypeEnum::IntType(t) => (t.get_bit_width() / 8) as u64,
                _ => 8,
            };
            if offset % align != 0 {
                offset += align - (offset % align);
            }
            field.offset = offset;
            offset += align.max(1);
            fields.push(field);
        }

        let payload_size = offset;
        let header_size = 16u64;
        let total_size = header_size + payload_size;

        let ptr_offsets: Vec<u64> = fields
            .iter()
            .filter(|f| f.is_ptr)
            .map(|f| header_size + f.offset)
            .collect();

        let meta = self.create_type_info(full_name, 0, total_size, &ptr_offsets)?;

        Ok(StructLayout {
            size: total_size,
            fields,
            meta,
        })
    }

    fn compute_enum_layout(&mut self, full_name: &str, e: &EnumDecl) -> Result<EnumLayout<'ctx>, CodegenError> {
        let ptr_size = std::mem::size_of::<usize>() as u64;
        let header_size = 16u64;
        let mut variants = HashMap::new();
        let mut max_ptrs = 0usize;
        let mut max_raw = 0u64;

        for (tag, variant) in e.variants.iter().enumerate() {
            let mut ptr_index = 0usize;
            let mut raw_offset = 4u64; // tag at offset 0 in raw section
            let mut fields = Vec::new();
            let field_specs: Vec<(Option<String>, Type)> = match &variant.kind {
                EnumVariantKind::Unit => Vec::new(),
                EnumVariantKind::Tuple(types) => types.iter().map(|t| (None, t.clone())).collect(),
                EnumVariantKind::Struct(fields) => fields.iter().map(|f| (Some(f.name.clone()), f.ty.clone())).collect(),
            };
            for (name, ty) in field_specs {
                let is_ptr = self.is_gc_ref_type(&ty);
                let llvm_ty = self.field_llvm_type(&ty)?;
                if is_ptr {
                    fields.push(VariantFieldLayout {
                        name,
                        ty: ty.clone(),
                        llvm_ty,
                        is_ptr: true,
                        ptr_index,
                        raw_offset: 0,
                    });
                    ptr_index += 1;
                } else {
                    let align = match llvm_ty {
                        BasicTypeEnum::IntType(t) if t.get_bit_width() == 1 => 1,
                        BasicTypeEnum::IntType(t) => (t.get_bit_width() / 8) as u64,
                        _ => 8,
                    };
                    if raw_offset % align != 0 {
                        raw_offset += align - (raw_offset % align);
                    }
                    fields.push(VariantFieldLayout {
                        name,
                        ty: ty.clone(),
                        llvm_ty,
                        is_ptr: false,
                        ptr_index: 0,
                        raw_offset,
                    });
                    raw_offset += align.max(1);
                }
            }
            let variant_ptrs = ptr_index;
            let variant_raw = raw_offset;
            if variant_ptrs > max_ptrs {
                max_ptrs = variant_ptrs;
            }
            if variant_raw > max_raw {
                max_raw = variant_raw;
            }
            variants.insert(
                variant.name.clone(),
                VariantLayout {
                    tag: tag as u32,
                    fields,
                },
            );
        }

        let raw_size = if max_raw == 0 { 4 } else { max_raw };
        let raw_base = header_size + (max_ptrs as u64) * ptr_size;
        let total_size = raw_base + raw_size;

        let ptr_offsets: Vec<u64> = (0..max_ptrs)
            .map(|i| header_size + (i as u64) * ptr_size)
            .collect();
        let meta = self.create_type_info(full_name, 1, total_size, &ptr_offsets)?;

        Ok(EnumLayout {
            size: total_size,
            ptr_count: max_ptrs,
            raw_size,
            raw_base,
            meta,
            variants,
        })
    }

    fn collect_tuple_types(&mut self, ty: &Type) -> Result<(), CodegenError> {
        match ty {
            Type::Tuple(items) => {
                self.ensure_tuple_layout(items)?;
                for item in items {
                    self.collect_tuple_types(item)?;
                }
            }
            Type::RawPointer(inner) => self.collect_tuple_types(inner)?,
            Type::Path(_, _) => {}
        }
        Ok(())
    }

    fn collect_tuple_types_in_block(&mut self, block: &Block) -> Result<(), CodegenError> {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => self.collect_tuple_types_in_expr(expr)?,
                Stmt::Return(expr, _) => {
                    if let Some(expr) = expr {
                        self.collect_tuple_types_in_expr(expr)?;
                    }
                }
            }
        }
        if let Some(tail) = &block.tail {
            self.collect_tuple_types_in_expr(tail)?;
        }
        Ok(())
    }

    fn collect_tuple_types_in_expr(&mut self, expr: &Expr) -> Result<(), CodegenError> {
        match expr {
            Expr::Let { ty, value, .. } => {
                if let Some(ty) = ty {
                    self.collect_tuple_types(ty)?;
                }
                self.collect_tuple_types_in_expr(value)?;
            }
            Expr::If { cond, then_branch, else_branch, .. } => {
                self.collect_tuple_types_in_expr(cond)?;
                self.collect_tuple_types_in_block(then_branch)?;
                if let Some(else_branch) = else_branch {
                    self.collect_tuple_types_in_block(else_branch)?;
                }
            }
            Expr::While { cond, body, .. } => {
                self.collect_tuple_types_in_expr(cond)?;
                self.collect_tuple_types_in_block(body)?;
            }
            Expr::Match { scrutinee, arms, .. } => {
                self.collect_tuple_types_in_expr(scrutinee)?;
                for arm in arms {
                    self.collect_tuple_types_in_expr(&arm.body)?;
                }
            }
            Expr::Assign { target, value, .. } => {
                self.collect_tuple_types_in_expr(target)?;
                self.collect_tuple_types_in_expr(value)?;
            }
            Expr::Binary { left, right, .. } => {
                self.collect_tuple_types_in_expr(left)?;
                self.collect_tuple_types_in_expr(right)?;
            }
            Expr::Unary { expr, .. } => self.collect_tuple_types_in_expr(expr)?,
            Expr::Call { callee, args, .. } => {
                self.collect_tuple_types_in_expr(callee)?;
                for arg in args {
                    self.collect_tuple_types_in_expr(arg)?;
                }
            }
            Expr::Field { base, .. } => self.collect_tuple_types_in_expr(base)?,
            Expr::StructLit { fields, .. } => {
                for (_, expr) in fields {
                    self.collect_tuple_types_in_expr(expr)?;
                }
            }
            Expr::Tuple { items, .. } => {
                for item in items {
                    self.collect_tuple_types_in_expr(item)?;
                }
            }
            Expr::Block(block) => {
                self.collect_tuple_types_in_block(block)?;
            }
            Expr::Literal(_, _) | Expr::Path(_, _) => {}
            Expr::Break { .. } | Expr::Continue { .. } => {}
        }
        Ok(())
    }

    fn ensure_tuple_layout(&mut self, items: &[Type]) -> Result<String, CodegenError> {
        let key = tuple_key(items);
        if !self.tuples.contains_key(&key) {
            let layout = self.compute_tuple_layout(&key, items)?;
            self.tuples.insert(key.clone(), layout);
        }
        Ok(key)
    }

    fn compute_tuple_layout(&mut self, name: &str, items: &[Type]) -> Result<StructLayout<'ctx>, CodegenError> {
        let ptr_size = std::mem::size_of::<usize>() as u64;
        let mut ptr_fields = Vec::new();
        let mut raw_fields = Vec::new();

        for (i, ty) in items.iter().enumerate() {
            let is_ptr = self.is_gc_ref_type(ty);
            let llvm_ty = self.field_llvm_type(ty)?;
            let entry = FieldLayout {
                name: i.to_string(),
                ty: ty.clone(),
                llvm_ty,
                offset: 0,
                is_ptr,
            };
            if is_ptr {
                ptr_fields.push(entry);
            } else {
                raw_fields.push(entry);
            }
        }

        let mut fields = Vec::new();
        let mut offset = 0u64;

        for mut field in ptr_fields {
            field.offset = offset;
            offset += ptr_size;
            fields.push(field);
        }

        for mut field in raw_fields {
            let align = match field.llvm_ty {
                BasicTypeEnum::IntType(t) if t.get_bit_width() == 1 => 1,
                BasicTypeEnum::IntType(t) => (t.get_bit_width() / 8) as u64,
                _ => 8,
            };
            if offset % align != 0 {
                offset += align - (offset % align);
            }
            field.offset = offset;
            offset += align.max(1);
            fields.push(field);
        }

        let payload_size = offset;
        let header_size = 16u64;
        let total_size = header_size + payload_size;

        let ptr_offsets: Vec<u64> = fields
            .iter()
            .filter(|f| f.is_ptr)
            .map(|f| header_size + f.offset)
            .collect();

        let meta = self.create_type_info(name, 0, total_size, &ptr_offsets)?;

        Ok(StructLayout {
            size: total_size,
            fields,
            meta,
        })
    }

    fn create_type_info(
        &self,
        name: &str,
        kind: u32,
        size: u64,
        ptr_offsets: &[u64],
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        let i32_ty = self.context.i32_type();
        let i64_ty = self.context.i64_type();
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());

        let offsets_const = i64_ty.const_array(
            &ptr_offsets
                .iter()
                .map(|v| i64_ty.const_int(*v, false))
                .collect::<Vec<_>>(),
        );
        let sym = mangle_name(name);
        let offsets_global = self.module.add_global(
            offsets_const.get_type(),
            None,
            &format!("__ptr_offsets_{}", sym),
        );
        offsets_global.set_initializer(&offsets_const);
        let zero = i32_ty.const_zero();
        let offsets_ptr = unsafe {
            offsets_global
                .as_pointer_value()
                .const_gep(offsets_const.get_type(), &[zero, zero])
        };

        let name_const = self.context.const_string(name.as_bytes(), true);
        let name_global = self
            .module
            .add_global(name_const.get_type(), None, &format!("__type_name_{}", sym));
        name_global.set_initializer(&name_const);
        let name_ptr = unsafe {
            name_global
                .as_pointer_value()
                .const_gep(name_const.get_type(), &[zero, zero])
        };

        let typeinfo_ty = self.context.struct_type(
            &[
                i32_ty.into(), // kind
                i32_ty.into(), // pad
                i64_ty.into(), // size
                i32_ty.into(), // num_ptrs
                i32_ty.into(), // pad
                ptr_ty.into(), // name
                ptr_ty.into(), // ptr_offsets
            ],
            false,
        );

        let kind = i32_ty.const_int(kind as u64, false);
        let size_val = i64_ty.const_int(size, false);
        let num_ptrs = i32_ty.const_int(ptr_offsets.len() as u64, false);
        let init = typeinfo_ty.const_named_struct(&[
            kind.into(),
            i32_ty.const_zero().into(),
            size_val.into(),
            num_ptrs.into(),
            i32_ty.const_zero().into(),
            name_ptr.into(),
            offsets_ptr.into(),
        ]);

        let global = self
            .module
            .add_global(typeinfo_ty, None, &format!("__typeinfo_{}", sym));
        global.set_initializer(&init);
        Ok(global.as_pointer_value())
    }
    fn new(context: &'ctx Context, name: &str, mode: CodegenMode) -> Self {
        let module = context.create_module(name);
        let builder = context.create_builder();
        let ptr_ty = context.ptr_type(inkwell::AddressSpace::default());
        let thread_struct_ty = context.struct_type(
            &[
                ptr_ty.into(),               // top_frame
                context.i32_type().into(),   // state
                context.i32_type().into(),   // padding
            ],
            false,
        );
        let thread_ty = thread_struct_ty.ptr_type(inkwell::AddressSpace::default());
        let object_header_ty = context.struct_type(
            &[
                ptr_ty.into(),             // meta
                context.i32_type().into(), // gc_flags
                context.i32_type().into(), // aux
            ],
            false,
        );
        let gc_pollcheck = Self::declare_gc_pollcheck(&module, thread_ty, context);
        let gc_allocate = Self::declare_gc_allocate(&module, thread_ty, context, "gc_allocate");
        let gc_allocate_array = Self::declare_gc_allocate(&module, thread_ty, context, "gc_allocate_array");
        let gc_write_barrier = Self::declare_gc_write_barrier(&module, thread_ty, context);
        Self {
            context,
            module,
            builder,
            thread_struct_ty,
            thread_ty,
            mode,
            functions: HashMap::new(),
            externs: HashMap::new(),
            gc_pollcheck,
            gc_allocate,
            gc_allocate_array,
            gc_write_barrier,
            object_header_ty,
            structs: HashMap::new(),
            enums: HashMap::new(),
            tuples: HashMap::new(),
            str_lit_id: 0,
        }
    }

    fn llvm_fn_name(&self, full_name: &str) -> String {
        if self.mode == CodegenMode::Aot && full_name == "main" {
            "__lang_main".to_string()
        } else {
            mangle_name(full_name)
        }
    }

    fn declare_gc_pollcheck(
        module: &Module<'ctx>,
        thread_ty: inkwell::types::PointerType<'ctx>,
        context: &'ctx Context,
    ) -> FunctionValue<'ctx> {
        let void_ty = context.void_type();
        let ptr_ty = context.ptr_type(inkwell::AddressSpace::default());
        let fn_ty = void_ty.fn_type(&[thread_ty.into(), ptr_ty.into()], false);
        module.add_function("gc_pollcheck_slow", fn_ty, None)
    }

    fn declare_gc_allocate(
        module: &Module<'ctx>,
        thread_ty: inkwell::types::PointerType<'ctx>,
        context: &'ctx Context,
        name: &str,
    ) -> FunctionValue<'ctx> {
        let ptr_ty = context.ptr_type(inkwell::AddressSpace::default());
        let i64_ty = context.i64_type();
        let fn_ty = ptr_ty.fn_type(&[thread_ty.into(), ptr_ty.into(), i64_ty.into()], false);
        module.add_function(name, fn_ty, None)
    }

    fn declare_gc_write_barrier(
        module: &Module<'ctx>,
        thread_ty: inkwell::types::PointerType<'ctx>,
        context: &'ctx Context,
    ) -> FunctionValue<'ctx> {
        let ptr_ty = context.ptr_type(inkwell::AddressSpace::default());
        let fn_ty = context.void_type().fn_type(&[thread_ty.into(), ptr_ty.into(), ptr_ty.into()], false);
        module.add_function("gc_write_barrier", fn_ty, None)
    }

    fn declare_functions(&mut self, modules: &[crate::ast::Module]) -> Result<(), CodegenError> {
        for module in modules {
            let module_path = module.path.clone().unwrap_or_default();
            for item in &module.items {
                match item {
                    Item::Fn(f) => {
                        let full_name = full_item_name(&module_path, &f.name);
                        let fn_val = self.declare_fn(&full_name, f)?;
                        self.functions.insert(full_name, fn_val);
                    }
                    Item::ExternFn(f) => {
                        let full_name = full_item_name(&module_path, &f.name);
                        let fn_val = self.declare_extern_fn(&full_name, f)?;
                        self.externs.insert(full_name, fn_val);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn declare_fn(&self, full_name: &str, f: &FnDecl) -> Result<FunctionValue<'ctx>, CodegenError> {
        let mut param_tys = Vec::new();
        param_tys.push(self.thread_ty.into());
        for param in &f.params {
            param_tys.push(self.llvm_type(&param.ty)?.into());
        }
        let ret_ty = self.llvm_type(&f.ret_type)?;
        let fn_ty = match ret_ty {
            BasicTypeEnum::IntType(t) => t.fn_type(&param_tys, false),
            BasicTypeEnum::PointerType(t) => t.fn_type(&param_tys, false),
            _ => return Err(CodegenError { message: "unsupported return type".to_string() }),
        };
        let llvm_name = self.llvm_fn_name(full_name);
        Ok(self.module.add_function(&llvm_name, fn_ty, None))
    }

    fn declare_extern_fn(&self, full_name: &str, f: &ExternFnDecl) -> Result<FunctionValue<'ctx>, CodegenError> {
        let symbol = extern_symbol_name(full_name);
        // Reuse existing LLVM function if already declared by another module
        if let Some(existing) = self.module.get_function(symbol) {
            return Ok(existing);
        }
        let mut param_tys = Vec::new();
        for param in &f.params {
            param_tys.push(self.llvm_type(&param.ty)?.into());
        }
        let ret_ty = self.llvm_type(&f.ret_type)?;
        let fn_ty = match ret_ty {
            BasicTypeEnum::IntType(t) => t.fn_type(&param_tys, f.varargs),
            BasicTypeEnum::PointerType(t) => t.fn_type(&param_tys, f.varargs),
            _ => return Err(CodegenError { message: "unsupported extern return type".to_string() }),
        };
        Ok(self.module.add_function(symbol, fn_ty, None))
    }

    fn define_functions(&mut self, modules: &[crate::ast::Module]) -> Result<(), CodegenError> {
        for module in modules {
            let module_path = module.path.clone().unwrap_or_default();
            for item in &module.items {
                if let Item::Fn(f) = item {
                    let full_name = full_item_name(&module_path, &f.name);
                    self.define_fn(&full_name, f)?;
                }
            }
        }
        Ok(())
    }

    fn define_fn(&mut self, full_name: &str, f: &FnDecl) -> Result<(), CodegenError> {
        let function = *self
            .functions
            .get(full_name)
            .ok_or_else(|| CodegenError { message: "missing function".to_string() })?;
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        let mut locals: HashMap<String, Local<'ctx>> = HashMap::new();
        let thread = function.get_nth_param(0).unwrap().into_pointer_value();

        let num_roots = self.count_roots_in_fn(f);
        let frame_ty = self.frame_type(num_roots);
        let (frame_ptr, frame_origin, root_base) = self.emit_prologue(thread, full_name, num_roots, frame_ty)?;

        let mut param_root_index = 0usize;
        for (i, param) in f.params.iter().enumerate() {
            let local = self.create_local_storage_in_ctx(
                function,
                &param.name,
                &param.ty,
                root_base,
                param_root_index,
            )?;
            let arg = function.get_nth_param((i + 1) as u32).unwrap();
            self.store_local(&local, arg)?;
            locals.insert(param.name.clone(), local);
            if self.is_gc_ref_type(&param.ty) {
                param_root_index += 1;
            }
        }

        let mut ctx = FnCtx {
            function,
            locals,
            scopes: Vec::new(),
            thread,
            frame_ptr,
            frame_origin,
            frame_ty,
            root_base,
            next_root: self.count_param_roots(&f.params),
            loop_stack: Vec::new(),
        };

        let value = self.codegen_block_value(&f.body, &mut ctx, Some(&f.ret_type))
            .map_err(|e| CodegenError { message: format!("{} (in fn {})", e.message, full_name) })?;
        let terminated = self
            .builder
            .get_insert_block()
            .and_then(|bb| bb.get_terminator())
            .is_some();
        if !terminated {
            let ret_val: BasicValueEnum = match value {
                Some(v) => v,
                None => match self.llvm_type(&f.ret_type)? {
                    BasicTypeEnum::IntType(_) => self.context.i64_type().const_int(0, false).into(),
                    _ => return Err(CodegenError { message: "unsupported return type".to_string() }),
                },
            };
            self.emit_epilogue(&ctx)?;
            self.map_builder(self.builder.build_return(Some(&ret_val)), "return")?;
        }
        Ok(())
    }

    fn emit_aot_wrapper_main(&mut self) -> Result<(), CodegenError> {
        if self.mode != CodegenMode::Aot {
            return Ok(());
        }
        let lang_main = *self
            .functions
            .get("main")
            .ok_or_else(|| CodegenError { message: "missing main function".to_string() })?;

        let i32_ty = self.context.i32_type();
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let fn_ty = i32_ty.fn_type(&[i32_ty.into(), ptr_ty.into()], false);
        let wrapper = self.module.add_function("main", fn_ty, None);
        let entry = self.context.append_basic_block(wrapper, "entry");
        self.builder.position_at_end(entry);

        let argc = wrapper.get_nth_param(0).unwrap();
        let argv = wrapper.get_nth_param(1).unwrap();
        let init_ty = self.context.void_type().fn_type(&[i32_ty.into(), ptr_ty.into()], false);
        let init_fn = self
            .module
            .get_function("rt_init_args")
            .unwrap_or_else(|| self.module.add_function("rt_init_args", init_ty, None));
        self.map_builder(
            self.builder.build_call(init_fn, &[argc.into(), argv.into()], "init_args"),
            "call",
        )?;

        // Call gc_init() before any allocations
        let gc_init_ty = self.context.void_type().fn_type(&[], false);
        let gc_init_fn = self
            .module
            .get_function("gc_init")
            .unwrap_or_else(|| self.module.add_function("gc_init", gc_init_ty, None));
        self.map_builder(
            self.builder.build_call(gc_init_fn, &[], "gc_init"),
            "call",
        )?;

        let thread_alloca = self.map_builder(self.builder.build_alloca(self.thread_struct_ty, "thread"), "alloca")?;
        let top_ptr = self.map_builder(
            self.builder
                .build_struct_gep(self.thread_struct_ty, thread_alloca, 0, "thread.top"),
            "gep",
        )?;
        let state_ptr = self.map_builder(
            self.builder
                .build_struct_gep(self.thread_struct_ty, thread_alloca, 1, "thread.state"),
            "gep",
        )?;
        let pad_ptr = self.map_builder(
            self.builder
                .build_struct_gep(self.thread_struct_ty, thread_alloca, 2, "thread.pad"),
            "gep",
        )?;
        self.map_builder(self.builder.build_store(top_ptr, ptr_ty.const_null()), "store")?;
        self.map_builder(self.builder.build_store(state_ptr, i32_ty.const_zero()), "store")?;
        self.map_builder(self.builder.build_store(pad_ptr, i32_ty.const_zero()), "store")?;

        let thread_ptr = self.map_builder(self.builder.build_bit_cast(thread_alloca, ptr_ty, "thread_ptr"), "bitcast")?;
        let call = self.map_builder(self.builder.build_call(lang_main, &[thread_ptr.into()], "lang_main"), "call")?;
        let ret_val = call
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError { message: "main returned void".to_string() })?;
        let ret_i32 = self.map_builder(
            self.builder
                .build_int_truncate(ret_val.into_int_value(), i32_ty, "ret"),
            "trunc",
        )?;
        self.map_builder(self.builder.build_return(Some(&ret_i32)), "return")?;
        Ok(())
    }

    fn codegen_block(&mut self, block: &Block, ctx: &mut FnCtx<'ctx>, expected: Option<&Type>) -> Result<bool, CodegenError> {
        self.push_scope(ctx);
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => {
                    let _ = self.codegen_expr(expr, ctx, None)?;
                }
                Stmt::Return(expr, _) => {
                    let value = if let Some(expr) = expr {
                        self.codegen_expr(expr, ctx, expected)?
                    } else {
                        None
                    };
                    if let Some(v) = value {
                        self.emit_epilogue(ctx)?;
                        self.map_builder(self.builder.build_return(Some(&v)), "return")?;
                    } else {
                        let zero = self.context.i64_type().const_int(0, false);
                        self.emit_epilogue(ctx)?;
                        self.map_builder(self.builder.build_return(Some(&zero)), "return")?;
                    }
                    self.pop_scope(ctx);
                    return Ok(true);
                }
            }
        }
        if let Some(tail) = &block.tail {
            let _ = self.codegen_expr(tail, ctx, expected)?;
        }
        self.pop_scope(ctx);
        Ok(false)
    }

    fn codegen_expr(
        &mut self,
        expr: &Expr,
        ctx: &mut FnCtx<'ctx>,
        expected: Option<&Type>,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        match expr {
            Expr::Let { name, ty, value, .. } => {
                let val = self.codegen_expr(value, ctx, ty.as_ref().or(expected))?;
                let ty = ty.as_ref().or(expected).ok_or_else(|| CodegenError { message: "let requires type".to_string() })?;
                let local = self.create_local_storage_in_ctx(ctx.function, name, ty, ctx.root_base, ctx.next_root)?;
                if let Some(v) = val {
                    self.store_local(&local, v)?;
                }
                ctx.locals.insert(name.clone(), local);
                if let Some(scope) = ctx.scopes.last_mut() {
                    scope.push(name.clone());
                }
                if self.is_gc_ref_type(ty) {
                    ctx.next_root += 1;
                    self.emit_pollcheck(ctx)?;
                }
                Ok(None)
            }
            Expr::Assign { target, value, .. } => {
                let v = self.codegen_expr(value, ctx, None)?;
                match &**target {
                    Expr::Path(path, _) => {
                        let name = path.last().ok_or_else(|| CodegenError { message: "invalid assignment".to_string() })?;
                        let local = ctx.locals.get(name).ok_or_else(|| CodegenError { message: format!("unknown local '{}' in assign", name) })?;
                        if let Some(v) = v {
                            self.store_local(local, v)?;
                            if self.is_gc_ref_type(&local.lang_ty) {
                                self.emit_pollcheck(ctx)?;
                            }
                        }
                        Ok(None)
                    }
                    Expr::Field { base, name, .. } => {
                        let (obj_ptr, _layout, field) = self.resolve_field(base, name, ctx)?;
                        let field_ptr = self.field_ptr(obj_ptr, &field)?;
                        if let Some(v) = v {
                            let store_val = if self.is_bool_type(&field.ty) {
                                let zext = self.map_builder(
                                    self.builder.build_int_z_extend(v.into_int_value(), self.context.i8_type(), "boolz"),
                                    "zext",
                                )?;
                                zext.into()
                            } else {
                                v
                            };
                            self.map_builder(self.builder.build_store(field_ptr, store_val), "store")?;
                            if self.is_gc_ref_type(&field.ty) {
                                let args = &[ctx.thread.into(), obj_ptr.into(), v.into()];
                                let _ = self.map_builder(self.builder.build_call(self.gc_write_barrier, args, "wb"), "call")?;
                            }
                        }
                        Ok(None)
                    }
                    _ => Err(CodegenError { message: "unsupported assignment target".to_string() }),
                }
            }
            Expr::Literal(lit, _) => match lit {
                Literal::Int(text) => Ok(Some(self.const_i64(text)?)),
                Literal::Char(value) => Ok(Some(self.context.i64_type().const_int(*value as u64, false).into())),
                Literal::Float(text) => {
                    let is_f32 = matches!(
                        expected,
                        Some(Type::Path(path, _)) if path.last().map(|s| s.as_str()) == Some("F32")
                    );
                    if is_f32 {
                        let value = text.parse::<f32>().map_err(|_| CodegenError { message: "invalid float literal".to_string() })?;
                        Ok(Some(self.context.f32_type().const_float(value as f64).into()))
                    } else {
                        let value = text.parse::<f64>().map_err(|_| CodegenError { message: "invalid float literal".to_string() })?;
                        Ok(Some(self.context.f64_type().const_float(value).into()))
                    }
                }
                Literal::Bool(b) => Ok(Some(self.context.bool_type().const_int(*b as u64, false).into())),
                Literal::Unit => Ok(Some(self.context.i64_type().const_zero().into())),
                Literal::Str(text) => {
                    let bytes = text.as_bytes();
                    let const_str = self.context.const_string(bytes, true);
                    let name = format!("__str_lit_{}", self.str_lit_id);
                    self.str_lit_id += 1;
                    let global = self
                        .module
                        .add_global(const_str.get_type(), None, &name);
                    global.set_initializer(&const_str);
                    let zero = self.context.i32_type().const_zero();
                    let ptr = unsafe {
                        global
                            .as_pointer_value()
                            .const_gep(const_str.get_type(), &[zero, zero])
                    };
                    Ok(Some(ptr.into()))
                }
                _ => Err(CodegenError { message: "unsupported literal".to_string() }),
            },
            Expr::Path(path, _) => {
                let name = path.last().ok_or_else(|| CodegenError { message: "empty path".to_string() })?;
                let local = ctx.locals.get(name).ok_or_else(|| CodegenError { message: format!("unknown local '{}' in path expr", name) })?;
                let loaded = self.load_local(local, name)?;
                Ok(Some(loaded))
            }
            Expr::Field { base, name, .. } => {
                let (obj_ptr, _layout, field) = self.resolve_field(base, name, ctx)?;
                let field_ptr = self.field_ptr(obj_ptr, &field)?;
                let loaded = self.map_builder(self.builder.build_load(field.llvm_ty, field_ptr, "field"), "load")?;
                if self.is_bool_type(&field.ty) {
                    let zero = self.context.i8_type().const_zero();
                    let cmp = self.map_builder(
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::NE, loaded.into_int_value(), zero, "bool"),
                        "icmp",
                    )?;
                    Ok(Some(cmp.into()))
                } else {
                    Ok(Some(loaded))
                }
            }
            Expr::Binary { op, left, right, .. } => {
                let l = self.codegen_expr(left, ctx, None)?.ok_or_else(|| CodegenError { message: "missing lhs".to_string() })?;
                let r = self.codegen_expr(right, ctx, None)?.ok_or_else(|| CodegenError { message: "missing rhs".to_string() })?;
                Ok(Some(self.codegen_binary(op, l, r)?))
            }
            Expr::Unary { op, expr, .. } => {
                let v = self.codegen_expr(expr, ctx, None)?.ok_or_else(|| CodegenError { message: "missing operand".to_string() })?;
                match op {
                    UnaryOp::Neg => {
                        if v.is_float_value() {
                            let value = self.map_builder(self.builder.build_float_neg(v.into_float_value(), "fneg"), "fneg")?;
                            Ok(Some(value.into()))
                        } else {
                            let zero = self.context.i64_type().const_int(0, false);
                            let value = self.map_builder(self.builder.build_int_sub(zero, v.into_int_value(), "neg"), "neg")?;
                            Ok(Some(value.into()))
                        }
                    }
                    UnaryOp::Not => {
                        let one = self.context.bool_type().const_int(1, false);
                        let value = self.map_builder(self.builder.build_xor(v.into_int_value(), one, "not"), "not")?;
                        Ok(Some(value.into()))
                    }
                }
            }
            Expr::If { cond, then_branch, else_branch, .. } => {
                let cond_val = self.codegen_expr(cond, ctx, None)?.ok_or_else(|| CodegenError { message: "missing cond".to_string() })?;
                let func = ctx.function;
                let then_bb = self.context.append_basic_block(func, "then");
                let else_bb = self.context.append_basic_block(func, "else");
                let cont_bb = self.context.append_basic_block(func, "cont");
                self.map_builder(
                    self.builder
                        .build_conditional_branch(cond_val.into_int_value(), then_bb, else_bb),
                    "condbr",
                )?;

                self.builder.position_at_end(then_bb);
                let then_val = self.codegen_block_value(then_branch, ctx, expected)?;
                let then_val_bb = self.builder.get_insert_block().unwrap();
                if !then_val_bb.get_terminator().is_some() {
                    self.map_builder(self.builder.build_unconditional_branch(cont_bb), "br")?;
                }

                self.builder.position_at_end(else_bb);
                let else_val = if let Some(else_branch) = else_branch {
                    self.codegen_block_value(else_branch, ctx, expected)?
                } else {
                    None
                };
                let else_val_bb = self.builder.get_insert_block().unwrap();
                if !else_val_bb.get_terminator().is_some() {
                    self.map_builder(self.builder.build_unconditional_branch(cont_bb), "br")?;
                }

                self.builder.position_at_end(cont_bb);
                if let (Some(tv), Some(ev)) = (then_val, else_val) {
                    let phi = self.map_builder(self.builder.build_phi(tv.get_type(), "iftmp"), "phi")?;
                    phi.add_incoming(&[(&tv, then_val_bb), (&ev, else_val_bb)]);
                    Ok(Some(phi.as_basic_value()))
                } else {
                    Ok(None)
                }
            }
            Expr::While { cond, body, .. } => {
                let func = ctx.function;
                let cond_bb = self.context.append_basic_block(func, "while.cond");
                let body_bb = self.context.append_basic_block(func, "while.body");
                let cont_bb = self.context.append_basic_block(func, "while.cont");
                self.map_builder(self.builder.build_unconditional_branch(cond_bb), "br")?;

                self.builder.position_at_end(cond_bb);
                let cond_val = self.codegen_expr(cond, ctx, None)?.ok_or_else(|| CodegenError { message: "missing cond".to_string() })?;
                self.map_builder(
                    self.builder
                        .build_conditional_branch(cond_val.into_int_value(), body_bb, cont_bb),
                    "condbr",
                )?;

                self.builder.position_at_end(body_bb);
                ctx.loop_stack.push((cond_bb, cont_bb));
                let _ = self.codegen_block(body, ctx, None)?;
                ctx.loop_stack.pop();
                if !self.builder.get_insert_block().unwrap().get_terminator().is_some() {
                    self.emit_pollcheck(ctx)?;
                    self.map_builder(self.builder.build_unconditional_branch(cond_bb), "br")?;
                }

                self.builder.position_at_end(cont_bb);
                Ok(None)
            }
            Expr::Match { scrutinee, arms, .. } => {
                let mut enum_name = None;
                for arm in arms {
                    match &arm.pattern {
                        Pattern::Path(path, _) | Pattern::Struct { path, .. } => {
                            if path.len() >= 2 {
                                let (enum_path, _) = enum_path_and_variant(path);
                                enum_name = Some(enum_path);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let enum_name = enum_name.ok_or_else(|| CodegenError { message: "match missing enum pattern".to_string() })?;
                let enum_layout = self
                    .enums
                    .get(&enum_name)
                    .cloned()
                    .ok_or_else(|| CodegenError { message: "unknown enum layout".to_string() })?;
                let obj_val = self.codegen_expr(scrutinee, ctx, None)?.ok_or_else(|| CodegenError { message: "missing match scrutinee".to_string() })?;
                let obj_ptr = obj_val.into_pointer_value();
                let tag_val = self.load_enum_tag(obj_ptr, &enum_layout)?;
                let func = ctx.function;
                let cont_bb = self.context.append_basic_block(func, "match.cont");
                let mut default_bb = cont_bb;
                let switch_bb = self.builder.get_insert_block().unwrap();
                let mut cases: Vec<(inkwell::values::IntValue<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();
                let mut arm_blocks = Vec::new();

                for arm in arms {
                    let arm_bb = self.context.append_basic_block(func, "match.arm");
                    match &arm.pattern {
                        Pattern::Wildcard(_) => {
                            default_bb = arm_bb;
                        }
                        Pattern::Path(path, _) => {
                            let variant = path.last().ok_or_else(|| CodegenError { message: "invalid pattern".to_string() })?;
                            let vlayout = enum_layout
                                .variants
                                .get(variant)
                                .ok_or_else(|| CodegenError { message: "unknown variant".to_string() })?;
                            let tag = self.context.i32_type().const_int(vlayout.tag as u64, false);
                            cases.push((tag, arm_bb));
                        }
                        Pattern::Struct { path, .. } => {
                            let variant = path.last().ok_or_else(|| CodegenError { message: "invalid pattern".to_string() })?;
                            let vlayout = enum_layout
                                .variants
                                .get(variant)
                                .ok_or_else(|| CodegenError { message: "unknown variant".to_string() })?;
                            let tag = self.context.i32_type().const_int(vlayout.tag as u64, false);
                            cases.push((tag, arm_bb));
                        }
                    }
                    arm_blocks.push((arm_bb, arm));
                }

                self.map_builder(
                    self.builder.build_switch(tag_val, default_bb, &cases),
                    "switch",
                )?;

                let mut incoming = Vec::new();
                for (bb, arm) in arm_blocks {
                    self.builder.position_at_end(bb);
                    self.push_scope(ctx);
                    if let Pattern::Struct { path, fields, .. } = &arm.pattern {
                        let variant_name = path.last().ok_or_else(|| CodegenError { message: "invalid pattern".to_string() })?;
                        let variant = enum_layout
                            .variants
                            .get(variant_name)
                            .ok_or_else(|| CodegenError { message: "unknown variant".to_string() })?;
                        let raw_base = self.enum_raw_base_ptr(obj_ptr, enum_layout.raw_base)?;
                        for pat_field in fields {
                            let binding = match &pat_field.binding {
                                Some(b) => b,
                                None => continue,
                            };
                            let field = variant
                                .fields
                                .iter()
                                .find(|f| f.name.as_deref() == Some(pat_field.name.as_str()))
                                .ok_or_else(|| CodegenError { message: "unknown field".to_string() })?;
                            let loaded = if field.is_ptr {
                                let slot_ptr = self.enum_ptr_slot(obj_ptr, field.ptr_index)?;
                                self.map_builder(
                                    self.builder.build_load(self.context.ptr_type(inkwell::AddressSpace::default()), slot_ptr, "enum_field"),
                                    "load",
                                )?
                            } else {
                                let field_ptr = self.enum_raw_field_ptr(raw_base, field.raw_offset, field.llvm_ty)?;
                                let raw = self.map_builder(self.builder.build_load(field.llvm_ty, field_ptr, "enum_field"), "load")?;
                                if self.is_bool_type(&field.ty) {
                                    let zero = self.context.i8_type().const_zero();
                                    let cmp = self.map_builder(
                                        self.builder
                                            .build_int_compare(inkwell::IntPredicate::NE, raw.into_int_value(), zero, "bool"),
                                        "icmp",
                                    )?;
                                    cmp.into()
                                } else {
                                    raw
                                }
                            };
                            let local = self.create_local_storage_in_ctx(ctx.function, binding, &field.ty, ctx.root_base, ctx.next_root)?;
                            self.store_local(&local, loaded)?;
                            if let Some(scope) = ctx.scopes.last_mut() {
                                scope.push(binding.clone());
                            }
                            ctx.locals.insert(binding.clone(), local);
                            if self.is_gc_ref_type(&field.ty) {
                                ctx.next_root += 1;
                            }
                        }
                    }
                    let value = self.codegen_expr(&arm.body, ctx, expected)?;
                    let value_bb = self.builder.get_insert_block().unwrap();
                    let terminated = value_bb.get_terminator().is_some();
                    if !terminated {
                        self.map_builder(self.builder.build_unconditional_branch(cont_bb), "br")?;
                    }
                    if let Some(v) = value {
                        incoming.push((v, value_bb));
                    }
                    self.pop_scope(ctx);
                }

                self.builder.position_at_end(cont_bb);
                if default_bb == cont_bb && !incoming.is_empty() {
                    let zero = self.zero_value(incoming[0].0.get_type())?;
                    incoming.push((zero, switch_bb));
                }
                if !incoming.is_empty() {
                    let ty = incoming[0].0.get_type();
                    let phi = self.map_builder(self.builder.build_phi(ty, "matchtmp"), "phi")?;
                    let mut pairs: Vec<(&dyn BasicValue<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();
                    for (v, bb) in &incoming {
                        pairs.push((v as &dyn BasicValue, *bb));
                    }
                    phi.add_incoming(&pairs);
                    Ok(Some(phi.as_basic_value()))
                } else {
                    Ok(None)
                }
            }
            Expr::Call { callee, args, .. } => {
                if let Expr::Path(path, _) = &**callee {
                    if path.len() >= 2 {
                        let (enum_name, variant_name) = enum_path_and_variant(path);
                        if let Some(layout) = self.enums.get(&enum_name).cloned() {
                            let variant = layout
                                .variants
                                .get(&variant_name)
                                .cloned()
                                .ok_or_else(|| CodegenError { message: "unknown enum variant".to_string() })?;
                            if variant.fields.iter().any(|f| f.name.is_some()) {
                                return Err(CodegenError { message: "struct variant requires field syntax".to_string() });
                            }
                            let size_val = self.context.i64_type().const_int(layout.size, false);
                            let meta_ptr = self.map_builder(
                                self.builder.build_bit_cast(layout.meta, self.context.ptr_type(inkwell::AddressSpace::default()), "meta"),
                                "bitcast",
                            )?;
                            let args_alloc = &[
                                ctx.thread.into(),
                                meta_ptr.into(),
                                size_val.into(),
                            ];
                            let call = self.map_builder(self.builder.build_call(self.gc_allocate, args_alloc, "alloc_enum"), "call")?;
                            let obj_ptr = call
                                .try_as_basic_value()
                                .basic()
                                .ok_or_else(|| CodegenError { message: "alloc returned void".to_string() })?
                                .into_pointer_value();
                            self.init_header(obj_ptr, meta_ptr.into_pointer_value())?;

                            let raw_base = self.enum_raw_base_ptr(obj_ptr, layout.raw_base)?;
                            let tag_ptr = self.map_builder(
                                self.builder.build_bit_cast(raw_base, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "tag"),
                                "bitcast",
                            )?;
                            let tag_val = self.context.i32_type().const_int(variant.tag as u64, false);
                            self.map_builder(self.builder.build_store(tag_ptr.into_pointer_value(), tag_val), "store")?;

                            if args.len() != variant.fields.len() {
                                return Err(CodegenError { message: "arg count mismatch".to_string() });
                            }
                            for (i, arg) in args.iter().enumerate() {
                                let arg_val = self.codegen_expr(arg, ctx, None)?.ok_or_else(|| CodegenError { message: "missing arg".to_string() })?;
                                let arg_layout = variant.fields.get(i).ok_or_else(|| CodegenError { message: "arg count mismatch".to_string() })?;
                                if arg_layout.is_ptr {
                                    let slot_ptr = self.enum_ptr_slot(obj_ptr, arg_layout.ptr_index)?;
                                    self.map_builder(self.builder.build_store(slot_ptr, arg_val), "store")?;
                                } else {
                                    let field_ptr = self.enum_raw_field_ptr(raw_base, arg_layout.raw_offset, arg_layout.llvm_ty)?;
                                    let store_val = if self.is_bool_type(&arg_layout.ty) {
                                        let zext = self.map_builder(
                                            self.builder.build_int_z_extend(arg_val.into_int_value(), self.context.i8_type(), "boolz"),
                                            "zext",
                                        )?;
                                        zext.into()
                                    } else {
                                        arg_val
                                    };
                                    self.map_builder(self.builder.build_store(field_ptr, store_val), "store")?;
                                }
                            }
                            self.emit_pollcheck(ctx)?;
                            return Ok(Some(obj_ptr.into()));
                        }
                    }
                }

                let (func, pass_thread) = match &**callee {
                    Expr::Path(path, _) => {
                        let key = path_to_string(path);
                        if let Some(f) = self.functions.get(&key) {
                            (*f, true)
                        } else if let Some(f) = self.externs.get(&key) {
                            (*f, false)
                        } else {
                            return Err(CodegenError { message: "unknown function".to_string() });
                        }
                    }
                    _ => return Err(CodegenError { message: "callee must be path".to_string() }),
                };
                let mut arg_vals: Vec<BasicMetadataValueEnum> = Vec::new();
                if pass_thread {
                    arg_vals.push(ctx.thread.into());
                }
                for arg in args {
                    let v = self.codegen_expr(arg, ctx, None)?.ok_or_else(|| CodegenError { message: "missing arg".to_string() })?;
                    arg_vals.push(v.into());
                }
                let call = self.map_builder(self.builder.build_call(func, &arg_vals, "call"), "call")?;
                self.emit_pollcheck(ctx)?;
                Ok(call.try_as_basic_value().basic())
            }
            Expr::StructLit { path, fields, .. } => {
                if path.len() >= 2 {
                    let (enum_name, variant_name) = enum_path_and_variant(path);
                    if let Some(layout) = self.enums.get(&enum_name).cloned() {
                        let variant = layout
                            .variants
                            .get(&variant_name)
                            .cloned()
                            .ok_or_else(|| CodegenError { message: "unknown enum variant".to_string() })?;
                        if variant.fields.iter().all(|f| f.name.is_some()) {
                            let size_val = self.context.i64_type().const_int(layout.size, false);
                            let meta_ptr = self.map_builder(
                                self.builder.build_bit_cast(layout.meta, self.context.ptr_type(inkwell::AddressSpace::default()), "meta"),
                                "bitcast",
                            )?;
                            let args_alloc = &[
                                ctx.thread.into(),
                                meta_ptr.into(),
                                size_val.into(),
                            ];
                            let call = self.map_builder(self.builder.build_call(self.gc_allocate, args_alloc, "alloc_enum"), "call")?;
                            let obj_ptr = call
                                .try_as_basic_value()
                                .basic()
                                .ok_or_else(|| CodegenError { message: "alloc returned void".to_string() })?
                                .into_pointer_value();
                            self.init_header(obj_ptr, meta_ptr.into_pointer_value())?;

                            let raw_base = self.enum_raw_base_ptr(obj_ptr, layout.raw_base)?;
                            let tag_ptr = self.map_builder(
                                self.builder.build_bit_cast(raw_base, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "tag"),
                                "bitcast",
                            )?;
                            let tag_val = self.context.i32_type().const_int(variant.tag as u64, false);
                            self.map_builder(self.builder.build_store(tag_ptr.into_pointer_value(), tag_val), "store")?;

                            for (field_name, field_expr) in fields {
                                let field = variant
                                    .fields
                                    .iter()
                                    .find(|f| f.name.as_deref() == Some(field_name.as_str()))
                                    .cloned()
                                    .ok_or_else(|| CodegenError { message: "unknown field".to_string() })?;
                                let value = self.codegen_expr(field_expr, ctx, Some(&field.ty))?.ok_or_else(|| CodegenError { message: "missing field value".to_string() })?;
                                if field.is_ptr {
                                    let slot_ptr = self.enum_ptr_slot(obj_ptr, field.ptr_index)?;
                                    self.map_builder(self.builder.build_store(slot_ptr, value), "store")?;
                                } else {
                                    let field_ptr = self.enum_raw_field_ptr(raw_base, field.raw_offset, field.llvm_ty)?;
                                    let store_val = if self.is_bool_type(&field.ty) {
                                        let zext = self.map_builder(
                                            self.builder.build_int_z_extend(value.into_int_value(), self.context.i8_type(), "boolz"),
                                            "zext",
                                        )?;
                                        zext.into()
                                    } else {
                                        value
                                    };
                                    self.map_builder(self.builder.build_store(field_ptr, store_val), "store")?;
                                }
                            }
                            self.emit_pollcheck(ctx)?;
                            return Ok(Some(obj_ptr.into()));
                        }
                    }
                }

                let name = path_to_string(path);
                let layout = self
                    .structs
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| CodegenError { message: "unknown struct".to_string() })?;
                let size_val = self.context.i64_type().const_int(layout.size, false);
                let meta_ptr = self.map_builder(
                    self.builder.build_bit_cast(layout.meta, self.context.ptr_type(inkwell::AddressSpace::default()), "meta"),
                    "bitcast",
                )?;
                let args = &[
                    ctx.thread.into(),
                    meta_ptr.into(),
                    size_val.into(),
                ];
                let call = self.map_builder(self.builder.build_call(self.gc_allocate, args, "alloc"), "call")?;
                let obj_ptr = call
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError { message: "alloc returned void".to_string() })?
                    .into_pointer_value();
                self.init_header(obj_ptr, meta_ptr.into_pointer_value())?;
                for (field_name, field_expr) in fields {
                    let field = layout
                        .fields
                        .iter()
                        .find(|f| f.name == *field_name)
                        .cloned()
                        .ok_or_else(|| CodegenError { message: "unknown field".to_string() })?;
                    let value = self.codegen_expr(field_expr, ctx, Some(&field.ty))?.ok_or_else(|| CodegenError { message: "missing field value".to_string() })?;
                    let field_ptr = self.field_ptr(obj_ptr, &field)?;
                    let store_val = if self.is_bool_type(&field.ty) {
                        let zext = self.map_builder(
                            self.builder.build_int_z_extend(value.into_int_value(), self.context.i8_type(), "boolz"),
                            "zext",
                        )?;
                        zext.into()
                    } else {
                        value
                    };
                    self.map_builder(self.builder.build_store(field_ptr, store_val), "store")?;
                }
                self.emit_pollcheck(ctx)?;
                Ok(Some(obj_ptr.into()))
            }
            Expr::Tuple { items, .. } => {
                let expected = expected.ok_or_else(|| CodegenError { message: "tuple requires type context".to_string() })?;
                let elem_types = match expected {
                    Type::Tuple(types) => types,
                    _ => return Err(CodegenError { message: "tuple requires tuple type".to_string() }),
                };
                if elem_types.len() != items.len() {
                    return Err(CodegenError { message: "tuple arity mismatch".to_string() });
                }
                let key = self.ensure_tuple_layout(elem_types)?;
                let layout = self
                    .tuples
                    .get(&key)
                    .ok_or_else(|| CodegenError { message: "unknown tuple layout".to_string() })?
                    .clone();
                let size_val = self.context.i64_type().const_int(layout.size, false);
                let meta_ptr = self.map_builder(
                    self.builder.build_bit_cast(layout.meta, self.context.ptr_type(inkwell::AddressSpace::default()), "meta"),
                    "bitcast",
                )?;
                let args = &[
                    ctx.thread.into(),
                    meta_ptr.into(),
                    size_val.into(),
                ];
                let call = self.map_builder(self.builder.build_call(self.gc_allocate, args, "alloc_tuple"), "call")?;
                let obj_ptr = call
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError { message: "alloc returned void".to_string() })?
                    .into_pointer_value();
                self.init_header(obj_ptr, meta_ptr.into_pointer_value())?;
                for (i, item) in items.iter().enumerate() {
                    let field_name = i.to_string();
                    let field = layout
                        .fields
                        .iter()
                        .find(|f| f.name == field_name)
                        .cloned()
                        .ok_or_else(|| CodegenError { message: "unknown tuple field".to_string() })?;
                    let value = self.codegen_expr(item, ctx, Some(&field.ty))?.ok_or_else(|| CodegenError { message: "missing tuple value".to_string() })?;
                    let field_ptr = self.field_ptr(obj_ptr, &field)?;
                    let store_val = if self.is_bool_type(&field.ty) {
                        let zext = self.map_builder(
                            self.builder.build_int_z_extend(value.into_int_value(), self.context.i8_type(), "boolz"),
                            "zext",
                        )?;
                        zext.into()
                    } else {
                        value
                    };
                    self.map_builder(self.builder.build_store(field_ptr, store_val), "store")?;
                    if field.is_ptr {
                        let args = &[ctx.thread.into(), obj_ptr.into(), value.into()];
                        let _ = self.map_builder(self.builder.build_call(self.gc_write_barrier, args, "wb"), "call")?;
                    }
                }
                self.emit_pollcheck(ctx)?;
                Ok(Some(obj_ptr.into()))
            }
            Expr::Block(block) => self.codegen_block_value(block, ctx, expected),
            Expr::Break { .. } => {
                let (_, break_bb) = ctx.loop_stack.last().ok_or_else(|| CodegenError { message: "break outside of loop".to_string() })?;
                let break_bb = *break_bb;
                self.map_builder(self.builder.build_unconditional_branch(break_bb), "br")?;
                Ok(None)
            }
            Expr::Continue { .. } => {
                let (continue_bb, _) = ctx.loop_stack.last().ok_or_else(|| CodegenError { message: "continue outside of loop".to_string() })?;
                let continue_bb = *continue_bb;
                self.emit_pollcheck(ctx)?;
                self.map_builder(self.builder.build_unconditional_branch(continue_bb), "br")?;
                Ok(None)
            }
            _ => Err(CodegenError { message: "unsupported expression".to_string() }),
        }
    }

    fn codegen_block_value(
        &mut self,
        block: &Block,
        ctx: &mut FnCtx<'ctx>,
        expected: Option<&Type>,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        self.push_scope(ctx);
        let mut last = None;
        let mut terminated = false;
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => {
                    last = self.codegen_expr(expr, ctx, expected)?;
                }
                Stmt::Return(expr, _) => {
                    let val = if let Some(expr) = expr {
                        self.codegen_expr(expr, ctx, expected)?
                    } else {
                        None
                    };
                    if let Some(v) = val {
                        self.emit_epilogue(ctx)?;
                        self.map_builder(self.builder.build_return(Some(&v)), "return")?;
                    } else {
                        let zero = self.context.i64_type().const_int(0, false);
                        self.emit_epilogue(ctx)?;
                        self.map_builder(self.builder.build_return(Some(&zero)), "return")?;
                    }
                    terminated = true;
                    break;
                }
            }
        }
        if !terminated {
            if let Some(tail) = &block.tail {
                last = self.codegen_expr(tail, ctx, expected)?;
            }
        }
        self.pop_scope(ctx);
        Ok(last)
    }

    fn codegen_binary(
        &self,
        op: &BinaryOp,
        left: BasicValueEnum<'ctx>,
        right: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        if left.is_float_value() || right.is_float_value() {
            let l = left.into_float_value();
            let r = right.into_float_value();
            let res = match op {
                BinaryOp::Add => self.map_builder(self.builder.build_float_add(l, r, "fadd"), "fadd")?.into(),
                BinaryOp::Sub => self.map_builder(self.builder.build_float_sub(l, r, "fsub"), "fsub")?.into(),
                BinaryOp::Mul => self.map_builder(self.builder.build_float_mul(l, r, "fmul"), "fmul")?.into(),
                BinaryOp::Div => self.map_builder(self.builder.build_float_div(l, r, "fdiv"), "fdiv")?.into(),
                BinaryOp::Rem => self.map_builder(self.builder.build_float_rem(l, r, "frem"), "frem")?.into(),
                BinaryOp::Eq => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::OEQ, l, r, "feq"),
                    "feq",
                )?
                .into(),
                BinaryOp::NotEq => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::ONE, l, r, "fne"),
                    "fne",
                )?
                .into(),
                BinaryOp::Lt => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::OLT, l, r, "flt"),
                    "flt",
                )?
                .into(),
                BinaryOp::LtEq => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::OLE, l, r, "fle"),
                    "fle",
                )?
                .into(),
                BinaryOp::Gt => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::OGT, l, r, "fgt"),
                    "fgt",
                )?
                .into(),
                BinaryOp::GtEq => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::OGE, l, r, "fge"),
                    "fge",
                )?
                .into(),
                BinaryOp::AndAnd | BinaryOp::OrOr => {
                    return Err(CodegenError { message: "invalid float logical op".to_string() })
                }
            };
            Ok(res)
        } else {
            let l = left.into_int_value();
            let r = right.into_int_value();
            let res = match op {
                BinaryOp::Add => self.map_builder(self.builder.build_int_add(l, r, "add"), "add")?,
                BinaryOp::Sub => self.map_builder(self.builder.build_int_sub(l, r, "sub"), "sub")?,
                BinaryOp::Mul => self.map_builder(self.builder.build_int_mul(l, r, "mul"), "mul")?,
                BinaryOp::Div => self.map_builder(self.builder.build_int_signed_div(l, r, "div"), "div")?,
                BinaryOp::Rem => self.map_builder(self.builder.build_int_signed_rem(l, r, "rem"), "rem")?,
                BinaryOp::Eq => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::EQ, l, r, "eq"), "eq")?,
                BinaryOp::NotEq => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::NE, l, r, "ne"), "ne")?,
                BinaryOp::Lt => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::SLT, l, r, "lt"), "lt")?,
                BinaryOp::LtEq => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::SLE, l, r, "le"), "le")?,
                BinaryOp::Gt => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::SGT, l, r, "gt"), "gt")?,
                BinaryOp::GtEq => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::SGE, l, r, "ge"), "ge")?,
                BinaryOp::AndAnd => self.map_builder(self.builder.build_and(l, r, "and"), "and")?,
                BinaryOp::OrOr => self.map_builder(self.builder.build_or(l, r, "or"), "or")?,
            };
            Ok(res.into())
        }
    }

    fn llvm_type(&self, ty: &Type) -> Result<BasicTypeEnum<'ctx>, CodegenError> {
        match ty {
            Type::Path(path, _) => {
                let name = path.last().ok_or_else(|| CodegenError { message: "empty type".to_string() })?;
                match name.as_str() {
                    "I64" => Ok(self.context.i64_type().into()),
                    "F32" => Ok(self.context.f32_type().into()),
                    "F64" => Ok(self.context.f64_type().into()),
                    "Bool" => Ok(self.context.bool_type().into()),
                    "Unit" => Ok(self.context.i64_type().into()),
                    "String" => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
                    _ if self.is_gc_ref_type(ty) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
                    _ => Err(CodegenError { message: "unsupported type".to_string() }),
                }
            }
            Type::RawPointer(_) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
            Type::Tuple(_) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
        }
    }

    fn const_i64(&self, text: &str) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        let cleaned = text.replace('_', "");
        let value = cleaned.parse::<i64>().map_err(|_| CodegenError { message: "invalid int literal".to_string() })?;
        Ok(self.context.i64_type().const_int(value as u64, true).into())
    }

    fn create_entry_alloca(
        &self,
        function: FunctionValue<'ctx>,
        name: &str,
        ty: &Type,
    ) -> Result<Local<'ctx>, CodegenError> {
        let builder = self.context.create_builder();
        let entry = function.get_first_basic_block().unwrap();
        match entry.get_first_instruction() {
            Some(inst) => builder.position_before(&inst),
            None => builder.position_at_end(entry),
        }
        let llvm_ty = self.llvm_type(ty)?;
        let ptr = self.map_builder(builder.build_alloca(llvm_ty, name), "alloca")?;
        Ok(Local {
            ptr,
            ty: llvm_ty,
            lang_ty: ty.clone(),
        })
    }

    fn create_local_storage_in_ctx(
        &self,
        function: FunctionValue<'ctx>,
        name: &str,
        ty: &Type,
        root_base: Option<PointerValue<'ctx>>,
        root_index: usize,
    ) -> Result<Local<'ctx>, CodegenError> {
        if self.is_gc_ref_type(ty) {
            let root_base = root_base.ok_or_else(|| CodegenError { message: "missing root base".to_string() })?;
            let idx = self.context.i32_type().const_int(root_index as u64, false);
            let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
            let slot_ptr = unsafe {
                self.map_builder(
                    self.builder.build_gep(ptr_ty, root_base, &[idx], "root_slot"),
                    "gep",
                )?
            };
            Ok(Local {
                ptr: slot_ptr,
                ty: ptr_ty.into(),
                lang_ty: ty.clone(),
            })
        } else {
            self.create_entry_alloca(function, name, ty)
        }
    }

    fn load_local(&self, local: &Local<'ctx>, name: &str) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        let loaded = self.map_builder(self.builder.build_load(local.ty, local.ptr, name), "load")?;
        if self.is_gc_ref_type(&local.lang_ty) {
            if let Some(inst) = loaded.as_instruction_value() {
                inst.set_volatile(true)
                    .map_err(|e| CodegenError { message: format!("volatile load failed: {e}") })?;
            }
        }
        Ok(loaded)
    }

    fn store_local(&self, local: &Local<'ctx>, value: BasicValueEnum<'ctx>) -> Result<(), CodegenError> {
        let inst = self.map_builder(self.builder.build_store(local.ptr, value), "store")?;
        if self.is_gc_ref_type(&local.lang_ty) {
            inst.set_volatile(true)
                .map_err(|e| CodegenError { message: format!("volatile store failed: {e}") })?;
        }
        Ok(())
    }

    fn push_scope(&self, ctx: &mut FnCtx<'ctx>) {
        ctx.scopes.push(Vec::new());
    }

    fn pop_scope(&self, ctx: &mut FnCtx<'ctx>) {
        if let Some(names) = ctx.scopes.pop() {
            for name in names {
                ctx.locals.remove(&name);
            }
        }
    }

    fn map_builder<T>(&self, res: Result<T, BuilderError>, label: &str) -> Result<T, CodegenError> {
        res.map_err(|e| CodegenError { message: format!("{label} failed: {e}") })
    }

    fn is_gc_ref_type(&self, ty: &Type) -> bool {
        match ty {
            Type::RawPointer(_) => false,
            Type::Tuple(_) => true,
            Type::Path(path, _) => {
                let name = path.last().map(|s| s.as_str()).unwrap_or("");
                !matches!(
                    name,
                    "I8" | "I16" | "I32" | "I64" | "U8" | "U16" | "U32" | "U64" | "F32" | "F64" | "Bool" | "Unit" | "String"
                )
            }
        }
    }

    fn count_param_roots(&self, params: &[Param]) -> usize {
        params.iter().filter(|p| self.is_gc_ref_type(&p.ty)).count()
    }

    fn count_roots_in_fn(&self, f: &FnDecl) -> usize {
        let mut count = self.count_param_roots(&f.params);
        count += self.count_roots_in_block(&f.body);
        count
    }

    fn count_roots_in_block(&self, block: &Block) -> usize {
        let mut count = 0;
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => count += self.count_roots_in_expr(expr),
                Stmt::Return(expr, _) => {
                    if let Some(expr) = expr {
                        count += self.count_roots_in_expr(expr);
                    }
                }
            }
        }
        if let Some(tail) = &block.tail {
            count += self.count_roots_in_expr(tail);
        }
        count
    }

    fn count_roots_in_expr(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Let { ty, value, .. } => {
                let mut count = 0;
                if let Some(ty) = ty {
                    if self.is_gc_ref_type(ty) {
                        count += 1;
                    }
                }
                count + self.count_roots_in_expr(value)
            }
            Expr::If { cond, then_branch, else_branch, .. } => {
                let mut count = self.count_roots_in_expr(cond);
                count += self.count_roots_in_block(then_branch);
                if let Some(else_branch) = else_branch {
                    count += self.count_roots_in_block(else_branch);
                }
                count
            }
            Expr::While { cond, body, .. } => self.count_roots_in_expr(cond) + self.count_roots_in_block(body),
            Expr::Match { scrutinee, arms, .. } => {
                let mut count = self.count_roots_in_expr(scrutinee);
                for arm in arms {
                    count += self.count_roots_in_pattern(&arm.pattern);
                    count += self.count_roots_in_expr(&arm.body);
                }
                count
            }
            Expr::Assign { target, value, .. } => {
                self.count_roots_in_expr(target) + self.count_roots_in_expr(value)
            }
            Expr::Binary { left, right, .. } => self.count_roots_in_expr(left) + self.count_roots_in_expr(right),
            Expr::Unary { expr, .. } => self.count_roots_in_expr(expr),
            Expr::Call { callee, args, .. } => {
                let mut count = self.count_roots_in_expr(callee);
                for arg in args {
                    count += self.count_roots_in_expr(arg);
                }
                count
            }
            Expr::Field { base, .. } => self.count_roots_in_expr(base),
            Expr::StructLit { fields, .. } => fields.iter().map(|(_, e)| self.count_roots_in_expr(e)).sum(),
            Expr::Tuple { items, .. } => items.iter().map(|e| self.count_roots_in_expr(e)).sum(),
            Expr::Block(block) => self.count_roots_in_block(block),
            _ => 0,
        }
    }

    fn count_roots_in_pattern(&self, pattern: &Pattern) -> usize {
        match pattern {
            Pattern::Struct { path, fields, .. } => {
                if path.len() < 2 {
                    return 0;
                }
                let (enum_name, variant_name) = enum_path_and_variant(path);
                let layout = match self.enums.get(&enum_name) {
                    Some(l) => l,
                    None => return 0,
                };
                let variant = match layout.variants.get(&variant_name) {
                    Some(v) => v,
                    None => return 0,
                };
                let mut count = 0;
                for pat_field in fields {
                    if pat_field.binding.is_none() {
                        continue;
                    }
                    if let Some(field) = variant
                        .fields
                        .iter()
                        .find(|f| f.name.as_deref() == Some(pat_field.name.as_str()))
                    {
                        if field.is_ptr {
                            count += 1;
                        }
                    }
                }
                count
            }
            _ => 0,
        }
    }

    fn emit_prologue(
        &mut self,
        thread: PointerValue<'ctx>,
        fn_name: &str,
        num_roots: usize,
        frame_ty: StructType<'ctx>,
    ) -> Result<(PointerValue<'ctx>, PointerValue<'ctx>, Option<PointerValue<'ctx>>), CodegenError> {
        let frame_origin = self.create_frame_origin(fn_name, num_roots)?;
        let frame_ptr = self.map_builder(self.builder.build_alloca(frame_ty, "frame"), "alloca")?;

        let parent_ptr = self.map_builder(
            self.builder.build_struct_gep(frame_ty, frame_ptr, 0, "frame.parent"),
            "gep",
        )?;
        let origin_ptr = self.map_builder(
            self.builder.build_struct_gep(frame_ty, frame_ptr, 1, "frame.origin"),
            "gep",
        )?;

        let top_frame_ptr = self.thread_top_frame_ptr(thread)?;
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let parent = self.map_builder(self.builder.build_load(ptr_ty, top_frame_ptr, "parent"), "load")?;

        self.map_builder(self.builder.build_store(parent_ptr, parent), "store")?;
        self.map_builder(self.builder.build_store(origin_ptr, frame_origin), "store")?;

        let root_base = if num_roots > 0 {
            let roots_ptr = self.map_builder(
                self.builder.build_struct_gep(frame_ty, frame_ptr, 2, "frame.roots"),
                "gep",
            )?;
            let roots_ty = ptr_ty.array_type(num_roots as u32);
            let zero = self.context.i32_type().const_zero();
            let root0 = unsafe {
                self.map_builder(
                    self.builder.build_gep(roots_ty, roots_ptr, &[zero, zero], "root0"),
                    "gep",
                )?
            };
            let ptr_size = std::mem::size_of::<usize>() as u64;
            let size = self.context.i64_type().const_int(ptr_size * num_roots as u64, false);
            let val = self.context.i8_type().const_int(0, false);
            let dest = self.map_builder(
                self.builder.build_bit_cast(root0, ptr_ty, "root0_cast"),
                "bitcast",
            )?;
            let dest_ptr = dest.into_pointer_value();
            let _ = self.map_builder(self.builder.build_memset(dest_ptr, 8, val, size), "memset")?;
            Some(root0)
        } else {
            None
        };

        self.map_builder(self.builder.build_store(top_frame_ptr, frame_ptr), "store")?;

        Ok((frame_ptr, frame_origin, root_base))
    }

    fn emit_epilogue(&self, ctx: &FnCtx<'ctx>) -> Result<(), CodegenError> {
        let parent_ptr = self.map_builder(
            self.builder.build_struct_gep(ctx.frame_ty, ctx.frame_ptr, 0, "frame.parent"),
            "gep",
        )?;
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let parent = self.map_builder(self.builder.build_load(ptr_ty, parent_ptr, "load"), "load")?;
        let top_frame_ptr = self.thread_top_frame_ptr(ctx.thread)?;
        self.map_builder(self.builder.build_store(top_frame_ptr, parent), "store")?;
        Ok(())
    }

    fn emit_pollcheck(&self, ctx: &FnCtx<'ctx>) -> Result<(), CodegenError> {
        let state_ptr = self.thread_state_ptr(ctx.thread)?;
        let state = self.map_builder(
            self.builder.build_load(self.context.i32_type(), state_ptr, "gc_state"),
            "load",
        )?;
        let zero = self.context.i32_type().const_zero();
        let needs = self.map_builder(
            self.builder
                .build_int_compare(inkwell::IntPredicate::NE, state.into_int_value(), zero, "needs_gc"),
            "icmp",
        )?;

        let func = ctx.function;
        let slow_bb = self.context.append_basic_block(func, "poll.slow");
        let cont_bb = self.context.append_basic_block(func, "poll.cont");
        self.map_builder(
            self.builder
                .build_conditional_branch(needs, slow_bb, cont_bb),
            "condbr",
        )?;

        self.builder.position_at_end(slow_bb);
        let args = &[ctx.thread.into(), ctx.frame_origin.into()];
        let _ = self.map_builder(self.builder.build_call(self.gc_pollcheck, args, "poll"), "call")?;
        self.map_builder(self.builder.build_unconditional_branch(cont_bb), "br")?;

        self.builder.position_at_end(cont_bb);
        Ok(())
    }

    fn thread_top_frame_ptr(&self, thread: PointerValue<'ctx>) -> Result<PointerValue<'ctx>, CodegenError> {
        self.map_builder(
            self.builder.build_struct_gep(self.thread_struct_ty, thread, 0, "thread.top"),
            "gep",
        )
    }

    fn thread_state_ptr(&self, thread: PointerValue<'ctx>) -> Result<PointerValue<'ctx>, CodegenError> {
        self.map_builder(
            self.builder.build_struct_gep(self.thread_struct_ty, thread, 1, "thread.state"),
            "gep",
        )
    }

    fn frame_type(&self, num_roots: usize) -> StructType<'ctx> {
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let roots_ty = ptr_ty.array_type(num_roots as u32);
        self.context.struct_type(&[ptr_ty.into(), ptr_ty.into(), roots_ty.into()], false)
    }

    fn create_frame_origin(&self, fn_name: &str, num_roots: usize) -> Result<PointerValue<'ctx>, CodegenError> {
        let i32_ty = self.context.i32_type();
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());

        let name_const = self.context.const_string(fn_name.as_bytes(), true);
        let sym = mangle_name(fn_name);
        let name_global = self.module.add_global(name_const.get_type(), None, &format!("__fn_name_{}", sym));
        name_global.set_initializer(&name_const);
        let zero = i32_ty.const_zero();
        let name_ptr = unsafe { name_global.as_pointer_value().const_gep(name_const.get_type(), &[zero, zero]) };

        let origin_ty = self.context.struct_type(&[i32_ty.into(), ptr_ty.into()], false);
        let origin_global = self.module.add_global(origin_ty, None, &format!("__frame_origin_{}", sym));
        let num = i32_ty.const_int(num_roots as u64, false);
        let init = origin_ty.const_named_struct(&[num.into(), name_ptr.into()]);
        origin_global.set_initializer(&init);
        Ok(origin_global.as_pointer_value())
    }

    fn field_llvm_type(&self, ty: &Type) -> Result<BasicTypeEnum<'ctx>, CodegenError> {
        match ty {
            Type::Path(path, _) => {
                let name = path.last().ok_or_else(|| CodegenError { message: "empty type".to_string() })?;
                match name.as_str() {
                    "I8" | "U8" => Ok(self.context.i8_type().into()),
                    "I16" | "U16" => Ok(self.context.i16_type().into()),
                    "I32" | "U32" => Ok(self.context.i32_type().into()),
                    "I64" | "U64" => Ok(self.context.i64_type().into()),
                    "F32" => Ok(self.context.f32_type().into()),
                    "F64" => Ok(self.context.f64_type().into()),
                    "Bool" => Ok(self.context.i8_type().into()),
                    "String" => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
                    "Unit" => Ok(self.context.i64_type().into()),
                    _ if self.is_gc_ref_type(ty) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
                    _ => Err(CodegenError { message: format!("unsupported field type '{}'", name) }),
                }
            }
            Type::RawPointer(_) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
            Type::Tuple(_) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
        }
    }

    fn zero_value(&self, ty: BasicTypeEnum<'ctx>) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        match ty {
            BasicTypeEnum::IntType(t) => Ok(t.const_zero().into()),
            BasicTypeEnum::FloatType(t) => Ok(t.const_zero().into()),
            BasicTypeEnum::PointerType(t) => Ok(t.const_null().into()),
            _ => Err(CodegenError { message: "unsupported zero value".to_string() }),
        }
    }

    fn struct_name_from_type(&self, ty: &Type) -> Result<String, CodegenError> {
        match ty {
            Type::Path(path, _) => Ok(path_to_string(path)),
            _ => Err(CodegenError { message: "expected struct type".to_string() }),
        }
    }

    fn layout_for_type(&mut self, ty: &Type) -> Result<StructLayout<'ctx>, CodegenError> {
        match ty {
            Type::Path(path, _) => {
                let name = path_to_string(path);
                self.structs
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| CodegenError { message: "unknown struct layout".to_string() })
            }
            Type::Tuple(items) => {
                let key = self.ensure_tuple_layout(items)?;
                self.tuples
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| CodegenError { message: "unknown tuple layout".to_string() })
            }
            _ => Err(CodegenError { message: "expected struct type".to_string() }),
        }
    }

    fn is_bool_type(&self, ty: &Type) -> bool {
        matches!(ty, Type::Path(path, _) if path.last().map(|s| s.as_str()) == Some("Bool"))
    }

    fn resolve_struct_base(
        &mut self,
        base: &Expr,
        ctx: &FnCtx<'ctx>,
    ) -> Result<(PointerValue<'ctx>, StructLayout<'ctx>), CodegenError> {
        match base {
            Expr::Path(path, _) => {
                let var = path.last().ok_or_else(|| CodegenError { message: "empty base".to_string() })?;
                let local = ctx.locals.get(var).ok_or_else(|| CodegenError { message: "unknown local".to_string() })?;
                let layout = self.layout_for_type(&local.lang_ty)?;
                let obj_ptr = self.load_local(local, "obj")?;
                Ok((obj_ptr.into_pointer_value(), layout))
            }
            Expr::Field { base: inner, name, .. } => {
                let (obj_ptr, _layout, field) = self.resolve_field(inner, name, ctx)?;
                if !self.is_gc_ref_type(&field.ty) {
                    return Err(CodegenError { message: "field base must be gc ref".to_string() });
                }
                let field_ptr = self.field_ptr(obj_ptr, &field)?;
                let loaded = self.map_builder(self.builder.build_load(field.llvm_ty, field_ptr, "field_obj"), "load")?;
                let layout = self.layout_for_type(&field.ty)?;
                Ok((loaded.into_pointer_value(), layout))
            }
            _ => Err(CodegenError { message: "field base must be path or field in codegen".to_string() }),
        }
    }

    fn resolve_field(
        &mut self,
        base: &Expr,
        name: &str,
        ctx: &FnCtx<'ctx>,
    ) -> Result<(PointerValue<'ctx>, StructLayout<'ctx>, FieldLayout<'ctx>), CodegenError> {
        let (obj_ptr, layout) = self.resolve_struct_base(base, ctx)?;
        let field = layout
            .fields
            .iter()
            .find(|f| f.name == name)
            .cloned()
            .ok_or_else(|| CodegenError { message: "unknown field".to_string() })?;
        Ok((obj_ptr, layout, field))
    }

    fn field_ptr(
        &self,
        obj_ptr: PointerValue<'ctx>,
        field: &FieldLayout<'ctx>,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        let i8_ty = self.context.i8_type();
        let header_size = self.context.i64_type().const_int(16, false);
        let payload_ptr = unsafe {
            self.map_builder(
                self.builder.build_gep(i8_ty, obj_ptr, &[header_size], "payload"),
                "gep",
            )?
        };
        let offset = self.context.i64_type().const_int(field.offset, false);
        let field_i8 = unsafe {
            self.map_builder(
                self.builder.build_gep(i8_ty, payload_ptr, &[offset], "field_i8"),
                "gep",
            )?
        };
        let field_ptr_ty = field.llvm_ty.ptr_type(inkwell::AddressSpace::default());
        let cast = self.map_builder(self.builder.build_bit_cast(field_i8, field_ptr_ty, "field_cast"), "bitcast")?;
        Ok(cast.into_pointer_value())
    }

    fn enum_raw_base_ptr(&self, obj_ptr: PointerValue<'ctx>, raw_base: u64) -> Result<PointerValue<'ctx>, CodegenError> {
        let i8_ty = self.context.i8_type();
        let offset = self.context.i64_type().const_int(raw_base, false);
        let raw_ptr = unsafe {
            self.map_builder(
                self.builder.build_gep(i8_ty, obj_ptr, &[offset], "enum_raw"),
                "gep",
            )?
        };
        Ok(raw_ptr)
    }

    fn enum_ptr_slot(&self, obj_ptr: PointerValue<'ctx>, index: usize) -> Result<PointerValue<'ctx>, CodegenError> {
        let i8_ty = self.context.i8_type();
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let header = self.context.i64_type().const_int(16, false);
        let ptr_base = unsafe {
            self.map_builder(
                self.builder.build_gep(i8_ty, obj_ptr, &[header], "enum_ptrs"),
                "gep",
            )?
        };
        let ptr_ptr = self.map_builder(self.builder.build_bit_cast(ptr_base, ptr_ty.ptr_type(inkwell::AddressSpace::default()), "ptr_base"), "bitcast")?;
        let idx = self.context.i64_type().const_int(index as u64, false);
        let slot = unsafe {
            self.map_builder(
                self.builder.build_gep(ptr_ty, ptr_ptr.into_pointer_value(), &[idx], "enum_slot"),
                "gep",
            )?
        };
        Ok(slot)
    }

    fn enum_raw_field_ptr(
        &self,
        raw_base: PointerValue<'ctx>,
        offset: u64,
        ty: BasicTypeEnum<'ctx>,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        let i8_ty = self.context.i8_type();
        let off = self.context.i64_type().const_int(offset, false);
        let field_i8 = unsafe {
            self.map_builder(
                self.builder.build_gep(i8_ty, raw_base, &[off], "enum_raw_field"),
                "gep",
            )?
        };
        let field_ptr_ty = ty.ptr_type(inkwell::AddressSpace::default());
        let cast = self.map_builder(self.builder.build_bit_cast(field_i8, field_ptr_ty, "enum_raw_cast"), "bitcast")?;
        Ok(cast.into_pointer_value())
    }

    fn resolve_enum_scrutinee(
        &self,
        scrutinee: &Expr,
        ctx: &FnCtx<'ctx>,
    ) -> Result<(EnumLayout<'ctx>, PointerValue<'ctx>), CodegenError> {
        match scrutinee {
            Expr::Path(path, _) => {
                let name = path.last().ok_or_else(|| CodegenError { message: "empty scrutinee".to_string() })?;
                let local = ctx.locals.get(name).ok_or_else(|| CodegenError { message: "unknown local".to_string() })?;
                let enum_name = self.struct_name_from_type(&local.lang_ty)?;
                let layout = self.enums.get(&enum_name).cloned().ok_or_else(|| CodegenError { message: "unknown enum layout".to_string() })?;
                let obj = self.load_local(local, "enum_obj")?;
                Ok((layout, obj.into_pointer_value()))
            }
            _ => Err(CodegenError { message: "match scrutinee must be path in codegen".to_string() }),
        }
    }

    fn load_enum_tag(&self, obj_ptr: PointerValue<'ctx>, layout: &EnumLayout<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, CodegenError> {
        let raw_base = self.enum_raw_base_ptr(obj_ptr, layout.raw_base)?;
        let tag_ptr = self.map_builder(
            self.builder.build_bit_cast(raw_base, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "tag_ptr"),
            "bitcast",
        )?;
        let tag = self.map_builder(
            self.builder.build_load(self.context.i32_type(), tag_ptr.into_pointer_value(), "tag"),
            "load",
        )?;
        Ok(tag.into_int_value())
    }

    fn init_header(&self, obj_ptr: PointerValue<'ctx>, meta_ptr: PointerValue<'ctx>) -> Result<(), CodegenError> {
        let header_ptr = self.map_builder(
            self.builder.build_bit_cast(
                obj_ptr,
                self.object_header_ty.ptr_type(inkwell::AddressSpace::default()),
                "header",
            ),
            "bitcast",
        )?;
        let header_ptr = header_ptr.into_pointer_value();
        let meta_slot = self.map_builder(
            self.builder
                .build_struct_gep(self.object_header_ty, header_ptr, 0, "meta"),
            "gep",
        )?;
        let flags_slot = self.map_builder(
            self.builder
                .build_struct_gep(self.object_header_ty, header_ptr, 1, "flags"),
            "gep",
        )?;
        let aux_slot = self.map_builder(
            self.builder
                .build_struct_gep(self.object_header_ty, header_ptr, 2, "aux"),
            "gep",
        )?;
        self.map_builder(self.builder.build_store(meta_slot, meta_ptr), "store")?;
        let zero = self.context.i32_type().const_zero();
        self.map_builder(self.builder.build_store(flags_slot, zero), "store")?;
        self.map_builder(self.builder.build_store(aux_slot, zero), "store")?;
        Ok(())
    }
}

fn path_to_string(path: &[String]) -> String {
    path.join("::")
}

fn full_item_name(module_path: &[String], name: &str) -> String {
    if module_path.is_empty() {
        name.to_string()
    } else {
        let mut parts = module_path.to_vec();
        parts.push(name.to_string());
        path_to_string(&parts)
    }
}

fn enum_path_and_variant(path: &[String]) -> (String, String) {
    let variant = path.last().cloned().unwrap_or_default();
    let enum_path = path[..path.len() - 1].to_vec();
    (path_to_string(&enum_path), variant)
}

fn last_segment(full: &str) -> &str {
    full.rsplit("::").next().unwrap_or(full)
}

fn extern_symbol_name(full: &str) -> &str {
    last_segment(full)
}

fn mangle_name(name: &str) -> String {
    name.replace("::", "__")
}

fn tuple_key(items: &[Type]) -> String {
    let mut out = String::from("__tuple__");
    for (i, ty) in items.iter().enumerate() {
        if i > 0 {
            out.push('_');
        }
        out.push_str(&type_key(ty));
    }
    out
}

fn type_key(ty: &Type) -> String {
    match ty {
        Type::Path(path, _) => mangle_name(&path_to_string(path)),
        Type::RawPointer(inner) => format!("ptr_{}", type_key(inner)),
        Type::Tuple(items) => {
            let mut out = String::from("tup");
            for item in items {
                out.push('_');
                out.push_str(&type_key(item));
            }
            out
        }
    }
}

struct Local<'ctx> {
    ptr: PointerValue<'ctx>,
    ty: BasicTypeEnum<'ctx>,
    lang_ty: Type,
}

struct FnCtx<'ctx> {
    function: FunctionValue<'ctx>,
    locals: HashMap<String, Local<'ctx>>,
    scopes: Vec<Vec<String>>,
    thread: PointerValue<'ctx>,
    frame_ptr: PointerValue<'ctx>,
    frame_origin: PointerValue<'ctx>,
    frame_ty: StructType<'ctx>,
    root_base: Option<PointerValue<'ctx>>,
    next_root: usize,
    loop_stack: Vec<(inkwell::basic_block::BasicBlock<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::resolve::resolve_module;
    use crate::typecheck::typecheck_module;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn compile_run(src: &str) -> i64 {
        let tokens = Lexer::new(src).lex_all().unwrap();
        let module = Parser::new(tokens).parse_module().unwrap();
        resolve_module(&module).unwrap();
        typecheck_module(&module).unwrap();
        compile_and_run(std::slice::from_ref(&module)).unwrap()
    }

    fn compile_object(src: &str) -> std::path::PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let tokens = Lexer::new(src).lex_all().unwrap();
        let module = Parser::new(tokens).parse_module().unwrap();
        resolve_module(&module).unwrap();
        typecheck_module(&module).unwrap();
        let mut path = std::env::temp_dir();
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        path.push(format!("langc_test_{id}.o"));
        compile_to_object(std::slice::from_ref(&module), &path).unwrap();
        path
    }

    fn compile_link_run(src: &str) -> i32 {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let tokens = Lexer::new(src).lex_all().unwrap();
        let module = Parser::new(tokens).parse_module().unwrap();
        resolve_module(&module).unwrap();
        typecheck_module(&module).unwrap();

        let mut obj_path = std::env::temp_dir();
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        obj_path.push(format!("langc_e2e_{id}.o"));
        compile_to_object(std::slice::from_ref(&module), &obj_path).unwrap();

        let mut exe_path = std::env::temp_dir();
        exe_path.push(format!("langc_e2e_{id}"));

        let status = std::process::Command::new("cargo")
            .arg("build")
            .arg("--quiet")
            .arg("--lib")
            .status()
            .expect("failed to build runtime lib");
        assert!(status.success());

        let target_dir = std::env::var("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
        let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
        let lib_path = target_dir.join(profile).join("liblang_runtime.a");

        let status = std::process::Command::new("cc")
            .arg(&obj_path)
            .arg(&lib_path)
            .arg("-O2")
            .arg("-o")
            .arg(&exe_path)
            .status()
            .expect("failed to invoke cc");
        assert!(status.success());

        let output = std::process::Command::new(&exe_path)
            .output()
            .expect("failed to run exe");
        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(&exe_path);
        output.status.code().unwrap_or(-1)
    }

    #[test]
    fn codegen_arithmetic() {
        let src = r#"
            fn main() -> I64 {
                let x: I64 = 4;
                x * 2 + 1
            }
        "#;
        assert_eq!(compile_run(src), 9);
    }

    #[test]
    fn codegen_if() {
        let src = r#"
            fn main() -> I64 {
                if true { 1 } else { 2 }
            }
        "#;
        assert_eq!(compile_run(src), 1);
    }

    #[test]
    fn codegen_while_loop() {
        let src = r#"
            fn main() -> I64 {
                let mut i: I64 = 0;
                let mut sum: I64 = 0;
                while i < 3 { sum = sum + i; i = i + 1; };
                sum
            }
        "#;
        assert_eq!(compile_run(src), 3);
    }

    #[test]
    fn codegen_struct_field() {
        let src = r#"
            struct User { id: I64 }
            fn main() -> I64 {
                let u: User = User { id: 7 };
                u.id
            }
        "#;
        assert_eq!(compile_run(src), 7);
    }

    #[test]
    fn codegen_nested_struct_field() {
        let src = r#"
            struct Box { value: I64 }
            struct Holder { inner: Box }
            fn main() -> I64 {
                let b: Box = Box { value: 4 };
                let h: Holder = Holder { inner: b };
                h.inner.value
            }
        "#;
        assert_eq!(compile_run(src), 4);
    }

    #[test]
    fn codegen_field_assign_gc_ref() {
        let src = r#"
            struct Box { value: I64 }
            struct Holder { inner: Box }
            fn main() -> I64 {
                let b1: Box = Box { value: 1 };
                let b2: Box = Box { value: 9 };
                let h: Holder = Holder { inner: b1 };
                h.inner = b2;
                h.inner.value
            }
        "#;
        assert_eq!(compile_run(src), 9);
    }

    #[test]
    fn codegen_aot_object() {
        let src = r#"
            fn main() -> I64 { 3 }
        "#;
        let path = compile_object(src);
        assert!(path.exists());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn codegen_e2e_jit_all_features() {
        let src = r#"
            extern fn add_i64(a: I64, b: I64) -> I64;
            extern fn null_ptr() -> RawPointer<I8>;
            extern fn ptr_is_null(p: RawPointer<I8>) -> I64;

            struct Pair { a: I64, b: I64 }
            enum Flag { On {}, Off {} }

            fn inc(x: I64) -> I64 { x + 1 }

            fn main() -> I64 {
                let mut i: I64 = 0;
                let mut sum: I64 = 0;
                while i < 3 { sum = sum + i; i = inc(i); };

                let mut p: Pair = Pair { a: sum, b: add_i64(1, 2) };
                p.b = p.b + 1;
                let mut total: I64 = p.a + p.b;

                let f: Flag = Flag::On {};
                total = match f { Flag::On {} => total + 10, Flag::Off {} => total + 1 };

                let ptr: RawPointer<I8> = null_ptr();
                if ptr_is_null(ptr) == 1 { total } else { 0 }
            }
        "#;
        assert_eq!(compile_run(src), 17);
    }

    #[test]
    fn codegen_e2e_aot_all_features() {
        let src = r#"
            extern fn add_i64(a: I64, b: I64) -> I64;
            extern fn null_ptr() -> RawPointer<I8>;
            extern fn ptr_is_null(p: RawPointer<I8>) -> I64;

            struct Pair { a: I64, b: I64 }
            enum Flag { On {}, Off {} }

            fn main() -> I64 {
                let mut i: I64 = 0;
                let mut acc: I64 = 0;
                while i < 4 { acc = acc + i; i = i + 1; };

                let mut p: Pair = Pair { a: acc, b: add_i64(11, 20) };
                p.b = p.b + 0;
                let mut total: I64 = p.a + p.b;

                let f: Flag = Flag::On {};
                total = match f { Flag::On {} => total + 1, Flag::Off {} => total + 2 };

                let extra: I64 = add_i64(1, 2) + 3;
                total = total + extra;

                let ptr: RawPointer<I8> = null_ptr();
                if ptr_is_null(ptr) == 1 { total - 2 } else { 0 }
            }
        "#;
        assert_eq!(compile_link_run(src), 42);
    }

    #[test]
    fn codegen_enum_match() {
        let src = r#"
            enum Result { Ok(I64), Err(I64) }
            fn main() -> I64 {
                let r: Result = Result::Ok(1);
                match r { Result::Ok => 5, Result::Err => 2 }
            }
        "#;
        assert_eq!(compile_run(src), 5);
    }

    #[test]
    fn codegen_enum_payload_alloc() {
        let src = r#"
            struct Box { value: I64 }
            enum Maybe { Some(Box, I64), None }
            fn main() -> I64 {
                let b: Box = Box { value: 6 };
                let m: Maybe = Maybe::Some(b, 1);
                match m { Maybe::Some => 2, Maybe::None => 3 }
            }
        "#;
        assert_eq!(compile_run(src), 2);
    }

    #[test]
    fn codegen_struct_variant_match_binding() {
        let src = r#"
            struct Box { value: I64 }
            enum Wrap { Some { inner: Box }, None {} }
            fn main() -> I64 {
                let b: Box = Box { value: 8 };
                let w: Wrap = Wrap::Some { inner: b };
                match w { Wrap::Some { inner } => inner.value, Wrap::None {} => 0 }
            }
        "#;
        assert_eq!(compile_run(src), 8);
    }

    #[test]
    fn codegen_return_struct() {
        let src = r#"
            struct Box { value: I64 }
            fn make() -> Box {
                let one: I64 = 1;
                Box { value: one }
            }
            fn main() -> I64 {
                let b: Box = make();
                b.value
            }
        "#;
        assert_eq!(compile_run(src), 1);
    }

    #[test]
    fn codegen_return_enum() {
        let src = r#"
            enum Flag { On {}, Off {} }
            fn make() -> Flag { Flag::On {} }
            fn main() -> I64 {
                let one: I64 = 1;
                let two: I64 = 2;
                let f: Flag = make();
                match f { Flag::On {} => one, Flag::Off {} => two }
            }
        "#;
        assert_eq!(compile_run(src), 1);
    }

    #[test]
    fn codegen_string_ops() {
        let src = r#"
            extern fn string_len(s: String) -> I64;
            extern fn string_eq(a: String, b: String) -> I64;
            extern fn string_concat(a: String, b: String) -> String;
            extern fn string_slice(s: String, start: I64, end_pos: I64) -> String;

            fn main() -> I64 {
                let one: I64 = 1;
                let zero: I64 = 0;
                let two: I64 = 2;
                let four: I64 = 4;
                let three: I64 = 3;
                let a: String = "hi";
                let b: String = "hi";
                let c: String = "ho";
                let ab: String = string_concat(a, c);
                let s: String = "hello";
                let sub: String = string_slice(s, one, four);
                let len_a: I64 = string_len(a);
                let len_ab: I64 = string_len(ab);
                let len_sub: I64 = string_len(sub);
                let eq1: I64 = string_eq(a, b);
                let eq2: I64 = string_eq(a, c);
                if len_a == two && len_ab == four && len_sub == three && eq1 == one && eq2 == zero { one } else { zero }
            }
        "#;
        assert_eq!(compile_run(src), 1);
    }

    #[test]
    fn codegen_read_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("langc_read_file_test.txt");
        std::fs::write(&path, b"abc").expect("write temp file");
        let path_str = path.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
            let src = format!(
            r#"
            extern fn read_file(path: String) -> String;
            extern fn string_len(s: String) -> I64;

            fn main() -> I64 {{
                let one: I64 = 1;
                let zero: I64 = 0;
                let three: I64 = 3;
                let data: String = read_file("{path}");
                if string_len(data) == three {{ one }} else {{ zero }}
            }}
        "#,
            path = path_str
        );
        assert_eq!(compile_run(&src), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn codegen_string_byte_at() {
        let src = r#"
            extern fn string_byte_at(s: String, index: I64) -> I64;
            fn main() -> I64 {
                let one: I64 = 1;
                let ninety_eight: I64 = 98;
                let s: String = "abc";
                if string_byte_at(s, one) == ninety_eight { 1 } else { 0 }
            }
        "#;
        assert_eq!(compile_run(src), 1);
    }

    #[test]
    fn codegen_vec_ptr() {
        let src = r#"
            extern fn vec_new() -> RawPointer<I8>;
            extern fn vec_len(v: RawPointer<I8>) -> I64;
            extern fn vec_get(v: RawPointer<I8>, index: I64) -> String;
            extern fn vec_push(v: RawPointer<I8>, item: String) -> I64;
            extern fn string_eq(a: String, b: String) -> I64;

            fn main() -> I64 {
                let one: I64 = 1;
                let zero: I64 = 0;
                let two: I64 = 2;
                let v: RawPointer<I8> = vec_new();
                let a: String = "a";
                let b: String = "b";
                vec_push(v, a);
                vec_push(v, b);
                let first: String = vec_get(v, zero);
                if vec_len(v) == two && string_eq(first, a) == one { one } else { zero }
            }
        "#;
        assert_eq!(compile_run(src), 1);
    }

    #[test]
    fn codegen_float_arithmetic() {
        let src = r#"
            fn main() -> I64 {
                let one: I64 = 1;
                let zero: I64 = 0;
                let a: F64 = 1.5;
                let b: F64 = 2.0;
                let three: F64 = 3.0;
                let sum: F64 = a + b;
                if sum > three { one } else { zero }
            }
        "#;
        assert_eq!(compile_run(src), 1);
    }

    #[test]
    fn codegen_tree_sum_jit() {
        let src = r#"
            struct Node { left: OptNode, right: OptNode, value: I64 }
            enum OptNode { Some { node: Node }, None {} }

            fn sum(opt: OptNode) -> I64 {
                match opt {
                    OptNode::Some { node } => node.value + sum(node.left) + sum(node.right),
                    OptNode::None {} => 0
                }
            }

            fn main() -> I64 {
                let zero: I64 = 0;
                let one: I64 = 1;
                let two: I64 = 2;
                let three: I64 = 3;
                let four: I64 = 4;

                let leaf1: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: one };
                let leaf2: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: two };
                let leaf3: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: three };
                let leaf4: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: four };

                let n1: Node = Node {
                    left: OptNode::Some { node: leaf1 },
                    right: OptNode::Some { node: leaf2 },
                    value: zero
                };
                let n2: Node = Node {
                    left: OptNode::Some { node: leaf3 },
                    right: OptNode::Some { node: leaf4 },
                    value: zero
                };
                let root: Node = Node {
                    left: OptNode::Some { node: n1 },
                    right: OptNode::Some { node: n2 },
                    value: zero
                };

                sum(OptNode::Some { node: root })
            }
        "#;
        assert_eq!(compile_run(src), 10);
    }

    #[test]
    fn codegen_tree_sum_aot() {
        let src = r#"
            struct Node { left: OptNode, right: OptNode, value: I64 }
            enum OptNode { Some { node: Node }, None {} }

            fn sum(opt: OptNode) -> I64 {
                match opt {
                    OptNode::Some { node } => node.value + sum(node.left) + sum(node.right),
                    OptNode::None {} => 0
                }
            }

            fn main() -> I64 {
                let zero: I64 = 0;
                let one: I64 = 1;
                let two: I64 = 2;
                let three: I64 = 3;
                let four: I64 = 4;

                let leaf1: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: one };
                let leaf2: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: two };
                let leaf3: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: three };
                let leaf4: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: four };

                let n1: Node = Node {
                    left: OptNode::Some { node: leaf1 },
                    right: OptNode::Some { node: leaf2 },
                    value: zero
                };
                let n2: Node = Node {
                    left: OptNode::Some { node: leaf3 },
                    right: OptNode::Some { node: leaf4 },
                    value: zero
                };
                let root: Node = Node {
                    left: OptNode::Some { node: n1 },
                    right: OptNode::Some { node: n2 },
                    value: zero
                };

                sum(OptNode::Some { node: root })
            }
        "#;
        assert_eq!(compile_link_run(src), 10);
    }
}
