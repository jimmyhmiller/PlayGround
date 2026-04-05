//! Lox VM: parse → resolve → lower → optimize → run (interpreter or JIT).

use std::cell::Cell;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use dynir::interp::{ExternCallResult, InterpResult, ModuleInterpreter, NoGcRoots};
use dynir::ir::{FuncDef, Module};
use dynir::opt;
use dynlang::gc::DynGcRuntime;
use dynlang::{GcConfig, NanBoxTags};
use dynlower::{JitModule, JitOutcome};
use dynvalue::NanBox;

use crate::lower::{self, LoxGcTypes};
use crate::parser::Parser;
use crate::value::*;

pub enum InterpretResult {
    Ok,
    CompileError,
    RuntimeError,
}

pub struct VM {
    globals: Vec<u64>,
    defined_globals: std::collections::HashSet<usize>,
    had_error: bool,
    pub use_jit: bool,
    jit_call_table: Vec<*const u8>,

    // GC runtime and type info
    gc: Option<DynGcRuntime>,
    gc_types: Option<LoxGcTypes>,

    // String tables
    string_table: Vec<u64>,         // compile-time ID → NanBox GC ptr
    string_intern: HashMap<String, u64>,  // text → NanBox GC ptr
    compile_strings: Vec<String>,   // compile-time string pool text
}

// ── Thread-local VM pointer for JIT extern "C" callbacks ──────────

thread_local! {
    static ACTIVE_VM: Cell<*mut VM> = const { Cell::new(std::ptr::null_mut()) };
}

fn with_vm<R>(f: impl FnOnce(&mut VM) -> R) -> R {
    ACTIVE_VM.with(|cell| {
        let ptr = cell.get();
        assert!(!ptr.is_null(), "no active VM");
        f(unsafe { &mut *ptr })
    })
}

// ── GC object helpers on VM ──────────────────────────────────────

impl VM {
    fn gc(&mut self) -> &mut DynGcRuntime {
        self.gc.as_mut().expect("GC runtime not initialized")
    }

    fn types(&self) -> &LoxGcTypes {
        self.gc_types.as_ref().expect("GC types not initialized")
    }

    /// Allocate a GC object, return the raw pointer.
    fn gc_alloc(&mut self, type_id: usize, varlen_len: usize) -> *mut u8 {
        self.gc().alloc(type_id, varlen_len)
    }

    /// Read the TypeInfo pointer from a GC object header.
    fn obj_type_info(&self, val: u64) -> *const dynobj::TypeInfo {
        if !is_obj(val) { return std::ptr::null(); }
        let ptr = obj_ptr(val);
        if ptr.is_null() { return std::ptr::null(); }
        unsafe { gc_read_type_info(ptr) }
    }

    /// Determine the object type tag (matching the old ObjType enum values).
    /// Returns: 0=String, 1=Closure, 2=Upvalue, 3=Class, 4=Instance, 5=BoundMethod, 6=NativeFn, 255=unknown
    fn obj_type_tag(&self, val: u64) -> u64 {
        let ti = self.obj_type_info(val);
        if ti.is_null() { return 255; }
        let types = self.types();
        if ti == types.type_infos[types.string_id] as *const _ { 0 }
        else if ti == types.type_infos[types.closure_id] as *const _ { 1 }
        else if ti == types.type_infos[types.upvalue_id] as *const _ { 2 }
        else if ti == types.type_infos[types.class_id] as *const _ { 3 }
        else if ti == types.type_infos[types.instance_id] as *const _ { 4 }
        else if ti == types.type_infos[types.bound_method_id] as *const _ { 5 }
        else if ti == types.type_infos[types.native_fn_id] as *const _ { 6 }
        else { 255 }
    }

    fn is_closure(&self, val: u64) -> bool {
        let tag = self.obj_type_tag(val);
        tag == 1 || tag == 6 // Closure or NativeFn
    }

    fn is_class(&self, val: u64) -> bool { self.obj_type_tag(val) == 3 }
    fn is_instance(&self, val: u64) -> bool { self.obj_type_tag(val) == 4 }
    fn is_bound_method(&self, val: u64) -> bool { self.obj_type_tag(val) == 5 }

    // ── String helpers ───────────────────────────────────────────

    /// Allocate a GC string from a Rust &str. Returns NanBox-tagged ptr.
    fn alloc_string(&mut self, s: &str) -> u64 {
        let bytes = s.as_bytes();
        let types = self.types();
        let string_id = types.string_id;
        let len_off = types.string_len_off;
        let data_off = types.string_data_base_off;

        let ptr = self.gc_alloc(string_id, bytes.len());
        unsafe {
            gc_write_field(ptr, len_off, bytes.len() as u64);
            gc_write_bytes(ptr, data_off, bytes);
        }
        obj_val(ptr)
    }

    /// Intern a string: same content → same NanBox value.
    fn intern_string(&mut self, s: &str) -> u64 {
        if let Some(&val) = self.string_intern.get(s) {
            return val;
        }
        let val = self.alloc_string(s);
        self.string_intern.insert(s.to_string(), val);
        val
    }

    /// Read a GC string as a Rust String. `val` must be a string object.
    fn read_string(&self, val: u64) -> String {
        let ptr = obj_ptr(val);
        let types = self.types();
        unsafe {
            let len = gc_read_field(ptr, types.string_len_off) as usize;
            let bytes = gc_read_bytes(ptr, types.string_data_base_off, len);
            String::from_utf8_lossy(bytes).into_owned()
        }
    }

    /// Resolve a compile-time string ID to a NanBox GC string value.
    fn resolve_string(&mut self, id: usize) -> u64 {
        if id < self.string_table.len() {
            return self.string_table[id];
        }
        // Shouldn't happen, but handle gracefully
        self.intern_string(&format!("#{}", id))
    }

    /// Get the text for a compile-time string ID (for error messages).
    fn string_text(&self, id: usize) -> String {
        if id < self.compile_strings.len() {
            self.compile_strings[id].clone()
        } else {
            format!("#{}", id)
        }
    }

    // ── Upvalue helpers ──────────────────────────────────────────

    fn alloc_upvalue(&mut self, init: u64) -> u64 {
        let types = self.types();
        let upvalue_id = types.upvalue_id;
        let off = types.upvalue_value_off;
        let ptr = self.gc_alloc(upvalue_id, 0);
        unsafe { gc_write_field(ptr, off, init); }
        obj_val(ptr)
    }

    fn get_upvalue(&self, cell: u64) -> u64 {
        let ptr = obj_ptr(cell);
        unsafe { gc_read_field(ptr, self.types().upvalue_value_off) }
    }

    fn set_upvalue(&self, cell: u64, val: u64) {
        let ptr = obj_ptr(cell);
        unsafe { gc_write_field(ptr, self.types().upvalue_value_off, val); }
    }

    // ── Closure helpers ──────────────────────────────────────────

    fn alloc_closure(&mut self, func_idx: u64, num_upvalues: usize, arity: u64, name_val: u64) -> u64 {
        let types = self.types();
        let closure_id = types.closure_id;
        let func_idx_off = types.closure_func_idx_off;
        let arity_off = types.closure_arity_off;
        let name_off = types.closure_name_off;
        let upval_base = types.closure_upval_base_off;

        let ptr = self.gc_alloc(closure_id, num_upvalues);
        unsafe {
            gc_write_field(ptr, func_idx_off, func_idx);
            gc_write_field(ptr, arity_off, arity);
            gc_write_field(ptr, name_off, name_val);
            // Initialize all upvalue slots to nil
            for i in 0..num_upvalues {
                gc_write_elem(ptr, upval_base, i, nil_val());
            }
        }
        obj_val(ptr)
    }

    fn closure_func_idx(&self, closure: u64) -> u64 {
        let ptr = obj_ptr(closure);
        unsafe { gc_read_field(ptr, self.types().closure_func_idx_off) }
    }

    fn closure_arity(&self, closure: u64) -> u64 {
        let ptr = obj_ptr(closure);
        unsafe { gc_read_field(ptr, self.types().closure_arity_off) }
    }

    fn closure_name(&self, closure: u64) -> u64 {
        let ptr = obj_ptr(closure);
        unsafe { gc_read_field(ptr, self.types().closure_name_off) }
    }

    fn closure_upvalue(&self, closure: u64, idx: usize) -> u64 {
        let ptr = obj_ptr(closure);
        unsafe { gc_read_elem(ptr, self.types().closure_upval_base_off, idx) }
    }

    fn set_closure_upvalue(&self, closure: u64, idx: usize, cell: u64) {
        let ptr = obj_ptr(closure);
        unsafe { gc_write_elem(ptr, self.types().closure_upval_base_off, idx, cell); }
    }

    // ── GC Table helpers ─────────────────────────────────────────
    // Table stores [key, val, key, val, ...] as varlen values with a count field.

    /// Allocate an empty table with the given capacity (number of key-value pairs).
    fn alloc_table(&mut self, capacity: usize) -> u64 {
        let types = self.types();
        let table_id = types.table_id;
        let count_off = types.table_count_off;
        let base_off = types.table_data_base_off;

        let ptr = self.gc_alloc(table_id, capacity * 2); // 2 slots per pair
        unsafe {
            gc_write_field(ptr, count_off, 0);
            // Initialize all varlen slots to nil so GC doesn't trace garbage
            for i in 0..(capacity * 2) {
                gc_write_elem(ptr, base_off, i, nil_val());
            }
        }
        obj_val(ptr)
    }

    /// Read the count (number of key-value pairs) from a table.
    fn table_count(&self, table: u64) -> usize {
        let ptr = obj_ptr(table);
        unsafe { gc_read_field(ptr, self.types().table_count_off) as usize }
    }

    /// Look up a key in a table. Returns Some(value) or None.
    fn table_get(&self, table: u64, key: u64) -> Option<u64> {
        let ptr = obj_ptr(table);
        let types = self.types();
        let count = unsafe { gc_read_field(ptr, types.table_count_off) as usize };
        let base = types.table_data_base_off;
        for i in 0..count {
            let k = unsafe { gc_read_elem(ptr, base, i * 2) };
            if k == key {
                return Some(unsafe { gc_read_elem(ptr, base, i * 2 + 1) });
            }
        }
        None
    }

    /// Set a key-value pair in a table. If key exists, updates in place.
    /// If key doesn't exist, may need to grow the table — returns the
    /// (possibly new) table value.
    fn table_set(&mut self, table: u64, key: u64, value: u64) -> u64 {
        let ptr = obj_ptr(table);
        let types = self.types();
        let count_off = types.table_count_off;
        let base = types.table_data_base_off;
        let count = unsafe { gc_read_field(ptr, count_off) as usize };

        // Check if key already exists
        for i in 0..count {
            let k = unsafe { gc_read_elem(ptr, base, i * 2) };
            if k == key {
                unsafe { gc_write_elem(ptr, base, i * 2 + 1, value); }
                return table;
            }
        }

        // Key not found — need to add. Check if there's capacity.
        // The varlen_count stored in the object is the total number of varlen slots.
        // We allocated capacity*2 slots. If count < capacity, we can add in place.
        let varlen_count = unsafe {
            let vc_off = types.type_infos[types.table_id].varlen_count_offset();
            gc_read_field(ptr, vc_off as i32) as usize
        };
        let capacity = varlen_count / 2;

        if count < capacity {
            // Room to add in place
            unsafe {
                gc_write_elem(ptr, base, count * 2, key);
                gc_write_elem(ptr, base, count * 2 + 1, value);
                gc_write_field(ptr, count_off, (count + 1) as u64);
            }
            return table;
        }

        // Need to grow — allocate new table with double capacity (min 4)
        let new_cap = if capacity == 0 { 4 } else { capacity * 2 };
        let new_table = self.alloc_table(new_cap);
        let new_ptr = obj_ptr(new_table);
        let new_base = self.types().table_data_base_off;
        let new_count_off = self.types().table_count_off;

        // Copy old entries
        unsafe {
            for i in 0..count {
                let k = gc_read_elem(ptr, base, i * 2);
                let v = gc_read_elem(ptr, base, i * 2 + 1);
                gc_write_elem(new_ptr, new_base, i * 2, k);
                gc_write_elem(new_ptr, new_base, i * 2 + 1, v);
            }
            // Add new entry
            gc_write_elem(new_ptr, new_base, count * 2, key);
            gc_write_elem(new_ptr, new_base, count * 2 + 1, value);
            gc_write_field(new_ptr, new_count_off, (count + 1) as u64);
        }
        new_table
    }

    /// Copy all entries from src table into dst table (for inheritance).
    /// Only inserts keys that don't already exist in dst.
    /// Returns the (possibly grown) dst table.
    fn table_merge(&mut self, dst: u64, src: u64) -> u64 {
        let src_ptr = obj_ptr(src);
        let types = self.types();
        let count = unsafe { gc_read_field(src_ptr, types.table_count_off) as usize };
        let base = types.table_data_base_off;

        let mut result = dst;
        for i in 0..count {
            let k = unsafe { gc_read_elem(src_ptr, base, i * 2) };
            let v = unsafe { gc_read_elem(src_ptr, base, i * 2 + 1) };
            // Only insert if not already present
            if self.table_get(result, k).is_none() {
                result = self.table_set(result, k, v);
            }
        }
        result
    }

    // ── Class helpers ────────────────────────────────────────────

    fn alloc_class(&mut self, name_val: u64) -> u64 {
        let empty_table = self.alloc_table(0);
        let types = self.types();
        let class_id = types.class_id;
        let name_off = types.class_name_off;
        let super_off = types.class_super_off;
        let methods_off = types.class_methods_off;

        let ptr = self.gc_alloc(class_id, 0);
        unsafe {
            gc_write_field(ptr, name_off, name_val);
            gc_write_field(ptr, super_off, nil_val());
            gc_write_field(ptr, methods_off, empty_table);
        }
        obj_val(ptr)
    }

    fn class_name(&self, class: u64) -> u64 {
        let ptr = obj_ptr(class);
        unsafe { gc_read_field(ptr, self.types().class_name_off) }
    }

    fn class_superclass(&self, class: u64) -> u64 {
        let ptr = obj_ptr(class);
        unsafe { gc_read_field(ptr, self.types().class_super_off) }
    }

    fn class_methods_table(&self, class: u64) -> u64 {
        let ptr = obj_ptr(class);
        unsafe { gc_read_field(ptr, self.types().class_methods_off) }
    }

    fn set_class_methods_table(&self, class: u64, table: u64) {
        let ptr = obj_ptr(class);
        unsafe { gc_write_field(ptr, self.types().class_methods_off, table); }
    }

    fn class_get_method(&self, class: u64, key: u64) -> Option<u64> {
        let table = self.class_methods_table(class);
        self.table_get(table, key)
    }

    fn class_set_method(&mut self, class: u64, key: u64, value: u64) {
        let table = self.class_methods_table(class);
        let new_table = self.table_set(table, key, value);
        if new_table != table {
            self.set_class_methods_table(class, new_table);
        }
    }

    fn set_class_superclass(&mut self, class: u64, super_val: u64) {
        let ptr = obj_ptr(class);
        unsafe { gc_write_field(ptr, self.types().class_super_off, super_val); }
    }

    // ── Instance helpers ─────────────────────────────────────────

    fn alloc_instance(&mut self, class: u64) -> u64 {
        let empty_table = self.alloc_table(0);
        let types = self.types();
        let instance_id = types.instance_id;
        let class_off = types.instance_class_off;
        let fields_off = types.instance_fields_off;

        let ptr = self.gc_alloc(instance_id, 0);
        unsafe {
            gc_write_field(ptr, class_off, class);
            gc_write_field(ptr, fields_off, empty_table);
        }
        obj_val(ptr)
    }

    fn instance_class(&self, inst: u64) -> u64 {
        let ptr = obj_ptr(inst);
        unsafe { gc_read_field(ptr, self.types().instance_class_off) }
    }

    fn instance_fields_table(&self, inst: u64) -> u64 {
        let ptr = obj_ptr(inst);
        unsafe { gc_read_field(ptr, self.types().instance_fields_off) }
    }

    fn set_instance_fields_table(&self, inst: u64, table: u64) {
        let ptr = obj_ptr(inst);
        unsafe { gc_write_field(ptr, self.types().instance_fields_off, table); }
    }

    fn instance_get_field(&self, inst: u64, key: u64) -> Option<u64> {
        let table = self.instance_fields_table(inst);
        self.table_get(table, key)
    }

    fn instance_set_field(&mut self, inst: u64, key: u64, value: u64) {
        let table = self.instance_fields_table(inst);
        let new_table = self.table_set(table, key, value);
        if new_table != table {
            self.set_instance_fields_table(inst, new_table);
        }
    }

    // ── BoundMethod helpers ──────────────────────────────────────

    fn alloc_bound_method(&mut self, receiver: u64, method: u64) -> u64 {
        let types = self.types();
        let bm_id = types.bound_method_id;
        let recv_off = types.bound_receiver_off;
        let method_off = types.bound_method_off;

        let ptr = self.gc_alloc(bm_id, 0);
        unsafe {
            gc_write_field(ptr, recv_off, receiver);
            gc_write_field(ptr, method_off, method);
        }
        obj_val(ptr)
    }

    fn bound_receiver(&self, bm: u64) -> u64 {
        let ptr = obj_ptr(bm);
        unsafe { gc_read_field(ptr, self.types().bound_receiver_off) }
    }

    fn bound_method_closure(&self, bm: u64) -> u64 {
        let ptr = obj_ptr(bm);
        unsafe { gc_read_field(ptr, self.types().bound_method_off) }
    }

    // ── NativeFn helpers ─────────────────────────────────────────

    fn alloc_native_fn(&mut self, name_val: u64, func_ptr: u64) -> u64 {
        let types = self.types();
        let nf_id = types.native_fn_id;
        let name_off = types.native_name_off;
        let fp_off = types.native_func_ptr_off;

        let ptr = self.gc_alloc(nf_id, 0);
        unsafe {
            gc_write_field(ptr, name_off, name_val);
            gc_write_field(ptr, fp_off, func_ptr);
        }
        obj_val(ptr)
    }

    fn native_func_ptr(&self, nf: u64) -> u64 {
        let ptr = obj_ptr(nf);
        unsafe { gc_read_field(ptr, self.types().native_func_ptr_off) }
    }

    // ── Value formatting ─────────────────────────────────────────

    fn value_to_string(&self, v: u64) -> String {
        if is_nil(v) {
            "nil".to_string()
        } else if is_bool(v) {
            if as_bool(v) { "true".to_string() } else { "false".to_string() }
        } else if is_number(v) {
            format_number(as_number(v))
        } else if is_obj(v) {
            let tag = self.obj_type_tag(v);
            match tag {
                0 => self.read_string(v), // String
                1 => { // Closure
                    let name_val = self.closure_name(v);
                    if is_obj(name_val) {
                        format!("<fn {}>", self.read_string(name_val))
                    } else {
                        "<fn>".to_string()
                    }
                }
                2 => "<upvalue>".to_string(), // Upvalue
                3 => { // Class
                    let name_val = self.class_name(v);
                    if is_obj(name_val) {
                        self.read_string(name_val)
                    } else {
                        "<class>".to_string()
                    }
                }
                4 => { // Instance
                    let class = self.instance_class(v);
                    let name_val = self.class_name(class);
                    if is_obj(name_val) {
                        format!("{} instance", self.read_string(name_val))
                    } else {
                        "<instance>".to_string()
                    }
                }
                5 => { // BoundMethod
                    let method = self.bound_method_closure(v);
                    let name_val = self.closure_name(method);
                    if is_obj(name_val) {
                        format!("<fn {}>", self.read_string(name_val))
                    } else {
                        "<fn>".to_string()
                    }
                }
                6 => "<native fn>".to_string(), // NativeFn
                _ => format!("<obj>"),
            }
        } else {
            "unknown".to_string()
        }
    }
}

// ── JIT extern "C" functions ──────────────────────────────────────

extern "C" fn jit_lox_print(v: u64) {
    with_vm(|vm| {
        if vm.had_error { return; }
        let s = vm.value_to_string(v);
        println!("{}", s);
    });
}

extern "C" fn jit_lox_define_global(name_id: u64, value: u64) {
    with_vm(|vm| {
        let id = name_id as usize;
        vm.ensure_global(id);
        vm.globals[id] = value;
        vm.defined_globals.insert(id);
    });
}

extern "C" fn jit_lox_get_global(name_id: u64) -> u64 {
    with_vm(|vm| {
        let id = name_id as usize;
        if id < vm.globals.len() && vm.defined_globals.contains(&id) {
            vm.globals[id]
        } else {
            vm.global_error(id);
            nil_val()
        }
    })
}

extern "C" fn jit_lox_set_global(name_id: u64, value: u64) -> u64 {
    with_vm(|vm| {
        let id = name_id as usize;
        if id < vm.globals.len() && vm.defined_globals.contains(&id) {
            vm.globals[id] = value;
            value
        } else {
            vm.global_error(id);
            nil_val()
        }
    })
}

extern "C" fn jit_lox_add(a: u64, b: u64) -> u64 {
    with_vm(|vm| {
        if is_obj(a) && is_obj(b) {
            let sa = vm.read_string(a);
            let sb = vm.read_string(b);
            return vm.intern_string(&format!("{}{}", sa, sb));
        }
        vm.runtime_error("Operands must be two numbers or two strings.");
        nil_val()
    })
}

extern "C" fn jit_lox_sub(_: u64, _: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operands must be numbers."); nil_val() }) }
extern "C" fn jit_lox_mul(_: u64, _: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operands must be numbers."); nil_val() }) }
extern "C" fn jit_lox_div(_: u64, _: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operands must be numbers."); nil_val() }) }
extern "C" fn jit_lox_neg(_: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operand must be a number."); nil_val() }) }
extern "C" fn jit_lox_eq(a: u64, b: u64) -> u64 { bool_val(values_equal(a, b)) }
extern "C" fn jit_lox_lt(_: u64, _: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operands must be numbers."); nil_val() }) }
extern "C" fn jit_lox_gt(_: u64, _: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operands must be numbers."); nil_val() }) }
extern "C" fn jit_lox_not(v: u64) -> u64 { bool_val(is_falsey(v)) }

extern "C" fn jit_lox_clock() -> u64 {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    number_val(t)
}

// ── Upvalue / Closure JIT externs ─────────────────────────────────

extern "C" fn jit_lox_alloc_upvalue(init: u64) -> u64 {
    with_vm(|vm| vm.alloc_upvalue(init))
}

extern "C" fn jit_lox_get_upvalue(cell: u64) -> u64 {
    with_vm(|vm| vm.get_upvalue(cell))
}

extern "C" fn jit_lox_set_upvalue(cell: u64, val: u64) {
    with_vm(|vm| vm.set_upvalue(cell, val));
}

extern "C" fn jit_lox_make_closure(func_idx: u64, num_upvalues: u64, arity: u64, name_id: u64) -> u64 {
    with_vm(|vm| {
        let name_val = vm.resolve_string(name_id as usize);
        vm.alloc_closure(func_idx, num_upvalues as usize, arity, name_val)
    })
}

extern "C" fn jit_lox_closure_upvalue(closure: u64, idx: u64) -> u64 {
    with_vm(|vm| vm.closure_upvalue(closure, idx as usize))
}

extern "C" fn jit_lox_set_closure_upvalue(closure: u64, idx: u64, cell: u64) {
    with_vm(|vm| vm.set_closure_upvalue(closure, idx as usize, cell));
}

extern "C" fn jit_lox_closure_func_ptr(closure: u64) -> u64 {
    with_vm(|vm| {
        let tag = vm.obj_type_tag(closure);
        match tag {
            1 => { // Closure
                let idx = vm.closure_func_idx(closure) as usize;
                if idx < vm.jit_call_table.len() {
                    vm.jit_call_table[idx] as u64
                } else {
                    idx as u64
                }
            }
            6 => { // NativeFn
                vm.native_func_ptr(closure)
            }
            _ => 0,
        }
    })
}

extern "C" fn jit_lox_obj_type(v: u64) -> u64 {
    with_vm(|vm| vm.obj_type_tag(v))
}

// ── Class JIT externs ─────────────────────────────────────────────

extern "C" fn jit_lox_make_class(name_id: u64) -> u64 {
    with_vm(|vm| {
        let name_val = vm.resolve_string(name_id as usize);
        vm.alloc_class(name_val)
    })
}

extern "C" fn jit_lox_make_native_fn(name_id: u64) -> u64 {
    with_vm(|vm| {
        let name_val = vm.resolve_string(name_id as usize);
        // Store the clock function pointer for native fns
        vm.alloc_native_fn(name_val, jit_lox_clock as u64)
    })
}

extern "C" fn jit_lox_check_arity(_callee: u64, expected: u64, got: u64) {
    with_vm(|vm| {
        if expected != got {
            vm.runtime_error(&format!("Expected {} arguments but got {}.", expected, got));
        }
    });
}

extern "C" fn jit_lox_call_non_callable() {
    with_vm(|vm| {
        vm.runtime_error("Can only call functions and classes.");
    });
}

extern "C" fn jit_lox_get_closure_arity(callee: u64) -> u64 {
    with_vm(|vm| {
        if !is_obj(callee) { return 0; }
        let tag = vm.obj_type_tag(callee);
        match tag {
            1 => vm.closure_arity(callee), // Closure
            6 => 0, // NativeFn (clock takes 0 args)
            _ => 0,
        }
    })
}

extern "C" fn jit_lox_get_class_init_arity(callee: u64) -> u64 {
    with_vm(|vm| {
        let init_name = vm.intern_string("init");
        if let Some(closure_val) = vm.class_get_method(callee, init_name) {
            return vm.closure_arity(closure_val);
        }
        255 // sentinel: no init
    })
}

extern "C" fn jit_lox_get_bound_arity(callee: u64) -> u64 {
    with_vm(|vm| {
        let method = vm.bound_method_closure(callee);
        vm.closure_arity(method)
    })
}

extern "C" fn jit_lox_class_inherit(class: u64, super_val: u64) {
    with_vm(|vm| {
        if !is_obj(super_val) || !vm.is_class(super_val) {
            vm.runtime_error("Superclass must be a class.");
            return;
        }
        let super_table = vm.class_methods_table(super_val);
        let own_table = vm.class_methods_table(class);
        let merged = vm.table_merge(own_table, super_table);
        vm.set_class_methods_table(class, merged);
        vm.set_class_superclass(class, super_val);
    });
}

extern "C" fn jit_lox_class_add_method(class: u64, name_id: u64, method_closure: u64) {
    with_vm(|vm| {
        let name_val = vm.resolve_string(name_id as usize);
        vm.class_set_method(class, name_val, method_closure);
    });
}

extern "C" fn jit_lox_construct_instance(class: u64) -> u64 {
    with_vm(|vm| vm.alloc_instance(class))
}

extern "C" fn jit_lox_class_init_ptr(class: u64) -> u64 {
    with_vm(|vm| {
        let init_name = vm.intern_string("init");
        if let Some(closure_val) = vm.class_get_method(class, init_name) {
            let idx = vm.closure_func_idx(closure_val) as usize;
            if idx < vm.jit_call_table.len() {
                return vm.jit_call_table[idx] as u64;
            }
            return idx as u64;
        }
        0
    })
}

extern "C" fn jit_lox_class_init_closure(class: u64) -> u64 {
    with_vm(|vm| {
        let init_name = vm.intern_string("init");
        vm.class_get_method(class, init_name).unwrap_or(nil_val())
    })
}

// ── Property JIT externs ──────────────────────────────────────────

extern "C" fn jit_lox_get_property(obj: u64, name_id: u64) -> u64 {
    with_vm(|vm| {
        let prop_name = vm.string_text(name_id as usize);
        let name_val = vm.resolve_string(name_id as usize);

        if !is_obj(obj) || !vm.is_instance(obj) {
            vm.runtime_error("Only instances have properties.");
            return nil_val();
        }

        // Check fields first
        if let Some(val) = vm.instance_get_field(obj, name_val) {
            return val;
        }

        // Check methods on the class
        let class = vm.instance_class(obj);
        if let Some(method) = vm.class_get_method(class, name_val) {
            return vm.alloc_bound_method(obj, method);
        }

        vm.runtime_error(&format!("Undefined property '{}'.", prop_name));
        nil_val()
    })
}

extern "C" fn jit_lox_set_property(obj: u64, name_id: u64, val: u64) -> u64 {
    with_vm(|vm| {
        let name_val = vm.resolve_string(name_id as usize);

        if !is_obj(obj) || !vm.is_instance(obj) {
            vm.runtime_error("Only instances have fields.");
            return nil_val();
        }

        vm.instance_set_field(obj, name_val, val);
        val
    })
}

// ── Bound method / Super JIT externs ─────────────────────────────

extern "C" fn jit_lox_get_super(this: u64, class_val: u64, method_name_id: u64) -> u64 {
    with_vm(|vm| {
        let method_name_str = vm.string_text(method_name_id as usize);
        let method_name = vm.resolve_string(method_name_id as usize);

        let superclass = vm.class_superclass(class_val);
        if is_nil(superclass) { return nil_val(); }

        if let Some(method_closure) = vm.class_get_method(superclass, method_name) {
            return vm.alloc_bound_method(this, method_closure);
        }

        vm.runtime_error(&format!("Undefined property '{}'.", method_name_str));
        nil_val()
    })
}

extern "C" fn jit_lox_bound_receiver(bm: u64) -> u64 {
    with_vm(|vm| vm.bound_receiver(bm))
}

extern "C" fn jit_lox_bound_method_closure(bm: u64) -> u64 {
    with_vm(|vm| vm.bound_method_closure(bm))
}

extern "C" fn jit_lox_bound_closure_func_ptr(bm: u64) -> u64 {
    with_vm(|vm| {
        let method = vm.bound_method_closure(bm);
        let idx = vm.closure_func_idx(method) as usize;
        if idx < vm.jit_call_table.len() {
            vm.jit_call_table[idx] as u64
        } else {
            idx as u64
        }
    })
}

extern "C" fn jit_lox_resolve_string(id: u64) -> u64 {
    with_vm(|vm| vm.resolve_string(id as usize))
}

/// Build JIT extern pointers matching module.func_table order.
fn build_jit_externs(module: &Module) -> Vec<*const u8> {
    let mut ptrs = Vec::new();
    for def in &module.func_table {
        if let FuncDef::Extern(ext) = def {
            let ptr: *const u8 = match ext.name.as_str() {
                "lox_add" => jit_lox_add as *const u8,
                "lox_sub" => jit_lox_sub as *const u8,
                "lox_mul" => jit_lox_mul as *const u8,
                "lox_div" => jit_lox_div as *const u8,
                "lox_neg" => jit_lox_neg as *const u8,
                "lox_eq" => jit_lox_eq as *const u8,
                "lox_lt" => jit_lox_lt as *const u8,
                "lox_gt" => jit_lox_gt as *const u8,
                "lox_not" => jit_lox_not as *const u8,
                "lox_print" => jit_lox_print as *const u8,
                "lox_define_global" => jit_lox_define_global as *const u8,
                "lox_get_global" => jit_lox_get_global as *const u8,
                "lox_set_global" => jit_lox_set_global as *const u8,
                "lox_clock" => jit_lox_clock as *const u8,
                "lox_alloc_upvalue" => jit_lox_alloc_upvalue as *const u8,
                "lox_get_upvalue" => jit_lox_get_upvalue as *const u8,
                "lox_set_upvalue" => jit_lox_set_upvalue as *const u8,
                "lox_make_closure" => jit_lox_make_closure as *const u8,
                "lox_check_arity" => jit_lox_check_arity as *const u8,
                "lox_call_non_callable" => jit_lox_call_non_callable as *const u8,
                "lox_get_closure_arity" => jit_lox_get_closure_arity as *const u8,
                "lox_get_class_init_arity" => jit_lox_get_class_init_arity as *const u8,
                "lox_get_bound_arity" => jit_lox_get_bound_arity as *const u8,
                "lox_closure_upvalue" => jit_lox_closure_upvalue as *const u8,
                "lox_closure_func_ptr" => jit_lox_closure_func_ptr as *const u8,
                "lox_set_closure_upvalue" => jit_lox_set_closure_upvalue as *const u8,
                "lox_obj_type" => jit_lox_obj_type as *const u8,
                "lox_make_class" => jit_lox_make_class as *const u8,
                "lox_class_inherit" => jit_lox_class_inherit as *const u8,
                "lox_class_add_method" => jit_lox_class_add_method as *const u8,
                "lox_construct_instance" => jit_lox_construct_instance as *const u8,
                "lox_class_init_ptr" => jit_lox_class_init_ptr as *const u8,
                "lox_class_init_closure" => jit_lox_class_init_closure as *const u8,
                "lox_get_property" => jit_lox_get_property as *const u8,
                "lox_set_property" => jit_lox_set_property as *const u8,
                "lox_get_super" => jit_lox_get_super as *const u8,
                "lox_bound_receiver" => jit_lox_bound_receiver as *const u8,
                "lox_bound_method_closure" => jit_lox_bound_method_closure as *const u8,
                "lox_bound_closure_func_ptr" => jit_lox_bound_closure_func_ptr as *const u8,
                "lox_make_native_fn" => jit_lox_make_native_fn as *const u8,
                "lox_resolve_string" => jit_lox_resolve_string as *const u8,
                "__gc_alloc__" => jit_gc_alloc as *const u8,
                other => panic!("unknown extern for JIT: {other}"),
            };
            ptrs.push(ptr);
        }
    }
    ptrs
}

extern "C" fn jit_gc_alloc(type_id: u64, varlen_len: u64) -> u64 {
    with_vm(|vm| {
        let ptr = vm.gc_alloc(type_id as usize, varlen_len as usize);
        ptr as u64
    })
}

// ── VM implementation ─────────────────────────────────────────────

impl VM {
    pub fn new() -> Self {
        VM {
            globals: Vec::new(),
            defined_globals: std::collections::HashSet::new(),
            had_error: false,
            use_jit: false,
            jit_call_table: Vec::new(),
            gc: None,
            gc_types: None,
            string_table: Vec::new(),
            string_intern: HashMap::new(),
            compile_strings: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.globals.clear();
        self.defined_globals.clear();
        self.had_error = false;
        self.jit_call_table.clear();
        self.gc = None;
        self.gc_types = None;
        self.string_table.clear();
        self.string_intern.clear();
        self.compile_strings.clear();
    }

    fn ensure_global(&mut self, id: usize) {
        if id >= self.globals.len() {
            self.globals.resize(id + 1, nil_val());
        }
    }

    fn global_error(&mut self, id: usize) {
        let name = self.string_text(id);
        self.runtime_error(&format!("Undefined variable '{}'.", name));
    }

    pub fn interpret(&mut self, source: &str) -> InterpretResult {
        let program = match Parser::parse(source) {
            Some(p) => p,
            None => return InterpretResult::CompileError,
        };

        let mut lowered = lower::lower(&program);

        // Optimize
        for func in &mut lowered.module.functions {
            opt::mem2reg(func);
            opt::constant_fold(func);
            opt::gvn(func);
            opt::dce(func);
        }

        // Debug IR dump
        if std::env::var("DUMP_IR").is_ok() {
            for func in &lowered.module.functions {
                eprintln!("{}", func);
                eprintln!("---");
            }
        }

        // Initialize GC runtime
        self.gc_types = Some(lowered.gc_types);
        self.compile_strings = lowered.strings.clone();
        let type_infos = &self.gc_types.as_ref().unwrap().type_infos;
        self.gc = Some(create_gc_runtime(type_infos));

        // Intern compile-time strings
        self.string_table.clear();
        self.string_intern.clear();
        for s in &lowered.strings {
            let val = self.intern_string(s);
            self.string_table.push(val);
        }

        self.had_error = false;

        if self.use_jit {
            self.run_jit(&lowered.module, lowered.entry)
        } else {
            self.run_interp(&lowered.module, lowered.entry)
        }
    }

    fn run_interp(&mut self, module: &Module, entry: dynlang::FuncRef) -> InterpretResult {
        let roots = NoGcRoots;
        let mut interp = ModuleInterpreter::<NanBox, _>::new(module, &roots);
        self.bind_runtime(&mut interp);

        match interp.run(entry, &[nil_val()]) {
            Ok(InterpResult::Value(_)) | Ok(InterpResult::Void) => {
                if self.had_error { InterpretResult::RuntimeError }
                else { InterpretResult::Ok }
            }
            Ok(InterpResult::Deopt { .. }) => InterpretResult::RuntimeError,
            Err(e) => {
                eprintln!("Internal error: {:?}", e);
                InterpretResult::RuntimeError
            }
        }
    }

    fn run_jit(&mut self, module: &Module, entry: dynlang::FuncRef) -> InterpretResult {
        let externs = build_jit_externs(module);
        let jit = JitModule::compile_linear_scan::<NanBox>(module, &externs);

        self.jit_call_table = jit.call_table().to_vec();

        ACTIVE_VM.with(|cell| cell.set(self as *mut VM));
        let result = jit.call_outcome(entry, &[nil_val()]);
        ACTIVE_VM.with(|cell| cell.set(std::ptr::null_mut()));
        self.jit_call_table.clear();

        match result {
            JitOutcome::Value(_) | JitOutcome::Void => {
                if self.had_error { InterpretResult::RuntimeError }
                else { InterpretResult::Ok }
            }
            _ => { eprintln!("JIT error: {:?}", result); InterpretResult::RuntimeError }
        }
    }

    fn runtime_error(&mut self, msg: &str) {
        eprintln!("{}", msg);
        self.had_error = true;
    }

    fn bind_runtime<'a>(&mut self, interp: &mut ModuleInterpreter<'a, NanBox, NoGcRoots>) {
        let rt = self as *mut VM;

        interp.bind_by_name("lox_print", move |args| {
            let vm = unsafe { &*rt };
            if !vm.had_error {
                println!("{}", vm.value_to_string(args[0]));
            }
            ExternCallResult::Value(None)
        });

        interp.bind_by_name("lox_define_global", move |args| {
            let vm = unsafe { &mut *rt };
            let id = args[0] as usize;
            vm.ensure_global(id);
            vm.globals[id] = args[1];
            vm.defined_globals.insert(id);
            ExternCallResult::Value(None)
        });

        interp.bind_by_name("lox_get_global", move |args| {
            let vm = unsafe { &mut *rt };
            let id = args[0] as usize;
            if id < vm.globals.len() && vm.defined_globals.contains(&id) {
                ExternCallResult::Value(Some(vm.globals[id]))
            } else {
                vm.global_error(id);
                ExternCallResult::Value(Some(nil_val()))
            }
        });

        interp.bind_by_name("lox_set_global", move |args| {
            let vm = unsafe { &mut *rt };
            let id = args[0] as usize;
            let val = args[1];
            if id < vm.globals.len() && vm.defined_globals.contains(&id) {
                vm.globals[id] = val;
                ExternCallResult::Value(Some(val))
            } else {
                vm.global_error(id);
                ExternCallResult::Value(Some(nil_val()))
            }
        });

        // Arithmetic slow paths
        interp.bind_by_name("lox_add", move |args| {
            let vm = unsafe { &mut *rt };
            let (a, b) = (args[0], args[1]);
            if is_obj(a) && is_obj(b) {
                let sa = vm.read_string(a);
                let sb = vm.read_string(b);
                let r = vm.intern_string(&format!("{}{}", sa, sb));
                return ExternCallResult::Value(Some(r));
            }
            vm.runtime_error("Operands must be two numbers or two strings.");
            ExternCallResult::Value(Some(nil_val()))
        });
        interp.bind_by_name("lox_sub", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operands must be numbers."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_mul", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operands must be numbers."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_div", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operands must be numbers."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_neg", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operand must be a number."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_eq", |args| { ExternCallResult::Value(Some(bool_val(values_equal(args[0], args[1])))) });
        interp.bind_by_name("lox_lt", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operands must be numbers."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_gt", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operands must be numbers."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_not", |args| { ExternCallResult::Value(Some(bool_val(is_falsey(args[0])))) });
        interp.bind_by_name("lox_clock", |_| {
            let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
            ExternCallResult::Value(Some(number_val(t)))
        });

        // Upvalue / closure
        interp.bind_by_name("lox_alloc_upvalue", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.alloc_upvalue(args[0])))
        });
        interp.bind_by_name("lox_get_upvalue", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.get_upvalue(args[0])))
        });
        interp.bind_by_name("lox_set_upvalue", move |args| {
            let vm = unsafe{&mut*rt};
            vm.set_upvalue(args[0], args[1]);
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_make_closure", move |args| {
            let vm = unsafe{&mut*rt};
            let name_val = vm.resolve_string(args[3] as usize);
            let v = vm.alloc_closure(args[0], args[1] as usize, args[2], name_val);
            ExternCallResult::Value(Some(v))
        });
        interp.bind_by_name("lox_closure_upvalue", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.closure_upvalue(args[0], args[1] as usize)))
        });
        interp.bind_by_name("lox_set_closure_upvalue", move |args| {
            let vm = unsafe{&mut*rt};
            vm.set_closure_upvalue(args[0], args[1] as usize, args[2]);
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_closure_func_ptr", move |args| {
            let vm = unsafe{&mut*rt};
            let tag = vm.obj_type_tag(args[0]);
            let v = match tag {
                1 => vm.closure_func_idx(args[0]), // Closure: return func_table_idx for interp
                6 => 0, // NativeFn: not used in interp mode
                _ => 0,
            };
            ExternCallResult::Value(Some(v))
        });
        interp.bind_by_name("lox_check_arity", move |args| {
            let vm = unsafe{&mut*rt};
            let expected = args[1];
            let got = args[2];
            if expected != got {
                vm.runtime_error(&format!("Expected {} arguments but got {}.", expected, got));
            }
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_call_non_callable", move |_args| {
            let vm = unsafe{&mut*rt};
            vm.runtime_error("Can only call functions and classes.");
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_get_closure_arity", move |args| {
            let vm = unsafe{&mut*rt};
            if !is_obj(args[0]) { return ExternCallResult::Value(Some(0)); }
            let tag = vm.obj_type_tag(args[0]);
            let v = match tag {
                1 => vm.closure_arity(args[0]),
                6 => 0,
                _ => 0,
            };
            ExternCallResult::Value(Some(v))
        });
        interp.bind_by_name("lox_get_class_init_arity", move |args| {
            let vm = unsafe{&mut*rt};
            let init_name = vm.intern_string("init");
            let v = if let Some(closure_val) = vm.class_get_method(args[0], init_name) {
                vm.closure_arity(closure_val)
            } else { 255 };
            ExternCallResult::Value(Some(v))
        });
        interp.bind_by_name("lox_get_bound_arity", move |args| {
            let vm = unsafe{&mut*rt};
            let method = vm.bound_method_closure(args[0]);
            ExternCallResult::Value(Some(vm.closure_arity(method)))
        });
        interp.bind_by_name("lox_obj_type", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.obj_type_tag(args[0])))
        });

        // Class
        interp.bind_by_name("lox_make_class", move |args| {
            let vm = unsafe{&mut*rt};
            let name_val = vm.resolve_string(args[0] as usize);
            ExternCallResult::Value(Some(vm.alloc_class(name_val)))
        });
        interp.bind_by_name("lox_class_inherit", move |args| {
            let vm = unsafe{&mut*rt};
            if !is_obj(args[1]) || !vm.is_class(args[1]) {
                vm.runtime_error("Superclass must be a class.");
                return ExternCallResult::Value(None);
            }
            let super_table = vm.class_methods_table(args[1]);
            let own_table = vm.class_methods_table(args[0]);
            let merged = vm.table_merge(own_table, super_table);
            vm.set_class_methods_table(args[0], merged);
            vm.set_class_superclass(args[0], args[1]);
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_class_add_method", move |args| {
            let vm = unsafe{&mut*rt};
            let name_val = vm.resolve_string(args[1] as usize);
            vm.class_set_method(args[0], name_val, args[2]);
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_construct_instance", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.alloc_instance(args[0])))
        });
        interp.bind_by_name("lox_class_init_ptr", move |args| {
            let vm = unsafe{&mut*rt};
            let init_name = vm.intern_string("init");
            let v = if let Some(c) = vm.class_get_method(args[0], init_name) {
                vm.closure_func_idx(c)
            } else { 0 };
            ExternCallResult::Value(Some(v))
        });
        interp.bind_by_name("lox_class_init_closure", move |args| {
            let vm = unsafe{&mut*rt};
            let init_name = vm.intern_string("init");
            let v = vm.class_get_method(args[0], init_name).unwrap_or(nil_val());
            ExternCallResult::Value(Some(v))
        });

        // Properties
        interp.bind_by_name("lox_get_property", move |args| {
            let vm = unsafe{&mut*rt};
            let name_id = args[1] as usize;
            let prop_name = vm.string_text(name_id);
            let name_val = vm.resolve_string(name_id);

            if !is_obj(args[0]) || !vm.is_instance(args[0]) {
                vm.runtime_error("Only instances have properties.");
                return ExternCallResult::Value(Some(nil_val()));
            }
            if let Some(val) = vm.instance_get_field(args[0], name_val) {
                return ExternCallResult::Value(Some(val));
            }
            let class = vm.instance_class(args[0]);
            if let Some(method) = vm.class_get_method(class, name_val) {
                let bm = vm.alloc_bound_method(args[0], method);
                return ExternCallResult::Value(Some(bm));
            }
            vm.runtime_error(&format!("Undefined property '{}'.", prop_name));
            ExternCallResult::Value(Some(nil_val()))
        });
        interp.bind_by_name("lox_set_property", move |args| {
            let vm = unsafe{&mut*rt};
            let name_val = vm.resolve_string(args[1] as usize);

            if !is_obj(args[0]) || !vm.is_instance(args[0]) {
                vm.runtime_error("Only instances have fields.");
                return ExternCallResult::Value(Some(nil_val()));
            }
            vm.instance_set_field(args[0], name_val, args[2]);
            ExternCallResult::Value(Some(args[2]))
        });

        // Super
        interp.bind_by_name("lox_get_super", move |args| {
            let vm = unsafe{&mut*rt};
            let method_name_str = vm.string_text(args[2] as usize);
            let method_name = vm.resolve_string(args[2] as usize);

            let superclass = vm.class_superclass(args[1]);
            if is_nil(superclass) { return ExternCallResult::Value(Some(nil_val())); }

            if let Some(mc) = vm.class_get_method(superclass, method_name) {
                let bm = vm.alloc_bound_method(args[0], mc);
                return ExternCallResult::Value(Some(bm));
            }
            vm.runtime_error(&format!("Undefined property '{}'.", method_name_str));
            ExternCallResult::Value(Some(nil_val()))
        });

        // Native fn
        interp.bind_by_name("lox_make_native_fn", move |args| {
            let vm = unsafe{&mut*rt};
            let name_val = vm.resolve_string(args[0] as usize);
            let v = vm.alloc_native_fn(name_val, 0);
            ExternCallResult::Value(Some(v))
        });

        // Bound methods
        interp.bind_by_name("lox_bound_receiver", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.bound_receiver(args[0])))
        });
        interp.bind_by_name("lox_bound_method_closure", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.bound_method_closure(args[0])))
        });
        interp.bind_by_name("lox_bound_closure_func_ptr", move |args| {
            let vm = unsafe{&mut*rt};
            let method = vm.bound_method_closure(args[0]);
            ExternCallResult::Value(Some(vm.closure_func_idx(method)))
        });

        // String resolution
        interp.bind_by_name("lox_resolve_string", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.resolve_string(args[0] as usize)))
        });

        // GC alloc
        interp.bind_by_name("__gc_alloc__", move |args| {
            let vm = unsafe{&mut*rt};
            let ptr = vm.gc_alloc(args[0] as usize, args[1] as usize);
            ExternCallResult::Value(Some(ptr as u64))
        });
    }
}

/// Create a DynGcRuntime from a list of TypeInfo pointers.
fn create_gc_runtime(type_infos: &[&'static dynobj::TypeInfo]) -> DynGcRuntime {
    use dynlang::ObjType;
    // Create minimal ObjType entries to pass to DynGcRuntime::new
    let obj_types: Vec<ObjType> = type_infos.iter().enumerate().map(|(i, &ti)| {
        ObjType {
            name: format!("type_{}", i),
            type_info: ti,
            field_offsets: HashMap::new(),
            varlen: dynobj::VarLenKind::None,
        }
    }).collect();
    let tags = NanBoxTags { nil: TAG_NIL, bool_tag: TAG_BOOL, ptr: TAG_OBJ };
    DynGcRuntime::new(&GcConfig::leak(), &tags, &obj_types)
}
