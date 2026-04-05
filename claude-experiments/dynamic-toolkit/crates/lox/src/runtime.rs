/// Lox runtime functions operating on NanBox-encoded u64 values.
///
/// These are bound as extern functions in the dynir interpreter.
/// They handle all dynamic operations (type checking, dispatch, etc).
use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr;

use dynvalue::{NanBox, TagScheme};

use crate::object::*;
use crate::value::*;

/// Thread-local runtime context for extern callbacks.
thread_local! {
    static RUNTIME: RefCell<Option<*mut LoxRuntime>> = const { RefCell::new(None) };
}

pub fn with_runtime<R>(f: impl FnOnce(&mut LoxRuntime) -> R) -> R {
    RUNTIME.with(|r| {
        let ptr = r.borrow().expect("LoxRuntime not installed");
        f(unsafe { &mut *ptr })
    })
}

pub fn install_runtime<R>(rt: &mut LoxRuntime, f: impl FnOnce() -> R) -> R {
    RUNTIME.with(|r| {
        let prev = r.borrow().clone();
        *r.borrow_mut() = Some(rt as *mut LoxRuntime);
        let result = f();
        *r.borrow_mut() = prev;
        result
    });
    unreachable!()
}

// Workaround: install returns a guard
pub struct RuntimeGuard {
    prev: Option<*mut LoxRuntime>,
}

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        RUNTIME.with(|r| {
            *r.borrow_mut() = self.prev;
        });
    }
}

pub fn install_runtime_guard(rt: &mut LoxRuntime) -> RuntimeGuard {
    let prev = RUNTIME.with(|r| {
        let prev = *r.borrow();
        *r.borrow_mut() = Some(rt as *mut LoxRuntime);
        prev
    });
    RuntimeGuard { prev }
}

/// Core Lox runtime state, shared across all extern function calls.
pub struct LoxRuntime {
    pub heap: GcHeap,
    pub globals: HashMap<*mut ObjString, Value>,
    pub open_upvalues: *mut ObjUpvalue,
    pub init_string: *mut ObjString,
    pub output: String,
    /// Argument staging area for function calls
    pub arg_buffer: Vec<Value>,
    /// Callback to re-enter the interpreter for a given function
    pub call_handler: Option<Box<dyn Fn(usize, &[Value]) -> Value>>,
    /// Track if a runtime error occurred
    pub had_error: bool,
    pub error_message: String,
}

impl LoxRuntime {
    pub fn new() -> Self {
        let mut heap = GcHeap::new();
        let init_string = heap.alloc_string("init".to_string());
        LoxRuntime {
            heap,
            globals: HashMap::new(),
            open_upvalues: ptr::null_mut(),
            init_string,
            output: String::new(),
            arg_buffer: Vec::with_capacity(256),
            call_handler: None,
            had_error: false,
            error_message: String::new(),
        }
    }

    pub fn runtime_error(&mut self, msg: &str) {
        self.had_error = true;
        self.error_message = msg.to_string();
        eprintln!("{}", msg);
    }

    // ── Arithmetic ──────────────────────────────────────────────

    pub fn lox_add(&mut self, a: Value, b: Value) -> Value {
        if is_number(a) && is_number(b) {
            return number_val(as_number(a) + as_number(b));
        }
        if is_string(a) && is_string(b) {
            let result = unsafe {
                let sa = &(*(as_obj(a) as *mut ObjString)).chars;
                let sb = &(*(as_obj(b) as *mut ObjString)).chars;
                format!("{}{}", sa, sb)
            };
            let s = self.heap.alloc_string(result);
            return obj_val(s as *mut Obj);
        }
        self.runtime_error("Operands must be two numbers or two strings.");
        nil_val()
    }

    pub fn lox_sub(&mut self, a: Value, b: Value) -> Value {
        if is_number(a) && is_number(b) {
            return number_val(as_number(a) - as_number(b));
        }
        self.runtime_error("Operands must be numbers.");
        nil_val()
    }

    pub fn lox_mul(&mut self, a: Value, b: Value) -> Value {
        if is_number(a) && is_number(b) {
            return number_val(as_number(a) * as_number(b));
        }
        self.runtime_error("Operands must be numbers.");
        nil_val()
    }

    pub fn lox_div(&mut self, a: Value, b: Value) -> Value {
        if is_number(a) && is_number(b) {
            return number_val(as_number(a) / as_number(b));
        }
        self.runtime_error("Operands must be numbers.");
        nil_val()
    }

    pub fn lox_negate(&mut self, a: Value) -> Value {
        if is_number(a) {
            return number_val(-as_number(a));
        }
        self.runtime_error("Operand must be a number.");
        nil_val()
    }

    pub fn lox_not(&self, a: Value) -> Value {
        bool_val(is_falsey(a))
    }

    // ── Comparison ──────────────────────────────────────────────

    pub fn lox_equal(&self, a: Value, b: Value) -> Value {
        bool_val(values_equal(a, b))
    }

    pub fn lox_greater(&mut self, a: Value, b: Value) -> Value {
        if is_number(a) && is_number(b) {
            return bool_val(as_number(a) > as_number(b));
        }
        self.runtime_error("Operands must be numbers.");
        nil_val()
    }

    pub fn lox_less(&mut self, a: Value, b: Value) -> Value {
        if is_number(a) && is_number(b) {
            return bool_val(as_number(a) < as_number(b));
        }
        self.runtime_error("Operands must be numbers.");
        nil_val()
    }

    pub fn lox_is_falsey(&self, a: Value) -> u64 {
        if is_falsey(a) { 1 } else { 0 }
    }

    // ── I/O ─────────────────────────────────────────────────────

    pub fn lox_print(&mut self, a: Value) {
        let s = value_to_string(a);
        println!("{}", s);
        self.output.push_str(&s);
        self.output.push('\n');
    }

    pub fn lox_clock(&self) -> Value {
        let duration = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        number_val(duration.as_secs_f64())
    }

    // ── Globals ─────────────────────────────────────────────────

    pub fn lox_get_global(&mut self, name: Value) -> Value {
        let name_str = as_obj(name) as *mut ObjString;
        if let Some(&val) = self.globals.get(&name_str) {
            val
        } else {
            let name_s = unsafe { (*name_str).chars.clone() };
            self.runtime_error(&format!("Undefined variable '{}'.", name_s));
            nil_val()
        }
    }

    pub fn lox_set_global(&mut self, name: Value, value: Value) -> Value {
        let name_str = as_obj(name) as *mut ObjString;
        if self.globals.contains_key(&name_str) {
            self.globals.insert(name_str, value);
        } else {
            let name_s = unsafe { (*name_str).chars.clone() };
            self.runtime_error(&format!("Undefined variable '{}'.", name_s));
        }
        value
    }

    pub fn lox_define_global(&mut self, name: Value, value: Value) {
        let name_str = as_obj(name) as *mut ObjString;
        self.globals.insert(name_str, value);
    }

    // ── Properties ──────────────────────────────────────────────

    pub fn lox_get_property(&mut self, instance_val: Value, name: Value) -> Value {
        if !is_instance(instance_val) {
            self.runtime_error("Only instances have properties.");
            return nil_val();
        }
        let instance = as_obj(instance_val) as *mut ObjInstance;
        let name_str = as_obj(name) as *mut ObjString;

        unsafe {
            if let Some(&val) = (*instance).fields.get(&name_str) {
                return val;
            }
            // Try methods
            let class = (*instance).class;
            if let Some(&method) = (*class).methods.get(&name_str) {
                let bound = self.heap.alloc_bound_method(
                    instance_val,
                    as_obj(method) as *mut ObjClosure,
                );
                return obj_val(bound as *mut Obj);
            }
            self.runtime_error(&format!(
                "Undefined property '{}'.",
                (*name_str).chars
            ));
            nil_val()
        }
    }

    pub fn lox_set_property(&mut self, instance_val: Value, name: Value, value: Value) -> Value {
        if !is_instance(instance_val) {
            self.runtime_error("Only instances have fields.");
            return nil_val();
        }
        let instance = as_obj(instance_val) as *mut ObjInstance;
        let name_str = as_obj(name) as *mut ObjString;
        unsafe { (*instance).fields.insert(name_str, value) };
        value
    }

    // ── Calls ───────────────────────────────────────────────────

    pub fn lox_call(&mut self, callee: Value, arg_count: u64) -> Value {
        let argc = arg_count as u8;
        if !is_obj(callee) {
            self.runtime_error("Can only call functions and classes.");
            return nil_val();
        }
        let obj = as_obj(callee);
        unsafe {
            match (*obj).obj_type {
                ObjType::Closure => {
                    let closure = obj as *mut ObjClosure;
                    let function = (*closure).function;
                    let arity = (*function).arity;
                    if argc != arity {
                        self.runtime_error(&format!(
                            "Expected {} arguments but got {}.",
                            arity, argc
                        ));
                        return nil_val();
                    }
                    // Get the function ID (stored as the function index in our system)
                    let func_id = (*function).func_id;
                    let mut args = vec![callee]; // closure is first arg
                    for i in 0..argc as usize {
                        args.push(self.arg_buffer[i]);
                    }
                    if let Some(ref handler) = self.call_handler {
                        let handler_fn = handler.as_ref() as *const dyn Fn(usize, &[Value]) -> Value;
                        let result = (*handler_fn)(func_id, &args);
                        return result;
                    }
                    nil_val()
                }
                ObjType::Native => {
                    let native = obj as *mut ObjNative;
                    let func = (*native).function;
                    func(argc, self.arg_buffer.as_ptr())
                }
                ObjType::Class => {
                    let class = obj as *mut ObjClass;
                    let instance = self.heap.alloc_instance(class);
                    let instance_val = obj_val(instance as *mut Obj);

                    if let Some(&initializer) = (*class).methods.get(&self.init_string) {
                        let closure = as_obj(initializer) as *mut ObjClosure;
                        let function = (*closure).function;
                        let arity = (*function).arity;
                        if argc != arity {
                            self.runtime_error(&format!(
                                "Expected {} arguments but got {}.",
                                arity, argc
                            ));
                            return nil_val();
                        }
                        let func_id = (*function).func_id;
                        let mut args = vec![initializer]; // closure
                        // Override slot 0 with instance (for "this")
                        args[0] = initializer;
                        // But actually, we need to pass instance as the receiver...
                        // In clox, the instance is stored at stack[0] of the called frame
                        // For our calling convention: args[0] = closure, but the instance
                        // is what slot 0 should resolve to for "this"
                        // Let's pass instance as the first "arg" implicitly
                        // Actually, let's just put instance in arg_buffer position -1
                        // No — let me use a simpler approach: store instance on the arg buffer
                        self.arg_buffer.insert(0, instance_val);
                        let mut call_args = vec![initializer];
                        for i in 0..=argc as usize {
                            call_args.push(self.arg_buffer[i]);
                        }
                        if let Some(ref handler) = self.call_handler {
                            let handler_fn = handler.as_ref() as *const dyn Fn(usize, &[Value]) -> Value;
                            (*handler_fn)(func_id, &call_args);
                        }
                        self.arg_buffer.drain(..1); // remove inserted instance
                        return instance_val;
                    } else if argc != 0 {
                        self.runtime_error(&format!(
                            "Expected 0 arguments but got {}.",
                            argc
                        ));
                        return nil_val();
                    }
                    instance_val
                }
                ObjType::BoundMethod => {
                    let bound = obj as *mut ObjBoundMethod;
                    let closure = (*bound).method;
                    let function = (*closure).function;
                    let arity = (*function).arity;
                    if argc != arity {
                        self.runtime_error(&format!(
                            "Expected {} arguments but got {}.",
                            arity, argc
                        ));
                        return nil_val();
                    }
                    let func_id = (*function).func_id;
                    let mut args = vec![obj_val(closure as *mut Obj)];
                    // Replace slot 0 with receiver for "this"
                    args.push((*bound).receiver); // this goes in "slot 0" position
                    // Hmm, this calling convention is tricky...
                    // Let me simplify: the call handler gets func_id and a flat args array
                    // args[0] = closure value
                    // For methods, we need "this" to be accessible
                    // In clox, "this" is at slot 0 of the frame
                    // In our IR, slot 0 = closure param, which we override
                    // Actually, just pass the receiver as if it's the closure
                    let mut call_args = vec![obj_val(closure as *mut Obj)];
                    for i in 0..argc as usize {
                        call_args.push(self.arg_buffer[i]);
                    }
                    // But we need "this" — it should be in local 0
                    // For methods, local 0 = "this" (the receiver)
                    // Let me overwrite call_args[0] with the receiver
                    // Wait no, args[0] is the closure. For "this", the compiler
                    // uses GetLocal(0) which in clox points to the function slot
                    // which is overwritten with the receiver for methods.
                    // In our translation, slot 0 of the frame IS the first param,
                    // which is the closure. We need to replace it with the receiver.
                    // So: call_args[0] should be the receiver when it's a bound method.
                    call_args[0] = (*bound).receiver;
                    if let Some(ref handler) = self.call_handler {
                        let handler_fn = handler.as_ref() as *const dyn Fn(usize, &[Value]) -> Value;
                        return (*handler_fn)(func_id, &call_args);
                    }
                    nil_val()
                }
                _ => {
                    self.runtime_error("Can only call functions and classes.");
                    nil_val()
                }
            }
        }
    }

    pub fn lox_push_arg(&mut self, index: u64, value: Value) {
        let idx = index as usize;
        if idx >= self.arg_buffer.len() {
            self.arg_buffer.resize(idx + 1, nil_val());
        }
        self.arg_buffer[idx] = value;
    }

    // ── Invoke ──────────────────────────────────────────────────

    pub fn lox_invoke(&mut self, receiver: Value, name: Value, arg_count: u64) -> Value {
        if !is_instance(receiver) {
            self.runtime_error("Only instances have methods.");
            return nil_val();
        }
        let instance = as_obj(receiver) as *mut ObjInstance;
        let name_str = as_obj(name) as *mut ObjString;

        unsafe {
            // Check fields first
            if let Some(&field) = (*instance).fields.get(&name_str) {
                return self.lox_call(field, arg_count);
            }
            // Method call
            let class = (*instance).class;
            if let Some(&method) = (*class).methods.get(&name_str) {
                let closure = as_obj(method) as *mut ObjClosure;
                let function = (*closure).function;
                let arity = (*function).arity;
                let argc = arg_count as u8;
                if argc != arity {
                    self.runtime_error(&format!(
                        "Expected {} arguments but got {}.",
                        arity, argc
                    ));
                    return nil_val();
                }
                let func_id = (*function).func_id;
                let mut args = vec![receiver]; // "this" is the receiver
                for i in 0..argc as usize {
                    args.push(self.arg_buffer[i]);
                }
                if let Some(ref handler) = self.call_handler {
                    let handler_fn = handler.as_ref() as *const dyn Fn(usize, &[Value]) -> Value;
                    return (*handler_fn)(func_id, &args);
                }
                nil_val()
            } else {
                self.runtime_error(&format!(
                    "Undefined property '{}'.",
                    (*name_str).chars
                ));
                nil_val()
            }
        }
    }

    pub fn lox_super_invoke(
        &mut self,
        receiver: Value,
        name: Value,
        superclass: Value,
        arg_count: u64,
    ) -> Value {
        let class = as_obj(superclass) as *mut ObjClass;
        let name_str = as_obj(name) as *mut ObjString;

        unsafe {
            if let Some(&method) = (*class).methods.get(&name_str) {
                let closure = as_obj(method) as *mut ObjClosure;
                let function = (*closure).function;
                let func_id = (*function).func_id;
                let argc = arg_count as u8;
                let mut args = vec![receiver];
                for i in 0..argc as usize {
                    args.push(self.arg_buffer[i]);
                }
                if let Some(ref handler) = self.call_handler {
                    let handler_fn = handler.as_ref() as *const dyn Fn(usize, &[Value]) -> Value;
                    return (*handler_fn)(func_id, &args);
                }
                nil_val()
            } else {
                self.runtime_error(&format!(
                    "Undefined property '{}'.",
                    (*name_str).chars
                ));
                nil_val()
            }
        }
    }

    // ── Closures/Upvalues ───────────────────────────────────────

    pub fn lox_get_upvalue(&self, closure: Value, index: u64) -> Value {
        let closure_obj = as_obj(closure) as *mut ObjClosure;
        unsafe {
            let uv = (*closure_obj).upvalues[index as usize];
            *(*uv).location
        }
    }

    pub fn lox_set_upvalue(&self, closure: Value, index: u64, value: Value) {
        let closure_obj = as_obj(closure) as *mut ObjClosure;
        unsafe {
            let uv = (*closure_obj).upvalues[index as usize];
            *(*uv).location = value;
        }
    }

    pub fn lox_close_upvalue(&mut self, slot_ptr: *mut Value) {
        while !self.open_upvalues.is_null()
            && unsafe { (*self.open_upvalues).location as *const Value } >= slot_ptr as *const Value
        {
            let upvalue = self.open_upvalues;
            unsafe {
                (*upvalue).closed = *(*upvalue).location;
                (*upvalue).location = &mut (*upvalue).closed;
                self.open_upvalues = (*upvalue).next;
            }
        }
    }

    pub fn lox_capture_upvalue(&mut self, slot_ptr: *mut Value) -> *mut ObjUpvalue {
        let mut prev: *mut ObjUpvalue = ptr::null_mut();
        let mut uv = self.open_upvalues;
        while !uv.is_null() && unsafe { (*uv).location } > slot_ptr {
            prev = uv;
            uv = unsafe { (*uv).next };
        }
        if !uv.is_null() && unsafe { (*uv).location } == slot_ptr {
            return uv;
        }
        let created = self.heap.alloc_upvalue(slot_ptr);
        unsafe { (*created).next = uv };
        if prev.is_null() {
            self.open_upvalues = created;
        } else {
            unsafe { (*prev).next = created };
        }
        created
    }

    // ── Classes ─────────────────────────────────────────────────

    pub fn lox_make_class(&mut self, name: Value) -> Value {
        let name_str = as_obj(name) as *mut ObjString;
        let class = self.heap.alloc_class(name_str);
        obj_val(class as *mut Obj)
    }

    pub fn lox_inherit(&mut self, subclass: Value, superclass: Value) -> Value {
        if !is_class(superclass) {
            self.runtime_error("Superclass must be a class.");
            return nil_val();
        }
        let super_cls = as_obj(superclass) as *mut ObjClass;
        let sub_cls = as_obj(subclass) as *mut ObjClass;
        unsafe {
            (*sub_cls).methods = (*super_cls).methods.clone();
        }
        nil_val()
    }

    pub fn lox_define_method(&mut self, class: Value, name: Value, method: Value) {
        let cls = as_obj(class) as *mut ObjClass;
        let name_str = as_obj(name) as *mut ObjString;
        unsafe { (*cls).methods.insert(name_str, method) };
    }

    pub fn lox_get_super(&mut self, receiver: Value, name: Value, superclass: Value) -> Value {
        let class = as_obj(superclass) as *mut ObjClass;
        let name_str = as_obj(name) as *mut ObjString;
        unsafe {
            if let Some(&method) = (*class).methods.get(&name_str) {
                let bound = self.heap.alloc_bound_method(
                    receiver,
                    as_obj(method) as *mut ObjClosure,
                );
                obj_val(bound as *mut Obj)
            } else {
                self.runtime_error(&format!(
                    "Undefined property '{}'.",
                    (*name_str).chars
                ));
                nil_val()
            }
        }
    }
}
