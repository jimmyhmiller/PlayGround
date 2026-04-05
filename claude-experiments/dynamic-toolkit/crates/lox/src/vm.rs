/// Lox VM: parse → AST → lower to dynir → run via dynir interpreter.
use std::collections::HashMap;

use dynir::interp::{ExternCallResult, InterpResult, ModuleInterpreter, NoGcRoots};
use dynvalue::NanBox;

use crate::ast;
use crate::lower;
use crate::object::*;
use crate::parser::Parser;
use crate::value::*;

pub enum InterpretResult {
    Ok,
    CompileError,
    RuntimeError,
}

pub struct VM {
    heap: GcHeap,
    globals: HashMap<*mut ObjString, Value>,
    /// String table: hash ID → interned ObjString
    string_table: HashMap<u64, *mut ObjString>,
    had_error: bool,
}

impl VM {
    pub fn new() -> Self {
        VM {
            heap: GcHeap::new(),
            globals: HashMap::new(),
            string_table: HashMap::new(),
            had_error: false,
        }
    }

    pub fn interpret(&mut self, source: &str) -> InterpretResult {
        // Phase 1: Parse
        let program = match Parser::parse(source) {
            Some(p) => p,
            None => return InterpretResult::CompileError,
        };

        // Phase 2: Lower AST → dynir Module
        let lowered = lower::lower(&program);

        // Phase 3: Execute via dynir interpreter
        let roots = NoGcRoots;
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&lowered.module, &roots);

        // Bind all extern runtime functions
        let rt = self as *mut VM;
        bind_runtime(&mut interp, rt);

        match interp.run(lowered.entry, &[]) {
            Ok(InterpResult::Value(_)) | Ok(InterpResult::Void) => {
                if self.had_error {
                    InterpretResult::RuntimeError
                } else {
                    InterpretResult::Ok
                }
            }
            Ok(InterpResult::Deopt { .. }) => InterpretResult::RuntimeError,
            Err(e) => {
                eprintln!("Internal error: {:?}", e);
                InterpretResult::RuntimeError
            }
        }
    }

    fn intern_string(&mut self, s: &str) -> *mut ObjString {
        let id = crate::lower::string_id(s);
        if let Some(&existing) = self.string_table.get(&id) {
            return existing;
        }
        let obj = self.heap.alloc_string(s.to_string());
        self.string_table.insert(id, obj);
        obj
    }

    fn runtime_error(&mut self, msg: &str) {
        eprintln!("{}", msg);
        self.had_error = true;
    }
}

fn bind_runtime<'a>(interp: &mut ModuleInterpreter<'a, NanBox, NoGcRoots>, rt: *mut VM) {
    interp.bind_by_name("lox_add", move |args| {
        let vm = unsafe { &mut *rt };
        let a = args[0];
        let b = args[1];
        if is_number(a) && is_number(b) {
            return ExternCallResult::Value(Some(number_val(as_number(a) + as_number(b))));
        }
        if is_string(a) && is_string(b) {
            let result = unsafe {
                format!("{}{}", (*(as_obj(a) as *mut ObjString)).chars, (*(as_obj(b) as *mut ObjString)).chars)
            };
            let s = vm.heap.alloc_string(result);
            return ExternCallResult::Value(Some(obj_val(s as *mut Obj)));
        }
        vm.runtime_error("Operands must be two numbers or two strings.");
        ExternCallResult::Value(Some(nil_val()))
    });

    interp.bind_by_name("lox_sub", move |args| {
        let vm = unsafe { &mut *rt };
        if is_number(args[0]) && is_number(args[1]) {
            ExternCallResult::Value(Some(number_val(as_number(args[0]) - as_number(args[1]))))
        } else {
            vm.runtime_error("Operands must be numbers.");
            ExternCallResult::Value(Some(nil_val()))
        }
    });

    interp.bind_by_name("lox_mul", move |args| {
        let vm = unsafe { &mut *rt };
        if is_number(args[0]) && is_number(args[1]) {
            ExternCallResult::Value(Some(number_val(as_number(args[0]) * as_number(args[1]))))
        } else {
            vm.runtime_error("Operands must be numbers.");
            ExternCallResult::Value(Some(nil_val()))
        }
    });

    interp.bind_by_name("lox_div", move |args| {
        let vm = unsafe { &mut *rt };
        if is_number(args[0]) && is_number(args[1]) {
            ExternCallResult::Value(Some(number_val(as_number(args[0]) / as_number(args[1]))))
        } else {
            vm.runtime_error("Operands must be numbers.");
            ExternCallResult::Value(Some(nil_val()))
        }
    });

    interp.bind_by_name("lox_negate", move |args| {
        let vm = unsafe { &mut *rt };
        if is_number(args[0]) {
            ExternCallResult::Value(Some(number_val(-as_number(args[0]))))
        } else {
            vm.runtime_error("Operand must be a number.");
            ExternCallResult::Value(Some(nil_val()))
        }
    });

    interp.bind_by_name("lox_not", move |_args| {
        ExternCallResult::Value(Some(bool_val(is_falsey(_args[0]))))
    });

    interp.bind_by_name("lox_equal", move |args| {
        ExternCallResult::Value(Some(bool_val(values_equal(args[0], args[1]))))
    });

    interp.bind_by_name("lox_greater", move |args| {
        let vm = unsafe { &mut *rt };
        if is_number(args[0]) && is_number(args[1]) {
            ExternCallResult::Value(Some(bool_val(as_number(args[0]) > as_number(args[1]))))
        } else {
            vm.runtime_error("Operands must be numbers.");
            ExternCallResult::Value(Some(nil_val()))
        }
    });

    interp.bind_by_name("lox_less", move |args| {
        let vm = unsafe { &mut *rt };
        if is_number(args[0]) && is_number(args[1]) {
            ExternCallResult::Value(Some(bool_val(as_number(args[0]) < as_number(args[1]))))
        } else {
            vm.runtime_error("Operands must be numbers.");
            ExternCallResult::Value(Some(nil_val()))
        }
    });

    interp.bind_by_name("lox_is_falsey", move |args| {
        ExternCallResult::Value(Some(if is_falsey(args[0]) { 1u64 } else { 0u64 }))
    });

    interp.bind_by_name("lox_print", move |args| {
        print_value(args[0]);
        println!();
        ExternCallResult::Value(None)
    });

    interp.bind_by_name("lox_define_global", move |args| {
        let vm = unsafe { &mut *rt };
        let name = as_obj(args[0]) as *mut ObjString;
        vm.globals.insert(name, args[1]);
        ExternCallResult::Value(None)
    });

    interp.bind_by_name("lox_get_global", move |args| {
        let vm = unsafe { &mut *rt };
        let name = as_obj(args[0]) as *mut ObjString;
        if let Some(&val) = vm.globals.get(&name) {
            ExternCallResult::Value(Some(val))
        } else {
            let name_s = unsafe { (*name).chars.clone() };
            vm.runtime_error(&format!("Undefined variable '{}'.", name_s));
            ExternCallResult::Value(Some(nil_val()))
        }
    });

    interp.bind_by_name("lox_set_global", move |args| {
        let vm = unsafe { &mut *rt };
        let name = as_obj(args[0]) as *mut ObjString;
        if vm.globals.contains_key(&name) {
            vm.globals.insert(name, args[1]);
            ExternCallResult::Value(Some(args[1]))
        } else {
            let name_s = unsafe { (*name).chars.clone() };
            vm.runtime_error(&format!("Undefined variable '{}'.", name_s));
            ExternCallResult::Value(Some(nil_val()))
        }
    });

    interp.bind_by_name("lox_make_string", move |args| {
        // args[0] is the string hash ID. Look up or create the string.
        // This is a placeholder — in a real implementation, we'd have the actual
        // string content available. For now, this won't work correctly.
        // We need to pass the actual string content somehow.
        let vm = unsafe { &mut *rt };
        // The ID is the FNV hash of the string content. We need to map this back.
        // This is a fundamental design issue — let me store strings differently.
        ExternCallResult::Value(Some(nil_val()))
    });

    // Stubs for features that need more work
    interp.bind_by_name("lox_get_property", move |_args| {
        let vm = unsafe { &mut *rt };
        vm.runtime_error("Properties not yet supported in dynir mode.");
        ExternCallResult::Value(Some(nil_val()))
    });
    interp.bind_by_name("lox_set_property", move |_args| {
        let vm = unsafe { &mut *rt };
        vm.runtime_error("Properties not yet supported in dynir mode.");
        ExternCallResult::Value(Some(nil_val()))
    });
    interp.bind_by_name("lox_get_super", move |_args| {
        ExternCallResult::Value(Some(nil_val()))
    });
    interp.bind_by_name("lox_call_value", move |_args| {
        let vm = unsafe { &mut *rt };
        vm.runtime_error("Function calls not yet supported in dynir mode.");
        ExternCallResult::Value(Some(nil_val()))
    });
    interp.bind_by_name("lox_invoke", move |_args| {
        ExternCallResult::Value(Some(nil_val()))
    });
    interp.bind_by_name("lox_super_invoke", move |_args| {
        ExternCallResult::Value(Some(nil_val()))
    });
    interp.bind_by_name("lox_make_class", move |_args| {
        ExternCallResult::Value(Some(nil_val()))
    });
    interp.bind_by_name("lox_inherit", move |_args| {
        ExternCallResult::Value(None)
    });
    interp.bind_by_name("lox_define_method", move |_args| {
        ExternCallResult::Value(None)
    });
}
