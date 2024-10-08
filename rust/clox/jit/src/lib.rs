use std::{slice::from_raw_parts, io::Write};

use include_bindings::{ObjClosure, VM, Obj, ObjString};

use crate::include_bindings::{printValue, ObjType};

mod include_bindings;

#[derive(Debug, Copy, Clone)]
enum OpCode {
    Constant(u8),
    Nil,
    True,
    False,
    Pop,
    GetLocal(u8),
    SetLocal(u8),
    GetGlobal(u8),
    DefineGlobal(u8),
    SetGlobal(u8),
    GetUpvalue(u8),
    SetUpvalue(u8),
    GetProperty(u8),
    SetProperty(u8),
    GetSuper(u8),
    Equal,
    Greater,
    Less,
    Add,
    Subtract,
    Multiply,
    Divide,
    Not,
    Negate,
    Print,
    Jump(u8),
    JumpIfFalse(u8),
    Loop(u8),
    Call(u8),
    Invoke(u8),
    SuperInvoke(u8),
    Closure(u8, u8),
    CloseUpvalue,
    Return,
    Class(u8),
    Inherit,
    Method,
}

impl OpCode {
    fn from_bytes(bytes: &[u8]) -> (Self, u8) {
        match bytes[0] {
            0 => (OpCode::Constant(bytes[1]), 2),
            1 => (OpCode::Nil, 1),
            2 => (OpCode::True, 1),
            3 => (OpCode::False, 1),
            4 => (OpCode::Pop, 1),
            5 => (OpCode::GetLocal(bytes[1]), 2),
            6 => (OpCode::SetLocal(bytes[1]), 2),
            7 => (OpCode::GetGlobal(bytes[1]), 2),
            8 => (OpCode::DefineGlobal(bytes[1]), 2),
            9 => (OpCode::SetGlobal(bytes[1]), 2),
            10 => (OpCode::GetUpvalue(bytes[1]), 2),
            11 => (OpCode::SetUpvalue(bytes[1]), 2),
            12 => (OpCode::GetProperty(bytes[1]), 2),
            13 => (OpCode::SetProperty(bytes[1]), 2),
            14 => (OpCode::GetSuper(bytes[1]), 2),
            15 => (OpCode::Equal, 1),
            16 => (OpCode::Greater, 1),
            17 => (OpCode::Less, 1),
            18 => (OpCode::Add, 1),
            19 => (OpCode::Subtract, 1),
            20 => (OpCode::Multiply, 1),
            21 => (OpCode::Divide, 1),
            22 => (OpCode::Not, 1),
            23 => (OpCode::Negate, 1),
            24 => (OpCode::Print, 1),
            25 => (OpCode::Jump(bytes[1]), 2),
            26 => (OpCode::JumpIfFalse(bytes[1]), 2),
            27 => (OpCode::Loop(bytes[1]), 2),
            28 => (OpCode::Call(bytes[1]), 2),
            29 => (OpCode::Invoke(bytes[1]), 2),
            30 => (OpCode::SuperInvoke(bytes[1]), 2),
            31 => (OpCode::Closure(bytes[1], bytes[2]), 3),
            32 => (OpCode::CloseUpvalue, 1),
            33 => (OpCode::Return, 1),
            34 => (OpCode::Class(bytes[1]), 2),
            35 => (OpCode::Inherit, 1),
            36 => (OpCode::Method, 1),
            _ => panic!("Unknown opcode: {}", bytes[0]),
        }
    }
}


const SIGN_BIT: u64 = 0x8000000000000000;
const QNAN: u64 = 0x7ffc000000000000;

const TAG_NIL: u64 = 1;
const TAG_FALSE: u64 = 2;
const TAG_TRUE: u64 = 3;

type Value = u64;

const TRUE_VAL: Value = QNAN | TAG_TRUE;
const NIL_VAL: Value = QNAN | TAG_NIL;

fn is_bool(value: Value) -> bool {
    (value | 1) == TRUE_VAL
}

fn is_nil(value: Value) -> bool {
    value == NIL_VAL
}

fn is_number(value: Value) -> bool {
    (value & QNAN) != QNAN
}

fn is_obj(value: Value) -> bool {
    (value & (QNAN | SIGN_BIT)) == (QNAN | SIGN_BIT)
}

fn as_obj(value: u64) -> *mut Obj {
    let mask = !(SIGN_BIT | QNAN);
    (value & mask) as *mut Obj
}

fn as_string(value: *mut Obj) -> *mut ObjString {
    let string = unsafe { std::mem::transmute::<*mut Obj, *mut ObjString>(value) };
    string
}


fn value_to_num(value: Value) -> f64 {
    let num = unsafe { std::mem::transmute::<Value, f64>(value) };
    num
}


#[no_mangle]
pub extern "C" fn on_closure_call(vm: *mut VM, obj_closure: ObjClosure) {

    unsafe {
        if vm.is_null() {
            return;
        }
        let vm = *vm;
        let frame = vm.frames[(vm.frameCount as usize).saturating_sub(1)];
        let closure = obj_closure;
        let function = *closure.function;
        if function.name == std::ptr::null_mut() {
            return;
        }
        let code = function.chunk.code;
        let code = from_raw_parts(code as *mut u8, function.chunk.count as usize);
        let mut decoded = Vec::new();
        let mut i = 0;
        while i < code.len() {
            let (op, len) = OpCode::from_bytes(&code[i..]);
            decoded.push(op);
            match op {
                // Looks like I'm going to need to deal with nan-boxing for my jitted code.
                OpCode::Constant(i) => {
                    let values = function.chunk.constants.values;
                    println!("constants: {:?}", function.chunk.constants);

                    let values: &[u64] = from_raw_parts(values as *mut u64, function.chunk.constants.count as usize);
                    let value: u64 = values[i as usize];
                    if is_number(value) {
                        println!("constant({}): {:?}", i, value_to_num(value));
                    } else if is_bool(value) {
                        println!("constant({}): {:?}", i, value == TRUE_VAL);
                    } else if is_nil(value) {
                        println!("constant({}): {:?}", i, "nil");
                    } else if is_obj(value) {
                        let obj_pointer = as_obj(value);
                        let obj = *obj_pointer;
                        match obj.type_ {
                            ObjType::OBJ_BOUND_METHOD => println!("constant({}): {:?}", i, "bound method"),
                            ObjType::OBJ_CLASS => println!("constant({}): {:?}", i, "class"),
                            ObjType::OBJ_CLOSURE => println!("constant({}): {:?}", i, "closure"),
                            ObjType::OBJ_FUNCTION => println!("constant({}): {:?}", i, "function"),
                            ObjType::OBJ_INSTANCE => println!("constant({}): {:?}", i, "instance"),
                            ObjType::OBJ_NATIVE => println!("constant({}): {:?}", i, "native"),
                            ObjType::OBJ_STRING => {
                                let string_pointer = as_string(obj_pointer);
                                let c_string = (*string_pointer).chars;
                                let rust_slice = from_raw_parts(c_string as *mut u8, (*string_pointer).length as usize);
                                let rust_string = std::str::from_utf8(rust_slice).unwrap();
                                println!("constant({}): {:?}", i, rust_string);
                            },
                            ObjType::OBJ_UPVALUE => println!("constant({}): {:?}", i, "upvalue"),
                        }
                    } else {
                        println!("constant({}): {:?}", i, "unknown");
                    }
                    
                }
                _ => {}
            }
            i += len as usize;
        }

        println!("decoded: {:?}", decoded);

        let name = *function.name;
        let s_name = from_raw_parts(name.chars as *mut u8, name.length as usize);
        let s_name = std::str::from_utf8(s_name).unwrap();
        println!("on_closure_call: {}", s_name);
    }
}

