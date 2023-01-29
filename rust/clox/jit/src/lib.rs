use std::slice::from_raw_parts;

use bindings::{ObjClosure, VM};

use crate::bindings::OpCode;

mod bindings;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[no_mangle]
pub extern "C" fn on_closure_call(vm: *mut VM, obj_closure: ObjClosure) {
    unsafe {
        let closure = obj_closure;
        let function = *closure.function;
        if function.name == std::ptr::null_mut() {
            return;
        }
        let code = function.chunk.code;
        let code = from_raw_parts(code as *mut u8, function.chunk.count as usize);
        let name = *function.name;
        let s_name = from_raw_parts(name.chars as *mut u8, name.length as usize);
        let s_name = std::str::from_utf8(s_name).unwrap();
        println!("on_closure_call: {}", s_name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
