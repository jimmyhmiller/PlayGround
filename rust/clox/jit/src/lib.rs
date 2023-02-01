use std::slice::from_raw_parts;

use include_bindings::{ObjClosure, VM, *};


mod include_bindings;





struct OpCodeParser {
    opcode_encoding: Vec<(OpCode, usize)>,
}

impl OpCodeParser {

    const fn new() -> OpCodeParser {
        OpCodeParser {
            opcode_encoding: Vec::new(),
        }
    }

    fn get_instance() -> &'static mut OpCodeParser {
        unsafe {
            &mut OP_CODE_PARSER
        }
    }

    fn load_opcode_and_length_from_csv(csv: &str) -> Vec<(OpCode, usize)> {
        let mut result = Vec::new();
        let lines = csv.split("\n");
        for line in lines {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split(",");
            let opcode = parts.next().unwrap();
            let length = parts.next().unwrap();
            let opcode = OpCode::from_str(opcode);
            let length = length.parse::<usize>().unwrap();
            result.push((OpCode::from(opcode), length));
        }
        result
    }

    fn cache_opcode_and_length(&mut self) {
        // Because I'm including I have to rebuild to reload.
        let csv = include_str!("../resources/opcodes.csv");
        self.opcode_encoding = OpCodeParser::load_opcode_and_length_from_csv(&csv);
    }

    fn get_opcode_length(&self, opcode: OpCode) -> usize {
        for (op, length) in &self.opcode_encoding {
            if *op == opcode {
                return *length;
            }
        }
        panic!("opcode not found: {:?}", opcode);
    }

    fn decode_opcodes(&mut self, code: &[u8]) -> Vec<(OpCode, Vec<u8>)> {
        self.cache_opcode_and_length();
        let mut result = Vec::new();
        let mut index = 0;
        while index < code.len() {
            let opcode = code[index];
            // TODO
            let opcode = OpCode::from_u32(opcode as u32);
            let length = self.get_opcode_length(opcode);
            // read length bytes
            let mut bytes = Vec::new();
            for i in 0..(length - 1) {
                bytes.push(code[index + 1 + i]);
            }
            result.push((opcode, bytes));
            index += length;
        }
        result
    }

}



static mut OP_CODE_PARSER: OpCodeParser = OpCodeParser::new();



#[no_mangle]
pub extern "C" fn on_closure_call(vm: *mut VM, obj_closure: ObjClosure) {
    let op_code_parser = OpCodeParser::get_instance();

    unsafe {
        let closure = obj_closure;
        let function = *closure.function;
        if function.name == std::ptr::null_mut() {
            return;
        }
        let code = function.chunk.code;
        let code = from_raw_parts(code as *mut u8, function.chunk.count as usize);

        let decoded = op_code_parser.decode_opcodes(code);
        println!("decoded: {:?}", decoded.iter().map(|(op, len)| (*op, len)).collect::<Vec<_>>());

        let name = *function.name;
        let s_name = from_raw_parts(name.chars as *mut u8, name.length as usize);
        let s_name = std::str::from_utf8(s_name).unwrap();
        println!("on_closure_call: {}", s_name);
    }
}

