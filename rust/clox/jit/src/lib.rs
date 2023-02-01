use std::slice::from_raw_parts;

use bindings::{ObjClosure, VM, *};

use crate::bindings::OpCode;

mod bindings;



fn op_code_from_str(opcode: &str) -> OpCode {
    match opcode {
        "OP_CONSTANT" => OpCode_OP_CONSTANT,
        "OP_NIL" => OpCode_OP_NIL,
        "OP_TRUE" => OpCode_OP_TRUE,
        "OP_FALSE" => OpCode_OP_FALSE,
        "OP_POP" => OpCode_OP_POP,
        "OP_GET_LOCAL" => OpCode_OP_GET_LOCAL,
        "OP_SET_LOCAL" => OpCode_OP_SET_LOCAL,
        "OP_GET_GLOBAL" => OpCode_OP_GET_GLOBAL,
        "OP_DEFINE_GLOBAL" => OpCode_OP_DEFINE_GLOBAL,
        "OP_SET_GLOBAL" => OpCode_OP_SET_GLOBAL,
        "OP_GET_UPVALUE" => OpCode_OP_GET_UPVALUE,
        "OP_SET_UPVALUE" => OpCode_OP_SET_UPVALUE,
        "OP_GET_PROPERTY" => OpCode_OP_GET_PROPERTY,
        "OP_SET_PROPERTY" => OpCode_OP_SET_PROPERTY,
        "OP_GET_SUPER" => OpCode_OP_GET_SUPER,
        "OP_EQUAL" => OpCode_OP_EQUAL,
        "OP_GREATER" => OpCode_OP_GREATER,
        "OP_LESS" => OpCode_OP_LESS,
        "OP_ADD" => OpCode_OP_ADD,
        "OP_SUBTRACT" => OpCode_OP_SUBTRACT,
        "OP_MULTIPLY" => OpCode_OP_MULTIPLY,
        "OP_DIVIDE" => OpCode_OP_DIVIDE,
        "OP_NOT" => OpCode_OP_NOT,
        "OP_NEGATE" => OpCode_OP_NEGATE,
        "OP_PRINT" => OpCode_OP_PRINT,
        "OP_JUMP" => OpCode_OP_JUMP,
        "OP_JUMP_IF_FALSE" => OpCode_OP_JUMP_IF_FALSE,
        "OP_LOOP" => OpCode_OP_LOOP,
        "OP_CALL" => OpCode_OP_CALL,
        "OP_INVOKE" => OpCode_OP_INVOKE,
        "OP_SUPER_INVOKE" => OpCode_OP_SUPER_INVOKE,
        "OP_CLOSURE" => OpCode_OP_CLOSURE,
        "OP_CLOSE_UPVALUE" => OpCode_OP_CLOSE_UPVALUE,
        "OP_RETURN" => OpCode_OP_RETURN,
        "OP_CLASS" => OpCode_OP_CLASS,
        "OP_INHERIT" => OpCode_OP_INHERIT,
        "OP_METHOD" => OpCode_OP_METHOD,
        _ => panic!("unknown opcode: {}", opcode),
    }
}

#[allow(non_upper_case_globals)]
fn op_code_to_str(opcode: OpCode) -> &'static str {
    match opcode {
        OpCode_OP_CONSTANT => "OP_CONSTANT",
        OpCode_OP_NIL => "OP_NIL",
        OpCode_OP_TRUE => "OP_TRUE",
        OpCode_OP_FALSE => "OP_FALSE",
        OpCode_OP_POP => "OP_POP",
        OpCode_OP_GET_LOCAL => "OP_GET_LOCAL",
        OpCode_OP_SET_LOCAL => "OP_SET_LOCAL",
        OpCode_OP_GET_GLOBAL => "OP_GET_GLOBAL",
        OpCode_OP_DEFINE_GLOBAL => "OP_DEFINE_GLOBAL",
        OpCode_OP_SET_GLOBAL => "OP_SET_GLOBAL",
        OpCode_OP_GET_UPVALUE => "OP_GET_UPVALUE",
        OpCode_OP_SET_UPVALUE => "OP_SET_UPVALUE",
        OpCode_OP_GET_PROPERTY => "OP_GET_PROPERTY",
        OpCode_OP_SET_PROPERTY => "OP_SET_PROPERTY",
        OpCode_OP_GET_SUPER => "OP_GET_SUPER",
        OpCode_OP_EQUAL => "OP_EQUAL",
        OpCode_OP_GREATER => "OP_GREATER",
        OpCode_OP_LESS => "OP_LESS",
        OpCode_OP_ADD => "OP_ADD",
        OpCode_OP_SUBTRACT => "OP_SUBTRACT",
        OpCode_OP_MULTIPLY => "OP_MULTIPLY",
        OpCode_OP_DIVIDE => "OP_DIVIDE",
        OpCode_OP_NOT => "OP_NOT",
        OpCode_OP_NEGATE => "OP_NEGATE",
        OpCode_OP_PRINT => "OP_PRINT",
        OpCode_OP_JUMP => "OP_JUMP",
        OpCode_OP_JUMP_IF_FALSE => "OP_JUMP_IF_FALSE",
        OpCode_OP_LOOP => "OP_LOOP",
        OpCode_OP_CALL => "OP_CALL",
        OpCode_OP_INVOKE => "OP_INVOKE",
        OpCode_OP_SUPER_INVOKE => "OP_SUPER_INVOKE",
        OpCode_OP_CLOSURE => "OP_CLOSURE",
        OpCode_OP_CLOSE_UPVALUE => "OP_CLOSE_UPVALUE",
        OpCode_OP_RETURN => "OP_RETURN",
        OpCode_OP_CLASS => "OP_CLASS",
        OpCode_OP_INHERIT => "OP_INHERIT",
        OpCode_OP_METHOD => "OP_METHOD",
        _ => panic!("unknown opcode: {}", opcode),
    }
}





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
            let opcode = op_code_from_str(opcode);
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
        panic!("opcode not found: {}", opcode);
    }

    fn decode_opcodes(&mut self, code: &[u8]) -> Vec<(OpCode, Vec<u8>)> {
        self.cache_opcode_and_length();
        let mut result = Vec::new();
        let mut index = 0;
        while index < code.len() {
            let opcode = code[index];
            let opcode = OpCode::from(opcode);
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
        println!("decoded: {:?}", decoded.iter().map(|(op, len)| (op_code_to_str(*op), len)).collect::<Vec<_>>());

        let name = *function.name;
        let s_name = from_raw_parts(name.chars as *mut u8, name.length as usize);
        let s_name = std::str::from_utf8(s_name).unwrap();
        println!("on_closure_call: {}", s_name);
    }
}

