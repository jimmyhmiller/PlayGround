#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
// The code just generates things that do that
// kind of annoying
#![allow(deref_nullptr)]

include!("bindings.rs");


impl OpCode {
    pub fn from_u32(code: u32) -> OpCode {
        match code {
            0 => OpCode::OP_CONSTANT,
            1 => OpCode::OP_NIL,
            2 => OpCode::OP_TRUE,
            3 => OpCode::OP_FALSE,
            4 => OpCode::OP_POP,
            5 => OpCode::OP_GET_LOCAL,
            6 => OpCode::OP_SET_LOCAL,
            7 => OpCode::OP_GET_GLOBAL,
            8 => OpCode::OP_DEFINE_GLOBAL,
            9 => OpCode::OP_SET_GLOBAL,
            10 => OpCode::OP_GET_UPVALUE,
            11 => OpCode::OP_SET_UPVALUE,
            12 => OpCode::OP_GET_PROPERTY,
            13 => OpCode::OP_SET_PROPERTY,
            14 => OpCode::OP_GET_SUPER,
            15 => OpCode::OP_EQUAL,
            16 => OpCode::OP_GREATER,
            17 => OpCode::OP_LESS,
            18 => OpCode::OP_ADD,
            19 => OpCode::OP_SUBTRACT,
            20 => OpCode::OP_MULTIPLY,
            21 => OpCode::OP_DIVIDE,
            22 => OpCode::OP_NOT,
            23 => OpCode::OP_NEGATE,
            24 => OpCode::OP_PRINT,
            25 => OpCode::OP_JUMP,
            26 => OpCode::OP_JUMP_IF_FALSE,
            27 => OpCode::OP_LOOP,
            28 => OpCode::OP_CALL,
            29 => OpCode::OP_INVOKE,
            30 => OpCode::OP_SUPER_INVOKE,
            31 => OpCode::OP_CLOSURE,
            32 => OpCode::OP_CLOSE_UPVALUE,
            33 => OpCode::OP_RETURN,
            34 => OpCode::OP_CLASS,
            35 => OpCode::OP_INHERIT,
            36 => OpCode::OP_METHOD,
            _ => panic!("unknown opcode: {}", code),
        }
    }


    pub fn from_str(opcode: &str) -> OpCode {
        match opcode {
            "OP_CONSTANT" => OpCode::OP_CONSTANT,
            "OP_NIL" => OpCode::OP_NIL,
            "OP_TRUE" => OpCode::OP_TRUE,
            "OP_FALSE" => OpCode::OP_FALSE,
            "OP_POP" => OpCode::OP_POP,
            "OP_GET_LOCAL" => OpCode::OP_GET_LOCAL,
            "OP_SET_LOCAL" => OpCode::OP_SET_LOCAL,
            "OP_GET_GLOBAL" => OpCode::OP_GET_GLOBAL,
            "OP_DEFINE_GLOBAL" => OpCode::OP_DEFINE_GLOBAL,
            "OP_SET_GLOBAL" => OpCode::OP_SET_GLOBAL,
            "OP_GET_UPVALUE" => OpCode::OP_GET_UPVALUE,
            "OP_SET_UPVALUE" => OpCode::OP_SET_UPVALUE,
            "OP_GET_PROPERTY" => OpCode::OP_GET_PROPERTY,
            "OP_SET_PROPERTY" => OpCode::OP_SET_PROPERTY,
            "OP_GET_SUPER" => OpCode::OP_GET_SUPER,
            "OP_EQUAL" => OpCode::OP_EQUAL,
            "OP_GREATER" => OpCode::OP_GREATER,
            "OP_LESS" => OpCode::OP_LESS,
            "OP_ADD" => OpCode::OP_ADD,
            "OP_SUBTRACT" => OpCode::OP_SUBTRACT,
            "OP_MULTIPLY" => OpCode::OP_MULTIPLY,
            "OP_DIVIDE" => OpCode::OP_DIVIDE,
            "OP_NOT" => OpCode::OP_NOT,
            "OP_NEGATE" => OpCode::OP_NEGATE,
            "OP_PRINT" => OpCode::OP_PRINT,
            "OP_JUMP" => OpCode::OP_JUMP,
            "OP_JUMP_IF_FALSE" => OpCode::OP_JUMP_IF_FALSE,
            "OP_LOOP" => OpCode::OP_LOOP,
            "OP_CALL" => OpCode::OP_CALL,
            "OP_INVOKE" => OpCode::OP_INVOKE,
            "OP_SUPER_INVOKE" => OpCode::OP_SUPER_INVOKE,
            "OP_CLOSURE" => OpCode::OP_CLOSURE,
            "OP_CLOSE_UPVALUE" => OpCode::OP_CLOSE_UPVALUE,
            "OP_RETURN" => OpCode::OP_RETURN,
            "OP_CLASS" => OpCode::OP_CLASS,
            "OP_INHERIT" => OpCode::OP_INHERIT,
            "OP_METHOD" => OpCode::OP_METHOD,
            _ => panic!("unknown opcode: {}", opcode),
        }
    }
}
