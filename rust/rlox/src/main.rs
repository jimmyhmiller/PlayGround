
#[derive(Debug)]
enum OpCode {
    Constant = 0,
    Return = 1,
}

type Value = f64;

fn store_constant(constant_pool: &mut Vec<Value>, value: Value) {
    constant_pool.push(value);
}



fn assemble(instructions: &mut Vec<u8>, op_code: OpCode) {
    match op_code {
        OpCode::Constant => {
            instructions.push(OpCode::Constant as u8);
        },
        OpCode::Return => {
            instructions.push(OpCode::Return as u8);
        }
    }
}


fn disassemble(instructions: &mut Vec<u8>, constant_pool: &Vec<Value>) {
    let op_code = instructions.pop().unwrap();
    match op_code {
        0 => println!("Constant {}", constant_pool[instructions.pop().unwrap() as usize]),
        1 => println!("Return"),
        _ => println!("Unknown opcode: {}", op_code),
    }
}



fn main() {
    
}
