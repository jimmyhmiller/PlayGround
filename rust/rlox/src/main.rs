
#[derive(Debug)]
enum OpCode {
    Constant = 0,
    Return = 1,
}

type Value = f64;

fn store_constant(constant_pool: &mut Vec<Value>, value: Value) -> u8 {
    constant_pool.push(value);
    return (constant_pool.len() - 1).try_into().unwrap();
}


struct Code {
    instructions: Vec<u8>,
    lines: Vec<usize>,
}


fn write_byte(code: &mut Code, byte: u8, line: usize) {
    code.instructions.push(byte);
    code.lines.push(line);
}

// fn assemble(code: &mut Code, op_code: OpCode) {
//     match op_code {
//         OpCode::Constant => {
//             code.instructions.push(OpCode::Constant as u8);
//             // TODO: Should I take advantage of rust enums?
//         },
//         OpCode::Return => {
//             code.instructions.push(OpCode::Return as u8);
//         }
//     }
// }


fn disassemble(code: &Code, offset: usize, constant_pool: &Vec<Value>) -> usize {
    let mut new_offset = offset;

    print!("{:04} ", offset);
    if offset > 0 && code.lines[offset] == code.lines[offset-1] {
        print!("   | ");
    } else {
        print!("{:4} ", code.lines[offset]);
    }

    let op_code = code.instructions[offset];
    
    match op_code {
        0 => { 
            print!("Constant {}", constant_pool[code.instructions[offset+1] as usize]);
            new_offset += 2;
        },
        1 => {
            print!("Return");
            new_offset += 1;
        },
        _ => {
            print!("Unknown opcode: {}", op_code);
            new_offset += 1;
        },
    }

    println!("");

    new_offset
}



fn main() {
    let mut code = Code {
        instructions: vec![],
        lines: vec![],
    };

    let mut constant_pool = vec![];

    let constant = store_constant(&mut constant_pool, 3.1);
    
    write_byte(&mut code, OpCode::Constant as u8, 123);
    write_byte(&mut code, constant, 123);
    write_byte(&mut code, OpCode::Return as u8, 123);

    let mut offset = 0;
    while offset < code.instructions.len() {
        offset = disassemble(&code, offset, &constant_pool);
    }
    

}
