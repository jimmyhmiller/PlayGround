
#[derive(Debug)]
enum OpCode {
    Constant = 0,
    Return = 1,
}

type Value = f64;


struct Chunk {
    instructions: Vec<u8>,
    lines: Vec<usize>,
    constants: Vec<Value>,
}

impl Chunk {
    fn add_constant(&mut self, value: Value) -> u8 {
        self.constants.push(value);
        return (self.constants.len() - 1).try_into().unwrap();
    }

    fn write_byte(&mut self, byte: u8, line: usize) {
        self.instructions.push(byte);
        self.lines.push(line);
    }

    fn disassemble(&self, offset: usize) -> usize {
        let mut new_offset = offset;
    
        print!("{:04} ", offset);
        if offset > 0 && self.lines[offset] == self.lines[offset-1] {
            print!("   | ");
        } else {
            print!("{:4} ", self.lines[offset]);
        }
    
        let op_code = self.instructions[offset];
        
        match op_code {
            0 => { 
                print!("Constant {}", self.constants[self.instructions[offset+1] as usize]);
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





fn main() {
    let mut chunk = Chunk {
        instructions: vec![],
        lines: vec![],
        constants: vec![],
    };

    let constant = chunk.add_constant(3.1);
    
    chunk.write_byte(OpCode::Constant as u8, 123);
    chunk.write_byte(constant, 123);
    chunk.write_byte(OpCode::Return as u8, 123);

    let mut offset = 0;
    while offset < chunk.instructions.len() {
        offset = chunk.disassemble(offset);
    }
    

}
