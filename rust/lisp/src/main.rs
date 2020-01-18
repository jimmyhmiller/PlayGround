

struct Machine {
    stack: [u8; 4048],
    stack_pointer: usize,
    program: [u16; 4048],
    program_counter: usize,
}

enum Instruction {
    Next,
    Halt,
    Jump(u8),
}

impl Machine {
    fn new() -> Machine {
        Machine{
            stack: [0; 4048],
            stack_pointer: 0,
            program: [0; 4048],
            program_counter: 0,
        }
    }

    fn current(& self) -> u8 {
        self.stack[self.stack_pointer - 1]
    }

    fn previous(& self) -> u8 {
        self.stack[self.stack_pointer - 2]
    }

    fn u8const(& mut self, num : u8) -> Instruction {
        println!("const");
        self.stack[self.stack_pointer] = num;
        self.stack_pointer += 1;
        Instruction::Next
    }

    fn add(& mut self) -> Instruction {
        println!("add");
        self.stack[self.stack_pointer - 2] = self.current().wrapping_add(self.previous());
        self.stack_pointer -= 1;
        Instruction::Next
    }

    fn sub(& mut self) -> Instruction {
        println!("sub");
        self.stack[self.stack_pointer - 2] = self.previous().wrapping_sub(self.current());
        self.stack_pointer -= 1;
        Instruction::Next
    }


    fn jump_if_zero(& mut self, jump_address: u8) -> Instruction {
        println!("jumpif {:?}", self.current());
        if self.current() == 0 {
            Instruction::Jump(jump_address)
        } else {
            Instruction::Next
        }
    }

    fn jump(& mut self, jump_address: u8) -> Instruction {
        println!("jump");
        Instruction::Jump(jump_address)
    }

    fn decode(& mut self, opcode: [u8; 2]) -> Instruction {
        match opcode {
            [0x0, 0x0] => Instruction::Halt,
            [0x1, x] => self.u8const(x),
            [0x2, _] => self.add(),
            [0x3, _] => self.sub(),
            [0x4, x] => self.jump(x),
            [0x5, x] => self.jump_if_zero(x),
            // Need op codes for copy and drop
            // Maybe swap or rotate?
            // Then we might want an assembler
            _ => Instruction::Halt
        }
    }

    fn run_once(& mut self) -> Instruction {
        self.decode(self.program[self.program_counter].to_ne_bytes())
    }

    fn run(& mut self) {
        loop {
            match self.run_once() {
                Instruction::Next => {
                    self.program_counter += 1;
                    continue
                },
                Instruction::Halt => break,
                Instruction::Jump(n) => {
                    self.program_counter = n as usize;
                    continue
                }
            }
        }
    }
}

// 2


fn main() {
    let mut machine = Machine::new();
    machine.program[0] = 0x0A01;
    machine.program[1] = 0x0505;
    machine.program[2] = 0x0101;
    machine.program[3] = 0x0003;
    machine.program[4] = 0x0104;

    machine.run();
    let result = machine.current();

    println!("{:?}", result);
}
