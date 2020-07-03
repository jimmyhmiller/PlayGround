use std::collections::HashMap;
use std::hash::{Hash};
use std::io::{self, BufRead};


#[derive(Debug, Clone)]
enum Op<T> {
    Match(T, T),
    CopyArgument(usize),
    CopyTraversal(usize),
    Push(T),
    DropArgument(usize),
    DropTraversal(usize),
    Skip(usize),
    Retract(usize),
    Build(T, usize),
    Goto(T),
    Recycle
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum Program {
    Zero,
    Succ,
    Plus,
    PlusZero,
    PlusSucc,
    PlusC,
}



#[derive(Debug, Clone)]
struct Machine<T> {
    argument_stack: Vec<T>,
    control_stack: Vec<T>,
    traversal_stack: Vec<T>,
    executable_stack: Vec<Op<T>>,
    code_table: HashMap<T, Vec<Op<T>>>
}

impl<T> Machine<T> where T: PartialEq + Eq + Hash + Clone {
    fn step(&mut self) -> bool {
        // Maybe I should have a bottom value? Not sure?
        if self.executable_stack.is_empty() && self.control_stack.is_empty() {
            return false;
        }
        match self.executable_stack.pop().unwrap() {
            Op::Match(g, h) => {
                if let Some(elem) = self.argument_stack.last_mut() {
                    if *elem == g {
                        self.argument_stack.remove(self.argument_stack.len() - 1);
                        // Figure out better way to do this.
                        self.executable_stack = self.code_table.get(&h).unwrap().clone();
                    }
                }
            }
            Op::CopyArgument(k) => {
                self.argument_stack.push(self.argument_stack.get(k).unwrap().clone());
            }
            Op::CopyTraversal(k) => {
                self.argument_stack.push(self.traversal_stack.get(k).unwrap().clone());
            }
            Op::Push(h) => {
                self.control_stack.push(h);
            }
            Op::DropArgument(k) => {
                for _ in 0..k {
                    self.argument_stack.pop();
                }
            }
            Op::DropTraversal(k) => {
                for _ in 0..k {
                    self.traversal_stack.pop();
                }
            }

            Op::Skip(k) => {
                for i in (self.argument_stack.len()-k..self.argument_stack.len()).rev() {
                    let elem = self.argument_stack.remove(i);
                    self.traversal_stack.push(elem);
                }
            }
            Op::Retract(k) => {
                for i in (self.traversal_stack.len()-k..self.traversal_stack.len()).rev() {
                    let elem = self.traversal_stack.remove(i);
                    self.argument_stack.push(elem);
                }
            }
            Op::Build(f,_k) => {
                // This says to replace t1...tk with f(t1...tk)
                // I have been having call implicit is that oka/
                self.argument_stack.push(f);
            }
            Op::Goto(h) => {
                self.executable_stack = self.code_table.get(&h).unwrap().clone();
            }
            Op::Recycle => {
                // Instead of clone can I change a pointer?
                if let Some(elem) = self.control_stack.pop() {
                    self.executable_stack = self.code_table.get(&elem).unwrap().clone();
                }
            }
        };
        true
    }
}



pub fn main_arm() -> io::Result<()> {

    let mut code_table = HashMap::new();
    code_table.insert(Program::Zero, vec![Op::Recycle, Op::Build(Program::Zero, 0)]);
    code_table.insert(Program::Succ, vec![Op::Recycle, Op::Build(Program::Succ, 1)]);
    code_table.insert(Program::Plus, vec![Op::Goto(Program::PlusC), Op::Match(Program::Succ, Program::PlusSucc), Op::Match(Program::Zero, Program::PlusZero) ]);
    code_table.insert(Program::PlusZero, vec![Op::Recycle]);
    code_table.insert(Program::PlusSucc, vec![Op::Goto(Program::Plus), Op::Push(Program::Succ)]);
    code_table.insert(Program::PlusC, vec![Op::Recycle, Op::Build(Program::PlusC, 2), ]);

    let mut machine = Machine {
        control_stack: vec![ Program::Plus, Program::Succ, Program::Zero, Program::Succ, Program::Zero],
        executable_stack: vec![Op::Recycle],
        argument_stack: vec![],
        traversal_stack: vec![],
        code_table: code_table
    };

    loop {

        println!("{:?}   |   {:?}   |   {:?}   |   {:?}", machine.control_stack, machine.executable_stack, machine.argument_stack, machine.traversal_stack);
        let result = machine.step();
        if !result {
            break;
        }
    }
    Ok(())

}

// P, C, E, A T