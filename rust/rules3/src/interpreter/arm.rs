use std::collections::HashMap;
use std::hash::{Hash};


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

#[derive(Debug, Clone)]
enum Nat {
    Z,
    S,
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
    fn step(&mut self, op: Op<T>) {
        match op {
            Op::Match(g, h) => {
                if let Some(elem) = self.argument_stack.last_mut() {
                    if *elem == g {
                        self.argument_stack.remove(self.argument_stack.len() - 1);
                        for op in self.code_table.get(&h).unwrap() {
                            // This might be backwards
                            self.executable_stack.push(op.clone());
                        }
                    }
                }
            }
            Op::CopyArgument(_) => {}
            Op::CopyTraversal(_) => {}
            Op::Push(_) => {}
            Op::DropArgument(_) => {}
            Op::DropTraversal(_) => {}
            Op::Skip(_) => {}
            Op::Retract(_) => {}
            Op::Build(_, _) => {}
            Op::Goto(_) => {}
            Op::Recycle => {}
        }
    }
}



pub fn main_arm() {
    println!("Arm");
}