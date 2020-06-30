
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
    executable_stack: Vec<T>,
    // code_table: HashTable<T, Vec<Op>>
}

impl<T> Machine<T> where T: PartialEq + Eq {
    fn step(&mut self, op: Op<T>) {
        match op {
            Op::Match(g, h) => {
                if let Some(elem) = self.argument_stack.last_mut() {
                    if *elem == g {
                        self.argument_stack.remove(self.argument_stack.len() - 1);
                        // self.executable_stack()
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