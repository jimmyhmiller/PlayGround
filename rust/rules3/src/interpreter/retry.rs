#![allow(dead_code)]

use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Expr2 {
    Undefined,
    Num{val: i64, exhausted: bool},
    Symbol{val: String, exhausted: bool},
    Scope{val: String, exhausted: bool},
    LogicVariable{val: String,  exhausted: bool},
    Call{f: Rc<RefCell<Expr2>>, args: RefCell<Vec<Rc<RefCell<Expr2>>>>, exhausted: bool},
    Map{content: Vec<(Rc<RefCell<Expr2>>, Rc<RefCell<Expr2>>)>, exhausted: bool},
    Array{content: Rc<RefCell<Vec<Rc<RefCell<Expr2>>>>>, exhausted: bool},
    Do{exprs: Rc<RefCell<Vec<Rc<RefCell<Expr2>>>>>, exhausted: bool},
}

impl Expr2 {
    fn call(f: Expr2, args: Vec<Expr2>) -> Expr2 {
        Expr2::Call{
            f: Rc::new(RefCell::new(f)), 
            args: RefCell::new(args.into_iter().map(|x| Rc::new(RefCell::new(x))).collect()),
            exhausted: false,
        }
    }
    fn symbol(s: &str) -> Expr2 {
        Expr2::Symbol{val: s.to_string(), exhausted: false}
    }

    fn is_exhausted(&self) -> bool {
        match self {
            Expr2::Undefined => true,
            Expr2::Num { val, exhausted } => *exhausted,
            Expr2::Symbol { val, exhausted } => *exhausted,
            Expr2::Scope { val, exhausted } => *exhausted,
            Expr2::LogicVariable { val, exhausted } => *exhausted,
            Expr2::Call { f, args, exhausted } => *exhausted,
            Expr2::Map { content, exhausted } => *exhausted,
            Expr2::Array { content, exhausted } => *exhausted,
            Expr2::Do { exprs, exhausted } => *exhausted,
        }
    }
    fn exhaust(&mut self) {
        match self {
            Expr2::Undefined => {}
            Expr2::Num { val, ref mut exhausted } => *exhausted = true,
            Expr2::Symbol { val, ref mut exhausted } => *exhausted = true,
            Expr2::Scope { val, ref mut exhausted } => *exhausted = true,
            Expr2::LogicVariable { val, ref mut exhausted } => *exhausted = true,
            Expr2::Call { f, args, ref mut exhausted } => *exhausted = true,
            Expr2::Map { content, ref mut exhausted } => *exhausted = true,
            Expr2::Array { content, ref mut exhausted } => *exhausted = true,
            Expr2::Do { exprs, ref mut exhausted } => *exhausted = true,
        }
    }
}


fn traverse(expr: & Expr2) {
    if expr.is_exhausted() {
        return;
    }
    match expr {
        Expr2::Call { f, args, .. } => {
            let x = args.borrow();
            for i in x.iter() {
                let mut q = i.borrow_mut();
                q.exhaust();
            }
        }
        _ => {}

    }
}

pub fn doit() {
    let s = Rc::new( RefCell::new(Expr2::Symbol{val: "test".to_string(), exhausted: false}));
    let my_expr = Expr2::call(Expr2::symbol("test1"), vec![Expr2::symbol("test2"), Expr2::symbol("test3"), Expr2::symbol("test4")]);
    print!("{:#?}", my_expr);
    traverse(&my_expr);
    print!("{:#?}", my_expr);

}


fn main() {
    doit()
}