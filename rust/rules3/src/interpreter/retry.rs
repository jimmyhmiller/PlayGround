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
    Call{f: Box<Expr2>, args: Vec<Expr2>, exhausted: bool},
    Map{content: Vec<(Expr2, Expr2)>, exhausted: bool},
    Array{content: Vec<Expr2>, exhausted: bool},
    Do{exprs: Vec<Expr2>, exhausted: bool},
}

impl Expr2 {
    fn call(f: Expr2, args: Vec<Expr2>) -> Expr2 {
        Expr2::Call{
            f: box f,
            args: args,
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


fn traverse(expr: &mut Expr2) {
    if expr.is_exhausted() {
        return;
    }
    match expr {
        Expr2::Symbol{..} => {
            expr.exhaust();
        }
        Expr2::Call { f, args, .. } => {
            for i in args {
                traverse(i);
                i.exhaust();
            }
            expr.exhaust();
        }
        _ => {}

    }
}


fn build_big_expr(depth: i32) -> Expr2 {
    let mut i = 0;
    let mut expr = Expr2::call(Expr2::symbol("bottom"), vec![]);
    while i < depth {
       let x =  std::iter::repeat(expr);
       expr = Expr2::call(Expr2::symbol("call"), x.take(10).collect());
       i += 1;
    }
    expr
}

fn count_elems(expr: Expr2) -> i64 {
    let mut i = 0;
    match expr {
        Expr2::Symbol{..} => {
            i += 1;
        }
        Expr2::Call { f, args, .. } => {
            for x in args {
                i += count_elems(x);
            }
            i += 1;
        }
        _ => {}

    }
    i
}


pub fn doit() {
    let s = Rc::new( RefCell::new(Expr2::Symbol{val: "test".to_string(), exhausted: false}));
    let my_expr = Expr2::call(Expr2::symbol("test1"), vec![Expr2::symbol("test2"), Expr2::symbol("test3"), Expr2::symbol("test4")]);
    let mut my_expr2 = Expr2::call(Expr2::symbol("test1"), vec![Expr2::symbol("test2"), Expr2::symbol("test3"), my_expr]);
    let mut my_expr3 = build_big_expr(4);
    println!("Finished creating");
    // print!("{:#?}", my_expr3);
    traverse(&mut my_expr3);
    println!("{}", count_elems(my_expr3));
    // print!("{:#?}", my_expr3);
}


// Before I know if this is really good enough I need to try out doing a bunch 
// rewrites and see if it can handle the performance.

fn main() {
    doit()
}


// This seems to be much much faster for what I need to do.
// I should be able to port things over here and get stuff working.
// I do need to think about when and where I should clone.
// For example, meta should ideally be a reference. 
// That is true in general for the expr vs sub_expr split.
// So I need to keep those things in mind as I build this.


// Think about making exhausting O(1) in many cases.
// For example, what if we built up a struct where there was a bool
// for each type of value in the ast? Then, we could look at the rules
// and see what they match on at the top level. If there is a catch all,
// we know every thing will match. If there isn't and there are for example,
// only rules for Call, we can exhaust everything without doing any lookups
// over the rules.