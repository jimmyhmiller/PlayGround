#![allow(dead_code, unused_variables)]

use std::collections::HashMap;
use std::time::{Instant};
use std::thread;
use std::mem;
use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Expr {
    Undefined,
    Num{val: i64, exhausted: bool},
    Symbol{val: String, exhausted: bool},
    Scope{val: String, exhausted: bool},
    LogicVariable{val: String,  exhausted: bool},
    Call{f: Box<Expr>, args: Vec<Expr>, exhausted: bool},
    Map{content: Vec<(Expr, Expr)>, exhausted: bool},
    Array{content: Vec<Expr>, exhausted: bool},
    Do{exprs: Vec<Expr>, exhausted: bool},
}

impl Expr {
    fn call(f: Expr, args: Vec<Expr>) -> Expr {
        Expr::Call{
            f: box f,
            args: args,
            exhausted: false,
        }
    }
    fn symbol(s: &str) -> Expr {
        Expr::Symbol{val: s.to_string(), exhausted: false}
    }

    fn is_exhausted(&self) -> bool {
        match self {
            Expr::Undefined => true,
            Expr::Num { val, exhausted } => *exhausted,
            Expr::Symbol { val, exhausted } => *exhausted,
            Expr::Scope { val, exhausted } => *exhausted,
            Expr::LogicVariable { val, exhausted } => *exhausted,
            Expr::Call { f, args, exhausted } => *exhausted,
            Expr::Map { content, exhausted } => *exhausted,
            Expr::Array { content, exhausted } => *exhausted,
            Expr::Do { exprs, exhausted } => *exhausted,
        }
    }
    fn exhaust(&mut self) {
        match self {
            Expr::Undefined => {}
            Expr::Num { val, ref mut exhausted } => *exhausted = true,
            Expr::Symbol { val, ref mut exhausted } => *exhausted = true,
            Expr::Scope { val, ref mut exhausted } => *exhausted = true,
            Expr::LogicVariable { val, ref mut exhausted } => *exhausted = true,
            Expr::Call { f, args, ref mut exhausted } => *exhausted = true,
            Expr::Map { content, ref mut exhausted } => *exhausted = true,
            Expr::Array { content, ref mut exhausted } => *exhausted = true,
            Expr::Do { exprs, ref mut exhausted } => *exhausted = true,
        }
    }
}

#[derive(Debug)]
struct Clause {
    left: Expr,
    right: Expr,
}

// Should scopes just be on the rules like this?
// Or should scopes be a derived property? 
// What about rules that are on multiple scopes?
#[derive(Debug)]
struct Rule<'a> {
    name: &'a str,
    clauses: Vec<Clause>,
    in_scope: &'a str,
    out_scope: &'a str,
}

// The reference vs value here is probably wrong.
#[derive(Debug)]
struct Meta<'a> {
    current_expr: &'a Expr,
    new_expr: Expr,
    current_sub_expr: &'a Expr,
    new_sub_expr: &'a Expr,
}

struct Interpreter<'a> {
    scopes: HashMap<&'a str, Expr>,
    meta: Option<Meta<'a>>,
    rules: Vec<Rule<'a>>,
    substitutions: Vec<HashMap<&'a str, &'a Expr>>,
}

// This needs to have a reference to the rule as well.
struct MatchResult<'a> {
    env: HashMap<&'a str, &'a Expr>
}

impl<'a> Interpreter<'a> {

    fn new() -> Interpreter<'a> {
        let mut scopes = HashMap::new();
        scopes.insert("main", Expr::Undefined);
        Interpreter {
            scopes,
            meta: None,
            rules: vec![],
            substitutions: vec![HashMap::new()],
        }
    }

    fn matches(&mut self, expr: &mut Expr) -> Option<MatchResult<'a>> {
        // Need to be handle multiple environments and make sure they exist.
        // Also need to ensure we reinsert them so we aren't deallocating.
        let mut env: HashMap<&'a str, &'a Expr> = self.substitutions.remove(0);
        let result = match expr {
            Expr::Undefined => {
                self.substitutions.push(env);
                None
            },
            Expr::Num { val: 3, exhausted } => {
                env.insert("?x", & Expr::Num{val: 4, exhausted: false});
                Some(env)
            },
            Expr::Num { val: 2, exhausted } => {
                env.insert("?x", & Expr::Num{val: 3, exhausted: false});
                Some(env)
            },
            Expr::Num { val: 1, exhausted } => {
                env.insert("?x", & Expr::Num{val: 2, exhausted: false});
                Some(env)
            },
            Expr::Num { val, exhausted } => {
                self.substitutions.push(env);
                None
            },
            Expr::Symbol { val, exhausted } => {
                self.substitutions.push(env);
                None
            },
            Expr::Scope { val, exhausted } => {
                self.substitutions.push(env);
                None
            },
            Expr::LogicVariable { val, exhausted } => {
                self.substitutions.push(env);
                None
            },
            Expr::Call { f, args, exhausted } => {
                self.substitutions.push(env);
                None
            },
            Expr::Map { content, exhausted } => {
                self.substitutions.push(env);
                None
            },
            Expr::Array { content, exhausted } => {
                self.substitutions.push(env);
                None
            },
            Expr::Do { exprs, exhausted } => {
                self.substitutions.push(env);
                None
            },
        };
        if result.is_none() {
            expr.exhaust();

        }
        result.map(|r| MatchResult{ env: r})
    }

    fn submit(&mut self, scope: &'a str, expr: Expr) {
        self.scopes.insert(scope, expr);
    }

    fn print_scope(& self, scope: &'a str) {
        print!("{:#?}", self.scopes.get(scope).unwrap());
    }

    fn substitute(&self, expr: &mut Expr, result: &mut MatchResult) {
        *expr = result.env.remove("?x").unwrap().clone();
    }


    fn interpret(&mut self, scope: &'a str) {
        let (_, mut expr) = self.scopes.remove_entry(scope).unwrap();
        if expr.is_exhausted() {
            self.scopes.insert(scope, expr);
            return;
        }
        let mut queue = VecDeque::new();
        // queue.push_front(&mut expr);
        // let mut e = queue.pop_front().unwrap();
        match expr {
            Expr::Undefined => {}
            Expr::Num { val, exhausted } => {
                if let Some(mut result) = self.matches(&mut expr) {
                    self.substitute(&mut expr, &mut result);
                    self.scopes.insert(scope, expr);
                    result.env.clear();
                    self.substitutions.push(result.env);
                } else {
                    self.scopes.insert(scope, expr);
                }
            }
            Expr::Symbol { val, exhausted } => {}
            Expr::Scope { val, exhausted } => {}
            Expr::LogicVariable { val, exhausted } => {}
            Expr::Call { f, args, exhausted } => {
                // I need to figure out how in the world I deal with recursion here.
                // Maybe I need to bite the bullet and just do a zipper instead?
                // https://sachanganesh.com/programming/graph-tree-traversals-in-rust/
                // Seems like I will have to do something I consider crazy?
                // What are my other options? Are there better languages for this?
                // Maybe I have to reach for something like: https://github.com/oooutlk/trees?


                // So, yeah, I need to do an arena allocated tree. That gives me the zipper
                // like setup where I can go to parent nodes quickly.
                // At first this seemd like I was contorting things too much for rust,
                // and maybe I am. But I am okay with that for now. This is an interesting
                // enough way of doing things that I am willing to try it out and see if it
                //
            }
            Expr::Map { content, exhausted } => {}
            Expr::Array { content, exhausted } => {}
            Expr::Do { exprs, exhausted } => {}
        }
    }

    fn until_exhausted(&mut self, scope: &'a str) {
        while !self.scopes.get(scope).unwrap().is_exhausted() {
            self.interpret(scope);
        }
    }
}







fn build_big_expr(depth: i32) -> Expr {
    let mut i = 0;
    let mut expr = Expr::call(Expr::symbol("bottom"), vec![Expr::Num{val: 1, exhausted: false}, Expr::Num{val: 1, exhausted: false}]);
    while i < depth {
       let x =  std::iter::repeat(expr);
       expr = Expr::call(Expr::symbol("call"), x.take(1).collect());
       i += 1;
    }
    expr
}

fn count_elems(expr: & Expr) -> i64 {
    let mut i = 0;
    match expr {
        Expr::Symbol{..} => {
            i += 1;
        }
        Expr::Num{..} => {
            i += 1;
        }
        Expr::Call { f, args, .. } => {
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
    let now = Instant::now();
    let mut my_expr3 = build_big_expr(1);
    println!("{}", now.elapsed().as_millis());
    println!("Finished creating");
    let now = Instant::now();
    let mut interpreter = Interpreter::new();
    let example = Expr::call(Expr::symbol("bottom"), vec![Expr::Num{val: 1, exhausted: false}, Expr::Num{val: 1, exhausted: false}]);
    interpreter.submit("main", example);
    interpreter.until_exhausted("main");
    // interpreter.print_scope("main");
    println!("{}", now.elapsed().as_millis());
    let now = Instant::now();
    println!("{}", count_elems(interpreter.scopes.get("main").unwrap()));
    println!("{}", now.elapsed().as_millis());


    // thread::spawn(move || drop(interpreter));
    
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