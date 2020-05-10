#![feature(box_syntax, box_patterns)]
#![allow(dead_code)]


use std::collections::HashMap;
use std::io::{self, BufRead};
mod interpreter;


fn main() {
    let main_scope = "main".to_string();
    let rule_sub = interpreter::Rule {
        left: ("-", "?a", "?b").into(),
        right: ("builtin/-", "?a", "?b").into(),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule_plus = interpreter::Rule {
        left: ("+", "?a", "?b").into(),
        right: ("builtin/+", "?a", "?b").into(),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule_mult = interpreter::Rule {
        left: ("*", "?a", "?b").into(),
        right: ("builtin/*", "?a", "?b").into(),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule1 = interpreter::Rule {
        left: ("fact", 0).into(),
        right: 1.into(),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule2 = interpreter::Rule {
        left: ("fact", "?n").into(),
        right: ("*", "?n", ("fact", ("-", "?n", 1))).into(),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule3 = interpreter::Rule {
        left: vec![("stuff", "?x")].into(),
        right: vec![("thing", "?x")].into(),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule4 = interpreter::Rule {
        left: vec![("phase", "rewrite"), ("expr", "?x"), ("scope", "main"), ("new_expr", "?y"), ("sub_expr", "?sub"), ("new_sub_expr", "?new_sub")].into(),
        right: ("builtin/println", ("quote", ("?sub", "=>", "?new_sub"))).into(),
        in_scope: "main".to_string(),
        out_scope: "io".to_string(),
    };
    let rule5 = interpreter::Rule {
        left: vec![("phase", "rewrite"), ("expr", "?expr"), ("scope", "main"), ("new_expr", "?new_expr"), ("sub_expr", "?sub"), ("new_sub_expr", "?new_sub")].into(),
        right: ("builtin/println", ("quote", ("?expr", "=>", "?new_expr"))).into(),
        in_scope: "main".to_string(),
        out_scope: "io".to_string(),
    };


    // println!("{:?}", interpreter::read("fib(0)"));


    
    // let mut expr : Expr = ("fact", 20).into();
    // // let mut expr : Expr = vec![("stuff", "hello")].into();
    let interpreter = interpreter::Interpreter::new(vec![rule_sub, rule_mult, rule_plus, rule1, rule2, rule3], main_scope);
    let meta_interpreter = interpreter::Interpreter::new(vec![rule4, rule5], "meta".to_string());
    // println!("{}", expr.pretty_print());

    let mut h = HashMap::new();


    h.insert("main".to_string(), interpreter);
    h.insert("meta".to_string(), meta_interpreter);
    h.insert("io".to_string(), interpreter::Interpreter::new(vec![], "io".to_string()));

    let stdin = io::stdin();
    let mut iterator = stdin.lock().lines();
    loop {
        let line = iterator.next().unwrap().unwrap();
        if line == "exit" {
            std::process::exit(0);
        }
        let mut expr = interpreter::read(&line);
        expr = interpreter::eval(&h, expr);
        interpreter::print(expr);
    }
   
   

    // println!("{}", expr.pretty_print());
}



// need to think about nil
// need to fix map matching
// need to make a parser
// need to make a repl
// need to make rules be matchable
// need to think about repeats
// need to make scopes a real thing
