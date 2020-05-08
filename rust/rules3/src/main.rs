#![feature(box_syntax, box_patterns)]
#![allow(dead_code)]

mod parser;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::mem;



#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum Expr {
    Undefined,
    Num(i64),
    Symbol(String),
    LogicVariable(String),
    Exhausted(Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Map(Vec<(Expr, Expr)>)
}

impl Expr {
    fn is_exhausted(& self) -> bool {
        match self {
            Expr::Exhausted(_) => true,
            _ => false
        }
    }

    fn pretty_print(& self) -> String {
        match self {
            Expr::Num(n) => format!("{:?}", n),
            Expr::Symbol(n) => n.to_string(),
            Expr::LogicVariable(n) => n.to_string(),
            Expr::Undefined => "Undefined".to_string(),
            Expr::Exhausted(box e) => e.pretty_print(),
            Expr::Call(box f, args) => {
                let p_f = f.pretty_print();
                let p_args : Vec<String> = args.iter().map(|x| x.pretty_print()).collect();
                format!("({} {})", p_f, p_args.join(" "))
            }
            Expr::Map(entries) => {
                let mut entries = entries.clone();
                let mut result = "{".to_string();
                if entries.is_empty() {
                    return "{}".to_string();
                }
                let (last_key, last_value) = entries.pop().unwrap();
                for (key, value) in entries {
                    result = format!("{}{}: {}, ", result, key.pretty_print(), value.pretty_print());
                }
                result = format!("{}{}: {}}}", result, last_key.pretty_print(), last_value.pretty_print());
                result
            }
        }
    }

    fn get_num(self) -> Option<i64> {
        match self {
            Expr::Exhausted(x) => x.get_num(),
            Expr::Num(i) => Some(i),
            _ => None
        }
    }
}


fn builtins(expr: Expr) -> InterpreterResult {

    let sub_rule : Rule = Rule {
        left: Expr::Undefined,
        right: Expr::Undefined,
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let mult_rule : Rule = Rule {
        left: Expr::Undefined,
        right: Expr::Undefined,
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let expr_clone = expr.clone();
    match expr {
        Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), args) if f == "builtin/-" => {
            let a = args[0].clone().get_num();
            let b = args[1].clone().get_num();
            if let (Some(x), Some(y)) = (a,b) {
                let result = Expr::Num(x - y);
                let result_clone = result.clone();
                InterpreterResult::rewrote(result, expr_clone, result_clone, sub_rule)
            } else {
                InterpreterResult::no_change(expr_clone)
            }
        }
        Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), args) if f == "builtin/+" => {
            let a = args[0].clone().get_num();
            let b = args[1].clone().get_num();
            if let (Some(x), Some(y)) = (a,b) {
                let result = Expr::Num(x + y);
                let result_clone = result.clone();
                InterpreterResult::rewrote(result, expr_clone, result_clone, sub_rule)
            } else {
                InterpreterResult::no_change(expr_clone)
            }
        }
        Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), args) if f == "builtin/*" => {
            let a = args[0].clone().get_num();
            let b = args[1].clone().get_num();
            if let (Some(x), Some(y)) = (a,b) {
                let result = Expr::Num(x * y);
                let result_clone = result.clone();
                InterpreterResult::rewrote(result, expr_clone, result_clone, mult_rule)
            } else {
                InterpreterResult::no_change(expr_clone)
            }
        }
        Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), args) if f == "builtin/println" => {
            let a = &args[0].clone();
            let a_clone = a.clone();
            let a_clone2 = a.clone();
            println!("{}", a.pretty_print());
            // Maybe I need the idea of a void rule?
            // How else am I going to make multiple rules match?
            InterpreterResult::rewrote(a_clone, expr_clone, a_clone2, mult_rule)
        }
        _ => InterpreterResult::no_change(expr_clone)
    }
}



impl Into<Expr> for &i64 {
    fn into(self) -> Expr {
        Expr::Num(*self)
    }
}
impl Into<Expr> for i64 {
    fn into(self) -> Expr {
        Expr::Num(self)
    }
}
impl Into<Expr> for &str {
    fn into(self) -> Expr {
        self.to_string().into()
    }
}
impl Into<Expr> for String {
    fn into(self) -> Expr {
        if self.starts_with('?') {
            Expr::LogicVariable(self)
        } else {
            Expr::Symbol(self)
        }
    }
}

impl Into<Expr> for &String {
    fn into(self) -> Expr {
        Expr::Symbol(self.to_string())
    }
}

impl<T : Into<Expr>, S : Into<Expr>> Into<Expr> for (T, S) {
    fn into(self) -> Expr {
        let f = self.0.into();
        match f {
            Expr::Symbol(s) if s == "quote" => Expr::Exhausted(box self.1.into()),
            _ => Expr::Call(box f, vec![self.1.into()])
        }

    }
}

impl<T : Into<Expr>, S : Into<Expr>, R : Into<Expr>> Into<Expr> for (T, S, R) {
    fn into(self) -> Expr {
        Expr::Call(box self.0.into(), vec![self.1.into(), self.2.into()])
    }
}

impl<T : Into<Expr>, S : Into<Expr>> Into<Expr> for Vec<(T, S)> {
    fn into(self) -> Expr {
        let mut results = Vec::with_capacity(self.len());
        for (key, value) in self {
            results.push((key.into(), value.into()));
        }
        Expr::Map(results)
    }
}


#[derive(Debug, Clone)]
struct Rule {
    left: Expr,
    right: Expr,
    in_scope: String,
    out_scope: String,
}

impl Rule {
    fn build_env(& self, expr : & Expr) -> Option<HashMap<String, Expr>> {
        let mut env : HashMap<String, Expr> = HashMap::new();
        let mut queue = VecDeque::new();
        let expr_clone = expr.clone();
        queue.push_front((self.left.clone(), expr_clone));
        let mut failed = false;
        while !queue.is_empty() && !failed {
            let elem = queue.pop_front().unwrap();
            match elem {
                // Need to handle non-linear

                (x, Expr::Exhausted(box t)) => {
                    queue.push_front((x, t));
                },
                (Expr::LogicVariable(s), t) => {
                    env.insert(s.to_string(), t.clone());
                },
                (Expr::Symbol(ref s1), Expr::Symbol(ref s2)) if s1 == s2 => {},
                (Expr::Num(ref n1), Expr::Num(ref n2)) if n1 == n2 => {},
                (Expr::Call(box f1, args1), Expr::Call(box f2, args2)) => {
                    if args1.len() != args2.len() {
                        failed = true;
                    } else {
                        let mut args1_clone = args1.clone();
                        let mut args2_clone = args2.clone();
                        queue.push_front((f1, f2));
                        for _ in 0..args1.len() {
                            queue.push_front((args1_clone.pop().unwrap(), args2_clone.pop().unwrap()))
                        }
                    }
                }
                // This is depending on key order which is wrong. Need to fix.
                (Expr::Map(args1), Expr::Map(args2)) => {
                    if args1.len() != args2.len() {
                        failed = true;
                    } else {
                        let mut args1_clone = args1.clone();
                        let mut args2_clone = args2.clone();
                        for _ in 0..args1.len() {
                            let (k1, v1) = args1_clone.pop().unwrap();
                            let (k2, v2) = args2_clone.pop().unwrap();
                            queue.push_front((k1, k2));
                            queue.push_front((v1, v2));
                        }
                    }
                }
                _ => {
                    failed = true;
                }
            };
        };
        if failed {
            None
        } else {
            Some(env)
        }
    }
    fn pretty_print(& self) -> String {
        format!("{} => {}", self.left.pretty_print(), self.right.pretty_print())
    }
}


#[derive(Debug, Clone)]
enum InterpreterResult {
    NoChange{
        expr: Expr,
        sub_expr: Expr,
        side_effects: Vec<InterpreterResult>,
    },
    Rewrote{
        new_expr: Expr,
        sub_expr: Expr,
        new_sub_expr: Expr,
        rule: Rule,
        side_effects: Vec<InterpreterResult>,
    },
}

impl InterpreterResult {
    fn expr(self) -> Expr {
        match self {
            InterpreterResult::NoChange{expr, ..} => expr,
            InterpreterResult::Rewrote{new_expr, ..} => new_expr,
        }
    }
    fn wrap(& self, expr: Expr) -> InterpreterResult {
        match self {
            InterpreterResult::Rewrote{sub_expr, new_sub_expr, rule, ..} =>
                InterpreterResult::rewrote(expr, sub_expr.clone(), new_sub_expr.clone(), rule.clone()),
            InterpreterResult::NoChange{sub_expr, side_effects, ..} => InterpreterResult::NoChange{expr, sub_expr: sub_expr.clone(), side_effects: side_effects.clone()}
        }
    }
    fn rewrote(new_expr: Expr, sub_expr: Expr, new_sub_expr: Expr, rule: Rule) -> InterpreterResult {
        InterpreterResult::Rewrote{new_expr, sub_expr, new_sub_expr, rule, side_effects: vec![]}
    }

    fn no_change(expr: Expr) -> InterpreterResult {
        let expr_clone = expr.clone();
        InterpreterResult::NoChange{expr, sub_expr: expr_clone, side_effects: vec![]}
    }

    fn side_effects(self) -> Vec<InterpreterResult> {
        match self {
            InterpreterResult::Rewrote{side_effects, ..} => side_effects,
            InterpreterResult::NoChange{side_effects, ..} => side_effects,
        }
    }
    fn to_meta_expr(&self, scope: String, expr: Expr) -> Expr {
        match self {
            InterpreterResult::Rewrote{new_expr, new_sub_expr, sub_expr, ..} => {
                let meta_expr : Expr = vec![("phase", ("quote", "rewrite".into())),
                                            ("expr", ("quote", expr)),
                                            ("scope", ("quote", scope.into())),
                                            ("new_expr", ("quote", new_expr.clone())),
                                            ("sub_expr", ("quote", sub_expr.clone())),
                                            ("new_sub_expr", ("quote", new_sub_expr.clone()))].into();
                meta_expr
            },
            InterpreterResult::NoChange{..} => {
                // TODO: Need to fix this.
                Expr::Undefined
            }
        }
    }

    fn get_scope(&self) -> String {
        match self {
            InterpreterResult::Rewrote{rule, ..} => {
                rule.out_scope.clone()
            },
            InterpreterResult::NoChange{..} => {
                "main".to_string()
            }
        }
    }
}


#[derive(Debug, Clone)]
struct Interpreter {
    rules: Vec<Rule>,
    scope: String,
}





impl Interpreter {
    fn new(rules: Vec<Rule>, scope: String) -> Interpreter {

        // When I construct the Interpreter (or add new rules)
        // I can compute what things I care about matching on.
        // So if all my rules are on calls, I can only match on those.
        Interpreter {
            rules,
            scope,
        }
    }

    fn match_rule(& self, rhs : & Expr, env : & HashMap<String, Expr>) -> Expr {
        match rhs {
            Expr::Num(_) => rhs.clone(),
            Expr::Symbol(_) => rhs.clone(),
            Expr::LogicVariable(s) => {
                if let Some(x) = env.get(s) {
                    x.clone()
                } else {
                    panic!("Some invalid environment! {} {:?}", s, rhs);
                }
            }
            Expr::Undefined => {
                panic!("Undefined! {:?}", rhs)
            }
            Expr::Exhausted(x) => {
                Expr::Exhausted(box self.match_rule(x, env))
            }
            Expr::Call(box f, args) => {
                let new_args = args.iter().map(|x| self.match_rule(x, env)).collect();
                Expr::Call(box self.match_rule(f, env), new_args)
            }
            Expr::Map(args) => {
                let new_args = args.iter().map(|(x, y)| (self.match_rule(x, env), self.match_rule(y, env))).collect();
                Expr::Map(new_args)
            }
        }
    }

    fn match_all(& self, e : & Expr) -> InterpreterResult {
        let mut main_result = None;
        let mut side_effects = vec![];
        // Need to handle multiple matches in different scopes
        for rule in &self.rules {
            if let Some(env) = rule.build_env(e) {
                if rule.out_scope == self.scope && main_result.is_none() {
                    main_result = Some((rule.clone(), self.match_rule(&rule.right, &env)));
                } else if rule.out_scope != self.scope {
                    let new_expr =self.match_rule(&rule.right, &env);
                    let result = InterpreterResult::Rewrote{
                        sub_expr: e.clone(),
                        new_expr: new_expr.clone(),
                        new_sub_expr: new_expr.clone(),
                        side_effects: vec![],
                        rule: rule.clone()
                    };
                    side_effects.push(result);
                }
            }
        }
        if let Some((rule, expr)) = main_result {
            let expr_clone = expr.clone();
            InterpreterResult::Rewrote{
                sub_expr: e.clone(),
                new_expr: expr,
                new_sub_expr: expr_clone,
                rule,
                side_effects,
            }
        } else {
            InterpreterResult::NoChange{
                expr: Expr::Exhausted(box e.clone()),
                sub_expr: Expr::Exhausted(box e.clone()),
                side_effects,
            }
        }
    }


    // Really we should be able to understand what our rules do to know
    // what we need to match on.
    fn step(&self, e : & Expr) -> InterpreterResult {
        let e = e.clone();
        match e {
            Expr::Undefined => InterpreterResult::no_change(Expr::Exhausted(box e)),
            Expr::Num(_) => self.match_all(&e),
            Expr::Symbol(_) => self.match_all(&e),
            Expr::LogicVariable(_) => self.match_all(&e),
            Expr::Exhausted(_) => InterpreterResult::no_change(e),
            Expr::Call(f @ box Expr::Exhausted(_), mut args) => {
                if let Some(index) = args.iter().position(|e| {
                    match e {
                        Expr::Exhausted(_) => false,
                        _ => true,
                    }
                }) {
                    let expr = mem::replace(&mut args[index], Expr::Undefined);
                    let step = self.step(&expr);
                    args[index] = step.clone().expr();
                    step.wrap(Expr::Call(f, args))
                } else {
                    let f_clone = f.clone();
                    if let box Expr::Exhausted(box Expr::Symbol(fn_name)) = f_clone {
                        if fn_name.starts_with("builtin/") {
                            builtins(Expr::Call(f, args))
                        } else {
                            self.match_all(&Expr::Call(f, args))
                        }
                    }  else {
                        self.match_all(&Expr::Call(f, args))
                    }
                }

            }
            Expr::Call(box f, args) => {
                let step = self.step(&f);
                step.wrap(Expr::Call(box step.clone().expr(), args))
            }
            Expr::Map(mut args) => {
                if let Some(index) = args.iter().position(|(k, v)| {
                    match (k, v) {
                        (&Expr::Exhausted(_), &Expr::Exhausted(_))  => false,
                        _ => true,
                    }
                }) {
                    let (k, v) = mem::replace(&mut args[index], (Expr::Undefined, Expr::Undefined));
                    if let Expr::Exhausted(_) = k {
                        let step = self.step(&v);
                        args[index] = (k, step.clone().expr());
                        step.wrap(Expr::Map(args))
                    } else {
                        let step = self.step(&k);
                        args[index] = (step.clone().expr(), v);
                        step.wrap(Expr::Map(args))
                    }
                } else {
                    self.match_all(&Expr::Map(args))
                }
            }
        }
    }
}

fn print_result(_expr: &Expr, result: &InterpreterResult) {
    match result {
        // Gives the idea of phase, but we need to know what subexpression stepped.
        InterpreterResult::Rewrote{new_expr: _, sub_expr, new_sub_expr, rule: _, side_effects: _} => {
            // println!("{} => {}", sub_expr.pretty_print(), new_sub_expr.pretty_print());
            println!("Rewrote {} => {}", sub_expr.pretty_print(), new_sub_expr.pretty_print());
        },
        InterpreterResult::NoChange{expr: _, sub_expr, side_effects: _} => {
            println!("NoChange {}", sub_expr.pretty_print())
        }
    }
}

fn run_interpreter_loop(expr: Expr, scope: String, interpreters: & HashMap<String, Interpreter>) -> Expr {
    let mut next_results = VecDeque::new();

    let main = interpreters.get(&scope).unwrap();
    let first_step = main.step(&expr);
    let first_step_clone = first_step.clone();
    let meta_expr = first_step.to_meta_expr(scope, expr);
    for effect in first_step.side_effects() {
        next_results.push_back((effect.get_scope(), effect.expr()));
    }
    next_results.push_front(("meta".to_string(), meta_expr));
    
    while !next_results.is_empty() {
        let (scope, expr) = next_results.pop_front().unwrap();
        // Need to deal with scopes that don't exist, or just create them before this loop?
        let main = interpreters.get(&scope).expect("Didn't have an interpreter for the scope");
        let main_result = main.step(&expr);
        let main_expr = main_result.clone().expr();
        let side_effects = main_result.side_effects();
        for effect in side_effects {
            next_results.push_back((effect.get_scope(), effect.expr()));
        }
        if !main_expr.is_exhausted() {
            next_results.push_back((scope, main_expr));
        }
    }
   
    first_step_clone.expr()

}


fn main() {

    println!("{:?}", parser::parse(parser::tokenize("fib(0)")));
    // let main_scope = "main".to_string();
    // let rule_sub = Rule {
    //     left: ("-", "?a", "?b").into(),
    //     right: ("builtin/-", "?a", "?b").into(),
    //     in_scope: "main".to_string(),
    //     out_scope: "main".to_string(),
    // };
    // let rule_plus = Rule {
    //     left: ("+", "?a", "?b").into(),
    //     right: ("builtin/+", "?a", "?b").into(),
    //     in_scope: "main".to_string(),
    //     out_scope: "main".to_string(),
    // };
    // let rule_mult = Rule {
    //     left: ("*", "?a", "?b").into(),
    //     right: ("builtin/*", "?a", "?b").into(),
    //     in_scope: "main".to_string(),
    //     out_scope: "main".to_string(),
    // };
    // let rule1 = Rule {
    //     left: ("fact", 0).into(),
    //     right: 1.into(),
    //     in_scope: "main".to_string(),
    //     out_scope: "main".to_string(),
    // };
    // let rule2 = Rule {
    //     left: ("fact", "?n").into(),
    //     right: ("*", "?n", ("fact", ("-", "?n", 1))).into(),
    //     in_scope: "main".to_string(),
    //     out_scope: "main".to_string(),
    // };
    // let rule3 = Rule {
    //     left: vec![("stuff", "?x")].into(),
    //     right: vec![("thing", "?x")].into(),
    //     in_scope: "main".to_string(),
    //     out_scope: "main".to_string(),
    // };
    // let rule4 = Rule {
    //     left: vec![("phase", "rewrite"), ("expr", "?x"), ("scope", "main"), ("new_expr", "?y"), ("sub_expr", "?sub"), ("new_sub_expr", "?new_sub")].into(),
    //     right: ("builtin/println", ("quote", ("?sub", "=>", "?new_sub"))).into(),
    //     in_scope: "main".to_string(),
    //     out_scope: "io".to_string(),
    // };
    // let rule5 = Rule {
    //     left: vec![("phase", "rewrite"), ("expr", "?expr"), ("scope", "main"), ("new_expr", "?new_expr"), ("sub_expr", "?sub"), ("new_sub_expr", "?new_sub")].into(),
    //     right: ("builtin/println", ("quote", ("?expr", "=>", "?new_expr"))).into(),
    //     in_scope: "main".to_string(),
    //     out_scope: "io".to_string(),
    // };



    
    // let mut expr : Expr = ("fact", 20).into();
    // // let mut expr : Expr = vec![("stuff", "hello")].into();
    // let interpreter = Interpreter::new(vec![rule_sub, rule_mult, rule_plus, rule1, rule2, rule3], main_scope);
    // let meta_interpreter = Interpreter::new(vec![rule4, rule5], "meta".to_string());
    // println!("{}", expr.pretty_print());

    // let mut h = HashMap::new();
    // h.insert("main".to_string(), interpreter);
    // h.insert("meta".to_string(), meta_interpreter);
    // h.insert("io".to_string(), Interpreter::new(vec![], "io".to_string()));

    // while !expr.is_exhausted() {
    //     expr = run_interpreter_loop(expr, "main".to_string(), &h);
    // }
   

    // println!("{}", expr.pretty_print());
}



// need to think about nil
// need to fix map matching
// need to make a parser
// need to make a repl
// need to make rules be matchable
// need to think about repeats
// need to make scopes a real thing
