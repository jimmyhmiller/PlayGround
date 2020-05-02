#![feature(box_syntax, box_patterns)]

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

    fn de_exhaust(self) -> Expr {
        // This is pretty terrible.
        // Should probably make exhaust an attribute instead
        // of a node in the tree.
        match self {
            Expr::Exhausted(x) => x.de_exhaust(),
            Expr::Call(box f, args) => Expr::Call(box f.de_exhaust(), args.into_iter().map(|x| x.de_exhaust()).collect()),
            Expr::Map(args) => Expr::Map(args.into_iter().map(|(k, v)| (k.de_exhaust(), v.de_exhaust())).collect()),
            e => e
        }
    }
    fn pretty_print(& self) -> String {
        match self {
            Expr::Num(n) => format!("{:?}", n),
            Expr::Symbol(n) => format!("{}", n),
            Expr::LogicVariable(n) => format!("{}", n),
            Expr::Undefined => "Undefined".to_string(),
            Expr::Exhausted(box e) => e.pretty_print(),
            Expr::Call(box f, args) => {
                let p_f = f.pretty_print();
                let p_args : Vec<String> = args.into_iter().map(|x| x.pretty_print()).collect();
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
        if let Expr::Num(i) = self {
            Some(i)
        } else {
            None
        }
    }
}


fn builtins(expr: Expr) -> InterpreterResult {
    let expr_clone = expr.clone();
    match expr {
        Expr::Call(box Expr::Symbol(f), args) if f == "builtin/-" => {
            let a = args[0].clone().get_num();
            let b = args[1].clone().get_num();
            if let (Some(x), Some(y)) = (a,b) {
                let result = Expr::Num(x - y);
                let result_clone = result.clone();
                InterpreterResult::rewrote(result, expr_clone.clone(), result_clone, SUB_RULE)
            } else {
                InterpreterResult::no_change(expr_clone)
            }
        }
        Expr::Call(box Expr::Symbol(f), args) if f == "builtin/+" => {
            let a = args[0].clone().get_num();
            let b = args[1].clone().get_num();
            if let (Some(x), Some(y)) = (a,b) {
                let result = Expr::Num(x + y);
                let result_clone = result.clone();
                InterpreterResult::rewrote(result, expr_clone.clone(), result_clone, SUB_RULE)
            } else {
                InterpreterResult::no_change(expr_clone)
            }
        }
        Expr::Call(box Expr::Symbol(f), args) if f == "builtin/*" => {
            let a = args[0].clone().get_num();
            let b = args[1].clone().get_num();
            if let (Some(x), Some(y)) = (a,b) {
                let result = Expr::Num(x * y);
                let result_clone = result.clone();
                InterpreterResult::rewrote(result, expr_clone.clone(), result_clone, MULT_RULE)
            } else {
                InterpreterResult::no_change(expr_clone)
            }
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
        if self.starts_with("?") {
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
        Expr::Call(box self.0.into(), vec![self.1.into()])
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
    },
    Rewrote{
        new_expr: Expr,
        sub_expr: Expr,
        new_sub_expr: Expr,
        rule: Rule,
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
            InterpreterResult::NoChange{sub_expr, ..} => InterpreterResult::NoChange{expr, sub_expr: sub_expr.clone()}
        }
    }
    fn rewrote(new_expr: Expr, sub_expr: Expr, new_sub_expr: Expr, rule: Rule) -> InterpreterResult {
        InterpreterResult::Rewrote{new_expr, sub_expr, new_sub_expr, rule}
    }

    fn no_change(expr: Expr) -> InterpreterResult {
        let expr_clone = expr.clone();
        InterpreterResult::NoChange{expr, sub_expr: expr_clone}
    }
}


#[derive(Debug, Clone)]
struct Interpreter {
    rules: Vec<Rule>
}


// Placeholders till I figure out what to do
// with builtins.
// Probably want to do these just as standard calls.
// But they are prefixed named somehow?
const SUB_RULE : Rule = Rule {
    left: Expr::Undefined,
    right: Expr::Undefined,
};
const MULT_RULE : Rule = Rule {
    left: Expr::Undefined,
    right: Expr::Undefined,
};



impl Interpreter {
    fn new(rules: Vec<Rule>) -> Interpreter {
        Interpreter {
            rules: rules
        }
    }

    fn match_rule(& self, rhs : & Expr, env : & HashMap<String, Expr>) -> Expr {
        match rhs {
            Expr::Num(_) => rhs.clone(),
            Expr::Symbol(_) => rhs.clone(),
            Expr::LogicVariable(s) => {
                if let Some(x) = env.get(s) {
                    let y = x.clone();
                    y
                } else {
                    panic!("Some invalid environment! {} {:?}", s, rhs);
                }
            }
            Expr::Undefined => {
                panic!("Undefined! {:?}", rhs)
            }
            Expr::Exhausted(_) => {
                panic!("Some exhausted! {:?}", rhs);
            }
            Expr::Call(box f, args) => {
                let new_args = args.into_iter().map(|x| self.match_rule(x, env)).collect();
                Expr::Call(box self.match_rule(f, env), new_args)
            }
            Expr::Map(args) => {
                let new_args = args.into_iter().map(|(x, y)| (self.match_rule(x, env), self.match_rule(y, env))).collect();
                Expr::Map(new_args)
            }
        }
    }

    fn match_all(& self, e : & Expr) -> InterpreterResult {
        let mut result = None;
        let mut i = 0;
        // Need to handle multiple matches in different scopes
        while result.is_none() && i < self.rules.len() {
            if let Some(env) = &self.rules[i].build_env(e) {
                result = Some((&self.rules[i], self.match_rule(&self.rules[i].right, env)));
            }
            i += 1;
        }
        if let Some((rule, expr)) = result {
            let expr_clone = expr.clone();
            InterpreterResult::Rewrote{
                sub_expr: e.clone(),
                new_expr: expr,
                new_sub_expr: expr_clone,
                rule: rule.clone(),
            }
        } else {
            InterpreterResult::no_change(Expr::Exhausted(box e.clone()))
        }
    }

    // NoChange vs Rewrote is not quite right here.
    // Anytime we do step on the interior or something and
    // don't check the result, we are doing it wrong.
    fn step(&mut self, e : & Expr) -> InterpreterResult {
        let e = e.clone();
        match e {
            Expr::Undefined => InterpreterResult::no_change(e),
            Expr::Num(_) => self.match_all(&e.de_exhaust()),
            Expr::Symbol(_) => self.match_all(&e.de_exhaust()),
            Expr::LogicVariable(_) => self.match_all(&e.de_exhaust()),
            Expr::Exhausted(_) => InterpreterResult::no_change(e),
            Expr::Call(f @ box Expr::Exhausted(_), mut args) => {
                if let Some(index) = args.iter().position(|e| {
                    match e {
                        &Expr::Exhausted(_) => false,
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
                            builtins(Expr::Call(f, args).de_exhaust())
                        } else {
                            self.match_all(&Expr::Call(f, args).de_exhaust())
                        }
                    }  else {
                        self.match_all(&Expr::Call(f, args).de_exhaust())
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
                    self.match_all(&Expr::Map(args).de_exhaust())
                }
            }
        }
    }
}



fn main() {

    let rule_sub = Rule {
        left: ("-", "?a", "?b").into(),
        right: ("builtin/-", "?a", "?b").into()
    };
    let rule_plus = Rule {
        left: ("+", "?a", "?b").into(),
        right: ("builtin/+", "?a", "?b").into()
    };
    let rule_mult = Rule {
        left: ("*", "?a", "?b").into(),
        right: ("builtin/*", "?a", "?b").into()
    };
    let rule1 = Rule {
        left: ("fact", 0).into(),
        right: 1.into()
    };
    let rule2 = Rule {
        left: ("fact", "?n").into(),
        right: ("*", "?n", ("fact", ("-", "?n", 1))).into()
    };
    let rule3 = Rule {
        left: vec![("stuff", "?x")].into(),
        right: vec![("thing", "?x")].into(),
    };


    // let mut expr : Expr = ("fact", 20).into();
    let mut expr : Expr = vec![("stuff", "hello")].into();
    let mut interpreter = Interpreter::new(vec![rule_sub, rule_mult, rule_plus, rule1, rule2, rule3]);
    println!("{:?}", expr);
    let mut fuel = 0;
    while !expr.is_exhausted() {
        if fuel > 100 {
            println!("Ran out of fuel");
            break;
        }
        fuel += 1;
        let result = interpreter.step(&expr);
        match result.clone() {
            // Gives the idea of phase, but we need to know what subexpression stepped.
            InterpreterResult::Rewrote{new_expr, sub_expr, new_sub_expr, rule} => {
                println!("{} => {}", sub_expr.pretty_print(), new_sub_expr.pretty_print());
                println!("Rewrote {} => {} via {}", expr.pretty_print(), new_expr.pretty_print(), rule.pretty_print())
            },
            InterpreterResult::NoChange{expr: _, sub_expr} => {
                println!("NoChange {}", sub_expr.pretty_print())
            }
        }
        expr = result.expr();
        // println!("{}", expr.pretty_print());
    }
    println!("{}", expr.pretty_print());
}



// Need to make exhausted be a property
// Need to make return from step be an enum
// Need to handle multiple rules matching in different contexts
// Need to add matching on phases
// Need to think about nil/unit
