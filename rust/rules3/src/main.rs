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
    Subtract(Box<Expr>, Box<Expr>),
    Multiply(Box<Expr>, Box<Expr>),
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
            e => e
        }
    }
    fn pretty_print(& self) -> String {
        match self {
            Expr::Num(n) => format!("{:?}", n),
            Expr::Symbol(n) => format!("{}", n),
            Expr::Subtract(x, y) => format!("(- {} {})", x.pretty_print(), y.pretty_print()),
            Expr::Multiply(x, y) => format!("(* {} {})", x.pretty_print(), y.pretty_print()),
            Expr::LogicVariable(n) => format!("{}", n),
            Expr::Undefined => "Undefined".to_string(),
            Expr::Exhausted(box e) => e.pretty_print(),
            Expr::Call(box f, args) => {
                let p_f = f.pretty_print();
                let p_args : Vec<String> = args.into_iter().map(|x| x.pretty_print()).collect();
                format!("({} {})", p_f, p_args.join(" "))
            }
        }
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
        let first = self.0.into();
        if first == Expr::Symbol("-".to_string()) {
            Expr::Subtract(box self.1.into(), box self.2.into())
        } else if first == Expr::Symbol("*".to_string()) {
            Expr::Multiply(box self.1.into(), box self.2.into())
        } else {
            Expr::Call(box first, vec![self.1.into(), self.2.into()])
        }
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
                    if args2.len() != args2.len() {
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
}

#[derive(Debug, Clone)]
enum InterpreterResult {
    NoChange(Expr),
    Rewrote(Expr),
}

impl InterpreterResult {
    fn expr(self) -> Expr {
        match self {
            InterpreterResult::NoChange(e) => e,
            InterpreterResult::Rewrote(e) => e,
        }
    }
    fn rewrote(& self) -> bool {
        match self {
            InterpreterResult::Rewrote(_) => true,
            InterpreterResult::NoChange(_) => false,
        }
    }
    fn wrap(& self, expr: Expr) -> InterpreterResult {
        if self.rewrote() {
            InterpreterResult::Rewrote(expr)
        } else {
            InterpreterResult::NoChange(expr)
        }
    }
}


#[derive(Debug, Clone)]
struct Interpreter {
    rules: Vec<Rule>
}



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
            Expr::Subtract(x, y) => Expr::Subtract(box self.match_rule(x, env), box self.match_rule(y, env)),
            Expr::Multiply(x, y) => Expr::Multiply(box self.match_rule(x, env), box self.match_rule(y, env)),
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
        }
    }

    fn match_all(& self, e : & Expr) -> InterpreterResult {
        let mut result = None;
        let mut i = 0;
        // Need to handle multiple matches in different scopes
        while result.is_none() && i < self.rules.len() {
            if let Some(env) = &self.rules[i].build_env(e) {
                result = Some(self.match_rule(&self.rules[i].right, env));
            }
            i += 1;
        }
        if let Some(expr) = result {
            InterpreterResult::Rewrote(expr)
        } else {
            InterpreterResult::NoChange(Expr::Exhausted(box e.clone()))
        }
    }

    // NoChange vs Rewrote is not quite right here.
    // Anytime we do step on the interior or something and
    // don't check the result, we are doing it wrong.
    fn step(&mut self, e : & Expr) -> InterpreterResult {
        let e = e.clone();
        match e {
            Expr::Undefined => InterpreterResult::NoChange(e),
            Expr::Num(_) => self.match_all(&e.de_exhaust()),
            Expr::Subtract(box Expr::Num(x), box Expr::Num(y)) => InterpreterResult::Rewrote(Expr::Num(x - y)),
            Expr::Subtract(box x, y @ box Expr::Num(_)) => {
                let step = self.step(&x);
                let expr = Expr::Subtract(box step.clone().expr(), y);
                step.wrap(expr)
            },
            Expr::Subtract(x @ box Expr::Num(_), box y) => {
                let step = self.step(&y);
                let expr = Expr::Subtract(x, box step.clone().expr());
                step.wrap(expr)
            },
            Expr::Subtract(box x, y) => {
                let step = self.step(&x);
                let expr = Expr::Subtract(box step.clone().expr(), y);
                step.wrap(expr)
            },
            Expr::Multiply(box Expr::Num(x), box Expr::Num(y)) => InterpreterResult::Rewrote(Expr::Num(x * y)),
            Expr::Multiply(box x, y @ box Expr::Num(_)) => {
                let step = self.step(&x);
                let expr = Expr::Multiply(box step.clone().expr(), y);
                step.wrap(expr)
            },
            Expr::Multiply(x @ box Expr::Num(_), box y) => {
                let step = self.step(&y);
                let expr = Expr::Multiply(x, box step.clone().expr());
                step.wrap(expr)
            },
            Expr::Multiply(box x, y) => {
                let step = self.step(&x);
                let expr = Expr::Multiply(box step.clone().expr(), y);
                step.wrap(expr)
            },
            Expr::Symbol(_) => self.match_all(&e.de_exhaust()),
            Expr::LogicVariable(_) => self.match_all(&e.de_exhaust()),
            Expr::Exhausted(_) => InterpreterResult::NoChange(e),
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
                    self.match_all(&Expr::Call(f, args).de_exhaust())
                }
            }
            Expr::Call(box f, args) => {
                let step = self.step(&f);
                step.wrap(Expr::Call(box step.clone().expr(), args))
            }
        }
    }
}

fn main() {
    let mut expr : Expr = ("fact", 20).into();
    let rule1 = Rule {
        left: ("fact", 0).into(),
        right: 1.into()
    };
    let rule2 = Rule {
        left: ("fact", "?n").into(),
        right: ("*", "?n", ("fact", ("-", "?n", 1))).into()
    };


    let mut interpreter = Interpreter::new(vec![rule1, rule2]);
    println!("{:?}", expr);
    // let mut fuel = 0;
    while !expr.is_exhausted() {
        // if fuel > 1000 {
        //     break;
        // }
        // fuel += 1;
        let result = interpreter.step(&expr);
        match result.clone() {
            // Gives the idea of phase, but we need to know what subexpression stepped.
            InterpreterResult::Rewrote(expr2) => {
                println!("Rewrote {} => {}", expr.pretty_print(), expr2.pretty_print())
            }
            InterpreterResult::NoChange(_expr2) => {
                // println!("NoChange {} => {}", expr.pretty_print(), expr2.pretty_print())
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
