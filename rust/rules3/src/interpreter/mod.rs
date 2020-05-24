pub mod parser;
mod helpers;

pub use self::parser::parse;
pub use self::parser::tokenize;
pub use self::parser::read;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::mem;
use std::io::{self, BufRead};



#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Undefined,
    Num(i64),
    Symbol(String),
    LogicVariable(String),
    Exhausted(Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Map(Vec<(Expr, Expr)>),
    Array(Vec<Expr>),
    Do(Vec<Expr>),
}


impl Expr {
    fn is_exhausted(& self) -> bool {
        match self {
            Expr::Exhausted(_) => true,
            _ => false
        }
    }

    fn get_symbol(self) -> Option<String> {
        match self {
            Expr::Symbol(s) => Some(s),
            _ => None
        }
    }

    // really need to change exhausted
    fn de_exhaust(self) -> Expr {
        match self {
            Expr::Exhausted(box x) => x,
            _ => self
        }
    }

    fn pretty_print(& self) -> String {
        match self {
            Expr::Num(n) => format!("{:?}", n),
            Expr::Symbol(n) => n.to_string(),
            Expr::LogicVariable(n) => n.to_string(),
            Expr::Undefined => "Undefined".to_string(),
            Expr::Exhausted(box e) => {
                match e {
                    Expr::Call(_, _) => format!("'{}", e.pretty_print()),
                    _ => e.pretty_print()
                }
            },
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
            Expr::Array(args) => {
                let p_args : Vec<String> = args.iter().map(|x| x.pretty_print()).collect();
                format!("[{}]", p_args.join(", "))
            }
            Expr::Do(args) => {
                let mut result = "do {\n".to_string();
                for arg in args {
                    result.push_str(format!("{:ident$}{}\n", "", arg.pretty_print(), ident=4).as_ref());
                }
                result.push_str("}");
                // Should really indent here
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
                InterpreterResult::rewrote(result, expr_clone, result_clone, sub_rule, "main".to_string())
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
                InterpreterResult::rewrote(result, expr_clone, result_clone, sub_rule, "main".to_string())
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
                InterpreterResult::rewrote(result, expr_clone, result_clone, mult_rule, "main".to_string())
            } else {
                InterpreterResult::no_change(expr_clone)
            }
        }
        Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), args) if f == "builtin/println" => {
            let print_string = args.iter().map(|x | x.pretty_print()).collect::<Vec<String>>().join(" ");
            println!("{}", print_string);
            // Maybe I need the idea of a void rule?
            // How else am I going to make multiple rules match?
            InterpreterResult::rewrote( 
                Expr::Exhausted(box Expr::Undefined), 
                expr_clone,  
                Expr::Exhausted(box Expr::Undefined), 
                mult_rule, 
                "io".to_string())
        }
        Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), args) if f == "builtin/add-rule" => {
            let rule = args[0].clone().de_exhaust();
            InterpreterResult::NoChange{
                expr: Expr::Exhausted(box expr_clone.clone()),
                sub_expr: Expr::Exhausted(box expr_clone.clone()),
                side_effects: vec![
                    InterpreterResult::Rewrote{
                        new_expr: rule.clone(),
                        new_sub_expr: rule,
                        sub_expr: expr_clone,
                        rule: mult_rule,
                        out_scope: "rules".to_string(),
                        side_effects: vec![],
                    }
                ]
            }
        }
        Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), _args) if f == "builtin/read-line" => {
            let stdin = io::stdin();
            let mut iterator = stdin.lock().lines();
            let line = iterator.next().unwrap().unwrap();
            // Need to handle just whitespace.
            // Might be nice if I could make the real repl thing here.
            if line == "" {
                return InterpreterResult::Rewrote{
                    new_expr: Expr::Exhausted(box Expr::Undefined),
                    new_sub_expr: Expr::Exhausted(box Expr::Undefined),
                    sub_expr: expr_clone,
                    rule: mult_rule,
                    out_scope: "io".to_string(),
                    side_effects: vec![],
                }
            }
            let new_expr = read(line.as_ref());
            InterpreterResult::Rewrote{
                new_expr: new_expr.clone(),
                new_sub_expr: new_expr,
                sub_expr: expr_clone,
                rule: mult_rule,
                out_scope: "rules".to_string(),
                side_effects: vec![],
            }
        }
        
        _ => InterpreterResult::no_change(expr_clone)
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub left: Expr,
    pub right: Expr,
    pub in_scope: String,
    pub out_scope: String,
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
                    // @performance Don't allocate here.
                    // Maybe there is some temp area? Would that be in interpreter then?
                    let rules_map : HashMap<Expr, Expr> = args1.into_iter().collect();
                    let expr_map : HashMap<Expr, Expr> = args2.into_iter().map(|(k, v)| (k.de_exhaust(), v.de_exhaust())).collect();
                    for (key, value) in rules_map {
                        // We don't allow logic variables in keys yet.
                        if let Some(expr_value) = expr_map.get(&key) {
                            queue.push_front((value, expr_value.clone()))
                        } else {
                            failed = true;
                            break;
                        }
                    }
                }
                (Expr::Array(args1), Expr::Array(args2)) => {
                    // Need to get rid of this once we have repeats.
                    if args1.len() != args2.len() {
                        failed = true;
                    } else {
                        let mut args1_clone = args1.clone();
                        let mut args2_clone = args2.clone();
                        for _ in 0..args1.len() {
                            queue.push_front((args1_clone.pop().unwrap(), args2_clone.pop().unwrap()))
                        }
                    }
                }
                (Expr::Do(args1), Expr::Do(args2)) => {
                    // Need to get rid of this once we have repeats.
                    if args1.len() != args2.len() {
                        failed = true;
                    } else {
                        let mut args1_clone = args1.clone();
                        let mut args2_clone = args2.clone();
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
        out_scope: String,
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
            InterpreterResult::Rewrote{sub_expr, new_sub_expr, rule, out_scope,..} =>
                InterpreterResult::rewrote(expr, sub_expr.clone(), new_sub_expr.clone(), rule.clone(), out_scope.clone()),
            InterpreterResult::NoChange{sub_expr, side_effects, ..} => InterpreterResult::NoChange{expr, sub_expr: sub_expr.clone(), side_effects: side_effects.clone()}
        }
    }
    fn rewrote(new_expr: Expr, sub_expr: Expr, new_sub_expr: Expr, rule: Rule, out_scope: String) -> InterpreterResult {
        InterpreterResult::Rewrote{new_expr, sub_expr, new_sub_expr, rule, side_effects: vec![], out_scope}
    }

    fn no_change(expr: Expr) -> InterpreterResult {
       let expr = match expr {
            Expr::Exhausted(_) => expr,
            _ => Expr::Exhausted(box expr)
        };
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

    // Probably need to make this Option
    fn get_scope(&self) -> String {
        match self {
            InterpreterResult::Rewrote{out_scope, ..} => {
                out_scope.clone()
            },
            InterpreterResult::NoChange{..} => {
                "".to_string()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Interpreter {
    rules: Vec<Rule>,
    scope: String,
}


impl Interpreter {
    pub fn new(rules: Vec<Rule>, scope: String) -> Interpreter {

        // When I construct the Interpreter (or add new rules)
        // I can compute what things I care about matching on.
        // So if all my rules are on calls, I can only match on those.
        Interpreter {
            rules,
            scope,
        }
    }

    fn substitute(& self, rhs : & Expr, env : & HashMap<String, Expr>) -> Expr {
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
                Expr::Exhausted(box self.substitute(x, env))
            }
            Expr::Call(box f, args) => {
                let new_args = args.iter().map(|x| self.substitute(x, env)).collect();
                Expr::Call(box self.substitute(f, env), new_args)
            }
            Expr::Map(args) => {
                
                let new_args = args.iter().map(|(x, y)| (self.substitute(x, env), self.substitute(y, env))).collect();
                Expr::Map(new_args)
            }
            Expr::Array(args) => {
                let new_args = args.iter().map(|x| self.substitute(x, env)).collect();
                Expr::Array(new_args)
            }
            Expr::Do(args) => {
                let new_args = args.iter().map(|x| self.substitute(x, env)).collect();
                Expr::Do(new_args)
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
                    main_result = Some((rule.clone(), self.substitute(&rule.right, &env)));
                } else if rule.out_scope != self.scope {
                    let new_expr = self.substitute(&rule.right, &env);
                    let result = InterpreterResult::Rewrote{
                        sub_expr: e.clone(),
                        new_expr: new_expr.clone(),
                        new_sub_expr: new_expr.clone(),
                        side_effects: vec![],
                        rule: rule.clone(),
                        out_scope: rule.out_scope.clone()
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
                out_scope: rule.out_scope.clone(),
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
        let e_clone = e.clone();
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
            Expr::Array(mut args) => {
                if let Some(index) = args.iter().position(|e| {
                    match e {
                        Expr::Exhausted(_) => false,
                        _ => true,
                    }
                }) {
                    let expr = mem::replace(&mut args[index], Expr::Undefined);
                    let step = self.step(&expr);
                    args[index] = step.clone().expr();
                    step.wrap(Expr::Array(args))
                } else {
                    self.match_all(&Expr::Array(args))
                }
            }
            Expr::Do(mut args) => {
                if let Some(index) = args.iter().position(|e| {
                    match e {
                        Expr::Exhausted(_) => false,
                        _ => true,
                    }
                }) {
                    let expr = mem::replace(&mut args[index], Expr::Undefined);
                    let step = self.step(&expr);
                    args[index] = step.clone().expr();
                    step.wrap(Expr::Do(args))
                } else {
                   match self.match_all(&Expr::Do(args.clone())) {
                       InterpreterResult::NoChange{..} => {
                            let dummy_rule = Rule {
                                left: Expr::Undefined,
                                right: Expr::Undefined,
                                in_scope: "main".to_string(),
                                out_scope: "main".to_string(),
                            };
                            // deal with empty do?
                            let new_expr = args.last().unwrap().clone();
                            InterpreterResult::rewrote(new_expr.clone(), e_clone.clone(), new_expr, dummy_rule, self.scope.clone())
                       }
                       x => x
                   }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    scopes: HashMap<String, Expr>,
    interpreters: HashMap<String, Interpreter>,
}

impl Program {
    // Are rules an expr? 
    // Or are they are different datatype that we convert for matching?
    // Or do we provide special rules for matching on rules?
    pub fn new(rules: Vec<Rule>) -> Program {
        let mut scopes = HashMap::new();
        scopes.insert("main".to_string(), Expr::Undefined);
        scopes.insert("meta".to_string(), Expr::Undefined);
        scopes.insert("rules".to_string(), Program::make_rules(rules.clone()));
        scopes.insert("io".to_string(), Expr::Undefined);
        Program {
            scopes: scopes,
            interpreters: Program::make_interpreters(rules),
        }
    }


    fn make_rules(rules: Vec<Rule>) -> Expr {
        Expr::Array(rules.iter().map(|r| {
            let rule_clone = r.clone();
            vec![("left", rule_clone.left), 
                 ("right", rule_clone.right), 
                 ("in_scope", rule_clone.in_scope.into()),
                 ("out_scope", rule_clone.out_scope.into())].into()
        }).collect())
    }


    fn from_rules_expr(expr: Expr) -> Vec<Rule> {
        let expr = expr.de_exhaust();
        if let Expr::Array(rs) = expr {
            let mut rules = Vec::with_capacity(rs.len());
            for rule in rs {
                if let Expr::Map(rule_expr) = rule {
                    // Going to depend on order here
                   rules.push(Rule {
                        left: rule_expr[0].1.clone().de_exhaust(),
                        right: rule_expr[1].1.clone().de_exhaust(),
                        in_scope: rule_expr[2].1.clone().get_symbol().unwrap(),
                        out_scope: rule_expr[3].1.clone().get_symbol().unwrap(),
                    })
                } else {
                    panic!("Rule not a map {:?}", rule);
                }
            }
            rules
        } else {
            panic!("Rules aren't an array! {:?}", expr);
        }
       
    }


    fn make_interpreters(rules: Vec<Rule>) -> HashMap<String, Interpreter> {
        let mut interpreters: HashMap<String, Interpreter> = HashMap::new();
        interpreters.insert("main".to_string(), Interpreter::new(vec![], "main".to_string()));
        interpreters.insert("meta".to_string(), Interpreter::new(vec![], "meta".to_string()));
        interpreters.insert("rules".to_string(), Interpreter::new(vec![], "rules".to_string()));
        interpreters.insert("io".to_string(), Interpreter::new(vec![], "io".to_string()));
        for rule in rules {
            if interpreters.contains_key(&rule.in_scope) {
                let interpreter = interpreters.get_mut(&rule.in_scope).unwrap();
                interpreter.rules.push(rule);
            } else {
                let scope = rule.in_scope.clone();
                interpreters.insert(scope.clone(), Interpreter::new(vec![rule], scope));
            }
        }
        interpreters
    }


    fn run_interpreter_loop(&mut self, scope: String) -> () {
        let expr = self.scopes.get(&scope).unwrap().clone();
        // println!("{}: {}", scope, expr.pretty_print());
        let interpreter = self.interpreters.get(&scope).unwrap();
        let result = interpreter.step(&expr);
        let meta_expr = result.to_meta_expr(scope.clone(), expr.clone());
        // Make sure we don't try to meta eval our meta.
        if scope != "meta" {
            self.scopes.insert("meta".to_string(), meta_expr);
            self.eval_scope("meta".to_string());
        }


        // need to handle out_scope properly?

        // meta scope could have replaced our expression.
        // If that happened, we don't want to override.
        // Need a better way other than checking equality.
        if self.scopes.get(&scope).unwrap().clone() != expr {
            // meta intervened. If that is the case, we don't want to finish
            // evaling this expression and we don't want to do any side effects.
            return;
        }

        if scope == "rules" {
            // Need to be able to do more than append.
            if let Expr::Array(current_rules) = self.scopes.get("rules").clone().unwrap() {
                let mut current_rules = current_rules.clone();
                current_rules.push(result.clone().expr());
                // two places rules live is a bit awkward
                // println!("Updating rules");
                self.interpreters = Program::make_interpreters(Program::from_rules_expr(Expr::Array(current_rules.clone())));
                self.scopes.insert(scope.clone(), Expr::Array(current_rules));
            } else {
                panic!("Updating rules failed");
            }
        } else {
            self.scopes.insert(scope.clone(), result.clone().expr());
        }
        

        for side_effect in result.side_effects() {
            let expr = side_effect.clone().expr();
            let scope = side_effect.get_scope();
            if scope == "rules" {
                // Need to be able to do more than append.
                if let Expr::Array(current_rules) = self.scopes.get("rules").clone().unwrap() {
                    let mut current_rules = current_rules.clone();
                    current_rules.push(expr);
                    // two places rules live is a bit awkward
                    let rules = Program::from_rules_expr(Expr::Array(current_rules.clone()));
                    // println!("Updating rules {:?}", rules);
                    self.interpreters = Program::make_interpreters(rules);
                    self.scopes.insert(scope.clone(), Expr::Array(current_rules));
                } else {
                    panic!("Updating rules failed");
                }

                continue;
            }
            // I don't think we put no_changes as side-effects, so I'm not worried about no out_scope
            self.scopes.insert(scope.clone(), expr);
            self.eval_scope(scope);
        }
       
    }

    pub fn eval_scope(&mut self, scope: String) {
        while !self.scopes.get(&scope).unwrap().is_exhausted() {
            self.run_interpreter_loop(scope.clone());
        }
    }

    // Maybe scope is passed?
    pub fn submit(&mut self, expr: Expr) {
        // Really need to get better at references vs not especially for string.
        self.scopes.insert("main".to_string(), expr);
        self.eval_scope("main".to_string());
    }


    pub fn get_main(&self) -> Expr {
        self.scopes.get("main").unwrap().clone()
    }
}



pub fn print(expr : Expr) {
    println!("{}", expr.pretty_print());
}
