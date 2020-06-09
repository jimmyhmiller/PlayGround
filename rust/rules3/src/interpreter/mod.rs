

pub mod parser;
mod helpers;

pub use self::parser::parse;
pub use self::parser::tokenize;
pub use self::parser::read;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::mem;
use std::io::{self, BufRead};



// #[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Undefined,
    Num(i64),
    Symbol(String),
    Scope(String),
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

    fn exhaust(self) -> Expr {
        match self {
            Expr::Exhausted(_) => self,
            _ => Expr::Exhausted(box self)
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
            Expr::Exhausted(box x) => x.de_exhaust(),
            _ => self
        }
    }

    fn de_exhaust_ref(&self) -> &Expr {
        match self {
            Expr::Exhausted(box x) => x,
            _ => self
        }
    }

    fn pretty_print(& self) -> String {
        match self {
            Expr::Num(n) => format!("{:?}", n),
            Expr::Symbol(n) => n.to_string(),
            Expr::Scope(n) => format!("@{:}", n.to_string()),
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
                format!("{}({})", p_f, p_args.join(" "))
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




#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub left: Expr,
    pub right: Expr,
    pub in_scope: String,
    pub out_scope: String,
}

impl Rule {

    fn noop_rule() -> Rule {
        Rule {
            left: Expr::Undefined,
            right: Expr::Undefined,
            in_scope: "main".to_string(),
            out_scope: "main".to_string(),
        }
    }

    fn build_env<'a>(& self, expr : &'a Expr, mut env : HashMap<String, &'a Expr>) -> Option<HashMap<String, &'a Expr>> {
        let mut queue: VecDeque<(&Expr, &Expr)> = VecDeque::new();
        queue.push_front((&self.left, &expr));
        let mut failed = false;
        while !queue.is_empty() && !failed {
            let elem = queue.pop_front().unwrap();
            match elem {
                // Need to handle non-linear

                (x, Expr::Exhausted(box t)) => {
                    queue.push_front((x, t));
                },
                (Expr::LogicVariable(s), t) => {
                    env.insert(s.to_string(), t);
                },
                (Expr::Symbol(ref s1), Expr::Symbol(ref s2)) if s1 == s2 => {},
                // What does it mean to match on a scope? Probably should think more about this.
                (Expr::Scope(ref s1), Expr::Scope(ref s2)) if s1 == s2 => {},
                (Expr::Num(ref n1), Expr::Num(ref n2)) if n1 == n2 => {},
                (Expr::Call(box f1, args1), Expr::Call(box f2, args2)) => {
                    if args1.len() != args2.len() {
                        failed = true;
                    } else {
                        queue.push_front((f1, f2));
                        for i in 0..args1.len() {
                            queue.push_front((&args1[i], &args2[i]))
                        }
                    }
                }
                (Expr::Map(args1), Expr::Map(args2)) => {
                    // @performance Don't allocate here.
                    let expr_map : HashMap<&Expr, &Expr> = args2.into_iter().map(|(k, v)| (k.de_exhaust_ref(), v.de_exhaust_ref())).collect();
                    for (key, value) in args1 {
                        // We don't allow logic variables in keys yet.
                        if let Some(expr_value) = expr_map.get(&key) {
                            queue.push_front((value, expr_value))
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
                        for i in 0..args1.len() {
                            queue.push_front((&args1[i], &args2[i]))
                        }
                    }
                }
                (Expr::Do(args1), Expr::Do(args2)) => {
                    // Need to get rid of this once we have repeats.
                    if args1.len() != args2.len() {
                        failed = true;
                    } else {
                        for i in 0..args1.len() {
                            queue.push_front((&args1[i], &args2[i]))
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
pub struct Program {
    current_rules: Vec<Rule>,
    current_scope: String,
    rules_by_scope: HashMap<String, Vec<Rule>>,
    scopes: HashMap<String, Expr>,
    previous_scope_value: Expr,
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
        let initial_rules = Program::make_rules_by_scope(rules);
        Program {
            current_rules: initial_rules.get("main").unwrap().to_vec(),
            current_scope: "main".to_string(),
            scopes: scopes.clone(),
            rules_by_scope: initial_rules,
            previous_scope_value: Expr::Undefined,
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

    fn make_rules_by_scope(rules: Vec<Rule>) -> HashMap<String, Vec<Rule>> {
        let mut rules_by_scope: HashMap<String, Vec<Rule>> = HashMap::new();
        // Need to not be cloning scopes here. That's a really bad idea.
        rules_by_scope.insert("main".to_string(), vec![]);
        rules_by_scope.insert("meta".to_string(), vec![]);
        rules_by_scope.insert("rules".to_string(), vec![]);
        rules_by_scope.insert("io".to_string(), vec![]);
        for rule in rules {
            if rules_by_scope.contains_key(&rule.in_scope) {
                let scope = rules_by_scope.get_mut(&rule.in_scope).unwrap();
                scope.push(rule);
            } else {
                let scope = rule.in_scope.clone();
                rules_by_scope.insert(scope.clone(), vec![rule]);
            }
        }
        rules_by_scope
    }

    fn set_scope_and_rules(&mut self, scope_entry: (String, Expr)) {
        // current scope should probably be a ref?
        self.current_scope = scope_entry.0.clone();
        self.current_rules = self.rules_by_scope.get(&scope_entry.0).unwrap().clone();
        self.previous_scope_value = scope_entry.1;
    } 



    fn run_interpreter_loop(&mut self, scope_entry: & (String, Expr)) -> () {
        let expr = self.scopes.get(&scope_entry.0).unwrap().clone();
        // println!("{}:  {:?}\n{:?}\n\n", self.current_scope, expr, self.current_rules);
        // if scope_entry.0 == "io" {
        //     println!("{}", expr.pretty_print());
        // }

        // let interpreter = self.interpreters.get(&scope_entry.0).unwrap();
        // Need to handle scope not existing
        // self.set_scope_and_rules(&scope_entry.0);
        let result = self.step(&expr);

        // Not a huge fan of this current scope stuff. I also don't

        let meta_expr = result.to_meta_expr(scope_entry.0.clone(), expr.clone());
        // Make sure we don't try to meta eval our meta.
        if scope_entry.0 != "meta" {
            let scope_entry = self.scopes.remove_entry("meta").unwrap();
            self.scopes.insert("meta".to_string(), meta_expr);
            self.eval_scope(scope_entry);
        }

        // This is an important setting of scope entry, because
        // otherwise we would be stuck in meta. I ultimately want to
        // get rid of current scope and rules.
        self.set_scope_and_rules(scope_entry.clone());


        // need to handle out_scope properly?

        // meta scope could have replaced our expression.
        // If that happened, we don't want to override.
        // Need a better way other than checking equality.
        if self.scopes.get(&scope_entry.0).unwrap().clone() != expr {
            // meta intervened. If that is the case, we don't want to finish
            // evaling this expression and we don't want to do any side effects.
            return;
        }

        if scope_entry.0 == "rules" {
            // Need to be able to do more than append.
            if let Expr::Array(current_rules) = self.scopes.get("rules").clone().unwrap() {
                let mut current_rules = current_rules.clone();
                current_rules.push(result.clone().expr());
                // two places rules live is a bit awkward
                // println!("Updating rules");
                let new_rules = Program::from_rules_expr(Expr::Array(current_rules.clone()));
                // self.interpreters = Program::make_interpreters(new_rules, self.scopes.clone());
                self.rules_by_scope = Program::make_rules_by_scope(new_rules.clone());
                self.scopes.insert("rules".to_string(), Expr::Array(current_rules.clone()));
            } else {
                panic!("Updating rules failed");
            }
        } else {
            self.scopes.insert(scope_entry.0.clone(), result.clone().expr());
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
                    let new_rules = Program::from_rules_expr(Expr::Array(current_rules.clone()));
                    // This is ugly. Figure out a better way to do all of this rule handling.
                    self.rules_by_scope = Program::make_rules_by_scope(new_rules);
                    self.scopes.insert("rules".to_string(), Expr::Array(current_rules.clone()));
                } else {
                    panic!("Updating rules failed");
                }

                continue;
            }
            // I don't think we put no_changes as side-effects, so I'm not worried about no out_scope

            // What do I do about scopes that don't exist?
            // Also how do I initialize scopes?
            let scope_entry = self.scopes.remove_entry(&scope).unwrap_or((scope.clone(), Expr::Undefined));
            self.scopes.insert(scope.clone(), expr);
            self.eval_scope(scope_entry);
        }
       
    }



    pub fn eval_scope(&mut self, scope_entry: (String, Expr)) {
        let scope = &scope_entry.clone().0;
        self.set_scope_and_rules(scope_entry.clone());
        while !self.scopes.get(scope).unwrap().is_exhausted() {
            self.run_interpreter_loop(&scope_entry);
        }
    }

    // Maybe scope is passed?
    pub fn submit(&mut self, expr: Expr) {
        // Really need to get better at references vs not especially for string.
        let scope_entry = self.scopes.remove_entry("main").unwrap();
        self.scopes.insert("main".to_string(), expr);
        self.eval_scope(scope_entry);
    }


    pub fn get_main(&self) -> Expr {
        self.scopes.get("main").unwrap().clone()
    }



    fn substitute(& self, rhs : & Expr, env : & HashMap<String, &Expr>) -> Expr {
        match rhs {
            Expr::Num(_) => rhs.clone(),
            Expr::Symbol(_) => rhs.clone(),
            Expr::Scope(n) if n == &self.current_scope => self.previous_scope_value.clone(),
            Expr::Scope(n) => {
                if let Some(new_expr) = self.scopes.get(n) {
                    new_expr.clone()
                } else {
                    Expr::Undefined
                }
            }
            Expr::LogicVariable(s) => {
                if let Some(x) = env.get(s) {
                    x.clone().clone()
                } else {
                    panic!("Some invalid environment! {} {:?}", s, rhs);
                }
            }
            Expr::Undefined => {
                panic!("Undefined! {:?}", rhs)
            }
            Expr::Exhausted(box Expr::Scope(n)) => {
                Expr::Exhausted(box Expr::Scope(n.clone()))
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
        for rule in &self.current_rules {
            if self.current_scope == "meta" {
                // println!("{:?}, {:?}", rule, e);
            }
            let env : HashMap<String, &Expr> = HashMap::new();
            if let Some(env) = rule.build_env(e, env) {
                if rule.out_scope == self.current_scope && main_result.is_none() {
                    main_result = Some((rule.clone(), self.substitute(&rule.right, &env)));
                } else if rule.out_scope != self.current_scope {
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
                expr: e.clone().exhaust(),
                sub_expr: e.clone().exhaust(),
                side_effects,
            }
        }
    }


    // Really we should be able to understand what our rules do to know
    // what we need to match on.
    fn step(&mut self, e : & Expr) -> InterpreterResult {
        let e = e.clone();
        // println!("{:?}", e);
        let e_clone = e.clone();
        match e {
            Expr::Undefined => InterpreterResult::no_change(e.exhaust()),
            Expr::Num(_) => self.match_all(&e),
            Expr::Symbol(_) => self.match_all(&e),
            // Need to think about what to do when stepping on scope.
            // This might be the right place to resolve it? 
            // Is it always exhausted?
            Expr::Scope(n) if n == self.current_scope => InterpreterResult::rewrote(
                self.previous_scope_value.clone().exhaust(), 
                e_clone,
                self.previous_scope_value.clone().exhaust(),
                Rule::noop_rule(),
                self.current_scope.clone()),
            Expr::Scope(n) => {
                // println!("{}", n);
                if let Some(new_expr) = self.scopes.get(&n) {
                    // println!("{:?}", new_expr);
                    InterpreterResult::rewrote(
                        new_expr.clone().exhaust(),
                        e_clone,
                        new_expr.clone().exhaust(),
                        Rule::noop_rule(),
                        self.current_scope.clone())
                } else {
                    InterpreterResult::no_change(e_clone)
                }
            }
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
                            self.builtins(Expr::Call(f, args))
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
                            // deal with empty do?
                            let new_expr = args.last().unwrap().clone();
                            InterpreterResult::rewrote(new_expr.clone(), e_clone.clone(), new_expr, Rule::noop_rule(), self.current_scope.clone())
                       }
                       x => x
                   }
                }
            }
        }
    }

    fn builtins(&mut self, expr: Expr) -> InterpreterResult {

        let expr_clone = expr.clone();
        match expr {
            Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), args) if f == "builtin/-" => {
                let a = args[0].clone().get_num();
                let b = args[1].clone().get_num();
                if let (Some(x), Some(y)) = (a,b) {
                    let result = Expr::Num(x - y);
                    let result_clone = result.clone();
                    InterpreterResult::rewrote(result, expr_clone, result_clone, Rule::noop_rule(), self.current_scope.clone())
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
                    InterpreterResult::rewrote(result, expr_clone, result_clone, Rule::noop_rule(), self.current_scope.clone())
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
                    InterpreterResult::rewrote(result, expr_clone, result_clone, Rule::noop_rule(), self.current_scope.clone())
                } else {
                    InterpreterResult::no_change(expr_clone)
                }
            }
            Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), args) if f == "builtin/push-back" => {
                // Probably don't actually want to de_exhaust here?
                // Maybe just take one layer off?
                let expr = args[0].clone().de_exhaust();
                let array = args[1].clone().de_exhaust();
                if let Expr::Array(mut array) = array {
                    array.push(expr);
                    let result = Expr::Array(array);
                    let result_clone = result.clone();
                    InterpreterResult::rewrote(result, expr_clone, result_clone, Rule::noop_rule(), self.current_scope.clone())
                } else {
                    InterpreterResult::no_change(expr_clone)
                }
            }
            Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), args) if f == "builtin/println" => {
                // println!("{:?}", args);
                let print_string = args.iter().map(|x | x.pretty_print()).collect::<Vec<String>>().join(" ");
                println!("{}", print_string);
                // Maybe I need the idea of a void rule?
                // How else am I going to make multiple rules match?
                InterpreterResult::rewrote( 
                    Expr::Exhausted(box Expr::Undefined), 
                    expr_clone,  
                    Expr::Exhausted(box Expr::Undefined), 
                    Rule::noop_rule(), 
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
                            rule: Rule::noop_rule(),
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
                        rule: Rule::noop_rule(),
                        out_scope: "io".to_string(),
                        side_effects: vec![],
                    }
                }
                let new_expr = read(line.as_ref());
                InterpreterResult::Rewrote{
                    new_expr: new_expr.clone(),
                    new_sub_expr: new_expr,
                    sub_expr: expr_clone,
                    rule: Rule::noop_rule(),
                    out_scope: "rules".to_string(),
                    side_effects: vec![],
                }
            }
            
            Expr::Call(box Expr::Exhausted(box Expr::Symbol(f)), args) if f == "builtin/set-scope" => {
                let scope = args[0].clone().de_exhaust();
                let expr = args[1].clone().de_exhaust();
                
                if let Expr::Scope(name) = scope {

                    InterpreterResult::Rewrote{
                        new_expr: Expr::Exhausted(box Expr::Undefined),
                        new_sub_expr: Expr::Exhausted(box Expr::Undefined),
                        sub_expr: expr_clone,
                        rule: Rule::noop_rule(),
                        out_scope: self.current_scope.clone(),
                        side_effects: vec![InterpreterResult::Rewrote{
                            new_expr: expr.clone(),
                            new_sub_expr: expr,
                            sub_expr: self.scopes.get(&name).unwrap_or(&Expr::Undefined).clone(),
                            rule: Rule::noop_rule(),
                            out_scope: name.clone(),
                            side_effects: vec![],
                        }],
                    }
                } else {
                    InterpreterResult::no_change(expr_clone)
                }
            }
            
            _ => InterpreterResult::no_change(expr_clone)
        }
    }
}



pub fn print(expr : Expr) {
    println!("{}", expr.pretty_print());
}




// So it seems in order to do the things I need to do, 
// I have to get rid of interpreter and just use program.
// I should have realized this. I mean technically I thought about
// it but I thought that returning results was all I needed, but that
// definitely isn't the case. Interpreters need access to all the scopes
// to be able to set values and things.
// So instead of just a static scope, I will need a current scope.
// And instead of each interpreter having rules I will need rules by scope.
