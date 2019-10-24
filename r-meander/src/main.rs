use std::collections::HashMap;


#[derive(Debug, Clone)]
enum Pattern {
    Wildcard,
    LogicVariable(String),
    MemoryVariable(String),
    RepeatZeroOrMore(Box<Pattern>),
    Concat(Box<Pattern>, Box<Pattern>),
    StringConstant(String),
    IntegerConstant(i32),
}

#[derive(Debug, Eq, PartialEq)]
enum Output {
    IntegerConstant(i32),
    StringConstant(String),
    MemoryVariable(Vec<Box<Output>>)
}

impl Into<Option<Term>> for Output {
    fn into(self) -> Option<Term> {
        match self {
            Output::IntegerConstant(i) => Option::Some(Term::IntegerConstant(i)),
            Output::StringConstant(s) => Option::Some(Term::StringConstant(s)),
            _ => Option::None
        }
    }
}

impl Into<Option<Output>> for Term {
    fn into(self) -> Option<Output> {
        match self {
            Term::IntegerConstant(i) => Option::Some(Output::IntegerConstant(i)),
            Term::StringConstant(s) => Option::Some(Output::StringConstant(s)),
            _ => Option::None
        }
    }
}


// What other sorts of values do I want?
// Vectors, Seq, Etc?
// How do I match with json?
// How do people define custom matching operators?
// Can this work with plan text?
#[derive(Debug, Eq, PartialEq, Clone)]
enum Term {
    Concat(Box<Term>, Box<Term>),
    StringConstant(String),
    IntegerConstant(i32),
}



#[derive(Debug)]
enum PatternResult<'a> {
    Error(String, &'a mut HashMap<String, Output>),
    // Should this be a variable type instead of string?
    // Does that mean I need a new enum?
    Result(&'a mut HashMap<String, Output>),
}

fn interpret<'a>(
    pattern: Pattern,
    term: Term,
    result: PatternResult<'a>,
) -> PatternResult<'a> {
    match result {
        PatternResult::Error(s, env) => PatternResult::Error(s.to_string(), env),
        PatternResult::Result(env) => match (pattern, term) {
            (Pattern::RepeatZeroOrMore(p), Term::Concat(t1, t2)) => {
                println!("Repeat!");
                let p1 = *p.clone();
                let p2 = *p.clone();
                match interpret(p1, *t1, PatternResult::Result(env)) {
                    PatternResult::Error(_, env) => PatternResult::Result(env),
                    PatternResult::Result(env) => interpret(Pattern::RepeatZeroOrMore(Box::new(p2)), *t2, PatternResult::Result(env))
                }
            }

            (Pattern::RepeatZeroOrMore(p), term) => {
                println!("Repeat!");
                match interpret(*p, term, PatternResult::Result(env)) {
                    PatternResult::Error(_, env) => PatternResult::Result(env),
                    PatternResult::Result(env) => PatternResult::Result(env)
                }
            }
            (Pattern::Wildcard, _) => PatternResult::Result(env),
            (Pattern::MemoryVariable(name), term) => {
                println!("MemoryVariable! {}", name);
                let entry : &mut Output  = env.entry(name).or_insert(Output::MemoryVariable(vec!()));
                if let Output::MemoryVariable(v) = entry {
                    if let Some(output) = Into::into(term) {
                        v.push(Box::new(output));
                    }
                     PatternResult::Result(env)
                } else {
                    PatternResult::Error(format!("Unexpected {:?}", term), env)
                }

            }

            (Pattern::LogicVariable(name), term) => match env.get(&name) {
                Some(val) => {
                    let term_to_output : Option<Output> = Into::into(term);

                    if let Some(t) = term_to_output {
                        if t == *val {
                            return PatternResult::Result(env)
                        }
                    }
                    // borrow checker doesn't like using term here. How do I fix that?
                    PatternResult::Error(format!("Unexpected1"), env)
                }
                None => {
                    println!("{}", name);
                    match Into::into(term) {
                        Some(output) => {
                            env.insert(name.to_string(), output);
                            PatternResult::Result(env)
                        }
                        // borrow checker doesn't like using term here. How do I fix that?
                        None => PatternResult::Error(format!("Unexpected2"), env)
                    }
                }
            },
            (Pattern::Concat(pattern1, pattern2), Term::Concat(term1, term2)) => {
                println!("Concat!");
                let pattern1_1 = *pattern1.clone();
                if let Pattern::RepeatZeroOrMore(p1) = pattern1_1 {
                    let p1_1 = *p1.clone();
                    let p1_2 = *p1.clone();
                    match interpret(p1_1, *term1, PatternResult::Result(env)) {
                        PatternResult::Error(_res, env) => interpret(*pattern2, *term2, PatternResult::Result(env)),
                        PatternResult::Result(env) => interpret(Pattern::RepeatZeroOrMore(Box::new(p1_2)), *term2, PatternResult::Result(env))
                    }
                } else {

                    match interpret(pattern1_1, *term1, PatternResult::Result(env)) {
                        PatternResult::Error(res, env) => PatternResult::Error(res.to_string(), env),
                        PatternResult::Result(env) => interpret(*pattern2, *term2, PatternResult::Result(env))
                    }
                }
            }
            (Pattern::Concat(_, _), _) => {
                PatternResult::Error("Didn't match".to_string(), env)
            }
            (Pattern::StringConstant(str1), Term::StringConstant(str2)) => {
                if str1 == str2 {
                    PatternResult::Result(env)
                 } else {
                     PatternResult::Error("Didn't match".to_string(), env)
                 }
            }
            (Pattern::StringConstant(_), _) => {
                PatternResult::Error("Didn't match".to_string(), env)
            }
            (Pattern::IntegerConstant(i1), Term::IntegerConstant(i2)) => {
                if i1 == i2 {
                    PatternResult::Result(env)
                 } else {
                     PatternResult::Error("Didn't match".to_string(), env)
                 }
            }
            (Pattern::IntegerConstant(_), _) => {
                PatternResult::Error("Didn't match".to_string(), env)
            }
        },
    }
}


fn lvar(s : &'static str) -> Pattern {
     Pattern::LogicVariable(s.to_string())
}

fn mvar(s : &'static str) -> Pattern {
     Pattern::MemoryVariable(s.to_string())
}

fn wc() -> Pattern {
     Pattern::Wildcard
}

fn ps(s : &'static str) -> Pattern {
    Pattern::StringConstant(s.to_string())
}

fn pc(p1: Pattern, p2: Pattern) -> Pattern {
    Pattern::Concat(Box::new(p1), Box::new(p2))
}

fn ts(s : &'static str) -> Term {
    Term::StringConstant(s.to_string())
}

fn tc(p1: Term, p2: Term) -> Term {
    Term::Concat(Box::new(p1), Box::new(p2))
}

fn rpz(p1 : Pattern) -> Pattern {
    Pattern::RepeatZeroOrMore(Box::new(p1))
}



fn main() {

    let env: &mut HashMap<String, Output> = &mut HashMap::new();

    // We don't handle length mismatch
    // Do we need an end node to do that?
    // Maybe we want to allow partial matches?
    // Should lvars and mvars have the same namespace?
    // Should Zero or more be greedy?
    // This is currently match, how would I do search?
    println!(
        "{:?}",
        interpret(
            pc(lvar("x"), rpz(mvar("xs"))),
            tc(ts("x"), tc(ts("y"), ts("Hey"))),

            PatternResult::Result(env))
    );
}
