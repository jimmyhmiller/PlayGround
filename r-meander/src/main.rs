use std::collections::HashMap;

#[derive(Debug, Clone)]
enum Pattern<'a> {
    Wildcard,
    LogicVariable(String),
    Concat(&'a Pattern<'a>, &'a Pattern<'a>),
    StringConstant(String),
    IntegerConstant(i32),
}

// What other sorts of values do I want?
// Vectors, Seq, Etc?
// How do I match with json?
// How do people define custom matching operators?
// Can this work with plan text?
#[derive(Debug, Eq, PartialEq)]
enum Term<'a> {
    Concat(&'a Term<'a>, &'a Term<'a>),
    StringConstant(String),
    IntegerConstant(i32),
}

#[derive(Debug)]
enum PatternResult<'a> {
    Error(String),
    // Should this be a variable type instead of string?
    // Does that mean I need a new enum?
    Result(&'a mut HashMap<String, &'a Term<'a>>),
}

fn interpret<'a>(
    pattern: &'a Pattern<'a>,
    term: &'a Term<'a>,
    result: PatternResult<'a>,
) -> PatternResult<'a> {
    match result {
        PatternResult::Error(s) => PatternResult::Error(s.to_string()),
        PatternResult::Result(env) => match (pattern, term) {
            (Pattern::Wildcard, _) => PatternResult::Result(env),
            (Pattern::LogicVariable(name), term) => match env.get(name) {
                Some(val) => {
                    if val == &term {
                        PatternResult::Result(env)
                    } else {
                        PatternResult::Error(format!(
                            "Failed to match {0}, found value {1:?} expected {2:?}",
                            name, val, term
                        ))
                    }
                }
                None => {
                    env.insert(name.to_string(), term);
                    PatternResult::Result(env)
                }
            },
            (Pattern::Concat(pattern1, pattern2), Term::Concat(term1, term2)) => {
                match interpret(pattern1, term1, PatternResult::Result(env)) {
                    PatternResult::Error(res) => PatternResult::Error(res.to_string()),
                    PatternResult::Result(env) => interpret(pattern2, term2, PatternResult::Result(env))
                }
            }
            (Pattern::Concat(_, _), _) => {
                PatternResult::Error("Didn't match".to_string())
            }
            (Pattern::StringConstant(str1), Term::StringConstant(str2)) => {
                if str1 == str2 {
                    PatternResult::Result(env)
                 } else {
                     PatternResult::Error("Didn't match".to_string())
                 }
            }
            (Pattern::StringConstant(_), _) => {
                PatternResult::Error("Didn't match".to_string())
            }
            (Pattern::IntegerConstant(i1), Term::IntegerConstant(i2)) => {
                if i1 == i2 {
                    PatternResult::Result(env)
                 } else {
                     PatternResult::Error("Didn't match".to_string())
                 }
            }
            (Pattern::IntegerConstant(_), _) => {
                PatternResult::Error("Didn't match".to_string())
            }
        },
    }
}

fn main() {
    // I thought I could deal with raw constructors, but not if I have to let bind everything.alloc
    // I guess I'm going to need to make a macro sooner rather than later
    let c1 = &Pattern::LogicVariable("x".to_string());
    let c2 = &Pattern::Concat(&Pattern::Wildcard, c1);
    let c3 = &Pattern::StringConstant("Hello".to_string());
    let example_pattern: Pattern = Pattern::Concat(c2, c3);
    let c4 = &Term::StringConstant("test".to_string());
    let c5 = &Term::Concat(c4, c4);
    let c6 = &Term::StringConstant("Hello".to_string());
    let example_term: Term = Term::Concat(c5, c6);
    let env: &mut HashMap<String, &Term> = &mut HashMap::new();

    println!(
        "{:?}\n {:?}\n {:?}",
        example_pattern,
        example_term,
        interpret(&example_pattern, &example_term, PatternResult::Result(env))
    );
    println!("Hello, world!");
}
