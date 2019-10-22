use std::collections::HashMap;

#[derive(Debug)]
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
    Result(&'a mut HashMap<String, Term<'a>>),
}

fn interpret<'a>(
    pattern: Pattern<'a>,
    term: Term<'a>,
    result: PatternResult<'a>,
) -> PatternResult<'a> {
    match result {
        PatternResult::Error(s) => PatternResult::Error(s),
        PatternResult::Result(env) => match (pattern, term) {
            (Pattern::Wildcard, _) => PatternResult::Result(env),
            (Pattern::LogicVariable(name), term) => match env.get(&name) {
                Some(val) => {
                    if val == &term {
                        PatternResult::Result(env)
                    } else {
                        PatternResult::Error(format!(
                            "Failed to match {0}, found value {1:?} expected {2:?}",
                            name, val, &term
                        ))
                    }
                }
                None => {
                    env.insert(name, term);
                    PatternResult::Result(env)
                }
            },
            // Temp
            _ => PatternResult::Result(env),
        },
    }
}

fn main() {
    // I thought I could deal with raw constructors, but not if I have to let bind everything.alloc
    // I guess I'm going to need to make a macro sooner rather than later
    let c1 = &Pattern::LogicVariable("x".to_string());
    let c2 = &Pattern::Concat(&Pattern::Wildcard, c1);
    let c3 = &Pattern::StringConstant("Hello".to_string());
    let examplePattern: Pattern = Pattern::Concat(c2, c3);
    let c4 = &Term::StringConstant("test".to_string());
    let c5 = &Term::Concat(c4, c4);
    let c6 = &Term::StringConstant("Hello".to_string());
    let exampleTerm: Term = Term::Concat(c5, c6);
    let env: &mut HashMap<String, Term> = &mut HashMap::new();

    println!(
        "{:?}",
        interpret(examplePattern, exampleTerm, PatternResult::Result(env))
    );
    println!("Hello, world!");
}
