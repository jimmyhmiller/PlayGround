use std::collections::HashMap;
use std::collections::VecDeque;


#[derive(Debug, Clone)]
#[allow(dead_code)]
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
#[allow(dead_code)]
enum Term {
    Concat(Box<Term>, Box<Term>),
    StringConstant(String),
    IntegerConstant(i32),
    End
}


#[allow(dead_code)]
#[derive(Debug)]
enum PatternResult<'a> {
    Error(String, &'a mut HashMap<String, Output>),
    // Should this be a variable type instead of string?
    // Does that mean I need a new enum?
    Result(&'a mut HashMap<String, Output>),
}


#[allow(dead_code)]
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
                    PatternResult::Error(format!("Unexpected"), env)
                }
                None => {
                    println!("{}", name);
                    let t1 = term.clone();
                    match Into::into(term) {
                        Some(output) => {
                            env.insert(name.to_string(), output);
                            PatternResult::Result(env)
                        }
                        // Need to extract a function for dealing with errors here
                        None => PatternResult::Error(format!("Can't convert {:?}", t1), env)
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
                        PatternResult::Result(env) => interpret(Pattern::Concat(Box::new(Pattern::RepeatZeroOrMore(Box::new(p1_2))), pattern2), *term2, PatternResult::Result(env))
                    }
                } else {

                    match interpret(pattern1_1, *term1, PatternResult::Result(env)) {
                        PatternResult::Error(res, env) => PatternResult::Error(res.to_string(), env),
                        PatternResult::Result(env) => interpret(*pattern2, *term2, PatternResult::Result(env))
                    }
                }
            }
            (Pattern::Concat(pattern1, pattern2), term) => {
                println!("Concat!");
                let pattern1_1 = *pattern1.clone();
                if let Pattern::RepeatZeroOrMore(p1) = pattern1_1 {
                    match interpret(*p1, term, PatternResult::Result(env)) {
                        PatternResult::Result(env) => interpret(*pattern2, Term::End, PatternResult::Result(env)),
                        PatternResult::Error(res, env) => PatternResult::Error(res.to_string(), env),
                    }
                } else {
                    PatternResult::Error(format!("Didn't match concat {:?} {:?} {:?}", pattern1, pattern2, term), env)
                }
            }
            (Pattern::StringConstant(str1), Term::StringConstant(str2)) => {
                if str1 == str2 {
                    PatternResult::Result(env)
                 } else {
                     PatternResult::Error(format!("Didn't match string {} {}", str1, str2), env)
                 }
            }
            (Pattern::StringConstant(s), term) => {
                PatternResult::Error(format!("{:?} is not a string and not {}", term, s), env)
            }
            (Pattern::IntegerConstant(i1), Term::IntegerConstant(i2)) => {
                if i1 == i2 {
                    PatternResult::Result(env)
                 } else {
                     PatternResult::Error(format!("Didn't match integer {} {}", i1, i2), env)
                 }
            }
            (Pattern::IntegerConstant(i), term) => {
                PatternResult::Error(format!("{:?} is not an integer and not {}", term, i), env)
            }
        },
    }
}

#[allow(dead_code)]
fn lvar(s : &'static str) -> Pattern {
     Pattern::LogicVariable(s.to_string())
}

#[allow(dead_code)]
fn mvar(s : &'static str) -> Pattern {
     Pattern::MemoryVariable(s.to_string())
}

#[allow(dead_code)]
fn wc() -> Pattern {
     Pattern::Wildcard
}

#[allow(dead_code)]
fn ps(s : &'static str) -> Pattern {
    Pattern::StringConstant(s.to_string())
}

#[allow(dead_code)]
fn pc(p1: Pattern, p2: Pattern) -> Pattern {
    Pattern::Concat(Box::new(p1), Box::new(p2))
}

#[allow(dead_code)]
fn ts(s : &'static str) -> Term {
    Term::StringConstant(s.to_string())
}

#[allow(dead_code)]
fn tc(p1: Term, p2: Term) -> Term {
    Term::Concat(Box::new(p1), Box::new(p2))
}

#[allow(dead_code)]
fn rpz(p1 : Pattern) -> Pattern {
    Pattern::RepeatZeroOrMore(Box::new(p1))
}



fn main() {

    let env: &mut HashMap<String, Output> = &mut HashMap::new();

    // We don't handle length mismatch when repeat
    // Do we need an end node to do that?
    // Maybe we want to allow partial matches?
    // Should lvars and mvars have the same namespace?
    // Should Zero or more be greedy?
    // This is currently match, how would I do search?

    // Not super happy with ending things.
    // How does zero or interact with things?
    // Maybe I should look at regex and derivatives?

    // println!(
    //     "{:?}",
    //     interpret(
    //         pc(lvar("x"), pc(lvar("y"), lvar("z"))),
    //         tc(ts("x"), ts("y")),
    //
    //         PatternResult::Result(env))
    // );
    let mut states = to_state_machine(Pattern::Concat(Box::new(Pattern::LogicVariable("x".to_string())), Box::new(Pattern::IntegerConstant(2))));
    states.sort_by_key(|s| s.label);
    for state in states {
        println!("{:?}", state)
    }

}

type Label = i32;

#[derive(Debug)]
#[allow(dead_code)]
enum Op {
    NOOP,
    Return,
    SKIP,
    CheckString(String),
    CheckInteger(i32),
    Accumulate(String),
    Assign(String),
    CheckVar(String),
}

#[derive(Debug)]
struct State {
    label: Label,
    operation: Op,
    then_case: Label, // Maybe Option?
    else_case: Label // Maybe Option?
    // Or should the be an enum? There are states that can't fail and that is good to know
}

// Should make an interator for patterns
fn to_state_machine(pattern : Pattern) -> Vec<State> {
    let mut current_label = 0;
    let start = State { label: current_label, operation: Op::NOOP, then_case: current_label + 1, else_case: -1 };
    const END_STATE : i32 = std::i32::MAX;
    let end = State { label: END_STATE, operation: Op::Return, then_case: -1, else_case: -1};
    let mut states = vec!(start, end);
    current_label += 1;

    let mut queue = VecDeque::new();
    queue.push_front(pattern);

    let mut else_context = VecDeque::new();
    else_context.push_front(END_STATE);

    // How to deal with last node before end?

    loop {
        if let Some(pattern) = queue.pop_front() {
            match pattern {
                Pattern::Wildcard => {
                    states.push(State { label: current_label, operation: Op::SKIP, then_case: current_label +1, else_case: *else_context.front().unwrap_or(&END_STATE) });
                    current_label += 1;
                }
                // Really should go to current fail not end
                Pattern::IntegerConstant(i) => {
                    states.push(State { label: current_label, operation: Op::CheckInteger(i), then_case: current_label +1, else_case: *else_context.front().unwrap_or(&END_STATE) });
                    current_label += 1;
                }
                Pattern::StringConstant(s) => {
                    states.push(State { label: current_label, operation: Op::CheckString(s), then_case: current_label +1, else_case: *else_context.front().unwrap_or(&END_STATE) });
                    current_label += 1;
                }
                Pattern::MemoryVariable(name) => {
                    states.push(State { label: current_label, operation: Op::Accumulate(name), then_case: current_label +1, else_case: *else_context.front().unwrap_or(&END_STATE) });
                    current_label += 1;
                }
                Pattern::LogicVariable(name) => {
                    states.push(State { label: current_label, operation: Op::CheckVar(name.clone()), then_case: current_label +1, else_case: *else_context.front().unwrap_or(&END_STATE) });
                    current_label += 1;
                    states.push(State { label: current_label, operation: Op::Assign(name), then_case: current_label +1, else_case: *else_context.front().unwrap_or(&END_STATE) });
                    current_label += 1;
                }
                Pattern::RepeatZeroOrMore(pattern) => {
                    queue.push_front(*pattern);
                    else_context.push_front(current_label);

                    // I guess in order for this to work I need to change the whole way then works as well.
                    // Then isn't the current_label + 1. How does this work with a concat?
                    // I need to think about this part more.
                }
                Pattern::Concat(p1, p2) => {
                    queue.push_back(*p1);
                    queue.push_back(*p2);
                }
                _ => break
            }
        } else {
            break;
        }
    }
    return states
}







// State Machine Thoughts

// Operation
// Then
// Else
