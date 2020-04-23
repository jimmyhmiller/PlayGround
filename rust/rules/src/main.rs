use std::collections::VecDeque;
use std::collections::HashMap;

// Need to think about term vs expr?
// I've come up with phases and ways to match on
// terms. But really I need a way to match on phases as well.
// Do I do this by making phases terms? Or do I introduce a pattern
// type of matching a phase?

// After I rewrite I need to then see if there are any rules that match
// the thing I just rewrote to, before I continue on with the other exprs.
// That needs to go through all the phase as well.

// Should I have a left vs right side pattern?
// Should I even have pattern vs term?

// I should probably not do recursion at all?


#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum Term {
    Literal(String),
    Call(Box<Term>, Vec<Term>),
    Block(Vec<Term>)
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum Pattern {
    Literal(String),
    LogicVariable(String),
    Call(Box<Pattern>, Vec<Pattern>),
    Block(Vec<Pattern>)
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct Clause {
    left: Pattern,
    right: Pattern
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct Rule<'a> {
    name: String,
    clauses: Vec<&'a Clause>,
    // Need scope things here
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct Submit<'a> {
    term: &'a Term
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct SelectExpr<'a> {
    term: &'a Term,
    expr: &'a Term
 }

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct SelectRule<'a> {
    term: &'a Term,
    expr: &'a Term,
    rule: &'a Rule<'a>,
    clause: &'a Clause
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct Rewrite<'a> {
    term: &'a Term,
    expr: &'a Term,
    rule: &'a Rule<'a>,
    clause: &'a Clause,
    result: Term
}


// Really this is Submit -> SelectExpr
// Why there is no good way to say these things in a statically
// typed language is something I will never understand.
fn select_terms<'a>(submit: Submit<'a>) -> VecDeque<SelectExpr<'a>> {
    let mut terms = VecDeque::new();
    let term = &submit.term;
    match term {
        Term::Literal(_) => terms.push_back(SelectExpr {term: term, expr: term}),
        Term::Call(f, args) => {
            for expr in args {
                terms.push_back(SelectExpr {term: term, expr: expr});
            }
            terms.push_back(SelectExpr {term: term, expr: f});
            terms.push_back(SelectExpr {term: term, expr: term});
        }
        Term::Block(body) => {
            for expr in body {
                terms.push_back(SelectExpr {term: term, expr: expr});
            }
            terms.push_back(SelectExpr {term: term, expr: term});
        }
    };
    terms
}

fn matches(clause: & zPattern, term: & Term) -> bool {
    // Instead of doing this, I should probably just capture the environment.
    // This is a big waste
    // @Optimize
    make_env(clause, term).is_some()
}

// Really this is SelectExpr -> [Rule] -> SelectRule
fn select_rules<'a>(selectExpr : & SelectExpr<'a>, rules : &'a Vec<&'a Rule<'a>>) -> Vec<SelectRule<'a>> {
    // This is going to be a slow linear look up, but we can optimize later
    // @Optimize
    let SelectExpr{term, expr} = selectExpr;
    let mut selected_rules = vec![];
    for rule in rules {
        for clause in &rule.clauses {
            if matches(&clause.left, expr) {
                selected_rules.push(SelectRule {term: term, expr: expr, rule: rule, clause: clause})
            }
        }
    }
    selected_rules
}

// If I have this, do I need matches?
fn make_env<'a>(clause: &'a Pattern, term: &'a Term) -> Option<HashMap<&'a Pattern, &'a Term>> {
    let mut env : HashMap<&'a Pattern, &'a Term> = HashMap::new();
    let mut queue : VecDeque<(&'a Pattern, &'a Term)> = VecDeque::new();
    queue.push_back((clause, term));
    while !queue.is_empty() {
        if let Some((clause, term)) = queue.pop_front() {
            match (clause, term) {
                // Other places I'm assuming things match.
                // Maybe I shouldn't?
                // My match code is broken right now for logic variables
                (Pattern::LogicVariable(s), x) => {
                    let val = env.get(&Pattern::LogicVariable(s.to_string()));
                    if let Some(val) = val {
                        if *val != x {
                            return Option::None;
                        }
                    }
                    env.insert(clause, x);
                }
                // Have to revisit with repeats
                (Pattern::Call(f1, args1), Term::Call(f2, args2)) => {
                    if args1.len() != args2.len() {
                        return Option::None;
                    }
                    queue.push_back((f1, f2));
                    for i in 0..args1.len() {
                        queue.push_back((&args1[i], &args2[i]))
                    }
                }
                (Pattern::Block(body1), Term::Block(body2)) => {
                    // Have to revisit with repeats
                    if body1.len() != body2.len() {
                        return Option::None;
                    }
                    for i in 0..body1.len() {
                        queue.push_back((&body1[i], &body2[i]))
                    }
                }
                _ => {}
            };
        }
    };
    if env.is_empty() {
        Option::None
    } else {
        Option::Some(env)
    }
}

fn substitute<'a>(pattern : &'a Pattern, env: &'a HashMap<&'a Pattern, &'a Term>) -> Term {
    // Assumes a complete environment
    match pattern {
        Pattern::LogicVariable(_) => (*env.get(pattern).unwrap()).clone(),
        Pattern::Call(f, args) => {
            let args = args.into_iter().map(|p| substitute(p, env)).collect();
            Term::Call(Box::new(substitute(f,env)), args)
        },
        Pattern::Block(body) => {
            let body = body.into_iter().map(|p| substitute(p, env)).collect();
            Term::Block(body)
        }
        Pattern::Literal(s) => Term::Literal(s.to_string())
    }
}

// Really this is SelectRule -> Rewrite
fn rewrite<'a>(select_rule: & SelectRule<'a>) -> Rewrite<'a> {
    let SelectRule{term, expr, rule, clause} = select_rule;
    if let Some(env) = make_env(&clause.left, expr) {
        let result = substitute(&clause.right, &env);
        Rewrite{term: term, expr: expr, rule: rule, clause: clause, result: result}
    } else {
        panic!("Rewrite called with invalid env {:?}", select_rule)
    }
}




fn main() {
    let lhs_pattern = Pattern::Call(Box::new(Pattern::LogicVariable("?x".to_string())), vec![]);
    let rhs_pattern = Pattern::Literal("Replaced!".to_string());
    let clause = Clause{left: lhs_pattern, right: rhs_pattern};
    let rule = &Rule {name: "my-rule".to_string(), clauses: vec![&clause]};
    let rules = vec![rule];
    let mut term = Term::Call(Box::new(Term::Literal("A".to_string())), vec![]);

    let mut fuel = 0;
    let mut submit = &mut Submit{term: &term};
    let mut queue = select_terms(submit.clone());
    while let Some(selected) = queue.pop_front() {
        if fuel >= 10 {
            break;
        } else {
            fuel += 1;
        }
        let selected_rules = select_rules(&selected, &rules);
        if selected_rules.is_empty() {
            continue;
        }
        for mut rule in selected_rules {
            // println!("{:?}", result);
            // I just don't see a way around this. Do I have to clone every single time?
            // Probably related to the fact that result is not a reference but everything else is.
            // Should terms not be references?

            // queue.push_front(rewrite_to_submit(&rewrite(&rule)));
            // let mut terms = select_terms(submit);
            // while let Some(term) = terms.pop_back() {
            //     queue.push_front(term);
            // }
            // let result = rewrite(&rule);

            println!("rule before {:?}", rule);
            if let Some(env) = make_env(&clause.left, rule.expr) {
                let result = substitute(&clause.right, &env);
                let x = &mut rule.expr;
                *x = &result;
                println!("rule {:?}", rule);
            } else {
                panic!("Rewrite called with invalid env {:?}", rule)
            }


            // Conceptually what I need to do is take the result from rewrite
            // and then turn it into a submit and push it on the queue.
            // Freaking rust makes that hard.
            println!("{:?}", 1);
        }
    }

}

// I'm hoplessly lost and wasting time.
