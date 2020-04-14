use std::collections::VecDeque;
use std::collections::HashMap;

// Need to think about term vs expr?
// I've come up with phases and ways to match on
// terms. But really I need a way to match on phases as well.
// Do I do this by making phases terms? Or do I introduce a pattern
// type of matching a phase?

// Should I have a left vs right side pattern?

#[derive(Debug, PartialEq, Eq, Hash)]
enum Term {
    Literal(String),
    Call(Box<Term>, Vec<Term>),
    Block(Vec<Term>)
}

#[derive(Debug, PartialEq, Eq, Hash)]
enum Pattern {
    Literal(String),
    LogicVariable(String),
    Call(Box<Pattern>, Vec<Pattern>),
    Block(Vec<Pattern>)
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct Rule<'a> {
    name: String,
    clauses: Vec<&'a Pattern>,
    // Need scope things here
}

// In some ways I like phase being one thing here.
// But maybe the structs should be separate? Really not sure.
#[derive(Debug, PartialEq, Eq, Hash)]
enum Phase<'a> {
    Submit {term: &'a Term},
    SelectExpr {term: &'a Term, expr: &'a Term},
    SelectRule {term: &'a Term, expr: &'a Term, rule: &'a Rule<'a>, clause: &'a Pattern},
    Rewrite {term: Term, expr: Term, result: Term, rule: Rule<'a>, clause: Term}
}

// Really this is Submit -> SelectExpr
// Why there is no good way to say these things in a statically
// typed language is something I will never understand.
fn select_term<'a>(phase: & Phase<'a>) -> Phase<'a> {
    if let Phase::Submit { term } = phase {
        match term {
            Term::Literal(_) => Phase::SelectExpr {term: term, expr: term},
            Term::Call(f, args) if args.len() == 0 => Phase::SelectExpr {term: term, expr: f},
            Term::Call(f, args) => Phase::SelectExpr {term: term, expr: args.last().unwrap()},
            Term::Block(body)  => Phase::SelectExpr {term: term, expr: body.first().unwrap()}
        }
    } else {
        panic!("Passed something that wasn't a submit to SelectTerm {:?}", phase);
    }
}

fn matches_all(patterns: & Vec<Pattern>, terms: & Vec<Term>) -> bool {
    if patterns.len() != terms.len() {
        return false
    }
    for i in 0..patterns.len() {
        if !matches(&patterns[i], &terms[i]) {
            return false
        }
    }
    true
}

// Need some notion of an environment here to have correct semantics
// for logic variables.
fn matches(clause: & Pattern, term: & Term) -> bool {

    match (clause, term) {
        (Pattern::LogicVariable(_), _) => true,
        (Pattern::Literal(s1), Term::Literal(s2)) => s1 == s2,
        (Pattern::Call(f1, args1), Term::Call(f2, args2)) => matches(f1, f2) &&
                                                             matches_all(args1, args2),
        (Pattern::Block(body1), Term::Block(body2)) => matches_all(body1, body2),
        (_, _) => false
    }
}

// Really this is SelectExpr -> [Rule] -> SelectRule
fn select_rules<'a>(phase : Phase<'a>, rules: std::vec::Vec<&'a Rule<'a>>) -> Vec<Phase<'a>> {
    // This is going to be a slow linear look up, but we can optimize later
    // @Optimize
    if let Phase::SelectExpr{term, expr} = phase {
        let mut selected_rules = vec![];
        for rule in rules {
            for clause in &rule.clauses {
                if matches(clause, expr) {
                    selected_rules.push(Phase::SelectRule {term: term, expr: expr, rule: rule, clause: clause})
                }
            }
        }
        selected_rules
    } else {
        panic!("Passed something that wasn't a submit to SelectExpr {:?}", phase);
    }
}

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
    Option::Some(env)
}

// Need subst next


fn main() {
    let pattern = &Pattern::LogicVariable("?x".to_string());
    let rule = &Rule {name: "my-rule".to_string(), clauses: vec![pattern]};
    let rules = vec![rule];
    let term = &Term::Literal("A".to_string());

    let submit = &Phase::Submit{term: term};
    let selected = select_term(submit);
    let rules = select_rules(selected, rules);
    if let Phase::SelectRule{expr, clause, term: _, rule: _} = rules[0] {
        let env = make_env(clause, expr);
        println!("{:?}", rules);
        println!("{:?}", env);
    }
}
