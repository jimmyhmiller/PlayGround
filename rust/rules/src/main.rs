

// Need to think about term vs expr?
// I've come up with phases and ways to match on
// terms. But really I need a way to match on phases as well.
// Do I do this by making phases terms? Or do I introduce a pattern
// type of matching a phase?

// Should I have a left vs right side pattern?

#[derive(Debug)]
enum Term {
    Literal(String),
    Call(Box<Term>, Vec<Term>),
    Block(Vec<Term>)
}

#[derive(Debug)]
enum Pattern {
    Literal(String),
    LogicVariable(String),
    Call(Box<Pattern>, Vec<Pattern>),
    Block(Vec<Pattern>)
}

#[derive(Debug)]
struct Rule<'a> {
    name: String,
    clauses: Vec<&'a Pattern>,
    // Need scope things here
}

#[derive(Debug)]
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

// Maybe build environment when matching?
fn matches(clause: & Pattern, term: & Term, ) -> bool {

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



fn main() {
    let pattern = &Pattern::LogicVariable("?x".to_string());
    let rule = &Rule {name: "my-rule".to_string(), clauses: vec![pattern]};
    let rules = vec![rule];
    let term = &Term::Literal("A".to_string());

    let submit = &Phase::Submit{term: term};
    let selected = select_term(submit);
    let rules = select_rules(selected, rules);

    println!("{:?}", rules);
}
