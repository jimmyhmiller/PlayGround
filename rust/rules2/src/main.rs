#![feature(box_syntax, box_patterns)]
use core::fmt::Debug;
use std::collections::VecDeque;
use std::collections::HashMap;


#[derive(Debug, Clone)]
enum Term {
    Literal(String),
    Int(i64),
    LogicVariable(String),
    Pair(Box<Term>, Box<Term>),
    Three(Box<Term>, Box<Term>, Box<Term>)
}


impl Into<Term> for &i64 {
    fn into(self) -> Term {
        Term::Int(*self)
    }
}
impl Into<Term> for i64 {
    fn into(self) -> Term {
        Term::Int(self)
    }
}
impl Into<Term> for &str {
    fn into(self) -> Term {
        self.to_string().into()
    }
}
impl Into<Term> for String {
    fn into(self) -> Term {
        if self.starts_with("?") {
            Term::LogicVariable(self)
        } else {
            Term::Literal(self)
        }
    }
}

impl Into<Term> for &String {
    fn into(self) -> Term {
        Term::Literal(self.to_string())
    }
}

impl<T : Into<Term>, S : Into<Term>> Into<Term> for (T, S) {
    fn into(self) -> Term {
        Term::Pair(Box::new(self.0.into()), Box::new(self.1.into()))
    }
}
impl<T : Into<Term>, S : Into<Term>, R : Into<Term>> Into<Term> for (T, S, R) {
    fn into(self) -> Term {
        Term::Three(Box::new(self.0.into()), Box::new(self.1.into()), Box::new(self.2.into()))
    }
}

impl<T : Into<Term>, S : Into<Term>> Into<Rule> for (T, S) {
    fn into(self) -> Rule {
        Rule {
            left: self.0.into(),
            right: self.1.into(),
        }
    }
}

struct TermIter {
    queue: VecDeque<Box<Term>>,
}

impl TermIter {

    fn new(term: Term) -> TermIter {
        let mut queue = VecDeque::new();
        for child in term.children_not_mut() {
            queue.push_front(child);
        };
        TermIter {
            queue: queue,
        }
    }
    // I didn't implement a real iterator. But hopefully
    // this is some start to one.
    // Going to run into problems with lifetimes in associated types?
    // https://lukaskalbertodt.github.io/2018/08/03/solving-the-generalized-streaming-iterator-problem-without-gats.html

    fn next(&mut self) -> Option<Box<Term>> {
        let mut fuel = 0;
        loop {
            fuel += 1;
            if fuel > 20 {
                println!("Fuel Ran Out!");
                return None
            }
            if let Some(child) = self.queue.pop_front() {
                match child {
                    box Term::Int(_) => return Some(child),
                    box Term::Literal(_) => return Some(child),
                    box Term::LogicVariable(_) => return Some(child),
                    box Term::Pair(l, r) => {
                        self.queue.push_front(r);
                        if l.is_atom() {
                            return Some(l);
                        } else {
                            for child in l.children_not_mut() {
                                self.queue.push_front(child)
                            }
                            continue;
                        }
                    },

                    box Term::Three(l, m, r) => {
                        self.queue.push_front(r);
                        self.queue.push_front(m);
                        if l.is_atom() {
                            return Some(l);
                        } else {
                            for child in l.children_not_mut() {
                                self.queue.push_front(child)
                            }
                        }
                    }
                };
            } else {
                return None
            }
        }
    }
}


impl Term {

    fn is_atom(&self) -> bool {
        match self {
            Term::Int(_) => true,
            Term::Literal(_) => true,
            Term::LogicVariable(_) => true,
            _ => false,
        }
    }

    fn children<'a>(&'a mut self) -> Vec<&'a mut Box<Term>> {
        match self {
            Term::Pair(ref mut t1, ref mut t2) => vec![t2, t1],
            Term::Three(ref mut t1, ref mut t2, ref mut t3) => vec![t3, t2, t1],
            _ => vec![]
        }
    }
    fn children_not_mut<'a>(self) -> Vec<Box<Term>> {
        match self {
            Term::Pair(t1, t2) => vec![t2, t1],
            Term::Three(t1, t2, t3) => vec![t3, t2, t1],
            _ => vec![]
        }
    }

    // This strategy is wrong sadly. See comment below
    // So maybe, I keep state between calls in the form of the queue?
    // not 100% sure how that would work out, but seems doable.
    fn rewrite_once<F>(&mut self, f: F) -> bool where
            F: Fn(& Term) -> Option<Term>{
        let mut queue = VecDeque::new();
        let result = f(self);
        if result.is_some() {
            *self = result.unwrap();
            return true;
        }
        for child in self.children() {
            queue.push_front(child);
        }
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            let result = f(node);
            if result.is_some() {
                **node = result.unwrap();
                return true;
            } else {
                for child in node.children() {
                    queue.push_front(child)
                }
            }
        }
        return false;
    }
}

#[derive(Debug, Clone)]
struct TermContext {
    finished: bool
}

impl Default for TermContext {
    fn default() -> Self {
        TermContext { finished: false }
     }
}

#[derive(Debug, Clone)]
struct Rule {
    left: Term,
    right: Term,
}


// Maybe I do something like this.
// get next node
// rewrite
// if rule applied next is self
// else next iterates through children

// So I can say, give me node, rewrite, give me node rewrite, forever until done.alloc
// Rewrite does need to go through the phases. But that seems reasonable.


// So this works. I'm not sure if it is right or not.
// Intuitively it does seem like I need to always start from the outside and work my way in.
// What if a rewrite rule makes it so that some more general rule matches?

// I need to actual make rules and logic variables, with real substitution.
// I need meta-evaluation. I need phases. I need some ffi?

impl Rule {
    fn run(self, term : & Term) -> Option<Term> {
        let mut env : HashMap<String, Term> = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_front((&self.left, term));
        let mut failed = false;
        while !queue.is_empty() && !failed {
            let elem = queue.pop_front().unwrap();
            match elem {
                // Need to handle non-linear
                (Term::LogicVariable(s), t) => {
                    env.insert(s.to_string(), t.clone());
                },
                (Term::Literal(ref s1), Term::Literal(ref s2)) if s1 == s2 => {},
                (Term::Int(ref n1), Term::Int(ref n2)) if n1 == n2 => {},
                (Term::Pair(a1, b1), Term::Pair(a2, b2)) => {
                     queue.push_front((&*a1, &*a2));
                     queue.push_front((&*b1, &*b2));
                 },
                 (Term::Three(a1, b1, c1), Term::Three(a2, b2, c2)) => {
                      queue.push_front((&*a1, &*a2));
                      queue.push_front((&*b1, &*b2));
                      queue.push_front((&*c1, &*c2));
                  },
                _ => {
                    failed = true;
                }
            };
        };

        let f = |term : & Term| {
            match term {
                Term::LogicVariable(s) => Some(env.get(s).unwrap().clone()),
                _ => None
            }
        };
        let mut right = self.right.clone();
        if !failed {
            while right.rewrite_once(f) {}
            Some(right)
        } else {
            None
        }

    }
}


fn main() {


    let fib = |term : & Term| {
        match term {
            Term::Pair(box Term::Literal(ref fib), box Term::Int(ref number)) if fib == "fib" && *number == 0 => Some(0.into()),
            Term::Pair(box Term::Literal(ref fib), box Term::Int(ref number)) if fib == "fib" && *number == 1 => Some(1.into()),
            Term::Pair(box Term::Literal(ref fib), box Term::Int(n)) if fib == "fib"=>
                Some(("+", ("fib", ("-", n, 1),
                           ("fib", ("-", n, 2)))).into()),

            Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "-" => Some(Term::Int(n - m)),
            Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "+" => Some(Term::Int(n + m)),
            _ => None
        }
    };

    let fact = |term : & Term| {
        match term {
            Term::Pair(box Term::Literal(ref fact), box Term::Int(ref number)) if fact == "fact" && *number == 0 => Some(1.into()),
            Term::Pair(box Term::Literal(ref fact), box Term::Int(n)) if fact == "fact"=>
                Some(("*", n, ("fact", ("-", n, 1))).into()),


            Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "-" => Some(Term::Int(n - m)),
            Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "+" => Some(Term::Int(n + m)),
            Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "*" => Some(Term::Int(n * m)),
            _ => None
        }
    };

    let my_rule = Rule {
        left: "?x".into(),
        right: "thing".into(),
    };

    // let t = my_rule.run(& "test".into());
    // println!("{:?}", t);

    // It seems that this proves that my evaluation strategy is wrong.
    // If we do fact(1) => 1 * fact(1 - 1)
    // then our fact(?n) rule is going to match on fact(1 - 1)
    // which will then expand infinitely.

    // let fact2 = |term : & Term| {
    //     let fact0 : Rule = (("fact", 0), 1).into();
    //     let fact_n : Rule = (("fact", "?x"), ("*", "?x", ("fact", ("-", "?x", 1)))).into();
    //     let math = |term : & Term| {
    //         match term {
    //             Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "-" => Some(Term::Int(n - m)),
    //             Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "+" => Some(Term::Int(n + m)),
    //             Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "*" => Some(Term::Int(n * m)),
    //             _ => None
    //         }
    //     };
    //     fact0.run(term).or_else(|| math(term)).or_else(|| fact_n.run(term))
    // };
    // let fact0 : Rule = (("fact", 0), 1).into();
    // let t1 = fact0.run(& ("fact", 0).into());
    // println!("{:?}", t1);

    let my_fact_test : Term = (("fact", 1)).into();

    let mut iter = TermIter::new(my_fact_test);
    let mut x = iter.next().unwrap();
    let y = iter.next();
    *x = Term::Int(2);
    let z = iter.next();
    println!("{:?}, {:?}, {:?}", x, y, z);
    // println!("{:?}", my_fact_test);
    // let y = iter.next();


    // println!("{:?}", my_fact_test);
    // let mut fuel = 0;
    // while my_fact_test.rewrite_once(fact) && fuel < 3 {
    //     println!("{:?}", my_fact_test);
    //     fuel += 1;
    // }
    // println!("{:?}", my_fact_test);

}
