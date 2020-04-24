#![feature(box_syntax, box_patterns)]
use std::collections::VecDeque;


#[derive(Debug, Clone)]
enum Term {
    Literal(String),
    Int(i64),
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
        Term::Literal(self.to_string())
    }
}
impl Into<Term> for String {
    fn into(self) -> Term {
        Term::Literal(self)
    }
}
impl Into<Term> for &String {
    fn into(self) -> Term {
        Term::Literal(self.to_string())
    }
}


impl Term {
    fn literal(s : &str) -> Term {
        Term::Literal(s.to_string())
    }

    fn pair<T : Into<Term>, S : Into<Term>>(term1 : T, term2 : S) -> Term {
        Term::Pair(Box::new(term1.into()), Box::new(term2.into()))
    }
    fn three<T : Into<Term>, S : Into<Term>, R : Into<Term>>(term1 : T, term2 : S, term3 : R) -> Term {
        Term::Three(Box::new(term1.into()), Box::new(term2.into()), Box::new(term3.into()))
    }

    fn children(&mut self) -> Vec<&mut Box<Term>> {
        match self {
            Term::Pair(ref mut t1, ref mut t2) => vec![t2, t1],
            Term::Three(ref mut t1, ref mut t2, ref mut t3) => vec![t3, t2, t1],
            _ => vec![]
        }
    }

    fn rewrite_once(&mut self, f: fn(& Term) -> Option<Term>) -> bool {
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


fn main() {


    let fib = |term : & Term| {
        match term {
            Term::Pair(box Term::Literal(ref fib), box Term::Int(ref number)) if fib == "fib" && *number == 0 => Some(0.into()),
            Term::Pair(box Term::Literal(ref fib), box Term::Int(ref number)) if fib == "fib" && *number == 1 => Some(1.into()),
            Term::Pair(box Term::Literal(ref fib), box Term::Int(n)) if fib == "fib"=>
                Some(Term::three("+", Term::pair("fib", Term::three("-", n, 1)),
                                      Term::pair("fib", Term::three("-", n, 2)))),

            Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "-" => Some(Term::Int(n - m)),
            Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "+" => Some(Term::Int(n + m)),
            _ => None
        }
    };

    let fact = |term : & Term| {
        match term {
            Term::Pair(box Term::Literal(ref fact), box Term::Int(ref number)) if fact == "fact" && *number == 0 => Some(1.into()),
            Term::Pair(box Term::Literal(ref fact), box Term::Int(n)) if fact == "fact"=>
                Some(Term::three("*", n,Term::pair("fact", Term::three("-", n, 1)))),

            Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "-" => Some(Term::Int(n - m)),
            Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "+" => Some(Term::Int(n + m)),
            Term::Three(box Term::Literal(ref f), box Term::Int(n), box Term::Int(m)) if f == "*" => Some(Term::Int(n * m)),
            _ => None
        }
    };

    let mut my_fact_test = Term::pair("fact", 10);

    println!("{:?}", my_fact_test);
    while my_fact_test.rewrite_once(fact) {
        println!("{:?}", my_fact_test);
    }
    println!("{:?}", my_fact_test);

}
