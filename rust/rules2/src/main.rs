use std::collections::VecDeque;


#[derive(Debug, Clone)]
enum Term {
    Literal(String, TermContext),
    Pair(Box<Term>, Box<Term>, TermContext)
}

impl Term {
    fn literal(s : &str) -> Term {
        Term::Literal(s.to_string(), Default::default())
    }

    fn pair(term1 : Term, term2 : Term) -> Term {
        Term::Pair(Box::new(term1), Box::new(term2), Default::default())
    }

    fn update_first(&mut self, new_term : Term) {
        if let Term::Pair(ref mut t1,_t2,_data) = self {
            **t1 = new_term;
        }
    }

    fn update_second(&mut self, new_term : Term) {
        if let Term::Pair(_t1, ref mut t2, _data) = self {
            **t2 = new_term;
        }
    }

    fn first_mut(&mut self) -> Option<&mut Box<Term>> {
        if let Term::Pair(ref mut t1, _t2, _data) = self {
            Some(t1)
        } else {
            None
        }
    }

    fn children(&mut self) -> Vec<&mut Box<Term>> {
        match self {
            Term::Pair(ref mut t1, ref mut t2, _) => vec![t2, t1],
            _ => vec![]
        }
    }

    fn traverse(&mut self, f: fn(& Term) -> Option<Term>) {
        let mut queue = VecDeque::new();
        let result = f(self);
        if result.is_some() {
            *self = result.unwrap();
            self.traverse(f);
        }
        for child in self.children() {
            queue.push_front(child);
        }
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            println!("{:?}", node);
            let result = f(node);
            if result.is_some() {
                **node = result.unwrap();
                // println!("{:?}", node);
                queue.push_front(node);
            } else {
                for child in node.children() {
                    queue.push_front(child)
                }
            }
        }
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

fn main() {

    // let mut queue = VecDeque::new();
    let mut my_term = Term::pair(Term::pair(Term::literal("first"), Term::literal("second")), Term::literal("third"));
    my_term.traverse(|term| {
        println!("{:?}", term);
        match term {
            Term::Literal(ref s, _) if s == "first" => Some(Term::literal("thing")),
            Term::Literal(ref s, _) if s == "second" => Some(Term::literal("stuff")),
            Term::Pair(_, _, _) => Some(Term::literal("second")),
            _ => None
        }
    });

    println!("{:?}", my_term);


}
