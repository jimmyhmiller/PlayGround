use std::time::{Instant};

#[derive(Debug, Clone)]
struct Node<T>  where T: Clone {
    index: usize,
    child_index: Option<usize>,
    val: T,
    parent: Option<usize>,
    children: Vec<usize>,
}


#[derive(Debug, Clone)]
enum Expr {
    Call,
    Symbol(usize),
    Num(usize),
}

#[derive(Debug, Clone)]
struct Forest<T> where T : Clone {
    arena: Vec<Node<T>>,
    current_index: usize,
}

type Index = usize;

impl<T> Forest<T> where T : Clone {
    fn insert_root(&mut self, t: T) -> Index {
        let index = self.current_index;
        let n = Node {
            index,
            child_index: None,
            val: t,
            parent: None,
            children: vec![],
        };
        self.arena.push(n);
        self.current_index += 1;
        index
    }

    // What if parent doesn't exist?
    fn insert(&mut self, t: T, parent: Index) -> Index {
        let index = self.current_index;
        let p = self.arena.get_mut(parent).unwrap();
        p.children.push(self.current_index);
        let child_index = (&*p).children.len() - 1;
        self.arena.push(Node {
            index,
            child_index: Some(child_index),
            val: t,
            parent: Some(parent),
            children: vec![],
        });
        self.current_index += 1;
        index
    }

    fn insert_node(&mut self, mut node: Node<T>) -> Index {
        let index = self.current_index;
        node.index = index;
        self.arena.push(node);
        self.current_index += 1;
        index
    }

    fn get(&self, index: Index) -> Option<&Node<T>> {
        self.arena.get(index)
    }

    // Deal with node not existing
    fn persistent_change(&mut self, t : T, index: Index) -> Index {
        let mut node =  self.get(index).unwrap().clone();
        let child_index = node.child_index;
        let mut parent_index = node.parent;
        node.val = t;

        // Instead of doing the double updating stuff here,
        // I can just use current_index to premake these nodes.
        // Then no need to do multiple array lookups.
        let new_index = self.insert_node(node);
        while let Some(index) = parent_index {
            let mut parent = self.get(index).unwrap().clone();
            parent_index = parent.parent;
            parent.children[child_index.unwrap()] = new_index;
            let p = self.insert_node(parent);
            self.arena.get_mut(new_index).unwrap().parent = Some(p);
        }
        new_index
    }
}



pub fn run_new() {


    let mut f = Forest::<Expr> {
        arena: Vec::with_capacity(2),
        current_index: 0,
    };
    let now = Instant::now();
    let depth = 3000;
    let width = 3000;
    let root = f.insert_root(Expr::Call);
    let mut p = root;
    for _ in 0..depth {
        for _ in 0..width{
            p = f.insert(Expr::Symbol(0), p);
        }
    }
    println!("{}", now.elapsed().as_micros());

    // let n1 = f.insert_root(Expr::Call);
    // let n2 = f.insert(Expr::Symbol(0), n1);
    // let n3 = f.insert(Expr::Call, n1);
    // let n4 = f.insert(Expr::Symbol(1), n3);
    // let n5 = f.insert(Expr::Num(2), n4);

    let now = Instant::now();
    
    f.persistent_change(Expr::Symbol(1), depth*width);
    println!("{:#?} : length: {}", f.get(depth*width), f.arena.len());
    println!("{}", now.elapsed().as_micros());
    std::thread::spawn(move || drop(f));



}