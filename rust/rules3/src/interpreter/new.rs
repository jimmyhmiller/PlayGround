use std::time::{Instant};
use std::fmt::Debug;
use std::collections::HashMap;

type Index = usize;

#[derive(Debug, Clone)]
struct Node<T>  where T: Clone {
    index: usize,
    child_index: Option<usize>,
    val: T,
    parent: Option<usize>,
    children: Vec<usize>,
    exhausted: bool,
}


#[derive(Debug, Clone)]
enum Expr {
    Call,
    Symbol(Index),
    Scope(Index),
    LogicVariable(Index),
    Num(usize),
    Do,
    Array,
    Map, // How do I do this one? I guess its children are just even?
    Quote,
}

// Before I've had it so that clauses are in scopes.
// but really that is a matter of their content.
// they can even match with multiple scopes.
// So I think that will be changed with Expr, 
// not clause or rule.
#[derive(Debug, Clone)]
struct Clause {
    left: Index,
    right: Index
}


#[derive(Debug, Clone)]
struct Rule {
    name: Index,
    // This is really a set but it will probably be small so vec is fine?
    // Also, the scopes here are really derived from the clauses,
    // but not having to check each clause just makes sense.
    scopes: Vec<Index>, 
    clauses: Vec<Clause>,
}

#[derive(Debug, Clone)]
struct Meta {
    original_expr: Index,
    original_sub_expr: Index,
    new_expr: Index,
    new_sub_expr: Index,
}

// Allocates and uses more storage than I technically need.
// But I doubt this will ever be the bottleneck.
struct Interner {
    lookup: HashMap<String, Index>,
    storage: Vec<String>
}

impl Interner {
    fn intern(&mut self, symbol : &str) -> Index {
        if let Some(index) = self.lookup.get(symbol) {
            return *index
        }
        let index = self.storage.len();
        self.lookup.insert(symbol.to_string(), index);
        self.storage.push(symbol.to_string());
        index
    }

    fn lookup(&self, index: Index) -> Option<&String> {
        self.storage.get(index)
    }
}

#[derive(Debug, Clone)]
struct Forest<T> where T : Clone {
    arena: Vec<Node<T>>,
    current_index: usize,
}



impl<T> Forest<T> where T : Clone + Debug {
    fn insert_root(&mut self, t: T) -> Index {
        let index = self.next_index();
        let n = Node {
            index,
            child_index: None,
            val: t,
            parent: None,
            children: vec![],
            exhausted: false
        };
        self.arena.push(n);
        index
    }

    fn next_index(&mut self) -> usize {
        let index = self.current_index;
        self.current_index += 1;
        index
    }

    // What if parent doesn't exist?
    fn insert(&mut self, t: T, parent: Index) -> Option<Index> {
        let index = self.next_index();
        let p = self.arena.get_mut(parent)?;
        p.children.push(index);
        let child_index = (&*p).children.len() - 1;
        self.arena.push(Node {
            index,
            child_index: Some(child_index),
            val: t,
            parent: Some(parent),
            children: vec![],
            exhausted: false
        });
        Some(index)
    }

    fn insert_node(&mut self, mut node: Node<T>) -> Index {
        let index = self.next_index();
        node.index = index;
        self.arena.push(node);
        index
    }

    fn get(&self, index: Index) -> Option<&Node<T>> {
        self.arena.get(index)
    }
    fn print_tree_inner(&self, node: &Node<T>, prefix: String, last: bool) {
        let current_prefix = if last { "`- " } else { "|- " };
        println!("{}{}{:?}", prefix, current_prefix, node.val);

        let child_prefix = if last { "   " } else { "|  " };
        let prefix = prefix + child_prefix;
        if !node.children.is_empty() {
            let last_child = node.children.len() - 1;

            for (i, child_index) in node.children.iter().enumerate() {
                if let Some(child) = self.get(*child_index) {
                    self.print_tree_inner(&child, prefix.to_string(), i == last_child);
                }
            }
        }
    }
    fn copy_tree_helper(&self, node_index: Index, parent_index: Option<Index>, forest: &mut Forest<T>) {
        if let Some(node) = self.get(node_index) {
            let new_index = if let Some(parent_index) = parent_index {
                forest.insert(node.val.clone(), parent_index)
            } else {
                Some(node_index)
            };
            for child_index in node.children.clone() {
                self.copy_tree_helper(child_index, new_index, forest)
            }
        }
    }

    // Might need to do this for a list of indexes?
    fn garbage_collect(&mut self, index: Index) {
        let mut forest = Forest {
            arena: vec![],
            current_index: 0,
        };

        if let Some(node) = self.get(index) {
            let root = forest.insert_root(node.val.clone());
            self.copy_tree_helper(root, None, &mut forest);
        }
        *self = forest
    }

    // https://vallentin.dev/2019/05/14/pretty-print-tree
    fn print_tree(&self, index: Index) {
        if let Some(node) = self.get(index) {
            self.print_tree_inner(node, "".to_string(), true)
        }

    }

    /// Returns new Index of node modified and its new root.
    fn persistent_change(&mut self, t : T, index: Index) -> Option<(Index, Index)> {

        let mut node = self.get(index)?.clone();
        node.val = t;

        let mut child_index = node.child_index;
        let mut parent_index = node.parent;
        let mut new_index = self.next_index();
        let original_index = new_index;

        node.index = new_index;
        self.arena.push(node);
        // If this element doesn't have any parent,
        // it is the root and we need to return it.
        let mut last_parent = new_index;
        while let (Some(index), Some(child_index_value)) = (parent_index, child_index) {
            let new_parent_index = self.next_index();
            let mut parent = self.get(index)?.clone();
            
            parent.index = new_parent_index;
            parent.children[child_index_value] = new_index;
            last_parent = new_parent_index;
            
            parent_index = parent.parent;
            child_index = parent.child_index;
            new_index = parent.index;

            self.arena.push(parent);

        }
        Some((original_index, last_parent))
    }
}



pub fn run_new() {


    let mut f = Forest::<Expr> {
        arena: Vec::with_capacity(2),
        current_index: 0,
    };
    let now = Instant::now();
    let depth = 10;
    let width = 10;
    let root = f.insert_root(Expr::Call);
    let mut p = root;
    for _ in 0..depth {
        if let Some(index) = f.insert(Expr::Symbol(0), p) {
            p = index;
            for _ in 0..width{
                f.insert(Expr::Symbol(0), p);
            }
        }
    }
    // println!("{}", now.elapsed().as_millis());

    // let n1 = f.insert_root(Expr::Call);
    // let n2 = f.insert(Expr::Symbol(0), n1);
    // let n3 = f.insert(Expr::Call, n1);
    // let n4 = f.insert(Expr::Symbol(1), n3);
    // let n5 = f.insert(Expr::Num(2), n4);

    let now = Instant::now();

    let mut result = f.persistent_change(Expr::Symbol(1), depth*width);
    for _ in 0..100000 {
        result = f.persistent_change(Expr::Symbol(1), depth*width);
    }
    println!("{}", now.elapsed().as_millis());
    
    let now = Instant::now();
    if let Some((node, root)) = result {
        // println!("length: {}, {}, {:#?}", f.arena.len(), node, f.get(root));

        // println!("{}", root);
        // f.print_tree(root);
        println!("{}", f.arena.len());
        f.garbage_collect(root);
        println!("{}", f.arena.len());
    }
    println!("{}", now.elapsed().as_millis());
    // f.print_tree(root);


    
    // Silly litle trick that does speed things up.
    std::thread::spawn(move || drop(f));



}


// I need to hook up the parser to these trees.
// I need to process rules
// I need to do an precompute for exhaustion.
// Then I could either exhaust as I parse, or exhaust across the whole tree in linear time.
// I need to get basic evaluation working. Starting at the left most node.
// Can I keep track of that node as I parse? So then there is no traversing?
// If so, can I really keep track of the left most node? What about keeping track
// of only the nodes that aren't obviously exhausted?
// I need to think about patterns of rules that create new nodes vs using existing ones.
// I need to create built in rules.
// I need to think about whether I need need an environment for logic variables.
// If things are left linear, then a logic variable just means reference this node.
// If things aren't left linear I would need node equality (will need that eventually anyways.)
// One obvious optimization for equality would be to do graph reduction, but that complicates things.
// Can I reduce rules into some form of bytecode?
// Am I that far way from a compiler? Can I eliminate these vectors of children a the nodes?
// Is there a good model of stack computation to be had here?

