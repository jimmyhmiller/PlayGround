use std::time::{Instant};
use std::fmt::Debug;
use std::collections::HashMap;

type Index = usize;

#[derive(Debug, Clone)]
pub struct Node<T>  where T: Clone {
    index: usize,
    child_index: Option<usize>,
    val: T,
    parent: Option<usize>,
    children: Vec<usize>,
    exhausted: bool,
}


#[derive(Debug, Clone)]
pub enum Expr {
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

// Basic idea here is that rules are represented in a forest.
// There is a Rule node which has as its children many clauses.
// Clauses then have as children left and right.
// Maybe I should have these just as a struct instead? But this
// does give me some nice things for uniformity.
#[derive(Debug, Clone)]
pub enum Rule {
    Rule{name: Index, scopes: Vec<Index>},
    Clause,
    Left(Expr),
    Right(Expr),
}

#[derive(Debug, Clone)]
pub struct Meta {
    original_expr: Index,
    original_sub_expr: Index,
    new_expr: Index,
    new_sub_expr: Index,
}

// Allocates and uses more storage than I technically need.
// But I doubt this will ever be the bottleneck.
#[derive(Debug, Clone)]
pub struct Interner {
    lookup: HashMap<String, Index>,
    storage: Vec<String>
}

impl Interner {
    fn new() -> Interner {
        Interner {
            lookup: HashMap::new(),
            storage: vec![],
        }
    }

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
pub struct Forest<T> where T : Clone {
    arena: Vec<Node<T>>,
    current_index: usize,
}



impl<T> Forest<T> where T : Clone + Debug {

    fn new() -> Forest<T> {
        Forest {
            arena: vec![],
            current_index: 0,
        }
    }

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

    pub fn print_tree_inner(&self, node: &Node<T>, prefix: String, last: bool) {
        let current_prefix = if last { "`- " } else { "|- " };
        println!("{}{}{:?} {}", prefix, current_prefix, node.val, node.exhausted);

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
    fn copy_tree_helper(&self, mut sub_index: Index, node_index: Index, parent_index: Option<Index>, forest: &mut Forest<T>) -> Option<Index> {
        if let Some(node) = self.get(node_index) {
            let new_index = if let Some(parent_index) = parent_index {
                forest.insert(node.val.clone(), parent_index)
            } else {
                // If there is no parent it is the root which is always 0.
                Some(0)
            };
            if node_index == sub_index {
                if let Some(i) = new_index {
                    sub_index = i;
                }
            }
            for child_index in node.children.clone() {
                if let Some(i) = self.copy_tree_helper(sub_index, child_index, new_index, forest) {
                    sub_index = i;
                }
            }
            Some(sub_index)
        } else {
            None
        }
    }

    // Might need to do this for a list of indexes?
    fn garbage_collect(&mut self, index: Index, sub_index: Index) -> Option<Index> {
        let mut forest = Forest {
            arena: vec![],
            current_index: 0,
        };

        let mut result_index = None;
        if let Some(node) = self.get(index) {
            let root = forest.insert_root(node.val.clone());
            result_index = self.copy_tree_helper(sub_index, index, None, &mut forest);
        }
        *self = forest;
        result_index

    }

    // https://vallentin.dev/2019/05/14/pretty-print-tree
    pub fn print_tree(&self, index: Index) {
        if let Some(node) = self.get(index) {
            self.print_tree_inner(node, "".to_string(), true)
        }

    }

    fn persistent_change(&mut self, t : T, index: Index) {

        // Need to capture location of old node we copied.
        if let Some(node) = self.arena.get_mut(index) {
            node.val = t;
            let node_clone = node.clone();
            self.insert_node(node_clone);
        }

    }

}

#[derive(Debug, Clone)]
pub struct RootedForest<T> where T : Clone {
    pub root: Index,
    pub focus: Index,
    pub forest: Forest<T>,
}


impl<T> RootedForest<T> where T : Clone + Debug {

    fn new() -> RootedForest<T> {
        RootedForest {
            root: 0,
            focus: 0,
            forest: Forest::new(),
        }
    }

    // I should probably cache this root?
    // I can easily do this if changing root
    // goes through some method.
    fn get_root(&self) -> Option<&Node<T>> {
        self.forest.get(self.root)
    }

    fn exhaust_focus(&mut self) {
        if let Some(node) = self.forest.arena.get_mut(self.focus) {
            node.exhausted = true
        }
    }


    pub fn insert_child(&mut self, t : T) -> Option<Index> {
        // I could exhaust here if I did this
        //  from program and new things about rules.
        self.forest.insert(t, self.focus)
    }

    pub fn make_last_child_focus(&mut self) {
        if let Some(parent) = self.get_focus_parent() {
            if let Some(node) = self.forest.get(parent) {
                if let Some(last_child) = node.children.last() {
                    // I could keep track of last inserted,
                    // or I could assume it is last index
                    self.focus = *last_child;
                }
            }
        }
    }
    pub fn make_parent_focus(&mut self) {
        if let Some(parent) = self.get_focus_parent() {
            self.focus = parent;
        }
    }

    pub fn swap_and_insert(&mut self, t: T) {
        if let Some(node) = self.forest.arena.get_mut(self.focus) {
            let node_value = node.val.clone();
            node.val = t;
            self.insert_child(node_value);
        }
    }

    pub fn insert_root(&mut self, t: T) {
        let root = self.forest.insert_root(t);
        self.root = root;
        self.focus = root;
    }

    fn get_focus(&self) -> Option<&Node<T>> {
        self.forest.get(self.focus)
    }
    
    fn get(&self, index: Index) -> Option<&Node<T>> {
        self.forest.get(index)
    }

    // Could I cache this?
    // Trying to cache the root is actually harder than it seems.
    // I would need to store a reference in the struct.
    // But that reference needs a lifetime.
    fn root_is_exhausted(&self) -> bool {
        let root = self.get_root();
        root.is_none() || root.unwrap().exhausted
    }

    fn focus_is_exhausted(&self) -> bool {
        let focus = self.get_focus();
        focus.is_none() || focus.unwrap().exhausted
    }

    pub fn get_focus_parent(&self) -> Option<Index> {
        self.get_focus().and_then(|x| x.parent)
    }


    /// Should move to the next expr that needs evaluation
    fn move_to_next_expr(&mut self) {
        // let mut fuel = 0;
        loop {
            // fuel += 1;
            // if fuel > 100 {
            //     break;
            // }
            if self.root_is_exhausted() {
                return
            }
            if self.focus_is_exhausted() {
                if let Some(index) = self.get_focus_parent() {
                    // println!("Moving to parent");
                    self.focus = index;
                    continue;
                } else {
                    return
                }
            }
            if let Some(focus) = self.get_focus() {
                if focus.children.len() == 0 {
                    return
                }

                let mut all_children_exhausted = true;
                for child in focus.children.iter() {
                    if let Some(c) = self.get(*child) {
                        if !c.exhausted {
                            self.focus = *child;
                            all_children_exhausted = false;
                            break;
                        }
                    }
                }
                if all_children_exhausted {
                    break;
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    // Can we rewrite meta? We can match on meta and rewrite else where.
    // But what would it mean to rewrite meta?
    pub meta: Meta,
    // Technically these chould just be in the hashmap.
    // But it seems worthwhile to bring them to the top level.
    pub main: RootedForest<Expr>,
    pub io: RootedForest<Expr>,
    // We will need some structure for the preprocessed rules.
    // Like keeping a list of clauses by scope and type.
    pub rules: RootedForest<Rule>,
    pub symbols: Interner,
    pub scopes: HashMap<Index, RootedForest<Expr>>,
}


impl Program {

    pub fn new() -> Program {
        Program {
            meta: Meta {
                original_expr: 0,
                original_sub_expr: 0,
                new_expr: 0,
                new_sub_expr: 0,
            },
            main: RootedForest::new(),
            io: RootedForest::new(),
            rules: RootedForest::new(),
            symbols: Interner::new(),
            scopes: HashMap::new(),
        }
    }

    fn rewrite(scope: &mut RootedForest<Expr>) {
        if let Some(focus) = scope.get_focus() {
            match focus.val {
                Expr::Symbol(0) => {
                    scope.forest.persistent_change(Expr::Symbol(1),scope.focus);
                }
               _ => {
                //    println!("Exhausting");
                   scope.exhaust_focus()
               }
            }
        } else {
            scope.exhaust_focus();
        }
    }

    // Making this work for main right now even though
    // we actually need multiple scopes.
    fn step(&mut self) {
        let scope = &mut self.main;
        scope.move_to_next_expr();
        if scope.root_is_exhausted() {
            return
        }
        // rewrite should return old node location.
        // meta needs to change.
        Program::rewrite(scope)
        // if let Some((sub, root)) = Program::rewrite(scope) {
        //     let meta = Meta{
        //         original_expr: scope.root,
        //         original_sub_expr: scope.focus,
        //         new_expr: root,
        //         new_sub_expr: sub
        //     };
        //     let new_focus = scope.forest.garbage_collect(root, sub);
        //     scope.root = 0;
        //     if let Some(focus) = new_focus {
        //         scope.focus = focus;
        //     }
        //     // What do we do if this is false??
        // } else {
        //     scope.exhaust_focus();
        // }
    }

    pub fn full_step(&mut self) {
        // let mut fuel = 0;
        while !self.main.root_is_exhausted() {
            // fuel +=1;
            // if fuel > 3000 {
            //     println!("break");
            //     break;
            // }
            self.step();
        }
    }
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
// Need to add builtin symbols/rules



// Write old node to new part of arena
// Mutate existing node.
// Then meta returns a Tree. A Tree has a reference to an arena.
// I can pass the old and new node index to this tree that says, if someone
// looksup this node, then I return the other one.





// submit expr to main
// find the first reducible expr
// if not matches exhaust.
// if match:
    // check against meta
    // if meta match assign first half meta
    // based on structure of rule vs subtree insert/mutate
    // fill out second half of meta
    // eval meta
    // eval any side-effects