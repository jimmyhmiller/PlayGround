use std::time::{Instant};
use std::fmt::Debug;
use std::collections::HashMap;

type Index = usize;

#[derive(Debug, Clone)]
struct Node<T>  where T: Clone{
    index: usize,
    val: T,
    parent: Option<usize>,
    child: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    exhausted: bool,
}

impl<T> Copy for Node<T> where T : Copy {}


#[derive(Debug, Clone, Copy)]
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

// Basic idea here is that rules are represented in a forest.
// There is a Rule node which has as its children many clauses.
// Clauses then have as children left and right.
// Maybe I should have these just as a struct instead? But this
// does give me some nice things for uniformity.
#[derive(Debug, Clone)]
enum Rule {
    Rule{name: Index, scopes: Vec<Index>},
    Clause,
    Left(Expr),
    Right(Expr),
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
#[derive(Debug, Clone)]
struct Interner {
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
struct Forest<T> where T : Clone {
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
            val: t,
            parent: None,
            child: None,
            left: None,
            right: None,
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

    fn insert_child(&mut self, parent: & Node<T>, sibling_index: Index) {
        if let Some(child_index) = parent.child {
            if let Some(mut child) = self.get(child_index) {
                // let mut fuel = 0;
                while let Some(sibling) = child.right {
                    // println!("{:?} {:?}", child.index, sibling);
                    // fuel += 1;
                    // if fuel > 10 {
                    //     println!("Fuel! {:?}", child);
                    //     break;
                    // }
                    if let Some(node) = self.get(sibling) {
                        child = node;
                    } else {
                        panic!("Some sibling doesn't exist");
                    }
                    
                }
                if let Some(node) = self.arena.get_mut(child_index) {
                    node.right = Some(sibling_index);
                }
            }
        }
    }

    // What if parent doesn't exist?
    fn insert(&mut self, t: T, parent_index: Index) -> Option<Index> {
        let index = self.next_index();
        let mut parent = self.arena.get_mut(parent_index)?;
        if parent.child.is_none() {
            parent.child = Some(index);
        } else {
            let parent = parent.clone();
            self.insert_child(& parent, index);
        }

        self.arena.push(Node {
            index,
            val: t,
            parent: Some(parent_index),
            child: None,
            left: None,
            right: None,
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
            if let Some(child_index) = node.child {
                if let Some(i) = self.copy_tree_helper(sub_index, child_index, new_index, forest) {
                    sub_index = i;
                }
                if let Some(mut child) = self.get(child_index) {
                    while let Some(sibling_index) = child.right {
                        if let Some(i) = self.copy_tree_helper(sub_index, sibling_index, new_index, forest) {
                            sub_index = i;
                        }
                        if let Some(sibling) = self.get(child_index) {
                            child = sibling;
                        }
                    }
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

    fn print_tree_inner(&self, node: &Node<T>, prefix: String, last: bool) {
        let current_prefix = if last { "`- " } else { "|- " };
        println!("{}{}{:?} {}", prefix, current_prefix, node.val, node.exhausted);

        let child_prefix = if last { "   " } else { "|  " };
        let prefix = prefix + child_prefix;

        // Should consider making a iter method for children
        if node.child.is_some() {

            if let Some(mut child) = self.get(node.child.unwrap()) {
                while child.right.is_some() {
                    self.print_tree_inner(&child, prefix.to_string(), false);
                    if let Some(sibling) = self.get(child.right.unwrap()) {
                        child = sibling
                    } else {
                        panic!(format!("We have a node with a right that doesn't exist {:?}", node));
                    } 
                }
                self.print_tree_inner(&child, prefix.to_string(), true);
            }
        }
    }

    // https://vallentin.dev/2019/05/14/pretty-print-tree
    fn print_tree(&self, index: Index) {
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
struct RootedForest<T> where T : Clone {
    root: Index,
    focus: Index,
    forest: Forest<T>,
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

    fn get_focus(&self) -> Option<&Node<T>> {
        self.forest.get(self.focus)
    }
    
    fn get(&self, index: Index) -> Option<&Node<T>> {
        self.forest.get(index)
    }

    // I could cache this?
    fn root_is_exhausted(&self) -> bool {
        let root = self.get_root();
        root.is_none() || root.unwrap().exhausted
    }

    fn focus_is_exhausted(&self) -> bool {
        let focus = self.get_focus();
        focus.is_none() || focus.unwrap().exhausted
    }

    fn get_focus_parent(&self) -> Option<Index> {
        self.get_focus().and_then(|x| x.parent)
    }

    fn move_focus_if_exhausted(&mut self) {
    }

    /// Should move to the next expr that needs evaluation
    fn move_to_next_expr(&mut self) {
        let mut fuel = 0;
        loop {
            fuel += 1;
            if fuel > 10 {
                return;
            }
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
                if focus.child.is_none() {
                    return
                }

                // This isn't right yet. We aren skipping some nodes.
                let mut all_children_exhausted = true;
                if let Some(child_index) = focus.child {
                    if let Some(mut child) = self.get(child_index) {
                        while child.exhausted {
                            if let Some(right_index) = child.right {
                                if let Some(sibling) = self.get(right_index) {
                                    child = sibling;
                                }
                            } else {
                                break;
                            }
                        }
                        if !child.exhausted {
                            self.focus = child.index;
                            all_children_exhausted = false;
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
struct Program {
    // Can we rewrite meta? We can match on meta and rewrite else where.
    // But what would it mean to rewrite meta?
    meta: Meta,
    // Technically these chould just be in the hashmap.
    // But it seems worthwhile to bring them to the top level.
    main: RootedForest<Expr>,
    io: RootedForest<Expr>,
    // We will need some structure for the preprocessed rules.
    // Like keeping a list of clauses by scope and type.
    rules: RootedForest<Rule>,
    symbols: Interner,
    scopes: HashMap<Index, RootedForest<Expr>>,
}


impl Program {

    fn new() -> Program {
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

    fn full_step(&mut self) {
        let mut fuel = 0;
        while !self.main.root_is_exhausted() {
            fuel +=1;
            if fuel > 30 {
                println!("break");
                break;
            }
            self.step();
        }
    }
}





pub fn run_new() {

    let mut program = Program::new();
    let f = &mut program.main.forest;
    let now = Instant::now();
    let depth = 30;
    let width = 30;
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
    println!("{}", now.elapsed().as_millis());


    // if let Some((focus, root)) = program.main.forest.persistent_change(Expr::Symbol(2), n4.unwrap()) {
    //     let result = program.main.forest.garbage_collect(root, focus);
    //     println!("{}: {:?}", root, result);
    // }
    let now = Instant::now();
    // println!("{:#?}", program.main.forest.arena);
    // program.main.forest.print_tree(program.main.root);
    program.full_step();
    let x : Vec<& Node<Expr>> = program.main.forest.arena.iter().filter(|x| x.exhausted == false).collect();
    println!("not exhausted {}", x.len());
    println!("{}", now.elapsed().as_millis());
    println!("{} {}", program.main.root, program.main.forest.arena.len());
    program.main.forest.print_tree(program.main.root);


    // Silly litle trick that does speed things up.
    // std::thread::spawn(move || drop(program));

}

fn main() {}

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