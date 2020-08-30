use std::fmt::Debug;
use std::collections::HashMap;
use std::collections::VecDeque;

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


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Call,
    Symbol(Index),
    Scope(Index),
    LogicVariable(Index),
    Num(isize),
    Do,
    Array,
    Map, // How do I do this one? I guess its children are just even?
    Quote,
}


#[derive(Debug, Clone)]
pub struct Meta<T> where T : Debug + Clone {
    original_expr: T,
    original_sub_expr: T,
    new_expr: T,
    new_sub_expr: T,
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

    pub fn intern(&mut self, symbol : &str) -> Index {
        if let Some(index) = self.lookup.get(symbol) {
            return *index
        }
        let index = self.storage.len();
        self.lookup.insert(symbol.to_string(), index);
        self.storage.push(symbol.to_string());
        index
    }

    pub fn lookup(&self, index: Index) -> Option<&String> {
        self.storage.get(index)
    }

    pub fn get_index(&self, symbol : &str) -> Option<&Index> {
        self.lookup.get(symbol)
    }
}


#[derive(Debug, Clone)]
pub struct Forest<T> where T : Clone {
    pub root: Index,
    pub focus: Index,
    pub arena: Vec<Node<T>>,
    current_index: usize,
}

// Probably going to need to extend this to have root and focus
pub trait ForestLike<T> where T : Clone + Debug {
    fn get_children(&self, index: Index) -> Option<&Vec<Index>>;
    fn get(&self, index: Index) -> Option<&Node<T>>;
    fn get_focus_node(&self) -> Option<&Node<T>>;
    fn get_focus(&self) -> Index;
    fn get_root(&self) -> Index;
    
    fn print_tree_inner<F>(&self, node: &Node<T>, prefix: String, last: bool, formatter: &F) where F : Fn(&T) -> String {
        let current_prefix = if last { "`- " } else { "|- " };
        println!("{}{}{} {} {}", prefix, current_prefix, formatter(&node.val), node.index, node.exhausted);

        let child_prefix = if last { "   " } else { "|  " };
        let prefix = prefix + child_prefix;
        let children = self.get_children(node.index).unwrap();
        if !children.is_empty() {
            let last_child = children.len() - 1;

            for (i, child_index) in children.iter().enumerate() {
                if let Some(child) = self.get(*child_index) {
                    self.print_tree_inner(&child, prefix.to_string(), i == last_child, formatter);
                }
            }
        }
    }

    // https://vallentin.dev/2019/05/14/pretty-print-tree
    fn print_tree<F>(&self, index: Index, formatter: F) where F : Fn(&T) -> String {
        if let Some(node) = self.get(index) {
            self.print_tree_inner(node, "".to_string(), true, &formatter)
        }
    }

    fn copy_tree_helper(&self, mut focus_index: Index, from_index: Index, into_parent_index: Option<Index>, forest: &mut Forest<T>) -> Option<Index> {
        if let Some(node) = self.get(from_index) {
            let new_index = if let Some(parent_index) = into_parent_index {
                forest.insert(node.val.clone(), parent_index, node.exhausted)
            } else {
                Some(forest.insert_root(node.val.clone(), node.exhausted))
            };
            if from_index == focus_index {
                if let Some(i) = new_index {
                    focus_index = i;
                }
            }
            for child_index in self.get_children(node.index).unwrap() {
                if let Some(i) = self.copy_tree_helper(focus_index, *child_index, new_index, forest) {
                    focus_index = i;
                }
            }
            Some(focus_index)
        } else {
            None
        }
    }
    
}
// Maybe this should string build instead of print?
fn print_expr_inner(forest : &impl ForestLike<Expr>, node: &Node<Expr>, formatter: &impl FormatExpr) {
    match &node.val {
        Expr::Call => {
            let children = forest.get_children(node.index).unwrap();
            let mut is_first = true;
            let last_child = children.len().saturating_sub(1);
            for (i, child_index) in children.iter().enumerate() {
                print_expr_inner(forest, forest.get(*child_index).unwrap(), formatter);
                if is_first {
                    print!("(");
                    is_first = false;
                } else if i == last_child {
                    print!(")");
                } else {
                    print!(", ");
                }
            }
        }
        Expr::Array => {
            let children = forest.get_children(node.index).unwrap();
            let last_child = children.len() - 1;
            print!("[");
            for (i, child_index) in children.iter().enumerate() {
                print_expr_inner(forest, forest.get(*child_index).unwrap(), formatter);
                 if i == last_child {
                    print!("]");
                } else {
                    print!(", ");
                }
            }
            print!("]");
        }
        Expr::Map => {
            let children = forest.get_children(node.index).unwrap();
            let last_child = children.len() - 1;
            print!("{{");
            for (i, child_index) in children.iter().enumerate() {
                print_expr_inner(forest, forest.get(*child_index).unwrap(), formatter);
                if i % 2 == 0 {
                    print!(": ");
                } else if i != last_child {
                    print!(", ");
                }
            }
            print!("}}");
        }
        Expr::Quote => {
            print!("'");
            let children = forest.get_children(node.index).unwrap();
            for (i, child_index) in children.iter().enumerate() {
                print_expr_inner(forest, forest.get(*child_index).unwrap(), formatter);
            }
        }
        x => {
            print!("{}", formatter.format_expr(&x));
            let children = forest.get_children(node.index).unwrap();
            for (i, child_index) in children.iter().enumerate() {
                print_expr_inner(forest, forest.get(*child_index).unwrap(), formatter);
            }
        }
    }
}

pub fn print_expr(forest : &impl ForestLike<Expr>, index: Index, formatter: &impl FormatExpr) {
    if let Some(node) = forest.get(index) {
        print_expr_inner(forest, node, formatter);

        println!("");
    }
}


impl<T> ForestLike<T> for Forest<T> where T : Clone + Debug {
    fn get_children(&self, index: Index) -> Option<&Vec<Index>> {
        self.get(index).map(|x | &x.children)
    }

    fn get(&self, index: Index) -> Option<&Node<T>> {
        self.arena.get(index)
    }

    fn get_focus_node(&self) -> Option<&Node<T>> {
        self.get(self.focus)
    }

    fn get_focus(&self) -> Index {
        self.focus
    }

    fn get_root(&self) -> Index {
        self.root
    }
}

#[derive(Debug, Clone)]
pub struct MetaForest<'a, T> where T : Clone + Debug {
    pub forest: &'a Forest<T>,
    pub meta: Meta<Index>,
    pub meta_parents: Meta<Option<Index>>,
    pub meta_index_start: Index,
    pub meta_nodes: Forest<T>,
}


impl<'a> MetaForest<'a, Expr> {

    fn new(meta: Meta<Index>, forest: &'a Forest<Expr>, symbols: & Interner) -> MetaForest<'a, Expr> {
        let meta_index_start = forest.arena.len();
        let mut meta_nodes = Forest::new();
        meta_nodes.current_index = meta_index_start;
        let mut meta_forest: MetaForest<'a, Expr> = MetaForest {
            meta,
            forest,
            meta_parents: Meta { original_expr: None, original_sub_expr: None, new_expr: None, new_sub_expr: None},
            meta_index_start,
            meta_nodes: meta_nodes,
        };

        meta_forest.setup(symbols);
        meta_forest
    }

    pub fn setup(&mut self, symbols: & Interner) {

        let original_expr = self.forest.get(self.meta.original_expr);
        let original_sub_expr =self.forest.get(self.meta.original_sub_expr);
        let new_expr = self.forest.get(self.meta.new_expr);
        let new_sub_expr =  self.forest.get(self.meta.new_sub_expr);
        self.meta_parents = Meta {
            original_expr: original_expr.and_then(|x| x.parent),
            original_sub_expr: original_sub_expr.and_then(|x| x.parent),
            new_expr: new_expr.and_then(|x| x.parent),
            new_sub_expr: new_expr.and_then(|x| x.parent),
        };
        // I'm really not sure about all of this.
        // It definitely doesn't feel right.
        let location = self.meta_nodes.insert_root_val(Expr::Map);
        // I've messed with the index here. Need to fix it back to 0.
        self.meta_nodes.root = 0;
        self.meta_nodes.focus = 0;
        self.meta_nodes.insert_child(Expr::Symbol(*symbols.get_index("original_expr").unwrap()));
        let root = self.meta_nodes.arena.get_mut(self.meta_nodes.root ).unwrap();
        root.children.push(self.meta_index_start + 2);
        self.meta_nodes.insert_node(original_expr.unwrap().clone());
        self.meta_nodes.insert_child(Expr::Symbol(*symbols.get_index("original_sub_expr").unwrap()));
        let root = self.meta_nodes.arena.get_mut(self.meta_nodes.root).unwrap();
        root.children.push(self.meta_index_start + 4);
        self.meta_nodes.insert_node(original_sub_expr.unwrap().clone());
        self.meta_nodes.insert_child(Expr::Symbol(*symbols.get_index("new_expr").unwrap()));
        let root = self.meta_nodes.arena.get_mut(self.meta_nodes.root).unwrap();
        root.children.push(self.meta_index_start + 6);
        self.meta_nodes.insert_node(new_expr.unwrap().clone());
        self.meta_nodes.insert_child(Expr::Symbol(*symbols.get_index("new_sub_expr").unwrap()));
        let root = self.meta_nodes.arena.get_mut(self.meta_nodes.root).unwrap();
        root.children.push(self.meta_index_start + 8);
        self.meta_nodes.insert_node(new_sub_expr.unwrap().clone());
        // let new_children = self.meta_nodes.get_root().unwrap().children.iter().map(|i| i + self.meta_index_start).collect();
        // let root = self.meta_nodes.arena.get_mut(self.meta_nodes.root).unwrap();
        // root.children = new_children;
    }
}


// What does it mean to mutate meta? Does it mean mutating the actual scope?
// If I get to the point of mutating meta, I could track the offset of the meta
// information and if you are past the meta_index_start subtract the length
// of meta to find the underlying data.


// Need to build up all these structures so they give coherent answers
impl<'a> ForestLike<Expr> for MetaForest<'a, Expr> {
    fn get_children(&self, index: Index) -> Option<&Vec<Index>> {
        if Some(index) == self.meta_parents.original_expr {
            self.meta_nodes.get(2).map(|x| &x.children)
        } else if Some(index) == self.meta_parents.original_sub_expr {
            self.meta_nodes.get(4).map(|x| &x.children)
        } else if Some(index) == self.meta_parents.new_expr {
            self.meta_nodes.get(6).map(|x| &x.children)
        } else if Some(index) == self.meta_parents.new_sub_expr {
            self.meta_nodes.get(8).map(|x| &x.children)
        } else if index >= self.meta_index_start {
            let offset = index - self.meta_index_start;
            if offset <= 8 {
                self.meta_nodes.get(offset).map(|x | &x.children)
            } else {
                panic!("Asking for meta_children that is too big");
            }
        } else {
            self.forest.get(index).map(|x | &x.children)
        }
    }

    fn get(&self, index: Index) -> Option<&Node<Expr>> {
        if index >= self.meta_index_start {
            let offset = index - self.meta_index_start;
            if offset <= 8 {
                self.meta_nodes.get(offset)  
            } else {
                println!("{:?}, {:?}", self.meta_index_start, index);
                panic!("Asking for meta that is too big");
            }
        } else if index == self.meta.new_expr {
            self.forest.get(self.meta.original_expr)
        } else if index == self.meta.new_sub_expr {
            self.forest.get(self.meta.original_sub_expr)
        } else {
            self.forest.get(index)
        }
    }

    fn get_focus_node(&self) -> Option<&Node<Expr>> {
        self.get(self.meta_index_start)
    }

    fn get_focus(&self) -> Index {
        self.meta_index_start
    }

    fn get_root(&self) -> Index {
        self.meta_index_start
    }
}


impl<T> Forest<T> where T : Clone + Debug {

    fn new() -> Forest<T> {
        Forest {
            root: 0,
            focus: 0,
            arena: vec![],
            current_index: 0,
        }
    }


    fn insert_root(&mut self, t: T, exhausted: bool) -> Index {
        let index = self.next_index();
        let n = Node {
            index,
            child_index: None,
            val: t,
            parent: None,
            children: vec![],
            exhausted,
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
    fn insert(&mut self, t: T, parent: Index, exhausted: bool) -> Option<Index> {
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
            exhausted,
        });
        Some(index)
    }

    fn insert_node(&mut self, mut node: Node<T>) -> Index {
        let index = self.next_index();
        node.index = index;
        self.arena.push(node);
        index
    }

    // focus_index is here so you can keep your focus after the tree is copied.
    fn copy_tree_helper(&self, mut focus_index: Index, from_index: Index, into_parent_index: Option<Index>, forest: &mut Forest<T>) -> Option<Index> {
        if let Some(node) = self.get(from_index) {
            let new_index = if let Some(parent_index) = into_parent_index {
                forest.insert(node.val.clone(), parent_index, node.exhausted)
            } else {
                forest.insert_root(node.val.clone(), node.exhausted);
                Some(0)
            };
            if from_index == focus_index {
                if let Some(i) = new_index {
                    focus_index = i;
                }
            }
            for child_index in node.children.clone() {
                if let Some(i) = self.copy_tree_helper(focus_index, child_index, new_index, forest) {
                    focus_index = i;
                }
            }
            Some(focus_index)
        } else {
            None
        }
    }

    fn copy_tree_helper_f<F>(&self, from_index: Index, current_index: Index, parent_index: Option<Index>, forest: &mut Forest<T>, index_f : & F)
    where F : Fn(&T) -> Option<Index> {
        if let Some(node) = self.get(current_index) {
            let node = if let Some(new_index) = index_f(&node.val) {
                // println!("{:?} {:?} {:?}", new_index, current_index, forest);
                forest.get(new_index).unwrap().clone()
            } else {
                node.clone()
            };


            let new_parent_index = if from_index == current_index {
                // I don't understand this. I thought I did.
                // The idea here is that the first call shouldn't do anything.
                // I thought I could just set this to parent_index,
                // but that doesn't work even though when this function is called,
                // the value of parent index is forest.focus
                // This might only work in some weird case and actually fail otherwise.
                Some(forest.focus)
            } else if parent_index.is_none() {
                // println!("No parent {:?}", node.val);
                let new_root = forest.insert_root(node.val, node.exhausted);
                forest.root = new_root;
                forest.focus = new_root;
                Some(new_root)
            } else {
                // println!("Parent {:?}", node.val);
                forest.insert(node.val, parent_index.unwrap(), node.exhausted)
            };

            for child_index in node.children.clone() {
                self.copy_tree_helper_f(from_index, child_index, new_parent_index, forest, index_f);
            }
        }
    }

    // Might need to do this for a list of indexes?
    // sub_index is basically focus. I maybe don't need it?
    fn garbage_collect(&mut self, index: Index, sub_index: Index) -> Option<Index> {
        let mut forest = Forest::new();

        let mut result_index = None;
        if let Some(node) = self.get(index) {
            result_index = self.copy_tree_helper(sub_index, index, None, &mut forest);
        }
        *self = forest;
        result_index

    }


    // Returns old nodes new location
    fn persistent_change(&mut self, t : T, index: Index) -> Option<Index> {
        let node = self.arena.get_mut(index)?;
        let node_clone = node.clone();
        node.val = t;
        Some(self.insert_node(node_clone))
    }

    fn clear_children(&mut self, index: Index) {
        if let Some(node) = self.arena.get_mut(index) {
            node.children.clear();
        }
    }


    fn exhaust_focus(&mut self) {
        if let Some(node) = self.arena.get_mut(self.focus) {
            node.exhausted = true
        }
    }



    pub fn insert_child(&mut self, t : T) -> Option<Index> {
        // I could exhaust here if I did this
        //  from program and new things about rules.
        self.insert(t, self.focus, false)
    }

    pub fn make_last_child_focus(&mut self) {
        let mut new_focus = None;
        if let Some(node) = self.get_focus_node() {
            if let Some(index) = node.children.last() {
                new_focus = Some(*index);
            }
        }
        if let Some(new_focus) = new_focus {
            self.move_focus(new_focus);
        }
    }

    pub fn make_parent_focus(&mut self) {
        if let Some(parent) = self.get_focus_parent() {
            self.focus = parent;
        }
    }

    pub fn get_last_inserted_val(&self) -> Option<&T> {
        let node = self.arena.get(self.focus)?;
        let index = if node.children.len() > 0 {
            *node.children.last().unwrap()
        } else {
            self.focus
        };
        self.get(index).map(|x| &x.val)
    }

    pub fn make_last_inserted_focus(&mut self) {
        if let Some(node) = self.arena.get(self.focus) {
            let index = if node.children.len() > 0 {
                *node.children.last().unwrap()
            } else {
                self.focus
            };
            self.focus = index;
        }
    }

    pub fn swap_and_insert(&mut self, t: T) {
        if let Some(node) = self.arena.get(self.focus) {
            let index = if node.children.len() > 0 {
                *node.children.last().unwrap()
            } else {
                self.focus  
            };
            let mut node = self.arena.get_mut(index).unwrap();
            let node_value = node.val.clone();
            node.val = t;
            self.focus = index;
            self.insert_child(node_value);
        }
    }

    pub fn insert_root_val(&mut self, t: T) {
        let root = self.insert_root(t, false);
        self.root = root;
        self.focus = root;
    }

    // I should probably cache this root?
    // I can easily do this if changing root
    // goes through some method.
    fn get_root(&self) -> Option<&Node<T>> {
        self.get(self.root)
    }


    pub fn get_focus_val(&self) -> Option<&T> {
        self.get(self.focus).map(|x| &x.val)
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
        let focus = self.get_focus_node();
        focus.is_none() || focus.unwrap().exhausted
    }

    pub fn get_focus_parent(&self) -> Option<Index> {
        self.get_focus_node().and_then(|x| x.parent)
    }

    pub fn move_focus(&mut self, focus: Index) {
        self.focus = focus;
    }

    pub fn garbage_collect_self(&mut self) {
        if let Some(new_focus) = self.garbage_collect(self.root, self.focus) {
            self.root = 0;
            self.focus = new_focus;
        }

    }
}


impl Forest<Expr> {

    fn get_child_nums_binary(&self) -> Option<(Index, isize, isize)> {

        let focus = self.get_focus_node()?;
        if focus.children.len() != 3 {
            return None;
        }
        let node = self.get(*focus.children.get(0)?)?;
        match node {
            Node{val: Expr::Symbol(i), ..} => {
                let second_child = self.get(*focus.children.get(1)?)?;
                let third_child = self.get(*focus.children.get(2)?)?;
                if let (Node{val: Expr::Num(x), ..}, Node{val: Expr::Num(y), ..}) = (second_child, third_child) {
                    return Some((*i, *x, *y));
                }
                return None;
            }
            _ => return None
        }
    }

    pub fn focus_is_quote(&self) -> bool {
        let focus = self.get_focus_node();
        focus.is_none() || focus.unwrap().val == Expr::Quote
    }

    pub fn pretty_print_tree(&self) {

        self.print_tree(self.root, |expr| {
            match expr {
                _ => format!("{:?}", expr)
            }
        })
    }

       /// Should move to the next expr that needs evaluation
       fn move_to_next_reducible_expr(&mut self) {
        // let mut fuel = 0;
        loop {
            // fuel += 1;
            // if fuel > 100 {
            //     break;
            // }
            if self.root_is_exhausted() {
                return
            }
            // The addition of quote here makes it so that everything below
            // a quote is not exhausted. Is that the correct behavior? Not sure.
            if self.focus_is_exhausted() || self.focus_is_quote() {
                if let Some(index) = self.get_focus_parent() {
                    // println!("Moving to parent");
                    self.focus = index;
                    continue;
                } else {
                    return
                }
            }
            if let Some(focus) = self.get_focus_node() {
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
pub struct Clause {
    pub in_scopes: Vec<Index>,
    pub out_scopes: Vec<Index>,
    pub left: Index,
    pub right: Index,
}
#[derive(Debug, Clone)]
pub struct MatchingClause {
    pub out_scope: Index,
    pub rule_index: Index,
    pub environment: HashMap<Index, Index>,
}

#[derive(Debug, Clone)]
pub struct Program {
    // Can we rewrite meta? We can match on meta and rewrite else where.
    // But what would it mean to rewrite meta?
    pub meta: Meta<Index>,
    // Technically these chould just be in the hashmap.
    // But it seems worthwhile to bring them to the top level.
    pub main: Forest<Expr>,
    pub io: Forest<Expr>,
    // We will need some structure for the preprocessed rules.
    // Like keeping a list of clauses by scope and type.
    pub rules: Forest<Expr>,
    pub symbols: Interner,
    pub scopes: HashMap<Index, Forest<Expr>>,
    pub clause_indexes: Vec<Clause>,
    pub main_scope_index: Index,
    pub meta_scope_index: Index,
    pub rules_scope_index: Index,
    pub io_scope_index: Index,
}


pub trait FormatExpr {
    fn format_expr(&self, expr: &Expr) -> String {
        format!("{:?}", expr)
    }
}

impl FormatExpr for Interner {
    fn format_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Symbol(index) | Expr::LogicVariable(index) | Expr::Scope(index) => {
                let value = self.lookup(*index).unwrap().clone();
                if value.len() == 1 && !value.chars().next().unwrap().is_alphanumeric() {
                    format!("({})", value)
                } else {
                    value
                }
            }
            _ => format!("{:?}", expr)
        }
    }
}


// These macros exist to appease the borrow checker. Can't extract them
// out into functions, but as macros the borrow check understands what is happening.
macro_rules! get_scope_mut_for_index {
    ($program:expr, $scope_index:expr) => {
        if $scope_index == $program.main_scope_index {
            &mut $program.main
        } else if $scope_index == $program.io_scope_index {
            &mut $program.io
        }  else if $scope_index == $program.rules_scope_index {
            &mut $program.rules
        } else if $program.scopes.contains_key(&$scope_index) {
            $program.scopes.get_mut(&$scope_index).unwrap()
        } else {
            panic!("Scope does not exist");
        }
    };
}

macro_rules! get_scope_mut_for_index_no_rules {
    ($program:expr, $scope_index:expr) => {
        if $scope_index == $program.main_scope_index {
            &mut $program.main
        } else if $scope_index == $program.io_scope_index {
            &mut $program.io
        } else if $program.scopes.contains_key(&$scope_index) {
            $program.scopes.get_mut(&$scope_index).unwrap()
        } else {
            panic!("Scope does not exist");
        }
    };
}

macro_rules! get_scope_pairs {
    ($program:expr, $scope_index:expr, $effect_index:expr) => {
        if $scope_index == $program.main_scope_index && $effect_index == $program.io_scope_index {
            (&$program.main, &mut $program.io)
        } else if $scope_index == $program.io_scope_index && $effect_index == $program.main_scope_index {
            (&$program.io, &mut $program.main)
        } else {
            panic!("Pair scopes incomplete. Also, have no idea how to extend it any not anger the borrow checker.");
        };
    };
}


impl Program {
    

    pub fn new() -> Program {

        let mut symbols = Interner::new();
        symbols.intern("builtin/+");
        symbols.intern("builtin/-");
        symbols.intern("builtin/*");
        symbols.intern("builtin/div");
        symbols.intern("fact");
        let main_scope_index = symbols.intern("@main");
        let io_scope_index = symbols.intern("@io");
        let rules_scope_index = symbols.intern("@rules");
        let meta_scope_index = symbols.intern("@meta");
        symbols.intern("original_expr");
        symbols.intern("original_sub_expr");
        symbols.intern("new_expr");
        symbols.intern("new_sub_expr");
        let meta = Meta {
            original_expr: 0,
            original_sub_expr: 0,
            new_expr: 0,
            new_sub_expr: 0,
        };
        let mut program = Program {
            meta: meta.clone(),
            main: Forest::new(),
            io: Forest::new(),
            rules: Forest::new(),
            symbols: symbols,
            scopes: HashMap::new(),
            clause_indexes: Vec::new(),
            main_scope_index,
            meta_scope_index,
            io_scope_index,
            rules_scope_index,
        };

        // Hack to make it so there is always a root.
        program.main.insert_root_val(Expr::Num(0));
        program.io.insert_root_val(Expr::Num(0));
        program.rules.insert_root_val(Expr::Num(0));
        program
        
    }

    fn construct_scopes(&self, scope_attribute_index : Index) -> Option<Vec<Index> >{
        let array = self.rules.get(scope_attribute_index)?;
        let mut results = Vec::with_capacity(array.children.len());
        for index in &array.children {
            if let Expr::Scope(i) = self.rules.get(*index)?.val {
                results.push(i)
            }
        }

        Some(results)
    }

    // So as I add rules I could totally index incrementally by keeping the last index
    // of how far I indexed and then only indexing more from there.
    pub fn set_clause_indexes(&mut self) -> Option<()> {
        // This is returning option of void because I'm tired of not being able to use ?.
        let in_scope_symbol = Expr::Symbol(*self.symbols.get_index("in_scopes")?);
        let out_scope_symbol = Expr::Symbol(*self.symbols.get_index("out_scopes")?);
        let clauses_symbol = Expr::Symbol(*self.symbols.get_index("clauses")?);

        // The rules are quoted so it is two levels deep.
        // Need to make better things for traversing.
        let rules = &self.rules.get(*self.rules.get_root()?.children.get(0)?)?.children;

        let mut clauses : Vec<Clause> = vec![];
        for rule_index in rules {
            let rule = self.rules.get(*rule_index)?;
            let mut in_scope_index = None;
            let mut out_scope_index = None;
            // This is super ugly. But it kind of makes sense?
            let mut next_is_clauses = false;
            for attribute_index in &rule.children {
                let node = self.rules.get(*attribute_index)?;
                // Right now I'm kind of assuming these come before clauses
                if node.val == in_scope_symbol {
                    in_scope_index = Some(attribute_index + 1);
                };
                if node.val == out_scope_symbol {
                    out_scope_index = Some(attribute_index + 1);
                };
                if next_is_clauses {
                    next_is_clauses = false;
                    for clause_index in &node.children {
                        let clause = self.rules.get(*clause_index)?;
                        clauses.push(Clause{
                            left: *clause.children.get(1)?,
                            right: *clause.children.get(3)?,
                            // need to construct these from the indexes
                            in_scopes: self.construct_scopes(in_scope_index?)?,
                            out_scopes: self.construct_scopes(out_scope_index?)?,
                        });
                    }
                }
                if node.val == clauses_symbol {
                    next_is_clauses = true;
                }

            }
        }
        self.clause_indexes = clauses;

        Some(())
    }

    // I want to minimize allocation here, so I might look at having
    // some object that will give me new environments and keep them around
    // so I don't have to allocate and deallocate a new one everytime.
    // Also will allocate the queue at a higher level instead of each function call.
    pub fn build_env(&self, scope: &impl ForestLike<Expr>, left_hand_index : Index, expr_index : Index) -> Option<HashMap<Index, Index>> {
        let mut env = HashMap::new();
        let mut queue: VecDeque<(Index, Index)> = VecDeque::new();
        queue.push_front((left_hand_index, expr_index));
        let mut failed = false;
        while !queue.is_empty() && !failed {
            let (left_hand_index, expr_index) = queue.pop_front().unwrap();
            let elems = (self.rules.get(left_hand_index)?, scope.get(expr_index)?);
            match elems {
                (Node{ val: Expr::LogicVariable(l_index), ..}, _) => {
                    env.insert(*l_index, expr_index);
                }

                (Node {val: Expr::Map, index: l_index, ..}, Node{ val: Expr::Map, index: e_index, ..}) => {
                    let l_children = self.rules.get_children(*l_index)?;
                    let e_children = scope.get_children(*e_index)?;
                    for i in (0..l_children.len()).step_by(2) {
                        for j in (0..e_children.len()).step_by(2) {

                            let i_index = *l_children.get(i)?;
                            let j_index = *e_children.get(j)?;
                            // This is a suboptimal way of doing these things.
                            // TODO: Make better
                            if self.rules.get(i_index)?.val == scope.get(j_index)?.val {
                                // println!("{:?}, {:?}", self.rules.get(i_index)?.val, scope.get(j_index)?.val);
                                queue.push_front((*l_children.get(i+1)?, *e_children.get(j+1)?));
                            }
                        }
                    }
                },
                (Node{ val: l_val, index: l_index, ..}, Node{ val: e_val, index: e_index, ..}) => {
                    let l_children = self.rules.get_children(*l_index)?;
                    let e_children = scope.get_children(*e_index)?;
                    if l_val != e_val {
                        failed = true;
                        break;
                    }
                    // Children length will have to change once repeats exist.
                    // Children length is also wrong for maps.
                    // In general, I need to handle maps differently.
                    // Really this representation is just wrong for maps,
                    // but I will probably ignore that for now.
                    if l_children.len() != e_children.len() {
                        failed = true;
                        break;
                    }
                    for i in 0..l_children.len() {
                        queue.push_front((*l_children.get(i)?, *e_children.get(i)?))
                    }
                }
            }

        };

        if failed {
            None
        } else {
            Some(env)
        }
    }

    pub fn pretty_print_scope(&self, scope : &impl ForestLike<Expr>) {
        // Need to refactor to use some non closure formatter like I did with FormatExpr
        let index = scope.get_focus();
        scope.print_tree(index, |expr| {
            match expr {
                Expr::Symbol(index) | Expr::LogicVariable(index) | Expr::Scope(index) => {
                    let value = self.symbols.lookup(*index).unwrap().clone();
                    if value.len() == 1 && !value.chars().next().unwrap().is_alphanumeric() {
                        format!("({})", value)
                    } else {
                        value
                    }
                }
                _ => format!("{:?}", expr)
            }
        })
    }

    pub fn pretty_print_main(&self) {
        self.pretty_print_scope(&self.main)
    }

    pub fn substitute(scope: &mut Forest<Expr>, rule_scope: &Forest<Expr>, right_index: Index, env: &HashMap<Index, Index>) -> Option<Index> {
        let right = rule_scope.get(right_index).unwrap().clone();
        let right_replace = match right.val {
            Expr::LogicVariable(index) => {
                scope.get(*env.get(&index).unwrap()).unwrap().clone().val
            }
            val => val
        };
        let node = scope.get_focus_node().unwrap();
        let parent = node.parent.clone();

        let result = scope.persistent_change(right_replace, scope.focus);
        let focus = scope.arena.get_mut(scope.focus).unwrap();
        focus.children.clear();
        rule_scope.copy_tree_helper_f(right_index, right_index, parent, scope, & |val| {
            match val {
                Expr::LogicVariable(index) => {
                    Some(*env.get(&index).unwrap())
                }
                _ => None
            }
        });

        // What about the root? Does it actually matter or just the focus?
        // I do need to know the root for meta evaluation, but root should only
        // change here if the focus is the root. Or at least I think so.
        result
       
    }

    pub fn get_node_for_substitution(right: &Node<Expr>, in_scope: &impl ForestLike<Expr>, env: &HashMap<Index, Index>) -> Option<Node<Expr>> {
        match &right.val {
            Expr::LogicVariable(index) => {
                Some(in_scope.get(*env.get(&index)?)?.clone())
            }
            _ => None
        }
    }


    pub fn transfer_and_substitute(in_scope: &impl ForestLike<Expr>, mut out_scope: &mut Forest<Expr>, rule_scope: &Forest<Expr>, right_index: Index, env: &HashMap<Index, Index>, parent_index: Option<Index>) -> Option<Index> {
        let node = rule_scope.get(right_index)?;
        let new_node = Program::get_node_for_substitution(node, in_scope, &env);

        let is_root = parent_index.is_none();
        // Not sure if this should be root or focus?
        let new_location = if new_node.is_some() {
            let index = new_node?.index;
            let new_location = in_scope.copy_tree_helper(index, index, parent_index, &mut out_scope)?;
            if is_root {
                out_scope.root = new_location;
            }
            new_location
        } else {
            if is_root {
                out_scope.insert_root_val(node.val.clone());
                out_scope.root
            } else {
                out_scope.insert(node.val.clone(), parent_index?, false)?
            }
        };
        for child_index in rule_scope.get_children(right_index)?  {
            Program::transfer_and_substitute(in_scope, out_scope, rule_scope, *child_index, env, Some(new_location));
        }

        // We changed the focus as we were going down the tree,
        // but we need to reset it to our root.
        if is_root {
            out_scope.focus = out_scope.root;
        }

        None

    }

    // Needs to return output scope
    pub fn find_matching_rules(&self, scope_symbol_index: Index, expr_index: Index, scope: &impl ForestLike<Expr>) -> (Option<MatchingClause>, Vec<MatchingClause>) {
        let mut matching_rules = vec![];
        let mut matching_rule = None;
        for Clause{left,right, in_scopes, out_scopes} in &self.clause_indexes {
            if !in_scopes.contains(&scope_symbol_index) { continue };
            let env = self.build_env(scope, *left, expr_index);
            if env.is_none() { continue };

            let clause = MatchingClause {
                // Right now we are assuming a singular out scope.
                // In general, I need to think about rules with multiple scopes.
                out_scope: *out_scopes.first().unwrap(),
                environment: env.unwrap(),
                rule_index: *right
            };
            let out_scope_matches = *out_scopes.first().unwrap() == scope_symbol_index;
            if matching_rule.is_none() && out_scope_matches {
                matching_rule = Some(clause)
            } else if !out_scope_matches {
                matching_rules.push(clause);
            }
            

            // Need to check outscope and do side effects here.
            // The first element of the vector will be our main rule
            // and then the rest will be side effects.
        };

        (matching_rule, matching_rules)
        
    }

    pub fn build_meta_forest(&self, scope_index: Index, meta_original_focus: Index, meta_original_root: Index, scope_root: Index, scope_focus: Index) -> MetaForest<Expr>{
        let meta = Meta {
            original_expr: meta_original_root,
            original_sub_expr: meta_original_focus,
            new_expr: scope_root,
            new_sub_expr: scope_focus,
        };

        let scope = self.get_scope_ref_from_index(scope_index);
        let symbols = &self.symbols;
        let meta_scope_index = self.symbols.get_index("@meta").unwrap();
        let meta_forest = MetaForest::new(meta.clone(), scope, symbols);
        meta_forest
    }

    pub fn handle_builtin(&mut self, scope_index: Index) -> Option<(Index, Index)> {
        let scope = get_scope_mut_for_index!(self, scope_index);
        let node = scope.get_focus_node()?;
        if node.val != Expr::Call { return None };
        let children = scope.get_children(node.index)?;
        let first_child = scope.get(*children.first()?)?;
        if let Expr::Symbol(symbol_index) = first_child.val {
            let symbol_value = self.symbols.lookup(symbol_index)?;
            match symbol_value.as_str() {
                "builtin/println" => {
                    print_expr(scope, *children.get(1)?, &self.symbols);
                    scope.exhaust_focus();
                    // Returning none here might not be the right option.
                    // That means we can't meta on a print statement, which seems wrong.
                    // But I also don't currently rewrite it. Need to revisit.
                    return None
                },
                "builtin/add-rule" => {
                    let rules = &mut self.rules;
                    let outer_quote = *rules.get_children(rules.root)?.first()?;
                    let array = *rules.get_children(outer_quote)?.first()?;
                    // Need to actually add a rule
                    // This means copying some tree into the rules
                    // adding a new child node to the array of rules
                    // and reindexing clauses

                    // Maybe if I created builtin append that would be easier? Maybe?
                    self.set_clause_indexes();

                }
                _ => {
                    if let Some((symbol_index, x, y)) = scope.get_child_nums_binary() {
                        if symbol_index < 4 {
                           // These are implicit right now based on the
                            // order I inserted them in the constructor.
                            let val = match symbol_index {
                                0 => Expr::Num(x + y),
                                1 => Expr::Num(x - y),
                                2 => Expr::Num(x * y),
                                3 => Expr::Num(x / y),
                                _ => panic!("Not possible because of if above.")
                            };
                            let meta_original_focus = scope.persistent_change(val, scope.focus)?;
                            let meta_original_root = if scope.focus == scope.root { meta_original_focus } else { scope.root };
                            scope.clear_children(scope.focus);
                            return Some((meta_original_focus, meta_original_root))
                        }
                    }
                }
            }
        };
        None
        
    }

    pub fn get_scope_ref_from_index(&self, scope_index: Index) -> &Forest<Expr> {
        if scope_index == self.main_scope_index {
            &self.main
        } else if scope_index == self.io_scope_index {
            &self.io
        }  else if scope_index == self.rules_scope_index {
            &self.rules
        } else if self.scopes.contains_key(&scope_index) {
            self.scopes.get(&scope_index).unwrap()
        } else {
            panic!("Scope does not exist");
        }
    }

    pub fn get_scope_index_for_scope_name(&self, scope_name: &str) -> Index {
        *self.symbols.get_index(scope_name).unwrap()
    }

    pub fn rewrite(&mut self, scope_index: Index) -> Option<()> {


        let rules = &self.rules;


        let scope = self.get_scope_ref_from_index(scope_index);
        let original_root = scope.root;
        let original_sub_expr = scope.focus;
        let (matching_rule, original_side_effects) = self.find_matching_rules(scope_index, scope.focus, scope);

       

        // This is one thing I don't like about rust. I have to make these
        // variable or else self is now borrowed both mutabily and immutabily.
        let scope_root = scope.root;
        let scope_focus = scope.focus;

        let meta_info =  if let Some(MatchingClause{environment: env, rule_index: right, out_scope}) = &matching_rule {
            let scope = get_scope_mut_for_index_no_rules!(self, scope_index);
    
            let meta_original_focus = Program::substitute(scope, rules, *right, env)?;
            let meta_original_root = if scope.focus == scope.root { meta_original_focus } else { scope.root };
            Some((meta_original_focus, meta_original_root))
        } else {
            let result = self.handle_builtin(scope_index);
            result
        };

        if let Some((meta_original_focus, meta_original_root)) = meta_info {
            let meta_forest = self.build_meta_forest(
                scope_index,
                meta_original_focus,
                meta_original_root,
                scope_root,
                scope_focus,
            );
            
            let meta_scope_index = self.symbols.get_index("@meta").unwrap();
            let (matching_rule, side_effects) = self.find_matching_rules(*meta_scope_index, meta_forest.meta_index_start, &meta_forest);

            let meta = meta_forest.meta;
            for effect in &side_effects {
                let effect_index = effect.out_scope;

                let symbols = &self.symbols;
                let rules = &self.rules;
                let (scope, out_scope) = get_scope_pairs!(self, scope_index, effect_index);

                let meta_forest = MetaForest::new(meta.clone(), scope, symbols);

                Program::transfer_and_substitute(&meta_forest, out_scope, rules, effect.rule_index, &effect.environment, None);

                self.rewrite(effect_index);

            }
            // println!("{:?}", matching_rule);
        } 

        if let Some((meta_original_focus, meta_original_root)) = meta_info {
            let meta = Meta {
                original_expr: meta_original_root,
                original_sub_expr: meta_original_focus,
                new_expr: scope_root,
                new_sub_expr: scope_focus,
            };
            
            for effect in &original_side_effects {
                let rules = &self.rules;
                let symbols = &self.symbols;
                let (scope, out_scope) = get_scope_pairs!(self, scope_index, effect.out_scope);
                let meta_forest = MetaForest::new(meta.clone(), scope, symbols);
                Program::transfer_and_substitute(&meta_forest, out_scope, rules, effect.rule_index, &effect.environment, None);
                self.rewrite(effect.out_scope);
            }
        }
        
        if meta_info.is_none() {
            let scope = get_scope_mut_for_index!(self, scope_index);
            // No rules matched, so we exhaust
            scope.exhaust_focus();
        }

       
        None
    }

    fn step(&mut self) {
        // println!("step");
        let scope = &mut self.main;
        scope.move_to_next_reducible_expr();
        if scope.root_is_exhausted() {
            return
        }
        // rewrite should return old node location.
        // meta needs to change.
        self.rewrite(self.get_scope_index_for_scope_name("@main"));
    }

    pub fn full_step(&mut self) {
        // let mut fuel = 0;
        // println!("Full step");
        // self.rules.pretty_print_tree();
        // println!("{:?}", self.clause_indexes);
        while !self.main.root_is_exhausted() {
            // fuel +=1;
            // if fuel > 100 {
            //     println!("break");
            //     break;
            // }
            self.step();
            // self.pretty_print_main();

        }
    }
}








// I need to do a precompute for exhaustion.
// Then I could either exhaust as I parse, or exhaust across the whole tree in linear time.
// Can I reduce rules into some form of bytecode?
// Am I that far way from a compiler? Can I eliminate these vectors of children a the nodes?
// Is there a good model of stack computation to be had here?




// Need to create a way to add rules
// Need to make scopes lazily evaluated.
// Need to add an append builtin
// Need to think about how input/repling should work
// Need to think about how scoped temporary rules would work.