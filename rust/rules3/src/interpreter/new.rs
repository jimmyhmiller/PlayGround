use std::time::{Instant};
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
    pub arena: Vec<Node<T>>,
    current_index: usize,
}



impl<T> Forest<T> where T : Clone + Debug {

    fn new() -> Forest<T> {
        Forest {
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

    fn get(&self, index: Index) -> Option<&Node<T>> {
        self.arena.get(index)
    }

    pub fn print_tree_inner<F>(&self, node: &Node<T>, prefix: String, last: bool, formatter: &F) where F : Fn(&T) -> String {
        let current_prefix = if last { "`- " } else { "|- " };
        println!("{}{}{} {} {}", prefix, current_prefix, formatter(&node.val), node.index, node.exhausted);

        let child_prefix = if last { "   " } else { "|  " };
        let prefix = prefix + child_prefix;
        if !node.children.is_empty() {
            let last_child = node.children.len() - 1;

            for (i, child_index) in node.children.iter().enumerate() {
                if let Some(child) = self.get(*child_index) {
                    self.print_tree_inner(&child, prefix.to_string(), i == last_child, formatter);
                }
            }
        }
    }

    // https://vallentin.dev/2019/05/14/pretty-print-tree
    pub fn print_tree<F>(&self, index: Index, formatter: F) where F : Fn(&T) -> String {
        if let Some(node) = self.get(index) {
            self.print_tree_inner(node, "".to_string(), true, &formatter)
        }

    }
    
    fn copy_tree_helper(&self, mut sub_index: Index, node_index: Index, parent_index: Option<Index>, forest: &mut Forest<T>) -> Option<Index> {
        if let Some(node) = self.get(node_index) {
            let new_index = if let Some(parent_index) = parent_index {
                forest.insert(node.val.clone(), parent_index, node.exhausted)
            } else {
                forest.insert_root(node.val.clone(), node.exhausted);
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

    fn copy_tree_helper_f<F>(&self, from_index: Index, current_index: Index, parent_index: Option<Index>, rooted_forest: &mut RootedForest<T>, index_f : & F)
    where F : Fn(&T) -> Option<Index> {
        if let Some(node) = self.get(current_index) {
            let node = if let Some(new_index) = index_f(&node.val) {
                rooted_forest.get(new_index).unwrap().clone()
            } else {
                node.clone()
            };


            let new_parent_index = if from_index == current_index {
                // I don't understand this. I thought I did.
                // The idea here is that the first call shouldn't do anything.
                // I thought I could just set this to parent_index,
                // but that doesn't work even though when this function is called,
                // the value of parent index is rooted_forest.focus
                // This might only work in some weird case and actually fail otherwise.
                Some(rooted_forest.focus)
            } else if parent_index.is_none() {
                // println!("No parent {:?}", node.val);
                let new_root = rooted_forest.forest.insert_root(node.val, node.exhausted);
                rooted_forest.root = new_root;
                rooted_forest.focus = new_root;
                Some(new_root)
            } else {
                // println!("Parent {:?}", node.val);
                rooted_forest.forest.insert(node.val, parent_index.unwrap(), node.exhausted)
            };

            for child_index in node.children.clone() {
                self.copy_tree_helper_f(from_index, child_index, new_parent_index, rooted_forest, index_f);
            }
        }
    }

    // Might need to do this for a list of indexes?
    // sub_index is basically focus. I maybe don't need it?
    fn garbage_collect(&mut self, index: Index, sub_index: Index) -> Option<Index> {
        let mut forest = Forest {
            arena: vec![],
            current_index: 0,
        };

        let mut result_index = None;
        if let Some(node) = self.get(index) {
            result_index = self.copy_tree_helper(sub_index, index, None, &mut forest);
        }
        *self = forest;
        result_index

    }


    // Returns old nodes new location
    fn persistent_change(&mut self, t : T, index: Index) -> Option<Index> {

        // Need to capture location of old node we copied.
        if let Some(node) = self.arena.get_mut(index) {
            node.val = t;
            let node_clone = node.clone();
            Some(self.insert_node(node_clone))
        } else {
            None
        }

    }

    fn clear_children(&mut self, index: Index) {
        if let Some(node) = self.arena.get_mut(index) {
            node.children.clear();
        }
    }

}


pub struct Cursor<'a, T> where T : Clone {
    pub focus: &'a Node<T>,
    pub focus_parent: &'a Node<T>,
    pub child_index: Index,
}
#[derive(Debug, Clone)]
pub struct RootedForest<T> where T : Clone {
    pub root: Index,
    pub focus: Index,
    pub forest: Forest<T>,
}


pub trait ForestLike<T> where T : Clone + Debug {
    fn get_children(&self, index: Index) -> Option<&Vec<Index>>;
    fn get(&self, index: Index) -> Option<&Node<T>>;
}




impl<T> ForestLike<T> for RootedForest<T> where T : Clone + Debug {
    fn get_children(&self, index: Index) -> Option<&Vec<Index>> {
        self.get(index).map(|x | &x.children)
    }

    fn get(&self, index: Index) -> Option<&Node<T>> {
        self.forest.get(index)
    }
}

#[derive(Debug, Clone)]
pub struct MetaForest<T> where T : Clone + Debug {
    pub rooted_forest: RootedForest<T>,
    pub meta: Meta<Index>,
    pub meta_parents: Meta<Index>,
    pub meta_index_start: Index,
    pub meta_nodes: Vec<Node<T>>,
    pub meta_children: Meta<Vec<Index>>,
}

// So what I'm thinking is that the meta is going to go on the end
// I won't actually add it just virtually

// What does it mean to mutate meta? Does it mean mutating the actual scope?
// If I get to the point of mutating meta, I could track the offset of the meta
// information and if you are past the meta_index_start subtract the length
// of meta to find the underlying data.


// Need to build up all these structures so they give coherent answers
impl ForestLike<Expr> for MetaForest<Expr> {
    fn get_children(&self, index: Index) -> Option<&Vec<Index>> {
        if index == self.meta_parents.original_expr {
            Some(&self.meta_children.original_expr)
        } else if index == self.meta_parents.new_expr {
            Some(&self.meta_children.new_expr)
        } else if index == self.meta_parents.original_sub_expr {
            Some(&self.meta_children.original_sub_expr)
        } else if index == self.meta_parents.new_sub_expr {
            Some(&self.meta_children.new_sub_expr)
        } else if index >= self.meta_index_start {
            let offset = index - self.meta_index_start;
            match offset {
                2 => self.get_children(self.meta.original_expr),
                4 => self.get_children(self.meta.original_sub_expr),
                6 => self.get_children(self.meta.new_expr),
                8 => self.get_children(self.meta.new_sub_expr),
                _ => None
            }
        } else {
            self.rooted_forest.get(index).map(|x | &x.children)
        }
    }

    fn get(&self, index: Index) -> Option<&Node<Expr>> {
        if index >= self.meta_index_start {
            let offset = index - self.meta_index_start;
            if offset <= 8 {
                self.meta_nodes.get(offset)  
            } else {
                panic!("Asking for meta that is too big");
            }
        } else {
            self.rooted_forest.forest.get(index)
        }
    }
}


impl<T> RootedForest<T> where T : Clone + Debug {

    fn new() -> RootedForest<T> {
        RootedForest {
            root: 0,
            focus: 0,
            forest: Forest::new(),
        }
    }

    fn exhaust_focus(&mut self) {
        if let Some(node) = self.forest.arena.get_mut(self.focus) {
            node.exhausted = true
        }
    }



    pub fn insert_child(&mut self, t : T) -> Option<Index> {
        // I could exhaust here if I did this
        //  from program and new things about rules.
        self.forest.insert(t, self.focus, false)
    }

    pub fn make_last_child_focus(&mut self) {
        let mut new_focus = None;
        if let Some(node) = self.get_focus() {
            if let Some(index) = node.children.last() {
                new_focus = Some(*index);
            }
        }
        if let Some(new_focus) = new_focus {
            self.move_focus(new_focus);
        }
    }

    pub fn make_last_sibling_focus(&mut self) {
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

    pub fn get_last_inserted_val(&self) -> Option<&T> {
        let node = self.forest.arena.get(self.focus)?;
        let index = if node.children.len() > 0 {
            *node.children.last().unwrap()
        } else {
            self.focus
        };
        self.get(index).map(|x| &x.val)
    }

    pub fn make_last_inserted_focus(&mut self) {
        if let Some(node) = self.forest.arena.get(self.focus) {
            let index = if node.children.len() > 0 {
                *node.children.last().unwrap()
            } else {
                self.focus
            };
            self.focus = index;
        }
    }

    pub fn swap_and_insert(&mut self, t: T) {
        if let Some(node) = self.forest.arena.get(self.focus) {
            let index = if node.children.len() > 0 {
                *node.children.last().unwrap()
            } else {
                self.focus  
            };
            let mut node = self.forest.arena.get_mut(index).unwrap();
            let node_value = node.val.clone();
            node.val = t;
            self.focus = index;
            self.insert_child(node_value);
        }
    }

    pub fn insert_root(&mut self, t: T) {
        let root = self.forest.insert_root(t, false);
        self.root = root;
        self.focus = root;
    }

    // I should probably cache this root?
    // I can easily do this if changing root
    // goes through some method.
    fn get_root(&self) -> Option<&Node<T>> {
        self.forest.get(self.root)
    }

    pub fn get_focus(&self) -> Option<&Node<T>> {
        self.forest.get(self.focus)
    }
    pub fn get_focus_val(&self) -> Option<&T> {
        self.forest.get(self.focus).map(|x| &x.val)
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
    pub fn get_focus_parent_val(&self) -> Option<&T> {
        self.get_focus_parent()
            .and_then(|x| self.get(x))
            .map(|x| &x.val)
    }

    // This might just be a bad idea.
    fn next_expr_top_down<'a>(&'a self, cursor: Cursor<'a, T>) -> Cursor<'a, T> {
        // Need to change this so parent is an Option
        // Also, if I cursor with two things, how do I know they have the same structure?
        // Do I need to encode the step they took as well?
        // Or maybe I shouldn't do this cursory thing but actually just do a recursive
        // step on each thing?
        // I'm really not sure.
        if let Some(child_position) = cursor.focus_parent.children.get(cursor.child_index + 1) {
            let new_focus = self.get(*child_position).unwrap();
            Cursor{
                focus_parent: cursor.focus_parent,
                focus: new_focus,
                child_index: cursor.child_index + 1,
            }
        } else if cursor.focus.children.len() > 0 {
            let new_focus = self.get(*cursor.focus.children.get(0).unwrap()).unwrap();
            Cursor{
                focus_parent: cursor.focus_parent,
                focus: new_focus,
                child_index: 0,
            }
        } else {
         cursor
        }
    }


    pub fn replace_focus_val(&mut self, t : T) {
        let index = self.focus;
        let node = self.forest.arena.get_mut(index);
    }

    pub fn move_focus(&mut self, focus: Index) {
        self.focus = focus;
    }

    pub fn garbage_collect(&mut self) {
        if let Some(new_focus) = self.forest.garbage_collect(self.root, self.focus) {
            self.root = 0;
            self.focus = new_focus;
        }

    }
}


impl RootedForest<Expr> {
    fn is_call(&self) -> bool {
        if let Some(Node{val: Expr::Call, ..}) = self.get_focus() {
            true
        } else {
            false
        }
    }

    fn get_child_nums_binary(&self) -> Option<(Index, isize, isize)> {

        let focus = self.get_focus()?;
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
        let focus = self.get_focus();
        focus.is_none() || focus.unwrap().val == Expr::Quote
    }

    pub fn pretty_print_tree(&self) {

        self.forest.print_tree(self.root, |expr| {
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
pub struct Clause {
    pub in_scopes: Vec<Index>,
    pub out_scopes: Vec<Index>,
    pub left: Index,
    pub right: Index,
}

#[derive(Debug, Clone)]
pub struct Program {
    // Can we rewrite meta? We can match on meta and rewrite else where.
    // But what would it mean to rewrite meta?
    pub meta: Meta<Index>,
    // Technically these chould just be in the hashmap.
    // But it seems worthwhile to bring them to the top level.
    pub main: RootedForest<Expr>,
    pub io: RootedForest<Expr>,
    // We will need some structure for the preprocessed rules.
    // Like keeping a list of clauses by scope and type.
    pub rules: RootedForest<Expr>,
    pub symbols: Interner,
    pub scopes: HashMap<Index, RootedForest<Expr>>,
    pub clause_indexes: Vec<Clause>,
}



// To do some matches, I probably just want to do a queue and 
// go through the nodes pushing all the children in the same order.
// I can check the length of the children
// Of course once I have repeats that will be more complicated.
// But I don't keep track of anywhere I am, so I don't need to mess around
// with the focus. I will need the focus to establish where to look,
// but that might/probably will be separate.
// The read only rooted tree could be good for that or could be unnecessary.


impl Program {

    pub fn new() -> Program {

        let mut symbols = Interner::new();
        symbols.intern("builtin/+");
        symbols.intern("builtin/-");
        symbols.intern("builtin/*");
        symbols.intern("builtin/div");
        symbols.intern("fact");
        symbols.intern("@main");
        symbols.intern("@io");
        symbols.intern("@rules");
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
            symbols: symbols,
            scopes: HashMap::new(),
            clause_indexes: Vec::new(),
        }
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
    pub fn build_env(&self, scope: &dyn ForestLike<Expr>, left_hand_index : Index, expr_index : Index) -> Option<HashMap<Index, Index>> {
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
                (Node{ val: l_val, index: l_index, ..}, Node{ val: e_val, index: e_index, ..}) => {
                    let l_children = self.rules.get_children(*l_index)?;
                    let e_children = scope.get_children(*e_index)?;
                    if l_val != e_val {
                        failed = true;
                        break;
                    }
                    // Children length will have to change once repeats exist.
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

    pub fn pretty_print_scope(&self, scope : &RootedForest<Expr>) {
        scope.forest.print_tree(scope.root, |expr| {
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

    // Returns old nodes new location
    pub fn substitute(mut scope: &mut RootedForest<Expr>, rule_scope: &RootedForest<Expr>, right_index: Index, env: HashMap<Index, Index>) -> Option<Index> {
        let right = rule_scope.get(right_index).unwrap().clone();
        let right_replace = match right.val {
            Expr::LogicVariable(index) => {
                scope.get(*env.get(&index).unwrap()).unwrap().clone().val
            }
            val => val
        };
        let node = scope.get_focus().unwrap();
        let parent = node.parent.clone();

        let result = scope.forest.persistent_change(right_replace, scope.focus);
        let focus = scope.forest.arena.get_mut(scope.focus).unwrap();
        focus.children.clear();
        rule_scope.forest.copy_tree_helper_f(right_index, right_index, parent, &mut scope, & |val| {
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

    // What I really want to is to pass the scope as a value, 
    // but then I am borrowing self multiple times. I am going to instead have this stupid
    // janking indexing thing.
    pub fn rewrite(&mut self, scope_index: Index) -> Option<()> {

        let rules = &self.rules;
        let scope = match scope_index {
            0 => &self.main,
            1 => &self.io,
            2 => &self.rules,
            _ => self.scopes.get(&scope_index).unwrap()
        };
        let scope_symbol_index = match scope_index {
            0 => &self.symbols.get_index("@main").unwrap(),
            1 => &self.symbols.get_index("@io").unwrap(),
            2 => &self.symbols.get_index("@rules").unwrap(),
            _ => &scope_index
        };
        // This all needs refactoring, but it is nice to see factorial
        // working with this code base. I need to do meta eval stuff and see
        // how fast we still are. But so far, speed is much better.

        let original_root = scope.root;
        let original_sub_expr = scope.focus;

        // We first check if there is a matching rule.
        // We really need to handle meta stuff here.
        let mut matching_rule = None;
        for Clause{left,right, in_scopes, ..} in &self.clause_indexes {
            if !in_scopes.contains(&scope_symbol_index) { continue };
            let env = self.build_env(scope, *left, scope.focus);
            if env.is_none() { continue };
            // Need some notion of output scopes
            matching_rule = Some((*right, env.unwrap()));
            break;
        };

        let scope = match scope_index {
            0 => &mut self.main,
            1 => &mut self.io,
            // I can do rules here because it is already borrowed.
            // Going to have to figure that out.
            // 2 => &mut self.rules,
            _ => self.scopes.get_mut(&scope_index).unwrap()
        };


        if let Some((right, env)) = matching_rule {
            let old_focus_new_location = Program::substitute(scope, rules, right, env)?;
            let meta_original_root = if scope.focus == scope.root { old_focus_new_location } else { scope.root };
            let meta_original_focus = old_focus_new_location;
            let meta = Meta {
                original_expr: meta_original_root,
                original_sub_expr: meta_original_focus,
                new_expr: scope.root,
                new_sub_expr: scope.focus,
            };
            // We need to now try to match on meta.
            // We need some tree we can do a build_env on.
            // So the basic idea is we have a tree that is the shape of meta above.
            // but those indexes really dereference to the scope we are meta evaling on.
            // So we need a get_method but we also need to be doing something like get_children.
            // But the get_children can't be on a node, they need to be on the forest itself.
            // That way when code asks for the children we can intercept that call and
            // check if the node is a parent of our meta nodes and return the correct thing.
            // Otherwise we'd have to have a full copy of the tress which would be crazy.
            // So I either need an interface for build_env or I need to mutate the scope and just
            // add these metaoverrides to it. The latter might be the quickest change.


            return None
        }

        // Then we check if there is a builtin rule.
        // Also need to handle meta stuff here.

        if let Some(focus) = scope.get_focus() {
            match focus.val {
                Expr::Call => {
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
                            scope.forest.persistent_change(val, scope.focus);
                            scope.forest.clear_children(scope.focus);
                            return None;
                        }
                    }
                    
                    
                    scope.exhaust_focus();
                }
                _ => {
                    scope.exhaust_focus()
                }
            }
        } else {
            scope.exhaust_focus();
        }
        None
    }

    // Making this work for main right now even though
    // we actually need multiple scopes.
    fn step(&mut self) {
        // println!("step");
        let scope = &mut self.main;
        scope.move_to_next_reducible_expr();
        if scope.root_is_exhausted() {
            return
        }
        // rewrite should return old node location.
        // meta needs to change.
        self.rewrite(0);
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