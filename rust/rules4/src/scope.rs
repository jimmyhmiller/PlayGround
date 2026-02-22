use crate::term::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::rc::Rc;

// ── Scope Handler ──

pub trait ScopeHandler {
    fn try_eval(&mut self, store: &mut TermStore, term: TermId) -> Option<TermId>;
}

// ── Scope ──
// An independent actor with its own dynamic rules, work queue, and pending buffer.

pub struct Scope {
    pub name: String,
    pub dynamic_rules: Vec<(TermId, TermId)>,
    pub dynamic_rules_map: HashMap<TermId, TermId>,
    pub work_queue: VecDeque<TermId>,
    pub handler: Option<Box<dyn ScopeHandler>>,
    pub pending: Rc<RefCell<Vec<TermId>>>,
    pub output_only: bool, // true = buffer-only (DOM), false = full actor
}

impl Scope {
    pub fn new(name: String, output_only: bool) -> Self {
        Scope {
            name,
            dynamic_rules: Vec::new(),
            dynamic_rules_map: HashMap::new(),
            work_queue: VecDeque::new(),
            handler: None,
            pending: Rc::new(RefCell::new(Vec::new())),
            output_only,
        }
    }
}

// ── Browser/DOM Scope Handler ──
// Buffers terms written via @dom for the JS runtime to pick up.
// Uses a shared buffer so the Engine can read pending terms.

pub struct BrowserHandler {
    pub pending: Rc<RefCell<Vec<TermId>>>,
}

impl BrowserHandler {
    pub fn new(pending: Rc<RefCell<Vec<TermId>>>) -> Self {
        BrowserHandler { pending }
    }
}

impl ScopeHandler for BrowserHandler {
    fn try_eval(&mut self, _store: &mut TermStore, term: TermId) -> Option<TermId> {
        self.pending.borrow_mut().push(term);
        Some(term)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct IoHandler {
    pub println_sym: SymId,
    pub readline_sym: SymId,
}

#[cfg(not(target_arch = "wasm32"))]
impl ScopeHandler for IoHandler {
    fn try_eval(&mut self, store: &mut TermStore, term: TermId) -> Option<TermId> {
        use std::io::{self, BufRead};

        if let TermData::Call { head, args_start, args_len } = store.get(term) {
            if let TermData::Sym(s) = store.get(head) {
                if s == self.println_sym {
                    for i in 0..args_len as usize {
                        let arg = store.args_pool[args_start as usize + i];
                        print!("{}", store.display(arg));
                    }
                    println!();
                    return Some(store.num(0));
                }
                if s == self.readline_sym && args_len == 0 {
                    let mut line = String::new();
                    io::stdin().lock().read_line(&mut line).unwrap();
                    let trimmed = line.trim();
                    let sym = store.sym(trimmed);
                    return Some(store.sym_term(sym));
                }
            }
        }
        None
    }
}
