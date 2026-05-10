//! Symbol intern table for the bootstrap Rust reader.
//!
//! Symbols are addressed by a `u32` id; identity is integer compare.
//! In Clojure, symbols can be qualified (`ns/name`) — we store the
//! qualified printed form as a single string and split on lookup if
//! needed. The compiler treats the head-of-list "symbol" as just an
//! id to dispatch on; the runtime Var system later resolves the
//! `ns/name` split into proper namespace lookups.

use std::collections::HashMap;

pub struct SymbolTable {
    name_to_id: HashMap<String, u32>,
    id_to_name: Vec<String>,
    gensym_counter: u32,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            name_to_id: HashMap::new(),
            id_to_name: Vec::new(),
            gensym_counter: 0,
        }
    }

    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = self.id_to_name.len() as u32;
        self.id_to_name.push(name.to_string());
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    pub fn name(&self, id: u32) -> &str {
        &self.id_to_name[id as usize]
    }

    pub fn gensym(&mut self, tag: &str) -> u32 {
        self.gensym_counter += 1;
        let n = self.gensym_counter;
        let name = format!("#:{tag}{n}");
        let id = self.id_to_name.len() as u32;
        self.id_to_name.push(name);
        id
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}
