//! Symbol intern table. Symbols are addressed by a u32 id; identity is by id
//! (so `eq?` on symbols is integer compare).
//!
//! Stored in the host context. Reader and macroexpander look up names; the
//! compiler embeds ids as IR constants.

use std::collections::HashMap;

pub struct SymbolTable {
    name_to_id: HashMap<String, u32>,
    id_to_name: Vec<String>,
    /// Counter for `gensym` — each call returns a fresh id with a printable
    /// "#:tag<n>" name that no source program could read back.
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
        // Insert directly; do not register in name_to_id so source code can
        // never collide.
        let id = self.id_to_name.len() as u32;
        self.id_to_name.push(name);
        id
    }
}
