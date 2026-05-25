//! In-memory state store. v0.1 has no versioning; the scheduler is
//! single-threaded so reads observe the latest write.

use indexmap::IndexMap;

use ir::manifest::{Manifest, StateDecl};

use crate::value::Value;

#[derive(Debug, Default, Clone)]
pub struct StateStore {
    atoms: IndexMap<String, Value>,
    maps: IndexMap<String, Vec<(Value, Value)>>,
}

impl StateStore {
    pub fn from_manifest(manifest: &Manifest) -> Self {
        let mut store = Self::default();
        for (name, decl) in &manifest.state {
            match decl {
                StateDecl::Atom { .. } => {
                    store.atoms.insert(name.clone(), Value::Null);
                }
                StateDecl::Map { .. } => {
                    store.maps.insert(name.clone(), Vec::new());
                }
            }
        }
        store
    }

    pub fn atoms(&self) -> &IndexMap<String, Value> {
        &self.atoms
    }

    pub fn maps(&self) -> &IndexMap<String, Vec<(Value, Value)>> {
        &self.maps
    }

    pub fn get_atom(&self, name: &str) -> Option<&Value> {
        self.atoms.get(name)
    }

    pub fn set_atom(&mut self, name: &str, value: Value) {
        self.atoms.insert(name.to_string(), value);
    }

    pub fn get_map_entry(&self, name: &str, key: &Value) -> Option<&Value> {
        self.maps
            .get(name)
            .and_then(|m| m.iter().find(|(k, _)| k == key).map(|(_, v)| v))
    }

    pub fn list_map(&self, name: &str) -> Vec<(Value, Value)> {
        self.maps
            .get(name)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    }

    pub fn put_map_entry(&mut self, name: &str, key: Value, value: Value) {
        if let Some(m) = self.maps.get_mut(name) {
            if let Some(slot) = m.iter_mut().find(|(k, _)| k == &key) {
                slot.1 = value;
            } else {
                m.push((key, value));
            }
        }
    }

    pub fn delete_map_entry(&mut self, name: &str, key: &Value) {
        if let Some(m) = self.maps.get_mut(name) {
            m.retain(|(k, _)| k != key);
        }
    }
}
