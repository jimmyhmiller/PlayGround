//! Runtime values and the polymorphic `Coll` (collection) type.
//!
//! The whole point of this language is that the *programmer* never picks a
//! concrete data structure. The analyzer assigns each abstract collection a
//! `Kind` (Set / Map / Sequence / Queue, plus whether ordering matters), and
//! the runtime instantiates a concrete `Store` for that kind. We keep TWO
//! backends for every kind:
//!
//!   * the *specialized* backend  — the best structure for the job
//!   * the *naive* backend        — a linear structure of the same family
//!
//! Both obey identical observable semantics, so a program produces the same
//! answer either way; only the performance differs. That difference is the
//! entire value proposition, and the `bench` subcommand measures it.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::rc::Rc;

/// The abstract "shape" of a collection, as decided by usage analysis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Set { ordered: bool },
    Map { ordered: bool },
    Sequence,
    Queue,
}

/// A runtime value. Collections are reference-counted and mutable in place.
#[derive(Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Str(String),
    Nil,
    Coll(Rc<RefCell<Coll>>),
}

/// A scalar usable as a set element or map key.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Key {
    Int(i64),
    Bool(bool),
    Str(String),
}

impl Key {
    pub fn to_value(&self) -> Value {
        match self {
            Key::Int(i) => Value::Int(*i),
            Key::Bool(b) => Value::Bool(*b),
            Key::Str(s) => Value::Str(s.clone()),
        }
    }
}

impl Value {
    pub fn to_key(&self) -> Result<Key, String> {
        match self {
            Value::Int(i) => Ok(Key::Int(*i)),
            Value::Bool(b) => Ok(Key::Bool(*b)),
            Value::Str(s) => Ok(Key::Str(s.clone())),
            Value::Nil => Err("nil cannot be used as a set element or map key".into()),
            Value::Coll(_) => Err("a collection cannot be used as a set element or map key".into()),
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Str(s) => !s.is_empty(),
            Value::Nil => false,
            Value::Coll(c) => c.borrow().size() > 0,
        }
    }
}

/// Concrete storage. The same `Kind` maps to a specialized or naive variant.
pub enum Store {
    // Set family
    HashSet(HashSet<Key>),
    BTreeSet(BTreeSet<Key>),
    LinSet(Vec<Key>),
    // Map family
    HashMap(HashMap<Key, Value>),
    BTreeMap(BTreeMap<Key, Value>),
    LinMap(Vec<(Key, Value)>),
    // Sequence family (also backs stacks and list literals)
    Vec(Vec<Value>),
    // Queue family
    Deque(VecDeque<Value>),
    LinQueue(Vec<Value>),
}

pub struct Coll {
    pub store: Store,
}

impl Coll {
    /// Build an empty collection for a kind, choosing specialized vs naive.
    pub fn new(kind: Kind, naive: bool) -> Coll {
        let store = match (kind, naive) {
            (Kind::Set { ordered: false }, false) => Store::HashSet(HashSet::new()),
            (Kind::Set { ordered: true }, false) => Store::BTreeSet(BTreeSet::new()),
            (Kind::Set { .. }, true) => Store::LinSet(Vec::new()),
            (Kind::Map { ordered: false }, false) => Store::HashMap(HashMap::new()),
            (Kind::Map { ordered: true }, false) => Store::BTreeMap(BTreeMap::new()),
            (Kind::Map { .. }, true) => Store::LinMap(Vec::new()),
            (Kind::Sequence, _) => Store::Vec(Vec::new()),
            (Kind::Queue, false) => Store::Deque(VecDeque::new()),
            (Kind::Queue, true) => Store::LinQueue(Vec::new()),
        };
        Coll { store }
    }

    /// A bare sequence (used for list literals, `range`, and `sorted` results).
    pub fn seq(values: Vec<Value>) -> Coll {
        Coll { store: Store::Vec(values) }
    }

    pub fn into_value(self) -> Value {
        Value::Coll(Rc::new(RefCell::new(self)))
    }

    // ----- set / map: membership and keyed access -----------------------

    pub fn add(&mut self, v: Value) -> Result<Value, String> {
        let k = v.to_key()?;
        match &mut self.store {
            Store::HashSet(s) => {
                s.insert(k);
            }
            Store::BTreeSet(s) => {
                s.insert(k);
            }
            Store::LinSet(s) => {
                if !s.iter().any(|kk| kk == &k) {
                    s.push(k);
                }
            }
            Store::HashMap(m) => {
                m.entry(k).or_insert(Value::Nil);
            }
            Store::BTreeMap(m) => {
                m.entry(k).or_insert(Value::Nil);
            }
            Store::LinMap(m) => {
                if !m.iter().any(|(kk, _)| kk == &k) {
                    m.push((k, Value::Nil));
                }
            }
            Store::Vec(seq) => seq.push(v),
            Store::Deque(q) => q.push_back(v),
            Store::LinQueue(q) => q.push(v),
        }
        Ok(Value::Nil)
    }

    pub fn has(&self, v: &Value) -> Result<Value, String> {
        let found = match &self.store {
            Store::HashSet(s) => s.contains(&v.to_key()?),
            Store::BTreeSet(s) => s.contains(&v.to_key()?),
            Store::LinSet(s) => {
                let k = v.to_key()?;
                s.iter().any(|kk| kk == &k)
            }
            Store::HashMap(m) => m.contains_key(&v.to_key()?),
            Store::BTreeMap(m) => m.contains_key(&v.to_key()?),
            Store::LinMap(m) => {
                let k = v.to_key()?;
                m.iter().any(|(kk, _)| kk == &k)
            }
            Store::Vec(seq) => seq.iter().any(|x| values_eq(x, v)),
            Store::Deque(q) => q.iter().any(|x| values_eq(x, v)),
            Store::LinQueue(q) => q.iter().any(|x| values_eq(x, v)),
        };
        Ok(Value::Bool(found))
    }

    pub fn del(&mut self, v: &Value) -> Result<Value, String> {
        match &mut self.store {
            Store::HashSet(s) => {
                s.remove(&v.to_key()?);
            }
            Store::BTreeSet(s) => {
                s.remove(&v.to_key()?);
            }
            Store::LinSet(s) => {
                let k = v.to_key()?;
                s.retain(|kk| kk != &k);
            }
            Store::HashMap(m) => {
                m.remove(&v.to_key()?);
            }
            Store::BTreeMap(m) => {
                m.remove(&v.to_key()?);
            }
            Store::LinMap(m) => {
                let k = v.to_key()?;
                m.retain(|(kk, _)| kk != &k);
            }
            Store::Vec(seq) => seq.retain(|x| !values_eq(x, v)),
            Store::Deque(q) => q.retain(|x| !values_eq(x, v)),
            Store::LinQueue(q) => q.retain(|x| !values_eq(x, v)),
        }
        Ok(Value::Nil)
    }

    pub fn put(&mut self, k: Value, val: Value) -> Result<Value, String> {
        let key = k.to_key()?;
        match &mut self.store {
            Store::HashMap(m) => {
                m.insert(key, val);
            }
            Store::BTreeMap(m) => {
                m.insert(key, val);
            }
            Store::LinMap(m) => {
                if let Some(e) = m.iter_mut().find(|(kk, _)| kk == &key) {
                    e.1 = val;
                } else {
                    m.push((key, val));
                }
            }
            _ => return Err("`put` requires a map-kind collection".into()),
        }
        Ok(Value::Nil)
    }

    pub fn get(&self, k: &Value) -> Result<Value, String> {
        let key = k.to_key()?;
        Ok(match &self.store {
            Store::HashMap(m) => m.get(&key).cloned().unwrap_or(Value::Nil),
            Store::BTreeMap(m) => m.get(&key).cloned().unwrap_or(Value::Nil),
            Store::LinMap(m) => m
                .iter()
                .find(|(kk, _)| kk == &key)
                .map(|(_, v)| v.clone())
                .unwrap_or(Value::Nil),
            _ => return Err("`get` requires a map-kind collection".into()),
        })
    }

    // ----- sequence / stack ---------------------------------------------

    pub fn append(&mut self, v: Value) -> Result<Value, String> {
        match &mut self.store {
            Store::Vec(seq) => seq.push(v),
            Store::Deque(q) => q.push_back(v),
            Store::LinQueue(q) => q.push(v),
            _ => return Err("`append` requires a sequence-kind collection".into()),
        }
        Ok(Value::Nil)
    }

    pub fn at(&self, i: i64) -> Result<Value, String> {
        let idx = i as usize;
        match &self.store {
            Store::Vec(seq) => seq
                .get(idx)
                .cloned()
                .ok_or_else(|| format!("index {} out of bounds (len {})", i, seq.len())),
            Store::Deque(q) => q
                .get(idx)
                .cloned()
                .ok_or_else(|| format!("index {} out of bounds (len {})", i, q.len())),
            _ => Err("`at` requires a sequence-kind collection".into()),
        }
    }

    pub fn set_at(&mut self, i: i64, v: Value) -> Result<Value, String> {
        let idx = i as usize;
        match &mut self.store {
            Store::Vec(seq) => {
                if idx < seq.len() {
                    seq[idx] = v;
                    Ok(Value::Nil)
                } else {
                    Err(format!("index {} out of bounds (len {})", i, seq.len()))
                }
            }
            _ => Err("`set_at` requires a sequence-kind collection".into()),
        }
    }

    pub fn push(&mut self, v: Value) -> Result<Value, String> {
        match &mut self.store {
            Store::Vec(seq) => {
                seq.push(v);
                Ok(Value::Nil)
            }
            _ => Err("`push` requires a sequence/stack-kind collection".into()),
        }
    }

    pub fn pop(&mut self) -> Result<Value, String> {
        match &mut self.store {
            Store::Vec(seq) => Ok(seq.pop().unwrap_or(Value::Nil)),
            _ => Err("`pop` requires a sequence/stack-kind collection".into()),
        }
    }

    pub fn peek(&self) -> Result<Value, String> {
        match &self.store {
            Store::Vec(seq) => Ok(seq.last().cloned().unwrap_or(Value::Nil)),
            _ => Err("`peek` requires a sequence/stack-kind collection".into()),
        }
    }

    // ----- queue ---------------------------------------------------------

    pub fn enqueue(&mut self, v: Value) -> Result<Value, String> {
        match &mut self.store {
            Store::Deque(q) => q.push_back(v),
            Store::LinQueue(q) => q.push(v),
            _ => return Err("`enqueue` requires a queue-kind collection".into()),
        }
        Ok(Value::Nil)
    }

    pub fn dequeue(&mut self) -> Result<Value, String> {
        match &mut self.store {
            Store::Deque(q) => Ok(q.pop_front().unwrap_or(Value::Nil)),
            // The naive queue pays O(n) to shift every element down. This is
            // exactly the cost the specialized ring buffer avoids.
            Store::LinQueue(q) => {
                if q.is_empty() {
                    Ok(Value::Nil)
                } else {
                    Ok(q.remove(0))
                }
            }
            _ => Err("`dequeue` requires a queue-kind collection".into()),
        }
    }

    pub fn front(&self) -> Result<Value, String> {
        match &self.store {
            Store::Deque(q) => Ok(q.front().cloned().unwrap_or(Value::Nil)),
            Store::LinQueue(q) => Ok(q.first().cloned().unwrap_or(Value::Nil)),
            _ => Err("`front` requires a queue-kind collection".into()),
        }
    }

    // ----- universal --------------------------------------------------

    pub fn size(&self) -> usize {
        match &self.store {
            Store::HashSet(s) => s.len(),
            Store::BTreeSet(s) => s.len(),
            Store::LinSet(s) => s.len(),
            Store::HashMap(m) => m.len(),
            Store::BTreeMap(m) => m.len(),
            Store::LinMap(m) => m.len(),
            Store::Vec(s) => s.len(),
            Store::Deque(q) => q.len(),
            Store::LinQueue(q) => q.len(),
        }
    }

    /// Natural iteration order: keys for sets/maps, values for sequences/queues.
    pub fn iter_values(&self) -> Vec<Value> {
        match &self.store {
            Store::HashSet(s) => s.iter().map(Key::to_value).collect(),
            Store::BTreeSet(s) => s.iter().map(Key::to_value).collect(),
            Store::LinSet(s) => s.iter().map(Key::to_value).collect(),
            Store::HashMap(m) => m.keys().map(Key::to_value).collect(),
            Store::BTreeMap(m) => m.keys().map(Key::to_value).collect(),
            Store::LinMap(m) => m.iter().map(|(k, _)| k.to_value()).collect(),
            Store::Vec(s) => s.clone(),
            Store::Deque(q) => q.iter().cloned().collect(),
            Store::LinQueue(q) => q.clone(),
        }
    }

    /// Elements in sorted order (keys for sets/maps, values for sequences).
    pub fn sorted_values(&self) -> Result<Vec<Value>, String> {
        // Sets/maps sort by key; sequences sort by value-as-key.
        let mut keys: Vec<Key> = match &self.store {
            Store::HashSet(s) => s.iter().cloned().collect(),
            Store::BTreeSet(s) => return Ok(s.iter().map(Key::to_value).collect()),
            Store::LinSet(s) => s.clone(),
            Store::HashMap(m) => m.keys().cloned().collect(),
            Store::BTreeMap(m) => return Ok(m.keys().map(Key::to_value).collect()),
            Store::LinMap(m) => m.iter().map(|(k, _)| k.clone()).collect(),
            Store::Vec(s) => {
                let mut ks = Vec::with_capacity(s.len());
                for v in s {
                    ks.push(v.to_key()?);
                }
                ks
            }
            Store::Deque(q) => {
                let mut ks = Vec::with_capacity(q.len());
                for v in q {
                    ks.push(v.to_key()?);
                }
                ks
            }
            Store::LinQueue(q) => {
                let mut ks = Vec::with_capacity(q.len());
                for v in q {
                    ks.push(v.to_key()?);
                }
                ks
            }
        };
        keys.sort();
        Ok(keys.iter().map(Key::to_value).collect())
    }

    pub fn min(&self) -> Result<Value, String> {
        Ok(self.sorted_values()?.into_iter().next().unwrap_or(Value::Nil))
    }

    pub fn max(&self) -> Result<Value, String> {
        Ok(self.sorted_values()?.into_iter().last().unwrap_or(Value::Nil))
    }

    pub fn keys(&self) -> Vec<Value> {
        self.iter_values()
    }

    /// (key, value) pairs for maps; used only for pretty-printing.
    fn entries(&self) -> Option<Vec<(Value, Value)>> {
        match &self.store {
            Store::HashMap(m) => Some(m.iter().map(|(k, v)| (k.to_value(), v.clone())).collect()),
            Store::BTreeMap(m) => Some(m.iter().map(|(k, v)| (k.to_value(), v.clone())).collect()),
            Store::LinMap(m) => Some(m.iter().map(|(k, v)| (k.to_value(), v.clone())).collect()),
            _ => None,
        }
    }

    pub fn display(&self) -> String {
        if let Some(entries) = self.entries() {
            let body: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", format_value(k), format_value(v)))
                .collect();
            return format!("{{{}}}", body.join(", "));
        }
        let is_set = matches!(
            self.store,
            Store::HashSet(_) | Store::BTreeSet(_) | Store::LinSet(_)
        );
        let body: Vec<String> = self.iter_values().iter().map(format_value).collect();
        if is_set {
            format!("{{{}}}", body.join(", "))
        } else {
            format!("[{}]", body.join(", "))
        }
    }
}

pub fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        (Value::Coll(x), Value::Coll(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

pub fn format_value(v: &Value) -> String {
    match v {
        Value::Int(i) => i.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Str(s) => s.clone(),
        Value::Nil => "nil".to_string(),
        Value::Coll(c) => c.borrow().display(),
    }
}
