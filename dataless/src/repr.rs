//! Data representations.
//!
//! Every collection is, logically, an ordered list of records indexed 1..n
//! (exactly as in Balzer's paper: "the data representations are ordered
//! lists"). The *representation* decides how that ordered list is stored, and
//! therefore what each operation costs — but never what it computes. All
//! representations expose the identical interface, so the program that uses
//! them is byte-for-byte the same.
//!
//!   ARRAY        positional lookup O(1), insert/delete in middle O(n)
//!   LIST         positional lookup O(n), insert-after-handle O(1),
//!                delete via handle O(n) (must find predecessor)
//!   DOUBLE_LIST  positional lookup O(n), insert/delete via handle O(1)
//!
//! A `steps` counter records the work done so the cost difference is visible
//! and testable even though the output is identical.

use crate::value::Value;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rep {
    Array,
    List,
    DoubleList,
}

impl Rep {
    pub fn name(self) -> &'static str {
        match self {
            Rep::Array => "ARRAY",
            Rep::List => "LIST (forward links)",
            Rep::DoubleList => "DOUBLE_LIST (forward + backward links)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Type {
    Int,
    Text,
    Bool,
}

pub struct FieldLayout {
    pub name: String,
    pub default: Value,
}

struct Node {
    fields: Vec<Value>,
    next: Option<usize>,
    prev: Option<usize>,
    alive: bool,
}

enum Store {
    /// Stable slab of records + an explicit order vector (position -> slab id).
    Array {
        slab: Vec<Vec<Value>>,
        order: Vec<usize>,
    },
    /// Linked slab. `doubly` enables O(1) delete via back-links.
    Linked {
        nodes: Vec<Node>,
        head: Option<usize>,
        tail: Option<usize>,
        len: usize,
        doubly: bool,
    },
}

pub struct Collection {
    pub name: String,
    pub rep: Rep,
    pub fields: Vec<FieldLayout>,
    store: Store,
    pub steps: u64,
}

impl Collection {
    pub fn new(name: String, rep: Rep, fields: Vec<FieldLayout>) -> Collection {
        let store = match rep {
            Rep::Array => Store::Array {
                slab: Vec::new(),
                order: Vec::new(),
            },
            Rep::List => Store::Linked {
                nodes: Vec::new(),
                head: None,
                tail: None,
                len: 0,
                doubly: false,
            },
            Rep::DoubleList => Store::Linked {
                nodes: Vec::new(),
                head: None,
                tail: None,
                len: 0,
                doubly: true,
            },
        };
        Collection {
            name,
            rep,
            fields,
            store,
            steps: 0,
        }
    }

    pub fn field_id(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    fn defaults(&self) -> Vec<Value> {
        self.fields.iter().map(|f| f.default.clone()).collect()
    }

    pub fn len(&self) -> usize {
        match &self.store {
            Store::Array { order, .. } => order.len(),
            Store::Linked { len, .. } => *len,
        }
    }

    /// Append a fresh member at the end; returns its stable id (a handle).
    /// O(1) for every representation.
    pub fn append(&mut self) -> usize {
        let defaults = self.defaults();
        match &mut self.store {
            Store::Array { slab, order } => {
                let id = slab.len();
                slab.push(defaults);
                order.push(id);
                id
            }
            Store::Linked {
                nodes,
                head,
                tail,
                len,
                ..
            } => {
                let id = nodes.len();
                nodes.push(Node {
                    fields: defaults,
                    next: None,
                    prev: *tail,
                    alive: true,
                });
                match *tail {
                    Some(t) => nodes[t].next = Some(id),
                    None => *head = Some(id),
                }
                *tail = Some(id);
                *len += 1;
                id
            }
        }
    }

    /// O(1) field read via a stable handle, for any representation.
    pub fn get(&self, id: usize, field: usize) -> Value {
        match &self.store {
            Store::Array { slab, .. } => slab[id][field].clone(),
            Store::Linked { nodes, .. } => nodes[id].fields[field].clone(),
        }
    }

    /// O(1) field write via a stable handle.
    pub fn set(&mut self, id: usize, field: usize, v: Value) {
        match &mut self.store {
            Store::Array { slab, .. } => slab[id][field] = v,
            Store::Linked { nodes, .. } => nodes[id].fields[field] = v,
        }
    }

    /// The stable handle of the member at 1-based position `pos`.
    /// This is where representations diverge: O(1) for ARRAY, O(pos) for lists.
    pub fn member_at(&mut self, pos: usize) -> Result<usize, String> {
        if pos < 1 || pos > self.len() {
            return Err(format!(
                "position {} out of range for `{}` (1..{})",
                pos,
                self.name,
                self.len()
            ));
        }
        match &self.store {
            Store::Array { order, .. } => {
                self.steps += 1;
                Ok(order[pos - 1])
            }
            Store::Linked { nodes, head, .. } => {
                let mut cur = head.unwrap();
                let mut steps = 1u64;
                for _ in 1..pos {
                    cur = nodes[cur].next.unwrap();
                    steps += 1;
                }
                self.steps += steps;
                Ok(cur)
            }
        }
    }

    /// All member handles in positional order (single O(n) walk).
    pub fn members(&mut self) -> Vec<usize> {
        match &self.store {
            Store::Array { order, .. } => {
                self.steps += order.len() as u64;
                order.clone()
            }
            Store::Linked { nodes, head, len, .. } => {
                let mut out = Vec::with_capacity(*len);
                let mut cur = *head;
                while let Some(c) = cur {
                    out.push(c);
                    cur = nodes[c].next;
                }
                self.steps += out.len() as u64;
                out
            }
        }
    }

    /// Insert a fresh member immediately *after* the given handle; returns the
    /// new handle. O(1) for lists (the paper's selling point), O(n) for arrays.
    pub fn insert_after(&mut self, after: usize) -> usize {
        let defaults = self.defaults();
        match &mut self.store {
            Store::Array { slab, order } => {
                let id = slab.len();
                slab.push(defaults);
                let pos = order.iter().position(|&x| x == after).unwrap();
                self.steps += (order.len() - pos) as u64; // shifting cost
                order.insert(pos + 1, id);
                id
            }
            Store::Linked {
                nodes,
                tail,
                len,
                ..
            } => {
                let id = nodes.len();
                let nxt = nodes[after].next;
                nodes.push(Node {
                    fields: defaults,
                    next: nxt,
                    prev: Some(after),
                    alive: true,
                });
                nodes[after].next = Some(id);
                match nxt {
                    Some(n) => nodes[n].prev = Some(id),
                    None => *tail = Some(id),
                }
                *len += 1;
                self.steps += 1;
                id
            }
        }
    }

    /// Delete a member by handle. ARRAY shifts (O(n)); DOUBLE_LIST is O(1);
    /// singly-linked LIST must find the predecessor (O(n)).
    pub fn delete(&mut self, id: usize) -> Result<(), String> {
        match &mut self.store {
            Store::Array { slab: _, order } => {
                let pos = order
                    .iter()
                    .position(|&x| x == id)
                    .ok_or_else(|| "delete: stale member handle".to_string())?;
                self.steps += (order.len() - pos) as u64;
                order.remove(pos);
                Ok(())
            }
            Store::Linked {
                nodes,
                head,
                tail,
                len,
                doubly,
            } => {
                if !nodes[id].alive {
                    return Err("delete: member already deleted".into());
                }
                let prev = if *doubly {
                    self.steps += 1;
                    nodes[id].prev
                } else {
                    // singly linked: walk from head to find the predecessor
                    let mut p = None;
                    let mut cur = *head;
                    let mut steps = 0u64;
                    while let Some(c) = cur {
                        steps += 1;
                        if c == id {
                            break;
                        }
                        p = Some(c);
                        cur = nodes[c].next;
                    }
                    self.steps += steps;
                    p
                };
                let next = nodes[id].next;
                match prev {
                    Some(p) => nodes[p].next = next,
                    None => *head = next,
                }
                match next {
                    Some(n) => {
                        if *doubly {
                            nodes[n].prev = prev;
                        }
                    }
                    None => *tail = prev,
                }
                nodes[id].alive = false;
                *len -= 1;
                Ok(())
            }
        }
    }
}
