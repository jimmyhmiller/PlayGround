use std::collections::HashMap;
use std::fmt;

// ── Newtypes ──

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TermId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SymId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

// ── TermData (Copy — args live in a pool, no heap alloc per term) ──

#[derive(Clone, Copy, Debug)]
pub enum TermData {
    Num(i64),
    Float(u64), // f64 stored as bits for Copy+Eq+Hash
    Sym(SymId),
    Call { head: TermId, args_start: u32, args_len: u16 },
}

#[inline]
pub fn float_val(bits: u64) -> f64 {
    f64::from_bits(bits)
}

// ── TermStore ──

const NUM_CACHE_SIZE: usize = 64;

pub struct TermStore {
    terms: Vec<TermData>,
    pub args_pool: Vec<TermId>,
    num_cache: [TermId; NUM_CACHE_SIZE],
    num_cache_valid: u64,
    num_dedup: HashMap<i64, TermId>,
    float_dedup: HashMap<u64, TermId>,
    sym_term_dedup: HashMap<SymId, TermId>,
    call_dedup: HashMap<[u32; 6], TermId>,
    symbols: Vec<String>,
    sym_dedup: HashMap<String, SymId>,
}

impl TermStore {
    pub fn new() -> Self {
        TermStore {
            terms: Vec::with_capacity(1024),
            args_pool: Vec::with_capacity(2048),
            num_cache: [TermId(0); NUM_CACHE_SIZE],
            num_cache_valid: 0,
            num_dedup: HashMap::new(),
            float_dedup: HashMap::new(),
            sym_term_dedup: HashMap::new(),
            call_dedup: HashMap::new(),
            symbols: Vec::new(),
            sym_dedup: HashMap::new(),
        }
    }

    #[inline]
    pub fn get(&self, id: TermId) -> TermData {
        self.terms[id.0 as usize]
    }

    pub fn sym(&mut self, name: &str) -> SymId {
        if let Some(&id) = self.sym_dedup.get(name) {
            return id;
        }
        let id = SymId(self.symbols.len() as u32);
        self.symbols.push(name.to_string());
        self.sym_dedup.insert(name.to_string(), id);
        id
    }

    pub fn sym_name(&self, id: SymId) -> &str {
        &self.symbols[id.0 as usize]
    }

    #[inline]
    pub fn num(&mut self, n: i64) -> TermId {
        if n >= 0 && (n as usize) < NUM_CACHE_SIZE {
            let i = n as usize;
            if self.num_cache_valid & (1u64 << i) != 0 {
                return self.num_cache[i];
            }
            let id = TermId(self.terms.len() as u32);
            self.terms.push(TermData::Num(n));
            self.num_cache[i] = id;
            self.num_cache_valid |= 1u64 << i;
            id
        } else {
            if let Some(&id) = self.num_dedup.get(&n) {
                return id;
            }
            let id = TermId(self.terms.len() as u32);
            self.terms.push(TermData::Num(n));
            self.num_dedup.insert(n, id);
            id
        }
    }

    pub fn float(&mut self, f: f64) -> TermId {
        let bits = f.to_bits();
        if let Some(&id) = self.float_dedup.get(&bits) {
            return id;
        }
        let id = TermId(self.terms.len() as u32);
        self.terms.push(TermData::Float(bits));
        self.float_dedup.insert(bits, id);
        id
    }

    pub fn symbol(&mut self, name: &str) -> TermId {
        let s = self.sym(name);
        self.sym_term(s)
    }

    #[inline]
    pub fn sym_term(&mut self, s: SymId) -> TermId {
        if let Some(&id) = self.sym_term_dedup.get(&s) {
            return id;
        }
        let id = TermId(self.terms.len() as u32);
        self.terms.push(TermData::Sym(s));
        self.sym_term_dedup.insert(s, id);
        id
    }

    #[inline]
    pub fn call(&mut self, head: TermId, args: &[TermId]) -> TermId {
        if args.len() <= 4 {
            let mut key = [0u32; 6];
            key[0] = args.len() as u32;
            key[1] = head.0;
            for (i, a) in args.iter().enumerate() {
                key[i + 2] = a.0;
            }
            if let Some(&id) = self.call_dedup.get(&key) {
                return id;
            }
            let args_start = self.args_pool.len() as u32;
            self.args_pool.extend_from_slice(args);
            let id = TermId(self.terms.len() as u32);
            self.terms.push(TermData::Call { head, args_start, args_len: args.len() as u16 });
            self.call_dedup.insert(key, id);
            id
        } else {
            let args_start = self.args_pool.len() as u32;
            self.args_pool.extend_from_slice(args);
            let id = TermId(self.terms.len() as u32);
            self.terms.push(TermData::Call { head, args_start, args_len: args.len() as u16 });
            id
        }
    }

    pub fn display(&self, id: TermId) -> TermDisplay<'_> {
        TermDisplay { store: self, id }
    }
}

pub struct TermDisplay<'a> {
    store: &'a TermStore,
    id: TermId,
}

impl fmt::Display for TermDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.store.get(self.id) {
            TermData::Num(n) => write!(f, "{}", n),
            TermData::Float(bits) => {
                let v = float_val(bits);
                if v.fract() == 0.0 {
                    write!(f, "{:.1}", v)
                } else {
                    write!(f, "{}", v)
                }
            }
            TermData::Sym(s) => write!(f, "{}", self.store.sym_name(s)),
            TermData::Call { head, args_start, args_len } => {
                // quote(x) displays transparently as x
                if args_len == 1 {
                    if let TermData::Sym(s) = self.store.get(head) {
                        if self.store.sym_name(s) == "quote" {
                            let inner = self.store.args_pool[args_start as usize];
                            return write!(f, "{}", self.store.display(inner));
                        }
                    }
                }
                write!(f, "{}(", self.store.display(head))?;
                for i in 0..args_len as usize {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    let arg = self.store.args_pool[args_start as usize + i];
                    write!(f, "{}", self.store.display(arg))?;
                }
                write!(f, ")")
            }
        }
    }
}
