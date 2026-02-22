use std::cmp::Ordering;
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
    transient_sym_pool: Vec<SymId>,
    transient_sym_next: usize,
}

impl TermStore {
    pub fn new() -> Self {
        let mut symbols = Vec::new();
        let mut transient_sym_pool = Vec::with_capacity(16);
        for _ in 0..16 {
            let id = SymId(symbols.len() as u32);
            symbols.push(String::new());
            transient_sym_pool.push(id);
        }
        TermStore {
            terms: Vec::with_capacity(1024),
            args_pool: Vec::with_capacity(2048),
            num_cache: [TermId(0); NUM_CACHE_SIZE],
            num_cache_valid: 0,
            num_dedup: HashMap::new(),
            float_dedup: HashMap::new(),
            sym_term_dedup: HashMap::new(),
            call_dedup: HashMap::new(),
            symbols,
            sym_dedup: HashMap::new(),
            transient_sym_pool,
            transient_sym_next: 0,
        }
    }

    /// Allocate a transient symbol term: overwrites a pooled SymId's name and
    /// creates a fresh (non-deduped) TermId so the vdom sees changes between frames.
    pub fn transient_sym_term(&mut self, content: String) -> TermId {
        let pool_idx = self.transient_sym_next % self.transient_sym_pool.len();
        self.transient_sym_next += 1;
        let sym_id = self.transient_sym_pool[pool_idx];
        self.symbols[sym_id.0 as usize] = content;
        // Fresh TermId (no dedup) so vdom sees changes between frames
        let term_id = TermId(self.terms.len() as u32);
        self.terms.push(TermData::Sym(sym_id));
        term_id
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

    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Total ordering on terms for canonical sorting of sets and map keys.
    /// Num < Float < Sym < Call. Within each: natural ordering.
    pub fn term_cmp(&self, a: TermId, b: TermId) -> Ordering {
        if a == b { return Ordering::Equal; }
        let ta = self.get(a);
        let tb = self.get(b);
        match (ta, tb) {
            (TermData::Num(na), TermData::Num(nb)) => na.cmp(&nb),
            (TermData::Num(_), _) => Ordering::Less,
            (_, TermData::Num(_)) => Ordering::Greater,

            (TermData::Float(fa), TermData::Float(fb)) => {
                f64::from_bits(fa).partial_cmp(&f64::from_bits(fb)).unwrap_or(Ordering::Equal)
            }
            (TermData::Float(_), _) => Ordering::Less,
            (_, TermData::Float(_)) => Ordering::Greater,

            (TermData::Sym(sa), TermData::Sym(sb)) => {
                self.sym_name(sa).cmp(self.sym_name(sb))
            }
            (TermData::Sym(_), _) => Ordering::Less,
            (_, TermData::Sym(_)) => Ordering::Greater,

            (TermData::Call { head: ha, args_start: asa, args_len: ala },
             TermData::Call { head: hb, args_start: asb, args_len: alb }) => {
                let hcmp = self.term_cmp(ha, hb);
                if hcmp != Ordering::Equal { return hcmp; }
                let lcmp = ala.cmp(&alb);
                if lcmp != Ordering::Equal { return lcmp; }
                for i in 0..ala as usize {
                    let aa = self.args_pool[asa as usize + i];
                    let ab = self.args_pool[asb as usize + i];
                    let c = self.term_cmp(aa, ab);
                    if c != Ordering::Equal { return c; }
                }
                Ordering::Equal
            }
        }
    }

    /// Mark-compact garbage collection.  Takes root TermIds, compacts the
    /// store, and returns a remap table (indexed by old TermId.0).
    pub fn gc(&mut self, roots: &[TermId]) -> Vec<TermId> {
        let n = self.terms.len();
        let mut marked = vec![false; n];
        let mut stack: Vec<u32> = Vec::with_capacity(roots.len() * 2);

        for root in roots {
            let i = root.0 as usize;
            if i < n && !marked[i] {
                stack.push(root.0);
            }
        }

        // Mark phase — BFS through Call children
        while let Some(idx) = stack.pop() {
            let i = idx as usize;
            if i >= n || marked[i] { continue; }
            marked[i] = true;
            if let TermData::Call { head, args_start, args_len } = self.terms[i] {
                stack.push(head.0);
                for j in 0..args_len as usize {
                    stack.push(self.args_pool[args_start as usize + j].0);
                }
            }
        }

        // Build remap table
        let mut remap = vec![TermId(u32::MAX); n];
        let mut new_count = 0u32;
        for (old_idx, &is_marked) in marked.iter().enumerate() {
            if is_marked {
                remap[old_idx] = TermId(new_count);
                new_count += 1;
            }
        }

        // Compact terms and args_pool
        let mut new_terms = Vec::with_capacity(new_count as usize);
        let mut new_args_pool = Vec::new();

        for (old_idx, &is_marked) in marked.iter().enumerate() {
            if !is_marked { continue; }
            match self.terms[old_idx] {
                TermData::Num(val) => new_terms.push(TermData::Num(val)),
                TermData::Float(bits) => new_terms.push(TermData::Float(bits)),
                TermData::Sym(s) => new_terms.push(TermData::Sym(s)),
                TermData::Call { head, args_start, args_len } => {
                    let new_head = remap[head.0 as usize];
                    let new_start = new_args_pool.len() as u32;
                    for j in 0..args_len as usize {
                        let old_arg = self.args_pool[args_start as usize + j];
                        new_args_pool.push(remap[old_arg.0 as usize]);
                    }
                    new_terms.push(TermData::Call {
                        head: new_head, args_start: new_start, args_len,
                    });
                }
            }
        }

        self.terms = new_terms;
        self.args_pool = new_args_pool;

        // Rebuild all dedup maps and caches from the compacted store
        self.num_dedup.clear();
        self.float_dedup.clear();
        self.sym_term_dedup.clear();
        self.call_dedup.clear();
        self.num_cache_valid = 0;
        self.transient_sym_next = 0;

        for (idx, term) in self.terms.iter().enumerate() {
            let tid = TermId(idx as u32);
            match *term {
                TermData::Num(val) => {
                    if val >= 0 && (val as usize) < NUM_CACHE_SIZE {
                        self.num_cache[val as usize] = tid;
                        self.num_cache_valid |= 1u64 << (val as usize);
                    } else {
                        self.num_dedup.insert(val, tid);
                    }
                }
                TermData::Float(bits) => { self.float_dedup.insert(bits, tid); }
                TermData::Sym(s) => { self.sym_term_dedup.entry(s).or_insert(tid); }
                TermData::Call { head, args_start, args_len } => {
                    if args_len <= 4 {
                        let mut key = [0u32; 6];
                        key[0] = args_len as u32;
                        key[1] = head.0;
                        for i in 0..args_len as usize {
                            key[i + 2] = self.args_pool[args_start as usize + i].0;
                        }
                        self.call_dedup.insert(key, tid);
                    }
                }
            }
        }

        remap
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
                if let TermData::Sym(s) = self.store.get(head) {
                    let name = self.store.sym_name(s);
                    // quote(x) displays transparently as x
                    if name == "quote" && args_len == 1 {
                        let inner = self.store.args_pool[args_start as usize];
                        return write!(f, "{}", self.store.display(inner));
                    }
                    // vec(...) displays as [...]
                    if name == "vec" {
                        write!(f, "[")?;
                        for i in 0..args_len as usize {
                            if i > 0 { write!(f, ", ")?; }
                            let arg = self.store.args_pool[args_start as usize + i];
                            write!(f, "{}", self.store.display(arg))?;
                        }
                        return write!(f, "]");
                    }
                    // set(...) displays as #{...}
                    if name == "set" {
                        write!(f, "#{{")?;
                        for i in 0..args_len as usize {
                            if i > 0 { write!(f, ", ")?; }
                            let arg = self.store.args_pool[args_start as usize + i];
                            write!(f, "{}", self.store.display(arg))?;
                        }
                        return write!(f, "}}");
                    }
                    // map(entry(k,v), ...) displays as {:k v ...}
                    if name == "map" {
                        write!(f, "{{")?;
                        for i in 0..args_len as usize {
                            if i > 0 { write!(f, " ")?; }
                            let arg = self.store.args_pool[args_start as usize + i];
                            if let TermData::Call { head: eh, args_start: eas, args_len: 2 } = self.store.get(arg) {
                                if let TermData::Sym(es) = self.store.get(eh) {
                                    if self.store.sym_name(es) == "entry" {
                                        let k = self.store.args_pool[eas as usize];
                                        let v = self.store.args_pool[eas as usize + 1];
                                        write!(f, "{} {}", self.store.display(k), self.store.display(v))?;
                                        continue;
                                    }
                                }
                            }
                            // fallback: display the arg directly
                            write!(f, "{}", self.store.display(arg))?;
                        }
                        return write!(f, "}}");
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
