//! The runtime: heap, symbol table, global environment, reader, macroexpander,
//! analyzer, and the value-model-aware primitives.
//!
//! `encode`/`decode` are the seam where the value axis meets everything else:
//! they box a non-immediate category and unbox on the way out, and `allocs`
//! counts the boxing so the micro-languages can *show* the cost. The reader
//! produces `Val` (code is data); `macroexpand` re-enters compiled code
//! through `CodeSpace::invoke`; `analyze` lowers a macroexpanded `Val` to `Ir`.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::code::CodeSpace;
use crate::ir::{Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::value::{Cat, Frame, HeapId, Locals, Obj, RawTag, Sym, Val};

/// A global binding. `is_macro` is what makes `macroexpand` treat a call head
/// specially — the single resolution table the compiler and runtime share.
pub struct Var {
    pub val: u64,
    pub is_macro: bool,
}

struct Specials {
    quote: Sym,
    if_: Sym,
    do_: Sym,
    def: Sym,
    defmacro: Sym,
    fn_: Sym,
    let_: Sym,
    amp: Sym,
}

pub struct Runtime<M: ValueModel> {
    pub heap: Vec<Obj>,
    /// Count of heap allocations. Instrumentation so the value axis is visible.
    pub allocs: u64,
    sym_names: Vec<String>,
    sym_ids: HashMap<String, Sym>,
    pub globals: HashMap<Sym, Var>,
    sf: Specials,
    prims: HashMap<Sym, Prim>,
    /// Compile-time lexical scope, innermost frame last. Used only during
    /// `analyze` to resolve names to `(up, idx)` slots; empty between forms.
    scope: Vec<Vec<Sym>>,
    _pd: PhantomData<fn() -> M>,
}

impl<M: ValueModel> Runtime<M> {
    pub fn new() -> Self {
        let mut rt = Runtime {
            heap: Vec::new(),
            allocs: 0,
            sym_names: Vec::new(),
            sym_ids: HashMap::new(),
            globals: HashMap::new(),
            sf: Specials {
                quote: 0,
                if_: 0,
                do_: 0,
                def: 0,
                defmacro: 0,
                fn_: 0,
                let_: 0,
                amp: 0,
            },
            prims: HashMap::new(),
            scope: Vec::new(),
            _pd: PhantomData,
        };
        rt.sf = Specials {
            quote: rt.intern("quote"),
            if_: rt.intern("if"),
            do_: rt.intern("do"),
            def: rt.intern("def"),
            defmacro: rt.intern("defmacro"),
            fn_: rt.intern("fn"),
            let_: rt.intern("let"),
            amp: rt.intern("&"),
        };
        use Prim::*;
        for (name, p) in [
            ("+", Add),
            ("-", Sub),
            ("*", Mul),
            ("<", Lt),
            ("=", Eq),
            ("list", List),
            ("cons", Cons),
            ("first", First),
            ("rest", Rest),
            ("nil?", IsNil),
            ("println", Println),
        ] {
            let s = rt.intern(name);
            rt.prims.insert(s, p);
        }
        rt
    }

    // ── symbols ─────────────────────────────────────────────
    pub fn intern(&mut self, s: &str) -> Sym {
        if let Some(&id) = self.sym_ids.get(s) {
            return id;
        }
        let id = self.sym_names.len() as Sym;
        self.sym_names.push(s.to_string());
        self.sym_ids.insert(s.to_string(), id);
        id
    }
    pub fn sym_name(&self, s: Sym) -> &str {
        &self.sym_names[s as usize]
    }

    // ── heap ────────────────────────────────────────────────
    pub fn alloc(&mut self, o: Obj) -> HeapId {
        self.allocs += 1;
        self.heap.push(o);
        (self.heap.len() - 1) as HeapId
    }

    /// Box a non-immediate category, encode an immediate one. THE value-axis
    /// seam: whether Int or Float takes the heap path is the model's call.
    pub fn encode(&mut self, v: Val) -> u64 {
        match v {
            Val::Int(i) => {
                if M::R::is_immediate(Cat::Int) {
                    M::R::enc_int(i)
                } else {
                    let id = self.alloc(Obj::BoxInt(i));
                    M::R::enc_ref(id)
                }
            }
            Val::Float(f) => {
                if M::R::is_immediate(Cat::Float) {
                    M::R::enc_float(f)
                } else {
                    let id = self.alloc(Obj::BoxFloat(f));
                    M::R::enc_ref(id)
                }
            }
            Val::Bool(b) => M::R::enc_bool(b),
            Val::Nil => M::R::enc_nil(),
            Val::Sym(s) => M::R::enc_sym(s),
            Val::Ref(id) => M::R::enc_ref(id),
        }
    }

    pub fn decode(&self, bits: u64) -> Val {
        match M::R::tag_of(bits) {
            RawTag::Int => Val::Int(M::R::imm_int(bits)),
            RawTag::Float => Val::Float(M::R::imm_float(bits)),
            RawTag::Bool => Val::Bool(M::R::as_bool(bits)),
            RawTag::Nil => Val::Nil,
            RawTag::Sym => Val::Sym(M::R::as_sym(bits)),
            RawTag::Ref => {
                let id = M::R::as_ref(bits);
                match &self.heap[id as usize] {
                    Obj::BoxInt(i) => Val::Int(*i),
                    Obj::BoxFloat(f) => Val::Float(*f),
                    _ => Val::Ref(id),
                }
            }
        }
    }

    // ── lists ───────────────────────────────────────────────
    pub fn cons(&mut self, head: u64, tail: u64) -> u64 {
        let id = self.alloc(Obj::Cons { head, tail });
        M::R::enc_ref(id)
    }
    pub fn as_cons(&self, bits: u64) -> Option<(u64, u64)> {
        if let RawTag::Ref = M::R::tag_of(bits) {
            if let Obj::Cons { head, tail } = &self.heap[M::R::as_ref(bits) as usize] {
                return Some((*head, *tail));
            }
        }
        None
    }
    pub fn list_to_vec(&self, mut bits: u64) -> Vec<u64> {
        let mut out = Vec::new();
        while let Some((h, t)) = self.as_cons(bits) {
            out.push(h);
            bits = t;
        }
        out
    }
    pub fn vec_to_list(&mut self, items: &[u64]) -> u64 {
        let mut tail = self.encode(Val::Nil);
        for &it in items.iter().rev() {
            tail = self.cons(it, tail);
        }
        tail
    }

    // ── equality / compare ──────────────────────────────────
    pub fn equal(&self, a: u64, b: u64) -> bool {
        match (self.decode(a), self.decode(b)) {
            (Val::Int(x), Val::Int(y)) => x == y,
            (Val::Float(x), Val::Float(y)) => x == y,
            (Val::Bool(x), Val::Bool(y)) => x == y,
            (Val::Nil, Val::Nil) => true,
            (Val::Sym(x), Val::Sym(y)) => x == y,
            (Val::Ref(_), Val::Ref(_)) => match (self.as_cons(a), self.as_cons(b)) {
                (Some((ha, ta)), Some((hb, tb))) => self.equal(ha, hb) && self.equal(ta, tb),
                _ => a == b, // identity for other objects
            },
            _ => false,
        }
    }
    fn num_lt(&self, a: u64, b: u64) -> bool {
        let f = |v: Val| match v {
            Val::Int(i) => i as f64,
            Val::Float(x) => x,
            _ => panic!("< on non-number"),
        };
        f(self.decode(a)) < f(self.decode(b))
    }

    // ── primitives (value-model fast paths live here) ───────
    pub fn prim(&mut self, op: Prim, args: &[u64]) -> u64 {
        match op {
            Prim::Add => self.arith(args[0], args[1], |a, b| a.wrapping_add(b), |a, b| a + b),
            Prim::Sub => self.arith(args[0], args[1], |a, b| a.wrapping_sub(b), |a, b| a - b),
            Prim::Mul => self.arith(args[0], args[1], |a, b| a.wrapping_mul(b), |a, b| a * b),
            Prim::Lt => {
                let r = self.num_lt(args[0], args[1]);
                self.encode(Val::Bool(r))
            }
            Prim::Eq => {
                let r = self.equal(args[0], args[1]);
                self.encode(Val::Bool(r))
            }
            Prim::List => self.vec_to_list(args),
            Prim::Cons => self.cons(args[0], args[1]),
            Prim::First => self.as_cons(args[0]).map(|(h, _)| h).unwrap_or_else(|| self.enc_nil()),
            Prim::Rest => self.as_cons(args[0]).map(|(_, t)| t).unwrap_or_else(|| self.enc_nil()),
            Prim::IsNil => {
                let r = matches!(self.decode(args[0]), Val::Nil);
                self.encode(Val::Bool(r))
            }
            Prim::Println => {
                let s = self.print(args[0]);
                println!("{s}");
                self.enc_nil()
            }
        }
    }

    fn enc_nil(&self) -> u64 {
        M::R::enc_nil()
    }

    /// The generic arithmetic path. Written ONCE, correct for every model:
    /// it tries each immediate numeric category in turn, and only falls to the
    /// boxing slow path when neither operand pair is an immediate number.
    ///
    ///   LowBit + ints  -> integer fast path fires,  0 allocations
    ///   NanBox + ints  -> no immediate int, slow path, boxes the result
    ///   NanBox + floats-> float fast path fires,     0 allocations
    ///   LowBit + floats-> no immediate float, slow path, boxes the result
    ///
    /// This is the exact shape of the toolkit's `dyn_add`, minus IR emission.
    fn arith(&mut self, a: u64, b: u64, iop: fn(i64, i64) -> i64, fop: fn(f64, f64) -> f64) -> u64 {
        if M::R::is_immediate(Cat::Int)
            && M::R::tag_of(a) == RawTag::Int
            && M::R::tag_of(b) == RawTag::Int
        {
            return M::R::enc_int(iop(M::R::imm_int(a), M::R::imm_int(b)));
        }
        if M::R::is_immediate(Cat::Float)
            && M::R::tag_of(a) == RawTag::Float
            && M::R::tag_of(b) == RawTag::Float
        {
            return M::R::enc_float(fop(M::R::imm_float(a), M::R::imm_float(b)));
        }
        let r = match (self.decode(a), self.decode(b)) {
            (Val::Int(x), Val::Int(y)) => Val::Int(iop(x, y)),
            (Val::Float(x), Val::Float(y)) => Val::Float(fop(x, y)),
            (Val::Int(x), Val::Float(y)) => Val::Float(fop(x as f64, y)),
            (Val::Float(x), Val::Int(y)) => Val::Float(fop(x, y as f64)),
            (va, vb) => panic!("arith on non-numbers: {va:?} {vb:?}"),
        };
        self.encode(r)
    }

    // ── printing ────────────────────────────────────────────
    pub fn print(&self, bits: u64) -> String {
        match self.decode(bits) {
            Val::Int(i) => i.to_string(),
            Val::Float(f) => {
                if f.is_finite() && f == f.trunc() {
                    format!("{f:.1}")
                } else {
                    format!("{f}")
                }
            }
            Val::Bool(b) => b.to_string(),
            Val::Nil => "nil".to_string(),
            Val::Sym(s) => self.sym_name(s).to_string(),
            Val::Ref(id) => match &self.heap[id as usize] {
                Obj::Cons { .. } => {
                    let items = self.list_to_vec(bits);
                    let inner: Vec<String> = items.iter().map(|&x| self.print(x)).collect();
                    format!("({})", inner.join(" "))
                }
                Obj::Str(s) => format!("\"{s}\""),
                Obj::Closure { .. } => "#<closure>".to_string(),
                Obj::BoxInt(i) => i.to_string(),
                Obj::BoxFloat(f) => format!("{f}"),
            },
        }
    }

    // ── reader: &str -> Vec<Val> (code is data) ─────────────
    pub fn read_all(&mut self, src: &str) -> Vec<u64> {
        let toks = tokenize(src);
        let mut p = 0;
        let mut out = Vec::new();
        while p < toks.len() {
            let (v, np) = self.read_form(&toks, p);
            out.push(v);
            p = np;
        }
        out
    }

    fn read_form(&mut self, toks: &[Tok], p: usize) -> (u64, usize) {
        match &toks[p] {
            Tok::LParen => {
                let mut items = Vec::new();
                let mut q = p + 1;
                while !matches!(toks.get(q), Some(Tok::RParen)) {
                    if q >= toks.len() {
                        panic!("unbalanced (");
                    }
                    let (v, nq) = self.read_form(toks, q);
                    items.push(v);
                    q = nq;
                }
                (self.vec_to_list(&items), q + 1)
            }
            Tok::RParen => panic!("unexpected )"),
            Tok::Quote => {
                let (v, nq) = self.read_form(toks, p + 1);
                let qs = self.sf.quote;
                let qsym = self.encode(Val::Sym(qs));
                let inner = self.vec_to_list(&[qsym, v]);
                (inner, nq)
            }
            Tok::Atom(a) => (self.read_atom(a), p + 1),
        }
    }

    fn read_atom(&mut self, a: &str) -> u64 {
        let v = if a == "nil" {
            Val::Nil
        } else if a == "true" {
            Val::Bool(true)
        } else if a == "false" {
            Val::Bool(false)
        } else if let Ok(i) = a.parse::<i64>() {
            Val::Int(i)
        } else if let Ok(f) = a.parse::<f64>() {
            Val::Float(f)
        } else {
            Val::Sym(self.intern(a))
        };
        self.encode(v)
    }

    // ── macroexpand: re-enters compiled code mid-compile ────
    //
    // The backend `cs` is passed as a value, not baked into `Runtime`'s type.
    // `cs` and `self` are disjoint borrows, so `cs.invoke(self, ...)` type-
    // checks with any backend — including a wrapping one (see `code::Traced`).
    pub fn macroexpand(&mut self, cs: &dyn CodeSpace<M>, mut form: u64) -> u64 {
        loop {
            let Some((head, _)) = self.as_cons(form) else {
                return form;
            };
            let Val::Sym(hs) = self.decode(head) else {
                return form;
            };
            let (is_macro, mfn) = match self.globals.get(&hs) {
                Some(v) => (v.is_macro, v.val),
                None => return form,
            };
            if !is_macro {
                return form;
            }
            let args = self.list_to_vec(form);
            // args[0] is the macro symbol; pass the UNEVALUATED argument forms.
            // `invoke` runs a closure while we are mid-compilation. This is
            // the re-entrancy the batch "declare-all-then-define-all" model
            // structurally forbids.
            form = cs.invoke(cs, self, mfn, &args[1..]);
        }
    }

    // ── analyze: macroexpanded Val -> Ir ────────────────────
    pub fn analyze(&mut self, cs: &dyn CodeSpace<M>, form: u64) -> Ir {
        let form = self.macroexpand(cs, form);
        match self.decode(form) {
            Val::Int(_) | Val::Float(_) | Val::Bool(_) | Val::Nil => Ir::Const(form),
            Val::Sym(s) => match self.resolve_local(s) {
                Some((up, idx)) => Ir::Local { up, idx },
                None => Ir::Global(s),
            },
            Val::Ref(_) => {
                if self.as_cons(form).is_some() {
                    self.analyze_list(cs, form)
                } else {
                    Ir::Const(form)
                }
            }
        }
    }

    fn analyze_list(&mut self, cs: &dyn CodeSpace<M>, form: u64) -> Ir {
        let items = self.list_to_vec(form);
        let head = items[0];
        if let Val::Sym(hs) = self.decode(head) {
            if hs == self.sf.quote {
                return Ir::Quote(items[1]);
            }
            if hs == self.sf.if_ {
                let c = self.analyze(cs, items[1]);
                let t = self.analyze(cs, items[2]);
                let e = if items.len() > 3 {
                    self.analyze(cs, items[3])
                } else {
                    Ir::Const(self.enc_nil())
                };
                return Ir::If(Box::new(c), Box::new(t), Box::new(e));
            }
            if hs == self.sf.do_ {
                let body: Vec<Ir> = items[1..].iter().map(|&f| self.analyze(cs, f)).collect();
                return Ir::Do(body);
            }
            if hs == self.sf.def {
                let Val::Sym(name) = self.decode(items[1]) else {
                    panic!("def: name must be a symbol");
                };
                let init = self.analyze(cs, items[2]);
                return Ir::Def {
                    name,
                    init: Box::new(init),
                    is_macro: false,
                };
            }
            if hs == self.sf.defmacro {
                let Val::Sym(name) = self.decode(items[1]) else {
                    panic!("defmacro: name must be a symbol");
                };
                let lam = self.analyze_fn(cs, items[2], &items[3..]);
                return Ir::Def {
                    name,
                    init: Box::new(lam),
                    is_macro: true,
                };
            }
            if hs == self.sf.fn_ {
                return self.analyze_fn(cs, items[1], &items[2..]);
            }
            if hs == self.sf.let_ {
                let bl = self.list_to_vec(items[1]);
                // Open a fresh let frame; each binding occupies the next slot
                // and becomes visible to later inits and the body (let*).
                self.scope.push(Vec::new());
                let mut inits = Vec::new();
                let mut i = 0;
                while i + 1 < bl.len() {
                    let Val::Sym(s) = self.decode(bl[i]) else {
                        panic!("let: binding name must be a symbol");
                    };
                    inits.push(self.analyze(cs, bl[i + 1]));
                    self.scope.last_mut().unwrap().push(s);
                    i += 2;
                }
                let body: Vec<Ir> = items[2..].iter().map(|&f| self.analyze(cs, f)).collect();
                self.scope.pop();
                return Ir::Let(inits, Box::new(Ir::Do(body)));
            }
            if let Some(&p) = self.prims.get(&hs) {
                let a: Vec<Ir> = items[1..].iter().map(|&f| self.analyze(cs, f)).collect();
                return Ir::Prim(p, a);
            }
        }
        let f = self.analyze(cs, head);
        let a: Vec<Ir> = items[1..].iter().map(|&f| self.analyze(cs, f)).collect();
        Ir::Call(Box::new(f), a)
    }

    /// Resolve a name to `(up, idx)` against the compile-time scope, innermost
    /// frame first (so inner bindings shadow outer). `None` => it is a global.
    fn resolve_local(&self, sym: Sym) -> Option<(u16, u16)> {
        for (up, frame) in self.scope.iter().rev().enumerate() {
            if let Some(idx) = frame.iter().rposition(|&s| s == sym) {
                return Some((up as u16, idx as u16));
            }
        }
        None
    }

    /// Analyze an `fn`/`defmacro` body under a fresh param frame. Params occupy
    /// slots `0..nparams`; a variadic rest param is the slot after them.
    fn analyze_fn(&mut self, cs: &dyn CodeSpace<M>, params_form: u64, body_forms: &[u64]) -> Ir {
        let (params, variadic) = self.parse_params(params_form);
        let nparams = params.len();
        let mut frame = params;
        if let Some(rest) = variadic {
            frame.push(rest);
        }
        self.scope.push(frame);
        let body: Vec<Ir> = body_forms.iter().map(|&f| self.analyze(cs, f)).collect();
        self.scope.pop();
        Ir::Lambda {
            nparams,
            variadic: variadic.is_some(),
            body: std::rc::Rc::new(Ir::Do(body)),
        }
    }

    fn parse_params(&self, form: u64) -> (Vec<Sym>, Option<Sym>) {
        let items = self.list_to_vec(form);
        let mut params = Vec::new();
        let mut variadic = None;
        let mut i = 0;
        while i < items.len() {
            let Val::Sym(s) = self.decode(items[i]) else {
                panic!("param must be a symbol");
            };
            if s == self.sf.amp {
                if let Some(&rest) = items.get(i + 1) {
                    let Val::Sym(r) = self.decode(rest) else {
                        panic!("variadic param must be a symbol");
                    };
                    variadic = Some(r);
                }
                break;
            }
            params.push(s);
            i += 1;
        }
        (params, variadic)
    }

    /// Build a callee's slot frame from evaluated args. Slots `0..nparams` are
    /// the positional args; a variadic rest arg is the slot after them, holding
    /// the collected list. Shared by every backend so the frame layout has one
    /// definition matching what `analyze` assigned.
    pub fn build_call_frame(
        &mut self,
        nparams: usize,
        variadic: bool,
        args: &[u64],
        env: Locals,
    ) -> Locals {
        let mut slots: Vec<u64> = Vec::with_capacity(nparams + variadic as usize);
        if variadic {
            assert!(
                args.len() >= nparams,
                "arity: expected at least {nparams}, got {}",
                args.len()
            );
            slots.extend_from_slice(&args[..nparams]);
            let restlist = self.vec_to_list(&args[nparams..]);
            slots.push(restlist);
        } else {
            assert!(
                args.len() == nparams,
                "arity: expected {nparams}, got {}",
                args.len()
            );
            slots.extend_from_slice(args);
        }
        Some(Rc::new(Frame { slots, parent: env }))
    }

    // ── top-level eval ──────────────────────────────────────
    pub fn eval_top(&mut self, cs: &dyn CodeSpace<M>, form: u64) -> u64 {
        let ir = self.analyze(cs, form);
        cs.eval_ir(cs, self, &ir, &None)
    }

    pub fn eval_str(&mut self, cs: &dyn CodeSpace<M>, src: &str) -> u64 {
        let forms = self.read_all(src);
        let mut last = self.enc_nil();
        for f in forms {
            last = self.eval_top(cs, f);
        }
        last
    }
}

// ── tokenizer ───────────────────────────────────────────────
enum Tok {
    LParen,
    RParen,
    Quote,
    Atom(String),
}

fn tokenize(src: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            c if c.is_whitespace() => i += 1,
            ';' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '(' | '[' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' | ']' => {
                out.push(Tok::RParen);
                i += 1;
            }
            '\'' => {
                out.push(Tok::Quote);
                i += 1;
            }
            _ => {
                let start = i;
                while i < chars.len()
                    && !chars[i].is_whitespace()
                    && !matches!(chars[i], '(' | ')' | '[' | ']' | '\'' | ';')
                {
                    i += 1;
                }
                out.push(Tok::Atom(chars[start..i].iter().collect()));
            }
        }
    }
    out
}
