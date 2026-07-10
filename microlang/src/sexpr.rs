//! An OPTIONAL s-expression frontend — NOT part of the core.
//!
//! The core toolkit is the axes: `Ir` (what a program means), `CodeSpace` (how it
//! runs), `ValueModel` (values), and `Runtime` (heap/GC/globals/execution). None
//! of that knows about s-expressions or "special forms". THIS module is a
//! convenience layer on top: a reader (`&str -> Val`) and `Sexpr` — a small,
//! opinionated Lisp compiler (`Val -> Ir`) with a fixed set of forms
//! (`quote/if/do/def/fn/let/set!/defmethod`). A frontend MAY use it to avoid
//! hand-building `Ir` (Scheme does); a frontend that owns its own surface
//! compiles to `Ir` directly and ignores this entirely (the Clojure frontend
//! does). It holds its OWN compile state and only borrows `&mut Runtime`, so the
//! core `Runtime` is free of it. (Uses only `Runtime`'s public API — root stack,
//! interning, decode/encode — so it could move to its own crate unchanged.)

use std::collections::HashMap;
use std::sync::Arc;

use crate::code::CodeSpace;
use crate::ir::{Ir, Prim};
use crate::model::ValueModel;
use crate::runtime::Runtime;
use crate::value::{Sym, Val};

/// The interned special-form symbols this little Lisp recognizes.
struct Specials {
    quote: Sym,
    if_: Sym,
    do_: Sym,
    def: Sym,
    defmethod: Sym,
    fn_: Sym,
    let_: Sym,
    set_: Sym,
    amp: Sym,
}

/// The s-expr compiler: its own scope, special forms, prim-name map, and dispatch
/// site counter. Borrows `&mut Runtime` per call for the shared state.
pub struct Sexpr {
    sf: Specials,
    prims: HashMap<Sym, Prim>,
    scope: Vec<Vec<Sym>>,
    next_site: usize,
}

impl Sexpr {
    pub fn new<M: ValueModel>(rt: &mut Runtime<M>) -> Self {
        let sf = Specials {
            quote: rt.intern("quote"),
            if_: rt.intern("if"),
            do_: rt.intern("do"),
            def: rt.intern("def"),
            defmethod: rt.intern("defmethod"),
            fn_: rt.intern("fn"),
            let_: rt.intern("let"),
            set_: rt.intern("set!"),
            amp: rt.intern("&"),
        };
        use Prim::*;
        let mut prims = HashMap::new();
        for (name, p) in [
            ("+", Add), ("-", Sub), ("*", Mul), ("<", Lt), ("=", Eq),
            ("list", List), ("cons", Cons), ("first", First), ("rest", Rest),
            ("nil?", IsNil), ("println", Println), ("gc", Gc),
            ("record", Record), ("field", Field), ("type-of", TypeOf), ("nfields", NFields), ("throw", Throw), ("%eq", Identical),
            ("%add", Add), ("%sub", Sub), ("%mul", Mul), ("%lt", Lt), ("%num-eq", Eq),
            ("%first", First), ("%rest", Rest), ("%cons", Cons),
            ("string-length", StrLen), ("char->integer", CharToInt), ("integer->char", IntToChar),
            ("vector", Vector), ("vector-ref", VectorRef), ("vector-set!", VectorSet),
            ("vector-length", VectorLen), ("values", Values), ("%values->list", ValuesToList),
            ("apply", Apply), ("%callec", CallEc), ("%callcc", CallCc),
            ("%reset", Reset), ("%shift", Shift),
        ] {
            let s = rt.intern(name);
            prims.insert(s, p);
        }
        Sexpr { sf, prims, scope: Vec::new(), next_site: 0 }
    }

    /// Read, then compile-and-run each form. The value of the last form.
    pub fn eval_str<M: ValueModel>(&mut self, rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, src: &str) -> u64 {
        let forms = read_all(rt, src);
        // Root the whole read buffer: a GC during an earlier form must not
        // reclaim (nor dangle a pointer to) later, not-yet-compiled source.
        let base = rt.root_depth();
        for &f in &forms {
            rt.push_root(f);
        }
        let mut last = rt.encode(Val::Nil);
        for k in 0..forms.len() {
            let f = rt.root_get(base + k);
            last = self.eval_top(rt, cs, f);
        }
        rt.truncate_roots(base);
        last
    }

    pub fn eval_top<M: ValueModel>(&mut self, rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, form: u64) -> u64 {
        let ir = self.analyze(rt, cs, form);
        cs.eval_ir(cs, rt, &ir, &None)
    }

    /// Lower one `Val` to `Ir`. Roots the form so a GC during a nested compile
    /// relocates it and children are re-read at their new addresses.
    pub fn analyze<M: ValueModel>(&mut self, rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, form: u64) -> Ir {
        let slot = rt.push_root(form);
        let r = match rt.decode(rt.root_get(slot)) {
            Val::Int(_) | Val::Float(_) | Val::Bool(_) | Val::Nil => {
                let f = rt.root_get(slot);
                Ir::Const(rt.intern_const(f))
            }
            Val::Sym(s) => match self.resolve_local(s) {
                Some((up, idx)) => Ir::Local { up, idx },
                None => Ir::Global(s),
            },
            Val::Ref(_) => {
                if rt.as_cons(rt.root_get(slot)).is_some() {
                    self.analyze_list(rt, cs, slot)
                } else {
                    let f = rt.root_get(slot);
                    Ir::Const(rt.intern_const(f))
                }
            }
        };
        rt.truncate_roots(slot);
        r
    }

    fn child<M: ValueModel>(&self, rt: &Runtime<M>, slot: usize, k: usize) -> u64 {
        rt.list_to_vec(rt.root_get(slot))[k]
    }
    fn child_count<M: ValueModel>(&self, rt: &Runtime<M>, slot: usize) -> usize {
        rt.list_to_vec(rt.root_get(slot)).len()
    }

    fn analyze_list<M: ValueModel>(&mut self, rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, slot: usize) -> Ir {
        let head = self.child(rt, slot, 0);
        if let Val::Sym(hs) = rt.decode(head) {
            if hs == self.sf.quote {
                let q = self.child(rt, slot, 1);
                return Ir::Quote(rt.intern_const(q));
            }
            if hs == self.sf.if_ {
                let c1 = self.child(rt, slot, 1);
                let c = self.analyze(rt, cs, c1);
                let t1 = self.child(rt, slot, 2);
                let t = self.analyze(rt, cs, t1);
                let e = if self.child_count(rt, slot) > 3 {
                    let e1 = self.child(rt, slot, 3);
                    self.analyze(rt, cs, e1)
                } else {
                    let nil = rt.encode(Val::Nil);
                    Ir::Const(rt.intern_const(nil))
                };
                return Ir::If(Box::new(c), Box::new(t), Box::new(e));
            }
            if hs == self.sf.do_ {
                let n = self.child_count(rt, slot);
                let mut body = Vec::new();
                for k in 1..n {
                    let f = self.child(rt, slot, k);
                    body.push(self.analyze(rt, cs, f));
                }
                return Ir::Do(body);
            }
            if hs == self.sf.def {
                let Val::Sym(name) = rt.decode(self.child(rt, slot, 1)) else {
                    panic!("def: name must be a symbol");
                };
                let i1 = self.child(rt, slot, 2);
                let init = self.analyze(rt, cs, i1);
                return Ir::Def { name, init: Box::new(init) };
            }
            if hs == self.sf.defmethod {
                let Val::Sym(name) = rt.decode(self.child(rt, slot, 1)) else {
                    panic!("defmethod: method name must be a symbol");
                };
                let Val::Sym(ty) = rt.decode(self.child(rt, slot, 2)) else {
                    panic!("defmethod: type must be a symbol");
                };
                let imp_form = self.child(rt, slot, 3);
                let imp = self.analyze(rt, cs, imp_form);
                return Ir::DefMethod { name, ty, imp: Box::new(imp) };
            }
            if hs == self.sf.fn_ {
                return self.analyze_fn(rt, cs, slot, 1, 2);
            }
            if hs == self.sf.let_ {
                return self.analyze_let(rt, cs, slot);
            }
            if hs == self.sf.set_ {
                let Val::Sym(name) = rt.decode(self.child(rt, slot, 1)) else {
                    panic!("set!: target must be a symbol");
                };
                let vform = self.child(rt, slot, 2);
                let val = Box::new(self.analyze(rt, cs, vform));
                return match self.resolve_local(name) {
                    Some((up, idx)) => Ir::SetLocal { up, idx, val },
                    None => Ir::SetGlobal { name, val },
                };
            }
            // A prim is a name's DEFAULT meaning; a local/global binding shadows it.
            let shadowed = self.resolve_local(hs).is_some() || rt.global_defined(hs);
            if !shadowed {
                if let Some(&p) = self.prims.get(&hs) {
                    let n = self.child_count(rt, slot);
                    let mut a = Vec::new();
                    for k in 1..n {
                        let f = self.child(rt, slot, k);
                        a.push(self.analyze(rt, cs, f));
                    }
                    return Ir::Prim(p, a);
                }
            }
            if rt.is_method_name(hs) {
                let site = self.fresh_site();
                let n = self.child_count(rt, slot);
                let mut a = Vec::new();
                for k in 1..n {
                    let f = self.child(rt, slot, k);
                    a.push(self.analyze(rt, cs, f));
                }
                return Ir::Dispatch { site, method: hs, args: a };
            }
        }
        let h = self.child(rt, slot, 0);
        let f = self.analyze(rt, cs, h);
        let n = self.child_count(rt, slot);
        let mut a = Vec::new();
        for k in 1..n {
            let c = self.child(rt, slot, k);
            a.push(self.analyze(rt, cs, c));
        }
        Ir::Call(Box::new(f), a)
    }

    fn analyze_let<M: ValueModel>(&mut self, rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, slot: usize) -> Ir {
        let binds_form = self.child(rt, slot, 1);
        let bslot = rt.push_root(binds_form);
        self.scope.push(Vec::new());
        let mut inits = Vec::new();
        let mut i = 0;
        while i + 1 < rt.list_to_vec(rt.root_get(bslot)).len() {
            let bl = rt.list_to_vec(rt.root_get(bslot));
            let Val::Sym(s) = rt.decode(bl[i]) else {
                panic!("let: binding name must be a symbol");
            };
            let initform = bl[i + 1];
            inits.push(self.analyze(rt, cs, initform));
            self.scope.last_mut().unwrap().push(s);
            i += 2;
        }
        rt.truncate_roots(bslot);
        let n = self.child_count(rt, slot);
        let mut body = Vec::new();
        for k in 2..n {
            let f = self.child(rt, slot, k);
            body.push(self.analyze(rt, cs, f));
        }
        self.scope.pop();
        Ir::Let(inits, Box::new(Ir::Do(body)))
    }

    fn resolve_local(&self, sym: Sym) -> Option<(u16, u16)> {
        for (up, frame) in self.scope.iter().rev().enumerate() {
            if let Some(idx) = frame.iter().rposition(|&s| s == sym) {
                return Some((up as u16, idx as u16));
            }
        }
        None
    }

    fn analyze_fn<M: ValueModel>(
        &mut self,
        rt: &mut Runtime<M>,
        cs: &dyn CodeSpace<M>,
        slot: usize,
        params_k: usize,
        body_start: usize,
    ) -> Ir {
        let params_form = self.child(rt, slot, params_k);
        let (params, variadic) = self.parse_params(rt, params_form);
        let nparams = params.len();
        let mut frame = params;
        if let Some(rest) = variadic {
            frame.push(rest);
        }
        self.scope.push(frame);
        let n = self.child_count(rt, slot);
        let mut body = Vec::new();
        for k in body_start..n {
            let f = self.child(rt, slot, k);
            body.push(self.analyze(rt, cs, f));
        }
        self.scope.pop();
        Ir::Lambda { nparams, variadic: variadic.is_some(), body: Arc::new(Ir::Do(body)) }
    }

    fn parse_params<M: ValueModel>(&self, rt: &Runtime<M>, form: u64) -> (Vec<Sym>, Option<Sym>) {
        let items = rt.list_to_vec(form);
        let mut params = Vec::new();
        let mut variadic = None;
        let mut i = 0;
        while i < items.len() {
            let Val::Sym(s) = rt.decode(items[i]) else {
                panic!("param must be a symbol");
            };
            if s == self.sf.amp {
                if let Some(&rest) = items.get(i + 1) {
                    let Val::Sym(r) = rt.decode(rest) else {
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

    fn fresh_site(&mut self) -> usize {
        let s = self.next_site;
        self.next_site += 1;
        s
    }
}

// ── convenience: one-shot read/analyze/eval with a transient `Sexpr` ────────

/// Read + compile + run a whole program. Value of the last form.
pub fn eval_str<M: ValueModel>(rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, src: &str) -> u64 {
    let mut s = Sexpr::new(rt);
    s.eval_str(rt, cs, src)
}

/// Compile one form to `Ir`.
pub fn analyze<M: ValueModel>(rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, form: u64) -> Ir {
    let mut s = Sexpr::new(rt);
    s.analyze(rt, cs, form)
}

// ── reader: &str -> Vec<Val> (code is data) ─────────────────────────────────

enum Tok {
    LParen,
    RParen,
    Quote,
    Atom(String),
}

pub fn read_all<M: ValueModel>(rt: &mut Runtime<M>, src: &str) -> Vec<u64> {
    let toks = tokenize(src);
    let mut p = 0;
    let mut out = Vec::new();
    while p < toks.len() {
        let (v, np) = read_form(rt, &toks, p);
        out.push(v);
        p = np;
    }
    out
}

fn read_form<M: ValueModel>(rt: &mut Runtime<M>, toks: &[Tok], p: usize) -> (u64, usize) {
    match &toks[p] {
        Tok::LParen => {
            let mut items = Vec::new();
            let mut q = p + 1;
            while !matches!(toks.get(q), Some(Tok::RParen)) {
                if q >= toks.len() {
                    panic!("unbalanced (");
                }
                let (v, nq) = read_form(rt, toks, q);
                items.push(v);
                q = nq;
            }
            (rt.vec_to_list(&items), q + 1)
        }
        Tok::RParen => panic!("unexpected )"),
        Tok::Quote => {
            let (v, nq) = read_form(rt, toks, p + 1);
            let qs = rt.intern("quote");
            let qsym = rt.encode(Val::Sym(qs));
            let inner = rt.vec_to_list(&[qsym, v]);
            (inner, nq)
        }
        Tok::Atom(a) => (read_atom(rt, a), p + 1),
    }
}

fn read_atom<M: ValueModel>(rt: &mut Runtime<M>, a: &str) -> u64 {
    let v = if a == "nil" {
        Val::Nil
    } else if a == "true" {
        Val::Bool(true)
    } else if a == "false" {
        Val::Bool(false)
    } else if let Ok(i) = a.parse::<i128>() {
        Val::Int(i)
    } else if let Ok(f) = a.parse::<f64>() {
        Val::Float(f)
    } else {
        Val::Sym(rt.intern(a))
    };
    rt.encode(v)
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
