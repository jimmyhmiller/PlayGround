//! A Clojure reader: text -> data. Distinguishes the collection literals the
//! core toolkit reader cannot — `[]` vectors, `{}` maps, `#{}` sets — plus
//! `:keywords`, `"strings"`, `\chars`, and `'quote`. Everything is built as a
//! toolkit heap value (list / `Obj::Vector` / tagged `Obj::Record`), so the GC
//! traces it and `analyze` self-evaluates the literal ones.
//!
//! Collection literals hold their sub-FORMS; the frontend's `expand` turns a
//! code-position vector/map/set into a constructor call (evaluating elements)
//! and leaves it literal under `quote`. Keywords/strings/chars/numbers are
//! self-evaluating (non-cons refs → `Ir::Const`).

use microlang::value::Sym;
use microlang::{Obj, Repr, Runtime, Val, ValueModel};

/// Type tags for the Clojure data types the frontend builds on `Obj::Record`.
/// Collections are LIST-BACKED (a record holding one field: a cons-list of its
/// contents), so every operation is simple recursive list code — no `apply`
/// (which is CEK-only) and no vector prims. `type-of` reports these tags, so
/// `clojure.core` predicates dispatch on them.
pub const KEYWORD: &str = "Keyword";
pub const VECTOR: &str = "Vector";
pub const MAP: &str = "Map";
pub const SET: &str = "Set";

pub fn read_all<M: ValueModel>(rt: &mut Runtime<M>, src: &str) -> Vec<u64> {
    let toks = tokenize(src);
    let mut p = Parser { toks, pos: 0 };
    let mut forms = Vec::new();
    while p.pos < p.toks.len() {
        forms.push(p.form(rt));
    }
    forms
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Open(char),  // ( [ {
    Close(char), // ) ] }
    HashBrace,   // #{
    HashParen,   // #(  anonymous-fn literal
    Quote,
    Backtick,      // `  syntax-quote
    Unquote,       // ~
    UnquoteSplice, // ~@
    Deref,         // @  -> (deref x)
    Caret,         // ^  metadata (parsed, then discarded for now)
    Str(String),
    Char(char),
    Atom(String), // symbol / keyword / number / nil / true / false
}

fn tokenize(src: &str) -> Vec<Tok> {
    let cs: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < cs.len() {
        let c = cs[i];
        match c {
            c if c.is_whitespace() || c == ',' => i += 1, // commas are whitespace in Clojure
            ';' => {
                while i < cs.len() && cs[i] != '\n' {
                    i += 1;
                }
            }
            '(' | '[' | '{' => {
                out.push(Tok::Open(c));
                i += 1;
            }
            ')' | ']' | '}' => {
                out.push(Tok::Close(c));
                i += 1;
            }
            '#' if i + 1 < cs.len() && cs[i + 1] == '{' => {
                out.push(Tok::HashBrace);
                i += 2;
            }
            '#' if i + 1 < cs.len() && cs[i + 1] == '(' => {
                // `#(...)` anonymous-fn literal: the `(` is consumed here so the
                // body reads as a normal parenthesized form under `HashParen`.
                out.push(Tok::HashParen);
                out.push(Tok::Open('('));
                i += 2;
            }
            '\'' => {
                out.push(Tok::Quote);
                i += 1;
            }
            '^' => {
                out.push(Tok::Caret);
                i += 1;
            }
            '`' => {
                out.push(Tok::Backtick);
                i += 1;
            }
            '@' => {
                out.push(Tok::Deref);
                i += 1;
            }
            '~' => {
                if i + 1 < cs.len() && cs[i + 1] == '@' {
                    out.push(Tok::UnquoteSplice);
                    i += 2;
                } else {
                    out.push(Tok::Unquote);
                    i += 1;
                }
            }
            '"' => {
                i += 1;
                let mut s = String::new();
                while i < cs.len() && cs[i] != '"' {
                    if cs[i] == '\\' && i + 1 < cs.len() {
                        i += 1;
                        s.push(match cs[i] {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            other => other,
                        });
                    } else {
                        s.push(cs[i]);
                    }
                    i += 1;
                }
                i += 1; // closing quote
                out.push(Tok::Str(s));
            }
            '\\' => {
                // a character literal: \a, \newline, \space, \tab
                i += 1;
                let start = i;
                while i < cs.len() && !cs[i].is_whitespace() && !matches!(cs[i], '(' | ')' | '[' | ']' | '{' | '}') {
                    i += 1;
                }
                let name: String = cs[start..i].iter().collect();
                let ch = match name.as_str() {
                    "newline" => '\n',
                    "space" => ' ',
                    "tab" => '\t',
                    "return" => '\r',
                    _ => name.chars().next().unwrap_or(' '),
                };
                out.push(Tok::Char(ch));
            }
            _ => {
                let start = i;
                while i < cs.len()
                    && !cs[i].is_whitespace()
                    && !matches!(cs[i], '(' | ')' | '[' | ']' | '{' | '}' | '\'' | '"' | ';' | ',' | '`' | '~' | '^')
                {
                    i += 1;
                }
                out.push(Tok::Atom(cs[start..i].iter().collect()));
            }
        }
    }
    out
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn form<M: ValueModel>(&mut self, rt: &mut Runtime<M>) -> u64 {
        let tok = self.toks[self.pos].clone();
        self.pos += 1;
        match tok {
            Tok::Open('(') => {
                let items = self.until(rt, ')');
                rt.vec_to_list(&items)
            }
            Tok::Open('[') => {
                let items = self.until(rt, ']');
                let lst = rt.vec_to_list(&items);
                record(rt, VECTOR, vec![lst])
            }
            Tok::Open('{') => {
                let items = self.until(rt, '}');
                let lst = rt.vec_to_list(&items);
                record(rt, MAP, vec![lst])
            }
            Tok::HashBrace => {
                let items = self.until(rt, '}');
                let lst = rt.vec_to_list(&items);
                record(rt, SET, vec![lst])
            }
            Tok::HashParen => {
                // `#(...)` -> `(fn [%1 %2 … & %&] (...))`. Read the body (the
                // following `(...)` form), scan it for the implicit params `%`,
                // `%1`…`%N`, `%&`, then synthesize the param vector. Bare `%` is
                // an alias for `%1`, so it is rewritten before emission.
                let mut body = self.form(rt);
                let mut max_n = 0usize;
                let mut has_rest = false;
                let mut has_bare = false;
                scan_pct(rt, body, &mut max_n, &mut has_rest, &mut has_bare);
                if has_bare {
                    if max_n < 1 {
                        max_n = 1;
                    }
                    let pct1 = sym(rt, "%1");
                    body = rewrite_bare_pct(rt, body, pct1);
                }
                let mut params = Vec::new();
                for i in 1..=max_n {
                    params.push(sym(rt, &format!("%{i}")));
                }
                if has_rest {
                    params.push(sym(rt, "&"));
                    params.push(sym(rt, "%&"));
                }
                let plist = rt.vec_to_list(&params);
                let pvec = record(rt, VECTOR, vec![plist]);
                let fnsym = sym(rt, "fn");
                rt.vec_to_list(&[fnsym, pvec, body])
            }
            Tok::Quote => self.wrap(rt, "quote"),
            Tok::Caret => {
                // ^meta target : we discard metadata for now, EXCEPT `:macro`
                // (as `^{:macro true}` or `^:macro`), which real core.clj uses to
                // define macros — wrap it so `def` can register the macro.
                let meta = self.form(rt);
                let target = self.form(rt);
                if is_macro_meta(rt, meta) {
                    let m = sym(rt, "-macro-meta");
                    rt.vec_to_list(&[m, target])
                } else if is_dynamic_meta(rt, meta) {
                    // `^:dynamic name` -> `(-dynamic-meta name)`; the compiler's
                    // `def` marks the var dynamic so refs read the binding stack.
                    let m = sym(rt, "-dynamic-meta");
                    rt.vec_to_list(&[m, target])
                } else if meta_has_key(rt, meta, "private") {
                    // `^:private name` -> `(-private-meta name)`; the compiler marks
                    // the var private (cross-namespace access errors).
                    let m = sym(rt, "-private-meta");
                    rt.vec_to_list(&[m, target])
                } else {
                    target
                }
            }
            Tok::Backtick => self.wrap(rt, "syntax-quote"),
            Tok::Deref => self.wrap(rt, "deref"),
            Tok::Unquote => self.wrap(rt, "unquote"),
            Tok::UnquoteSplice => self.wrap(rt, "unquote-splice"),
            Tok::Str(s) => alloc(rt, Obj::Str(s)),
            Tok::Char(c) => alloc(rt, Obj::Char(c)),
            Tok::Atom(a) => self.atom(rt, &a),
            Tok::Open(c) | Tok::Close(c) => panic!("reader: unexpected {c}"),
        }
    }

    /// `(head <next-form>)` — for reader macros `'`, `` ` ``, `~`, `~@`.
    fn wrap<M: ValueModel>(&mut self, rt: &mut Runtime<M>, head: &str) -> u64 {
        let inner = self.form(rt);
        let h = sym(rt, head);
        rt.vec_to_list(&[h, inner])
    }

    fn until<M: ValueModel>(&mut self, rt: &mut Runtime<M>, close: char) -> Vec<u64> {
        let mut items = Vec::new();
        loop {
            match self.toks.get(self.pos) {
                Some(Tok::Close(c)) if *c == close => {
                    self.pos += 1;
                    return items;
                }
                Some(_) => items.push(self.form(rt)),
                None => panic!("reader: unbalanced, expected {close}"),
            }
        }
    }

    fn atom<M: ValueModel>(&mut self, rt: &mut Runtime<M>, a: &str) -> u64 {
        if a == "nil" {
            return rt.encode(Val::Nil);
        }
        if a == "true" {
            return rt.encode(Val::Bool(true));
        }
        if a == "false" {
            return rt.encode(Val::Bool(false));
        }
        if let Ok(i) = a.parse::<i128>() {
            return rt.encode(Val::Int(i));
        }
        if let Ok(f) = a.parse::<f64>() {
            return rt.encode(Val::Float(f));
        }
        if let Some(kw) = a.strip_prefix(':') {
            // (keyword) -> a `Keyword` record holding the interned name symbol.
            let name = rt.intern(kw);
            let name_v = rt.encode(Val::Sym(name));
            return record(rt, KEYWORD, vec![name_v]);
        }
        sym(rt, a)
    }
}

fn sym<M: ValueModel>(rt: &mut Runtime<M>, n: &str) -> u64 {
    let s = rt.intern(n);
    rt.encode(Val::Sym(s))
}

fn field0<M: ValueModel>(rt: &Runtime<M>, v: u64, tag: &str) -> Option<u64> {
    if let Val::Ref(id) = rt.decode(v) {
        if let Obj::Record { type_id, fields } = &rt.heap()[id as usize] {
            if rt.sym_name(*type_id) == tag {
                return fields.first().copied();
            }
        }
    }
    None
}

/// Is `meta` a macro marker — `^:macro` / `^{:macro true}`?
fn is_macro_meta<M: ValueModel>(rt: &Runtime<M>, meta: u64) -> bool {
    meta_has_key(rt, meta, "macro")
}

/// Is `meta` a `^:dynamic` marker — `^:dynamic` / `^{:dynamic true}`?
fn is_dynamic_meta<M: ValueModel>(rt: &Runtime<M>, meta: u64) -> bool {
    meta_has_key(rt, meta, "dynamic")
}

/// Does the reader metadata form carry keyword `key` — either as the bare
/// keyword `^:key` or as a `^{:key true}` map entry?
fn meta_has_key<M: ValueModel>(rt: &Runtime<M>, meta: u64, key: &str) -> bool {
    if let Some(name) = field0(rt, meta, KEYWORD) {
        return matches!(rt.decode(name), Val::Sym(s) if rt.sym_name(s) == key);
    }
    if let Some(kvlist) = field0(rt, meta, MAP) {
        let kvs = rt.list_to_vec(kvlist);
        let mut i = 0;
        while i + 1 < kvs.len() {
            if let Some(kname) = field0(rt, kvs[i], KEYWORD) {
                if matches!(rt.decode(kname), Val::Sym(s) if rt.sym_name(s) == key) {
                    return true;
                }
            }
            i += 2;
        }
    }
    false
}

fn alloc<M: ValueModel>(rt: &mut Runtime<M>, o: Obj) -> u64 {
    let id = rt.alloc(o);
    <M::R as Repr>::enc_ref(id)
}

fn record<M: ValueModel>(rt: &mut Runtime<M>, ty: &str, fields: Vec<u64>) -> u64 {
    let type_id: Sym = rt.intern(ty);
    alloc(rt, Obj::Record { type_id, fields })
}

/// Walk a `#(...)` body form (cons lists + record contents) collecting the
/// implicit anonymous-fn params it mentions: the largest `%N`, whether `%&`
/// (rest) appears, and whether the bare `%` appears. Purely inspects — shared
/// borrows only.
fn scan_pct<M: ValueModel>(
    rt: &Runtime<M>,
    form: u64,
    max_n: &mut usize,
    has_rest: &mut bool,
    has_bare: &mut bool,
) {
    match rt.decode(form) {
        Val::Sym(s) => {
            let name = rt.sym_name(s);
            if name == "%&" {
                *has_rest = true;
            } else if name == "%" {
                *has_bare = true;
            } else if let Some(rest) = name.strip_prefix('%') {
                if let Ok(n) = rest.parse::<usize>() {
                    if n > *max_n {
                        *max_n = n;
                    }
                }
            }
        }
        Val::Ref(id) => {
            let (head, tail, fields) = match &rt.heap()[id as usize] {
                Obj::Cons { head, tail } => (Some(*head), Some(*tail), Vec::new()),
                Obj::Record { fields, .. } => (None, None, fields.clone()),
                _ => (None, None, Vec::new()),
            };
            if let Some(h) = head {
                scan_pct(rt, h, max_n, has_rest, has_bare);
            }
            if let Some(t) = tail {
                scan_pct(rt, t, max_n, has_rest, has_bare);
            }
            for f in fields {
                scan_pct(rt, f, max_n, has_rest, has_bare);
            }
        }
        _ => {}
    }
}

/// Rebuild `form`, replacing every bare `%` symbol with `pct1` (the `%1` symbol).
/// Cons/record structure is copied; leaves are returned unchanged.
fn rewrite_bare_pct<M: ValueModel>(rt: &mut Runtime<M>, form: u64, pct1: u64) -> u64 {
    match rt.decode(form) {
        Val::Sym(s) if rt.sym_name(s) == "%" => pct1,
        Val::Ref(id) => {
            let obj = rt.heap()[id as usize].clone();
            match obj {
                Obj::Cons { head, tail } => {
                    let h = rewrite_bare_pct(rt, head, pct1);
                    let t = rewrite_bare_pct(rt, tail, pct1);
                    alloc(rt, Obj::Cons { head: h, tail: t })
                }
                Obj::Record { type_id, fields } => {
                    let nf: Vec<u64> =
                        fields.iter().map(|&f| rewrite_bare_pct(rt, f, pct1)).collect();
                    alloc(rt, Obj::Record { type_id, fields: nf })
                }
                _ => form,
            }
        }
        _ => form,
    }
}
