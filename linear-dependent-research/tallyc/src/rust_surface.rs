//! The v1.0 SURFACE language (`docs/10-surface-syntax.md`): ML-flavored type
//! signatures, Rust-flavored terms. Elaborates to the `dep.rs` kernel, which
//! does all type-checking.
//!
//! ```text
//! enum Nat { Zero : Nat, Succ : Nat -> Nat }
//!
//! enum Vec (a : Type) : Nat -> Type {
//!     Nil  : Vec a Zero,
//!     Cons : {0 k : Nat} -> a -> Vec a k -> Vec a (Succ k),
//! }
//!
//! append : {0 a : Type} -> {0 m n : Nat} -> Vec a m -> Vec a n -> Vec a (add m n)
//! fn append(xs, ys) {
//!     match xs {
//!         Nil        => ys,
//!         Cons(h, t) => Cons(h, append(t, ys)),   // implicit a, k inferred; recursion ↦ IH
//!     }
//! }
//! ```
//!
//! Phase 2: **implicit `{..}` arguments**. A binder written with braces is erased
//! (multiplicity 0) and *inferred*; explicit `(..)` binders are passed as before.
//! Implicit constructor/datatype arguments are solved by matching the head's
//! result type against the expected type (first-order); the kernel re-checks
//! everything, so elaboration stays untrusted. `match` still compiles to the
//! dependent eliminator with the motive inferred from the return type and
//! structural recursion rewritten to the induction hypothesis.

use crate::dep::{self, Constructor, DataDecl, Signature, Term, Value};
use crate::mult::Mult;
use std::collections::HashMap;
use std::rc::Rc;

/// Sentinel de Bruijn level for an elaboration hole (metavariable). Real bound
/// variables never reach this range.
const HOLE_BASE: usize = usize::MAX / 2;

// ===========================================================================
// tokens + lexer
// ===========================================================================

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(u64),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Semi,
    Colon,
    Eq,
    FatArrow,
    Arrow,
    KwLet,
    KwFn,
    KwEnum,
    KwStruct,
    KwMatch,
    KwType,
    KwPostulate,
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if (c == '-' && i + 1 < b.len() && b[i + 1] as char == '-')
            || (c == '/' && i + 1 < b.len() && b[i + 1] as char == '/')
        {
            while i < b.len() && b[i] as char != '\n' {
                i += 1;
            }
        } else if c == '-' && i + 1 < b.len() && b[i + 1] as char == '>' {
            out.push(Tok::Arrow);
            i += 2;
        } else if c == '=' && i + 1 < b.len() && b[i + 1] as char == '>' {
            out.push(Tok::FatArrow);
            i += 2;
        } else if c.is_ascii_digit() {
            let s = i;
            while i < b.len() && (b[i] as char).is_ascii_digit() {
                i += 1;
            }
            out.push(Tok::Num(src[s..i].parse().map_err(|_| "bad number")?));
        } else if c.is_alphabetic() || c == '_' {
            let s = i;
            while i < b.len() && {
                let ch = b[i] as char;
                ch.is_alphanumeric() || ch == '_' || ch == '\''
            } {
                i += 1;
            }
            let w = &src[s..i];
            out.push(match w {
                "fn" => Tok::KwFn,
                "enum" => Tok::KwEnum,
                "struct" => Tok::KwStruct,
                "match" => Tok::KwMatch,
                "let" => Tok::KwLet,
                "Type" => Tok::KwType,
                "postulate" => Tok::KwPostulate,
                _ => Tok::Ident(w.to_string()),
            });
        } else {
            let t = match c {
                '(' => Tok::LParen,
                ')' => Tok::RParen,
                '{' => Tok::LBrace,
                '}' => Tok::RBrace,
                ',' => Tok::Comma,
                ';' => Tok::Semi,
                ':' => Tok::Colon,
                '=' => Tok::Eq,
                _ => return Err(format!("unexpected character {c:?}")),
            };
            out.push(t);
            i += 1;
        }
    }
    Ok(out)
}

// ===========================================================================
// surface AST
// ===========================================================================

#[derive(Clone, Debug)]
enum Ty {
    Var(String),
    Type,
    App(Box<Ty>, Box<Ty>),
    /// (mult, implicit?, name, domain, codomain)
    Arrow(Mult, bool, Option<String>, Box<Ty>, Box<Ty>),
}

#[derive(Clone, Debug)]
enum Tm {
    Var(String),
    Call(String, Vec<Tm>),
    Match(String, Vec<Arm>),
    /// `let (a, b) = e; body` — destructure a single-constructor value
    LetPair(Vec<String>, Box<Tm>, Box<Tm>),
}

#[derive(Clone, Debug)]
struct Arm {
    ctor: String,
    binders: Vec<String>,
    body: Tm,
}

#[derive(Clone, Debug)]
struct Binder {
    mult: Option<Mult>,
    implicit: bool,
    name: String,
    ty: Ty,
}

#[derive(Clone, Debug)]
enum Item {
    Sig(String, Ty),
    Fn(String, Vec<String>, Tm),
    Enum {
        name: String,
        params: Vec<Binder>,
        index_ty: Option<Ty>,
        variants: Vec<(String, Ty)>,
    },
    Struct {
        name: String,
        params: Vec<Binder>,
        fields: Vec<(String, Ty)>,
    },
    Postulate(String, Ty),
}

// ===========================================================================
// parser
// ===========================================================================

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }
    fn eat(&mut self, t: &Tok) -> Result<(), String> {
        if self.peek() == Some(t) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {:?}, found {:?}", t, self.peek()))
        }
    }
    fn ident(&mut self) -> Result<String, String> {
        match self.next() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(format!("expected an identifier, found {other:?}")),
        }
    }

    fn parse_program(&mut self) -> Result<Vec<Item>, String> {
        let mut items = Vec::new();
        while self.peek().is_some() {
            items.push(self.parse_item()?);
        }
        Ok(items)
    }

    fn parse_item(&mut self) -> Result<Item, String> {
        match self.peek() {
            Some(Tok::KwEnum) => self.parse_enum(),
            Some(Tok::KwStruct) => self.parse_struct(),
            Some(Tok::KwFn) => self.parse_fn(),
            Some(Tok::KwPostulate) => {
                self.next();
                let name = self.ident()?;
                self.eat(&Tok::Colon)?;
                let ty = self.parse_ty()?;
                Ok(Item::Postulate(name, ty))
            }
            Some(Tok::Ident(_)) => {
                let name = self.ident()?;
                self.eat(&Tok::Colon)?;
                let ty = self.parse_ty()?;
                Ok(Item::Sig(name, ty))
            }
            other => Err(format!("expected an item, found {other:?}")),
        }
    }

    /// Does a binder `( … : … )` or `{ … : … }` start here?
    fn binder_open(&self) -> Option<bool> {
        let open = match self.peek() {
            Some(Tok::LParen) => false,
            Some(Tok::LBrace) => true,
            _ => return None,
        };
        let k = self.pos + 1;
        let ok = ((matches!(self.toks.get(k), Some(Tok::Num(0)) | Some(Tok::Num(1)))
            || matches!(self.toks.get(k), Some(Tok::Ident(w)) if w == "w"))
            && matches!(self.toks.get(k + 1), Some(Tok::Ident(_)))
            && self.toks.get(k + 2) == Some(&Tok::Colon))
            || (matches!(self.toks.get(k), Some(Tok::Ident(_)))
                && self.toks.get(k + 1) == Some(&Tok::Colon));
        if ok {
            Some(open)
        } else {
            None
        }
    }

    fn parse_mult(&mut self) -> Option<Mult> {
        match self.peek() {
            Some(Tok::Num(0)) => {
                self.next();
                Some(Mult::Zero)
            }
            Some(Tok::Num(1)) => {
                self.next();
                Some(Mult::One)
            }
            Some(Tok::Ident(w)) if w == "w" => {
                self.next();
                Some(Mult::Omega)
            }
            _ => None,
        }
    }

    /// Parse a single binder. Returns the binder and whether more may follow.
    fn parse_binder(&mut self) -> Result<Binder, String> {
        let implicit = match self.next() {
            Some(Tok::LParen) => false,
            Some(Tok::LBrace) => true,
            other => return Err(format!("expected a binder, found {other:?}")),
        };
        let mult = self.parse_mult();
        let name = self.ident()?;
        self.eat(&Tok::Colon)?;
        let ty = self.parse_ty()?;
        self.eat(if implicit { &Tok::RBrace } else { &Tok::RParen })?;
        Ok(Binder { mult, implicit, name, ty })
    }

    fn parse_ty(&mut self) -> Result<Ty, String> {
        if self.binder_open().is_some() {
            let mut binders = Vec::new();
            while self.binder_open().is_some() {
                binders.push(self.parse_binder()?);
            }
            self.eat(&Tok::Arrow)?;
            let body = self.parse_ty()?;
            let mut out = body;
            for b in binders.into_iter().rev() {
                let m = b.mult.unwrap_or(if b.implicit { Mult::Zero } else { Mult::Omega });
                out = Ty::Arrow(m, b.implicit, Some(b.name), Box::new(b.ty), Box::new(out));
            }
            Ok(out)
        } else {
            let lhs = self.parse_ty_app()?;
            if self.peek() == Some(&Tok::Arrow) {
                self.next();
                let rhs = self.parse_ty()?;
                Ok(Ty::Arrow(Mult::Omega, false, None, Box::new(lhs), Box::new(rhs)))
            } else {
                Ok(lhs)
            }
        }
    }

    fn parse_ty_app(&mut self) -> Result<Ty, String> {
        let mut e = self.parse_ty_atom()?;
        while matches!(self.peek(), Some(Tok::Ident(_)) | Some(Tok::LParen) | Some(Tok::KwType)) {
            // stop before a new top-level signature `Ident :` (no statement
            // terminator, so an application would otherwise swallow it)
            if matches!(self.peek(), Some(Tok::Ident(_))) && self.toks.get(self.pos + 1) == Some(&Tok::Colon) {
                break;
            }
            let arg = self.parse_ty_atom()?;
            e = Ty::App(Box::new(e), Box::new(arg));
        }
        Ok(e)
    }

    fn parse_ty_atom(&mut self) -> Result<Ty, String> {
        match self.peek() {
            Some(Tok::KwType) => {
                self.next();
                Ok(Ty::Type)
            }
            Some(Tok::Ident(_)) => Ok(Ty::Var(self.ident()?)),
            Some(Tok::LParen) => {
                self.next();
                let t = self.parse_ty()?;
                self.eat(&Tok::RParen)?;
                Ok(t)
            }
            other => Err(format!("expected a type atom, found {other:?}")),
        }
    }

    fn parse_enum(&mut self) -> Result<Item, String> {
        self.eat(&Tok::KwEnum)?;
        let name = self.ident()?;
        let mut params = Vec::new();
        while self.binder_open() == Some(false) {
            params.push(self.parse_binder()?);
        }
        let index_ty = if self.peek() == Some(&Tok::Colon) {
            self.next();
            Some(self.parse_ty()?)
        } else {
            None
        };
        self.eat(&Tok::LBrace)?;
        let mut variants = Vec::new();
        while self.peek() != Some(&Tok::RBrace) {
            let cname = self.ident()?;
            self.eat(&Tok::Colon)?;
            let cty = self.parse_ty()?;
            variants.push((cname, cty));
            if self.peek() == Some(&Tok::Comma) {
                self.next();
            }
        }
        self.eat(&Tok::RBrace)?;
        Ok(Item::Enum { name, params, index_ty, variants })
    }

    fn parse_struct(&mut self) -> Result<Item, String> {
        self.eat(&Tok::KwStruct)?;
        let name = self.ident()?;
        let mut params = Vec::new();
        while self.binder_open() == Some(false) {
            params.push(self.parse_binder()?);
        }
        self.eat(&Tok::LBrace)?;
        let mut fields = Vec::new();
        while self.peek() != Some(&Tok::RBrace) {
            let fname = self.ident()?;
            self.eat(&Tok::Colon)?;
            let fty = self.parse_ty()?;
            fields.push((fname, fty));
            if self.peek() == Some(&Tok::Comma) {
                self.next();
            }
        }
        self.eat(&Tok::RBrace)?;
        Ok(Item::Struct { name, params, fields })
    }

    fn parse_fn(&mut self) -> Result<Item, String> {
        self.eat(&Tok::KwFn)?;
        let name = self.ident()?;
        self.eat(&Tok::LParen)?;
        let mut params = Vec::new();
        while self.peek() != Some(&Tok::RParen) {
            params.push(self.ident()?);
            if self.peek() == Some(&Tok::Comma) {
                self.next();
            }
        }
        self.eat(&Tok::RParen)?;
        self.eat(&Tok::LBrace)?;
        let body = self.parse_tm()?;
        self.eat(&Tok::RBrace)?;
        Ok(Item::Fn(name, params, body))
    }

    fn parse_tm(&mut self) -> Result<Tm, String> {
        match self.peek() {
            Some(Tok::KwMatch) => self.parse_match(),
            Some(Tok::KwLet) => {
                self.next();
                self.eat(&Tok::LParen)?;
                let mut names = Vec::new();
                while self.peek() != Some(&Tok::RParen) {
                    names.push(self.ident()?);
                    if self.peek() == Some(&Tok::Comma) {
                        self.next();
                    }
                }
                self.eat(&Tok::RParen)?;
                self.eat(&Tok::Eq)?;
                let rhs = self.parse_call()?;
                self.eat(&Tok::Semi)?;
                let body = self.parse_tm()?;
                Ok(Tm::LetPair(names, Box::new(rhs), Box::new(body)))
            }
            _ => self.parse_call(),
        }
    }

    fn parse_call(&mut self) -> Result<Tm, String> {
        let name = match self.next() {
            Some(Tok::Ident(s)) => s,
            Some(Tok::LParen) => {
                let t = self.parse_tm()?;
                self.eat(&Tok::RParen)?;
                return Ok(t);
            }
            other => return Err(format!("expected a term, found {other:?}")),
        };
        if self.peek() == Some(&Tok::LParen) {
            self.next();
            let mut args = Vec::new();
            while self.peek() != Some(&Tok::RParen) {
                args.push(self.parse_tm()?);
                if self.peek() == Some(&Tok::Comma) {
                    self.next();
                }
            }
            self.eat(&Tok::RParen)?;
            Ok(Tm::Call(name, args))
        } else {
            Ok(Tm::Var(name))
        }
    }

    fn parse_match(&mut self) -> Result<Tm, String> {
        self.eat(&Tok::KwMatch)?;
        let scrut = self.ident()?;
        self.eat(&Tok::LBrace)?;
        let mut arms = Vec::new();
        while self.peek() != Some(&Tok::RBrace) {
            let ctor = self.ident()?;
            let mut binders = Vec::new();
            if self.peek() == Some(&Tok::LParen) {
                self.next();
                while self.peek() != Some(&Tok::RParen) {
                    binders.push(self.ident()?);
                    if self.peek() == Some(&Tok::Comma) {
                        self.next();
                    }
                }
                self.eat(&Tok::RParen)?;
            }
            self.eat(&Tok::FatArrow)?;
            let body = self.parse_tm()?;
            arms.push(Arm { ctor, binders, body });
            if self.peek() == Some(&Tok::Comma) {
                self.next();
            }
        }
        self.eat(&Tok::RBrace)?;
        Ok(Tm::Match(scrut, arms))
    }
}

// ===========================================================================
// elaboration
// ===========================================================================

pub struct Program {
    pub sig: Signature,
    pub defs: Vec<(String, Term, Term)>,
}

/// per-constructor surface info (implicitness + binder names lost by the kernel)
#[derive(Clone)]
struct CtorInfo {
    data: String,
    param_implicit: Vec<bool>,
    arg_implicit: Vec<bool>,
    arg_names: Vec<Option<String>>,
}

struct Rec<'a> {
    fnname: &'a str,
    /// position of the scrutinee among the fn's EXPLICIT parameters
    scrut_pos: usize,
    fields: &'a HashMap<String, String>,
}

/// A typing context: each bound variable's name and type (as a Value).
#[derive(Clone, Default)]
struct Cx {
    names: Vec<String>,
    types: Vec<Value>,
}

impl Cx {
    fn len(&self) -> usize {
        self.names.len()
    }
    fn push(&mut self, name: String, ty: Value) {
        self.names.push(name);
        self.types.push(ty);
    }
    fn debruijn(&self, name: &str) -> Option<usize> {
        self.names.iter().rev().position(|s| s == name)
    }
    fn var_type(&self, name: &str) -> Option<Value> {
        let lvl = self.names.iter().rposition(|s| s == name)?;
        Some(self.types[lvl].clone())
    }
}

struct Elab {
    rc: Rc<Signature>,
    data_arity: HashMap<String, usize>,
    /// total argument count for a constructor (params + args)
    ctor_arity: HashMap<String, usize>,
    ctor_info: HashMap<String, CtorInfo>,
    defs: HashMap<String, (Term, Term)>,
    /// per-definition implicit flags (one per parameter)
    def_implicit: HashMap<String, Vec<bool>>,
}

fn neutral_env(n: usize) -> Vec<Value> {
    (0..n).map(dep::nvar).collect()
}

impl Elab {
    fn debruijn(scope: &[String], name: &str) -> Option<usize> {
        scope.iter().rev().position(|s| s == name)
    }

    fn ctor_has_implicits(&self, name: &str) -> bool {
        self.ctor_info
            .get(name)
            .map(|ci| ci.param_implicit.iter().chain(&ci.arg_implicit).any(|b| *b))
            .unwrap_or(false)
    }

    fn def_has_implicits(&self, name: &str) -> bool {
        self.def_implicit.get(name).map(|f| f.iter().any(|b| *b)).unwrap_or(false)
    }

    fn eval(&self, scope_len: usize, t: &Term) -> Value {
        dep::eval_rc(&self.rc, &neutral_env(scope_len), t)
    }

    /// Resolve a name applied to elaborated args (no implicit solving).
    fn resolve(&self, name: &str, args: Vec<Term>, scope: &[String], prefer_ctor: bool) -> Result<Term, String> {
        match name {
            "Eq" if args.len() == 3 => {
                let mut a = args.into_iter();
                return Ok(Term::Eq(Box::new(a.next().unwrap()), Box::new(a.next().unwrap()), Box::new(a.next().unwrap())));
            }
            "refl" if args.len() == 1 => return Ok(Term::Refl(Box::new(args.into_iter().next().unwrap()))),
            _ => {}
        }
        if let Some(i) = Self::debruijn(scope, name) {
            return Ok(args.into_iter().fold(Term::Var(i), |f, a| Term::App(Box::new(f), Box::new(a))));
        }
        let is_data = self.data_arity.contains_key(name);
        let is_ctor = self.ctor_arity.contains_key(name);
        let mk_data = |args: Vec<Term>| {
            let ar = self.data_arity[name];
            if args.len() != ar {
                Err(format!("datatype `{name}` expects {ar} argument(s), got {}", args.len()))
            } else {
                Ok(Term::Data(name.to_string(), args))
            }
        };
        let mk_ctor = |args: Vec<Term>| {
            if self.ctor_has_implicits(name) {
                return Err(format!("`{name}` has implicit arguments; it must be used where its type is known"));
            }
            let ar = self.ctor_arity[name];
            if args.len() != ar {
                Err(format!("constructor `{name}` expects {ar} argument(s), got {}", args.len()))
            } else {
                Ok(Term::Constr(name.to_string(), args))
            }
        };
        if is_data && is_ctor {
            return if prefer_ctor { mk_ctor(args) } else { mk_data(args) };
        }
        if is_data {
            return mk_data(args);
        }
        if is_ctor {
            return mk_ctor(args);
        }
        if let Some((body, ty)) = self.defs.get(name) {
            let head = Term::Ann(Box::new(body.clone()), Box::new(ty.clone()));
            return Ok(args.into_iter().fold(head, |f, a| Term::App(Box::new(f), Box::new(a))));
        }
        Err(format!("unbound name `{name}`"))
    }

    fn elab_ty(&self, t: &Ty, scope: &[String]) -> Result<Term, String> {
        match t {
            Ty::Type => Ok(Term::Type),
            Ty::Arrow(m, _, name, a, b) => {
                let ta = self.elab_ty(a, scope)?;
                let mut s2 = scope.to_vec();
                s2.push(name.clone().unwrap_or_else(|| "_".into()));
                let tb = self.elab_ty(b, &s2)?;
                Ok(Term::Pi(*m, Box::new(ta), Box::new(tb)))
            }
            Ty::Var(_) | Ty::App(_, _) => {
                let (head, args) = flatten_ty(t);
                let eargs = args.iter().map(|a| self.elab_ty(a, scope)).collect::<Result<Vec<_>, _>>()?;
                match head {
                    Ty::Var(name) => self.resolve(name, eargs, scope, false),
                    _ => Err("the head of a type application must be a name".into()),
                }
            }
        }
    }

    /// Elaborate a term in CHECK mode against `expected` (a Value in `scope`).
    fn check(&self, tm: &Tm, expected: &Value, cx: &Cx, rec: Option<&Rec>) -> Result<Term, String> {
        match tm {
            Tm::LetPair(names, e, body) => self.elab_let_pair(names, e, body, expected, cx, rec),
            Tm::Var(name) if self.ctor_has_implicits(name) => self.solve_ctor(name, &[], expected, cx, rec),
            Tm::Call(name, args) => {
                if let Some(r) = rec {
                    if name == r.fnname {
                        return self.ih_for(r, args, cx);
                    }
                }
                if self.ctor_has_implicits(name) {
                    return self.solve_ctor(name, args, expected, cx, rec);
                }
                if self.def_has_implicits(name) {
                    return Ok(self.solve_fn_call(name, args, Some(expected), cx, rec)?.0);
                }
                self.elab_tm(tm, cx, rec)
            }
            _ => self.elab_tm(tm, cx, rec),
        }
    }

    fn ih_for(&self, r: &Rec, args: &[Tm], cx: &Cx) -> Result<Term, String> {
        if args.len() <= r.scrut_pos {
            return Err(format!("recursive call to `{}` has too few arguments", r.fnname));
        }
        if let Tm::Var(v) = &args[r.scrut_pos] {
            if let Some(ih) = r.fields.get(v) {
                let i = cx.debruijn(ih).expect("ih var in scope");
                return Ok(Term::Var(i));
            }
        }
        Err(format!(
            "non-structural recursion: `{}` must recurse on a sub-component of the matched argument",
            r.fnname
        ))
    }

    /// Plain (no-implicit) elaboration; used for vars, functions, explicit ctors.
    fn elab_tm(&self, t: &Tm, cx: &Cx, rec: Option<&Rec>) -> Result<Term, String> {
        match t {
            Tm::Var(name) => self.resolve(name, vec![], &cx.names, true),
            Tm::Call(name, args) => {
                if let Some(r) = rec {
                    if name == r.fnname {
                        return self.ih_for(r, args, cx);
                    }
                }
                if self.def_has_implicits(name) {
                    return Ok(self.solve_fn_call(name, args, None, cx, rec)?.0);
                }
                let eargs = args.iter().map(|a| self.elab_tm(a, cx, rec)).collect::<Result<Vec<_>, _>>()?;
                self.resolve(name, eargs, &cx.names, true)
            }
            Tm::Match(_, _) => Err("nested `match` is not supported".into()),
            Tm::LetPair(_, _, _) => Err("`let` must appear in a checked position".into()),
        }
    }

    /// Infer the type of a simple argument (a variable or a definition reference,
    /// or a call/constructor that can itself be solved). Returns `(term, type)`.
    fn infer_arg(&self, t: &Tm, cx: &Cx, rec: Option<&Rec>) -> Result<(Term, Value), String> {
        match t {
            Tm::Var(name) => {
                if let Some(ty) = cx.var_type(name) {
                    let i = cx.debruijn(name).unwrap();
                    Ok((Term::Var(i), ty))
                } else if self.ctor_arity.get(name) == Some(&0) && !self.ctor_has_implicits(name) {
                    // a nullary constructor of a parameterless family (e.g. Zero)
                    let ci = &self.ctor_info[name];
                    let decl = self.rc.data(&ci.data).unwrap();
                    let ctor = decl.ctors.iter().find(|c| &c.name == name).unwrap();
                    let idxv: Vec<Value> =
                        ctor.idxs.iter().map(|t| dep::eval_rc(&self.rc, &[], t)).collect();
                    Ok((Term::Constr(name.clone(), vec![]), Value::VData(ci.data.clone(), idxv)))
                } else if let Some((body, ty)) = self.defs.get(name) {
                    let head = Term::Ann(Box::new(body.clone()), Box::new(ty.clone()));
                    Ok((head, dep::eval_rc(&self.rc, &[], ty)))
                } else {
                    Err(format!("cannot infer the type of `{name}` (Phase 2)"))
                }
            }
            Tm::Call(name, args) if self.defs.contains_key(name) => {
                self.solve_fn_call(name, args, None, cx, rec)
            }
            _ => Err("cannot infer the type of this argument (Phase 2)".into()),
        }
    }

    /// Solve a constructor's implicit arguments by matching its result type
    /// against `expected`, then elaborate the explicit arguments.
    fn solve_ctor(&self, cname: &str, user_args: &[Tm], expected: &Value, cx: &Cx, rec: Option<&Rec>) -> Result<Term, String> {
        let info = self.ctor_info[cname].clone();
        let decl = self.rc.data(&info.data).unwrap().clone();
        let ctor = decl.ctors.iter().find(|c| c.name == cname).unwrap().clone();
        let np = decl.params.len();
        let nargs = ctor.args.len();
        let total = np + nargs;
        let n = cx.len();

        let implicit_of = |pos: usize| {
            if pos < np {
                info.param_implicit[pos]
            } else {
                info.arg_implicit[pos - np]
            }
        };

        let nexplicit = (0..total).filter(|&p| !implicit_of(p)).count();
        if user_args.len() != nexplicit {
            return Err(format!(
                "constructor `{cname}` expects {nexplicit} explicit argument(s), got {}",
                user_args.len()
            ));
        }

        // fresh holes for every position; build the result type with holes
        let mut holes: Vec<Option<Value>> = (0..total).map(|_| None).collect();
        let hole_env: Vec<Value> = (0..total).map(|id| dep::nvar(HOLE_BASE + id)).collect();
        let mut result_args: Vec<Value> = (0..np).map(|p| hole_env[p].clone()).collect();
        for idx in &ctor.idxs {
            result_args.push(dep::eval_rc(&self.rc, &hole_env, idx));
        }
        let result = Value::VData(info.data.clone(), result_args);
        solve(&mut holes, &result, expected);

        // walk positions left to right, filling values + terms
        let mut env: Vec<Value> = Vec::with_capacity(total);
        let mut terms: Vec<Term> = Vec::with_capacity(total);
        let mut next_user = 0;
        for pos in 0..total {
            let dom_tm = if pos < np { &decl.params[pos].1 } else { &ctor.args[pos - np].1 };
            if implicit_of(pos) {
                let sol = holes[pos]
                    .clone()
                    .ok_or_else(|| format!("cannot infer implicit argument of `{cname}`"))?;
                terms.push(dep::quote_at(n, &sol));
                env.push(sol);
            } else {
                let dom_val = dep::eval_rc(&self.rc, &env, dom_tm);
                let arg_tm = self.check(&user_args[next_user], &dom_val, cx, rec)?;
                next_user += 1;
                let v = self.eval(n, &arg_tm);
                terms.push(arg_tm);
                env.push(v);
            }
        }
        Ok(Term::Constr(cname.to_string(), terms))
    }

    /// Solve a function's implicit arguments from the explicit arguments' types
    /// (and the expected result type, if known). Returns `(term, result_type)`.
    fn solve_fn_call(&self, fname: &str, user_args: &[Tm], expected: Option<&Value>, cx: &Cx, rec: Option<&Rec>) -> Result<(Term, Value), String> {
        let (body, fty) = self.defs[fname].clone();
        let flags = self.def_implicit[fname].clone();
        let total = flags.len();
        let nexplicit = flags.iter().filter(|b| !**b).count();
        if user_args.len() != nexplicit {
            return Err(format!(
                "`{fname}` expects {nexplicit} explicit argument(s), got {}",
                user_args.len()
            ));
        }
        // collect the parameter domains and the codomain from the function's type
        let mut domains: Vec<Term> = Vec::new();
        let mut t = fty.clone();
        for _ in 0..total {
            match t {
                Term::Pi(_, a, b) => {
                    domains.push(*a);
                    t = *b;
                }
                _ => return Err(format!("internal: `{fname}` has too few arrows")),
            }
        }
        let codomain = t;
        let n = cx.len();

        let mut holes: Vec<Option<Value>> = (0..total).map(|_| None).collect();
        let mut env: Vec<Value> = Vec::with_capacity(total); // arg values, in `cx`'s context
        let mut terms: Vec<Option<Term>> = vec![None; total];
        let mut next_user = 0;
        for pos in 0..total {
            let dom_val = dep::eval_rc(&self.rc, &env, &domains[pos]);
            if flags[pos] {
                env.push(dep::nvar(HOLE_BASE + pos)); // a hole, solved below
            } else {
                // try to infer the arg's type (to pin implicits); else check it
                // against the (current) domain — for args that pin nothing
                let arg_tm = match self.infer_arg(&user_args[next_user], cx, rec) {
                    Ok((arg_tm, arg_ty)) => {
                        solve(&mut holes, &dom_val, &arg_ty);
                        arg_tm
                    }
                    Err(_) => self.check(&user_args[next_user], &dom_val, cx, rec)?,
                };
                next_user += 1;
                let v = self.eval(n, &arg_tm);
                env.push(v);
                terms[pos] = Some(arg_tm);
            }
        }
        if let Some(exp) = expected {
            let cod_val = dep::eval_rc(&self.rc, &env, &codomain);
            solve(&mut holes, &cod_val, exp);
        }
        for pos in 0..total {
            if flags[pos] {
                let sol = holes[pos]
                    .clone()
                    .ok_or_else(|| format!("cannot infer implicit argument {pos} of `{fname}`"))?;
                terms[pos] = Some(dep::quote_at(n, &sol));
            }
        }
        // re-evaluate the codomain with solved holes for the returned type
        let solved_env: Vec<Value> = (0..total)
            .map(|p| {
                if flags[p] {
                    holes[p].clone().unwrap()
                } else {
                    env[p].clone()
                }
            })
            .collect();
        let result_ty = dep::eval_rc(&self.rc, &solved_env, &codomain);
        let head = Term::Ann(Box::new(body), Box::new(fty));
        let term = terms
            .into_iter()
            .map(Option::unwrap)
            .fold(head, |f, a| Term::App(Box::new(f), Box::new(a)));
        Ok((term, result_ty))
    }

    /// `let (a, b) = e; body` — destructure a single-constructor value `e` by
    /// eliminating it (a *linear* pair is consumed once and its fields bound at
    /// their declared multiplicities).
    fn elab_let_pair(&self, names: &[String], e: &Tm, body: &Tm, expected: &Value, cx: &Cx, rec: Option<&Rec>) -> Result<Term, String> {
        let (e_term, e_ty) = self.infer_arg(e, cx, rec)?;
        let (dname, dargs) = match &e_ty {
            Value::VData(d, a) => (d.clone(), a.clone()),
            _ => return Err("`let (..)` requires a value of a datatype".into()),
        };
        let decl = self.rc.data(&dname).unwrap().clone();
        if decl.ctors.len() != 1 {
            return Err(format!(
                "`let (..)` needs a single-constructor type, but `{dname}` has {} constructors",
                decl.ctors.len()
            ));
        }
        let arm = Arm { ctor: decl.ctors[0].name.clone(), binders: names.to_vec(), body: body.clone() };
        self.elab_nested_match(&e_term, &dname, &dargs, std::slice::from_ref(&arm), expected, cx, rec)
    }

    /// A `match`/`let` in CHECK mode (the motive is the constant expected type —
    /// a non-dependent elimination). The scrutinee is an arbitrary elaborated term.
    fn elab_nested_match(&self, e_term: &Term, dname: &str, dargs: &[Value], arms: &[Arm], expected: &Value, cx: &Cx, rec: Option<&Rec>) -> Result<Term, String> {
        let decl = self.rc.data(dname).unwrap().clone();
        let np = decl.params.len();
        let ni = decl.indices.len();
        let n = cx.len();
        let sparam_tms: Vec<Term> = dargs[..np].iter().map(|v| dep::quote_at(n, v)).collect();

        // constant motive: λ indices. λ _. expected
        let exp_tm = dep::quote_at(n, expected);
        let mut motive = dep::shift_term(ni + 1, &exp_tm);
        for _ in 0..(ni + 1) {
            motive = Term::Lam(Box::new(motive));
        }
        let motive_tm = motive.clone();

        let mut methods = Vec::with_capacity(decl.ctors.len());
        for ctor in &decl.ctors {
            let info = &self.ctor_info[&ctor.name];
            let nargs = ctor.args.len();
            let rec_fields: Vec<usize> = ctor
                .args
                .iter()
                .enumerate()
                .filter(|(_, (_, a))| matches!(a, Term::Data(d, _) if d == dname))
                .map(|(i, _)| i)
                .collect();
            let arm = arms
                .iter()
                .find(|a| a.ctor == ctor.name)
                .ok_or_else(|| format!("missing a case for `{}`", ctor.name))?;
            let nexplicit = info.arg_implicit.iter().filter(|b| !**b).count();
            if arm.binders.len() != nexplicit {
                return Err(format!(
                    "pattern `{}`: expected {nexplicit} binder(s), got {}",
                    ctor.name,
                    arm.binders.len()
                ));
            }
            let mut binder_names = Vec::new();
            let mut next_pat = 0;
            for j in 0..nargs {
                if info.arg_implicit[j] {
                    binder_names.push(info.arg_names[j].clone().unwrap_or_else(|| format!("$imp{j}")));
                } else {
                    binder_names.push(arm.binders[next_pat].clone());
                    next_pat += 1;
                }
            }
            for kk in 0..rec_fields.len() {
                binder_names.push(format!("$ih{kk}"));
            }
            let (binder_tys, _) =
                dep::elim_method_telescope(&self.rc, dname, &sparam_tms, &motive_tm, &ctor.name)?;
            let mut arm_cx = cx.clone();
            for (bn, (_, bty)) in binder_names.iter().zip(&binder_tys) {
                let v = dep::eval_rc(&self.rc, &neutral_env(arm_cx.len()), bty);
                arm_cx.push(bn.clone(), v);
            }
            let mut body = self.check(&arm.body, expected, &arm_cx, rec)?;
            for _ in 0..(nargs + rec_fields.len()) {
                body = Term::Lam(Box::new(body));
            }
            methods.push(body);
        }
        Ok(Term::Elim(dname.to_string(), Box::new(motive), methods, Box::new(e_term.clone())))
    }
}

/// First-order matching: bind holes in `holes` so that `pat` matches `target`.
/// Non-hole mismatches are ignored (the kernel re-checks the final term).
fn solve(holes: &mut Vec<Option<Value>>, pat: &Value, target: &Value) {
    let pat = deref(holes, pat);
    let target = deref(holes, target);
    if let Some(id) = hole_id(&pat) {
        holes[id] = Some(target);
        return;
    }
    if let Some(id) = hole_id(&target) {
        holes[id] = Some(pat);
        return;
    }
    match (&pat, &target) {
        (Value::VData(n1, a1), Value::VData(n2, a2)) if n1 == n2 && a1.len() == a2.len() => {
            for (x, y) in a1.iter().zip(a2) {
                solve(holes, x, y);
            }
        }
        (Value::VConstr(n1, a1), Value::VConstr(n2, a2)) if n1 == n2 && a1.len() == a2.len() => {
            for (x, y) in a1.iter().zip(a2) {
                solve(holes, x, y);
            }
        }
        (Value::VSuc(a), Value::VSuc(b)) => solve(holes, a, b),
        (Value::VNeu(n1), Value::VNeu(n2)) => solve_neu(holes, n1, n2),
        _ => {}
    }
}

/// Descend into a neutral spine (e.g. `Own ?a` vs `Own Nat`) to bind holes.
fn solve_neu(holes: &mut Vec<Option<Value>>, n1: &crate::dep::Neutral, n2: &crate::dep::Neutral) {
    use crate::dep::Neutral::NApp;
    if let (NApp(f1, a1), NApp(f2, a2)) = (n1, n2) {
        solve_neu(holes, f1, f2);
        solve(holes, a1, a2);
    }
}

fn hole_id(v: &Value) -> Option<usize> {
    match v {
        Value::VNeu(crate::dep::Neutral::NVar(l)) if *l >= HOLE_BASE => Some(*l - HOLE_BASE),
        _ => None,
    }
}

fn deref(holes: &[Option<Value>], v: &Value) -> Value {
    if let Some(id) = hole_id(v) {
        if let Some(sol) = &holes[id] {
            return deref(holes, sol);
        }
    }
    v.clone()
}

fn flatten_ty(t: &Ty) -> (&Ty, Vec<&Ty>) {
    let mut args = Vec::new();
    let mut head = t;
    while let Ty::App(f, a) = head {
        args.push(&**a);
        head = f;
    }
    args.reverse();
    (head, args)
}

fn decompose_ctor(mut t: Term, data: &str, nparams: usize) -> Result<(Vec<(Mult, Term)>, Vec<Term>), String> {
    let mut args = Vec::new();
    while let Term::Pi(m, a, b) = t {
        args.push((m, *a));
        t = *b;
    }
    let nargs = args.len();
    match t {
        Term::Data(name, spine) if name == data => {
            if spine.len() < nparams {
                return Err(format!("constructor of `{data}` returns too few arguments"));
            }
            for (i, sp) in spine.iter().take(nparams).enumerate() {
                let expected = Term::Var(nargs + nparams - 1 - i);
                if *sp != expected {
                    return Err(format!("a constructor of `{data}` must return it at its parameters (param {i})"));
                }
            }
            Ok((args, spine[nparams..].to_vec()))
        }
        other => Err(format!("a constructor of `{data}` must return `{data} …`, found {other:?}")),
    }
}

/// Peel a `Ty` arrow chain into (mult, implicit, name?, domain) entries.
fn peel_arrows(t: &Ty) -> (Vec<(Mult, bool, Option<String>, Ty)>, Ty) {
    let mut out = Vec::new();
    let mut t = t.clone();
    while let Ty::Arrow(m, imp, name, a, b) = t {
        out.push((m, imp, name, *a));
        t = *b;
    }
    (out, t)
}

fn count_index_pis(t: &Ty) -> Result<usize, String> {
    let (arrows, ret) = peel_arrows(t);
    match ret {
        Ty::Type => Ok(arrows.len()),
        _ => Err("a datatype's index telescope must end in `Type`".into()),
    }
}

// ---- the eliminator translation for a recursive `fn … { match p { … } }` ----

impl Elab {
    #[allow(clippy::too_many_arguments)]
    fn elab_match_body(
        &self,
        fnname: &str,
        fn_cx: &Cx, // the fn's parameters (names + types), in signature order
        explicit_params: &[String],
        param_tys: &[Ty], // surface type of each FULL param
        ret: &Ty,
        scrut: &str,
        arms: &[Arm],
    ) -> Result<Term, String> {
        let full_params: &[String] = &fn_cx.names;
        let full_pos = full_params.iter().position(|p| p == scrut)
            .ok_or_else(|| format!("`match` scrutinee `{scrut}` is not a parameter"))?;
        let explicit_pos = explicit_params.iter().position(|p| p == scrut)
            .ok_or_else(|| format!("`match` scrutinee `{scrut}` must be an explicit parameter"))?;

        let (dhead, dargs) = flatten_ty(&param_tys[full_pos]);
        let data = match dhead {
            Ty::Var(n) => n.clone(),
            _ => return Err("the scrutinee's type must be a datatype".into()),
        };
        let decl = self.rc.data(&data).ok_or_else(|| format!("`{data}` is not a datatype"))?.clone();
        let np = decl.params.len();
        let ni = decl.indices.len();
        if dargs.len() != np + ni {
            return Err(format!("scrutinee type `{data}` applied to the wrong number of arguments"));
        }
        let index_names: Vec<String> = dargs[np..]
            .iter()
            .map(|a| match a {
                Ty::Var(v) => Ok(v.clone()),
                _ => Err("a matched value's indices must be variables (Phase 1/2)".to_string()),
            })
            .collect::<Result<_, _>>()?;

        // motive = λ indices. λ scrut. ret      (rebinding shadows the params)
        let mut motive_scope = full_params.to_vec();
        motive_scope.extend(index_names.iter().cloned());
        motive_scope.push(scrut.to_string());
        let ret_term = self.elab_ty(ret, &motive_scope)?;
        let mut motive = ret_term;
        for _ in 0..(ni + 1) {
            motive = Term::Lam(Box::new(motive));
        }

        let sparam_tms: Vec<Term> = dargs[..np]
            .iter()
            .map(|a| self.elab_ty(a, full_params))
            .collect::<Result<_, _>>()?;

        let mut methods = Vec::with_capacity(decl.ctors.len());
        for ctor in &decl.ctors {
            let info = &self.ctor_info[&ctor.name];
            let nargs = ctor.args.len();
            let rec_fields: Vec<usize> = ctor
                .args
                .iter()
                .enumerate()
                .filter(|(_, (_, aty))| matches!(aty, Term::Data(dn, _) if *dn == data))
                .map(|(i, _)| i)
                .collect();
            let arm = arms
                .iter()
                .find(|a| a.ctor == ctor.name)
                .ok_or_else(|| format!("`match` is missing a case for `{}`", ctor.name))?;

            // build the method-binder scope: each ctor arg (implicit→its name,
            // explicit→the pattern's name), then one IH per recursive arg
            let nexplicit = info.arg_implicit.iter().filter(|b| !**b).count();
            if arm.binders.len() != nexplicit {
                return Err(format!(
                    "case `{}`: expected {nexplicit} pattern binder(s), got {}",
                    ctor.name,
                    arm.binders.len()
                ));
            }
            // binder names: each ctor arg (implicit→its name, explicit→pattern
            // name), then one IH per recursive arg
            let mut binder_names: Vec<String> = Vec::new();
            let mut next_pat = 0;
            for j in 0..nargs {
                if info.arg_implicit[j] {
                    binder_names.push(info.arg_names[j].clone().unwrap_or_else(|| format!("$imp{j}")));
                } else {
                    binder_names.push(arm.binders[next_pat].clone());
                    next_pat += 1;
                }
            }
            let mut fields: HashMap<String, String> = HashMap::new();
            for (kk, &fi) in rec_fields.iter().enumerate() {
                let ih = format!("$ih{kk}");
                fields.insert(binder_names[fi].clone(), ih.clone());
                binder_names.push(ih);
            }

            // typing context: the fn params, then the method binders with their
            // kernel types (from the eliminator method telescope)
            let motive_tm = motive.clone();
            let (binder_tys, ret_ty_tm) =
                dep::elim_method_telescope(&self.rc, &data, &sparam_tms, &motive_tm, &ctor.name)?;
            let mut arm_cx = fn_cx.clone();
            for (bn, (_, bty)) in binder_names.iter().zip(&binder_tys) {
                let v = dep::eval_rc(&self.rc, &neutral_env(arm_cx.len()), bty);
                arm_cx.push(bn.clone(), v);
            }
            let expected = dep::eval_rc(&self.rc, &neutral_env(arm_cx.len()), &ret_ty_tm);

            let r = Rec { fnname, scrut_pos: explicit_pos, fields: &fields };
            let mut body = self.check(&arm.body, &expected, &arm_cx, Some(&r))?;
            for _ in 0..(nargs + rec_fields.len()) {
                body = Term::Lam(Box::new(body));
            }
            methods.push(body);
        }
        for a in arms {
            if !decl.ctors.iter().any(|c| c.name == a.ctor) {
                return Err(format!("`{}` is not a constructor of `{data}`", a.ctor));
            }
        }
        let scrut_idx = Self::debruijn(full_params, scrut).unwrap();
        Ok(Term::Elim(data, Box::new(motive), methods, Box::new(Term::Var(scrut_idx))))
    }
}

/// Build the full ordered parameter list of a `fn` from its signature and the
/// `fn`'s explicit parameter names. Returns (full_names, full_tys, ret_ty).
fn split_signature(sig: &Ty, explicit: &[String]) -> Result<(Vec<String>, Vec<bool>, Vec<Ty>, Ty), String> {
    let mut names = Vec::new();
    let mut imps = Vec::new();
    let mut tys = Vec::new();
    let mut t = sig.clone();
    let mut next_explicit = 0;
    loop {
        match t {
            Ty::Arrow(_, implicit, name, a, b) => {
                let nm = if implicit {
                    name.clone().ok_or("implicit parameters must be named in the signature")?
                } else if next_explicit < explicit.len() {
                    let nm = explicit[next_explicit].clone();
                    next_explicit += 1;
                    nm
                } else {
                    // no more fn params: this arrow belongs to the return type
                    return Ok((names, imps, tys, Ty::Arrow(Mult::Omega, implicit, name, a, b)));
                };
                let rest = match &name {
                    Some(sn) if Some(&nm) != name.as_ref() => rename_ty(&b, sn, &nm),
                    _ => *b,
                };
                names.push(nm);
                imps.push(implicit);
                tys.push(*a);
                t = rest;
            }
            other => {
                if next_explicit != explicit.len() {
                    return Err("more `fn` parameters than the signature has arrows".into());
                }
                return Ok((names, imps, tys, other));
            }
        }
    }
}

fn rename_ty(t: &Ty, from: &str, to: &str) -> Ty {
    match t {
        Ty::Type => Ty::Type,
        Ty::Var(v) => Ty::Var(if v == from { to.to_string() } else { v.clone() }),
        Ty::App(f, a) => Ty::App(Box::new(rename_ty(f, from, to)), Box::new(rename_ty(a, from, to))),
        Ty::Arrow(m, i, name, a, b) => {
            let a2 = rename_ty(a, from, to);
            let b2 = if name.as_deref() == Some(from) { (**b).clone() } else { rename_ty(b, from, to) };
            Ty::Arrow(*m, *i, name.clone(), Box::new(a2), Box::new(b2))
        }
    }
}

pub fn elaborate(src: &str) -> Result<Program, String> {
    let toks = lex(src)?;
    let items = Parser { toks, pos: 0 }.parse_program()?;

    let mut elab = Elab {
        rc: Rc::new(Signature::default()),
        data_arity: HashMap::new(),
        ctor_arity: HashMap::new(),
        ctor_info: HashMap::new(),
        defs: HashMap::new(),
        def_implicit: HashMap::new(),
    };

    // pass B: arities
    for it in &items {
        match it {
            Item::Enum { name, params, index_ty, variants } => {
                let np = params.len();
                let ni = match index_ty {
                    Some(e) => count_index_pis(e)?,
                    None => 0,
                };
                elab.data_arity.insert(name.clone(), np + ni);
                for (cn, cty) in variants {
                    let (arrows, _) = peel_arrows(cty);
                    elab.ctor_arity.insert(cn.clone(), np + arrows.len());
                }
            }
            Item::Struct { name, params, fields } => {
                elab.data_arity.insert(name.clone(), params.len());
                elab.ctor_arity.insert(name.clone(), params.len() + fields.len());
            }
            _ => {}
        }
    }

    // pass C: datatypes → signature, recording ctor implicit/name info
    let mut sig = Signature::default();
    for it in &items {
        match it {
            Item::Enum { name, params, index_ty, variants } => {
                let np = params.len();
                // family parameters are always solved (implicit) at constructor use sites
                let param_implicit: Vec<bool> = vec![true; np];
                let mut scope = Vec::new();
                let mut kparams = Vec::new();
                for b in params {
                    let ty = elab.elab_ty(&b.ty, &scope)?;
                    kparams.push((b.mult.unwrap_or(Mult::Zero), ty));
                    scope.push(b.name.clone());
                }
                let mut indices = Vec::new();
                if let Some(e) = index_ty {
                    let mut it = elab.elab_ty(e, &scope)?;
                    while let Term::Pi(_, a, b) = it {
                        indices.push((Mult::Zero, *a));
                        it = *b;
                    }
                }
                let mut ctors = Vec::new();
                for (cn, cty) in variants {
                    let (arrows, _) = peel_arrows(cty);
                    let arg_implicit: Vec<bool> = arrows.iter().map(|(_, i, _, _)| *i).collect();
                    let arg_names: Vec<Option<String>> = arrows.iter().map(|(_, _, n, _)| n.clone()).collect();
                    let ct = elab.elab_ty(cty, &scope)?;
                    let (args, idxs) = decompose_ctor(ct, name, np)?;
                    elab.ctor_info.insert(cn.clone(), CtorInfo {
                        data: name.clone(),
                        param_implicit: param_implicit.clone(),
                        arg_implicit,
                        arg_names,
                    });
                    ctors.push(Constructor { name: cn.clone(), args, idxs });
                }
                sig.datas.push(DataDecl { name: name.clone(), params: kparams, indices, ctors });
            }
            Item::Struct { name, params, fields } => {
                let param_implicit: Vec<bool> = vec![true; params.len()];
                let mut scope = Vec::new();
                let mut kparams = Vec::new();
                for b in params {
                    let ty = elab.elab_ty(&b.ty, &scope)?;
                    kparams.push((b.mult.unwrap_or(Mult::Zero), ty));
                    scope.push(b.name.clone());
                }
                let mut args = Vec::new();
                let mut fscope = scope.clone();
                let mut arg_names = Vec::new();
                for (fname, fty) in fields {
                    let ty = elab.elab_ty(fty, &fscope)?;
                    args.push((Mult::Omega, ty));
                    arg_names.push(Some(fname.clone()));
                    fscope.push(fname.clone());
                }
                elab.ctor_info.insert(name.clone(), CtorInfo {
                    data: name.clone(),
                    param_implicit,
                    arg_implicit: vec![false; fields.len()],
                    arg_names,
                });
                let ctor = Constructor { name: name.clone(), args, idxs: vec![] };
                sig.datas.push(DataDecl { name: name.clone(), params: kparams, indices: vec![], ctors: vec![ctor] });
            }
            // postulates (opaque typed constants — the memory primitives, etc.),
            // processed in source order so each sees the earlier declarations
            Item::Postulate(name, pty) => {
                let ty_term = elab.elab_ty(pty, &[])?;
                let (arrows, _) = peel_arrows(pty);
                let flags: Vec<bool> = arrows.iter().map(|(_, i, _, _)| *i).collect();
                elab.defs.insert(name.clone(), (Term::Const(name.clone()), ty_term.clone()));
                elab.def_implicit.insert(name.clone(), flags);
                sig.postulates.push((name.clone(), ty_term));
            }
            _ => {}
        }
    }
    elab.rc = Rc::new(sig);

    // pass D: fns
    let mut sigs: HashMap<String, Ty> = HashMap::new();
    for it in &items {
        if let Item::Sig(n, t) = it {
            sigs.insert(n.clone(), t.clone());
        }
    }
    let mut out_defs = Vec::new();
    for it in &items {
        if let Item::Fn(name, params, body) = it {
            let sig = sigs.get(name).ok_or_else(|| format!("`fn {name}` has no type signature"))?.clone();
            let ty_term = elab.elab_ty(&sig, &[])?;
            let (full_names, full_imps, full_tys, ret) = split_signature(&sig, params)?;
            elab.def_implicit.insert(name.clone(), full_imps.clone());

            // the fn's typing context: each parameter's name + kernel type
            let mut fn_cx = Cx::default();
            for (i, (pn, pty)) in full_names.iter().zip(&full_tys).enumerate() {
                let kty = elab.elab_ty(pty, &full_names[..i])?;
                let v = dep::eval_rc(&elab.rc, &neutral_env(i), &kty);
                fn_cx.push(pn.clone(), v);
            }

            let body_inner = match body {
                Tm::Match(scrut, arms) => {
                    elab.elab_match_body(name, &fn_cx, params, &full_tys, &ret, scrut, arms)?
                }
                other => {
                    let expected = dep::eval_rc(&elab.rc, &neutral_env(full_names.len()), &elab.elab_ty(&ret, &full_names)?);
                    elab.check(other, &expected, &fn_cx, None)?
                }
            };
            let mut term = body_inner;
            for _ in 0..full_names.len() {
                term = Term::Lam(Box::new(term));
            }
            elab.defs.insert(name.clone(), (term.clone(), ty_term.clone()));
            out_defs.push((name.clone(), ty_term, term));
        }
    }

    Ok(Program { sig: (*elab.rc).clone(), defs: out_defs })
}

pub fn check_program(src: &str) -> Result<Program, Vec<String>> {
    let prog = elaborate(src).map_err(|e| vec![e])?;
    let mut diags = Vec::new();
    if let Err(e) = dep::check_signature(&prog.sig) {
        diags.push(format!("signature: {e}"));
    }
    for (name, ty, body) in &prog.defs {
        if let Err(e) = dep::check_closed_in(prog.sig.clone(), body, ty) {
            diags.push(format!("fn {name}: {e}"));
        }
    }
    if diags.is_empty() {
        Ok(prog)
    } else {
        Err(diags)
    }
}

impl Program {
    pub fn normalize(&self, name: &str) -> Option<Term> {
        let (_, _, body) = self.defs.iter().find(|(n, _, _)| n == name)?;
        Some(dep::normalize_closed_in(self.sig.clone(), body))
    }
}

pub fn pretty(t: &Term) -> String {
    crate::surface::pretty(t)
}

#[cfg(test)]
mod tests;
