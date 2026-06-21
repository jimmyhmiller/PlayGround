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
use crate::totality::{self, ArmInfo, Call as TCall, FnClauses, Totality};
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
    Plus,
    FatArrow,
    Arrow,
    KwLet,
    KwFn,
    KwEnum,
    KwStruct,
    KwMatch,
    KwType,
    KwPostulate,
    KwBuiltin,
    KwTotal,
    KwPartial,
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
        } else if c == '%' {
            // a pragma keyword, e.g. `%builtin`.
            let s = i + 1;
            let mut j = s;
            while j < b.len() && (b[j] as char).is_alphabetic() {
                j += 1;
            }
            match &src[s..j] {
                "builtin" => out.push(Tok::KwBuiltin),
                "total" => out.push(Tok::KwTotal),
                "partial" => out.push(Tok::KwPartial),
                other => return Err(format!("unknown pragma `%{other}`")),
            }
            i = j;
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
                '+' => Tok::Plus,
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Tm {
    Var(String),
    Call(String, Vec<Tm>),
    Match(String, Vec<Arm>),
    /// a built-in `Nat` literal, e.g. `0`, `5`, `1000000`.
    Lit(u64),
    /// built-in `Nat` addition `a + b`.
    Add(Box<Tm>, Box<Tm>),
    /// `let (a, b) = e; body` — destructure a single-constructor value
    LetPair(Vec<String>, Box<Tm>, Box<Tm>),
    /// `let x = e; body` — bind a single value (1a surface expressiveness).
    Let(String, Box<Tm>, Box<Tm>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Arm {
    pub(crate) ctor: String,
    pub(crate) binders: Vec<String>,
    pub(crate) body: Tm,
}

/// A `%total` / `%partial` annotation on a `fn`. `%total` DEMANDS the totality
/// checker certify the function (coverage + termination); failure is a hard
/// error. `%partial` documents an intentionally-partial (general-recursion) fn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TotAnnot {
    Total,
    Partial,
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
    /// `(name, params, body, totality annotation)`
    Fn(String, Vec<String>, Tm, Option<TotAnnot>),
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
    /// `%builtin Nat <Type>` — opt `<Type>` into the packed integer
    /// representation (it must be a Nat-shaped enum). Holds the type name.
    BuiltinNat(String),
}

// ===========================================================================
// parser
// ===========================================================================

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    fresh: usize,
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
            Some(Tok::KwBuiltin) => {
                // `%builtin Nat <TypeName>`
                self.next();
                let kind = self.ident()?;
                if kind != "Nat" && kind != "Natural" {
                    return Err(format!("unknown `%builtin` kind `{kind}` (expected `Nat`)"));
                }
                let ty = self.ident()?;
                Ok(Item::BuiltinNat(ty))
            }
            Some(Tok::KwTotal) => {
                self.next();
                self.parse_fn(Some(TotAnnot::Total))
            }
            Some(Tok::KwPartial) => {
                self.next();
                self.parse_fn(Some(TotAnnot::Partial))
            }
            Some(Tok::KwEnum) => self.parse_enum(),
            Some(Tok::KwStruct) => self.parse_struct(),
            Some(Tok::KwFn) => self.parse_fn(None),
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

    fn parse_fn(&mut self, annot: Option<TotAnnot>) -> Result<Item, String> {
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
        Ok(Item::Fn(name, params, body, annot))
    }

    fn parse_tm(&mut self) -> Result<Tm, String> {
        match self.peek() {
            Some(Tok::KwMatch) => self.parse_match(),
            Some(Tok::KwLet) => {
                self.next();
                // `let (a, b) = e; body`  (pair destructure)  OR  `let x = e; body`.
                if self.peek() == Some(&Tok::LParen) {
                    self.next();
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
                } else {
                    let name = self.ident()?;
                    self.eat(&Tok::Eq)?;
                    let rhs = self.parse_call()?;
                    self.eat(&Tok::Semi)?;
                    let body = self.parse_tm()?;
                    Ok(Tm::Let(name, Box::new(rhs), Box::new(body)))
                }
            }
            _ => self.parse_add(),
        }
    }

    /// `parse_call (+ parse_call)*` — left-associative built-in Nat addition.
    fn parse_add(&mut self) -> Result<Tm, String> {
        let mut lhs = self.parse_call()?;
        while self.peek() == Some(&Tok::Plus) {
            self.next();
            let rhs = self.parse_call()?;
            lhs = Tm::Add(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_call(&mut self) -> Result<Tm, String> {
        let name = match self.next() {
            Some(Tok::Ident(s)) => s,
            Some(Tok::Num(n)) => return Ok(Tm::Lit(n)),
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
        // the scrutinee may be an EXPRESSION (a call / paren), not just a var; a
        // non-var scrutinee is desugared to `let $s = <expr>; match $s { … }`.
        let scrut_tm = self.parse_call()?;
        let (scrut, bind): (String, Option<Tm>) = match scrut_tm {
            Tm::Var(v) => (v, None),
            other => {
                let s = format!("$m{}", self.fresh);
                self.fresh += 1;
                (s, Some(other))
            }
        };
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
        let m = Tm::Match(scrut.clone(), arms);
        // wrap the desugared scrutinee binding, if any.
        match bind {
            Some(e) => Ok(Tm::Let(scrut, Box::new(e), Box::new(m))),
            None => Ok(m),
        }
    }
}

// ===========================================================================
// elaboration
// ===========================================================================

pub struct Program {
    pub sig: Signature,
    pub defs: Vec<(String, Term, Term)>,
    /// per-`fn` totality status: `(name, is_total, reason_if_partial)`. Reported
    /// to the user; a `%total`-annotated `fn` that is not total is a hard error
    /// raised during elaboration (it never reaches here).
    pub totality: Vec<(String, bool, Option<String>)>,
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
    /// ACCUMULATOR-FOLD mode (Phase 1a′). `None` ⇒ verbatim-arg structural fold:
    /// a recursive call uses the induction hypothesis BARE (the IH is a value of
    /// the result type), and the verdict has already guaranteed every non-scrutinee
    /// argument is passed verbatim. `Some(acc_tys)` ⇒ the IH is itself a FUNCTION of
    /// the accumulators (`T₁ → … → T_K → R`), so a recursive call `f(smaller, e₁…e_K)`
    /// lowers to `App(…App(ih, e₁′)…, e_K′)`; `acc_tys` are the (closed) kernel types
    /// of the non-scrutinee params in param order, used to check each new acc arg.
    acc_tys: Option<&'a [Value]>,
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

/// The role a constructor of a `%builtin Nat` type plays.
#[derive(Clone, Copy, PartialEq)]
enum NatRole {
    Zero,
    Succ,
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
    /// type names opted into the packed built-in `Nat` (via `%builtin Nat T`).
    /// These are NOT registered as datatypes; they alias the kernel's `Nat`.
    nat_types: std::collections::HashSet<String>,
    /// constructor name → its `Nat` role, for the `%builtin Nat` types.
    nat_ctor: HashMap<String, NatRole>,
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
        // `%builtin Nat` types alias the kernel's packed `Nat`: the type name → the
        // `Nat` type, its nullary ctor → `Zero`, its successor ctor → `Suc`.
        if self.nat_types.contains(name) && args.is_empty() {
            return Ok(Term::Nat);
        }
        if let Some(role) = self.nat_ctor.get(name) {
            return match (role, args.len()) {
                (NatRole::Zero, 0) => Ok(Term::Zero),
                (NatRole::Succ, 1) => Ok(Term::Suc(Box::new(args.into_iter().next().unwrap()))),
                (NatRole::Zero, n) => Err(format!("`{name}` (Nat zero) takes no arguments, got {n}")),
                (NatRole::Succ, n) => Err(format!("`{name}` (Nat successor) takes 1 argument, got {n}")),
            };
        }
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
            // Surface `Type` is the base universe `Type 0`. (Universe POLYMORPHISM
            // in definitions is not yet surfaced — see `elab_ty` note / kernel
            // `Term::Type`. Until it is, a definition that genuinely needs a higher
            // universe must be written at the kernel level; the surface stays in
            // `Type 0`, which is sound, just less polymorphic.)
            Ty::Type => Ok(Term::Type(0)),
            Ty::Arrow(m, _, name, a, b) => {
                let ta = self.elab_ty(a, scope)?;
                let mut s2 = scope.to_vec();
                s2.push(name.clone().unwrap_or_else(|| "_".into()));
                let tb = self.elab_ty(b, &s2)?;
                // A binder of a LINEAR type (`Own`/`Σ[1]`) defaults to multiplicity 1,
                // not ω — the SAME fail-toward-linearity rule as the `let` binder. An ω
                // linear parameter would launder a double-free: `fn f(x : Own Nat) {
                // free(x); free(x) }` (param ω, used twice = ω ≤ ω) is otherwise
                // ACCEPTED. An EXPLICIT `(1 x : …)` is already 1; an abstract type
                // parameter (`a` for `{0 a : Type}`) stays ω — that is the §13
                // polymorphism case (a linear value flowing through an abstract-typed ω
                // param leaks, not double-frees; it needs real surface linear params,
                // Phase A). A bare `->` and an explicit `(ω …)` are indistinguishable in
                // the surface, but an ω linear binder is never legitimately wanted.
                let m = if *m == Mult::Omega && type_is_linear(&ta, &self.rc) { Mult::One } else { *m };
                Ok(Term::Pi(m, Box::new(ta), Box::new(tb)))
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
            Tm::Let(name, e, body) => self.elab_let(name, e, body, expected, cx, rec),
            // a `match` reached in CHECK position is a NESTED / expression case
            // split (non-recursive — the recursive fold is the fn body, handled in
            // pass D). Lower it to a non-dependent eliminator via elab_nested_match.
            Tm::Match(scrut, arms) => self.elab_case(scrut, arms, expected, cx, rec),
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
        match &args[r.scrut_pos] {
            // DIRECT recursive field: the matched-position argument is a strict-subterm
            // binder `v` whose induction hypothesis is in scope as `r.fields[v]`.
            Tm::Var(v) if r.fields.contains_key(v) => {
                let ih = Term::Var(cx.debruijn(&r.fields[v]).expect("ih var in scope"));
                match r.acc_tys {
                    // verbatim-arg fold: the IH IS the recursive result (the verdict has
                    // guaranteed every other argument is passed verbatim).
                    None => Ok(ih),
                    // accumulator fold (Phase 1a′): the IH is a function of the
                    // accumulators, so apply it to the NEW accumulator arguments — every
                    // position but the scrutinee, in param order — each checked against
                    // its accumulator type.
                    Some(acc_tys) => {
                        let acc_args: Vec<&Tm> = args
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != r.scrut_pos)
                            .map(|(_, a)| a)
                            .collect();
                        if acc_args.len() != acc_tys.len() {
                            return Err(format!(
                                "recursive call to `{}` has {} accumulator argument(s), expected {}",
                                r.fnname,
                                acc_args.len(),
                                acc_tys.len()
                            ));
                        }
                        let mut t = ih;
                        for (a, ty) in acc_args.iter().zip(acc_tys) {
                            let ea = self.check(a, ty, cx, Some(r))?;
                            t = Term::App(Box::new(t), Box::new(ea));
                        }
                        Ok(t)
                    }
                }
            }
            // HIGHER-ORDER recursive field (Phase 1b / well-founded recursion): the
            // matched-position argument is `f(callargs…)` where `f` is a higher-order
            // recursive field (a W-type child-function / `Acc`'s accessibility fn). The
            // kernel's IH for such a field is itself a function `λz…. elim (f z…)`, so a
            // recursive call `g(f(callargs…))` lowers to `App(…App(ih, c₁)…, cₙ)`.
            Tm::Call(f, callargs) if r.fields.contains_key(f) => {
                // VALUE-CORRECTNESS GUARD: the IH is a function OF the field-application
                // arguments, so the lowering computes `ih(callargs…)` — it does NOT
                // thread the recursive call's OTHER arguments. Each non-scrutinee
                // argument must therefore MATCH the field-application argument it
                // descends through; otherwise the written value is silently dropped, a
                // well-typed WRONG-VALUE term the (non-dependent) kernel re-check can't
                // catch. (For `f(y, h(y, prf))` the new `y` must equal `h`'s `y`; an
                // accumulator that varies independently is correctly REJECTED — a
                // higher-order fold's IH carries no extra accumulator.)
                let other: Vec<&Tm> =
                    args.iter().enumerate().filter(|(i, _)| *i != r.scrut_pos).map(|(_, a)| a).collect();
                for (i, oa) in other.iter().enumerate() {
                    if callargs.get(i) != Some(*oa) {
                        let pos = if i < r.scrut_pos { i } else { i + 1 };
                        return Err(format!(
                            "well-founded recursion in `{}`: argument #{pos} of the recursive \
                             call must match the accessibility-function argument it descends \
                             through (the induction hypothesis is a function of those \
                             arguments) — a different value there is silently dropped and \
                             would compute the wrong result",
                            r.fnname
                        ));
                    }
                }
                let mut t = Term::Var(cx.debruijn(&r.fields[f]).expect("ih var in scope"));
                for a in callargs {
                    let ea = self.elab_tm(a, cx, Some(r))?;
                    t = Term::App(Box::new(t), Box::new(ea));
                }
                Ok(t)
            }
            _ => Err(format!(
                "non-structural recursion: `{}` must recurse on a sub-component of the matched argument",
                r.fnname
            )),
        }
    }

    /// Plain (no-implicit) elaboration; used for vars, functions, explicit ctors.
    fn elab_tm(&self, t: &Tm, cx: &Cx, rec: Option<&Rec>) -> Result<Term, String> {
        match t {
            Tm::Lit(n) => Ok(Term::NatLit(*n)),
            Tm::Add(a, b) => Ok(Term::Add(
                Box::new(self.elab_tm(a, cx, rec)?),
                Box::new(self.elab_tm(b, cx, rec)?),
            )),
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
            Tm::Match(_, _) | Tm::Let(_, _, _) | Tm::LetPair(_, _, _) => {
                Err("`match`/`let` must appear in a checked position (annotate the \
                     surrounding expression's type)".into())
            }
        }
    }

    /// Infer the type of a simple argument (a variable or a definition reference,
    /// or a call/constructor that can itself be solved). Returns `(term, type)`.
    fn infer_arg(&self, t: &Tm, cx: &Cx, rec: Option<&Rec>) -> Result<(Term, Value), String> {
        match t {
            // built-in Nat: literals, addition, and the `%builtin Nat` intro forms
            // all have type `Nat` (packed), so they infer with no expected type.
            Tm::Lit(n) => Ok((Term::NatLit(*n), Value::VNat)),
            Tm::Add(a, b) => {
                let ta = self.check(a, &Value::VNat, cx, rec)?;
                let tb = self.check(b, &Value::VNat, cx, rec)?;
                Ok((Term::Add(Box::new(ta), Box::new(tb)), Value::VNat))
            }
            Tm::Var(name) if self.nat_ctor.get(name) == Some(&NatRole::Zero) => {
                Ok((Term::Zero, Value::VNat))
            }
            Tm::Call(name, args)
                if self.nat_ctor.get(name) == Some(&NatRole::Succ) && args.len() == 1 =>
            {
                let a = self.check(&args[0], &Value::VNat, cx, rec)?;
                Ok((Term::Suc(Box::new(a)), Value::VNat))
            }
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
            // a constructor application of a NON-indexed, NON-parameterized family
            // (e.g. `Succ(Zero)` : Nat). Its result type is the bare family, so we
            // can infer it with no expected type — and pinning it lets an enclosing
            // call solve an implicit through it (e.g. `alloc(Succ(Zero))`).
            Tm::Call(name, args) if self.ctor_info.contains_key(name) => {
                let info = self.ctor_info[name].clone();
                let decl = self.rc.data(&info.data).unwrap();
                if decl.params.is_empty() && decl.indices.is_empty() {
                    let ty = Value::VData(info.data.clone(), vec![]);
                    let tm = self.solve_ctor(name, args, &ty, cx, rec)?;
                    Ok((tm, ty))
                } else {
                    Err(format!(
                        "cannot infer the type of constructor `{name}` of an indexed or \
                         parameterized family without an expected type"
                    ))
                }
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
                // A solution that still mentions an unsolved hole means the implicit
                // wasn't fully determined (it unified with another open metavariable).
                // Report it cleanly rather than quoting it (which would underflow on
                // the synthetic hole de Bruijn level).
                if value_has_hole(&sol) {
                    return Err(format!(
                        "cannot infer implicit argument {pos} of `{fname}` (it depends on \
                         another argument's type that is not yet determined)"
                    ));
                }
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
    /// `let x = e; body` — infer `e`, bind `x`, check `body`, lower to the
    /// β-redex `(λx. body) e` (the kernel reduces and re-checks it). No kernel
    /// change — pure surface desugaring (Phase 1a).
    fn elab_let(&self, name: &str, e: &Tm, body: &Tm, expected: &Value, cx: &Cx, rec: Option<&Rec>) -> Result<Term, String> {
        let n = cx.len();
        let (e_term, e_ty) = self.infer_arg(e, cx, rec)?;
        let e_ty_tm = dep::quote_at(n, &e_ty);
        let mut body_cx = cx.clone();
        body_cx.push(name.to_string(), e_ty);
        let body_term = self.check(body, expected, &body_cx, rec)?;
        // lower to `(λx. body : (x:E) → BODY) e` — the lambda is ANNOTATED so the
        // kernel can check the β-redex (a bare lambda is not inferrable). The
        // codomain doesn't depend on x, so it is the (shifted) expected type.
        //
        // The binder's QUANTITY is the load-bearing soundness bit: a `let` binding a
        // LINEAR value (one whose type carries an `Own`/`Σ[1]` component) must bind at
        // `1`, so using it twice is `ω ⋢ 1` (a double-free) and dropping it is `0 ⋢ 1`
        // (a leak). An ω-binder here LAUNDERS linearity — it would accept
        // `let o = alloc(Zero); free(o); free(o)`. A copyable value (no linear
        // component) binds at `ω` so ordinary `let x = e; … x … x …` still works.
        // FAIL-SAFE toward linearity: `contains_linear` flags any reachable concrete
        // `Own`/`Σ[1]`; the abstract-type-parameter and field-hidden cases are the
        // §13 polymorphism corner, tightened when generic linear collections land.
        let exp_tm = dep::quote_at(n, expected);
        let binder_mult = if type_is_linear(&e_ty_tm, &self.rc) { Mult::One } else { Mult::Omega };
        let pi_ty = Term::Pi(binder_mult, Box::new(e_ty_tm), Box::new(dep::shift_term(1, &exp_tm)));
        let lam = Term::Ann(Box::new(Term::Lam(Box::new(body_term))), Box::new(pi_ty));
        Ok(Term::App(Box::new(lam), Box::new(e_term)))
    }

    /// A `match <scrut> { … }` in CHECK position — a NON-recursive case split on
    /// an in-scope variable (a parameter or a `let`-bound value), lowered to a
    /// non-dependent eliminator. The recursive structural fold stays the fn body
    /// (pass D / elab_match_body); this handles every OTHER `match` (nested, after
    /// a `let`, on an expression desugared through a `let`). (Phase 1a.)
    fn elab_case(&self, scrut: &str, arms: &[Arm], expected: &Value, cx: &Cx, rec: Option<&Rec>) -> Result<Term, String> {
        let (e_term, e_ty) = self.infer_arg(&Tm::Var(scrut.to_string()), cx, rec)?;
        match &e_ty {
            Value::VData(d, a) => {
                let a = a.clone();
                self.elab_nested_match(&e_term, d, &a, arms, expected, cx, rec)
            }
            // a nested/expression case split on a built-in `Nat` ⇒ a non-recursive
            // `NatCase` (no induction hypothesis); the constant motive is the
            // expected type. `rec` is threaded so a recursive self-call inside the
            // arm still maps to the OUTER fn's IH.
            Value::VNat => self.elab_nested_nat_case(&e_term, arms, expected, cx, rec),
            _ => Err(format!("`match {scrut}`: cannot case-split a value of this type")),
        }
    }

    fn elab_nested_nat_case(&self, scrut_tm: &Term, arms: &[Arm], expected: &Value, cx: &Cx, rec: Option<&Rec>) -> Result<Term, String> {
        let mut zero_arm = None;
        let mut succ_arm = None;
        for a in arms {
            match self.nat_ctor.get(&a.ctor) {
                Some(NatRole::Zero) if zero_arm.is_some() => {
                    return Err(format!("`match` has a redundant/duplicate arm for `{}`", a.ctor))
                }
                Some(NatRole::Succ) if succ_arm.is_some() => {
                    return Err(format!("`match` has a redundant/duplicate arm for `{}`", a.ctor))
                }
                Some(NatRole::Zero) => zero_arm = Some(a),
                Some(NatRole::Succ) => succ_arm = Some(a),
                None => return Err(format!("`{}` is not a constructor of the Nat scrutinee", a.ctor)),
            }
        }
        let zero_arm = zero_arm.ok_or("`match` on Nat is missing the zero case")?;
        let succ_arm = succ_arm.ok_or("`match` on Nat is missing the successor case")?;
        if succ_arm.binders.len() != 1 {
            return Err("the successor case binds exactly one predecessor".into());
        }
        // constant motive λ_. expected; methods checked at the same expected type.
        let n = cx.len();
        let motive = Term::Lam(Box::new(dep::shift_term(1, &dep::quote_at(n, expected))));
        let z = self.check(&zero_arm.body, expected, cx, rec)?;
        let mut succ_cx = cx.clone();
        succ_cx.push(succ_arm.binders[0].clone(), Value::VNat);
        let s_body = self.check(&succ_arm.body, expected, &succ_cx, rec)?;
        Ok(Term::NatCase(
            Box::new(motive),
            Box::new(z),
            Box::new(Term::Lam(Box::new(s_body))),
            Box::new(scrut_tm.clone()),
        ))
    }

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
            // USE-SITE LINEARITY (see `rebind_linear_fields`): re-bind each linear
            // field at 1 so a hidden `Own` (incl. a generic instantiated at `Own`)
            // can't be used twice in a nested/expression `match` either.
            let arm_body = rebind_linear_fields(&arm.body, &binder_names, &info.arg_implicit, &binder_tys, nargs, &self.rc);
            let mut body = self.check(&arm_body, expected, &arm_cx, rec)?;
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
        // reconcile the two `Nat` representations: a constructor's result index
        // `Succ n` (with `n` a hole) evaluates to `VSuc(hole)`, but a literal like
        // `Succ Zero` is PACKED by the evaluator into `VNatLit(1)`. Unify `Succ`-spine
        // against the packed literal so implicit indices solve from the expected type
        // (e.g. `ltZ : {0 n} -> Lt Zero (Succ n)` against `Lt Zero (Succ Zero)` ⇒
        // `n = Zero`). Exact (peels one `Succ` per step); never over-infers.
        (Value::VSuc(a), Value::VNatLit(k)) if *k > 0 => solve(holes, a, &Value::VNatLit(k - 1)),
        (Value::VNatLit(k), Value::VSuc(b)) if *k > 0 => solve(holes, &Value::VNatLit(k - 1), b),
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

/// Does `v` still mention an unsolved elaboration hole (a synthetic `NVar` at or
/// above `HOLE_BASE`)? Used to reject an under-determined implicit cleanly instead
/// of quoting the synthetic level (which would underflow). Covers the value shapes
/// an implicit solution can take; closure bodies are not traversed (a hole there
/// would not have escaped solving).
fn value_has_hole(v: &Value) -> bool {
    use crate::dep::{Neutral, Value as V};
    fn neu(n: &Neutral) -> bool {
        match n {
            Neutral::NVar(l) => *l >= HOLE_BASE,
            Neutral::NApp(f, a) => neu(f) || value_has_hole(a),
            Neutral::NAdd(a, b) => value_has_hole(a) || value_has_hole(b),
            Neutral::NFst(p) | Neutral::NSnd(p) => neu(p),
            Neutral::NNatElim(p, z, s, sc) => {
                value_has_hole(p) || value_has_hole(z) || value_has_hole(s) || neu(sc)
            }
            Neutral::NElim(_, m, ms, sc) => {
                value_has_hole(m) || ms.iter().any(value_has_hole) || neu(sc)
            }
            Neutral::NNatCase(p, z, s, sc) => {
                value_has_hole(p) || value_has_hole(z) || value_has_hole(s) || neu(sc)
            }
            // a `Fix` holds closed terms — no elaboration holes can survive in it.
            Neutral::NConst(_) | Neutral::NFix(_, _) => false,
        }
    }
    match v {
        V::VNeu(n) => neu(n),
        V::VData(_, args) | V::VConstr(_, args) => args.iter().any(value_has_hole),
        V::VSuc(a) | V::VRefl(a) => value_has_hole(a),
        V::VPair(a, b) => value_has_hole(a) || value_has_hole(b),
        V::VPi(_, a, _) | V::VSigma(_, a, _) => value_has_hole(a),
        V::VEq(a, b, c) => value_has_hole(a) || value_has_hole(b) || value_has_hole(c),
        V::VType(_) | V::VNat | V::VNatLit(_) | V::VLam(_) | V::VLamNative(_) => false,
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
    /// Try to discharge a `match` whose scrutinee type is EMPTY because its
    /// concrete index makes every constructor impossible (an absurd case, e.g.
    /// `Fin Zero`). Returns `Some(term)` if all constructors are absurd (the
    /// match must then have ZERO arms), else `None` (fall back to ordinary
    /// coverage). v1 scope: a single index whose constructor heads are decidably
    /// disjoint from the scrutinee's; otherwise CONSERVATIVELY `None` (reachable).
    ///
    /// THE DISCHARGE (no kernel change, no `Void`/prelude needed): the result type
    /// is computed through builtin `Nat` —
    ///   motive `m = λ i. λ _. NatCase i T (λ_. Nat)`   (T at index, Nat elsewhere)
    /// and each constructor method just returns `0 : Nat` (because at every
    /// constructor's `Succ`-headed index `m` computes `Nat`). The eliminator then
    /// has type `m idx scrut = NatCase idx T (λ_.Nat)`, which is `T` exactly when
    /// `idx` is the absurd (`Zero`) index. THE LINCHPIN: if the unifier WRONGLY
    /// called a reachable scrutinee absurd, `idx` is `Succ k`, `m` computes `Nat`,
    /// and the term has type `Nat ≠ T` — the kernel re-check REJECTS it. So the
    /// classifier is untrusted; it can only cause a rejection, never an unsound
    /// accept (FUTURE_WORK §4.2).
    fn try_absurd_match(
        &self,
        data: &str,
        decl: &DataDecl,
        dargs: &[&Ty],
        full_params: &[String],
        scrut: &str,
        ret: &Ty,
        arms: &[Arm],
    ) -> Result<Option<Term>, String> {
        let np = decl.params.len();
        if decl.indices.len() != 1 {
            return Ok(None); // v1: single-index families only
        }
        let p = full_params.len();
        // scrutinee parameter + index values, in the function-parameter context.
        let sparam_vals: Vec<Value> = dargs[..np]
            .iter()
            .map(|a| Ok(dep::eval_rc(&self.rc, &neutral_env(p), &self.elab_ty(a, full_params)?)))
            .collect::<Result<_, String>>()?;
        let sidx_tm = self.elab_ty(dargs[np], full_params)?;
        let sidx_nf = dep::quote_at(p, &dep::eval_rc(&self.rc, &neutral_env(p), &sidx_tm));
        let s_head = match ctor_head(&sidx_nf) {
            Some(h) => h,
            None => return Ok(None), // index not a known constructor ⇒ reachable
        };
        // every constructor's result index must be decidably disjoint from it.
        for ctor in &decl.ctors {
            let mut env = sparam_vals.clone();
            for j in 0..ctor.args.len() {
                env.push(dep::nvar(p + j));
            }
            let cidx_nf = dep::quote_at(p + ctor.args.len(), &dep::eval_rc(&self.rc, &env, &ctor.idxs[0]));
            let disjoint = matches!(ctor_head(&cidx_nf), Some(h) if h != s_head);
            if !disjoint {
                return Ok(None); // some constructor may be reachable ⇒ ordinary coverage
            }
        }
        // ALL constructors absurd ⇒ the type is empty. Reject any arms (the cases
        // are impossible and cannot be written), then synthesize the discharge.
        if !arms.is_empty() {
            return Err(format!(
                "`match` on `{data}` at this index is absurd (every constructor is \
                 impossible) — it must have NO arms; remove `{}`",
                arms[0].ctor
            ));
        }
        // The discharge below builds a `NatCase` over the index, so it requires a
        // `%builtin Nat` index. For a family indexed by a BOXED type the derived
        // term would fail the kernel re-check with a cryptic `expected Nat, found
        // Data(..)`; give the honest message instead (sound either way — this is a
        // completeness limit, not an unsoundness).
        if !matches!(decl.indices[0].1, Term::Nat) {
            return Err(format!(
                "`match` on `{data}` here is an absurd (empty) case, but absurd \
                 discharge currently requires the family to be indexed by a \
                 `%builtin Nat` — this one is indexed by a boxed type. Mark the index \
                 type `%builtin Nat`, or wait for the general Phase E2 discharge."
            ));
        }
        let t_term = self.elab_ty(ret, full_params)?;
        // ⚠️ BACKSTOP CAVEAT (read before extending to MIXED / per-constructor
        // absurd discharge): the Nat sentinel below makes the kernel-rejection
        // backstop T-DEPENDENT. At a mis-classified-REACHABLE index the discharge
        // term has type `Nat`, so the kernel rejects it ONLY when the match's
        // result type `T ≠ Nat`. That is SOUND HERE because v1 fires *only* when
        // EVERY constructor is decidably absurd (the loop above returns `None`
        // otherwise), so there is no reachable case to mis-classify and the
        // verdict does not lean on the backstop at all. But a per-constructor /
        // mixed discharge, where a reachable constructor COULD be wrongly called
        // absurd, would SILENTLY accept the mistake when `T = Nat`. Before that
        // extension, switch to a for-all-T sentinel (a fresh uninhabited type via
        // the two-step `→ Void`, `elim Void`), or prove the classifier sound
        // independently. Do NOT carry the Nat sentinel into mixed coverage.
        // motive = λ idx. λ _scrut. NatCase[λ_.Type] idx (shift² T) (λ_. Nat) idx
        let motive = Term::Lam(Box::new(Term::Lam(Box::new(Term::NatCase(
            Box::new(Term::Lam(Box::new(Term::Type(0)))),
            Box::new(dep::shift_term(2, &t_term)),
            Box::new(Term::Lam(Box::new(Term::Nat))),
            Box::new(Term::Var(1)), // the index binder
        )))));
        // one method per constructor: λ(args)…λ(IHs). 0   (: Nat = the Succ branch)
        let methods: Vec<Term> = decl
            .ctors
            .iter()
            .map(|ctor| {
                let nrec = ctor
                    .args
                    .iter()
                    .filter(|(_, a)| matches!(a, Term::Data(dn, _) if dn == data))
                    .count();
                let mut m = Term::NatLit(0);
                for _ in 0..(ctor.args.len() + nrec) {
                    m = Term::Lam(Box::new(m));
                }
                m
            })
            .collect();
        let scrut_idx = Self::debruijn(full_params, scrut)
            .ok_or_else(|| format!("`match` scrutinee `{scrut}` is not in scope"))?;
        Ok(Some(Term::Elim(
            data.to_string(),
            Box::new(motive),
            methods,
            Box::new(Term::Var(scrut_idx)),
        )))
    }

    /// Which of a `match` arm's pattern binders are STRICT SUBTERMS of the
    /// scrutinee — i.e. recursive fields of the matched constructor (arguments
    /// whose type is the family itself). These are the only legal structural
    /// decrease targets for the termination check. (Works uniformly for a
    /// `%builtin Nat`, whose `Succ` constructor's argument IS the family.)
    /// Returns `(direct, higher_order)`: the arm's pattern binders that are
    /// DIRECT recursive fields (strict subterms — first-order `data idxs`), and
    /// those that are HIGHER-ORDER recursive fields (`(z…) → data idxs`, e.g. a
    /// W-type's child-function or `Acc`'s accessibility fn — Phase 1b).
    fn smaller_binders(&self, data: &str, arm: &Arm) -> (Vec<String>, Vec<String>) {
        // `%builtin Nat` types are NOT in the kernel signature (they are natively
        // packed), so look up the constructor's recorded role: the successor's
        // single binder is the predecessor — the strict subterm. GATE this on the
        // SCRUTINEE'S actual type being a built-in Nat: the role table is keyed by
        // constructor NAME, so an unrelated enum that happens to reuse the name
        // `Succ` must NOT be treated as a Nat here (it would over-approximate the
        // strict-subterm set). With the gate, the structural verdict agrees with
        // the eliminator lowering instead of relying on its hard-error backstop.
        if self.nat_types.contains(data) {
            if let Some(role) = self.nat_ctor.get(&arm.ctor) {
                return match role {
                    NatRole::Succ => (arm.binders.first().cloned().into_iter().collect(), vec![]),
                    NatRole::Zero => (vec![], vec![]),
                };
            }
        }
        let decl = match self.rc.data(data) {
            Some(d) => d,
            None => return (vec![], vec![]),
        };
        let ctor = match decl.ctors.iter().find(|c| c.name == arm.ctor) {
            Some(c) => c,
            None => return (vec![], vec![]),
        };
        let info = match self.ctor_info.get(&arm.ctor) {
            Some(i) => i,
            None => return (vec![], vec![]),
        };
        let mut smaller = Vec::new();
        let mut ho_smaller = Vec::new();
        let mut explicit_idx = 0;
        for (ai, (_, aty)) in ctor.args.iter().enumerate() {
            let is_explicit = !info.arg_implicit.get(ai).copied().unwrap_or(false);
            if is_explicit {
                // a recursive field is DIRECT (`data idxs`, arity 0) or HIGHER-ORDER
                // (`(z…) → data idxs`, arity > 0). The kernel reads this off the
                // declared type via `rec_spine`; mirror it here.
                if let Some(arity) = dep::rec_field_arity(data, aty) {
                    if let Some(n) = arm.binders.get(explicit_idx) {
                        if arity == 0 {
                            smaller.push(n.clone());
                        } else {
                            ho_smaller.push(n.clone());
                        }
                    }
                }
                explicit_idx += 1;
            }
        }
        (smaller, ho_smaller)
    }

    /// Distil one `fn` into the totality analyzer's input. The analysis works in
    /// EXPLICIT-parameter space (surface recursive calls only list explicit
    /// arguments — implicits are inferred), so `params` is the user-written
    /// parameter list. The scrutinee's datatype is recovered from the full
    /// signature (`full_names`/`full_tys`, which interleave implicits).
    fn fn_clauses(
        &self,
        name: &str,
        params: &[String],
        full_names: &[String],
        full_tys: &[Ty],
        body: &Tm,
    ) -> FnClauses {
        match body {
            Tm::Match(scrut, arms) => {
                let scrut_pos = params.iter().position(|p| p == scrut);
                // the scrutinee's datatype: find it among ALL params (it may sit
                // after some implicits) and read its type head.
                let data = full_names
                    .iter()
                    .position(|p| p == scrut)
                    .and_then(|fi| match flatten_ty(&full_tys[fi]).0 {
                        Ty::Var(n) => Some(n.clone()),
                        _ => None,
                    });
                let arm_infos = arms
                    .iter()
                    .map(|arm| {
                        let (smaller, ho_smaller) =
                            data.as_ref().map(|d| self.smaller_binders(d, arm)).unwrap_or_default();
                        let mut calls = Vec::new();
                        collect_all_calls(&arm.body, &mut calls);
                        ArmInfo { smaller, ho_smaller, calls }
                    })
                    .collect();
                let scrut_is_nat =
                    data.as_ref().map(|d| self.nat_types.contains(d)).unwrap_or(false);
                FnClauses {
                    name: name.to_string(),
                    params: params.to_vec(),
                    scrut_pos,
                    arms: arm_infos,
                    body_calls: Vec::new(),
                    scrut_is_nat,
                }
            }
            other => {
                let mut calls = Vec::new();
                collect_all_calls(other, &mut calls);
                FnClauses {
                    name: name.to_string(),
                    params: params.to_vec(),
                    scrut_pos: None,
                    arms: Vec::new(),
                    body_calls: calls,
                    scrut_is_nat: false,
                }
            }
        }
    }

    /// Lower `match n { Zero => z, Succ(k) => s }` on a `%builtin Nat` scrutinee to
    /// the kernel's `NatElim(motive, z, λk.λih. s, n)`. A recursive call on the
    /// predecessor `k` inside the `Succ` arm becomes the induction hypothesis `ih`.
    #[allow(clippy::too_many_arguments)]
    fn elab_nat_match(
        &self,
        fnname: &str,
        fn_cx: &Cx,
        ret: &Ty,
        scrut: &str,
        arms: &[Arm],
        full_params: &[String],
        explicit_pos: usize,
    ) -> Result<Term, String> {
        // identify the Zero / Succ arms by their constructors' recorded roles.
        let mut zero_arm = None;
        let mut succ_arm = None;
        for a in arms {
            match self.nat_ctor.get(&a.ctor) {
                // a redundant/duplicate arm for the same role is rejected (E2).
                Some(NatRole::Zero) if zero_arm.is_some() => {
                    return Err(format!("`match` has a redundant/duplicate arm for `{}`", a.ctor))
                }
                Some(NatRole::Succ) if succ_arm.is_some() => {
                    return Err(format!("`match` has a redundant/duplicate arm for `{}`", a.ctor))
                }
                Some(NatRole::Zero) => zero_arm = Some(a),
                Some(NatRole::Succ) => succ_arm = Some(a),
                None => return Err(format!("`{}` is not a constructor of the Nat scrutinee", a.ctor)),
            }
        }
        let zero_arm = zero_arm.ok_or("`match` on Nat is missing the zero case")?;
        let succ_arm = succ_arm.ok_or("`match` on Nat is missing the successor case")?;
        // the zero case is nullary; the successor case binds exactly one predecessor.
        if !zero_arm.binders.is_empty() {
            return Err(format!(
                "the zero case binds no pattern variables, got {}",
                zero_arm.binders.len()
            ));
        }
        if succ_arm.binders.len() != 1 {
            return Err(format!(
                "the successor case binds exactly one predecessor, got {}",
                succ_arm.binders.len()
            ));
        }
        let pred_name = succ_arm.binders[0].clone();

        // motive = λn. ret   (ret elaborated with the scrutinee rebound to `n`).
        let mut motive_scope = full_params.to_vec();
        motive_scope.push(scrut.to_string());
        let ret_term = self.elab_ty(ret, &motive_scope)?;
        let motive = Term::Lam(Box::new(ret_term.clone()));

        let n = fn_cx.len();
        let base_env = neutral_env(n);
        // expected types come from the motive applied to Zero / k / (Suc k).
        let p_at = |scrut_val: Value| {
            let mut env = base_env.clone();
            env.push(scrut_val);
            dep::eval_rc(&self.rc, &env, &ret_term)
        };

        // zero method : P Zero
        let z_expected = p_at(Value::VNatLit(0));
        let z_method = self.check(&zero_arm.body, &z_expected, fn_cx, None)?;

        // succ method : λk. λih. body  with body : P (Suc k), ih : P k
        let k_val = dep::nvar(n); // the predecessor, a fresh neutral at level n
        let mut succ_cx = fn_cx.clone();
        succ_cx.push(pred_name.clone(), Value::VNat);
        let ih_ty = p_at(k_val.clone());
        let ih_name = "$ih".to_string();
        succ_cx.push(ih_name.clone(), ih_ty);
        // body : P (Suc k) — the motive at the successor of the predecessor `k`.
        let s_expected = p_at(Value::VSuc(Box::new(k_val)));
        let mut fields: HashMap<String, String> = HashMap::new();
        fields.insert(pred_name, ih_name);
        let r = Rec { fnname, scrut_pos: explicit_pos, fields: &fields, acc_tys: None };
        let body = self.check(&succ_arm.body, &s_expected, &succ_cx, Some(&r))?;
        let s_method = Term::Lam(Box::new(Term::Lam(Box::new(body))));

        let scrut_idx = Self::debruijn(full_params, scrut).unwrap();
        Ok(Term::NatElim(
            Box::new(motive),
            Box::new(z_method),
            Box::new(s_method),
            Box::new(Term::Var(scrut_idx)),
        ))
    }

    /// Does this `Nat`-scrutinee `fn` recurse ACCUMULATOR-style — i.e. does some
    /// recursive call vary a non-scrutinee argument (rather than pass it verbatim)?
    /// If so it must use the function-typed-motive lowering (`elab_nat_match_acc`),
    /// not the plain verbatim fold (`elab_nat_match`). Detection mirrors the
    /// totality verdict's accumulator check, in EXPLICIT-parameter space (surface
    /// recursive calls list only explicit arguments).
    fn is_acc_fold(&self, fnname: &str, scrut: &str, explicit_params: &[String], arms: &[Arm]) -> bool {
        let sp = match explicit_params.iter().position(|p| p == scrut) {
            Some(p) => p,
            None => return false,
        };
        let mut calls = Vec::new();
        for arm in arms {
            collect_all_calls(&arm.body, &mut calls);
        }
        for c in &calls {
            if c.callee != fnname || c.args.len() != explicit_params.len() {
                continue;
            }
            for (i, a) in c.args.iter().enumerate() {
                if i == sp {
                    continue;
                }
                let verbatim = matches!(a, Tm::Var(v) if v == &explicit_params[i]);
                if !verbatim {
                    return true;
                }
            }
        }
        false
    }

    /// Lower an ACCUMULATOR-style `match m { Zero => …, Succ(k) => f(k, e₁…e_K) }`
    /// fold on a `%builtin Nat` scrutinee (Phase 1a′). The standard fold-into-function
    /// encoding: the eliminator's motive is a FUNCTION TYPE, so the induction
    /// hypothesis is itself a function of the accumulators and a recursive call applies
    /// it to the NEW accumulators.
    ///
    ///   motive = λscrut'. (T₁ → … → T_K → R)          -- a `Nat → Type`
    ///   z      = λa₁…λa_K. <Zero arm body>            -- : T₁ → … → T_K → R
    ///   s      = λk. λih. λa₁…λa_K. <Succ arm body>    -- ih : T₁ → … → T_K → R
    ///   body   = (NatElim motive z s scrut) a₁ … a_K   -- apply to the real accs
    ///
    /// where a recursive call `f(k, e₁…e_K)` in the Succ arm becomes
    /// `App(…App(ih, e₁′)…, e_K′)` (see `ih_for`'s accumulator mode). The kernel
    /// re-checks this function-typed-motive `NatElim`, so a lowering bug can only
    /// produce a rejected program, never an unsound one.
    ///
    /// v1 RESTRICTIONS (errored clearly otherwise): all params explicit; the
    /// accumulator types and the result type `R` are NON-DEPENDENT (independent of
    /// the scrutinee and the accumulators — true for `div`/`gcd`/`lt`/`sub`), hence
    /// CLOSED kernel types. The accumulators are every non-scrutinee param, in order.
    #[allow(clippy::too_many_arguments)]
    fn elab_nat_match_acc(
        &self,
        fnname: &str,
        fn_cx: &Cx,         // the fn's full params (names + types), in signature order
        param_tys: &[Ty],   // surface type of each FULL param
        ret: &Ty,
        scrut: &str,
        arms: &[Arm],
        full_pos: usize,    // scrutinee's position among the full params
    ) -> Result<Term, String> {
        let full_params: &[String] = &fn_cx.names;
        let n = fn_cx.len();

        // the accumulators = every non-scrutinee param, in param order.
        let acc_positions: Vec<usize> = (0..n).filter(|&i| i != full_pos).collect();
        let acc_names: Vec<String> = acc_positions.iter().map(|&i| full_params[i].clone()).collect();
        let k = acc_names.len();

        // v1: elaborate each accumulator type and the result type as CLOSED kernel
        // terms (independent of every parameter). Elaborating in the EMPTY scope
        // makes a type that references a parameter fail with `unbound name …`, which
        // we surface as the v1 non-dependence restriction.
        let r_term = self.elab_ty(ret, &[]).map_err(|e| {
            format!(
                "Phase 1a′ accumulator fold `{fnname}`: the result type must be \
                 non-dependent (independent of every parameter) in v1 — {e}"
            )
        })?;
        let mut acc_ty_terms: Vec<Term> = Vec::with_capacity(k);
        for &i in &acc_positions {
            let t = self.elab_ty(&param_tys[i], &[]).map_err(|e| {
                format!(
                    "Phase 1a′ accumulator fold `{fnname}`: accumulator `{}`'s type must \
                     be non-dependent (independent of every parameter) in v1 — {e}",
                    full_params[i]
                )
            })?;
            acc_ty_terms.push(t);
        }

        // motive = λscrut'. (T₁ → … → T_K → R). The Pi chain (and R) are closed, so
        // they sit unshifted under the motive's binder; the binder itself is unused
        // (the result is non-dependent on the scrutinee).
        let mut pi_chain = r_term.clone();
        for t in acc_ty_terms.iter().rev() {
            pi_chain = Term::Pi(Mult::Omega, Box::new(t.clone()), Box::new(pi_chain));
        }
        let motive = Term::Lam(Box::new(pi_chain.clone()));

        // closed type Values for checking method bodies and accumulator args.
        let r_val = dep::eval_rc(&self.rc, &[], &r_term);
        let ih_ty_val = dep::eval_rc(&self.rc, &[], &pi_chain); // p k = p Zero = the Pi chain
        let acc_ty_vals: Vec<Value> =
            acc_ty_terms.iter().map(|t| dep::eval_rc(&self.rc, &[], t)).collect();

        // identify the Zero / Succ arms by their constructors' recorded roles.
        let mut zero_arm = None;
        let mut succ_arm = None;
        for a in arms {
            match self.nat_ctor.get(&a.ctor) {
                Some(NatRole::Zero) if zero_arm.is_some() => {
                    return Err(format!("`match` has a redundant/duplicate arm for `{}`", a.ctor))
                }
                Some(NatRole::Succ) if succ_arm.is_some() => {
                    return Err(format!("`match` has a redundant/duplicate arm for `{}`", a.ctor))
                }
                Some(NatRole::Zero) => zero_arm = Some(a),
                Some(NatRole::Succ) => succ_arm = Some(a),
                None => return Err(format!("`{}` is not a constructor of the Nat scrutinee", a.ctor)),
            }
        }
        let zero_arm = zero_arm.ok_or("`match` on Nat is missing the zero case")?;
        let succ_arm = succ_arm.ok_or("`match` on Nat is missing the successor case")?;
        if !zero_arm.binders.is_empty() {
            return Err(format!(
                "the zero case binds no pattern variables, got {}",
                zero_arm.binders.len()
            ));
        }
        if succ_arm.binders.len() != 1 {
            return Err(format!(
                "the successor case binds exactly one predecessor, got {}",
                succ_arm.binders.len()
            ));
        }
        let pred_name = succ_arm.binders[0].clone();

        // zero method : T₁ → … → T_K → R  =  λa₁…λa_K. <Zero body>. The accumulators
        // are FRESH binders pushed atop the fn params, so a surface reference to an
        // accumulator name resolves (innermost-first) to the method's own binder.
        let mut z_cx = fn_cx.clone();
        for (an, av) in acc_names.iter().zip(&acc_ty_vals) {
            z_cx.push(an.clone(), av.clone());
        }
        let mut z_method = self.check(&zero_arm.body, &r_val, &z_cx, None)?;
        for _ in 0..k {
            z_method = Term::Lam(Box::new(z_method));
        }

        // succ method : Π(k:Nat). (p k) → p (Suc k)  =  λk. λih. λa₁…λa_K. <Succ body>.
        // The IH is a function of the accumulators (type `T₁ → … → T_K → R`).
        let mut succ_cx = fn_cx.clone();
        succ_cx.push(pred_name.clone(), Value::VNat);
        let ih_name = "$ih".to_string();
        succ_cx.push(ih_name.clone(), ih_ty_val);
        for (an, av) in acc_names.iter().zip(&acc_ty_vals) {
            succ_cx.push(an.clone(), av.clone());
        }
        let mut fields: HashMap<String, String> = HashMap::new();
        fields.insert(pred_name, ih_name);
        let r = Rec { fnname, scrut_pos: full_pos, fields: &fields, acc_tys: Some(&acc_ty_vals) };
        let mut s_method = self.check(&succ_arm.body, &r_val, &succ_cx, Some(&r))?;
        for _ in 0..k {
            s_method = Term::Lam(Box::new(s_method)); // a_K … a₁
        }
        s_method = Term::Lam(Box::new(s_method)); // ih
        s_method = Term::Lam(Box::new(s_method)); // k

        // body = (NatElim motive z s scrut) a₁ … a_K — apply the built function to
        // the REAL accumulator params (pass D λ-wraps the whole thing afterwards).
        let scrut_idx = fn_cx.debruijn(scrut).expect("scrutinee param in scope");
        let mut body = Term::NatElim(
            Box::new(motive),
            Box::new(z_method),
            Box::new(s_method),
            Box::new(Term::Var(scrut_idx)),
        );
        for an in &acc_names {
            let idx = fn_cx.debruijn(an).expect("accumulator param in scope");
            body = Term::App(Box::new(body), Box::new(Term::Var(idx)));
        }
        Ok(body)
    }

    /// A recursive `fn` matching on a `Nat` whose recursion is not structural
    /// (e.g. `build(k, leftLabel)` / `build(k, rightLabel)`, or a non-decreasing
    /// call) is GENERAL recursion, not a fold: compile it to `Fix(ty, λparams.
    /// NatCase…)`, where recursive calls are real self-calls and the case-split
    /// provides no induction hypothesis. Used only when the totality analyzer's
    /// STRUCTURAL verdict is `Partial`, so folds (`add`, `mul`, …) still lower to
    /// eliminators (reducible in types, iterative). See `totality.rs`.
    fn elab_fix_nat(
        &self,
        fnname: &str,
        ty_term: &Term,
        full_names: &[String],
        full_tys: &[Ty],
        ret: &Ty,
        scrut: &str,
        arms: &[Arm],
    ) -> Result<Term, String> {
        // context: `self` (named like the fn) : the fn's type, then the params.
        let vty = dep::eval_rc(&self.rc, &[], ty_term);
        let mut cx = Cx::default();
        cx.push(fnname.to_string(), vty);
        let mut scope: Vec<String> = vec![fnname.to_string()];
        for (pn, pty) in full_names.iter().zip(full_tys) {
            let kty = self.elab_ty(pty, &scope)?;
            let v = dep::eval_rc(&self.rc, &neutral_env(cx.len()), &kty);
            cx.push(pn.clone(), v);
            scope.push(pn.clone());
        }
        let natcase = self.elab_nat_case(&cx, ret, scrut, arms, &scope)?;
        let mut body = natcase;
        for _ in 0..full_names.len() {
            body = Term::Lam(Box::new(body));
        }
        Ok(Term::Fix(Box::new(ty_term.clone()), Box::new(body)))
    }

    /// Build `NatCase(motive, z, λk. s, scrut)` for a `match` on a `Nat`, in a
    /// context `cx` that already binds `self` and the params. Recursion is by
    /// explicit self-call (no `Rec`/IH), so the successor branch has no IH binder.
    fn elab_nat_case(
        &self,
        cx: &Cx,
        ret: &Ty,
        scrut: &str,
        arms: &[Arm],
        scope: &[String],
    ) -> Result<Term, String> {
        let scrut_idx = cx
            .debruijn(scrut)
            .ok_or_else(|| format!("`match` scrutinee `{scrut}` is not in scope"))?;
        let mut zero_arm = None;
        let mut succ_arm = None;
        for a in arms {
            match self.nat_ctor.get(&a.ctor) {
                // a redundant/duplicate arm for the same role is rejected (E2).
                Some(NatRole::Zero) if zero_arm.is_some() => {
                    return Err(format!("`match` has a redundant/duplicate arm for `{}`", a.ctor))
                }
                Some(NatRole::Succ) if succ_arm.is_some() => {
                    return Err(format!("`match` has a redundant/duplicate arm for `{}`", a.ctor))
                }
                Some(NatRole::Zero) => zero_arm = Some(a),
                Some(NatRole::Succ) => succ_arm = Some(a),
                None => return Err(format!("`{}` is not a constructor of the Nat scrutinee", a.ctor)),
            }
        }
        let zero_arm = zero_arm.ok_or("`match` on Nat is missing the zero case")?;
        let succ_arm = succ_arm.ok_or("`match` on Nat is missing the successor case")?;
        if succ_arm.binders.len() != 1 {
            return Err("the successor case binds exactly one predecessor".into());
        }
        let pred_name = succ_arm.binders[0].clone();

        let mut motive_scope = scope.to_vec();
        motive_scope.push(scrut.to_string());
        let ret_term = self.elab_ty(ret, &motive_scope)?;
        let motive = Term::Lam(Box::new(ret_term.clone()));

        let n = cx.len();
        let base = neutral_env(n);
        let p_at = |scrut_val: Value| {
            let mut env = base.clone();
            env.push(scrut_val);
            dep::eval_rc(&self.rc, &env, &ret_term)
        };

        let z_expected = p_at(Value::VNatLit(0));
        let z = self.check(&zero_arm.body, &z_expected, cx, None)?;

        let k_val = dep::nvar(n);
        let mut succ_cx = cx.clone();
        succ_cx.push(pred_name, Value::VNat);
        let s_expected = p_at(Value::VSuc(Box::new(k_val)));
        let s_body = self.check(&succ_arm.body, &s_expected, &succ_cx, None)?;
        let s = Term::Lam(Box::new(s_body));

        Ok(Term::NatCase(
            Box::new(motive),
            Box::new(z),
            Box::new(s),
            Box::new(Term::Var(scrut_idx)),
        ))
    }

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
        // a `match` on a `%builtin Nat` scrutinee compiles to the kernel's native
        // `NatElim` (a counting loop), not a boxed datatype eliminator.
        if self.nat_types.contains(&data) {
            // an ACCUMULATOR-style fold (a recursive call varies a non-scrutinee
            // argument) needs the function-typed-motive lowering (Phase 1a′); a
            // verbatim-arg fold stays on the plain `NatElim` path (no regression).
            if self.is_acc_fold(fnname, scrut, explicit_params, arms) {
                if explicit_params.len() != full_params.len() {
                    return Err(format!(
                        "Phase 1a′ accumulator fold `{fnname}`: implicit parameters are not \
                         yet supported (v1 requires all-explicit parameters)"
                    ));
                }
                return self.elab_nat_match_acc(fnname, fn_cx, param_tys, ret, scrut, arms, full_pos);
            }
            return self.elab_nat_match(fnname, fn_cx, ret, scrut, arms, full_params, explicit_pos);
        }
        let decl = self.rc.data(&data).ok_or_else(|| format!("`{data}` is not a datatype"))?.clone();
        let np = decl.params.len();
        let ni = decl.indices.len();
        if dargs.len() != np + ni {
            return Err(format!("scrutinee type `{data}` applied to the wrong number of arguments"));
        }
        // E2: if the scrutinee's (concrete) index makes EVERY constructor
        // impossible (e.g. `Fin Zero` — both `fz`/`fs` need a `Succ` index), the
        // type is EMPTY: discharge the match with zero arms (an absurd case). The
        // derived term is kernel-rechecked, so a mis-classification can only be
        // REJECTED, never silently accepted (see `try_absurd_match`).
        if let Some(t) = self.try_absurd_match(&data, &decl, &dargs, full_params, scrut, ret, arms)? {
            return Ok(t);
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

        // COVERAGE HYGIENE (E2): every arm must name a REAL constructor of the
        // scrutinee's family, and no constructor may be matched twice. (The
        // per-constructor loop below finds the FIRST arm for each ctor, so without
        // this an unknown-ctor arm or a duplicate would be silently ignored.)
        let mut seen: Vec<&str> = Vec::new();
        for arm in arms {
            if !decl.ctors.iter().any(|c| c.name == arm.ctor) {
                return Err(format!(
                    "`match` arm `{}` is not a constructor of `{data}`",
                    arm.ctor
                ));
            }
            if seen.contains(&arm.ctor.as_str()) {
                return Err(format!(
                    "`match` has a redundant/duplicate arm for `{}`",
                    arm.ctor
                ));
            }
            seen.push(arm.ctor.as_str());
        }

        let mut methods = Vec::with_capacity(decl.ctors.len());
        for ctor in &decl.ctors {
            let info = &self.ctor_info[&ctor.name];
            let nargs = ctor.args.len();
            // recursive fields: DIRECT (`data idxs`) OR HIGHER-ORDER (`(z…) → data
            // idxs`), detected via `rec_spine` exactly as the kernel's method-type /
            // eliminator do — so the IH count here matches `elim_method_telescope`.
            let rec_fields: Vec<usize> = ctor
                .args
                .iter()
                .enumerate()
                .filter(|(_, (_, aty))| dep::rec_field_arity(&data, aty).is_some())
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
            // binder names: ALL the constructor's arguments, then one induction
            // hypothesis per recursive argument — the standard (Coq-style) method
            // telescope `∀ a l, P l → P (c a l)`, matching `method_ty_tm`, `velim`,
            // and the backend (all kept consistent on args-then-IHs).
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

            let r = Rec { fnname, scrut_pos: explicit_pos, fields: &fields, acc_tys: None };
            // USE-SITE LINEARITY (the convergent, whack-a-mole-proof check): re-bind
            // each EXPLICIT field whose INSTANTIATED type is linear via `let f = f`, so
            // the let-binder rule binds it at `1` and the kernel enforces exactly-once
            // use through it. Because it checks the field's ACTUAL (post-substitution)
            // type at the USE SITE, it catches a hidden `Own` no matter HOW it is hidden
            // — a generic container instantiated at `Own` (`Pair (Own Nat) Unit`), a
            // nested generic, an alias — not just a syntactic field. (Non-linear fields
            // are untouched, so copyable fields stay ω/multi-usable.)
            let arm_body = rebind_linear_fields(&arm.body, &binder_names, &info.arg_implicit, &binder_tys, nargs, &self.rc);
            let mut body = self.check(&arm_body, &expected, &arm_cx, Some(&r))?;
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

/// Collect the argument lists of every call to `fnname` inside a term.
/// The CONSTRUCTOR HEAD of a (normalized) index term, for decidable index
/// disjointness: two indices with KNOWN, DIFFERENT heads cannot be equal (e.g.
/// `Zero` vs `Succ _`). Returns `None` for a neutral/variable/non-constructor
/// term — meaning "unknown", so disjointness is NOT concluded (conservative:
/// when unsure, treat as possibly-reachable). Built-in `Nat` literals fold to
/// `Zero`/`Succ`; boxed constructors report their own name.
fn ctor_head(t: &Term) -> Option<String> {
    match t {
        Term::NatLit(0) | Term::Zero => Some("Zero".into()),
        Term::NatLit(_) | Term::Suc(_) => Some("Succ".into()),
        Term::Constr(name, _) => Some(name.clone()),
        _ => None,
    }
}

/// Collect EVERY call (to any function/constructor) in `t`, for the totality
/// call graph: a `Call(name, args)` becomes `TCall { callee: name, args }`.
/// Constructor applications are included too — harmless, since they are not in
/// the user-function set and so are treated as total leaves by the analyzer.
/// Is this fully-instantiated TYPE LINEAR — does it carry an `Own` pointer or a linear
/// `Σ[1]` pair anywhere reachable? This is the ONE convergent linearity test, used at
/// every binding site (`let`, function parameter, `match` field) to decide whether the
/// binder is linear (bind at `1`: use-twice = `ω ⋢ 1`, drop = `0 ⋢ 1`) or copyable
/// (bind at `ω`). It is FIELD-AWARE and INSTANTIATION-aware, so it catches a hidden
/// `Own` HOWEVER it is hidden — directly (`Own T`), in a type ARGUMENT (`Vec (Own T) n`),
/// behind a datatype's FIELD definitions (`struct Box { p : Own Nat }`, recursively),
/// or via a generic container instantiated at `Own` (`Pair (Own Nat) Unit`). This
/// replaces the earlier whack-a-mole forbid-per-hiding-spot: there is no syntactic
/// hiding-spot to miss, because we check the actual resolved type. An abstract type
/// VARIABLE (a `Var`, e.g. an un-instantiated `{0 a}`) reads non-linear — that is the
/// FUTURE_WORK §13 polymorphism corner (a leak through an abstract-typed ω param, NOT a
/// double-free), handled when real surface linear params land.
fn type_is_linear(ty: &Term, rc: &Rc<Signature>) -> bool {
    contains_linear(ty, rc, &mut std::collections::HashSet::new())
}

fn contains_linear(ty: &Term, rc: &Rc<Signature>, seen: &mut std::collections::HashSet<String>) -> bool {
    match ty {
        Term::Const(n) => n == "Own",
        Term::Sigma(crate::mult::Mult::One, _, _) => true,
        Term::Sigma(_, a, b) | Term::Pi(_, a, b) | Term::App(a, b) | Term::Pair(a, b) | Term::Add(a, b) => {
            contains_linear(a, rc, seen) || contains_linear(b, rc, seen)
        }
        Term::Eq(a, b, c) => {
            contains_linear(a, rc, seen) || contains_linear(b, rc, seen) || contains_linear(c, rc, seen)
        }
        Term::Ann(e, _) => contains_linear(e, rc, seen),
        Term::Suc(x) | Term::Fst(x) | Term::Snd(x) | Term::Refl(x) => contains_linear(x, rc, seen),
        Term::Data(name, args) => {
            if name == "Own" {
                return true;
            }
            // a linear type ARGUMENT (e.g. `Vec (Own T) n`, `Pair (Own Nat) Unit`).
            if args.iter().any(|a| contains_linear(a, rc, seen)) {
                return true;
            }
            // a linear FIELD hidden behind the datatype's name — recurse into the
            // constructor field DEFINITIONS (the `seen` guard handles recursive types).
            if seen.insert(name.clone()) {
                if let Some(decl) = rc.data(name) {
                    for ctor in &decl.ctors {
                        for (_, fty) in &ctor.args {
                            if contains_linear(fty, rc, seen) {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        }
        Term::Constr(_, args) => args.iter().any(|a| contains_linear(a, rc, seen)),
        // Var / Type / Nat / NatLit / Zero / Lam / Fix / NatElim / NatCase / Elim:
        // no linear component (an abstract `Var` is the deferred §13 polymorphism case).
        _ => false,
    }
}

/// USE-SITE LINEARITY for a `match` arm: re-bind each EXPLICIT field whose INSTANTIATED
/// type `is_linear` via `let f = f`, so the let-binder rule binds it at `1` and the
/// kernel enforces exactly-once use through it. This is the convergent, whack-a-mole-
/// proof check — it inspects the field's ACTUAL (post-substitution) type at the USE
/// SITE, so a hidden `Own` is caught however it is hidden (generic instantiated at
/// `Own`, nested, behind a struct name). Non-linear fields are untouched (copyable
/// fields stay ω, freely multi-usable). `binder_tys` is parallel to `binder_names`
/// (ctor args, then IHs), the args' types INSTANTIATED by `elim_method_telescope`.
fn rebind_linear_fields(
    body: &Tm,
    binder_names: &[String],
    arg_implicit: &[bool],
    binder_tys: &[(crate::mult::Mult, Term)],
    nargs: usize,
    rc: &Rc<Signature>,
) -> Tm {
    let mut wrapped = body.clone();
    for j in (0..nargs).rev() {
        let explicit = !arg_implicit.get(j).copied().unwrap_or(false);
        // A 0-ERASED field (a `{0}` proof/index, even of a linear type) stays 0 — it
        // has no runtime representation and so no double-free risk; forcing it to 1
        // would OVER-reject. Only an ω/1 field of an instantiated-linear type is bound
        // at 1.
        let erased = binder_tys[j].0 == crate::mult::Mult::Zero;
        if explicit && !erased && type_is_linear(&binder_tys[j].1, rc) {
            let nm = binder_names[j].clone();
            wrapped = Tm::Let(nm.clone(), Box::new(Tm::Var(nm)), Box::new(wrapped));
        }
    }
    wrapped
}

fn collect_all_calls(t: &Tm, out: &mut Vec<TCall>) {
    match t {
        Tm::Call(n, args) => {
            out.push(TCall { callee: n.clone(), args: args.clone() });
            for a in args {
                collect_all_calls(a, out);
            }
        }
        Tm::Add(a, b) => {
            collect_all_calls(a, out);
            collect_all_calls(b, out);
        }
        Tm::Match(_, arms) => {
            for a in arms {
                collect_all_calls(&a.body, out);
            }
        }
        Tm::LetPair(_, e, body) | Tm::Let(_, e, body) => {
            collect_all_calls(e, out);
            collect_all_calls(body, out);
        }
        Tm::Var(_) | Tm::Lit(_) => {}
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

/// The built-in memory prelude. These are the trusted L3 primitives the native
/// backend gives a real `malloc`/`free` lowering. They are provided automatically
/// so a program need not re-declare the boilerplate — but ONLY for names the user
/// has not declared themselves, so explicit (re)declarations always win and any
/// existing program keeps compiling unchanged. The implementations live in
/// `dep_codegen::compile_postulate`; the kernel checks every USE of them against
/// these types (linearity ⇒ no leak / no use-after-free).
const PRELUDE: &str = r#"
enum Unit { U : Unit }
postulate Own   : Type -> Type
postulate alloc : {0 a : Type} -> a -> Own a
postulate free  : {0 a : Type} -> (1 o : Own a) -> Unit
"#;

/// The primary name a top-level item declares (`None` for a `%builtin` pragma).
fn item_name(it: &Item) -> Option<&str> {
    match it {
        Item::Sig(n, _) | Item::Fn(n, _, _, _) | Item::Postulate(n, _) => Some(n),
        Item::Enum { name, .. } | Item::Struct { name, .. } => Some(name),
        Item::BuiltinNat(_) => None,
    }
}

pub fn elaborate(src: &str) -> Result<Program, String> {
    let toks = lex(src)?;
    let mut items = Parser { toks, pos: 0, fresh: 0 }.parse_program()?;

    // Prepend the memory prelude — but ALL-OR-NOTHING: if the program declares any
    // of the prelude's own names (Unit/Own/alloc/free), it is managing the memory
    // layer itself, so we inject nothing and leave it exactly as written (this is
    // how every pre-prelude program keeps compiling, and it avoids a partial
    // injection that could reference a name the user declares later).
    let declared: std::collections::HashSet<&str> = items.iter().filter_map(item_name).collect();
    let prelude_items = Parser { toks: lex(PRELUDE)?, pos: 0, fresh: 0 }.parse_program()?;
    let collides = prelude_items
        .iter()
        .filter_map(item_name)
        .any(|n| declared.contains(n));
    if !collides {
        let mut merged = prelude_items;
        merged.append(&mut items);
        items = merged;
    }

    let mut elab = Elab {
        rc: Rc::new(Signature::default()),
        data_arity: HashMap::new(),
        ctor_arity: HashMap::new(),
        ctor_info: HashMap::new(),
        defs: HashMap::new(),
        def_implicit: HashMap::new(),
        nat_types: std::collections::HashSet::new(),
        nat_ctor: HashMap::new(),
    };

    // pass A: `%builtin Nat T` pragmas. Validate each names a Nat-shaped enum (one
    // nullary constructor + one single-self-recursive constructor) and record the
    // type + its two constructors as the packed built-in `Nat`. Such a type is
    // NOT registered as a datatype — it aliases the kernel's `Nat`.
    for it in &items {
        if let Item::BuiltinNat(tyname) = it {
            let decl = items.iter().find_map(|x| match x {
                Item::Enum { name, params, index_ty, variants } if name == tyname => {
                    Some((params, index_ty, variants))
                }
                _ => None,
            });
            let Some((params, index_ty, variants)) = decl else {
                return Err(format!("`%builtin Nat {tyname}`: no `enum {tyname}` is declared"));
            };
            if !params.is_empty() || index_ty.is_some() || variants.len() != 2 {
                return Err(format!(
                    "`%builtin Nat {tyname}`: a built-in Nat must be a parameterless, \
                     unindexed enum with exactly two constructors"
                ));
            }
            let mut zero = None;
            let mut succ = None;
            for (cn, cty) in variants {
                let (arrows, _) = peel_arrows(cty);
                match arrows.as_slice() {
                    [] => zero = Some(cn.clone()),
                    [(_, _, _, dom)] if matches!(dom, Ty::Var(d) if d == tyname) => {
                        succ = Some(cn.clone())
                    }
                    _ => {
                        return Err(format!(
                            "`%builtin Nat {tyname}`: constructor `{cn}` is not Nat-shaped \
                             (need one nullary `Zero` and one `{tyname} -> {tyname}` `Succ`)"
                        ))
                    }
                }
            }
            let (Some(zero), Some(succ)) = (zero, succ) else {
                return Err(format!(
                    "`%builtin Nat {tyname}`: need exactly one nullary and one \
                     single-recursive constructor"
                ));
            };
            elab.nat_types.insert(tyname.clone());
            elab.nat_ctor.insert(zero, NatRole::Zero);
            elab.nat_ctor.insert(succ, NatRole::Succ);
        }
    }

    // pass B: arities (skipping the `%builtin Nat` types — they alias the kernel Nat)
    for it in &items {
        match it {
            Item::Enum { name, params, index_ty, variants } if !elab.nat_types.contains(name) => {
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

    // pass C: datatypes → signature, recording ctor implicit/name info.
    // `%builtin Nat` types are skipped — they are the kernel's `Nat`, not data.
    let mut sig = Signature::default();
    for it in &items {
        match it {
            Item::Enum { name, params, index_ty, variants } if !elab.nat_types.contains(name) => {
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
                    // (Phase A) LINEAR FIELDS ARE ALLOWED: the use-site linearity check
                    // (`type_is_linear` field-aware + `rebind_linear_fields` at every
                    // match) enforces exactly-once use through a stored `Own`, so a
                    // hidden-field double-free/leak is caught where the value is USED —
                    // no declaration-time forbid needed (it was whack-a-mole).
                    elab.ctor_info.insert(cn.clone(), CtorInfo {
                        data: name.clone(),
                        param_implicit: param_implicit.clone(),
                        arg_implicit,
                        arg_names,
                    });
                    ctors.push(Constructor { name: cn.clone(), args, idxs });
                }
                // Surface datatypes live in `Type 0` (all surface `Type`s are
                // `Type 0`). A constructor that tried to store a `Type` would have
                // an argument in `Type 1`, which `check_signature` rejects against
                // this `universe: 0` — the predicativity/Girard guard. Until the
                // surface gains universe annotations, such datatypes are written at
                // the kernel level.
                sig.datas.push(DataDecl {
                    name: name.clone(),
                    universe: 0,
                    params: kparams,
                    indices,
                    ctors,
                });
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
                // (Phase A) linear struct fields are ALLOWED — see the enum note above:
                // the use-site linearity check enforces exactly-once use through them.
                elab.ctor_info.insert(name.clone(), CtorInfo {
                    data: name.clone(),
                    param_implicit,
                    arg_implicit: vec![false; fields.len()],
                    arg_names,
                });
                let ctor = Constructor { name: name.clone(), args, idxs: vec![] };
                sig.datas.push(DataDecl {
                    name: name.clone(),
                    universe: 0, // surface structs live in `Type 0` (see enum note)
                    params: kparams,
                    indices: vec![],
                    ctors: vec![ctor],
                });
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
    // PHASE E1 — totality (termination) PRE-PASS: distil every `fn` and run the
    // structural-descent / mutual-recursion analyzer ONCE, before lowering. The
    // verdict then DECIDES the lowering: `Total` ⇒ a kernel eliminator (which the
    // kernel re-checks total-by-construction); `Partial` ⇒ an opaque `Fix`. This
    // does not grow the trusted base — see `totality.rs`.
    let mut clauses: Vec<FnClauses> = Vec::new();
    for it in &items {
        if let Item::Fn(name, params, body, _) = it {
            let sig = sigs.get(name).ok_or_else(|| format!("`fn {name}` has no type signature"))?;
            let (full_names, _imps, full_tys, _ret) = split_signature(sig, params)?;
            clauses.push(elab.fn_clauses(name, params, &full_names, &full_tys, body));
        }
    }
    let verdicts = totality::analyze(&clauses);

    let mut out_defs = Vec::new();
    let mut totality_status: Vec<(String, bool, Option<String>)> = Vec::new();
    for it in &items {
        if let Item::Fn(name, params, body, annot) = it {
            let sig = sigs.get(name).ok_or_else(|| format!("`fn {name}` has no type signature"))?.clone();
            let ty_term = elab.elab_ty(&sig, &[])?;
            let (full_names, full_imps, full_tys, ret) = split_signature(&sig, params)?;
            elab.def_implicit.insert(name.clone(), full_imps.clone());

            // the totality verdicts for this fn (analyzed in the pre-pass):
            //   `full`       — end-to-end (own recursion ∧ callees total): the
            //                  CERTIFICATE and the reported status.
            //   `structural` — own recursion only: drives LOWERING (eliminator
            //                  vs Fix); a non-recursive fn is always structurally
            //                  total, so it lowers normally even calling a partial.
            let info = verdicts.get(name).cloned();
            let full = info.as_ref().map(|i| i.full.clone()).unwrap_or(Totality::Total);
            let structural =
                info.as_ref().map(|i| i.structural.clone()).unwrap_or(Totality::Total);
            // `%total` is a CERTIFICATE, not a hint: a `%total` fn that the checker
            // cannot certify (its own recursion, OR a partial callee) is a HARD
            // ERROR — annotation ≠ proof.
            if *annot == Some(TotAnnot::Total) {
                if let Some(reason) = full.reason() {
                    return Err(format!("`%total fn {name}` is not total: {reason}"));
                }
            }
            totality_status.push((
                name.clone(),
                full.is_total(),
                full.reason().map(|s| s.to_string()),
            ));

            // the fn's typing context: each parameter's name + kernel type
            let mut fn_cx = Cx::default();
            for (i, (pn, pty)) in full_names.iter().zip(&full_tys).enumerate() {
                let kty = elab.elab_ty(pty, &full_names[..i])?;
                let v = dep::eval_rc(&elab.rc, &neutral_env(i), &kty);
                fn_cx.push(pn.clone(), v);
            }

            // PARTIAL ⇒ general recursion via an opaque `Fix` (the kernel never
            // unfolds it, so it cannot reduce in a type — the partial/total
            // boundary). Currently only `%builtin Nat` case-splits have a `Fix`
            // lowering; a partial recursion on any other shape is an honest hard
            // error (NOT silently accepted), pending the Phase E2 machinery.
            if !structural.is_total() {
                if let Tm::Match(scrut, arms) = body {
                    if let Some(sp) = full_names.iter().position(|p| p == scrut) {
                        let scrut_is_nat = matches!(
                            flatten_ty(&full_tys[sp]).0,
                            Ty::Var(n) if elab.nat_types.contains(n)
                        );
                        if scrut_is_nat {
                            let term = elab.elab_fix_nat(name, &ty_term, &full_names, &full_tys, &ret, scrut, arms)?;
                            elab.defs.insert(name.clone(), (term.clone(), ty_term.clone()));
                            out_defs.push((name.clone(), ty_term, term));
                            continue;
                        }
                    }
                }
                return Err(format!(
                    "`fn {name}` is partial ({}) but cannot be lowered: general/mutual \
                     recursion is only supported on a `%builtin Nat` scrutinee so far \
                     (Phase E2). Restructure as a structural fold, or wait for E2.",
                    structural.reason().unwrap_or("not total")
                ));
            }

            // TOTAL ⇒ lower to a kernel eliminator (or a non-recursive body).
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

    Ok(Program { sig: (*elab.rc).clone(), defs: out_defs, totality: totality_status })
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

/// Pretty-print a kernel `Term` (used to display the normal form of `main`).
pub fn pretty(t: &Term) -> String {
    fn atom(t: &Term) -> String {
        match t {
            Term::Data(_, args) | Term::Constr(_, args) if !args.is_empty() => {
                format!("({})", pretty(t))
            }
            Term::App(_, _) | Term::Suc(_) => format!("({})", pretty(t)),
            _ => pretty(t),
        }
    }
    match t {
        Term::Type(0) => "Type".into(),
        Term::Type(i) => format!("Type {i}"),
        Term::Nat => "Nat".into(),
        Term::Var(i) => format!("#{i}"),
        Term::NatLit(n) => n.to_string(),
        Term::Suc(k) => format!("suc {}", atom(k)),
        Term::Data(name, args) | Term::Constr(name, args) => {
            let mut s = name.clone();
            for a in args {
                s.push(' ');
                s.push_str(&atom(a));
            }
            s
        }
        Term::Const(c) => c.clone(),
        Term::App(f, a) => format!("{} {}", pretty(f), atom(a)),
        Term::Lam(_) => "<fun>".into(),
        Term::Pi(_, _, _) => "<pi>".into(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests;
