//! The v1.0 SURFACE language (`docs/10-surface-syntax.md`): ML-flavored type
//! signatures, Rust-flavored terms. Elaborates to the `dep.rs` kernel, which
//! does all type-checking.
//!
//! ```text
//! enum Nat { Zero : Nat, Succ : Nat -> Nat }
//!
//! boxed enum Vec (a : Type) : Nat -> Type {
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
    /// a suffixed integer literal `123u8` / `-` handled separately — carries the
    /// magnitude and the scalar type name (`u8`…`i64`). Desugars to `u8(123)`.
    NumSuffixed(u64, String),
    /// a float literal `3.14` / `3.14f32` — carries the IEEE bit pattern and
    /// whether it is single precision. Desugars to `f64_bits(<bits>)`.
    FloatLit(u64, bool),
    Str(String),
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
    Backslash,
    KwRewrite,
    KwIn,
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
    KwLinear,
    KwBoxed,
    KwForeign,
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
                "foreign" => out.push(Tok::KwForeign),
                other => return Err(format!("unknown pragma `%{other}`")),
            }
            i = j;
        } else if c == '"' {
            i += 1;
            let mut lit = String::new();
            loop {
                if i >= b.len() {
                    return Err("unterminated string literal".into());
                }
                match b[i] as char {
                    '"' => {
                        i += 1;
                        break;
                    }
                    '\\' => {
                        i += 1;
                        let e = *b.get(i).ok_or("unterminated string escape")? as char;
                        lit.push(match e {
                            'n' => '\n',
                            't' => '\t',
                            '\\' => '\\',
                            '"' => '"',
                            '0' => '\0',
                            other => return Err(format!("unknown string escape `\\{other}`")),
                        });
                        i += 1;
                    }
                    ch => {
                        lit.push(ch);
                        i += ch.len_utf8();
                    }
                }
            }
            out.push(Tok::Str(lit));
        } else if c.is_ascii_digit() {
            let s = i;
            while i < b.len() && (b[i] as char).is_ascii_digit() {
                i += 1;
            }
            // FLOAT literal: `digits '.' digits`, optional `f32`/`f64` suffix.
            if i + 1 < b.len() && b[i] as char == '.' && (b[i + 1] as char).is_ascii_digit() {
                i += 1; // consume '.'
                while i < b.len() && (b[i] as char).is_ascii_digit() {
                    i += 1;
                }
                let numstr = &src[s..i];
                let is_f32 = if src[i..].starts_with("f32") {
                    i += 3;
                    true
                } else if src[i..].starts_with("f64") {
                    i += 3;
                    false
                } else {
                    false
                };
                let v: f64 = numstr.parse().map_err(|_| "bad float literal")?;
                let bits = if is_f32 {
                    (v as f32).to_bits() as u64
                } else {
                    v.to_bits()
                };
                out.push(Tok::FloatLit(bits, is_f32));
            } else {
                // integer, with an optional scalar suffix `u8`…`i64` (no space).
                let mut j = i;
                while j < b.len() && {
                    let ch = b[j] as char;
                    ch.is_alphanumeric() || ch == '_'
                } {
                    j += 1;
                }
                let suffix = &src[i..j];
                if matches!(
                    suffix,
                    "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64"
                ) {
                    let n: u64 = src[s..i].parse().map_err(|_| "bad number")?;
                    out.push(Tok::NumSuffixed(n, suffix.to_string()));
                    i = j;
                } else {
                    out.push(Tok::Num(src[s..i].parse().map_err(|_| "bad number")?));
                }
            }
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
                "rewrite" => Tok::KwRewrite,
                "in" => Tok::KwIn,
                "Type" => Tok::KwType,
                "postulate" => Tok::KwPostulate,
                "boxed" => Tok::KwBoxed,
                "linear" => Tok::KwLinear,
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
                '\\' => Tok::Backslash,
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

/// A surface multiplicity: a concrete rig element, or a variable bound by a
/// `(m : Mult)` parameter (multiplicity polymorphism). A `Var` is substituted to a
/// concrete `Mult` at each call site before elaboration — the kernel never sees a
/// multiplicity variable (see `docs/MULT_POLY_PLAN.md`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SMult {
    Lit(Mult),
    Var(String),
}

impl SMult {
    /// Resolve to a concrete `Mult`, given a substitution for variables. A residual
    /// variable is a hard error — it means a mult-poly function reached elaboration
    /// without being monomorphized, which is a compiler bug, not a user error.
    fn resolve(&self, env: &HashMap<String, Mult>) -> Result<Mult, String> {
        match self {
            SMult::Lit(m) => Ok(*m),
            SMult::Var(v) => env.get(v).copied().ok_or_else(|| {
                format!("unresolved multiplicity variable `{v}` (mult-poly not monomorphized)")
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Ty {
    Var(String),
    /// a Nat literal at the INDEX level, e.g. `Slice Nat 4` / `Arr U8 16`.
    Lit(u64),
    /// `Type l` with a LEVEL VARIABLE `l` — exists only between parsing and
    /// the level-monomorphization pre-pass (PHASE B2), which replaces it with
    /// a concrete `Ty::Type(n)` per instantiation.
    TypeV(String),
    /// `Type` (level 0) or `Type i` — the i-th universe. The kernel's hierarchy
    /// (`Type i : Type (i+1)`, cumulative, predicative) is now surface-reachable.
    Type(usize),
    App(Box<Ty>, Box<Ty>),
    /// built-in `Nat` addition at the index level, e.g. `Vec a (m + n)`. Binds
    /// looser than application and tighter than `->`. Elaborates to `Term::Add`,
    /// which the kernel decides up to linear-Nat equality (see `src/solver.rs`).
    Add(Box<Ty>, Box<Ty>),
    /// (mult, implicit?, name, domain, codomain)
    Arrow(SMult, bool, Option<String>, Box<Ty>, Box<Ty>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Tm {
    Var(String),
    Call(String, Vec<Tm>),
    Match(String, Vec<Arm>),
    /// a MULTI-SCRUTINEE match, `match a, b { p, q => e, … }` (Phase A1). Each
    /// row holds one full pattern per scrutinee. Exists only between parsing
    /// and `desugar_patterns`, which compiles it through the pattern matrix
    /// into nested single-scrutinee `Match`es (so coverage, dead-arm
    /// rejection, and kernel re-checking all come from the existing paths).
    MatchN(Vec<String>, Vec<(Vec<Pat>, Tm)>),
    /// a built-in `Nat` literal, e.g. `0`, `5`, `1000000`.
    Lit(u64),
    /// a string literal `"…"` — a value of the prelude postulate `Str`.
    Str(String),
    /// built-in `Nat` addition `a + b`.
    Add(Box<Tm>, Box<Tm>),
    /// `let (a, b) = e; body` — destructure a single-constructor value
    LetPair(Vec<String>, Box<Tm>, Box<Tm>),
    /// `let x = e; body` — bind a single value (1a surface expressiveness).
    Let(String, Box<Tm>, Box<Tm>),
    /// `(e : T)` type ASCRIPTION — check `e` against `T`. Surfaces via the
    /// type-annotated `let x : T = e` (drives inference for `cast` and other
    /// context-typed forms).
    Ann(Box<Tm>, Box<Ty>),
    /// an anonymous function `\x => e` / `\x, y => e` (P5 lambda sugar). Checked
    /// against an expected function type (its parameter types come from context);
    /// the body is elaborated with ONLY the lambda's parameters in scope, so a
    /// reference to an enclosing LOCAL (a capture) is a clear "unbound name"
    /// error — capturing closures use the explicit representation types (§3),
    /// pending the `->`/`=>` surface decision. A non-capturing lambda is a closed
    /// function value, so it inlines/specialises exactly like a named `fn` (P1/P4).
    Lam(Vec<String>, Box<Tm>),
}

/// A surface match PATTERN: a variable, or a constructor applied to
/// sub-patterns (`Cons(h, Cons(h2, t))`). A bare name that is a declared
/// constructor is a NULLARY constructor pattern (`Cons(h, Nil)`), not a binder.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Pat {
    Var(String),
    Ctor(String, Vec<Pat>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Arm {
    pub(crate) ctor: String,
    pub(crate) binders: Vec<String>,
    /// the arm's argument patterns, parallel to `binders` (a nested pattern's
    /// binder is a fresh placeholder until `desugar_patterns` flattens the
    /// match). Empty after desugaring — the invariant every later pass assumes.
    pub(crate) pats: Vec<Pat>,
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
    mult: Option<SMult>,
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
        /// declared `linear`: values of this type are resources (no drop/dup).
        linear: bool,
        /// `boxed enum` — the explicit opt-in to the heap-cell representation
        /// (constructors allocate one cell per value). REQUIRED for recursive
        /// families; without it an enum is a flat tagged-union VALUE.
        boxed: bool,
    },
    Struct {
        name: String,
        params: Vec<Binder>,
        fields: Vec<(String, Ty)>,
    },
    /// `(name, type, linear?)` — `linear` marks a resource postulate type.
    Postulate(String, Ty, bool),
    /// `%foreign ["c_symbol"] name : ty` — an opaque postulate whose native
    /// lowering is a direct call to an extern C symbol (`c_symbol`, defaulting
    /// to `name`) at the i64 ABI: one `i64` per RUNTIME (non-erased) argument,
    /// an `i64` result. THE audited escape hatch (FUTURE_WORK §6/§8): the
    /// kernel checks every use against the declared type, but the C side of
    /// the boundary is trusted — like Rust's `extern`, wrap it in a safe,
    /// honestly-typed declaration.
    Foreign(String, String, Ty),
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
            Some(Tok::KwEnum) => self.parse_enum(false, false),
            Some(Tok::KwStruct) => self.parse_struct(),
            Some(Tok::KwFn) => self.parse_fn(None),
            Some(Tok::KwPostulate) => self.parse_postulate(false),
            // `%foreign ["c_symbol"] name : ty` — an extern-C postulate.
            Some(Tok::KwForeign) => {
                self.next();
                let sym = match self.peek() {
                    Some(Tok::Str(_)) => match self.next() {
                        Some(Tok::Str(s)) => Some(s),
                        _ => unreachable!(),
                    },
                    _ => None,
                };
                let name = self.ident()?;
                self.eat(&Tok::Colon)?;
                let ty = self.parse_ty()?;
                Ok(Item::Foreign(name.clone(), sym.unwrap_or(name), ty))
            }
            // `linear` prefixes a resource `postulate` or `enum` — a value of that
            // type may not be dropped or duplicated (an un-annotated binder of it
            // defaults to multiplicity 1). See `docs/VIEW_LAYER_PLAN.md`.
            Some(Tok::KwLinear) => {
                self.next();
                let boxed = self.peek() == Some(&Tok::KwBoxed);
                if boxed {
                    self.next();
                }
                match self.peek() {
                    Some(Tok::KwPostulate) if !boxed => self.parse_postulate(true),
                    Some(Tok::KwEnum) => self.parse_enum(true, boxed),
                    other => Err(format!(
                        "`linear` must be followed by `postulate` or `[boxed] enum`, found {other:?}"
                    )),
                }
            }
            // `boxed enum` / `boxed linear enum` — the explicit opt-in to the
            // heap-cell representation (constructors allocate).
            Some(Tok::KwBoxed) => {
                self.next();
                match self.peek() {
                    Some(Tok::KwEnum) => self.parse_enum(false, true),
                    Some(Tok::KwLinear) => {
                        self.next();
                        match self.peek() {
                            Some(Tok::KwEnum) => self.parse_enum(true, true),
                            other => Err(format!(
                                "`boxed linear` must be followed by `enum`, found {other:?}"
                            )),
                        }
                    }
                    _ => unreachable!(),
                }
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
                && self.toks.get(k + 1) == Some(&Tok::Colon))
            // a multiplicity-VARIABLE binder `( m x : … )`: two identifiers then `:`
            // (the first is the mult var, the second the binder name).
            || (matches!(self.toks.get(k), Some(Tok::Ident(_)))
                && matches!(self.toks.get(k + 1), Some(Tok::Ident(_)))
                && self.toks.get(k + 2) == Some(&Tok::Colon));
        if ok {
            Some(open)
        } else {
            None
        }
    }

    fn parse_mult(&mut self) -> Option<SMult> {
        match self.peek() {
            Some(Tok::Num(0)) => {
                self.next();
                Some(SMult::Lit(Mult::Zero))
            }
            Some(Tok::Num(1)) => {
                self.next();
                Some(SMult::Lit(Mult::One))
            }
            Some(Tok::Ident(w)) if w == "w" => {
                self.next();
                Some(SMult::Lit(Mult::Omega))
            }
            // a multiplicity VARIABLE: an identifier that is the binder's
            // multiplicity (a `(m x : T)` binder), disambiguated from the binder
            // NAME by a following identifier. `(x : T)` has one ident (the name);
            // `(m x : T)` has two (mult var, then name).
            Some(Tok::Ident(v))
                if matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(_))) =>
            {
                let v = v.clone();
                self.next();
                Some(SMult::Var(v))
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
                let m = b.mult.clone().unwrap_or(SMult::Lit(
                    if b.implicit { Mult::Zero } else { Mult::Omega },
                ));
                out = Ty::Arrow(m, b.implicit, Some(b.name), Box::new(b.ty), Box::new(out));
            }
            Ok(out)
        } else {
            // index-level `+` binds looser than application, tighter than `->`.
            let mut lhs = self.parse_ty_app()?;
            while self.peek() == Some(&Tok::Plus) {
                self.next();
                let rhs = self.parse_ty_app()?;
                lhs = Ty::Add(Box::new(lhs), Box::new(rhs));
            }
            if self.peek() == Some(&Tok::Arrow) {
                self.next();
                let rhs = self.parse_ty()?;
                Ok(Ty::Arrow(SMult::Lit(Mult::Omega), false, None, Box::new(lhs), Box::new(rhs)))
            } else {
                Ok(lhs)
            }
        }
    }

    fn parse_ty_app(&mut self) -> Result<Ty, String> {
        let mut e = self.parse_ty_atom()?;
        while matches!(
            self.peek(),
            Some(Tok::Ident(_)) | Some(Tok::LParen) | Some(Tok::KwType) | Some(Tok::Num(_))
        ) {
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
                // an optional LEVEL literal: `Type 1`, `Type 2`, … (default 0),
                // or a LEVEL VARIABLE `Type l` (an enclosing `(l : Level)`
                // parameter, resolved by monomorphization — Phase B2).
                if let Some(Tok::Num(l)) = self.peek() {
                    let l = *l as usize;
                    self.next();
                    Ok(Ty::Type(l))
                } else if let Some(Tok::Ident(v)) = self.peek() {
                    // only a LOWERCASE single name reads as a level variable —
                    // `Type` followed by a datatype/ctor application must not
                    // swallow it. Levels are conventionally `l`, `l1`, …
                    if v.starts_with(|c: char| c.is_ascii_lowercase())
                        && self.toks.get(self.pos + 1) != Some(&Tok::Colon)
                    {
                        let v = v.clone();
                        self.next();
                        Ok(Ty::TypeV(v))
                    } else {
                        Ok(Ty::Type(0))
                    }
                } else {
                    Ok(Ty::Type(0))
                }
            }
            Some(Tok::Ident(_)) => Ok(Ty::Var(self.ident()?)),
            Some(Tok::Num(n)) => {
                let n = *n;
                self.next();
                Ok(Ty::Lit(n))
            }
            Some(Tok::LParen) => {
                self.next();
                let t = self.parse_ty()?;
                self.eat(&Tok::RParen)?;
                Ok(t)
            }
            other => Err(format!("expected a type atom, found {other:?}")),
        }
    }

    fn parse_postulate(&mut self, linear: bool) -> Result<Item, String> {
        self.eat(&Tok::KwPostulate)?;
        let name = self.ident()?;
        self.eat(&Tok::Colon)?;
        let ty = self.parse_ty()?;
        Ok(Item::Postulate(name, ty, linear))
    }

    fn parse_enum(&mut self, linear: bool, boxed: bool) -> Result<Item, String> {
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
        Ok(Item::Enum { name, params, index_ty, variants, linear, boxed })
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
            // an anonymous function `\x => e` / `\x y => e` (space-separated
            // parameters, so it needs no parentheses inside a comma-separated
            // argument list — `map(\x => x + x, xs)`).
            Some(Tok::Backslash) => {
                self.next();
                let mut params = Vec::new();
                while let Some(Tok::Ident(_)) = self.peek() {
                    params.push(self.ident()?);
                }
                if params.is_empty() {
                    return Err("a lambda `\\… => …` needs at least one parameter".into());
                }
                self.eat(&Tok::FatArrow)?;
                let body = self.parse_tm()?;
                Ok(Tm::Lam(params, Box::new(body)))
            }
            // `rewrite <prf> in <expr>` (Idris-style) — rewrite the expected goal
            // along the equality `prf`, then check `expr` against the rewritten
            // goal. Desugared to a reserved call `@rewrite(prf, expr)`, handled in
            // `check` (it needs the expected type).
            Some(Tok::KwRewrite) => {
                self.next();
                let prf = self.parse_add()?;
                self.eat(&Tok::KwIn)?;
                let body = self.parse_tm()?;
                Ok(Tm::Call("@rewrite".to_string(), vec![prf, body]))
            }
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
                    let rhs = self.parse_add()?;
                    self.eat(&Tok::Semi)?;
                    let body = self.parse_tm()?;
                    Ok(Tm::LetPair(names, Box::new(rhs), Box::new(body)))
                } else {
                    let name = self.ident()?;
                    // optional type ascription: `let x : T = e` checks `e` at `T`
                    // (drives `cast`/context-typed inference).
                    let ann = if self.peek() == Some(&Tok::Colon) {
                        self.next();
                        Some(self.parse_ty_app()?)
                    } else {
                        None
                    };
                    self.eat(&Tok::Eq)?;
                    let rhs = self.parse_add()?;
                    self.eat(&Tok::Semi)?;
                    let body = self.parse_tm()?;
                    let rhs = match ann {
                        Some(ty) => Tm::Ann(Box::new(rhs), Box::new(ty)),
                        None => rhs,
                    };
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
            // `123u8` ⇒ `u8(123)`; `3.14` ⇒ `f64_bits(<bits>)` (reinterpret the
            // Nat's bit pattern as a float — codegen identity, floats ARE i64 bits).
            Some(Tok::NumSuffixed(n, suf)) => {
                return Ok(Tm::Call(suf, vec![Tm::Lit(n)]))
            }
            Some(Tok::FloatLit(bits, is_f32)) => {
                let f = if is_f32 { "f32_bits" } else { "f64_bits" };
                return Ok(Tm::Call(f.to_string(), vec![Tm::Lit(bits)]));
            }
            Some(Tok::Str(x)) => return Ok(Tm::Str(x)),
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

    /// A match-arm argument pattern: `name` or `Ctor(pat, …)` (recursive).
    fn parse_pattern(&mut self) -> Result<Pat, String> {
        let name = self.ident()?;
        if self.peek() == Some(&Tok::LParen) {
            self.next();
            let mut subs = Vec::new();
            while self.peek() != Some(&Tok::RParen) {
                subs.push(self.parse_pattern()?);
                if self.peek() == Some(&Tok::Comma) {
                    self.next();
                }
            }
            self.eat(&Tok::RParen)?;
            Ok(Pat::Ctor(name, subs))
        } else {
            Ok(Pat::Var(name))
        }
    }

    fn parse_match(&mut self) -> Result<Tm, String> {
        self.eat(&Tok::KwMatch)?;
        // the scrutinee may be an EXPRESSION (a call / paren), not just a var; a
        // non-var scrutinee is desugared to `let $s = <expr>; match $s { … }`.
        // MULTIPLE comma-separated scrutinees (Phase A1) parse to `MatchN`, whose
        // arms are full pattern ROWS compiled by the pattern matrix.
        let mut scrut_tms = vec![self.parse_call()?];
        while self.peek() == Some(&Tok::Comma) {
            self.next();
            scrut_tms.push(self.parse_call()?);
        }
        if scrut_tms.len() > 1 {
            return self.parse_match_n(scrut_tms);
        }
        let scrut_tm = scrut_tms.pop().unwrap();
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
            let mut pats = Vec::new();
            if self.peek() == Some(&Tok::LParen) {
                self.next();
                while self.peek() != Some(&Tok::RParen) {
                    let pat = self.parse_pattern()?;
                    // a NESTED pattern gets a fresh placeholder binder; the
                    // pattern-matrix desugar (`desugar_patterns`) replaces it.
                    binders.push(match &pat {
                        Pat::Var(v) => v.clone(),
                        Pat::Ctor(_, _) => {
                            let n = format!("$p{}", self.fresh);
                            self.fresh += 1;
                            n
                        }
                    });
                    pats.push(pat);
                    if self.peek() == Some(&Tok::Comma) {
                        self.next();
                    }
                }
                self.eat(&Tok::RParen)?;
            }
            self.eat(&Tok::FatArrow)?;
            let body = self.parse_tm()?;
            arms.push(Arm { ctor, binders, pats, body });
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

    /// The MULTI-SCRUTINEE match body (Phase A1): each arm is a row of exactly
    /// one full pattern per scrutinee, `p, q => body`. Non-var scrutinees are
    /// let-bound left-to-right (evaluation order preserved).
    fn parse_match_n(&mut self, scrut_tms: Vec<Tm>) -> Result<Tm, String> {
        let mut binds: Vec<(String, Tm)> = Vec::new();
        let mut cols: Vec<String> = Vec::new();
        for t in scrut_tms {
            match t {
                Tm::Var(v) => cols.push(v),
                other => {
                    let s = format!("$m{}", self.fresh);
                    self.fresh += 1;
                    binds.push((s.clone(), other));
                    cols.push(s);
                }
            }
        }
        self.eat(&Tok::LBrace)?;
        let n = cols.len();
        let mut rows = Vec::new();
        while self.peek() != Some(&Tok::RBrace) {
            let mut pats = Vec::with_capacity(n);
            for k in 0..n {
                pats.push(self.parse_pattern()?);
                if k + 1 < n {
                    self.eat(&Tok::Comma)?;
                }
            }
            self.eat(&Tok::FatArrow)?;
            let body = self.parse_tm()?;
            rows.push((pats, body));
            if self.peek() == Some(&Tok::Comma) {
                self.next();
            }
        }
        self.eat(&Tok::RBrace)?;
        if rows.is_empty() {
            return Err(
                "a multi-scrutinee `match` needs at least one arm (write an absurd \
                 match on the single empty scrutinee instead)"
                    .into(),
            );
        }
        let mut m = Tm::MatchN(cols, rows);
        for (s, e) in binds.into_iter().rev() {
            m = Tm::Let(s, Box::new(e), Box::new(m));
        }
        Ok(m)
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
    /// the type of a FULLY-APPLIED recursive call (`ih_for`'s result): the bare
    /// IH's type in verbatim-fold mode, `R` in accumulator mode. All these
    /// motives are CONSTANT (non-dependent), so one value suffices — it lets a
    /// recursive call be INFERRED (e.g. as a `let` right-hand side), not only
    /// checked.
    result_ty: Value,
    /// CONVOY-FOLD mode: the recursive call's varying arguments are exactly the
    /// convoy's index-dependent deps, and the IH is a FUNCTION of them (the
    /// dependent motive abstracts them) — apply it, checked against the IH's
    /// refined Π domains.
    fold: Option<&'a FnFold>,
}

/// A structurally-descending recursive fn whose non-scrutinee arguments vary
/// — legal when they are exactly the scrutinee's index-dependent values (the
/// convoy deps): the dependent motive abstracts them, so the eliminator's IH
/// is a function of them and the recursion is a kernel-checked TOTAL fold.
struct FnFold {
    fnname: String,
    /// position of the scrutinee among the EXPLICIT parameters.
    scrut_pos: usize,
    /// explicit-argument positions of the convoy deps, in motive dep order.
    dep_args: Vec<usize>,
    explicit_params: Vec<String>,
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
    /// BASE constructor names of mult-poly datatype families: a family
    /// `FlatClo (m : Mult) …` is monomorphised to `FlatClo$1`/`FlatClo$w`, so the
    /// base name `FlatClo` (what the user writes) is not itself a constructor.
    /// It resolves to an instance constructor via the expected/scrutinee type
    /// (`resolve_mono_ctor`); this set lets `match` desugaring accept the base
    /// name instead of rejecting it as an unknown constructor.
    poly_ctor_base: std::collections::HashSet<String>,
    /// set while elaborating the body of a `%partial` `Fix` (general recursion). In
    /// this mode a boxed `match` lowers to a NON-recursive `Term::Case` (recursion is
    /// the `Fix` self-call), not the recursive `Term::Elim` (whose implicit IHs would
    /// blow up exponentially with the self-calls). Interior mutability so the
    /// `&self` elaboration methods can scope it around a `Fix` body.
    in_fix: std::cell::Cell<bool>,
    /// current substitution for multiplicity variables (`(m : Mult)` params),
    /// set while elaborating a monomorphized instance of a mult-poly function.
    /// Empty for ordinary code, where every `SMult` is already `Lit`.
    mult_env: std::cell::RefCell<HashMap<String, Mult>>,
    /// type-parameter names currently assumed LINEAR: the pessimistic mode of
    /// the linear-capability check (`def_linear_capable`). Empty for ordinary
    /// elaboration.
    pess_linear: std::cell::RefCell<std::collections::HashSet<String>>,
    /// per-definition LINEAR-CAPABILITY of each parameter position: `true` when
    /// the body stays usage-valid with that (Type-)parameter assumed linear,
    /// INFERRED by re-elaborating the body pessimistically. Instantiating a
    /// non-capable parameter at a linear type is rejected at the call site
    /// (closing the FUTURE_WORK s13 "generic code drops a linear element" hole).
    def_linear_capable: std::cell::RefCell<HashMap<String, Vec<bool>>>,
    /// allocator for GLOBALLY-UNIQUE elaboration-hole ids (offsets above
    /// `HOLE_BASE`). Every in-flight solver (`solve_ctor`/`solve_fn_call`,
    /// nested calls included) draws a disjoint block, so a hole leaked into a
    /// nested elaboration is FOREIGN there (opaque), never aliased into the
    /// inner solver's slots (which used to corrupt them and panic on quote).
    hole_ctr: std::cell::Cell<usize>,
}

fn neutral_env(n: usize) -> Vec<Value> {
    (0..n).map(dep::nvar).collect()
}

impl Elab {
    fn debruijn(scope: &[String], name: &str) -> Option<usize> {
        scope.iter().rev().position(|s| s == name)
    }

    /// Allocate a DISJOINT block of `len` elaboration holes (see `Holes`).
    fn fresh_hole_block(&self, len: usize) -> Holes {
        let base = self.hole_ctr.get();
        self.hole_ctr.set(base + len);
        Holes { base, slots: vec![None; len] }
    }

    /// A fresh synthetic neutral backed by NO hole block: opaque to every
    /// solver (nothing can bind it). Used as the placeholder value of a
    /// deferred argument.
    fn fresh_opaque(&self) -> Value {
        let base = self.hole_ctr.get();
        self.hole_ctr.set(base + 1);
        dep::nvar(HOLE_BASE + base)
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
    /// Would applying a first-class function VALUE of type `vty` to `nargs`
    /// arguments leave a `Π` result — a PARTIAL application, or one that returns a
    /// function? The native backend has no currying and cannot form such a
    /// closure, so this is a codegen-soundness gate (see the `elab_tm` call site).
    /// `level` is the context depth, for the fresh vars that peel dependent codomains.
    fn value_app_is_partial(&self, vty: &Value, nargs: usize, level: usize) -> bool {
        let mut t = vty.clone();
        for i in 0..nargs {
            match t {
                Value::VPi(_, _, cod) => t = cod.apply(dep::nvar(level + i)),
                _ => return false, // over-applied — the kernel/other paths handle it
            }
        }
        matches!(t, Value::VPi(_, _, _))
    }

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
        // The linear-Nat order built-ins are a FALLBACK — a user `enum Lt`,
        // `fn lt`, etc. always wins (checked above). Types `Le a b` / `Lt a b`
        // (see `mk_ineq_type`); proofs `le(a,b)` / `lt(a,b)` run the stratum-(A)
        // solver and emit an LCF-checked witness (see `mk_ineq_proof`). All four
        // need `%builtin Nat` (index `+`).
        match (name, args.len()) {
            ("Le", 2) => {
                let mut a = args.into_iter();
                return Ok(mk_ineq_type(&a.next().unwrap(), &a.next().unwrap(), false));
            }
            ("Lt", 2) => {
                let mut a = args.into_iter();
                return Ok(mk_ineq_type(&a.next().unwrap(), &a.next().unwrap(), true));
            }
            ("le", 2) => {
                let mut a = args.into_iter();
                return mk_ineq_proof(&a.next().unwrap(), &a.next().unwrap(), false);
            }
            ("lt", 2) => {
                let mut a = args.into_iter();
                return mk_ineq_proof(&a.next().unwrap(), &a.next().unwrap(), true);
            }
            _ => {}
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
            Ty::Type(l) => Ok(Term::Type(*l)),
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
                let m = m.resolve(&self.mult_env.borrow())?;
                let m = if m == Mult::Omega && type_is_linear(&ta, &self.rc) { Mult::One } else { m };
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
            Ty::Add(a, b) => Ok(Term::Add(
                Box::new(self.elab_ty(a, scope)?),
                Box::new(self.elab_ty(b, scope)?),
            )),
            Ty::Lit(n) => Ok(Term::NatLit(*n)),
            Ty::TypeV(v) => Err(format!(
                "unresolved level variable `{v}` — a `Type {v}` may only mention an \
                 enclosing `(l : Level)` parameter (instantiated per call site)"
            )),
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
            Tm::Var(name) if self.ctor_has_implicits(&self.resolve_ctor_expected(name, expected)) => {
                let rn = self.resolve_ctor_expected(name, expected);
                self.solve_ctor(&rn, &[], expected, cx, rec)
            }
            // `rewrite prf in expr` (desugared to `@rewrite(prf, expr)`) — needs the
            // expected goal, so it is a CHECK-position form (see `elab_rewrite`).
            Tm::Call(name, args) if name == "@rewrite" && args.len() == 2 => {
                self.elab_rewrite(&args[0], &args[1], expected, cx, rec)
            }
            Tm::Call(name, args) => {
                if let Some(r) = rec {
                    if name == r.fnname {
                        return self.ih_for(r, args, cx);
                    }
                }
                // A mult-poly BASE constructor (`FlatClo`) resolves to the
                // instance (`FlatClo$1`) named by the expected datatype.
                let rn = self.resolve_ctor_expected(name, expected);
                if self.ctor_has_implicits(&rn) {
                    return self.solve_ctor(&rn, args, expected, cx, rec);
                }
                // A LOCAL BINDER shadows a global def (see `elab_tm`): don't route
                // a shadowed name through the implicit solver — e.g. `foldr`'s
                // param `f` must win over a top-level `f : {0 n} -> …` (which would
                // otherwise be solved with the wrong arity). The self-binder
                // (registered implicits, absent from `defs`) still solves here.
                let shadowed = cx.debruijn(name).is_some();
                if self.def_has_implicits(name) && (!shadowed || !self.defs.contains_key(name)) {
                    return Ok(self.solve_fn_call(name, args, Some(expected), cx, rec)?.0);
                }
                self.elab_tm(tm, cx, rec)
            }
            // an anonymous function `\x => e` (P5): elaborate against the expected
            // function type, binding each parameter in a FRESH context (only the
            // lambda's own parameters). A reference to an enclosing LOCAL is thus
            // an "unbound name" error — capturing needs the explicit closure
            // representations (§3); a non-capturing lambda is a CLOSED function
            // value, wrapped in its type so P1 `fn_value`/P4 treat it exactly like
            // a named `fn` (materialise + specialise, zero-cost).
            Tm::Lam(params, body) => {
                let mut lam_cx = Cx { names: Vec::new(), types: Vec::new() };
                let mut ety = expected.clone();
                for p in params {
                    match ety {
                        Value::VPi(_m, dom, cod) => {
                            let lvl = lam_cx.len();
                            lam_cx.push(p.clone(), (*dom).clone());
                            ety = cod.apply(dep::nvar(lvl));
                        }
                        _ => {
                            return Err(format!(
                                "a lambda with {} parameter(s) is checked against a \
                                 non-function type — the call/annotation must give it an \
                                 expected function type",
                                params.len()
                            ))
                        }
                    }
                }
                let mut out = self.check(body, &ety, &lam_cx, None)?;
                for _ in params {
                    out = Term::Lam(Box::new(out));
                }
                Ok(Term::Ann(Box::new(out), Box::new(dep::quote_at(cx.len(), expected))))
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
                if let Some(fo) = r.fold {
                    // CONVOY FOLD: apply the IH to the arguments at the DEP
                    // positions, checked against its (per-arm REFINED) Π
                    // domains. Every other non-scrutinee argument must be
                    // passed VERBATIM — it is captured, not threaded, so a
                    // changed value there would be silently dropped (the
                    // 1a′ value-correctness guard).
                    for (i, a) in args.iter().enumerate() {
                        if i == r.scrut_pos || fo.dep_args.contains(&i) {
                            continue;
                        }
                        let verbatim = matches!(a, Tm::Var(av)
                            if Some(av.as_str()) == fo.explicit_params.get(i).map(|p| p.as_str()));
                        if !verbatim {
                            return Err(format!(
                                "recursive call to `{}`: argument #{} must be passed                                  through unchanged (only the index-dependent arguments                                  the match refines may vary)",
                                r.fnname,
                                i + 1
                            ));
                        }
                    }
                    let mut t = ih;
                    let mut ty = cx.var_type(&r.fields[v]).expect("ih type in scope");
                    for p in &fo.dep_args {
                        let a = args.get(*p).ok_or_else(|| {
                            format!("recursive call to `{}` has too few arguments", r.fnname)
                        })?;
                        let Value::VPi(_, dom, cod) = ty else {
                            return Err(format!(
                                "recursive call to `{}`: the induction hypothesis accepts                                  fewer varying arguments than supplied",
                                r.fnname
                            ));
                        };
                        let ea = self.check(a, &dom, cx, Some(r))?;
                        let va = self.eval(cx.len(), &ea);
                        t = Term::App(Box::new(t), Box::new(ea));
                        ty = cod.apply(va);
                    }
                    return Ok(t);
                }
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
            // a type ascription infers via `infer_arg` (checks against the annotation).
            Tm::Ann(_, _) => self.infer_arg(t, cx, rec).map(|(tm, _)| tm),
            // a lambda in INFER position: its parameter types are unknown. Use it
            // where a function type is expected (a HOF argument, or annotated).
            Tm::Lam(_, _) => Err(
                "a lambda `\\… => …` needs an expected function type — pass it where \
                 a function is expected (e.g. a higher-order argument) or annotate it"
                    .into(),
            ),
            Tm::MatchN(_, _) => Err(
                "internal error: a multi-scrutinee `match` survived pattern \
                 desugaring — this is a compiler bug, please report it"
                    .into(),
            ),
            Tm::Lit(n) => Ok(Term::NatLit(*n)),
            Tm::Str(x) => Ok(Term::StrLit(x.clone())),
            Tm::Add(a, b) => Ok(Term::Add(
                Box::new(self.elab_tm(a, cx, rec)?),
                Box::new(self.elab_tm(b, cx, rec)?),
            )),
            Tm::Var(name) => self.resolve(name, vec![], &cx.names, true),
            Tm::Call(name, args) => {
                if name == "J" && args.len() == 3 {
                    return self.elab_j(&args[0], &args[1], &args[2], cx, rec).map(|(tm, _)| tm);
                }
                if name == "@rewrite" {
                    return Err("`rewrite … in …` needs an expected type — use it where the \
                                goal is known (a function body, a `let` with an annotation, \
                                or an argument position), not in an inferred position"
                        .into());
                }
                if let Some(r) = rec {
                    if name == r.fnname {
                        return self.ih_for(r, args, cx);
                    }
                }
                // SOUNDNESS: a first-class function VALUE (a local binder of function
                // type) applied so that the RESULT is still a `Π` — a partial
                // application, or one that returns a function — cannot be lowered:
                // the native backend has no currying, so it would emit an indirect
                // call with the wrong arity (a wild call / memory fault), and codegen
                // is trusted (not kernel-re-checked). Reject it. Saturated and
                // over-applied calls fall through unchanged.
                if let Some(vty) = cx.var_type(name) {
                    if !args.is_empty() && self.value_app_is_partial(&vty, args.len(), cx.len()) {
                        return Err(format!(
                            "partial application of the first-class function `{name}`: it \
                             is applied to fewer arguments than its type takes (or returns \
                             a function), which the native backend cannot form. Apply it \
                             to all its arguments, or build an explicit closure."
                        ));
                    }
                }
                // A LOCAL BINDER shadows a global def of the same name — an
                // ordinary parameter `f` must win over a top-level `f`. Both the
                // implicit-solver and known-Π-telescope branches below resolve
                // the GLOBAL def, so gate them on the context: skip when `name` is
                // bound locally (unless it is the in-flight self-binder, which
                // carries registered implicits and is absent from `defs`). Without
                // this, `foldr`'s param `f` (applied to 2 args) resolved to a
                // top-level `f : Nat -> Nat` ⇒ "`f` is applied to too many
                // arguments". Falls through to `resolve`, which builds the
                // context-variable application.
                let shadowed = cx.debruijn(name).is_some();
                if self.def_has_implicits(name) && (!shadowed || !self.defs.contains_key(name)) {
                    return Ok(self.solve_fn_call(name, args, None, cx, rec)?.0);
                }
                // A NO-implicit definition still has a KNOWN Π telescope: CHECK
                // each argument against its domain (so a constructor-with-
                // implicits argument — `mixed(LCons(5, LNil))` — elaborates),
                // instead of blindly inferring it.
                if let Some((_, fty)) = self.defs.get(name).filter(|_| !shadowed) {
                    let mut fty = fty.clone();
                    let mut dom_env: Vec<Value> = Vec::new();
                    let mut eargs = Vec::with_capacity(args.len());
                    for a in args {
                        let Term::Pi(_, dom, cod) = fty else {
                            return Err(format!("`{name}` is applied to too many arguments"));
                        };
                        let dom_val = dep::eval_rc(&self.rc, &dom_env, &dom);
                        let ta = self.check(a, &dom_val, cx, rec)?;
                        dom_env.push(self.eval(cx.len(), &ta));
                        eargs.push(ta);
                        fty = *cod;
                    }
                    return self.resolve(name, eargs, &cx.names, true);
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
    /// `J(\z e => P, base, proof)` — path induction (the eliminator of `Eq`),
    /// surfaced here (the kernel implements `Term::J`; this is the elaboration).
    /// `proof : Eq A x y`; the MOTIVE `\z e => P` abstracts the varying endpoint
    /// `z : A` and the proof `e : Eq A x z` — so it may mention the enclosing
    /// context (it is a type-level, erased abstraction, elaborated UNDER `cx`, not
    /// in a fresh one like a runtime lambda). `base : P x (refl x)`, and the whole
    /// thing has type `P y proof`. `cong`/`sym`/`trans`/`subst`/`rewrite` are all
    /// this. The kernel re-checks the emitted `Term::J`, so a wrong motive is a
    /// rejected program, never unsound.
    fn elab_j(&self, motive: &Tm, base: &Tm, proof: &Tm, cx: &Cx, rec: Option<&Rec>) -> Result<(Term, Value), String> {
        let (proof_tm, proof_ty) = self.infer_arg(proof, cx, rec)?;
        let (a, x, y) = match &proof_ty {
            Value::VEq(a, x, y) => ((**a).clone(), (**x).clone(), (**y).clone()),
            _ => return Err("`J`: the third argument must be an equality proof `Eq A x y`".into()),
        };
        let (params, body) = match motive {
            Tm::Lam(ps, b) => (ps, b),
            _ => return Err("`J`: the first argument must be a motive lambda `\\z e => <type>`".into()),
        };
        if params.len() != 2 {
            return Err(
                "`J`: the motive lambda takes two parameters `\\z e => <type>` — the varying \
                 endpoint and the proof"
                    .into(),
            );
        }
        // elaborate the motive body under `z : A`, `e : Eq A x z` (extending `cx`,
        // so it may reference the enclosing context).
        let n = cx.len();
        let mut mcx = cx.clone();
        mcx.push(params[0].clone(), a.clone());
        let z_val = dep::nvar(n);
        mcx.push(
            params[1].clone(),
            Value::VEq(Box::new(a.clone()), Box::new(x.clone()), Box::new(z_val)),
        );
        let body_tm = self.elab_tm(body, &mcx, None)?;
        let motive_tm = Term::Lam(Box::new(Term::Lam(Box::new(body_tm))));
        // base : P x (refl x)
        let vp = self.eval(cx.len(), &motive_tm);
        let base_ty = dep::vapp(dep::vapp(vp.clone(), x.clone()), Value::VRefl(Box::new(x.clone())));
        let base_tm = self.check(base, &base_ty, cx, rec)?;
        // result : P y proof
        let proof_val = self.eval(cx.len(), &proof_tm);
        let result_ty = dep::vapp(dep::vapp(vp, y), proof_val);
        Ok((Term::J(Box::new(motive_tm), Box::new(base_tm), Box::new(proof_tm)), result_ty))
    }

    /// `rewrite prf in expr` — Idris-style equational rewriting, in CHECK
    /// position (it needs the goal). `prf : Eq A x y`; the goal `G` is rewritten
    /// by ABSTRACTING the right-hand side `y` — motive `M w = G[y := w]` — so
    /// `expr` is checked against `M x = G[y := x]` (the goal with `y` replaced by
    /// `x`), and the result `J(M, expr, prf) : M y = G` is the original goal. This
    /// is the motive inference `rewrite` gives you: you write neither the motive
    /// nor `J`. The kernel re-checks the `Term::J`, so it is sound.
    fn elab_rewrite(&self, prf: &Tm, expr: &Tm, expected: &Value, cx: &Cx, rec: Option<&Rec>) -> Result<Term, String> {
        let (prf_tm, prf_ty) = self.infer_arg(prf, cx, rec)?;
        let (x, y) = match &prf_ty {
            Value::VEq(_, x, y) => ((**x).clone(), (**y).clone()),
            _ => return Err("`rewrite`: the proof must be an equality `Eq A x y`".into()),
        };
        let n = cx.len();
        // motive body = `goal[y := w]` (abstract the RHS out of the goal),
        // shifted under the two motive binders `\w e => …`; `w` is `Var(1)`.
        let gt = dep::shift_term(2, &dep::quote_at(n, expected));
        let yt = dep::shift_term(2, &dep::quote_at(n, &y));
        let body = dep::abstract_subterm(&gt, &yt, 1, 0);
        let motive_tm = Term::Lam(Box::new(Term::Lam(Box::new(body))));
        // expr : `M x (refl x)` — the goal with the RHS replaced by the LHS.
        let vp = self.eval(cx.len(), &motive_tm);
        let base_ty = dep::vapp(dep::vapp(vp, x.clone()), Value::VRefl(Box::new(x)));
        let expr_tm = self.check(expr, &base_ty, cx, rec)?;
        // `J(M, expr, prf) : M y prf = goal`.
        Ok(Term::J(Box::new(motive_tm), Box::new(expr_tm), Box::new(prf_tm)))
    }

    fn infer_arg(&self, t: &Tm, cx: &Cx, rec: Option<&Rec>) -> Result<(Term, Value), String> {
        match t {
            // `J(motive, base, proof)` — path induction (see `elab_j`).
            Tm::Call(name, args) if name == "J" && args.len() == 3 => {
                self.elab_j(&args[0], &args[1], &args[2], cx, rec)
            }
            // `(e : T)` — elaborate the annotation, CHECK `e` against it, and
            // report `T` as the inferred type (the ascription IS the inference
            // for a context-typed form like `cast`).
            Tm::Ann(e, ty) => {
                let ty_tm = self.elab_ty(ty, &cx.names)?;
                let ty_val = self.eval(cx.len(), &ty_tm);
                let e_tm = self.check(e, &ty_val, cx, rec)?;
                Ok((e_tm, ty_val))
            }
            // built-in Nat: literals, addition, and the `%builtin Nat` intro forms
            // all have type `Nat` (packed), so they infer with no expected type.
            Tm::Lit(n) => Ok((Term::NatLit(*n), Value::VNat)),
            Tm::Str(x) => Ok((
                Term::StrLit(x.clone()),
                Value::VNeu(dep::Neutral::NConst("Str".into())),
            )),
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
            // a RECURSIVE call in INFER position (e.g. `let r = f(x, t)`): the same
            // IH mapping `check` uses; its fully-applied type is the (constant)
            // motive result carried on `Rec`.
            Tm::Call(name, args) if rec.is_some_and(|r| r.fnname == name) => {
                let r = rec.unwrap();
                let t = self.ih_for(r, args, cx)?;
                // For a DIRECT structural IH (verbatim fold), the hypothesis
                // variable already carries its correctly-SUBSTITUTED dependent type
                // in the context (`P k` for the actual predecessor `k`), whereas
                // `r.result_ty` is only the motive at a fresh NEUTRAL — good enough
                // to CHECK against a known expected type, but it mis-informs a
                // caller that must INFER an implicit endpoint from this call's type
                // (e.g. `cong(f, plusZeroR(k))`). Prefer the context type.
                let ty = match (r.acc_tys, r.fold, args.get(r.scrut_pos)) {
                    (None, None, Some(Tm::Var(v))) => r
                        .fields
                        .get(v)
                        .and_then(|f| cx.var_type(f))
                        .unwrap_or_else(|| r.result_ty.clone()),
                    _ => r.result_ty.clone(),
                };
                Ok((t, ty))
            }
            // a call to a def — or to a CONTEXT VARIABLE WITH REGISTERED IMPLICITS
            // (the `Fix` self-binder of the function being defined): both go through
            // the implicit solver (`solve_fn_call` resolves a not-in-`defs` callee as
            // the in-scope self binder). Without the second disjunct, a self-call in
            // INFER position (e.g. `let va = eval(env, a)`) fell to the curried
            // context-variable path below, which never solves implicits — it consumed
            // the explicit args against the IMPLICIT Πs and built a wrong partial
            // application the kernel then rejected.
            //
            // A LOCAL BINDER SHADOWS A GLOBAL DEF of the same name: an ordinary
            // parameter `p` must win over a top-level `p`, so both disjuncts are
            // gated on the context — the first fires only when `name` is NOT in
            // scope as a local, the second only for the self-binder (registered
            // implicits, in the context, and NOT a def — the in-flight function is
            // absent from `defs`). Without this, elaborating e.g. `filter`'s body
            // `match p(h) { … }` when a global `p : Nat -> Unit` exists resolved
            // `p` to the global (so `p(h) : Unit`, and `True`/`False` were rejected
            // as non-constructors of `Unit`). The `check` path already prefers the
            // local binder; this brings the INFER path to parity.
            Tm::Call(name, args)
                if (self.defs.contains_key(name) && cx.var_type(name).is_none())
                    || (!self.defs.contains_key(name)
                        && self.def_has_implicits(name)
                        && cx.var_type(name).is_some()) =>
            {
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
            // a call to a CONTEXT VARIABLE — most importantly the `Fix` self-binder inside
            // a `%partial` recursive body (`let s = self(x); …`). Infer it as a curried
            // application: peel one `Π` domain per argument, check the argument, and
            // instantiate the codomain with the argument value (so a dependent self-type
            // stays correct). The kernel re-checks the produced application (incl. its
            // linearity); this only ELABORATES it. (Self-binders carry no implicits.)
            Tm::Call(name, args) if cx.var_type(name).is_some() => {
                let idx = cx.debruijn(name).unwrap();
                let mut head = Term::Var(idx);
                let mut head_ty = cx.var_type(name).unwrap();
                for arg in args {
                    let (dom, cod) = match head_ty {
                        Value::VPi(_, dom, cod) => (dom, cod),
                        _ => {
                            return Err(format!(
                                "`{name}` is applied to more arguments than its type allows"
                            ))
                        }
                    };
                    let arg_tm = self.check(arg, &dom, cx, rec)?;
                    let arg_v = self.eval(cx.len(), &arg_tm);
                    head = Term::App(Box::new(head), Box::new(arg_tm));
                    head_ty = cod.apply(arg_v);
                }
                // SOUNDNESS: a first-class function VALUE applied to FEWER arguments
                // than its arity (result still a `Π`) is a PARTIAL application. The
                // native backend has no currying — it would emit an indirect call
                // with the wrong number of arguments (a wild call / memory fault),
                // and codegen is trusted (not re-checked). Reject it here.
                if matches!(head_ty, Value::VPi(_, _, _)) {
                    return Err(format!(
                        "partial application of the first-class function `{name}`: it is \
                         applied to fewer arguments than its type takes, which would need \
                         currying / a closure the native backend cannot form. Apply it to \
                         all its arguments, or build an explicit closure."
                    ));
                }
                Ok((head, head_ty))
            }
            _ => Err("cannot infer the type of this argument (Phase 2)".into()),
        }
    }

    /// Solve a constructor's implicit arguments by matching its result type
    /// against `expected`, then elaborate the explicit arguments.
    /// Resolve a possibly-BASE constructor name against a target datatype NAME.
    /// A mult-poly datatype instance `D$k` has mangled constructors `C$k`, but
    /// the user writes the base `C`; if `cname` is not itself a registered
    /// constructor and `data` carries a mult suffix (`…$k`), retry `C$k`.
    fn resolve_mono_ctor(&self, cname: &str, data: &str) -> String {
        if self.ctor_info.contains_key(cname) {
            return cname.to_string();
        }
        if let Some(idx) = data.rfind('$') {
            let mangled = format!("{cname}{}", &data[idx..]);
            if self.ctor_info.contains_key(&mangled) {
                return mangled;
            }
        }
        cname.to_string()
    }

    /// As `resolve_mono_ctor`, taking the datatype name from an EXPECTED value
    /// (the check-position constructor case).
    fn resolve_ctor_expected(&self, cname: &str, expected: &Value) -> String {
        match expected {
            Value::VData(dname, _) => self.resolve_mono_ctor(cname, dname),
            _ => cname.to_string(),
        }
    }

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
        let mut holes = self.fresh_hole_block(total);
        let hole_env: Vec<Value> =
            (0..total).map(|id| dep::nvar(HOLE_BASE + holes.base + id)).collect();
        let mut result_args: Vec<Value> = (0..np).map(|p| hole_env[p].clone()).collect();
        for idx in &ctor.idxs {
            result_args.push(dep::eval_rc(&self.rc, &hole_env, idx));
        }
        let result = Value::VData(info.data.clone(), result_args);
        solve(&mut holes, &result, expected);

        // PRE-SOLVE from the explicit arguments' inferred types. `solve(result,
        // expected)` above only pins implicits that appear in the RESULT; an
        // EXISTENTIAL constructor implicit — `{0 e}` in
        // `MkOwnClo : {0 e} -> (1 env : Own e) -> (code : e -> a -> b) -> OwnClo a b`
        // — appears only in a LATER explicit argument's type (`Own e`, `e -> a -> b`),
        // never in the result, so the expected type cannot determine it. Recover it
        // the way `solve_fn_call` does: infer each explicit argument's type and unify
        // it against its (hole-)domain. Each domain is evaluated with holes for the
        // EARLIER positions only (`hole_env[0..pos]`), matching the de Bruijn depth of
        // the incremental walk below. Purely additive — it can only BIND MORE holes,
        // and the kernel re-check remains the backstop for any mis-inference.
        {
            let mut nu = 0;
            for pos in 0..total {
                if implicit_of(pos) {
                    continue;
                }
                let dom_tm = if pos < np { &decl.params[pos].1 } else { &ctor.args[pos - np].1 };
                let dom_val = dep::eval_rc(&self.rc, &hole_env[0..pos], dom_tm);
                if let Ok((_, arg_ty)) = self.infer_arg(&user_args[nu], cx, rec) {
                    solve(&mut holes, &dom_val, &arg_ty);
                }
                nu += 1;
            }
        }

        // walk positions left to right, filling values + terms
        let mut env: Vec<Value> = Vec::with_capacity(total);
        let mut terms: Vec<Term> = Vec::with_capacity(total);
        let mut next_user = 0;
        for pos in 0..total {
            let dom_tm = if pos < np { &decl.params[pos].1 } else { &ctor.args[pos - np].1 };
            if implicit_of(pos) {
                let sol = holes.slots[pos]
                    .clone()
                    .ok_or_else(|| format!("cannot infer implicit argument of `{cname}`"))?;
                // A "solution" that still mentions a hole (its own, or one leaked
                // from an ENCLOSING call's open implicit) is not a solution —
                // report it cleanly rather than quoting it (which would underflow
                // on the synthetic hole level). Same guard as `solve_fn_call`.
                if value_has_hole(&sol) {
                    return Err(format!(
                        "cannot infer implicit argument of `{cname}` (it depends on an \
                         implicit of the enclosing call that is not yet determined — \
                         bind the argument to a named, annotated definition first)"
                    ));
                }
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
        // `fname` has implicits (the caller checked `def_has_implicits`). It is either a
        // top-level def, OR — inside a `%partial` `Fix` body — the recursive SELF-binder,
        // which is in `cx` but NOT in `defs` (the function being defined). For the self,
        // the "body" is the self VARIABLE (`Var`) and the type is its context type; the
        // implicit-solving below is identical (it reads `def_implicit[fname]`, which is
        // registered for the function regardless). This makes a recursive self-call WITH
        // IMPLICITS in a `Fix` body elaborate (the `(A)` heap-recursion self-call extended
        // to implicits + check-position). The kernel re-checks the resulting application,
        // so a wrong self-resolution / mis-inferred implicit is caught.
        let (body, fty) = match self.defs.get(fname) {
            Some(bf) => bf.clone(),
            None => {
                let idx = cx.debruijn(fname).ok_or_else(|| {
                    format!("cannot resolve the call `{fname}(…)`: not a defined function or in-scope binder")
                })?;
                let fty_tm = dep::quote_at(cx.len(), &cx.var_type(fname).unwrap());
                (Term::Var(idx), fty_tm)
            }
        };
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

        let mut holes = self.fresh_hole_block(total);
        let mut env: Vec<Value> = Vec::with_capacity(total); // arg values, in `cx`'s context
        let mut terms: Vec<Option<Term>> = vec![None; total];
        // explicit args whose elaboration is DEFERRED: their domain still
        // mentioned an open hole in pass 1 (a nested call/constructor with its
        // own implicits can't be checked against a hole), so they are checked
        // AFTER the other args and the expected type have pinned the implicits.
        let mut deferred: Vec<(usize, usize)> = Vec::new(); // (position, user index)
        let mut next_user = 0;
        for pos in 0..total {
            let dom_val = dep::eval_rc(&self.rc, &env, &domains[pos]);
            if flags[pos] {
                env.push(dep::nvar(HOLE_BASE + holes.base + pos)); // a hole, solved below
            } else {
                // try to infer the arg's type (to pin implicits); else check it
                // against the (current) domain — for args that pin nothing —
                // unless that domain still has holes, in which case DEFER it.
                let arg_tm = match self.infer_arg(&user_args[next_user], cx, rec) {
                    Ok((arg_tm, arg_ty)) => {
                        solve(&mut holes, &dom_val, &arg_ty);
                        arg_tm
                    }
                    Err(_) if value_has_hole(&dom_val) => {
                        deferred.push((pos, next_user));
                        next_user += 1;
                        // an OPAQUE placeholder value (a fresh id backed by no
                        // block, so nothing can bind it) — later domains that
                        // mention this argument's VALUE stay neutral.
                        env.push(self.fresh_opaque());
                        continue;
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
                let sol = holes.slots[pos]
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
        // LINEAR-CAPABILITY gate (see `def_linear_capable`): a Type-implicit
        // solved at a LINEAR type is only sound when the callee's body was
        // verified usage-valid with that parameter assumed linear — otherwise a
        // generic body could silently DROP or DUPLICATE the linear values it
        // abstracts over (the FUTURE_WORK §13 parametricity hole). Postulates
        // and the in-flight definition itself (recursive calls during its own
        // capability check) default to capable.
        for pos in 0..total {
            if !flags[pos] || !matches!(domains[pos], Term::Type(_)) {
                continue;
            }
            let Some(sol_tm) = &terms[pos] else { continue };
            if type_is_linear_in(sol_tm, &self.rc, &self.pess_linear.borrow(), &cx.names, n) {
                let capable = self
                    .def_linear_capable
                    .borrow()
                    .get(fname)
                    .map(|v| v.get(pos).copied().unwrap_or(true))
                    .unwrap_or(true);
                if !capable {
                    return Err(format!(
                        "`{fname}` cannot be instantiated at a LINEAR type here: its \
                         body is not verified to use values of that type parameter \
                         exactly once (implicit argument {pos}), which only copyable \
                         types allow. In `{fname}`: consume each such value exactly \
                         once, and sequence a consumption feeding an unrestricted \
                         position through `let` (`let y = f(x); C(y, …)`, not \
                         `C(f(x), …)`)"
                    ));
                }
            }
        }
        // final pass: re-evaluate the telescope with everything solved, checking
        // each DEFERRED argument against its now-concrete domain as we go (an
        // earlier deferred arg's value feeds a later deferred arg's domain).
        let mut final_env: Vec<Value> = Vec::with_capacity(total);
        for pos in 0..total {
            if flags[pos] {
                final_env.push(holes.slots[pos].clone().unwrap());
            } else if let Some((_, ui)) = deferred.iter().find(|(p, _)| *p == pos) {
                let dom_val = dep::eval_rc(&self.rc, &final_env, &domains[pos]);
                if value_has_hole(&dom_val) {
                    return Err(format!(
                        "cannot determine the type of argument {} of `{fname}` \
                         (an implicit it depends on was never pinned)",
                        ui + 1
                    ));
                }
                let arg_tm = self.check(&user_args[*ui], &dom_val, cx, rec)?;
                let v = self.eval(n, &arg_tm);
                terms[pos] = Some(arg_tm);
                final_env.push(v);
            } else {
                final_env.push(env[pos].clone());
            }
        }
        let result_ty = dep::eval_rc(&self.rc, &final_env, &codomain);
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
        // Lower to the kernel's CALL-BY-VALUE `Let` (NOT the β-redex `(λx.body) e`):
        // the β-redex SCALES `e`'s usage by the binder multiplicity, which over-counts
        // the linear resources an effectful `e` consumes — e.g. `let u = free(x);
        // free(y)` (u : Unit, ω) would scale `free(x)` by ω and wrongly reject `x`.
        // The CBV `Let` counts `e` exactly once (`U_e ⊕ U_body`).
        //
        // The binder's QUANTITY is the load-bearing soundness bit: a `let` binding a
        // LINEAR value (whose type carries an `Own`/`Σ[1]`) binds at `1`, so using it
        // twice is `ω ⋢ 1` (double-free) and dropping it is `0 ⋢ 1` (leak); a copyable
        // value binds at `ω` so ordinary `let x = e; … x … x …` works.
        let binder_mult = if type_is_linear_in(&e_ty_tm, &self.rc, &self.pess_linear.borrow(), &cx.names, n) { Mult::One } else { Mult::Omega };
        Ok(Term::Let(binder_mult, Box::new(e_ty_tm), Box::new(e_term), Box::new(body_term)))
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
                // Resolve mult-poly BASE constructor patterns (`FlatClo`) to the
                // instance (`FlatClo$w`) named by the scrutinee datatype `d`.
                let resolved: Vec<Arm>;
                let arms: &[Arm] = if self.poly_ctor_base.contains(&arms[0].ctor)
                    || arms.iter().any(|arm| self.poly_ctor_base.contains(&arm.ctor))
                {
                    resolved = arms
                        .iter()
                        .map(|arm| Arm { ctor: self.resolve_mono_ctor(&arm.ctor, d), ..arm.clone() })
                        .collect();
                    &resolved
                } else {
                    arms
                };
                self.elab_nested_match(&e_term, d, &a, arms, expected, cx, rec, None)
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
        let n = cx.len();
        // THE NAT CONVOY: a context variable whose type mentions the scrutinee, or
        // a linear variable consumed inside the arms, needs the dependent lowering.
        let exp_tm = dep::quote_at(n, expected);
        if let Some(t) =
            self.try_nat_convoy_case(scrut_tm, &exp_tm, zero_arm, succ_arm, cx, rec)?
        {
            return Ok(t);
        }
        // constant motive λ_. expected; methods checked at the same expected type.
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

    /// THE NAT CONVOY — a dependent `NatCase` over a *variable* scrutinee.
    ///
    /// Two things go wrong when a `match` on a `Nat` variable `b` is lowered with
    /// the constant motive: (1) an in-scope value whose TYPE mentions `b`
    /// (`arr : Arr Nat (b + m)`) is stuck at the unrefined type inside the arms —
    /// the `Succ` arm never learns `b = Succ k`; and (2) the kernel's `NatCase`
    /// usage rule ω-scales the methods, so a LINEAR value consumed inside an arm
    /// is rejected (`ω ⋢ 1`) even though exactly one arm runs. Both are the
    /// problem the boxed CONVOY solves (docs/CONVOY_HANDOFF.md), and this is the
    /// same fix at the `Nat` scrutinee: λ-abstract those context variables into a
    /// FUNCTION-TYPED motive `λb'. Π(π₁ d₁ : T₁[b'])… Π(π_K d_K : T_K[b']). R[b']`,
    /// check each arm as a function of the re-bound variables at their REFINED
    /// types (`b' := Zero` / `Succ k`), and apply the whole `NatCase` to the
    /// original variables ONCE — a linear dep is consumed exactly once at the
    /// application, and each method binder carries the per-arm obligation (so
    /// consuming a linear exactly once per arm is accepted, and missing an arm's
    /// consumption is still `0 ⋢ 1`). The elaborator stays untrusted: the kernel
    /// re-checks the result, so a wrong convoy can only be rejected. Codegen
    /// commutes the application back into the arms (the same case-commuting
    /// conversion as `Case`/`Elim`).
    ///
    /// Returns `None` when there is nothing to abstract (the constant-motive
    /// paths are unchanged) or the scrutinee is not a context variable.
    fn try_nat_convoy_case(
        &self,
        scrut_tm: &Term,
        exp_tm: &Term, // the expected/return type, in cx-space
        zero_arm: &Arm,
        succ_arm: &Arm,
        cx: &Cx,
        rec: Option<&Rec>,
    ) -> Result<Option<Term>, String> {
        let n = cx.len();
        let mut sv = scrut_tm;
        while let Term::Ann(inner, _) = sv {
            sv = inner;
        }
        let Term::Var(si) = sv else { return Ok(None) };
        if *si >= n {
            return Ok(None);
        }
        let scrut_lvl = n - 1 - *si;

        // Candidate dependents: referenced by some arm (an unreferenced linear was
        // consumed before the match — re-applying it would double-use it), the
        // innermost binding of their name (an outer shadowed binding cannot be
        // re-bound by a method lambda without capturing the inner one), and the
        // type either mentions the scrutinee (transitively through other
        // dependents) or is linear (the ω-scaling fix).
        let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
        collect_tm_names(&zero_arm.body, &mut referenced);
        collect_tm_names(&succ_arm.body, &mut referenced);
        let mut dep_lvls: Vec<usize> = vec![scrut_lvl];
        let mut deps: Vec<ConvoyDep> = Vec::new();
        for l in 0..n {
            if l == scrut_lvl {
                continue;
            }
            let name = &cx.names[l];
            if !referenced.contains(name) && !name.starts_with("$ih") {
                continue;
            }
            if cx.names.iter().rposition(|s2| s2 == name) != Some(l) {
                continue;
            }
            let ty = dep::quote_at(n, &cx.types[l]);
            let linear =
                type_is_linear_in(&ty, &self.rc, &self.pess_linear.borrow(), &cx.names, n);
            if !linear && !term_mentions_levels(&ty, n, &dep_lvls) {
                continue;
            }
            let mult = if linear { Mult::One } else { Mult::Omega };
            dep_lvls.push(l);
            deps.push(ConvoyDep { lvl: l, name: name.clone(), ty, mult });
        }
        if deps.is_empty() {
            return Ok(None);
        }
        let k = deps.len();

        // motive = λb'. Π deps[b := b']. R[b := b'] — re-target the cx-space terms
        // under the inserted binders (the same arithmetic as `Convoy::remap`, with
        // the scrutinee itself as the index variable).
        let remap = |t: &Term, d: usize, repl: usize, j: usize| -> Term {
            dep::map_vars(t, 0, &|i, depth| {
                if i < depth {
                    return Term::Var(i);
                }
                let outer = i - depth;
                if outer < n {
                    let l = n - 1 - outer;
                    if l == scrut_lvl {
                        return Term::Var(repl + depth);
                    }
                    if let Some(m) = deps[..j].iter().position(|dp| dp.lvl == l) {
                        return Term::Var(j - 1 - m + depth);
                    }
                }
                Term::Var(i + d)
            })
        };
        let mut mbody = remap(exp_tm, 1 + k, k, k);
        for j in (0..k).rev() {
            let ty = remap(&deps[j].ty, 1 + j, j, j);
            mbody = Term::Pi(deps[j].mult, Box::new(ty), Box::new(mbody));
        }
        let motive = Term::Lam(Box::new(mbody));

        // zero method: peel the motive's Πs at b' := Zero, re-binding each dep
        // under its original name (shadowing the outer binding) at the refined type.
        let mut z_cx = cx.clone();
        let mut z_exp = dep::eval_rc(
            &self.rc,
            &neutral_env(n),
            &Term::App(Box::new(motive.clone()), Box::new(Term::NatLit(0))),
        );
        for d in &deps {
            match z_exp {
                Value::VPi(_, dom, clo) => {
                    z_cx.push(d.name.clone(), *dom);
                    z_exp = clo.apply(dep::nvar(z_cx.len() - 1));
                }
                _ => {
                    return Err("internal: nat-convoy zero method type did not reduce \
                                to a Π over the abstracted dependents"
                        .into())
                }
            }
        }
        let mut z = self.check(&zero_arm.body, &z_exp, &z_cx, rec)?;
        for _ in 0..k {
            z = Term::Lam(Box::new(z));
        }

        // succ method: λk. λd₁…λd_K. body, deps re-bound at b' := Succ k.
        let pred_name = succ_arm.binders[0].clone();
        let mut s_cx = cx.clone();
        s_cx.push(pred_name, Value::VNat);
        let mut env_s = neutral_env(n);
        env_s.push(dep::nvar(n));
        let mut s_exp = dep::eval_rc(
            &self.rc,
            &env_s,
            &Term::App(
                Box::new(dep::shift_term(1, &motive)),
                Box::new(Term::Suc(Box::new(Term::Var(0)))),
            ),
        );
        for d in &deps {
            match s_exp {
                Value::VPi(_, dom, clo) => {
                    s_cx.push(d.name.clone(), *dom);
                    s_exp = clo.apply(dep::nvar(s_cx.len() - 1));
                }
                _ => {
                    return Err("internal: nat-convoy succ method type did not reduce \
                                to a Π over the abstracted dependents"
                        .into())
                }
            }
        }
        let mut s_body = self.check(&succ_arm.body, &s_exp, &s_cx, rec)?;
        for _ in 0..k {
            s_body = Term::Lam(Box::new(s_body));
        }
        let s = Term::Lam(Box::new(s_body));

        let mut out = Term::NatCase(
            Box::new(motive),
            Box::new(z),
            Box::new(s),
            Box::new(scrut_tm.clone()),
        );
        for d in &deps {
            out = Term::App(Box::new(out), Box::new(Term::Var(n - 1 - d.lvl)));
        }
        Ok(Some(out))
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
        let arm = Arm { ctor: decl.ctors[0].name.clone(), binders: names.to_vec(), pats: vec![], body: body.clone() };
        self.elab_nested_match(&e_term, &dname, &dargs, std::slice::from_ref(&arm), expected, cx, rec, None)
    }

    /// A `match`/`let` in CHECK mode. The motive is normally the constant
    /// expected type (a non-dependent elimination) — but when the scrutinee's
    /// type is INDEXED by (`Succ` of) a context variable and other context
    /// variables' types depend on that index, the CONVOY (docs/CONVOY_HANDOFF.md;
    /// "the view from the left") abstracts those dependents into the motive so
    /// each arm re-binds them at their per-constructor REFINED types.
    #[allow(clippy::too_many_arguments)]
    fn elab_nested_match(&self, e_term: &Term, dname: &str, dargs: &[Value], arms: &[Arm], expected: &Value, cx: &Cx, rec: Option<&Rec>, fold: Option<&FnFold>) -> Result<Term, String> {
        let decl = self.rc.data(dname).unwrap().clone();
        let np = decl.params.len();
        let ni = decl.indices.len();
        let n = cx.len();
        let sparam_tms: Vec<Term> = dargs[..np].iter().map(|v| dep::quote_at(n, v)).collect();

        // COVERAGE HYGIENE: every arm must name a real constructor of the family
        // (the per-constructor loop below would silently ignore an unknown one).
        for a in arms {
            if !decl.ctors.iter().any(|c| c.name == a.ctor) {
                return Err(format!("`{}` is not a constructor of `{dname}`", a.ctor));
            }
        }
        let exp_tm = dep::quote_at(n, expected);
        let convoy = self.detect_convoy(e_term, &decl, dargs, &exp_tm, arms, cx);
        let motive = match &convoy {
            // constant motive: λ indices. λ _. expected
            None => {
                let mut m = dep::shift_term(ni + 1, &exp_tm);
                for _ in 0..(ni + 1) {
                    m = Term::Lam(Box::new(m));
                }
                m
            }
            Some(cv) => cv.build_motive(&exp_tm, n),
        };
        let motive_tm = motive.clone();

        // In a `Fix` body (`in_fix`) the match lowers to a NON-recursive `Case` (no IH,
        // recursion via the `Fix` self-call). With NO recursion in scope at all
        // (`rec` is None) a `Case` is also the right lowering — an `Elim` would
        // bind (and its backend EAGERLY compute) induction hypotheses nobody can
        // reference, which for a LINEAR recursive field would re-traverse a
        // resource the body consumes directly. Only a genuine structural fold
        // (`rec` Some) needs `Elim`.
        let as_case = self.in_fix.get() || (rec.is_none() && fold.is_none());
        let mut methods = Vec::with_capacity(decl.ctors.len());
        for (ci, ctor) in decl.ctors.iter().enumerate() {
            let info = &self.ctor_info[&ctor.name];
            let nargs = ctor.args.len();
            let rec_fields: Vec<usize> = ctor
                .args
                .iter()
                .enumerate()
                .filter(|(_, (_, a))| matches!(a, Term::Data(d, _) if d == dname))
                .map(|(i, _)| i)
                .collect();
            let nlam = if as_case { nargs } else { nargs + rec_fields.len() };
            // An arm the index REFUTES (a `Zero`-headed constructor index against the
            // scrutinee's `Succ`-headed index) is impossible: it must be OMITTED, and
            // its method is the `Nat` sentinel of the convoy motive's `Zero` branch —
            // typed dead code the kernel accepts and no run can reach (the scrutinee's
            // type rules the constructor out).
            if convoy.as_ref().is_some_and(|cv| cv.absurd_ctor[ci]) {
                if arms.iter().any(|a| a.ctor == ctor.name) {
                    return Err(format!(
                        "`match`: the case `{}` is impossible here — the scrutinee's \
                         index is `Succ`-headed and this constructor's is `Zero` — \
                         remove the arm",
                        ctor.name
                    ));
                }
                let mut m = Term::NatLit(0);
                for _ in 0..nlam {
                    m = Term::Lam(Box::new(m));
                }
                methods.push(m);
                continue;
            }
            let arm = arms
                .iter()
                .find(|a| a.ctor == ctor.name)
                .ok_or_else(|| format!("missing a case for `{}`", ctor.name))?;
            let nexplicit = info.arg_implicit.iter().filter(|b| !**b).count();
            // FULL-ARITY patterns may NAME the implicit binders too (Phase A4
            // provenance fix): `MkASplit(llo, lhi, lo, hi, rj)` binds the two
            // erased `Loc`s, so annotations in the arm can mention them (the
            // existential OPENS with a user name — the Idris/ATS idiom).
            let full_arity = nargs != nexplicit && arm.binders.len() == nargs;
            if arm.binders.len() != nexplicit && !full_arity {
                return Err(format!(
                    "pattern `{}`: expected {nexplicit} binder(s) (or all {nargs} to \
                     bind the implicits too), got {}",
                    ctor.name,
                    arm.binders.len()
                ));
            }
            let mut binder_names = Vec::new();
            let mut next_pat = 0;
            for j in 0..nargs {
                if full_arity {
                    binder_names.push(arm.binders[j].clone());
                } else if info.arg_implicit[j] {
                    binder_names.push(info.arg_names[j].clone().unwrap_or_else(|| format!("$imp{j}")));
                } else {
                    binder_names.push(arm.binders[next_pat].clone());
                    next_pat += 1;
                }
            }
            // In a `Fix` body the methods bind ONLY the constructor's args (no IHs);
            // for an `Elim` they also bind one IH per recursive field.
            if !as_case {
                for kk in 0..rec_fields.len() {
                    binder_names.push(format!("$ih{kk}"));
                }
            }
            let (binder_tys, result_tm) =
                dep::elim_method_telescope(&self.rc, dname, &sparam_tms, &motive_tm, &ctor.name, !as_case)?;
            let mut arm_cx = cx.clone();
            for (bn, (_, bty)) in binder_names.iter().zip(&binder_tys) {
                let v = dep::eval_rc(&self.rc, &neutral_env(arm_cx.len()), bty);
                arm_cx.push(bn.clone(), v);
            }
            // CONVOY: the method's result type is `motive ctor-index (ctor args)`,
            // which β/ι-reduces to `Π (dep : refined-ty) …. expected'` — peel those
            // Πs, RE-BINDING each dependent under its ORIGINAL name (it shadows the
            // outer binding) at its refined type, and check the arm at the reduced
            // codomain. The refinement is computed by the kernel's own method-type
            // machinery, and the final term is re-checked by the kernel — a wrong
            // convoy can only be rejected, never unsoundly accepted.
            let ndep = convoy.as_ref().map_or(0, |cv| cv.deps.len());
            let arm_expected = match &convoy {
                None => expected.clone(),
                Some(cv) => {
                    let mut res_v = dep::eval_rc(&self.rc, &neutral_env(arm_cx.len()), &result_tm);
                    for d in &cv.deps {
                        match res_v {
                            Value::VPi(_, dom, clo) => {
                                arm_cx.push(d.name.clone(), *dom);
                                res_v = clo.apply(dep::nvar(arm_cx.len() - 1));
                            }
                            _ => {
                                return Err(format!(
                                    "internal: convoy method type for `{}` did not reduce \
                                     to a Π over the abstracted dependents",
                                    ctor.name
                                ))
                            }
                        }
                    }
                    res_v
                }
            };
            // CONVOY FOLD: bind this arm's recursive fields to their IHs so a
            // self-call maps to `IH(new-deps)` (see `ih_for`'s fold branch).
            let mut arm_fields: HashMap<String, String> = HashMap::new();
            if fold.is_some() {
                for (kk, &fi) in rec_fields.iter().enumerate() {
                    arm_fields.insert(binder_names[fi].clone(), format!("$ih{kk}"));
                }
            }
            let arm_rec = fold.map(|fo| Rec {
                fnname: &fo.fnname,
                scrut_pos: fo.scrut_pos,
                fields: &arm_fields,
                acc_tys: None,
                result_ty: arm_expected.clone(),
                fold: Some(fo),
            });
            // USE-SITE LINEARITY (see `rebind_linear_fields`): re-bind each linear
            // field at 1 so a hidden `Own` (incl. a generic instantiated at `Own`)
            // can't be used twice in a nested/expression `match` either.
            let arm_body = rebind_linear_fields(&arm.body, &binder_names, &info.arg_implicit, &binder_tys, nargs, &self.rc, &self.pess_linear.borrow(), &arm_cx.names, n);
            let mut body = self.check(&arm_body, &arm_expected, &arm_cx, arm_rec.as_ref().or(rec))?;
            for _ in 0..(nlam + ndep) {
                body = Term::Lam(Box::new(body));
            }
            methods.push(body);
        }
        let mut out = if as_case {
            Term::Case(dname.to_string(), Box::new(motive), methods, Box::new(e_term.clone()))
        } else {
            Term::Elim(dname.to_string(), Box::new(motive), methods, Box::new(e_term.clone()))
        };
        // apply the eliminator's Π-over-dependents result to the ACTUAL dependents
        // (whose per-arm refinements the methods just consumed).
        if let Some(cv) = &convoy {
            for d in &cv.deps {
                out = Term::App(Box::new(out), Box::new(Term::Var(n - 1 - d.lvl)));
            }
        }
        Ok(out)
    }

    /// Detect a CONVOY opportunity for a `match` on `e_term : dname dargs`
    /// (docs/CONVOY_HANDOFF.md). v1 scope, all misses CONSERVATIVE (constant
    /// motive, as before):
    ///
    /// - the family has exactly ONE index, of builtin `Nat` type;
    /// - the scrutinee's index value is a context VARIABLE (`r = 0`) or `Succ`
    ///   of one (`r = 1`);
    /// - the dependents are the context variables whose types mention that
    ///   variable (transitively through other dependents); none may mention the
    ///   scrutinee itself;
    /// - for `r = 1`, every constructor's result index must be `Zero`- or
    ///   `Succ`-headed (so the motive's `NatCase` reduces in every method type);
    ///   `Zero`-headed constructors are REFUTED arms (must be omitted).
    ///
    /// Fires when there are dependents, or (`r = 1`) when it refutes an arm or
    /// the EXPECTED type mentions the index variable (the `Succ`-inversion in
    /// the motive is then what types `vhead`/`vtail`-style index projections).
    ///
    /// A candidate dependent is abstracted only if some arm body REFERENCES its
    /// name: abstracting an unused one is never needed for typing, and for a
    /// LINEAR dependent already consumed before the match (e.g. the `Own` the
    /// scrutinee was just `unbox`ed from) the re-application would double-use it.
    fn detect_convoy(&self, e_term: &Term, decl: &dep::DataDecl, dargs: &[Value], exp_tm: &Term, arms: &[Arm], cx: &Cx) -> Option<Convoy> {
        let np = decl.params.len();
        if decl.indices.len() != 1 || !matches!(decl.indices[0].1, Term::Nat) {
            return None;
        }
        let n = cx.len();
        // peel `Succ` layers off the index value down to a context variable.
        let mut r = 0usize;
        let mut v = &dargs[np];
        while let Value::VSuc(inner) = v {
            r += 1;
            v = inner;
        }
        // v1: `Succ`-depth ≤ 1 (deeper would need nested `NatCase`s whose inner
        // case is STUCK on the constructor's own neutral index).
        if r > 1 {
            return None;
        }
        let lvl = match v {
            Value::VNeu(dep::Neutral::NVar(l)) if *l < n => *l,
            _ => return None,
        };
        let mut scrut = e_term;
        while let Term::Ann(inner, _) = scrut {
            scrut = inner;
        }
        let scrut_lvl = match scrut {
            Term::Var(i) if *i < n => Some(n - 1 - *i),
            _ => None,
        };
        let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
        for a in arms {
            collect_tm_names(&a.body, &mut referenced);
        }
        let mut dep_lvls: Vec<usize> = vec![lvl];
        let mut deps: Vec<ConvoyDep> = Vec::new();
        for l in 0..n {
            if l == lvl || Some(l) == scrut_lvl {
                continue;
            }
            // the referenced-names filter skips deps no arm uses (abstracting an
            // already-consumed linear value would double-use it) — but an
            // INDUCTION HYPOTHESIS binder (`$ih…`, a nested match inside a
            // convoy fold) is referenced only through the recursive-call
            // mapping, which the surface names can't show: always a candidate
            // (it is a function — never linear, harmless if unused).
            if !referenced.contains(&cx.names[l]) && !cx.names[l].starts_with("$ih") {
                continue;
            }
            let ty = dep::quote_at(n, &cx.types[l]);
            if !term_mentions_levels(&ty, n, &dep_lvls) {
                continue;
            }
            if let Some(sl) = scrut_lvl {
                if term_mentions_levels(&ty, n, &[sl]) {
                    return None; // a dependent typed BY the scrutinee: out of v1 scope
                }
            }
            let mult = if type_is_linear_in(&ty, &self.rc, &self.pess_linear.borrow(), &cx.names, n) { Mult::One } else { Mult::Omega };
            dep_lvls.push(l);
            deps.push(ConvoyDep { lvl: l, name: cx.names[l].clone(), ty, mult });
        }
        let mut absurd_ctor = vec![false; decl.ctors.len()];
        if r == 1 {
            for (ci, ctor) in decl.ctors.iter().enumerate() {
                // the constructor's result index, evaluated with params at their
                // actual values and the ctor's own args neutral. `Zero` vs the
                // scrutinee's `Succ`-headed index is DECIDABLY disjoint, so the
                // refuted-arm classification cannot be wrong (there is no
                // reachable arm to misclassify; cf. the `try_absurd_match`
                // Nat-sentinel caveat, which this satisfies).
                let mut env: Vec<Value> = dargs[..np].to_vec();
                for j in 0..ctor.args.len() {
                    env.push(dep::nvar(n + j));
                }
                let idx_v = dep::eval_rc(&self.rc, &env, &ctor.idxs[0]);
                match idx_v {
                    Value::VNatLit(0) => absurd_ctor[ci] = true,
                    Value::VNatLit(_) | Value::VSuc(_) => {}
                    _ => return None, // non-ctor-headed index: the NatCase would stick
                }
            }
        }
        let worthwhile = !deps.is_empty()
            || (r == 1
                && (absurd_ctor.iter().any(|b| *b) || term_mentions_levels(exp_tm, n, &[lvl])));
        if !worthwhile {
            return None;
        }
        Some(Convoy { r, lvl, deps, absurd_ctor })
    }
}

/// The CONVOY: a dependent-match motive that Π-abstracts the index-dependent
/// context variables, so each arm sees them REFINED by that constructor's
/// index. For `r = 0` the index variable itself becomes the motive's index
/// binder; for `r = 1` the variable sits UNDER a `Succ`, so the motive computes
/// it back from the index binder by LARGE ELIMINATION (`NatCase` — the
/// predecessor; the same J-free technique as the absurd discharge, and what
/// connects a tail's index to a constructor's own implicit without an identity
/// eliminator).
struct Convoy {
    /// `Succ`-depth of the scrutinee's index over the variable (0 or 1).
    r: usize,
    /// context LEVEL of the index variable.
    lvl: usize,
    /// the abstracted dependents, ascending by context level.
    deps: Vec<ConvoyDep>,
    /// per-constructor: is the arm refuted by the index? (`r = 1` only)
    absurd_ctor: Vec<bool>,
}

struct ConvoyDep {
    lvl: usize,
    name: String,
    /// the dependent's type, quoted at the match's context length.
    ty: Term,
    /// `1` for a linear dependent (the kernel then enforces exactly-once use of
    /// the re-bound dependent in every arm), `ω` otherwise — same rule as `let`.
    mult: Mult,
}

impl Convoy {
    /// `r = 0`:  λ idx. λ scrut. Π deps[x := idx]. expected[x := idx]
    /// `r = 1`:  λ idx. λ scrut. NatCase[λ_.Type] idx Nat (λ p. Π deps[x := p]. expected[x := p])
    /// (the `Nat` in the `Zero` branch is the refuted-arm sentinel: no reachable
    /// use site computes it, and a misclassification yields a kernel-rejected term).
    fn build_motive(&self, exp_tm: &Term, n: usize) -> Term {
        let k = self.deps.len();
        // binders between the original context and the Π-chain: [idx, scrut] for
        // r = 0; [idx, scrut, p] for r = 1 (p = the NatCase predecessor binder).
        let base = 2 + self.r;
        // the index variable's replacement, at Π-chain depth j: the idx binder
        // (r = 0) or the predecessor binder p (r = 1).
        let idx_repl = |j: usize| if self.r == 0 { j + 1 } else { j };
        let mut body = self.remap(exp_tm, n, base + k, idx_repl(k), k);
        for j in (0..k).rev() {
            let ty = self.remap(&self.deps[j].ty, n, base + j, idx_repl(j), j);
            body = Term::Pi(self.deps[j].mult, Box::new(ty), Box::new(body));
        }
        if self.r == 0 {
            Term::Lam(Box::new(Term::Lam(Box::new(body))))
        } else {
            Term::Lam(Box::new(Term::Lam(Box::new(Term::NatCase(
                Box::new(Term::Lam(Box::new(Term::Type(0)))),
                Box::new(Term::Nat),
                Box::new(Term::Lam(Box::new(body))),
                Box::new(Term::Var(1)), // the index binder
            )))))
        }
    }

    /// Re-target a term quoted at level `n` to sit under `d` inserted binders:
    /// the index variable ↦ `Var(idx_repl)`, the first `j` dependents ↦ their
    /// own Π binders, every other free variable shifted by `d`. (Replacement
    /// indices are relative to the insertion point; `map_vars` supplies the
    /// local binder depth.)
    fn remap(&self, t: &Term, n: usize, d: usize, idx_repl: usize, j: usize) -> Term {
        dep::map_vars(t, 0, &|i, depth| {
            if i < depth {
                return Term::Var(i);
            }
            let outer = i - depth;
            if outer < n {
                let l = n - 1 - outer;
                if l == self.lvl {
                    return Term::Var(idx_repl + depth);
                }
                if let Some(m) = self.deps[..j].iter().position(|dp| dp.lvl == l) {
                    return Term::Var(j - 1 - m + depth);
                }
            }
            Term::Var(i + d)
        })
    }
}

// ===========================================================================
// MULT-POLY MONOMORPHIZATION (docs/MULT_POLY_PLAN.md slice 2)
// ===========================================================================

/// A function declared with explicit `(m : Mult)` parameter(s), stored
/// UN-elaborated (its usage-validity depends on `m`).
struct PolyDef {
    sig: Ty,
    params: Vec<String>,
    body: Tm,
    annot: Option<TotAnnot>,
    /// (explicit-argument index, parameter name) of each `Mult` parameter.
    mult_params: Vec<(usize, String)>,
}

fn is_mult_sort(t: &Ty) -> bool {
    matches!(t, Ty::Var(n) if n == "Mult")
}

/// The explicit-position `(m : Mult)` parameters of a signature.
fn mult_params_of(sig: &Ty) -> Result<Vec<(usize, String)>, String> {
    let (arrows, _) = peel_arrows(sig);
    let mut out = Vec::new();
    let mut expl = 0usize;
    for (_, imp, name, dom) in &arrows {
        if *imp {
            if is_mult_sort(dom) {
                return Err("an implicit `{m : Mult}` parameter is not supported yet — \
                            make it explicit: `(m : Mult)` (explicit-first)"
                    .into());
            }
            continue;
        }
        if is_mult_sort(dom) {
            let n = name.clone().ok_or("a `Mult` parameter must be named: `(m : Mult)`")?;
            out.push((expl, n));
        }
        expl += 1;
    }
    Ok(out)
}

/// `0`/`1`/`w` — or an enclosing instance's own `(x : Mult)` parameter.
fn resolve_mult_arg(t: &Tm, subst: &HashMap<String, Mult>) -> Result<Mult, String> {
    match t {
        Tm::Lit(0) => Ok(Mult::Zero),
        Tm::Lit(1) => Ok(Mult::One),
        Tm::Var(v) if v == "w" => Ok(Mult::Omega),
        Tm::Var(v) => subst.get(v).copied().ok_or_else(|| {
            format!(
                "`{v}` is not a multiplicity — a `Mult` argument must be `0`, `1`, `w`, \
                 or an enclosing `(x : Mult)` parameter"
            )
        }),
        _ => Err("a `Mult` argument must be `0`, `1`, or `w`".into()),
    }
}

fn mult_char(m: Mult) -> char {
    match m {
        Mult::Zero => '0',
        Mult::One => '1',
        Mult::Omega => 'w',
    }
}

fn mono_name(base: &str, mults: &[Mult]) -> String {
    let suffix: String = mults.iter().map(|m| mult_char(*m)).collect();
    format!("{base}${suffix}")
}

/// If `name` is a mult-poly INSTANCE constructor `base$<mults>` (a non-empty
/// suffix all in `{0,1,w}`), return its base name — the inverse of `mono_name`.
fn mono_ctor_base(name: &str) -> Option<&str> {
    let (base, suffix) = name.rsplit_once('$')?;
    (!base.is_empty()
        && !suffix.is_empty()
        && suffix.chars().all(|c| matches!(c, '0' | '1' | 'w')))
    .then_some(base)
}

/// Substitute concrete multiplicities for `SMult::Var`s throughout a type.
fn subst_mult_ty(t: &Ty, subst: &HashMap<String, Mult>) -> Ty {
    match t {
        Ty::Arrow(m, imp, n, a, b) => {
            let m2 = match m {
                SMult::Var(v) => match subst.get(v) {
                    Some(mm) => SMult::Lit(*mm),
                    None => SMult::Var(v.clone()),
                },
                SMult::Lit(l) => SMult::Lit(*l),
            };
            Ty::Arrow(
                m2,
                *imp,
                n.clone(),
                Box::new(subst_mult_ty(a, subst)),
                Box::new(subst_mult_ty(b, subst)),
            )
        }
        Ty::App(f, a) => {
            Ty::App(Box::new(subst_mult_ty(f, subst)), Box::new(subst_mult_ty(a, subst)))
        }
        Ty::Add(a, b) => {
            Ty::Add(Box::new(subst_mult_ty(a, subst)), Box::new(subst_mult_ty(b, subst)))
        }
        Ty::Var(_) | Ty::Type(_) | Ty::Lit(_) | Ty::TypeV(_) => t.clone(),
    }
}

/// An instance's signature: the original with the explicit `(m : Mult)` arrows
/// REMOVED and every mult variable substituted concretely.
fn strip_mult_arrows(t: &Ty, subst: &HashMap<String, Mult>) -> Ty {
    match t {
        Ty::Arrow(_, false, _, dom, cod) if is_mult_sort(dom) => strip_mult_arrows(cod, subst),
        Ty::Arrow(m, imp, n, a, b) => {
            let m2 = match m {
                SMult::Var(v) => match subst.get(v) {
                    Some(mm) => SMult::Lit(*mm),
                    None => SMult::Var(v.clone()),
                },
                SMult::Lit(l) => SMult::Lit(*l),
            };
            Ty::Arrow(
                m2,
                *imp,
                n.clone(),
                Box::new(subst_mult_ty(a, subst)),
                Box::new(strip_mult_arrows(b, subst)),
            )
        }
        other => subst_mult_ty(other, subst),
    }
}

/// Rewrite calls to mult-poly functions into calls to their concrete instances,
/// queuing each newly-needed instance. `subst` maps the enclosing instance's own
/// mult parameters (empty outside one).
fn mono_rewrite_tm(
    t: &Tm,
    polys: &HashMap<String, PolyDef>,
    subst: &HashMap<String, Mult>,
    queue: &mut Vec<(String, Vec<Mult>)>,
) -> Result<Tm, String> {
    match t {
        Tm::Var(name) if polys.contains_key(name) => Err(format!(
            "`{name}` is multiplicity-polymorphic — it must be applied to its `Mult` \
             argument(s) (a bare reference cannot be monomorphized)"
        )),
        Tm::Var(_) | Tm::Lit(_) | Tm::Str(_) => Ok(t.clone()),
        Tm::Ann(e, ty) => Ok(Tm::Ann(
            Box::new(mono_rewrite_tm(e, polys, subst, queue)?),
            ty.clone(),
        )),
        Tm::Call(name, args) => {
            if let Some(pd) = polys.get(name) {
                let mut mults = Vec::new();
                let mut rest = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    if pd.mult_params.iter().any(|(mi, _)| *mi == i) {
                        mults.push(resolve_mult_arg(a, subst)?);
                    } else {
                        rest.push(mono_rewrite_tm(a, polys, subst, queue)?);
                    }
                }
                if mults.len() != pd.mult_params.len() {
                    return Err(format!(
                        "`{name}` expects {} explicit argument(s) including its `Mult` \
                         parameter(s), got {}",
                        pd.mult_params.last().map(|(i, _)| i + 1).unwrap_or(0).max(args.len()),
                        args.len()
                    ));
                }
                if !queue.iter().any(|(n, ms)| n == name && *ms == mults) {
                    queue.push((name.clone(), mults.clone()));
                }
                Ok(Tm::Call(mono_name(name, &mults), rest))
            } else {
                let eargs = args
                    .iter()
                    .map(|a| mono_rewrite_tm(a, polys, subst, queue))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Tm::Call(name.clone(), eargs))
            }
        }
        Tm::Match(s, arms) => {
            let arms2 = arms
                .iter()
                .map(|a| {
                    Ok(Arm {
                        ctor: a.ctor.clone(),
                        binders: a.binders.clone(),
                        pats: a.pats.clone(),
                        body: mono_rewrite_tm(&a.body, polys, subst, queue)?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Tm::Match(s.clone(), arms2))
        }
        Tm::MatchN(cols, rows) => {
            let rows2 = rows
                .iter()
                .map(|(ps, b)| Ok((ps.clone(), mono_rewrite_tm(b, polys, subst, queue)?)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Tm::MatchN(cols.clone(), rows2))
        }
        Tm::Add(a, b) => Ok(Tm::Add(
            Box::new(mono_rewrite_tm(a, polys, subst, queue)?),
            Box::new(mono_rewrite_tm(b, polys, subst, queue)?),
        )),
        Tm::Let(n, e, b) => Ok(Tm::Let(
            n.clone(),
            Box::new(mono_rewrite_tm(e, polys, subst, queue)?),
            Box::new(mono_rewrite_tm(b, polys, subst, queue)?),
        )),
        Tm::LetPair(ns, e, b) => Ok(Tm::LetPair(
            ns.clone(),
            Box::new(mono_rewrite_tm(e, polys, subst, queue)?),
            Box::new(mono_rewrite_tm(b, polys, subst, queue)?),
        )),
        Tm::Lam(ps, b) => {
            Ok(Tm::Lam(ps.clone(), Box::new(mono_rewrite_tm(b, polys, subst, queue)?)))
        }
    }
}

// --- MULT-POLY DATATYPES (slice: closures over linearity) ------------------
// A datatype declared with a `(m : Mult)` parameter — e.g.
//   struct FlatClo (m : Mult) (e : Type) (a : Type) (b : Type) { code : (m x : a) -> b }
// is monomorphised the same way a function is: each concrete reference
// `FlatClo 1 e a b` becomes `FlatClo$1 e a b` (a fresh datatype with `m := 1`
// substituted through the constructor field arrows), so the kernel's rig stays
// concrete. This is the datatype counterpart of `monomorphize_mult_poly`; it
// runs in the same pass because a *function's* own `(m : Mult)` parameter flows
// into a datatype reference (`applyFlat : (m : Mult) -> FlatClo m e a b -> …`),
// so the datatype instance is only pinned once the function's `m` is resolved.

/// The parameter positions of a datatype declared as `(m : Mult)`.
fn data_mult_positions(params: &[Binder]) -> Vec<usize> {
    params
        .iter()
        .enumerate()
        .filter_map(|(i, b)| is_mult_sort(&b.ty).then_some(i))
        .collect()
}

/// Flatten a left-nested type application `App(App(App(D, x), y), z)` into its
/// head `D` and argument list `[x, y, z]`.
fn flatten_ty_app(t: &Ty) -> (&Ty, Vec<&Ty>) {
    let mut head = t;
    let mut args: Vec<&Ty> = Vec::new();
    while let Ty::App(f, a) = head {
        args.push(a);
        head = f;
    }
    args.reverse();
    (head, args)
}

/// Decode a `Mult` written as a TYPE-application argument: `1`/`0` parse as a
/// `Ty::Lit` (the index-literal), `w` and an enclosing `(x : Mult)` parameter as
/// a `Ty::Var` (resolved through `subst`).
fn resolve_mult_ty(t: &Ty, subst: &HashMap<String, Mult>) -> Result<Mult, String> {
    match t {
        Ty::Lit(0) => Ok(Mult::Zero),
        Ty::Lit(1) => Ok(Mult::One),
        Ty::Var(v) if v == "w" => Ok(Mult::Omega),
        Ty::Var(v) => subst.get(v).copied().ok_or_else(|| {
            format!(
                "`{v}` is not a multiplicity — a datatype's `Mult` argument must be \
                 `0`, `1`, `w`, or an enclosing `(x : Mult)` parameter"
            )
        }),
        _ => Err("a datatype's `Mult` argument must be `0`, `1`, or `w`".into()),
    }
}

/// Rewrite every reference to a mult-poly datatype in a type: `D <mults…> args…`
/// ⟶ `D$<mults> args…`, dropping the `Mult` arguments and queuing the needed
/// instance. `poly` maps a datatype name to its `Mult` parameter positions.
fn mangle_ty(
    t: &Ty,
    poly: &HashMap<String, Vec<usize>>,
    subst: &HashMap<String, Mult>,
    queue: &mut Vec<(String, Vec<Mult>)>,
) -> Result<Ty, String> {
    match t {
        Ty::App(_, _) => {
            let (head, args) = flatten_ty_app(t);
            if let Ty::Var(name) = head {
                if let Some(positions) = poly.get(name).cloned() {
                    let mut mults: Vec<Mult> = Vec::new();
                    let mut rest: Vec<Ty> = Vec::new();
                    for (i, a) in args.iter().enumerate() {
                        if positions.contains(&i) {
                            mults.push(resolve_mult_ty(a, subst)?);
                        } else {
                            rest.push(mangle_ty(a, poly, subst, queue)?);
                        }
                    }
                    if mults.len() != positions.len() {
                        return Err(format!(
                            "`{name}` is multiplicity-polymorphic and must be applied to \
                             its `Mult` argument(s) at every reference"
                        ));
                    }
                    if !queue.iter().any(|(n, ms)| n == name && *ms == mults) {
                        queue.push((name.clone(), mults.clone()));
                    }
                    let mut out = Ty::Var(mono_name(name, &mults));
                    for r in rest {
                        out = Ty::App(Box::new(out), Box::new(r));
                    }
                    return Ok(out);
                }
            }
            // not a mult-poly head: recurse into both halves.
            if let Ty::App(f, a) = t {
                return Ok(Ty::App(
                    Box::new(mangle_ty(f, poly, subst, queue)?),
                    Box::new(mangle_ty(a, poly, subst, queue)?),
                ));
            }
            unreachable!()
        }
        Ty::Arrow(m, imp, n, a, b) => Ok(Ty::Arrow(
            m.clone(),
            *imp,
            n.clone(),
            Box::new(mangle_ty(a, poly, subst, queue)?),
            Box::new(mangle_ty(b, poly, subst, queue)?),
        )),
        Ty::Add(a, b) => Ok(Ty::Add(
            Box::new(mangle_ty(a, poly, subst, queue)?),
            Box::new(mangle_ty(b, poly, subst, queue)?),
        )),
        Ty::Var(_) | Ty::Lit(_) | Ty::Type(_) | Ty::TypeV(_) => Ok(t.clone()),
    }
}

/// Build the concrete instance of a mult-poly datatype at `mults`: substitute the
/// `Mult` parameters through the field/variant arrows, mangle datatype references
/// in them, drop the `Mult` parameters, and mangle each constructor name.
fn generate_data_instance(
    item: &Item,
    positions: &[usize],
    mults: &[Mult],
    poly: &HashMap<String, Vec<usize>>,
    queue: &mut Vec<(String, Vec<Mult>)>,
) -> Result<Item, String> {
    // subst: the datatype's own `Mult` parameter names ⟶ the concrete mults.
    let subst_of = |params: &[Binder]| -> HashMap<String, Mult> {
        positions
            .iter()
            .zip(mults)
            .map(|(&p, m)| (params[p].name.clone(), *m))
            .collect()
    };
    // rewrite a field/variant type: substitute arrow mults, then mangle refs.
    let rewrite = |ty: &Ty, subst: &HashMap<String, Mult>, q: &mut Vec<(String, Vec<Mult>)>| {
        mangle_ty(&subst_mult_ty(ty, subst), poly, subst, q)
    };
    match item {
        Item::Struct { name, params, fields } => {
            let subst = subst_of(params);
            let nparams: Vec<Binder> = params
                .iter()
                .enumerate()
                .filter(|(i, _)| !positions.contains(i))
                .map(|(_, b)| b.clone())
                .collect();
            let nfields: Vec<(String, Ty)> = fields
                .iter()
                .map(|(fna, fty)| Ok((fna.clone(), rewrite(fty, &subst, queue)?)))
                .collect::<Result<_, String>>()?;
            Ok(Item::Struct { name: mono_name(name, mults), params: nparams, fields: nfields })
        }
        Item::Enum { name, params, index_ty, variants, linear, boxed } => {
            let subst = subst_of(params);
            let nparams: Vec<Binder> = params
                .iter()
                .enumerate()
                .filter(|(i, _)| !positions.contains(i))
                .map(|(_, b)| b.clone())
                .collect();
            let nvariants: Vec<(String, Ty)> = variants
                .iter()
                .map(|(cn, cty)| Ok((mono_name(cn, mults), rewrite(cty, &subst, queue)?)))
                .collect::<Result<_, String>>()?;
            Ok(Item::Enum {
                name: mono_name(name, mults),
                params: nparams,
                index_ty: index_ty.clone(),
                variants: nvariants,
                linear: *linear,
                boxed: *boxed,
            })
        }
        _ => Err("internal: a mult-poly datatype instance of a non-datatype".into()),
    }
}

/// The slice-2 pre-pass: pull out every `fn` with a `(m : Mult)` parameter,
/// rewrite all remaining bodies (queuing needed instances), instantiate each
/// requested (function × multiplicities) pair ONCE — a poly body's own calls
/// are rewritten under its substitution, so poly-calls-poly chains resolve —
/// and splice the instances in at the original definition's position (they
/// precede every caller, since a definition precedes its uses).
fn monomorphize_mult_poly(items: &mut Vec<Item>) -> Result<(), String> {
    let mut sigs: HashMap<String, Ty> = HashMap::new();
    for it in items.iter() {
        if let Item::Sig(n, t) = it {
            sigs.insert(n.clone(), t.clone());
        }
    }
    let mut polys: HashMap<String, PolyDef> = HashMap::new();
    for it in items.iter() {
        if let Item::Fn(name, params, body, annot) = it {
            let Some(sig) = sigs.get(name) else { continue };
            let mps = mult_params_of(sig)?;
            if !mps.is_empty() {
                polys.insert(
                    name.clone(),
                    PolyDef {
                        sig: sig.clone(),
                        params: params.clone(),
                        body: body.clone(),
                        annot: *annot,
                        mult_params: mps,
                    },
                );
            }
        }
    }
    // mult-poly DATATYPES: name -> (original Item, `Mult`-parameter positions).
    let mut data_polys: HashMap<String, (Item, Vec<usize>)> = HashMap::new();
    for it in items.iter() {
        let (name, params) = match it {
            Item::Struct { name, params, .. } => (name, params),
            Item::Enum { name, params, .. } => (name, params),
            _ => continue,
        };
        let pos = data_mult_positions(params);
        if !pos.is_empty() {
            data_polys.insert(name.clone(), (it.clone(), pos));
        }
    }
    if polys.is_empty() && data_polys.is_empty() {
        return Ok(());
    }
    let poly_map: HashMap<String, Vec<usize>> =
        data_polys.iter().map(|(n, (_, p))| (n.clone(), p.clone())).collect();
    let mut data_queue: Vec<(String, Vec<Mult>)> = Vec::new();
    let empty: HashMap<String, Mult> = HashMap::new();

    // Mangle mult-poly datatype references in every RETAINED item's types
    // (literal-mult references). Skip poly `fn` sigs — their datatype refs use
    // the fn's own `Mult` parameter, resolved per-instance below — and the poly
    // datatypes themselves (replaced by their instances).
    for it in items.iter_mut() {
        match it {
            Item::Sig(n, t) if !polys.contains_key(n) => {
                *t = mangle_ty(t, &poly_map, &empty, &mut data_queue)?;
            }
            Item::Postulate(_, t, _) => *t = mangle_ty(t, &poly_map, &empty, &mut data_queue)?,
            Item::Foreign(_, _, t) => *t = mangle_ty(t, &poly_map, &empty, &mut data_queue)?,
            Item::Struct { name, fields, .. } if !data_polys.contains_key(name) => {
                for (_, ft) in fields.iter_mut() {
                    *ft = mangle_ty(ft, &poly_map, &empty, &mut data_queue)?;
                }
            }
            Item::Enum { name, variants, .. } if !data_polys.contains_key(name) => {
                for (_, vt) in variants.iter_mut() {
                    *vt = mangle_ty(vt, &poly_map, &empty, &mut data_queue)?;
                }
            }
            _ => {}
        }
    }

    // rewrite every non-poly body, queuing the needed FUNCTION instances
    let mut queue: Vec<(String, Vec<Mult>)> = Vec::new();
    for it in items.iter_mut() {
        if let Item::Fn(name, _, body, _) = it {
            if !polys.contains_key(name) {
                *body = mono_rewrite_tm(body, &polys, &empty, &mut queue)?;
            }
        }
    }
    // instantiate function polys (drain to fixpoint), MANGLING each instance's
    // sig so a datatype reference at the fn's own `Mult` parameter is resolved.
    let mut made: Vec<(String, Item, Item)> = Vec::new(); // (original, Sig, Fn)
    let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut qi = 0;
    while qi < queue.len() {
        let (orig, mults) = queue[qi].clone();
        qi += 1;
        let iname = mono_name(&orig, &mults);
        if !done.insert(iname.clone()) {
            continue;
        }
        let pd = &polys[&orig];
        let subst: HashMap<String, Mult> = pd
            .mult_params
            .iter()
            .zip(&mults)
            .map(|((_, n), m)| (n.clone(), *m))
            .collect();
        let isig =
            mangle_ty(&strip_mult_arrows(&pd.sig, &subst), &poly_map, &subst, &mut data_queue)?;
        let iparams: Vec<String> = pd
            .params
            .iter()
            .filter(|p| !pd.mult_params.iter().any(|(_, n)| n == *p))
            .cloned()
            .collect();
        let ibody = mono_rewrite_tm(&pd.body, &polys, &subst, &mut queue)?;
        made.push((
            orig.clone(),
            Item::Sig(iname.clone(), isig),
            Item::Fn(iname, iparams, ibody, pd.annot),
        ));
    }

    // instantiate DATATYPE polys (fixpoint — an instance's fields may reference
    // further mult-poly datatypes, which `generate_data_instance` queues).
    let mut data_made: Vec<(String, Item)> = Vec::new();
    let mut data_done: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut di = 0;
    while di < data_queue.len() {
        let (orig, mults) = data_queue[di].clone();
        di += 1;
        if !data_done.insert(mono_name(&orig, &mults)) {
            continue;
        }
        let (item, positions) = data_polys
            .get(&orig)
            .ok_or_else(|| format!("internal: mult-poly datatype `{orig}` not found"))?
            .clone();
        let inst = generate_data_instance(&item, &positions, &mults, &poly_map, &mut data_queue)?;
        data_made.push((orig.clone(), inst));
    }

    // splice: drop each poly fn Sig/Fn AND each poly datatype, inserting its
    // instances at the original definition's position (which precedes every use).
    let mut out: Vec<Item> = Vec::with_capacity(items.len() + made.len() + data_made.len());
    for it in items.drain(..) {
        match &it {
            Item::Sig(n, _) if polys.contains_key(n) => {}
            Item::Fn(n, _, _, _) if polys.contains_key(n) => {
                for (orig, s, f) in &made {
                    if orig == n {
                        out.push(s.clone());
                        out.push(f.clone());
                    }
                }
            }
            Item::Struct { name, .. } | Item::Enum { name, .. } if data_polys.contains_key(name) => {
                for (orig, inst) in &data_made {
                    if orig == name {
                        out.push(inst.clone());
                    }
                }
            }
            _ => out.push(it),
        }
    }
    *items = out;
    Ok(())
}

// ===========================================================================
// PHASE B2 — UNIVERSE (LEVEL) POLYMORPHISM by surface monomorphization.
// `def`s may take explicit `(l : Level)` parameters and mention `Type l`;
// each call site passes a LITERAL level and gets a per-level instance
// (`id$L1`), exactly like the mult-poly pre-pass. The kernel's rig of
// universes stays concrete and strict — the hierarchy cannot collapse, and
// the Girard/Hurkens guard is untouched (each instance is checked at its
// concrete levels).
// ===========================================================================

fn is_level_sort(t: &Ty) -> bool {
    matches!(t, Ty::Var(n) if n == "Level")
}

fn level_params_of(sig: &Ty) -> Result<Vec<(usize, String)>, String> {
    let (arrows, _) = peel_arrows(sig);
    let mut out = Vec::new();
    let mut expl = 0usize;
    for (_, imp, name, dom) in &arrows {
        if *imp {
            if is_level_sort(dom) {
                return Err("an implicit `{l : Level}` parameter is not supported yet — \
                            make it explicit: `(l : Level)`"
                    .into());
            }
            continue;
        }
        if is_level_sort(dom) {
            let n =
                name.clone().ok_or("a `Level` parameter must be named: `(l : Level)`")?;
            out.push((expl, n));
        }
        expl += 1;
    }
    Ok(out)
}

fn subst_level_ty(t: &Ty, subst: &HashMap<String, usize>) -> Ty {
    match t {
        Ty::TypeV(v) => match subst.get(v) {
            Some(l) => Ty::Type(*l),
            None => Ty::TypeV(v.clone()),
        },
        Ty::Arrow(m, imp, n, a, b) => Ty::Arrow(
            m.clone(),
            *imp,
            n.clone(),
            Box::new(subst_level_ty(a, subst)),
            Box::new(subst_level_ty(b, subst)),
        ),
        Ty::App(f, a) => {
            Ty::App(Box::new(subst_level_ty(f, subst)), Box::new(subst_level_ty(a, subst)))
        }
        Ty::Add(a, b) => {
            Ty::Add(Box::new(subst_level_ty(a, subst)), Box::new(subst_level_ty(b, subst)))
        }
        Ty::Var(_) | Ty::Type(_) | Ty::Lit(_) => t.clone(),
    }
}

/// An instance's signature: the original with the explicit `(l : Level)`
/// arrows REMOVED and every level variable substituted concretely.
fn strip_level_arrows(t: &Ty, subst: &HashMap<String, usize>) -> Ty {
    match t {
        Ty::Arrow(_, false, _, dom, cod) if is_level_sort(dom) => strip_level_arrows(cod, subst),
        Ty::Arrow(m, imp, n, a, b) => Ty::Arrow(
            m.clone(),
            *imp,
            n.clone(),
            Box::new(subst_level_ty(a, subst)),
            Box::new(strip_level_arrows(b, subst)),
        ),
        other => subst_level_ty(other, subst),
    }
}

struct LevelPolyDef {
    sig: Ty,
    params: Vec<String>,
    body: Tm,
    annot: Option<TotAnnot>,
    level_params: Vec<(usize, String)>,
}

fn resolve_level_arg(t: &Tm, subst: &HashMap<String, usize>) -> Result<usize, String> {
    match t {
        Tm::Lit(n) => Ok(*n as usize),
        Tm::Var(v) => subst.get(v).copied().ok_or_else(|| {
            format!(
                "`{v}` is not a universe level — a `Level` argument must be a literal \
                 (`0`, `1`, …) or an enclosing `(l : Level)` parameter"
            )
        }),
        _ => Err("a `Level` argument must be a literal level".into()),
    }
}

fn level_mono_name(base: &str, levels: &[usize]) -> String {
    let suffix: Vec<String> = levels.iter().map(|l| l.to_string()).collect();
    format!("{base}$L{}", suffix.join("_"))
}

/// rewrite level-poly calls in a term, substituting the enclosing instance's
/// own level parameters in `Ann` types along the way.
fn level_rewrite_tm(
    t: &Tm,
    polys: &HashMap<String, LevelPolyDef>,
    subst: &HashMap<String, usize>,
    queue: &mut Vec<(String, Vec<usize>)>,
) -> Result<Tm, String> {
    Ok(match t {
        Tm::Var(name) if polys.contains_key(name) => {
            return Err(format!(
                "`{name}` is level-polymorphic — it must be applied to its `Level` \
                 argument(s) (a bare reference cannot be monomorphized)"
            ))
        }
        Tm::Var(_) | Tm::Lit(_) | Tm::Str(_) => t.clone(),
        Tm::Ann(e, ty) => Tm::Ann(
            Box::new(level_rewrite_tm(e, polys, subst, queue)?),
            Box::new(subst_level_ty(ty, subst)),
        ),
        Tm::Call(name, args) => {
            if let Some(pd) = polys.get(name) {
                let np = pd.level_params.len();
                if args.len() < np {
                    return Err(format!(
                        "`{name}` needs {np} `Level` argument(s) first"
                    ));
                }
                let mut levels = Vec::with_capacity(np);
                for (k, (pos, _)) in pd.level_params.iter().enumerate() {
                    let _ = pos;
                    levels.push(resolve_level_arg(&args[k], subst)?);
                }
                let rest = args[np..]
                    .iter()
                    .map(|a| level_rewrite_tm(a, polys, subst, queue))
                    .collect::<Result<Vec<_>, _>>()?;
                if !queue.iter().any(|(n, ls)| n == name && *ls == levels) {
                    queue.push((name.clone(), levels.clone()));
                }
                Tm::Call(level_mono_name(name, &levels), rest)
            } else {
                Tm::Call(
                    name.clone(),
                    args.iter()
                        .map(|a| level_rewrite_tm(a, polys, subst, queue))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        }
        Tm::Add(a, b) => Tm::Add(
            Box::new(level_rewrite_tm(a, polys, subst, queue)?),
            Box::new(level_rewrite_tm(b, polys, subst, queue)?),
        ),
        Tm::Let(n, e, b) => Tm::Let(
            n.clone(),
            Box::new(level_rewrite_tm(e, polys, subst, queue)?),
            Box::new(level_rewrite_tm(b, polys, subst, queue)?),
        ),
        Tm::LetPair(ns, e, b) => Tm::LetPair(
            ns.clone(),
            Box::new(level_rewrite_tm(e, polys, subst, queue)?),
            Box::new(level_rewrite_tm(b, polys, subst, queue)?),
        ),
        Tm::Match(s, arms) => Tm::Match(
            s.clone(),
            arms.iter()
                .map(|a| {
                    Ok(Arm {
                        ctor: a.ctor.clone(),
                        binders: a.binders.clone(),
                        pats: a.pats.clone(),
                        body: level_rewrite_tm(&a.body, polys, subst, queue)?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        ),
        Tm::MatchN(cols, rows) => Tm::MatchN(
            cols.clone(),
            rows.iter()
                .map(|(ps, b)| Ok((ps.clone(), level_rewrite_tm(b, polys, subst, queue)?)))
                .collect::<Result<Vec<_>, String>>()?,
        ),
        Tm::Lam(ps, b) => Tm::Lam(ps.clone(), Box::new(level_rewrite_tm(b, polys, subst, queue)?)),
    })
}

fn monomorphize_level_poly(items: &mut Vec<Item>) -> Result<(), String> {
    let mut sigs: HashMap<String, Ty> = HashMap::new();
    for it in items.iter() {
        if let Item::Sig(n, t) = it {
            sigs.insert(n.clone(), t.clone());
        }
    }
    let mut polys: HashMap<String, LevelPolyDef> = HashMap::new();
    for it in items.iter() {
        if let Item::Fn(name, params, body, annot) = it {
            let Some(sig) = sigs.get(name) else { continue };
            let lps = level_params_of(sig)?;
            if !lps.is_empty() {
                polys.insert(
                    name.clone(),
                    LevelPolyDef {
                        sig: sig.clone(),
                        params: params.clone(),
                        body: body.clone(),
                        annot: *annot,
                        level_params: lps,
                    },
                );
            }
        }
    }
    if polys.is_empty() {
        return Ok(());
    }
    let mut queue: Vec<(String, Vec<usize>)> = Vec::new();
    for it in items.iter_mut() {
        if let Item::Fn(name, _, body, _) = it {
            if !polys.contains_key(name) {
                *body = level_rewrite_tm(body, &polys, &HashMap::new(), &mut queue)?;
            }
        }
    }
    let mut made: Vec<(String, Item, Item)> = Vec::new();
    let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut qi = 0;
    while qi < queue.len() {
        let (orig, levels) = queue[qi].clone();
        qi += 1;
        let iname = level_mono_name(&orig, &levels);
        if !done.insert(iname.clone()) {
            continue;
        }
        let pd = &polys[&orig];
        let subst: HashMap<String, usize> = pd
            .level_params
            .iter()
            .zip(&levels)
            .map(|((_, n), l)| (n.clone(), *l))
            .collect();
        let isig = strip_level_arrows(&pd.sig, &subst);
        let iparams: Vec<String> = pd
            .params
            .iter()
            .filter(|p| !pd.level_params.iter().any(|(_, n)| n == *p))
            .cloned()
            .collect();
        let ibody = level_rewrite_tm(&pd.body, &polys, &subst, &mut queue)?;
        made.push((
            orig.clone(),
            Item::Sig(iname.clone(), isig),
            Item::Fn(iname, iparams, ibody, pd.annot),
        ));
    }
    let mut out: Vec<Item> = Vec::with_capacity(items.len() + made.len());
    for it in items.drain(..) {
        match &it {
            Item::Sig(n, _) if polys.contains_key(n) => {}
            Item::Fn(n, _, _, _) if polys.contains_key(n) => {
                for (orig, s, f) in &made {
                    if orig == n {
                        out.push(s.clone());
                        out.push(f.clone());
                    }
                }
            }
            _ => out.push(it),
        }
    }
    *items = out;
    Ok(())
}

// ===========================================================================
// NESTED PATTERNS — the pattern-matrix desugar (classic column splitting)
// ===========================================================================

impl Elab {
    /// A pattern normalized against the constructor table: a bare name that IS a
    /// declared constructor is a (nullary) constructor pattern, not a binder.
    fn norm_pat(&self, p: &Pat) -> Result<Pat, String> {
        match p {
            Pat::Var(v) if self.ctor_info.contains_key(v) || self.nat_ctor.contains_key(v) => {
                let nexpl = self.pat_ctor_arity(v);
                if nexpl != 0 {
                    return Err(format!(
                        "pattern `{v}` is a constructor with {nexpl} argument(s) — write \
                         `{v}(…)`, or rename the binder (it shadows the constructor)"
                    ));
                }
                Ok(Pat::Ctor(v.clone(), vec![]))
            }
            _ => Ok(p.clone()),
        }
    }

    /// EXPLICIT-argument count of a constructor (what a pattern binds).
    fn pat_ctor_arity(&self, c: &str) -> usize {
        if let Some(role) = self.nat_ctor.get(c) {
            return match role {
                NatRole::Zero => 0,
                NatRole::Succ => 1,
            };
        }
        self.ctor_info
            .get(c)
            .map(|i| i.arg_implicit.iter().filter(|b| !**b).count())
            .unwrap_or(0)
    }

    /// Rewrite every `match` in `t` so that all arms are FLAT (variable-only
    /// patterns, one arm per constructor): nested patterns compile to nested
    /// matches by the pattern-matrix algorithm; arms sharing an outer
    /// constructor are merged. Runs once, after the constructor table exists
    /// and before anything else consumes match arms.
    fn desugar_patterns(&self, t: &Tm, fresh: &mut usize) -> Result<Tm, String> {
        Ok(match t {
            Tm::Var(_) | Tm::Lit(_) | Tm::Str(_) => t.clone(),
            Tm::Ann(e, ty) => Tm::Ann(Box::new(self.desugar_patterns(e, fresh)?), ty.clone()),
            Tm::Lam(ps, b) => Tm::Lam(ps.clone(), Box::new(self.desugar_patterns(b, fresh)?)),
            Tm::Call(n, args) => Tm::Call(
                n.clone(),
                args.iter().map(|a| self.desugar_patterns(a, fresh)).collect::<Result<_, _>>()?,
            ),
            Tm::Add(a, b) => Tm::Add(
                Box::new(self.desugar_patterns(a, fresh)?),
                Box::new(self.desugar_patterns(b, fresh)?),
            ),
            Tm::Let(n, e, b) => Tm::Let(
                n.clone(),
                Box::new(self.desugar_patterns(e, fresh)?),
                Box::new(self.desugar_patterns(b, fresh)?),
            ),
            Tm::LetPair(ns, e, b) => Tm::LetPair(
                ns.clone(),
                Box::new(self.desugar_patterns(e, fresh)?),
                Box::new(self.desugar_patterns(b, fresh)?),
            ),
            Tm::Match(scrut, arms) => {
                let mut arms2 = Vec::with_capacity(arms.len());
                for a in arms {
                    let pats = a
                        .pats
                        .iter()
                        .map(|p| self.norm_pat(p))
                        .collect::<Result<Vec<_>, _>>()?;
                    arms2.push(Arm {
                        ctor: a.ctor.clone(),
                        binders: a.binders.clone(),
                        pats,
                        body: self.desugar_patterns(&a.body, fresh)?,
                    });
                }
                self.compile_arms(scrut, arms2, fresh)?
            }
            // a MULTI-SCRUTINEE match: straight into the pattern matrix over
            // all scrutinee columns at once — coverage then falls out of the
            // flat exhaustiveness check on every produced single-scrutinee
            // match (and the kernel's method-count re-check backstops it).
            Tm::MatchN(cols, rows) => {
                let mut rows2: Vec<(Vec<Pat>, Tm, usize)> = Vec::with_capacity(rows.len());
                for (i, (ps, b)) in rows.iter().enumerate() {
                    if ps.len() != cols.len() {
                        return Err(format!(
                            "`match` over {} scrutinees: arm {} has {} pattern(s)",
                            cols.len(),
                            i + 1,
                            ps.len()
                        ));
                    }
                    let nps =
                        ps.iter().map(|p| self.norm_pat(p)).collect::<Result<Vec<_>, _>>()?;
                    rows2.push((nps, self.desugar_patterns(b, fresh)?, i));
                }
                let nrows = rows2.len();
                let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
                let out = self.compile_matrix(cols, rows2, fresh, &mut used)?;
                if let Some(i) = (0..nrows).find(|i| !used.contains(i)) {
                    return Err(format!(
                        "unreachable `match` arm (arm {}): earlier patterns already cover it",
                        i + 1
                    ));
                }
                out
            }
        })
    }

    /// One `match`: if every arm is already flat (variable patterns only, no
    /// repeated constructor), pass it through untouched (binder names, arm
    /// order, everything — zero behavior change for existing programs).
    /// Otherwise group the arms by outer constructor and compile each group's
    /// pattern matrix.
    fn compile_arms(&self, scrut: &str, arms: Vec<Arm>, fresh: &mut usize) -> Result<Tm, String> {
        let is_ctor = |n: &str| {
            self.ctor_info.contains_key(n)
                || self.nat_ctor.contains_key(n)
                || self.poly_ctor_base.contains(n)
        };
        let mut seen: Vec<&str> = Vec::new();
        let mut needs = false;
        for a in &arms {
            if seen.contains(&a.ctor.as_str()) {
                needs = true; // repeated outer ctor: merge via the matrix
            }
            seen.push(&a.ctor);
            if a.pats.iter().any(|p| matches!(p, Pat::Ctor(_, _))) {
                needs = true;
            }
            // a TOP-LEVEL catch-all arm (`x => …` / `_ => …`, Phase A1): a
            // non-constructor name with no arguments is a variable row —
            // route through the matrix, which expands it over the rest of
            // the family. An ARGFUL unknown name stays a hard error (it can
            // only be a typo, never a binder).
            if !is_ctor(&a.ctor) {
                if a.binders.is_empty() {
                    needs = true;
                } else {
                    return Err(format!(
                        "`{}` is not a declared constructor (a pattern with \
                         arguments must name one)",
                        a.ctor
                    ));
                }
            }
        }
        if !needs {
            let flat = arms
                .into_iter()
                .map(|mut a| {
                    a.pats = vec![];
                    a
                })
                .collect();
            return Ok(Tm::Match(scrut.to_string(), flat));
        }
        // ONE-COLUMN pattern matrix over the scrutinee: each source arm is a
        // row whose single pattern is its (top-level) constructor pattern, or
        // a variable for a catch-all arm. First-match-wins WITHIN a branch (a
        // shadowed row is silently dropped there), but a source arm that wins
        // in NO branch is genuinely dead — track which arms were ever used
        // and reject the rest.
        let mut rows: Vec<(Vec<Pat>, Tm, usize)> = Vec::with_capacity(arms.len());
        for (i, a) in arms.iter().enumerate() {
            let pat = if is_ctor(&a.ctor) {
                let want = self.pat_ctor_arity(&a.ctor);
                if a.pats.len() != want {
                    return Err(format!(
                        "pattern `{}`: expected {} binder(s), got {}",
                        a.ctor,
                        want,
                        a.pats.len()
                    ));
                }
                Pat::Ctor(a.ctor.clone(), a.pats.clone())
            } else {
                Pat::Var(a.ctor.clone())
            };
            rows.push((vec![pat], a.body.clone(), i));
        }
        let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let out = self.compile_matrix(&[scrut.to_string()], rows, fresh, &mut used)?;
        if let Some(i) = (0..arms.len()).find(|i| !used.contains(i)) {
            return Err(format!(
                "unreachable `match` arm for `{}`: earlier patterns already cover it",
                arms[i].ctor
            ));
        }
        Ok(out)
    }

    /// Column binder names: keep the user's name when a single row binds a plain
    /// variable there; otherwise a fresh placeholder (user variables are then
    /// re-bound by `let` inside `compile_matrix`).
    fn matrix_binders(&self, k: usize, rows: &[&[Pat]], fresh: &mut usize) -> Vec<String> {
        (0..k)
            .map(|j| {
                if rows.len() == 1 {
                    if let Pat::Var(v) = &rows[0][j] {
                        return v.clone();
                    }
                }
                *fresh += 1;
                format!("$q{fresh}")
            })
            .collect()
    }

    /// The pattern matrix: `cols` are the in-scope occurrence names, `rows` the
    /// remaining (patterns, body) alternatives in source order. Splits on the
    /// leftmost column holding a constructor pattern; variable rows join every
    /// constructor group (their variable re-bound to the whole column value).
    /// All-variable rows: the FIRST wins and any later row is an unreachable-arm
    /// error (same redundancy discipline as flat matches).
    fn compile_matrix(
        &self,
        cols: &[String],
        rows: Vec<(Vec<Pat>, Tm, usize)>,
        fresh: &mut usize,
        used: &mut std::collections::HashSet<usize>,
    ) -> Result<Tm, String> {
        let split = (0..cols.len())
            .find(|j| rows.iter().any(|(ps, _, _)| matches!(ps[*j], Pat::Ctor(_, _))));
        let Some(j) = split else {
            // all variables: the FIRST row wins; later rows are shadowed HERE
            // (they may still win in another branch — `compile_arms` rejects the
            // ones that never do).
            let (ps, body, id) = rows.into_iter().next().expect("non-empty pattern matrix");
            used.insert(id);
            let mut b = body;
            for (i, p) in ps.iter().enumerate().rev() {
                if let Pat::Var(v) = p {
                    if v != &cols[i] && !v.starts_with('$') {
                        b = Tm::Let(v.clone(), Box::new(Tm::Var(cols[i].clone())), Box::new(b));
                    }
                }
            }
            return Ok(b);
        };
        // constructors split on, in first-appearance order — plus, when a
        // variable row exists, every OTHER constructor of the same family (the
        // variable covers them; the family is read off any named constructor).
        let mut ctors: Vec<String> = Vec::new();
        for (ps, _, _) in &rows {
            if let Pat::Ctor(c, _) = &ps[j] {
                if !ctors.contains(c) {
                    ctors.push(c.clone());
                }
            }
        }
        let has_var_row = rows.iter().any(|(ps, _, _)| matches!(ps[j], Pat::Var(_)));
        if has_var_row {
            let named = ctors[0].clone();
            let family: Vec<String> = if self.nat_ctor.contains_key(&named) {
                self.nat_ctor.keys().cloned().collect()
            } else if let Some(info) = self.ctor_info.get(&named) {
                self.rc
                    .data(&info.data)
                    .map(|d| d.ctors.iter().map(|c| c.name.clone()).collect())
                    .unwrap_or_default()
            } else {
                vec![]
            };
            for c in family {
                if !ctors.contains(&c) {
                    ctors.push(c);
                }
            }
        }
        let mut inner_arms = Vec::with_capacity(ctors.len());
        for c in &ctors {
            let k2 = self.pat_ctor_arity(c);
            let mut sub_rows: Vec<(Vec<Pat>, Tm, usize)> = Vec::new();
            for (ps, body, id) in &rows {
                match &ps[j] {
                    Pat::Ctor(c2, subs) if c2 == c => {
                        if subs.len() != k2 {
                            return Err(format!(
                                "pattern `{c}`: expected {k2} argument(s), got {}",
                                subs.len()
                            ));
                        }
                        let mut nps = ps[..j].to_vec();
                        nps.extend(subs.iter().map(|sp| self.norm_pat(sp)).collect::<Result<Vec<_>, _>>()?);
                        nps.extend_from_slice(&ps[j + 1..]);
                        sub_rows.push((nps, body.clone(), *id));
                    }
                    Pat::Ctor(_, _) => {}
                    Pat::Var(v) => {
                        // the variable covers this constructor too: wildcards for
                        // the sub-positions, the variable re-bound to the column.
                        let mut nps = ps[..j].to_vec();
                        for _ in 0..k2 {
                            *fresh += 1;
                            nps.push(Pat::Var(format!("$w{fresh}")));
                        }
                        nps.extend_from_slice(&ps[j + 1..]);
                        let b = if v.starts_with('$') {
                            body.clone()
                        } else {
                            Tm::Let(v.clone(), Box::new(Tm::Var(cols[j].clone())), Box::new(body.clone()))
                        };
                        sub_rows.push((nps, b, *id));
                    }
                }
            }
            if sub_rows.is_empty() {
                continue; // no row reaches this constructor: absent arm (coverage decides)
            }
            let sub_pats: Vec<&[Pat]> = sub_rows
                .iter()
                .map(|(ps, _, _)| &ps[j..j + k2])
                .collect();
            let binders = self.matrix_binders(k2, &sub_pats, fresh);
            let mut ncols = cols[..j].to_vec();
            ncols.extend(binders.iter().cloned());
            ncols.extend_from_slice(&cols[j + 1..]);
            let body = self.compile_matrix(&ncols, sub_rows, fresh, used)?;
            inner_arms.push(Arm { ctor: c.clone(), binders, pats: vec![], body });
        }
        Ok(Tm::Match(cols[j].clone(), inner_arms))
    }
}

/// Stable topological sort of the `Item::Fn` entries by their call graph:
/// callees first, original order preserved among independent definitions.
/// Non-`Fn` items keep their positions (the `Fn`s permute among their own
/// slots). On a cycle the remaining functions stay in source order.
fn reorder_fns_by_calls(items: &mut [Item]) {
    let slots: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, it)| matches!(it, Item::Fn(_, _, _, _)))
        .map(|(i, _)| i)
        .collect();
    if slots.len() < 2 {
        return;
    }
    let fn_names: std::collections::HashSet<String> = slots
        .iter()
        .filter_map(|&i| match &items[i] {
            Item::Fn(n, _, _, _) => Some(n.clone()),
            _ => None,
        })
        .collect();
    // per fn: the OTHER fns its body calls (self-calls excluded).
    let deps: Vec<(String, std::collections::HashSet<String>)> = slots
        .iter()
        .map(|&i| {
            let Item::Fn(n, _, body, _) = &items[i] else { unreachable!() };
            let mut calls = Vec::new();
            collect_all_calls(body, &mut calls);
            let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
            for c in &calls {
                if c.callee != *n && fn_names.contains(&c.callee) {
                    names.insert(c.callee.clone());
                }
            }
            // a bare `Var` reference to a fn (e.g. passing it as a value, or a
            // zero-arg def used as a value) is a dependency too.
            let mut refs = std::collections::HashSet::new();
            collect_tm_names(body, &mut refs);
            for r in refs {
                if r != *n && fn_names.contains(&r) {
                    names.insert(r);
                }
            }
            (n.clone(), names)
        })
        .collect();
    let mut emitted: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut order: Vec<usize> = Vec::with_capacity(slots.len()); // indices into `deps`
    let mut remaining: Vec<usize> = (0..deps.len()).collect();
    while !remaining.is_empty() {
        let pick = remaining
            .iter()
            .position(|&k| deps[k].1.iter().all(|d| emitted.contains(d) || !remaining.iter().any(|&r| deps[r].0 == *d)))
            .unwrap_or(0); // cycle: earliest remaining, original order
        let k = remaining.remove(pick);
        emitted.insert(deps[k].0.clone());
        order.push(k);
    }
    // permute the Fn items into the slots per `order`.
    let originals: Vec<Item> = slots.iter().map(|&i| items[i].clone()).collect();
    for (slot_pos, &k) in order.iter().enumerate() {
        items[slots[slot_pos]] = originals[k].clone();
    }
}

/// Every identifier a surface term references (variables, callees, match
/// scrutinees — recursively). Shadowing is ignored: a shadowed name still
/// counts as referenced, which over-approximates (safe — over-abstraction is
/// caught by the kernel; the filter's job is only to skip CLEARLY-unused deps).
fn collect_tm_names(t: &Tm, out: &mut std::collections::HashSet<String>) {
    match t {
        Tm::Ann(e, _) => collect_tm_names(e, out),
        Tm::Lam(_, b) => collect_tm_names(b, out),
        Tm::Var(nm) => {
            out.insert(nm.clone());
        }
        Tm::Call(nm, args) => {
            out.insert(nm.clone());
            for a in args {
                collect_tm_names(a, out);
            }
        }
        Tm::Match(s, arms) => {
            out.insert(s.clone());
            for a in arms {
                collect_tm_names(&a.body, out);
            }
        }
        Tm::MatchN(cols, rows) => {
            for c in cols {
                out.insert(c.clone());
            }
            for (_, b) in rows {
                collect_tm_names(b, out);
            }
        }
        Tm::Lit(_) | Tm::Str(_) => {}
        Tm::Add(a, b) => {
            collect_tm_names(a, out);
            collect_tm_names(b, out);
        }
        Tm::LetPair(_, e, b) | Tm::Let(_, e, b) => {
            collect_tm_names(e, out);
            collect_tm_names(b, out);
        }
    }
}

/// Does `t` (quoted at level `n`) mention any of the context levels in `lvls`?
fn term_mentions_levels(t: &Term, n: usize, lvls: &[usize]) -> bool {
    let hit = std::cell::Cell::new(false);
    dep::map_vars(t, 0, &|i, depth| {
        if i >= depth {
            let outer = i - depth;
            if outer < n && lvls.contains(&(n - 1 - outer)) {
                hit.set(true);
            }
        }
        Term::Var(i)
    });
    hit.get()
}

/// A solver's hole block: `slots[i]` is the hole with GLOBAL id `base + i`
/// (level `HOLE_BASE + base + i`). Blocks are allocated from `Elab::hole_ctr`,
/// so every in-flight solver owns a DISJOINT id range: a FOREIGN hole (an
/// enclosing call's still-open implicit appearing in `expected`) is simply
/// opaque here — bindable as a solution's CONTENT (then caught by
/// `value_has_hole`) but never aliased onto this block's slots.
struct Holes {
    base: usize,
    slots: Vec<Option<Value>>,
}

impl Holes {
    /// the solved value at GLOBAL id `gid`, if this block owns it and it is solved.
    fn lookup(&self, gid: usize) -> Option<&Value> {
        gid.checked_sub(self.base)
            .and_then(|i| self.slots.get(i))
            .and_then(|s| s.as_ref())
    }
    fn owns(&self, gid: usize) -> bool {
        gid.checked_sub(self.base).is_some_and(|i| i < self.slots.len())
    }
    fn set(&mut self, gid: usize, v: Value) {
        let i = gid - self.base;
        self.slots[i] = Some(v);
    }
}

/// First-order matching: bind holes in `holes` so that `pat` matches `target`.
/// Non-hole mismatches are ignored (the kernel re-checks the final term).
fn solve(holes: &mut Holes, pat: &Value, target: &Value) {
    let pat = deref(holes, pat);
    let target = deref(holes, target);
    if let Some(id) = hole_id(&pat) {
        if holes.owns(id) {
            holes.set(id, target);
            return;
        }
    }
    if let Some(id) = hole_id(&target) {
        if holes.owns(id) {
            holes.set(id, pat);
            return;
        }
    }
    // two foreign holes / a foreign hole vs. structure: nothing to bind here —
    // fall through (mismatches are ignored; the kernel re-check is the backstop).
    if hole_id(&pat).is_some() || hole_id(&target).is_some() {
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
        // FUNCTION types — descend into both the domain and the codomain so an
        // implicit that appears ONLY in a higher-order argument's type is pinned
        // (e.g. `map`'s result-element `b` from the callback `f : a -> b` when it
        // is passed `dbl : Nat -> Nat`). The codomain is a closure; instantiate
        // both sides at one shared RIGID sentinel (a level ≥ `HOLE_BASE` that no
        // block owns, so it acts as a foreign hole: it never mis-binds this
        // block's slots, and if it ever leaked into a solution `value_has_hole`
        // rejects it cleanly). Non-dependent arrows — every stdlib signature —
        // ignore the sentinel entirely, so the codomain bodies compare directly.
        (Value::VPi(_, d1, c1), Value::VPi(_, d2, c2)) => {
            solve(holes, d1, d2);
            let fresh = crate::dep::nvar(HOLE_BASE + (usize::MAX / 4));
            solve(holes, &c1.apply(fresh.clone()), &c2.apply(fresh));
        }
        // IDENTITY types — structural, like `VData`: unify the carrier and the two
        // endpoints. This lets a proof combinator recover its implicit endpoints
        // (`cong`/`sym`/`trans`'s `{0 x}{0 y}`) from the `Eq A x y` type of the
        // equality argument it is given — so they read like Idris (the endpoints
        // are inferred, not written).
        (Value::VEq(a1, x1, y1), Value::VEq(a2, x2, y2)) => {
            solve(holes, a1, a2);
            solve(holes, x1, x2);
            solve(holes, y1, y2);
        }
        (Value::VNeu(n1), Value::VNeu(n2)) => solve_neu(holes, n1, n2),
        _ => {}
    }
}

/// Descend into a neutral spine (e.g. `Own ?a` vs `Own Nat`) to bind holes.
fn solve_neu(holes: &mut Holes, n1: &crate::dep::Neutral, n2: &crate::dep::Neutral) {
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
            Neutral::NCase(_, m, ms, sc) => {
                value_has_hole(m) || ms.iter().any(value_has_hole) || neu(sc)
            }
            Neutral::NNatCase(p, z, s, sc) => {
                value_has_hole(p) || value_has_hole(z) || value_has_hole(s) || neu(sc)
            }
            Neutral::NJ(p, b, e) => value_has_hole(p) || value_has_hole(b) || neu(e),
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
        V::VType(_) | V::VNat | V::VNatLit(_) | V::VStr(_) | V::VLam(_) | V::VLamNative(_) => false,
    }
}

fn deref(holes: &Holes, v: &Value) -> Value {
    if let Some(id) = hole_id(v) {
        if let Some(sol) = holes.lookup(id) {
            let sol = sol.clone();
            return deref(holes, &sol);
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
fn peel_arrows(t: &Ty) -> (Vec<(SMult, bool, Option<String>, Ty)>, Ty) {
    let mut out = Vec::new();
    let mut t = t.clone();
    while let Ty::Arrow(m, imp, name, a, b) = t {
        out.push((m, imp, name, *a));
        t = *b;
    }
    (out, t)
}

/// A datatype head's index telescope: the arrow count and the UNIVERSE the
/// family is declared to live in (`… -> Type i`, default `Type` = 0).
fn count_index_pis(t: &Ty) -> Result<(usize, usize), String> {
    let (arrows, ret) = peel_arrows(t);
    match ret {
        Ty::Type(l) => Ok((arrows.len(), l)),
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
        let ni = decl.indices.len();
        if ni == 0 {
            return Ok(None);
        }
        let p = full_params.len();
        // scrutinee parameter + index values, in the function-parameter context.
        let sparam_vals: Vec<Value> = dargs[..np]
            .iter()
            .map(|a| Ok(dep::eval_rc(&self.rc, &neutral_env(p), &self.elab_ty(a, full_params)?)))
            .collect::<Result<_, String>>()?;
        // v2 (Phase A1): find ANY index position `k` that REFUTES the whole
        // family — the scrutinee's index-k value has a constructor head and
        // EVERY constructor's result index at k has a decidably different
        // head. (v1 was the single-index Nat-only special case.)
        let mut refuting: Option<(usize, String)> = None;
        'positions: for k in 0..ni {
            let sidx_tm = self.elab_ty(dargs[np + k], full_params)?;
            let sidx_nf = dep::quote_at(p, &dep::eval_rc(&self.rc, &neutral_env(p), &sidx_tm));
            let Some(s_head) = ctor_head(&sidx_nf) else {
                continue; // index not a known constructor: not refuting here
            };
            for ctor in &decl.ctors {
                let mut env = sparam_vals.clone();
                for j in 0..ctor.args.len() {
                    env.push(dep::nvar(p + j));
                }
                let cidx_nf =
                    dep::quote_at(p + ctor.args.len(), &dep::eval_rc(&self.rc, &env, &ctor.idxs[k]));
                let disjoint = matches!(ctor_head(&cidx_nf), Some(h) if h != s_head);
                if !disjoint {
                    continue 'positions; // some constructor may be reachable at k
                }
            }
            refuting = Some((k, s_head));
            break;
        }
        let Some((k, s_head)) = refuting else {
            return Ok(None); // no refuting index ⇒ ordinary coverage
        };
        // ALL constructors absurd ⇒ the type is empty. Reject any arms (the cases
        // are impossible and cannot be written), then synthesize the discharge.
        if !arms.is_empty() {
            return Err(format!(
                "`match` on `{data}` at this index is absurd (every constructor is \
                 impossible) — it must have NO arms; remove `{}`",
                arms[0].ctor
            ));
        }
        let t_term = self.elab_ty(ret, full_params)?;
        // ⚠️ BACKSTOP CAVEAT (read before extending to MIXED / per-constructor
        // absurd discharge): the sentinel below makes the kernel-rejection
        // backstop T-DEPENDENT. At a mis-classified-REACHABLE index the discharge
        // term has type `Nat`, so the kernel rejects it ONLY when the match's
        // result type `T ≠ Nat`. That is SOUND HERE because this path fires *only*
        // when EVERY constructor is decidably absurd (the loop above returns
        // `None` otherwise), so there is no reachable case to mis-classify and the
        // verdict does not lean on the backstop at all. But a per-constructor /
        // mixed discharge, where a reachable constructor COULD be wrongly called
        // absurd, would SILENTLY accept the mistake when `T = Nat`. Before that
        // extension, switch to a for-all-T sentinel (a fresh uninhabited type via
        // the two-step `→ Void`, `elim Void`), or prove the classifier sound
        // independently. Do NOT carry the Nat sentinel into mixed coverage.
        //
        // THE SENTINEL, generalized (v2): a type function over index k that
        // computes `T` exactly at the scrutinee's (refuting) head and `Nat` at
        // every other head. Under the motive's `ni+1` lambdas the index-k
        // binder is Var(ni - k); `T` shifts by `ni+1` (+ any sentinel-local
        // binders). For a `%builtin Nat` index it is the v1 `NatCase`; for a
        // SIMPLE datatype index (no params, no indices) it is a LARGE-
        // ELIMINATING kernel `Case` (allowed — the kernel validates the
        // motive at any level), which the kernel re-checks like everything
        // else.
        let t_shift = dep::shift_term(ni + 1, &t_term);
        let e = Term::Var(ni - k);
        let sentinel = match &decl.indices[k].1 {
            Term::Nat => Term::NatCase(
                Box::new(Term::Lam(Box::new(Term::Type(0)))),
                Box::new(if s_head == "Zero" { t_shift.clone() } else { Term::Nat }),
                Box::new(Term::Lam(Box::new(if s_head == "Succ" {
                    dep::shift_term(1, &t_shift)
                } else {
                    Term::Nat
                }))),
                Box::new(e),
            ),
            Term::Data(iname, iargs) if iargs.is_empty() => {
                let idecl = self
                    .rc
                    .data(iname)
                    .ok_or_else(|| format!("unknown index datatype `{iname}`"))?
                    .clone();
                if !idecl.params.is_empty() || !idecl.indices.is_empty() {
                    return Err(format!(
                        "`match` on `{data}` here is an absurd (empty) case, but absurd \
                         discharge over an index of the PARAMETERIZED/INDEXED type \
                         `{iname}` is not supported yet — only `Nat` and simple enum \
                         indices are (this is a completeness limit, not an unsoundness)."
                    ));
                }
                let methods: Vec<Term> = idecl
                    .ctors
                    .iter()
                    .map(|c| {
                        let mut m = if c.name == s_head {
                            dep::shift_term(c.args.len(), &t_shift)
                        } else {
                            Term::Nat
                        };
                        for _ in 0..c.args.len() {
                            m = Term::Lam(Box::new(m));
                        }
                        m
                    })
                    .collect();
                Term::Case(
                    iname.clone(),
                    Box::new(Term::Lam(Box::new(Term::Type(0)))),
                    methods,
                    Box::new(e),
                )
            }
            _ => {
                return Err(format!(
                    "`match` on `{data}` here is an absurd (empty) case, but its \
                     refuting index has a type absurd discharge cannot case on \
                     (only `Nat` and simple enum indices are supported)."
                ));
            }
        };
        // motive = λ idx_0 … λ idx_{ni-1}. λ _scrut. sentinel
        let mut motive = sentinel;
        for _ in 0..(ni + 1) {
            motive = Term::Lam(Box::new(motive));
        }
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
        let r = Rec { fnname, scrut_pos: explicit_pos, fields: &fields, acc_tys: None, result_ty: p_at(Value::VNeu(dep::Neutral::NVar(succ_cx.len()))) , fold: None };
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
        // NOTE (Phase #1 arbitrary-length, see docs/PHASE_ARBITRARY_LENGTH_PLAN.md): a
        // LINEAR accumulator needs `Mult::One` here (threading `ω` scales its usage to
        // `ω` ⇒ `ω⋢1`). That one-line change is necessary but NOT sufficient — clean
        // arbitrary-length LINEAR traversal needs structural-unbox descent (recurse on
        // the unbox'd tail), not a fuel fold (whose base case can't free the remaining
        // linear list). Landing the mult change + the descent + the routing together.
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
        let r = Rec { fnname, scrut_pos: full_pos, fields: &fields, acc_tys: Some(&acc_ty_vals), result_ty: r_val.clone() , fold: None };
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

    /// GENERAL `%partial` recursion on a HEAP/boxed structure: compile the WHOLE body to
    /// `Fix(ty, λparams. <body>)` in FIX MODE (`in_fix`), so every boxed `match` lowers
    /// to a NON-recursive `Term::Case` and a recursive call resolves to the `Fix`
    /// self-binder (`fnname` is in scope). This is what lets a function recurse on an
    /// owned linked list / tree / AST (the interpreter) — dispatch is `Case`, recursion
    /// is the self-call, no implicit-IH blow-up. Used when the structural verdict is
    /// `Partial` and the scrutinee is NOT a `%builtin Nat` (which keeps `elab_fix_nat`).
    fn elab_fix(
        &self,
        fnname: &str,
        ty_term: &Term,
        full_names: &[String],
        full_tys: &[Ty],
        ret: &Ty,
        body: &Tm,
    ) -> Result<Term, String> {
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
        let ret_tm = self.elab_ty(ret, &scope)?;
        let expected = dep::eval_rc(&self.rc, &neutral_env(cx.len()), &ret_tm);
        // FIX MODE: boxed matches → `Case` (no IH); recursion is the self-call (rec=None,
        // `fnname` resolves to the self-binder). Restore the flag even on error.
        let prev = self.in_fix.replace(true);
        let checked = self.check(body, &expected, &cx, None);
        self.in_fix.set(prev);
        let mut body_term = checked?;
        for _ in 0..full_names.len() {
            body_term = Term::Lam(Box::new(body_term));
        }
        Ok(Term::Fix(Box::new(ty_term.clone()), Box::new(body_term)))
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

        // THE NAT CONVOY: if a context variable's type mentions the scrutinee, or a
        // linear variable is consumed inside the arms, lower to a dependent
        // `NatCase` with a function-typed motive (see `try_nat_convoy_case`).
        let ret_cx_tm = self.elab_ty(ret, scope)?;
        if let Some(t) = self.try_nat_convoy_case(
            &Term::Var(scrut_idx),
            &ret_cx_tm,
            zero_arm,
            succ_arm,
            cx,
            None,
        )? {
            return Ok(t);
        }

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
            // FULL-ARITY patterns may NAME the implicit binders too (open the
            // existentials — see the nested elaborator's twin).
            let full_arity = nargs != nexplicit && arm.binders.len() == nargs;
            if arm.binders.len() != nexplicit && !full_arity {
                return Err(format!(
                    "case `{}`: expected {nexplicit} pattern binder(s) (or all \
                     {nargs} to bind the implicits too), got {}",
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
                if full_arity {
                    binder_names.push(arm.binders[j].clone());
                } else if info.arg_implicit[j] {
                    binder_names.push(info.arg_names[j].clone().unwrap_or_else(|| format!("$imp{j}")));
                } else {
                    binder_names.push(arm.binders[next_pat].clone());
                    next_pat += 1;
                }
            }
            // typing context: the fn params, then the method binders with their
            // kernel types (from the eliminator method telescope)
            let motive_tm = motive.clone();
            let (binder_tys, ret_ty_tm) =
                dep::elim_method_telescope(&self.rc, &data, &sparam_tms, &motive_tm, &ctor.name, true)?;
            let mut fields: HashMap<String, String> = HashMap::new();
            for (kk, &fi) in rec_fields.iter().enumerate() {
                let ih = format!("$ih{kk}");
                fields.insert(binder_names[fi].clone(), ih.clone());
                // A LINEAR recursive field is consumed by the FOLD ITSELF: its one
                // legal use is through the induction hypothesis (`fields` maps the
                // surface name there), so HIDE the direct binder — a second, direct
                // consumption of the tail (a double-free the eliminator's own
                // traversal would compound) becomes unrepresentable, and the
                // use-site rebind correctly does not demand a direct use.
                if type_is_linear_in(
                    &binder_tys[fi].1,
                    &self.rc,
                    &self.pess_linear.borrow(),
                    &fn_cx.names,
                    fn_cx.len() + fi,
                ) {
                    binder_names[fi] = format!("$folded{fi}");
                }
                binder_names.push(ih);
            }
            let mut arm_cx = fn_cx.clone();
            for (bn, (_, bty)) in binder_names.iter().zip(&binder_tys) {
                let v = dep::eval_rc(&self.rc, &neutral_env(arm_cx.len()), bty);
                arm_cx.push(bn.clone(), v);
            }
            let expected = dep::eval_rc(&self.rc, &neutral_env(arm_cx.len()), &ret_ty_tm);

            let r = Rec { fnname, scrut_pos: explicit_pos, fields: &fields, acc_tys: None, result_ty: expected.clone() , fold: None };
            // USE-SITE LINEARITY (the convergent, whack-a-mole-proof check): re-bind
            // each EXPLICIT field whose INSTANTIATED type is linear via `let f = f`, so
            // the let-binder rule binds it at `1` and the kernel enforces exactly-once
            // use through it. Because it checks the field's ACTUAL (post-substitution)
            // type at the USE SITE, it catches a hidden `Own` no matter HOW it is hidden
            // — a generic container instantiated at `Own` (`Pair (Own Nat) Unit`), a
            // nested generic, an alias — not just a syntactic field. (Non-linear fields
            // are untouched, so copyable fields stay ω/multi-usable.)
            let arm_body = rebind_linear_fields(&arm.body, &binder_names, &info.arg_implicit, &binder_tys, nargs, &self.rc, &self.pess_linear.borrow(), &arm_cx.names, fn_cx.len());
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
                    return Ok((names, imps, tys, Ty::Arrow(SMult::Lit(Mult::Omega), implicit, name, a, b)));
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
    let no_pess = std::collections::HashSet::new();
    contains_linear(ty, rc, &mut std::collections::HashSet::new(), &no_pess, &[], 0, 0)
}

/// `type_is_linear` UNDER a pessimistic assumption: the context variables whose
/// NAMES are in `pess` (resolved through `cx_names`; `lvl` = the level `ty` was
/// quoted at, `depth` = binders crossed inside `ty`) are treated as LINEAR
/// types. This is how a generic function's body is checked for LINEAR-CAPABILITY
/// of a type parameter (see `def_linear_capable`): with the parameter assumed
/// linear, every use-site rule (match-field rebind, `let` binder multiplicity,
/// convoy dependents) binds abstract-typed values at `1`.
fn type_is_linear_in(
    ty: &Term,
    rc: &Rc<Signature>,
    pess: &std::collections::HashSet<String>,
    cx_names: &[String],
    lvl: usize,
) -> bool {
    contains_linear(ty, rc, &mut std::collections::HashSet::new(), pess, cx_names, lvl, 0)
}

fn contains_linear(
    ty: &Term,
    rc: &Rc<Signature>,
    seen: &mut std::collections::HashSet<String>,
    pess: &std::collections::HashSet<String>,
    cx_names: &[String],
    lvl: usize,
    depth: usize,
) -> bool {
    let go = |t: &Term, seen: &mut std::collections::HashSet<String>| {
        contains_linear(t, rc, seen, pess, cx_names, lvl, depth)
    };
    let go1 = |t: &Term, seen: &mut std::collections::HashSet<String>| {
        contains_linear(t, rc, seen, pess, cx_names, lvl, depth + 1)
    };
    match ty {
        Term::Const(n) => rc.linear_types.contains(n),
        Term::Sigma(crate::mult::Mult::One, _, _) => true,
        Term::Sigma(_, a, b) => go(a, seen) || go1(b, seen),
        // A FUNCTION is not a resource: a Π CONSUMING a linear argument (or
        // producing one per call) is itself freely copyable — every call gets a
        // fresh argument. (Propagating through Π made `(w f : (1 x : Own T) -> b)`
        // silently linear, breaking higher-order code over linear data. A genuine
        // one-shot function is declared with an explicit `(1 f : …)`.)
        Term::Pi(_, _, _) => false,
        Term::App(a, b) | Term::Pair(a, b) | Term::Add(a, b) => go(a, seen) || go(b, seen),
        Term::Eq(a, b, c) => go(a, seen) || go(b, seen) || go(c, seen),
        Term::Ann(e, _) => go(e, seen),
        Term::Suc(x) | Term::Fst(x) | Term::Snd(x) | Term::Refl(x) => go(x, seen),
        Term::Data(name, args) => {
            if rc.linear_types.contains(name) {
                return true;
            }
            // a linear type ARGUMENT (e.g. `Vec (Own T) n`, `Pair (Own Nat) Unit`).
            if args.iter().any(|a| go(a, seen)) {
                return true;
            }
            // a linear FIELD hidden behind the datatype's name — recurse into the
            // constructor field DEFINITIONS (the `seen` guard handles recursive types).
            if seen.insert(name.clone()) {
                if let Some(decl) = rc.data(name) {
                    for ctor in &decl.ctors {
                        for (_, fty) in &ctor.args {
                            if contains_linear(fty, rc, seen, pess, cx_names, lvl, depth) {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        }
        Term::Constr(_, args) => args.iter().any(|a| go(a, seen)),
        // an abstract type VARIABLE: linear exactly when the pessimistic set says
        // so (the linear-capability check); otherwise the §13 default (not linear —
        // an instantiation at a linear type is then gated by `def_linear_capable`).
        Term::Var(i) => {
            if pess.is_empty() || *i < depth {
                return false;
            }
            let outer = i - depth;
            if outer >= lvl {
                return false;
            }
            // context level of this variable; a level past `cx_names` is a local
            // binder introduced after the named context (e.g. a method-telescope
            // binder) — never a pessimistic type parameter.
            let level = lvl - 1 - outer;
            level < cx_names.len() && pess.contains(&cx_names[level])
        }
        // Type / Nat / NatLit / Zero / Lam / Fix / NatElim / NatCase / Elim:
        // no linear component.
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
    pess: &std::collections::HashSet<String>,
    cx_names: &[String],
    base_lvl: usize,
) -> Tm {
    let mut wrapped = body.clone();
    for j in (0..nargs).rev() {
        let explicit = !arg_implicit.get(j).copied().unwrap_or(false);
        // A 0-ERASED field (a `{0}` proof/index, even of a linear type) stays 0 — it
        // has no runtime representation and so no double-free risk; forcing it to 1
        // would OVER-reject. Only an ω/1 field of an instantiated-linear type is bound
        // at 1.
        let erased = binder_tys[j].0 == crate::mult::Mult::Zero;
        // A HIDDEN binder (`$folded…` — a linear RECURSIVE field whose consumption
        // belongs to the eliminator; see `elab_match_body`) is not directly usable,
        // so demanding a direct use of it would always fail: skip it.
        if binder_names[j].starts_with('$') {
            continue;
        }
        // `binder_tys[j]` sits under `j` earlier telescope binders: it is quoted at
        // level `base_lvl + j` for the pessimistic (linear-capability) resolution.
        if explicit
            && !erased
            && type_is_linear_in(&binder_tys[j].1, rc, pess, cx_names, base_lvl + j)
        {
            let nm = binder_names[j].clone();
            wrapped = Tm::Let(nm.clone(), Box::new(Tm::Var(nm)), Box::new(wrapped));
        }
    }
    wrapped
}

fn collect_all_calls(t: &Tm, out: &mut Vec<TCall>) {
    collect_calls_wf(t, &HashMap::new(), &[], out)
}

/// PHASE B3 — inline the bodies of FORWARD-referenced mutual-recursion
/// members: a call `g(args…)` where `g` is in `defining`'s SCC and appears
/// later in item order becomes `let p1 = a1; …; <g's body, itself unrolled>`
/// (let-scoping does the hygiene; calls back to `defining` remain and become
/// the `Fix` self-reference). A member repeating on the inline path means the
/// cycle is not a simple loop — declined honestly.
fn unroll_forward_scc_calls(
    defining: &str,
    t: &Tm,
    scc: &std::collections::HashSet<String>,
    order: &HashMap<String, usize>,
    defs: &HashMap<String, (Vec<String>, Tm)>,
    path: &mut Vec<String>,
) -> Result<Tm, String> {
    let go = |t: &Tm, path: &mut Vec<String>| -> Result<Tm, String> {
        unroll_forward_scc_calls(defining, t, scc, order, defs, path)
    };
    Ok(match t {
        Tm::Var(_) | Tm::Lit(_) | Tm::Str(_) => t.clone(),
        Tm::Ann(e, ty) => Tm::Ann(Box::new(go(e, path)?), ty.clone()),
        Tm::Lam(ps, b) => Tm::Lam(ps.clone(), Box::new(go(b, path)?)),
        Tm::Call(g, args) => {
            let forward = scc.contains(g)
                && order.get(g).zip(order.get(defining)).is_some_and(|(og, od)| og > od);
            let args2 = args
                .iter()
                .map(|a| go(a, path))
                .collect::<Result<Vec<_>, _>>()?;
            if !forward {
                return Ok(Tm::Call(g.clone(), args2));
            }
            if path.iter().any(|p| p == g) {
                return Err(format!(
                    "the mutual-recursion cycle through `{defining}` re-enters `{g}` \
                     while inlining — only simple mutual cycles can be lowered so \
                     far (Phase B3 v1); mark the group `%partial` and restructure, \
                     or break the cycle"
                ));
            }
            let (gparams, gbody) = defs
                .get(g)
                .ok_or_else(|| format!("unknown mutual-recursion member `{g}`"))?;
            if gparams.len() != args2.len() {
                return Err(format!(
                    "`{g}` expects {} argument(s), got {} (mutual-recursion inlining)",
                    gparams.len(),
                    args2.len()
                ));
            }
            path.push(g.clone());
            let inlined = go(gbody, path)?;
            path.pop();
            // let-bind right-to-left so evaluation order matches the call.
            let mut out = inlined;
            for (p, a) in gparams.iter().zip(args2).rev() {
                out = Tm::Let(p.clone(), Box::new(a), Box::new(out));
            }
            out
        }
        Tm::Add(a, b) => Tm::Add(Box::new(go(a, path)?), Box::new(go(b, path)?)),
        Tm::Let(n, e, b) => {
            Tm::Let(n.clone(), Box::new(go(e, path)?), Box::new(go(b, path)?))
        }
        Tm::LetPair(ns, e, b) => {
            Tm::LetPair(ns.clone(), Box::new(go(e, path)?), Box::new(go(b, path)?))
        }
        Tm::Match(s, arms) => Tm::Match(
            s.clone(),
            arms.iter()
                .map(|a| {
                    Ok(Arm {
                        ctor: a.ctor.clone(),
                        binders: a.binders.clone(),
                        pats: a.pats.clone(),
                        body: go(&a.body, path)?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        ),
        Tm::MatchN(cols, rows) => Tm::MatchN(
            cols.clone(),
            rows.iter()
                .map(|(ps, b)| Ok((ps.clone(), go(b, path)?)))
                .collect::<Result<Vec<_>, String>>()?,
        ),
    })
}

/// Does `t` mention the variable `v` anywhere (shadowing-unaware — an
/// over-approximation, used only to CONSERVATIVELY drop `dlt` facts)?
fn tm_mentions(t: &Tm, v: &str) -> bool {
    let mut found = false;
    let mut names = std::collections::HashSet::new();
    collect_tm_names(t, &mut names);
    if names.contains(v) {
        found = true;
    }
    found
}

/// `collect_all_calls`, threading the RUNTIME-WITNESSED `Lt` facts (Phase
/// E3/B1): inside the `DYes` arm of a `match` on a `let`-bound `dlt(e1, e2)`,
/// the fact `e1 < e2` holds whenever that arm runs (dlt IS the machine
/// compare). Facts and dlt-bindings are dropped conservatively when any
/// variable they mention is rebound.
fn collect_calls_wf(
    t: &Tm,
    dlt_binds: &HashMap<String, (Tm, Tm)>,
    facts: &[(Tm, Tm)],
    out: &mut Vec<TCall>,
) {
    // remove every binding/fact that mentions (or is) a rebound name.
    let shadow = |names: &[&String],
                  db: &HashMap<String, (Tm, Tm)>,
                  fs: &[(Tm, Tm)]|
     -> (HashMap<String, (Tm, Tm)>, Vec<(Tm, Tm)>) {
        let db2 = db
            .iter()
            .filter(|(k, (e1, e2))| {
                !names.iter().any(|n| {
                    *n == *k || tm_mentions(e1, n) || tm_mentions(e2, n)
                })
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let fs2 = fs
            .iter()
            .filter(|(e1, e2)| !names.iter().any(|n| tm_mentions(e1, n) || tm_mentions(e2, n)))
            .cloned()
            .collect();
        (db2, fs2)
    };
    match t {
        Tm::Ann(e, _) => collect_calls_wf(e, dlt_binds, facts, out),
        Tm::Call(n, args) => {
            out.push(TCall { callee: n.clone(), args: args.clone(), lt_facts: facts.to_vec() });
            for a in args {
                collect_calls_wf(a, dlt_binds, facts, out);
            }
        }
        Tm::Add(a, b) => {
            collect_calls_wf(a, dlt_binds, facts, out);
            collect_calls_wf(b, dlt_binds, facts, out);
        }
        Tm::Match(s, arms) => {
            for a in arms {
                let binders: Vec<&String> = a.binders.iter().collect();
                let (db, mut fs) = shadow(&binders, dlt_binds, facts);
                if a.ctor == "DYes" {
                    if let Some((e1, e2)) = dlt_binds.get(s) {
                        if !a.binders.iter().any(|b| tm_mentions(e1, b) || tm_mentions(e2, b)) {
                            fs.push((e1.clone(), e2.clone()));
                        }
                    }
                }
                collect_calls_wf(&a.body, &db, &fs, out);
            }
        }
        Tm::MatchN(_, rows) => {
            // pattern binders shadow; no fact survives conservatively.
            for (_, b) in rows {
                collect_calls_wf(b, &HashMap::new(), &[], out);
            }
        }
        Tm::LetPair(ns, e, body) => {
            collect_calls_wf(e, dlt_binds, facts, out);
            let binders: Vec<&String> = ns.iter().collect();
            let (db, fs) = shadow(&binders, dlt_binds, facts);
            collect_calls_wf(body, &db, &fs, out);
        }
        Tm::Let(n, e, body) => {
            collect_calls_wf(e, dlt_binds, facts, out);
            let (mut db, fs) = shadow(&[n], dlt_binds, facts);
            let mut rhs = &**e;
            while let Tm::Ann(inner, _) = rhs {
                rhs = inner;
            }
            if let Tm::Call(c, args) = rhs {
                if c == "dlt" && args.len() == 2 {
                    db.insert(n.clone(), (args[0].clone(), args[1].clone()));
                }
            }
            collect_calls_wf(body, &db, &fs, out);
        }
        Tm::Lam(_, b) => collect_calls_wf(b, dlt_binds, facts, out),
        Tm::Var(_) | Tm::Lit(_) | Tm::Str(_) => {}
    }
}

fn rename_ty(t: &Ty, from: &str, to: &str) -> Ty {
    match t {
        Ty::Type(l) => Ty::Type(*l),
        Ty::Var(v) => Ty::Var(if v == from { to.to_string() } else { v.clone() }),
        Ty::App(f, a) => Ty::App(Box::new(rename_ty(f, from, to)), Box::new(rename_ty(a, from, to))),
        Ty::Arrow(m, i, name, a, b) => {
            let a2 = rename_ty(a, from, to);
            let b2 = if name.as_deref() == Some(from) { (**b).clone() } else { rename_ty(b, from, to) };
            Ty::Arrow(m.clone(), *i, name.clone(), Box::new(a2), Box::new(b2))
        }
        Ty::Add(a, b) => Ty::Add(
            Box::new(rename_ty(a, from, to)),
            Box::new(rename_ty(b, from, to)),
        ),
        Ty::Lit(n) => Ty::Lit(*n),
        Ty::TypeV(v) => Ty::TypeV(v.clone()),
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
linear postulate Own : Type -> Type
postulate alloc : {0 a : Type} -> (1 x : a) -> Own a
postulate free  : {0 a : Type} -> (1 o : Own a) -> Unit
postulate unbox : {0 a : Type} -> (1 o : Own a) -> a
postulate Loc   : Type
postulate Ptr   : Loc -> Type
linear postulate PtsTo : Loc -> Type -> Type
postulate Hole  : Type -> Type
enum Cell (a : Type) { MkCell : {0 l : Loc} -> (p : Ptr l) -> (1 v : PtsTo l a) -> Cell a }
enum Taken (a : Type) (l : Loc) { MkTaken : a -> (1 v : PtsTo l (Hole a)) -> Taken a l }
postulate valloc : {0 a : Type} -> (1 x : a) -> Cell a
postulate vwrite : {0 a : Type} -> {0 b : Type} -> {0 l : Loc} -> Ptr l -> (1 v : PtsTo l a) -> b -> PtsTo l b
postulate vtake  : {0 a : Type} -> {0 l : Loc} -> Ptr l -> (1 v : PtsTo l a) -> Taken a l
postulate vread  : {0 a : Type} -> {0 l : Loc} -> Ptr l -> (1 v : PtsTo l a) -> a
postulate vfree  : {0 a : Type} -> {0 l : Loc} -> Ptr l -> (1 v : PtsTo l a) -> Unit
linear postulate Loan : Loc -> Type -> Type
enum Borrowed (a : Type) { MkBorrowed : {0 l : Loc} -> (p : Ptr l) -> (1 v : PtsTo l a) -> (1 ln : Loan l a) -> Borrowed a }
postulate borrow  : {0 a : Type} -> (1 o : Own a) -> Borrowed a
postulate restore : {0 a : Type} -> {0 l : Loc} -> Ptr l -> (1 v : PtsTo l a) -> (1 ln : Loan l a) -> Own a
linear postulate RawTo : Loc -> Type -> Type
enum RawCell (a : Type) { MkRawCell : {0 l : Loc} -> (p : Ptr l) -> (1 v : RawTo l a) -> RawCell a }
postulate ralloc : {0 a : Type} -> Unit -> RawCell a
postulate winit  : {0 a : Type} -> {0 l : Loc} -> Ptr l -> (1 v : RawTo l a) -> a -> PtsTo l a
postulate rfree  : {0 a : Type} -> {0 l : Loc} -> Ptr l -> (1 v : RawTo l a) -> Unit
linear postulate SRead : Loc -> Type -> Type
linear postulate SLoan : Loc -> Type -> Type
enum Shared (a : Type) { MkShared : {0 l : Loc} -> (p : Ptr l) -> (1 s : SRead l a) -> (1 ln : SLoan l a) -> Shared a }
enum SPair (a : Type) (l : Loc) { MkSPair : (p : Ptr l) -> (1 s1 : SRead l a) -> (1 s2 : SRead l a) -> SPair a l }
enum SGot (a : Type) (l : Loc) { MkSGot : a -> (1 s : SRead l a) -> SGot a l }
postulate share   : {0 a : Type} -> (1 o : Own a) -> Shared a
postulate sdup    : {0 a : Type} -> {0 l : Loc} -> Ptr l -> (1 s : SRead l a) -> SPair a l
postulate sjoin   : {0 a : Type} -> {0 l : Loc} -> (1 s1 : SRead l a) -> (1 s2 : SRead l a) -> SRead l a
postulate sread   : {0 a : Type} -> {0 l : Loc} -> Ptr l -> (1 s : SRead l a) -> SGot a l
postulate unshare : {0 a : Type} -> {0 l : Loc} -> Ptr l -> (1 s : SRead l a) -> (1 ln : SLoan l a) -> Own a
linear postulate JoinHandle : Type -> Type
postulate spawn : {0 e : Type} -> {0 a : Type} -> (1 work : (1 x : e) -> a) -> (1 env : e) -> JoinHandle a
postulate join  : {0 a : Type} -> (1 h : JoinHandle a) -> a
postulate Str    : Type
postulate prints : Str -> Unit
postulate Region : Type
linear postulate RegionCap : Region -> Type
enum RegionPack { MkRegionPack : {0 r : Region} -> (1 cap : RegionCap r) -> RegionPack }
postulate rnew : Unit -> RegionPack
linear postulate Pool : Region -> Type -> Type
postulate RPtr : Region -> Type
enum PAlloc (a : Type) (r : Region) { MkPAlloc : (p : RPtr r) -> (1 P : Pool r a) -> PAlloc a r }
enum PGot (a : Type) (r : Region) { MkPGot : a -> (1 P : Pool r a) -> PGot a r }
postulate pnew : {0 r : Region} -> {0 a : Type} -> (1 cap : RegionCap r) -> Pool r a
postulate palloc : {0 a : Type} -> {0 r : Region} -> (1 P : Pool r a) -> a -> PAlloc a r
postulate pget : {0 a : Type} -> {0 r : Region} -> (1 P : Pool r a) -> RPtr r -> PGot a r
postulate pset : {0 a : Type} -> {0 r : Region} -> (1 P : Pool r a) -> RPtr r -> a -> Pool r a
postulate pfree : {0 a : Type} -> {0 r : Region} -> (1 P : Pool r a) -> RPtr r -> Pool r a
postulate prelease : {0 a : Type} -> {0 r : Region} -> (1 P : Pool r a) -> Unit
"#;

/// The CONTIGUOUS-ARRAY prelude — the C `a[i]` on a flat buffer, made safe by an
/// ERASED bound. `Arr a n` lowers to ONE `malloc(n*8)` flat block (no header, no
/// per-element cell, no stored length — the length lives only in the erased
/// index). `aget`/`aset` take a machine-integer index plus a multiplicity-0
/// `Lt i n` proof (dischargeable by the stratum-(A) solver's `lt(i, n)`), so the
/// emitted code is a bare indexed load/store with NO bounds branch — out-of-
/// bounds is unrepresentable, not checked. The array itself is linear: it must
/// be freed exactly once (`afree`), reads thread it back (`ARead`), and the
/// element type is only ever used at ω positions, so a linear element type
/// cannot be smuggled in (`anew`/`aset` would duplicate it — the checker
/// rejects `ω ⋢ 1` at the call). Injected only for `%builtin Nat` programs:
/// the index domain and the `Lt` elaboration are the packed machine `Nat`.
const ARR_PRELUDE: &str = r#"
linear postulate Arr : Type -> Nat -> Type
enum ARead (a : Type) (n : Nat) { MkARead : a -> (1 arr : Arr a n) -> ARead a n }
postulate anew  : {0 a : Type} -> (n : Nat) -> a -> Arr a n
postulate aget  : {0 a : Type} -> {0 n : Nat} -> (i : Nat) -> (0 p : Lt i n) -> (1 arr : Arr a n) -> ARead a n
postulate aset  : {0 a : Type} -> {0 n : Nat} -> (i : Nat) -> (0 p : Lt i n) -> a -> (1 arr : Arr a n) -> Arr a n
postulate afree : {0 a : Type} -> {0 n : Nat} -> (1 arr : Arr a n) -> Unit
enum DecLt (i : Nat) (n : Nat) { DYes : (0 p : Lt i n) -> DecLt i n, DNo : DecLt i n }
postulate dlt : (i : Nat) -> (n : Nat) -> DecLt i n
linear postulate Slice : Loc -> Type -> Nat -> Type
enum SliceRead (l : Loc) (a : Type) (n : Nat) { MkSliceRead : a -> (1 s : Slice l a n) -> SliceRead l a n }
postulate sget : {0 a : Type} -> {0 l : Loc} -> {0 n : Nat} -> (i : Nat) -> (0 p : Lt i n) -> (1 s : Slice l a n) -> SliceRead l a n
postulate sset : {0 a : Type} -> {0 l : Loc} -> {0 n : Nat} -> (i : Nat) -> (0 p : Lt i n) -> a -> (1 s : Slice l a n) -> Slice l a n
linear postulate Rejoin : Loc -> Loc -> Type -> Nat -> Nat -> Type
enum ASplit (a : Type) (k : Nat) (m : Nat) { MkASplit : {0 llo : Loc} -> {0 lhi : Loc} -> (1 lo : Slice llo a k) -> (1 hi : Slice lhi a m) -> (1 rj : Rejoin llo lhi a k m) -> ASplit a k m }
postulate asplit : {0 a : Type} -> (k : Nat) -> (0 m : Nat) -> (1 arr : Arr a (k + m)) -> ASplit a k m
postulate ajoin  : {0 a : Type} -> {0 llo : Loc} -> {0 lhi : Loc} -> {0 k : Nat} -> {0 m : Nat} -> (1 lo : Slice llo a k) -> (1 hi : Slice lhi a m) -> (1 rj : Rejoin llo lhi a k m) -> Arr a (k + m)
"#;

/// `Nat`, auto-provided as the packed machine integer (Idris 2's default). A
/// program that declares its OWN `Nat` — or uses `Zero`/`Succ` for its own
/// constructors — opts out entirely and keeps its representation choice (a plain
/// Peano `enum Nat` for type-level `Succ`-towers / proofs, say). Only injected
/// when none of `Nat`/`Zero`/`Succ` is already a user-declared name.
const NAT_PRELUDE: &str = r#"
%builtin Nat Nat
enum Nat { Zero : Nat, Succ : Nat -> Nat }
"#;

/// The Core standard prelude — a faithful, lean slice of the Idris 2 `Prelude`
/// (`Bool`, `Maybe`, `Either`, `Ordering`, `Pair`, `List`, plus the usual
/// combinators). Auto-provided like the memory layer, ALL-OR-NOTHING: a program
/// that declares any of these names owns that layer and gets none of the block
/// (so every pre-stdlib program keeps compiling unchanged). The two `Nat`-typed
/// helpers (`length`, `compareNat`) name `Zero`/`Succ`, so the block is injected
/// only when those constructors are in scope (they always are once `Nat` is —
/// injected or user-declared with the standard shape). NOTE: the higher-order
/// members (`map`/`filter`/`foldr`/…) type-check and are correct, but running
/// them needs first-class-function codegen (see `dep_codegen`'s clear error);
/// the first-order members compile and run today.
const STD_PRELUDE: &str = r#"
enum Bool { False : Bool, True : Bool }
not : Bool -> Bool
fn not(b) { match b { False => True, True => False } }
and : Bool -> Bool -> Bool
fn and(x, y) { match x { False => False, True => y } }
or : Bool -> Bool -> Bool
fn or(x, y) { match x { False => y, True => True } }
ifThenElse : {0 a : Type} -> Bool -> a -> a -> a
fn ifThenElse(c, t, e) { match c { True => t, False => e } }

enum Ordering { LT : Ordering, EQ : Ordering, GT : Ordering }
compareNat : Nat -> Nat -> Ordering
fn compareNat(m, n) {
  match m {
    Zero    => match n { Zero => EQ, Succ(k) => LT },
    Succ(j) => match n { Zero => GT, Succ(k) => compareNat(j, k) },
  }
}

enum Maybe (a : Type) { Nothing : Maybe a, Just : a -> Maybe a }
maybe : {0 a : Type} -> {0 b : Type} -> b -> (f : a -> b) -> Maybe a -> b
fn maybe(dflt, f, m) { match m { Nothing => dflt, Just(x) => f(x) } }
fromMaybe : {0 a : Type} -> a -> Maybe a -> a
fn fromMaybe(dflt, m) { match m { Nothing => dflt, Just(x) => x } }
isJust : {0 a : Type} -> Maybe a -> Bool
fn isJust(m) { match m { Nothing => False, Just(x) => True } }
isNothing : {0 a : Type} -> Maybe a -> Bool
fn isNothing(m) { match m { Nothing => True, Just(x) => False } }
mapMaybe : {0 a : Type} -> {0 b : Type} -> (f : a -> b) -> Maybe a -> Maybe b
fn mapMaybe(f, m) { match m { Nothing => Nothing, Just(x) => Just(f(x)) } }

enum Either (a : Type) (b : Type) { Left : a -> Either a b, Right : b -> Either a b }
either : {0 a : Type} -> {0 b : Type} -> {0 c : Type} -> (f : a -> c) -> (g : b -> c) -> Either a b -> c
fn either(f, g, e) { match e { Left(x) => f(x), Right(y) => g(y) } }
mirror : {0 a : Type} -> {0 b : Type} -> Either a b -> Either b a
fn mirror(e) { match e { Left(x) => Right(x), Right(y) => Left(y) } }

enum Pair (a : Type) (b : Type) { MkPair : a -> b -> Pair a b }
fst : {0 a : Type} -> {0 b : Type} -> Pair a b -> a
fn fst(p) { match p { MkPair(x, y) => x } }
snd : {0 a : Type} -> {0 b : Type} -> Pair a b -> b
fn snd(p) { match p { MkPair(x, y) => y } }
swap : {0 a : Type} -> {0 b : Type} -> Pair a b -> Pair b a
fn swap(p) { match p { MkPair(x, y) => MkPair(y, x) } }

boxed enum List (a : Type) { Nil : List a, Cons : a -> List a -> List a }
map : {0 a : Type} -> {0 b : Type} -> (f : a -> b) -> List a -> List b
fn map(f, xs) { match xs { Nil => Nil, Cons(h, t) => Cons(f(h), map(f, t)) } }
append : {0 a : Type} -> List a -> List a -> List a
fn append(xs, ys) { match xs { Nil => ys, Cons(h, t) => Cons(h, append(t, ys)) } }
length : {0 a : Type} -> List a -> Nat
fn length(xs) { match xs { Nil => Zero, Cons(h, t) => Succ(length(t)) } }
foldr : {0 a : Type} -> {0 b : Type} -> (f : a -> b -> b) -> b -> List a -> b
fn foldr(f, z, xs) { match xs { Nil => z, Cons(h, t) => f(h, foldr(f, z, t)) } }
foldl : {0 a : Type} -> {0 b : Type} -> (f : b -> a -> b) -> b -> List a -> b
fn foldl(f, z, xs) { match xs { Nil => z, Cons(h, t) => foldl(f, f(z, h), t) } }
reverseOnto : {0 a : Type} -> List a -> List a -> List a
fn reverseOnto(acc, xs) { match xs { Nil => acc, Cons(h, t) => reverseOnto(Cons(h, acc), t) } }
reverse : {0 a : Type} -> List a -> List a
fn reverse(xs) { reverseOnto(Nil, xs) }
filter : {0 a : Type} -> (p : a -> Bool) -> List a -> List a
fn filter(p, xs) {
  match xs {
    Nil => Nil,
    Cons(h, t) => match p(h) { True => Cons(h, filter(p, t)), False => filter(p, t) },
  }
}

id : {0 a : Type} -> a -> a
fn id(x) { x }
const : {0 a : Type} -> {0 b : Type} -> a -> b -> a
fn const(x, y) { x }
flip : {0 a : Type} -> {0 b : Type} -> {0 c : Type} -> (f : a -> b -> c) -> b -> a -> c
fn flip(f, y, x) { f(x, y) }
compose : {0 a : Type} -> {0 b : Type} -> {0 c : Type} -> (g : b -> c) -> (f : a -> b) -> a -> c
fn compose(g, f, x) { g(f(x)) }
the : (0 a : Type) -> a -> a
fn the(a, x) { x }
"#;

/// Every type/constructor/function/postulate name the stdlib block introduces —
/// the collision set that all-or-nothing gates `STD_PRELUDE` (a user declaring
/// any of these owns that layer, and the block is skipped).
const STD_NAMES: &[&str] = &[
    "Bool", "False", "True", "not", "and", "or", "ifThenElse",
    "Ordering", "LT", "EQ", "GT", "compareNat",
    "Maybe", "Nothing", "Just", "maybe", "fromMaybe", "isJust", "isNothing", "mapMaybe",
    "Either", "Left", "Right", "either", "mirror",
    "Pair", "MkPair", "fst", "snd", "swap",
    "List", "Nil", "Cons", "map", "append", "length", "foldr", "foldl",
    "reverseOnto", "reverse", "filter",
    "id", "const", "flip", "compose", "the",
];

/// The primary name a top-level item declares (`None` for a `%builtin` pragma).
fn item_name(it: &Item) -> Option<&str> {
    match it {
        Item::Sig(n, _) | Item::Fn(n, _, _, _) | Item::Postulate(n, _, _) => Some(n),
        Item::Foreign(n, _, _) => Some(n),
        Item::Enum { name, .. } | Item::Struct { name, .. } => Some(name),
        Item::BuiltinNat(_) => None,
    }
}

pub fn elaborate(src: &str) -> Result<Program, String> {
    let toks = lex(src)?;
    let mut items = Parser { toks, pos: 0, fresh: 0 }.parse_program()?;

    // ---- AUTO-INJECTED STANDARD PRELUDE (Nat + Core Idris-2 Prelude) --------
    // Everything the USER declared: item names PLUS enum CONSTRUCTOR names (a
    // constructor `Zero`/`Cons`/… is as much a "taken" name as a type or fn).
    // Used to decide, all-or-nothing per layer, whether an auto-injected block
    // would collide — so any explicit user (re)declaration wins and every
    // pre-stdlib program keeps compiling byte-for-byte unchanged.
    let user_names: std::collections::HashSet<String> = items
        .iter()
        .flat_map(|it| {
            let mut ns: Vec<String> = item_name(it).map(str::to_string).into_iter().collect();
            if let Item::Enum { variants, .. } = it {
                ns.extend(variants.iter().map(|(cn, _)| cn.clone()));
            }
            ns
        })
        .collect();

    // `Nat`, packed (Idris 2's default). Skipped if the program declares its own
    // `Nat` — or uses `Zero`/`Succ` for its own constructors — so it keeps its
    // representation choice (a plain Peano `enum Nat` for `Succ`-towers / proofs).
    let nat_taken = ["Nat", "Zero", "Succ"].iter().any(|n| user_names.contains(*n))
        || items.iter().any(|it| matches!(it, Item::BuiltinNat(_)));
    if !nat_taken {
        let nat_items = Parser { toks: lex(NAT_PRELUDE)?, pos: 0, fresh: 0 }.parse_program()?;
        let mut merged = nat_items;
        merged.append(&mut items);
        items = merged;
    }

    // The Core prelude — injected only when NONE of its names collide AND the
    // Nat constructors its `length`/`compareNat` name (`Zero`/`Succ`) are in
    // scope (always true once Nat is, with the standard shape; a program with an
    // oddly-shaped `Nat` is left alone rather than broken). Spliced right AFTER
    // the `Nat` enum so its types precede any user signature that uses them.
    let std_collides = STD_NAMES.iter().any(|n| user_names.contains(*n));
    let (mut has_zero, mut has_succ) = (false, false);
    for it in &items {
        if let Item::Enum { variants, .. } = it {
            for (cn, _) in variants {
                has_zero |= cn == "Zero";
                has_succ |= cn == "Succ";
            }
        }
    }
    if !std_collides && has_zero && has_succ {
        let std_items = Parser { toks: lex(STD_PRELUDE)?, pos: 0, fresh: 0 }.parse_program()?;
        let nat_pos = items
            .iter()
            .position(|it| matches!(it, Item::Enum { name, .. } if name == "Nat"))
            .map(|i| i + 1)
            .unwrap_or(0);
        items.splice(nat_pos..nat_pos, std_items);
    }

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
    // `print : Nat -> Unit` — the one observable effect. Its type references the
    // PROGRAM'S `Nat` (an enum or `%builtin`), so it is appended AFTER the user
    // items (postulates are elaborated in source order); skipped when the program
    // declares no `Nat`, owns the memory layer (no prelude `Unit`), or declares
    // its own `print`.
    let has_nat = items.iter().any(|it| {
        item_name(it) == Some("Nat") || matches!(it, Item::BuiltinNat(n) if n == "Nat")
    });
    let has_print = items.iter().any(|it| item_name(it) == Some("print"));
    if !collides && has_nat && !has_print {
        let p2 = Parser { toks: lex("postulate print : Nat -> Unit\n")?, pos: 0, fresh: 0 }
            .parse_program()?;
        items.extend(p2);
    }
    // Char I/O, the same way: `putc n` writes one byte (n truncated to a char);
    // `getc U` reads one byte from stdin and returns `Succ(byte)` — or `Zero` at
    // EOF, so ONE ordinary `match` both tests for EOF and binds the character
    // (its predecessor). Injected per-name so a program may own either.
    // NATIVE INTEGER ARITHMETIC — the C instructions, as KERNEL-OPAQUE
    // postulates: they run as single machine ops (wrapping mod 2^64; div/mod
    // are zero-TOTAL: n/0 = 0, n%0 = n — no UB, no trap) and never reduce in
    // types. INDEX arithmetic in types stays the total, solver-decided
    // fragment (`+`, `Lt`/`le`/`lt`); a fold-defined `mul` still reduces in
    // types if you write one — these are the PARTIAL-fragment runtime ops.
    // Injected per-name, so a program defining its own `mul` keeps it.
    // a name is TAKEN if any item declares it — including as an enum
    // CONSTRUCTOR (`enum Expr { mul : … }` must keep its `mul`).
    let taken: std::collections::HashSet<String> = items
        .iter()
        .flat_map(|it| {
            let mut ns: Vec<String> =
                item_name(it).map(|n| n.to_string()).into_iter().collect();
            if let Item::Enum { variants, .. } = it {
                ns.extend(variants.iter().map(|(cn, _)| cn.clone()));
            }
            ns
        })
        .collect();
    let mut scalar_type_decls: Vec<Item> = Vec::new();
    for (nm, decl) in [
        // PRIMITIVE SCALAR TYPES (Phase Scalars, S1) — opaque `Type` constants,
        // kernel never reduces them. In S1 they are STORAGE annotations: a value
        // of a scalar type lives as an i64 in registers (Nat-compatible, so the
        // native ops work on it) but is stored at its true byte width inside an
        // `Arr` — so `Arr U8 n` is a real `malloc(n)` byte buffer, load widens
        // (zext for U-, sext for I-), store truncates. Injected per-name so a
        // program may declare its own.
        ("U8", "postulate U8 : Type\n"),
        ("U16", "postulate U16 : Type\n"),
        ("U32", "postulate U32 : Type\n"),
        ("U64", "postulate U64 : Type\n"),
        ("I8", "postulate I8 : Type\n"),
        ("I16", "postulate I16 : Type\n"),
        ("I32", "postulate I32 : Type\n"),
        ("I64", "postulate I64 : Type\n"),
        ("F32", "postulate F32 : Type\n"),
        ("F64", "postulate F64 : Type\n"),
        ("putc", "postulate putc : Nat -> Unit\n"),
        ("getc", "postulate getc : Unit -> Nat\n"),
        ("sub", "postulate sub : Nat -> Nat -> Nat\n"),
        ("mul", "postulate mul : Nat -> Nat -> Nat\n"),
        ("div", "postulate div : Nat -> Nat -> Nat\n"),
        ("mod", "postulate mod : Nat -> Nat -> Nat\n"),
        ("ltb", "postulate ltb : Nat -> Nat -> Nat\n"),
        ("leb", "postulate leb : Nat -> Nat -> Nat\n"),
        ("eqb", "postulate eqb : Nat -> Nat -> Nat\n"),
        ("band", "postulate band : Nat -> Nat -> Nat\n"),
        ("bor", "postulate bor : Nat -> Nat -> Nat\n"),
        ("bxor", "postulate bxor : Nat -> Nat -> Nat\n"),
        ("shl", "postulate shl : Nat -> Nat -> Nat\n"),
        ("shr", "postulate shr : Nat -> Nat -> Nat\n"),
        // SCALAR CONVERSIONS (S1): Nat -> T (mask to width) and T -> Nat
        // (reinterpret). The only way to make/observe a scalar VALUE in S1 —
        // arithmetic still happens in the Nat register (C's integer promotion,
        // made explicit). Codegen: trunc/extend by the width in the name.
        ("u8", "postulate u8 : Nat -> U8\n"),
        ("u16", "postulate u16 : Nat -> U16\n"),
        ("u32", "postulate u32 : Nat -> U32\n"),
        ("u64", "postulate u64 : Nat -> U64\n"),
        ("i8", "postulate i8 : Nat -> I8\n"),
        ("i16", "postulate i16 : Nat -> I16\n"),
        ("i32", "postulate i32 : Nat -> I32\n"),
        ("i64", "postulate i64 : Nat -> I64\n"),
        ("nat_u8", "postulate nat_u8 : U8 -> Nat\n"),
        ("nat_u16", "postulate nat_u16 : U16 -> Nat\n"),
        ("nat_u32", "postulate nat_u32 : U32 -> Nat\n"),
        ("nat_u64", "postulate nat_u64 : U64 -> Nat\n"),
        ("nat_i8", "postulate nat_i8 : I8 -> Nat\n"),
        ("nat_i16", "postulate nat_i16 : I16 -> Nat\n"),
        ("nat_i32", "postulate nat_i32 : I32 -> Nat\n"),
        ("nat_i64", "postulate nat_i64 : I64 -> Nat\n"),
        // FIRST-CLASS SCALAR ARITHMETIC (S2) — the C machine ops, polymorphic
        // over ANY scalar type `a` (width + signedness recovered at codegen from
        // the erased `a`). Semantics are C's, NOT Nat's: `ssub` WRAPS (two's
        // complement), not monus; `sdiv`/`smod`/`sshr`/`slt`/`sle` are
        // signedness-directed by `a` (I* signed, U* unsigned). Total edges:
        // `/0 = 0`, `%0 = x`, shift ≥ width = 0. `slt`/`sle`/`seq` return Nat
        // 0/1 (match as `Zero`/`Succ`). Using them at a non-scalar `a` (e.g.
        // `Nat`) is a guided compile error — use `+`/`sub`/`mul` for `Nat`.
        ("sadd", "postulate sadd : {0 a : Type} -> a -> a -> a\n"),
        ("ssub", "postulate ssub : {0 a : Type} -> a -> a -> a\n"),
        ("smul", "postulate smul : {0 a : Type} -> a -> a -> a\n"),
        ("sdiv", "postulate sdiv : {0 a : Type} -> a -> a -> a\n"),
        ("smod", "postulate smod : {0 a : Type} -> a -> a -> a\n"),
        ("sand", "postulate sand : {0 a : Type} -> a -> a -> a\n"),
        ("sor", "postulate sor : {0 a : Type} -> a -> a -> a\n"),
        ("sxor", "postulate sxor : {0 a : Type} -> a -> a -> a\n"),
        ("sshl", "postulate sshl : {0 a : Type} -> a -> a -> a\n"),
        ("sshr", "postulate sshr : {0 a : Type} -> a -> a -> a\n"),
        ("sneg", "postulate sneg : {0 a : Type} -> a -> a\n"),
        ("slt", "postulate slt : {0 a : Type} -> a -> a -> Nat\n"),
        ("sle", "postulate sle : {0 a : Type} -> a -> a -> Nat\n"),
        ("seq", "postulate seq : {0 a : Type} -> a -> a -> Nat\n"),
        // CASTS (S2/S4) — one polymorphic conversion between ANY two scalars
        // (int↔int width change, int↔float, float↔float). The target `b` is
        // inferred from context (an annotated `let` or a typed argument).
        ("cast", "postulate cast : {0 a : Type} -> {0 b : Type} -> a -> b\n"),
        // FLOATS (S4) — IEEE-754 `F32`/`F64`, riding the i64 register as their
        // bit pattern (decoded only at these ops). `flt`/`fle`/`feq` are ORDERED
        // comparisons → Nat 0/1. Build a float from a Nat with `f64_of_nat`
        // (`sitofp`) and read it back with `nat_of_f64` (`fptosi`); `cast`
        // covers every int↔float and float↔float conversion.
        ("fadd", "postulate fadd : {0 a : Type} -> a -> a -> a\n"),
        ("fsub", "postulate fsub : {0 a : Type} -> a -> a -> a\n"),
        ("fmul", "postulate fmul : {0 a : Type} -> a -> a -> a\n"),
        ("fdiv", "postulate fdiv : {0 a : Type} -> a -> a -> a\n"),
        ("fneg", "postulate fneg : {0 a : Type} -> a -> a\n"),
        ("flt", "postulate flt : {0 a : Type} -> a -> a -> Nat\n"),
        ("fle", "postulate fle : {0 a : Type} -> a -> a -> Nat\n"),
        ("feq", "postulate feq : {0 a : Type} -> a -> a -> Nat\n"),
        // float LITERAL desugaring targets: reinterpret a Nat's bit pattern as a
        // float (`3.14` ⇒ `f64_bits(<bits>)`). Codegen identity — floats ride the
        // i64 register as their bits.
        ("f64_bits", "postulate f64_bits : Nat -> F64\n"),
        ("f32_bits", "postulate f32_bits : Nat -> F32\n"),
        ("f64_of_nat", "postulate f64_of_nat : Nat -> F64\n"),
        ("f32_of_nat", "postulate f32_of_nat : Nat -> F32\n"),
        ("nat_of_f64", "postulate nat_of_f64 : F64 -> Nat\n"),
        ("nat_of_f32", "postulate nat_of_f32 : F32 -> Nat\n"),
    ] {
        if !collides && has_nat && !taken.contains(nm) {
            let p = Parser { toks: lex(decl)?, pos: 0, fresh: 0 }.parse_program()?;
            // The scalar TYPE declarations (`U8`…`F64` : Type) are PREPENDED — a
            // user `postulate`/`%foreign` signature may reference them (e.g.
            // `%foreign "sqrt" : F64 -> F64`), and postulate signatures are
            // registered in source order (no forward reference), so the type must
            // already be in scope. The ops/conversions stay appended (they are
            // called from fn bodies, elaborated in a later pass).
            if matches!(
                nm,
                "U8" | "U16" | "U32" | "U64" | "I8" | "I16" | "I32" | "I64" | "F32" | "F64"
            ) {
                scalar_type_decls.extend(p);
            } else {
                items.extend(p);
            }
        }
    }
    if !scalar_type_decls.is_empty() {
        let mut merged = std::mem::take(&mut scalar_type_decls);
        merged.append(&mut items);
        items = merged;
    }
    // The contiguous-array prelude (see `ARR_PRELUDE`): appended AFTER the user
    // items because its types reference the program's `Nat`, and only for
    // `%builtin Nat` programs (the `Lt` bound elaborates through the linear-Nat
    // solver, which needs the packed index domain). ALL-OR-NOTHING on its own
    // names: a program declaring any of them owns its array layer.
    let has_bnat = items
        .iter()
        .any(|it| matches!(it, Item::BuiltinNat(n) if n == "Nat"));
    const ARR_NAMES: [&str; 16] = [
        "Arr", "ARead", "anew", "aget", "aset", "afree", "DecLt", "dlt",
        "Slice", "SliceRead", "sget", "sset", "Rejoin", "ASplit", "asplit", "ajoin",
    ];
    let arr_collides = items
        .iter()
        .filter_map(item_name)
        .any(|n| ARR_NAMES.contains(&n));
    if !collides && has_bnat && !arr_collides {
        let p3 = Parser { toks: lex(ARR_PRELUDE)?, pos: 0, fresh: 0 }.parse_program()?;
        // insert right AFTER the `%builtin Nat` enum declaration (the first
        // item the prelude's types reference), NOT at the end — so USER
        // datatypes and signatures below can hold `Arr`/`Slice` fields.
        let nat_pos = items
            .iter()
            .position(|it| matches!(it, Item::Enum { name, .. } if name == "Nat"))
            .map(|i| i + 1)
            .unwrap_or(items.len());
        items.splice(nat_pos..nat_pos, p3);
    }
    // `peq : RPtr equality` references the program's `Nat`, so like `print` it
    // is APPENDED (the rest of the pool layer lives in the main prelude, which
    // user declarations may reference).
    let has_peq = items.iter().any(|it| item_name(it) == Some("peq"));
    if !collides && has_nat && !has_peq {
        let p4 = Parser {
            toks: lex("postulate peq : {0 r : Region} -> RPtr r -> RPtr r -> Nat
")?,
            pos: 0,
            fresh: 0,
        }
        .parse_program()?;
        items.extend(p4);
    }

    // MULT-POLY MONOMORPHIZATION (docs/MULT_POLY_PLAN.md slice 2) — a SURFACE
    // pre-pass: functions with an explicit `(m : Mult)` parameter are replaced
    // by per-call-site concrete instances (`f$1`, `f$w`, …), so the kernel's rig
    // stays `{0,1,ω}` and every checked instance is concrete (its usage-validity
    // is decided by the ordinary checker — no symbolic-rig reasoning, no new
    // trusted code). Behavior-neutral for programs with no `Mult` parameters.
    // LEVEL-POLY MONOMORPHIZATION (Phase B2) first — a level instance may
    // itself be mult-polymorphic; then the mult pass sees concrete levels.
    monomorphize_level_poly(&mut items)?;
    monomorphize_mult_poly(&mut items)?;

    let mut elab = Elab {
        rc: Rc::new(Signature::default()),
        data_arity: HashMap::new(),
        ctor_arity: HashMap::new(),
        ctor_info: HashMap::new(),
        defs: HashMap::new(),
        def_implicit: HashMap::new(),
        nat_types: std::collections::HashSet::new(),
        nat_ctor: HashMap::new(),
        poly_ctor_base: std::collections::HashSet::new(),
        in_fix: std::cell::Cell::new(false),
        mult_env: std::cell::RefCell::new(HashMap::new()),
        pess_linear: std::cell::RefCell::new(std::collections::HashSet::new()),
        def_linear_capable: std::cell::RefCell::new(HashMap::new()),
        hole_ctr: std::cell::Cell::new(0),
    };

    // pass A: `%builtin Nat T` pragmas. Validate each names a Nat-shaped enum (one
    // nullary constructor + one single-self-recursive constructor) and record the
    // type + its two constructors as the packed built-in `Nat`. Such a type is
    // NOT registered as a datatype — it aliases the kernel's `Nat`.
    for it in &items {
        if let Item::BuiltinNat(tyname) = it {
            let decl = items.iter().find_map(|x| match x {
                Item::Enum { name, params, index_ty, variants, .. } if name == tyname => {
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
            Item::Enum { name, params, index_ty, variants, .. } if !elab.nat_types.contains(name) => {
                let np = params.len();
                let ni = match index_ty {
                    Some(e) => count_index_pis(e)?.0,
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
            Item::Enum { name, params, index_ty, variants, linear, boxed } if !elab.nat_types.contains(name) => {
                if *linear {
                    sig.linear_types.insert(name.clone());
                }
                if *boxed {
                    sig.boxed_types.insert(name.clone());
                }
                let np = params.len();
                // family parameters are always solved (implicit) at constructor use sites
                let param_implicit: Vec<bool> = vec![true; np];
                let mut scope = Vec::new();
                let mut kparams = Vec::new();
                for b in params {
                    let ty = elab.elab_ty(&b.ty, &scope)?;
                    let pm = match &b.mult {
                        Some(sm) => sm.resolve(&elab.mult_env.borrow())?,
                        None => Mult::Zero,
                    };
                    kparams.push((pm, ty));
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
                    if let Some(base) = mono_ctor_base(&cn) {
                        elab.poly_ctor_base.insert(base.to_string());
                    }
                    ctors.push(Constructor { name: cn.clone(), args, idxs });
                }
                // The datatype's UNIVERSE is what it declares (`enum T : Type 1
                // { … }` — default `Type` = 0). `check_signature`'s predicativity
                // side-condition (every field/param level ≤ the universe, never a
                // universe quantifying over itself — the Girard guard) then checks
                // the declaration genuinely: a `Type`-storing container must say
                // `Type 1`, and saying it makes it legal.
                let universe = match index_ty {
                    Some(e) => count_index_pis(e)?.1,
                    None => 0,
                };
                sig.datas.push(DataDecl {
                    name: name.clone(),
                    universe,
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
                    let pm = match &b.mult {
                        Some(sm) => sm.resolve(&elab.mult_env.borrow())?,
                        None => Mult::Zero,
                    };
                    kparams.push((pm, ty));
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
                if let Some(base) = mono_ctor_base(name) {
                    elab.poly_ctor_base.insert(base.to_string());
                }
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
            Item::Postulate(name, pty, linear) => {
                let ty_term = elab.elab_ty(pty, &[])?;
                let (arrows, _) = peel_arrows(pty);
                let flags: Vec<bool> = arrows.iter().map(|(_, i, _, _)| *i).collect();
                elab.defs.insert(name.clone(), (Term::Const(name.clone()), ty_term.clone()));
                elab.def_implicit.insert(name.clone(), flags);
                sig.postulates.push((name.clone(), ty_term));
                if *linear {
                    sig.linear_types.insert(name.clone());
                }
            }
            // `%foreign` — an ordinary opaque postulate to the kernel (every use
            // is checked against the declared type; it can never reduce in a
            // type), plus a registry entry telling codegen which extern C symbol
            // its runtime applications call.
            Item::Foreign(name, sym, pty) => {
                let ty_term = elab.elab_ty(pty, &[])?;
                let (arrows, _) = peel_arrows(pty);
                let flags: Vec<bool> = arrows.iter().map(|(_, i, _, _)| *i).collect();
                elab.defs.insert(name.clone(), (Term::Const(name.clone()), ty_term.clone()));
                elab.def_implicit.insert(name.clone(), flags);
                sig.postulates.push((name.clone(), ty_term));
                sig.foreigns.insert(name.clone(), sym.clone());
            }
            _ => {}
        }
    }
    elab.rc = Rc::new(sig);

    // pass C½ — NESTED PATTERNS: flatten every match's pattern matrix (needs the
    // constructor table from pass C; must run before totality/elaboration, which
    // assume flat arms).
    let mut pat_fresh = 0usize;
    for it in items.iter_mut() {
        if let Item::Fn(_, _, body, _) = it {
            *body = elab.desugar_patterns(body, &mut pat_fresh)?;
        }
    }
    // pass C¾ — FORWARD REFERENCES: reorder the `fn` definitions so every
    // callee precedes its callers (definitions elaborate in sequence, and a
    // call inlines the already-elaborated callee — source order used to be a
    // hard requirement). A dependency CYCLE (mutual recursion) keeps the
    // original order and errors exactly as before, rather than silently
    // changing behavior.
    reorder_fns_by_calls(&mut items);
    let items = items; // freeze

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

    // PHASE B3 — MUTUAL RECURSION lowering by FORWARD-CALL UNROLLING: within
    // a call-graph SCC, a call to a member that appears LATER in item order
    // (a forward reference, which per-definition inlining cannot resolve) is
    // replaced by that member's body with its explicit parameters let-bound
    // to the call's arguments (scoping does the hygiene). Back-edges to the
    // fn being defined remain as calls, which the `Fix` machinery turns into
    // the self-reference. Backward calls (already-elaborated members) inline
    // as ordinary defs.
    let fn_order: HashMap<String, usize> = items
        .iter()
        .filter_map(|it| match it {
            Item::Fn(n, _, _, _) => Some(n.clone()),
            _ => None,
        })
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let fn_defs: HashMap<String, (Vec<String>, Tm)> = items
        .iter()
        .filter_map(|it| match it {
            Item::Fn(n, ps, b, _) => Some((n.clone(), (ps.clone(), b.clone()))),
            _ => None,
        })
        .collect();
    // reachability over fn names for SCC membership.
    let fn_reach: HashMap<String, std::collections::HashSet<String>> = {
        let mut edges: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
        for (n, (_, b)) in &fn_defs {
            let mut calls = Vec::new();
            collect_all_calls(b, &mut calls);
            let e = edges.entry(n.clone()).or_default();
            for c in calls {
                if fn_defs.contains_key(&c.callee) {
                    e.insert(c.callee);
                }
            }
        }
        // naive transitive closure (small graphs).
        loop {
            let mut changed = false;
            let keys: Vec<String> = edges.keys().cloned().collect();
            for k in &keys {
                let succs: Vec<String> = edges[k].iter().cloned().collect();
                for s in succs {
                    let add: Vec<String> =
                        edges.get(&s).map(|x| x.iter().cloned().collect()).unwrap_or_default();
                    let e = edges.get_mut(k).unwrap();
                    for a in add {
                        if e.insert(a) {
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        edges
    };
    let scc_of = |f: &str| -> std::collections::HashSet<String> {
        fn_reach
            .get(f)
            .map(|r| {
                r.iter()
                    .filter(|g| {
                        g.as_str() != f
                            && fn_reach.get(g.as_str()).is_some_and(|rg| rg.contains(f))
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    };

    let mut out_defs = Vec::new();
    let mut totality_status: Vec<(String, bool, Option<String>)> = Vec::new();
    // fns whose recursion was certified by a kernel-checked CONVOY FOLD after the
    // analyzer (conservatively) declined them.
    let mut promoted: std::collections::HashSet<String> = std::collections::HashSet::new();
    for it in &items {
        if let Item::Fn(name, params, body, annot) = it {
            // unroll forward SCC calls in this body (no-op outside a cycle).
            let scc = scc_of(name);
            let unrolled_body;
            let body: &Tm = if scc.is_empty() {
                body
            } else {
                let mut path = vec![name.clone()];
                unrolled_body =
                    unroll_forward_scc_calls(name, body, &scc, &fn_order, &fn_defs, &mut path)?;
                &unrolled_body
            };
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
            // `%partial` is a LOWERING DIRECTIVE as well as documentation: it forces
            // the general-recursion (`Fix`) lowering even when the recursion is
            // structurally total. The observable difference is EFFECT ORDER: a total
            // fold is a call-by-value ELIMINATOR (the recursive result is computed
            // before the step body runs — `print` in a fold emits bottom-up), while a
            // `Fix` is direct recursion (source order). Effectful recursion should be
            // `%partial` (FUTURE_WORK §8: effects live in the partial fragment).
            let structural = if *annot == Some(TotAnnot::Partial) {
                Totality::Partial("marked `%partial`".into())
            } else {
                structural
            };
            // CONVOY-FOLD PROMOTION: a structurally-descending recursion whose
            // varying arguments are the scrutinee's INDEX-DEPENDENT values was
            // declined only for want of a lowering — try the dependent-motive
            // eliminator; a kernel-checked success IS the totality certificate.
            // Never attempted under `%partial` (that forces `Fix`).
            let promoted_term = if !structural.is_total() && *annot != Some(TotAnnot::Partial) {
                elab.try_convoy_fold(name, params, body, &ty_term, &full_names, &full_tys, &ret)
            } else {
                None
            };
            if promoted_term.is_some() {
                promoted.insert(name.clone());
            }
            if *annot == Some(TotAnnot::Total) && promoted_term.is_none() {
                if let Some(reason) = full.reason() {
                    // a caller of a PROMOTED fn: its analyzer verdict predates the
                    // promotion — treat a direct calls-X reason as satisfied.
                    let excused = reason
                        .strip_prefix("calls `")
                        .and_then(|r| r.split('`').next())
                        .is_some_and(|callee| promoted.contains(callee));
                    if !excused {
                        return Err(format!("`%total fn {name}` is not total: {reason}"));
                    }
                }
            }
            if let Some(t) = &promoted_term {
                totality_status.push((name.clone(), true, None));
                elab.defs.insert(name.clone(), (t.clone(), ty_term.clone()));
                out_defs.push((name.clone(), ty_term, t.clone()));
                continue;
            }
            totality_status.push((
                name.clone(),
                full.is_total(),
                full.reason().map(|s| s.to_string()),
            ));

            let term = elab.elab_fn_term(
                name,
                params,
                body,
                &ty_term,
                &full_names,
                &full_tys,
                &ret,
                matches!(structural, Totality::Total),
            )?;

            // LINEAR-CAPABILITY inference (see `def_linear_capable`): for each
            // implicit Type-parameter, re-elaborate + kernel-check the body with
            // that parameter assumed LINEAR. Success ⇒ the body never drops or
            // duplicates values of it ⇒ instantiating it at a linear type is
            // sound. (The pessimistic term is discarded — only pass/fail counts;
            // the definition the program uses is the ordinary one above.)
            let mut capable = vec![true; full_names.len()];
            for i in 0..full_names.len() {
                if full_imps[i] && matches!(full_tys[i], Ty::Type(_)) {
                    elab.pess_linear.borrow_mut().insert(full_names[i].clone());
                    let ok = elab
                        .elab_fn_term(
                            name,
                            params,
                            body,
                            &ty_term,
                            &full_names,
                            &full_tys,
                            &ret,
                            matches!(structural, Totality::Total),
                        )
                        .and_then(|t| {
                            dep::check_closed_in((*elab.rc).clone(), &t, &ty_term)
                                .map_err(|e| e)
                        })
                        .map_err(|e| {
                            if std::env::var("TALLY_DEBUG_CAP").is_ok() {
                                eprintln!("[cap] {name} not capable in {}: {e}", full_names[i]);
                            }
                            e
                        })
                        .is_ok();
                    elab.pess_linear.borrow_mut().clear();
                    capable[i] = ok;
                }
            }
            elab.def_linear_capable.borrow_mut().insert(name.clone(), capable);

            elab.defs.insert(name.clone(), (term.clone(), ty_term.clone()));
            out_defs.push((name.clone(), ty_term, term));
        }
    }

    // the analyzer's `full` verdicts predate any convoy-fold promotions: flip
    // the now-stale "calls `X`, which is not total" reasons (to a fixpoint, so
    // chains of callers promote too).
    loop {
        let total_now: std::collections::HashSet<String> = totality_status
            .iter()
            .filter(|(_, t, _)| *t)
            .map(|(n, _, _)| n.clone())
            .collect();
        let mut changed = false;
        for entry in totality_status.iter_mut() {
            if entry.1 {
                continue;
            }
            let stale = entry.2.as_deref().is_some_and(|r| {
                r.strip_prefix("calls `")
                    .and_then(|x| x.split('`').next())
                    .is_some_and(|callee| total_now.contains(callee))
            });
            if stale {
                entry.1 = true;
                entry.2 = None;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    Ok(Program { sig: (*elab.rc).clone(), defs: out_defs, totality: totality_status })
}

impl Elab {
    /// Attempt the CONVOY-FOLD lowering for a structurally-descending recursion
    /// whose non-scrutinee arguments vary (declined by the verdict as "not yet
    /// lowerable"): if the varying arguments are exactly the scrutinee's
    /// index-dependent deps, the dependent motive abstracts them and the
    /// recursion is an ordinary eliminator — which the KERNEL RE-CHECKS here,
    /// making success a genuine totality certificate. Any miss returns `None`
    /// (the caller falls back to the honest `Fix` path unchanged).
    fn try_convoy_fold(
        &self,
        name: &str,
        params: &[String],
        body: &Tm,
        ty_term: &Term,
        full_names: &[String],
        full_tys: &[Ty],
        ret: &Ty,
    ) -> Option<Term> {
        let Tm::Match(scrut, arms) = body else { return None };
        if arms.is_empty() {
            return None;
        }
        let mut fn_cx = Cx::default();
        for (i, (pn, pty)) in full_names.iter().zip(full_tys).enumerate() {
            let kty = self.elab_ty(pty, &full_names[..i]).ok()?;
            let v = dep::eval_rc(&self.rc, &neutral_env(i), &kty);
            fn_cx.push(pn.clone(), v);
        }
        let sidx = fn_cx.debruijn(scrut)?;
        let Some(Value::VData(d, dargs)) = fn_cx.var_type(scrut) else { return None };
        let decl = self.rc.data(&d)?.clone();
        let ret_tm = self.elab_ty(ret, full_names).ok()?;
        let expected = dep::eval_rc(&self.rc, &neutral_env(fn_cx.len()), &ret_tm);
        let exp_tm = dep::quote_at(fn_cx.len(), &expected);
        let cv = self.detect_convoy(&Term::Var(sidx), &decl, &dargs, &exp_tm, arms, &fn_cx)?;
        if cv.deps.is_empty() {
            return None;
        }
        let mut dep_args = Vec::new();
        for dp in &cv.deps {
            let pname = fn_cx.names.get(dp.lvl)?;
            dep_args.push(params.iter().position(|p| p == pname)?);
        }
        let fold = FnFold {
            fnname: name.to_string(),
            scrut_pos: params.iter().position(|p| p == scrut)?,
            dep_args,
            explicit_params: params.to_vec(),
        };
        let body_t = self
            .elab_nested_match(&Term::Var(sidx), &d, &dargs, arms, &expected, &fn_cx, None, Some(&fold))
            .ok()?;
        let mut term = body_t;
        for _ in 0..full_names.len() {
            term = Term::Lam(Box::new(term));
        }
        // the un-fakeable gate: only a kernel-checked fold is a certificate.
        dep::check_closed_in((*self.rc).clone(), &term, ty_term).ok()?;
        Some(term)
    }

    /// Elaborate one `fn`'s body to its final definition term (the pass-D
    /// per-function work, factored out so the linear-capability check can run it
    /// a second time pessimistically). Inserts nothing; the caller registers the
    /// def.
    #[allow(clippy::too_many_arguments)]
    fn elab_fn_term(
        &self,
        name: &str,
        params: &[String],
        body: &Tm,
        ty_term: &Term,
        full_names: &[String],
        full_tys: &[Ty],
        ret: &Ty,
        structural_total: bool,
    ) -> Result<Term, String> {
        // the fn's typing context: each parameter's name + kernel type
        let mut fn_cx = Cx::default();
        for (i, (pn, pty)) in full_names.iter().zip(full_tys).enumerate() {
            let kty = self.elab_ty(pty, &full_names[..i])?;
            let v = dep::eval_rc(&self.rc, &neutral_env(i), &kty);
            fn_cx.push(pn.clone(), v);
        }

        // PARTIAL ⇒ general recursion via an opaque `Fix` (the kernel never
        // unfolds it, so it cannot reduce in a type — the partial/total
        // boundary). Currently only `%builtin Nat` case-splits have a `Fix`
        // lowering; a partial recursion on any other shape is an honest hard
        // error (NOT silently accepted), pending the Phase E2 machinery.
        if !structural_total {
            // A `%builtin Nat`-scrutinee match keeps the proven `NatCase` `Fix`
            // lowering. ANY other partial recursion — on a boxed/heap structure, or a
            // body that is not a direct param-match (e.g. `match unbox(e)`) — lowers
            // via the GENERAL `Case`-based `Fix` (`in_fix` mode): heap recursion RUNS
            // as `%partial`. (`%partial` relaxes TERMINATION, not LINEARITY: the `Fix`
            // body is still fully linearity-checked by the kernel; and the kernel
            // treats `Fix` OPAQUELY, so no `%partial` term reduces during type-checking.)
            let nat_scrut_match = matches!(body, Tm::Match(scrut, _)
                if full_names.iter().position(|p| p == scrut).is_some_and(|sp| matches!(
                    flatten_ty(&full_tys[sp]).0, Ty::Var(n) if self.nat_types.contains(n))));
            let term = if let (true, Tm::Match(scrut, arms)) = (nat_scrut_match, body) {
                self.elab_fix_nat(name, ty_term, full_names, full_tys, ret, scrut, arms)?
            } else {
                self.elab_fix(name, ty_term, full_names, full_tys, ret, body)?
            };
            return Ok(term);
        }

        // TOTAL ⇒ lower to a kernel eliminator (or a non-recursive body).
        let body_inner = match body {
            Tm::Match(scrut, arms) => {
                // CONVOY at the fn level: when the scrutinee's index refines
                // context dependents / refutes arms, route through the
                // nested-match elaborator (which builds the dependent motive)
                // instead of `elab_match_body`'s constant one. v1: only for a
                // body with NO self-call (a recursive convoy body would need
                // dep-applied induction hypotheses — not yet built; such fns
                // are `%partial` and take the `Fix` path above).
                let no_self_call = {
                    let mut calls = Vec::new();
                    collect_all_calls(body, &mut calls);
                    calls.iter().all(|c| c.callee != *name)
                };
                // Route EVERY non-self-recursive match body (with arms — the
                // zero-arm absurd discharge stays on `elab_match_body`'s
                // `try_absurd_match`) through the nested-match elaborator: it
                // builds the CONVOY motive when the index refines dependents,
                // and lowers to a `Case` (no eager induction hypotheses — which
                // for a linear recursive field would re-traverse a resource the
                // body consumes directly, e.g. a nested-pattern inner match).
                let route_nested = no_self_call
                    && !arms.is_empty()
                    && fn_cx.debruijn(scrut).is_some()
                    // a CONCRETE literal index (e.g. `Fin Zero`) keeps the
                    // `elab_match_body` path — its absurd-discharge machinery
                    // (`try_absurd_match`) owns index-refuted matches there.
                    && !matches!(
                        fn_cx.var_type(scrut),
                        Some(Value::VData(ref d, ref a))
                            if self.rc.data(d).is_some_and(|decl| decl.indices.len() == 1)
                                && matches!(a.last(), Some(Value::VNatLit(_))))
                ;
                if route_nested {
                    let expected = dep::eval_rc(&self.rc, &neutral_env(full_names.len()), &self.elab_ty(ret, full_names)?);
                    self.check(body, &expected, &fn_cx, None)?
                } else {
                    self.elab_match_body(name, &fn_cx, params, full_tys, ret, scrut, arms)?
                }
            }
            other => {
                let expected = dep::eval_rc(&self.rc, &neutral_env(full_names.len()), &self.elab_ty(ret, full_names)?);
                self.check(other, &expected, &fn_cx, None)?
            }
        };
        let mut term = body_inner;
        for _ in 0..full_names.len() {
            term = Term::Lam(Box::new(term));
        }
        Ok(term)
    }
}

/// The COPYING-CONTAINER gate: `Arr` and `Pool` hand copies of their elements
/// out (`aget`/`pget`) while the container keeps them — sound only for
/// unrestricted elements. The ω element parameters reject linear VARIABLES,
/// but an ANONYMOUS linear value (`anew(3, alloc(0))`) binds nothing and slips
/// the usage accounting — so the element TYPE itself is checked here, on the
/// fully-elaborated terms (where solved implicits are real spine arguments).
/// Generic (`Var`-typed) elements are covered elsewhere: non-`%partial`
/// generic fns inline with substituted types (checked here in the caller),
/// and `%partial` ones hit the abstract-layout error in codegen.
const COPYING_CONTAINER_OPS: [&str; 14] = [
    "anew", "aget", "aset", "afree", "pnew", "palloc", "pget", "pset", "pfree", "prelease",
    "sget", "sset", "asplit", "ajoin",
];

/// The DROPPING-DESTRUCTOR gate (Phase A3): `free` and `vfree` dispose of a
/// cell WITHOUT running any destructor on its contents — sound only for
/// unrestricted payloads. At a LINEAR payload type the inner resource would
/// be silently leaked (the Phase 0/A3 audit found `free(alloc(alloc(0)))`
/// compiled). The discipline: consume the payload FIRST — `unbox`/`vread`
/// return it (and free the cell), after which the inner token is yours to
/// consume — so no expressiveness is lost.
const DROPPING_OPS: [&str; 2] = ["free", "vfree"];

/// The COPYING-READ gate (Phase A2): `sread` hands out a COPY of the payload
/// while the cell keeps it — sound only for unrestricted payloads (a linear
/// payload would be duplicated). `share` is gated too, so the mistake is
/// reported where the sharing starts, not at the first read.
const SHARING_OPS: [&str; 2] = ["share", "sread"];

fn validate_copying_containers(sig: &Rc<Signature>, t: &Term) -> Result<(), String> {
    // the element-type argument is always the FIRST argument of the op:
    // `App(Const(op), elem_ty)` at the bottom of the spine.
    if let Term::App(f, a) = t {
        if let Term::Const(c) = strip_walk(f) {
            if COPYING_CONTAINER_OPS.contains(&c.as_str()) && type_is_linear(a, sig) {
                return Err(format!(
                    "`{c}`: the element type here is LINEAR — a copying container \
                     (`Arr`, `Pool`) hands copies of its elements out while keeping \
                     them, which would duplicate the resource (double-free). Store \
                     linear values in a concrete datatype, an `Own`, or a `Cell` \
                     instead."
                ));
            }
            if DROPPING_OPS.contains(&c.as_str()) && type_is_linear(a, sig) {
                return Err(format!(
                    "`{c}`: the payload type here is LINEAR — dropping the cell \
                     would silently LEAK the resource inside it. Consume the \
                     payload first: `unbox`/`vread` return it (and free the cell), \
                     then dispose of the inner resource explicitly."
                ));
            }
            if SHARING_OPS.contains(&c.as_str()) && type_is_linear(a, sig) {
                return Err(format!(
                    "`{c}`: the payload type here is LINEAR — a shared read-only \
                     borrow hands COPIES of the payload out (`sread`), which would \
                     duplicate the resource (double-free). Share unrestricted data \
                     only; keep linear payloads behind the unique `borrow`."
                ));
            }
        }
    }
    walk_children(t, &mut |child| validate_copying_containers(sig, child))
}

fn strip_walk(t: &Term) -> &Term {
    match t {
        Term::Ann(e, _) => strip_walk(e),
        other => other,
    }
}

/// Apply `f` to every immediate child term (full coverage of the Term shape).
fn walk_children(
    t: &Term,
    f: &mut dyn FnMut(&Term) -> Result<(), String>,
) -> Result<(), String> {
    match t {
        Term::Var(_)
        | Term::Type(_)
        | Term::Nat
        | Term::NatLit(_)
        | Term::Zero
        | Term::Const(_)
        | Term::StrLit(_) => Ok(()),
        Term::Pi(_, a, b) | Term::Sigma(_, a, b) | Term::App(a, b) | Term::Add(a, b)
        | Term::Pair(a, b) => {
            f(a)?;
            f(b)
        }
        Term::Lam(a) | Term::Suc(a) | Term::Fst(a) | Term::Snd(a) | Term::Refl(a) => f(a),
        Term::Ann(a, b) | Term::Fix(b, a) => {
            f(a)?;
            f(b)
        }
        Term::Let(_, ty, e, body) => {
            f(ty)?;
            f(e)?;
            f(body)
        }
        Term::Eq(a, b, c) | Term::J(a, b, c) => {
            f(a)?;
            f(b)?;
            f(c)
        }
        Term::NatElim(a, b, c, d) | Term::NatCase(a, b, c, d) => {
            f(a)?;
            f(b)?;
            f(c)?;
            f(d)
        }
        Term::Data(_, args) | Term::Constr(_, args) => {
            for a in args {
                f(a)?;
            }
            Ok(())
        }
        Term::Case(_, m, methods, sc) | Term::Elim(_, m, methods, sc) => {
            f(m)?;
            for x in methods {
                f(x)?;
            }
            f(sc)
        }
    }
}

pub fn check_program(src: &str) -> Result<Program, Vec<String>> {
    let prog = elaborate(src).map_err(|e| vec![e])?;
    let mut diags = Vec::new();
    if let Err(e) = dep::check_signature(&prog.sig) {
        diags.push(format!("signature: {e}"));
    }
    // ZERO NON-EXPLICIT ALLOCATION: every datatype must have a finite VALUE
    // layout unless it opts into the heap-cell representation (`boxed enum`) —
    // a recursive family without indirection is rejected at declaration.
    if let Err(e) = crate::dep_codegen::validate_layouts(&prog.sig) {
        diags.push(format!("layout: {e}"));
    }
    // the copying-container gate (linear element types) — see above.
    let rc = Rc::new(prog.sig.clone());
    for (name, _, body) in &prog.defs {
        if let Err(e) = validate_copying_containers(&rc, body) {
            diags.push(format!("fn {name}: {e}"));
        }
    }

    // PARTIAL EVALUATION pass — runs BEFORE the usage check, so specialising a
    // recursive function applied to a static tree (an interpreter applied to a
    // fixed program) removes it from every caller's body before that body is
    // usage-checked. UNTRUSTED and SELF-VERIFYING: each rewritten body is
    // re-checked (against its type, or — for a template — the erased-params-
    // bumped type) and reverted on failure. So PE can only make a well-typed
    // program faster, never break it.
    let mut prog = prog;
    let rc = Rc::new(prog.sig.clone());
    for (name, ty, body) in prog.defs.iter_mut() {
        let ped = dep::pe_reduce_body(&rc, body);
        if ped == *body {
            continue;
        }
        let ok = dep::check_closed_in((*rc).clone(), &ped, ty).is_ok()
            || (dep::has_leading_zero_pi(ty)
                && dep::check_closed_in(
                    (*rc).clone(),
                    &dep::bump_template_body(&ped),
                    &dep::bump_leading_zero_pis(ty),
                )
                .is_ok());
        if ok {
            if std::env::var("TALLY_PE_LOG").is_ok() {
                eprintln!("[pe] specialised `{name}`");
            }
            *body = ped;
        }
    }

    // Type/usage-check every (possibly specialised) def. A function that FAILS
    // ONLY because it matches on an ERASED (`0`) parameter is not an error — it
    // is a PARTIAL-EVALUATION TEMPLATE (`0` on the argument IS the binding-time
    // annotation): it checks exactly when its static parameters are available,
    // and PE specialises every *use* of it away. So an interpreter over an
    // erased program type-checks as a template, and any call that PE could NOT
    // specialise is correctly rejected here (the erased `match` survives) — you
    // cannot run an interpreter over a program that does not exist at runtime.
    for (name, ty, body) in &prog.defs {
        if let Err(e) = dep::check_closed_in(prog.sig.clone(), body, ty) {
            if dep::has_leading_zero_pi(ty)
                && dep::check_closed_in(
                    prog.sig.clone(),
                    &dep::bump_template_body(body),
                    &dep::bump_leading_zero_pis(ty),
                )
                .is_ok()
            {
                continue; // a static template — legal (PE specialises every use)
            }
            diags.push(format!("fn {name}: {e}"));
        }
    }
    if !diags.is_empty() {
        return Err(diags);
    }

    Ok(prog)
}

impl Program {
    pub fn normalize(&self, name: &str) -> Option<Term> {
        let (_, _, body) = self.defs.iter().find(|(n, _, _)| n == name)?;
        Some(dep::normalize_closed_in(self.sig.clone(), body))
    }
}

/// Pretty-print a kernel `Term` (used to display the normal form of `main`).
/// The order proposition `a ≤ b` (`strict=false`) or `a < b` (`strict=true`) as a
/// kernel type, encoded existentially over the difference — NO new kernel
/// primitive, just `Σ` + the identity type whose equation the linear-Nat solver
/// decides (`src/solver.rs`, `docs/03` stratum A):
/// ```text
///   Le a b := Σ (0 d : Nat). Eq Nat (a + d)      b
///   Lt a b := Σ (0 d : Nat). Eq Nat (Succ a + d) b
/// ```
/// `a`/`b` live in the current context; under the fresh `Σ`-binder `d` (de Bruijn
/// 0) they are shifted up by one.
fn mk_ineq_type(a: &Term, b: &Term, strict: bool) -> Term {
    let a1 = dep::shift_term(1, a);
    let b1 = dep::shift_term(1, b);
    let lhs = if strict { Term::Suc(Box::new(a1)) } else { a1 };
    let eqn = Term::Eq(
        Box::new(Term::Nat),
        Box::new(Term::Add(Box::new(lhs), Box::new(Term::Var(0)))),
        Box::new(b1),
    );
    Term::Sigma(Mult::Zero, Box::new(Term::Nat), Box::new(eqn))
}

/// EXPLICIT, proof-producing discharge of `a ≤ b` / `a < b`: the stratum-(A)
/// solver decides the bound and, if it holds, emits the kernel proof `(d, refl)`
/// which the kernel re-checks INDEPENDENTLY (LCF discipline — a solver bug fails
/// to compile, it can never mint an unsound proof). If the bound does not hold it
/// is a hard error, never a silent stub. The result is annotated with its
/// `Le`/`Lt` type so it also type-checks in inference position (a bare pair
/// cannot be inferred). Requires `%builtin Nat` (index `+`).
fn mk_ineq_proof(a: &Term, b: &Term, strict: bool) -> Result<Term, String> {
    let lhs = if strict { Term::Suc(Box::new(a.clone())) } else { a.clone() };
    let d = crate::solver::diff_witness(b, &lhs).ok_or_else(|| {
        format!(
            "`{}`: cannot prove  {}  {}  {}  — it does not hold for all Nat",
            if strict { "lt" } else { "le" },
            pretty(a),
            if strict { "<" } else { "<=" },
            pretty(b),
        )
    })?;
    // (d, refl(lhs + d)): the solver chose d so that `lhs + d ≡ b`, so the kernel's
    // refl re-check (via `conv`/`canon`) succeeds.
    let eq = Term::Refl(Box::new(Term::Add(Box::new(lhs), Box::new(d.clone()))));
    let pair = Term::Pair(Box::new(d), Box::new(eq));
    Ok(Term::Ann(Box::new(pair), Box::new(mk_ineq_type(a, b, strict))))
}

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
