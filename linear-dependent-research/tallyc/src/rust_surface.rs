//! The v1.0 SURFACE language (`docs/10-surface-syntax.md`): ML-flavored type
//! signatures, Rust-flavored terms. Elaborates to the `dep.rs` kernel, which
//! does all type-checking.
//!
//! ```text
//! enum Nat { Zero : Nat, Succ : Nat -> Nat }
//!
//! add : Nat -> Nat -> Nat
//! fn add(m, n) {
//!     match m {
//!         Zero    => n,
//!         Succ(k) => Succ(add(k, n)),   // recursive call ↦ the induction hypothesis
//!     }
//! }
//! ```
//!
//! This is Phase 1 of the roadmap: the front end, `enum`/`struct`/signature/`fn`,
//! and `match` → the dependent eliminator (motive inferred from the return type;
//! structural self-recursion translated to the IH; coverage checked). Implicit
//! `{..}` arguments and full index-unification are Phase 2 — here every argument
//! is explicit. Types use ML juxtaposition (`Vec a n`); terms use Rust calls
//! (`Cons(a, k, h, t)`).

use crate::dep::{self, Constructor, DataDecl, Signature, Term};
use crate::mult::Mult;
use std::collections::HashMap;

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
    Colon,
    Eq,
    FatArrow, // =>
    Arrow,    // ->
    KwFn,
    KwEnum,
    KwStruct,
    KwMatch,
    KwLet,
    KwType,
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
                _ => Tok::Ident(w.to_string()),
            });
        } else {
            let t = match c {
                '(' => Tok::LParen,
                ')' => Tok::RParen,
                '{' => Tok::LBrace,
                '}' => Tok::RBrace,
                ',' => Tok::Comma,
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

/// Type expressions (ML-style: juxtaposition application, arrows, binders).
#[derive(Clone, Debug)]
enum Ty {
    Var(String),
    Type,
    App(Box<Ty>, Box<Ty>),
    Arrow(Mult, Option<String>, Box<Ty>, Box<Ty>),
}

/// Term expressions (Rust-style: calls, match, let).
#[derive(Clone, Debug)]
enum Tm {
    Var(String),
    Call(String, Vec<Tm>), // f(a, b, ...)  — head is always a name in Phase 1
    Match(String, Vec<Arm>),
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
            Some(Tok::Ident(_)) => {
                let name = self.ident()?;
                self.eat(&Tok::Colon)?;
                let ty = self.parse_ty()?;
                Ok(Item::Sig(name, ty))
            }
            other => Err(format!("expected an item, found {other:?}")),
        }
    }

    // ---- types (ML) ----

    fn looks_like_binder(&self) -> bool {
        if self.peek() != Some(&Tok::LParen) {
            return false;
        }
        let k = self.pos + 1;
        if (matches!(self.toks.get(k), Some(Tok::Num(0)) | Some(Tok::Num(1)))
            || matches!(self.toks.get(k), Some(Tok::Ident(w)) if w == "w"))
            && matches!(self.toks.get(k + 1), Some(Tok::Ident(_)))
            && self.toks.get(k + 2) == Some(&Tok::Colon)
        {
            return true;
        }
        matches!(self.toks.get(k), Some(Tok::Ident(_)))
            && self.toks.get(k + 1) == Some(&Tok::Colon)
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

    fn parse_binder(&mut self) -> Result<Binder, String> {
        self.eat(&Tok::LParen)?;
        let mult = self.parse_mult();
        let name = self.ident()?;
        self.eat(&Tok::Colon)?;
        let ty = self.parse_ty()?;
        self.eat(&Tok::RParen)?;
        Ok(Binder { mult, name, ty })
    }

    fn parse_ty(&mut self) -> Result<Ty, String> {
        if self.looks_like_binder() {
            let mut binders = Vec::new();
            while self.looks_like_binder() {
                binders.push(self.parse_binder()?);
            }
            self.eat(&Tok::Arrow)?;
            let body = self.parse_ty()?;
            let mut out = body;
            for b in binders.into_iter().rev() {
                out = Ty::Arrow(b.mult.unwrap_or(Mult::Omega), Some(b.name), Box::new(b.ty), Box::new(out));
            }
            Ok(out)
        } else {
            let lhs = self.parse_ty_app()?;
            if self.peek() == Some(&Tok::Arrow) {
                self.next();
                let rhs = self.parse_ty()?;
                Ok(Ty::Arrow(Mult::Omega, None, Box::new(lhs), Box::new(rhs)))
            } else {
                Ok(lhs)
            }
        }
    }

    fn parse_ty_app(&mut self) -> Result<Ty, String> {
        let mut e = self.parse_ty_atom()?;
        while matches!(self.peek(), Some(Tok::Ident(_)) | Some(Tok::LParen) | Some(Tok::KwType)) {
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

    // ---- declarations ----

    fn parse_enum(&mut self) -> Result<Item, String> {
        self.eat(&Tok::KwEnum)?;
        let name = self.ident()?;
        let mut params = Vec::new();
        while self.looks_like_binder() {
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
        while self.looks_like_binder() {
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

    // ---- terms (Rust) ----

    fn parse_tm(&mut self) -> Result<Tm, String> {
        match self.peek() {
            Some(Tok::KwMatch) => self.parse_match(),
            Some(Tok::KwLet) => Err("`let` is not yet supported in Phase 1 bodies".into()),
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
    pub defs: Vec<(String, Term, Term)>, // (name, type, body)
}

/// per-`fn` recursion context, present only inside the eliminator translation.
struct Rec<'a> {
    fnname: &'a str,
    scrut_pos: usize,
    /// recursive-field name → the IH variable name bound for it
    fields: &'a HashMap<String, String>,
}

struct Elab {
    data_arity: HashMap<String, usize>,
    ctor_arity: HashMap<String, usize>,
    defs: HashMap<String, (Term, Term)>,
    sig: Signature,
    fresh: usize,
}

impl Elab {
    fn gensym(&mut self, base: &str) -> String {
        self.fresh += 1;
        format!("${base}{}", self.fresh)
    }

    fn debruijn(scope: &[String], name: &str) -> Option<usize> {
        scope.iter().rev().position(|s| s == name)
    }

    /// Resolve a name applied to (already-elaborated) arguments. `prefer_ctor`
    /// disambiguates a name that is both a datatype and a constructor (a
    /// `struct`): the constructor wins in term position, the datatype in type
    /// position.
    fn resolve(&self, name: &str, args: Vec<Term>, scope: &[String], prefer_ctor: bool) -> Result<Term, String> {
        match name {
            "Eq" if args.len() == 3 => {
                let mut a = args.into_iter();
                return Ok(Term::Eq(
                    Box::new(a.next().unwrap()),
                    Box::new(a.next().unwrap()),
                    Box::new(a.next().unwrap()),
                ));
            }
            "refl" if args.len() == 1 => {
                return Ok(Term::Refl(Box::new(args.into_iter().next().unwrap())));
            }
            _ => {}
        }
        if let Some(i) = Self::debruijn(scope, name) {
            return Ok(args.into_iter().fold(Term::Var(i), |f, a| Term::App(Box::new(f), Box::new(a))));
        }
        let as_data = |args: Vec<Term>| -> Result<Term, String> {
            let ar = self.data_arity[name];
            if args.len() != ar {
                return Err(format!("datatype `{name}` expects {ar} argument(s), got {}", args.len()));
            }
            Ok(Term::Data(name.to_string(), args))
        };
        let as_ctor = |args: Vec<Term>| -> Result<Term, String> {
            let ar = self.ctor_arity[name];
            if args.len() != ar {
                return Err(format!("constructor `{name}` expects {ar} argument(s), got {}", args.len()));
            }
            Ok(Term::Constr(name.to_string(), args))
        };
        let is_data = self.data_arity.contains_key(name);
        let is_ctor = self.ctor_arity.contains_key(name);
        if is_data && is_ctor {
            return if prefer_ctor { as_ctor(args) } else { as_data(args) };
        }
        if is_data {
            return as_data(args);
        }
        if is_ctor {
            return as_ctor(args);
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
            Ty::Arrow(m, name, a, b) => {
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

    fn elab_tm(&mut self, t: &Tm, scope: &[String], rec: Option<&Rec>) -> Result<Term, String> {
        match t {
            Tm::Var(name) => self.resolve(name, vec![], scope, true),
            Tm::Call(name, args) => {
                // a structural recursive call ↦ the induction hypothesis
                if let Some(r) = rec {
                    if name == r.fnname {
                        if args.len() <= r.scrut_pos {
                            return Err(format!("recursive call to `{name}` has too few arguments"));
                        }
                        if let Tm::Var(v) = &args[r.scrut_pos] {
                            if let Some(ih) = r.fields.get(v) {
                                // the IH already stands for `fnname (… this field …)`
                                return self.resolve(ih, vec![], scope, false);
                            }
                        }
                        return Err(format!(
                            "non-structural recursion: `{name}` must recurse on a sub-component \
                             of the matched argument (Phase 1)"
                        ));
                    }
                }
                let eargs = args
                    .iter()
                    .map(|a| self.elab_tm(a, scope, rec))
                    .collect::<Result<Vec<_>, _>>()?;
                self.resolve(name, eargs, scope, true)
            }
            Tm::Match(_, _) => Err("nested `match` is not supported in Phase 1".into()),
        }
    }
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

/// Decompose an elaborated constructor type `Π(args). D params idxs`.
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
                    return Err(format!(
                        "a constructor of `{data}` must return it at its parameters (param {i})"
                    ));
                }
            }
            Ok((args, spine[nparams..].to_vec()))
        }
        other => Err(format!("a constructor of `{data}` must return `{data} …`, found {other:?}")),
    }
}

// ---- the eliminator translation for a recursive `fn … { match p { … } }` ----

impl Elab {
    /// Elaborate a `fn` body that is a top-level `match` on a parameter `scrut`,
    /// producing the body term in the context of `params` (its names). `param_tys`
    /// gives each parameter's surface type; `ret` is the declared return type.
    #[allow(clippy::too_many_arguments)]
    fn elab_match_body(
        &mut self,
        fnname: &str,
        params: &[String],
        param_tys: &[Ty],
        ret: &Ty,
        scrut: &str,
        arms: &[Arm],
    ) -> Result<Term, String> {
        let scrut_pos = params
            .iter()
            .position(|p| p == scrut)
            .ok_or_else(|| format!("`match` scrutinee `{scrut}` is not a parameter"))?;
        // the scrutinee's type: D params indices
        let (dhead, dargs) = flatten_ty(&param_tys[scrut_pos]);
        let data = match dhead {
            Ty::Var(n) => n.clone(),
            _ => return Err("the scrutinee's type must be a datatype".into()),
        };
        let decl = self
            .sig
            .datas
            .iter()
            .find(|d| d.name == data)
            .ok_or_else(|| format!("`{data}` is not a datatype"))?
            .clone();
        let np = decl.params.len();
        let ni = decl.indices.len();
        if dargs.len() != np + ni {
            return Err(format!("scrutinee type `{data}` applied to the wrong number of arguments"));
        }
        // index arguments must be variables (Phase 1) so we can build the motive
        let index_names: Vec<String> = dargs[np..]
            .iter()
            .map(|a| match a {
                Ty::Var(v) => Ok(v.clone()),
                _ => Err("Phase 1: a matched value's indices must be variables".to_string()),
            })
            .collect::<Result<_, _>>()?;

        // motive = λ index_names. λ scrut. ret      (rebinding shadows the params)
        let mut motive_scope = params.to_vec();
        motive_scope.extend(index_names.iter().cloned());
        motive_scope.push(scrut.to_string());
        let ret_term = self.elab_ty(ret, &motive_scope)?;
        let mut motive = ret_term;
        for _ in 0..(ni + 1) {
            motive = Term::Lam(Box::new(motive));
        }

        // methods, in constructor-declaration order
        let mut methods = Vec::with_capacity(decl.ctors.len());
        for ctor in &decl.ctors {
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
            if arm.binders.len() != nargs {
                return Err(format!(
                    "case `{}`: expected {nargs} binder(s), got {}",
                    ctor.name,
                    arm.binders.len()
                ));
            }
            // method binders: the constructor args, then one IH per recursive arg
            let mut scope = params.to_vec();
            for b in &arm.binders {
                scope.push(b.clone());
            }
            let mut fields: HashMap<String, String> = HashMap::new();
            let mut ih_names = Vec::new();
            for &fi in &rec_fields {
                let ih = self.gensym("ih");
                fields.insert(arm.binders[fi].clone(), ih.clone());
                ih_names.push(ih);
            }
            for ih in &ih_names {
                scope.push(ih.clone());
            }
            let r = Rec { fnname, scrut_pos, fields: &fields };
            let mut body = self.elab_tm(&arm.body, &scope, Some(&r))?;
            for _ in 0..(nargs + rec_fields.len()) {
                body = Term::Lam(Box::new(body));
            }
            methods.push(body);
        }
        // reject arms naming a constructor outside this family
        for a in arms {
            if !decl.ctors.iter().any(|c| c.name == a.ctor) {
                return Err(format!("`{}` is not a constructor of `{data}`", a.ctor));
            }
        }
        let scrut_idx = Elab::debruijn(params, scrut).unwrap();
        Ok(Term::Elim(data, Box::new(motive), methods, Box::new(Term::Var(scrut_idx))))
    }
}

/// Peel a signature `Ty` into its parameter telescope (using the `fn`'s names)
/// and the return type, renaming each named domain to the `fn`'s parameter name.
fn split_signature(sig: &Ty, params: &[String]) -> Result<(Vec<(Mult, Ty)>, Ty), String> {
    let mut doms = Vec::new();
    let mut t = sig.clone();
    let mut i = 0;
    while i < params.len() {
        match t {
            Ty::Arrow(m, name, a, b) => {
                doms.push((m, *a));
                t = match name {
                    Some(sn) => rename_ty(&b, &sn, &params[i]),
                    None => *b,
                };
                i += 1;
            }
            _ => return Err(format!("`fn` has {} parameter(s) but its type has fewer arrows", params.len())),
        }
    }
    Ok((doms, t))
}

/// Rename free occurrences of `from` to `to` in a type (shadow-aware).
fn rename_ty(t: &Ty, from: &str, to: &str) -> Ty {
    match t {
        Ty::Type => Ty::Type,
        Ty::Var(v) => Ty::Var(if v == from { to.to_string() } else { v.clone() }),
        Ty::App(f, a) => Ty::App(Box::new(rename_ty(f, from, to)), Box::new(rename_ty(a, from, to))),
        Ty::Arrow(m, name, a, b) => {
            let a2 = rename_ty(a, from, to);
            // a binder of the same name shadows `from` in the body
            let b2 = if name.as_deref() == Some(from) {
                (**b).clone()
            } else {
                rename_ty(b, from, to)
            };
            Ty::Arrow(*m, name.clone(), Box::new(a2), Box::new(b2))
        }
    }
}

fn count_index_pis(t: &Ty) -> Result<usize, String> {
    match t {
        Ty::Type => Ok(0),
        Ty::Arrow(_, _, _, b) => Ok(1 + count_index_pis(b)?),
        _ => Err("a datatype's index telescope must end in `Type`".into()),
    }
}
fn count_arrows(t: &Ty) -> usize {
    match t {
        Ty::Arrow(_, _, _, b) => 1 + count_arrows(b),
        _ => 0,
    }
}

pub fn elaborate(src: &str) -> Result<Program, String> {
    let toks = lex(src)?;
    let items = Parser { toks, pos: 0 }.parse_program()?;

    let mut elab = Elab {
        data_arity: HashMap::new(),
        ctor_arity: HashMap::new(),
        defs: HashMap::new(),
        sig: Signature::default(),
        fresh: 0,
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
                    elab.ctor_arity.insert(cn.clone(), np + count_arrows(cty));
                }
            }
            Item::Struct { name, params, fields } => {
                elab.data_arity.insert(name.clone(), params.len());
                elab.ctor_arity.insert(name.clone(), params.len() + fields.len());
            }
            _ => {}
        }
    }

    // pass C: datatypes → signature
    for it in &items {
        match it {
            Item::Enum { name, params, index_ty, variants } => {
                let np = params.len();
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
                    let ct = elab.elab_ty(cty, &scope)?;
                    let (args, idxs) = decompose_ctor(ct, name, np)?;
                    ctors.push(Constructor { name: cn.clone(), args, idxs });
                }
                elab.sig.datas.push(DataDecl { name: name.clone(), params: kparams, indices, ctors });
            }
            Item::Struct { name, params, fields } => {
                let mut scope = Vec::new();
                let mut kparams = Vec::new();
                for b in params {
                    let ty = elab.elab_ty(&b.ty, &scope)?;
                    kparams.push((b.mult.unwrap_or(Mult::Zero), ty));
                    scope.push(b.name.clone());
                }
                // one constructor (same name) taking the fields, returning `Struct params`
                let mut args = Vec::new();
                let mut fscope = scope.clone();
                for (fname, fty) in fields {
                    let ty = elab.elab_ty(fty, &fscope)?;
                    args.push((Mult::Omega, ty));
                    fscope.push(fname.clone());
                }
                let ctor = Constructor { name: name.clone(), args, idxs: vec![] };
                elab.sig.datas.push(DataDecl { name: name.clone(), params: kparams, indices: vec![], ctors: vec![ctor] });
            }
            _ => {}
        }
    }

    // pass D: pair signatures with fns, elaborate in order
    let mut sigs: HashMap<String, Ty> = HashMap::new();
    for it in &items {
        if let Item::Sig(n, t) = it {
            sigs.insert(n.clone(), t.clone());
        }
    }
    let mut out_defs = Vec::new();
    for it in &items {
        if let Item::Fn(name, params, body) = it {
            let sig = sigs
                .get(name)
                .ok_or_else(|| format!("`fn {name}` has no type signature"))?
                .clone();
            let ty_term = elab.elab_ty(&sig, &[])?;
            let (doms, ret) = split_signature(&sig, params)?;
            let param_tys: Vec<Ty> = doms.iter().map(|(_, t)| t.clone()).collect();

            // elaborate the body
            let body_inner = match body {
                Tm::Match(scrut, arms) => {
                    elab.elab_match_body(name, params, &param_tys, &ret, scrut, arms)?
                }
                other => elab.elab_tm(other, params, None)?,
            };
            let mut term = body_inner;
            for _ in 0..params.len() {
                term = Term::Lam(Box::new(term));
            }
            elab.defs.insert(name.clone(), (term.clone(), ty_term.clone()));
            out_defs.push((name.clone(), ty_term, term));
        }
    }

    Ok(Program { sig: elab.sig, defs: out_defs })
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

/// Readable rendering of a closed normal term (reuses the v0.9 pretty-printer style).
pub fn pretty(t: &Term) -> String {
    crate::surface::pretty(t)
}

#[cfg(test)]
mod tests;
