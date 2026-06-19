//! A SURFACE language for the dependent+linear core (`dep.rs`), with a parser
//! and an elaborator. You write named syntax — `data` declarations, `def`s,
//! `fun`/`->`/application, and a `match`-style `elim` — and it is elaborated to
//! the de Bruijn kernel `Term`/`Signature`. The kernel does ALL type checking;
//! elaboration is purely syntactic (name resolution + the eliminator assembly).
//!
//! ```text
//! data N where
//!   | z : N
//!   | s : N -> N
//!
//! def add : (m : N) -> (n : N) -> N
//!   = fun m n => elim m to (fun _ => N) {
//!       | z      => n
//!       | s k ih => s ih
//!     }
//! ```
//!
//! Multiplicities use Idris's QTT notation in binders: `(0 x : A)` erased,
//! `(1 x : A)` linear, `(w x : A)` or plain `(x : A)` unrestricted. `data`
//! parameters/indices default to erased (`0`).
//!
//! Built-in heads the elaborator recognises: `Type`, and `Eq A x y` / `refl x`
//! (the identity type, for proofs). Everything else is a datatype, constructor,
//! definition, or bound variable. Recursion is ONLY via `elim`, so every `def`
//! is total by construction.

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
    Colon,
    Eq,
    FatArrow, // =>
    Arrow,    // ->
    Bar,      // |
    Semi,
    KwData,
    KwDef,
    KwFun,
    KwWhere,
    KwElim,
    KwTo,
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
            let word = &src[s..i];
            out.push(match word {
                "data" => Tok::KwData,
                "def" => Tok::KwDef,
                "fun" => Tok::KwFun,
                "where" => Tok::KwWhere,
                "elim" => Tok::KwElim,
                "to" => Tok::KwTo,
                "Type" => Tok::KwType,
                _ => Tok::Ident(word.to_string()),
            });
        } else {
            let t = match c {
                '(' => Tok::LParen,
                ')' => Tok::RParen,
                '{' => Tok::LBrace,
                '}' => Tok::RBrace,
                ':' => Tok::Colon,
                '=' => Tok::Eq,
                '|' => Tok::Bar,
                ';' => Tok::Semi,
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
enum SExpr {
    Var(String),
    Type,
    Pi(Mult, Option<String>, Box<SExpr>, Box<SExpr>),
    Lam(Vec<String>, Box<SExpr>),
    App(Box<SExpr>, Box<SExpr>),
    Elim(Box<SExpr>, Box<SExpr>, Vec<SBranch>),
}

#[derive(Clone, Debug)]
struct SBranch {
    ctor: String,
    binders: Vec<String>,
    body: SExpr,
}

#[derive(Clone, Debug)]
struct Binder {
    mult: Option<Mult>,
    name: String,
    ty: SExpr,
}

#[derive(Clone, Debug)]
struct SData {
    name: String,
    params: Vec<Binder>,
    index_ty: Option<SExpr>, // an arrow chain ending in Type
    ctors: Vec<(String, SExpr)>,
}

#[derive(Clone, Debug)]
struct SDef {
    name: String,
    ty: SExpr,
    body: SExpr,
}

#[derive(Clone, Debug)]
enum SDecl {
    Data(SData),
    Def(SDef),
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
    fn peek2(&self) -> Option<&Tok> {
        self.toks.get(self.pos + 1)
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

    fn parse_program(&mut self) -> Result<Vec<SDecl>, String> {
        let mut decls = Vec::new();
        while self.peek().is_some() {
            decls.push(self.parse_decl()?);
        }
        Ok(decls)
    }

    fn parse_decl(&mut self) -> Result<SDecl, String> {
        match self.peek() {
            Some(Tok::KwData) => self.parse_data().map(SDecl::Data),
            Some(Tok::KwDef) => self.parse_def().map(SDecl::Def),
            other => Err(format!("expected `data` or `def`, found {other:?}")),
        }
    }

    fn parse_data(&mut self) -> Result<SData, String> {
        self.eat(&Tok::KwData)?;
        let name = self.ident()?;
        let mut params = Vec::new();
        while self.looks_like_binder() {
            params.push(self.parse_binder()?);
        }
        let index_ty = if self.peek() == Some(&Tok::Colon) {
            self.next();
            Some(self.parse_term()?)
        } else {
            None
        };
        self.eat(&Tok::KwWhere)?;
        let mut ctors = Vec::new();
        while self.peek() == Some(&Tok::Bar) {
            self.next();
            let cname = self.ident()?;
            self.eat(&Tok::Colon)?;
            let cty = self.parse_term()?;
            ctors.push((cname, cty));
        }
        Ok(SData { name, params, index_ty, ctors })
    }

    fn parse_def(&mut self) -> Result<SDef, String> {
        self.eat(&Tok::KwDef)?;
        let name = self.ident()?;
        self.eat(&Tok::Colon)?;
        let ty = self.parse_term()?;
        self.eat(&Tok::Eq)?;
        let body = self.parse_term()?;
        Ok(SDef { name, ty, body })
    }

    /// A binder is `( [mult] name : type )`. Peek for `(` then optional mult then
    /// ident then `:`.
    fn looks_like_binder(&self) -> bool {
        if self.peek() != Some(&Tok::LParen) {
            return false;
        }
        let k = self.pos + 1;
        // `( mult name : ...`
        if (matches!(self.toks.get(k), Some(Tok::Num(0)) | Some(Tok::Num(1)))
            || matches!(self.toks.get(k), Some(Tok::Ident(w)) if w == "w"))
            && matches!(self.toks.get(k + 1), Some(Tok::Ident(_)))
            && self.toks.get(k + 2) == Some(&Tok::Colon)
        {
            return true;
        }
        // `( name : ...`
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
        // a multiplicity only if an ident + ':' follow it
        let mult = if (matches!(self.peek(), Some(Tok::Num(0)) | Some(Tok::Num(1)))
            || matches!(self.peek(), Some(Tok::Ident(w)) if w == "w"))
            && matches!(self.peek2(), Some(Tok::Ident(_)))
        {
            self.parse_mult()
        } else {
            None
        };
        let name = self.ident()?;
        self.eat(&Tok::Colon)?;
        let ty = self.parse_term()?;
        self.eat(&Tok::RParen)?;
        Ok(Binder { mult, name, ty })
    }

    fn parse_term(&mut self) -> Result<SExpr, String> {
        match self.peek() {
            Some(Tok::KwFun) => {
                self.next();
                let mut names = Vec::new();
                while let Some(Tok::Ident(_)) = self.peek() {
                    names.push(self.ident()?);
                }
                if names.is_empty() {
                    return Err("`fun` needs at least one parameter".into());
                }
                self.eat(&Tok::FatArrow)?;
                let body = self.parse_term()?;
                Ok(SExpr::Lam(names, Box::new(body)))
            }
            Some(Tok::KwElim) => self.parse_elim(),
            _ => self.parse_pi(),
        }
    }

    fn parse_pi(&mut self) -> Result<SExpr, String> {
        if self.looks_like_binder() {
            let mut binders = Vec::new();
            while self.looks_like_binder() {
                binders.push(self.parse_binder()?);
            }
            self.eat(&Tok::Arrow)?;
            let body = self.parse_term()?;
            let mut out = body;
            for bnd in binders.into_iter().rev() {
                out = SExpr::Pi(
                    bnd.mult.unwrap_or(Mult::Omega),
                    Some(bnd.name),
                    Box::new(bnd.ty),
                    Box::new(out),
                );
            }
            Ok(out)
        } else {
            let lhs = self.parse_app()?;
            if self.peek() == Some(&Tok::Arrow) {
                self.next();
                let rhs = self.parse_term()?;
                Ok(SExpr::Pi(Mult::Omega, None, Box::new(lhs), Box::new(rhs)))
            } else {
                Ok(lhs)
            }
        }
    }

    fn parse_app(&mut self) -> Result<SExpr, String> {
        let mut e = self.parse_atom()?;
        while self.starts_atom() {
            let arg = self.parse_atom()?;
            e = SExpr::App(Box::new(e), Box::new(arg));
        }
        Ok(e)
    }

    fn starts_atom(&self) -> bool {
        matches!(
            self.peek(),
            Some(Tok::Ident(_)) | Some(Tok::LParen) | Some(Tok::KwType)
        )
    }

    fn parse_atom(&mut self) -> Result<SExpr, String> {
        match self.peek() {
            Some(Tok::KwType) => {
                self.next();
                Ok(SExpr::Type)
            }
            Some(Tok::Ident(_)) => Ok(SExpr::Var(self.ident()?)),
            Some(Tok::LParen) => {
                self.next();
                let e = self.parse_term()?;
                self.eat(&Tok::RParen)?;
                Ok(e)
            }
            other => Err(format!("expected an atom, found {other:?}")),
        }
    }

    fn parse_elim(&mut self) -> Result<SExpr, String> {
        self.eat(&Tok::KwElim)?;
        let scrut = self.parse_app()?;
        self.eat(&Tok::KwTo)?;
        let motive = self.parse_atom()?;
        self.eat(&Tok::LBrace)?;
        let mut branches = Vec::new();
        while self.peek() == Some(&Tok::Bar) {
            self.next();
            let ctor = self.ident()?;
            let mut binders = Vec::new();
            while let Some(Tok::Ident(_)) = self.peek() {
                binders.push(self.ident()?);
            }
            self.eat(&Tok::FatArrow)?;
            let body = self.parse_term()?;
            branches.push(SBranch { ctor, binders, body });
            if self.peek() == Some(&Tok::Semi) {
                self.next();
            }
        }
        self.eat(&Tok::RBrace)?;
        Ok(SExpr::Elim(Box::new(scrut), Box::new(motive), branches))
    }
}

// peel helpers on the surface AST (for computing arities before elaboration)
fn count_index_pis(e: &SExpr) -> Result<usize, String> {
    match e {
        SExpr::Type => Ok(0),
        SExpr::Pi(_, _, _, b) => Ok(1 + count_index_pis(b)?),
        _ => Err("a datatype's index telescope must end in `Type`".into()),
    }
}
fn count_ctor_pis(e: &SExpr) -> usize {
    match e {
        SExpr::Pi(_, _, _, b) => 1 + count_ctor_pis(b),
        _ => 0,
    }
}

// ===========================================================================
// elaboration: surface AST -> kernel Signature + definitions
// ===========================================================================

pub struct Program {
    pub sig: Signature,
    pub defs: Vec<(String, Term, Term)>, // (name, type, body)
}

struct Elab {
    data_arity: HashMap<String, usize>,  // params + indices
    ctor_arity: HashMap<String, usize>,  // params + args
    defs: HashMap<String, (Term, Term)>, // body, type
    sig: Signature,
}

impl Elab {
    /// Resolve a name used as the head of a spine.
    fn elab_spine(&self, head: &str, args: &[SExpr], scope: &[String]) -> Result<Term, String> {
        // built-in heads
        match head {
            "Eq" => {
                if args.len() != 3 {
                    return Err("`Eq` takes exactly 3 arguments (Eq A x y)".into());
                }
                return Ok(Term::Eq(
                    Box::new(self.elab(&args[0], scope)?),
                    Box::new(self.elab(&args[1], scope)?),
                    Box::new(self.elab(&args[2], scope)?),
                ));
            }
            "refl" => {
                if args.len() != 1 {
                    return Err("`refl` takes exactly 1 argument (refl x)".into());
                }
                return Ok(Term::Refl(Box::new(self.elab(&args[0], scope)?)));
            }
            _ => {}
        }
        // a bound variable?
        if let Some(rev) = scope.iter().rev().position(|s| s == head) {
            let mut e = Term::Var(rev);
            for a in args {
                e = Term::App(Box::new(e), Box::new(self.elab(a, scope)?));
            }
            return Ok(e);
        }
        // a datatype?
        if let Some(&arity) = self.data_arity.get(head) {
            if args.len() != arity {
                return Err(format!(
                    "datatype `{head}` expects {arity} argument(s), got {}",
                    args.len()
                ));
            }
            let eargs = args
                .iter()
                .map(|a| self.elab(a, scope))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Term::Data(head.to_string(), eargs));
        }
        // a constructor?
        if let Some(&arity) = self.ctor_arity.get(head) {
            if args.len() != arity {
                return Err(format!(
                    "constructor `{head}` expects {arity} argument(s), got {}",
                    args.len()
                ));
            }
            let eargs = args
                .iter()
                .map(|a| self.elab(a, scope))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Term::Constr(head.to_string(), eargs));
        }
        // a previously-defined definition (inlined, annotated for inference)?
        if let Some((body, ty)) = self.defs.get(head) {
            let mut e = Term::Ann(Box::new(body.clone()), Box::new(ty.clone()));
            for a in args {
                e = Term::App(Box::new(e), Box::new(self.elab(a, scope)?));
            }
            return Ok(e);
        }
        Err(format!("unbound name `{head}`"))
    }

    fn elab(&self, e: &SExpr, scope: &[String]) -> Result<Term, String> {
        match e {
            SExpr::Type => Ok(Term::Type),
            SExpr::Pi(m, name, a, b) => {
                let ta = self.elab(a, scope)?;
                let mut s2 = scope.to_vec();
                s2.push(name.clone().unwrap_or_else(|| "_".into()));
                let tb = self.elab(b, &s2)?;
                Ok(Term::Pi(*m, Box::new(ta), Box::new(tb)))
            }
            SExpr::Lam(names, body) => {
                let mut s2 = scope.to_vec();
                for n in names {
                    s2.push(n.clone());
                }
                let mut t = self.elab(body, &s2)?;
                for _ in names {
                    t = Term::Lam(Box::new(t));
                }
                Ok(t)
            }
            SExpr::Elim(scrut, motive, branches) => self.elab_elim(scrut, motive, branches, scope),
            SExpr::Var(_) | SExpr::App(_, _) => {
                let (head, args) = flatten(e);
                match head {
                    SExpr::Var(name) => self.elab_spine(name, &args, scope),
                    other => {
                        let mut t = self.elab(other, scope)?;
                        for a in &args {
                            t = Term::App(Box::new(t), Box::new(self.elab(a, scope)?));
                        }
                        Ok(t)
                    }
                }
            }
        }
    }

    fn elab_elim(
        &self,
        scrut: &SExpr,
        motive: &SExpr,
        branches: &[SBranch],
        scope: &[String],
    ) -> Result<Term, String> {
        if branches.is_empty() {
            return Err("`elim` needs at least one branch".into());
        }
        let s_term = self.elab(scrut, scope)?;
        let m_term = self.elab(motive, scope)?;
        // which family? (from the constructors named in the branches)
        let (decl, _) = self
            .sig
            .ctor(&branches[0].ctor)
            .ok_or_else(|| format!("`{}` is not a constructor", branches[0].ctor))?;
        let data = decl.name.clone();
        let order: Vec<String> = decl.ctors.iter().map(|c| c.name.clone()).collect();

        let mut methods = Vec::with_capacity(order.len());
        for cname in &order {
            let ctor = self
                .sig
                .data(&data)
                .unwrap()
                .ctors
                .iter()
                .find(|c| &c.name == cname)
                .unwrap()
                .clone();
            let nrec = ctor
                .args
                .iter()
                .filter(|(_, aty)| matches!(aty, Term::Data(dn, _) if *dn == data))
                .count();
            let expected = ctor.args.len() + nrec;
            let br = branches
                .iter()
                .find(|b| &b.ctor == cname)
                .ok_or_else(|| format!("elim is missing a case for `{cname}`"))?;
            if br.binders.len() != expected {
                return Err(format!(
                    "case `{cname}`: expected {expected} binder(s) ({} arg(s) + {nrec} IH(s)), got {}",
                    ctor.args.len(),
                    br.binders.len()
                ));
            }
            let mut s2 = scope.to_vec();
            for bn in &br.binders {
                s2.push(bn.clone());
            }
            let mut body = self.elab(&br.body, &s2)?;
            for _ in &br.binders {
                body = Term::Lam(Box::new(body));
            }
            methods.push(body);
        }
        // reject branches naming a constructor outside this family
        for b in branches {
            if !order.contains(&b.ctor) {
                return Err(format!("`{}` is not a constructor of `{data}`", b.ctor));
            }
        }
        Ok(Term::Elim(data, Box::new(m_term), methods, Box::new(s_term)))
    }
}

fn flatten(e: &SExpr) -> (&SExpr, Vec<SExpr>) {
    let mut args = Vec::new();
    let mut head = e;
    while let SExpr::App(f, a) = head {
        args.push((**a).clone());
        head = f;
    }
    args.reverse();
    (head, args)
}

/// Decompose an elaborated constructor type `Π(args). D params idxs` into the
/// kernel's argument telescope and result indices.
fn decompose_ctor(
    mut t: Term,
    data: &str,
    nparams: usize,
) -> Result<(Vec<(Mult, Term)>, Vec<Term>), String> {
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
            // the first nparams must be exactly the parameter variables
            for (i, sp) in spine.iter().take(nparams).enumerate() {
                let expected = Term::Var(nargs + nparams - 1 - i);
                if *sp != expected {
                    return Err(format!(
                        "a constructor of `{data}` must return it at its parameters \
                         (parameter {i} mismatched)"
                    ));
                }
            }
            let idxs = spine[nparams..].to_vec();
            Ok((args, idxs))
        }
        other => Err(format!(
            "a constructor of `{data}` must return `{data} ...`, found {other:?}"
        )),
    }
}

/// Parse + elaborate a whole program into a kernel signature and definitions.
pub fn elaborate(src: &str) -> Result<Program, String> {
    let toks = lex(src)?;
    let decls = Parser { toks, pos: 0 }.parse_program()?;

    // pass B: compute arities (purely syntactic)
    let mut elab = Elab {
        data_arity: HashMap::new(),
        ctor_arity: HashMap::new(),
        defs: HashMap::new(),
        sig: Signature::default(),
    };
    // store decomposition info for pass C
    let mut data_nparams: HashMap<String, usize> = HashMap::new();
    for d in &decls {
        if let SDecl::Data(sd) = d {
            let np = sd.params.len();
            let ni = match &sd.index_ty {
                Some(e) => count_index_pis(e)?,
                None => 0,
            };
            elab.data_arity.insert(sd.name.clone(), np + ni);
            data_nparams.insert(sd.name.clone(), np);
            for (cname, cty) in &sd.ctors {
                elab.ctor_arity.insert(cname.clone(), np + count_ctor_pis(cty));
            }
        }
    }

    // pass C: elaborate datatype declarations into the signature
    for d in &decls {
        if let SDecl::Data(sd) = d {
            let np = sd.params.len();
            // parameter telescope (in growing scope)
            let mut scope: Vec<String> = Vec::new();
            let mut params = Vec::new();
            for bnd in &sd.params {
                let ty = elab.elab(&bnd.ty, &scope)?;
                params.push((bnd.mult.unwrap_or(Mult::Zero), ty));
                scope.push(bnd.name.clone());
            }
            // index telescope (in scope [params]); peel the arrow chain
            let mut indices = Vec::new();
            if let Some(e) = &sd.index_ty {
                let mut it = elab.elab(e, &scope)?;
                while let Term::Pi(_, a, b) = it {
                    indices.push((Mult::Zero, *a));
                    it = *b;
                }
            }
            // constructors
            let mut ctors = Vec::new();
            for (cname, cty) in &sd.ctors {
                // ctor type elaborated in scope [params]
                let cty_term = elab.elab(cty, &scope)?;
                let (args, idxs) = decompose_ctor(cty_term, &sd.name, np)?;
                ctors.push(Constructor { name: cname.clone(), args, idxs });
            }
            elab.sig.datas.push(DataDecl {
                name: sd.name.clone(),
                params,
                indices,
                ctors,
            });
        }
    }

    // pass E: elaborate definitions (each may reference earlier ones)
    let mut out_defs = Vec::new();
    for d in &decls {
        if let SDecl::Def(sd) = d {
            let ty = elab.elab(&sd.ty, &[])?;
            let body = elab.elab(&sd.body, &[])?;
            elab.defs.insert(sd.name.clone(), (body.clone(), ty.clone()));
            out_defs.push((sd.name.clone(), ty, body));
        }
    }

    Ok(Program { sig: elab.sig, defs: out_defs })
}

/// Elaborate and type-check a whole program. Returns the checked program.
pub fn check_program(src: &str) -> Result<Program, Vec<String>> {
    let prog = elaborate(src).map_err(|e| vec![e])?;
    let mut diags = Vec::new();
    if let Err(e) = dep::check_signature(&prog.sig) {
        diags.push(format!("signature: {e}"));
    }
    for (name, ty, body) in &prog.defs {
        if let Err(e) = dep::check_closed_in(prog.sig.clone(), body, ty) {
            diags.push(format!("def {name}: {e}"));
        }
    }
    if diags.is_empty() {
        Ok(prog)
    } else {
        Err(diags)
    }
}

impl Program {
    /// The normal form of a checked definition's body.
    pub fn normalize(&self, name: &str) -> Option<Term> {
        let (_, _, body) = self.defs.iter().find(|(n, _, _)| n == name)?;
        Some(dep::normalize_closed_in(self.sig.clone(), body))
    }
}

/// A readable rendering of a (closed, normal) kernel term — datatype values,
/// constructors, `Type`, etc. — for CLI output.
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
        Term::Type => "Type".into(),
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
