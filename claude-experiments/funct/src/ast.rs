//! AST. Pipes and UFCS sugar are desugared by the parser, so the compiler
//! never sees `|>`.

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Fn(FnDef),
    Type(TypeDef),
    /// Top-level `let pattern = expr` (no `mut` allowed).
    Let {
        pattern: Pattern,
        expr: Expr,
        exported: bool,
        line: u32,
    },
    Expr(Expr),
    Import(ImportDef),
    /// `extern fn name(args)` — declares a host-provided function. Binds to
    /// the registered native when present; otherwise calling it faults.
    Extern {
        name: String,
        params: Vec<String>,
        line: u32,
    },
    /// `extern let name` — declares a host-injected value (vm.set_global);
    /// reading it before the host provides one faults.
    ExternLet {
        name: String,
        line: u32,
    },
}

#[derive(Debug, Clone)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<Pattern>,
    pub body: Expr,
    pub exported: bool,
    /// `#[...]` attributes; currently only "test" is defined
    pub attrs: Vec<String>,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct ImportDef {
    /// module path, e.g. "math/vec" → `<root>/math/vec.ft` (or a host module)
    pub path: String,
    pub kind: ImportKind,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub enum ImportKind {
    /// `import { a, b as c } from "m"` — (name, alias)
    Named(Vec<(String, Option<String>)>),
    /// `import "m" as alias` — alias defaults to the last path segment
    Qualified(Option<String>),
}

#[derive(Debug, Clone)]
pub struct TypeDef {
    pub name: String,
    pub variants: Vec<VariantDef>,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct VariantDef {
    pub tag: String,
    /// Field names for record-style variants; None for bare tags.
    pub fields: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        mutable: bool,
        pattern: Pattern,
        expr: Expr,
        line: u32,
    },
    /// `x = e` / `x += e` (desugared to plain assign by parser)
    Assign {
        name: String,
        expr: Expr,
        line: u32,
    },
    While {
        cond: Expr,
        body: Expr,
        line: u32,
    },
    For {
        pattern: Pattern,
        iter: Expr,
        body: Expr,
        line: u32,
    },
    Return {
        expr: Option<Expr>,
        line: u32,
    },
    Break {
        line: u32,
    },
    Continue {
        line: u32,
    },
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// String interpolation: concat of parts (each coerced via str()).
    Interp(Vec<InterpPart>),
    Ident(String),
    /// `Foo`, `Foo(a, b)`, `Foo { x: 1 }`
    Variant {
        tag: String,
        payload: VariantCtor,
    },
    List(Vec<Expr>),
    Tuple(Vec<Expr>),
    /// Record literal; `spread` is the `{ ..base, x: 1 }` base.
    Record {
        spread: Option<Box<Expr>>,
        fields: Vec<(String, Expr)>,
    },
    Lambda {
        params: Vec<Pattern>,
        body: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    /// UFCS / method call: `x.name(args)`
    MethodCall {
        recv: Box<Expr>,
        name: String,
        args: Vec<Expr>,
    },
    Field {
        recv: Box<Expr>,
        name: String,
    },
    Index {
        recv: Box<Expr>,
        index: Box<Expr>,
    },
    Unary {
        op: UnOp,
        operand: Box<Expr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Range {
        lo: Box<Expr>,
        hi: Box<Expr>,
        inclusive: bool,
    },
    If {
        cond: Box<Expr>,
        then: Box<Expr>,
        els: Option<Box<Expr>>,
    },
    Match {
        subject: Box<Expr>,
        arms: Vec<Arm>,
    },
    Block(Vec<Stmt>, Option<Box<Expr>>),
    /// postfix `?`
    Try(Box<Expr>),
    /// `@a`
    Deref(Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum InterpPart {
    Lit(String),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub enum VariantCtor {
    Unit,
    Positional(Vec<Expr>),
    Named(Vec<(String, Expr)>),
}

#[derive(Debug, Clone)]
pub struct Arm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
    pub line: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Bind(String),
    LitInt(i64),
    LitFloat(f64),
    LitStr(String),
    LitBool(bool),
    LitUnit,
    /// `Some(x)` / bare `None`
    VariantPos {
        tag: String,
        items: Vec<Pattern>,
    },
    /// `Circle { radius: r, .. }`
    VariantNamed {
        tag: String,
        fields: Vec<(String, Pattern)>,
        rest: bool,
    },
    /// `{ x, y: p }` record pattern
    Record {
        fields: Vec<(String, Pattern)>,
        rest: bool,
    },
    Tuple(Vec<Pattern>),
    /// `[a, b, ..rest]`; rest: None = exact, Some(None) = `..`, Some(Some(name)) = `..rest`
    List {
        items: Vec<Pattern>,
        rest: Option<Option<String>>,
    },
    Range {
        lo: i64,
        hi: i64,
        inclusive: bool,
    },
    Or(Vec<Pattern>),
    As(Box<Pattern>, String),
}

impl Pattern {
    /// All names bound by this pattern, in deterministic order.
    pub fn bound_names(&self, out: &mut Vec<String>) {
        match self {
            Pattern::Wildcard
            | Pattern::LitInt(_)
            | Pattern::LitFloat(_)
            | Pattern::LitStr(_)
            | Pattern::LitBool(_)
            | Pattern::LitUnit
            | Pattern::Range { .. } => {}
            Pattern::Bind(n) => {
                if !out.contains(n) {
                    out.push(n.clone());
                }
            }
            Pattern::VariantPos { items, .. } | Pattern::Tuple(items) => {
                for p in items {
                    p.bound_names(out);
                }
            }
            Pattern::VariantNamed { fields, .. } | Pattern::Record { fields, .. } => {
                for (_, p) in fields {
                    p.bound_names(out);
                }
            }
            Pattern::List { items, rest } => {
                for p in items {
                    p.bound_names(out);
                }
                if let Some(Some(n)) = rest {
                    if !out.contains(n) {
                        out.push(n.clone());
                    }
                }
            }
            Pattern::Or(alts) => {
                // both sides must bind the same names; take the first
                if let Some(first) = alts.first() {
                    first.bound_names(out);
                }
            }
            Pattern::As(inner, name) => {
                inner.bound_names(out);
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
        }
    }
}
