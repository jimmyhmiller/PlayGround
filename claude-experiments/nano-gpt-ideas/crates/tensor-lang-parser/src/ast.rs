#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    FnDef(FnDef),
    Let(LetBinding),
    /// Declare a symbolic dimension parameter: `dim T`
    DimDecl(String),
    /// Loop unrolled at compile time: `for i in 0..N { ... }`.
    /// `start` and `end` must evaluate to concrete integers.
    ForLoop {
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Item>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let(LetBinding),
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetBinding {
    pub name: String,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    Ident(String),
    Array(Vec<Expr>),
    Call {
        name: String,
        args: Vec<Arg>,
    },
    BinOp {
        op: BinOpKind,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    Positional(Expr),
    Named { name: String, value: Expr },
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOpKind {
    Mul,
    Div,
    Add,
    Sub,
}
