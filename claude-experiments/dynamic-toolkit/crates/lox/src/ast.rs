/// Lox AST produced by the parser.

#[derive(Debug, Clone)]
pub enum Expr {
    Number(f64),
    String(std::string::String),
    Bool(bool),
    Nil,
    Var(std::string::String),
    Assign(std::string::String, Box<Expr>),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnaryOp, Box<Expr>),
    Logical(Box<Expr>, LogicalOp, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Get(Box<Expr>, std::string::String),
    Set(Box<Expr>, std::string::String, Box<Expr>),
    This,
    Super(std::string::String),
    Grouping(Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy)]
pub enum LogicalOp {
    And,
    Or,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Expr(Expr),
    Print(Expr),
    Var(std::string::String, Option<Expr>),
    Block(Vec<Stmt>),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    While(Expr, Box<Stmt>),
    Return(Option<Expr>),
    Fun(FunDecl),
    Class(ClassDecl),
}

#[derive(Debug, Clone)]
pub struct FunDecl {
    pub name: std::string::String,
    pub params: Vec<std::string::String>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub struct ClassDecl {
    pub name: std::string::String,
    pub superclass: Option<std::string::String>,
    pub methods: Vec<FunDecl>,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub stmts: Vec<Stmt>,
}
