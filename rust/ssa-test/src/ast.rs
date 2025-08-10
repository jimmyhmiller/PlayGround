#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(i32),
    Variable(String),
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Assignment {
        variable: String,
        value: Expr,
    },
    If {
        condition: Expr,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
    },
    While {
        condition: Expr,
        body: Vec<Stmt>,
    },
    Expression(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub statements: Vec<Stmt>,
}

impl Expr {
    pub fn literal(value: i32) -> Self {
        Expr::Literal(value)
    }

    pub fn variable(name: &str) -> Self {
        Expr::Variable(name.to_string())
    }

    pub fn binary_op(left: Expr, op: BinaryOperator, right: Expr) -> Self {
        Expr::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    pub fn unary_op(op: UnaryOperator, operand: Expr) -> Self {
        Expr::UnaryOp {
            op,
            operand: Box::new(operand),
        }
    }
}

impl Stmt {
    pub fn assignment(variable: &str, value: Expr) -> Self {
        Stmt::Assignment {
            variable: variable.to_string(),
            value,
        }
    }

    pub fn if_stmt(condition: Expr, then_branch: Vec<Stmt>, else_branch: Option<Vec<Stmt>>) -> Self {
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        }
    }

    pub fn while_stmt(condition: Expr, body: Vec<Stmt>) -> Self {
        Stmt::While { condition, body }
    }

    pub fn expression(expr: Expr) -> Self {
        Stmt::Expression(expr)
    }
}