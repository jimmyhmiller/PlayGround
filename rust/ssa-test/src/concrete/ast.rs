//! AST types for the example language.

#[derive(Debug, Clone, PartialEq)]
pub enum Ast {
    // Literals
    Literal(i32),
    Variable(String),

    // Operations
    BinaryOp {
        left: Box<Ast>,
        op: BinaryOperator,
        right: Box<Ast>,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Ast>,
    },

    // Statements
    Assignment {
        variable: String,
        value: Box<Ast>,
    },
    If {
        condition: Box<Ast>,
        then_branch: Vec<Ast>,
        else_branch: Option<Vec<Ast>>,
    },
    While {
        condition: Box<Ast>,
        body: Vec<Ast>,
    },

    // Program/Block
    Block(Vec<Ast>),

    // Print statement
    Print(Box<Ast>),
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

impl Ast {
    pub fn literal(value: i32) -> Self {
        Ast::Literal(value)
    }

    pub fn variable(name: &str) -> Self {
        Ast::Variable(name.to_string())
    }

    pub fn binary_op(left: Ast, op: BinaryOperator, right: Ast) -> Self {
        Ast::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    pub fn unary_op(op: UnaryOperator, operand: Ast) -> Self {
        Ast::UnaryOp {
            op,
            operand: Box::new(operand),
        }
    }

    pub fn assignment(variable: &str, value: Ast) -> Self {
        Ast::Assignment {
            variable: variable.to_string(),
            value: Box::new(value),
        }
    }

    pub fn if_stmt(condition: Ast, then_branch: Vec<Ast>, else_branch: Option<Vec<Ast>>) -> Self {
        Ast::If {
            condition: Box::new(condition),
            then_branch,
            else_branch,
        }
    }

    pub fn while_stmt(condition: Ast, body: Vec<Ast>) -> Self {
        Ast::While {
            condition: Box::new(condition),
            body,
        }
    }

    pub fn block(statements: Vec<Ast>) -> Self {
        Ast::Block(statements)
    }

    pub fn print(value: Ast) -> Self {
        Ast::Print(Box::new(value))
    }
}
