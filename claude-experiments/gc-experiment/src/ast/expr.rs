use crate::ast::types::Type;

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    Shl, // Shift left
}

/// Expressions in our language
#[derive(Debug, Clone)]
pub enum Expr {
    /// Integer literal
    IntLit(i64),
    /// Boolean literal
    BoolLit(bool),
    /// Null pointer literal (for struct/array types)
    Null,
    /// Variable reference
    Var(String),
    /// Binary operation
    BinOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Allocate a new struct: new StructName
    NewStruct(String),
    /// Allocate a new array: new Type[size]
    NewArray {
        element_type: Type,
        size: Box<Expr>,
    },
    /// Field access: expr.field
    FieldGet {
        object: Box<Expr>,
        struct_name: String, // Needed to look up field index
        field: String,
    },
    /// Array element access: expr[index]
    ArrayGet {
        array: Box<Expr>,
        index: Box<Expr>,
    },
    /// Array length: expr.length
    ArrayLen(Box<Expr>),
    /// Function call: func(args...)
    Call {
        function: String,
        args: Vec<Expr>,
    },
    /// Print an integer value (for debugging)
    PrintInt(Box<Expr>),
}

/// Statements in our language
#[derive(Debug, Clone)]
pub enum Stmt {
    /// Variable declaration with optional initializer: let name: type = expr
    Let {
        name: String,
        typ: Type,
        init: Option<Expr>,
    },
    /// Assignment to a variable: name = expr
    Assign {
        name: String,
        value: Expr,
    },
    /// Field assignment: obj.field = expr
    FieldSet {
        object: Expr,
        struct_name: String,
        field: String,
        value: Expr,
    },
    /// Array element assignment: arr[index] = expr
    ArraySet {
        array: Expr,
        index: Expr,
        value: Expr,
    },
    /// Return statement: return expr
    Return(Option<Expr>),
    /// If statement: if cond { then } else { else }
    If {
        condition: Expr,
        then_block: Vec<Stmt>,
        else_block: Vec<Stmt>,
    },
    /// While loop: while cond { body }
    While {
        condition: Expr,
        body: Vec<Stmt>,
    },
    /// Expression statement (for side effects, e.g., function calls)
    Expr(Expr),
}

impl Expr {
    // Convenience constructors

    pub fn int(n: i64) -> Self {
        Expr::IntLit(n)
    }

    pub fn bool_lit(b: bool) -> Self {
        Expr::BoolLit(b)
    }

    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var(name.into())
    }

    pub fn add(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn sub(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Sub,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn mul(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Mul,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn eq(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Eq,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn ne(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Ne,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn lt(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Lt,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn le(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Le,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn gt(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Gt,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn ge(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Ge,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn shl(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Shl,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn new_struct(name: impl Into<String>) -> Self {
        Expr::NewStruct(name.into())
    }

    pub fn new_array(element_type: Type, size: Expr) -> Self {
        Expr::NewArray {
            element_type,
            size: Box::new(size),
        }
    }

    pub fn field_get(object: Expr, struct_name: impl Into<String>, field: impl Into<String>) -> Self {
        Expr::FieldGet {
            object: Box::new(object),
            struct_name: struct_name.into(),
            field: field.into(),
        }
    }

    pub fn array_get(array: Expr, index: Expr) -> Self {
        Expr::ArrayGet {
            array: Box::new(array),
            index: Box::new(index),
        }
    }

    pub fn array_len(array: Expr) -> Self {
        Expr::ArrayLen(Box::new(array))
    }

    pub fn call(function: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::Call {
            function: function.into(),
            args,
        }
    }

    pub fn print_int(value: Expr) -> Self {
        Expr::PrintInt(Box::new(value))
    }
}

impl Stmt {
    // Convenience constructors

    pub fn let_decl(name: impl Into<String>, typ: Type, init: Option<Expr>) -> Self {
        Stmt::Let {
            name: name.into(),
            typ,
            init,
        }
    }

    pub fn assign(name: impl Into<String>, value: Expr) -> Self {
        Stmt::Assign {
            name: name.into(),
            value,
        }
    }

    pub fn field_set(
        object: Expr,
        struct_name: impl Into<String>,
        field: impl Into<String>,
        value: Expr,
    ) -> Self {
        Stmt::FieldSet {
            object,
            struct_name: struct_name.into(),
            field: field.into(),
            value,
        }
    }

    pub fn array_set(array: Expr, index: Expr, value: Expr) -> Self {
        Stmt::ArraySet {
            array,
            index,
            value,
        }
    }

    pub fn ret(expr: Option<Expr>) -> Self {
        Stmt::Return(expr)
    }

    pub fn if_stmt(condition: Expr, then_block: Vec<Stmt>, else_block: Vec<Stmt>) -> Self {
        Stmt::If {
            condition,
            then_block,
            else_block,
        }
    }

    pub fn while_loop(condition: Expr, body: Vec<Stmt>) -> Self {
        Stmt::While { condition, body }
    }

    pub fn expr(e: Expr) -> Self {
        Stmt::Expr(e)
    }
}
