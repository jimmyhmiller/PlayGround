#[macro_export]
macro_rules! expr {
    // Binary operations
    ((+ $left:tt $right:tt)) => { 
        crate::ast::Expr::binary_op(expr!($left), crate::ast::BinaryOperator::Add, expr!($right)) 
    };
    ((- $left:tt $right:tt)) => { 
        crate::ast::Expr::binary_op(expr!($left), crate::ast::BinaryOperator::Subtract, expr!($right)) 
    };
    ((* $left:tt $right:tt)) => { 
        crate::ast::Expr::binary_op(expr!($left), crate::ast::BinaryOperator::Multiply, expr!($right)) 
    };
    ((/ $left:tt $right:tt)) => { 
        crate::ast::Expr::binary_op(expr!($left), crate::ast::BinaryOperator::Divide, expr!($right)) 
    };
    
    ((== $left:tt $right:tt)) => { 
        crate::ast::Expr::binary_op(expr!($left), crate::ast::BinaryOperator::Equal, expr!($right)) 
    };
    ((!= $left:tt $right:tt)) => { 
        crate::ast::Expr::binary_op(expr!($left), crate::ast::BinaryOperator::NotEqual, expr!($right)) 
    };
    ((< $left:tt $right:tt)) => { 
        crate::ast::Expr::binary_op(expr!($left), crate::ast::BinaryOperator::LessThan, expr!($right)) 
    };
    ((<= $left:tt $right:tt)) => { 
        crate::ast::Expr::binary_op(expr!($left), crate::ast::BinaryOperator::LessThanOrEqual, expr!($right)) 
    };
    ((> $left:tt $right:tt)) => { 
        crate::ast::Expr::binary_op(expr!($left), crate::ast::BinaryOperator::GreaterThan, expr!($right)) 
    };
    ((>= $left:tt $right:tt)) => { 
        crate::ast::Expr::binary_op(expr!($left), crate::ast::BinaryOperator::GreaterThanOrEqual, expr!($right)) 
    };
    
    // Unary operations
    ((neg $operand:tt)) => { 
        crate::ast::Expr::unary_op(crate::ast::UnaryOperator::Negate, expr!($operand)) 
    };
    ((not $operand:tt)) => { 
        crate::ast::Expr::unary_op(crate::ast::UnaryOperator::Not, expr!($operand)) 
    };
    
    // Variables
    ((var $name:ident)) => { 
        crate::ast::Expr::variable(stringify!($name)) 
    };
    
    // Literals
    ($int:literal) => { crate::ast::Expr::literal($int) };
}

#[macro_export]
macro_rules! stmt {
    ((set $var:ident $value:tt)) => {
        crate::ast::Stmt::assignment(stringify!($var), expr!($value))
    };
    
    ((if $condition:tt $then_branch:tt)) => {
        crate::ast::Stmt::if_stmt(expr!($condition), vec![stmt!($then_branch)], None)
    };
    
    ((if $condition:tt $then_branch:tt $else_branch:tt)) => {
        crate::ast::Stmt::if_stmt(
            expr!($condition), 
            vec![stmt!($then_branch)], 
            Some(vec![stmt!($else_branch)])
        )
    };
    
    ((while $condition:tt $body:tt)) => {
        crate::ast::Stmt::while_stmt(expr!($condition), vec![stmt!($body)])
    };
    
    ((begin $($stmts:tt)*)) => {
        vec![$(stmt!($stmts)),*]
    };
    
    ($expr:tt) => {
        crate::ast::Stmt::expression(expr!($expr))
    };
}

#[macro_export]
macro_rules! program {
    ($($stmt:tt)*) => {
        crate::ast::Program {
            statements: vec![$(stmt!($stmt)),*]
        }
    };
}