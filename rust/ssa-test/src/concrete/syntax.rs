//! Macro-based syntax for creating AST nodes.

#[macro_export]
macro_rules! ast {
    // Binary operations
    ((+ $left:tt $right:tt)) => {
        $crate::concrete::ast::Ast::binary_op(ast!($left), $crate::concrete::ast::BinaryOperator::Add, ast!($right))
    };
    ((- $left:tt $right:tt)) => {
        $crate::concrete::ast::Ast::binary_op(ast!($left), $crate::concrete::ast::BinaryOperator::Subtract, ast!($right))
    };
    ((* $left:tt $right:tt)) => {
        $crate::concrete::ast::Ast::binary_op(ast!($left), $crate::concrete::ast::BinaryOperator::Multiply, ast!($right))
    };
    ((/ $left:tt $right:tt)) => {
        $crate::concrete::ast::Ast::binary_op(ast!($left), $crate::concrete::ast::BinaryOperator::Divide, ast!($right))
    };

    ((== $left:tt $right:tt)) => {
        $crate::concrete::ast::Ast::binary_op(ast!($left), $crate::concrete::ast::BinaryOperator::Equal, ast!($right))
    };
    ((!= $left:tt $right:tt)) => {
        $crate::concrete::ast::Ast::binary_op(ast!($left), $crate::concrete::ast::BinaryOperator::NotEqual, ast!($right))
    };
    ((< $left:tt $right:tt)) => {
        $crate::concrete::ast::Ast::binary_op(ast!($left), $crate::concrete::ast::BinaryOperator::LessThan, ast!($right))
    };
    ((<= $left:tt $right:tt)) => {
        $crate::concrete::ast::Ast::binary_op(ast!($left), $crate::concrete::ast::BinaryOperator::LessThanOrEqual, ast!($right))
    };
    ((> $left:tt $right:tt)) => {
        $crate::concrete::ast::Ast::binary_op(ast!($left), $crate::concrete::ast::BinaryOperator::GreaterThan, ast!($right))
    };
    ((>= $left:tt $right:tt)) => {
        $crate::concrete::ast::Ast::binary_op(ast!($left), $crate::concrete::ast::BinaryOperator::GreaterThanOrEqual, ast!($right))
    };

    // Unary operations
    ((neg $operand:tt)) => {
        $crate::concrete::ast::Ast::unary_op($crate::concrete::ast::UnaryOperator::Negate, ast!($operand))
    };
    ((not $operand:tt)) => {
        $crate::concrete::ast::Ast::unary_op($crate::concrete::ast::UnaryOperator::Not, ast!($operand))
    };

    // Variables
    ((var $name:ident)) => {
        $crate::concrete::ast::Ast::variable(stringify!($name))
    };

    // Literals
    ($int:literal) => { $crate::concrete::ast::Ast::literal($int) };

    // Statements
    ((set $var:ident $value:tt)) => {
        $crate::concrete::ast::Ast::assignment(stringify!($var), ast!($value))
    };

    ((if $condition:tt $then_branch:tt)) => {
        $crate::concrete::ast::Ast::if_stmt(ast!($condition), vec![ast!($then_branch)], None)
    };

    ((if $condition:tt $then_branch:tt $else_branch:tt)) => {
        $crate::concrete::ast::Ast::if_stmt(
            ast!($condition),
            vec![ast!($then_branch)],
            Some(vec![ast!($else_branch)])
        )
    };

    ((while $condition:tt $body:tt)) => {
        $crate::concrete::ast::Ast::while_stmt(ast!($condition), vec![ast!($body)])
    };

    ((begin $($stmts:tt)*)) => {
        $crate::concrete::ast::Ast::block(vec![$(ast!($stmts)),*])
    };

    // Print statement
    ((print $value:tt)) => {
        $crate::concrete::ast::Ast::print(ast!($value))
    };
}

#[macro_export]
macro_rules! program {
    ($($stmt:tt)*) => {
        $crate::concrete::ast::Ast::Block(vec![$(ast!($stmt)),*])
    };
}
