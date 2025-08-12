#[macro_export]
macro_rules! ast {
    // Binary operations
    ((+ $left:tt $right:tt)) => {
        $crate::ast::Ast::binary_op(ast!($left), $crate::ast::BinaryOperator::Add, ast!($right))
    };
    ((- $left:tt $right:tt)) => {
        crate::ast::Ast::binary_op(ast!($left), crate::ast::BinaryOperator::Subtract, ast!($right))
    };
    ((* $left:tt $right:tt)) => {
        crate::ast::Ast::binary_op(ast!($left), crate::ast::BinaryOperator::Multiply, ast!($right))
    };
    ((/ $left:tt $right:tt)) => {
        crate::ast::Ast::binary_op(ast!($left), crate::ast::BinaryOperator::Divide, ast!($right))
    };

    ((== $left:tt $right:tt)) => {
        crate::ast::Ast::binary_op(ast!($left), crate::ast::BinaryOperator::Equal, ast!($right))
    };
    ((!= $left:tt $right:tt)) => {
        crate::ast::Ast::binary_op(ast!($left), crate::ast::BinaryOperator::NotEqual, ast!($right))
    };
    ((< $left:tt $right:tt)) => {
        crate::ast::Ast::binary_op(ast!($left), crate::ast::BinaryOperator::LessThan, ast!($right))
    };
    ((<= $left:tt $right:tt)) => {
        crate::ast::Ast::binary_op(ast!($left), crate::ast::BinaryOperator::LessThanOrEqual, ast!($right))
    };
    ((> $left:tt $right:tt)) => {
        crate::ast::Ast::binary_op(ast!($left), crate::ast::BinaryOperator::GreaterThan, ast!($right))
    };
    ((>= $left:tt $right:tt)) => {
        crate::ast::Ast::binary_op(ast!($left), crate::ast::BinaryOperator::GreaterThanOrEqual, ast!($right))
    };

    // Unary operations
    ((neg $operand:tt)) => {
        crate::ast::Ast::unary_op(crate::ast::UnaryOperator::Negate, ast!($operand))
    };
    ((not $operand:tt)) => {
        crate::ast::Ast::unary_op(crate::ast::UnaryOperator::Not, ast!($operand))
    };

    // Variables
    ((var $name:ident)) => {
        crate::ast::Ast::variable(stringify!($name))
    };

    // Literals
    ($int:literal) => { crate::ast::Ast::literal($int) };

    // Statements
    ((set $var:ident $value:tt)) => {
        crate::ast::Ast::assignment(stringify!($var), ast!($value))
    };

    ((if $condition:tt $then_branch:tt)) => {
        crate::ast::Ast::if_stmt(ast!($condition), vec![ast!($then_branch)], None)
    };

    ((if $condition:tt $then_branch:tt $else_branch:tt)) => {
        crate::ast::Ast::if_stmt(
            ast!($condition),
            vec![ast!($then_branch)],
            Some(vec![ast!($else_branch)])
        )
    };

    ((while $condition:tt $body:tt)) => {
        crate::ast::Ast::while_stmt(ast!($condition), vec![ast!($body)])
    };

    ((begin $($stmts:tt)*)) => {
        crate::ast::Ast::block(vec![$(ast!($stmts)),*])
    };

    // Print statement
    ((print $value:tt)) => {
        crate::ast::Ast::print(ast!($value))
    };
}

#[macro_export]
macro_rules! program {
    ($($stmt:tt)*) => {
        $crate::ast::Ast::Block(vec![$(ast!($stmt)),*])
    };
}
