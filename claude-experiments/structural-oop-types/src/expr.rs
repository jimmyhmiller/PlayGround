//! Expression AST for the object-oriented language
//!
//! This represents expressions in a small language with:
//! - Literals (bool, int)
//! - Variables and let bindings
//! - Functions (lambda) and application
//! - Objects with methods
//! - Field access
//! - Self-reference (this)

/// An expression in the language
#[derive(Debug, Clone)]
pub enum Expr {
    // === Literals ===
    /// Boolean literal
    Bool(bool),
    /// Integer literal
    Int(i64),
    /// String literal
    String(String),

    // === Variables ===
    /// Variable reference
    Var(String),

    // === Functions ===
    /// Lambda abstraction: λx. body
    Lambda(String, Box<Expr>),
    /// Function application: f(arg)
    App(Box<Expr>, Box<Expr>),

    // === Objects ===
    /// Object literal: { method1 = expr1, method2 = expr2, ... }
    /// Each method body has access to `this` referring to the object itself
    Object(Vec<(String, Expr)>),
    /// Field access: expr.field
    FieldAccess(Box<Expr>, String),
    /// Self-reference within an object
    This,

    // === Control ===
    /// Conditional: if cond then t else e
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// Let binding: let x = e1 in e2
    Let(String, Box<Expr>, Box<Expr>),
    /// Recursive let binding: let rec x = e1 in e2
    LetRec(String, Box<Expr>, Box<Expr>),
    /// Mutually recursive let bindings: let rec x1 = e1 and x2 = e2 in body
    LetRecMutual(Vec<(String, Expr)>, Box<Expr>),

    // === Binary Operators ===
    /// Equality: e1 == e2 (works on int and string)
    Eq(Box<Expr>, Box<Expr>),
    /// Boolean AND: e1 && e2
    And(Box<Expr>, Box<Expr>),
    /// Boolean OR: e1 || e2
    Or(Box<Expr>, Box<Expr>),
    /// Integer addition: e1 + e2
    Add(Box<Expr>, Box<Expr>),
    /// Integer subtraction: e1 - e2
    Sub(Box<Expr>, Box<Expr>),
    /// Integer multiplication: e1 * e2
    Mul(Box<Expr>, Box<Expr>),
    /// Integer division: e1 / e2
    Div(Box<Expr>, Box<Expr>),
    /// String concatenation: e1 ++ e2
    Concat(Box<Expr>, Box<Expr>),
}

impl Expr {
    // === Constructors for cleaner AST building ===

    pub fn bool(b: bool) -> Self {
        Expr::Bool(b)
    }

    pub fn int(n: i64) -> Self {
        Expr::Int(n)
    }

    pub fn string(s: impl Into<String>) -> Self {
        Expr::String(s.into())
    }

    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var(name.into())
    }

    pub fn lambda(param: impl Into<String>, body: Expr) -> Self {
        Expr::Lambda(param.into(), Box::new(body))
    }

    pub fn app(func: Expr, arg: Expr) -> Self {
        Expr::App(Box::new(func), Box::new(arg))
    }

    pub fn object(methods: Vec<(impl Into<String>, Expr)>) -> Self {
        Expr::Object(methods.into_iter().map(|(n, e)| (n.into(), e)).collect())
    }

    pub fn field(expr: Expr, field: impl Into<String>) -> Self {
        Expr::FieldAccess(Box::new(expr), field.into())
    }

    pub fn this() -> Self {
        Expr::This
    }

    pub fn if_(cond: Expr, then_: Expr, else_: Expr) -> Self {
        Expr::If(Box::new(cond), Box::new(then_), Box::new(else_))
    }

    pub fn let_(name: impl Into<String>, value: Expr, body: Expr) -> Self {
        Expr::Let(name.into(), Box::new(value), Box::new(body))
    }

    pub fn let_rec(name: impl Into<String>, value: Expr, body: Expr) -> Self {
        Expr::LetRec(name.into(), Box::new(value), Box::new(body))
    }

    pub fn eq(left: Expr, right: Expr) -> Self {
        Expr::Eq(Box::new(left), Box::new(right))
    }

    pub fn and(left: Expr, right: Expr) -> Self {
        Expr::And(Box::new(left), Box::new(right))
    }

    pub fn or(left: Expr, right: Expr) -> Self {
        Expr::Or(Box::new(left), Box::new(right))
    }

    // === Multi-argument helpers ===

    /// Create a multi-argument lambda: λx. λy. λz. body
    pub fn lambda_n(params: &[&str], body: Expr) -> Self {
        params
            .iter()
            .rev()
            .fold(body, |acc, &p| Expr::lambda(p, acc))
    }

    /// Create a multi-argument application: f(x)(y)(z)
    pub fn app_n(func: Expr, args: Vec<Expr>) -> Self {
        args.into_iter().fold(func, |f, arg| Expr::app(f, arg))
    }
}

/// Helper macro for building objects more concisely
///
/// Usage:
/// ```
/// object! {
///     "isEmpty" => Expr::bool(true),
///     "contains" => Expr::lambda("i", Expr::bool(false)),
/// }
/// ```
#[macro_export]
macro_rules! object {
    ($($field:expr => $value:expr),* $(,)?) => {
        Expr::Object(vec![
            $(($field.to_string(), $value)),*
        ])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_object() {
        let empty = Expr::object(vec![
            ("isEmpty", Expr::bool(true)),
            ("contains", Expr::lambda("i", Expr::bool(false))),
        ]);

        match empty {
            Expr::Object(methods) => {
                assert_eq!(methods.len(), 2);
                assert_eq!(methods[0].0, "isEmpty");
                assert_eq!(methods[1].0, "contains");
            }
            _ => panic!("Expected Object"),
        }
    }

    #[test]
    fn test_lambda_n() {
        let f = Expr::lambda_n(&["x", "y", "z"], Expr::var("x"));
        // Should be λx. λy. λz. x
        match f {
            Expr::Lambda(x, body) => {
                assert_eq!(x, "x");
                match *body {
                    Expr::Lambda(y, body2) => {
                        assert_eq!(y, "y");
                        match *body2 {
                            Expr::Lambda(z, _) => assert_eq!(z, "z"),
                            _ => panic!("Expected inner lambda"),
                        }
                    }
                    _ => panic!("Expected middle lambda"),
                }
            }
            _ => panic!("Expected outer lambda"),
        }
    }
}
