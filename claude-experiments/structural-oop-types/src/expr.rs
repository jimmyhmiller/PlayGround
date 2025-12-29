//! Expression AST for the object-oriented language
//!
//! This represents expressions in a small language with:
//! - Literals (bool, int)
//! - Variables and let bindings
//! - Functions (lambda) and application
//! - Objects with methods (classes)
//! - Field access
//! - Self-reference (this)
//! - Spread operator for objects (...expr)

/// A field in an object literal - either a named field or a spread
#[derive(Debug, Clone)]
pub enum ObjectField {
    /// Named field: name: expr
    Field(String, Expr),
    /// Spread: ...expr (copies all fields from expr)
    Spread(Expr),
}

/// A class definition: class Name(params) { fields }
/// Fields can be named (field: expr) or spreads (...expr)
#[derive(Debug, Clone)]
pub struct ClassDef {
    pub name: String,
    pub params: Vec<String>,
    pub fields: Vec<ObjectField>,
}

impl ClassDef {
    pub fn new(name: impl Into<String>, params: Vec<String>, fields: Vec<ObjectField>) -> Self {
        ClassDef {
            name: name.into(),
            params,
            fields,
        }
    }

    /// Create a ClassDef from plain fields (for backwards compatibility)
    pub fn from_plain_fields(name: impl Into<String>, params: Vec<String>, fields: Vec<(String, Expr)>) -> Self {
        let obj_fields = fields.into_iter()
            .map(|(n, e)| ObjectField::Field(n, e))
            .collect();
        ClassDef {
            name: name.into(),
            params,
            fields: obj_fields,
        }
    }

    /// Convert a class definition to a lambda that returns an object.
    /// class F(a, b) { x: e1, y: e2 } -> F = a => b => { x: e1, y: e2 }
    /// class F() { x: e1 } -> F = _unit => { x: e1 }
    pub fn to_lambda(&self) -> Expr {
        let obj = Expr::Object(self.fields.clone());

        // Handle zero-param class: wrap in a _unit lambda (thunk)
        if self.params.is_empty() {
            return Expr::Lambda("_unit".to_string(), Box::new(obj));
        }

        // Wrap in lambdas for each parameter (curried)
        self.params.iter().rev().fold(obj, |body, param| {
            Expr::Lambda(param.clone(), Box::new(body))
        })
    }
}

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
    /// Object literal: { method1 = expr1, method2 = expr2, ...spread, ... }
    /// Each method body has access to `this` referring to the object itself
    /// Fields can be named (Field) or spread (Spread) to copy all fields from another object
    Object(Vec<ObjectField>),
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
    /// Block with class definitions: { class A ... class B ... expr }
    /// All classes are mutually recursive within the block
    Block(Vec<ClassDef>, Box<Expr>),
    /// Multi-argument function call: f(a, b, c)
    Call(Box<Expr>, Vec<Expr>),

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
    /// Integer modulo: e1 % e2
    Mod(Box<Expr>, Box<Expr>),
    /// String concatenation: e1 ++ e2
    Concat(Box<Expr>, Box<Expr>),
    /// Less than: e1 < e2
    Lt(Box<Expr>, Box<Expr>),
    /// Less than or equal: e1 <= e2
    LtEq(Box<Expr>, Box<Expr>),
    /// Greater than: e1 > e2
    Gt(Box<Expr>, Box<Expr>),
    /// Greater than or equal: e1 >= e2
    GtEq(Box<Expr>, Box<Expr>),

    // === Unary Operators ===
    /// Boolean NOT: !e
    Not(Box<Expr>),
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
        Expr::Object(methods.into_iter().map(|(n, e)| ObjectField::Field(n.into(), e)).collect())
    }

    pub fn object_with_spreads(fields: Vec<ObjectField>) -> Self {
        Expr::Object(fields)
    }

    pub fn spread(expr: Expr) -> ObjectField {
        ObjectField::Spread(expr)
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

    pub fn block(classes: Vec<ClassDef>, body: Expr) -> Self {
        Expr::Block(classes, Box::new(body))
    }

    pub fn call(func: Expr, args: Vec<Expr>) -> Self {
        Expr::Call(Box::new(func), args)
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

    pub fn add(left: Expr, right: Expr) -> Self {
        Expr::Add(Box::new(left), Box::new(right))
    }

    pub fn sub(left: Expr, right: Expr) -> Self {
        Expr::Sub(Box::new(left), Box::new(right))
    }

    pub fn mul(left: Expr, right: Expr) -> Self {
        Expr::Mul(Box::new(left), Box::new(right))
    }

    pub fn div(left: Expr, right: Expr) -> Self {
        Expr::Div(Box::new(left), Box::new(right))
    }

    pub fn concat(left: Expr, right: Expr) -> Self {
        Expr::Concat(Box::new(left), Box::new(right))
    }

    pub fn mod_(left: Expr, right: Expr) -> Self {
        Expr::Mod(Box::new(left), Box::new(right))
    }

    pub fn lt(left: Expr, right: Expr) -> Self {
        Expr::Lt(Box::new(left), Box::new(right))
    }

    pub fn lt_eq(left: Expr, right: Expr) -> Self {
        Expr::LtEq(Box::new(left), Box::new(right))
    }

    pub fn gt(left: Expr, right: Expr) -> Self {
        Expr::Gt(Box::new(left), Box::new(right))
    }

    pub fn gt_eq(left: Expr, right: Expr) -> Self {
        Expr::GtEq(Box::new(left), Box::new(right))
    }

    pub fn not(expr: Expr) -> Self {
        Expr::Not(Box::new(expr))
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
            $(ObjectField::Field($field.to_string(), $value)),*
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
            Expr::Object(fields) => {
                assert_eq!(fields.len(), 2);
                match &fields[0] {
                    ObjectField::Field(name, _) => assert_eq!(name, "isEmpty"),
                    _ => panic!("Expected Field"),
                }
                match &fields[1] {
                    ObjectField::Field(name, _) => assert_eq!(name, "contains"),
                    _ => panic!("Expected Field"),
                }
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
