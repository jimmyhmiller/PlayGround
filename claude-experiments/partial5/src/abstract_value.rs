//! Abstract values for partial evaluation
//!
//! The key insight: we don't need to track ALL possible values, just whether
//! we KNOW the value at partial evaluation time or not.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use swc_ecma_ast as ast;
use swc_ecma_ast::{BinaryOp, UnaryOp};

/// A JavaScript value that we know at partial evaluation time
#[derive(Debug, Clone)]
pub enum JsValue {
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Rc<RefCell<Vec<AbstractValue>>>),
    Object(Rc<RefCell<HashMap<String, AbstractValue>>>),
    Function(FunctionValue),
}

/// A function value - either a known function body or an opaque reference
#[derive(Debug, Clone)]
pub enum FunctionValue {
    /// A function whose body we can inline/trace
    Known {
        params: Vec<String>,
        body: ast::BlockStmt,
    },
    /// A reference to a function in a dispatch table (e.g., arr[N])
    DispatchHandler(String, usize),
    /// An opaque function we can't trace into
    Opaque(String),
}

/// Abstract value - either known statically or dynamic
#[derive(Debug, Clone)]
pub enum AbstractValue {
    /// Value is known at partial evaluation time
    Known(JsValue),
    /// Value is dynamic - we have an expression that computes it
    /// The String is a description/debug name, the Expr is optional residual code
    Dynamic(String, Option<Box<ast::Expr>>),
    /// Value could be anything - we've lost track
    Top,
}

impl AbstractValue {
    pub fn known_number(n: f64) -> Self {
        AbstractValue::Known(JsValue::Number(n))
    }

    pub fn known_bool(b: bool) -> Self {
        AbstractValue::Known(JsValue::Bool(b))
    }

    pub fn known_string(s: String) -> Self {
        AbstractValue::Known(JsValue::String(s))
    }

    pub fn known_undefined() -> Self {
        AbstractValue::Known(JsValue::Undefined)
    }

    pub fn known_null() -> Self {
        AbstractValue::Known(JsValue::Null)
    }

    pub fn known_array(elements: Vec<AbstractValue>) -> Self {
        AbstractValue::Known(JsValue::Array(Rc::new(RefCell::new(elements))))
    }

    pub fn known_object(props: HashMap<String, AbstractValue>) -> Self {
        AbstractValue::Known(JsValue::Object(Rc::new(RefCell::new(props))))
    }

    pub fn dynamic(name: &str) -> Self {
        AbstractValue::Dynamic(name.to_string(), None)
    }

    pub fn dynamic_with_expr(name: &str, expr: ast::Expr) -> Self {
        AbstractValue::Dynamic(name.to_string(), Some(Box::new(expr)))
    }

    /// Check if the value is statically known
    pub fn is_known(&self) -> bool {
        matches!(self, AbstractValue::Known(_))
    }

    /// Check if the value is dynamic
    pub fn is_dynamic(&self) -> bool {
        matches!(self, AbstractValue::Dynamic(_, _) | AbstractValue::Top)
    }

    /// Try to get as a known number
    pub fn as_number(&self) -> Option<f64> {
        match self {
            AbstractValue::Known(JsValue::Number(n)) => Some(*n),
            _ => None,
        }
    }

    /// Try to get as a known integer
    pub fn as_int(&self) -> Option<i64> {
        self.as_number().map(|n| n as i64)
    }

    /// Try to get as a known boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            AbstractValue::Known(JsValue::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as a known string
    pub fn as_string(&self) -> Option<&String> {
        match self {
            AbstractValue::Known(JsValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Try to get as a known array (shared reference)
    pub fn as_array(&self) -> Option<Rc<RefCell<Vec<AbstractValue>>>> {
        match self {
            AbstractValue::Known(JsValue::Array(arr)) => Some(arr.clone()),
            _ => None,
        }
    }

    /// Try to get as a known object (shared reference)
    pub fn as_object(&self) -> Option<Rc<RefCell<HashMap<String, AbstractValue>>>> {
        match self {
            AbstractValue::Known(JsValue::Object(obj)) => Some(obj.clone()),
            _ => None,
        }
    }

    /// Try to get as a known function
    pub fn as_function(&self) -> Option<&FunctionValue> {
        match self {
            AbstractValue::Known(JsValue::Function(f)) => Some(f),
            _ => None,
        }
    }

    /// Convert to JavaScript-like truthiness
    pub fn is_truthy(&self) -> Option<bool> {
        match self {
            AbstractValue::Known(JsValue::Undefined) => Some(false),
            AbstractValue::Known(JsValue::Null) => Some(false),
            AbstractValue::Known(JsValue::Bool(b)) => Some(*b),
            AbstractValue::Known(JsValue::Number(n)) => Some(*n != 0.0 && !n.is_nan()),
            AbstractValue::Known(JsValue::String(s)) => Some(!s.is_empty()),
            AbstractValue::Known(JsValue::Array(_)) => Some(true),
            AbstractValue::Known(JsValue::Object(_)) => Some(true),
            AbstractValue::Known(JsValue::Function(_)) => Some(true),
            _ => None, // Dynamic values - we don't know
        }
    }

    /// Convert abstract value to an AST expression for residual code generation
    pub fn to_expr(&self) -> ast::Expr {
        crate::emit::value_to_expr(self)
    }
}

impl fmt::Display for AbstractValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbstractValue::Known(v) => write!(f, "{}", v),
            AbstractValue::Dynamic(name, _) => write!(f, "?{}", name),
            AbstractValue::Top => write!(f, "⊤"),
        }
    }
}

impl fmt::Display for JsValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsValue::Undefined => write!(f, "undefined"),
            JsValue::Null => write!(f, "null"),
            JsValue::Bool(b) => write!(f, "{}", b),
            JsValue::Number(n) => write!(f, "{}", n),
            JsValue::String(s) => write!(f, "\"{}\"", s),
            JsValue::Array(arr) => {
                write!(f, "[")?;
                let arr = arr.borrow();
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    if i > 5 {
                        write!(f, "...")?;
                        break;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            JsValue::Object(obj) => {
                write!(f, "{{")?;
                let obj = obj.borrow();
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    if i > 3 {
                        write!(f, "...")?;
                        break;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            JsValue::Function(fv) => match fv {
                FunctionValue::Known { params, .. } => {
                    write!(f, "fn({})", params.join(", "))
                }
                FunctionValue::DispatchHandler(name, n) => write!(f, "{}[{}]", name, n),
                FunctionValue::Opaque(name) => write!(f, "<fn:{}>", name),
            },
        }
    }
}

/// Helper to coerce a JsValue to i32 for bitwise operations (JavaScript semantics)
fn js_to_int32(val: &JsValue) -> i32 {
    match val {
        JsValue::Number(n) => {
            // JavaScript ToInt32: NaN and Infinity become 0
            if n.is_nan() || n.is_infinite() {
                0
            } else {
                *n as i32
            }
        }
        JsValue::Bool(b) => if *b { 1 } else { 0 },
        JsValue::String(s) => {
            // Try to parse as number, otherwise NaN -> 0
            s.parse::<f64>().map(|n| n as i32).unwrap_or(0)
        }
        JsValue::Null => 0,
        JsValue::Undefined => 0,
        // Arrays and objects coerce to NaN -> 0
        _ => 0,
    }
}

/// Wrap a CondExpr in parens to avoid precedence issues when used as a binary operand
fn paren_if_cond(expr: ast::Expr) -> ast::Expr {
    if matches!(&expr, ast::Expr::Cond(_)) {
        ast::Expr::Paren(ast::ParenExpr {
            span: Default::default(),
            expr: Box::new(expr),
        })
    } else {
        expr
    }
}

/// Helper to create a binary expression for residual code
fn make_binop_expr(op: BinaryOp, left: &AbstractValue, right: &AbstractValue) -> ast::Expr {
    ast::Expr::Bin(ast::BinExpr {
        span: Default::default(),
        op,
        left: Box::new(paren_if_cond(left.to_expr())),
        right: Box::new(paren_if_cond(right.to_expr())),
    })
}

/// Helper to create a unary expression for residual code
fn make_unop_expr(op: UnaryOp, arg: &AbstractValue) -> ast::Expr {
    ast::Expr::Unary(ast::UnaryExpr {
        span: Default::default(),
        op,
        arg: Box::new(arg.to_expr()),
    })
}

/// Binary operations on abstract values
impl AbstractValue {
    pub fn add(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_number(a + b)
            }
            (AbstractValue::Known(JsValue::String(a)), AbstractValue::Known(JsValue::String(b))) => {
                AbstractValue::known_string(format!("{}{}", a, b))
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::Add, self, other);
                AbstractValue::dynamic_with_expr("add_result", expr)
            }
        }
    }

    pub fn sub(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_number(a - b)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::Sub, self, other);
                AbstractValue::dynamic_with_expr("sub_result", expr)
            }
        }
    }

    pub fn mul(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_number(a * b)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::Mul, self, other);
                AbstractValue::dynamic_with_expr("mul_result", expr)
            }
        }
    }

    pub fn div(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_number(a / b)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::Div, self, other);
                AbstractValue::dynamic_with_expr("div_result", expr)
            }
        }
    }

    pub fn rem(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_number(a % b)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::Mod, self, other);
                AbstractValue::dynamic_with_expr("rem_result", expr)
            }
        }
    }

    pub fn bitand(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(a), AbstractValue::Known(b)) => {
                let a_int = js_to_int32(a);
                let b_int = js_to_int32(b);
                AbstractValue::known_number((a_int & b_int) as f64)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::BitAnd, self, other);
                AbstractValue::dynamic_with_expr("bitand_result", expr)
            }
        }
    }

    pub fn bitor(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(a), AbstractValue::Known(b)) => {
                let a_int = js_to_int32(a);
                let b_int = js_to_int32(b);
                AbstractValue::known_number((a_int | b_int) as f64)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::BitOr, self, other);
                AbstractValue::dynamic_with_expr("bitor_result", expr)
            }
        }
    }

    pub fn bitxor(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(a), AbstractValue::Known(b)) => {
                let a_int = js_to_int32(a);
                let b_int = js_to_int32(b);
                AbstractValue::known_number((a_int ^ b_int) as f64)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::BitXor, self, other);
                AbstractValue::dynamic_with_expr("bitxor_result", expr)
            }
        }
    }

    pub fn lshift(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(a), AbstractValue::Known(b)) => {
                let a_int = js_to_int32(a);
                let b_int = js_to_int32(b) as u32;
                AbstractValue::known_number((a_int << (b_int & 0x1f)) as f64)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::LShift, self, other);
                AbstractValue::dynamic_with_expr("lshift_result", expr)
            }
        }
    }

    pub fn rshift(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(a), AbstractValue::Known(b)) => {
                let a_int = js_to_int32(a);
                let b_int = js_to_int32(b) as u32;
                AbstractValue::known_number((a_int >> (b_int & 0x1f)) as f64)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::RShift, self, other);
                AbstractValue::dynamic_with_expr("rshift_result", expr)
            }
        }
    }

    pub fn lt(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_bool(a < b)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::Lt, self, other);
                AbstractValue::dynamic_with_expr("lt_result", expr)
            }
        }
    }

    pub fn le(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_bool(a <= b)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::LtEq, self, other);
                AbstractValue::dynamic_with_expr("le_result", expr)
            }
        }
    }

    pub fn gt(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_bool(a > b)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::Gt, self, other);
                AbstractValue::dynamic_with_expr("gt_result", expr)
            }
        }
    }

    pub fn ge(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_bool(a >= b)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::GtEq, self, other);
                AbstractValue::dynamic_with_expr("ge_result", expr)
            }
        }
    }

    pub fn eq(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_bool((a - b).abs() < f64::EPSILON)
            }
            (AbstractValue::Known(JsValue::String(a)), AbstractValue::Known(JsValue::String(b))) => {
                AbstractValue::known_bool(a == b)
            }
            (AbstractValue::Known(JsValue::Bool(a)), AbstractValue::Known(JsValue::Bool(b))) => {
                AbstractValue::known_bool(a == b)
            }
            (AbstractValue::Known(JsValue::Null), AbstractValue::Known(JsValue::Null)) => {
                AbstractValue::known_bool(true)
            }
            (AbstractValue::Known(JsValue::Undefined), AbstractValue::Known(JsValue::Undefined)) => {
                AbstractValue::known_bool(true)
            }
            // Different known types are not strictly equal
            (AbstractValue::Known(_), AbstractValue::Known(_)) => {
                AbstractValue::known_bool(false)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::EqEq, self, other);
                AbstractValue::dynamic_with_expr("eq_result", expr)
            }
        }
    }

    pub fn strict_eq(&self, other: &AbstractValue) -> AbstractValue {
        match (self, other) {
            (AbstractValue::Known(JsValue::Number(a)), AbstractValue::Known(JsValue::Number(b))) => {
                AbstractValue::known_bool((a - b).abs() < f64::EPSILON)
            }
            (AbstractValue::Known(JsValue::String(a)), AbstractValue::Known(JsValue::String(b))) => {
                AbstractValue::known_bool(a == b)
            }
            (AbstractValue::Known(JsValue::Bool(a)), AbstractValue::Known(JsValue::Bool(b))) => {
                AbstractValue::known_bool(a == b)
            }
            (AbstractValue::Known(JsValue::Null), AbstractValue::Known(JsValue::Null)) => {
                AbstractValue::known_bool(true)
            }
            (AbstractValue::Known(JsValue::Undefined), AbstractValue::Known(JsValue::Undefined)) => {
                AbstractValue::known_bool(true)
            }
            // Different known types are not strictly equal
            (AbstractValue::Known(_), AbstractValue::Known(_)) => {
                AbstractValue::known_bool(false)
            }
            _ => {
                let expr = make_binop_expr(BinaryOp::EqEqEq, self, other);
                AbstractValue::dynamic_with_expr("strict_eq_result", expr)
            }
        }
    }

    pub fn neq(&self, other: &AbstractValue) -> AbstractValue {
        match self.eq(other) {
            AbstractValue::Known(JsValue::Bool(b)) => AbstractValue::known_bool(!b),
            _ => {
                let expr = make_binop_expr(BinaryOp::NotEq, self, other);
                AbstractValue::dynamic_with_expr("neq_result", expr)
            }
        }
    }

    pub fn not(&self) -> AbstractValue {
        match self.is_truthy() {
            Some(b) => AbstractValue::known_bool(!b),
            None => {
                let expr = make_unop_expr(UnaryOp::Bang, self);
                AbstractValue::dynamic_with_expr("not_result", expr)
            }
        }
    }

    pub fn bitnot(&self) -> AbstractValue {
        match self {
            AbstractValue::Known(JsValue::Number(n)) => {
                AbstractValue::known_number((!(*n as i32)) as f64)
            }
            _ => {
                let expr = make_unop_expr(UnaryOp::Tilde, self);
                AbstractValue::dynamic_with_expr("bitnot_result", expr)
            }
        }
    }

    pub fn neg(&self) -> AbstractValue {
        match self {
            AbstractValue::Known(JsValue::Number(n)) => AbstractValue::known_number(-n),
            _ => {
                let expr = make_unop_expr(UnaryOp::Minus, self);
                AbstractValue::dynamic_with_expr("neg_result", expr)
            }
        }
    }
}
