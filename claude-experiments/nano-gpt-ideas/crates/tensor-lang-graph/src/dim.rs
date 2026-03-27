use std::collections::HashMap;
use std::fmt;

/// A dimension value that may be concrete or symbolic.
///
/// Used in tensor shapes, strides, and buffer sizes. `Lit` values are known
/// at compile time. `Param` values are supplied at runtime. Arithmetic
/// variants represent computed dimensions (e.g., strides = product of
/// trailing dims).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    /// A compile-time constant.
    Lit(usize),
    /// A runtime parameter, e.g. "T" for sequence length.
    Param(String),
    /// Sum of two dimensions.
    Add(Box<Dim>, Box<Dim>),
    /// Product of two dimensions.
    Mul(Box<Dim>, Box<Dim>),
    /// Integer division (floor).
    Div(Box<Dim>, Box<Dim>),
    /// Subtraction.
    Sub(Box<Dim>, Box<Dim>),
}

impl Dim {
    /// True only for `Lit(1)`. Used for broadcast detection.
    pub fn is_one(&self) -> bool {
        matches!(self, Dim::Lit(1))
    }

    /// True only for `Lit(0)`.
    pub fn is_zero(&self) -> bool {
        matches!(self, Dim::Lit(0))
    }

    /// Extract a concrete value, if this is `Lit`.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Dim::Lit(n) => Some(*n),
            _ => None,
        }
    }

    /// True if this is a compile-time constant.
    pub fn is_lit(&self) -> bool {
        matches!(self, Dim::Lit(_))
    }

    /// True if this contains any `Param`.
    pub fn is_symbolic(&self) -> bool {
        match self {
            Dim::Lit(_) => false,
            Dim::Param(_) => true,
            Dim::Add(a, b) | Dim::Mul(a, b) | Dim::Div(a, b) | Dim::Sub(a, b) => {
                a.is_symbolic() || b.is_symbolic()
            }
        }
    }

    /// Evaluate with concrete parameter values.
    pub fn eval(&self, params: &HashMap<String, usize>) -> usize {
        match self {
            Dim::Lit(n) => *n,
            Dim::Param(name) => *params
                .get(name)
                .unwrap_or_else(|| panic!("missing dim param: {name}")),
            Dim::Add(a, b) => a.eval(params) + b.eval(params),
            Dim::Mul(a, b) => a.eval(params) * b.eval(params),
            Dim::Div(a, b) => a.eval(params) / b.eval(params),
            Dim::Sub(a, b) => a.eval(params) - b.eval(params),
        }
    }

    /// Render as an AssemblyScript expression.
    pub fn to_code(&self) -> String {
        match self {
            Dim::Lit(n) => format!("{n}"),
            Dim::Param(name) => name.clone(),
            Dim::Add(a, b) => format!("({} + {})", a.to_code(), b.to_code()),
            Dim::Sub(a, b) => format!("({} - {})", a.to_code(), b.to_code()),
            Dim::Mul(a, b) => {
                // Simplify display: Lit(1) * x => x
                if a.is_one() {
                    return b.to_code();
                }
                if b.is_one() {
                    return a.to_code();
                }
                format!("({} * {})", a.to_code(), b.to_code())
            }
            Dim::Div(a, b) => format!("({} / {})", a.to_code(), b.to_code()),
        }
    }

    /// Algebraic simplification. Folds constants, eliminates identity operations.
    pub fn simplify(&self) -> Dim {
        match self {
            Dim::Lit(_) | Dim::Param(_) => self.clone(),
            Dim::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Dim::Lit(x), Dim::Lit(y)) => Dim::Lit(x + y),
                    (Dim::Lit(0), _) => b,
                    (_, Dim::Lit(0)) => a,
                    _ => Dim::Add(Box::new(a), Box::new(b)),
                }
            }
            Dim::Sub(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Dim::Lit(x), Dim::Lit(y)) => Dim::Lit(x - y),
                    (_, Dim::Lit(0)) => a,
                    _ if a == b => Dim::Lit(0),
                    _ => Dim::Sub(Box::new(a), Box::new(b)),
                }
            }
            Dim::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Dim::Lit(x), Dim::Lit(y)) => Dim::Lit(x * y),
                    (Dim::Lit(0), _) | (_, Dim::Lit(0)) => Dim::Lit(0),
                    (Dim::Lit(1), _) => b,
                    (_, Dim::Lit(1)) => a,
                    _ => Dim::Mul(Box::new(a), Box::new(b)),
                }
            }
            Dim::Div(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Dim::Lit(x), Dim::Lit(y)) => Dim::Lit(x / y),
                    (Dim::Lit(0), _) => Dim::Lit(0),
                    (_, Dim::Lit(1)) => a,
                    _ if a == b => Dim::Lit(1),
                    _ => Dim::Div(Box::new(a), Box::new(b)),
                }
            }
        }
    }

    /// Compute the product of a list of dimensions, folding literals.
    pub fn product(dims: &[Dim]) -> Dim {
        if dims.is_empty() {
            return Dim::Lit(1);
        }
        let mut result = dims[0].clone();
        for d in &dims[1..] {
            result = Dim::Mul(Box::new(result), Box::new(d.clone())).simplify();
        }
        result
    }

    /// Compute row-major strides for a shape.
    /// strides[i] = product of dims[i+1..]
    pub fn strides(shape: &[Dim]) -> Vec<Dim> {
        let n = shape.len();
        let mut s = vec![Dim::Lit(1); n];
        for i in (0..n.saturating_sub(1)).rev() {
            s[i] = Dim::Mul(Box::new(s[i + 1].clone()), Box::new(shape[i + 1].clone())).simplify();
        }
        s
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Lit(n) => write!(f, "{n}"),
            Dim::Param(name) => write!(f, "{name}"),
            Dim::Add(a, b) => write!(f, "({a} + {b})"),
            Dim::Sub(a, b) => write!(f, "({a} - {b})"),
            Dim::Mul(a, b) => write!(f, "({a} * {b})"),
            Dim::Div(a, b) => write!(f, "({a} / {b})"),
        }
    }
}

/// Convenience: convert a concrete usize to Dim::Lit.
impl From<usize> for Dim {
    fn from(n: usize) -> Self {
        Dim::Lit(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lit_basics() {
        let d = Dim::Lit(768);
        assert!(d.is_lit());
        assert!(!d.is_one());
        assert!(!d.is_zero());
        assert_eq!(d.as_usize(), Some(768));
        assert!(!d.is_symbolic());
        assert_eq!(d.to_code(), "768");
    }

    #[test]
    fn test_param_basics() {
        let d = Dim::Param("T".into());
        assert!(!d.is_lit());
        assert!(!d.is_one());
        assert_eq!(d.as_usize(), None);
        assert!(d.is_symbolic());
        assert_eq!(d.to_code(), "T");
    }

    #[test]
    fn test_eval() {
        let expr = Dim::Mul(
            Box::new(Dim::Param("T".into())),
            Box::new(Dim::Lit(768)),
        );
        let mut params = HashMap::new();
        params.insert("T".into(), 3);
        assert_eq!(expr.eval(&params), 2304);
    }

    #[test]
    fn test_simplify_lit_arithmetic() {
        let a = Dim::Add(Box::new(Dim::Lit(3)), Box::new(Dim::Lit(5)));
        assert_eq!(a.simplify(), Dim::Lit(8));

        let m = Dim::Mul(Box::new(Dim::Lit(4)), Box::new(Dim::Lit(768)));
        assert_eq!(m.simplify(), Dim::Lit(3072));

        let d = Dim::Div(Box::new(Dim::Lit(768)), Box::new(Dim::Lit(12)));
        assert_eq!(d.simplify(), Dim::Lit(64));
    }

    #[test]
    fn test_simplify_identity() {
        let t = Dim::Param("T".into());
        // 1 * T => T
        assert_eq!(
            Dim::Mul(Box::new(Dim::Lit(1)), Box::new(t.clone())).simplify(),
            t
        );
        // T * 1 => T
        assert_eq!(
            Dim::Mul(Box::new(t.clone()), Box::new(Dim::Lit(1))).simplify(),
            t
        );
        // 0 + T => T
        assert_eq!(
            Dim::Add(Box::new(Dim::Lit(0)), Box::new(t.clone())).simplify(),
            t
        );
        // T - 0 => T
        assert_eq!(
            Dim::Sub(Box::new(t.clone()), Box::new(Dim::Lit(0))).simplify(),
            t
        );
        // 0 * T => 0
        assert_eq!(
            Dim::Mul(Box::new(Dim::Lit(0)), Box::new(t.clone())).simplify(),
            Dim::Lit(0)
        );
        // T / T => 1
        assert_eq!(
            Dim::Div(Box::new(t.clone()), Box::new(t.clone())).simplify(),
            Dim::Lit(1)
        );
    }

    #[test]
    fn test_to_code_mul_identity() {
        // Lit(1) * Param("T") should render as just "T"
        let expr = Dim::Mul(Box::new(Dim::Lit(1)), Box::new(Dim::Param("T".into())));
        assert_eq!(expr.to_code(), "T");
    }

    #[test]
    fn test_to_code_complex() {
        // T * 768
        let expr = Dim::Mul(
            Box::new(Dim::Param("T".into())),
            Box::new(Dim::Lit(768)),
        );
        assert_eq!(expr.to_code(), "(T * 768)");
    }

    #[test]
    fn test_product() {
        // Product of [1, T, 768] = T * 768
        let shape = vec![Dim::Lit(1), Dim::Param("T".into()), Dim::Lit(768)];
        let p = Dim::product(&shape);
        let mut params = HashMap::new();
        params.insert("T".into(), 3);
        assert_eq!(p.eval(&params), 2304);
    }

    #[test]
    fn test_product_all_lit() {
        let shape = vec![Dim::Lit(2), Dim::Lit(3), Dim::Lit(4)];
        let p = Dim::product(&shape);
        assert_eq!(p, Dim::Lit(24));
    }

    #[test]
    fn test_strides() {
        // Shape [B, T, 768] where B=1
        let shape = vec![Dim::Lit(1), Dim::Param("T".into()), Dim::Lit(768)];
        let s = Dim::strides(&shape);
        assert_eq!(s.len(), 3);
        // stride[2] = 1
        assert_eq!(s[2], Dim::Lit(1));
        // stride[1] = 768
        assert_eq!(s[1], Dim::Lit(768));
        // stride[0] = T * 768
        let mut params = HashMap::new();
        params.insert("T".into(), 5);
        assert_eq!(s[0].eval(&params), 5 * 768);
    }

    #[test]
    fn test_strides_all_lit() {
        let shape = vec![Dim::Lit(2), Dim::Lit(3), Dim::Lit(4)];
        let s = Dim::strides(&shape);
        assert_eq!(s[0], Dim::Lit(12));
        assert_eq!(s[1], Dim::Lit(4));
        assert_eq!(s[2], Dim::Lit(1));
    }

    #[test]
    fn test_quadratic_eval() {
        // T * T * 12 (attention score buffer)
        let expr = Dim::Mul(
            Box::new(Dim::Mul(
                Box::new(Dim::Param("T".into())),
                Box::new(Dim::Param("T".into())),
            )),
            Box::new(Dim::Lit(12)),
        );
        let mut params = HashMap::new();
        params.insert("T".into(), 10);
        assert_eq!(expr.eval(&params), 1200);
    }

    #[test]
    fn test_display() {
        let d = Dim::Mul(
            Box::new(Dim::Param("T".into())),
            Box::new(Dim::Lit(768)),
        );
        assert_eq!(format!("{d}"), "(T * 768)");
    }
}
