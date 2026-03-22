/// Compile-time arithmetic operations on widths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComptimeOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// A compile-time expression representing a vector width.
///
/// Every width position in the language (type annotations, struct literals, etc.)
/// contains a `Width`. It is always evaluated to a concrete `u64` before codegen,
/// either immediately (for `Fixed`) or during monomorphization (for `Param`/`BinOp`).
#[derive(Debug, Clone, PartialEq)]
pub enum Width {
    /// A concrete integer literal: `[8]`, `[1024]`
    Fixed(u64),
    /// Native machine SIMD width: `[_]`
    /// Resolved to Fixed early in compilation, per target.
    Native,
    /// A reference to a comptime parameter: `[N]`
    Param(String),
    /// Arithmetic on comptime values: `[N * 2]`, `[N + M]`
    BinOp {
        op: ComptimeOp,
        lhs: Box<Width>,
        rhs: Box<Width>,
    },
}

impl Width {
    /// Evaluate this width expression to a concrete integer.
    ///
    /// `env` maps comptime parameter names to their concrete values.
    /// `native_width` is the target's SIMD width (for resolving `_`).
    ///
    /// Panics if a parameter is not found in `env`.
    pub fn eval(&self, env: &std::collections::HashMap<String, u64>, native_width: u64) -> u64 {
        match self {
            Width::Fixed(n) => *n,
            Width::Native => native_width,
            Width::Param(name) => *env
                .get(name)
                .unwrap_or_else(|| panic!("unbound comptime parameter: {}", name)),
            Width::BinOp { op, lhs, rhs } => {
                let l = lhs.eval(env, native_width);
                let r = rhs.eval(env, native_width);
                match op {
                    ComptimeOp::Add => l + r,
                    ComptimeOp::Sub => l.checked_sub(r).unwrap_or_else(|| {
                        panic!("comptime subtraction underflow: {} - {}", l, r)
                    }),
                    ComptimeOp::Mul => l * r,
                    ComptimeOp::Div => {
                        assert!(r != 0, "comptime division by zero");
                        l / r
                    }
                }
            }
        }
    }

    /// Returns true if this width is fully concrete (no params, no native).
    pub fn is_concrete(&self) -> bool {
        match self {
            Width::Fixed(_) => true,
            Width::Native => false,
            Width::Param(_) => false,
            Width::BinOp { lhs, rhs, .. } => lhs.is_concrete() && rhs.is_concrete(),
        }
    }

    /// Collect all comptime parameter names referenced in this width.
    pub fn params(&self) -> Vec<String> {
        match self {
            Width::Fixed(_) | Width::Native => vec![],
            Width::Param(name) => vec![name.clone()],
            Width::BinOp { lhs, rhs, .. } => {
                let mut ps = lhs.params();
                ps.extend(rhs.params());
                ps
            }
        }
    }

    /// Substitute all params using `env`, returning a new Width.
    /// Params not in `env` are left as-is.
    pub fn substitute(&self, env: &std::collections::HashMap<String, u64>) -> Width {
        match self {
            Width::Fixed(_) | Width::Native => self.clone(),
            Width::Param(name) => match env.get(name) {
                Some(val) => Width::Fixed(*val),
                None => self.clone(),
            },
            Width::BinOp { op, lhs, rhs } => {
                let new_lhs = lhs.substitute(env);
                let new_rhs = rhs.substitute(env);
                // If both sides are now fixed, evaluate immediately
                if let (Width::Fixed(l), Width::Fixed(r)) = (&new_lhs, &new_rhs) {
                    let result = match op {
                        ComptimeOp::Add => l + r,
                        ComptimeOp::Sub => l.checked_sub(*r).unwrap_or_else(|| {
                            panic!("comptime subtraction underflow: {} - {}", l, r)
                        }),
                        ComptimeOp::Mul => l * r,
                        ComptimeOp::Div => {
                            assert!(*r != 0, "comptime division by zero");
                            l / r
                        }
                    };
                    Width::Fixed(result)
                } else {
                    Width::BinOp {
                        op: *op,
                        lhs: Box::new(new_lhs),
                        rhs: Box::new(new_rhs),
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub name: String,
    pub width: Option<Width>,
}

// --- Compile-time parameters ---

#[derive(Debug, Clone, PartialEq)]
pub struct ComptimeParam {
    pub name: String,
}

// --- Top-level items ---

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Struct(StructDef),
    Fn(FnDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub name: String,
    pub comptime_params: Vec<ComptimeParam>,
    pub fields: Vec<FieldDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnDef {
    pub name: String,
    pub comptime_params: Vec<ComptimeParam>,
    pub params: Vec<Param>,
    pub ret_ty: Option<Type>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

// --- Statements ---

/// Source location for error reporting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

impl Span {
    pub fn new(line: usize, col: usize) -> Self {
        Span { line, col }
    }

    /// A dummy span for contexts where source location is unavailable.
    pub fn dummy() -> Self {
        Span { line: 0, col: 0 }
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line == 0 {
            write!(f, "<unknown>")
        } else {
            write!(f, "{}:{}", self.line, self.col)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CarryDef {
    pub name: String,
    pub ty: Type,
    pub init: Expr,
}

/// A statement with source location.
#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

/// Equality compares only the `kind`, ignoring source spans.
impl PartialEq for Stmt {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Assign {
        target: AssignTarget,
        ty: Option<Type>,
        value: Expr,
    },
    Return(Expr),
    Expr(Expr),
    Stream {
        chunk_name: String,
        chunk_ty: Type,
        buffer: String,
        carry: Vec<CarryDef>,
        body: Vec<Stmt>,
        carry_updates: Vec<(String, Expr)>,
    },
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignTarget {
    Ident(String),
    Destructure(Vec<String>),
    Scatter {
        base: Box<Expr>,
        index: Box<Expr>,
        mask: Option<Box<Expr>>,
    },
}

// --- Expressions ---

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    CharLit(char),
    Ident(String),

    /// [type: val, val, ...] — constant vector literal
    VecLit {
        elem_type: String,
        values: Vec<i64>,
    },

    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
    },

    /// [mask] body : fallback
    Masked {
        mask: Box<Expr>,
        body: Box<Expr>,
        fallback: Option<Box<Expr>>,
    },

    /// +/ expr, */ expr, etc.
    Reduction {
        op: ReductionOp,
        operand: Box<Expr>,
    },

    /// scan.add(expr) or scan.xor(expr, seed)
    Scan {
        op: ScanOp,
        operand: Box<Expr>,
        seed: Option<Box<Expr>>,
    },

    /// expr.field
    Field {
        base: Box<Expr>,
        field: String,
    },

    /// expr(args)
    Call {
        func: Box<Expr>,
        args: Vec<CallArg>,
    },

    /// expr.[index] or expr.[index, mask]
    Gather {
        base: Box<Expr>,
        index: Box<Expr>,
        mask: Option<Box<Expr>>,
    },

    /// Name[width] { field: value, ... }
    StructLit {
        name: String,
        width: Width,
        fields: Vec<(String, Expr)>,
    },

    /// load[Type](ptr) or loadu[Type](ptr) or load_at[Type](ptr, offset)
    Load {
        aligned: bool,
        ty: Type,
        ptr: Box<Expr>,
        offset: Option<Box<Expr>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallArg {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Gt,
    Lt,
    GtEq,
    LtEq,
    EqEq,
    NotEq,
    And,
    Or,
    Xor,
    BitShl,
    BitShr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReductionOp {
    Add,
    Mul,
    Or,
    And,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScanOp {
    Add,
    Xor,
    Max,
    PrecedingAny,
}

// --- Tests for Width evaluation ---

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_eval_fixed() {
        assert_eq!(Width::Fixed(8).eval(&HashMap::new(), 8), 8);
    }

    #[test]
    fn test_eval_native() {
        assert_eq!(Width::Native.eval(&HashMap::new(), 16), 16);
    }

    #[test]
    fn test_eval_param() {
        assert_eq!(Width::Param("N".into()).eval(&env(&[("N", 32)]), 8), 32);
    }

    #[test]
    #[should_panic(expected = "unbound comptime parameter: N")]
    fn test_eval_unbound_param() {
        Width::Param("N".into()).eval(&HashMap::new(), 8);
    }

    #[test]
    fn test_eval_add() {
        let w = Width::BinOp {
            op: ComptimeOp::Add,
            lhs: Box::new(Width::Param("N".into())),
            rhs: Box::new(Width::Fixed(4)),
        };
        assert_eq!(w.eval(&env(&[("N", 8)]), 8), 12);
    }

    #[test]
    fn test_eval_mul() {
        let w = Width::BinOp {
            op: ComptimeOp::Mul,
            lhs: Box::new(Width::Param("N".into())),
            rhs: Box::new(Width::Fixed(2)),
        };
        assert_eq!(w.eval(&env(&[("N", 8)]), 8), 16);
    }

    #[test]
    fn test_eval_sub() {
        let w = Width::BinOp {
            op: ComptimeOp::Sub,
            lhs: Box::new(Width::Param("N".into())),
            rhs: Box::new(Width::Fixed(2)),
        };
        assert_eq!(w.eval(&env(&[("N", 8)]), 8), 6);
    }

    #[test]
    fn test_eval_div() {
        let w = Width::BinOp {
            op: ComptimeOp::Div,
            lhs: Box::new(Width::Param("N".into())),
            rhs: Box::new(Width::Fixed(2)),
        };
        assert_eq!(w.eval(&env(&[("N", 16)]), 8), 8);
    }

    #[test]
    #[should_panic(expected = "comptime division by zero")]
    fn test_eval_div_by_zero() {
        let w = Width::BinOp {
            op: ComptimeOp::Div,
            lhs: Box::new(Width::Fixed(8)),
            rhs: Box::new(Width::Fixed(0)),
        };
        w.eval(&HashMap::new(), 8);
    }

    #[test]
    #[should_panic(expected = "comptime subtraction underflow")]
    fn test_eval_sub_underflow() {
        let w = Width::BinOp {
            op: ComptimeOp::Sub,
            lhs: Box::new(Width::Fixed(2)),
            rhs: Box::new(Width::Fixed(8)),
        };
        w.eval(&HashMap::new(), 8);
    }

    #[test]
    fn test_eval_nested() {
        // (N + M) * 2
        let w = Width::BinOp {
            op: ComptimeOp::Mul,
            lhs: Box::new(Width::BinOp {
                op: ComptimeOp::Add,
                lhs: Box::new(Width::Param("N".into())),
                rhs: Box::new(Width::Param("M".into())),
            }),
            rhs: Box::new(Width::Fixed(2)),
        };
        assert_eq!(w.eval(&env(&[("N", 4), ("M", 4)]), 8), 16);
    }

    #[test]
    fn test_eval_native_in_expr() {
        // _ * 2
        let w = Width::BinOp {
            op: ComptimeOp::Mul,
            lhs: Box::new(Width::Native),
            rhs: Box::new(Width::Fixed(2)),
        };
        assert_eq!(w.eval(&HashMap::new(), 8), 16);
    }

    #[test]
    fn test_is_concrete() {
        assert!(Width::Fixed(8).is_concrete());
        assert!(!Width::Native.is_concrete());
        assert!(!Width::Param("N".into()).is_concrete());
        assert!(Width::BinOp {
            op: ComptimeOp::Add,
            lhs: Box::new(Width::Fixed(4)),
            rhs: Box::new(Width::Fixed(4)),
        }
        .is_concrete());
        assert!(!Width::BinOp {
            op: ComptimeOp::Mul,
            lhs: Box::new(Width::Param("N".into())),
            rhs: Box::new(Width::Fixed(2)),
        }
        .is_concrete());
    }

    #[test]
    fn test_params() {
        assert_eq!(Width::Fixed(8).params(), Vec::<String>::new());
        assert_eq!(Width::Param("N".into()).params(), vec!["N".to_string()]);
        let w = Width::BinOp {
            op: ComptimeOp::Add,
            lhs: Box::new(Width::Param("N".into())),
            rhs: Box::new(Width::Param("M".into())),
        };
        assert_eq!(w.params(), vec!["N".to_string(), "M".to_string()]);
    }

    #[test]
    fn test_substitute_param() {
        let w = Width::Param("N".into());
        assert_eq!(w.substitute(&env(&[("N", 8)])), Width::Fixed(8));
    }

    #[test]
    fn test_substitute_unbound_left_alone() {
        let w = Width::Param("N".into());
        assert_eq!(w.substitute(&HashMap::new()), Width::Param("N".into()));
    }

    #[test]
    fn test_substitute_evaluates_when_concrete() {
        // N * 2 with N=8 → Fixed(16)
        let w = Width::BinOp {
            op: ComptimeOp::Mul,
            lhs: Box::new(Width::Param("N".into())),
            rhs: Box::new(Width::Fixed(2)),
        };
        assert_eq!(w.substitute(&env(&[("N", 8)])), Width::Fixed(16));
    }

    #[test]
    fn test_substitute_partial() {
        // N + M with only N=8 → Fixed(8) + Param("M")
        let w = Width::BinOp {
            op: ComptimeOp::Add,
            lhs: Box::new(Width::Param("N".into())),
            rhs: Box::new(Width::Param("M".into())),
        };
        assert_eq!(
            w.substitute(&env(&[("N", 8)])),
            Width::BinOp {
                op: ComptimeOp::Add,
                lhs: Box::new(Width::Fixed(8)),
                rhs: Box::new(Width::Param("M".into())),
            }
        );
    }
}
