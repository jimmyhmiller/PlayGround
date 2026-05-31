//! MLIR-free dataflow analyses over the JSIR IR.
//!
//! First analysis: **constant propagation**, a port of upstream
//! `maldoca/js/ir/analyses/constant_propagation`. It is a forward dataflow
//! analysis over the JSHIR with a 3-level constant lattice
//! (`Uninitialized` ⊑ constant ⊑ `Unknown`) and a per-variable state that
//! tracks which symbols hold known constants.

use jsir_ir::attr::{format_mlir_f64, quote_mlir_string};

pub mod constprop;
pub mod dataflow;
pub mod sign;
pub use constprop::{analyze_constants, ConstProp};
pub use dataflow::{Analysis, Lattice as LatticeTrait};
pub use sign::{analyze_signs, SignAnalysis};

/// A concrete constant value a JS expression can fold to.
#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    /// `f64` stored as bits so `NaN`/`-0` compare structurally like MLIR attrs.
    Num(u64),
    Str(String),
    Bool(bool),
    Null,
    /// `value`, verbatim `raw` (`1n`), and parsed `rawValue`.
    BigInt { value: String, raw: String, raw_value: String },
    /// A regular-expression literal: `pattern` and `flags`.
    RegExp { pattern: String, flags: String },
}

impl Const {
    pub fn num(f: f64) -> Const {
        Const::Num(f.to_bits())
    }
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Const::Num(b) => Some(f64::from_bits(*b)),
            _ => None,
        }
    }

    /// JS `ToNumber` for the primitive constants we model.
    pub fn to_number(&self) -> Option<f64> {
        match self {
            Const::Num(b) => Some(f64::from_bits(*b)),
            Const::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            Const::Null => Some(0.0),
            Const::Str(s) => {
                let t = s.trim();
                if t.is_empty() {
                    Some(0.0)
                } else {
                    t.parse::<f64>().ok()
                }
            }
            _ => None,
        }
    }

    /// JS `ToString` for the primitive constants we model.
    pub fn to_js_string(&self) -> Option<String> {
        Some(match self {
            Const::Num(b) => js_number_to_string(f64::from_bits(*b)),
            Const::Str(s) => s.clone(),
            Const::Bool(v) => if *v { "true" } else { "false" }.to_string(),
            Const::Null => "null".to_string(),
            Const::BigInt { value, .. } => value.clone(),
            Const::RegExp { .. } => return None,
        })
    }
}

/// JS `Number.prototype.toString` for the common (finite) cases: integers print
/// without a decimal point.
pub fn js_number_to_string(n: f64) -> String {
    if n == 0.0 {
        "0".to_string()
    } else if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e21 {
        format!("{}", n as i64)
    } else if n.is_nan() {
        "NaN".to_string()
    } else if n.is_infinite() {
        if n > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
    } else {
        format!("{n}")
    }
}

/// The constant-propagation lattice for a single value.
#[derive(Debug, Clone, PartialEq)]
pub enum Lattice {
    /// Bottom: never assigned / unreachable.
    Uninitialized,
    /// A known constant.
    Const(Const),
    /// Top: a value that is not a known single constant.
    Unknown,
}

impl Lattice {
    /// The numeric value if this is a numeric constant.
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Lattice::Const(c) => c.as_num(),
            _ => None,
        }
    }
}

/// The constant lattice plugged into the generic dataflow engine. `bottom` is
/// `Uninitialized`, `top` is `Unknown`; the join is upstream's exact rule.
impl dataflow::Lattice for Lattice {
    fn bottom() -> Self {
        Lattice::Uninitialized
    }
    fn top() -> Self {
        Lattice::Unknown
    }

    fn join(&mut self, other: &Lattice) -> bool {
        match (&*self, other) {
            (_, Lattice::Uninitialized) => false,
            (Lattice::Uninitialized, _) => {
                *self = other.clone();
                true
            }
            (Lattice::Unknown, _) => false,
            (Lattice::Const(a), Lattice::Const(b)) if a == b => false,
            (Lattice::Const(_), _) => {
                *self = Lattice::Unknown;
                true
            }
        }
    }

    /// Render exactly like upstream's analysis dump (`// %N = <value>`).
    fn render(&self) -> String {
        match self {
            Lattice::Uninitialized => "<uninitialized>".to_string(),
            Lattice::Unknown => "<unknown>".to_string(),
            Lattice::Const(c) => match c {
                Const::Num(b) => format!("{} : f64", format_mlir_f64(f64::from_bits(*b))),
                Const::Str(s) => quote_mlir_string(s),
                Const::Bool(v) => if *v { "true" } else { "false" }.to_string(),
                Const::Null => "#jsir.null_literal".to_string(),
                Const::BigInt { value, raw, raw_value } => format!(
                    "#jsir<big_int_literal {},  {}, {}>",
                    quote_mlir_string(value),
                    quote_mlir_string(raw),
                    quote_mlir_string(raw_value),
                ),
                Const::RegExp { pattern, flags } => format!(
                    "#jsir<reg_exp_literal {}, {}>",
                    quote_mlir_string(pattern),
                    quote_mlir_string(flags),
                ),
            },
        }
    }
}
