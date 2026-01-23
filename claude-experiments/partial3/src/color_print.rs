//! Colored pretty printing for partial evaluation results
//!
//! This module provides functions to print PValues with colors to distinguish
//! static (known at compile time) from dynamic (only known at runtime) parts.

use crate::ast::Expr;
use crate::partial::PValue;
use crate::value::Value;

// ANSI color codes
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";

/// Check if an expression is "static" (a literal value known at compile time)
fn is_static_expr(e: &Expr) -> bool {
    match e {
        Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Undefined | Expr::Null => true,
        Expr::Array(elements) => elements.iter().all(is_static_expr),
        Expr::Object(props) => props.iter().all(|(_, v)| is_static_expr(v)),
        Expr::Fn(_, _) => true, // Functions are static values
        _ => false,
    }
}


/// Configuration for colored printing
pub struct ColorConfig {
    pub static_color: &'static str,
    pub dynamic_color: &'static str,
    pub enabled: bool,
}

impl Default for ColorConfig {
    fn default() -> Self {
        Self {
            static_color: GREEN,
            dynamic_color: YELLOW,
            enabled: true,
        }
    }
}

impl ColorConfig {
    pub fn no_color() -> Self {
        Self {
            static_color: "",
            dynamic_color: "",
            enabled: false,
        }
    }
}

/// Pretty print a PValue with colors
pub fn pretty_print_colored(pv: &PValue) -> String {
    pretty_print_colored_with_config(pv, &ColorConfig::default())
}

/// Pretty print a PValue with custom color configuration
pub fn pretty_print_colored_with_config(pv: &PValue, config: &ColorConfig) -> String {
    let mut output = String::new();
    pretty_print_pvalue(&mut output, pv, 0, config);
    output
}

fn pretty_print_pvalue(out: &mut String, pv: &PValue, indent: usize, config: &ColorConfig) {
    match pv {
        PValue::Static(v) => {
            // Static values are shown in green
            if config.enabled {
                out.push_str(config.static_color);
            }
            pretty_print_value(out, v, indent, config);
            if config.enabled {
                out.push_str(RESET);
            }
        }
        PValue::StaticNamed { name, value } => {
            // Named static - show the name with a dim indicator of its value
            if config.enabled {
                out.push_str(config.static_color);
            }
            out.push_str(name);
            if config.enabled {
                out.push_str(DIM);
                out.push_str(" /* = ");
                pretty_print_value_inline(out, value);
                out.push_str(" */");
                out.push_str(RESET);
            }
        }
        PValue::Dynamic(e) => {
            // Dynamic expressions - colorize based on subexpression types
            pretty_print_expr_colored(out, e, indent, config);
        }
        PValue::DynamicNamed { name, expr } => {
            // Named dynamic - show the name (expr kept for optimization)
            if config.enabled {
                out.push_str(config.dynamic_color);
            }
            out.push_str(name);
            if config.enabled {
                out.push_str(DIM);
                out.push_str(" /* = ");
                out.push_str(&format!("{}", expr));
                out.push_str(" */");
                out.push_str(RESET);
            }
        }
    }
}

fn pretty_print_value(out: &mut String, v: &Value, indent: usize, config: &ColorConfig) {
    let _ = (indent, config); // For future use with nested structures
    match v {
        Value::Int(n) => out.push_str(&format!("{}", n)),
        Value::Bool(b) => out.push_str(&format!("{}", b)),
        Value::String(s) => out.push_str(&format!("\"{}\"", s.escape_default())),
        Value::Undefined => out.push_str("undefined"),
        Value::Null => out.push_str("null"),
        Value::Array(elements) => {
            out.push_str("(array");
            for elem in elements.borrow().iter() {
                out.push(' ');
                pretty_print_value_inline(out, elem);
            }
            out.push(')');
        }
        Value::Object(obj) => {
            out.push_str("(object");
            for (k, v) in obj.borrow().iter() {
                out.push_str(&format!(" ({} ", k));
                pretty_print_value_inline(out, v);
                out.push(')');
            }
            out.push(')');
        }
        Value::Closure { params, body, .. } => {
            out.push_str(&format!("(fn ({}) {})", params.join(" "), body));
        }
        Value::Opaque { label, .. } => {
            out.push_str(&format!("<opaque: {}>", label));
        }
    }
}

fn pretty_print_value_inline(out: &mut String, v: &Value) {
    match v {
        Value::Int(n) => out.push_str(&format!("{}", n)),
        Value::Bool(b) => out.push_str(&format!("{}", b)),
        Value::String(s) => out.push_str(&format!("\"{}\"", s.escape_default())),
        Value::Undefined => out.push_str("undefined"),
        Value::Null => out.push_str("null"),
        Value::Array(elements) => {
            out.push('[');
            for (i, elem) in elements.borrow().iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                pretty_print_value_inline(out, elem);
            }
            out.push(']');
        }
        Value::Object(obj) => {
            out.push('{');
            for (i, (k, val)) in obj.borrow().iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("{}: ", k));
                pretty_print_value_inline(out, val);
            }
            out.push('}');
        }
        Value::Closure { params, body, .. } => {
            out.push_str(&format!("(fn ({}) ...)", params.join(" ")));
            let _ = body;
        }
        Value::Opaque { label, .. } => {
            out.push_str(&format!("<{}>", label));
        }
    }
}

fn pretty_print_expr_colored(out: &mut String, e: &Expr, indent: usize, config: &ColorConfig) {
    let ind1 = "  ".repeat(indent + 1);

    match e {
        // Literals are always static (green)
        Expr::Int(n) => {
            if config.enabled {
                out.push_str(config.static_color);
            }
            out.push_str(&format!("{}", n));
            if config.enabled {
                out.push_str(RESET);
            }
        }
        Expr::Bool(b) => {
            if config.enabled {
                out.push_str(config.static_color);
            }
            out.push_str(&format!("{}", b));
            if config.enabled {
                out.push_str(RESET);
            }
        }
        Expr::String(s) => {
            if config.enabled {
                out.push_str(config.static_color);
            }
            out.push_str(&format!("\"{}\"", s.escape_default()));
            if config.enabled {
                out.push_str(RESET);
            }
        }
        Expr::Undefined => {
            if config.enabled {
                out.push_str(config.static_color);
            }
            out.push_str("undefined");
            if config.enabled {
                out.push_str(RESET);
            }
        }
        Expr::Null => {
            if config.enabled {
                out.push_str(config.static_color);
            }
            out.push_str("null");
            if config.enabled {
                out.push_str(RESET);
            }
        }

        // Variables are dynamic (yellow) - they reference runtime values
        Expr::Var(name) => {
            if config.enabled {
                out.push_str(config.dynamic_color);
            }
            out.push_str(name);
            if config.enabled {
                out.push_str(RESET);
            }
        }

        // Binary operations
        Expr::BinOp(op, left, right) => {
            out.push('(');
            out.push_str(&format!("{}", op));
            out.push(' ');
            pretty_print_expr_colored(out, left, indent, config);
            out.push(' ');
            pretty_print_expr_colored(out, right, indent, config);
            out.push(')');
        }

        Expr::BitNot(inner) => {
            out.push_str("(~ ");
            pretty_print_expr_colored(out, inner, indent, config);
            out.push(')');
        }

        Expr::LogNot(inner) => {
            out.push_str("(! ");
            pretty_print_expr_colored(out, inner, indent, config);
            out.push(')');
        }

        // Let bindings - color variable as static if value is static
        Expr::Let(name, value, body) => {
            out.push_str("(let ");
            if config.enabled {
                if is_static_expr(value) {
                    out.push_str(config.static_color);
                } else {
                    out.push_str(config.dynamic_color);
                }
            }
            out.push_str(name);
            if config.enabled {
                out.push_str(RESET);
            }
            out.push(' ');
            pretty_print_expr_colored(out, value, indent + 1, config);
            out.push('\n');
            out.push_str(&ind1);
            pretty_print_expr_colored(out, body, indent + 1, config);
            out.push(')');
        }

        // If expressions
        Expr::If(cond, then_branch, else_branch) => {
            out.push_str("(if ");
            pretty_print_expr_colored(out, cond, indent, config);
            out.push('\n');
            out.push_str(&ind1);
            pretty_print_expr_colored(out, then_branch, indent + 1, config);
            out.push('\n');
            out.push_str(&ind1);
            pretty_print_expr_colored(out, else_branch, indent + 1, config);
            out.push(')');
        }

        // Function definitions
        Expr::Fn(params, body) => {
            out.push_str("(fn (");
            for (i, param) in params.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                if config.enabled {
                    out.push_str(config.dynamic_color);
                }
                out.push_str(param);
                if config.enabled {
                    out.push_str(RESET);
                }
            }
            out.push_str(")\n");
            out.push_str(&ind1);
            pretty_print_expr_colored(out, body, indent + 1, config);
            out.push(')');
        }

        // Function calls
        Expr::Call(func, args) => {
            out.push_str("(call ");
            pretty_print_expr_colored(out, func, indent, config);
            for arg in args {
                out.push(' ');
                pretty_print_expr_colored(out, arg, indent, config);
            }
            out.push(')');
        }

        // Arrays - color as static if all elements are static
        Expr::Array(elements) => {
            let all_static = elements.iter().all(is_static_expr);
            if all_static && config.enabled {
                out.push_str(config.static_color);
            }
            out.push_str("(array");
            for elem in elements {
                out.push(' ');
                if all_static {
                    // Print elements without recursing into colored printing
                    pretty_print_expr_colored(out, elem, indent, config);
                } else {
                    pretty_print_expr_colored(out, elem, indent, config);
                }
            }
            out.push(')');
            if all_static && config.enabled {
                out.push_str(RESET);
            }
        }

        Expr::Index(arr, idx) => {
            out.push_str("(index ");
            pretty_print_expr_colored(out, arr, indent, config);
            out.push(' ');
            pretty_print_expr_colored(out, idx, indent, config);
            out.push(')');
        }

        Expr::Len(arr) => {
            out.push_str("(len ");
            pretty_print_expr_colored(out, arr, indent, config);
            out.push(')');
        }

        // Objects - color as static if all values are static
        Expr::Object(props) => {
            let all_static = props.iter().all(|(_, v)| is_static_expr(v));
            if all_static && config.enabled {
                out.push_str(config.static_color);
            }
            out.push_str("(object");
            for (key, val) in props {
                out.push_str(&format!(" ({} ", key));
                pretty_print_expr_colored(out, val, indent, config);
                out.push(')');
            }
            out.push(')');
            if all_static && config.enabled {
                out.push_str(RESET);
            }
        }

        Expr::PropAccess(obj, prop) => {
            out.push_str("(prop ");
            pretty_print_expr_colored(out, obj, indent, config);
            out.push_str(&format!(" \"{}\")", prop));
        }

        Expr::PropSet(obj, prop, val) => {
            out.push_str("(prop-set! ");
            pretty_print_expr_colored(out, obj, indent, config);
            out.push_str(&format!(" \"{}\" ", prop));
            pretty_print_expr_colored(out, val, indent, config);
            out.push(')');
        }

        Expr::ComputedAccess(obj, key) => {
            out.push_str("(computed-prop ");
            pretty_print_expr_colored(out, obj, indent, config);
            out.push(' ');
            pretty_print_expr_colored(out, key, indent, config);
            out.push(')');
        }

        Expr::ComputedSet(obj, key, val) => {
            out.push_str("(computed-set! ");
            pretty_print_expr_colored(out, obj, indent, config);
            out.push(' ');
            pretty_print_expr_colored(out, key, indent, config);
            out.push(' ');
            pretty_print_expr_colored(out, val, indent, config);
            out.push(')');
        }

        // Loops
        Expr::While(cond, body) => {
            out.push_str("(while ");
            pretty_print_expr_colored(out, cond, indent, config);
            out.push('\n');
            out.push_str(&ind1);
            pretty_print_expr_colored(out, body, indent + 1, config);
            out.push(')');
        }

        Expr::For { init, cond, update, body } => {
            out.push_str("(for");
            if let Some(i) = init {
                out.push(' ');
                pretty_print_expr_colored(out, i, indent, config);
            } else {
                out.push_str(" #f");
            }
            if let Some(c) = cond {
                out.push(' ');
                pretty_print_expr_colored(out, c, indent, config);
            } else {
                out.push_str(" #t");
            }
            if let Some(u) = update {
                out.push(' ');
                pretty_print_expr_colored(out, u, indent, config);
            } else {
                out.push_str(" #f");
            }
            out.push('\n');
            out.push_str(&ind1);
            pretty_print_expr_colored(out, body, indent + 1, config);
            out.push(')');
        }

        Expr::Break => out.push_str("(break)"),
        Expr::Continue => out.push_str("(continue)"),

        // Mutation - color variable as static if value is static
        Expr::Set(name, value) => {
            out.push_str("(set! ");
            if config.enabled {
                if is_static_expr(value) {
                    out.push_str(config.static_color);
                } else {
                    out.push_str(config.dynamic_color);
                }
            }
            out.push_str(name);
            if config.enabled {
                out.push_str(RESET);
            }
            out.push(' ');
            pretty_print_expr_colored(out, value, indent, config);
            out.push(')');
        }

        // Sequencing
        Expr::Begin(exprs) => {
            if exprs.is_empty() {
                out.push_str("(begin)");
            } else if exprs.len() == 1 {
                out.push_str("(begin ");
                pretty_print_expr_colored(out, &exprs[0], indent, config);
                out.push(')');
            } else {
                out.push_str("(begin");
                for e in exprs {
                    out.push('\n');
                    out.push_str(&ind1);
                    pretty_print_expr_colored(out, e, indent + 1, config);
                }
                out.push(')');
            }
        }

        // Error handling
        Expr::Throw(inner) => {
            out.push_str("(throw ");
            pretty_print_expr_colored(out, inner, indent, config);
            out.push(')');
        }

        Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
            out.push_str("(try\n");
            out.push_str(&ind1);
            pretty_print_expr_colored(out, try_block, indent + 1, config);
            out.push('\n');
            out.push_str(&ind1);
            if let Some(param) = catch_param {
                out.push_str("(catch ");
                if config.enabled {
                    out.push_str(config.dynamic_color);
                }
                out.push_str(param);
                if config.enabled {
                    out.push_str(RESET);
                }
                out.push(' ');
                pretty_print_expr_colored(out, catch_block, indent + 1, config);
                out.push(')');
            } else {
                out.push_str("(catch ");
                pretty_print_expr_colored(out, catch_block, indent + 1, config);
                out.push(')');
            }
            if let Some(finally) = finally_block {
                out.push('\n');
                out.push_str(&ind1);
                out.push_str("(finally ");
                pretty_print_expr_colored(out, finally, indent + 1, config);
                out.push(')');
            }
            out.push(')');
        }

        Expr::Switch { discriminant, cases, default } => {
            out.push_str("(switch ");
            pretty_print_expr_colored(out, discriminant, indent, config);
            for (val, body) in cases {
                out.push('\n');
                out.push_str(&ind1);
                out.push_str("(case ");
                pretty_print_expr_colored(out, val, indent + 1, config);
                for stmt in body {
                    out.push('\n');
                    out.push_str(&"  ".repeat(indent + 2));
                    pretty_print_expr_colored(out, stmt, indent + 2, config);
                }
                out.push(')');
            }
            if let Some(def) = default {
                out.push('\n');
                out.push_str(&ind1);
                out.push_str("(default");
                for stmt in def {
                    out.push('\n');
                    out.push_str(&"  ".repeat(indent + 2));
                    pretty_print_expr_colored(out, stmt, indent + 2, config);
                }
                out.push(')');
            }
            out.push(')');
        }

        // New expressions
        Expr::New(ctor, args) => {
            out.push_str("(new ");
            pretty_print_expr_colored(out, ctor, indent, config);
            for arg in args {
                out.push(' ');
                pretty_print_expr_colored(out, arg, indent, config);
            }
            out.push(')');
        }

        Expr::Opaque(label) => {
            if config.enabled {
                out.push_str(config.static_color);
            }
            out.push_str(&format!("<opaque: {}>", label));
            if config.enabled {
                out.push_str(RESET);
            }
        }
    }
}

/// Print a legend explaining the colors
pub fn print_legend() {
    println!("{}Static (known at compile time){}", GREEN, RESET);
    println!("{}Dynamic (only known at runtime){}", YELLOW, RESET);
}
