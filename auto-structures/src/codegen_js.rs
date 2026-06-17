//! Compile-to-JavaScript backend. Each abstract collection is lowered to the
//! concrete JS structure that analysis chose: `Set`, `Map`, `Array`, or a
//! `Queue` class. Operations are emitted as the native calls for that
//! structure, so the generated program is already specialized — there is no
//! generic dispatch at runtime.

use crate::analysis::Analysis;
use crate::ast::*;
use crate::value::Kind;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq)]
enum JsRepr {
    Set,
    Map,
    Array,
    Queue,
    Scalar,
}

pub struct Codegen<'a> {
    analysis: &'a Analysis,
    reprs: HashMap<String, JsRepr>,
    out: String,
    indent: usize,
}

const PRELUDE: &str = r#""use strict";
// ---- runtime prelude (specialized data-structure helpers) ----
class Queue {
  constructor() { this.items = []; this.head = 0; }
  enqueue(x) { this.items.push(x); }
  dequeue() {
    if (this.head >= this.items.length) return null;
    const v = this.items[this.head++];
    if (this.head > 64 && this.head * 2 > this.items.length) {
      this.items = this.items.slice(this.head);
      this.head = 0;
    }
    return v;
  }
  front() { return this.head < this.items.length ? this.items[this.head] : null; }
  size() { return this.items.length - this.head; }
  *[Symbol.iterator]() { for (let i = this.head; i < this.items.length; i++) yield this.items[i]; }
}
function __cmp(a, b) { if (a < b) return -1; if (a > b) return 1; return 0; }
function __sorted(it) { return [...it].sort(__cmp); }
function __range(n) { const a = []; for (let i = 0; i < n; i++) a.push(i); return a; }
function __min(it) { let m = null, first = true; for (const x of it) { if (first || x < m) { m = x; first = false; } } return m; }
function __max(it) { let m = null, first = true; for (const x of it) { if (first || x > m) { m = x; first = false; } } return m; }
function __show(v) {
  if (v === null || v === undefined) return "nil";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number" || typeof v === "string") return "" + v;
  if (v instanceof Set) return "{" + [...v].map(__show).join(", ") + "}";
  if (v instanceof Map) return "{" + [...v.entries()].map(([k, x]) => __show(k) + ": " + __show(x)).join(", ") + "}";
  if (v instanceof Queue) return "[" + [...v].map(__show).join(", ") + "]";
  if (Array.isArray(v)) return "[" + v.map(__show).join(", ") + "]";
  return "" + v;
}
function __print(args) { console.log(args.map(__show).join(" ")); }
// ---- generated program ----
"#;

pub fn generate(program: &Program, analysis: &Analysis) -> String {
    let mut cg = Codegen {
        analysis,
        reprs: HashMap::new(),
        out: String::new(),
        indent: 0,
    };
    cg.precompute_reprs(program);
    cg.out.push_str(PRELUDE);
    let stmts = program.clone();
    cg.emit_block(&stmts);
    cg.out
}

impl<'a> Codegen<'a> {
    /// Walk lets to learn each variable's JS representation up front.
    fn precompute_reprs(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            match s {
                Stmt::Let { name, value } => {
                    let repr = self.repr_of_expr(value, name);
                    self.reprs.insert(name.clone(), repr);
                }
                Stmt::If { then, els, .. } => {
                    self.precompute_reprs(then);
                    self.precompute_reprs(els);
                }
                Stmt::While { body, .. } => self.precompute_reprs(body),
                Stmt::For { body, .. } => self.precompute_reprs(body),
                _ => {}
            }
        }
    }

    fn repr_of_expr(&self, e: &Expr, binding: &str) -> JsRepr {
        match e {
            Expr::Call(name, args) if name == "collection" && args.is_empty() => {
                match self.analysis.kind_of(binding) {
                    Some(Kind::Set { .. }) => JsRepr::Set,
                    Some(Kind::Map { .. }) => JsRepr::Map,
                    Some(Kind::Sequence) => JsRepr::Array,
                    Some(Kind::Queue) => JsRepr::Queue,
                    None => JsRepr::Set,
                }
            }
            Expr::List(_) => JsRepr::Array,
            Expr::Call(name, _) if name == "sorted" || name == "keys" || name == "range" => {
                JsRepr::Array
            }
            Expr::Var(v) => self.reprs.get(v).copied().unwrap_or(JsRepr::Scalar),
            _ => JsRepr::Scalar,
        }
    }

    fn repr_of_first(&self, args: &[Expr]) -> JsRepr {
        match args.first() {
            Some(Expr::Var(v)) => self.reprs.get(v).copied().unwrap_or(JsRepr::Scalar),
            _ => JsRepr::Scalar,
        }
    }

    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("  ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn emit_block(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.emit_stmt(s);
        }
    }

    fn emit_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let { name, value } => {
                let rhs = self.emit_ctor_or_expr(value, name);
                self.line(&format!("let {} = {};", name, rhs));
            }
            Stmt::Assign { name, value } => {
                let rhs = self.emit_expr(value);
                self.line(&format!("{} = {};", name, rhs));
            }
            Stmt::Expr(e) => {
                let s = self.emit_expr(e);
                self.line(&format!("{};", s));
            }
            Stmt::If { cond, then, els } => {
                let c = self.emit_expr(cond);
                self.line(&format!("if ({}) {{", c));
                self.indent += 1;
                self.emit_block(then);
                self.indent -= 1;
                if els.is_empty() {
                    self.line("}");
                } else {
                    self.line("} else {");
                    self.indent += 1;
                    self.emit_block(els);
                    self.indent -= 1;
                    self.line("}");
                }
            }
            Stmt::While { cond, body } => {
                let c = self.emit_expr(cond);
                self.line(&format!("while ({}) {{", c));
                self.indent += 1;
                self.emit_block(body);
                self.indent -= 1;
                self.line("}");
            }
            Stmt::For { var, iter, body } => {
                let it = self.emit_iterable(iter);
                self.line(&format!("for (const {} of {}) {{", var, it));
                self.indent += 1;
                self.emit_block(body);
                self.indent -= 1;
                self.line("}");
            }
            Stmt::Break => self.line("break;"),
            Stmt::Continue => self.line("continue;"),
        }
    }

    fn emit_ctor_or_expr(&self, value: &Expr, binding: &str) -> String {
        if let Expr::Call(name, args) = value {
            if name == "collection" && args.is_empty() {
                return match self.reprs.get(binding).copied().unwrap_or(JsRepr::Set) {
                    JsRepr::Set => "new Set()".to_string(),
                    JsRepr::Map => "new Map()".to_string(),
                    JsRepr::Array => "[]".to_string(),
                    JsRepr::Queue => "new Queue()".to_string(),
                    JsRepr::Scalar => "new Set()".to_string(),
                };
            }
        }
        self.emit_expr(value)
    }

    fn emit_iterable(&self, e: &Expr) -> String {
        match e {
            Expr::Var(v) => match self.reprs.get(v).copied().unwrap_or(JsRepr::Scalar) {
                JsRepr::Map => format!("{}.keys()", v),
                _ => v.clone(),
            },
            _ => self.emit_expr(e),
        }
    }

    fn emit_expr(&self, e: &Expr) -> String {
        match e {
            Expr::Int(n) => n.to_string(),
            Expr::Str(s) => js_string(s),
            Expr::Bool(b) => b.to_string(),
            Expr::Nil => "null".to_string(),
            Expr::Var(name) => name.clone(),
            Expr::List(items) => {
                let parts: Vec<String> = items.iter().map(|i| self.emit_expr(i)).collect();
                format!("[{}]", parts.join(", "))
            }
            Expr::Unary(op, inner) => {
                let v = self.emit_expr(inner);
                match op {
                    UnOp::Neg => format!("(-{})", v),
                    UnOp::Not => format!("(!Boolean({}))", v),
                }
            }
            Expr::Binary(op, l, r) => {
                let a = self.emit_expr(l);
                let b = self.emit_expr(r);
                match op {
                    BinOp::Add => format!("({} + {})", a, b),
                    BinOp::Sub => format!("({} - {})", a, b),
                    BinOp::Mul => format!("({} * {})", a, b),
                    BinOp::Div => format!("Math.trunc({} / {})", a, b),
                    BinOp::Mod => format!("({} % {})", a, b),
                    BinOp::Eq => format!("({} === {})", a, b),
                    BinOp::Ne => format!("({} !== {})", a, b),
                    BinOp::Lt => format!("({} < {})", a, b),
                    BinOp::Le => format!("({} <= {})", a, b),
                    BinOp::Gt => format!("({} > {})", a, b),
                    BinOp::Ge => format!("({} >= {})", a, b),
                    BinOp::And => format!("(Boolean({}) && Boolean({}))", a, b),
                    BinOp::Or => format!("(Boolean({}) || Boolean({}))", a, b),
                }
            }
            Expr::Call(name, args) => self.emit_call(name, args),
        }
    }

    fn emit_call(&self, name: &str, args: &[Expr]) -> String {
        let a: Vec<String> = args.iter().map(|x| self.emit_expr(x)).collect();
        let repr = self.repr_of_first(args);
        let arg = |i: usize| a.get(i).cloned().unwrap_or_else(|| "undefined".to_string());

        match name {
            "print" => format!("__print([{}])", a.join(", ")),
            "range" => format!("__range({})", arg(0)),
            "str" => format!("__show({})", arg(0)),
            "sorted" => match repr {
                JsRepr::Map => format!("__sorted({}.keys())", arg(0)),
                _ => format!("__sorted({})", arg(0)),
            },
            "keys" => format!("[...{}.keys()]", arg(0)),
            "add" => format!("{}.add({})", arg(0), arg(1)),
            "has" => format!("{}.has({})", arg(0), arg(1)),
            "del" => format!("{}.delete({})", arg(0), arg(1)),
            "put" => format!("{}.set({}, {})", arg(0), arg(1), arg(2)),
            "get" => format!("({}.get({}) ?? null)", arg(0), arg(1)),
            "append" | "push" => format!("{}.push({})", arg(0), arg(1)),
            "pop" => format!("({}.pop() ?? null)", arg(0)),
            "peek" => format!("{}[{}.length - 1]", arg(0), arg(0)),
            "at" => format!("{}[{}]", arg(0), arg(1)),
            "set_at" => format!("({}[{}] = {})", arg(0), arg(1), arg(2)),
            "enqueue" => format!("{}.enqueue({})", arg(0), arg(1)),
            "dequeue" => format!("{}.dequeue()", arg(0)),
            "front" => format!("{}.front()", arg(0)),
            "min" => format!("__min({})", arg(0)),
            "max" => format!("__max({})", arg(0)),
            "size" | "len" => match repr {
                JsRepr::Set | JsRepr::Map => format!("{}.size", arg(0)),
                JsRepr::Queue => format!("{}.size()", arg(0)),
                _ => format!("{}.length", arg(0)),
            },
            other => format!("/* unknown:{} */ null", other),
        }
    }
}

fn js_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}
