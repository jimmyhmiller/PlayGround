//! Stage 3 — the staged shader backend.
//!
//! A `shader {}` body is split by **binding-time analysis**: every value is
//! `static` (depends only on tokens / variant / `let`s over those) or `dynamic`
//! (transitively touches a per-frame builtin: `time dt hover focus press mouse
//! size resolution`). The static stage is **partial-evaluated now** — folded to
//! constants; the dynamic stage is **lowered to WGSL** with its free builtins
//! declared as uniforms.
//!
//! The author never marks CPU vs GPU. The analysis decides — this is the same
//! binding-time analysis a partial evaluator uses, with the stage boundary set
//! to the GPU upload.

use crate::GlazeError;
use crate::ast::{Expr, ShaderBody};
use crate::eval::{Length, Value, eval_static_expr};
use std::collections::HashMap;

/// The per-frame inputs a Glaze shader may read. Each maps to a field of the
/// canonical `GlazeUniforms` block bound by the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Builtin {
    Time,
    Dt,
    Hover,
    Focus,
    Press,
    Mouse,
    Size,
    Resolution,
    /// per-fragment 0..1 coordinate within the element (the one non-uniform input)
    Uv,
}

impl Builtin {
    fn from_name(s: &str) -> Option<Builtin> {
        Some(match s {
            "time" => Builtin::Time,
            "dt" => Builtin::Dt,
            "hover" => Builtin::Hover,
            "focus" => Builtin::Focus,
            "press" => Builtin::Press,
            "mouse" => Builtin::Mouse,
            "size" => Builtin::Size,
            "resolution" => Builtin::Resolution,
            "uv" => Builtin::Uv,
            _ => return None,
        })
    }
    fn field(self) -> &'static str {
        match self {
            Builtin::Time => "u.time",
            Builtin::Dt => "u.dt",
            Builtin::Hover => "u.hover",
            Builtin::Focus => "u.focus",
            Builtin::Press => "u.press",
            Builtin::Mouse => "u.mouse",
            Builtin::Size => "u.size",
            Builtin::Resolution => "u.resolution",
            // comes from the vertex output, not the uniform block
            Builtin::Uv => "in.uv",
        }
    }
    /// component count (1 = scalar, 2 = vec2)
    fn dim(self) -> u8 {
        match self {
            Builtin::Mouse | Builtin::Size | Builtin::Resolution | Builtin::Uv => 2,
            _ => 1,
        }
    }
    /// `uv` rides the vertex output; everything else is a uniform field.
    pub fn is_uniform(self) -> bool {
        !matches!(self, Builtin::Uv)
    }
}

/// The result of compiling one `shader {}` layer.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledShader {
    pub overlay: bool,
    /// the fragment body: dynamic `let`s + a `return <vec4>;`
    pub wgsl_body: String,
    /// which per-frame builtins the body actually reads (host feeds these)
    pub used: Vec<Builtin>,
}

struct LetInfo {
    dynamic: bool,
    dim: u8,
}

/// A resolved member/swizzle access (`uv.x`, `resolution.xy`, `someLet.rgb`).
struct Member {
    /// the WGSL base expression (e.g. `in.uv`, `u.resolution`, or a let name)
    wgsl_base: String,
    /// the swizzle component string (`x`, `xy`, `rgb`, …)
    swizzle: String,
    dynamic: bool,
    /// the builtin being swizzled, if the base is one (so callers can mark it used)
    builtin: Option<Builtin>,
}

struct Sctx<'a> {
    program: &'a crate::Program,
    variant: &'a HashMap<String, String>,
    /// static `let` values, for folding
    static_lets: HashMap<String, Value>,
    /// binding-time + dim of every `let` seen so far
    lets: HashMap<String, LetInfo>,
}

pub(crate) fn compile_shader(
    program: &crate::Program,
    variant: &HashMap<String, String>,
    body: &ShaderBody,
    overlay: bool,
) -> Result<CompiledShader, GlazeError> {
    let mut cx = Sctx {
        program,
        variant,
        static_lets: HashMap::new(),
        lets: HashMap::new(),
    };
    let mut used: Vec<Builtin> = Vec::new();
    let mut out = String::new();

    // Process `let`s in order. Dynamic ones become WGSL `let` statements;
    // static ones are folded and never reach the GPU.
    for (name, expr) in &body.lets {
        let dim = cx.dim_of(expr)?;
        if cx.is_dynamic(expr) {
            let w = cx.lower(expr, &mut used)?;
            out.push_str(&format!("    let {name} = {w};\n"));
            cx.lets.insert(name.clone(), LetInfo { dynamic: true, dim });
        } else {
            let v = eval_static_expr(cx.program, cx.variant, &cx.static_lets, expr)?;
            cx.static_lets.insert(name.clone(), v);
            cx.lets.insert(name.clone(), LetInfo { dynamic: false, dim });
        }
    }

    // The emit: coerce to a vec4 rgba.
    let emit_dim = cx.dim_of(&body.emit)?;
    let emit_w = cx.lower(&body.emit, &mut used)?;
    match emit_dim {
        4 => out.push_str(&format!("    return {emit_w};\n")),
        1 => out.push_str(&format!(
            "    let _g = {emit_w};\n    return vec4<f32>(_g, _g, _g, _g);\n"
        )),
        n => {
            return Err(GlazeError::Eval(format!(
                "shader `emit` must be a scalar or vec4 (rgba), got a vec{n}"
            )));
        }
    }

    used.dedup();
    Ok(CompiledShader { overlay, wgsl_body: out, used })
}

impl Sctx<'_> {
    /// Resolve `base.swizzle` member access. Returns `None` for ordinary dotted
    /// identifiers (token names like `accent.solid`), which keeps the lexer's
    /// dotted-ident behaviour intact.
    fn member(&self, name: &str) -> Option<Member> {
        let (base, rest) = name.split_once('.')?;
        if rest.is_empty() || rest.len() > 4 || !rest.chars().all(|c| "xyzwrgba".contains(c)) {
            return None;
        }
        if let Some(b) = Builtin::from_name(base) {
            if b.dim() < 2 {
                return None; // can't swizzle a scalar builtin
            }
            return Some(Member {
                wgsl_base: b.field().to_string(),
                swizzle: rest.to_string(),
                dynamic: true,
                builtin: Some(b),
            });
        }
        if let Some(li) = self.lets.get(base) {
            return Some(Member {
                wgsl_base: base.to_string(),
                swizzle: rest.to_string(),
                dynamic: li.dynamic,
                builtin: None,
            });
        }
        None
    }

    /// If `name(args)` calls a user function, return its inlined body.
    fn inline_fn(&self, name: &str, args: &[Expr]) -> Option<Expr> {
        self.program
            .fns
            .iter()
            .find(|f| f.name == name && f.params.len() == args.len())
            .map(|f| crate::ast::subst_params(&f.body, &f.params, args))
    }

    /// Does this expression transitively depend on a per-frame builtin?
    fn is_dynamic(&self, e: &Expr) -> bool {
        match e {
            Expr::Num(..) | Expr::Hex(_) | Expr::Color { .. } => false,
            Expr::Ident(name) => {
                if Builtin::from_name(name).is_some() {
                    return true;
                }
                if let Some(l) = self.lets.get(name) {
                    return l.dynamic;
                }
                self.member(name).map(|m| m.dynamic).unwrap_or(false)
            }
            Expr::Unary(_, x) => self.is_dynamic(x),
            Expr::Bin(_, l, r) => self.is_dynamic(l) || self.is_dynamic(r),
            Expr::Call(name, args) => {
                if let Some(inl) = self.inline_fn(name, args) {
                    return self.is_dynamic(&inl);
                }
                args.iter().any(|a| self.is_dynamic(a))
            }
            Expr::Tern(c, a, b) => self.is_dynamic(c) || self.is_dynamic(a) || self.is_dynamic(b),
        }
    }

    /// Component count of an expression (1 = scalar … 4 = vec4).
    fn dim_of(&self, e: &Expr) -> Result<u8, GlazeError> {
        Ok(match e {
            Expr::Num(..) => 1,
            Expr::Hex(_) | Expr::Color { .. } => 4,
            Expr::Ident(name) => {
                if let Some(b) = Builtin::from_name(name) {
                    b.dim()
                } else if let Some(l) = self.lets.get(name) {
                    l.dim
                } else if let Some(m) = self.member(name) {
                    m.swizzle.len() as u8
                } else {
                    // a token or variant symbol: dim from its static value
                    match eval_static_expr(self.program, self.variant, &self.static_lets, e) {
                        Ok(Value::Color(_)) => 4,
                        _ => 1,
                    }
                }
            }
            Expr::Unary(_, x) => self.dim_of(x)?,
            Expr::Bin(op, l, r) => {
                let (dl, dr) = (self.dim_of(l)?, self.dim_of(r)?);
                match op.as_str() {
                    ">" | "<" | ">=" | "<=" | "==" => 1,
                    // scalar broadcasts against a vector
                    _ => dl.max(dr),
                }
            }
            Expr::Call(name, args) => {
                if let Some(inl) = self.inline_fn(name, args) {
                    self.dim_of(&inl)?
                } else {
                    self.call_dim(name, args)?
                }
            }
            Expr::Tern(_, a, _) => self.dim_of(a)?,
        })
    }

    fn call_dim(&self, name: &str, args: &[Expr]) -> Result<u8, GlazeError> {
        let arg = |i: usize| -> Result<u8, GlazeError> {
            args.get(i)
                .map(|a| self.dim_of(a))
                .unwrap_or(Ok(1))
        };
        Ok(match name {
            "vec2" => 2,
            "vec3" => 3,
            "vec4" => 4,
            "length" | "dot" => 1,
            // result follows the "value" argument
            "smoothstep" => arg(2)?,
            "step" => arg(1)?,
            "mix" | "clamp" | "min" | "max" | "pow" | "sin" | "cos" | "tan" | "abs" | "sqrt"
            | "floor" | "fract" | "sign" | "exp" | "log" | "normalize" => arg(0)?,
            _ => arg(0)?,
        })
    }

    /// Lower an expression to WGSL. Static subexpressions are folded to literals
    /// (constants baked into the shader); dynamic ones recurse.
    fn lower(&self, e: &Expr, used: &mut Vec<Builtin>) -> Result<String, GlazeError> {
        // User functions inline (substitute args into the body) on both sides of
        // the staging boundary. Vector constructors lower structurally — there's
        // no Vec value to fold to.
        if let Expr::Call(name, args) = e {
            if let Some(inl) = self.inline_fn(name, args) {
                return self.lower(&inl, used);
            }
            if matches!(name.as_str(), "vec2" | "vec3" | "vec4") {
                let parts: Result<Vec<_>, _> = args.iter().map(|a| self.lower(a, used)).collect();
                return Ok(format!("{}({})", wgsl_fn(name), parts?.join(", ")));
            }
        }
        if !self.is_dynamic(e) {
            let v = eval_static_expr(self.program, self.variant, &self.static_lets, e)?;
            return wgsl_literal(&v);
        }
        Ok(match e {
            Expr::Ident(name) => {
                if let Some(b) = Builtin::from_name(name) {
                    used.push(b);
                    b.field().to_string()
                } else if let Some(m) = self.member(name) {
                    if let Some(b) = m.builtin {
                        used.push(b);
                    }
                    format!("{}.{}", m.wgsl_base, m.swizzle)
                } else {
                    // a dynamic `let` — reference it by name
                    name.clone()
                }
            }
            Expr::Unary(op, x) => format!("({}{})", op, self.lower(x, used)?),
            Expr::Bin(op, l, r) => {
                format!("({} {} {})", self.lower(l, used)?, op, self.lower(r, used)?)
            }
            Expr::Call(name, args) => {
                let parts: Result<Vec<_>, _> = args.iter().map(|a| self.lower(a, used)).collect();
                format!("{}({})", wgsl_fn(name), parts?.join(", "))
            }
            // select(false_value, true_value, cond)
            Expr::Tern(c, a, b) => format!(
                "select({}, {}, {})",
                self.lower(b, used)?,
                self.lower(a, used)?,
                self.lower(c, used)?
            ),
            Expr::Num(..) | Expr::Hex(_) | Expr::Color { .. } => unreachable!("literals are static"),
        })
    }
}

fn wgsl_fn(name: &str) -> &str {
    match name {
        "vec2" => "vec2<f32>",
        "vec3" => "vec3<f32>",
        "vec4" => "vec4<f32>",
        other => other,
    }
}

/// Format an f64 as a WGSL float literal (always has a decimal point).
fn fnum(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 {
        format!("{:.1}", x)
    } else {
        let s = format!("{}", x);
        if s.contains('.') || s.contains('e') { s } else { format!("{s}.0") }
    }
}

fn wgsl_literal(v: &Value) -> Result<String, GlazeError> {
    Ok(match v {
        Value::Num(n) => fnum(*n),
        Value::Len(Length::Px(p)) => fnum(*p as f64),
        Value::Len(_) => return Err(GlazeError::Eval("non-px length in a shader expression".into())),
        Value::Color(c) => format!(
            "vec4<f32>({}, {}, {}, {})",
            fnum(c.r as f64),
            fnum(c.g as f64),
            fnum(c.b as f64),
            fnum(c.a as f64)
        ),
        Value::Bool(b) => b.to_string(),
        Value::Sym(s) => {
            return Err(GlazeError::Eval(format!(
                "symbol `{s}` has no numeric value in a shader expression"
            )));
        }
    })
}
