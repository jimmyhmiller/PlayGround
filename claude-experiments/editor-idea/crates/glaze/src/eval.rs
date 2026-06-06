//! Stage-1 static evaluation: resolve tokens, fold a style's static layers for a
//! given variant + active discrete states into a concrete `CompiledStyle` IR.
//!
//! This is the part a host renderer consumes. Dynamic (shader) layers are Stage 3
//! and currently produce a hard `Unsupported` error rather than being silently
//! dropped.

use crate::GlazeError;
use crate::ast::*;
use std::collections::HashMap;

// ---------- values ----------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Length {
    Px(f32),
    Pct(f32),
    Em(f32),
    Auto,
}

/// The runtime value of an evaluated expression during static folding.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Num(f64),
    Len(Length),
    Color(Rgba),
    Bool(bool),
    /// a bare identifier with no token/param binding — a keyword/variant literal
    Sym(String),
}

// ---------- compiled IR ----------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dim {
    Px(f32),
    Pct(f32),
    Auto,
}

/// Flex main-axis direction for a container.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dir {
    Row,
    Column,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BoxStyle {
    /// top, right, bottom, left
    pub padding: [f32; 4],
    pub gap: f32,
    pub radius: f32,
    pub width: Option<Dim>,
    pub height: Option<Dim>,
    pub min_width: Option<Dim>,
    pub max_width: Option<Dim>,
    pub min_height: Option<Dim>,
    pub max_height: Option<Dim>,
    /// flexbox grow/shrink factors (for children of a row/column)
    pub flex_grow: Option<f32>,
    pub flex_shrink: Option<f32>,
    /// override this container's main-axis direction (responsive layout)
    pub flex_direction: Option<Dir>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Layer {
    Fill(Rgba),
    Border { color: Rgba, width: f32 },
    Shadow { color: Rgba, blur: f32, offset_y: f32 },
    /// A compiled shader layer (Stage 3): generated WGSL + its dynamic inputs.
    Shader(crate::shader::CompiledShader),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompiledStyle {
    pub box_: BoxStyle,
    pub layers: Vec<Layer>,
}

// ---------- evaluation context ----------

struct Ctx<'a> {
    program: &'a Program,
    variant: &'a HashMap<String, String>,
    /// token names currently being resolved, for cycle detection
    resolving: Vec<String>,
    /// in-scope static `let` bindings (shader bodies)
    lets: HashMap<String, Value>,
    /// viewport width/height, available as `vw`/`vh` in `when` breakpoints
    vw: f32,
    vh: f32,
}

/// Evaluate a fully-static expression with a set of static `let` bindings in
/// scope. Used by the shader backend to fold the static stage.
pub(crate) fn eval_static_expr(
    program: &Program,
    variant: &HashMap<String, String>,
    lets: &HashMap<String, Value>,
    e: &Expr,
) -> Result<Value, GlazeError> {
    let mut ctx = Ctx {
        program,
        variant,
        resolving: Vec::new(),
        lets: lets.clone(),
        vw: f32::INFINITY,
        vh: f32::INFINITY,
    };
    eval(e, &mut ctx)
}

impl Program {
    /// Resolve a style for a variant + active discrete states. Viewport is
    /// treated as infinite (no `when` breakpoint fires).
    pub fn resolve(
        &self,
        style: &str,
        variant: &HashMap<String, String>,
        states: &[&str],
    ) -> Result<CompiledStyle, GlazeError> {
        self.resolve_at(style, variant, states, f32::INFINITY, f32::INFINITY)
    }

    /// Resolve a style at a given viewport size. `when <cond> { … }` blocks whose
    /// condition (over `vw`/`vh`) is true are applied as responsive overrides —
    /// this is how a stylesheet does media-query layout (e.g. row → column).
    ///
    /// Precedence: base props/shaders < matching `when` blocks (doc order) <
    /// matching state overlays.
    pub fn resolve_at(
        &self,
        style: &str,
        variant: &HashMap<String, String>,
        states: &[&str],
        vw: f32,
        vh: f32,
    ) -> Result<CompiledStyle, GlazeError> {
        let def = self
            .styles
            .iter()
            .find(|s| s.name == style)
            .ok_or_else(|| GlazeError::Eval(format!("no style named `{}`", style)))?;
        let mut ctx = Ctx {
            program: self,
            variant,
            resolving: Vec::new(),
            lets: HashMap::new(),
            vw,
            vh,
        };
        let mut out = CompiledStyle::default();
        // pass 1: base props/shaders + let-bindings (skip State/When)
        for item in &def.body {
            match item {
                Item::Prop { name, args } => apply_prop(name, args, &mut ctx, &mut out)?,
                Item::Let { name, value } => {
                    let v = eval(value, &mut ctx)?;
                    ctx.lets.insert(name.clone(), v);
                }
                Item::Shader { overlay, body } => {
                    let cs = crate::shader::compile_shader(self, variant, body, *overlay)?;
                    out.layers.push(Layer::Shader(cs));
                }
                Item::State { .. } | Item::When { .. } => {}
            }
        }
        // pass 2: matching `when` breakpoint overrides, in document order
        for item in &def.body {
            if let Item::When { cond, body } = item {
                if matches!(eval(cond, &mut ctx)?, Value::Bool(true)) {
                    apply_overlay(body, self, variant, &mut ctx, &mut out)?;
                }
            }
        }
        // pass 3: matching state overlays
        for item in &def.body {
            if let Item::State { state, body } = item {
                if states.contains(&state.as_str()) {
                    apply_overlay(body, self, variant, &mut ctx, &mut out)?;
                }
            }
        }
        Ok(out)
    }

    /// Evaluate a token by name (used by tests and for inspecting the token graph).
    pub fn eval_token(&self, name: &str) -> Result<Value, GlazeError> {
        let mut ctx = Ctx {
            program: self,
            variant: &EMPTY,
            resolving: Vec::new(),
            lets: HashMap::new(),
            vw: f32::INFINITY,
            vh: f32::INFINITY,
        };
        eval_ident(name, &mut ctx)
    }
}

use std::sync::LazyLock;
static EMPTY: LazyLock<HashMap<String, String>> = LazyLock::new(HashMap::new);

/// Apply the items of a `when`/state overlay block onto `out`. Nested
/// `when`/state blocks are rejected.
fn apply_overlay(
    items: &[Item],
    program: &Program,
    variant: &HashMap<String, String>,
    ctx: &mut Ctx,
    out: &mut CompiledStyle,
) -> Result<(), GlazeError> {
    for it in items {
        match it {
            Item::Prop { name, args } => apply_prop(name, args, ctx, out)?,
            Item::Let { name, value } => {
                let v = eval(value, ctx)?;
                ctx.lets.insert(name.clone(), v);
            }
            Item::Shader { overlay, body } => {
                let cs = crate::shader::compile_shader(program, variant, body, *overlay)?;
                out.layers.push(Layer::Shader(cs));
            }
            Item::State { .. } => {
                return Err(GlazeError::Parse("nested state blocks are not allowed".into()));
            }
            Item::When { .. } => {
                return Err(GlazeError::Parse("nested `when` blocks are not allowed".into()));
            }
        }
    }
    Ok(())
}

fn apply_prop(
    name: &str,
    args: &[Expr],
    ctx: &mut Ctx,
    out: &mut CompiledStyle,
) -> Result<(), GlazeError> {
    let vals: Vec<Value> = args.iter().map(|e| eval(e, ctx)).collect::<Result<_, _>>()?;
    let want = |n: usize| -> Result<(), GlazeError> {
        if vals.len() == n {
            Ok(())
        } else {
            Err(GlazeError::Eval(format!(
                "`{}` expects {} argument(s), got {}",
                name,
                n,
                vals.len()
            )))
        }
    };
    match name {
        "fill" => {
            want(1)?;
            out.layers.push(Layer::Fill(as_color(&vals[0], name)?));
        }
        "border" => {
            want(2)?;
            out.layers.push(Layer::Border {
                color: as_color(&vals[0], name)?,
                width: as_px(&vals[1], name)?,
            });
        }
        "shadow" => {
            want(1)?;
            out.layers.push(Layer::Shadow {
                color: as_color(&vals[0], name)?,
                blur: 0.0,
                offset_y: 0.0,
            });
        }
        "radius" => {
            want(1)?;
            out.box_.radius = as_px(&vals[0], name)?;
        }
        "gap" => {
            want(1)?;
            out.box_.gap = as_px(&vals[0], name)?;
        }
        "pad" => {
            let p = &mut out.box_.padding;
            match vals.len() {
                1 => *p = [as_px(&vals[0], name)?; 4],
                2 => {
                    let (v, h) = (as_px(&vals[0], name)?, as_px(&vals[1], name)?);
                    *p = [v, h, v, h];
                }
                4 => {
                    *p = [
                        as_px(&vals[0], name)?,
                        as_px(&vals[1], name)?,
                        as_px(&vals[2], name)?,
                        as_px(&vals[3], name)?,
                    ];
                }
                _ => return Err(GlazeError::Eval("`pad` takes 1, 2, or 4 lengths".into())),
            }
        }
        "width" => {
            want(1)?;
            out.box_.width = Some(as_dim(&vals[0], name)?);
        }
        "height" => {
            want(1)?;
            out.box_.height = Some(as_dim(&vals[0], name)?);
        }
        "min_width" => {
            want(1)?;
            out.box_.min_width = Some(as_dim(&vals[0], name)?);
        }
        "max_width" => {
            want(1)?;
            out.box_.max_width = Some(as_dim(&vals[0], name)?);
        }
        "min_height" => {
            want(1)?;
            out.box_.min_height = Some(as_dim(&vals[0], name)?);
        }
        "max_height" => {
            want(1)?;
            out.box_.max_height = Some(as_dim(&vals[0], name)?);
        }
        "direction" => {
            want(1)?;
            out.box_.flex_direction = Some(match &vals[0] {
                Value::Sym(s) if s == "row" => Dir::Row,
                Value::Sym(s) if s == "column" || s == "col" => Dir::Column,
                other => {
                    return Err(GlazeError::Eval(format!(
                        "`direction` expects `row` or `column`, got {other:?}"
                    )));
                }
            });
        }
        "grow" => {
            want(1)?;
            out.box_.flex_grow = Some(as_px(&vals[0], name)?);
        }
        "shrink" => {
            want(1)?;
            out.box_.flex_shrink = Some(as_px(&vals[0], name)?);
        }
        other => return Err(GlazeError::Eval(format!("unknown property `{}`", other))),
    }
    Ok(())
}

fn as_color(v: &Value, prop: &str) -> Result<Rgba, GlazeError> {
    match v {
        Value::Color(c) => Ok(*c),
        _ => Err(GlazeError::Eval(format!("`{}` expects a color, got {:?}", prop, v))),
    }
}
fn as_px(v: &Value, prop: &str) -> Result<f32, GlazeError> {
    match v {
        Value::Len(Length::Px(p)) => Ok(*p),
        Value::Num(n) => Ok(*n as f32),
        _ => Err(GlazeError::Eval(format!("`{}` expects a px length, got {:?}", prop, v))),
    }
}
fn as_dim(v: &Value, prop: &str) -> Result<Dim, GlazeError> {
    match v {
        Value::Len(Length::Px(p)) => Ok(Dim::Px(*p)),
        Value::Len(Length::Pct(p)) => Ok(Dim::Pct(*p)),
        Value::Len(Length::Auto) => Ok(Dim::Auto),
        Value::Num(n) => Ok(Dim::Px(*n as f32)),
        _ => Err(GlazeError::Eval(format!("`{}` expects a dimension, got {:?}", prop, v))),
    }
}

// ---------- expression evaluation ----------

fn eval(e: &Expr, ctx: &mut Ctx) -> Result<Value, GlazeError> {
    match e {
        Expr::Num(v, unit) => Ok(match unit.as_deref() {
            None => Value::Num(*v),
            Some("px") => Value::Len(Length::Px(*v as f32)),
            Some("%") => Value::Len(Length::Pct(*v as f32)),
            Some("em") => Value::Len(Length::Em(*v as f32)),
            Some(u) => return Err(GlazeError::Eval(format!("unknown unit `{}`", u))),
        }),
        Expr::Hex(h) => Ok(Value::Color(hex_to_linear(h)?)),
        Expr::Color { space, nums } => Ok(Value::Color(color_literal(space, nums)?)),
        Expr::Ident(name) => {
            if name == "auto" {
                return Ok(Value::Len(Length::Auto));
            }
            eval_ident(name, ctx)
        }
        Expr::Unary('-', x) => match eval(x, ctx)? {
            Value::Num(n) => Ok(Value::Num(-n)),
            Value::Len(Length::Px(p)) => Ok(Value::Len(Length::Px(-p))),
            other => Err(GlazeError::Eval(format!("cannot negate {:?}", other))),
        },
        Expr::Unary(op, _) => Err(GlazeError::Eval(format!("unknown unary `{}`", op))),
        Expr::Bin(op, l, r) => {
            let a = eval(l, ctx)?;
            let b = eval(r, ctx)?;
            eval_bin(op, a, b)
        }
        Expr::Tern(c, a, b) => match eval(c, ctx)? {
            Value::Bool(true) => eval(a, ctx),
            Value::Bool(false) => eval(b, ctx),
            other => Err(GlazeError::Eval(format!(
                "ternary condition must be a bool, got {:?}",
                other
            ))),
        },
        Expr::Call(name, args) => {
            // user-defined function: inline (substitute args into body) and eval
            let user = ctx
                .program
                .fns
                .iter()
                .find(|f| &f.name == name)
                .map(|f| (f.params.clone(), f.body.clone()));
            if let Some((params, body)) = user {
                if params.len() != args.len() {
                    return Err(GlazeError::Eval(format!(
                        "`{}` expects {} argument(s), got {}",
                        name,
                        params.len(),
                        args.len()
                    )));
                }
                let inlined = crate::ast::subst_params(&body, &params, args);
                return eval(&inlined, ctx);
            }
            let vs: Vec<Value> = args.iter().map(|a| eval(a, ctx)).collect::<Result<_, _>>()?;
            eval_call(name, &vs)
        }
    }
}

fn eval_ident(name: &str, ctx: &mut Ctx) -> Result<Value, GlazeError> {
    // viewport size — available in `when` breakpoint conditions
    if name == "vw" {
        return Ok(Value::Num(ctx.vw as f64));
    }
    if name == "vh" {
        return Ok(Value::Num(ctx.vh as f64));
    }
    // 0. in-scope static `let` binding (shader bodies)
    if let Some(v) = ctx.lets.get(name) {
        return Ok(v.clone());
    }
    // 1. variant parameter
    if let Some(v) = ctx.variant.get(name) {
        return Ok(Value::Sym(v.clone()));
    }
    // 2. token (with cycle detection)
    if let Some(def) = ctx.program.tokens.iter().find(|t| t.name == name) {
        if ctx.resolving.iter().any(|n| n == name) {
            return Err(GlazeError::Eval(format!(
                "token cycle: {} -> {}",
                ctx.resolving.join(" -> "),
                name
            )));
        }
        ctx.resolving.push(name.to_string());
        let def = def.clone();
        let v = eval(&def.value, ctx);
        ctx.resolving.pop();
        return v;
    }
    // 2.5 single-component swizzle of a color-valued token or `let`
    //     (`slate.x`, `accent.r`). Multi-component static swizzles have no
    //     Value representation, so only single channels fold here.
    if let Some((base, sw)) = name.split_once('.') {
        let ch = sw.chars().next();
        if sw.len() == 1 && matches!(ch, Some('x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a')) {
            let base_val = if ctx.lets.contains_key(base) {
                ctx.lets.get(base).cloned()
            } else if ctx.program.tokens.iter().any(|t| t.name == base) {
                Some(eval_ident(base, ctx)?)
            } else {
                None
            };
            if let Some(Value::Color(c)) = base_val {
                let comp = match ch.unwrap() {
                    'x' | 'r' => c.r,
                    'y' | 'g' => c.g,
                    'z' | 'b' => c.b,
                    _ => c.a,
                };
                return Ok(Value::Num(comp as f64));
            }
        }
    }
    // 3. otherwise a bare symbol (e.g. a variant literal like `danger`)
    Ok(Value::Sym(name.to_string()))
}

fn eval_bin(op: &str, a: Value, b: Value) -> Result<Value, GlazeError> {
    use Value::*;
    match op {
        "==" => Ok(Bool(values_eq(&a, &b))),
        ">" | "<" | ">=" | "<=" => {
            if let (Num(x), Num(y)) = (&a, &b) {
                let r = match op {
                    ">" => x > y,
                    "<" => x < y,
                    ">=" => x >= y,
                    _ => x <= y,
                };
                Ok(Bool(r))
            } else {
                Err(GlazeError::Eval(format!("cannot compare {:?} {} {:?}", a, op, b)))
            }
        }
        "+" | "-" | "*" | "/" => eval_arith(op, a, b),
        _ => Err(GlazeError::Eval(format!("unknown operator `{}`", op))),
    }
}

fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Sym(x), Value::Sym(y)) => x == y,
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        _ => false,
    }
}

fn eval_arith(op: &str, a: Value, b: Value) -> Result<Value, GlazeError> {
    use Length::*;
    use Value::*;
    let f = |x: f64, y: f64| match op {
        "+" => x + y,
        "-" => x - y,
        "*" => x * y,
        _ => x / y,
    };
    match (a, b) {
        (Num(x), Num(y)) => Ok(Num(f(x, y))),
        (Len(Px(x)), Len(Px(y))) if op == "+" || op == "-" => Ok(Len(Px(f(x as f64, y as f64) as f32))),
        (Len(Px(x)), Num(y)) if op == "*" || op == "/" => Ok(Len(Px(f(x as f64, y) as f32))),
        (Num(x), Len(Px(y))) if op == "*" => Ok(Len(Px((x * y as f64) as f32))),
        (Color(c), Num(s)) if op == "*" => Ok(Color(scale(c, s as f32))),
        (Num(s), Color(c)) if op == "*" => Ok(Color(scale(c, s as f32))),
        (Color(x), Color(y)) if op == "+" => Ok(Color(add(x, y))),
        (x, y) => Err(GlazeError::Eval(format!("cannot evaluate {:?} {} {:?}", x, op, y))),
    }
}

fn eval_call(name: &str, vs: &[Value]) -> Result<Value, GlazeError> {
    let num = |v: &Value| -> Result<f64, GlazeError> {
        match v {
            Value::Num(n) => Ok(*n),
            _ => Err(GlazeError::Eval(format!("expected number, got {:?}", v))),
        }
    };
    match name {
        // vector constructors fold to a linear Color (components are direct).
        "vec4" if vs.len() == 4 => Ok(Value::Color(Rgba {
            r: num(&vs[0])? as f32,
            g: num(&vs[1])? as f32,
            b: num(&vs[2])? as f32,
            a: num(&vs[3])? as f32,
        })),
        "vec3" if vs.len() == 3 => Ok(Value::Color(Rgba {
            r: num(&vs[0])? as f32,
            g: num(&vs[1])? as f32,
            b: num(&vs[2])? as f32,
            a: 1.0,
        })),
        "mix" if vs.len() == 3 => match (&vs[0], &vs[1]) {
            (Value::Color(a), Value::Color(b)) => {
                Ok(Value::Color(lerp(*a, *b, num(&vs[2])? as f32)))
            }
            (Value::Num(a), Value::Num(b)) => {
                let t = num(&vs[2])?;
                Ok(Value::Num(a + (b - a) * t))
            }
            _ => Err(GlazeError::Eval("mix(color,color,t) or mix(num,num,t)".into())),
        },
        "min" if vs.len() == 2 => Ok(Value::Num(num(&vs[0])?.min(num(&vs[1])?))),
        "max" if vs.len() == 2 => Ok(Value::Num(num(&vs[0])?.max(num(&vs[1])?))),
        "pow" if vs.len() == 2 => Ok(Value::Num(num(&vs[0])?.powf(num(&vs[1])?))),
        "abs" if vs.len() == 1 => Ok(Value::Num(num(&vs[0])?.abs())),
        "sqrt" if vs.len() == 1 => Ok(Value::Num(num(&vs[0])?.sqrt())),
        "sin" if vs.len() == 1 => Ok(Value::Num(num(&vs[0])?.sin())),
        "cos" if vs.len() == 1 => Ok(Value::Num(num(&vs[0])?.cos())),
        "tan" if vs.len() == 1 => Ok(Value::Num(num(&vs[0])?.tan())),
        "floor" if vs.len() == 1 => Ok(Value::Num(num(&vs[0])?.floor())),
        "fract" if vs.len() == 1 => {
            let x = num(&vs[0])?;
            Ok(Value::Num(x - x.floor()))
        }
        "sign" if vs.len() == 1 => Ok(Value::Num(num(&vs[0])?.signum())),
        "exp" if vs.len() == 1 => Ok(Value::Num(num(&vs[0])?.exp())),
        "log" if vs.len() == 1 => Ok(Value::Num(num(&vs[0])?.ln())),
        "clamp" if vs.len() == 3 => {
            Ok(Value::Num(num(&vs[0])?.clamp(num(&vs[1])?, num(&vs[2])?)))
        }
        "step" if vs.len() == 2 => {
            Ok(Value::Num(if num(&vs[1])? < num(&vs[0])? { 0.0 } else { 1.0 }))
        }
        "smoothstep" if vs.len() == 3 => {
            let (e0, e1, x) = (num(&vs[0])?, num(&vs[1])?, num(&vs[2])?);
            let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
            Ok(Value::Num(t * t * (3.0 - 2.0 * t)))
        }
        // These are valid Glaze functions but only meaningful once their inputs
        // can be dynamic / once we evaluate them — kept as explicit hard errors.
        "lighten" | "darken" | "saturate" | "pick_readable" => Err(GlazeError::Unsupported(format!(
            "`{}()` not implemented yet (Stage 1 supports literals + mix/min/max/pow/abs/sqrt)",
            name
        ))),
        _ => Err(GlazeError::Eval(format!(
            "unknown function `{}` (arity {})",
            name,
            vs.len()
        ))),
    }
}

// ---------- color helpers (linear-rgb storage) ----------

fn scale(c: Rgba, s: f32) -> Rgba {
    Rgba { r: c.r * s, g: c.g * s, b: c.b * s, a: c.a }
}
fn add(x: Rgba, y: Rgba) -> Rgba {
    Rgba {
        r: x.r + y.r,
        g: x.g + y.g,
        b: x.b + y.b,
        a: (x.a + y.a).min(1.0),
    }
}
fn lerp(a: Rgba, b: Rgba, t: f32) -> Rgba {
    Rgba {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

fn color_literal(space: &str, nums: &[f64]) -> Result<Rgba, GlazeError> {
    let a = if nums.len() == 4 { nums[3] as f32 } else { 1.0 };
    if nums.len() < 3 {
        return Err(GlazeError::Eval(format!("{}() needs 3 components", space)));
    }
    let (lr, lg, lb) = match space {
        "oklch" => oklch_to_linear(nums[0], nums[1], nums[2]),
        "oklab" => oklab_to_linear(nums[0], nums[1], nums[2]),
        _ => return Err(GlazeError::Eval(format!("unknown color space `{}`", space))),
    };
    Ok(Rgba { r: lr, g: lg, b: lb, a })
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn hex_to_linear(h: &str) -> Result<Rgba, GlazeError> {
    let parse = |s: &str| u8::from_str_radix(s, 16).map(|v| v as f32 / 255.0);
    let (r, g, b, a) = match h.len() {
        3 => {
            let d = |i: usize| {
                let c = &h[i..i + 1];
                u8::from_str_radix(&format!("{c}{c}"), 16).map(|v| v as f32 / 255.0)
            };
            (d(0)?, d(1)?, d(2)?, 1.0)
        }
        6 => (parse(&h[0..2])?, parse(&h[2..4])?, parse(&h[4..6])?, 1.0),
        8 => (
            parse(&h[0..2])?,
            parse(&h[2..4])?,
            parse(&h[4..6])?,
            parse(&h[6..8])?,
        ),
        _ => return Err(GlazeError::Eval(format!("bad hex color `#{}`", h))),
    };
    Ok(Rgba {
        r: srgb_to_linear(r),
        g: srgb_to_linear(g),
        b: srgb_to_linear(b),
        a,
    })
}

fn oklab_to_linear(l: f64, a: f64, b: f64) -> (f32, f32, f32) {
    let l_ = l + 0.396_337_777_4 * a + 0.215_803_757_3 * b;
    let m_ = l - 0.105_561_345_8 * a - 0.063_854_172_8 * b;
    let s_ = l - 0.089_484_177_5 * a - 1.291_485_548_0 * b;
    let (l3, m3, s3) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    let r = 4.076_741_662_1 * l3 - 3.307_711_591_3 * m3 + 0.230_969_929_2 * s3;
    let g = -1.268_438_004_6 * l3 + 2.609_757_401_1 * m3 - 0.341_319_396_5 * s3;
    let bb = -0.004_196_086_3 * l3 - 0.703_418_614_7 * m3 + 1.707_614_701_0 * s3;
    let cl = |x: f64| x.clamp(0.0, 1.0) as f32;
    (cl(r), cl(g), cl(bb))
}

fn oklch_to_linear(l: f64, c: f64, h_deg: f64) -> (f32, f32, f32) {
    let h = h_deg.to_radians();
    oklab_to_linear(l, c * h.cos(), c * h.sin())
}
