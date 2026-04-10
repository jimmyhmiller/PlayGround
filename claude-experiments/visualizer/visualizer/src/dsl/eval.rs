use std::collections::HashMap;

use crate::animated::{AnimatedColor, AnimatedPos, AnimatedValue};
use crate::scene::{CircleNode, GroupNode, Node, RectNode, SceneGraph, TriangleNode};

use super::parser::Value;

#[derive(Debug)]
pub struct EvalError {
    pub message: String,
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "eval error: {}", self.message)
    }
}

fn err(msg: impl Into<String>) -> EvalError {
    EvalError { message: msg.into() }
}

/// Environment for variable bindings during evaluation.
struct Env {
    bindings: Vec<HashMap<String, f64>>,
}

impl Env {
    fn new() -> Self {
        Self { bindings: vec![HashMap::new()] }
    }

    fn push_scope(&mut self) {
        self.bindings.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.bindings.pop();
    }

    fn set(&mut self, name: &str, value: f64) {
        self.bindings.last_mut().unwrap().insert(name.to_string(), value);
    }

    fn get(&self, name: &str) -> Option<f64> {
        for scope in self.bindings.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(*v);
            }
        }
        None
    }
}

/// Evaluate a numeric expression in the current environment.
fn eval_expr(val: &Value, env: &Env) -> Result<f64, EvalError> {
    match val {
        Value::Number(n) => Ok(*n),
        Value::Symbol(name) => {
            env.get(name).ok_or_else(|| err(format!("undefined variable: {name}")))
        }
        Value::List(items) => {
            if items.is_empty() {
                return Err(err("empty expression"));
            }
            let head = items[0].as_symbol().ok_or_else(|| err("expected function name"))?;
            match head {
                "+" => {
                    let mut sum = 0.0;
                    for item in &items[1..] {
                        sum += eval_expr(item, env)?;
                    }
                    Ok(sum)
                }
                "-" => {
                    if items.len() == 2 {
                        return Ok(-eval_expr(&items[1], env)?);
                    }
                    let mut result = eval_expr(&items[1], env)?;
                    for item in &items[2..] {
                        result -= eval_expr(item, env)?;
                    }
                    Ok(result)
                }
                "*" => {
                    let mut product = 1.0;
                    for item in &items[1..] {
                        product *= eval_expr(item, env)?;
                    }
                    Ok(product)
                }
                "/" => {
                    let mut result = eval_expr(&items[1], env)?;
                    for item in &items[2..] {
                        let d = eval_expr(item, env)?;
                        if d == 0.0 {
                            return Err(err("division by zero"));
                        }
                        result /= d;
                    }
                    Ok(result)
                }
                _ => Err(err(format!("unknown function in expression: {head}"))),
            }
        }
        _ => Err(err(format!("expected number or expression, got {val:?}"))),
    }
}

/// Parse keyword arguments from a list, starting after the head symbol.
/// Returns remaining positional args and a map of keyword -> Value.
fn parse_kwargs(items: &[Value]) -> (Vec<&Value>, HashMap<String, &Value>) {
    let mut positional = Vec::new();
    let mut kwargs = HashMap::new();
    let mut i = 0;
    while i < items.len() {
        if let Value::Keyword(k) = &items[i] {
            if i + 1 < items.len() {
                kwargs.insert(k.clone(), &items[i + 1]);
                i += 2;
                continue;
            }
        }
        positional.push(&items[i]);
        i += 1;
    }
    (positional, kwargs)
}

fn eval_color(val: &Value, env: &Env) -> Result<[f32; 4], EvalError> {
    match val {
        Value::String(s) => parse_hex_color(s),
        Value::List(items) => {
            if items.is_empty() {
                return Err(err("empty color expression"));
            }
            let head = items[0].as_symbol().ok_or_else(|| err("expected rgba/rgb"))?;
            match head {
                "rgba" => {
                    if items.len() != 5 {
                        return Err(err("rgba expects 4 args"));
                    }
                    Ok([
                        eval_expr(&items[1], env)? as f32,
                        eval_expr(&items[2], env)? as f32,
                        eval_expr(&items[3], env)? as f32,
                        eval_expr(&items[4], env)? as f32,
                    ])
                }
                "rgb" => {
                    if items.len() != 4 {
                        return Err(err("rgb expects 3 args"));
                    }
                    Ok([
                        eval_expr(&items[1], env)? as f32,
                        eval_expr(&items[2], env)? as f32,
                        eval_expr(&items[3], env)? as f32,
                        1.0,
                    ])
                }
                _ => Err(err(format!("unknown color function: {head}"))),
            }
        }
        _ => Err(err(format!("expected color string or (rgba ...), got {val:?}"))),
    }
}

fn parse_hex_color(s: &str) -> Result<[f32; 4], EvalError> {
    let s = s.trim_start_matches('#');
    if s.len() == 6 {
        let r = u8::from_str_radix(&s[0..2], 16).map_err(|_| err("bad hex color"))? as f32 / 255.0;
        let g = u8::from_str_radix(&s[2..4], 16).map_err(|_| err("bad hex color"))? as f32 / 255.0;
        let b = u8::from_str_radix(&s[4..6], 16).map_err(|_| err("bad hex color"))? as f32 / 255.0;
        Ok([r, g, b, 1.0])
    } else if s.len() == 8 {
        let r = u8::from_str_radix(&s[0..2], 16).map_err(|_| err("bad hex color"))? as f32 / 255.0;
        let g = u8::from_str_radix(&s[2..4], 16).map_err(|_| err("bad hex color"))? as f32 / 255.0;
        let b = u8::from_str_radix(&s[4..6], 16).map_err(|_| err("bad hex color"))? as f32 / 255.0;
        let a = u8::from_str_radix(&s[6..8], 16).map_err(|_| err("bad hex color"))? as f32 / 255.0;
        Ok([r, g, b, a])
    } else {
        Err(err(format!("hex color must be 6 or 8 digits, got {s}")))
    }
}

fn eval_easing(name: &str) -> Result<crate::anim::Easing, EvalError> {
    use crate::anim::Easing;
    match name {
        "linear" => Ok(Easing::Linear),
        "quad-in" => Ok(Easing::QuadIn),
        "quad-out" => Ok(Easing::QuadOut),
        "quad-in-out" => Ok(Easing::QuadInOut),
        "cubic-in" => Ok(Easing::CubicIn),
        "cubic-out" => Ok(Easing::CubicOut),
        "cubic-in-out" => Ok(Easing::CubicInOut),
        "elastic-in" => Ok(Easing::ElasticIn),
        "elastic-out" => Ok(Easing::ElasticOut),
        "elastic-in-out" => Ok(Easing::ElasticInOut),
        "bounce-in" => Ok(Easing::BounceIn),
        "bounce-out" => Ok(Easing::BounceOut),
        "bounce-in-out" => Ok(Easing::BounceInOut),
        "back-in" => Ok(Easing::BackIn),
        "back-out" => Ok(Easing::BackOut),
        "back-in-out" => Ok(Easing::BackInOut),
        _ => Err(err(format!("unknown easing: {name}"))),
    }
}

/// Evaluate an animated value expression.
/// Can be a plain number, (spring value :stiffness S :damping D), or (tween from to :duration D :easing E).
fn eval_animated(val: &Value, env: &Env) -> Result<AnimatedValue, EvalError> {
    match val {
        Value::Number(n) => Ok(AnimatedValue::constant(*n)),
        Value::Symbol(name) => {
            let v = env.get(name).ok_or_else(|| err(format!("undefined: {name}")))?;
            Ok(AnimatedValue::constant(v))
        }
        Value::List(items) => {
            if items.is_empty() {
                return Err(err("empty animated expression"));
            }
            let head = items[0].as_symbol().ok_or_else(|| err("expected spring/tween/expr"))?;
            match head {
                "spring" => {
                    if items.len() < 2 {
                        return Err(err("spring needs a value"));
                    }
                    let target = eval_expr(&items[1], env)?;
                    let (_, kwargs) = parse_kwargs(&items[2..]);
                    let stiffness = kwargs.get("stiffness")
                        .map(|v| eval_expr(v, env))
                        .transpose()?
                        .unwrap_or(300.0);
                    let damping = kwargs.get("damping")
                        .map(|v| eval_expr(v, env))
                        .transpose()?
                        .unwrap_or(18.0);
                    let mut s = AnimatedValue::spring(target, stiffness, damping);
                    s.set_target(target);
                    Ok(s)
                }
                "tween" => {
                    if items.len() < 3 {
                        return Err(err("tween needs from and to values"));
                    }
                    let from = eval_expr(&items[1], env)?;
                    let to = eval_expr(&items[2], env)?;
                    let (_, kwargs) = parse_kwargs(&items[3..]);
                    let duration = kwargs.get("duration")
                        .map(|v| eval_expr(v, env))
                        .transpose()?
                        .unwrap_or(0.5);
                    let easing_name = kwargs.get("easing")
                        .and_then(|v| v.as_symbol())
                        .unwrap_or("cubic-out");
                    let easing = eval_easing(easing_name)?;
                    let mut t = AnimatedValue::tween(from, to, duration, easing);
                    t.fire();
                    Ok(t)
                }
                // Arithmetic expressions evaluate to a constant
                "+" | "-" | "*" | "/" => {
                    let v = eval_expr(val, env)?;
                    Ok(AnimatedValue::constant(v))
                }
                _ => Err(err(format!("unknown animated form: {head}"))),
            }
        }
        _ => Err(err(format!("expected number or expression for animated value, got {val:?}"))),
    }
}

fn eval_animated_color(val: &Value, env: &Env) -> Result<AnimatedColor, EvalError> {
    let c = eval_color(val, env)?;
    Ok(AnimatedColor::constant(c[0], c[1], c[2], c[3]))
}

/// Collect body items from a list, skipping keyword-value pairs.
fn collect_body<'a>(items: &'a [Value]) -> Vec<&'a Value> {
    let mut body = Vec::new();
    let mut i = 0;
    while i < items.len() {
        if items[i].as_keyword().is_some() {
            i += 2; // skip kwarg pair
            continue;
        }
        if matches!(items[i], Value::List(_)) {
            body.push(&items[i]);
        }
        i += 1;
    }
    body
}

/// Evaluate a sequence of body expressions. `let` bindings persist for
/// subsequent siblings (they set in the current scope, not a new one).
fn eval_body(body: &[&Value], out: &mut Vec<Node>, env: &mut Env) -> Result<(), EvalError> {
    for item in body {
        if let Some(node) = eval_node(item, env)? {
            out.push(node);
        }
    }
    Ok(())
}

/// Evaluate a node expression and return a Node.
fn eval_node(val: &Value, env: &mut Env) -> Result<Option<Node>, EvalError> {
    let items = val.as_list().ok_or_else(|| err("expected list for node"))?;
    if items.is_empty() {
        return Err(err("empty node expression"));
    }

    let head = items[0].as_symbol().ok_or_else(|| err("expected node type"))?;
    match head {
        "rect" => {
            let (_, kwargs) = parse_kwargs(&items[1..]);

            let x = kwargs.get("x").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let y = kwargs.get("y").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let w = kwargs.get("w").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(50.0));
            let h = kwargs.get("h").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(50.0));
            let radius = kwargs.get("radius").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let fill = kwargs.get("fill").map(|v| eval_animated_color(v, env)).transpose()?.unwrap_or(AnimatedColor::constant(1.0, 1.0, 1.0, 1.0));
            let opacity = kwargs.get("opacity").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(1.0));
            let scale = kwargs.get("scale").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(1.0));

            let mut rect = RectNode::new(0.0, 0.0, 0.0, 0.0);
            rect.props.pos = AnimatedPos { x, y };
            rect.width = w;
            rect.height = h;
            rect.corner_radius = radius;
            rect.fill = fill;
            rect.props.opacity = opacity;
            rect.props.scale = scale;

            if let Some(id_val) = kwargs.get("id") {
                if let Some(id) = id_val.as_string().or(id_val.as_symbol()) {
                    rect.props.id = Some(id.to_string());
                }
            }

            Ok(Some(Node::Rect(rect)))
        }

        "circle" => {
            let (_, kwargs) = parse_kwargs(&items[1..]);

            let x = kwargs.get("x").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let y = kwargs.get("y").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let r = kwargs.get("r").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(20.0));
            let fill = kwargs.get("fill").map(|v| eval_animated_color(v, env)).transpose()?.unwrap_or(AnimatedColor::constant(1.0, 1.0, 1.0, 1.0));
            let opacity = kwargs.get("opacity").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(1.0));
            let scale = kwargs.get("scale").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(1.0));

            let mut circle = CircleNode::new(0.0, 0.0, 0.0);
            circle.props.pos = AnimatedPos { x, y };
            circle.radius = r;
            circle.fill = fill;
            circle.props.opacity = opacity;
            circle.props.scale = scale;

            if let Some(id_val) = kwargs.get("id") {
                if let Some(id) = id_val.as_string().or(id_val.as_symbol()) {
                    circle.props.id = Some(id.to_string());
                }
            }

            Ok(Some(Node::Circle(circle)))
        }

        "triangle" => {
            let (_, kwargs) = parse_kwargs(&items[1..]);

            let x = kwargs.get("x").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let y = kwargs.get("y").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let size = kwargs.get("size").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(10.0));
            let fill = kwargs.get("fill").map(|v| eval_animated_color(v, env)).transpose()?.unwrap_or(AnimatedColor::constant(1.0, 1.0, 1.0, 1.0));
            let opacity = kwargs.get("opacity").map(|v| eval_animated(v, env)).transpose()?.unwrap_or(AnimatedValue::constant(1.0));

            let mut tri = TriangleNode::new(0.0, 0.0, 0.0);
            tri.props.pos = AnimatedPos { x, y };
            tri.size = size;
            tri.fill = fill;
            tri.props.opacity = opacity;

            Ok(Some(Node::Triangle(tri)))
        }

        "group" => {
            let mut group = GroupNode::new();
            let (_, kwargs) = parse_kwargs(&items[1..]);

            if let Some(id_val) = kwargs.get("id") {
                if let Some(id) = id_val.as_string().or(id_val.as_symbol()) {
                    group.props.id = Some(id.to_string());
                }
            }

            // Collect body items (skip kwarg pairs)
            let body = collect_body(&items[1..]);
            eval_body(&body, &mut group.children, env)?;

            Ok(Some(Node::Group(group)))
        }

        "let" => {
            // (let name expr) — binds in the current scope
            // Handled by eval_body; if encountered standalone, just bind
            if items.len() < 3 {
                return Err(err("let needs name and value"));
            }
            let name = items[1].as_symbol().ok_or_else(|| err("let name must be a symbol"))?;
            let value = eval_expr(&items[2], env)?;
            env.set(name, value);
            Ok(None)
        }

        "scene" => {
            let mut group = GroupNode::new();
            let start = if items.len() > 1 && items[1].as_symbol().is_some() {
                if let Some(name) = items[1].as_symbol() {
                    group.props.id = Some(name.to_string());
                }
                2
            } else {
                1
            };

            let body: Vec<&Value> = items[start..].iter().filter(|v| matches!(v, Value::List(_))).collect();
            env.push_scope();
            eval_body(&body, &mut group.children, env)?;
            env.pop_scope();

            Ok(Some(Node::Group(group)))
        }

        "each" => {
            if items.len() < 4 {
                return Err(err("each needs: var, range, body"));
            }
            let var = items[1].as_symbol().ok_or_else(|| err("each var must be a symbol"))?;
            let range_expr = items[2].as_list().ok_or_else(|| err("each range must be (range N)"))?;
            if range_expr.is_empty() || range_expr[0].as_symbol() != Some("range") {
                return Err(err("expected (range N)"));
            }
            let n = eval_expr(&range_expr[1], env)? as usize;

            let body: Vec<&Value> = items[3..].iter().filter(|v| matches!(v, Value::List(_))).collect();
            let mut group = GroupNode::new();
            for i in 0..n {
                env.push_scope();
                env.set(var, i as f64);
                eval_body(&body, &mut group.children, env)?;
                env.pop_scope();
            }

            Ok(Some(Node::Group(group)))
        }

        _ => Err(err(format!("unknown node type: {head}"))),
    }
}

/// Evaluate parsed DSL expressions into a SceneGraph.
pub fn eval(values: &[Value]) -> Result<SceneGraph, EvalError> {
    let mut graph = SceneGraph::new();
    let mut env = Env::new();

    for val in values {
        if let Some(node) = eval_node(val, &mut env)? {
            graph.add(node);
        }
    }

    Ok(graph)
}
