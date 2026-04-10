use std::collections::HashMap;

use crate::anim::Easing;
use crate::animated::{AnimatedColor, AnimatedPos, AnimatedValue};
use crate::scene::{CircleNode, GroupNode, Node, RectNode, SceneGraph, TriangleNode};
use crate::tweakables::Tweakables;

use super::parser::Value;

// ── Error ──

#[derive(Debug)]
pub struct RuntimeError {
    pub message: String,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

fn err(msg: impl Into<String>) -> RuntimeError {
    RuntimeError { message: msg.into() }
}

// ── Runtime Value ──

#[derive(Debug, Clone)]
pub enum RtVal {
    Num(f64),
    Str(String),
    Bool(bool),
    Color([f32; 4]),
    List(Vec<HashMap<String, RtVal>>),
    Nil,
}

impl RtVal {
    fn as_num(&self) -> Result<f64, RuntimeError> {
        match self { RtVal::Num(n) => Ok(*n), _ => Err(err(format!("expected number, got {self:?}"))) }
    }
    fn as_str(&self) -> Result<&str, RuntimeError> {
        match self { RtVal::Str(s) => Ok(s), _ => Err(err(format!("expected string, got {self:?}"))) }
    }
    fn as_color(&self) -> Result<[f32; 4], RuntimeError> {
        match self { RtVal::Color(c) => Ok(*c), _ => Err(err(format!("expected color, got {self:?}"))) }
    }
}

// ── State Binding ──
// Tracks which node property should follow which state key.

struct Binding {
    node_id: String,
    property: PropPath,
    state_key: String,
}

#[derive(Clone)]
enum PropPath {
    X, Y, Opacity, Scale, Radius, Width, Height, Size,
    FillR, FillG, FillB, FillA,
}

// ── Sequence ──

struct Sequence {
    steps: Vec<Value>,
    current: usize,
    wait_remaining: f64,
    env_snapshot: HashMap<String, RtVal>, // captured let bindings
}

// ── Program ──

pub struct Program {
    // Source AST
    source: Vec<Value>,

    // Scene graph
    pub graph: SceneGraph,

    // Mutable state
    state: HashMap<String, RtVal>,

    // State -> node property bindings
    bindings: Vec<Binding>,

    // Event handlers (stored as AST)
    on_click: Vec<Vec<Value>>,

    // Active sequences
    sequences: Vec<Sequence>,

    // Color palette for entries
    color_index: usize,
}

fn entry_color(idx: usize) -> [f32; 4] {
    let colors: &[[f32; 4]] = &[
        [0.35, 0.65, 0.95, 1.0],
        [0.95, 0.45, 0.35, 1.0],
        [0.40, 0.85, 0.50, 1.0],
        [0.95, 0.75, 0.30, 1.0],
        [0.75, 0.45, 0.90, 1.0],
        [0.95, 0.55, 0.70, 1.0],
        [0.40, 0.85, 0.85, 1.0],
        [0.90, 0.65, 0.40, 1.0],
    ];
    colors[idx % colors.len()]
}

fn hash_key(key: &str, n: usize) -> usize {
    let mut h: u32 = 5381;
    for b in key.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h as usize % n
}

// ── Env (lexical + state) ──

struct Env<'a> {
    scopes: Vec<HashMap<String, RtVal>>,
    state: &'a HashMap<String, RtVal>,
}

impl<'a> Env<'a> {
    fn new(state: &'a HashMap<String, RtVal>) -> Self {
        Self { scopes: vec![HashMap::new()], state }
    }

    fn push(&mut self) { self.scopes.push(HashMap::new()); }
    fn pop(&mut self) { self.scopes.pop(); }

    fn set(&mut self, name: &str, val: RtVal) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), val);
    }

    fn get(&self, name: &str) -> Option<&RtVal> {
        // Check lexical scopes first
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v);
            }
        }
        // Then check state
        self.state.get(name)
    }

    fn get_num(&self, name: &str) -> Result<f64, RuntimeError> {
        self.get(name).ok_or_else(|| err(format!("undefined: {name}")))?.as_num()
    }
}

// ── Expression evaluation ──

fn eval_expr(val: &Value, env: &Env) -> Result<RtVal, RuntimeError> {
    match val {
        Value::Number(n) => Ok(RtVal::Num(*n)),
        Value::String(s) => Ok(RtVal::Str(s.clone())),
        Value::Bool(b) => Ok(RtVal::Bool(*b)),
        Value::Symbol(name) => {
            // @name syntax is handled by the parser as a symbol starting with @
            // but our parser doesn't do that yet. For now, just look up the name.
            env.get(name).cloned().ok_or_else(|| err(format!("undefined: {name}")))
        }
        Value::List(items) => {
            if items.is_empty() { return Err(err("empty expression")); }
            let head = items[0].as_symbol().ok_or_else(|| err("expected function"))?;
            match head {
                "+" => {
                    let mut sum = 0.0;
                    for item in &items[1..] { sum += eval_expr(item, env)?.as_num()?; }
                    Ok(RtVal::Num(sum))
                }
                "-" => {
                    if items.len() == 2 { return Ok(RtVal::Num(-eval_expr(&items[1], env)?.as_num()?)); }
                    let mut r = eval_expr(&items[1], env)?.as_num()?;
                    for item in &items[2..] { r -= eval_expr(item, env)?.as_num()?; }
                    Ok(RtVal::Num(r))
                }
                "*" => {
                    let mut r = 1.0;
                    for item in &items[1..] { r *= eval_expr(item, env)?.as_num()?; }
                    Ok(RtVal::Num(r))
                }
                "/" => {
                    let mut r = eval_expr(&items[1], env)?.as_num()?;
                    for item in &items[2..] {
                        let d = eval_expr(item, env)?.as_num()?;
                        if d == 0.0 { return Err(err("division by zero")); }
                        r /= d;
                    }
                    Ok(RtVal::Num(r))
                }
                "%" => {
                    let a = eval_expr(&items[1], env)?.as_num()?;
                    let b = eval_expr(&items[2], env)?.as_num()?;
                    Ok(RtVal::Num(a % b))
                }
                "=" => {
                    let a = eval_expr(&items[1], env)?.as_num()?;
                    let b = eval_expr(&items[2], env)?.as_num()?;
                    Ok(RtVal::Bool((a - b).abs() < 0.0001))
                }
                ">" => {
                    let a = eval_expr(&items[1], env)?.as_num()?;
                    let b = eval_expr(&items[2], env)?.as_num()?;
                    Ok(RtVal::Bool(a > b))
                }
                "<" => {
                    let a = eval_expr(&items[1], env)?.as_num()?;
                    let b = eval_expr(&items[2], env)?.as_num()?;
                    Ok(RtVal::Bool(a < b))
                }
                "if" => {
                    if items.len() < 4 { return Err(err("if needs cond, then, else")); }
                    let cond = eval_expr(&items[1], env)?;
                    let truthy = match &cond {
                        RtVal::Bool(b) => *b,
                        RtVal::Num(n) => *n != 0.0,
                        RtVal::Nil => false,
                        _ => true,
                    };
                    if truthy { eval_expr(&items[2], env) } else { eval_expr(&items[3], env) }
                }
                "rgba" => {
                    Ok(RtVal::Color([
                        eval_expr(&items[1], env)?.as_num()? as f32,
                        eval_expr(&items[2], env)?.as_num()? as f32,
                        eval_expr(&items[3], env)?.as_num()? as f32,
                        eval_expr(&items[4], env)?.as_num()? as f32,
                    ]))
                }
                "rgb" => {
                    Ok(RtVal::Color([
                        eval_expr(&items[1], env)?.as_num()? as f32,
                        eval_expr(&items[2], env)?.as_num()? as f32,
                        eval_expr(&items[3], env)?.as_num()? as f32,
                        1.0,
                    ]))
                }
                "hash" => {
                    let key = eval_expr(&items[1], env)?.as_str()?.to_string();
                    let n = eval_expr(&items[2], env)?.as_num()? as usize;
                    Ok(RtVal::Num(hash_key(&key, n) as f64))
                }
                "len" => {
                    let name = items[1].as_symbol().ok_or_else(|| err("len expects symbol"))?;
                    match env.get(name) {
                        Some(RtVal::List(l)) => Ok(RtVal::Num(l.len() as f64)),
                        _ => Ok(RtVal::Num(0.0)),
                    }
                }
                "nth" => {
                    let list_name = items[1].as_symbol().ok_or_else(|| err("nth expects list name"))?;
                    let idx = eval_expr(&items[2], env)?.as_num()? as usize;
                    match env.get(list_name) {
                        Some(RtVal::List(l)) => {
                            if idx < l.len() {
                                // Return as a string (for keys)
                                if let Some(RtVal::Str(s)) = l[idx].get("_value") {
                                    Ok(RtVal::Str(s.clone()))
                                } else {
                                    Ok(RtVal::Nil)
                                }
                            } else {
                                Ok(RtVal::Nil)
                            }
                        }
                        _ => Ok(RtVal::Nil),
                    }
                }
                "get" => {
                    // (get map "key")
                    let map_name = items[1].as_symbol().ok_or_else(|| err("get expects symbol"))?;
                    let key = eval_expr(&items[2], env)?;
                    let key_str = match &key {
                        RtVal::Str(s) => s.clone(),
                        _ => return Err(err("get key must be string")),
                    };
                    match env.get(map_name) {
                        Some(RtVal::List(l)) => {
                            // get on a list doesn't make sense, but...
                            Ok(RtVal::Nil)
                        }
                        _ => Ok(RtVal::Nil),
                    }
                }
                "color-for" => {
                    let idx = eval_expr(&items[1], env)?.as_num()? as usize;
                    Ok(RtVal::Color(entry_color(idx)))
                }
                "count-where" => {
                    // (count-where list-name "field" value)
                    let list_name = items[1].as_symbol().ok_or_else(|| err("count-where expects list name"))?;
                    let field = eval_expr(&items[2], env)?.as_str()?.to_string();
                    let target = eval_expr(&items[3], env)?.as_num()?;
                    match env.get(list_name) {
                        Some(RtVal::List(l)) => {
                            let count = l.iter().filter(|item| {
                                item.get(&field).and_then(|v| v.as_num().ok()) == Some(target)
                            }).count();
                            Ok(RtVal::Num(count as f64))
                        }
                        _ => Ok(RtVal::Num(0.0)),
                    }
                }
                _ => Err(err(format!("unknown function: {head}"))),
            }
        }
        _ => Err(err(format!("cannot evaluate: {val:?}"))),
    }
}

fn eval_num(val: &Value, env: &Env) -> Result<f64, RuntimeError> {
    eval_expr(val, env)?.as_num()
}

// ── Parse kwargs ──

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

// ── Easing ──

fn eval_easing(name: &str) -> Result<Easing, RuntimeError> {
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

// ── Hex color ──

fn parse_hex_color(s: &str) -> Result<[f32; 4], RuntimeError> {
    let s = s.trim_start_matches('#');
    let parse = |lo: usize, hi: usize| -> Result<f32, RuntimeError> {
        Ok(u8::from_str_radix(&s[lo..hi], 16).map_err(|_| err("bad hex"))? as f32 / 255.0)
    };
    match s.len() {
        6 => Ok([parse(0, 2)?, parse(2, 4)?, parse(4, 6)?, 1.0]),
        8 => Ok([parse(0, 2)?, parse(2, 4)?, parse(4, 6)?, parse(6, 8)?]),
        _ => Err(err("hex color must be 6 or 8 digits")),
    }
}

// ── Animated value from DSL ──

fn make_animated(val: &Value, env: &Env, bindings: &mut Vec<Binding>, node_id: &str, prop: PropPath) -> Result<AnimatedValue, RuntimeError> {
    match val {
        Value::Number(n) => Ok(AnimatedValue::constant(*n)),
        Value::Symbol(name) => {
            let v = env.get_num(name)?;
            // If it's a state reference, create a binding
            if env.state.contains_key(name) {
                bindings.push(Binding {
                    node_id: node_id.to_string(),
                    property: prop,
                    state_key: name.to_string(),
                });
                Ok(AnimatedValue::spring(v, 300.0, 18.0))
            } else {
                Ok(AnimatedValue::constant(v))
            }
        }
        Value::List(items) => {
            if items.is_empty() { return Err(err("empty")); }
            let head = items[0].as_symbol().ok_or_else(|| err("expected function"))?;
            match head {
                "spring" => {
                    let target = eval_num(&items[1], env)?;
                    let (_, kw) = parse_kwargs(&items[2..]);
                    let stiffness = kw.get("stiffness").map(|v| eval_num(v, env)).transpose()?.unwrap_or(300.0);
                    let damping = kw.get("damping").map(|v| eval_num(v, env)).transpose()?.unwrap_or(18.0);
                    // Check if the target is a state reference
                    if let Value::Symbol(name) = &items[1] {
                        if env.state.contains_key(name) {
                            bindings.push(Binding {
                                node_id: node_id.to_string(),
                                property: prop,
                                state_key: name.to_string(),
                            });
                        }
                    }
                    Ok(AnimatedValue::spring(target, stiffness, damping))
                }
                "tween" => {
                    let from = eval_num(&items[1], env)?;
                    let to = eval_num(&items[2], env)?;
                    let (_, kw) = parse_kwargs(&items[3..]);
                    let dur = kw.get("duration").map(|v| eval_num(v, env)).transpose()?.unwrap_or(0.5);
                    let easing = kw.get("easing").and_then(|v| v.as_symbol()).unwrap_or("cubic-out");
                    let mut t = AnimatedValue::tween(from, to, dur, eval_easing(easing)?);
                    t.fire();
                    Ok(t)
                }
                "+" | "-" | "*" | "/" | "%" | "if" => {
                    Ok(AnimatedValue::constant(eval_num(val, env)?))
                }
                _ => Err(err(format!("unknown in animated: {head}"))),
            }
        }
        _ => Err(err(format!("bad animated value: {val:?}"))),
    }
}

fn make_color(val: &Value, env: &Env) -> Result<AnimatedColor, RuntimeError> {
    match val {
        Value::String(s) => {
            let c = parse_hex_color(s)?;
            Ok(AnimatedColor::constant(c[0], c[1], c[2], c[3]))
        }
        Value::Symbol(name) => {
            match env.get(name) {
                Some(RtVal::Color(c)) => Ok(AnimatedColor::constant(c[0], c[1], c[2], c[3])),
                _ => Err(err(format!("expected color for {name}"))),
            }
        }
        Value::List(items) => {
            if items.is_empty() { return Err(err("empty color")); }
            let head = items[0].as_symbol().unwrap_or("");
            match head {
                "rgba" | "rgb" | "color-for" | "if" => {
                    let v = eval_expr(val, env)?;
                    let c = v.as_color()?;
                    Ok(AnimatedColor::constant(c[0], c[1], c[2], c[3]))
                }
                _ => Err(err(format!("unknown color form: {head}"))),
            }
        }
        _ => Err(err(format!("bad color: {val:?}"))),
    }
}

// ── Node building ──

fn build_node(val: &Value, env: &mut Env, bindings: &mut Vec<Binding>, id_counter: &mut usize) -> Result<Option<Node>, RuntimeError> {
    let items = val.as_list().ok_or_else(|| err("expected list"))?;
    if items.is_empty() { return Err(err("empty")); }
    let head = items[0].as_symbol().ok_or_else(|| err("expected symbol"))?;

    match head {
        "rect" => {
            let (_, kw) = parse_kwargs(&items[1..]);
            let id = kw.get("id").and_then(|v| v.as_string().or(v.as_symbol())).map(|s| s.to_string())
                .unwrap_or_else(|| { *id_counter += 1; format!("_n{}", id_counter) });

            let x = kw.get("x").map(|v| make_animated(v, env, bindings, &id, PropPath::X)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let y = kw.get("y").map(|v| make_animated(v, env, bindings, &id, PropPath::Y)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let w = kw.get("w").map(|v| make_animated(v, env, bindings, &id, PropPath::Width)).transpose()?.unwrap_or(AnimatedValue::constant(50.0));
            let h = kw.get("h").map(|v| make_animated(v, env, bindings, &id, PropPath::Height)).transpose()?.unwrap_or(AnimatedValue::constant(50.0));
            let radius = kw.get("radius").map(|v| make_animated(v, env, bindings, &id, PropPath::Radius)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let fill = kw.get("fill").map(|v| make_color(v, env)).transpose()?.unwrap_or(AnimatedColor::constant(1.0, 1.0, 1.0, 1.0));
            let opacity = kw.get("opacity").map(|v| make_animated(v, env, bindings, &id, PropPath::Opacity)).transpose()?.unwrap_or(AnimatedValue::constant(1.0));
            let scale = kw.get("scale").map(|v| make_animated(v, env, bindings, &id, PropPath::Scale)).transpose()?.unwrap_or(AnimatedValue::constant(1.0));

            let mut rect = RectNode::new(0.0, 0.0, 0.0, 0.0);
            rect.props.id = Some(id);
            rect.props.pos = AnimatedPos { x, y };
            rect.width = w;
            rect.height = h;
            rect.corner_radius = radius;
            rect.fill = fill;
            rect.props.opacity = opacity;
            rect.props.scale = scale;
            Ok(Some(Node::Rect(rect)))
        }

        "circle" => {
            let (_, kw) = parse_kwargs(&items[1..]);
            let id = kw.get("id").and_then(|v| v.as_string().or(v.as_symbol())).map(|s| s.to_string())
                .unwrap_or_else(|| { *id_counter += 1; format!("_n{}", id_counter) });

            let x = kw.get("x").map(|v| make_animated(v, env, bindings, &id, PropPath::X)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let y = kw.get("y").map(|v| make_animated(v, env, bindings, &id, PropPath::Y)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let r = kw.get("r").map(|v| make_animated(v, env, bindings, &id, PropPath::Radius)).transpose()?.unwrap_or(AnimatedValue::constant(20.0));
            let fill = kw.get("fill").map(|v| make_color(v, env)).transpose()?.unwrap_or(AnimatedColor::constant(1.0, 1.0, 1.0, 1.0));
            let opacity = kw.get("opacity").map(|v| make_animated(v, env, bindings, &id, PropPath::Opacity)).transpose()?.unwrap_or(AnimatedValue::constant(1.0));
            let scale = kw.get("scale").map(|v| make_animated(v, env, bindings, &id, PropPath::Scale)).transpose()?.unwrap_or(AnimatedValue::constant(1.0));

            let mut c = CircleNode::new(0.0, 0.0, 0.0);
            c.props.id = Some(id);
            c.props.pos = AnimatedPos { x, y };
            c.radius = r;
            c.fill = fill;
            c.props.opacity = opacity;
            c.props.scale = scale;
            Ok(Some(Node::Circle(c)))
        }

        "triangle" => {
            let (_, kw) = parse_kwargs(&items[1..]);
            let id = kw.get("id").and_then(|v| v.as_string().or(v.as_symbol())).map(|s| s.to_string())
                .unwrap_or_else(|| { *id_counter += 1; format!("_n{}", id_counter) });

            let x = kw.get("x").map(|v| make_animated(v, env, bindings, &id, PropPath::X)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let y = kw.get("y").map(|v| make_animated(v, env, bindings, &id, PropPath::Y)).transpose()?.unwrap_or(AnimatedValue::constant(0.0));
            let size = kw.get("size").map(|v| make_animated(v, env, bindings, &id, PropPath::Size)).transpose()?.unwrap_or(AnimatedValue::constant(10.0));
            let fill = kw.get("fill").map(|v| make_color(v, env)).transpose()?.unwrap_or(AnimatedColor::constant(1.0, 1.0, 1.0, 1.0));
            let opacity = kw.get("opacity").map(|v| make_animated(v, env, bindings, &id, PropPath::Opacity)).transpose()?.unwrap_or(AnimatedValue::constant(1.0));

            let mut t = TriangleNode::new(0.0, 0.0, 0.0);
            t.props.id = Some(id);
            t.props.pos = AnimatedPos { x, y };
            t.size = size;
            t.fill = fill;
            t.props.opacity = opacity;
            Ok(Some(Node::Triangle(t)))
        }

        "group" => {
            let (_, kw) = parse_kwargs(&items[1..]);
            let mut group = GroupNode::new();
            if let Some(id_val) = kw.get("id") {
                group.props.id = id_val.as_string().or(id_val.as_symbol()).map(|s| s.to_string());
            }
            build_body(&items[1..], &mut group.children, env, bindings, id_counter)?;
            Ok(Some(Node::Group(group)))
        }

        "scene" => {
            let mut group = GroupNode::new();
            let start = if items.len() > 1 && items[1].as_symbol().is_some() {
                group.props.id = items[1].as_symbol().map(|s| s.to_string());
                2
            } else {
                1
            };
            env.push();
            build_body(&items[start..], &mut group.children, env, bindings, id_counter)?;
            env.pop();
            Ok(Some(Node::Group(group)))
        }

        "let" => {
            if items.len() < 3 { return Err(err("let needs name and value")); }
            let name = items[1].as_symbol().ok_or_else(|| err("let name must be symbol"))?;
            let val = eval_expr(&items[2], env)?;
            env.set(name, val);
            Ok(None)
        }

        "def" => {
            // (def name value) — handled at top level to set state, here it's a no-op
            // because state was already set in the first pass
            Ok(None)
        }

        "on" => {
            // Event handlers are extracted in the first pass, not during node building
            Ok(None)
        }

        "each" => {
            if items.len() < 4 { return Err(err("each needs var, range, body")); }
            let var = items[1].as_symbol().ok_or_else(|| err("each var must be symbol"))?;

            // Handle (range N) specially, or evaluate as a number
            let n = if let Some(range_list) = items[2].as_list() {
                if !range_list.is_empty() && range_list[0].as_symbol() == Some("range") {
                    eval_num(&range_list[1], env)? as usize
                } else {
                    eval_num(&items[2], env)? as usize
                }
            } else {
                eval_num(&items[2], env)? as usize
            };

            let mut group = GroupNode::new();
            for i in 0..n {
                env.push();
                env.set(var, RtVal::Num(i as f64));
                build_body(&items[3..], &mut group.children, env, bindings, id_counter)?;
                env.pop();
            }
            Ok(Some(Node::Group(group)))
        }

        _ => Err(err(format!("unknown: {head}"))),
    }
}

fn build_body(items: &[Value], out: &mut Vec<Node>, env: &mut Env, bindings: &mut Vec<Binding>, id_counter: &mut usize) -> Result<(), RuntimeError> {
    let mut i = 0;
    while i < items.len() {
        // Skip keyword-value pairs
        if items[i].as_keyword().is_some() {
            i += 2;
            continue;
        }
        if let Value::List(_) = &items[i] {
            if let Some(node) = build_node(&items[i], env, bindings, id_counter)? {
                out.push(node);
            }
        }
        i += 1;
    }
    Ok(())
}

// ── Apply binding: set a node property's spring target ──

fn apply_binding(node: &mut Node, prop: &PropPath, value: f64) {
    match prop {
        PropPath::X => node.props_mut().pos.x.set_target(value),
        PropPath::Y => node.props_mut().pos.y.set_target(value),
        PropPath::Opacity => node.props_mut().opacity.set_target(value),
        PropPath::Scale => node.props_mut().scale.set_target(value),
        PropPath::Width => {
            if let Some(r) = node.as_rect_mut() { r.width.set_target(value); }
        }
        PropPath::Height => {
            if let Some(r) = node.as_rect_mut() { r.height.set_target(value); }
        }
        PropPath::Radius => {
            if let Some(r) = node.as_rect_mut() { r.corner_radius.set_target(value); }
            if let Some(c) = node.as_circle_mut() { c.radius.set_target(value); }
        }
        PropPath::Size => {
            // triangle
        }
        PropPath::FillR | PropPath::FillG | PropPath::FillB | PropPath::FillA => {
            // TODO: color bindings
        }
    }
}

// ── Program implementation ──

impl Program {
    pub fn compile(source: &str) -> Result<Self, RuntimeError> {
        let values = super::parser::parse(source).map_err(|e| err(e.to_string()))?;

        // First pass: extract (def ...) and (on ...) from top-level forms
        let mut state = HashMap::new();
        let mut on_click = Vec::new();

        fn extract_defs_and_handlers(vals: &[Value], state: &mut HashMap<String, RtVal>, on_click: &mut Vec<Vec<Value>>) -> Result<(), RuntimeError> {
            for val in vals {
                if let Some(items) = val.as_list() {
                    if items.is_empty() { continue; }
                    let head = items[0].as_symbol().unwrap_or("");
                    match head {
                        "def" => {
                            if items.len() < 3 { return Err(err("def needs name and value")); }
                            let name = items[1].as_symbol().ok_or_else(|| err("def name must be symbol"))?.to_string();
                            let init = match &items[2] {
                                Value::Number(n) => RtVal::Num(*n),
                                Value::String(s) => RtVal::Str(s.clone()),
                                Value::Bool(b) => RtVal::Bool(*b),
                                Value::List(inner) => {
                                    if inner.is_empty() || inner[0].as_symbol() == Some("list") {
                                        // (list) or (list "a" "b" ...) -> list of single-value maps
                                        let items: Vec<HashMap<String, RtVal>> = inner[1..].iter().map(|v| {
                                            let mut m = HashMap::new();
                                            match v {
                                                Value::String(s) => { m.insert("_value".into(), RtVal::Str(s.clone())); }
                                                Value::Number(n) => { m.insert("_value".into(), RtVal::Num(*n)); }
                                                _ => {}
                                            }
                                            m
                                        }).collect();
                                        RtVal::List(items)
                                    } else {
                                        RtVal::Nil
                                    }
                                }
                                _ => RtVal::Nil,
                            };
                            state.insert(name, init);
                        }
                        "on" => {
                            if items.len() < 3 { continue; }
                            if items[1].as_keyword() == Some("click") {
                                on_click.push(items[2..].to_vec());
                            }
                        }
                        "scene" => {
                            // Recurse into scene bodies
                            let start = if items.len() > 1 && items[1].as_symbol().is_some() { 2 } else { 1 };
                            extract_defs_and_handlers(&items[start..], state, on_click)?;
                        }
                        _ => {}
                    }
                }
            }
            Ok(())
        }

        extract_defs_and_handlers(&values, &mut state, &mut on_click)?;

        // Second pass: build scene graph
        let mut env = Env::new(&state);
        let mut bindings = Vec::new();
        let mut id_counter = 0;
        let mut graph = SceneGraph::new();

        for val in &values {
            if let Some(node) = build_node(val, &mut env, &mut bindings, &mut id_counter)? {
                graph.add(node);
            }
        }

        // Add an empty "spawned" group for dynamic nodes
        let mut spawned = GroupNode::new();
        spawned.props.id = Some("_spawned".into());
        graph.add(Node::Group(spawned));

        Ok(Self {
            source: values,
            graph,
            state,
            bindings,
            on_click,
            sequences: Vec::new(),
            color_index: 0,
        })
    }

    pub fn tick(&mut self, dt: f64, tw: &Tweakables) {
        // Advance sequences
        self.tick_sequences(dt);

        // Apply bindings: state -> node springs
        for binding in &self.bindings {
            if let Some(RtVal::Num(v)) = self.state.get(&binding.state_key) {
                if self.graph.find_mut(&binding.node_id).is_some() {
                    let node = self.graph.find_mut(&binding.node_id).unwrap();
                    apply_binding(node, &binding.property, *v);
                }
            }
        }

        // Tick the scene graph
        self.graph.tick(dt, tw);
    }

    pub fn draw(&self, scene: &mut vello::Scene, tw: &Tweakables) {
        self.graph.draw(scene, tw);
    }

    pub fn handle_click(&mut self) {
        log::info!("handle_click: {} handlers, state: op-index={:?}, arrow-x={:?}",
            self.on_click.len(),
            self.state.get("op-index"),
            self.state.get("arrow-x"));
        let handlers: Vec<Vec<Value>> = self.on_click.clone();
        for handler_body in &handlers {
            self.exec_body(handler_body);
        }
        log::info!("after click: op-index={:?}, arrow-x={:?}, sequences={}",
            self.state.get("op-index"),
            self.state.get("arrow-x"),
            self.sequences.len());
    }

    fn exec_body(&mut self, body: &[Value]) {
        let state_snapshot = self.state.clone();
        let mut env = Env::new(&state_snapshot);

        for form in body {
            if let Err(e) = self.exec_form(form, &mut env) {
                log::warn!("runtime: {e}");
                return;
            }
        }
    }

    fn exec_form(&mut self, form: &Value, env: &mut Env) -> Result<(), RuntimeError> {
        let items = form.as_list().ok_or_else(|| err("expected list in handler"))?;
        if items.is_empty() { return Ok(()); }
        let head = items[0].as_symbol().ok_or_else(|| err("expected symbol"))?;

        match head {
            "set!" => {
                if items.len() < 3 { return Err(err("set! needs name and value")); }
                let name = items[1].as_symbol().ok_or_else(|| err("set! name must be symbol"))?;
                let val = eval_expr(&items[2], env)?;
                self.state.insert(name.to_string(), val);
            }

            "let" => {
                if items.len() < 3 { return Err(err("let needs name and value")); }
                let name = items[1].as_symbol().ok_or_else(|| err("let name must be symbol"))?;
                let val = eval_expr(&items[2], env)?;
                env.set(name, val);
            }

            "sequence" => {
                self.sequences.push(Sequence {
                    steps: items[1..].to_vec(),
                    current: 0,
                    wait_remaining: 0.0,
                    env_snapshot: env.scopes.iter().flat_map(|s| s.iter()).map(|(k, v)| (k.clone(), v.clone())).collect(),
                });
            }

            "when" => {
                // (when cond body...)
                if items.len() < 3 { return Err(err("when needs cond and body")); }
                let cond = eval_expr(&items[1], env)?;
                let truthy = match &cond {
                    RtVal::Bool(b) => *b,
                    RtVal::Num(n) => *n != 0.0,
                    RtVal::Nil => false,
                    _ => true,
                };
                if truthy {
                    for item in &items[2..] {
                        self.exec_form(item, env)?;
                    }
                }
            }

            "push!" => {
                // (push! list-name (:field val ...))
                if items.len() < 3 { return Err(err("push! needs list name and item")); }
                let list_name = items[1].as_symbol().ok_or_else(|| err("push! list name must be symbol"))?.to_string();
                let item_list = items[2].as_list().ok_or_else(|| err("push! item must be a list"))?;
                let (_, kw) = parse_kwargs(item_list);
                let mut map = HashMap::new();
                for (k, v) in &kw {
                    map.insert(k.clone(), eval_expr(v, env)?);
                }
                if let Some(RtVal::List(list)) = self.state.get_mut(&list_name) {
                    list.push(map);
                }
            }

            "spawn!" => {
                // (spawn! group-id (rect ...))
                if items.len() < 3 { return Err(err("spawn! needs group-id and node")); }
                let group_id = items[1].as_symbol().or(items[1].as_string()).unwrap_or("_spawned");
                let mut new_bindings = Vec::new();
                let mut id_counter = self.graph.root.children.len() + 9000;
                if let Some(mut node) = build_node(&items[2], env, &mut new_bindings, &mut id_counter)? {
                    // Check for :target-x / :target-y kwargs to set spring targets
                    let node_items = items[2].as_list().unwrap_or(&[]);
                    let (_, kw) = parse_kwargs(&node_items[1..]);
                    if let Some(tx) = kw.get("target-x") {
                        let v = eval_num(tx, env)?;
                        node.props_mut().pos.x.set_target(v);
                    }
                    if let Some(ty) = kw.get("target-y") {
                        let v = eval_num(ty, env)?;
                        node.props_mut().pos.y.set_target(v);
                    }
                    self.bindings.extend(new_bindings);
                    if let Some(group) = self.graph.find_mut(group_id).and_then(|n| n.as_group_mut()) {
                        group.children.push(node);
                    }
                }
            }

            "set-prop!" => {
                // (set-prop! node-id :prop value)
                if items.len() < 4 { return Err(err("set-prop! needs id, prop, value")); }
                let node_id = items[1].as_symbol().or(items[1].as_string())
                    .ok_or_else(|| err("set-prop! id must be string/symbol"))?;
                let prop = items[2].as_keyword().ok_or_else(|| err("set-prop! prop must be keyword"))?;
                let state_snap = self.state.clone();
                let eval_env = Env::new(&state_snap);
                let val = eval_num(&items[3], &eval_env)?;
                if let Some(node) = self.graph.find_mut(node_id) {
                    let prop_path = match prop {
                        "x" => Some(PropPath::X),
                        "y" => Some(PropPath::Y),
                        "opacity" => Some(PropPath::Opacity),
                        "scale" => Some(PropPath::Scale),
                        _ => None,
                    };
                    if let Some(p) = prop_path {
                        apply_binding(node, &p, val);
                    }
                }
            }

            _ => {
                // Try evaluating as expression (side-effect-free)
                let state_snap = self.state.clone();
                let eval_env = Env::new(&state_snap);
                let _ = eval_expr(form, &eval_env);
            }
        }

        Ok(())
    }

    fn tick_sequences(&mut self, dt: f64) {
        // Take sequences out to avoid borrow conflicts with self.exec_form
        let mut sequences = std::mem::take(&mut self.sequences);
        let mut to_remove = Vec::new();

        for (i, seq) in sequences.iter_mut().enumerate() {
            if seq.wait_remaining > 0.0 {
                seq.wait_remaining -= dt;
                if seq.wait_remaining > 0.0 {
                    continue;
                }
                seq.current += 1;
            }

            // Execute steps until we hit a (wait) or run out
            while seq.current < seq.steps.len() {
                let step = seq.steps[seq.current].clone();
                if let Some(items) = step.as_list() {
                    if items.first().and_then(|v| v.as_symbol()) == Some("wait") {
                        if items.len() >= 2 {
                            let state_snap = self.state.clone();
                            let env = Env::new(&state_snap);
                            seq.wait_remaining = eval_num(&items[1], &env).unwrap_or(0.5);
                        }
                        break;
                    }
                }

                let env_snap = seq.env_snapshot.clone();
                let state_snap = self.state.clone();
                let mut env = Env::new(&state_snap);
                env.push();
                for (k, v) in &env_snap {
                    env.set(k, v.clone());
                }
                if let Err(e) = self.exec_form(&step, &mut env) {
                    log::warn!("sequence step error: {e}");
                }
                for (k, v) in env.scopes.last().unwrap().iter() {
                    seq.env_snapshot.insert(k.clone(), v.clone());
                }
                seq.current += 1;
            }

            if seq.current >= seq.steps.len() && seq.wait_remaining <= 0.0 {
                to_remove.push(i);
            }
        }

        // Remove finished sequences (reverse order to preserve indices)
        for i in to_remove.into_iter().rev() {
            sequences.remove(i);
        }

        self.sequences = sequences;
    }
}
