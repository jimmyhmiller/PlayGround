use std::collections::HashMap;

use crate::anim::Easing;
use crate::animated::{AnimatedColor, AnimatedPos, AnimatedValue};
use crate::scene::{CircleNode, GroupNode, Node, RectNode, SceneGraph, TriangleNode};
use crate::theme;
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

// ── Binding ──
// Tracks which node property should follow which source (state or tweakable).

struct Binding {
    node_id: String,
    property: PropPath,
    source: BindingSource,
}

enum BindingSource {
    State(String),
    Tweakable(String),
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

// ── Journal (undo/redo) ──

/// A single reversible mutation.
#[derive(Clone)]
enum Mutation {
    /// State variable changed: key, old value, new value
    State { key: String, old: RtVal, new: RtVal },
    /// Item pushed to a state list: list key, the pushed item
    PushList { key: String, item: HashMap<String, RtVal> },
    /// Node spawned into a group: group id, AST template of the node, bindings added
    Spawn { group_id: String, template: Value, bindings_added: usize },
}

/// A group of mutations committed together (one click = one step).
struct Step {
    mutations: Vec<Mutation>,
}

struct Journal {
    /// All recorded steps, both past and future (for redo).
    steps: Vec<Step>,
    /// Number of steps currently applied (cursor position).
    /// 0 means no steps applied; steps.len() means all applied.
    cursor: usize,
    /// Mutations accumulating for the in-progress step (being recorded live).
    recording: Option<Vec<Mutation>>,
}

impl Journal {
    fn new() -> Self {
        Self { steps: Vec::new(), cursor: 0, recording: None }
    }

    fn begin_step(&mut self) {
        // Starting a new step invalidates any future (redo) steps
        self.steps.truncate(self.cursor);
        self.recording = Some(Vec::new());
    }

    fn record(&mut self, m: Mutation) {
        if let Some(rec) = &mut self.recording {
            rec.push(m);
        }
    }

    fn commit_step(&mut self) {
        if let Some(rec) = self.recording.take() {
            if !rec.is_empty() {
                self.steps.push(Step { mutations: rec });
                self.cursor = self.steps.len();
            }
        }
    }

    fn can_back(&self) -> bool {
        self.cursor > 0 && self.recording.is_none()
    }

    fn can_forward(&self) -> bool {
        self.cursor < self.steps.len() && self.recording.is_none()
    }
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

    // Undo/redo journal
    journal: Journal,
}

fn entry_color(idx: usize) -> [f32; 4] {
    theme::accent(idx)
}

fn hash_key(key: &str, n: usize) -> usize {
    let mut h: u32 = 5381;
    for b in key.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h as usize % n
}

// ── Env (lexical + state + tweakables) ──

struct Env<'a> {
    scopes: Vec<HashMap<String, RtVal>>,
    /// Lexical bindings whose values are unresolved expressions (because they
    /// contain tweakable references). When a symbol resolves to a deferred
    /// expression, the caller should treat it as a live-derived value.
    deferred: Vec<HashMap<String, Value>>,
    state: &'a HashMap<String, RtVal>,
    tweaks: Option<&'a Tweakables>,
}

impl<'a> Env<'a> {
    fn new(state: &'a HashMap<String, RtVal>) -> Self {
        Self { scopes: vec![HashMap::new()], deferred: vec![HashMap::new()], state, tweaks: None }
    }

    fn new_with_tweaks(state: &'a HashMap<String, RtVal>, tweaks: &'a Tweakables) -> Self {
        Self { scopes: vec![HashMap::new()], deferred: vec![HashMap::new()], state, tweaks: Some(tweaks) }
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
        self.deferred.push(HashMap::new());
    }
    fn pop(&mut self) {
        self.scopes.pop();
        self.deferred.pop();
    }

    fn set(&mut self, name: &str, val: RtVal) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), val);
    }

    fn set_deferred(&mut self, name: &str, expr: Value) {
        self.deferred.last_mut().unwrap().insert(name.to_string(), expr);
    }

    fn get_deferred(&self, name: &str) -> Option<&Value> {
        for scope in self.deferred.iter().rev() {
            if let Some(e) = scope.get(name) {
                return Some(e);
            }
        }
        None
    }

    fn is_tweakable(&self, name: &str) -> bool {
        self.tweaks.map(|t| t.has(name)).unwrap_or(false)
    }

    fn get(&self, name: &str) -> Option<RtVal> {
        // Check lexical scopes first
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        // Then check state
        if let Some(v) = self.state.get(name) {
            return Some(v.clone());
        }
        // Then check tweakables (live value)
        if let Some(tw) = self.tweaks {
            if tw.has(name) {
                return Some(RtVal::Num(tw.get(name)));
            }
        }
        None
    }

    fn get_num(&self, name: &str) -> Result<f64, RuntimeError> {
        self.get(name).ok_or_else(|| err(format!("undefined: {name}")))?.as_num()
    }

    /// Collect all deferred expressions currently in scope (for capturing in closures).
    fn deferred_snapshot(&self) -> HashMap<String, Value> {
        let mut out = HashMap::new();
        for scope in &self.deferred {
            for (k, v) in scope {
                out.insert(k.clone(), v.clone());
            }
        }
        out
    }
}

// ── Expression evaluation ──

fn eval_expr(val: &Value, env: &Env) -> Result<RtVal, RuntimeError> {
    match val {
        Value::Number(n) => Ok(RtVal::Num(*n)),
        Value::String(s) => Ok(RtVal::Str(s.clone())),
        Value::Bool(b) => Ok(RtVal::Bool(*b)),
        Value::Symbol(name) => {
            env.get(name).ok_or_else(|| err(format!("undefined: {name}")))
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
                "color-for" | "palette" => {
                    let idx = eval_expr(&items[1], env)?.as_num()? as usize;
                    Ok(RtVal::Color(entry_color(idx)))
                }
                "bg" => Ok(RtVal::Color(theme::current().background)),
                "stroke-color" => Ok(RtVal::Color(theme::current().stroke)),
                "label-color" => Ok(RtVal::Color(theme::current().label)),
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

/// Check if an expression references any tweakables (live values).
fn expr_has_tweakable(val: &Value, env: &Env) -> bool {
    match val {
        Value::Symbol(name) => env.is_tweakable(name),
        Value::List(items) => items.iter().any(|v| expr_has_tweakable(v, env)),
        _ => false,
    }
}

/// Check if an expression references any tweakables OR deferred let bindings.
fn expr_has_tweakable_or_deferred(val: &Value, env: &Env) -> bool {
    match val {
        Value::Symbol(name) => env.is_tweakable(name) || env.get_deferred(name).is_some(),
        Value::List(items) => items.iter().any(|v| expr_has_tweakable_or_deferred(v, env)),
        _ => false,
    }
}

/// Fully inline deferred let bindings into an expression, so the resulting
/// AST can be evaluated without needing access to the deferred map.
fn inline_deferred(val: &Value, env: &Env) -> Value {
    match val {
        Value::Symbol(name) => {
            if let Some(expr) = env.get_deferred(name) {
                inline_deferred(&expr.clone(), env)
            } else {
                val.clone()
            }
        }
        Value::List(items) => {
            Value::List(items.iter().map(|v| inline_deferred(v, env)).collect())
        }
        _ => val.clone(),
    }
}

/// Freeze all lexical (non-state, non-tweakable) numeric bindings from env
/// so they can be captured by a closure.
fn freeze_lexicals(env: &Env) -> HashMap<String, f64> {
    let mut frozen = HashMap::new();
    for scope in &env.scopes {
        for (k, v) in scope.iter() {
            if !env.state.contains_key(k) && !env.is_tweakable(k) {
                if let RtVal::Num(n) = v {
                    frozen.insert(k.clone(), *n);
                }
            }
        }
    }
    frozen
}

fn make_animated(val: &Value, env: &Env, bindings: &mut Vec<Binding>, node_id: &str, prop: PropPath) -> Result<AnimatedValue, RuntimeError> {
    match val {
        Value::Number(n) => Ok(AnimatedValue::constant(*n)),
        Value::Symbol(name) => {
            // Deferred let binding: substitute the expression and recurse
            if env.get_deferred(name).is_some() {
                let inlined = inline_deferred(val, env);
                return make_animated(&inlined, env, bindings, node_id, prop);
            }
            let v = env.get_num(name)?;
            // State reference: reactive via spring binding
            if env.state.contains_key(name) {
                bindings.push(Binding {
                    node_id: node_id.to_string(),
                    property: prop,
                    source: BindingSource::State(name.to_string()),
                });
                let th = theme::current();
                Ok(AnimatedValue::spring(v, th.spring_stiffness, th.spring_damping))
            } else if env.is_tweakable(name) {
                // Tweakable reference: live reads
                Ok(AnimatedValue::tweakable(name))
            } else {
                Ok(AnimatedValue::constant(v))
            }
        }
        Value::List(items) => {
            if items.is_empty() { return Err(err("empty")); }
            let head = items[0].as_symbol().ok_or_else(|| err("expected function"))?;
            match head {
                "spring" => {
                    let (_, kw) = parse_kwargs(&items[2..]);
                    let th = theme::current();
                    let stiffness = kw.get("stiffness").map(|v| eval_num(v, env)).transpose()?.unwrap_or(th.spring_stiffness);
                    let damping = kw.get("damping").map(|v| eval_num(v, env)).transpose()?.unwrap_or(th.spring_damping);

                    // State-binding case: (spring some-state-symbol ...)
                    if let Value::Symbol(name) = &items[1] {
                        if env.state.contains_key(name) {
                            let target = env.get_num(name)?;
                            bindings.push(Binding {
                                node_id: node_id.to_string(),
                                property: prop,
                                source: BindingSource::State(name.to_string()),
                            });
                            return Ok(AnimatedValue::spring(target, stiffness, damping));
                        }
                    }

                    // Derived-target case: target expression references a tweakable or deferred let
                    if expr_has_tweakable_or_deferred(&items[1], env) {
                        let expr = inline_deferred(&items[1], env);
                        let frozen = freeze_lexicals(env);
                        // Compute initial target value now
                        let initial_target = {
                            let empty_state = HashMap::new();
                            let mut e = match env.tweaks {
                                Some(t) => Env::new_with_tweaks(&empty_state, t),
                                None => Env::new(&empty_state),
                            };
                            for (k, v) in &frozen {
                                e.set(k, RtVal::Num(*v));
                            }
                            eval_num(&expr, &e).unwrap_or(0.0)
                        };
                        // Optional :initial overrides the starting position
                        let initial = kw.get("initial").map(|v| eval_num(v, env)).transpose()?.unwrap_or(initial_target);
                        let source_frozen = frozen.clone();
                        let source_expr = expr.clone();
                        let mut av = AnimatedValue::derived(
                            move |tw| {
                                let empty_state = HashMap::new();
                                let mut e = Env::new_with_tweaks(&empty_state, tw);
                                for (k, v) in &source_frozen {
                                    e.set(k, RtVal::Num(*v));
                                }
                                eval_num(&source_expr, &e).unwrap_or(0.0)
                            },
                            stiffness,
                            damping,
                        );
                        av.set_immediate(initial);
                        return Ok(av);
                    }

                    // Plain spring: target is a constant expression
                    let target = eval_num(&items[1], env)?;
                    let initial = kw.get("initial").map(|v| eval_num(v, env)).transpose()?.unwrap_or(target);
                    let mut av = AnimatedValue::spring(initial, stiffness, damping);
                    av.set_target(target);
                    Ok(av)
                }
                "tween" => {
                    let from = eval_num(&items[1], env)?;
                    let to = eval_num(&items[2], env)?;
                    let (_, kw) = parse_kwargs(&items[3..]);
                    let th = theme::current();
                    let dur = kw.get("duration").map(|v| eval_num(v, env)).transpose()?.unwrap_or(th.tween_duration);
                    let theme_easing = th.tween_easing.clone();
                    let easing = kw.get("easing").and_then(|v| v.as_symbol()).unwrap_or(&theme_easing);
                    let mut t = AnimatedValue::tween(from, to, dur, eval_easing(easing)?);
                    t.fire();
                    Ok(t)
                }
                "+" | "-" | "*" | "/" | "%" | "if" => {
                    // If the expression references any tweakable (directly or via a
                    // deferred let), make it reactive by inlining and creating a
                    // derived-live animated value.
                    if expr_has_tweakable_or_deferred(val, env) {
                        let expr = inline_deferred(val, env);
                        let frozen = freeze_lexicals(env);
                        Ok(AnimatedValue::derived_live(move |tw| {
                            let empty_state = HashMap::new();
                            let mut e = Env::new_with_tweaks(&empty_state, tw);
                            for (k, v) in &frozen {
                                e.set(k, RtVal::Num(*v));
                            }
                            eval_num(&expr, &e).unwrap_or(0.0)
                        }))
                    } else {
                        Ok(AnimatedValue::constant(eval_num(val, env)?))
                    }
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
                "rgba" | "rgb" | "color-for" | "palette" | "bg" | "stroke-color" | "label-color" | "if" => {
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
            // If the expression references tweakables (or chains through a deferred let),
            // store it deferred so it gets re-evaluated live when referenced.
            if expr_has_tweakable_or_deferred(&items[2], env) {
                env.set_deferred(name, items[2].clone());
            } else {
                let val = eval_expr(&items[2], env)?;
                env.set(name, val);
            }
            Ok(None)
        }

        "def" | "defc" => {
            // Handled in the first pass; no-op during node building
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

/// Walk an AST and replace any symbol that resolves to a runtime number
/// with a literal Value::Number. This "freezes" the template so redo works
/// after the sequence env is gone.
fn resolve_template(val: &Value, env: &Env) -> Result<Value, RuntimeError> {
    match val {
        Value::Number(_) | Value::String(_) | Value::Bool(_) | Value::Keyword(_) => Ok(val.clone()),
        Value::Symbol(name) => {
            // Only resolve lexical variables, not state or tweakables (those are reactive)
            if !env.state.contains_key(name) && !env.is_tweakable(name) {
                match env.get(name) {
                    Some(RtVal::Num(n)) => return Ok(Value::Number(n)),
                    Some(RtVal::Str(s)) => return Ok(Value::String(s)),
                    _ => {}
                }
            }
            Ok(val.clone())
        }
        Value::List(items) => {
            let resolved: Result<Vec<Value>, _> = items.iter().map(|v| resolve_template(v, env)).collect();
            Ok(Value::List(resolved?))
        }
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
    pub fn compile(source: &str, tweaks: &mut Tweakables) -> Result<Self, RuntimeError> {
        let values = super::parser::parse(source).map_err(|e| err(e.to_string()))?;

        // Clear any existing tweakables from a previous compile
        tweaks.clear();

        // First pass: extract (def ...), (defc ...) and (on ...) from top-level forms
        let mut state = HashMap::new();
        let mut on_click = Vec::new();

        fn extract(
            vals: &[Value],
            state: &mut HashMap<String, RtVal>,
            on_click: &mut Vec<Vec<Value>>,
            tweaks: &mut Tweakables,
        ) -> Result<(), RuntimeError> {
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
                        "defc" => {
                            // (defc name default :min M :max X :step S :category C)
                            if items.len() < 3 { return Err(err("defc needs name and default")); }
                            let name = items[1].as_symbol().ok_or_else(|| err("defc name must be symbol"))?.to_string();
                            let default = match &items[2] {
                                Value::Number(n) => *n,
                                _ => return Err(err("defc default must be a number")),
                            };
                            let (_, kw) = parse_kwargs(&items[3..]);
                            let empty_state = HashMap::new();
                            let env = Env::new(&empty_state);
                            let min = kw.get("min").and_then(|v| match v {
                                Value::Number(n) => Some(*n),
                                _ => eval_num(v, &env).ok(),
                            }).unwrap_or_else(|| {
                                if default > 0.0 { 0.0 } else { default * 2.0 }
                            });
                            let max = kw.get("max").and_then(|v| match v {
                                Value::Number(n) => Some(*n),
                                _ => eval_num(v, &env).ok(),
                            }).unwrap_or_else(|| {
                                if default > 0.0 { default * 2.0 } else { 0.0 }
                            });
                            let step = kw.get("step").and_then(|v| match v {
                                Value::Number(n) => Some(*n),
                                _ => None,
                            }).unwrap_or_else(|| {
                                let range = max - min;
                                if range >= 10.0 { 1.0 }
                                else if range >= 1.0 { 0.1 }
                                else { 0.01 }
                            });
                            let category = kw.get("category")
                                .and_then(|v| v.as_string().or(v.as_symbol()))
                                .unwrap_or("scene")
                                .to_string();
                            tweaks.register(&name, default, min, max, step, &category);
                        }
                        "on" => {
                            if items.len() < 3 { continue; }
                            if items[1].as_keyword() == Some("click") {
                                on_click.push(items[2..].to_vec());
                            }
                        }
                        "scene" => {
                            let start = if items.len() > 1 && items[1].as_symbol().is_some() { 2 } else { 1 };
                            extract(&items[start..], state, on_click, tweaks)?;
                        }
                        _ => {}
                    }
                }
            }
            Ok(())
        }

        extract(&values, &mut state, &mut on_click, tweaks)?;

        // Build the HTML panel from whatever was registered
        tweaks.build_panel();

        // Second pass: build scene graph
        let mut env = Env::new_with_tweaks(&state, tweaks);
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
            journal: Journal::new(),
        })
    }

    pub fn tick(&mut self, dt: f64, tw: &Tweakables) {
        // Advance sequences
        self.tick_sequences(dt, tw);

        // Apply bindings: retarget node springs from state or tweakables
        for binding in &self.bindings {
            let value = match &binding.source {
                BindingSource::State(key) => {
                    if let Some(RtVal::Num(v)) = self.state.get(key) {
                        Some(*v)
                    } else {
                        None
                    }
                }
                BindingSource::Tweakable(key) => Some(tw.get(key)),
            };
            if let Some(v) = value {
                if let Some(node) = self.graph.find_mut(&binding.node_id) {
                    apply_binding(node, &binding.property, v);
                }
            }
        }

        // Tick the scene graph
        self.graph.tick(dt, tw);
    }

    pub fn draw(&self, scene: &mut vello::Scene, tw: &Tweakables) {
        self.graph.draw(scene, tw);
    }

    pub fn handle_click(&mut self, tw: &Tweakables) {
        if self.journal.recording.is_some() {
            return;
        }
        self.journal.begin_step();

        let handlers: Vec<Vec<Value>> = self.on_click.clone();
        for handler_body in &handlers {
            self.exec_body(handler_body, tw);
        }
    }

    pub fn step_back(&mut self) {
        log::info!("step_back: cursor={}/{} recording={}",
            self.journal.cursor,
            self.journal.steps.len(),
            self.journal.recording.is_some());
        if !self.journal.can_back() {
            log::info!("  can't step back");
            return;
        }
        self.journal.cursor -= 1;
        let step_idx = self.journal.cursor;
        let mutations = self.journal.steps[step_idx].mutations.clone();
        log::info!("  undoing {} mutations", mutations.len());
        for m in mutations.iter().rev() {
            self.apply_inverse(m);
        }
    }

    pub fn step_forward(&mut self) {
        log::info!("step_forward: cursor={}/{} recording={}",
            self.journal.cursor,
            self.journal.steps.len(),
            self.journal.recording.is_some());
        if !self.journal.can_forward() {
            log::info!("  can't step forward");
            return;
        }
        let step_idx = self.journal.cursor;
        self.journal.cursor += 1;
        let mutations = self.journal.steps[step_idx].mutations.clone();
        log::info!("  redoing {} mutations", mutations.len());
        for m in mutations.iter() {
            self.apply_forward(m);
        }
    }

    pub fn can_step_back(&self) -> bool {
        self.journal.can_back()
    }

    pub fn can_step_forward(&self) -> bool {
        self.journal.can_forward()
    }

    fn apply_inverse(&mut self, m: &Mutation) {
        match m {
            Mutation::State { key, old, .. } => {
                self.state.insert(key.clone(), old.clone());
            }
            Mutation::PushList { key, .. } => {
                if let Some(RtVal::List(l)) = self.state.get_mut(key) {
                    l.pop();
                }
            }
            Mutation::Spawn { group_id, bindings_added, .. } => {
                // Remove the last child from the group
                if let Some(group) = self.graph.find_mut(group_id).and_then(|n| n.as_group_mut()) {
                    group.children.pop();
                }
                // Remove the bindings that were added
                let new_len = self.bindings.len().saturating_sub(*bindings_added);
                self.bindings.truncate(new_len);
            }
        }
    }

    fn apply_forward(&mut self, m: &Mutation) {
        match m {
            Mutation::State { key, new, .. } => {
                self.state.insert(key.clone(), new.clone());
            }
            Mutation::PushList { key, item } => {
                if let Some(RtVal::List(l)) = self.state.get_mut(key) {
                    l.push(item.clone());
                }
            }
            Mutation::Spawn { group_id, template, .. } => {
                // Re-build the node from the template
                let state_snap = self.state.clone();
                let mut env = Env::new(&state_snap);
                let mut new_bindings = Vec::new();
                let mut id_counter = self.graph.root.children.len() + 9000;
                if let Some(node) = build_node(template, &mut env, &mut new_bindings, &mut id_counter).ok().flatten() {
                    let n_added = new_bindings.len();
                    self.bindings.extend(new_bindings);
                    if let Some(group) = self.graph.find_mut(group_id).and_then(|n| n.as_group_mut()) {
                        group.children.push(node);
                    }
                    // Update the stored count in case it changed (shouldn't)
                    let _ = n_added;
                }
            }
        }
    }

    fn exec_body(&mut self, body: &[Value], tw: &Tweakables) {
        let state_snapshot = self.state.clone();
        let mut env = Env::new_with_tweaks(&state_snapshot, tw);

        for form in body {
            if let Err(e) = self.exec_form(form, &mut env, tw) {
                log::warn!("runtime: {e}");
                return;
            }
        }
    }

    fn exec_form(&mut self, form: &Value, env: &mut Env, tw: &Tweakables) -> Result<(), RuntimeError> {
        let items = form.as_list().ok_or_else(|| err("expected list in handler"))?;
        if items.is_empty() { return Ok(()); }
        let head = items[0].as_symbol().ok_or_else(|| err("expected symbol"))?;

        match head {
            "set!" => {
                if items.len() < 3 { return Err(err("set! needs name and value")); }
                let name = items[1].as_symbol().ok_or_else(|| err("set! name must be symbol"))?;
                let val = eval_expr(&items[2], env)?;
                let old = self.state.get(name).cloned().unwrap_or(RtVal::Nil);
                self.journal.record(Mutation::State {
                    key: name.to_string(),
                    old,
                    new: val.clone(),
                });
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
                        self.exec_form(item, env, tw)?;
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
                self.journal.record(Mutation::PushList {
                    key: list_name.clone(),
                    item: map.clone(),
                });
                if let Some(RtVal::List(list)) = self.state.get_mut(&list_name) {
                    list.push(map);
                }
            }

            "spawn!" => {
                // (spawn! group-id (rect ...))
                if items.len() < 3 { return Err(err("spawn! needs group-id and node")); }
                let group_id = items[1].as_symbol().or(items[1].as_string()).unwrap_or("_spawned").to_string();
                // Resolve kwargs into a concrete template so undo/redo is deterministic
                let template = resolve_template(&items[2], env)?;
                let mut new_bindings = Vec::new();
                let mut id_counter = self.graph.root.children.len() + 9000;
                if let Some(mut node) = build_node(&template, env, &mut new_bindings, &mut id_counter)? {
                    let node_items = template.as_list().unwrap_or(&[]);
                    let (_, kw) = parse_kwargs(&node_items[1..]);
                    if let Some(tx) = kw.get("target-x") {
                        let v = eval_num(tx, env)?;
                        node.props_mut().pos.x.set_target(v);
                    }
                    if let Some(ty) = kw.get("target-y") {
                        let v = eval_num(ty, env)?;
                        node.props_mut().pos.y.set_target(v);
                    }
                    let bindings_added = new_bindings.len();
                    self.journal.record(Mutation::Spawn {
                        group_id: group_id.clone(),
                        template: template.clone(),
                        bindings_added,
                    });
                    self.bindings.extend(new_bindings);
                    if let Some(group) = self.graph.find_mut(&group_id).and_then(|n| n.as_group_mut()) {
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

    fn tick_sequences(&mut self, dt: f64, tw: &Tweakables) {
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

            while seq.current < seq.steps.len() {
                let step = seq.steps[seq.current].clone();
                if let Some(items) = step.as_list() {
                    if items.first().and_then(|v| v.as_symbol()) == Some("wait") {
                        if items.len() >= 2 {
                            let state_snap = self.state.clone();
                            let env = Env::new_with_tweaks(&state_snap, tw);
                            seq.wait_remaining = eval_num(&items[1], &env).unwrap_or(0.5);
                        }
                        break;
                    }
                }

                let env_snap = seq.env_snapshot.clone();
                let state_snap = self.state.clone();
                let mut env = Env::new_with_tweaks(&state_snap, tw);
                env.push();
                for (k, v) in &env_snap {
                    env.set(k, v.clone());
                }
                if let Err(e) = self.exec_form(&step, &mut env, tw) {
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

        // If no sequences are active and we're recording a step, commit it
        if self.sequences.is_empty() && self.journal.recording.is_some() {
            let mutations = self.journal.recording.as_ref().map(|r| r.len()).unwrap_or(0);
            log::info!("committing step with {mutations} mutations; total steps={}", self.journal.steps.len() + 1);
            self.journal.commit_step();
        }
    }
}
