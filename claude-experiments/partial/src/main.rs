use oxc_allocator::{Allocator, CloneIn, FromIn};
use oxc_ast::ast::*;
use oxc_ast::NONE;
use oxc_ast::AstBuilder;
use oxc_codegen::Codegen;
use oxc_parser::Parser;
use oxc_span::{Atom, SourceType, SPAN};
use oxc_syntax::operator::{AssignmentOperator, BinaryOperator, UnaryOperator, UpdateOperator};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::rc::Rc;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse command line arguments
    let mut filename: Option<String> = None;
    let mut output_file: Option<String> = None;
    let mut trace = false;
    let mut gas: usize = 10000;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if i + 1 < args.len() {
                    output_file = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: -o requires a filename");
                    std::process::exit(1);
                }
            }
            "-g" | "--gas" => {
                if i + 1 < args.len() {
                    gas = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: -g requires a number");
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("Error: -g requires a number");
                    std::process::exit(1);
                }
            }
            "-t" | "--trace" => {
                trace = true;
                i += 1;
            }
            arg if !arg.starts_with('-') => {
                filename = Some(arg.to_string());
                i += 1;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                i += 1;
            }
        }
    }

    let filename = match filename {
        Some(f) => f,
        None => {
            eprintln!("Usage: {} <file.js> [-o output.js] [-t] [-g gas]", args[0]);
            eprintln!("\nOptions:");
            eprintln!("  -o, --output <file>  Write residual program to file");
            eprintln!("  -t, --trace          Print decision trace");
            eprintln!("  -g, --gas <number>   Maximum evaluation steps (default: 10000)");
            eprintln!("\nExample:");
            eprintln!("  {} examples/vm.js", args[0]);
            eprintln!("  {} examples/vm.js -o residual.js -g 100000", args[0]);
            std::process::exit(1);
        }
    };

    let source = match fs::read_to_string(&filename) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", filename, e);
            std::process::exit(1);
        }
    };

    let allocator = Allocator::default();
    let source_type = SourceType::from_path(&filename).unwrap_or_default();
    let parser = Parser::new(&allocator, &source, source_type);
    let result = parser.parse();

    if !result.errors.is_empty() {
        for error in &result.errors {
            eprintln!("Parse error: {:?}", error);
        }
        return;
    }

    let mut pe = PartialEvaluator::new(&allocator, gas);
    pe.trace = trace;

    let eval_result = pe.evaluate(&result.program.body);

    // Always generate and output the residual, even on error
    let residual = pe.residual_code();

    // Print header
    println!("=== Partial evaluation: {} ===\n", filename);

    // Print error if any
    if let Err(ref e) = eval_result {
        println!("=== Stopped: {} ===\n", e);
    }

    pe.print_summary();
    pe.dump_globals();

    // Write to file if -o specified, otherwise print
    if let Some(out_path) = output_file {
        match fs::write(&out_path, &residual) {
            Ok(()) => println!("\nResidual written to: {}", out_path),
            Err(e) => eprintln!("\nError writing {}: {}", out_path, e),
        }
    } else {
        println!("\n=== Residual program ===");
        println!("{}", residual);
    }

    // Show the final computed value if fully evaluated
    if let Some(ref val) = pe.last_value {
        println!("\n=== Result ===");
        println!("{}", val.to_display(&pe));
    }
}

// === JavaScript Value representation ===

#[derive(Debug)]
enum JsValue<'a> {
    Undefined,
    Null,
    Boolean(bool),
    Number(f64),
    String(String),
    // Array with optional source tracking for reference semantics
    // source = (variable_name, index) means this is a reference to var_name[index]
    Array { elements: Vec<Value<'a>>, source: Option<(String, usize)> },
    // Special built-in objects we can track statically
    TextDecoder,  // A TextDecoder instance
    Date,         // The Date constructor/object
    // Typed arrays - all store as bytes/values
    Int8Array(Vec<i8>),
    // Uint8Array with optional backing ArrayBuffer variable name for mutation propagation
    Uint8Array { bytes: Vec<u8>, backing_var: Option<String> },
    Uint8ClampedArray(Vec<u8>),
    Int16Array(Vec<i16>),
    Uint16Array(Vec<u16>),
    Int32Array(Vec<i32>),
    Uint32Array(Vec<u32>),
    Float32Array(Vec<f32>),
    Float64Array(Vec<f64>),
    BigInt64Array(Vec<i64>),
    BigUint64Array(Vec<u64>),
    // ArrayBuffer for raw bytes
    ArrayBuffer(Vec<u8>),
    // DataView for reading typed data from ArrayBuffer
    DataView(Vec<u8>),
    // Plain JavaScript object
    Object(std::collections::HashMap<String, Value<'a>>),
}

impl<'a> JsValue<'a> {
    fn clone_in(&self, allocator: &'a Allocator) -> Self {
        match self {
            JsValue::Undefined => JsValue::Undefined,
            JsValue::Null => JsValue::Null,
            JsValue::Boolean(b) => JsValue::Boolean(*b),
            JsValue::Number(n) => JsValue::Number(*n),
            JsValue::String(s) => JsValue::String(s.clone()),
            JsValue::Array { elements, source } => JsValue::Array {
                elements: elements.iter().map(|v| v.clone_in(allocator)).collect(),
                source: source.clone(),
            },
            JsValue::TextDecoder => JsValue::TextDecoder,
            JsValue::Date => JsValue::Date,
            JsValue::Int8Array(v) => JsValue::Int8Array(v.clone()),
            JsValue::Uint8Array { bytes, backing_var } => JsValue::Uint8Array { bytes: bytes.clone(), backing_var: backing_var.clone() },
            JsValue::Uint8ClampedArray(v) => JsValue::Uint8ClampedArray(v.clone()),
            JsValue::Int16Array(v) => JsValue::Int16Array(v.clone()),
            JsValue::Uint16Array(v) => JsValue::Uint16Array(v.clone()),
            JsValue::Int32Array(v) => JsValue::Int32Array(v.clone()),
            JsValue::Uint32Array(v) => JsValue::Uint32Array(v.clone()),
            JsValue::Float32Array(v) => JsValue::Float32Array(v.clone()),
            JsValue::Float64Array(v) => JsValue::Float64Array(v.clone()),
            JsValue::BigInt64Array(v) => JsValue::BigInt64Array(v.clone()),
            JsValue::BigUint64Array(v) => JsValue::BigUint64Array(v.clone()),
            JsValue::ArrayBuffer(v) => JsValue::ArrayBuffer(v.clone()),
            JsValue::DataView(v) => JsValue::DataView(v.clone()),
            JsValue::Object(props) => JsValue::Object(
                props.iter().map(|(k, v)| (k.clone(), v.clone_in(allocator))).collect()
            ),
        }
    }
}

impl<'a> PartialEq for JsValue<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (JsValue::Undefined, JsValue::Undefined) => true,
            (JsValue::Null, JsValue::Null) => true,
            (JsValue::Boolean(a), JsValue::Boolean(b)) => a == b,
            (JsValue::Number(a), JsValue::Number(b)) => a == b,
            (JsValue::String(a), JsValue::String(b)) => a == b,
            // Objects use reference equality in JS, so different instances are never equal
            (JsValue::Array { .. }, JsValue::Array { .. }) => false,
            (JsValue::TextDecoder, JsValue::TextDecoder) => false,
            (JsValue::Int8Array(_), JsValue::Int8Array(_)) => false,
            (JsValue::Uint8Array { .. }, JsValue::Uint8Array { .. }) => false,
            (JsValue::Uint8ClampedArray(_), JsValue::Uint8ClampedArray(_)) => false,
            (JsValue::Int16Array(_), JsValue::Int16Array(_)) => false,
            (JsValue::Uint16Array(_), JsValue::Uint16Array(_)) => false,
            (JsValue::Int32Array(_), JsValue::Int32Array(_)) => false,
            (JsValue::Uint32Array(_), JsValue::Uint32Array(_)) => false,
            (JsValue::Float32Array(_), JsValue::Float32Array(_)) => false,
            (JsValue::Float64Array(_), JsValue::Float64Array(_)) => false,
            (JsValue::BigInt64Array(_), JsValue::BigInt64Array(_)) => false,
            (JsValue::BigUint64Array(_), JsValue::BigUint64Array(_)) => false,
            (JsValue::ArrayBuffer(_), JsValue::ArrayBuffer(_)) => false,
            (JsValue::DataView(_), JsValue::DataView(_)) => false,
            _ => false,
        }
    }
}

impl<'a> JsValue<'a> {
    fn is_truthy(&self) -> bool {
        match self {
            JsValue::Undefined => false,
            JsValue::Null => false,
            JsValue::Boolean(b) => *b,
            JsValue::Number(n) => *n != 0.0 && !n.is_nan(),
            JsValue::String(s) => !s.is_empty(),
            // All objects are truthy
            JsValue::Array { .. } => true,
            JsValue::TextDecoder => true,
            JsValue::Date => true,
            JsValue::Int8Array(_) => true,
            JsValue::Uint8Array { .. } => true,
            JsValue::Uint8ClampedArray(_) => true,
            JsValue::Int16Array(_) => true,
            JsValue::Uint16Array(_) => true,
            JsValue::Int32Array(_) => true,
            JsValue::Uint32Array(_) => true,
            JsValue::Float32Array(_) => true,
            JsValue::Float64Array(_) => true,
            JsValue::BigInt64Array(_) => true,
            JsValue::BigUint64Array(_) => true,
            JsValue::ArrayBuffer(_) => true,
            JsValue::DataView(_) => true,
            JsValue::Object(_) => true,
        }
    }

    fn as_number(&self) -> Option<f64> {
        match self {
            JsValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    fn is_primitive(&self) -> bool {
        matches!(
            self,
            JsValue::Undefined
                | JsValue::Null
                | JsValue::Boolean(_)
                | JsValue::Number(_)
                | JsValue::String(_)
        )
    }

    fn to_number(&self) -> Option<f64> {
        match self {
            JsValue::Number(n) => Some(*n),
            JsValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            JsValue::Null => Some(0.0),
            JsValue::Undefined => Some(f64::NAN),
            JsValue::String(s) => Some(Self::parse_js_number(s)),
            _ => None,
        }
    }

    fn to_js_string(&self) -> Option<String> {
        match self {
            JsValue::Undefined => Some("undefined".to_string()),
            JsValue::Null => Some("null".to_string()),
            JsValue::Boolean(b) => Some(if *b { "true" } else { "false" }.to_string()),
            JsValue::Number(n) => Some(Self::number_to_string(*n)),
            JsValue::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    fn number_to_string(n: f64) -> String {
        if n.is_nan() {
            "NaN".to_string()
        } else if n.is_infinite() {
            if n.is_sign_negative() {
                "-Infinity".to_string()
            } else {
                "Infinity".to_string()
            }
        } else if n.fract() == 0.0 && n.abs() < 1e15 {
            format!("{}", n as i64)
        } else {
            n.to_string()
        }
    }

    fn parse_js_number(s: &str) -> f64 {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return 0.0;
        }
        match trimmed {
            "NaN" => return f64::NAN,
            "Infinity" | "+Infinity" => return f64::INFINITY,
            "-Infinity" => return f64::NEG_INFINITY,
            _ => {}
        }

        let (sign, body) = if let Some(rest) = trimmed.strip_prefix('-') {
            (-1.0, rest)
        } else if let Some(rest) = trimmed.strip_prefix('+') {
            (1.0, rest)
        } else {
            (1.0, trimmed)
        };

        if let Some(rest) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
            if rest.is_empty() {
                return f64::NAN;
            }
            return match u64::from_str_radix(rest, 16) {
                Ok(v) => sign * v as f64,
                Err(_) => f64::NAN,
            };
        }
        if let Some(rest) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
            if rest.is_empty() {
                return f64::NAN;
            }
            return match u64::from_str_radix(rest, 2) {
                Ok(v) => sign * v as f64,
                Err(_) => f64::NAN,
            };
        }
        if let Some(rest) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
            if rest.is_empty() {
                return f64::NAN;
            }
            return match u64::from_str_radix(rest, 8) {
                Ok(v) => sign * v as f64,
                Err(_) => f64::NAN,
            };
        }

        match body.parse::<f64>() {
            Ok(v) => sign * v,
            Err(_) => f64::NAN,
        }
    }

    fn to_display(&self, pe: &PartialEvaluator<'a>) -> String {
        match self {
            JsValue::Undefined => "undefined".to_string(),
            JsValue::Null => "null".to_string(),
            JsValue::Boolean(b) => b.to_string(),
            JsValue::Number(n) => {
                if n.fract() == 0.0 {
                    format!("{}", *n as i64)
                } else {
                    n.to_string()
                }
            }
            JsValue::String(s) => format!("\"{}\"", s),
            JsValue::Array { elements: arr, source: _ } => {
                let items: Vec<String> = arr.iter().map(|v| v.to_display(pe)).collect();
                format!("[{}]", items.join(", "))
            }
            JsValue::TextDecoder => "TextDecoder {}".to_string(),
            JsValue::Date => "Date".to_string(),
            JsValue::Int8Array(v) => format!("Int8Array({})", v.len()),
            JsValue::Uint8Array { bytes, .. } => format!("Uint8Array({})", bytes.len()),
            JsValue::Uint8ClampedArray(v) => format!("Uint8ClampedArray({})", v.len()),
            JsValue::Int16Array(v) => format!("Int16Array({})", v.len()),
            JsValue::Uint16Array(v) => format!("Uint16Array({})", v.len()),
            JsValue::Int32Array(v) => format!("Int32Array({})", v.len()),
            JsValue::Uint32Array(v) => format!("Uint32Array({})", v.len()),
            JsValue::Float32Array(v) => format!("Float32Array({})", v.len()),
            JsValue::Float64Array(v) => format!("Float64Array({})", v.len()),
            JsValue::BigInt64Array(v) => format!("BigInt64Array({})", v.len()),
            JsValue::BigUint64Array(v) => format!("BigUint64Array({})", v.len()),
            JsValue::ArrayBuffer(v) => format!("ArrayBuffer({})", v.len()),
            JsValue::DataView(v) => format!("DataView({})", v.len()),
            JsValue::Object(props) => {
                let items: Vec<String> = props.iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_display(pe)))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
        }
    }
}

// === Lexical Environment ===

#[derive(Debug, Clone)]
struct Env<'a> {
    bindings: Rc<RefCell<HashMap<String, Value<'a>>>>,
    parent: Option<Rc<Env<'a>>>,
}

impl<'a> Env<'a> {
    fn new() -> Self {
        Env {
            bindings: Rc::new(RefCell::new(HashMap::new())),
            parent: None,
        }
    }

    fn extend(parent: Rc<Env<'a>>) -> Self {
        Env {
            bindings: Rc::new(RefCell::new(HashMap::new())),
            parent: Some(parent),
        }
    }

    fn lookup(&self, name: &str, allocator: &'a Allocator) -> Option<Value<'a>> {
        if let Some(val) = self.bindings.borrow().get(name) {
            return Some(val.clone_in(allocator));
        }
        if let Some(ref parent) = self.parent {
            return parent.lookup(name, allocator);
        }
        None
    }

    fn bind(&self, name: String, value: Value<'a>) {
        self.bindings.borrow_mut().insert(name, value);
    }

    fn set(&self, name: &str, value: Value<'a>) -> bool {
        if self.bindings.borrow().contains_key(name) {
            self.bindings.borrow_mut().insert(name.to_string(), value);
            return true;
        }
        if let Some(ref parent) = self.parent {
            return parent.set(name, value);
        }
        false
    }

    /// Collect all variable names in this environment and all parent environments
    fn collect_var_names(&self, names: &mut HashSet<String>) {
        for key in self.bindings.borrow().keys() {
            names.insert(key.clone());
        }
        if let Some(ref parent) = self.parent {
            parent.collect_var_names(names);
        }
    }
}

// === Partial Evaluation Value (Static or Dynamic) ===

#[derive(Debug)]
enum Value<'a> {
    /// A fully known value at PE time
    Static(JsValue<'a>),
    /// An unknown value - we keep the AST expression for residual code
    Dynamic(Expression<'a>),
    /// A function closure with its captured environment
    Closure {
        params: Vec<String>,
        body: &'a FunctionBody<'a>,
        env: Rc<Env<'a>>,
        original: Option<&'a Function<'a>>,
    },
}

impl<'a> Value<'a> {
    fn is_static(&self) -> bool {
        matches!(self, Value::Static(_))
    }

    /// Returns true if the value is fully known at PE time (static or closure)
    fn is_known(&self) -> bool {
        matches!(self, Value::Static(_) | Value::Closure { .. })
    }

    fn to_display(&self, pe: &PartialEvaluator<'a>) -> String {
        match self {
            Value::Static(v) => v.to_display(pe),
            Value::Dynamic(expr) => pe.expr_to_string(expr),
            Value::Closure { params, .. } => format!("<closure({})>", params.join(", ")),
        }
    }

    fn clone_in(&self, allocator: &'a Allocator) -> Self {
        match self {
            Value::Static(v) => Value::Static(v.clone_in(allocator)),
            Value::Dynamic(expr) => Value::Dynamic(expr.clone_in(allocator)),
            Value::Closure { params, body, env, original } => Value::Closure {
                params: params.clone(),
                body: *body,
                env: Rc::clone(env),
                original: *original,
            },
        }
    }
}

// === Stored Function ===

struct StoredFunction<'a> {
    params: Vec<String>,
    body: &'a oxc_ast::ast::FunctionBody<'a>,
}

// === Partial Evaluator ===

struct PartialEvaluator<'a> {
    allocator: &'a Allocator,
    ast: AstBuilder<'a>,
    functions: HashMap<String, StoredFunction<'a>>,
    env: Rc<Env<'a>>,            // Lexical environment with parent chain
    residual: Vec<Statement<'a>>,   // Generated AST statements
    return_value: Option<Value<'a>>,  // Track return value from function evaluation
    last_value: Option<Value<'a>>,    // Track the last evaluated expression value
    in_function: bool,           // Whether we're inside a function (vs top-level)
    in_specialization: bool,     // Whether we're currently specializing a closure (prevent recursion)
    external_vars: HashSet<String>,  // During specialization: vars that are external (mutations should be residual)
    trace: bool,                 // Whether to print decision trace
    depth: usize,                // Current nesting depth for trace indentation
    gas: usize,              // Fuel to prevent infinite loops
    emitted_top_level_vars: HashSet<String>,  // Track which vars have been emitted at top level
    // Statistics
    stats: Stats,
}

#[derive(Default)]
struct Stats {
    functions_defined: usize,
    functions_called: usize,
    functions_fully_evaluated: usize,
    functions_specialized: usize,
    iifes_encountered: usize,     // immediately invoked function expressions
    unknown_calls: usize,         // calls we couldn't resolve
    unknown_call_counts: HashMap<String, usize>,
    unknown_call_details: HashMap<String, Vec<String>>,
    vars_collapsed: usize,        // static vars that don't need residual
    vars_dynamic: usize,          // dynamic vars that remain in residual
    loops_fully_unrolled: usize,
    loops_emitted: usize,
    switches_resolved: usize,
    switches_emitted: usize,
    strings_decoded: usize,       // strings decoded via TextDecoder
    decoded_strings: HashSet<String>,  // unique strings decoded
}


impl<'a> PartialEvaluator<'a> {
    const UNKNOWN_CALL_TRACK_LIMIT: usize = 200;

    fn new(allocator: &'a Allocator, gas: usize) -> Self {
        Self {
            allocator,
            ast: AstBuilder::new(allocator),
            functions: HashMap::new(),
            env: Rc::new(Env::new()),
            residual: vec![],
            return_value: None,
            last_value: None,
            in_function: false,
            in_specialization: false,
            external_vars: HashSet::new(),
            trace: false,
            depth: 0,
            gas,
            emitted_top_level_vars: HashSet::new(),
            stats: Stats::default(),
        }
    }

    fn trace(&self, msg: &str) {
        if self.trace {
            let indent = "  ".repeat(self.depth);
            println!("{}{}", indent, msg);
        }
    }

    fn trace_decision(&self, what: &str, decision: &str) {
        if self.trace {
            let indent = "  ".repeat(self.depth);
            println!("{}{} -> {}", indent, what, decision);
        }
    }

    fn print_summary(&self) {
        println!("=== Summary ===");
        println!("Functions: {} defined, {} called",
            self.stats.functions_defined, self.stats.functions_called);
        if self.stats.functions_fully_evaluated > 0 {
            println!("  {} fully evaluated (no residual)", self.stats.functions_fully_evaluated);
        }
        if self.stats.functions_specialized > 0 {
            println!("  {} partially specialized", self.stats.functions_specialized);
        }
        if self.stats.iifes_encountered > 0 {
            println!("  {} IIFEs passed through", self.stats.iifes_encountered);
        }
        if self.stats.unknown_calls > 0 {
            println!("  {} unknown/dynamic calls", self.stats.unknown_calls);
            if !self.stats.unknown_call_counts.is_empty() {
                let mut counts: Vec<(&String, &usize)> = self.stats.unknown_call_counts.iter().collect();
                counts.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
                println!("  Top dynamic call sites:");
                for (call, count) in counts.into_iter().take(10) {
                    println!("    {}x {}", count, call);
                    if let Some(details) = self.stats.unknown_call_details.get(call) {
                        if !details.is_empty() {
                            println!("      vars: {}", details.join(", "));
                        }
                    }
                }
            }
        }
        println!("Variables: {} collapsed, {} remain dynamic",
            self.stats.vars_collapsed, self.stats.vars_dynamic);
        if self.stats.loops_fully_unrolled > 0 || self.stats.loops_emitted > 0 {
            println!("Loops: {} fully unrolled, {} emitted as residual",
                self.stats.loops_fully_unrolled, self.stats.loops_emitted);
        }
        if self.stats.switches_resolved > 0 || self.stats.switches_emitted > 0 {
            println!("Switches: {} resolved statically, {} emitted as residual",
                self.stats.switches_resolved, self.stats.switches_emitted);
        }
        if self.stats.strings_decoded > 0 {
            println!("Strings decoded: {}", self.stats.strings_decoded);
        }
    }

    fn record_unknown_call(&mut self, expr: &Expression<'a>) {
        let label = self.expr_to_string(expr);
        if self.stats.unknown_call_counts.len() < Self::UNKNOWN_CALL_TRACK_LIMIT
            || self.stats.unknown_call_counts.contains_key(&label)
        {
            *self.stats.unknown_call_counts.entry(label.clone()).or_insert(0) += 1;
            if !self.stats.unknown_call_details.contains_key(&label) {
                let mut details = Vec::new();
                for name in ["v23", "v24", "v30"] {
                    if label.contains(name) {
                        details.push(format!("{}={}", name, self.describe_var(name)));
                    }
                }
                if !details.is_empty() {
                    self.stats.unknown_call_details.insert(label, details);
                }
            }
        }
    }

    fn describe_var(&self, name: &str) -> String {
        match self.env.lookup(name, self.allocator) {
            Some(val) => self.describe_value_short(&val),
            None => "unbound".to_string(),
        }
    }

    fn describe_value_short(&self, val: &Value<'a>) -> String {
        match val {
            Value::Dynamic(_) => "dynamic".to_string(),
            Value::Closure { .. } => "closure".to_string(),
            Value::Static(js) => match js {
                JsValue::Undefined => "static undefined".to_string(),
                JsValue::Null => "static null".to_string(),
                JsValue::Boolean(b) => format!("static boolean({})", b),
                JsValue::Number(n) => {
                    if n.fract() == 0.0 {
                        format!("static number({})", *n as i64)
                    } else {
                        format!("static number({})", n)
                    }
                }
                JsValue::String(s) => format!("static string(len={})", s.len()),
                JsValue::Array { elements, .. } => {
                    format!("static array(len={})", elements.len())
                }
                JsValue::Object(_) => "static object".to_string(),
                JsValue::TextDecoder => "static TextDecoder".to_string(),
                JsValue::Date => "static Date".to_string(),
                JsValue::Int8Array(v) => format!("static Int8Array(len={})", v.len()),
                JsValue::Uint8Array { bytes, .. } => format!("static Uint8Array(len={})", bytes.len()),
                JsValue::Uint8ClampedArray(v) => format!("static Uint8ClampedArray(len={})", v.len()),
                JsValue::Int16Array(v) => format!("static Int16Array(len={})", v.len()),
                JsValue::Uint16Array(v) => format!("static Uint16Array(len={})", v.len()),
                JsValue::Int32Array(v) => format!("static Int32Array(len={})", v.len()),
                JsValue::Uint32Array(v) => format!("static Uint32Array(len={})", v.len()),
                JsValue::Float32Array(v) => format!("static Float32Array(len={})", v.len()),
                JsValue::Float64Array(v) => format!("static Float64Array(len={})", v.len()),
                JsValue::BigInt64Array(v) => format!("static BigInt64Array(len={})", v.len()),
                JsValue::BigUint64Array(v) => format!("static BigUint64Array(len={})", v.len()),
                JsValue::ArrayBuffer(v) => format!("static ArrayBuffer(len={})", v.len()),
                JsValue::DataView(v) => format!("static DataView(len={})", v.len()),
            },
        }
    }

    fn value_to_number(&self, val: &Value<'a>) -> Option<f64> {
        match val {
            Value::Static(js) => js.to_number(),
            _ => None,
        }
    }

    fn slice_index(&self, raw: f64, len: usize) -> usize {
        if raw.is_nan() {
            return 0;
        }
        if raw.is_infinite() {
            return if raw.is_sign_negative() { 0 } else { len };
        }
        let mut idx = raw.trunc() as isize;
        let len_i = len as isize;
        if idx < 0 {
            idx = len_i + idx;
            if idx < 0 {
                return 0;
            }
        }
        if idx > len_i {
            return len;
        }
        idx as usize
    }

    /// Dump final state of interesting globals (myGlobal, document listeners, etc.)
    fn dump_globals(&self) {
        // Check for myGlobal.listeners
        match self.env.lookup("myGlobal", self.allocator) {
            Some(Value::Static(JsValue::Object(ref props))) => {
                println!("\n=== myGlobal state ===");
                println!("  Properties: {:?}", props.keys().collect::<Vec<_>>());
                match props.get("listeners") {
                    Some(Value::Static(JsValue::Array { elements, .. })) => {
                        println!("  listeners.length = {}", elements.len());
                        if !elements.is_empty() {
                            println!("\n=== Event Listeners Registered ===");
                            for (i, elem) in elements.iter().enumerate() {
                                if let Value::Static(JsValue::Object(listener)) = elem {
                                    let event_type = listener.get("type")
                                        .map(|v| v.to_display(self))
                                        .unwrap_or_else(|| "unknown".to_string());
                                    let handler_info = listener.get("f")
                                        .map(|v| match v {
                                            Value::Closure { body, .. } => {
                                                // Try to extract what the handler does
                                                format!("function() {{ ... {} statements ... }}",
                                                    body.statements.len())
                                            }
                                            _ => v.to_display(self)
                                        })
                                        .unwrap_or_else(|| "unknown".to_string());
                                    println!("  {}. Event: {}, Handler: {}", i + 1, event_type, handler_info);
                                } else {
                                    println!("  {}. Listener: {:?}", i + 1, elem.to_display(self));
                                }
                            }
                        }
                    }
                    Some(other) => println!("  listeners = {:?}", other.to_display(self)),
                    None => println!("  No 'listeners' property"),
                }
            }
            Some(other) => println!("\n=== myGlobal = {:?} ===", other.to_display(self)),
            None => println!("\n=== myGlobal not found in environment ==="),
        }
    }

    /// Convert an Expression to a JavaScript source string using codegen
    fn expr_to_string(&self, expr: &Expression<'a>) -> String {
        // Create a minimal program with just an expression statement
        let expr_clone = expr.clone_in(self.allocator);
        let stmt = self.ast.statement_expression(SPAN, expr_clone);
        let stmts = self.ast.vec1(stmt);
        let program = self.ast.program(
            SPAN,
            SourceType::mjs(),
            "",  // source_text
            self.ast.vec(),  // comments
            None,  // hashbang
            self.ast.vec(),  // directives
            stmts,  // body
        );
        let codegen = Codegen::new().build(&program);
        // Remove trailing semicolon and newline
        codegen.code.trim().trim_end_matches(';').to_string()
    }

    /// Convert a JsValue to an AST Expression
    fn js_value_to_expr(&mut self, val: &JsValue<'a>) -> Expression<'a> {
        match val {
            JsValue::Undefined => {
                self.ast.expression_identifier(SPAN, "undefined")
            }
            JsValue::Null => {
                self.ast.expression_null_literal(SPAN)
            }
            JsValue::Boolean(b) => {
                self.ast.expression_boolean_literal(SPAN, *b)
            }
            JsValue::Number(n) => {
                // Handle negative numbers
                if *n < 0.0 {
                    let pos = self.ast.expression_numeric_literal(SPAN, -n, None, NumberBase::Decimal);
                    self.ast.expression_unary(SPAN, UnaryOperator::UnaryNegation, pos)
                } else {
                    self.ast.expression_numeric_literal(SPAN, *n, None, NumberBase::Decimal)
                }
            }
            JsValue::String(s) => {
                let atom: Atom<'a> = Atom::from_in(s.as_str(), self.allocator);
                self.ast.expression_string_literal(SPAN, atom, None)
            }
            JsValue::Array { elements: arr, source: _ } => {
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    arr.iter().map(|v| {
                        let expr = self.value_to_expr(v);
                        ArrayExpressionElement::from(expr)
                    })
                );
                self.ast.expression_array(SPAN, elements)
            }
            JsValue::TextDecoder => {
                // new TextDecoder()
                let callee = self.ast.expression_identifier(SPAN, "TextDecoder");
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec())
            }
            JsValue::Date => {
                // Just the Date identifier
                self.ast.expression_identifier(SPAN, "Date")
            }
            JsValue::Uint8Array { bytes, .. } => {
                // new Uint8Array([...bytes])
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    bytes.iter().map(|b| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *b as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "Uint8Array");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::Int8Array(vals) => {
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    vals.iter().map(|v| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *v as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "Int8Array");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::Uint8ClampedArray(bytes) => {
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    bytes.iter().map(|b| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *b as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "Uint8ClampedArray");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::Int16Array(vals) => {
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    vals.iter().map(|v| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *v as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "Int16Array");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::Uint16Array(vals) => {
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    vals.iter().map(|v| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *v as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "Uint16Array");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::Int32Array(vals) => {
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    vals.iter().map(|v| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *v as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "Int32Array");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::Uint32Array(vals) => {
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    vals.iter().map(|v| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *v as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "Uint32Array");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::Float32Array(vals) => {
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    vals.iter().map(|v| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *v as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "Float32Array");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::Float64Array(vals) => {
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    vals.iter().map(|v| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *v, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "Float64Array");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::BigInt64Array(vals) => {
                // BigInt64Array needs BigInt literals
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    vals.iter().map(|v| {
                        // For now, just use numeric literal - BigInt would need special handling
                        let expr = self.ast.expression_numeric_literal(SPAN, *v as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "BigInt64Array");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::BigUint64Array(vals) => {
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    vals.iter().map(|v| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *v as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "BigUint64Array");
                let arg = Argument::from(arr);
                self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg))
            }
            JsValue::ArrayBuffer(bytes) => {
                // ArrayBuffer can't be directly created from array literals,
                // but we can create from Uint8Array.buffer
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    bytes.iter().map(|b| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *b as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let callee = self.ast.expression_identifier(SPAN, "Uint8Array");
                let arg = Argument::from(arr);
                let typed_arr = self.ast.expression_new(SPAN, callee, NONE, self.ast.vec1(arg));
                // typed_arr.buffer
                self.make_static_member(typed_arr, "buffer")
            }
            JsValue::DataView(bytes) => {
                // new DataView(new Uint8Array([...]).buffer)
                let elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                    bytes.iter().map(|b| {
                        let expr = self.ast.expression_numeric_literal(SPAN, *b as f64, None, NumberBase::Decimal);
                        ArrayExpressionElement::from(expr)
                    })
                );
                let arr = self.ast.expression_array(SPAN, elements);
                let ua_callee = self.ast.expression_identifier(SPAN, "Uint8Array");
                let arg = Argument::from(arr);
                let typed_arr = self.ast.expression_new(SPAN, ua_callee, NONE, self.ast.vec1(arg));
                let buffer = self.make_static_member(typed_arr, "buffer");
                let dv_callee = self.ast.expression_identifier(SPAN, "DataView");
                let buffer_arg = Argument::from(buffer);
                self.ast.expression_new(SPAN, dv_callee, NONE, self.ast.vec1(buffer_arg))
            }
            JsValue::Object(props) => {
                // Build an object expression from properties
                let properties: oxc_allocator::Vec<'a, ObjectPropertyKind<'a>> = self.ast.vec_from_iter(
                    props.iter().map(|(key, value)| {
                        let key_atom: Atom<'a> = Atom::from_in(key.as_str(), self.allocator);
                        let key_expr = PropertyKey::StaticIdentifier(
                            self.ast.alloc(IdentifierName { span: SPAN, name: key_atom })
                        );
                        let value_expr = self.value_to_expr(value);
                        ObjectPropertyKind::ObjectProperty(self.ast.alloc_object_property(
                            SPAN, PropertyKind::Init, key_expr, value_expr, false, false, false
                        ))
                    })
                );
                self.ast.expression_object(SPAN, properties)
            }
        }
    }

    /// Convert a Value (Static or Dynamic) to an AST Expression
    fn value_to_expr(&mut self, val: &Value<'a>) -> Expression<'a> {
        match val {
            Value::Static(js) => self.js_value_to_expr(js),
            Value::Dynamic(expr) => expr.clone_in(self.allocator),
            Value::Closure { params, body, env, original } => {
                // Residualize the closure as a function expression
                // First, collect free variables that need to be captured
                let mut free_vars: HashSet<String> = HashSet::new();
                let mut local_vars: HashSet<String> = HashSet::new();

                if let Some(func) = original {
                    let func_params: HashSet<String> = func.params.items.iter()
                        .filter_map(|p| {
                            match &p.pattern {
                                BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
                                _ => None,
                            }
                        })
                        .collect();
                    if let Some(fn_body) = &func.body {
                        for stmt in &fn_body.statements {
                            self.collect_free_vars_stmt(stmt, &mut free_vars);
                        }
                        for stmt in &fn_body.statements {
                            self.collect_hoisted_vars_stmt(stmt, &mut local_vars);
                        }
                    }
                    for param in &func_params {
                        free_vars.remove(param);
                    }
                } else {
                    for stmt in &body.statements {
                        self.collect_free_vars_stmt(stmt, &mut free_vars);
                    }
                    for stmt in &body.statements {
                        self.collect_hoisted_vars_stmt(stmt, &mut local_vars);
                    }
                }
                for local in &local_vars {
                    free_vars.remove(local);
                }
                // Remove params
                for param in params {
                    free_vars.remove(param);
                }

                // Collect captured values from the closure's IMMEDIATE environment
                // (not from parent/global scopes - only from the function scope that created this closure)
                // This ensures each closure gets its own copy of locally-captured values
                //
                // IMPORTANT: Only do this for closures created inside functions (env has a parent).
                // Top-level closures (env has no parent) should reference globals directly.
                let mut captured_vars: Vec<(String, Value<'a>)> = Vec::new();
                if env.parent.is_some() {
                    // This closure was created inside a function - capture its local scope
                    let env_bindings = env.bindings.borrow();
                    for var_name in &free_vars {
                        // Only capture from the immediate env, not parent scopes
                        if let Some(var_value) = env_bindings.get(var_name) {
                            // Only capture static values (not closures or dynamics)
                            if matches!(&var_value, Value::Static(_)) {
                                captured_vars.push((var_name.clone(), var_value.clone_in(self.allocator)));
                            }
                        }
                    }
                    drop(env_bindings);
                }
                // For top-level closures (env.parent.is_none()), captured_vars stays empty
                // and the closure will reference globals directly

                // Try to specialize the closure body if:
                // 1. The closure has no parameters (or all params are known)
                // 2. We have captured static values
                // 3. We're not already specializing (prevent infinite recursion)
                // This allows us to unroll loops and resolve switches at compile time
                if params.is_empty() && !captured_vars.is_empty() && !self.in_specialization {
                    // Try to specialize the body with captured values
                    if let Some(spec_stmts) = self.try_specialize_closure_body(body, &captured_vars) {
                        // Successfully specialized! Build a function with the specialized body
                        let directives = self.ast.vec();
                        let spec_body_vec: oxc_allocator::Vec<'a, Statement<'a>> =
                            self.ast.vec_from_iter(spec_stmts.into_iter());
                        let spec_body = self.ast.function_body(SPAN, directives, spec_body_vec);

                        let empty_params = self.ast.formal_parameters(
                            SPAN, FormalParameterKind::FormalParameter, self.ast.vec(), None::<FormalParameterRest>,
                        );

                        let func = self.ast.expression_function(
                            SPAN, FunctionType::FunctionExpression, None::<BindingIdentifier>,
                            false, false, false,
                            None::<TSTypeParameterDeclaration>, None::<TSThisParameter>,
                            empty_params, None::<TSTypeAnnotation>, Some(spec_body),
                        );

                        return func;
                    }
                }

                // Build the inner function expression (fallback or no captured vars)
                let inner_func = if let Some(func) = original {
                    let func_clone = (*func).clone_in(self.allocator);
                    Expression::FunctionExpression(self.ast.alloc(func_clone))
                } else {
                    let param_items: oxc_allocator::Vec<'a, FormalParameter<'a>> = self.ast.vec_from_iter(
                        params.iter().map(|p| {
                            let pattern = self.ast.binding_pattern_binding_identifier(SPAN, self.alloc_str(p.as_str()));
                            self.ast.formal_parameter(
                                SPAN, self.ast.vec(), pattern,
                                None::<TSTypeAnnotation>, None::<Expression>, false, None, false, false,
                            )
                        })
                    );
                    let formal_params = self.ast.formal_parameters(
                        SPAN, FormalParameterKind::FormalParameter, param_items, None::<FormalParameterRest>,
                    );
                    let body_clone = (*body).clone_in(self.allocator);
                    let func = self.ast.function(
                        SPAN, FunctionType::FunctionExpression, None::<BindingIdentifier>,
                        false, false, false,
                        None::<TSTypeParameterDeclaration>, None::<TSThisParameter>,
                        formal_params, None::<TSTypeAnnotation>, Some(body_clone),
                    );
                    Expression::FunctionExpression(self.ast.alloc(func))
                };

                // If there are captured values, wrap in an IIFE that provides them
                if !captured_vars.is_empty() {
                    // Build: (function(var1, var2, ...) { return innerFunc; })(val1, val2, ...)
                    let mut iife_stmts: Vec<Statement<'a>> = Vec::new();

                    // Add var declarations for captured values
                    for (var_name, var_value) in &captured_vars {
                        let name_alloc: &'a str = self.alloc_str(var_name.as_str());
                        let init_expr = self.value_to_expr(var_value);
                        let stmt = self.build_var_decl_stmt(name_alloc, Some(init_expr));
                        iife_stmts.push(stmt);
                    }

                    // Add return statement with the inner function
                    let return_stmt = self.ast.statement_return(SPAN, Some(inner_func));
                    iife_stmts.push(return_stmt);

                    // Build the IIFE wrapper function
                    let directives = self.ast.vec();
                    let iife_body_vec: oxc_allocator::Vec<'a, Statement<'a>> = self.ast.vec_from_iter(iife_stmts.into_iter());
                    let iife_body = self.ast.function_body(SPAN, directives, iife_body_vec);

                    let empty_params = self.ast.formal_parameters(
                        SPAN, FormalParameterKind::FormalParameter, self.ast.vec(), None::<FormalParameterRest>,
                    );

                    let wrapper_func = self.ast.expression_function(
                        SPAN, FunctionType::FunctionExpression, None::<BindingIdentifier>,
                        false, false, false,
                        None::<TSTypeParameterDeclaration>, None::<TSThisParameter>,
                        empty_params, None::<TSTypeAnnotation>, Some(iife_body),
                    );

                    // Build the IIFE call with no arguments (vars are declared inside)
                    let iife_call = self.ast.expression_call(
                        SPAN, wrapper_func, None::<TSTypeParameterInstantiation>, self.ast.vec(), false
                    );

                    iife_call
                } else {
                    inner_func
                }
            }
        }
    }

    /// Create a static member expression (obj.prop)
    fn make_static_member(&self, obj: Expression<'a>, prop: &'a str) -> Expression<'a> {
        let prop_ident = IdentifierName { span: SPAN, name: Atom::from(prop) };
        Expression::StaticMemberExpression(self.ast.alloc_static_member_expression(SPAN, obj, prop_ident, false))
    }

    /// Create a computed member expression (obj[index])
    fn make_computed_member(&self, obj: Expression<'a>, index: Expression<'a>) -> Expression<'a> {
        Expression::ComputedMemberExpression(self.ast.alloc_computed_member_expression(SPAN, obj, index, false))
    }

    /// Build a variable declaration statement
    fn build_var_decl_stmt(&self, name: &'a str, init: Option<Expression<'a>>) -> Statement<'a> {
        let binding_pattern = self.ast.binding_pattern_binding_identifier(SPAN, name);
        let decl = self.ast.variable_declarator(SPAN, VariableDeclarationKind::Var, binding_pattern, None::<TSTypeAnnotation>, init, false);
        let decls = self.ast.vec1(decl);
        let var_decl = self.ast.alloc_variable_declaration(SPAN, VariableDeclarationKind::Var, decls, false);
        Statement::VariableDeclaration(var_decl)
    }

    /// Allocate a string in the allocator
    fn alloc_str(&self, s: &str) -> &'a str {
        self.allocator.alloc_str(s)
    }

    fn consume_gas(&mut self) -> Result<(), String> {
        if self.gas == 0 {
            return Err("Out of gas - possible infinite loop".to_string());
        }
        self.gas -= 1;
        Ok(())
    }

    fn evaluate(&mut self, stmts: &'a oxc_allocator::Vec<'a, Statement<'a>>) -> Result<(), String> {
        for stmt in stmts {
            self.eval_stmt(stmt)?;
        }
        Ok(())
    }

    fn eval_stmt(&mut self, stmt: &'a Statement<'a>) -> Result<(), String> {
        self.consume_gas()?;

        match stmt {
            Statement::FunctionDeclaration(func) => {
                // Bind function as a closure in the current environment
                // This captures the lexical environment at the point of definition
                if let Some(id) = &func.id {
                    let name = id.name.to_string();
                    if let Some(body) = &func.body {
                        let params: Vec<String> = func.params.items.iter()
                            .filter_map(|p| {
                                match &p.pattern {
                                    BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
                                    _ => None,
                                }
                            })
                            .collect();
                        self.stats.functions_defined += 1;

                        // Create a closure that captures the current environment
                        let closure = Value::Closure {
                            params,
                            body,
                            env: Rc::clone(&self.env),
                            original: Some(func.as_ref()),
                        };

                        // Bind in current environment
                        self.env.bind(name.clone(), closure);

                        // Also keep in functions map for backward compatibility
                        let params2: Vec<String> = func.params.items.iter()
                            .filter_map(|p| {
                                match &p.pattern {
                                    BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
                                    _ => None,
                                }
                            })
                            .collect();
                        self.functions.insert(name, StoredFunction {
                            params: params2,
                            body,
                        });
                    }
                }
            }

            Statement::VariableDeclaration(decl) => {
                for d in &decl.declarations {
                    let name = match &d.id {
                        BindingPattern::BindingIdentifier(id) => id.name.to_string(),
                        _ => continue,
                    };

                    let value = if let Some(init) = &d.init {
                        self.eval_expr(init)?
                    } else {
                        Value::Static(JsValue::Undefined)
                    };

                    let value_desc = match &value {
                        Value::Static(v) => format!("STATIC({})", v.to_display(self)),
                        Value::Dynamic(_) => "DYNAMIC".to_string(),
                        Value::Closure { .. } => "CLOSURE".to_string(),
                    };

                    if !self.in_function {
                        // At top-level: emit ALL variable declarations to residual
                        // But first, emit any captured variables from closures that aren't
                        // already declared at top level
                        self.emit_captured_vars(&value);

                        // Skip emitting if already emitted (by a closure's captured vars)
                        if !self.emitted_top_level_vars.contains(&name) {
                            self.trace_decision(&format!("var {} (top-level)", name), &format!("{} -> emit to residual", value_desc));
                            let name_alloc: &'a str = self.alloc_str(&name);
                            let init_expr = self.value_to_expr(&value);
                            let stmt = self.build_var_decl_stmt(name_alloc, Some(init_expr));
                            self.residual.push(stmt);
                            self.emitted_top_level_vars.insert(name.clone());
                        }
                    } else {
                        // Inside functions: track value in env, residualization handled later
                        let is_dynamic = matches!(&value, Value::Dynamic(_));
                        if is_dynamic {
                            self.trace_decision(&format!("var {}", name), &format!("{} -> track for potential residual", value_desc));
                            self.stats.vars_dynamic += 1;
                        } else {
                            self.trace_decision(&format!("var {}", name), &format!("{} -> collapse", value_desc));
                            self.stats.vars_collapsed += 1;
                        }
                    }

                    // Bind in current environment
                    self.env.bind(name, value);
                }
            }

            Statement::WhileStatement(w) => {
                self.trace("while loop");
                let mut iterations = 0;
                loop {
                    self.consume_gas()?;
                    let cond = self.eval_expr(&w.test)?;

                    match cond {
                        Value::Static(v) => {
                            if !v.is_truthy() {
                                self.trace_decision("while condition", &format!("STATIC(false) after {} iterations -> unroll complete", iterations));
                                if iterations > 0 {
                                    self.stats.loops_fully_unrolled += 1;
                                }
                                break;
                            }
                            iterations += 1;
                            self.depth += 1;
                            self.eval_stmt(&w.body)?;
                            self.depth -= 1;
                        }
                        Value::Closure { .. } => {
                            // Closures are truthy, so this would be an infinite loop
                            // Treat it as truthy and continue iterating
                            iterations += 1;
                            self.depth += 1;
                            self.eval_stmt(&w.body)?;
                            self.depth -= 1;
                        }
                        Value::Dynamic(cond_expr) => {
                            self.trace_decision("while condition", "DYNAMIC -> emit loop to residual");
                            self.stats.loops_emitted += 1;
                            // Build the while loop AST with the body
                            let body_stmt = w.body.clone_in(self.allocator);
                            let while_stmt = self.ast.statement_while(SPAN, cond_expr, body_stmt);
                            self.residual.push(while_stmt);
                            break;
                        }
                    }
                }
            }

            Statement::BlockStatement(b) => {
                for s in &b.body {
                    self.eval_stmt(s)?;
                }
            }

            Statement::SwitchStatement(sw) => {
                let discriminant = self.eval_expr(&sw.discriminant)?;

                match discriminant {
                    Value::Static(ref disc_val) => {
                        self.trace(&format!("switch on {} (static)", disc_val.to_display(self)));
                        for case in &sw.cases {
                            if let Some(test) = &case.test {
                                let test_val = self.eval_expr(test)?;
                                if let Value::Static(ref tv) = test_val {
                                    // JS uses === for switch comparison
                                    match Self::strict_equals(disc_val, tv) {
                                        Some(true) => {
                                            self.trace(&format!("case {} matched", tv.to_display(self)));
                                            self.stats.switches_resolved += 1;
                                            for s in &case.consequent {
                                                match s {
                                                    Statement::BreakStatement(_) => return Ok(()),
                                                    _ => self.eval_stmt(s)?,
                                                }
                                            }
                                            return Ok(());
                                        }
                                        Some(false) => {}
                                        None => {
                                            self.stats.switches_emitted += 1;
                                            let disc_expr = sw.discriminant.clone_in(self.allocator);
                                            let cases_clone = sw.cases.clone_in(self.allocator);
                                            let switch_stmt = self.ast.statement_switch(SPAN, disc_expr, cases_clone);
                                            self.residual.push(switch_stmt);
                                            return Ok(());
                                        }
                                    }
                                }
                            }
                        }
                        // No case matched - check for default
                        for case in &sw.cases {
                            if case.test.is_none() {
                                self.stats.switches_resolved += 1;
                                for s in &case.consequent {
                                    match s {
                                        Statement::BreakStatement(_) => return Ok(()),
                                        _ => self.eval_stmt(s)?,
                                    }
                                }
                                return Ok(());
                            }
                        }
                    }
                    Value::Closure { .. } => {
                        // Closures can't match any case (functions use reference equality)
                        // Just check for default case
                        for case in &sw.cases {
                            if case.test.is_none() {
                                self.stats.switches_resolved += 1;
                                for s in &case.consequent {
                                    match s {
                                        Statement::BreakStatement(_) => return Ok(()),
                                        _ => self.eval_stmt(s)?,
                                    }
                                }
                                return Ok(());
                            }
                        }
                    }
                    Value::Dynamic(disc_expr) => {
                        // Emit the entire switch statement as residual code
                        self.stats.switches_emitted += 1;
                        let cases_clone = sw.cases.clone_in(self.allocator);
                        let switch_stmt = self.ast.statement_switch(SPAN, disc_expr, cases_clone);
                        self.residual.push(switch_stmt);
                    }
                }
            }

            Statement::ExpressionStatement(expr) => {
                let val = self.eval_expr(&expr.expression)?;
                self.last_value = Some(val.clone_in(self.allocator));

                // Determine if we need to emit this expression to residual
                // Key insight: assignments to local variables we're tracking don't need
                // to be emitted - they're captured in the environment. We only emit if:
                // 1. The expression is dynamic
                // 2. The expression has "true" side effects (calls, member assignments, updates)
                //    that aren't captured by our environment tracking

                let needs_emit = match &expr.expression {
                    // Assignments to simple identifiers are tracked in env - don't emit if static
                    // EXCEPT at top-level where we need to emit to establish global state
                    Expression::AssignmentExpression(assign) => {
                        match &assign.left {
                            AssignmentTarget::AssignmentTargetIdentifier(_) => {
                                // At top-level: emit ALL assignments to establish global state
                                // Inside functions: only emit if value is dynamic
                                !self.in_function || matches!(&val, Value::Dynamic(_))
                            }
                            AssignmentTarget::ComputedMemberExpression(_) => {
                                // Array index assignments - emit only if we couldn't track statically
                                // (i.e., if the result is dynamic) OR at top-level
                                !self.in_function || matches!(&val, Value::Dynamic(_))
                            }
                            AssignmentTarget::StaticMemberExpression(_) => {
                                // Object property assignments - emit only if we couldn't track statically
                                // (i.e., if the result is dynamic) OR at top-level
                                !self.in_function || matches!(&val, Value::Dynamic(_))
                            }
                            _ => {
                                // Other member assignments - emit if dynamic
                                matches!(&val, Value::Dynamic(_)) || Self::expr_has_side_effects(&expr.expression)
                            }
                        }
                    }
                    // Update expressions (++, --) on identifiers are tracked in env
                    Expression::UpdateExpression(update) => {
                        match &update.argument {
                            SimpleAssignmentTarget::AssignmentTargetIdentifier(_) => {
                                matches!(&val, Value::Dynamic(_))
                            }
                            _ => true
                        }
                    }
                    // Call expressions that are array methods or closure calls we handle statically
                    Expression::CallExpression(call) => {
                        // Check if this is a call to a known closure that was fully evaluated
                        if let Expression::Identifier(id) = &call.callee {
                            // If the callee is a closure we track, and result is static, don't emit
                            // (the function call was fully evaluated internally)
                            if let Some(Value::Closure { .. }) = self.env.lookup(id.name.as_str(), self.allocator) {
                                matches!(&val, Value::Dynamic(_))
                            } else {
                                // Unknown function (might be external like console.log)
                                matches!(&val, Value::Dynamic(_)) || Self::expr_has_side_effects(&expr.expression)
                            }
                        } else if let Expression::StaticMemberExpression(mem) = &call.callee {
                            let method = mem.property.name.as_str();
                            // Check if this is a push/pop/unshift on a static array
                            if matches!(method, "push" | "pop" | "unshift") {
                                // Check if the object is a simple identifier (arr.push)
                                if let Expression::Identifier(id) = &mem.object {
                                    let var_name = id.name.as_str();
                                    // Check if this variable holds a static array
                                    if let Some(Value::Static(JsValue::Array { .. })) = self.env.lookup(var_name, self.allocator) {
                                        // We handled this statically - don't emit
                                        false
                                    } else {
                                        matches!(&val, Value::Dynamic(_)) || Self::expr_has_side_effects(&expr.expression)
                                    }
                                // Check if the object is a nested member (obj.prop.push)
                                } else if let Expression::StaticMemberExpression(nested_mem) = &mem.object {
                                    if let Expression::Identifier(obj_id) = &nested_mem.object {
                                        let obj_name = obj_id.name.as_str();
                                        let prop_name = nested_mem.property.name.as_str();
                                        // Check if obj.prop is a static array
                                        if let Some(Value::Static(JsValue::Object(props))) = self.env.lookup(obj_name, self.allocator) {
                                            if let Some(Value::Static(JsValue::Array { .. })) = props.get(prop_name) {
                                                // We handled this statically - don't emit
                                                false
                                            } else {
                                                matches!(&val, Value::Dynamic(_)) || Self::expr_has_side_effects(&expr.expression)
                                            }
                                        } else {
                                            matches!(&val, Value::Dynamic(_)) || Self::expr_has_side_effects(&expr.expression)
                                        }
                                    } else {
                                        matches!(&val, Value::Dynamic(_)) || Self::expr_has_side_effects(&expr.expression)
                                    }
                                // Handle computed member expression: arr[idx].push()
                                // If the result is static, we handled it statically with source tracking
                                } else if let Expression::ComputedMemberExpression(_) = &mem.object {
                                    // If push/pop/unshift returned a static value, the operation was
                                    // fully evaluated via source tracking - don't emit
                                    matches!(&val, Value::Dynamic(_))
                                } else {
                                    matches!(&val, Value::Dynamic(_)) || Self::expr_has_side_effects(&expr.expression)
                                }
                            // Check if this is .apply() or .call() on a closure method
                            } else if matches!(method, "apply" | "call") {
                                // Pattern: obj.method.apply(thisArg, args) or obj.method.call(thisArg, ...)
                                // If the result is static/closure, the call was fully evaluated
                                matches!(&val, Value::Dynamic(_))
                            } else {
                                // General member method call (e.g., obj.method())
                                // Check if the callee resolves to a closure - if so and result is static,
                                // the call was fully evaluated and we don't need to emit
                                if let Expression::Identifier(obj_id) = &mem.object {
                                    // obj.method() pattern
                                    if let Some(Value::Static(JsValue::Object(props))) = self.env.lookup(obj_id.name.as_str(), self.allocator) {
                                        if let Some(Value::Closure { .. }) = props.get(method) {
                                            // The method is a closure we track - if result is static, don't emit
                                            matches!(&val, Value::Dynamic(_))
                                        } else {
                                            // Method exists but isn't a closure we track
                                            // If call returned static, all side effects were handled
                                            matches!(&val, Value::Dynamic(_))
                                        }
                                    } else {
                                        // Object isn't in env - if call still returned static,
                                        // it was resolved and evaluated successfully
                                        matches!(&val, Value::Dynamic(_))
                                    }
                                } else {
                                    // Complex member expression (e.g., arr[i].method())
                                    // If the call fully evaluated to a static value, the callee was
                                    // resolved to a known closure and all side effects were handled.
                                    // Only emit if the result is dynamic.
                                    matches!(&val, Value::Dynamic(_))
                                }
                            }
                        } else if let Expression::ParenthesizedExpression(paren) = &call.callee {
                            // IIFE: (function() { ... })()
                            // If the IIFE was fully evaluated (result is static), don't emit
                            // The side effects have already been applied during evaluation
                            if let Expression::FunctionExpression(_) = &paren.expression {
                                matches!(&val, Value::Dynamic(_))
                            } else if let Expression::ArrowFunctionExpression(_) = &paren.expression {
                                matches!(&val, Value::Dynamic(_))
                            } else {
                                // Other parenthesized callee - if result is static, call was evaluated
                                matches!(&val, Value::Dynamic(_))
                            }
                        } else if let Expression::FunctionExpression(_) = &call.callee {
                            // Direct IIFE: function() { ... }()
                            matches!(&val, Value::Dynamic(_))
                        } else if let Expression::ArrowFunctionExpression(_) = &call.callee {
                            // Arrow IIFE: (() => { ... })()
                            matches!(&val, Value::Dynamic(_))
                        } else {
                            // Unknown callee pattern - if call returned static, it was fully evaluated
                            matches!(&val, Value::Dynamic(_))
                        }
                    }
                    _ => {
                        // For call expressions with other patterns, if the result is static,
                        // the call was fully evaluated and side effects were captured.
                        // If the callee was unknown, the result would be dynamic.
                        matches!(&val, Value::Dynamic(_))
                    }
                };

                if needs_emit {
                    // For assignments, we need to emit the full assignment expression
                    // (otherwise `myGlobal = {}` becomes just `{}`)
                    let emit_expr = if let Expression::AssignmentExpression(assign) = &expr.expression {
                        if matches!(&val, Value::Dynamic(_)) {
                            // Dynamic value: emit original assignment expression
                            expr.expression.clone_in(self.allocator)
                        } else if !self.in_function {
                            // Top-level static assignment: build assignment with evaluated value
                            // This preserves the target (e.g., `myGlobal`) while using the evaluated value
                            let target = assign.left.clone_in(self.allocator);
                            let value_expr = self.value_to_expr(&val);
                            self.ast.expression_assignment(
                                SPAN,
                                assign.operator,
                                target,
                                value_expr,
                            )
                        } else {
                            // Inside function, static value: just the value (for side effect tracking)
                            self.value_to_expr(&val)
                        }
                    } else {
                        match &val {
                            Value::Dynamic(_) => self.value_to_expr(&val),
                            _ => expr.expression.clone_in(self.allocator),
                        }
                    };
                    let stmt = self.ast.statement_expression(SPAN, emit_expr);
                    self.trace("emit ExpressionStatement to residual");
                    self.residual.push(stmt);
                }
            }

            Statement::ReturnStatement(ret) => {
                let val = if let Some(arg) = &ret.argument {
                    self.eval_expr(arg)?
                } else {
                    // Bare return; returns undefined
                    Value::Static(JsValue::Undefined)
                };

                // Only emit to residual if the return value is dynamic
                if let Value::Dynamic(_) = &val {
                    let return_expr = self.value_to_expr(&val);
                    let return_stmt = self.ast.statement_return(SPAN, Some(return_expr));
                    self.trace("emit ReturnStatement (Dynamic) to residual");
                    self.residual.push(return_stmt);
                }

                // Always track the return value (this breaks out of loops)
                self.return_value = Some(val);
            }

            Statement::IfStatement(if_stmt) => {
                let test = self.eval_expr(&if_stmt.test)?;
                match test {
                    Value::Static(v) => {
                        if v.is_truthy() {
                            self.eval_stmt(&if_stmt.consequent)?;
                        } else if let Some(alt) = &if_stmt.alternate {
                            self.eval_stmt(alt)?;
                        }
                    }
                    Value::Closure { .. } => {
                        // Closures are truthy
                        self.eval_stmt(&if_stmt.consequent)?;
                    }
                    Value::Dynamic(_) => {
                        // Can't determine branch, emit the whole if statement
                        self.residual.push(stmt.clone_in(self.allocator));
                    }
                }
            }

            Statement::ForStatement(for_stmt) => {
                // Handle for(;;) infinite loops and standard for loops
                self.trace("for loop");

                // Execute init if present
                if let Some(init) = &for_stmt.init {
                    if let ForStatementInit::VariableDeclaration(decl) = init {
                        // Handle var declarations in for init
                        for d in &decl.declarations {
                            let name = match &d.id {
                                BindingPattern::BindingIdentifier(id) => id.name.to_string(),
                                _ => continue,
                            };
                            let value = if let Some(init_expr) = &d.init {
                                self.eval_expr(init_expr)?
                            } else {
                                Value::Static(JsValue::Undefined)
                            };
                            self.env.bind(name, value);
                        }
                    } else if let Some(expr) = init.as_expression() {
                        self.eval_expr(expr)?;
                    }
                }

                let mut iterations = 0;
                loop {
                    self.consume_gas()?;

                    // Check test condition (if present)
                    if let Some(test) = &for_stmt.test {
                        let cond = self.eval_expr(test)?;
                        match cond {
                            Value::Static(v) => {
                                if !v.is_truthy() {
                                    break;
                                }
                            }
                            Value::Closure { .. } => {
                                // Closures are truthy, continue
                            }
                            Value::Dynamic(_) => {
                                // Can't determine, emit the loop
                                self.trace_decision("for condition", "DYNAMIC -> emit loop to residual");
                                self.residual.push(stmt.clone_in(self.allocator));
                                return Ok(());
                            }
                        }
                    }
                    // No test means infinite loop (for(;;))

                    iterations += 1;
                    self.depth += 1;
                    self.eval_stmt(&for_stmt.body)?;
                    self.depth -= 1;

                    // Check if we got a return
                    if self.return_value.is_some() {
                        break;
                    }

                    // Execute update if present
                    if let Some(update) = &for_stmt.update {
                        self.eval_expr(update)?;
                    }
                }
                self.trace_decision("for loop", &format!("unrolled {} iterations", iterations));
            }

            Statement::TryStatement(try_stmt) => {
                // Try to evaluate the try block
                // If it succeeds without throwing, we're done
                // If it throws, we'd need to evaluate the catch block
                // For now, evaluate try block and if there's an error, emit the whole thing

                let saved_residual = std::mem::take(&mut self.residual);
                let saved_return = self.return_value.take();

                let result = self.eval_block_statements(&try_stmt.block.body);

                match result {
                    Ok(()) => {
                        // Try block succeeded, merge residual
                        let try_residual = std::mem::replace(&mut self.residual, saved_residual);
                        self.residual.extend(try_residual);
                        // Keep any return value
                        if self.return_value.is_none() {
                            self.return_value = saved_return;
                        }
                    }
                    Err(_) => {
                        // Error during evaluation, emit the whole try statement as residual
                        self.residual = saved_residual;
                        self.return_value = saved_return;
                        self.residual.push(stmt.clone_in(self.allocator));
                    }
                }
            }

            Statement::ThrowStatement(throw) => {
                // Evaluate the thrown expression, then emit as residual
                // (we don't actually throw during PE)
                let _val = self.eval_expr(&throw.argument)?;
                self.residual.push(stmt.clone_in(self.allocator));
            }

            Statement::BreakStatement(_) => {
                // Break is handled by the loop constructs
                // If we reach here, just note it
            }

            Statement::ContinueStatement(_) => {
                // Continue is handled by loop constructs
            }

            _ => {
                // For any unhandled statement, emit it to residual to preserve it
                self.trace(&format!("unhandled statement type, emitting to residual"));
                self.residual.push(stmt.clone_in(self.allocator));
            }
        }
        Ok(())
    }

    fn eval_block_statements(&mut self, stmts: &'a oxc_allocator::Vec<'a, Statement<'a>>) -> Result<(), String> {
        for stmt in stmts {
            self.eval_stmt(stmt)?;
            if self.return_value.is_some() {
                break;
            }
        }
        Ok(())
    }

    /// Check if an expression has potential side effects that must be preserved
    /// even if we can compute its value. This is conservative - we assume
    /// any call, assignment, or update has side effects.
    fn expr_has_side_effects(expr: &Expression<'a>) -> bool {
        match expr {
            // Calls always potentially have side effects
            Expression::CallExpression(_) => true,
            Expression::NewExpression(_) => true,

            // Assignments have side effects (mutation)
            Expression::AssignmentExpression(_) => true,

            // Updates (++, --) have side effects
            Expression::UpdateExpression(_) => true,

            // Parenthesized - check inner
            Expression::ParenthesizedExpression(p) => Self::expr_has_side_effects(&p.expression),

            // Sequence - check all
            Expression::SequenceExpression(seq) => {
                seq.expressions.iter().any(|e| Self::expr_has_side_effects(e))
            }

            // Conditional - check all branches
            Expression::ConditionalExpression(cond) => {
                Self::expr_has_side_effects(&cond.test)
                    || Self::expr_has_side_effects(&cond.consequent)
                    || Self::expr_has_side_effects(&cond.alternate)
            }

            // Binary/Unary - check operands
            Expression::BinaryExpression(bin) => {
                Self::expr_has_side_effects(&bin.left) || Self::expr_has_side_effects(&bin.right)
            }
            Expression::UnaryExpression(un) => {
                // delete has side effects
                if un.operator == UnaryOperator::Delete {
                    return true;
                }
                Self::expr_has_side_effects(&un.argument)
            }

            // Logical expressions can short-circuit to side-effecting code
            Expression::LogicalExpression(log) => {
                Self::expr_has_side_effects(&log.left) || Self::expr_has_side_effects(&log.right)
            }

            // Pure expressions - no side effects
            Expression::Identifier(_)
            | Expression::NumericLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::ArrayExpression(_)
            | Expression::ObjectExpression(_)
            | Expression::FunctionExpression(_)
            | Expression::ArrowFunctionExpression(_)
            | Expression::StaticMemberExpression(_)
            | Expression::ComputedMemberExpression(_) => false,

            // Default to assuming side effects for anything else
            _ => true,
        }
    }

    fn strict_equals(a: &JsValue<'a>, b: &JsValue<'a>) -> Option<bool> {
        match (a, b) {
            (JsValue::Number(x), JsValue::Number(y)) => Some(x == y),
            (JsValue::Boolean(x), JsValue::Boolean(y)) => Some(x == y),
            (JsValue::String(x), JsValue::String(y)) => Some(x == y),
            (JsValue::Null, JsValue::Null) => Some(true),
            (JsValue::Undefined, JsValue::Undefined) => Some(true),
            (JsValue::Array { source: Some(ls), .. }, JsValue::Array { source: Some(rs), .. }) => {
                if ls == rs {
                    Some(true)
                } else {
                    None
                }
            }
            (a, b) if a.is_primitive() && b.is_primitive() => Some(false),
            _ => None,
        }
    }

    fn loose_equals(a: &JsValue<'a>, b: &JsValue<'a>) -> Option<bool> {
        if !a.is_primitive() || !b.is_primitive() {
            return None;
        }

        match (a, b) {
            (JsValue::Undefined, JsValue::Undefined) => Some(true),
            (JsValue::Null, JsValue::Null) => Some(true),
            (JsValue::Undefined, JsValue::Null) | (JsValue::Null, JsValue::Undefined) => Some(true),
            (JsValue::Number(_), JsValue::Number(_))
            | (JsValue::Boolean(_), JsValue::Boolean(_))
            | (JsValue::String(_), JsValue::String(_)) => Self::strict_equals(a, b),
            (JsValue::Number(n), JsValue::String(s)) => {
                Some(*n == JsValue::parse_js_number(s))
            }
            (JsValue::String(s), JsValue::Number(n)) => {
                Some(JsValue::parse_js_number(s) == *n)
            }
            (JsValue::Boolean(b), other) => {
                let left = JsValue::Number(if *b { 1.0 } else { 0.0 });
                Self::loose_equals(&left, other)
            }
            (other, JsValue::Boolean(b)) => {
                let right = JsValue::Number(if *b { 1.0 } else { 0.0 });
                Self::loose_equals(other, &right)
            }
            (JsValue::Null, _) | (JsValue::Undefined, _) | (_, JsValue::Null) | (_, JsValue::Undefined) => Some(false),
            _ => Some(false),
        }
    }

    fn eval_expr(&mut self, expr: &'a Expression<'a>) -> Result<Value<'a>, String> {
        self.consume_gas()?;

        match expr {
            Expression::NumericLiteral(n) => Ok(Value::Static(JsValue::Number(n.value))),

            Expression::BooleanLiteral(b) => Ok(Value::Static(JsValue::Boolean(b.value))),

            Expression::StringLiteral(s) => Ok(Value::Static(JsValue::String(s.value.to_string()))),

            Expression::NullLiteral(_) => Ok(Value::Static(JsValue::Null)),

            Expression::ArrayExpression(arr) => {
                let mut elements = Vec::new();
                let mut has_dynamic = false;
                for el in &arr.elements {
                    match el {
                        ArrayExpressionElement::SpreadElement(_) => {
                            // Return the original expression for spread
                            return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                        }
                        ArrayExpressionElement::Elision(_) => {
                            elements.push(Value::Static(JsValue::Undefined));
                        }
                        _ => {
                            if let Some(e) = el.as_expression() {
                                let val = self.eval_expr(e)?;
                                if !val.is_known() {
                                    has_dynamic = true;
                                }
                                elements.push(val);
                            }
                        }
                    }
                }
                if has_dynamic {
                    // Build an array expression with evaluated elements
                    let arr_elements: oxc_allocator::Vec<'a, ArrayExpressionElement<'a>> = self.ast.vec_from_iter(
                        elements.iter().map(|v| {
                            let e = self.value_to_expr(v);
                            ArrayExpressionElement::from(e)
                        })
                    );
                    Ok(Value::Dynamic(self.ast.expression_array(SPAN, arr_elements)))
                } else {
                    Ok(Value::Static(JsValue::Array { elements, source: None }))
                }
            }

            Expression::Identifier(id) => {
                let name = id.name.as_str();
                // Special case: JavaScript built-in values
                match name {
                    "undefined" => return Ok(Value::Static(JsValue::Undefined)),
                    "null" => return Ok(Value::Static(JsValue::Null)),
                    "true" => return Ok(Value::Static(JsValue::Boolean(true))),
                    "false" => return Ok(Value::Static(JsValue::Boolean(false))),
                    "NaN" => return Ok(Value::Static(JsValue::Number(f64::NAN))),
                    "Infinity" => return Ok(Value::Static(JsValue::Number(f64::INFINITY))),
                    "Date" => return Ok(Value::Static(JsValue::Date)),
                    _ => {}
                }
                if let Some(val) = self.env.lookup(name, self.allocator) {
                    // Debug: trace specific variable lookups
                    if self.trace && (name == "v12" || name == "v13" || name == "v14" || name == "v15" || name == "v16") {
                        self.trace(&format!("LOOKUP {} -> {:?}", name, val.to_display(self)));
                    }
                    Ok(val)
                } else {
                    // Debug: trace when variables are not found
                    if self.trace && (name == "v12" || name == "v13" || name == "v14" || name == "v15" || name == "v16") {
                        self.trace(&format!("LOOKUP {} -> NOT FOUND (dynamic)", name));
                    }
                    // Unknown identifier - return as dynamic AST node
                    Ok(Value::Dynamic(self.ast.expression_identifier(SPAN, name)))
                }
            }

            Expression::ConditionalExpression(cond) => {
                let test = self.eval_expr(&cond.test)?;
                match test {
                    Value::Static(v) => {
                        if v.is_truthy() {
                            self.eval_expr(&cond.consequent)
                        } else {
                            self.eval_expr(&cond.alternate)
                        }
                    }
                    Value::Closure { .. } => {
                        // Closures are truthy, take the consequent branch
                        self.eval_expr(&cond.consequent)
                    }
                    Value::Dynamic(test_expr) => {
                        let consequent = self.eval_expr(&cond.consequent)?;
                        let alternate = self.eval_expr(&cond.alternate)?;
                        let cons_expr = self.value_to_expr(&consequent);
                        let alt_expr = self.value_to_expr(&alternate);
                        Ok(Value::Dynamic(self.ast.expression_conditional(SPAN, test_expr, cons_expr, alt_expr)))
                    }
                }
            }

            Expression::BinaryExpression(bin) => {
                let left = self.eval_expr(&bin.left)?;
                let right = self.eval_expr(&bin.right)?;

                match (&left, &right) {
                    (Value::Static(l), Value::Static(r)) => {
                        match self.eval_binary_op(bin.operator, l, r)? {
                            Some(result) => Ok(Value::Static(result)),
                            None => Ok(Value::Dynamic(expr.clone_in(self.allocator))),
                        }
                    }
                    // Closure compared with Static using === or !==
                    // Closures are never equal to primitives (only equal to themselves by reference)
                    (Value::Closure { .. }, Value::Static(_)) | (Value::Static(_), Value::Closure { .. }) => {
                        match bin.operator {
                            BinaryOperator::StrictEquality | BinaryOperator::Equality => {
                                Ok(Value::Static(JsValue::Boolean(false)))
                            }
                            BinaryOperator::StrictInequality | BinaryOperator::Inequality => {
                                Ok(Value::Static(JsValue::Boolean(true)))
                            }
                            _ => {
                                let left_expr = self.value_to_expr(&left);
                                let right_expr = self.value_to_expr(&right);
                                Ok(Value::Dynamic(self.ast.expression_binary(SPAN, left_expr, bin.operator, right_expr)))
                            }
                        }
                    }
                    _ => {
                        let left_expr = self.value_to_expr(&left);
                        let right_expr = self.value_to_expr(&right);
                        Ok(Value::Dynamic(self.ast.expression_binary(SPAN, left_expr, bin.operator, right_expr)))
                    }
                }
            }

            Expression::AssignmentExpression(assign) => {
                match &assign.left {
                    AssignmentTarget::AssignmentTargetIdentifier(id) => {
                        // Simple identifier assignment: x = value
                        let name = id.name.to_string();

                        // Check if this is an external variable during specialization
                        let is_external_mutation = self.in_specialization && self.external_vars.contains(&name);

                        let value = match assign.operator {
                            AssignmentOperator::Assign => {
                                let val = self.eval_expr(&assign.right)?;
                                self.trace_decision(&format!("{} =", name), &format!("{}", val.to_display(self)));
                                val
                            }
                            AssignmentOperator::Addition => {
                                let current = self.env.lookup(&name, self.allocator)
                                    .unwrap_or(Value::Static(JsValue::Undefined));
                                let right = self.eval_expr(&assign.right)?;
                                match (&current, &right) {
                                    (Value::Static(l), Value::Static(r)) => {
                                        if let Some(result) = self.eval_binary_op(BinaryOperator::Addition, l, r)? {
                                            Value::Static(result)
                                        } else {
                                            let left_expr = self.value_to_expr(&current);
                                            let right_expr = self.value_to_expr(&right);
                                            Value::Dynamic(self.ast.expression_binary(SPAN, left_expr, BinaryOperator::Addition, right_expr))
                                        }
                                    }
                                    _ => {
                                        let left_expr = self.value_to_expr(&current);
                                        let right_expr = self.value_to_expr(&right);
                                        Value::Dynamic(self.ast.expression_binary(SPAN, left_expr, BinaryOperator::Addition, right_expr))
                                    }
                                }
                            }
                            _ => return Ok(Value::Dynamic(expr.clone_in(self.allocator))),
                        };

                        // If external, emit residual code for the side effect
                        if is_external_mutation {
                            let name_alloc: &'a str = self.alloc_str(&name);
                            let target = self.ast.simple_assignment_target_assignment_target_identifier(SPAN, name_alloc);
                            let value_expr = self.value_to_expr(&value);
                            let assign_expr = self.ast.expression_assignment(
                                SPAN,
                                AssignmentOperator::Assign,
                                target.into(),
                                value_expr,
                            );
                            let stmt = self.ast.statement_expression(SPAN, assign_expr);
                            self.residual.push(stmt);
                            self.trace(&format!("{} = ... (external assignment - emitting residual)", name));
                        }

                        let result = value.clone_in(self.allocator);
                        if !self.env.set(&name, value.clone_in(self.allocator)) {
                            self.env.bind(name, value);
                        }
                        Ok(result)
                    }
                    AssignmentTarget::ComputedMemberExpression(mem) => {
                        // Array index assignment: arr[i] = value
                        // Check if we can track this statically
                        if let Expression::Identifier(obj_id) = &mem.object {
                            let obj_name = obj_id.name.to_string();

                            // Check if this is an external variable during specialization
                            // If so, we must emit residual code for the side effect
                            let is_external_mutation = self.in_specialization && self.external_vars.contains(&obj_name);

                            let index_val = self.eval_expr(&mem.expression)?;
                            let right_val = self.eval_expr(&assign.right)?;

                            // For compound assignments like arr[i] ^= x, we need to compute the new value
                            let new_value = if assign.operator == AssignmentOperator::Assign {
                                right_val
                            } else {
                                // Get current element value
                                let current = if let Some(Value::Static(JsValue::Array { elements: ref arr, source: _ })) = self.env.lookup(&obj_name, self.allocator) {
                                    if let Value::Static(JsValue::Number(i)) = &index_val {
                                        let idx = *i as usize;
                                        if idx < arr.len() {
                                            arr[idx].clone_in(self.allocator)
                                        } else {
                                            Value::Static(JsValue::Undefined)
                                        }
                                    } else {
                                        return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                                    }
                                } else {
                                    return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                                };

                                // Apply the operator
                                match (&current, &right_val) {
                                    (Value::Static(l), Value::Static(r)) => {
                                        let op = match assign.operator {
                                            AssignmentOperator::BitwiseXOR => BinaryOperator::BitwiseXOR,
                                            AssignmentOperator::BitwiseOR => BinaryOperator::BitwiseOR,
                                            AssignmentOperator::BitwiseAnd => BinaryOperator::BitwiseAnd,
                                            AssignmentOperator::Addition => BinaryOperator::Addition,
                                            AssignmentOperator::Subtraction => BinaryOperator::Subtraction,
                                            AssignmentOperator::Multiplication => BinaryOperator::Multiplication,
                                            AssignmentOperator::Division => BinaryOperator::Division,
                                            _ => return Ok(Value::Dynamic(expr.clone_in(self.allocator))),
                                        };
                                        if let Some(result) = self.eval_binary_op(op, l, r)? {
                                            Value::Static(result)
                                        } else {
                                            return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                                        }
                                    }
                                    _ => return Ok(Value::Dynamic(expr.clone_in(self.allocator))),
                                }
                            };

                            // If this is an external variable mutation during specialization,
                            // we must emit residual code for the side effect
                            if is_external_mutation {
                                // Create residual assignment: obj[index] = value
                                let index_expr = self.value_to_expr(&index_val);
                                let right_expr = self.value_to_expr(&new_value);

                                let obj_name_alloc: &'a str = self.alloc_str(&obj_name);
                                let obj_expr = self.ast.expression_identifier(SPAN, obj_name_alloc);
                                let computed_member = self.ast.alloc_computed_member_expression(SPAN, obj_expr, index_expr, false);
                                let target = AssignmentTarget::ComputedMemberExpression(computed_member);

                                let assign_expr = self.ast.expression_assignment(
                                    SPAN,
                                    AssignmentOperator::Assign,  // Use simple assignment for the residual
                                    target,
                                    right_expr,
                                );
                                let stmt = self.ast.statement_expression(SPAN, assign_expr);
                                self.residual.push(stmt);
                                self.trace(&format!("{}[...] = ... (external mutation - emitting residual)", obj_name));
                                // Continue with normal mutation to keep specialization state correct
                            }

                            // If we have a static array and static index, mutate in place
                            // Value can be Static OR Closure (for function assignments like handlers[1] = function() {...})
                            if let Value::Static(JsValue::Number(idx)) = &index_val {
                                let idx = *idx as usize;
                                // Look up and mutate the array in the environment
                                if let Some(Value::Static(JsValue::Array { elements: arr, source })) = self.env.lookup(&obj_name, self.allocator) {
                                    let mut new_arr = arr;
                                    // Extend array if necessary
                                    while new_arr.len() <= idx {
                                        new_arr.push(Value::Static(JsValue::Undefined));
                                    }
                                    // Only store Static/Closure values in tracked array
                                    // Dynamic values are NOT stored - the residualized assignment handles them at runtime
                                    // This prevents issues where Dynamic values reference local variables that don't
                                    // exist in the residual output context
                                    match &new_value {
                                        Value::Static(_) | Value::Closure { .. } => {
                                            new_arr[idx] = new_value.clone_in(self.allocator);
                                            self.trace(&format!("{}[{}] = (static array index assignment)", obj_name, idx));
                                        }
                                        Value::Dynamic(_) => {
                                            // Don't track - let residualized assignment handle it
                                            self.trace(&format!("{}[{}] = (dynamic value - not tracked, residualized)", obj_name, idx));
                                        }
                                    }

                                    // If this array has a source (is a reference to another array's element),
                                    // propagate the change back to the source
                                    if let Some((ref source_var, source_idx)) = source {
                                        if let Some(Value::Static(JsValue::Array { elements: source_arr, source: source_source })) = self.env.lookup(source_var, self.allocator) {
                                            let mut new_source_arr = source_arr;
                                            // The source_idx element should be an array - update it
                                            if source_idx < new_source_arr.len() {
                                                // Clone new_arr for the source update
                                                let new_arr_for_source: Vec<Value<'a>> = new_arr.iter().map(|v| v.clone_in(self.allocator)).collect();
                                                new_source_arr[source_idx] = Value::Static(JsValue::Array { elements: new_arr_for_source, source: None });
                                                let updated_source = Value::Static(JsValue::Array { elements: new_source_arr, source: source_source });
                                                self.env.set(source_var, updated_source);
                                            }
                                        }
                                    }

                                    // Update the array in the environment
                                    let updated_array = Value::Static(JsValue::Array { elements: new_arr, source: source });
                                    if !self.env.set(&obj_name, updated_array.clone_in(self.allocator)) {
                                        self.env.bind(obj_name.clone(), updated_array);
                                    }

                                    return Ok(new_value);
                                }
                                // Handle Uint8Array index assignment
                                if let Some(Value::Static(JsValue::Uint8Array { bytes, backing_var })) = self.env.lookup(&obj_name, self.allocator) {
                                    if let Value::Static(JsValue::Number(n)) = &new_value {
                                        let mut new_bytes = bytes;
                                        // Extend if necessary
                                        while new_bytes.len() <= idx {
                                            new_bytes.push(0);
                                        }
                                        new_bytes[idx] = *n as u8;
                                        self.trace_decision(&format!("{}[{}] =", obj_name, idx), "STATIC typed array mutation");

                                        // Also update the backing ArrayBuffer if present
                                        if let Some(ref buf_var) = backing_var {
                                            let updated_buffer = Value::Static(JsValue::ArrayBuffer(new_bytes.clone()));
                                            if !self.env.set(buf_var, updated_buffer.clone_in(self.allocator)) {
                                                self.env.bind(buf_var.clone(), updated_buffer);
                                            }
                                        }

                                        let updated = Value::Static(JsValue::Uint8Array { bytes: new_bytes, backing_var });
                                        if !self.env.set(&obj_name, updated.clone_in(self.allocator)) {
                                            self.env.bind(obj_name.clone(), updated);
                                        }
                                        return Ok(new_value);
                                    }
                                }
                            }
                        }
                        // Couldn't track statically
                        Ok(Value::Dynamic(expr.clone_in(self.allocator)))
                    }
                    AssignmentTarget::StaticMemberExpression(mem) => {
                        // Object property assignment: obj.prop = value
                        if let Expression::Identifier(obj_id) = &mem.object {
                            let obj_name = obj_id.name.to_string();
                            let prop_name = mem.property.name.to_string();

                            if assign.operator == AssignmentOperator::Assign {
                                let new_value = self.eval_expr(&assign.right)?;

                                // Look up the object and update it
                                if let Some(Value::Static(JsValue::Object(mut props))) = self.env.lookup(&obj_name, self.allocator) {
                                    props.insert(prop_name.clone(), new_value.clone_in(self.allocator));
                                    let updated_obj = Value::Static(JsValue::Object(props));
                                    self.trace(&format!("{}.{} = (property assignment)", obj_name, prop_name));
                                    if !self.env.set(&obj_name, updated_obj.clone_in(self.allocator)) {
                                        self.env.bind(obj_name.clone(), updated_obj);
                                    }
                                    return Ok(new_value);
                                }
                            }
                        }
                        Ok(Value::Dynamic(expr.clone_in(self.allocator)))
                    }
                    _ => Ok(Value::Dynamic(expr.clone_in(self.allocator))),
                }
            }

            Expression::UpdateExpression(update) => {
                if let SimpleAssignmentTarget::AssignmentTargetIdentifier(id) = &update.argument {
                    let name = id.name.to_string();

                    // Check if this is an external variable during specialization
                    let is_external_mutation = self.in_specialization && self.external_vars.contains(&name);

                    // Check if the variable is a static number
                    let maybe_num = match self.env.lookup(&name, self.allocator) {
                        Some(Value::Static(JsValue::Number(n))) => Some(n),
                        _ => None,
                    };
                    if let Some(n) = maybe_num {
                        let new_val = match update.operator {
                            UpdateOperator::Increment => n + 1.0,
                            UpdateOperator::Decrement => n - 1.0,
                        };

                        // If external, emit residual code for the side effect
                        if is_external_mutation {
                            let stmt = self.ast.statement_expression(
                                SPAN,
                                expr.clone_in(self.allocator),
                            );
                            self.residual.push(stmt);
                            self.trace(&format!("{} (external update - emitting residual)", name));
                        }

                        // Update the variable in the environment
                        if !self.env.set(&name, Value::Static(JsValue::Number(new_val))) {
                            self.env.bind(name, Value::Static(JsValue::Number(new_val)));
                        }
                        // Postfix returns old value, prefix returns new value
                        if update.prefix {
                            return Ok(Value::Static(JsValue::Number(new_val)));
                        } else {
                            return Ok(Value::Static(JsValue::Number(n)));
                        }
                    } else {
                        // Variable not in env or dynamic, return the original update expression
                        return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                    }
                }
                Ok(Value::Dynamic(expr.clone_in(self.allocator)))
            }

            Expression::StaticMemberExpression(mem) => {
                let obj = self.eval_expr(&mem.object)?;
                let prop = mem.property.name.as_str();

                match (&obj, prop) {
                    (Value::Static(JsValue::Array { elements: arr, source: _ }), "length") => {
                        Ok(Value::Static(JsValue::Number(arr.len() as f64)))
                    }
                    // Handle .buffer on typed arrays - returns ArrayBuffer with same bytes
                    (Value::Static(JsValue::Uint8Array { bytes, .. }), "buffer") => {
                        Ok(Value::Static(JsValue::ArrayBuffer(bytes.clone())))
                    }
                    (Value::Static(JsValue::Int8Array(vals)), "buffer") => {
                        let bytes: Vec<u8> = vals.iter().map(|v| *v as u8).collect();
                        Ok(Value::Static(JsValue::ArrayBuffer(bytes)))
                    }
                    // Handle .length on typed arrays
                    (Value::Static(JsValue::Uint8Array { bytes, .. }), "length") => {
                        Ok(Value::Static(JsValue::Number(bytes.len() as f64)))
                    }
                    (Value::Static(JsValue::Int8Array(vals)), "length") => {
                        Ok(Value::Static(JsValue::Number(vals.len() as f64)))
                    }
                    (Value::Static(JsValue::Uint16Array(vals)), "length") => {
                        Ok(Value::Static(JsValue::Number(vals.len() as f64)))
                    }
                    (Value::Static(JsValue::Int16Array(vals)), "length") => {
                        Ok(Value::Static(JsValue::Number(vals.len() as f64)))
                    }
                    (Value::Static(JsValue::Uint32Array(vals)), "length") => {
                        Ok(Value::Static(JsValue::Number(vals.len() as f64)))
                    }
                    (Value::Static(JsValue::Int32Array(vals)), "length") => {
                        Ok(Value::Static(JsValue::Number(vals.len() as f64)))
                    }
                    (Value::Static(JsValue::Float32Array(vals)), "length") => {
                        Ok(Value::Static(JsValue::Number(vals.len() as f64)))
                    }
                    (Value::Static(JsValue::Float64Array(vals)), "length") => {
                        Ok(Value::Static(JsValue::Number(vals.len() as f64)))
                    }
                    (Value::Static(JsValue::ArrayBuffer(bytes)), "byteLength") => {
                        Ok(Value::Static(JsValue::Number(bytes.len() as f64)))
                    }
                    (Value::Static(JsValue::DataView(bytes)), "byteLength") => {
                        Ok(Value::Static(JsValue::Number(bytes.len() as f64)))
                    }
                    // Handle property access on static Objects
                    (Value::Static(JsValue::Object(props)), _) => {
                        if let Some(val) = props.get(prop) {
                            Ok(val.clone_in(self.allocator))
                        } else {
                            Ok(Value::Static(JsValue::Undefined))
                        }
                    }
                    _ => {
                        // If object is static but property access is dynamic, keep primitives
                        // as-is to avoid baking literal receivers into residual calls.
                        if let Value::Static(js) = &obj {
                            if js.is_primitive() {
                                return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                            }
                        }
                        // Preserve identifiers so later passes can still recognize patterns
                        // like v6.push(...).
                        if obj.is_static() {
                            let obj_expr = match &mem.object {
                                Expression::Identifier(_) => mem.object.clone_in(self.allocator),
                                _ => self.value_to_expr(&obj),
                            };
                            Ok(Value::Dynamic(self.make_static_member(obj_expr, prop)))
                        } else {
                            // Clone the original expression
                            Ok(Value::Dynamic(expr.clone_in(self.allocator)))
                        }
                    }
                }
            }

            Expression::ComputedMemberExpression(mem) => {
                // Track the source variable name if the object is an identifier
                let source_var = if let Expression::Identifier(id) = &mem.object {
                    Some(id.name.to_string())
                } else {
                    None
                };

                let obj = self.eval_expr(&mem.object)?;
                let index = self.eval_expr(&mem.expression)?;

                match (&obj, &index) {
                    (Value::Static(JsValue::Array { elements: arr, source: _ }), Value::Static(JsValue::Number(i))) => {
                        let idx = *i as usize;
                        if idx < arr.len() {
                            let element = arr[idx].clone_in(self.allocator);
                            // If the element is an array and we know the source variable,
                            // mark it with source tracking for reference semantics
                            if let Value::Static(JsValue::Array { elements: elem_arr, source: _ }) = element {
                                if let Some(var_name) = source_var {
                                    return Ok(Value::Static(JsValue::Array {
                                        elements: elem_arr,
                                        source: Some((var_name, idx)),
                                    }));
                                }
                            }
                            Ok(arr[idx].clone_in(self.allocator))
                        } else {
                            Ok(Value::Static(JsValue::Undefined))
                        }
                    }
                    // Handle array["length"] -> array.length
                    (Value::Static(JsValue::Array { elements: arr, .. }), Value::Static(JsValue::String(key))) if key == "length" => {
                        Ok(Value::Static(JsValue::Number(arr.len() as f64)))
                    }
                    // Handle string["length"] -> string.length
                    (Value::Static(JsValue::String(s)), Value::Static(JsValue::String(key))) if key == "length" => {
                        Ok(Value::Static(JsValue::Number(s.len() as f64)))
                    }
                    // Handle string[index] -> character at index
                    (Value::Static(JsValue::String(s)), Value::Static(JsValue::Number(i))) => {
                        let idx = *i as usize;
                        if idx < s.len() {
                            // JavaScript returns a single-character string
                            Ok(Value::Static(JsValue::String(s.chars().nth(idx).map(|c| c.to_string()).unwrap_or_default())))
                        } else {
                            Ok(Value::Static(JsValue::Undefined))
                        }
                    }
                    // Handle computed access on Object with string key
                    (Value::Static(JsValue::Object(props)), Value::Static(JsValue::String(key))) => {
                        if let Some(val) = props.get(key) {
                            Ok(val.clone_in(self.allocator))
                        } else {
                            Ok(Value::Static(JsValue::Undefined))
                        }
                    }
                    // Handle Date["now"] -> Date.now (returns Date for chaining)
                    (Value::Static(JsValue::Date), Value::Static(JsValue::String(key))) if key == "now" => {
                        // Return Date itself - the .apply() handler will deal with Date.now()
                        Ok(Value::Static(JsValue::Date))
                    }
                    _ => {
                        if let Value::Static(js) = &obj {
                            if js.is_primitive() {
                                return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                            }
                        }
                        let obj_expr = match &mem.object {
                            Expression::Identifier(_) => mem.object.clone_in(self.allocator),
                            _ => self.value_to_expr(&obj),
                        };
                        let idx_expr = self.value_to_expr(&index);
                        Ok(Value::Dynamic(self.make_computed_member(obj_expr, idx_expr)))
                    }
                }
            }

            Expression::CallExpression(call) => {
                // Avoid baking primitive values into member calls like v6.push(...)
                // where v6 is currently static but may change within residual control flow.
                if let Expression::StaticMemberExpression(mem) = &call.callee {
                    if matches!(&mem.object, Expression::Identifier(_)) {
                        let obj_val = self.eval_expr(&mem.object)?;
                        if let Value::Static(js) = &obj_val {
                            if js.is_primitive() {
                                return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                            }
                        }
                    }
                } else if let Expression::ComputedMemberExpression(mem) = &call.callee {
                    if matches!(&mem.object, Expression::Identifier(_)) {
                        let obj_val = self.eval_expr(&mem.object)?;
                        if let Value::Static(js) = &obj_val {
                            if js.is_primitive() {
                                return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                            }
                        }
                    }
                }

                // Check for method calls (e.g., stack.push, decoder.decode)
                if let Expression::StaticMemberExpression(mem) = &call.callee {
                    let method = mem.property.name.to_string();

                    // Handle .apply() calls: obj[prop].apply(thisArg, args) -> obj[prop](args)
                    // Evaluate directly instead of building new AST (avoids lifetime issues)
                    if method == "apply" || method == "call" {
                        // Evaluate the callee (e.g., obj[prop] or fakeWindow.Date)
                        let callee_val = self.eval_expr(&mem.object)?;

                        // Get args - for .apply() it's the second arg (array), for .call() it's args after first
                        let arg_vals: Vec<Value<'a>> = if method == "apply" {
                            if let Some(args_arg) = call.arguments.get(1) {
                                if let Some(args_expr) = args_arg.as_expression() {
                                    let args_val = self.eval_expr(args_expr)?;
                                    if let Value::Static(JsValue::Array { elements, .. }) = args_val {
                                        elements
                                    } else {
                                        vec![]
                                    }
                                } else {
                                    vec![]
                                }
                            } else {
                                vec![]
                            }
                        } else {
                            // .call(thisArg, arg1, arg2, ...) - evaluate args after first
                            let mut vals = vec![];
                            for arg in call.arguments.iter().skip(1) {
                                if let Some(arg_expr) = arg.as_expression() {
                                    vals.push(self.eval_expr(arg_expr)?);
                                }
                            }
                            vals
                        };

                        self.trace(&format!(".{}() transformed to direct call", method));

                        // Now handle the call based on what callee_val is
                        if let Value::Closure { params, body, env: closure_env, .. } = callee_val {
                            // It's a closure - invoke it directly
                            return self.call_closure(&params, body, &closure_env, arg_vals, "apply/call");
                        }

                        // Handle Date.now.apply() -> 0
                        if let Value::Static(JsValue::Date) = &callee_val {
                            self.trace_decision("Date.now.apply()", "STATIC -> 0");
                            return Ok(Value::Static(JsValue::Number(0.0)));
                        }

                        // Handle string/array methods via apply - common pattern in obfuscated code
                        // When callee_val is Dynamic(ComputedMemberExpression), extract the method name directly
                        if let Value::Dynamic(dyn_expr) = &callee_val {
                            if let Expression::ComputedMemberExpression(mem) = dyn_expr {
                                // Try to get the method name from a string literal in the expression
                                let method_name: Option<String> = match &mem.expression {
                                    Expression::StringLiteral(s) => Some(s.value.to_string()),
                                    Expression::Identifier(id) => {
                                        match self.env.lookup(id.name.as_str(), self.allocator) {
                                            Some(Value::Static(JsValue::String(s))) => Some(s),
                                            _ => None,
                                        }
                                    }
                                    _ => None
                                };

                                if let Some(method_name) = method_name {
                                    // Get thisArg from apply's first argument
                                    let this_arg = if let Some(this_expr) = call.arguments.first() {
                                        if let Some(expr) = this_expr.as_expression() {
                                            self.eval_expr(expr)?
                                        } else {
                                            Value::Static(JsValue::Undefined)
                                        }
                                    } else {
                                        Value::Static(JsValue::Undefined)
                                    };

                                    // Handle string.charCodeAt.apply(string, [index])
                                    if method_name == "charCodeAt" {
                                        if let Value::Static(JsValue::String(s)) = &this_arg {
                                            // Get index from arg_vals
                                            if let Some(Value::Static(JsValue::Number(idx))) = arg_vals.first() {
                                                let idx = *idx as usize;
                                                if idx < s.len() {
                                                    let char_code = s.chars().nth(idx).map(|c| c as u32 as f64).unwrap_or(f64::NAN);
                                                    self.trace_decision(&format!("\"{}\"[\"charCodeAt\"].apply(..., [{}])",
                                                        if s.len() > 10 { &s[..10] } else { s }, idx),
                                                        &format!("STATIC -> {}", char_code));
                                                    return Ok(Value::Static(JsValue::Number(char_code)));
                                                }
                                            }
                                        }
                                    }

                                    // Handle [].push.apply(target, items) - pushes items to target array
                                    if method_name == "push" {
                                        // thisArg should be the target array to push to
                                        if let Value::Static(JsValue::Array { elements: mut target, source }) = this_arg {
                                            // arg_vals contains the items to push
                                            for item in arg_vals.iter() {
                                                target.push(item.clone_in(self.allocator));
                                            }
                                            let new_len = target.len() as f64;
                                            self.trace_decision("[].push.apply(...)",
                                                &format!("STATIC -> {} items, returns {}", target.len(), new_len));

                                            // If we have a source variable, update it
                                            if let Some((ref var_name, _)) = source {
                                                let updated = Value::Static(JsValue::Array {
                                                    elements: target.iter().map(|v| v.clone_in(self.allocator)).collect(),
                                                    source: source.clone()
                                                });
                                                self.env.set(var_name, updated);
                                            }

                                            // push returns the new length
                                            return Ok(Value::Static(JsValue::Number(new_len)));
                                        }
                                    }

                                    // Handle String.fromCharCode.apply(null, charCodes)
                                    if method_name == "fromCharCode" {
                                        // arg_vals should be an array of char codes
                                        let mut result = String::new();
                                        let mut all_static = true;
                                        for arg in &arg_vals {
                                            if let Value::Static(JsValue::Number(n)) = arg {
                                                if let Some(c) = char::from_u32(*n as u32) {
                                                    result.push(c);
                                                }
                                            } else {
                                                all_static = false;
                                                break;
                                            }
                                        }
                                        if all_static && !result.is_empty() {
                                            self.trace_decision("String.fromCharCode.apply(...)",
                                                &format!("STATIC -> \"{}\"", if result.len() > 20 { &result[..20] } else { &result }));
                                            return Ok(Value::Static(JsValue::String(result)));
                                        }
                                    }
                                }
                            }
                        }

                        // Date or other - return dynamic but continue evaluation
                        self.trace(&format!("Non-closure callee in .{}() - callee_val: {} - returning dynamic",
                            method, callee_val.to_display(self)));
                        return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                    }

                    // First try to evaluate the object to see if it's a static value
                    let obj_val = self.eval_expr(&mem.object)?;

                    // Handle method calls on static objects
                    match &obj_val {
                        Value::Static(JsValue::TextDecoder) => {
                            if method == "decode" {
                                // TextDecoder.decode(uint8array)
                                if let Some(arg) = call.arguments.first() {
                                    if let Some(arg_expr) = arg.as_expression() {
                                        let arg_val = self.eval_expr(arg_expr)?;
                                        if let Value::Static(JsValue::Uint8Array { bytes, .. }) = arg_val {
                                            // Decode bytes to string (assuming UTF-8)
                                            let decoded = String::from_utf8_lossy(&bytes).to_string();
                                            self.trace_decision("TextDecoder.decode()", &format!("STATIC -> \"{}\"", decoded));
                                            self.stats.strings_decoded += 1;
                                            return Ok(Value::Static(JsValue::String(decoded)));
                                        }
                                    }
                                }
                            }
                        }
                        Value::Static(JsValue::Date) => {
                            // Handle Date methods
                            if method == "now" {
                                // Return a fixed timestamp for partial evaluation
                                // This allows the VM to proceed even though we don't have real time
                                self.trace_decision("Date.now()", "STATIC -> 0 (fixed for PE)");
                                return Ok(Value::Static(JsValue::Number(0.0)));
                            }
                        }
                        Value::Static(JsValue::DataView(bytes)) => {
                            // Handle DataView methods
                            match method.as_str() {
                                "getUint8" => {
                                    if let Some(arg) = call.arguments.first() {
                                        if let Some(arg_expr) = arg.as_expression() {
                                            let arg_val = self.eval_expr(arg_expr)?;
                                            if let Value::Static(JsValue::Number(offset)) = arg_val {
                                                let idx = offset as usize;
                                                if idx < bytes.len() {
                                                    return Ok(Value::Static(JsValue::Number(bytes[idx] as f64)));
                                                }
                                            }
                                        }
                                    }
                                }
                                "getInt8" => {
                                    if let Some(arg) = call.arguments.first() {
                                        if let Some(arg_expr) = arg.as_expression() {
                                            let arg_val = self.eval_expr(arg_expr)?;
                                            if let Value::Static(JsValue::Number(offset)) = arg_val {
                                                let idx = offset as usize;
                                                if idx < bytes.len() {
                                                    return Ok(Value::Static(JsValue::Number(bytes[idx] as i8 as f64)));
                                                }
                                            }
                                        }
                                    }
                                }
                                "getUint16" => {
                                    if let Some(arg) = call.arguments.first() {
                                        if let Some(arg_expr) = arg.as_expression() {
                                            let arg_val = self.eval_expr(arg_expr)?;
                                            if let Value::Static(JsValue::Number(offset)) = arg_val {
                                                let idx = offset as usize;
                                                // Check for little-endian flag
                                                let little_endian = call.arguments.get(1)
                                                    .and_then(|a| a.as_expression())
                                                    .and_then(|e| {
                                                        if let Expression::BooleanLiteral(b) = e {
                                                            Some(b.value)
                                                        } else {
                                                            None
                                                        }
                                                    })
                                                    .unwrap_or(false);
                                                if idx + 1 < bytes.len() {
                                                    let val = if little_endian {
                                                        u16::from_le_bytes([bytes[idx], bytes[idx + 1]])
                                                    } else {
                                                        u16::from_be_bytes([bytes[idx], bytes[idx + 1]])
                                                    };
                                                    return Ok(Value::Static(JsValue::Number(val as f64)));
                                                }
                                            }
                                        }
                                    }
                                }
                                "getInt32" => {
                                    if let Some(arg) = call.arguments.first() {
                                        if let Some(arg_expr) = arg.as_expression() {
                                            let arg_val = self.eval_expr(arg_expr)?;
                                            if let Value::Static(JsValue::Number(offset)) = arg_val {
                                                let idx = offset as usize;
                                                let little_endian = call.arguments.get(1)
                                                    .and_then(|a| a.as_expression())
                                                    .and_then(|e| {
                                                        if let Expression::BooleanLiteral(b) = e {
                                                            Some(b.value)
                                                        } else {
                                                            None
                                                        }
                                                    })
                                                    .unwrap_or(false);
                                                if idx + 3 < bytes.len() {
                                                    let val = if little_endian {
                                                        i32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]])
                                                    } else {
                                                        i32::from_be_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]])
                                                    };
                                                    return Ok(Value::Static(JsValue::Number(val as f64)));
                                                }
                                            }
                                        }
                                    }
                                }
                                "getFloat64" => {
                                    if let Some(arg) = call.arguments.first() {
                                        if let Some(arg_expr) = arg.as_expression() {
                                            let arg_val = self.eval_expr(arg_expr)?;
                                            if let Value::Static(JsValue::Number(offset)) = arg_val {
                                                let idx = offset as usize;
                                                let little_endian = call.arguments.get(1)
                                                    .and_then(|a| a.as_expression())
                                                    .and_then(|e| {
                                                        if let Expression::BooleanLiteral(b) = e {
                                                            Some(b.value)
                                                        } else {
                                                            None
                                                        }
                                                    })
                                                    .unwrap_or(false);
                                                if idx + 7 < bytes.len() {
                                                    let byte_arr = [
                                                        bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3],
                                                        bytes[idx + 4], bytes[idx + 5], bytes[idx + 6], bytes[idx + 7]
                                                    ];
                                                    let val = if little_endian {
                                                        f64::from_le_bytes(byte_arr)
                                                    } else {
                                                        f64::from_be_bytes(byte_arr)
                                                    };
                                                    return Ok(Value::Static(JsValue::Number(val)));
                                                }
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }

                    // Handle array methods (push, pop, etc.) on variables
                    if let Expression::Identifier(id) = &mem.object {
                        let var_name = id.name.to_string();

                        // Handle push/pop/unshift on static arrays in the environment
                        if let Some(Value::Static(JsValue::Array { elements: arr, source: _ })) = self.env.lookup(&var_name, self.allocator) {
                            match method.as_str() {
                                "push" => {
                                    let mut new_arr: Vec<Value<'a>> = arr.iter()
                                        .map(|v| v.clone_in(self.allocator))
                                        .collect();
                                    for arg in &call.arguments {
                                        if let Some(arg_expr) = arg.as_expression() {
                                            let val = self.eval_expr(arg_expr)?;
                                            new_arr.push(val);
                                        }
                                    }
                                    let new_len = new_arr.len();
                                    self.env.set(&var_name, Value::Static(JsValue::Array { elements: new_arr, source: None }));
                                    self.trace_decision(&format!("{}.push()", var_name), "STATIC array mutation");
                                    return Ok(Value::Static(JsValue::Number(new_len as f64)));
                                }
                                "unshift" => {
                                    let mut prefix: Vec<Value<'a>> = Vec::new();
                                    for arg in &call.arguments {
                                        if let Some(arg_expr) = arg.as_expression() {
                                            let val = self.eval_expr(arg_expr)?;
                                            prefix.push(val);
                                        }
                                    }
                                    let mut new_arr: Vec<Value<'a>> = prefix;
                                    new_arr.extend(arr.iter().map(|v| v.clone_in(self.allocator)));
                                    let new_len = new_arr.len();
                                    self.env.set(&var_name, Value::Static(JsValue::Array { elements: new_arr, source: None }));
                                    self.trace_decision(&format!("{}.unshift()", var_name), "STATIC array mutation");
                                    return Ok(Value::Static(JsValue::Number(new_len as f64)));
                                }
                                "pop" => {
                                    let mut new_arr: Vec<Value<'a>> = arr.iter().map(|v| v.clone_in(self.allocator)).collect();
                                    let popped = new_arr.pop().unwrap_or(Value::Static(JsValue::Undefined));
                                    self.env.set(&var_name, Value::Static(JsValue::Array { elements: new_arr, source: None }));
                                    self.trace_decision(&format!("{}.pop()", var_name), "STATIC array mutation");
                                    return Ok(popped);
                                }
                                "slice" => {
                                    let start_val = match call.arguments.get(0)
                                        .and_then(|arg| arg.as_expression())
                                    {
                                        Some(expr) => Some(self.eval_expr(expr)?),
                                        None => None,
                                    };
                                    let end_val = match call.arguments.get(1)
                                        .and_then(|arg| arg.as_expression())
                                    {
                                        Some(expr) => Some(self.eval_expr(expr)?),
                                        None => None,
                                    };
                                    if start_val.as_ref().map_or(false, |v| matches!(v, Value::Dynamic(_)))
                                        || end_val.as_ref().map_or(false, |v| matches!(v, Value::Dynamic(_)))
                                    {
                                        return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                                    }

                                    let len = arr.len();
                                    let start_num = start_val
                                        .as_ref()
                                        .and_then(|v| self.value_to_number(v))
                                        .unwrap_or(0.0);
                                    let end_num = end_val
                                        .as_ref()
                                        .and_then(|v| self.value_to_number(v))
                                        .unwrap_or(len as f64);
                                    let start = self.slice_index(start_num, len);
                                    let end = self.slice_index(end_num, len);
                                    let (start, end) = if end < start { (start, start) } else { (start, end) };
                                    let new_arr: Vec<Value<'a>> = arr[start..end]
                                        .iter()
                                        .map(|v| v.clone_in(self.allocator))
                                        .collect();
                                    self.trace_decision(&format!("{}.slice()", var_name), "STATIC array slice");
                                    return Ok(Value::Static(JsValue::Array { elements: new_arr, source: None }));
                                }
                                _ => {}
                            }
                        }
                    }

                    // Handle nested member expressions: obj.prop.push() where obj.prop is an array
                    // e.g., myGlobal.listeners.push(...)
                    if let Expression::StaticMemberExpression(nested_mem) = &mem.object {
                        if let Expression::Identifier(obj_id) = &nested_mem.object {
                            let obj_name = obj_id.name.to_string();
                            let prop_name = nested_mem.property.name.to_string();

                            // Look up the object
                            if let Some(Value::Static(JsValue::Object(props))) = self.env.lookup(&obj_name, self.allocator) {
                                // Get the property (should be an array)
                                if let Some(Value::Static(JsValue::Array { elements: arr, source: _ })) = props.get(&prop_name) {
                                    match method.as_str() {
                                        "push" => {
                                            let mut new_arr: Vec<Value<'a>> = arr.iter()
                                                .map(|v| v.clone_in(self.allocator))
                                                .collect();
                                            for arg in &call.arguments {
                                                if let Some(arg_expr) = arg.as_expression() {
                                                    let val = self.eval_expr(arg_expr)?;
                                                    new_arr.push(val);
                                                }
                                            }
                                            let new_len = new_arr.len();
                                            let mut new_props: HashMap<String, Value<'a>> = props.iter()
                                                .map(|(k, v)| (k.clone(), v.clone_in(self.allocator)))
                                                .collect();
                                            new_props.insert(prop_name.clone(), Value::Static(JsValue::Array { elements: new_arr, source: None }));
                                            self.env.set(&obj_name, Value::Static(JsValue::Object(new_props)));
                                            self.trace_decision(&format!("{}.{}.push()", obj_name, prop_name), "STATIC nested array mutation");
                                            return Ok(Value::Static(JsValue::Number(new_len as f64)));
                                        }
                                        "unshift" => {
                                            let mut prefix: Vec<Value<'a>> = Vec::new();
                                            for arg in &call.arguments {
                                                if let Some(arg_expr) = arg.as_expression() {
                                                    let val = self.eval_expr(arg_expr)?;
                                                    prefix.push(val);
                                                }
                                            }
                                            let mut new_arr: Vec<Value<'a>> = prefix;
                                            new_arr.extend(arr.iter().map(|v| v.clone_in(self.allocator)));
                                            let new_len = new_arr.len();
                                            let mut new_props: HashMap<String, Value<'a>> = props.iter()
                                                .map(|(k, v)| (k.clone(), v.clone_in(self.allocator)))
                                                .collect();
                                            new_props.insert(prop_name.clone(), Value::Static(JsValue::Array { elements: new_arr, source: None }));
                                            self.env.set(&obj_name, Value::Static(JsValue::Object(new_props)));
                                            self.trace_decision(&format!("{}.{}.unshift()", obj_name, prop_name), "STATIC nested array mutation");
                                            return Ok(Value::Static(JsValue::Number(new_len as f64)));
                                        }
                                        "slice" => {
                                            let start_val = match call.arguments.get(0)
                                                .and_then(|arg| arg.as_expression())
                                            {
                                                Some(expr) => Some(self.eval_expr(expr)?),
                                                None => None,
                                            };
                                            let end_val = match call.arguments.get(1)
                                                .and_then(|arg| arg.as_expression())
                                            {
                                                Some(expr) => Some(self.eval_expr(expr)?),
                                                None => None,
                                            };
                                            if start_val.as_ref().map_or(false, |v| matches!(v, Value::Dynamic(_)))
                                                || end_val.as_ref().map_or(false, |v| matches!(v, Value::Dynamic(_)))
                                            {
                                                return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                                            }

                                            let len = arr.len();
                                            let start_num = start_val
                                                .as_ref()
                                                .and_then(|v| self.value_to_number(v))
                                                .unwrap_or(0.0);
                                            let end_num = end_val
                                                .as_ref()
                                                .and_then(|v| self.value_to_number(v))
                                                .unwrap_or(len as f64);
                                            let start = self.slice_index(start_num, len);
                                            let end = self.slice_index(end_num, len);
                                            let (start, end) = if end < start { (start, start) } else { (start, end) };
                                            let new_arr: Vec<Value<'a>> = arr[start..end]
                                                .iter()
                                                .map(|v| v.clone_in(self.allocator))
                                                .collect();
                                            self.trace_decision(&format!("{}.{}.slice()", obj_name, prop_name), "STATIC nested array slice");
                                            return Ok(Value::Static(JsValue::Array { elements: new_arr, source: None }));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }

                // Handle array methods on computed expressions that evaluate to tracked arrays
                // e.g., v23[v24 - 2].push(value) where v23[v24 - 2] returns an array with source tracking
                if let Expression::StaticMemberExpression(mem) = &call.callee {
                    let method = mem.property.name.to_string();

                    // Evaluate the receiver to see if it's a tracked array
                    if let Ok(receiver_val) = self.eval_expr(&mem.object) {
                        if let Value::Static(JsValue::Array { elements: arr, source: Some((ref source_var, source_idx)) }) = receiver_val {
                            match method.as_str() {
                                "push" => {
                                    let mut new_arr: Vec<Value<'a>> = arr.iter()
                                        .map(|v| v.clone_in(self.allocator))
                                        .collect();
                                    for arg in &call.arguments {
                                        if let Some(arg_expr) = arg.as_expression() {
                                            let val = self.eval_expr(arg_expr)?;
                                            new_arr.push(val);
                                        }
                                    }
                                    let new_len = new_arr.len();

                                    // Update the source array element
                                    if let Some(Value::Static(JsValue::Array { elements: parent_arr, source: parent_source })) = self.env.lookup(source_var, self.allocator) {
                                        let mut new_parent_arr: Vec<Value<'a>> = parent_arr.iter().map(|v| v.clone_in(self.allocator)).collect();
                                        if source_idx < new_parent_arr.len() {
                                            new_parent_arr[source_idx] = Value::Static(JsValue::Array {
                                                elements: new_arr,
                                                source: None
                                            });
                                            self.env.set(source_var, Value::Static(JsValue::Array {
                                                elements: new_parent_arr,
                                                source: parent_source
                                            }));
                                            self.trace_decision(&format!("{}[{}].push()", source_var, source_idx), "STATIC tracked array mutation");
                                            return Ok(Value::Static(JsValue::Number(new_len as f64)));
                                        }
                                    }
                                }
                                "pop" => {
                                    let mut new_arr: Vec<Value<'a>> = arr.iter().map(|v| v.clone_in(self.allocator)).collect();
                                    let popped = new_arr.pop().unwrap_or(Value::Static(JsValue::Undefined));

                                    // Update the source array element
                                    if let Some(Value::Static(JsValue::Array { elements: parent_arr, source: parent_source })) = self.env.lookup(source_var, self.allocator) {
                                        let mut new_parent_arr: Vec<Value<'a>> = parent_arr.iter().map(|v| v.clone_in(self.allocator)).collect();
                                        if source_idx < new_parent_arr.len() {
                                            new_parent_arr[source_idx] = Value::Static(JsValue::Array {
                                                elements: new_arr,
                                                source: None
                                            });
                                            self.env.set(source_var, Value::Static(JsValue::Array {
                                                elements: new_parent_arr,
                                                source: parent_source
                                            }));
                                            self.trace_decision(&format!("{}[{}].pop()", source_var, source_idx), "STATIC tracked array mutation");
                                            return Ok(popped);
                                        }
                                    }
                                }
                                "unshift" => {
                                    let mut prefix: Vec<Value<'a>> = Vec::new();
                                    for arg in &call.arguments {
                                        if let Some(arg_expr) = arg.as_expression() {
                                            let val = self.eval_expr(arg_expr)?;
                                            prefix.push(val);
                                        }
                                    }
                                    let mut new_arr: Vec<Value<'a>> = prefix;
                                    new_arr.extend(arr.iter().map(|v| v.clone_in(self.allocator)));
                                    let new_len = new_arr.len();

                                    // Update the source array element
                                    if let Some(Value::Static(JsValue::Array { elements: parent_arr, source: parent_source })) = self.env.lookup(source_var, self.allocator) {
                                        let mut new_parent_arr: Vec<Value<'a>> = parent_arr.iter().map(|v| v.clone_in(self.allocator)).collect();
                                        if source_idx < new_parent_arr.len() {
                                            new_parent_arr[source_idx] = Value::Static(JsValue::Array {
                                                elements: new_arr,
                                                source: None
                                            });
                                            self.env.set(source_var, Value::Static(JsValue::Array {
                                                elements: new_parent_arr,
                                                source: parent_source
                                            }));
                                            self.trace_decision(&format!("{}[{}].unshift()", source_var, source_idx), "STATIC tracked array mutation");
                                            return Ok(Value::Static(JsValue::Number(new_len as f64)));
                                        }
                                    }
                                }
                                "slice" => {
                                    let start_val = match call.arguments.get(0)
                                        .and_then(|arg| arg.as_expression())
                                    {
                                        Some(expr) => Some(self.eval_expr(expr)?),
                                        None => None,
                                    };
                                    let end_val = match call.arguments.get(1)
                                        .and_then(|arg| arg.as_expression())
                                    {
                                        Some(expr) => Some(self.eval_expr(expr)?),
                                        None => None,
                                    };
                                    if start_val.as_ref().map_or(false, |v| matches!(v, Value::Dynamic(_)))
                                        || end_val.as_ref().map_or(false, |v| matches!(v, Value::Dynamic(_)))
                                    {
                                        return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                                    }

                                    let len = arr.len();
                                    let start_num = start_val
                                        .as_ref()
                                        .and_then(|v| self.value_to_number(v))
                                        .unwrap_or(0.0);
                                    let end_num = end_val
                                        .as_ref()
                                        .and_then(|v| self.value_to_number(v))
                                        .unwrap_or(len as f64);
                                    let start = self.slice_index(start_num, len);
                                    let end = self.slice_index(end_num, len);
                                    let (start, end) = if end < start { (start, start) } else { (start, end) };
                                    let new_arr: Vec<Value<'a>> = arr[start..end]
                                        .iter()
                                        .map(|v| v.clone_in(self.allocator))
                                        .collect();
                                    self.trace_decision(&format!("{}[{}].slice()", source_var, source_idx), "STATIC tracked array slice");
                                    return Ok(Value::Static(JsValue::Array { elements: new_arr, source: None }));
                                }
                                _ => {}
                            }
                        }
                    }
                }

                // Evaluate the callee to see if it's a closure or stored function
                let callee_val = self.eval_expr(&call.callee)?;

                // Evaluate arguments
                let mut arg_values = Vec::new();
                for arg in &call.arguments {
                    if let Some(e) = arg.as_expression() {
                        arg_values.push(self.eval_expr(e)?);
                    }
                }

                // Handle closure calls
                if let Value::Closure { params, body, env: closure_env, .. } = callee_val {
                    return self.call_closure(&params, body, &closure_env, arg_values, "closure");
                }

                // Check for user-defined function calls (e.g., run(program))
                if let Expression::Identifier(id) = &call.callee {
                    let fn_name = id.name.to_string();

                    // Look up and call the function from the stored functions map
                    if let Some(func) = self.functions.get(&fn_name) {
                        let params = func.params.clone();
                        let body = func.body;

                        // For top-level function declarations, use the global env as closure env
                        let closure_env = Rc::new(Env::new());
                        return self.call_closure(&params, body, &closure_env, arg_values, &fn_name);
                    }
                }

                // Check for IIFE (Immediately Invoked Function Expression) and evaluate it
                let maybe_func: Option<&'a Function<'a>> = match &call.callee {
                    Expression::FunctionExpression(f) => Some(f.as_ref()),
                    Expression::ParenthesizedExpression(p) => {
                        if let Expression::FunctionExpression(f) = &p.expression {
                            Some(f.as_ref())
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(func) = maybe_func {
                    if let Some(body) = &func.body {
                        // Get params
                        let params: Vec<String> = func.params.items.iter()
                            .filter_map(|p| {
                                match &p.pattern {
                                    BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
                                    _ => None,
                                }
                            })
                            .collect();

                        // IIFE captures current environment
                        let closure_env = Rc::clone(&self.env);
                        return self.call_closure(&params, body, &closure_env, arg_values, "IIFE");
                    }
                }

                self.trace_decision("unknown call", "DYNAMIC -> pass through");
                self.stats.unknown_calls += 1;
                self.record_unknown_call(expr);
                // Build a new call expression with specialized arguments
                // This ensures static arguments (like array values) are inlined
                let callee_expr = self.value_to_expr(&callee_val);
                let specialized_args: oxc_allocator::Vec<'a, Argument<'a>> = self.ast.vec_from_iter(
                    arg_values.iter().map(|v| Argument::from(self.value_to_expr(v)))
                );
                let new_call = self.ast.expression_call(
                    SPAN, callee_expr, None::<TSTypeParameterInstantiation>, specialized_args, false
                );
                Ok(Value::Dynamic(new_call))
            }

            Expression::FunctionExpression(func) => {
                // Create a closure capturing the current environment
                let params: Vec<String> = func.params.items.iter()
                    .filter_map(|p| {
                        match &p.pattern {
                            BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
                            _ => None,
                        }
                    })
                    .collect();

                if let Some(body) = &func.body {
                    Ok(Value::Closure {
                        params,
                        body,
                        env: Rc::clone(&self.env),
                        original: Some(func.as_ref()),
                    })
                } else {
                    Ok(Value::Dynamic(expr.clone_in(self.allocator)))
                }
            }

            Expression::ArrowFunctionExpression(arrow) => {
                // Create a closure for arrow functions
                let params: Vec<String> = arrow.params.items.iter()
                    .filter_map(|p| {
                        match &p.pattern {
                            BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
                            _ => None,
                        }
                    })
                    .collect();

                // Arrow function body is always a FunctionBody, even for expression bodies
                // The `expression` field tells us if it's `() => expr` vs `() => { ... }`
                Ok(Value::Closure {
                    params,
                    body: &arrow.body,
                    env: Rc::clone(&self.env),
                    original: None,
                })
            }

            Expression::NewExpression(new_expr) => {
                // Check for built-in constructors we can handle statically
                if let Expression::Identifier(id) = &new_expr.callee {
                    let constructor_name = id.name.as_str();
                    match constructor_name {
                        "TextDecoder" => {
                            // new TextDecoder() - create a static TextDecoder
                            self.trace_decision("new TextDecoder()", "STATIC");
                            return Ok(Value::Static(JsValue::TextDecoder));
                        }
                        "Uint8Array" => {
                            // new Uint8Array(array) or new Uint8Array(arraybuffer)
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    // Check if the argument is an identifier (for backing store tracking)
                                    let backing_var_name = if let Expression::Identifier(id) = arg_expr {
                                        Some(id.name.to_string())
                                    } else {
                                        None
                                    };

                                    let arg_val = self.eval_expr(arg_expr)?;
                                    match arg_val {
                                        Value::Static(JsValue::Array { elements: arr, source: _ }) => {
                                            // Convert array of numbers to Uint8Array
                                            let bytes: Vec<u8> = arr.iter().filter_map(|v| {
                                                if let Value::Static(JsValue::Number(n)) = v {
                                                    Some(*n as u8)
                                                } else {
                                                    None
                                                }
                                            }).collect();
                                            if bytes.len() == arr.len() {
                                                self.trace_decision("new Uint8Array([...])", "STATIC");
                                                return Ok(Value::Static(JsValue::Uint8Array { bytes, backing_var: None }));
                                            }
                                        }
                                        Value::Static(JsValue::ArrayBuffer(bytes)) => {
                                            // Create Uint8Array view of ArrayBuffer
                                            // Track the backing variable name for shared backing store semantics
                                            self.trace_decision("new Uint8Array(buffer)", "STATIC");
                                            return Ok(Value::Static(JsValue::Uint8Array { bytes, backing_var: backing_var_name }));
                                        }
                                        _ => {}
                                    }
                                }
                            } else {
                                // new Uint8Array() with no args = empty array
                                self.trace_decision("new Uint8Array()", "STATIC (empty)");
                                return Ok(Value::Static(JsValue::Uint8Array { bytes: Vec::new(), backing_var: None }));
                            }
                        }
                        "Int8Array" => {
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    let arg_val = self.eval_expr(arg_expr)?;
                                    if let Value::Static(JsValue::Array { elements: arr, source: _ }) = arg_val {
                                        let vals: Vec<i8> = arr.iter().filter_map(|v| {
                                            if let Value::Static(JsValue::Number(n)) = v {
                                                Some(*n as i8)
                                            } else {
                                                None
                                            }
                                        }).collect();
                                        if vals.len() == arr.len() {
                                            self.trace_decision("new Int8Array([...])", "STATIC");
                                            return Ok(Value::Static(JsValue::Int8Array(vals)));
                                        }
                                    }
                                }
                            } else {
                                return Ok(Value::Static(JsValue::Int8Array(Vec::new())));
                            }
                        }
                        "Uint8ClampedArray" => {
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    let arg_val = self.eval_expr(arg_expr)?;
                                    if let Value::Static(JsValue::Array { elements: arr, source: _ }) = arg_val {
                                        let bytes: Vec<u8> = arr.iter().filter_map(|v| {
                                            if let Value::Static(JsValue::Number(n)) = v {
                                                Some((*n).clamp(0.0, 255.0) as u8)
                                            } else {
                                                None
                                            }
                                        }).collect();
                                        if bytes.len() == arr.len() {
                                            self.trace_decision("new Uint8ClampedArray([...])", "STATIC");
                                            return Ok(Value::Static(JsValue::Uint8ClampedArray(bytes)));
                                        }
                                    }
                                }
                            } else {
                                return Ok(Value::Static(JsValue::Uint8ClampedArray(Vec::new())));
                            }
                        }
                        "Int16Array" => {
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    let arg_val = self.eval_expr(arg_expr)?;
                                    if let Value::Static(JsValue::Array { elements: arr, source: _ }) = arg_val {
                                        let vals: Vec<i16> = arr.iter().filter_map(|v| {
                                            if let Value::Static(JsValue::Number(n)) = v {
                                                Some(*n as i16)
                                            } else {
                                                None
                                            }
                                        }).collect();
                                        if vals.len() == arr.len() {
                                            self.trace_decision("new Int16Array([...])", "STATIC");
                                            return Ok(Value::Static(JsValue::Int16Array(vals)));
                                        }
                                    }
                                }
                            } else {
                                return Ok(Value::Static(JsValue::Int16Array(Vec::new())));
                            }
                        }
                        "Uint16Array" => {
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    let arg_val = self.eval_expr(arg_expr)?;
                                    if let Value::Static(JsValue::Array { elements: arr, source: _ }) = arg_val {
                                        let vals: Vec<u16> = arr.iter().filter_map(|v| {
                                            if let Value::Static(JsValue::Number(n)) = v {
                                                Some(*n as u16)
                                            } else {
                                                None
                                            }
                                        }).collect();
                                        if vals.len() == arr.len() {
                                            self.trace_decision("new Uint16Array([...])", "STATIC");
                                            return Ok(Value::Static(JsValue::Uint16Array(vals)));
                                        }
                                    }
                                }
                            } else {
                                return Ok(Value::Static(JsValue::Uint16Array(Vec::new())));
                            }
                        }
                        "Int32Array" => {
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    let arg_val = self.eval_expr(arg_expr)?;
                                    if let Value::Static(JsValue::Array { elements: arr, source: _ }) = arg_val {
                                        let vals: Vec<i32> = arr.iter().filter_map(|v| {
                                            if let Value::Static(JsValue::Number(n)) = v {
                                                Some(*n as i32)
                                            } else {
                                                None
                                            }
                                        }).collect();
                                        if vals.len() == arr.len() {
                                            self.trace_decision("new Int32Array([...])", "STATIC");
                                            return Ok(Value::Static(JsValue::Int32Array(vals)));
                                        }
                                    }
                                }
                            } else {
                                return Ok(Value::Static(JsValue::Int32Array(Vec::new())));
                            }
                        }
                        "Uint32Array" => {
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    let arg_val = self.eval_expr(arg_expr)?;
                                    if let Value::Static(JsValue::Array { elements: arr, source: _ }) = arg_val {
                                        let vals: Vec<u32> = arr.iter().filter_map(|v| {
                                            if let Value::Static(JsValue::Number(n)) = v {
                                                Some(*n as u32)
                                            } else {
                                                None
                                            }
                                        }).collect();
                                        if vals.len() == arr.len() {
                                            self.trace_decision("new Uint32Array([...])", "STATIC");
                                            return Ok(Value::Static(JsValue::Uint32Array(vals)));
                                        }
                                    }
                                }
                            } else {
                                return Ok(Value::Static(JsValue::Uint32Array(Vec::new())));
                            }
                        }
                        "Float32Array" => {
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    let arg_val = self.eval_expr(arg_expr)?;
                                    if let Value::Static(JsValue::Array { elements: arr, source: _ }) = arg_val {
                                        let vals: Vec<f32> = arr.iter().filter_map(|v| {
                                            if let Value::Static(JsValue::Number(n)) = v {
                                                Some(*n as f32)
                                            } else {
                                                None
                                            }
                                        }).collect();
                                        if vals.len() == arr.len() {
                                            self.trace_decision("new Float32Array([...])", "STATIC");
                                            return Ok(Value::Static(JsValue::Float32Array(vals)));
                                        }
                                    }
                                }
                            } else {
                                return Ok(Value::Static(JsValue::Float32Array(Vec::new())));
                            }
                        }
                        "Float64Array" => {
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    let arg_val = self.eval_expr(arg_expr)?;
                                    if let Value::Static(JsValue::Array { elements: arr, source: _ }) = arg_val {
                                        let vals: Vec<f64> = arr.iter().filter_map(|v| {
                                            if let Value::Static(JsValue::Number(n)) = v {
                                                Some(*n)
                                            } else {
                                                None
                                            }
                                        }).collect();
                                        if vals.len() == arr.len() {
                                            self.trace_decision("new Float64Array([...])", "STATIC");
                                            return Ok(Value::Static(JsValue::Float64Array(vals)));
                                        }
                                    }
                                }
                            } else {
                                return Ok(Value::Static(JsValue::Float64Array(Vec::new())));
                            }
                        }
                        "ArrayBuffer" => {
                            // new ArrayBuffer(length) - create empty buffer
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    let arg_val = self.eval_expr(arg_expr)?;
                                    if let Value::Static(JsValue::Number(len)) = arg_val {
                                        let buffer = vec![0u8; len as usize];
                                        self.trace_decision("new ArrayBuffer(len)", "STATIC");
                                        return Ok(Value::Static(JsValue::ArrayBuffer(buffer)));
                                    }
                                }
                            }
                        }
                        "DataView" => {
                            // new DataView(buffer) - wrap an ArrayBuffer
                            if let Some(arg) = new_expr.arguments.first() {
                                if let Some(arg_expr) = arg.as_expression() {
                                    let arg_val = self.eval_expr(arg_expr)?;
                                    if let Value::Static(JsValue::ArrayBuffer(bytes)) = arg_val {
                                        self.trace_decision("new DataView(buffer)", "STATIC");
                                        return Ok(Value::Static(JsValue::DataView(bytes)));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                // Return the new expression as-is if we can't handle it
                Ok(Value::Dynamic(expr.clone_in(self.allocator)))
            }

            Expression::SequenceExpression(seq) => {
                // Evaluate all expressions, return the last one
                let mut result = Value::Static(JsValue::Undefined);
                for expr in &seq.expressions {
                    result = self.eval_expr(expr)?;
                }
                Ok(result)
            }

            Expression::UnaryExpression(unary) => {
                let arg = self.eval_expr(&unary.argument)?;
                match arg {
                    Value::Static(v) => {
                        let result = match unary.operator {
                            UnaryOperator::UnaryNegation => {
                                if let Some(n) = v.to_number() {
                                    JsValue::Number(-n)
                                } else {
                                    return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                                }
                            }
                            UnaryOperator::UnaryPlus => {
                                if let Some(n) = v.to_number() {
                                    JsValue::Number(n)
                                } else {
                                    return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                                }
                            }
                            UnaryOperator::LogicalNot => {
                                JsValue::Boolean(!v.is_truthy())
                            }
                            UnaryOperator::BitwiseNot => {
                                if let Some(n) = v.to_number() {
                                    JsValue::Number((!(n as i32)) as f64)
                                } else {
                                    return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                                }
                            }
                            UnaryOperator::Typeof => {
                                let t = match v {
                                    JsValue::Undefined => "undefined",
                                    JsValue::Null => "object",
                                    JsValue::Boolean(_) => "boolean",
                                    JsValue::Number(_) => "number",
                                    JsValue::String(_) => "string",
                                    JsValue::Array { .. } => "object",
                                    // All typed arrays, TextDecoder, etc. are objects
                                    JsValue::TextDecoder => "object",
                                    JsValue::Date => "function",  // Date is a function/constructor
                                    JsValue::Int8Array(_) => "object",
                                    JsValue::Uint8Array { .. } => "object",
                                    JsValue::Uint8ClampedArray(_) => "object",
                                    JsValue::Int16Array(_) => "object",
                                    JsValue::Uint16Array(_) => "object",
                                    JsValue::Int32Array(_) => "object",
                                    JsValue::Uint32Array(_) => "object",
                                    JsValue::Float32Array(_) => "object",
                                    JsValue::Float64Array(_) => "object",
                                    JsValue::BigInt64Array(_) => "object",
                                    JsValue::BigUint64Array(_) => "object",
                                    JsValue::ArrayBuffer(_) => "object",
                                    JsValue::DataView(_) => "object",
                                    JsValue::Object(_) => "object",
                                };
                                JsValue::String(t.to_string())
                            }
                            _ => {
                                // For other unary ops, build AST
                                let arg_expr = self.js_value_to_expr(&v);
                                return Ok(Value::Dynamic(self.ast.expression_unary(SPAN, unary.operator, arg_expr)));
                            }
                        };
                        Ok(Value::Static(result))
                    }
                    Value::Closure { .. } => {
                        // typeof function is "function"
                        if unary.operator == UnaryOperator::Typeof {
                            Ok(Value::Static(JsValue::String("function".to_string())))
                        } else {
                            let arg_expr = self.value_to_expr(&arg);
                            Ok(Value::Dynamic(self.ast.expression_unary(SPAN, unary.operator, arg_expr)))
                        }
                    }
                    Value::Dynamic(arg_expr) => {
                        Ok(Value::Dynamic(self.ast.expression_unary(SPAN, unary.operator, arg_expr)))
                    }
                }
            }

            Expression::ParenthesizedExpression(paren) => {
                self.eval_expr(&paren.expression)
            }

            Expression::ObjectExpression(obj) => {
                // Evaluate all properties
                let mut props: std::collections::HashMap<String, Value<'a>> = std::collections::HashMap::new();
                let mut has_dynamic = false;

                for prop_kind in &obj.properties {
                    match prop_kind {
                        ObjectPropertyKind::ObjectProperty(prop) => {
                            // Get property key
                            let key = match &prop.key {
                                PropertyKey::StaticIdentifier(id) => Some(id.name.to_string()),
                                PropertyKey::StringLiteral(s) => Some(s.value.to_string()),
                                PropertyKey::NumericLiteral(n) => Some(n.value.to_string()),
                                _ => None,
                            };

                            if let Some(key_str) = key {
                                let value = self.eval_expr(&prop.value)?;
                                if matches!(&value, Value::Dynamic(_)) {
                                    has_dynamic = true;
                                }
                                props.insert(key_str, value);
                            } else {
                                // Computed or unsupported key - treat as dynamic
                                return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                            }
                        }
                        ObjectPropertyKind::SpreadProperty(_) => {
                            // Spread properties - treat entire object as dynamic
                            return Ok(Value::Dynamic(expr.clone_in(self.allocator)));
                        }
                    }
                }

                if has_dynamic {
                    // Build an object expression with evaluated elements
                    let properties: oxc_allocator::Vec<'a, ObjectPropertyKind<'a>> = self.ast.vec_from_iter(
                        props.into_iter().map(|(key, value)| {
                            let key_atom: Atom<'a> = Atom::from_in(key.as_str(), self.allocator);
                            let key_expr = PropertyKey::StaticIdentifier(
                                self.ast.alloc(IdentifierName { span: SPAN, name: key_atom })
                            );
                            let value_expr = self.value_to_expr(&value);
                            ObjectPropertyKind::ObjectProperty(self.ast.alloc_object_property(
                                SPAN, PropertyKind::Init, key_expr, value_expr, false, false, false
                            ))
                        })
                    );
                    Ok(Value::Dynamic(self.ast.expression_object(SPAN, properties)))
                } else {
                    Ok(Value::Static(JsValue::Object(props)))
                }
            }

            _ => Ok(Value::Dynamic(expr.clone_in(self.allocator))),
        }
    }

    /// Call a closure with proper lexical scoping
    fn call_closure(
        &mut self,
        params: &[String],
        body: &'a FunctionBody<'a>,
        closure_env: &Rc<Env<'a>>,
        arg_values: Vec<Value<'a>>,
        name: &str,
    ) -> Result<Value<'a>, String> {
        self.trace(&format!("call {}()", name));
        self.stats.functions_called += 1;
        self.depth += 1;

        // Create a new environment extending the closure's captured environment
        let call_env = Rc::new(Env::extend(Rc::clone(closure_env)));

        // Bind parameters to arguments
        for (param, arg) in params.iter().zip(arg_values.iter()) {
            call_env.bind(param.clone(), arg.clone_in(self.allocator));
        }
        // Handle missing arguments (bind to undefined)
        for param in params.iter().skip(arg_values.len()) {
            call_env.bind(param.clone(), Value::Static(JsValue::Undefined));
        }

        // Bind the `arguments` object - an array-like object containing all passed arguments
        let arguments_array: Vec<Value<'a>> = arg_values.iter()
            .map(|v| v.clone_in(self.allocator))
            .collect();
        call_env.bind("arguments".to_string(), Value::Static(JsValue::Array {
            elements: arguments_array,
            source: None,
        }));

        // Hoist all var declarations and function declarations - JavaScript semantics
        // This ensures closures defined before a `var` or function can access them
        let stmts_slice: &[Statement<'a>] = &body.statements;
        let mut hoisted_vars: HashSet<String> = HashSet::new();
        let mut hoisted_functions: Vec<(&'a Function<'a>, String)> = Vec::new();
        self.collect_hoisted_vars(stmts_slice, &mut hoisted_vars);
        self.collect_hoisted_functions(stmts_slice, &mut hoisted_functions);


        // Pre-bind hoisted vars to undefined (but don't overwrite params)
        for var_name in &hoisted_vars {
            if !params.contains(var_name) {
                // Only bind if not already bound (e.g., by a param)
                if call_env.bindings.borrow().get(var_name).is_none() {
                    call_env.bind(var_name.clone(), Value::Static(JsValue::Undefined));
                }
            }
        }

        // Now set env so hoisted functions capture it properly
        let saved_env = std::mem::replace(&mut self.env, Rc::clone(&call_env));

        // Hoist function declarations - they're fully hoisted (available at start)
        for (func, name) in &hoisted_functions {
            if !params.contains(name) {
                let func_params: Vec<String> = func.params.items.iter()
                    .filter_map(|p| {
                        match &p.pattern {
                            BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
                            _ => None,
                        }
                    })
                    .collect();
                if let Some(func_body) = &func.body {
                    let closure = Value::Closure {
                        params: func_params,
                        body: func_body,
                        env: Rc::clone(&self.env),  // capture the current call_env
                        original: Some(*func),
                    };
                    call_env.bind(name.clone(), closure);
                }
            }
        }

        // Restore env temporarily, will be swapped back when we save state
        self.env = saved_env;

        // Save current state
        let saved_env = std::mem::replace(&mut self.env, call_env);
        let saved_residual = std::mem::take(&mut self.residual);
        let saved_return_value = std::mem::take(&mut self.return_value);
        let saved_last_value = std::mem::take(&mut self.last_value);
        let saved_in_function = self.in_function;
        self.in_function = true;

        // Evaluate function body
        for stmt in &body.statements {
            self.eval_stmt(stmt)?;
            // Check if we got a return
            if self.return_value.is_some() {
                break;
            }
        }

        // Collect results
        let func_residual = std::mem::take(&mut self.residual);
        let func_return_value = std::mem::take(&mut self.return_value);

        // Collect bindings that were declared in this scope for potential residualization
        let func_env = std::mem::replace(&mut self.env, saved_env);

        // Restore parent state
        self.residual = saved_residual;
        self.return_value = saved_return_value;
        self.last_value = saved_last_value;
        self.in_function = saved_in_function;

        self.depth -= 1;

        // Determine what needs to be residualized
        // We need to emit code if:
        // 1. There are residual statements
        // 2. The return value is dynamic
        let has_residual = !func_residual.is_empty();
        let return_is_dynamic = matches!(&func_return_value, Some(Value::Dynamic(_)));

        if has_residual || return_is_dynamic {
            // Collect free variables from the residual code
            let mut free_vars: HashSet<String> = HashSet::new();
            for stmt in func_residual.iter() {
                self.collect_free_vars_stmt(stmt, &mut free_vars);
            }
            if let Some(Value::Dynamic(ref expr)) = func_return_value {
                self.collect_free_vars_expr(expr, &mut free_vars);
            }

            // Remove parameters from free vars (they're bound)
            for param in params {
                free_vars.remove(param);
            }

            let mut only_local_side_effects = true;
            for name in &free_vars {
                if func_env.bindings.borrow().contains_key(name) {
                    continue;
                }
                only_local_side_effects = false;
                break;
            }

            if has_residual && !return_is_dynamic && only_local_side_effects {
                self.trace_decision(&format!("call {}()", name),
                    "PARTIAL with local-only side effects -> drop residual");
                self.stats.functions_specialized += 1;
                if let Some(ret_val) = func_return_value {
                    return Ok(ret_val);
                }
                return Ok(Value::Static(JsValue::Undefined));
            }

            // Collect variable declarations that need to be residualized
            // We need to look in BOTH the local func_env AND the closure_env chain
            // because free variables may reference values from parent scopes
            let mut var_decls: Vec<(String, Option<Expression<'a>>)> = Vec::new();
            let mut seen_vars: HashSet<String> = HashSet::new();

            // First, add any declarations from the local function environment
            // BUT skip variables that are captured from outer scope (exist in closure_env)
            for (name, value) in func_env.bindings.borrow().iter() {
                // Skip parameters
                if params.contains(name) {
                    continue;
                }
                // Skip captured variables - they exist in the closure_env and shouldn't be re-declared
                // (they're outer-scope variables that the function references, not local vars)
                if closure_env.lookup(name, self.allocator).is_some() {
                    continue;
                }
                // Include this declaration if it's referenced in the residual
                if free_vars.contains(name) {
                    let init = match value {
                        Value::Static(js) => Some(self.js_value_to_expr(js)),
                        Value::Dynamic(expr) => Some(expr.clone_in(self.allocator)),
                        Value::Closure { .. } => Some(self.value_to_expr(value)),
                    };
                    var_decls.push((name.clone(), init));
                    seen_vars.insert(name.clone());
                }
            }

            // Now check for free variables that come from the closure's parent environment chain
            // Only add STATIC values (constants that can be inlined)
            // Dynamic values in closure_env mean the var was WRITTEN to, so it should remain
            // as a reference to outer scope, not be re-declared
            for name in &free_vars {
                // Skip if already handled or is a parameter
                if seen_vars.contains(name) || params.contains(name) {
                    continue;
                }
                // Look up in the closure environment chain
                if let Some(value) = closure_env.lookup(name, self.allocator) {
                    let init = match &value {
                        Value::Static(js) => {
                            if js.is_primitive() {
                                Some(self.js_value_to_expr(js))
                            } else {
                                None
                            }
                        }
                        Value::Closure { .. } => None,
                        // Dynamic values are writes to outer scope - don't re-declare
                        Value::Dynamic(_) => continue,
                    };
                    if let Some(init) = init {
                        var_decls.push((name.clone(), Some(init)));
                        seen_vars.insert(name.clone());
                    }
                }
            }

            self.trace_decision(&format!("call {}()", name),
                &format!("PARTIAL ({} var decls, {} residual stmts) -> specialize",
                    var_decls.len(), func_residual.len()));

            if name == "IIFE" {
                self.stats.iifes_encountered += 1;
            }
            self.stats.functions_specialized += 1;

            // Build the residual function body
            let mut body_stmts: Vec<Statement<'a>> = Vec::new();

            // Add variable declarations
            for (var_name, init) in &var_decls {
                let name_alloc: &'a str = self.alloc_str(var_name.as_str());
                let init_clone = init.as_ref().map(|e| e.clone_in(self.allocator));
                let stmt = self.build_var_decl_stmt(name_alloc, init_clone);
                body_stmts.push(stmt);
            }

            // Add residual statements
            for stmt in func_residual {
                body_stmts.push(stmt);
            }

            // Build IIFE for the residual
            let directives = self.ast.vec();
            let body_vec: oxc_allocator::Vec<'a, Statement<'a>> = self.ast.vec_from_iter(body_stmts.into_iter());
            let function_body = self.ast.function_body(SPAN, directives, body_vec);

            let param_items: oxc_allocator::Vec<'a, FormalParameter<'a>> = self.ast.vec_from_iter(
                params.iter().map(|p| {
                    let pattern = self.ast.binding_pattern_binding_identifier(SPAN, self.alloc_str(p.as_str()));
                    self.ast.formal_parameter(
                        SPAN, self.ast.vec(), pattern,
                        None::<TSTypeAnnotation>, None::<Expression>, false, None, false, false,
                    )
                })
            );
            let formal_params = self.ast.formal_parameters(
                SPAN, FormalParameterKind::FormalParameter, param_items, None::<FormalParameterRest>,
            );

            let func_expr = self.ast.expression_function(
                SPAN, FunctionType::FunctionExpression, None::<BindingIdentifier>,
                false, false, false,
                None::<TSTypeParameterDeclaration>, None::<TSThisParameter>,
                formal_params, None::<TSTypeAnnotation>, Some(function_body),
            );

            let call_args: oxc_allocator::Vec<'a, Argument<'a>> = self.ast.vec_from_iter(
                arg_values.iter().map(|v| Argument::from(self.value_to_expr(v)))
            );
            let iife_call = self.ast.expression_call(
                SPAN, func_expr, None::<TSTypeParameterInstantiation>, call_args, false
            );

            // IMPORTANT: After emitting an IIFE, any outer-scope variables that were modified
            // with Dynamic values need to be reset to Dynamic(Identifier). Otherwise, later
            // uses of these variables would substitute the internal expression which contains
            // references to local variables that don't exist outside the IIFE.
            // Collect names first to avoid borrow conflict
            let dynamic_vars: Vec<String> = closure_env.bindings.borrow().iter()
                .filter(|(_, v)| matches!(v, Value::Dynamic(_)))
                .map(|(n, _)| n.clone())
                .collect();
            for name in dynamic_vars {
                let name_alloc: &'a str = self.alloc_str(name.as_str());
                let ident_expr = self.ast.expression_identifier(SPAN, name_alloc);
                closure_env.bindings.borrow_mut().insert(
                    name,
                    Value::Dynamic(ident_expr)
                );
            }

            // KEY INSIGHT: If return value is static/closure but we have residual side effects,
            // emit the IIFE for side effects but return the actual value (not Dynamic)
            if !return_is_dynamic {
                // Emit IIFE as a statement for side effects
                let iife_stmt = self.ast.statement_expression(SPAN, iife_call);
                self.residual.push(iife_stmt);

                // Return the actual (static/closure) return value
                if let Some(ret_val) = func_return_value {
                    self.trace_decision(&format!("call {}()", name),
                        &format!("PARTIAL with static return -> emit IIFE for side effects"));
                    return Ok(ret_val);
                } else {
                    return Ok(Value::Static(JsValue::Undefined));
                }
            }

            return Ok(Value::Dynamic(iife_call));
        }

        // Fully evaluated
        if let Some(ret_val) = func_return_value {
            let val_desc = match &ret_val {
                Value::Static(v) => format!("STATIC({})", v.to_display(self)),
                Value::Dynamic(_) => "DYNAMIC".to_string(),
                Value::Closure { .. } => "CLOSURE".to_string(),
            };
            self.trace_decision(&format!("call {}()", name), &format!("FULLY EVALUATED -> {}", val_desc));
            self.stats.functions_fully_evaluated += 1;
            return Ok(ret_val);
        }

        self.trace_decision(&format!("call {}()", name), "FULLY EVALUATED -> undefined");
        self.stats.functions_fully_evaluated += 1;
        Ok(Value::Static(JsValue::Undefined))
    }

    /// Try to specialize a closure body given captured variable values.
    /// Returns specialized statements if successful, None otherwise.
    /// This is used during residualization to inline/unroll closure bodies.
    fn try_specialize_closure_body(
        &mut self,
        body: &'a FunctionBody<'a>,
        captured_vars: &[(String, Value<'a>)],
    ) -> Option<Vec<Statement<'a>>> {
        // Collect all variable names from the current environment as "external"
        // These are globals/outer scope vars - mutations to them should be residual
        let mut external: HashSet<String> = HashSet::new();
        self.env.collect_var_names(&mut external);

        // Create an environment with the captured vars bound
        let spec_env = Rc::new(Env::extend(Rc::clone(&self.env)));
        for (name, value) in captured_vars {
            spec_env.bind(name.clone(), value.clone_in(self.allocator));
            // Captured vars are NOT external - they're local to this closure
            external.remove(name);
        }

        // Save current state
        let saved_env = std::mem::replace(&mut self.env, spec_env);
        let saved_residual = std::mem::take(&mut self.residual);
        let saved_return_value = std::mem::take(&mut self.return_value);
        let saved_last_value = std::mem::take(&mut self.last_value);
        let saved_in_function = self.in_function;
        let saved_in_specialization = self.in_specialization;
        let saved_external_vars = std::mem::replace(&mut self.external_vars, external);
        let saved_gas = self.gas;
        self.in_function = true;
        self.in_specialization = true; // Prevent nested specialization
        self.gas = 500000; // Generous gas for specialization

        // Try to evaluate the body
        let mut success = true;
        for stmt in &body.statements {
            if let Err(_) = self.eval_stmt(stmt) {
                success = false;
                break;
            }
            if self.return_value.is_some() {
                break;
            }
        }

        // Collect results
        let spec_residual = std::mem::take(&mut self.residual);
        let spec_return = std::mem::take(&mut self.return_value);

        // Restore state
        self.env = saved_env;
        self.residual = saved_residual;
        self.return_value = saved_return_value;
        self.last_value = saved_last_value;
        self.in_function = saved_in_function;
        self.in_specialization = saved_in_specialization;
        self.external_vars = saved_external_vars;
        self.gas = saved_gas;

        if !success {
            return None;
        }

        // Build the specialized body statements
        let mut result_stmts = spec_residual;

        // Add return statement if there was a return value
        if let Some(ret_val) = spec_return {
            let ret_expr = self.value_to_expr(&ret_val);
            let ret_stmt = self.ast.statement_return(SPAN, Some(ret_expr));
            result_stmts.push(ret_stmt);
        }

        // Only return specialized body if we actually simplified something
        // (i.e., we have residual statements)
        if result_stmts.is_empty() {
            None
        } else {
            Some(result_stmts)
        }
    }

    /// Collect all hoisted var declarations from statements
    /// This scans for `var` declarations at any depth, but stops at function boundaries
    fn collect_hoisted_vars(&self, stmts: &[Statement<'a>], hoisted: &mut HashSet<String>) {
        for stmt in stmts {
            self.collect_hoisted_vars_stmt(stmt, hoisted);
        }
    }

    /// Collect all hoisted function declarations from statements
    /// Function declarations are hoisted to the top of their scope
    fn collect_hoisted_functions(&self, stmts: &'a [Statement<'a>], hoisted: &mut Vec<(&'a Function<'a>, String)>) {
        for stmt in stmts {
            self.collect_hoisted_functions_stmt(stmt, hoisted);
        }
    }

    fn collect_hoisted_functions_stmt(&self, stmt: &'a Statement<'a>, hoisted: &mut Vec<(&'a Function<'a>, String)>) {
        match stmt {
            Statement::FunctionDeclaration(func) => {
                if let Some(id) = &func.id {
                    hoisted.push((func.as_ref(), id.name.to_string()));
                }
            }
            Statement::BlockStatement(block) => {
                // In sloppy mode (non-strict), function declarations in blocks are hoisted
                // We'll treat them as hoisted for simplicity
                for s in &block.body {
                    self.collect_hoisted_functions_stmt(s, hoisted);
                }
            }
            Statement::IfStatement(if_stmt) => {
                self.collect_hoisted_functions_stmt(&if_stmt.consequent, hoisted);
                if let Some(ref alt) = if_stmt.alternate {
                    self.collect_hoisted_functions_stmt(alt, hoisted);
                }
            }
            Statement::WhileStatement(w) => {
                self.collect_hoisted_functions_stmt(&w.body, hoisted);
            }
            Statement::ForStatement(f) => {
                self.collect_hoisted_functions_stmt(&f.body, hoisted);
            }
            Statement::SwitchStatement(sw) => {
                for case in &sw.cases {
                    for s in &case.consequent {
                        self.collect_hoisted_functions_stmt(s, hoisted);
                    }
                }
            }
            Statement::TryStatement(try_stmt) => {
                for s in &try_stmt.block.body {
                    self.collect_hoisted_functions_stmt(s, hoisted);
                }
                if let Some(ref handler) = try_stmt.handler {
                    for s in &handler.body.body {
                        self.collect_hoisted_functions_stmt(s, hoisted);
                    }
                }
                if let Some(ref finalizer) = try_stmt.finalizer {
                    for s in &finalizer.body {
                        self.collect_hoisted_functions_stmt(s, hoisted);
                    }
                }
            }
            // Don't recurse into nested functions - they have their own scope
            _ => {}
        }
    }

    fn collect_hoisted_vars_stmt(&self, stmt: &Statement<'a>, hoisted: &mut HashSet<String>) {
        match stmt {
            Statement::VariableDeclaration(decl) => {
                // Only hoist `var`, not `let` or `const`
                if decl.kind == VariableDeclarationKind::Var {
                    for d in &decl.declarations {
                        if let BindingPattern::BindingIdentifier(id) = &d.id {
                            hoisted.insert(id.name.to_string());
                        }
                    }
                }
            }
            Statement::BlockStatement(block) => {
                // Recurse into blocks - var hoists out of them
                for s in &block.body {
                    self.collect_hoisted_vars_stmt(s, hoisted);
                }
            }
            Statement::IfStatement(if_stmt) => {
                self.collect_hoisted_vars_stmt(&if_stmt.consequent, hoisted);
                if let Some(ref alt) = if_stmt.alternate {
                    self.collect_hoisted_vars_stmt(alt, hoisted);
                }
            }
            Statement::WhileStatement(w) => {
                self.collect_hoisted_vars_stmt(&w.body, hoisted);
            }
            Statement::ForStatement(f) => {
                // Check the init for var declarations
                if let Some(ForStatementInit::VariableDeclaration(decl)) = &f.init {
                    if decl.kind == VariableDeclarationKind::Var {
                        for d in &decl.declarations {
                            if let BindingPattern::BindingIdentifier(id) = &d.id {
                                hoisted.insert(id.name.to_string());
                            }
                        }
                    }
                }
                self.collect_hoisted_vars_stmt(&f.body, hoisted);
            }
            Statement::SwitchStatement(sw) => {
                for case in &sw.cases {
                    for s in &case.consequent {
                        self.collect_hoisted_vars_stmt(s, hoisted);
                    }
                }
            }
            Statement::TryStatement(try_stmt) => {
                for s in &try_stmt.block.body {
                    self.collect_hoisted_vars_stmt(s, hoisted);
                }
                if let Some(ref handler) = try_stmt.handler {
                    for s in &handler.body.body {
                        self.collect_hoisted_vars_stmt(s, hoisted);
                    }
                }
                if let Some(ref finalizer) = try_stmt.finalizer {
                    for s in &finalizer.body {
                        self.collect_hoisted_vars_stmt(s, hoisted);
                    }
                }
            }
            // Note: we do NOT recurse into FunctionDeclaration - var doesn't hoist out of functions
            // Also not into ArrowFunctionExpression or FunctionExpression
            _ => {}
        }
    }

    /// Collect free variables from a statement
    fn collect_free_vars_stmt(&self, stmt: &Statement<'a>, free_vars: &mut HashSet<String>) {
        match stmt {
            Statement::ExpressionStatement(expr) => {
                self.collect_free_vars_expr(&expr.expression, free_vars);
            }
            Statement::ReturnStatement(ret) => {
                if let Some(ref arg) = ret.argument {
                    self.collect_free_vars_expr(arg, free_vars);
                }
            }
            Statement::VariableDeclaration(decl) => {
                for d in &decl.declarations {
                    if let Some(ref init) = d.init {
                        self.collect_free_vars_expr(init, free_vars);
                    }
                }
            }
            Statement::BlockStatement(block) => {
                for s in &block.body {
                    self.collect_free_vars_stmt(s, free_vars);
                }
            }
            Statement::WhileStatement(w) => {
                self.collect_free_vars_expr(&w.test, free_vars);
                self.collect_free_vars_stmt(&w.body, free_vars);
            }
            Statement::ForStatement(f) => {
                if let Some(ref init) = f.init {
                    if let ForStatementInit::VariableDeclaration(decl) = init {
                        for d in &decl.declarations {
                            if let Some(ref init_expr) = d.init {
                                self.collect_free_vars_expr(init_expr, free_vars);
                            }
                        }
                    } else if let Some(expr) = init.as_expression() {
                        self.collect_free_vars_expr(expr, free_vars);
                    }
                }
                if let Some(ref test) = f.test {
                    self.collect_free_vars_expr(test, free_vars);
                }
                if let Some(ref update) = f.update {
                    self.collect_free_vars_expr(update, free_vars);
                }
                self.collect_free_vars_stmt(&f.body, free_vars);
            }
            Statement::IfStatement(if_stmt) => {
                self.collect_free_vars_expr(&if_stmt.test, free_vars);
                self.collect_free_vars_stmt(&if_stmt.consequent, free_vars);
                if let Some(ref alt) = if_stmt.alternate {
                    self.collect_free_vars_stmt(alt, free_vars);
                }
            }
            Statement::TryStatement(try_stmt) => {
                for s in &try_stmt.block.body {
                    self.collect_free_vars_stmt(s, free_vars);
                }
                if let Some(ref handler) = try_stmt.handler {
                    for s in &handler.body.body {
                        self.collect_free_vars_stmt(s, free_vars);
                    }
                }
                if let Some(ref finalizer) = try_stmt.finalizer {
                    for s in &finalizer.body {
                        self.collect_free_vars_stmt(s, free_vars);
                    }
                }
            }
            Statement::ThrowStatement(throw) => {
                self.collect_free_vars_expr(&throw.argument, free_vars);
            }
            Statement::SwitchStatement(sw) => {
                self.collect_free_vars_expr(&sw.discriminant, free_vars);
                for case in &sw.cases {
                    if let Some(ref test) = case.test {
                        self.collect_free_vars_expr(test, free_vars);
                    }
                    for s in &case.consequent {
                        self.collect_free_vars_stmt(s, free_vars);
                    }
                }
            }
            _ => {}
        }
    }

    /// Collect free variables from an expression
    fn collect_free_vars_expr(&self, expr: &Expression<'a>, free_vars: &mut HashSet<String>) {
        match expr {
            Expression::Identifier(id) => {
                free_vars.insert(id.name.to_string());
            }
            Expression::BinaryExpression(bin) => {
                self.collect_free_vars_expr(&bin.left, free_vars);
                self.collect_free_vars_expr(&bin.right, free_vars);
            }
            Expression::UnaryExpression(un) => {
                self.collect_free_vars_expr(&un.argument, free_vars);
            }
            Expression::CallExpression(call) => {
                self.collect_free_vars_expr(&call.callee, free_vars);
                for arg in &call.arguments {
                    if let Some(e) = arg.as_expression() {
                        self.collect_free_vars_expr(e, free_vars);
                    }
                }
            }
            Expression::StaticMemberExpression(mem) => {
                self.collect_free_vars_expr(&mem.object, free_vars);
            }
            Expression::ComputedMemberExpression(mem) => {
                self.collect_free_vars_expr(&mem.object, free_vars);
                self.collect_free_vars_expr(&mem.expression, free_vars);
            }
            Expression::AssignmentExpression(assign) => {
                if let AssignmentTarget::AssignmentTargetIdentifier(id) = &assign.left {
                    free_vars.insert(id.name.to_string());
                }
                self.collect_free_vars_expr(&assign.right, free_vars);
            }
            Expression::ConditionalExpression(cond) => {
                self.collect_free_vars_expr(&cond.test, free_vars);
                self.collect_free_vars_expr(&cond.consequent, free_vars);
                self.collect_free_vars_expr(&cond.alternate, free_vars);
            }
            Expression::ArrayExpression(arr) => {
                for el in &arr.elements {
                    if let Some(e) = el.as_expression() {
                        self.collect_free_vars_expr(e, free_vars);
                    }
                }
            }
            Expression::ParenthesizedExpression(p) => {
                self.collect_free_vars_expr(&p.expression, free_vars);
            }
            Expression::SequenceExpression(seq) => {
                for e in &seq.expressions {
                    self.collect_free_vars_expr(e, free_vars);
                }
            }
            Expression::UpdateExpression(up) => {
                if let SimpleAssignmentTarget::AssignmentTargetIdentifier(id) = &up.argument {
                    free_vars.insert(id.name.to_string());
                }
            }
            Expression::ObjectExpression(obj) => {
                for prop in &obj.properties {
                    if let ObjectPropertyKind::ObjectProperty(p) = prop {
                        self.collect_free_vars_expr(&p.value, free_vars);
                        // Also collect from computed property keys
                        if p.computed {
                            self.collect_free_vars_expr(&p.key.as_expression().expect("computed key"), free_vars);
                        }
                    }
                }
            }
            Expression::FunctionExpression(func) => {
                // Collect free vars from the function body, but remove parameters
                if let Some(ref body) = func.body {
                    let params: HashSet<String> = func.params.items.iter()
                        .filter_map(|p| {
                            match &p.pattern {
                                BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
                                _ => None,
                            }
                        })
                        .collect();
                    let mut local_vars: HashSet<String> = HashSet::new();
                    let mut inner_free_vars: HashSet<String> = HashSet::new();
                    for stmt in &body.statements {
                        self.collect_free_vars_stmt(stmt, &mut inner_free_vars);
                    }
                    for stmt in &body.statements {
                        self.collect_hoisted_vars_stmt(stmt, &mut local_vars);
                    }
                    // Remove parameters from inner free vars, then add to outer
                    for name in inner_free_vars {
                        if !params.contains(&name) && !local_vars.contains(&name) {
                            free_vars.insert(name);
                        }
                    }
                }
            }
            Expression::ArrowFunctionExpression(func) => {
                // Collect free vars from the arrow function body, but remove parameters
                let params: HashSet<String> = func.params.items.iter()
                    .filter_map(|p| {
                        match &p.pattern {
                            BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
                            _ => None,
                        }
                    })
                    .collect();
                let mut inner_free_vars: HashSet<String> = HashSet::new();
                let mut local_vars: HashSet<String> = HashSet::new();
                for stmt in &func.body.statements {
                    self.collect_free_vars_stmt(stmt, &mut inner_free_vars);
                }
                for stmt in &func.body.statements {
                    self.collect_hoisted_vars_stmt(stmt, &mut local_vars);
                }
                // Remove parameters from inner free vars, then add to outer
                for name in inner_free_vars {
                    if !params.contains(&name) && !local_vars.contains(&name) {
                        free_vars.insert(name);
                    }
                }
            }
            Expression::LogicalExpression(log) => {
                self.collect_free_vars_expr(&log.left, free_vars);
                self.collect_free_vars_expr(&log.right, free_vars);
            }
            _ => {}
        }
    }

    /// Emit captured variables from closures contained in a value
    /// This is needed when residualizing closures at top-level - their captured vars must be emitted
    fn emit_captured_vars(&mut self, value: &Value<'a>) {
        self.emit_captured_vars_depth(value, 0);
    }

    fn emit_captured_vars_depth(&mut self, value: &Value<'a>, depth: usize) {
        // Prevent infinite recursion
        if depth > 50 {
            return;
        }

        match value {
            Value::Closure { body, env, original, .. } => {
                // Collect free variables from the closure body
                let mut free_vars: HashSet<String> = HashSet::new();
                let mut local_vars: HashSet<String> = HashSet::new();

                if let Some(func) = original {
                    // Use original function to get params
                    let params: HashSet<String> = func.params.items.iter()
                        .filter_map(|p| {
                            match &p.pattern {
                                BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
                                _ => None,
                            }
                        })
                        .collect();
                    if let Some(body) = &func.body {
                        // Collect all referenced identifiers
                        for stmt in &body.statements {
                            self.collect_free_vars_stmt(stmt, &mut free_vars);
                        }
                        // Collect locally declared variables
                        for stmt in &body.statements {
                            self.collect_hoisted_vars_stmt(stmt, &mut local_vars);
                        }
                    }
                    // Remove params from free_vars
                    for param in &params {
                        free_vars.remove(param);
                    }
                    // Remove local vars from free_vars
                    for local in &local_vars {
                        free_vars.remove(local);
                    }
                } else {
                    for stmt in &body.statements {
                        self.collect_free_vars_stmt(stmt, &mut free_vars);
                    }
                    for stmt in &body.statements {
                        self.collect_hoisted_vars_stmt(stmt, &mut local_vars);
                    }
                    for local in &local_vars {
                        free_vars.remove(local);
                    }
                }

                // Clone env to avoid borrow issues
                let env_clone = env.clone();

                // For each free variable, look it up in the closure env and emit if found
                for var_name in free_vars {
                    if matches!(
                        var_name.as_str(),
                        "Date"
                            | "String"
                            | "Uint8Array"
                            | "Array"
                            | "Object"
                            | "Number"
                            | "Boolean"
                            | "Math"
                            | "JSON"
                            | "console"
                            | "arguments"
                            | "undefined"
                    ) {
                        continue;
                    }
                    if self.emitted_top_level_vars.contains(&var_name) {
                        continue;
                    }
                    // Mark as emitted BEFORE looking up to prevent circular references
                    self.emitted_top_level_vars.insert(var_name.clone());

                    if let Some(var_value) = env_clone.lookup(&var_name, self.allocator) {
                        // Recursively emit captured vars from this value first
                        // For closures, we also need to emit their dependencies
                        self.emit_captured_vars_depth(&var_value, depth + 1);

                        // Emit the var declaration
                        let name_alloc: &'a str = self.alloc_str(&var_name);
                        let init_expr = self.value_to_expr(&var_value);
                        let stmt = self.build_var_decl_stmt(name_alloc, Some(init_expr));
                        self.residual.push(stmt);
                    }
                }
            }
            Value::Static(JsValue::Object(props)) => {
                // Check each property for closures
                // Clone to avoid borrow issues
                let props_clone: Vec<_> = props.iter()
                    .map(|(k, v)| (k.clone(), v.clone_in(self.allocator)))
                    .collect();
                for (_key, prop_value) in props_clone {
                    self.emit_captured_vars_depth(&prop_value, depth + 1);
                }
            }
            Value::Static(JsValue::Array { elements, source: None }) => {
                // Check each element for closures
                // Clone to avoid borrow issues
                let elems_clone: Vec<_> = elements.iter()
                    .map(|v| v.clone_in(self.allocator))
                    .collect();
                for elem in elems_clone {
                    self.emit_captured_vars_depth(&elem, depth + 1);
                }
            }
            Value::Dynamic(expr) => {
                let mut free_vars: HashSet<String> = HashSet::new();
                self.collect_free_vars_expr(expr, &mut free_vars);
                for var_name in free_vars {
                    if matches!(
                        var_name.as_str(),
                        "Date"
                            | "String"
                            | "Uint8Array"
                            | "Array"
                            | "Object"
                            | "Number"
                            | "Boolean"
                            | "Math"
                            | "JSON"
                            | "console"
                            | "arguments"
                            | "undefined"
                    ) {
                        continue;
                    }
                    if self.emitted_top_level_vars.contains(&var_name) {
                        continue;
                    }
                    self.emitted_top_level_vars.insert(var_name.clone());
                    if let Some(var_value) = self.env.lookup(&var_name, self.allocator) {
                        self.emit_captured_vars_depth(&var_value, depth + 1);
                        let name_alloc: &'a str = self.alloc_str(&var_name);
                        let init_expr = self.value_to_expr(&var_value);
                        let stmt = self.build_var_decl_stmt(name_alloc, Some(init_expr));
                        self.residual.push(stmt);
                    }
                }
            }
            _ => {}
        }
    }

    fn collect_assigned_vars_stmt(&self, stmt: &Statement<'a>, assigned: &mut HashSet<String>) {
        match stmt {
            Statement::ExpressionStatement(expr) => {
                self.collect_assigned_vars_expr(&expr.expression, assigned);
            }
            Statement::BlockStatement(block) => {
                for s in &block.body {
                    self.collect_assigned_vars_stmt(s, assigned);
                }
            }
            Statement::IfStatement(stmt) => {
                self.collect_assigned_vars_expr(&stmt.test, assigned);
                self.collect_assigned_vars_stmt(&stmt.consequent, assigned);
                if let Some(alt) = &stmt.alternate {
                    self.collect_assigned_vars_stmt(alt, assigned);
                }
            }
            Statement::WhileStatement(stmt) => {
                self.collect_assigned_vars_expr(&stmt.test, assigned);
                self.collect_assigned_vars_stmt(&stmt.body, assigned);
            }
            Statement::ForStatement(stmt) => {
                if let Some(init) = &stmt.init {
                    match init {
                        ForStatementInit::VariableDeclaration(decl) => {
                            for d in &decl.declarations {
                                if let BindingPattern::BindingIdentifier(id) = &d.id {
                                    assigned.insert(id.name.to_string());
                                }
                            }
                        }
                        _ => {
                            if let Some(expr) = init.as_expression() {
                                self.collect_assigned_vars_expr(expr, assigned);
                            }
                        }
                    }
                }
                if let Some(test) = &stmt.test {
                    self.collect_assigned_vars_expr(test, assigned);
                }
                if let Some(update) = &stmt.update {
                    self.collect_assigned_vars_expr(update, assigned);
                }
                self.collect_assigned_vars_stmt(&stmt.body, assigned);
            }
            Statement::ForInStatement(stmt) => {
                self.collect_assigned_vars_expr(&stmt.right, assigned);
                self.collect_assigned_vars_stmt(&stmt.body, assigned);
            }
            Statement::ForOfStatement(stmt) => {
                self.collect_assigned_vars_expr(&stmt.right, assigned);
                self.collect_assigned_vars_stmt(&stmt.body, assigned);
            }
            Statement::SwitchStatement(stmt) => {
                self.collect_assigned_vars_expr(&stmt.discriminant, assigned);
                for case in &stmt.cases {
                    if let Some(test) = &case.test {
                        self.collect_assigned_vars_expr(test, assigned);
                    }
                    for cons in &case.consequent {
                        self.collect_assigned_vars_stmt(cons, assigned);
                    }
                }
            }
            Statement::TryStatement(stmt) => {
                for s in &stmt.block.body {
                    self.collect_assigned_vars_stmt(s, assigned);
                }
                if let Some(handler) = &stmt.handler {
                    for s in &handler.body.body {
                        self.collect_assigned_vars_stmt(s, assigned);
                    }
                }
                if let Some(finalizer) = &stmt.finalizer {
                    for s in &finalizer.body {
                        self.collect_assigned_vars_stmt(s, assigned);
                    }
                }
            }
            _ => {}
        }
    }

    fn collect_assigned_vars_expr(&self, expr: &Expression<'a>, assigned: &mut HashSet<String>) {
        match expr {
            Expression::AssignmentExpression(assign) => {
                if let AssignmentTarget::AssignmentTargetIdentifier(id) = &assign.left {
                    assigned.insert(id.name.to_string());
                }
                self.collect_assigned_vars_expr(&assign.right, assigned);
            }
            Expression::UpdateExpression(up) => {
                if let SimpleAssignmentTarget::AssignmentTargetIdentifier(id) = &up.argument {
                    assigned.insert(id.name.to_string());
                }
            }
            Expression::BinaryExpression(bin) => {
                self.collect_assigned_vars_expr(&bin.left, assigned);
                self.collect_assigned_vars_expr(&bin.right, assigned);
            }
            Expression::UnaryExpression(un) => {
                self.collect_assigned_vars_expr(&un.argument, assigned);
            }
            Expression::ConditionalExpression(cond) => {
                self.collect_assigned_vars_expr(&cond.test, assigned);
                self.collect_assigned_vars_expr(&cond.consequent, assigned);
                self.collect_assigned_vars_expr(&cond.alternate, assigned);
            }
            Expression::LogicalExpression(log) => {
                self.collect_assigned_vars_expr(&log.left, assigned);
                self.collect_assigned_vars_expr(&log.right, assigned);
            }
            Expression::CallExpression(call) => {
                self.collect_assigned_vars_expr(&call.callee, assigned);
                for arg in &call.arguments {
                    if let Some(e) = arg.as_expression() {
                        self.collect_assigned_vars_expr(e, assigned);
                    }
                }
            }
            Expression::ArrayExpression(arr) => {
                for el in &arr.elements {
                    if let Some(e) = el.as_expression() {
                        self.collect_assigned_vars_expr(e, assigned);
                    }
                }
            }
            Expression::ObjectExpression(obj) => {
                for prop in &obj.properties {
                    if let ObjectPropertyKind::ObjectProperty(p) = prop {
                        self.collect_assigned_vars_expr(&p.value, assigned);
                        if p.computed {
                            self.collect_assigned_vars_expr(&p.key.as_expression().expect("computed key"), assigned);
                        }
                    }
                }
            }
            Expression::ParenthesizedExpression(p) => {
                self.collect_assigned_vars_expr(&p.expression, assigned);
            }
            Expression::SequenceExpression(seq) => {
                for e in &seq.expressions {
                    self.collect_assigned_vars_expr(e, assigned);
                }
            }
            _ => {}
        }
    }

    fn collect_usage_hints_stmt(&self, stmt: &Statement<'a>, array_hints: &mut HashSet<String>, number_hints: &mut HashSet<String>) {
        match stmt {
            Statement::ExpressionStatement(expr) => {
                self.collect_usage_hints_expr(&expr.expression, array_hints, number_hints);
            }
            Statement::BlockStatement(block) => {
                for s in &block.body {
                    self.collect_usage_hints_stmt(s, array_hints, number_hints);
                }
            }
            Statement::IfStatement(stmt) => {
                self.collect_usage_hints_expr(&stmt.test, array_hints, number_hints);
                self.collect_usage_hints_stmt(&stmt.consequent, array_hints, number_hints);
                if let Some(alt) = &stmt.alternate {
                    self.collect_usage_hints_stmt(alt, array_hints, number_hints);
                }
            }
            Statement::WhileStatement(stmt) => {
                self.collect_usage_hints_expr(&stmt.test, array_hints, number_hints);
                self.collect_usage_hints_stmt(&stmt.body, array_hints, number_hints);
            }
            Statement::ForStatement(stmt) => {
                if let Some(init) = &stmt.init {
                    if let Some(expr) = init.as_expression() {
                        self.collect_usage_hints_expr(expr, array_hints, number_hints);
                    }
                }
                if let Some(test) = &stmt.test {
                    self.collect_usage_hints_expr(test, array_hints, number_hints);
                }
                if let Some(update) = &stmt.update {
                    self.collect_usage_hints_expr(update, array_hints, number_hints);
                }
                self.collect_usage_hints_stmt(&stmt.body, array_hints, number_hints);
            }
            Statement::ForInStatement(stmt) => {
                self.collect_usage_hints_expr(&stmt.right, array_hints, number_hints);
                self.collect_usage_hints_stmt(&stmt.body, array_hints, number_hints);
            }
            Statement::ForOfStatement(stmt) => {
                self.collect_usage_hints_expr(&stmt.right, array_hints, number_hints);
                self.collect_usage_hints_stmt(&stmt.body, array_hints, number_hints);
            }
            Statement::SwitchStatement(stmt) => {
                self.collect_usage_hints_expr(&stmt.discriminant, array_hints, number_hints);
                for case in &stmt.cases {
                    if let Some(test) = &case.test {
                        self.collect_usage_hints_expr(test, array_hints, number_hints);
                    }
                    for cons in &case.consequent {
                        self.collect_usage_hints_stmt(cons, array_hints, number_hints);
                    }
                }
            }
            Statement::TryStatement(stmt) => {
                for s in &stmt.block.body {
                    self.collect_usage_hints_stmt(s, array_hints, number_hints);
                }
                if let Some(handler) = &stmt.handler {
                    for s in &handler.body.body {
                        self.collect_usage_hints_stmt(s, array_hints, number_hints);
                    }
                }
                if let Some(finalizer) = &stmt.finalizer {
                    for s in &finalizer.body {
                        self.collect_usage_hints_stmt(s, array_hints, number_hints);
                    }
                }
            }
            _ => {}
        }
    }

    fn collect_usage_hints_expr(&self, expr: &Expression<'a>, array_hints: &mut HashSet<String>, number_hints: &mut HashSet<String>) {
        match expr {
            Expression::ComputedMemberExpression(mem) => {
                self.collect_usage_hints_expr(&mem.object, array_hints, number_hints);
                self.collect_usage_hints_expr(&mem.expression, array_hints, number_hints);
            }
            Expression::StaticMemberExpression(mem) => {
                if let Expression::Identifier(id) = &mem.object {
                    let prop = mem.property.name.as_str();
                    if matches!(prop, "push" | "pop" | "unshift" | "length") {
                        array_hints.insert(id.name.to_string());
                    }
                } else if let Expression::ComputedMemberExpression(inner) = &mem.object {
                    if let Expression::Identifier(id) = &inner.object {
                        let prop = mem.property.name.as_str();
                        if matches!(prop, "push" | "pop" | "unshift" | "length") {
                            array_hints.insert(id.name.to_string());
                        }
                    }
                }
                self.collect_usage_hints_expr(&mem.object, array_hints, number_hints);
            }
            Expression::AssignmentExpression(assign) => {
                if let AssignmentTarget::ComputedMemberExpression(mem) = &assign.left {
                    if let Expression::Identifier(id) = &mem.object {
                        array_hints.insert(id.name.to_string());
                    }
                }
                self.collect_usage_hints_expr(&assign.right, array_hints, number_hints);
            }
            Expression::UpdateExpression(up) => {
                if let SimpleAssignmentTarget::AssignmentTargetIdentifier(id) = &up.argument {
                    number_hints.insert(id.name.to_string());
                }
            }
            Expression::BinaryExpression(bin) => {
                if let Expression::Identifier(id) = &bin.left {
                    if matches!(
                        bin.operator,
                        BinaryOperator::Addition
                            | BinaryOperator::Subtraction
                            | BinaryOperator::Multiplication
                            | BinaryOperator::Division
                            | BinaryOperator::Remainder
                            | BinaryOperator::ShiftLeft
                            | BinaryOperator::ShiftRight
                            | BinaryOperator::ShiftRightZeroFill
                            | BinaryOperator::BitwiseAnd
                            | BinaryOperator::BitwiseOR
                            | BinaryOperator::BitwiseXOR
                    ) {
                        number_hints.insert(id.name.to_string());
                    }
                }
                if let Expression::Identifier(id) = &bin.right {
                    if matches!(
                        bin.operator,
                        BinaryOperator::Addition
                            | BinaryOperator::Subtraction
                            | BinaryOperator::Multiplication
                            | BinaryOperator::Division
                            | BinaryOperator::Remainder
                            | BinaryOperator::ShiftLeft
                            | BinaryOperator::ShiftRight
                            | BinaryOperator::ShiftRightZeroFill
                            | BinaryOperator::BitwiseAnd
                            | BinaryOperator::BitwiseOR
                            | BinaryOperator::BitwiseXOR
                    ) {
                        number_hints.insert(id.name.to_string());
                    }
                }
                self.collect_usage_hints_expr(&bin.left, array_hints, number_hints);
                self.collect_usage_hints_expr(&bin.right, array_hints, number_hints);
            }
            Expression::UnaryExpression(un) => {
                self.collect_usage_hints_expr(&un.argument, array_hints, number_hints);
            }
            Expression::ConditionalExpression(cond) => {
                self.collect_usage_hints_expr(&cond.test, array_hints, number_hints);
                self.collect_usage_hints_expr(&cond.consequent, array_hints, number_hints);
                self.collect_usage_hints_expr(&cond.alternate, array_hints, number_hints);
            }
            Expression::LogicalExpression(log) => {
                self.collect_usage_hints_expr(&log.left, array_hints, number_hints);
                self.collect_usage_hints_expr(&log.right, array_hints, number_hints);
            }
            Expression::CallExpression(call) => {
                self.collect_usage_hints_expr(&call.callee, array_hints, number_hints);
                for arg in &call.arguments {
                    if let Some(e) = arg.as_expression() {
                        self.collect_usage_hints_expr(e, array_hints, number_hints);
                    }
                }
            }
            Expression::ArrayExpression(arr) => {
                for el in &arr.elements {
                    if let Some(e) = el.as_expression() {
                        self.collect_usage_hints_expr(e, array_hints, number_hints);
                    }
                }
            }
            Expression::ObjectExpression(obj) => {
                for prop in &obj.properties {
                    if let ObjectPropertyKind::ObjectProperty(p) = prop {
                        self.collect_usage_hints_expr(&p.value, array_hints, number_hints);
                        if p.computed {
                            self.collect_usage_hints_expr(&p.key.as_expression().expect("computed key"), array_hints, number_hints);
                        }
                    }
                }
            }
            Expression::FunctionExpression(func) => {
                if let Some(body) = &func.body {
                    for stmt in &body.statements {
                        self.collect_usage_hints_stmt(stmt, array_hints, number_hints);
                    }
                }
            }
            Expression::ArrowFunctionExpression(func) => {
                for stmt in &func.body.statements {
                    self.collect_usage_hints_stmt(stmt, array_hints, number_hints);
                }
            }
            Expression::ParenthesizedExpression(p) => {
                self.collect_usage_hints_expr(&p.expression, array_hints, number_hints);
            }
            Expression::SequenceExpression(seq) => {
                for e in &seq.expressions {
                    self.collect_usage_hints_expr(e, array_hints, number_hints);
                }
            }
            _ => {}
        }
    }

    fn eval_binary_op(&self, op: BinaryOperator, left: &JsValue<'a>, right: &JsValue<'a>) -> Result<Option<JsValue<'a>>, String> {
        match op {
            BinaryOperator::Addition => {
                if let (Some(l_str), Some(r_str)) = (left.to_js_string(), right.to_js_string()) {
                    if matches!(left, JsValue::String(_)) || matches!(right, JsValue::String(_)) {
                        return Ok(Some(JsValue::String(format!("{}{}", l_str, r_str))));
                    }
                }
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(l + r))),
                    _ => Ok(None),
                }
            }
            BinaryOperator::Subtraction => {
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(l - r))),
                    _ => Ok(None),
                }
            }
            BinaryOperator::Multiplication => {
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(l * r))),
                    _ => Ok(None),
                }
            }
            BinaryOperator::Division => {
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(l / r))),
                    _ => Ok(None),
                }
            }
            BinaryOperator::Remainder => {
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(l % r))),
                    _ => Ok(None),
                }
            }
            // Comparison operators
            BinaryOperator::LessThan => {
                match (left, right) {
                    (JsValue::String(l), JsValue::String(r)) => Ok(Some(JsValue::Boolean(l < r))),
                    _ => match (left.to_number(), right.to_number()) {
                        (Some(l), Some(r)) => Ok(Some(JsValue::Boolean(l < r))),
                        _ => Ok(None),
                    },
                }
            }
            BinaryOperator::LessEqualThan => {
                match (left, right) {
                    (JsValue::String(l), JsValue::String(r)) => Ok(Some(JsValue::Boolean(l <= r))),
                    _ => match (left.to_number(), right.to_number()) {
                        (Some(l), Some(r)) => Ok(Some(JsValue::Boolean(l <= r))),
                        _ => Ok(None),
                    },
                }
            }
            BinaryOperator::GreaterThan => {
                match (left, right) {
                    (JsValue::String(l), JsValue::String(r)) => Ok(Some(JsValue::Boolean(l > r))),
                    _ => match (left.to_number(), right.to_number()) {
                        (Some(l), Some(r)) => Ok(Some(JsValue::Boolean(l > r))),
                        _ => Ok(None),
                    },
                }
            }
            BinaryOperator::GreaterEqualThan => {
                match (left, right) {
                    (JsValue::String(l), JsValue::String(r)) => Ok(Some(JsValue::Boolean(l >= r))),
                    _ => match (left.to_number(), right.to_number()) {
                        (Some(l), Some(r)) => Ok(Some(JsValue::Boolean(l >= r))),
                        _ => Ok(None),
                    },
                }
            }
            BinaryOperator::StrictEquality => {
                Ok(Self::strict_equals(left, right).map(JsValue::Boolean))
            }
            BinaryOperator::StrictInequality => {
                Ok(Self::strict_equals(left, right).map(|v| JsValue::Boolean(!v)))
            }
            BinaryOperator::Equality => {
                Ok(Self::loose_equals(left, right).map(JsValue::Boolean))
            }
            BinaryOperator::Inequality => {
                Ok(Self::loose_equals(left, right).map(|v| JsValue::Boolean(!v)))
            }
            // Bitwise operators (convert to i32 like JS does)
            BinaryOperator::BitwiseAnd => {
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(((l as i32) & (r as i32)) as f64))),
                    _ => Ok(None),
                }
            }
            BinaryOperator::BitwiseOR => {
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(((l as i32) | (r as i32)) as f64))),
                    _ => Ok(None),
                }
            }
            BinaryOperator::BitwiseXOR => {
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(((l as i32) ^ (r as i32)) as f64))),
                    _ => Ok(None),
                }
            }
            BinaryOperator::ShiftLeft => {
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(((l as i32) << ((r as u32) & 0x1f)) as f64))),
                    _ => Ok(None),
                }
            }
            BinaryOperator::ShiftRight => {
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(((l as i32) >> ((r as u32) & 0x1f)) as f64))),
                    _ => Ok(None),
                }
            }
            BinaryOperator::ShiftRightZeroFill => {
                match (left.to_number(), right.to_number()) {
                    (Some(l), Some(r)) => Ok(Some(JsValue::Number(((l as u32) >> ((r as u32) & 0x1f)) as f64))),
                    _ => Ok(None),
                }
            }
            _ => Err(format!("Unsupported binary operator: {:?}", op)),
        }
    }

    fn residual_code(&mut self) -> String {
        // Collect all residual statements
        let mut all_stmts: Vec<Statement<'a>> = Vec::new();
        let mut captured_stmts: Vec<Statement<'a>> = Vec::new();

        // Add the original residual statements
        for s in &self.residual {
            all_stmts.push(s.clone_in(self.allocator));
        }

        // Add final state of event handlers if myGlobal.listeners exists
        if let Some(Value::Static(JsValue::Object(ref props))) = self.env.lookup("myGlobal", self.allocator) {
            if let Some(Value::Static(JsValue::Array { elements, .. })) = props.get("listeners") {
                if !elements.is_empty() {
                    // Add a comment separator
                    // (We can't add comments easily, so we'll just add the calls)

                    // For each listener, emit: document.addEventListener(type, handler)
                    for elem in elements {
                        if let Value::Static(JsValue::Object(listener)) = elem {
                            if let (Some(event_type), Some(handler)) = (listener.get("type"), listener.get("f")) {
                                let before = self.residual.len();
                                self.emit_captured_vars(handler);
                                if self.residual.len() > before {
                                    for stmt in self.residual.iter().skip(before) {
                                        captured_stmts.push(stmt.clone_in(self.allocator));
                                    }
                                }
                                // Build: document.addEventListener(type, handler)
                                let doc_ident = self.ast.expression_identifier(SPAN, "document");
                                let add_listener = IdentifierName { span: SPAN, name: Atom::from("addEventListener") };
                                let callee = Expression::StaticMemberExpression(
                                    self.ast.alloc_static_member_expression(SPAN, doc_ident, add_listener, false)
                                );

                                let type_expr = self.value_to_expr(event_type);
                                let handler_expr = self.value_to_expr(handler);

                                let args = self.ast.vec_from_iter([
                                    Argument::from(type_expr),
                                    Argument::from(handler_expr),
                                ].into_iter());

                                let call = self.ast.expression_call(
                                    SPAN, callee, None::<TSTypeParameterInstantiation>, args, false
                                );
                                let stmt = self.ast.statement_expression(SPAN, call);
                                all_stmts.push(stmt);
                            }
                        }
                    }
                }
            }
        }

        // Also check for a plain `listeners` array at top-level
        if let Some(Value::Static(JsValue::Array { elements, .. })) = self.env.lookup("listeners", self.allocator) {
            if !elements.is_empty() {
                // For each listener, emit: document.addEventListener(type, handler)
                // or if the element is just a closure, emit: listeners.push(handler)
                for elem in elements {
                    match elem {
                        Value::Static(JsValue::Object(listener)) => {
                            // Object with type/f structure (from document.addEventListener mock)
                            if let (Some(event_type), Some(handler)) = (listener.get("type"), listener.get("f")) {
                                let before = self.residual.len();
                                self.emit_captured_vars(handler);
                                if self.residual.len() > before {
                                    for stmt in self.residual.iter().skip(before) {
                                        captured_stmts.push(stmt.clone_in(self.allocator));
                                    }
                                }
                                let doc_ident = self.ast.expression_identifier(SPAN, "document");
                                let add_listener = IdentifierName { span: SPAN, name: Atom::from("addEventListener") };
                                let callee = Expression::StaticMemberExpression(
                                    self.ast.alloc_static_member_expression(SPAN, doc_ident, add_listener, false)
                                );

                                let type_expr = self.value_to_expr(event_type);
                                let handler_expr = self.value_to_expr(handler);

                                let args = self.ast.vec_from_iter([
                                    Argument::from(type_expr),
                                    Argument::from(handler_expr),
                                ].into_iter());

                                let call = self.ast.expression_call(
                                    SPAN, callee, None::<TSTypeParameterInstantiation>, args, false
                                );
                                let stmt = self.ast.statement_expression(SPAN, call);
                                all_stmts.push(stmt);
                            }
                        }
                        Value::Closure { .. } => {
                            // Plain closure pushed to listeners - emit listeners.push(handler)
                            let listeners_ident = self.ast.expression_identifier(SPAN, "listeners");
                            let push_name = IdentifierName { span: SPAN, name: Atom::from("push") };
                            let callee = Expression::StaticMemberExpression(
                                self.ast.alloc_static_member_expression(SPAN, listeners_ident, push_name, false)
                            );

                            let before = self.residual.len();
                            self.emit_captured_vars(&elem);
                            if self.residual.len() > before {
                                for stmt in self.residual.iter().skip(before) {
                                    captured_stmts.push(stmt.clone_in(self.allocator));
                                }
                            }
                            let handler_expr = self.value_to_expr(&elem);
                            let args = self.ast.vec_from_iter([
                                Argument::from(handler_expr),
                            ].into_iter());

                            let call = self.ast.expression_call(
                                SPAN, callee, None::<TSTypeParameterInstantiation>, args, false
                            );
                            let stmt = self.ast.statement_expression(SPAN, call);
                            all_stmts.push(stmt);
                        }
                        _ => {}
                    }
                }
            }
        }

        let mut frozen_vars: HashSet<String> = HashSet::new();
        for stmt in &captured_stmts {
            self.collect_hoisted_vars_stmt(stmt, &mut frozen_vars);
        }
        if !frozen_vars.is_empty() {
            all_stmts.retain(|stmt| {
                if let Statement::ExpressionStatement(expr_stmt) = stmt {
                    if let Expression::CallExpression(call) = &expr_stmt.expression {
                        let maybe_body: Option<&FunctionBody<'a>> = match &call.callee {
                            Expression::FunctionExpression(func) => func.body.as_ref().map(|v| &**v),
                            Expression::ArrowFunctionExpression(func) => Some(&func.body),
                            Expression::ParenthesizedExpression(paren) => {
                                if let Expression::FunctionExpression(func) = &paren.expression {
                                    func.body.as_ref().map(|v| &**v)
                                } else if let Expression::ArrowFunctionExpression(func) = &paren.expression {
                                    Some(&func.body)
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                        if let Some(body) = maybe_body {
                            let mut local_vars: HashSet<String> = HashSet::new();
                            let mut free_vars: HashSet<String> = HashSet::new();
                            for s in &body.statements {
                                self.collect_hoisted_vars_stmt(s, &mut local_vars);
                            }
                            for s in &body.statements {
                                self.collect_free_vars_stmt(s, &mut free_vars);
                            }
                            for local in &local_vars {
                                free_vars.remove(local);
                            }
                            if !free_vars.is_empty() && free_vars.is_subset(&frozen_vars) {
                                return false;
                            }
                        }
                    }
                }
                true
            });
        }

        // Ensure any free variables in the residual are declared or assigned
        let mut free_vars: HashSet<String> = HashSet::new();
        let mut declared_vars: HashSet<String> = HashSet::new();
        let mut assigned_vars: HashSet<String> = HashSet::new();

        for stmt in &all_stmts {
            self.collect_free_vars_stmt(stmt, &mut free_vars);
            self.collect_hoisted_vars_stmt(stmt, &mut declared_vars);
            self.collect_assigned_vars_stmt(stmt, &mut assigned_vars);
        }
        for stmt in &captured_stmts {
            self.collect_hoisted_vars_stmt(stmt, &mut declared_vars);
        }

        let mut missing_vars: Vec<String> = free_vars.into_iter()
            .filter(|name| !declared_vars.contains(name) && !assigned_vars.contains(name))
            .collect();
        missing_vars.sort();

        let mut array_hints: HashSet<String> = HashSet::new();
        let mut number_hints: HashSet<String> = HashSet::new();
        for stmt in &all_stmts {
            self.collect_usage_hints_stmt(stmt, &mut array_hints, &mut number_hints);
        }

        let mut missing_stmts: Vec<Statement<'a>> = Vec::new();
        for name in missing_vars {
            if matches!(
                name.as_str(),
                "Date"
                    | "String"
                    | "Uint8Array"
                    | "Array"
                    | "Object"
                    | "Number"
                    | "Boolean"
                    | "Math"
                    | "JSON"
                    | "console"
                    | "arguments"
                    | "undefined"
            ) {
                continue;
            }
            let init = if let Some(value) = self.env.lookup(&name, self.allocator) {
                match &value {
                    Value::Static(js) => Some(self.js_value_to_expr(js)),
                    Value::Closure { .. } => Some(self.value_to_expr(&value)),
                    Value::Dynamic(_) => None,
                }
            } else if array_hints.contains(&name) {
                Some(self.ast.expression_array(SPAN, self.ast.vec()))
            } else if number_hints.contains(&name) {
                Some(self.ast.expression_numeric_literal(SPAN, 0.0, None, NumberBase::Decimal))
            } else {
                None
            };
            let name_alloc: &'a str = self.alloc_str(&name);
            let stmt = self.build_var_decl_stmt(name_alloc, init);
            missing_stmts.push(stmt);
        }

        if !captured_stmts.is_empty() || !missing_stmts.is_empty() {
            let mut with_prefix = Vec::new();
            with_prefix.extend(captured_stmts.into_iter());
            with_prefix.extend(missing_stmts.into_iter());
            with_prefix.extend(all_stmts.into_iter());
            all_stmts = with_prefix;
        }

        if all_stmts.is_empty() {
            "// No residual code - fully evaluated!".to_string()
        } else {
            // Build a program from all statements and use codegen
            let stmts: oxc_allocator::Vec<'a, Statement<'a>> = self.ast.vec_from_iter(
                all_stmts.into_iter()
            );
            let program = self.ast.program(
                SPAN,
                SourceType::mjs(),
                "",  // source_text
                self.ast.vec(),  // comments
                None,  // hashbang
                self.ast.vec(),  // directives
                stmts,  // body
            );
            let codegen = Codegen::new().build(&program);
            codegen.code
        }
    }
}
