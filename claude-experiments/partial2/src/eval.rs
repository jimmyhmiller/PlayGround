use crate::env::{Env, Value, TypedArrayKind, SharedBuffer, SharedArray, SharedObject};
use crate::residual::{value_to_js, Binding, EvalTrace, StmtAction};
use oxc_ast::ast::*;
use oxc_codegen::{Codegen, Context, Gen};
use std::collections::HashMap;

/// Helper to create a dynamic value with its residual expression
fn dynamic(residual: String) -> Value {
    Value::Dynamic(residual)
}

/// Helper to get residual for any value
fn residual_of(value: &Value) -> Result<String, String> {
    value_to_js(value)
}

/// Escape a string for use in JSON
fn escape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result
}

/// Read a value from a DataView buffer
/// Returns None if the read would be out of bounds
fn read_dataview_value(buffer: &SharedBuffer, offset: usize, method_name: &str, little_endian: bool) -> Option<Value> {
    let bytes = buffer.to_vec();

    match method_name {
        "getInt8" => {
            if offset >= bytes.len() {
                return None;
            }
            Some(Value::Number(bytes[offset] as i8 as f64))
        }
        "getUint8" => {
            if offset >= bytes.len() {
                return None;
            }
            Some(Value::Number(bytes[offset] as f64))
        }
        "getInt16" => {
            if offset + 2 > bytes.len() {
                return None;
            }
            let b = &bytes[offset..offset + 2];
            let val = if little_endian {
                i16::from_le_bytes([b[0], b[1]])
            } else {
                i16::from_be_bytes([b[0], b[1]])
            };
            Some(Value::Number(val as f64))
        }
        "getUint16" => {
            if offset + 2 > bytes.len() {
                return None;
            }
            let b = &bytes[offset..offset + 2];
            let val = if little_endian {
                u16::from_le_bytes([b[0], b[1]])
            } else {
                u16::from_be_bytes([b[0], b[1]])
            };
            Some(Value::Number(val as f64))
        }
        "getInt32" => {
            if offset + 4 > bytes.len() {
                return None;
            }
            let b = &bytes[offset..offset + 4];
            let val = if little_endian {
                i32::from_le_bytes([b[0], b[1], b[2], b[3]])
            } else {
                i32::from_be_bytes([b[0], b[1], b[2], b[3]])
            };
            Some(Value::Number(val as f64))
        }
        "getUint32" => {
            if offset + 4 > bytes.len() {
                return None;
            }
            let b = &bytes[offset..offset + 4];
            let val = if little_endian {
                u32::from_le_bytes([b[0], b[1], b[2], b[3]])
            } else {
                u32::from_be_bytes([b[0], b[1], b[2], b[3]])
            };
            Some(Value::Number(val as f64))
        }
        "getFloat32" => {
            if offset + 4 > bytes.len() {
                return None;
            }
            let b = &bytes[offset..offset + 4];
            let val = if little_endian {
                f32::from_le_bytes([b[0], b[1], b[2], b[3]])
            } else {
                f32::from_be_bytes([b[0], b[1], b[2], b[3]])
            };
            Some(Value::Number(val as f64))
        }
        "getFloat64" => {
            if offset + 8 > bytes.len() {
                return None;
            }
            let b = &bytes[offset..offset + 8];
            let val = if little_endian {
                f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            } else {
                f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            };
            Some(Value::Number(val))
        }
        _ => None,
    }
}

/// Result of evaluating a statement
#[derive(Debug, Clone)]
pub enum StmtResult {
    /// Statement fully evaluated, no residual needed
    Consumed,
    /// Continue normal flow (no return/break)
    Continue,
    /// Return statement with value
    Return(Value),
    /// Break out of loop
    Break,
    /// Continue to next loop iteration
    ContinueLoop,
    /// Throw an exception with a value
    Throw(Value),
    /// Residual was emitted - containing loop should emit remaining loop and exit
    ResidualEmitted,
}

// ============================================================================
// Work Item Enums for Iterative Evaluation
// ============================================================================

/// Information about an assignment target for deferred assignment
#[derive(Debug, Clone)]
enum AssignTargetInfo {
    Identifier(String),
    StaticMember { object: String, property: String },
    ComputedMember { base: String, indices: Vec<Value> },
}

/// Work items for expression evaluation
#[derive(Debug, Clone)]
enum ExprWork<'a> {
    /// Evaluate an expression and push result onto value stack
    Eval(&'a Expression<'a>),
    /// Evaluate an identifier lazily (deferred lookup)
    EvalIdentifier { name: String },
    /// Apply a binary operator to top 2 values on stack
    ApplyBinary { operator: BinaryOperator },
    /// Apply a unary operator to top value on stack
    ApplyUnary { operator: UnaryOperator },
    /// Evaluate right side of logical expression if left didn't short-circuit
    LogicalRight { right: &'a Expression<'a>, operator: LogicalOperator, left_residual: Option<String> },
    /// Finish a logical expression by combining left and right residuals
    LogicalFinish { operator: LogicalOperator, left_residual: String },
    /// Handle conditional expression after test is evaluated
    ConditionalBranch { consequent: &'a Expression<'a>, alternate: &'a Expression<'a> },
    /// Collect array elements (remaining count, total count for building)
    ArrayCollect { remaining: usize },
    /// Collect object properties
    ObjectCollect { keys: Vec<String>, remaining: usize },
    /// Apply a function call after args are evaluated
    CallApply { func_name: String, arg_count: usize, call_expr: &'a CallExpression<'a> },
    /// Apply a method call after object and args are evaluated
    /// base_name is the root variable (e.g., "myGlobal")
    /// path is the property chain (e.g., ["listeners"] for myGlobal.listeners.push())
    MethodCallApply { method_name: String, base_name: String, path: Vec<String>, arg_count: usize, call_expr: &'a CallExpression<'a> },
    /// Static member access after object is evaluated
    MemberAccess { property: String },
    /// Computed member access after object and index are evaluated
    ComputedMemberApply,
    /// Apply a call where the callee is a computed member expression (e.g., funcs[0]())
    /// The callee value and args are on the stack
    ComputedCallApply { arg_count: usize, call_expr: &'a CallExpression<'a> },
    /// Apply assignment after value is evaluated
    AssignmentApply { target: AssignTargetInfo, operator: AssignmentOperator, original_expr: &'a AssignmentExpression<'a> },
    /// Apply update expression (++/--)
    UpdateApply { name: String, operator: UpdateOperator, prefix: bool },
}

/// Work items for statement evaluation
#[derive(Debug)]
#[allow(dead_code)]
enum StmtWork<'a> {
    /// Evaluate a statement
    Eval(&'a Statement<'a>),
    /// Continue a block after current statement completes
    BlockContinue { stmts: &'a [Statement<'a>], idx: usize },
    /// Pop a scope when leaving a block
    PopScope,
    /// Handle if branch after condition is evaluated (with condition value on stack)
    IfBranch { consequent: &'a Statement<'a>, alternate: Option<&'a Statement<'a>>, original_stmt: &'a Statement<'a> },
    /// Check while condition and potentially execute body
    WhileCheck { while_stmt: &'a WhileStatement<'a>, iterations: usize, original_stmt: &'a Statement<'a> },
    /// After while body completes, loop back to check
    WhileBodyDone { while_stmt: &'a WhileStatement<'a>, iterations: usize, original_stmt: &'a Statement<'a> },
    /// Decrement depth after while completes
    WhileExit,
    /// Handle switch case matching
    SwitchCase { switch_stmt: &'a SwitchStatement<'a>, discriminant: Value, case_idx: usize, matched: bool, fell_through: bool },
    /// For loop: check condition
    ForCheck { for_stmt: &'a ForStatement<'a>, iterations: usize, original_stmt: &'a Statement<'a> },
    /// For loop: execute update then check again
    ForUpdate { for_stmt: &'a ForStatement<'a>, iterations: usize, original_stmt: &'a Statement<'a> },
    /// Execute function body statements iteratively
    FunctionCallBody {
        func: &'a Function<'a>,
        stmt_idx: usize,
        func_name: String,
        saved_residual: Vec<String>,
        arg_values: Vec<Value>,
    },
    /// Finish function call, restore state
    FunctionCallFinish {
        func_name: String,
        saved_residual: Vec<String>,
        func: &'a Function<'a>,
    },
    /// Handle return value extraction
    ReturnValue,
    /// Try block evaluation
    TryBlock {
        try_stmt: &'a oxc_ast::ast::TryStatement<'a>,
        original_stmt: &'a Statement<'a>,
        saved_residual: Vec<String>,
        stmt_idx: usize,
    },
    /// Catch block evaluation
    CatchBlock {
        try_stmt: &'a oxc_ast::ast::TryStatement<'a>,
        original_stmt: &'a Statement<'a>,
        saved_residual: Vec<String>,
        stmt_idx: usize,
        thrown_value: Value,
    },
}

/// Unified work item for combined expression/statement evaluation
/// This allows function calls to be handled iteratively without Rust stack recursion
#[derive(Debug)]
enum UnifiedWork<'a> {
    /// Expression work item
    Expr(ExprWork<'a>),
    /// Statement work item (unused for now, keeping for future)
    Stmt(StmtWork<'a>),
    /// Start a function call - saves expression state and sets up function scope
    FunctionCallStart {
        func: &'a Function<'a>,
        func_name: String,
        arg_values: Vec<Value>,
    },
    /// Complete a function call - restores expression state and pushes return value
    FunctionCallComplete {
        func_name: String,
        func: &'a Function<'a>,
        saved_expr_work: Vec<ExprWork<'a>>,
        saved_values: Vec<Value>,
        saved_residual: Vec<String>,
    },
    /// Evaluate a sequence of statements (function body)
    EvalStatements {
        stmts: &'a [Statement<'a>],
        idx: usize,
    },
    // ============ Statement continuation work items ============
    // These allow statement evaluation to be trampolined through the work stack

    /// Continue return statement after expression is evaluated (value on stack)
    ReturnContinue,
    /// Continue expression statement after expression is evaluated (value on stack)
    ExprStmtContinue,
    /// Continue variable declaration after initializer is evaluated (value on stack)
    VarDeclContinue {
        name: String,
        kind: VariableDeclarationKind,
    },
    /// Continue if statement after condition is evaluated (value on stack)
    IfCondContinue {
        if_stmt: &'a IfStatement<'a>,
        original_stmt: &'a Statement<'a>,
    },
    /// Continue while loop after condition is evaluated (value on stack)
    WhileCondContinue {
        while_stmt: &'a WhileStatement<'a>,
        iterations: usize,
        original_stmt: &'a Statement<'a>,
    },
    /// Pop scope after block completes
    PopScope,
    /// Decrement depth after while completes
    WhileDepthDec,
    /// Continue switch statement after discriminant is evaluated (value on stack)
    SwitchDiscriminantContinue {
        switch_stmt: &'a SwitchStatement<'a>,
    },
    /// Continue switch case evaluation
    SwitchCaseContinue {
        switch_stmt: &'a SwitchStatement<'a>,
        discriminant: Value,
        case_idx: usize,
        matched: bool,
        fell_through: bool,
    },
    /// Process a single switch case test (value on stack after test evaluation)
    SwitchCaseTestContinue {
        switch_stmt: &'a SwitchStatement<'a>,
        discriminant: Value,
        case_idx: usize,
        matched: bool,
        fell_through: bool,
    },
    /// Continue after break statement
    BreakContinue,
    /// Marker for switch statement end - used to track switch context for break handling
    SwitchEnd,
}

/// Result from unified evaluation - either a value (for expression) or control flow (for statement)
#[derive(Debug, Clone)]
enum UnifiedResult {
    Value(Value),
    StmtResult(StmtResult),
}

/// Event types for execution tracing
#[derive(Debug, Clone)]
pub enum TraceEvent {
    /// A variable was defined
    BindingCreated {
        name: String,
        value_repr: String,
        is_static: bool,
        cause: Option<String>,  // What made it dynamic, if dynamic
    },
    /// A variable was updated
    BindingUpdated {
        name: String,
        old_repr: String,
        new_repr: String,
        was_static: bool,
        is_static: bool,
    },
    /// Entered a function
    FunctionEnter {
        name: String,
        args: Vec<(String, bool)>,  // (arg_repr, is_static)
    },
    /// Exited a function
    FunctionExit {
        name: String,
        result_repr: String,
        is_static: bool,
    },
    /// Started a loop iteration
    LoopIteration {
        loop_type: String,  // "while", "for"
        iteration: usize,
        condition_repr: String,
        condition_static: bool,
    },
    /// A value became dynamic due to an operation
    BecameDynamic {
        expr: String,
        reason: String,
    },
    /// Evaluation bailed out to residual
    BailedOut {
        reason: String,
        context: String,
    },
}

/// A timestamped trace entry
#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub seq: usize,
    pub depth: usize,  // call stack depth
    pub context: Vec<String>,  // current call stack
    pub event: TraceEvent,
}

/// Execution trace recorder
#[derive(Debug, Clone, Default)]
pub struct ExecutionTrace {
    pub entries: Vec<TraceEntry>,
    seq_counter: usize,
}

impl ExecutionTrace {
    pub fn new() -> Self {
        ExecutionTrace {
            entries: Vec::new(),
            seq_counter: 0,
        }
    }

    pub fn record(&mut self, depth: usize, context: Vec<String>, event: TraceEvent) {
        self.entries.push(TraceEntry {
            seq: self.seq_counter,
            depth,
            context,
            event,
        });
        self.seq_counter += 1;
    }

    pub fn to_json(&self) -> String {
        let mut output = String::from("{\n  \"events\": [\n");

        for (i, entry) in self.entries.iter().enumerate() {
            if i > 0 {
                output.push_str(",\n");
            }
            output.push_str(&format!("    {}", self.entry_to_json(entry)));
        }

        output.push_str("\n  ]\n}\n");
        output
    }

    fn entry_to_json(&self, entry: &TraceEntry) -> String {
        let event_json = match &entry.event {
            TraceEvent::BindingCreated { name, value_repr, is_static, cause } => {
                format!(
                    "{{\"type\": \"binding_created\", \"name\": \"{}\", \"value\": \"{}\", \"is_static\": {}, \"cause\": {}}}",
                    escape_json_string(name),
                    escape_json_string(&truncate_str(value_repr, 100)),
                    is_static,
                    cause.as_ref().map(|c| format!("\"{}\"", escape_json_string(c))).unwrap_or("null".to_string())
                )
            }
            TraceEvent::BindingUpdated { name, old_repr, new_repr, was_static, is_static } => {
                format!(
                    "{{\"type\": \"binding_updated\", \"name\": \"{}\", \"old\": \"{}\", \"new\": \"{}\", \"was_static\": {}, \"is_static\": {}}}",
                    escape_json_string(name),
                    escape_json_string(&truncate_str(old_repr, 50)),
                    escape_json_string(&truncate_str(new_repr, 50)),
                    was_static,
                    is_static
                )
            }
            TraceEvent::FunctionEnter { name, args } => {
                let args_json: Vec<String> = args.iter()
                    .map(|(repr, is_static)| format!("[\"{}\", {}]", escape_json_string(&truncate_str(repr, 30)), is_static))
                    .collect();
                format!(
                    "{{\"type\": \"function_enter\", \"name\": \"{}\", \"args\": [{}]}}",
                    escape_json_string(name),
                    args_json.join(", ")
                )
            }
            TraceEvent::FunctionExit { name, result_repr, is_static } => {
                format!(
                    "{{\"type\": \"function_exit\", \"name\": \"{}\", \"result\": \"{}\", \"is_static\": {}}}",
                    escape_json_string(name),
                    escape_json_string(&truncate_str(result_repr, 50)),
                    is_static
                )
            }
            TraceEvent::LoopIteration { loop_type, iteration, condition_repr, condition_static } => {
                format!(
                    "{{\"type\": \"loop_iteration\", \"loop_type\": \"{}\", \"iteration\": {}, \"condition\": \"{}\", \"condition_static\": {}}}",
                    loop_type,
                    iteration,
                    escape_json_string(&truncate_str(condition_repr, 50)),
                    condition_static
                )
            }
            TraceEvent::BecameDynamic { expr, reason } => {
                format!(
                    "{{\"type\": \"became_dynamic\", \"expr\": \"{}\", \"reason\": \"{}\"}}",
                    escape_json_string(&truncate_str(expr, 50)),
                    escape_json_string(reason)
                )
            }
            TraceEvent::BailedOut { reason, context } => {
                format!(
                    "{{\"type\": \"bailed_out\", \"reason\": \"{}\", \"context\": \"{}\"}}",
                    escape_json_string(reason),
                    escape_json_string(context)
                )
            }
        };

        format!(
            "{{\"seq\": {}, \"depth\": {}, \"stack\": {:?}, \"event\": {}}}",
            entry.seq,
            entry.depth,
            entry.context,
            event_json
        )
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Evaluator for partial evaluation.
/// Interprets the AST and tracks what values are statically known.
pub struct Evaluator<'a> {
    pub env: Env,
    /// Store function declarations by name for later invocation
    functions: HashMap<String, &'a Function<'a>>,
    /// Track what happens during evaluation for residual generation
    pub trace: EvalTrace,
    /// Top-level variable names in declaration order, with original expression source
    /// (name, original_expr_source or None if no initializer)
    top_level_vars: Vec<(String, Option<String>)>,
    /// Accumulated residual statements during evaluation
    residual_stmts: Vec<String>,
    /// Maximum iterations for loop unrolling (to prevent infinite loops)
    max_iterations: usize,
    /// Current nesting depth for while loops (to prevent stack overflow)
    depth: usize,
    /// Call stack tracking which functions are currently being evaluated
    /// Used to detect recursion and prevent infinite evaluation
    call_stack: Vec<String>,
    /// Maximum recursion depth before bailing out to residual
    pub max_recursion_depth: usize,
    /// Execution trace for debugging
    pub execution_trace: ExecutionTrace,
    /// Whether to record execution trace (can be expensive)
    pub trace_enabled: bool,
    /// Counter for generating unique names for anonymous functions
    anon_func_counter: usize,
}

impl<'a> Evaluator<'a> {
    pub fn new() -> Self {
        Evaluator {
            env: Env::new(),
            functions: HashMap::new(),
            trace: EvalTrace::new(),
            top_level_vars: Vec::new(),
            residual_stmts: Vec::new(),
            max_iterations: 100000,
            depth: 0,
            call_stack: Vec::new(),
            max_recursion_depth: 5000,
            execution_trace: ExecutionTrace::new(),
            trace_enabled: false,
            anon_func_counter: 0,
        }
    }

    /// Record a trace event
    fn trace_event(&mut self, event: TraceEvent) {
        if self.trace_enabled {
            self.execution_trace.record(
                self.call_stack.len(),
                self.call_stack.clone(),
                event,
            );
        }
    }

    /// Helper to get a short representation of a value
    fn value_repr(&self, value: &Value) -> String {
        self.value_to_source(value)
    }

    /// Emit a residual statement to the accumulator
    fn emit_residual(&mut self, stmt: String) {
        self.residual_stmts.push(stmt);
    }

    /// Output all variable bindings as JSON for debugging
    pub fn debug_bindings_json(&self) -> String {
        let mut output = String::from("{\n  \"bindings\": [\n");

        // Get all bindings from all scopes
        let bindings = self.env.all_bindings();
        let mut first = true;

        for (scope_idx, name, value) in bindings {
            if !first {
                output.push_str(",\n");
            }
            first = false;

            let binding_time = if value.is_static() { "static" } else { "dynamic" };
            let value_repr = self.value_to_json(&value);

            output.push_str(&format!(
                "    {{\n      \"name\": \"{}\",\n      \"scope\": {},\n      \"binding_time\": \"{}\",\n      \"value\": {}\n    }}",
                escape_json_string(&name),
                scope_idx,
                binding_time,
                value_repr
            ));
        }

        output.push_str("\n  ],\n  \"functions\": [\n");

        // List registered functions
        let mut first = true;
        for name in self.functions.keys() {
            if !first {
                output.push_str(",\n");
            }
            first = false;
            output.push_str(&format!("    \"{}\"", escape_json_string(name)));
        }

        output.push_str("\n  ],\n  \"residual_statements\": ");
        output.push_str(&format!("{}", self.residual_stmts.len()));
        output.push_str("\n}\n");

        output
    }

    fn value_to_json(&self, value: &Value) -> String {
        match value {
            Value::Number(n) => {
                if n.is_nan() {
                    "\"NaN\"".to_string()
                } else if n.is_infinite() {
                    if *n > 0.0 { "\"Infinity\"".to_string() } else { "\"-Infinity\"".to_string() }
                } else {
                    format!("{}", n)
                }
            }
            Value::String(s) => format!("\"{}\"", escape_json_string(s)),
            Value::Bool(b) => format!("{}", b),
            Value::Undefined => "\"undefined\"".to_string(),
            Value::Null => "null".to_string(),
            Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| self.value_to_json(&v)).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Object(obj) => {
                let items: Vec<String> = obj.iter()
                    .map(|(k, v)| format!("\"{}\": {}", escape_json_string(&k), self.value_to_json(&v)))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
            Value::Closure { params, .. } => {
                format!("{{\"type\": \"closure\", \"params\": {:?}}}", params)
            }
            Value::ArrayBuffer { buffer } => {
                format!("{{\"type\": \"ArrayBuffer\", \"byteLength\": {}}}", buffer.len())
            }
            Value::TypedArray { kind, length, .. } => {
                format!("{{\"type\": \"{}\", \"length\": {}}}", kind.name(), length)
            }
            Value::DataView { byte_length, .. } => {
                format!("{{\"type\": \"DataView\", \"byteLength\": {}}}", byte_length)
            }
            Value::TextDecoder { encoding } => {
                format!("{{\"type\": \"TextDecoder\", \"encoding\": \"{}\"}}", encoding)
            }
            Value::Dynamic(residual) => {
                format!("{{\"dynamic\": \"{}\"}}", escape_json_string(residual))
            }
        }
    }

    /// Output bindings as pretty-printed annotated source code
    /// Static values are shown in green, dynamic values in red
    pub fn debug_pretty_print(&self) -> String {
        // ANSI color codes
        const GREEN: &str = "\x1b[32m";
        const RED: &str = "\x1b[31m";
        const YELLOW: &str = "\x1b[33m";
        const CYAN: &str = "\x1b[36m";
        const DIM: &str = "\x1b[2m";
        const RESET: &str = "\x1b[0m";
        const BOLD: &str = "\x1b[1m";

        let mut output = String::new();

        // Header
        output.push_str(&format!("{}{}═══ Binding Analysis ═══{}\n\n", BOLD, CYAN, RESET));

        // Legend
        output.push_str(&format!("{}Legend:{} {}■ Static{} | {}■ Dynamic{}\n\n",
            DIM, RESET, GREEN, RESET, RED, RESET));

        // Get all bindings
        let bindings = self.env.all_bindings();

        // Group by scope
        let mut by_scope: std::collections::BTreeMap<usize, Vec<(String, Value)>> =
            std::collections::BTreeMap::new();
        for (scope_idx, name, value) in bindings {
            by_scope.entry(scope_idx).or_default().push((name, value));
        }

        // Count stats
        let mut static_count = 0;
        let mut dynamic_count = 0;

        for (scope_idx, vars) in &by_scope {
            output.push_str(&format!("{}{}── Scope {} ──{}\n", DIM, CYAN, scope_idx, RESET));

            for (name, value) in vars {
                let (color, status, value_str) = if value.is_static() {
                    static_count += 1;
                    (GREEN, "static", self.value_to_source(value))
                } else {
                    dynamic_count += 1;
                    let residual = match value {
                        Value::Dynamic(r) => r.clone(),
                        _ => "???".to_string(),
                    };
                    (RED, "dynamic", residual)
                };

                // Truncate long values for display
                let display_value = if value_str.len() > 80 {
                    format!("{}...", &value_str[..77])
                } else {
                    value_str.clone()
                };

                output.push_str(&format!(
                    "{}let {}{} = {}{}{}; {}// {}{}\n",
                    color, name, RESET,
                    color, display_value, RESET,
                    DIM, status, RESET
                ));

                // If it's a long dynamic value, show a preview of what it contains
                if value.is_dynamic() && value_str.len() > 80 {
                    // Count some interesting things in the residual
                    let func_count = value_str.matches("function").count();
                    let var_count = value_str.matches("var ").count();
                    output.push_str(&format!(
                        "  {}  └─ ({} chars, ~{} functions, ~{} vars){}\n",
                        DIM, value_str.len(), func_count, var_count, RESET
                    ));
                }
            }
            output.push_str("\n");
        }

        // Functions
        if !self.functions.is_empty() {
            output.push_str(&format!("{}{}── Functions ──{}\n", DIM, CYAN, RESET));
            for name in self.functions.keys() {
                output.push_str(&format!("{}function {}() {{ ... }}{}\n", YELLOW, name, RESET));
            }
            output.push_str("\n");
        }

        // Summary
        output.push_str(&format!("{}{}── Summary ──{}\n", DIM, CYAN, RESET));
        output.push_str(&format!(
            "{}Static:{} {}  {}Dynamic:{} {}  {}Functions:{} {}\n",
            GREEN, RESET, static_count,
            RED, RESET, dynamic_count,
            YELLOW, RESET, self.functions.len()
        ));

        if dynamic_count > 0 {
            output.push_str(&format!(
                "\n{}Tip: Dynamic values could not be fully evaluated at compile time.{}\n",
                DIM, RESET
            ));
        }

        output
    }

    fn value_to_source(&self, value: &Value) -> String {
        match value {
            Value::Number(n) => {
                if n.is_nan() {
                    "NaN".to_string()
                } else if n.is_infinite() {
                    if *n > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
                } else if n.fract() == 0.0 && n.abs() < 1e15 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
            Value::Bool(b) => format!("{}", b),
            Value::Undefined => "undefined".to_string(),
            Value::Null => "null".to_string(),
            Value::Array(arr) => {
                if arr.len() > 5 {
                    let items: Vec<String> = arr.iter().take(3).map(|v| self.value_to_source(&v)).collect();
                    format!("[{}, ... ({} more)]", items.join(", "), arr.len() - 3)
                } else {
                    let items: Vec<String> = arr.iter().map(|v| self.value_to_source(&v)).collect();
                    format!("[{}]", items.join(", "))
                }
            }
            Value::Object(obj) => {
                if obj.is_empty() {
                    "{}".to_string()
                } else if obj.len() > 3 {
                    let items: Vec<String> = obj.iter().take(2)
                        .map(|(k, v)| format!("{}: {}", k, self.value_to_source(&v)))
                        .collect();
                    format!("{{ {}, ... ({} more) }}", items.join(", "), obj.len() - 2)
                } else {
                    let items: Vec<String> = obj.iter()
                        .map(|(k, v)| format!("{}: {}", k, self.value_to_source(&v)))
                        .collect();
                    format!("{{ {} }}", items.join(", "))
                }
            }
            Value::Closure { params, .. } => {
                format!("function({}) {{ ... }}", params.join(", "))
            }
            Value::ArrayBuffer { buffer } => {
                format!("new ArrayBuffer({})", buffer.len())
            }
            Value::TypedArray { kind, length, .. } => {
                format!("new {}({})", kind.name(), length)
            }
            Value::DataView { byte_length, .. } => {
                format!("new DataView(<buffer>, {}))", byte_length)
            }
            Value::TextDecoder { encoding } => {
                format!("new TextDecoder(\"{}\")", encoding)
            }
            Value::Dynamic(residual) => residual.clone(),
        }
    }

    /// Take accumulated residual statements, clearing the accumulator
    fn take_residual(&mut self) -> Vec<String> {
        std::mem::take(&mut self.residual_stmts)
    }

    /// Evaluate a program, returning the final environment state
    pub fn eval_program(&mut self, program: &'a Program<'a>) -> Result<(), String> {
        // First pass: collect function declarations (hoisting)
        // Also record their positions for later
        let mut func_positions: HashMap<String, (usize, String)> = HashMap::new();
        for (idx, stmt) in program.body.iter().enumerate() {
            if let Statement::FunctionDeclaration(func) = stmt {
                self.register_function(func)?;
                if let Some(id) = &func.id {
                    let name = id.name.to_string();
                    let source = emit_function(func);
                    func_positions.insert(name, (idx, source));
                }
            }
        }

        // Track which variables are declared at which statement index
        // We'll collect their FINAL values after all evaluation is done
        let mut var_decl_info: Vec<(usize, Vec<(String, Option<String>)>)> = Vec::new();

        // Second pass: execute statements and record actions
        for (idx, stmt) in program.body.iter().enumerate() {
            match stmt {
                Statement::FunctionDeclaration(func) => {
                    // Record function declaration - will be preserved or consumed based on usage
                    if let Some(id) = &func.id {
                        let name = id.name.to_string();
                        let source = emit_function(func);
                        self.trace.set_stmt_action(idx, StmtAction::PreserveFunction { name, source });
                    }
                }
                Statement::VariableDeclaration(decl) => {
                    // Clear residual before evaluating this statement
                    let _ = self.take_residual();

                    // Evaluate the declaration
                    self.eval_variable_declaration(decl, stmt)?;

                    // Check if any residual was emitted during evaluation
                    // (e.g., from inlined function calls)
                    let residual = self.take_residual();
                    let has_residual = !residual.is_empty();

                    if has_residual {
                        // Emit the residual code before the variable binding
                        self.trace.set_stmt_action(idx, StmtAction::Preserve(residual.join("\n")));
                    }

                    // Remember which variables were declared here, with their original expressions
                    let mut var_names = Vec::new();
                    for declarator in &decl.declarations {
                        if let BindingPatternKind::BindingIdentifier(id) = &declarator.id.kind {
                            let name = id.name.to_string();
                            let orig_expr = declarator.init.as_ref().map(|e| emit_expr(e));
                            var_names.push((name, orig_expr));
                        }
                    }
                    // Use a different index if we already used idx for residual
                    // This ensures the variable binding comes after the residual
                    let binding_idx = if has_residual { idx + 10000 } else { idx };
                    var_decl_info.push((binding_idx, var_names));
                }
                _ => {
                    // Clear residual before evaluating this statement
                    let _ = self.take_residual();

                    // Other statements - evaluate and record action
                    match self.eval_statement(stmt) {
                        Ok(_) => {
                            // Check if any residual was emitted during evaluation
                            let residual = self.take_residual();
                            if residual.is_empty() {
                                // Statement was fully consumed
                                self.trace.set_stmt_action(idx, StmtAction::Consumed);
                            } else {
                                // Statement produced residual code
                                self.trace.set_stmt_action(idx, StmtAction::Preserve(residual.join("\n")));
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }

        // Third pass: collect FINAL values for all variable declarations
        for (idx, var_names) in var_decl_info {
            let mut bindings = Vec::new();
            for (name, _orig_expr) in var_names {
                if let Some(value) = self.env.get(&name) {
                    if let Value::Dynamic(residual_expr) = &value {
                        // Use the dynamic value's residual expression
                        // This is the specialized expression, not the original
                        bindings.push((name, Binding::Dynamic(residual_expr.clone())));
                    } else {
                        // Use the final static value
                        bindings.push((name, Binding::Static(value)));
                    }
                }
            }
            self.trace.set_stmt_action(idx, StmtAction::EmitBindings(bindings));
        }

        Ok(())
    }

    fn register_function(&mut self, func: &'a Function<'a>) -> Result<(), String> {
        let name = func
            .id
            .as_ref()
            .map(|id| id.name.to_string())
            .ok_or("Anonymous function not supported")?;

        let params: Vec<String> = func
            .params
            .items
            .iter()
            .filter_map(|p| {
                if let BindingPatternKind::BindingIdentifier(id) = &p.pattern.kind {
                    Some(id.name.to_string())
                } else {
                    None
                }
            })
            .collect();

        // Store function reference for later
        self.functions.insert(name.clone(), func);

        // Emit the function source for closure value
        let source = emit_function(func);

        // Also define closure value in env
        let closure = Value::Closure {
            params,
            body_id: 0,
            env: self.env.capture(),
            source,
            name: Some(name.clone()),
        };
        self.env.define(&name, closure);
        Ok(())
    }

    fn eval_statement(&mut self, stmt: &'a Statement<'a>) -> Result<StmtResult, String> {
        // Use the iterative implementation to avoid stack overflow
        self.eval_statement_iterative(stmt)
    }

    /// Old recursive eval_statement - kept for reference, but eval_statement now uses iterative version
    #[allow(dead_code)]
    fn eval_statement_recursive(&mut self, stmt: &'a Statement<'a>) -> Result<StmtResult, String> {
        match stmt {
            Statement::VariableDeclaration(decl) => {
                self.eval_variable_declaration_inner(decl)?;
                Ok(StmtResult::Continue)
            }
            Statement::ExpressionStatement(expr_stmt) => {
                // Conservative approach: evaluate for tracking but preserve side effects
                let value = self.eval_expression(&expr_stmt.expression)?;

                // Determine if this expression has side effects that must be preserved
                let must_preserve = match &expr_stmt.expression {
                    // Assignments to properties or unknown globals must be preserved
                    Expression::AssignmentExpression(assign) => {
                        match &assign.left {
                            // Assignment to a known local variable we declared - safe to consume
                            AssignmentTarget::AssignmentTargetIdentifier(id) => {
                                let name = id.name.to_string();
                                // Only consume if we have this variable in our environment
                                // AND it was declared (not just assigned like a global)
                                !self.env.exists(&name)
                            }
                            // Computed member assignments - check if we successfully tracked it
                            AssignmentTarget::ComputedMemberExpression(_) => {
                                // If the result is static, we tracked it; if dynamic, we didn't
                                value.is_dynamic()
                            }
                            // Static member assignments - check if we successfully tracked it
                            AssignmentTarget::StaticMemberExpression(_) => {
                                // If the result is static, we tracked it; if dynamic, we didn't
                                value.is_dynamic()
                            }
                            // Other property assignments must always be preserved
                            _ => true,
                        }
                    }
                    // Method calls might have side effects
                    Expression::CallExpression(call) => {
                        // Check if this is a call to one of our functions
                        if let Expression::Identifier(id) = &call.callee {
                            let func_name = id.name.to_string();
                            // If it's our function and fully evaluated, don't preserve
                            // Also check if the variable holds a closure with a known function name
                            let is_known_function = self.functions.contains_key(&func_name)
                                || matches!(
                                    self.env.get(&func_name),
                                    Some(Value::Closure { name: Some(ref n), .. }) if self.functions.contains_key(n.as_str())
                                );
                            !is_known_function
                        } else if let Expression::StaticMemberExpression(member) = &call.callee {
                            // Array methods on tracked arrays are safe
                            let method = member.property.name.to_string();
                            if matches!(method.as_str(), "push" | "pop" | "shift" | "unshift") {
                                if let Expression::Identifier(id) = &member.object {
                                    // Check if array is in our environment
                                    self.env.get(&id.name.to_string()).is_none()
                                } else {
                                    true
                                }
                            } else {
                                true // Other method calls preserved
                            }
                        } else {
                            true // Preserve other calls
                        }
                    }
                    // Update expressions on known locals are safe
                    Expression::UpdateExpression(update) => {
                        match &update.argument {
                            SimpleAssignmentTarget::AssignmentTargetIdentifier(id) => {
                                !self.env.exists(&id.name.to_string())
                            }
                            _ => true,
                        }
                    }
                    // Other expressions typically don't have side effects
                    _ => false,
                };

                if must_preserve {
                    // Emit the statement as residual
                    // Use the evaluated value's residual representation if dynamic,
                    // otherwise use the original statement
                    if let Value::Dynamic(residual_expr) = &value {
                        self.emit_residual(residual_expr.clone());
                    } else {
                        self.emit_residual(emit_stmt(stmt));
                    }
                }

                Ok(StmtResult::Continue)
            }
            Statement::FunctionDeclaration(func) => {
                // Register the function in current scope.
                // For top-level, this was already done in hoisting pass (harmless to re-do).
                // For nested functions (in IIFEs, etc.), this is required.
                self.register_function(func)?;
                Ok(StmtResult::Continue)
            }
            Statement::BlockStatement(block) => {
                self.eval_block_statement(block)
            }
            Statement::ReturnStatement(ret) => {
                if let Some(arg) = &ret.argument {
                    let val = self.eval_expression(arg)?;
                    Ok(StmtResult::Return(val))
                } else {
                    Ok(StmtResult::Return(Value::Undefined))
                }
            }
            Statement::IfStatement(if_stmt) => {
                self.eval_if_statement(if_stmt, stmt)
            }
            Statement::WhileStatement(while_stmt) => {
                self.eval_while_statement(while_stmt, stmt)
            }
            Statement::SwitchStatement(switch_stmt) => {
                self.eval_switch_statement(switch_stmt, stmt)
            }
            Statement::ForStatement(for_stmt) => {
                self.eval_for_statement(for_stmt, stmt)
            }
            Statement::BreakStatement(_) => {
                Ok(StmtResult::Break)
            }
            Statement::ContinueStatement(_) => {
                Ok(StmtResult::ContinueLoop)
            }
            Statement::ThrowStatement(throw_stmt) => {
                let val = self.eval_expression(&throw_stmt.argument)?;
                Ok(StmtResult::Throw(val))
            }
            Statement::TryStatement(try_stmt) => {
                self.eval_try_statement(try_stmt, stmt)
            }
            _ => {
                // Unknown statement - emit as residual
                self.emit_residual(emit_stmt(stmt));
                Ok(StmtResult::Continue)
            }
        }
    }

    fn eval_block_statement(&mut self, block: &'a BlockStatement<'a>) -> Result<StmtResult, String> {
        self.env.push_scope();
        let mut result = StmtResult::Continue;
        for stmt in &block.body {
            result = self.eval_statement(stmt)?;
            match &result {
                StmtResult::Return(_) | StmtResult::Break | StmtResult::ContinueLoop | StmtResult::Throw(_) | StmtResult::ResidualEmitted => break,
                _ => {}
            }
        }
        self.env.pop_scope();
        Ok(result)
    }

    fn eval_if_statement(&mut self, if_stmt: &'a IfStatement<'a>, _original_stmt: &'a Statement<'a>) -> Result<StmtResult, String> {
        let cond = self.eval_expression(&if_stmt.test)?;
        match cond.is_truthy() {
            Some(true) => self.eval_statement(&if_stmt.consequent),
            Some(false) => {
                if let Some(alt) = &if_stmt.alternate {
                    self.eval_statement(alt)
                } else {
                    Ok(StmtResult::Continue)
                }
            }
            None => {
                // Dynamic condition - emit residual if with specialized branches
                let cond_residual = residual_of(&cond)?;

                // Save current residual state and evaluate consequent
                let saved_residual = self.take_residual();
                let saved_env = self.env.capture();

                // Specialize the consequent branch
                let _consequent_result = self.eval_statement(&if_stmt.consequent)?;
                let consequent_residual = self.take_residual();
                let consequent_body = if consequent_residual.is_empty() {
                    // No residual from branch - emit the specialized statement
                    self.specialize_statement(&if_stmt.consequent)
                } else {
                    consequent_residual.join("\n")
                };

                // Restore env and specialize alternate branch if present
                self.env = saved_env.capture();
                let alternate_body = if let Some(alt) = &if_stmt.alternate {
                    let _ = self.eval_statement(alt)?;
                    let alt_residual = self.take_residual();
                    if alt_residual.is_empty() {
                        Some(self.specialize_statement(alt))
                    } else {
                        Some(alt_residual.join("\n"))
                    }
                } else {
                    None
                };

                // Restore original residual state
                self.residual_stmts = saved_residual;

                // Emit the if statement with specialized branches
                let if_residual = if let Some(alt) = alternate_body {
                    format!("if ({}) {{\n{}\n}} else {{\n{}\n}}", cond_residual, consequent_body, alt)
                } else {
                    format!("if ({}) {{\n{}\n}}", cond_residual, consequent_body)
                };
                self.emit_residual(if_residual);

                // The if statement was emitted as residual
                Ok(StmtResult::Continue)
            }
        }
    }

    fn eval_while_statement(&mut self, while_stmt: &'a WhileStatement<'a>, _original_stmt: &'a Statement<'a>) -> Result<StmtResult, String> {
        // Track nesting depth using Evaluator field
        self.depth += 1;

        // Bail out if we're too deep - likely recursive function calls
        // Each while loop adds ~10-20 stack frames for eval_statement/eval_switch/etc
        // Use the configurable max_recursion_depth for this limit as well
        if self.depth > self.max_recursion_depth {
            self.depth -= 1;
            let remaining_loop = self.build_while_residual(while_stmt);
            self.emit_residual(remaining_loop);
            return Ok(StmtResult::ResidualEmitted);
        }

        let mut iterations = 0;

        loop {
            if iterations >= self.max_iterations {
                // Prevent infinite unrolling - emit rest of loop as residual
                let remaining_loop = self.build_while_residual(while_stmt);
                self.emit_residual(remaining_loop);
                self.depth -= 1;
                return Ok(StmtResult::Continue);
            }
            iterations += 1;

            let cond = self.eval_expression(&while_stmt.test)?;

            match cond.is_truthy() {
                Some(true) => {
                    // Condition is statically true - execute body
                    let body_result = self.eval_statement(&while_stmt.body)?;
                    match body_result {
                        StmtResult::Break => break,
                        StmtResult::Return(v) => {
                            self.depth -= 1;
                            return Ok(StmtResult::Return(v));
                        }
                        StmtResult::Throw(v) => {
                            self.depth -= 1;
                            return Ok(StmtResult::Throw(v));
                        }
                        StmtResult::ResidualEmitted => {
                            // Body emitted residual (e.g., hit dynamic switch)
                            // Emit the remaining while loop and exit
                            let remaining_loop = self.build_while_residual(while_stmt);
                            self.emit_residual(remaining_loop);
                            // Propagate ResidualEmitted so containing constructs know
                            self.depth -= 1;
                            return Ok(StmtResult::ResidualEmitted);
                        }
                        StmtResult::ContinueLoop => continue,
                        _ => continue,
                    }
                }
                Some(false) => {
                    // Condition is statically false - exit loop
                    break;
                }
                None => {
                    // Condition is dynamic - emit rest of loop as residual
                    let remaining_loop = self.build_while_residual(while_stmt);
                    self.emit_residual(remaining_loop);
                    self.depth -= 1;
                    return Ok(StmtResult::ResidualEmitted);
                }
            }
        }

        self.depth -= 1;
        Ok(StmtResult::Continue)
    }

    fn eval_for_statement(&mut self, for_stmt: &'a ForStatement<'a>, original_stmt: &'a Statement<'a>) -> Result<StmtResult, String> {
        // Execute the init part (once)
        if let Some(init) = &for_stmt.init {
            match init {
                ForStatementInit::VariableDeclaration(decl) => {
                    // Handle variable declaration init
                    for declarator in &decl.declarations {
                        if let BindingPatternKind::BindingIdentifier(id) = &declarator.id.kind {
                            let name = id.name.to_string();
                            let value = if let Some(init_expr) = &declarator.init {
                                self.eval_expression(init_expr)?
                            } else {
                                Value::Undefined
                            };
                            self.env.define(&name, value);
                        }
                    }
                }
                _ => {
                    // Expression init
                    self.eval_expression(init.to_expression())?;
                }
            }
        }

        let mut iterations = 0;

        loop {
            if iterations >= self.max_iterations {
                // Prevent infinite unrolling - emit rest as residual
                self.emit_residual(emit_stmt(original_stmt));
                return Ok(StmtResult::Continue);
            }
            iterations += 1;

            // Evaluate test condition (if present)
            let should_continue = if let Some(test) = &for_stmt.test {
                let cond = self.eval_expression(test)?;
                match cond.is_truthy() {
                    Some(true) => true,
                    Some(false) => false,
                    None => {
                        // Condition is dynamic - emit rest as residual
                        self.emit_residual(emit_stmt(original_stmt));
                        return Ok(StmtResult::ResidualEmitted);
                    }
                }
            } else {
                // No test means always true (for(;;))
                true
            };

            if !should_continue {
                break;
            }

            // Execute body
            let body_result = self.eval_statement(&for_stmt.body)?;
            match body_result {
                StmtResult::Break => break,
                StmtResult::Return(v) => {
                    return Ok(StmtResult::Return(v));
                }
                StmtResult::Throw(v) => return Ok(StmtResult::Throw(v)),
                StmtResult::ResidualEmitted => {
                    // Body emitted residual
                    self.emit_residual(emit_stmt(original_stmt));
                    return Ok(StmtResult::ResidualEmitted);
                }
                StmtResult::ContinueLoop => {
                    // Continue - skip to update expression
                }
                _ => {}
            }

            // Execute update expression (if present)
            if let Some(update) = &for_stmt.update {
                self.eval_expression(update)?;
            }
        }

        Ok(StmtResult::Continue)
    }

    fn eval_try_statement(&mut self, try_stmt: &'a oxc_ast::ast::TryStatement<'a>, original_stmt: &'a Statement<'a>) -> Result<StmtResult, String> {
        // Save current residual state in case we need to emit the whole try-catch
        let saved_residual = self.take_residual();

        // Try to evaluate the try block
        self.env.push_scope();
        let mut try_result = StmtResult::Continue;
        for stmt in &try_stmt.block.body {
            try_result = self.eval_statement(stmt)?;
            match &try_result {
                StmtResult::Return(_) | StmtResult::Break | StmtResult::Throw(_) => break,
                StmtResult::ResidualEmitted => {
                    // Try block emitted residual - emit whole try-catch as residual
                    self.env.pop_scope();
                    self.residual_stmts = saved_residual;
                    self.emit_residual(emit_stmt(original_stmt));
                    return Ok(StmtResult::ResidualEmitted);
                }
                _ => {}
            }
        }
        self.env.pop_scope();

        // Check if we got a throw
        if let StmtResult::Throw(thrown_value) = try_result {
            // We have a throw - execute catch block if present
            if let Some(handler) = &try_stmt.handler {
                self.env.push_scope();

                // Bind the exception to the catch parameter
                if let Some(param) = &handler.param {
                    if let BindingPatternKind::BindingIdentifier(id) = &param.pattern.kind {
                        self.env.define(&id.name.to_string(), thrown_value);
                    }
                }

                // Execute catch block
                let mut catch_result = StmtResult::Continue;
                for stmt in &handler.body.body {
                    catch_result = self.eval_statement(stmt)?;
                    match &catch_result {
                        StmtResult::Return(_) | StmtResult::Break | StmtResult::Throw(_) => break,
                        StmtResult::ResidualEmitted => {
                            // Catch block emitted residual - emit whole try-catch
                            self.env.pop_scope();
                            self.residual_stmts = saved_residual;
                            self.emit_residual(emit_stmt(original_stmt));
                            return Ok(StmtResult::ResidualEmitted);
                        }
                        _ => {}
                    }
                }
                self.env.pop_scope();

                // Restore residual and return catch result
                self.residual_stmts = saved_residual;

                // If catch block threw, propagate it
                if let StmtResult::Throw(_) = catch_result {
                    return Ok(catch_result);
                }
                // Otherwise continue normally
                return Ok(StmtResult::Continue);
            } else {
                // No catch block - propagate the throw
                self.residual_stmts = saved_residual;
                return Ok(StmtResult::Throw(thrown_value));
            }
        }

        // No throw - restore residual and return try result
        self.residual_stmts = saved_residual;

        // Propagate return/break from try block
        match try_result {
            StmtResult::Return(_) | StmtResult::Break => Ok(try_result),
            _ => Ok(StmtResult::Continue),
        }
    }

    fn eval_switch_statement(&mut self, switch_stmt: &'a SwitchStatement<'a>, _original_stmt: &'a Statement<'a>) -> Result<StmtResult, String> {
        // Evaluate the discriminant
        let discriminant = self.eval_expression(&switch_stmt.discriminant)?;

        if discriminant.is_dynamic() {
            // Can't determine which case to take
            // Return ResidualEmitted so containing loop knows to emit remaining loop
            // Don't emit the switch here - the while loop will emit the complete loop
            return Ok(StmtResult::ResidualEmitted);
        }

        // Find matching case
        let mut matched = false;
        let mut fell_through = false;

        for case in &switch_stmt.cases {
            if matched || fell_through {
                // Execute this case (fell through or matched)
                for case_stmt in &case.consequent {
                    match self.eval_statement(case_stmt)? {
                        StmtResult::Return(v) => return Ok(StmtResult::Return(v)),
                        StmtResult::Break => return Ok(StmtResult::Continue),
                        _ => {}
                    }
                }
                fell_through = true;
                continue;
            }

            // Check if this case matches
            if let Some(test) = &case.test {
                let test_val = self.eval_expression(test)?;
                if discriminant == test_val {
                    matched = true;
                    // Execute this case
                    for case_stmt in &case.consequent {
                        match self.eval_statement(case_stmt)? {
                            StmtResult::Return(v) => return Ok(StmtResult::Return(v)),
                            StmtResult::Break => return Ok(StmtResult::Continue),
                            _ => {}
                        }
                    }
                    fell_through = true;
                }
            }
        }

        // Handle default case if no match
        if !matched {
            for case in &switch_stmt.cases {
                if case.test.is_none() {
                    // This is the default case
                    for case_stmt in &case.consequent {
                        match self.eval_statement(case_stmt)? {
                            StmtResult::Return(v) => return Ok(StmtResult::Return(v)),
                            StmtResult::Break => return Ok(StmtResult::Continue),
                            _ => {}
                        }
                    }
                    break;
                }
            }
        }

        Ok(StmtResult::Continue)
    }

    /// Build a residual while loop - emit the original loop structure
    /// The current state of variables will be emitted separately
    fn build_while_residual(&mut self, while_stmt: &'a WhileStatement<'a>) -> String {
        // Emit the original while loop structure using codegen
        // We can't easily specialize expressions inside because they depend on
        // loop variables that change at runtime
        let mut codegen = Codegen::new();
        use oxc_codegen::Gen;
        while_stmt.print(&mut codegen, Context::empty());
        codegen.into_source_text()
    }

    /// Specialize a statement by substituting known static values
    fn specialize_statement(&mut self, stmt: &'a Statement<'a>) -> String {
        match stmt {
            Statement::BlockStatement(block) => {
                let stmts: Vec<String> = block.body.iter()
                    .map(|s| self.specialize_statement(s))
                    .collect();
                stmts.join("\n")
            }
            Statement::ExpressionStatement(expr) => {
                let specialized = self.specialize_expression(&expr.expression);
                format!("{};", specialized)
            }
            Statement::VariableDeclaration(decl) => {
                self.specialize_var_decl(decl)
            }
            Statement::ReturnStatement(ret) => {
                if let Some(arg) = &ret.argument {
                    let specialized = self.specialize_expression(arg);
                    format!("return {};", specialized)
                } else {
                    "return;".to_string()
                }
            }
            Statement::IfStatement(if_stmt) => {
                let cond = self.specialize_expression(&if_stmt.test);
                let cons = self.specialize_statement(&if_stmt.consequent);
                if let Some(alt) = &if_stmt.alternate {
                    let alt_str = self.specialize_statement(alt);
                    format!("if ({}) {{\n{}\n}} else {{\n{}\n}}", cond, cons, alt_str)
                } else {
                    format!("if ({}) {{\n{}\n}}", cond, cons)
                }
            }
            Statement::WhileStatement(while_stmt) => {
                let cond = self.specialize_expression(&while_stmt.test);
                let body = self.specialize_statement(&while_stmt.body);
                format!("while ({}) {{\n{}\n}}", cond, body)
            }
            Statement::SwitchStatement(switch_stmt) => {
                let disc = self.specialize_expression(&switch_stmt.discriminant);
                let mut cases = Vec::new();
                for case in &switch_stmt.cases {
                    let case_body: Vec<String> = case.consequent.iter()
                        .map(|s| self.specialize_statement(s))
                        .collect();
                    if let Some(test) = &case.test {
                        let test_str = self.specialize_expression(test);
                        cases.push(format!("case {}:\n{}", test_str, case_body.join("\n")));
                    } else {
                        cases.push(format!("default:\n{}", case_body.join("\n")));
                    }
                }
                format!("switch ({}) {{\n{}\n}}", disc, cases.join("\n"))
            }
            Statement::BreakStatement(_) => "break;".to_string(),
            _ => emit_stmt(stmt),
        }
    }

    /// Specialize an expression by substituting known static values
    fn specialize_expression(&mut self, expr: &'a Expression<'a>) -> String {
        // Try to evaluate the expression
        match self.eval_expression(expr) {
            Ok(value) => {
                // If we got a static value, use it
                if value.is_static() {
                    residual_of(&value).unwrap_or_else(|_| emit_expr(expr))
                } else {
                    // Dynamic value - use its residual
                    residual_of(&value).unwrap_or_else(|_| emit_expr(expr))
                }
            }
            Err(_) => emit_expr(expr),
        }
    }

    /// Specialize a variable declaration
    fn specialize_var_decl(&mut self, decl: &'a VariableDeclaration<'a>) -> String {
        let mut parts = Vec::new();
        for declarator in &decl.declarations {
            let name = match &declarator.id.kind {
                BindingPatternKind::BindingIdentifier(id) => id.name.to_string(),
                _ => continue,
            };

            if let Some(init) = &declarator.init {
                let specialized = self.specialize_expression(init);
                parts.push(format!("let {} = {};", name, specialized));
            } else {
                parts.push(format!("let {};", name));
            }
        }
        parts.join("\n")
    }

    /// Inner variable declaration evaluation (doesn't track top-level)
    fn eval_variable_declaration_inner(&mut self, decl: &'a VariableDeclaration<'a>) -> Result<(), String> {
        for declarator in &decl.declarations {
            let name = match &declarator.id.kind {
                BindingPatternKind::BindingIdentifier(id) => id.name.to_string(),
                _ => {
                    // Unsupported binding pattern - emit as residual
                    // We can't properly handle destructuring yet
                    return Err(format!("Unsupported binding pattern"));
                }
            };

            let value = if let Some(init) = &declarator.init {
                // Special handling for function expressions - register them so they can be called
                if let Expression::FunctionExpression(func_expr) = init {
                    // Register the function so it can be called later
                    self.functions.insert(name.clone(), func_expr);

                    // Get params for closure value
                    let params: Vec<String> = func_expr
                        .params
                        .items
                        .iter()
                        .filter_map(|p| {
                            if let BindingPatternKind::BindingIdentifier(id) = &p.pattern.kind {
                                Some(id.name.to_string())
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Create closure value
                    let source = emit_function(func_expr);
                    Value::Closure {
                        params,
                        body_id: 0,
                        env: self.env.capture(),
                        source,
                        name: Some(name.clone()),
                    }
                } else {
                    self.eval_expression(init)?
                }
            } else {
                // No initializer - for var declarations, preserve existing value if hoisted
                // (JavaScript semantics: var declaration hoists, but doesn't reset value)
                if decl.kind == oxc_ast::ast::VariableDeclarationKind::Var {
                    if let Some(existing) = self.env.get(&name) {
                        existing
                    } else {
                        Value::Undefined
                    }
                } else {
                    Value::Undefined
                }
            };

            // Trace binding creation
            let cause = if value.is_dynamic() {
                Some(format!("initialized with dynamic expression"))
            } else {
                None
            };
            self.trace_event(TraceEvent::BindingCreated {
                name: name.clone(),
                value_repr: self.value_repr(&value),
                is_static: value.is_static(),
                cause,
            });

            // For var declarations, use set() to update the hoisted binding in the function scope
            // rather than define() which creates a new binding in the current (possibly nested) scope.
            // This is important because var is hoisted to function scope, and assignments inside
            // blocks/loops should update that hoisted binding, not create a shadowing one.
            if decl.kind == oxc_ast::ast::VariableDeclarationKind::Var && self.env.exists(&name) {
                self.env.set(&name, value);
            } else {
                self.env.define(&name, value);
            }
        }
        Ok(())
    }

    fn eval_variable_declaration(&mut self, decl: &'a VariableDeclaration<'a>, stmt: &'a Statement<'a>) -> Result<(), String> {
        for declarator in &decl.declarations {
            let name = match &declarator.id.kind {
                BindingPatternKind::BindingIdentifier(id) => id.name.to_string(),
                // Unsupported binding pattern - emit as residual
                _ => {
                    self.emit_residual(emit_stmt(stmt));
                    return Ok(());
                }
            };

            let value = if let Some(init) = &declarator.init {
                // Special handling for function expressions - register them so they can be called
                if let Expression::FunctionExpression(func_expr) = init {
                    // Register the function so it can be called later
                    self.functions.insert(name.clone(), func_expr);

                    // Get params for closure value
                    let params: Vec<String> = func_expr
                        .params
                        .items
                        .iter()
                        .filter_map(|p| {
                            if let BindingPatternKind::BindingIdentifier(id) = &p.pattern.kind {
                                Some(id.name.to_string())
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Create closure value
                    let source = emit_function(func_expr);
                    Value::Closure {
                        params,
                        body_id: 0,
                        env: self.env.capture(),
                        source,
                        name: Some(name.clone()),
                    }
                } else {
                    self.eval_expression(init)?
                }
            } else {
                Value::Undefined
            };

            // Trace binding creation
            let cause = if value.is_dynamic() {
                Some(format!("initialized with dynamic expression"))
            } else {
                None
            };
            self.trace_event(TraceEvent::BindingCreated {
                name: name.clone(),
                value_repr: self.value_repr(&value),
                is_static: value.is_static(),
                cause,
            });

            self.env.define(&name, value.clone());

            // Track top-level variable names with their original expression
            // Final values are captured at the end of eval_program
            if self.env.depth() == 1 {
                let original_expr = declarator.init.as_ref().map(|e| emit_expr(e));
                self.top_level_vars.push((name, original_expr));
            }
        }
        Ok(())
    }

    fn eval_expression(&mut self, expr: &'a Expression<'a>) -> Result<Value, String> {
        // Use the iterative implementation - this is the single code path for all expression evaluation
        self.eval_expression_iterative(expr)
    }

    /// Old recursive eval_expression - kept for reference, but eval_expression now uses iterative version
    #[allow(dead_code)]
    fn eval_expression_recursive(&mut self, expr: &'a Expression<'a>) -> Result<Value, String> {
        match expr {
            Expression::NumericLiteral(lit) => Ok(Value::Number(lit.value)),
            Expression::StringLiteral(lit) => Ok(Value::String(lit.value.to_string())),
            Expression::BooleanLiteral(lit) => Ok(Value::Bool(lit.value)),
            Expression::NullLiteral(_) => Ok(Value::Null),
            Expression::Identifier(id) => {
                let name = id.name.to_string();
                // Handle special JavaScript globals
                match name.as_str() {
                    "undefined" => Ok(Value::Undefined),
                    // Unknown variables are treated as dynamic (might be globals)
                    _ => Ok(self.env.get(&name).unwrap_or_else(|| dynamic(name)))
                }
            }
            Expression::ArrayExpression(arr) => {
                let mut elements = Vec::new();
                for elem in &arr.elements {
                    match elem {
                        ArrayExpressionElement::SpreadElement(_) => {
                            // Spread makes the whole array dynamic
                            return Ok(dynamic(emit_expr(expr)));
                        }
                        ArrayExpressionElement::Elision(_) => {
                            elements.push(Value::Undefined);
                        }
                        _ => {
                            let val = self.eval_expression(elem.to_expression())?;
                            elements.push(val);
                        }
                    }
                }
                // Keep the array as Array even if it has dynamic elements
                // This allows us to know the length statically
                Ok(Value::Array(SharedArray::new(elements)))
            }
            Expression::ObjectExpression(obj) => {
                use std::collections::HashMap;
                let mut props = HashMap::new();
                for prop in &obj.properties {
                    match prop {
                        oxc_ast::ast::ObjectPropertyKind::ObjectProperty(p) => {
                            // Get property key
                            let key = match &p.key {
                                oxc_ast::ast::PropertyKey::StaticIdentifier(id) => id.name.to_string(),
                                oxc_ast::ast::PropertyKey::StringLiteral(s) => s.value.to_string(),
                                // Complex keys make the object dynamic
                                _ => return Ok(dynamic(emit_expr(expr))),
                            };
                            // Evaluate the value
                            let val = self.eval_expression(&p.value)?;
                            props.insert(key, val);
                        }
                        oxc_ast::ast::ObjectPropertyKind::SpreadProperty(_) => {
                            // Spread makes the whole object dynamic
                            return Ok(dynamic(emit_expr(expr)));
                        }
                    }
                }
                Ok(Value::Object(SharedObject::new(props)))
            }
            Expression::BinaryExpression(bin) => self.eval_binary_expression(bin),
            Expression::UnaryExpression(unary) => self.eval_unary_expression(unary),
            Expression::CallExpression(call) => self.eval_call_expression(call),
            Expression::StaticMemberExpression(member) => {
                let obj = self.eval_expression(&member.object)?;
                let prop = member.property.name.to_string();

                match (&obj, prop.as_str()) {
                    (Value::Array(arr), "length") => Ok(Value::Number(arr.len() as f64)),
                    (Value::String(s), "length") => Ok(Value::Number(s.len() as f64)),
                    (Value::Object(props), _) => {
                        // Look up property in object
                        match props.get(&prop) {
                            Some(val) => Ok(val.clone()),
                            None => Ok(Value::Undefined),
                        }
                    }
                    (Value::Dynamic(obj_expr), _) => {
                        Ok(dynamic(format!("{}.{}", obj_expr, prop)))
                    }
                    // Unknown property access on static value - emit as residual
                    _ => {
                        let obj_js = residual_of(&obj)?;
                        Ok(dynamic(format!("{}.{}", obj_js, prop)))
                    }
                }
            }
            Expression::ComputedMemberExpression(member) => {
                let obj = self.eval_expression(&member.object)?;
                let idx = self.eval_expression(&member.expression)?;

                match (&obj, &idx) {
                    (Value::Array(arr), Value::Number(n)) => {
                        let i = *n as usize;
                        Ok(arr.get(i).unwrap_or(Value::Undefined))
                    }
                    _ => {
                        // At least one dynamic, or unknown types - emit residual
                        let obj_js = residual_of(&obj)?;
                        let idx_js = residual_of(&idx)?;
                        Ok(dynamic(format!("{}[{}]", obj_js, idx_js)))
                    }
                }
            }
            Expression::AssignmentExpression(assign) => self.eval_assignment_expression(assign),
            Expression::UpdateExpression(update) => self.eval_update_expression(update),
            Expression::ParenthesizedExpression(paren) => {
                // Just evaluate the inner expression
                self.eval_expression(&paren.expression)
            }
            Expression::ConditionalExpression(cond) => {
                let test = self.eval_expression(&cond.test)?;
                match test.is_truthy() {
                    Some(true) => self.eval_expression(&cond.consequent),
                    Some(false) => self.eval_expression(&cond.alternate),
                    None => {
                        // Condition is dynamic - evaluate both branches and build residual
                        let cons = self.eval_expression(&cond.consequent)?;
                        let alt = self.eval_expression(&cond.alternate)?;
                        let test_js = residual_of(&test)?;
                        let cons_js = residual_of(&cons)?;
                        let alt_js = residual_of(&alt)?;
                        Ok(dynamic(format!("{} ? {} : {}", test_js, cons_js, alt_js)))
                    }
                }
            }
            Expression::LogicalExpression(logic) => {
                let left = self.eval_expression(&logic.left)?;
                match logic.operator {
                    LogicalOperator::And => {
                        match left.is_truthy() {
                            Some(false) => Ok(left), // short-circuit
                            Some(true) => self.eval_expression(&logic.right),
                            None => {
                                let right = self.eval_expression(&logic.right)?;
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right)?;
                                Ok(dynamic(format!("{} && {}", left_js, right_js)))
                            }
                        }
                    }
                    LogicalOperator::Or => {
                        match left.is_truthy() {
                            Some(true) => Ok(left), // short-circuit
                            Some(false) => self.eval_expression(&logic.right),
                            None => {
                                let right = self.eval_expression(&logic.right)?;
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right)?;
                                Ok(dynamic(format!("{} || {}", left_js, right_js)))
                            }
                        }
                    }
                    LogicalOperator::Coalesce => {
                        match &left {
                            Value::Null | Value::Undefined => self.eval_expression(&logic.right),
                            _ if left.is_static() => Ok(left),
                            _ => {
                                let right = self.eval_expression(&logic.right)?;
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right)?;
                                Ok(dynamic(format!("{} ?? {}", left_js, right_js)))
                            }
                        }
                    }
                }
            }
            // Unknown expression type - emit as residual
            _ => Ok(dynamic(emit_expr(expr))),
        }
    }

    // ========================================================================
    // Iterative Expression Evaluation
    // ========================================================================

    /// Iteratively evaluate an expression using an explicit work stack.
    /// This avoids Rust stack overflow for deeply nested/recursive JS code.
    fn eval_expression_iterative(&mut self, expr: &'a Expression<'a>) -> Result<Value, String> {
        let mut work: Vec<ExprWork<'a>> = vec![ExprWork::Eval(expr)];
        let mut values: Vec<Value> = vec![];

        while let Some(item) = work.pop() {
            match item {
                ExprWork::Eval(e) => {
                    self.push_expr_work(e, &mut work, &mut values)?;
                }
                ExprWork::EvalIdentifier { name } => {
                    // Lazy identifier evaluation - look up current value now
                    let val = self.env.get(&name).unwrap_or(Value::Dynamic(name));
                    values.push(val);
                }
                ExprWork::ApplyBinary { operator } => {
                    let right = values.pop().ok_or("Missing right operand")?;
                    let left = values.pop().ok_or("Missing left operand")?;
                    let result = self.apply_binary_op(operator, left, right)?;
                    values.push(result);
                }
                ExprWork::ApplyUnary { operator } => {
                    let arg = values.pop().ok_or("Missing unary operand")?;
                    let result = self.apply_unary_op(operator, arg)?;
                    values.push(result);
                }
                ExprWork::LogicalRight { right, operator, left_residual } => {
                    let left = values.pop().ok_or("Missing left operand for logical")?;
                    match operator {
                        LogicalOperator::And => {
                            match left.is_truthy() {
                                Some(false) => values.push(left),
                                Some(true) => work.push(ExprWork::Eval(right)),
                                None => {
                                    // Need to evaluate right and build residual
                                    work.push(ExprWork::LogicalFinish {
                                        operator,
                                        left_residual: left_residual.unwrap_or_else(|| residual_of(&left).unwrap_or_default()),
                                    });
                                    work.push(ExprWork::Eval(right));
                                }
                            }
                        }
                        LogicalOperator::Or => {
                            match left.is_truthy() {
                                Some(true) => values.push(left),
                                Some(false) => work.push(ExprWork::Eval(right)),
                                None => {
                                    work.push(ExprWork::LogicalFinish {
                                        operator,
                                        left_residual: left_residual.unwrap_or_else(|| residual_of(&left).unwrap_or_default()),
                                    });
                                    work.push(ExprWork::Eval(right));
                                }
                            }
                        }
                        LogicalOperator::Coalesce => {
                            match &left {
                                Value::Null | Value::Undefined => work.push(ExprWork::Eval(right)),
                                _ if left.is_static() => values.push(left),
                                _ => {
                                    work.push(ExprWork::LogicalFinish {
                                        operator,
                                        left_residual: left_residual.unwrap_or_else(|| residual_of(&left).unwrap_or_default()),
                                    });
                                    work.push(ExprWork::Eval(right));
                                }
                            }
                        }
                    }
                }
                ExprWork::ConditionalBranch { consequent, alternate } => {
                    let test = values.pop().ok_or("Missing conditional test")?;
                    match test.is_truthy() {
                        Some(true) => work.push(ExprWork::Eval(consequent)),
                        Some(false) => work.push(ExprWork::Eval(alternate)),
                        None => {
                            // Dynamic condition - evaluate both and build residual
                            // We need to evaluate both branches and combine
                            // For simplicity, fall back to recursive for now
                            let cons = self.eval_expression_iterative(consequent)?;
                            let alt = self.eval_expression_iterative(alternate)?;
                            let test_js = residual_of(&test)?;
                            let cons_js = residual_of(&cons)?;
                            let alt_js = residual_of(&alt)?;
                            values.push(dynamic(format!("{} ? {} : {}", test_js, cons_js, alt_js)));
                        }
                    }
                }
                ExprWork::ArrayCollect { remaining } => {
                    // Pop `remaining` values and build array
                    let mut elements = Vec::with_capacity(remaining);
                    for _ in 0..remaining {
                        elements.push(values.pop().ok_or("Missing array element")?);
                    }
                    elements.reverse(); // They were pushed in reverse order
                    values.push(Value::Array(SharedArray::new(elements)));
                }
                ExprWork::ObjectCollect { keys, remaining } => {
                    // Pop values for each key
                    let mut props = std::collections::HashMap::new();
                    for key in keys.into_iter().rev().take(remaining) {
                        let val = values.pop().ok_or("Missing object property value")?;
                        props.insert(key, val);
                    }
                    values.push(Value::Object(SharedObject::new(props)));
                }
                ExprWork::CallApply { func_name, arg_count, call_expr } => {
                    // Pop arguments
                    let mut args = Vec::with_capacity(arg_count);
                    for _ in 0..arg_count {
                        args.push(values.pop().ok_or("Missing call argument")?);
                    }
                    args.reverse();

                    // Inline the call
                    let result = self.apply_function_call(&func_name, args, call_expr)?;
                    values.push(result);
                }
                ExprWork::MethodCallApply { method_name, base_name, path, arg_count, call_expr } => {
                    let mut args = Vec::with_capacity(arg_count);
                    for _ in 0..arg_count {
                        args.push(values.pop().ok_or("Missing method argument")?);
                    }
                    args.reverse();

                    let result = self.apply_method_call(&base_name, &path, &method_name, args, call_expr)?;
                    values.push(result);
                }
                ExprWork::MemberAccess { property } => {
                    let obj = values.pop().ok_or("Missing object for member access")?;
                    let result = self.apply_member_access(obj, &property)?;
                    values.push(result);
                }
                ExprWork::ComputedMemberApply => {
                    let idx = values.pop().ok_or("Missing index for computed member")?;
                    let obj = values.pop().ok_or("Missing object for computed member")?;
                    let result = self.apply_computed_member(obj, idx)?;
                    values.push(result);
                }
                ExprWork::ComputedCallApply { arg_count, call_expr } => {
                    // Pop the callee (result of computed member access)
                    let callee = values.pop().ok_or("Missing callee for computed call")?;

                    // Pop arguments
                    let mut args = Vec::with_capacity(arg_count);
                    for _ in 0..arg_count {
                        args.push(values.pop().ok_or("Missing argument for computed call")?);
                    }
                    args.reverse();

                    // Check if callee is a closure with a known function name
                    if let Value::Closure { name: Some(ref func_name), .. } = callee {
                        if self.functions.contains_key(func_name.as_str()) {
                            // Call the function
                            self.trace.mark_function_called(func_name);
                            let result = self.apply_function_call(func_name, args, call_expr)?;
                            values.push(result);
                            continue;
                        }
                    }

                    // Can't evaluate - emit as residual
                    let callee_js = residual_of(&callee)?;
                    let arg_strs: Result<Vec<_>, _> = args.iter().map(residual_of).collect();
                    values.push(dynamic(format!("{}({})", callee_js, arg_strs?.join(", "))));
                }
                ExprWork::AssignmentApply { target, operator, original_expr } => {
                    let right_value = values.pop().ok_or("Missing assignment value")?;
                    let result = self.apply_assignment(target, operator, right_value, original_expr)?;
                    values.push(result);
                }
                ExprWork::UpdateApply { name, operator, prefix } => {
                    let result = self.apply_update(&name, operator, prefix)?;
                    values.push(result);
                }
                ExprWork::LogicalFinish { operator, left_residual } => {
                    let right = values.pop().ok_or("Missing right operand for logical")?;
                    let right_js = residual_of(&right)?;
                    let op_str = match operator {
                        LogicalOperator::And => "&&",
                        LogicalOperator::Or => "||",
                        LogicalOperator::Coalesce => "??",
                    };
                    values.push(dynamic(format!("{} {} {}", left_residual, op_str, right_js)));
                }
            }
        }

        Ok(values.pop().unwrap_or(Value::Undefined))
    }

    /// Evaluate an expression using a unified work stack that handles function calls
    /// iteratively, avoiding Rust stack overflow for recursive JS functions.
    fn eval_expression_unified(&mut self, expr: &'a Expression<'a>) -> Result<Value, String> {
        let mut work: Vec<UnifiedWork<'a>> = vec![UnifiedWork::Expr(ExprWork::Eval(expr))];
        let mut values: Vec<Value> = vec![];
        let mut return_value: Option<Value> = None;

        while let Some(item) = work.pop() {
            match item {
                UnifiedWork::Expr(expr_work) => {
                    match expr_work {
                        ExprWork::Eval(e) => {
                            self.push_expr_work_unified(e, &mut work, &mut values)?;
                        }
                        ExprWork::EvalIdentifier { name } => {
                            // Lazy identifier evaluation - look up current value now
                            let val = self.env.get(&name).unwrap_or(Value::Dynamic(name));
                            values.push(val);
                        }
                        ExprWork::ApplyBinary { operator } => {
                            let right = values.pop().ok_or("Missing right operand")?;
                            let left = values.pop().ok_or("Missing left operand")?;
                            let result = self.apply_binary_op(operator, left, right)?;
                            values.push(result);
                        }
                        ExprWork::ApplyUnary { operator } => {
                            let arg = values.pop().ok_or("Missing unary operand")?;
                            let result = self.apply_unary_op(operator, arg)?;
                            values.push(result);
                        }
                        ExprWork::LogicalRight { right, operator, left_residual } => {
                            let left = values.pop().ok_or("Missing left operand for logical")?;
                            match operator {
                                LogicalOperator::And => {
                                    match left.is_truthy() {
                                        Some(false) => values.push(left),
                                        Some(true) => work.push(UnifiedWork::Expr(ExprWork::Eval(right))),
                                        None => {
                                            work.push(UnifiedWork::Expr(ExprWork::LogicalFinish {
                                                operator,
                                                left_residual: left_residual.unwrap_or_else(|| residual_of(&left).unwrap_or_default()),
                                            }));
                                            work.push(UnifiedWork::Expr(ExprWork::Eval(right)));
                                        }
                                    }
                                }
                                LogicalOperator::Or => {
                                    match left.is_truthy() {
                                        Some(true) => values.push(left),
                                        Some(false) => work.push(UnifiedWork::Expr(ExprWork::Eval(right))),
                                        None => {
                                            work.push(UnifiedWork::Expr(ExprWork::LogicalFinish {
                                                operator,
                                                left_residual: left_residual.unwrap_or_else(|| residual_of(&left).unwrap_or_default()),
                                            }));
                                            work.push(UnifiedWork::Expr(ExprWork::Eval(right)));
                                        }
                                    }
                                }
                                LogicalOperator::Coalesce => {
                                    match &left {
                                        Value::Null | Value::Undefined => work.push(UnifiedWork::Expr(ExprWork::Eval(right))),
                                        _ => {
                                            if left.is_static() {
                                                values.push(left);
                                            } else {
                                                work.push(UnifiedWork::Expr(ExprWork::LogicalFinish {
                                                    operator,
                                                    left_residual: left_residual.unwrap_or_else(|| residual_of(&left).unwrap_or_default()),
                                                }));
                                                work.push(UnifiedWork::Expr(ExprWork::Eval(right)));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        ExprWork::ConditionalBranch { consequent, alternate } => {
                            let test = values.pop().ok_or("Missing conditional test")?;
                            match test.is_truthy() {
                                Some(true) => work.push(UnifiedWork::Expr(ExprWork::Eval(consequent))),
                                Some(false) => work.push(UnifiedWork::Expr(ExprWork::Eval(alternate))),
                                None => {
                                    // Dynamic condition - evaluate both branches and build residual
                                    let cons = self.eval_expression_unified(consequent)?;
                                    let alt = self.eval_expression_unified(alternate)?;
                                    let test_js = residual_of(&test)?;
                                    let cons_js = residual_of(&cons)?;
                                    let alt_js = residual_of(&alt)?;
                                    values.push(dynamic(format!("{} ? {} : {}", test_js, cons_js, alt_js)));
                                }
                            }
                        }
                        ExprWork::ArrayCollect { remaining } => {
                            let mut elements = Vec::with_capacity(remaining);
                            for _ in 0..remaining {
                                elements.push(values.pop().ok_or("Missing array element")?);
                            }
                            elements.reverse();
                            values.push(Value::Array(SharedArray::new(elements)));
                        }
                        ExprWork::ObjectCollect { keys, remaining } => {
                            let mut props = std::collections::HashMap::new();
                            let mut vals = Vec::with_capacity(remaining);
                            for _ in 0..remaining {
                                vals.push(values.pop().ok_or("Missing object property value")?);
                            }
                            vals.reverse();
                            for (key, val) in keys.into_iter().zip(vals) {
                                props.insert(key, val);
                            }
                            values.push(Value::Object(SharedObject::new(props)));
                        }
                        ExprWork::CallApply { func_name, arg_count, call_expr: _ } => {
                            // Pop arguments
                            let mut args = Vec::with_capacity(arg_count);
                            for _ in 0..arg_count {
                                args.push(values.pop().ok_or("Missing call argument")?);
                            }
                            args.reverse();

                            // Mark function as called for dead code elimination
                            self.trace.mark_function_called(&func_name);

                            // Check if function exists - also check if it's a variable holding a closure
                            let (func, actual_func_name) = if let Some(func) = self.functions.get(&func_name) {
                                (*func, func_name.clone())
                            } else if let Some(Value::Closure { name: Some(ref closure_func_name), .. }) = self.env.get(&func_name) {
                                // Variable holds a closure with a known function name
                                if let Some(func) = self.functions.get(closure_func_name.as_str()) {
                                    self.trace.mark_function_called(closure_func_name);
                                    (*func, closure_func_name.clone())
                                } else {
                                    // Closure name doesn't map to a known function
                                    let arg_strs: Result<Vec<_>, _> = args.iter().map(residual_of).collect();
                                    values.push(dynamic(format!("{}({})", func_name, arg_strs?.join(", "))));
                                    continue;
                                }
                            } else {
                                // Function not defined - emit as dynamic call
                                let arg_strs: Result<Vec<_>, _> = args.iter().map(residual_of).collect();
                                values.push(dynamic(format!("{}({})", func_name, arg_strs?.join(", "))));
                                continue;
                            };
                            let func = func;

                            // Check recursion limit using the actual function name
                            let recursion_count = self.call_stack.iter().filter(|n| n.as_str() == actual_func_name).count();
                            if recursion_count >= self.max_recursion_depth {
                                self.trace.mark_function_live(&actual_func_name);
                                let arg_strs: Result<Vec<_>, _> = args.iter().map(residual_of).collect();
                                values.push(dynamic(format!("{}({})", actual_func_name, arg_strs?.join(", "))));
                                continue;
                            }

                            // Set up function call - save current state and push function body
                            let saved_expr_work: Vec<ExprWork<'a>> = work.iter()
                                .filter_map(|w| match w {
                                    UnifiedWork::Expr(e) => Some(e.clone()),
                                    _ => None,
                                })
                                .collect();
                            let saved_values = std::mem::take(&mut values);
                            let saved_residual = self.take_residual();

                            // Push completion marker (will be processed after function body)
                            work.push(UnifiedWork::FunctionCallComplete {
                                func_name: actual_func_name.clone(),
                                func,
                                saved_expr_work,
                                saved_values,
                                saved_residual,
                            });

                            // Push function call start (sets up scope and body)
                            work.push(UnifiedWork::FunctionCallStart {
                                func,
                                func_name: actual_func_name,
                                arg_values: args,
                            });
                        }
                        ExprWork::MethodCallApply { method_name, base_name, path, arg_count, call_expr } => {
                            let mut args = Vec::with_capacity(arg_count);
                            for _ in 0..arg_count {
                                args.push(values.pop().ok_or("Missing method argument")?);
                            }
                            args.reverse();
                            let result = self.apply_method_call(&base_name, &path, &method_name, args, call_expr)?;
                            values.push(result);
                        }
                        ExprWork::MemberAccess { property } => {
                            let obj = values.pop().ok_or("Missing object for member access")?;
                            let result = self.apply_member_access(obj, &property)?;
                            values.push(result);
                        }
                        ExprWork::ComputedMemberApply => {
                            let idx = values.pop().ok_or("Missing index for computed member")?;
                            let obj = values.pop().ok_or("Missing object for computed member")?;
                            let result = self.apply_computed_member(obj, idx)?;
                            values.push(result);
                        }
                        ExprWork::ComputedCallApply { arg_count, call_expr: _ } => {
                            // Pop the callee (result of computed member access)
                            let callee = values.pop().ok_or("Missing callee for computed call")?;

                            // Pop arguments
                            let mut args = Vec::with_capacity(arg_count);
                            for _ in 0..arg_count {
                                args.push(values.pop().ok_or("Missing argument for computed call")?);
                            }
                            args.reverse();

                            // Check if callee is a closure with a known function name
                            if let Value::Closure { name: Some(ref func_name), .. } = callee {
                                if let Some(func) = self.functions.get(func_name.as_str()) {
                                    // We can call this function!
                                    self.trace.mark_function_called(func_name);

                                    // Check recursion limit
                                    let recursion_count = self.call_stack.iter().filter(|n| n.as_str() == func_name).count();
                                    if recursion_count >= self.max_recursion_depth {
                                        self.trace.mark_function_live(func_name);
                                        let arg_strs: Result<Vec<_>, _> = args.iter().map(residual_of).collect();
                                        values.push(dynamic(format!("{}({})", func_name, arg_strs?.join(", "))));
                                        continue;
                                    }

                                    // Push function call onto work stack
                                    let func = *func;
                                    work.push(UnifiedWork::FunctionCallComplete {
                                        func_name: func_name.clone(),
                                        func,
                                        saved_expr_work: vec![],
                                        saved_values: values.clone(),
                                        saved_residual: self.take_residual(),
                                    });
                                    work.push(UnifiedWork::FunctionCallStart {
                                        func,
                                        func_name: func_name.clone(),
                                        arg_values: args,
                                    });
                                    continue;
                                }
                            }

                            // Can't evaluate - emit as residual
                            let callee_js = residual_of(&callee)?;
                            let arg_strs: Result<Vec<_>, _> = args.iter().map(residual_of).collect();
                            values.push(dynamic(format!("{}({})", callee_js, arg_strs?.join(", "))));
                        }
                        ExprWork::AssignmentApply { target, operator, original_expr } => {
                            let right_value = values.pop().ok_or("Missing assignment value")?;
                            let result = self.apply_assignment(target, operator, right_value, original_expr)?;
                            values.push(result);
                        }
                        ExprWork::UpdateApply { name, operator, prefix } => {
                            let result = self.apply_update(&name, operator, prefix)?;
                            values.push(result);
                        }
                        ExprWork::LogicalFinish { operator, left_residual } => {
                            let right = values.pop().ok_or("Missing right operand for logical")?;
                            let right_js = residual_of(&right)?;
                            let op_str = match operator {
                                LogicalOperator::And => "&&",
                                LogicalOperator::Or => "||",
                                LogicalOperator::Coalesce => "??",
                            };
                            values.push(dynamic(format!("{} {} {}", left_residual, op_str, right_js)));
                        }
                    }
                }
                UnifiedWork::FunctionCallStart { func, func_name, arg_values } => {
                    // Push onto call stack
                    self.call_stack.push(func_name.clone());

                    // Trace function entry
                    let args_info: Vec<(String, bool)> = arg_values.iter()
                        .map(|v| (self.value_repr(v), v.is_static()))
                        .collect();
                    self.trace_event(TraceEvent::FunctionEnter {
                        name: func_name.clone(),
                        args: args_info,
                    });

                    // Get params
                    let params: Vec<String> = func
                        .params
                        .items
                        .iter()
                        .filter_map(|p| {
                            if let BindingPatternKind::BindingIdentifier(id) = &p.pattern.kind {
                                Some(id.name.to_string())
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Create new scope
                    self.env.push_scope();

                    // Bind parameters
                    for (param, arg) in params.iter().zip(arg_values.iter()) {
                        self.env.define(param, arg.clone());
                    }

                    // IMPORTANT: Hoist function declarations from the body before executing statements
                    // This is required for JavaScript semantics - functions are hoisted within their scope
                    if let Some(body) = &func.body {
                        for stmt in &body.statements {
                            if let Statement::FunctionDeclaration(nested_func) = stmt {
                                self.register_function(nested_func)?;
                            }
                        }
                    }

                    // IMPORTANT: Hoist var declarations from nested structures (while, switch, if, etc.)
                    // JavaScript hoists ALL vars to the nearest function scope
                    if let Some(body) = &func.body {
                        let mut var_names = Vec::new();
                        for stmt in body.statements.iter() {
                            Self::collect_var_names(stmt, &mut var_names);
                        }
                        for name in var_names {
                            // Only hoist if not already defined in current scope (e.g., as a parameter)
                            // We use exists_in_current_scope because var should create a new local binding
                            // even if a variable with the same name exists in an outer scope
                            if !self.env.exists_in_current_scope(&name) {
                                self.env.define(&name, dynamic(name.clone()));
                            }
                        }
                    }

                    // Push function body statements (in reverse order so they execute in order)
                    if let Some(body) = &func.body {
                        if !body.statements.is_empty() {
                            work.push(UnifiedWork::EvalStatements {
                                stmts: &body.statements,
                                idx: 0,
                            });
                        }
                    }
                }
                UnifiedWork::FunctionCallComplete { func_name, func, saved_expr_work: _, saved_values, saved_residual } => {
                    // Get return value (if any)
                    let result = return_value.take().unwrap_or(Value::Undefined);

                    // Check if function produced residual
                    let func_residual = self.take_residual();

                    // Capture scope bindings BEFORE popping scope
                    let scope_bindings = self.env.current_scope_bindings();

                    // Pop scope
                    self.env.pop_scope();

                    // Pop call stack
                    self.call_stack.pop();

                    // Trace function exit
                    self.trace_event(TraceEvent::FunctionExit {
                        name: func_name.clone(),
                        result_repr: self.value_repr(&result),
                        is_static: result.is_static() && func_residual.is_empty(),
                    });

                    // Restore residual
                    self.residual_stmts = saved_residual;

                    // If function produced residual, wrap in IIFE
                    if !func_residual.is_empty() {
                        let mut iife_body = Vec::new();
                        for (name, value) in scope_bindings {
                            let value_str = residual_of(&value)?;
                            iife_body.push(format!("let {} = {};", name, value_str));
                        }

                        for stmt in &func_residual {
                            iife_body.push(stmt.clone());
                        }

                        // Add remaining statements from function body if any
                        if let Some(body) = &func.body {
                            for stmt in body.statements.iter() {
                                if let Statement::ReturnStatement(ret) = stmt {
                                    if let Some(arg) = &ret.argument {
                                        let return_expr = emit_expr(arg);
                                        iife_body.push(format!("return {};", return_expr));
                                    }
                                }
                            }
                        }

                        let iife = format!("(() => {{\n{}\n}})()", iife_body.join("\n"));

                        // Restore values and push IIFE as result
                        values = saved_values;
                        values.push(dynamic(iife));
                    } else {
                        // Restore values and push result
                        values = saved_values;
                        values.push(result);
                    }
                }
                UnifiedWork::EvalStatements { stmts, idx } => {
                    if idx < stmts.len() {
                        // Schedule next statement (pushed first so it's processed after current statement completes)
                        if idx + 1 < stmts.len() {
                            work.push(UnifiedWork::EvalStatements { stmts, idx: idx + 1 });
                        }

                        // Push work items for current statement (will be processed before next EvalStatements)
                        let stmt = &stmts[idx];
                        self.push_statement_work(stmt, &mut work, &mut values, &mut return_value)?;
                    }
                }
                UnifiedWork::Stmt(_) => {
                    // Statement work items are handled through EvalStatements
                    // This variant exists for future use
                }
                UnifiedWork::ReturnContinue => {
                    // Value is on the value stack
                    let val = values.pop().unwrap_or(Value::Undefined);
                    return_value = Some(val.clone());
                    // Skip remaining statements in THIS function only (up to the nearest FunctionCallComplete)
                    // Find the index of the nearest FunctionCallComplete marker
                    let func_complete_idx = work.iter().rposition(|w| matches!(w, UnifiedWork::FunctionCallComplete { .. }));
                    if let Some(idx) = func_complete_idx {
                        // Remove EvalStatements only from indices > idx (after the FunctionCallComplete marker)
                        // We need to keep work[0..=idx] intact and filter work[idx+1..]
                        let mut i = idx + 1;
                        while i < work.len() {
                            if matches!(work[i], UnifiedWork::EvalStatements { .. }) {
                                work.remove(i);
                            } else {
                                i += 1;
                            }
                        }
                    } else {
                        // No function call context - we're at top level, clear all EvalStatements
                        work.retain(|w| !matches!(w, UnifiedWork::EvalStatements { .. }));
                    }
                }
                UnifiedWork::ExprStmtContinue => {
                    // Value is on the value stack
                    let value = values.pop().unwrap_or(Value::Undefined);
                    // Side-effect only if dynamic
                    if !value.is_static() {
                        self.emit_residual(residual_of(&value)?);
                    }
                }
                UnifiedWork::VarDeclContinue { name, kind } => {
                    // Value is on the value stack
                    let value = values.pop().unwrap_or(Value::Undefined);
                    self.env.define(&name, value.clone());

                    // Record binding
                    self.trace_event(TraceEvent::BindingCreated {
                        name: name.clone(),
                        value_repr: self.value_repr(&value),
                        is_static: value.is_static(),
                        cause: None,
                    });

                    // Emit residual if dynamic
                    if !value.is_static() {
                        let value_js = residual_of(&value)?;
                        let kind_str = match kind {
                            VariableDeclarationKind::Var => "var",
                            VariableDeclarationKind::Let => "let",
                            VariableDeclarationKind::Const => "const",
                            _ => "let",
                        };
                        self.emit_residual(format!("{} {} = {};", kind_str, name, value_js));
                    }
                }
                UnifiedWork::IfCondContinue { if_stmt, original_stmt } => {
                    // Condition value is on the value stack
                    let test_val = values.pop().unwrap_or(Value::Undefined);
                    match test_val.is_truthy() {
                        Some(true) => {
                            // Push consequent for evaluation
                            self.push_statement_work(&if_stmt.consequent, &mut work, &mut values, &mut return_value)?;
                        }
                        Some(false) => {
                            if let Some(alt) = &if_stmt.alternate {
                                self.push_statement_work(alt, &mut work, &mut values, &mut return_value)?;
                            }
                        }
                        None => {
                            // Dynamic condition - emit as residual
                            let test_js = residual_of(&test_val)?;
                            let cons_js = self.specialize_statement(&if_stmt.consequent);
                            let if_residual = if let Some(alt) = &if_stmt.alternate {
                                let alt_js = self.specialize_statement(alt);
                                format!("if ({}) {{\n{}\n}} else {{\n{}\n}}", test_js, cons_js, alt_js)
                            } else {
                                format!("if ({}) {{\n{}\n}}", test_js, cons_js)
                            };
                            self.emit_residual(if_residual);
                        }
                    }
                }
                UnifiedWork::WhileCondContinue { while_stmt, iterations, original_stmt } => {
                    // Condition value is on the value stack
                    let test_val = values.pop().unwrap_or(Value::Undefined);
                    match test_val.is_truthy() {
                        Some(true) => {
                            if iterations >= self.max_iterations {
                                // Iteration limit reached - emit as residual
                                let test_js = emit_expr(&while_stmt.test);
                                let body_js = self.specialize_statement(&while_stmt.body);
                                self.emit_residual(format!("while ({}) {{\n{}\n}}", test_js, body_js));
                            } else {
                                // Push continuation for next iteration
                                work.push(UnifiedWork::WhileCondContinue {
                                    while_stmt,
                                    iterations: iterations + 1,
                                    original_stmt,
                                });
                                // Push condition evaluation
                                self.push_expr_work_unified(&while_stmt.test, &mut work, &mut values)?;
                                // Push body for evaluation
                                self.push_statement_work(&while_stmt.body, &mut work, &mut values, &mut return_value)?;
                            }
                        }
                        Some(false) => {
                            // Loop complete - decrement depth
                            self.depth -= 1;
                        }
                        None => {
                            // Dynamic condition - emit as residual
                            let test_js = emit_expr(&while_stmt.test);
                            let body_js = self.specialize_statement(&while_stmt.body);
                            self.emit_residual(format!("while ({}) {{\n{}\n}}", test_js, body_js));
                            self.depth -= 1;
                        }
                    }
                }
                UnifiedWork::PopScope => {
                    self.env.pop_scope();
                }
                UnifiedWork::WhileDepthDec => {
                    self.depth -= 1;
                }
                UnifiedWork::SwitchDiscriminantContinue { switch_stmt } => {
                    // Discriminant value is on the value stack
                    let discriminant = values.pop().unwrap_or(Value::Undefined);

                    // Start processing cases from the first one
                    if !switch_stmt.cases.is_empty() {
                        // Push first case processing
                        work.push(UnifiedWork::SwitchCaseContinue {
                            switch_stmt,
                            discriminant,
                            case_idx: 0,
                            matched: false,
                            fell_through: false,
                        });
                    }
                }
                UnifiedWork::SwitchCaseContinue { switch_stmt, discriminant, case_idx, matched, fell_through } => {
                    if case_idx >= switch_stmt.cases.len() {
                        // All cases processed
                        continue;
                    }

                    let case = &switch_stmt.cases[case_idx];

                    // Check if we already matched or fell through
                    if matched || fell_through {
                        // Execute case body then continue to next case
                        // Schedule next case first
                        if case_idx + 1 < switch_stmt.cases.len() {
                            work.push(UnifiedWork::SwitchCaseContinue {
                                switch_stmt,
                                discriminant: discriminant.clone(),
                                case_idx: case_idx + 1,
                                matched: true,
                                fell_through: true,
                            });
                        }

                        // Push case body statements in reverse
                        for stmt in case.consequent.iter().rev() {
                            self.push_statement_work(stmt, &mut work, &mut values, &mut return_value)?;
                        }
                    } else if let Some(test) = &case.test {
                        // Need to evaluate test expression
                        work.push(UnifiedWork::SwitchCaseTestContinue {
                            switch_stmt,
                            discriminant: discriminant.clone(),
                            case_idx,
                            matched,
                            fell_through,
                        });
                        self.push_expr_work_unified(test, &mut work, &mut values)?;
                    } else {
                        // Default case - match if nothing else matched yet
                        // Execute body and continue
                        if case_idx + 1 < switch_stmt.cases.len() {
                            work.push(UnifiedWork::SwitchCaseContinue {
                                switch_stmt,
                                discriminant: discriminant.clone(),
                                case_idx: case_idx + 1,
                                matched: true,
                                fell_through: true,
                            });
                        }

                        // Push case body statements in reverse
                        for stmt in case.consequent.iter().rev() {
                            self.push_statement_work(stmt, &mut work, &mut values, &mut return_value)?;
                        }
                    }
                }
                UnifiedWork::SwitchCaseTestContinue { switch_stmt, discriminant, case_idx, matched, fell_through } => {
                    // Test value is on the value stack
                    let test_val = values.pop().unwrap_or(Value::Undefined);

                    // Check if discriminant matches test
                    let matches = discriminant == test_val;

                    if matches {
                        // Match found - execute body then continue
                        let case = &switch_stmt.cases[case_idx];

                        // Schedule next case first (for fall-through)
                        if case_idx + 1 < switch_stmt.cases.len() {
                            work.push(UnifiedWork::SwitchCaseContinue {
                                switch_stmt,
                                discriminant: discriminant.clone(),
                                case_idx: case_idx + 1,
                                matched: true,
                                fell_through: true,
                            });
                        }

                        // Push case body statements in reverse
                        for stmt in case.consequent.iter().rev() {
                            self.push_statement_work(stmt, &mut work, &mut values, &mut return_value)?;
                        }
                    } else {
                        // No match - continue to next case
                        if case_idx + 1 < switch_stmt.cases.len() {
                            work.push(UnifiedWork::SwitchCaseContinue {
                                switch_stmt,
                                discriminant,
                                case_idx: case_idx + 1,
                                matched: false,
                                fell_through: false,
                            });
                        }
                    }
                }
                UnifiedWork::BreakContinue => {
                    // Break exits the innermost switch or loop.
                    // Find what kind of context we're in by looking for the first switch or while item.
                    // Include SwitchEnd to detect switch context even after all cases are processed.
                    let innermost_switch_idx = work.iter().rposition(|w| matches!(w,
                        UnifiedWork::SwitchCaseContinue { .. } |
                        UnifiedWork::SwitchCaseTestContinue { .. } |
                        UnifiedWork::SwitchEnd
                    ));
                    let innermost_while_idx = work.iter().rposition(|w| matches!(w,
                        UnifiedWork::WhileCondContinue { .. }
                    ));

                    match (innermost_switch_idx, innermost_while_idx) {
                        (Some(switch_idx), Some(while_idx)) if switch_idx > while_idx => {
                            // Switch is innermost - only remove switch items (those after while_idx)
                            work.retain(|w| !matches!(w,
                                UnifiedWork::SwitchCaseContinue { .. } |
                                UnifiedWork::SwitchCaseTestContinue { .. } |
                                UnifiedWork::SwitchEnd
                            ));
                        }
                        (Some(_), Some(_)) => {
                            // While is innermost - remove the while and any switch items inside it
                            work.retain(|w| !matches!(w,
                                UnifiedWork::SwitchCaseContinue { .. } |
                                UnifiedWork::SwitchCaseTestContinue { .. } |
                                UnifiedWork::SwitchEnd |
                                UnifiedWork::WhileCondContinue { .. }
                            ));
                        }
                        (Some(_), None) => {
                            // Only switch context
                            work.retain(|w| !matches!(w,
                                UnifiedWork::SwitchCaseContinue { .. } |
                                UnifiedWork::SwitchCaseTestContinue { .. } |
                                UnifiedWork::SwitchEnd
                            ));
                        }
                        (None, Some(_)) => {
                            // Only while context
                            work.retain(|w| !matches!(w,
                                UnifiedWork::WhileCondContinue { .. }
                            ));
                        }
                        (None, None) => {
                            // No context to break from (shouldn't happen in valid JS)
                        }
                    }
                }
                UnifiedWork::SwitchEnd => {
                    // Switch statement completed normally (without break)
                    // Nothing to do, just pop the marker
                }
            }
        }

        Ok(values.pop().unwrap_or(Value::Undefined))
    }

    /// Push work items for a statement to be evaluated iteratively
    fn push_statement_work(
        &mut self,
        stmt: &'a Statement<'a>,
        work: &mut Vec<UnifiedWork<'a>>,
        values: &mut Vec<Value>,
        return_value: &mut Option<Value>,
    ) -> Result<(), String> {
        match stmt {
            Statement::ExpressionStatement(expr_stmt) => {
                // Push continuation, then expression
                work.push(UnifiedWork::ExprStmtContinue);
                self.push_expr_work_unified(&expr_stmt.expression, work, values)?;
            }
            Statement::ReturnStatement(ret) => {
                work.push(UnifiedWork::ReturnContinue);
                if let Some(arg) = &ret.argument {
                    self.push_expr_work_unified(arg, work, values)?;
                } else {
                    values.push(Value::Undefined);
                }
            }
            Statement::VariableDeclaration(decl) => {
                // Handle each declarator
                for declarator in decl.declarations.iter().rev() {
                    if let BindingPatternKind::BindingIdentifier(id) = &declarator.id.kind {
                        let name = id.name.to_string();
                        work.push(UnifiedWork::VarDeclContinue {
                            name: name.clone(),
                            kind: decl.kind,
                        });
                        if let Some(init) = &declarator.init {
                            // Special handling for function expressions - register them so they can be called
                            if let Expression::FunctionExpression(func_expr) = init {
                                // Register the function so it can be called later
                                self.functions.insert(name.clone(), func_expr);

                                // Get params for closure value
                                let params: Vec<String> = func_expr
                                    .params
                                    .items
                                    .iter()
                                    .filter_map(|p| {
                                        if let BindingPatternKind::BindingIdentifier(id) = &p.pattern.kind {
                                            Some(id.name.to_string())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();

                                // Create closure value
                                let source = emit_function(func_expr);
                                values.push(Value::Closure {
                                    params,
                                    body_id: 0,
                                    env: self.env.capture(),
                                    source,
                                    name: Some(name.clone()),
                                });
                            } else {
                                self.push_expr_work_unified(init, work, values)?;
                            }
                        } else {
                            values.push(Value::Undefined);
                        }
                    }
                }
            }
            Statement::IfStatement(if_stmt) => {
                // Push continuation, then condition
                work.push(UnifiedWork::IfCondContinue {
                    if_stmt,
                    original_stmt: stmt,
                });
                self.push_expr_work_unified(&if_stmt.test, work, values)?;
            }
            Statement::WhileStatement(while_stmt) => {
                self.depth += 1;
                // Push continuation, then condition
                work.push(UnifiedWork::WhileCondContinue {
                    while_stmt,
                    iterations: 0,
                    original_stmt: stmt,
                });
                self.push_expr_work_unified(&while_stmt.test, work, values)?;
            }
            Statement::BlockStatement(block) => {
                // Push scope pop, then all statements in reverse
                work.push(UnifiedWork::PopScope);
                self.env.push_scope();
                for stmt in block.body.iter().rev() {
                    self.push_statement_work(stmt, work, values, return_value)?;
                }
            }
            Statement::SwitchStatement(switch_stmt) => {
                // Push SwitchEnd marker first (will be processed last, after all cases)
                // This marker stays on the stack to indicate we're in a switch context for break handling
                work.push(UnifiedWork::SwitchEnd);
                // Push continuation, then discriminant
                work.push(UnifiedWork::SwitchDiscriminantContinue { switch_stmt });
                self.push_expr_work_unified(&switch_stmt.discriminant, work, values)?;
            }
            Statement::BreakStatement(_) => {
                // Push break continuation which will clear switch/while work items
                work.push(UnifiedWork::BreakContinue);
            }
            Statement::EmptyStatement(_) => {
                // Nothing to do
            }
            Statement::ContinueStatement(_) => {
                // Continue in a loop - skip remaining statements in current iteration
                // The while continuation will handle the next iteration
                work.retain(|w| !matches!(w,
                    UnifiedWork::EvalStatements { .. } |
                    UnifiedWork::ExprStmtContinue |
                    UnifiedWork::VarDeclContinue { .. } |
                    UnifiedWork::ReturnContinue
                ));
            }
            Statement::ForStatement(for_stmt) => {
                // Push for loop handling - simplified for now
                self.depth += 1;
                self.env.push_scope();

                // Evaluate init if present
                if let Some(init) = &for_stmt.init {
                    match init {
                        ForStatementInit::VariableDeclaration(decl) => {
                            for declarator in &decl.declarations {
                                if let BindingPatternKind::BindingIdentifier(id) = &declarator.id.kind {
                                    let name = id.name.to_string();
                                    if let Some(init_expr) = &declarator.init {
                                        // Need to evaluate init expression synchronously for now
                                        // This is a limitation - deeply nested for loops might still overflow
                                        let value = self.eval_expression_in_unified(init_expr, work, values)?;
                                        self.env.define(&name, value);
                                    } else {
                                        self.env.define(&name, Value::Undefined);
                                    }
                                }
                            }
                        }
                        _ => {
                            // Expression init - use to_expression() method
                            let _ = self.eval_expression_in_unified(init.to_expression(), work, values)?;
                        }
                    }
                }

                // Simple iterative for loop execution
                let mut iterations = 0;
                loop {
                    // Check test
                    if let Some(test) = &for_stmt.test {
                        let test_val = self.eval_expression_in_unified(test, work, values)?;
                        match test_val.is_truthy() {
                            Some(false) => break,
                            None => {
                                // Dynamic test - emit residual
                                self.depth -= 1;
                                self.env.pop_scope();
                                return Ok(());
                            }
                            _ => {}
                        }
                    }

                    if iterations >= self.max_iterations {
                        // Too many iterations
                        self.depth -= 1;
                        self.env.pop_scope();
                        return Ok(());
                    }

                    // Execute body - use synchronous evaluation here to keep the loop simple
                    // This is a limitation but for now we need to finish the refactor
                    let result = self.eval_statement_for_unified(&for_stmt.body, work, values, return_value)?;
                    match result {
                        StmtResult::Break => break,
                        StmtResult::Return(v) => {
                            *return_value = Some(v);
                            self.depth -= 1;
                            self.env.pop_scope();
                            return Ok(());
                        }
                        _ => {}
                    }

                    // Execute update
                    if let Some(update) = &for_stmt.update {
                        let _ = self.eval_expression_in_unified(update, work, values)?;
                    }

                    iterations += 1;
                }

                self.depth -= 1;
                self.env.pop_scope();
            }
            _ => {
                // For other statements, fall back to synchronous evaluation
                let result = self.eval_statement_for_unified(stmt, work, values, return_value)?;
                // Handle result if needed
                match result {
                    StmtResult::Return(val) => {
                        *return_value = Some(val);
                        work.retain(|w| !matches!(w, UnifiedWork::EvalStatements { .. }));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Evaluate a single statement within the unified evaluation context
    fn eval_statement_for_unified(
        &mut self,
        stmt: &'a Statement<'a>,
        work: &mut Vec<UnifiedWork<'a>>,
        values: &mut Vec<Value>,
        return_value: &mut Option<Value>,
    ) -> Result<StmtResult, String> {
        match stmt {
            Statement::ExpressionStatement(expr_stmt) => {
                // For expression statements within a function, we need to evaluate
                // the expression but it might contain function calls
                let value = self.eval_expression_in_unified(&expr_stmt.expression, work, values)?;

                // Side-effect only if dynamic
                if !value.is_static() {
                    self.emit_residual(residual_of(&value)?);
                }
                Ok(StmtResult::Continue)
            }
            Statement::ReturnStatement(ret) => {
                let val = if let Some(arg) = &ret.argument {
                    self.eval_expression_in_unified(arg, work, values)?
                } else {
                    Value::Undefined
                };
                *return_value = Some(val.clone());
                Ok(StmtResult::Return(val))
            }
            Statement::VariableDeclaration(decl) => {
                for declarator in &decl.declarations {
                    if let BindingPatternKind::BindingIdentifier(id) = &declarator.id.kind {
                        let name = id.name.to_string();
                        let value = if let Some(init) = &declarator.init {
                            self.eval_expression_in_unified(init, work, values)?
                        } else {
                            Value::Undefined
                        };

                        self.env.define(&name, value.clone());

                        // Record binding
                        self.trace_event(TraceEvent::BindingCreated {
                            name: name.clone(),
                            value_repr: self.value_repr(&value),
                            is_static: value.is_static(),
                            cause: None,
                        });

                        // Emit residual if dynamic
                        if !value.is_static() {
                            let value_js = residual_of(&value)?;
                            let kind = match decl.kind {
                                VariableDeclarationKind::Var => "var",
                                VariableDeclarationKind::Let => "let",
                                VariableDeclarationKind::Const => "const",
                                _ => "let",
                            };
                            self.emit_residual(format!("{} {} = {};", kind, name, value_js));
                        }
                    }
                }
                Ok(StmtResult::Continue)
            }
            Statement::IfStatement(if_stmt) => {
                let test_val = self.eval_expression_in_unified(&if_stmt.test, work, values)?;
                match test_val.is_truthy() {
                    Some(true) => {
                        self.eval_statement_for_unified(&if_stmt.consequent, work, values, return_value)
                    }
                    Some(false) => {
                        if let Some(alt) = &if_stmt.alternate {
                            self.eval_statement_for_unified(alt, work, values, return_value)
                        } else {
                            Ok(StmtResult::Continue)
                        }
                    }
                    None => {
                        // Dynamic condition - emit as residual
                        let test_js = residual_of(&test_val)?;
                        let cons_js = self.specialize_statement(&if_stmt.consequent);
                        let if_residual = if let Some(alt) = &if_stmt.alternate {
                            let alt_js = self.specialize_statement(alt);
                            format!("if ({}) {{\n{}\n}} else {{\n{}\n}}", test_js, cons_js, alt_js)
                        } else {
                            format!("if ({}) {{\n{}\n}}", test_js, cons_js)
                        };
                        self.emit_residual(if_residual);
                        Ok(StmtResult::ResidualEmitted)
                    }
                }
            }
            Statement::WhileStatement(while_stmt) => {
                self.depth += 1;
                if self.depth > self.max_recursion_depth {
                    self.depth -= 1;
                    let remaining = self.build_while_residual(while_stmt);
                    self.emit_residual(remaining);
                    return Ok(StmtResult::ResidualEmitted);
                }

                let mut iterations = 0;
                loop {
                    if iterations >= self.max_iterations {
                        let remaining = self.build_while_residual(while_stmt);
                        self.emit_residual(remaining);
                        self.depth -= 1;
                        return Ok(StmtResult::Continue);
                    }
                    iterations += 1;

                    let cond = self.eval_expression_in_unified(&while_stmt.test, work, values)?;
                    match cond.is_truthy() {
                        Some(true) => {
                            let body_result = self.eval_statement_for_unified(&while_stmt.body, work, values, return_value)?;
                            match body_result {
                                StmtResult::Break => break,
                                StmtResult::Return(v) => {
                                    self.depth -= 1;
                                    return Ok(StmtResult::Return(v));
                                }
                                StmtResult::ResidualEmitted => {
                                    let remaining = self.build_while_residual(while_stmt);
                                    self.emit_residual(remaining);
                                    self.depth -= 1;
                                    return Ok(StmtResult::ResidualEmitted);
                                }
                                StmtResult::ContinueLoop => continue,
                                _ => {}
                            }
                        }
                        Some(false) => break,
                        None => {
                            let remaining = self.build_while_residual(while_stmt);
                            self.emit_residual(remaining);
                            self.depth -= 1;
                            return Ok(StmtResult::ResidualEmitted);
                        }
                    }
                }
                self.depth -= 1;
                Ok(StmtResult::Continue)
            }
            Statement::BlockStatement(block) => {
                self.env.push_scope();
                let mut result = StmtResult::Continue;
                for stmt in &block.body {
                    result = self.eval_statement_for_unified(stmt, work, values, return_value)?;
                    match &result {
                        StmtResult::Return(_) | StmtResult::Break | StmtResult::ContinueLoop |
                        StmtResult::Throw(_) | StmtResult::ResidualEmitted => break,
                        _ => {}
                    }
                }
                self.env.pop_scope();
                Ok(result)
            }
            Statement::BreakStatement(_) => Ok(StmtResult::Break),
            Statement::ContinueStatement(_) => Ok(StmtResult::ContinueLoop),
            Statement::SwitchStatement(switch_stmt) => {
                let discriminant = self.eval_expression_in_unified(&switch_stmt.discriminant, work, values)?;

                if !discriminant.is_static() {
                    // Dynamic discriminant - emit as residual
                    let disc_js = residual_of(&discriminant)?;
                    let mut cases_js = Vec::new();
                    for case in &switch_stmt.cases {
                        let case_body: Vec<String> = case.consequent.iter()
                            .map(|s| self.specialize_statement(s))
                            .collect();
                        if let Some(test) = &case.test {
                            let test_js = emit_expr(test);
                            cases_js.push(format!("case {}:\n{}", test_js, case_body.join("\n")));
                        } else {
                            cases_js.push(format!("default:\n{}", case_body.join("\n")));
                        }
                    }
                    self.emit_residual(format!("switch ({}) {{\n{}\n}}", disc_js, cases_js.join("\n")));
                    return Ok(StmtResult::ResidualEmitted);
                }

                // Static discriminant - evaluate matching case
                let mut matched = false;
                let mut fell_through = false;
                for case in &switch_stmt.cases {
                    if !matched && !fell_through {
                        if let Some(test) = &case.test {
                            let test_val = self.eval_expression_in_unified(test, work, values)?;
                            if discriminant == test_val {
                                matched = true;
                            }
                        } else {
                            // Default case - match if nothing else matched yet
                            matched = true;
                        }
                    }

                    if matched || fell_through {
                        fell_through = true;
                        for stmt in &case.consequent {
                            let result = self.eval_statement_for_unified(stmt, work, values, return_value)?;
                            match result {
                                StmtResult::Break => return Ok(StmtResult::Continue),
                                StmtResult::Return(v) => return Ok(StmtResult::Return(v)),
                                _ => {}
                            }
                        }
                    }
                }
                Ok(StmtResult::Continue)
            }
            Statement::ThrowStatement(throw_stmt) => {
                let val = self.eval_expression_in_unified(&throw_stmt.argument, work, values)?;
                Ok(StmtResult::Throw(val))
            }
            Statement::TryStatement(try_stmt) => {
                // Simplified try handling - evaluate block statements directly
                self.env.push_scope();
                let mut result = StmtResult::Continue;
                for stmt in &try_stmt.block.body {
                    result = self.eval_statement_for_unified(stmt, work, values, return_value)?;
                    match &result {
                        StmtResult::Return(_) | StmtResult::Break | StmtResult::Throw(_) => break,
                        _ => {}
                    }
                }
                self.env.pop_scope();
                Ok(result)
            }
            Statement::ForStatement(for_stmt) => {
                self.env.push_scope();

                // Init
                if let Some(init) = &for_stmt.init {
                    match init {
                        ForStatementInit::VariableDeclaration(decl) => {
                            for declarator in &decl.declarations {
                                if let BindingPatternKind::BindingIdentifier(id) = &declarator.id.kind {
                                    let name = id.name.to_string();
                                    let value = if let Some(init_expr) = &declarator.init {
                                        self.eval_expression_in_unified(init_expr, work, values)?
                                    } else {
                                        Value::Undefined
                                    };
                                    self.env.define(&name, value);
                                }
                            }
                        }
                        _ => {
                            self.eval_expression_in_unified(init.to_expression(), work, values)?;
                        }
                    }
                }

                let mut iterations = 0;
                loop {
                    if iterations >= self.max_iterations {
                        self.env.pop_scope();
                        return Ok(StmtResult::Continue);
                    }
                    iterations += 1;

                    // Test
                    if let Some(test) = &for_stmt.test {
                        let cond = self.eval_expression_in_unified(test, work, values)?;
                        match cond.is_truthy() {
                            Some(false) => break,
                            None => {
                                // Dynamic - emit residual
                                self.env.pop_scope();
                                return Ok(StmtResult::ResidualEmitted);
                            }
                            _ => {}
                        }
                    }

                    // Body
                    let body_result = self.eval_statement_for_unified(&for_stmt.body, work, values, return_value)?;
                    match body_result {
                        StmtResult::Break => break,
                        StmtResult::Return(v) => {
                            self.env.pop_scope();
                            return Ok(StmtResult::Return(v));
                        }
                        _ => {}
                    }

                    // Update
                    if let Some(update) = &for_stmt.update {
                        self.eval_expression_in_unified(update, work, values)?;
                    }
                }

                self.env.pop_scope();
                Ok(StmtResult::Continue)
            }
            _ => {
                // Fallback to regular statement evaluation
                self.eval_statement(stmt)
            }
        }
    }

    /// Evaluate an expression within the unified context, handling any function calls
    fn eval_expression_in_unified(
        &mut self,
        expr: &'a Expression<'a>,
        _work: &mut Vec<UnifiedWork<'a>>,
        _values: &mut Vec<Value>,
    ) -> Result<Value, String> {
        // Use the unified evaluation to handle function calls properly
        self.eval_expression_unified(expr)
    }

    /// Push work items for expression evaluation in unified context
    fn push_expr_work_unified(
        &mut self,
        expr: &'a Expression<'a>,
        work: &mut Vec<UnifiedWork<'a>>,
        values: &mut Vec<Value>,
    ) -> Result<(), String> {
        match expr {
            // Literals - push directly to values
            Expression::NumericLiteral(lit) => {
                values.push(Value::Number(lit.value));
            }
            Expression::StringLiteral(lit) => {
                values.push(Value::String(lit.value.to_string()));
            }
            Expression::BooleanLiteral(lit) => {
                values.push(Value::Bool(lit.value));
            }
            Expression::NullLiteral(_) => {
                values.push(Value::Null);
            }
            Expression::Identifier(id) => {
                let name = id.name.to_string();
                match name.as_str() {
                    "undefined" => values.push(Value::Undefined),
                    _ => {
                        // Push a work item for lazy evaluation
                        work.push(UnifiedWork::Expr(ExprWork::EvalIdentifier { name }));
                    }
                }
            }
            Expression::ArrayExpression(arr) => {
                let count = arr.elements.len();
                work.push(UnifiedWork::Expr(ExprWork::ArrayCollect { remaining: count }));
                for elem in arr.elements.iter().rev() {
                    if let Some(e) = elem.as_expression() {
                        work.push(UnifiedWork::Expr(ExprWork::Eval(e)));
                    } else {
                        values.push(Value::Undefined);
                    }
                }
            }
            Expression::ObjectExpression(obj) => {
                let mut keys = Vec::new();
                let count = obj.properties.len();
                for prop in &obj.properties {
                    if let ObjectPropertyKind::ObjectProperty(p) = prop {
                        let key = match &p.key {
                            PropertyKey::StaticIdentifier(id) => id.name.to_string(),
                            PropertyKey::StringLiteral(s) => s.value.to_string(),
                            _ => continue,
                        };
                        keys.push(key);
                    }
                }
                work.push(UnifiedWork::Expr(ExprWork::ObjectCollect { keys, remaining: count }));
                for prop in obj.properties.iter().rev() {
                    if let ObjectPropertyKind::ObjectProperty(p) = prop {
                        work.push(UnifiedWork::Expr(ExprWork::Eval(&p.value)));
                    }
                }
            }
            Expression::BinaryExpression(bin) => {
                work.push(UnifiedWork::Expr(ExprWork::ApplyBinary { operator: bin.operator }));
                work.push(UnifiedWork::Expr(ExprWork::Eval(&bin.right)));
                work.push(UnifiedWork::Expr(ExprWork::Eval(&bin.left)));
            }
            Expression::UnaryExpression(unary) => {
                work.push(UnifiedWork::Expr(ExprWork::ApplyUnary { operator: unary.operator }));
                work.push(UnifiedWork::Expr(ExprWork::Eval(&unary.argument)));
            }
            Expression::LogicalExpression(logic) => {
                work.push(UnifiedWork::Expr(ExprWork::LogicalRight {
                    right: &logic.right,
                    operator: logic.operator,
                    left_residual: None,
                }));
                work.push(UnifiedWork::Expr(ExprWork::Eval(&logic.left)));
            }
            Expression::ConditionalExpression(cond) => {
                work.push(UnifiedWork::Expr(ExprWork::ConditionalBranch {
                    consequent: &cond.consequent,
                    alternate: &cond.alternate,
                }));
                work.push(UnifiedWork::Expr(ExprWork::Eval(&cond.test)));
            }
            Expression::CallExpression(call) => {
                // Get function name
                let func_name = match &call.callee {
                    Expression::Identifier(id) => id.name.to_string(),
                    Expression::StaticMemberExpression(member) => {
                        // Method call - extract the full member chain
                        // e.g., myGlobal.listeners.push() -> base="myGlobal", path=["listeners"], method="push"
                        let method_name = member.property.name.to_string();

                        // Extract the base name and path by walking the member chain
                        let (base_name, path) = Self::extract_member_chain(&member.object);

                        if base_name.is_none() {
                            return self.fallback_to_iterative(expr, values);
                        }
                        let base_name = base_name.unwrap();

                        // Push method call apply
                        let arg_count = call.arguments.len();
                        work.push(UnifiedWork::Expr(ExprWork::MethodCallApply {
                            method_name,
                            base_name,
                            path,
                            arg_count,
                            call_expr: call,
                        }));

                        // Push arguments in reverse
                        for arg in call.arguments.iter().rev() {
                            work.push(UnifiedWork::Expr(ExprWork::Eval(arg.to_expression())));
                        }

                        return Ok(());
                    }
                    Expression::ComputedMemberExpression(member) => {
                        // Calling result of computed member access: funcs[0]()
                        let arg_count = call.arguments.len();

                        // Push the apply work item
                        work.push(UnifiedWork::Expr(ExprWork::ComputedCallApply {
                            arg_count,
                            call_expr: call,
                        }));

                        // Push arguments in reverse (will be popped first)
                        for arg in call.arguments.iter().rev() {
                            work.push(UnifiedWork::Expr(ExprWork::Eval(arg.to_expression())));
                        }

                        // Push computed member evaluation (callee)
                        // This will push the index, then the object
                        work.push(UnifiedWork::Expr(ExprWork::ComputedMemberApply));
                        work.push(UnifiedWork::Expr(ExprWork::Eval(&member.expression)));
                        work.push(UnifiedWork::Expr(ExprWork::Eval(&member.object)));

                        return Ok(());
                    }
                    _ => return self.fallback_to_iterative(expr, values),
                };

                // Push call apply
                let arg_count = call.arguments.len();
                work.push(UnifiedWork::Expr(ExprWork::CallApply {
                    func_name,
                    arg_count,
                    call_expr: call,
                }));

                // Push arguments in reverse order
                for arg in call.arguments.iter().rev() {
                    work.push(UnifiedWork::Expr(ExprWork::Eval(arg.to_expression())));
                }
            }
            Expression::StaticMemberExpression(member) => {
                work.push(UnifiedWork::Expr(ExprWork::MemberAccess {
                    property: member.property.name.to_string(),
                }));
                work.push(UnifiedWork::Expr(ExprWork::Eval(&member.object)));
            }
            Expression::ComputedMemberExpression(member) => {
                work.push(UnifiedWork::Expr(ExprWork::ComputedMemberApply));
                work.push(UnifiedWork::Expr(ExprWork::Eval(&member.expression)));
                work.push(UnifiedWork::Expr(ExprWork::Eval(&member.object)));
            }
            Expression::AssignmentExpression(assign) => {
                let target = match &assign.left {
                    AssignmentTarget::AssignmentTargetIdentifier(id) => {
                        AssignTargetInfo::Identifier(id.name.to_string())
                    }
                    AssignmentTarget::StaticMemberExpression(member) => {
                        let obj_name = match &member.object {
                            Expression::Identifier(id) => id.name.to_string(),
                            _ => return self.fallback_to_iterative(expr, values),
                        };
                        AssignTargetInfo::StaticMember {
                            object: obj_name,
                            property: member.property.name.to_string(),
                        }
                    }
                    AssignmentTarget::ComputedMemberExpression(member) => {
                        let obj_name = match &member.object {
                            Expression::Identifier(id) => id.name.to_string(),
                            _ => return self.fallback_to_iterative(expr, values),
                        };
                        let idx = self.eval_expression_iterative(&member.expression)?;
                        AssignTargetInfo::ComputedMember {
                            base: obj_name,
                            indices: vec![idx],
                        }
                    }
                    _ => return self.fallback_to_iterative(expr, values),
                };
                work.push(UnifiedWork::Expr(ExprWork::AssignmentApply {
                    target,
                    operator: assign.operator,
                    original_expr: assign,
                }));
                work.push(UnifiedWork::Expr(ExprWork::Eval(&assign.right)));
            }
            Expression::UpdateExpression(update) => {
                let name = match &update.argument {
                    SimpleAssignmentTarget::AssignmentTargetIdentifier(id) => id.name.to_string(),
                    _ => return self.fallback_to_iterative(expr, values),
                };
                work.push(UnifiedWork::Expr(ExprWork::UpdateApply {
                    name,
                    operator: update.operator,
                    prefix: update.prefix,
                }));
            }
            Expression::ParenthesizedExpression(paren) => {
                work.push(UnifiedWork::Expr(ExprWork::Eval(&paren.expression)));
            }
            Expression::SequenceExpression(seq) => {
                // Evaluate all expressions, keep last result
                for (i, expr) in seq.expressions.iter().enumerate() {
                    if i == seq.expressions.len() - 1 {
                        work.push(UnifiedWork::Expr(ExprWork::Eval(expr)));
                    } else {
                        // Evaluate for side effects only
                        let _ = self.eval_expression_iterative(expr)?;
                    }
                }
            }
            Expression::FunctionExpression(func_expr) => {
                // Create a closure value
                let params: Vec<String> = func_expr
                    .params
                    .items
                    .iter()
                    .filter_map(|p| {
                        if let BindingPatternKind::BindingIdentifier(id) = &p.pattern.kind {
                            Some(id.name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();

                // Get function name if it has one (named function expression)
                // For anonymous functions, generate a unique name so we can call them
                let func_name = if let Some(id) = &func_expr.id {
                    id.name.to_string()
                } else {
                    // Generate unique name for anonymous function
                    let name = format!("__anon_{}", self.anon_func_counter);
                    self.anon_func_counter += 1;
                    name
                };

                // Register the function so it can be called later
                self.functions.insert(func_name.clone(), func_expr);

                let source = emit_function(func_expr);
                values.push(Value::Closure {
                    params,
                    body_id: 0,
                    env: self.env.capture(),
                    source,
                    name: Some(func_name),
                });
            }
            _ => {
                return self.fallback_to_iterative(expr, values);
            }
        }
        Ok(())
    }

    /// Fallback to standard iterative evaluation for unsupported expressions
    fn fallback_to_iterative(&mut self, expr: &'a Expression<'a>, values: &mut Vec<Value>) -> Result<(), String> {
        let result = self.eval_expression_iterative(expr)?;
        values.push(result);
        Ok(())
    }

    /// Push work items for evaluating an expression
    fn push_expr_work(
        &mut self,
        expr: &'a Expression<'a>,
        work: &mut Vec<ExprWork<'a>>,
        values: &mut Vec<Value>,
    ) -> Result<(), String> {
        match expr {
            // Literals - push directly to values
            Expression::NumericLiteral(lit) => {
                values.push(Value::Number(lit.value));
            }
            Expression::StringLiteral(lit) => {
                values.push(Value::String(lit.value.to_string()));
            }
            Expression::BooleanLiteral(lit) => {
                values.push(Value::Bool(lit.value));
            }
            Expression::NullLiteral(_) => {
                values.push(Value::Null);
            }
            Expression::Identifier(id) => {
                let name = id.name.to_string();
                match name.as_str() {
                    "undefined" => values.push(Value::Undefined),
                    _ => {
                        // Push a work item for lazy evaluation
                        work.push(ExprWork::EvalIdentifier { name });
                    }
                }
            }
            Expression::ArrayExpression(arr) => {
                // Check for spread
                for elem in &arr.elements {
                    if matches!(elem, ArrayExpressionElement::SpreadElement(_)) {
                        values.push(dynamic(emit_expr(expr)));
                        return Ok(());
                    }
                }
                // Push collect, then eval each element (in reverse so they're processed in order)
                let count = arr.elements.len();
                work.push(ExprWork::ArrayCollect { remaining: count });
                for elem in arr.elements.iter().rev() {
                    match elem {
                        ArrayExpressionElement::Elision(_) => {
                            values.push(Value::Undefined);
                        }
                        _ => {
                            work.push(ExprWork::Eval(elem.to_expression()));
                        }
                    }
                }
            }
            Expression::ObjectExpression(obj) => {
                // Check for spread
                for prop in &obj.properties {
                    if matches!(prop, oxc_ast::ast::ObjectPropertyKind::SpreadProperty(_)) {
                        values.push(dynamic(emit_expr(expr)));
                        return Ok(());
                    }
                }
                // Collect keys and push work items
                let mut keys = Vec::new();
                for prop in &obj.properties {
                    if let oxc_ast::ast::ObjectPropertyKind::ObjectProperty(p) = prop {
                        let key = match &p.key {
                            oxc_ast::ast::PropertyKey::StaticIdentifier(id) => id.name.to_string(),
                            oxc_ast::ast::PropertyKey::StringLiteral(s) => s.value.to_string(),
                            _ => {
                                values.push(dynamic(emit_expr(expr)));
                                return Ok(());
                            }
                        };
                        keys.push(key);
                    }
                }
                let count = keys.len();
                work.push(ExprWork::ObjectCollect { keys, remaining: count });
                for prop in obj.properties.iter().rev() {
                    if let oxc_ast::ast::ObjectPropertyKind::ObjectProperty(p) = prop {
                        work.push(ExprWork::Eval(&p.value));
                    }
                }
            }
            Expression::BinaryExpression(bin) => {
                work.push(ExprWork::ApplyBinary { operator: bin.operator });
                work.push(ExprWork::Eval(&bin.right));
                work.push(ExprWork::Eval(&bin.left));
            }
            Expression::UnaryExpression(unary) => {
                work.push(ExprWork::ApplyUnary { operator: unary.operator });
                work.push(ExprWork::Eval(&unary.argument));
            }
            Expression::LogicalExpression(logic) => {
                // Evaluate left first, then decide whether to evaluate right
                work.push(ExprWork::LogicalRight {
                    right: &logic.right,
                    operator: logic.operator,
                    left_residual: None,
                });
                work.push(ExprWork::Eval(&logic.left));
            }
            Expression::ConditionalExpression(cond) => {
                work.push(ExprWork::ConditionalBranch {
                    consequent: &cond.consequent,
                    alternate: &cond.alternate,
                });
                work.push(ExprWork::Eval(&cond.test));
            }
            Expression::CallExpression(call) => {
                // Handle method calls
                if let Expression::StaticMemberExpression(member) = &call.callee {
                    let method_name = member.property.name.to_string();

                    // Handle method calls on NewExpression (like new TextDecoder().decode(...))
                    if let Expression::NewExpression(new_expr) = &member.object {
                        if let Expression::Identifier(id) = &new_expr.callee {
                            let ctor_name = id.name.to_string();

                            // Handle new TextDecoder().decode(typedArray)
                            if ctor_name == "TextDecoder" && method_name == "decode" {
                                // Get encoding from TextDecoder constructor
                                let encoding = if new_expr.arguments.is_empty() {
                                    "utf-8".to_string()
                                } else if let Some(Argument::StringLiteral(lit)) = new_expr.arguments.first() {
                                    lit.value.to_string()
                                } else {
                                    values.push(dynamic(emit_call_expr(call)));
                                    return Ok(());
                                };

                                // Evaluate the argument to decode
                                if call.arguments.len() == 1 {
                                    let arg = self.eval_expression_iterative(call.arguments[0].to_expression())?;
                                    if let Value::TypedArray { kind, buffer, byte_offset, length } = arg {
                                        // Get the raw bytes from the typed array view
                                        let element_size = kind.element_size();
                                        let byte_length = length * element_size;
                                        let data = buffer.get_bytes(byte_offset, byte_length);

                                        // Decode the bytes
                                        match encoding.as_str() {
                                            "utf-8" | "utf8" => {
                                                match String::from_utf8(data.clone()) {
                                                    Ok(s) => {
                                                        values.push(Value::String(s));
                                                        return Ok(());
                                                    }
                                                    Err(_) => {
                                                        // Lossy decode
                                                        let s = String::from_utf8_lossy(&data).to_string();
                                                        values.push(Value::String(s));
                                                        return Ok(());
                                                    }
                                                }
                                            }
                                            _ => {
                                                // Unsupported encoding - emit as residual
                                                values.push(dynamic(emit_call_expr(call)));
                                                return Ok(());
                                            }
                                        }
                                    }
                                }
                                // Fall through to emit as residual
                                values.push(dynamic(emit_call_expr(call)));
                                return Ok(());
                            }

                            // Handle new DataView(buffer).getFloat64(offset, littleEndian)
                            if ctor_name == "DataView" && (method_name == "getFloat64" || method_name == "getFloat32" ||
                                method_name == "getInt8" || method_name == "getUint8" ||
                                method_name == "getInt16" || method_name == "getUint16" ||
                                method_name == "getInt32" || method_name == "getUint32") {

                                // Need at least buffer arg for DataView and offset arg for get method
                                if new_expr.arguments.is_empty() || call.arguments.is_empty() {
                                    values.push(dynamic(emit_call_expr(call)));
                                    return Ok(());
                                }

                                // Evaluate the buffer argument
                                let buffer_arg = self.eval_expression_iterative(new_expr.arguments[0].to_expression())?;
                                let buffer = match buffer_arg {
                                    Value::ArrayBuffer { buffer } => buffer,
                                    _ => {
                                        values.push(dynamic(emit_call_expr(call)));
                                        return Ok(());
                                    }
                                };

                                // Evaluate offset
                                let offset_val = self.eval_expression_iterative(call.arguments[0].to_expression())?;
                                let offset = match offset_val {
                                    Value::Number(n) => n as usize,
                                    _ => {
                                        values.push(dynamic(emit_call_expr(call)));
                                        return Ok(());
                                    }
                                };

                                // Evaluate littleEndian (default is false / big-endian)
                                let little_endian = if call.arguments.len() > 1 {
                                    match self.eval_expression_iterative(call.arguments[1].to_expression())? {
                                        Value::Bool(b) => b,
                                        _ => false,
                                    }
                                } else {
                                    false
                                };

                                // Read the value from the buffer
                                let result = read_dataview_value(&buffer, offset, &method_name, little_endian);
                                match result {
                                    Some(val) => {
                                        values.push(val);
                                        return Ok(());
                                    }
                                    None => {
                                        values.push(dynamic(emit_call_expr(call)));
                                        return Ok(());
                                    }
                                }
                            }
                        }
                    }

                    // Use extract_member_chain to handle nested paths like myGlobal.listeners.push()
                    let (base_name, path) = Self::extract_member_chain(&member.object);
                    if let Some(base_name) = base_name {
                        // Check for spread
                        for arg in &call.arguments {
                            if matches!(arg, Argument::SpreadElement(_)) {
                                values.push(dynamic(emit_call_expr(call)));
                                return Ok(());
                            }
                        }

                        let arg_count = call.arguments.len();
                        work.push(ExprWork::MethodCallApply {
                            method_name,
                            base_name,
                            path,
                            arg_count,
                            call_expr: call,
                        });
                        for arg in call.arguments.iter().rev() {
                            work.push(ExprWork::Eval(arg.to_expression()));
                        }
                        return Ok(());
                    }
                }

                // Handle IIFE - for now, delegate to recursive version for simplicity
                let mut callee = &call.callee;
                while let Expression::ParenthesizedExpression(paren) = callee {
                    callee = &paren.expression;
                }
                if let Expression::FunctionExpression(func_expr) = callee {
                    match self.eval_iife(func_expr, &call.arguments) {
                        Ok(val) => values.push(val),
                        Err(_) => values.push(dynamic(emit_call_expr(call))),
                    }
                    return Ok(());
                }

                // Handle regular function calls
                if let Expression::Identifier(id) = &call.callee {
                    let func_name = id.name.to_string();
                    self.trace.mark_function_called(&func_name);

                    // Check for spread
                    for arg in &call.arguments {
                        if matches!(arg, Argument::SpreadElement(_)) {
                            self.trace.mark_function_live(&func_name);
                            values.push(dynamic(emit_call_expr(call)));
                            return Ok(());
                        }
                    }

                    let arg_count = call.arguments.len();
                    work.push(ExprWork::CallApply {
                        func_name,
                        arg_count,
                        call_expr: call,
                    });
                    for arg in call.arguments.iter().rev() {
                        work.push(ExprWork::Eval(arg.to_expression()));
                    }
                    return Ok(());
                }

                // Handle computed member call: arr[0]()
                if let Expression::ComputedMemberExpression(member) = &call.callee {
                    let arg_count = call.arguments.len();

                    // Push the apply work item
                    work.push(ExprWork::ComputedCallApply {
                        arg_count,
                        call_expr: call,
                    });

                    // Push arguments in reverse (will be popped first)
                    for arg in call.arguments.iter().rev() {
                        work.push(ExprWork::Eval(arg.to_expression()));
                    }

                    // Push computed member evaluation (callee)
                    work.push(ExprWork::ComputedMemberApply);
                    work.push(ExprWork::Eval(&member.expression));
                    work.push(ExprWork::Eval(&member.object));

                    return Ok(());
                }

                // Unsupported call
                values.push(dynamic(emit_call_expr(call)));
            }
            Expression::StaticMemberExpression(member) => {
                work.push(ExprWork::MemberAccess {
                    property: member.property.name.to_string(),
                });
                work.push(ExprWork::Eval(&member.object));
            }
            Expression::ComputedMemberExpression(member) => {
                work.push(ExprWork::ComputedMemberApply);
                work.push(ExprWork::Eval(&member.expression));
                work.push(ExprWork::Eval(&member.object));
            }
            Expression::AssignmentExpression(assign) => {
                // Build target info
                let target = match &assign.left {
                    AssignmentTarget::AssignmentTargetIdentifier(id) => {
                        AssignTargetInfo::Identifier(id.name.to_string())
                    }
                    AssignmentTarget::ComputedMemberExpression(member) => {
                        // Try to trace member chain
                        match self.trace_member_chain(&member.object, &member.expression) {
                            Some((base, indices)) => {
                                AssignTargetInfo::ComputedMember { base, indices }
                            }
                            None => {
                                // Can't trace - emit as residual
                                values.push(dynamic(emit_assign_expr(assign)));
                                return Ok(());
                            }
                        }
                    }
                    AssignmentTarget::StaticMemberExpression(member) => {
                        // Handle obj.property = value
                        let obj_name = match &member.object {
                            Expression::Identifier(id) => id.name.to_string(),
                            _ => {
                                // Complex object expression - emit as residual
                                values.push(dynamic(emit_assign_expr(assign)));
                                return Ok(());
                            }
                        };
                        AssignTargetInfo::StaticMember {
                            object: obj_name,
                            property: member.property.name.to_string(),
                        }
                    }
                    _ => {
                        values.push(dynamic(emit_assign_expr(assign)));
                        return Ok(());
                    }
                };

                work.push(ExprWork::AssignmentApply {
                    target,
                    operator: assign.operator,
                    original_expr: assign,
                });
                work.push(ExprWork::Eval(&assign.right));
            }
            Expression::UpdateExpression(update) => {
                let name = match &update.argument {
                    SimpleAssignmentTarget::AssignmentTargetIdentifier(id) => id.name.to_string(),
                    _ => {
                        values.push(dynamic(emit_update_expr(update)));
                        return Ok(());
                    }
                };
                // Update doesn't need to evaluate subexpressions
                work.push(ExprWork::UpdateApply {
                    name,
                    operator: update.operator,
                    prefix: update.prefix,
                });
            }
            Expression::ParenthesizedExpression(paren) => {
                work.push(ExprWork::Eval(&paren.expression));
            }
            Expression::NewExpression(new_expr) => {
                // Handle new TypedArray(...) and new TextDecoder(...)
                if let Expression::Identifier(id) = &new_expr.callee {
                    let ctor_name = id.name.to_string();

                    // Handle TextDecoder
                    if ctor_name == "TextDecoder" {
                        let encoding = if new_expr.arguments.is_empty() {
                            "utf-8".to_string()
                        } else if let Some(Argument::StringLiteral(lit)) = new_expr.arguments.first() {
                            lit.value.to_string()
                        } else {
                            // Dynamic encoding - emit as residual
                            values.push(dynamic(emit_new_expr(new_expr)));
                            return Ok(());
                        };
                        values.push(Value::TextDecoder { encoding });
                        return Ok(());
                    }

                    // Handle ArrayBuffer
                    if ctor_name == "ArrayBuffer" {
                        if new_expr.arguments.len() == 1 {
                            let arg_val = self.eval_expression_iterative(new_expr.arguments[0].to_expression())?;
                            if let Value::Number(size) = arg_val {
                                let buffer = SharedBuffer::new(size as usize);
                                values.push(Value::ArrayBuffer { buffer });
                                return Ok(());
                            }
                        }
                        // Dynamic size or no argument - emit as residual
                        values.push(dynamic(emit_new_expr(new_expr)));
                        return Ok(());
                    }

                    // Handle DataView
                    if ctor_name == "DataView" {
                        if !new_expr.arguments.is_empty() {
                            let arg_val = self.eval_expression_iterative(new_expr.arguments[0].to_expression())?;
                            if let Value::ArrayBuffer { buffer } = arg_val {
                                let byte_offset = if new_expr.arguments.len() > 1 {
                                    match self.eval_expression_iterative(new_expr.arguments[1].to_expression())? {
                                        Value::Number(n) => n as usize,
                                        _ => {
                                            values.push(dynamic(emit_new_expr(new_expr)));
                                            return Ok(());
                                        }
                                    }
                                } else {
                                    0
                                };
                                let byte_length = if new_expr.arguments.len() > 2 {
                                    match self.eval_expression_iterative(new_expr.arguments[2].to_expression())? {
                                        Value::Number(n) => n as usize,
                                        _ => {
                                            values.push(dynamic(emit_new_expr(new_expr)));
                                            return Ok(());
                                        }
                                    }
                                } else {
                                    buffer.len() - byte_offset
                                };
                                values.push(Value::DataView { buffer, byte_offset, byte_length });
                                return Ok(());
                            }
                        }
                        // Not an ArrayBuffer argument - emit as residual
                        values.push(dynamic(emit_new_expr(new_expr)));
                        return Ok(());
                    }

                    // Handle typed arrays
                    if let Some(kind) = TypedArrayKind::from_name(&ctor_name) {
                        if new_expr.arguments.len() == 1 {
                            // Evaluate the argument
                            let arg_val = self.eval_expression_iterative(new_expr.arguments[0].to_expression())?;
                            match arg_val {
                                Value::Array(elements) => {
                                    // Convert array of numbers to bytes
                                    let length = elements.len();
                                    let element_size = kind.element_size();
                                    let mut data = Vec::with_capacity(length * element_size);
                                    for elem in elements.iter() {
                                        if let Value::Number(n) = elem {
                                            // For now, just handle Uint8Array-style (1 byte per element)
                                            // TODO: Handle other element sizes properly
                                            data.push(n as u8);
                                        } else {
                                            // Non-number element - emit as residual
                                            values.push(dynamic(emit_new_expr(new_expr)));
                                            return Ok(());
                                        }
                                    }
                                    let buffer = SharedBuffer::from_bytes(data);
                                    values.push(Value::TypedArray { kind, buffer, byte_offset: 0, length });
                                    return Ok(());
                                }
                                Value::ArrayBuffer { buffer } => {
                                    // Create a view over the ArrayBuffer
                                    let element_size = kind.element_size();
                                    let length = buffer.len() / element_size;
                                    values.push(Value::TypedArray { kind, buffer, byte_offset: 0, length });
                                    return Ok(());
                                }
                                _ => {
                                    // Non-array argument - emit as residual
                                    values.push(dynamic(emit_new_expr(new_expr)));
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
                // Unsupported new expression - emit as residual
                values.push(dynamic(emit_new_expr(new_expr)));
            }
            Expression::FunctionExpression(func_expr) => {
                // Create a closure value
                let params: Vec<String> = func_expr
                    .params
                    .items
                    .iter()
                    .filter_map(|p| {
                        if let BindingPatternKind::BindingIdentifier(id) = &p.pattern.kind {
                            Some(id.name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();

                // Get function name if it has one (named function expression)
                // For anonymous functions, generate a unique name so we can call them
                let func_name = if let Some(id) = &func_expr.id {
                    id.name.to_string()
                } else {
                    // Generate unique name for anonymous function
                    let name = format!("__anon_{}", self.anon_func_counter);
                    self.anon_func_counter += 1;
                    name
                };

                // Register the function so it can be called later
                self.functions.insert(func_name.clone(), func_expr);

                let source = emit_function(func_expr);
                values.push(Value::Closure {
                    params,
                    body_id: 0,
                    env: self.env.capture(),
                    source,
                    name: Some(func_name),
                });
            }
            _ => {
                values.push(dynamic(emit_expr(expr)));
            }
        }
        Ok(())
    }

    /// Apply a binary operator to two values
    fn apply_binary_op(&self, operator: BinaryOperator, left: Value, right: Value) -> Result<Value, String> {
        // Both static - compute
        if left.is_static() && right.is_static() {
            match operator {
                BinaryOperator::Addition => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(a + b)),
                    (Value::String(a), Value::String(b)) => return Ok(Value::String(format!("{}{}", a, b))),
                    (Value::String(a), Value::Number(b)) => return Ok(Value::String(format!("{}{}", a, b))),
                    (Value::Number(a), Value::String(b)) => return Ok(Value::String(format!("{}{}", a, b))),
                    _ => {}
                },
                BinaryOperator::Subtraction => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(a - b)),
                    _ => {}
                },
                BinaryOperator::Multiplication => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(a * b)),
                    _ => {}
                },
                BinaryOperator::Division => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(a / b)),
                    _ => {}
                },
                BinaryOperator::LessThan => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Bool(a < b)),
                    _ => {}
                },
                BinaryOperator::GreaterThan => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Bool(a > b)),
                    _ => {}
                },
                BinaryOperator::LessEqualThan => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Bool(a <= b)),
                    _ => {}
                },
                BinaryOperator::GreaterEqualThan => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Bool(a >= b)),
                    _ => {}
                },
                BinaryOperator::Equality | BinaryOperator::StrictEquality => {
                    return Ok(Value::Bool(left == right));
                }
                BinaryOperator::Inequality | BinaryOperator::StrictInequality => {
                    return Ok(Value::Bool(left != right));
                }
                BinaryOperator::Remainder => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(*a as i64 as f64 % *b as i64 as f64)),
                    _ => {}
                },
                BinaryOperator::BitwiseAnd => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number((*a as i32 & *b as i32) as f64)),
                    _ => {}
                },
                BinaryOperator::BitwiseOR => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number((*a as i32 | *b as i32) as f64)),
                    _ => {}
                },
                BinaryOperator::BitwiseXOR => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number((*a as i32 ^ *b as i32) as f64)),
                    _ => {}
                },
                BinaryOperator::ShiftLeft => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(((*a as i32) << (*b as u32 & 0x1f)) as f64)),
                    _ => {}
                },
                BinaryOperator::ShiftRight => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(((*a as i32) >> (*b as u32 & 0x1f)) as f64)),
                    _ => {}
                },
                BinaryOperator::ShiftRightZeroFill => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(((*a as i32 as u32) >> (*b as u32 & 0x1f)) as f64)),
                    _ => {}
                },
                _ => {}
            }
        }

        // At least one side is dynamic - build residual expression
        let left_js = residual_of(&left)?;
        let right_js = residual_of(&right)?;

        let op_str = match operator {
            BinaryOperator::Addition => "+",
            BinaryOperator::Subtraction => "-",
            BinaryOperator::Multiplication => "*",
            BinaryOperator::Division => "/",
            BinaryOperator::LessThan => "<",
            BinaryOperator::GreaterThan => ">",
            BinaryOperator::LessEqualThan => "<=",
            BinaryOperator::GreaterEqualThan => ">=",
            BinaryOperator::Equality => "==",
            BinaryOperator::StrictEquality => "===",
            BinaryOperator::Inequality => "!=",
            BinaryOperator::StrictInequality => "!==",
            BinaryOperator::Remainder => "%",
            BinaryOperator::BitwiseAnd => "&",
            BinaryOperator::BitwiseOR => "|",
            BinaryOperator::BitwiseXOR => "^",
            BinaryOperator::ShiftLeft => "<<",
            BinaryOperator::ShiftRight => ">>",
            BinaryOperator::ShiftRightZeroFill => ">>>",
            _ => return Ok(dynamic(format!("{} ?? {}", left_js, right_js))),
        };

        Ok(dynamic(format!("{} {} {}", left_js, op_str, right_js)))
    }

    /// Apply a unary operator to a value
    fn apply_unary_op(&self, operator: UnaryOperator, arg: Value) -> Result<Value, String> {
        if arg.is_static() {
            match operator {
                UnaryOperator::UnaryNegation => match &arg {
                    Value::Number(n) => return Ok(Value::Number(-n)),
                    _ => {}
                },
                UnaryOperator::LogicalNot => match arg.is_truthy() {
                    Some(b) => return Ok(Value::Bool(!b)),
                    None => {}
                },
                UnaryOperator::UnaryPlus => match &arg {
                    Value::Number(n) => return Ok(Value::Number(*n)),
                    _ => {}
                },
                UnaryOperator::Typeof => {
                    let type_str = match &arg {
                        Value::Number(_) => "number",
                        Value::String(_) => "string",
                        Value::Bool(_) => "boolean",
                        Value::Undefined => "undefined",
                        Value::Null => "object",
                        Value::Array(_) => "object",
                        Value::Object(_) => "object",
                        Value::Closure { .. } => "function",
                        Value::ArrayBuffer { .. } => "object",
                        Value::TypedArray { .. } => "object",
                        Value::DataView { .. } => "object",
                        Value::TextDecoder { .. } => "object",
                        Value::Dynamic(_) => unreachable!(),
                    };
                    return Ok(Value::String(type_str.to_string()));
                }
                _ => {}
            }
        }

        let arg_js = residual_of(&arg)?;
        let op_str = match operator {
            UnaryOperator::UnaryNegation => "-",
            UnaryOperator::UnaryPlus => "+",
            UnaryOperator::LogicalNot => "!",
            UnaryOperator::BitwiseNot => "~",
            UnaryOperator::Typeof => "typeof ",
            UnaryOperator::Void => "void ",
            UnaryOperator::Delete => "delete ",
        };

        Ok(dynamic(format!("{}{}", op_str, arg_js)))
    }

    /// Apply member access
    fn apply_member_access(&self, obj: Value, property: &str) -> Result<Value, String> {
        match (&obj, property) {
            (Value::Array(arr), "length") => Ok(Value::Number(arr.len() as f64)),
            (Value::String(s), "length") => Ok(Value::Number(s.len() as f64)),
            (Value::Object(props), _) => {
                Ok(props.get(property).unwrap_or(Value::Undefined))
            }
            (Value::Dynamic(obj_expr), _) => {
                Ok(dynamic(format!("{}.{}", obj_expr, property)))
            }
            _ => {
                let obj_js = residual_of(&obj)?;
                Ok(dynamic(format!("{}.{}", obj_js, property)))
            }
        }
    }

    /// Apply computed member access
    fn apply_computed_member(&self, obj: Value, idx: Value) -> Result<Value, String> {
        match (&obj, &idx) {
            (Value::Array(arr), Value::Number(n)) => {
                let i = *n as usize;
                Ok(arr.get(i).unwrap_or(Value::Undefined))
            }
            _ => {
                let obj_js = residual_of(&obj)?;
                let idx_js = residual_of(&idx)?;
                Ok(dynamic(format!("{}[{}]", obj_js, idx_js)))
            }
        }
    }

    /// Apply a function call with pre-evaluated arguments
    fn apply_function_call(
        &mut self,
        func_name: &str,
        arg_values: Vec<Value>,
        _call_expr: &'a CallExpression<'a>,
    ) -> Result<Value, String> {
        // If function isn't defined, check if the variable holds a closure with a known function name
        let (func, actual_func_name) = if let Some(func) = self.functions.get(func_name) {
            (*func, func_name.to_string())
        } else {
            // Check if the variable holds a Closure value with a function name
            if let Some(Value::Closure { name: Some(ref closure_func_name), .. }) = self.env.get(func_name) {
                if let Some(func) = self.functions.get(closure_func_name.as_str()) {
                    self.trace.mark_function_called(closure_func_name);
                    (*func, closure_func_name.clone())
                } else {
                    // Closure name doesn't map to a known function
                    let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                    return Ok(dynamic(format!("{}({})", func_name, arg_strs?.join(", "))));
                }
            } else {
                // Not a closure - emit as residual
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                return Ok(dynamic(format!("{}({})", func_name, arg_strs?.join(", "))));
            }
        };

        // Check for recursion - allow up to max_recursion_depth recursive calls
        let recursion_count = self.call_stack.iter().filter(|n| n.as_str() == actual_func_name).count();
        if recursion_count >= self.max_recursion_depth {
            self.trace.mark_function_live(&actual_func_name);
            let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
            return Ok(dynamic(format!("{}({})", actual_func_name, arg_strs?.join(", "))));
        }

        // Push function onto call stack
        self.call_stack.push(actual_func_name.clone());

        // Trace function entry
        let args_info: Vec<(String, bool)> = arg_values.iter()
            .map(|v| (self.value_repr(v), v.is_static()))
            .collect();
        self.trace_event(TraceEvent::FunctionEnter {
            name: actual_func_name.clone(),
            args: args_info,
        });

        // Get params
        let params: Vec<String> = func
            .params
            .items
            .iter()
            .filter_map(|p| {
                if let BindingPatternKind::BindingIdentifier(id) = &p.pattern.kind {
                    Some(id.name.to_string())
                } else {
                    None
                }
            })
            .collect();

        // Save current residual state
        let saved_residual = self.take_residual();

        // Create new scope
        self.env.push_scope();

        // Bind parameters
        for (param, arg) in params.iter().zip(arg_values.iter()) {
            self.env.define(param, arg.clone());
        }

        // IMPORTANT: Hoist var declarations from nested structures (while, switch, if, etc.)
        // JavaScript hoists ALL vars to the nearest function scope
        if let Some(body) = &func.body {
            let mut var_names = Vec::new();
            for stmt in body.statements.iter() {
                Self::collect_var_names(stmt, &mut var_names);
            }
            for name in var_names {
                // Only hoist if not already defined in current scope (e.g., as a parameter)
                // We use exists_in_current_scope because var should create a new local binding
                // even if a variable with the same name exists in an outer scope
                if !self.env.exists_in_current_scope(&name) {
                    self.env.define(&name, dynamic(name.clone()));
                }
            }
        }

        // Execute function body using iterative statement evaluation
        #[allow(unused_assignments)]
        let mut body_result = StmtResult::Continue;
        let mut residual_emitted_at: Option<usize> = None;
        let result = if let Some(body) = &func.body {
            let mut return_val = None;
            for (idx, stmt) in body.statements.iter().enumerate() {
                body_result = self.eval_statement_iterative(stmt)?;
                match &body_result {
                    StmtResult::Return(v) => {
                        return_val = Some(v.clone());
                        break;
                    }
                    StmtResult::Break => {}
                    StmtResult::ResidualEmitted => {
                        residual_emitted_at = Some(idx);
                        break;
                    }
                    _ => {}
                }
            }
            return_val.unwrap_or(Value::Undefined)
        } else {
            Value::Undefined
        };

        // Check if function produced residual statements
        let mut func_residual = self.take_residual();

        // Handle residual emission
        let mut inlined_return_expr: Option<String> = None;
        if let (Some(emit_idx), Some(body)) = (residual_emitted_at, &func.body) {
            for stmt in body.statements.iter().skip(emit_idx + 1) {
                match stmt {
                    Statement::ReturnStatement(ret) => {
                        if let Some(arg) = &ret.argument {
                            inlined_return_expr = Some(emit_expr(arg));
                        }
                    }
                    _ => {
                        let specialized = self.specialize_statement(stmt);
                        func_residual.push(specialized);
                    }
                }
            }
        }

        // Capture scope bindings before popping
        let scope_bindings = self.env.current_scope_bindings();

        self.env.pop_scope();
        self.residual_stmts = saved_residual;

        // If function produced residual, wrap in IIFE
        if !func_residual.is_empty() {
            let mut iife_body = Vec::new();

            let mut bindings_map: std::collections::HashMap<String, Value> =
                scope_bindings.into_iter().collect();

            // Check for program/pc optimization
            if let (Some(Value::Array(arr)), Some(Value::Number(pc_val))) =
                (bindings_map.get("program"), bindings_map.get("pc"))
            {
                let pc_idx = *pc_val as usize;
                if pc_idx > 0 && pc_idx < arr.len() {
                    let truncated: Vec<Value> = arr.to_vec()[pc_idx..].to_vec();
                    bindings_map.insert("program".to_string(), Value::Array(SharedArray::new(truncated)));
                    bindings_map.insert("pc".to_string(), Value::Number(0.0));
                }
            }

            for (name, value) in bindings_map {
                let value_str = residual_of(&value)?;
                iife_body.push(format!("let {} = {};", name, value_str));
            }

            for stmt in &func_residual {
                iife_body.push(stmt.clone());
            }

            let return_expr = inlined_return_expr.unwrap_or_else(|| "undefined".to_string());
            iife_body.push(format!("return {};", return_expr));

            let iife = format!("(() => {{\n{}\n}})()", iife_body.join("\n"));

            // Trace function exit (dynamic/residual)
            self.trace_event(TraceEvent::FunctionExit {
                name: func_name.to_string(),
                result_repr: "<residual IIFE>".to_string(),
                is_static: false,
            });

            self.call_stack.pop();
            return Ok(dynamic(iife));
        }

        // Trace function exit (normal)
        self.trace_event(TraceEvent::FunctionExit {
            name: func_name.to_string(),
            result_repr: self.value_repr(&result),
            is_static: result.is_static(),
        });

        self.call_stack.pop();
        Ok(result)
    }

    /// Apply a method call with pre-evaluated arguments
    /// base_name is the root variable (e.g., "myGlobal")
    /// path is the property chain to the method receiver (e.g., ["listeners"] for myGlobal.listeners.push())
    fn apply_method_call(
        &mut self,
        base_name: &str,
        path: &[String],
        method_name: &str,
        arg_values: Vec<Value>,
        _call_expr: &'a CallExpression<'a>,
    ) -> Result<Value, String> {
        // Build the full path string for error messages/residual
        let full_path = if path.is_empty() {
            base_name.to_string()
        } else {
            format!("{}.{}", base_name, path.join("."))
        };

        // Get the base object
        let base_val = match self.env.get(base_name) {
            Some(v) => v,
            None => {
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                return Ok(dynamic(format!("{}.{}({})", full_path, method_name, arg_strs?.join(", "))));
            }
        };

        // Navigate the path to get the actual receiver
        let mut current = base_val.clone();
        for prop in path.iter() {
            match &current {
                Value::Object(map) => {
                    if let Some(v) = map.get(prop) {
                        current = v.clone();
                    } else {
                        let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                        return Ok(dynamic(format!("{}.{}({})", full_path, method_name, arg_strs?.join(", "))));
                    }
                }
                _ => {
                    let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                    return Ok(dynamic(format!("{}.{}({})", full_path, method_name, arg_strs?.join(", "))));
                }
            }
        }

        let obj_val = current;

        match (&obj_val, method_name) {
            (Value::Array(arr), "push") => {
                let mut new_arr = arr.clone();
                for val in arg_values {
                    new_arr.push(val);
                }
                let new_len = Value::Number(new_arr.len() as f64);
                // Now we need to update the nested property in the base object
                self.update_nested_property(base_name, path, Value::Array(new_arr));
                Ok(new_len)
            }
            (Value::Array(arr), "pop") => {
                let mut new_arr = arr.clone();
                let popped = new_arr.pop().unwrap_or(Value::Undefined);
                self.update_nested_property(base_name, path, Value::Array(new_arr));
                Ok(popped)
            }
            (Value::Array(arr), "length") => Ok(Value::Number(arr.len() as f64)),
            (Value::TextDecoder { encoding }, "decode") => {
                // Handle TextDecoder.decode(typedArray)
                if let Some(Value::TypedArray { kind, buffer, byte_offset, length }) = arg_values.first() {
                    // Get the raw bytes from the typed array view
                    let element_size = kind.element_size();
                    let byte_length = length * element_size;
                    let data = buffer.get_bytes(*byte_offset, byte_length);

                    match encoding.as_str() {
                        "utf-8" | "utf8" => {
                            match String::from_utf8(data.clone()) {
                                Ok(s) => return Ok(Value::String(s)),
                                Err(_) => {
                                    let s = String::from_utf8_lossy(&data).to_string();
                                    return Ok(Value::String(s));
                                }
                            }
                        }
                        _ => {
                            // Unsupported encoding
                        }
                    }
                }
                // Fall through to emit as residual
                let obj_js = residual_of(&obj_val)?;
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                Ok(dynamic(format!("{}.{}({})", obj_js, method_name, arg_strs?.join(", "))))
            }
            (Value::DataView { buffer, byte_offset, .. }, method)
                if method == "getFloat64" || method == "getFloat32" ||
                   method == "getInt8" || method == "getUint8" ||
                   method == "getInt16" || method == "getUint16" ||
                   method == "getInt32" || method == "getUint32" => {
                // Handle DataView.getXXX(offset, littleEndian) methods
                if let Some(Value::Number(offset)) = arg_values.first() {
                    let rel_offset = *byte_offset + (*offset as usize);
                    let little_endian = match arg_values.get(1) {
                        Some(Value::Bool(b)) => *b,
                        _ => false, // Default is big-endian
                    };
                    if let Some(result) = read_dataview_value(buffer, rel_offset, method, little_endian) {
                        return Ok(result);
                    }
                }
                // Fall through to emit as residual
                let obj_js = residual_of(&obj_val)?;
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                Ok(dynamic(format!("{}.{}({})", obj_js, method_name, arg_strs?.join(", "))))
            }
            (Value::Dynamic(obj_expr), _) => {
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                Ok(dynamic(format!("{}.{}({})", obj_expr, method_name, arg_strs?.join(", "))))
            }
            // Handle calling a closure stored in an object: obj.getValue() where getValue is a Closure
            (Value::Object(map), _) => {
                if let Some(prop_val) = map.get(method_name) {
                    if let Value::Closure { name: Some(ref func_name), .. } = prop_val {
                        // Check if we have the function definition
                        if self.functions.contains_key(func_name.as_str()) {
                            // Call the function
                            self.trace.mark_function_called(func_name);
                            return self.apply_function_call(func_name, arg_values, _call_expr);
                        }
                    }
                }
                // Fall through to emit as residual
                let obj_js = residual_of(&obj_val)?;
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                Ok(dynamic(format!("{}.{}({})", obj_js, method_name, arg_strs?.join(", "))))
            }
            _ => {
                let obj_js = residual_of(&obj_val)?;
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                Ok(dynamic(format!("{}.{}({})", obj_js, method_name, arg_strs?.join(", "))))
            }
        }
    }

    /// Extract the base variable name and property path from a member expression chain
    /// e.g., for `myGlobal.listeners`, returns (Some("myGlobal"), ["listeners"])
    /// Returns (None, []) if the chain cannot be resolved to a variable
    fn extract_member_chain(expr: &Expression<'_>) -> (Option<String>, Vec<String>) {
        match expr {
            Expression::Identifier(id) => (Some(id.name.to_string()), vec![]),
            Expression::StaticMemberExpression(member) => {
                let (base, mut path) = Self::extract_member_chain(&member.object);
                path.push(member.property.name.to_string());
                (base, path)
            }
            _ => (None, vec![]),
        }
    }

    /// Update a nested property within an object stored in the environment
    /// e.g., update_nested_property("myGlobal", ["listeners"], new_array)
    /// will update myGlobal.listeners to new_array
    fn update_nested_property(&mut self, base_name: &str, path: &[String], new_value: Value) {
        if path.is_empty() {
            // No nesting, just set directly
            self.env.set(base_name, new_value);
            return;
        }

        // Get the base object
        let base_val = match self.env.get(base_name) {
            Some(v) => v,
            None => return,
        };

        // Clone and modify the nested structure
        let updated = Self::update_value_at_path(base_val, path, new_value);
        self.env.set(base_name, updated);
    }

    /// Recursively update a value at a given path
    fn update_value_at_path(val: Value, path: &[String], new_value: Value) -> Value {
        if path.is_empty() {
            return new_value;
        }

        match val {
            Value::Object(map) => {
                let first = &path[0];
                let rest = &path[1..];
                if rest.is_empty() {
                    // This is the final property, set it directly
                    map.set(first.clone(), new_value);
                } else {
                    // Need to recurse deeper
                    if let Some(child) = map.get(first) {
                        let updated_child = Self::update_value_at_path(child.clone(), rest, new_value);
                        map.set(first.clone(), updated_child);
                    }
                }
                Value::Object(map)
            }
            _ => val, // Can't navigate into non-objects
        }
    }

    /// Apply an assignment
    fn apply_assignment(
        &mut self,
        target: AssignTargetInfo,
        operator: AssignmentOperator,
        right_value: Value,
        original_expr: &AssignmentExpression<'_>,
    ) -> Result<Value, String> {
        match target {
            AssignTargetInfo::Identifier(name) => {
                let final_value = match operator {
                    AssignmentOperator::Assign => right_value,
                    AssignmentOperator::Addition => {
                        let left = self.env.get(&name).unwrap_or_else(|| dynamic(name.clone()));
                        match (&left, &right_value) {
                            (Value::Number(a), Value::Number(b)) => Value::Number(a + b),
                            (Value::String(a), Value::String(b)) => Value::String(format!("{}{}", a, b)),
                            _ => {
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right_value)?;
                                dynamic(format!("{} + {}", left_js, right_js))
                            }
                        }
                    }
                    AssignmentOperator::Subtraction => {
                        let left = self.env.get(&name).unwrap_or_else(|| dynamic(name.clone()));
                        match (&left, &right_value) {
                            (Value::Number(a), Value::Number(b)) => Value::Number(a - b),
                            _ => {
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right_value)?;
                                dynamic(format!("{} - {}", left_js, right_js))
                            }
                        }
                    }
                    AssignmentOperator::Multiplication => {
                        let left = self.env.get(&name).unwrap_or_else(|| dynamic(name.clone()));
                        match (&left, &right_value) {
                            (Value::Number(a), Value::Number(b)) => Value::Number(a * b),
                            _ => {
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right_value)?;
                                dynamic(format!("{} * {}", left_js, right_js))
                            }
                        }
                    }
                    AssignmentOperator::Division => {
                        let left = self.env.get(&name).unwrap_or_else(|| dynamic(name.clone()));
                        match (&left, &right_value) {
                            (Value::Number(a), Value::Number(b)) => Value::Number(a / b),
                            _ => {
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right_value)?;
                                dynamic(format!("{} / {}", left_js, right_js))
                            }
                        }
                    }
                    _ => dynamic(emit_assign_expr(original_expr)),
                };

                if !self.env.set(&name, final_value.clone()) {
                    return Ok(dynamic(emit_assign_expr(original_expr)));
                }
                Ok(final_value)
            }
            AssignTargetInfo::ComputedMember { base, indices } => {
                if operator != AssignmentOperator::Assign {
                    return Ok(dynamic(emit_assign_expr(original_expr)));
                }

                if let Some(base_value) = self.env.get(&base) {
                    if let Some(updated) = Self::update_nested_value(&base_value, &indices, &right_value) {
                        self.env.set(&base, updated);
                        return Ok(right_value);
                    }
                }
                Ok(dynamic(emit_assign_expr(original_expr)))
            }
            AssignTargetInfo::StaticMember { object, property } => {
                if operator != AssignmentOperator::Assign {
                    return Ok(dynamic(emit_assign_expr(original_expr)));
                }

                // Get the object and update the property
                // With SharedObject, the mutation is visible to all aliases
                if let Some(obj_val) = self.env.get(&object) {
                    if let Value::Object(props) = obj_val {
                        props.set(property.clone(), right_value.clone());
                        return Ok(right_value);
                    }
                }
                Ok(dynamic(emit_assign_expr(original_expr)))
            }
        }
    }

    /// Apply an update expression (++/--)
    fn apply_update(&mut self, name: &str, operator: UpdateOperator, prefix: bool) -> Result<Value, String> {
        let current = match self.env.get(name) {
            Some(v) => v,
            None => return Ok(dynamic(format!("{}++", name))), // fallback
        };

        match &current {
            Value::Number(n) => {
                let new_val = match operator {
                    UpdateOperator::Increment => Value::Number(n + 1.0),
                    UpdateOperator::Decrement => Value::Number(n - 1.0),
                };
                self.env.set(name, new_val.clone());
                if prefix {
                    Ok(new_val)
                } else {
                    Ok(Value::Number(*n))
                }
            }
            Value::Dynamic(_) => {
                let op = match operator {
                    UpdateOperator::Increment => "++",
                    UpdateOperator::Decrement => "--",
                };
                let residual = if prefix {
                    format!("{}{}", op, name)
                } else {
                    format!("{}{}", name, op)
                };
                self.env.set(name, dynamic(name.to_string()));
                Ok(dynamic(residual))
            }
            _ => Ok(dynamic(format!("{}++", name))),
        }
    }

    // ========================================================================
    // Iterative Statement Evaluation
    // ========================================================================

    /// Iteratively evaluate a statement using an explicit work stack.
    /// This avoids Rust stack overflow for deeply nested/recursive JS code.
    fn eval_statement_iterative(&mut self, stmt: &'a Statement<'a>) -> Result<StmtResult, String> {
        let mut work: Vec<StmtWork<'a>> = vec![StmtWork::Eval(stmt)];
        let mut result = StmtResult::Continue;
        let mut expr_result: Option<Value> = None;

        while let Some(item) = work.pop() {
            // Check if we need to unwind due to control flow
            match &result {
                StmtResult::Return(_) | StmtResult::Break | StmtResult::Throw(_) => {
                    // Check if this work item is a marker we should stop at
                    match &item {
                        StmtWork::FunctionCallFinish { .. } => {
                            // Stop unwinding - function call caught the return
                        }
                        StmtWork::WhileCheck { .. } | StmtWork::ForCheck { .. } => {
                            // Break stops here
                            if matches!(result, StmtResult::Break) {
                                result = StmtResult::Continue;
                                continue;
                            }
                        }
                        StmtWork::WhileExit => {
                            self.depth -= 1;
                            continue;
                        }
                        StmtWork::PopScope => {
                            self.env.pop_scope();
                            continue;
                        }
                        _ => continue, // Skip other work items during unwind
                    }
                }
                StmtResult::ContinueLoop => {
                    // Continue unwinds to the loop check
                    match &item {
                        StmtWork::WhileBodyDone { while_stmt, iterations, original_stmt } => {
                            // Resume the while loop
                            result = StmtResult::Continue;
                            work.push(StmtWork::WhileCheck {
                                while_stmt,
                                iterations: *iterations + 1,
                                original_stmt,
                            });
                            continue;
                        }
                        StmtWork::ForUpdate { for_stmt, iterations, original_stmt } => {
                            // Resume the for loop at update
                            result = StmtResult::Continue;
                            // Execute update then check
                            if let Some(update) = &for_stmt.update {
                                self.eval_expression_iterative(update)?;
                            }
                            work.push(StmtWork::ForCheck {
                                for_stmt,
                                iterations: *iterations + 1,
                                original_stmt,
                            });
                            continue;
                        }
                        StmtWork::WhileExit => {
                            self.depth -= 1;
                            continue;
                        }
                        StmtWork::PopScope => {
                            self.env.pop_scope();
                            continue;
                        }
                        _ => continue,
                    }
                }
                StmtResult::ResidualEmitted => {
                    // ResidualEmitted should propagate up
                    match &item {
                        StmtWork::WhileExit => {
                            self.depth -= 1;
                            continue;
                        }
                        StmtWork::PopScope => {
                            self.env.pop_scope();
                            continue;
                        }
                        _ => continue,
                    }
                }
                _ => {}
            }

            match item {
                StmtWork::Eval(s) => {
                    result = self.push_stmt_work(s, &mut work, &mut expr_result)?;
                }
                StmtWork::BlockContinue { stmts, idx } => {
                    if idx < stmts.len() {
                        // More statements to process
                        work.push(StmtWork::BlockContinue { stmts, idx: idx + 1 });
                        work.push(StmtWork::Eval(&stmts[idx]));
                    }
                }
                StmtWork::PopScope => {
                    self.env.pop_scope();
                }
                StmtWork::IfBranch { consequent, alternate, original_stmt: _ } => {
                    let cond = expr_result.take().ok_or("Missing condition value")?;
                    match cond.is_truthy() {
                        Some(true) => {
                            work.push(StmtWork::Eval(consequent));
                        }
                        Some(false) => {
                            if let Some(alt) = alternate {
                                work.push(StmtWork::Eval(alt));
                            }
                        }
                        None => {
                            // Dynamic condition - emit residual if
                            let cond_residual = residual_of(&cond)?;
                            let saved_residual = self.take_residual();
                            let saved_env = self.env.capture();

                            // Evaluate consequent
                            let _cons_result = self.eval_statement_iterative(consequent)?;
                            let cons_residual = self.take_residual();
                            let cons_body = if cons_residual.is_empty() {
                                self.specialize_statement(consequent)
                            } else {
                                cons_residual.join("\n")
                            };

                            // Restore and evaluate alternate
                            self.env = saved_env.capture();
                            let alt_body = if let Some(alt) = alternate {
                                let _ = self.eval_statement_iterative(alt)?;
                                let alt_residual = self.take_residual();
                                if alt_residual.is_empty() {
                                    Some(self.specialize_statement(alt))
                                } else {
                                    Some(alt_residual.join("\n"))
                                }
                            } else {
                                None
                            };

                            self.residual_stmts = saved_residual;
                            let if_residual = if let Some(alt) = alt_body {
                                format!("if ({}) {{\n{}\n}} else {{\n{}\n}}", cond_residual, cons_body, alt)
                            } else {
                                format!("if ({}) {{\n{}\n}}", cond_residual, cons_body)
                            };
                            self.emit_residual(if_residual);
                        }
                    }
                }
                StmtWork::WhileCheck { while_stmt, iterations, original_stmt } => {
                    if iterations >= self.max_iterations {
                        let remaining = self.build_while_residual(while_stmt);
                        self.emit_residual(remaining);
                        self.depth -= 1;
                        result = StmtResult::Continue;
                        continue;
                    }

                    let cond = self.eval_expression_iterative(&while_stmt.test)?;
                    match cond.is_truthy() {
                        Some(true) => {
                            work.push(StmtWork::WhileBodyDone {
                                while_stmt,
                                iterations,
                                original_stmt,
                            });
                            work.push(StmtWork::Eval(&while_stmt.body));
                        }
                        Some(false) => {
                            self.depth -= 1;
                        }
                        None => {
                            let remaining = self.build_while_residual(while_stmt);
                            self.emit_residual(remaining);
                            self.depth -= 1;
                            result = StmtResult::ResidualEmitted;
                        }
                    }
                }
                StmtWork::WhileBodyDone { while_stmt, iterations, original_stmt } => {
                    // Body completed normally, loop back
                    work.push(StmtWork::WhileCheck {
                        while_stmt,
                        iterations: iterations + 1,
                        original_stmt,
                    });
                }
                StmtWork::WhileExit => {
                    self.depth -= 1;
                }
                StmtWork::SwitchCase { switch_stmt, discriminant, case_idx, matched, fell_through } => {
                    if case_idx >= switch_stmt.cases.len() {
                        // Done with all cases
                        if !matched {
                            // Handle default case
                            for case in &switch_stmt.cases {
                                if case.test.is_none() {
                                    for case_stmt in &case.consequent {
                                        let case_result = self.eval_statement_iterative(case_stmt)?;
                                        match case_result {
                                            StmtResult::Return(v) => {
                                                result = StmtResult::Return(v);
                                                break;
                                            }
                                            StmtResult::Break => break,
                                            _ => {}
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        continue;
                    }

                    let case = &switch_stmt.cases[case_idx];
                    let should_execute = matched || fell_through;

                    if should_execute {
                        // Execute case statements
                        let mut new_fell_through = true;
                        for case_stmt in &case.consequent {
                            let case_result = self.eval_statement_iterative(case_stmt)?;
                            match case_result {
                                StmtResult::Return(v) => {
                                    result = StmtResult::Return(v);
                                    return Ok(result);
                                }
                                StmtResult::Break => {
                                    new_fell_through = false;
                                    result = StmtResult::Continue;
                                    break;
                                }
                                _ => {}
                            }
                        }
                        if new_fell_through {
                            work.push(StmtWork::SwitchCase {
                                switch_stmt,
                                discriminant,
                                case_idx: case_idx + 1,
                                matched: true,
                                fell_through: true,
                            });
                        }
                    } else if let Some(test) = &case.test {
                        let test_val = self.eval_expression_iterative(test)?;
                        if discriminant == test_val {
                            // Match found
                            let mut new_fell_through = true;
                            for case_stmt in &case.consequent {
                                let case_result = self.eval_statement_iterative(case_stmt)?;
                                match case_result {
                                    StmtResult::Return(v) => {
                                        result = StmtResult::Return(v);
                                        return Ok(result);
                                    }
                                    StmtResult::Break => {
                                        new_fell_through = false;
                                        result = StmtResult::Continue;
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                            if new_fell_through {
                                work.push(StmtWork::SwitchCase {
                                    switch_stmt,
                                    discriminant,
                                    case_idx: case_idx + 1,
                                    matched: true,
                                    fell_through: true,
                                });
                            }
                        } else {
                            work.push(StmtWork::SwitchCase {
                                switch_stmt,
                                discriminant,
                                case_idx: case_idx + 1,
                                matched: false,
                                fell_through: false,
                            });
                        }
                    } else {
                        // Default case - skip for now, handle at end
                        work.push(StmtWork::SwitchCase {
                            switch_stmt,
                            discriminant,
                            case_idx: case_idx + 1,
                            matched: false,
                            fell_through: false,
                        });
                    }
                }
                StmtWork::ForCheck { for_stmt, iterations, original_stmt } => {
                    if iterations >= self.max_iterations {
                        self.emit_residual(emit_stmt(original_stmt));
                        result = StmtResult::Continue;
                        continue;
                    }

                    let should_continue = if let Some(test) = &for_stmt.test {
                        let cond = self.eval_expression_iterative(test)?;
                        match cond.is_truthy() {
                            Some(true) => true,
                            Some(false) => false,
                            None => {
                                self.emit_residual(emit_stmt(original_stmt));
                                result = StmtResult::ResidualEmitted;
                                continue;
                            }
                        }
                    } else {
                        true
                    };

                    if should_continue {
                        work.push(StmtWork::ForUpdate {
                            for_stmt,
                            iterations,
                            original_stmt,
                        });
                        work.push(StmtWork::Eval(&for_stmt.body));
                    }
                }
                StmtWork::ForUpdate { for_stmt, iterations, original_stmt } => {
                    if let Some(update) = &for_stmt.update {
                        self.eval_expression_iterative(update)?;
                    }
                    work.push(StmtWork::ForCheck {
                        for_stmt,
                        iterations: iterations + 1,
                        original_stmt,
                    });
                }
                StmtWork::FunctionCallBody { .. } | StmtWork::FunctionCallFinish { .. } => {
                    // These are handled in apply_function_call
                }
                StmtWork::ReturnValue => {
                    // Handled inline
                }
                StmtWork::TryBlock { .. } | StmtWork::CatchBlock { .. } => {
                    // Try-catch is complex, delegate to recursive for now
                }
            }
        }

        Ok(result)
    }

    /// Push work items for evaluating a statement
    fn push_stmt_work(
        &mut self,
        stmt: &'a Statement<'a>,
        work: &mut Vec<StmtWork<'a>>,
        expr_result: &mut Option<Value>,
    ) -> Result<StmtResult, String> {
        match stmt {
            Statement::VariableDeclaration(decl) => {
                self.eval_variable_declaration_inner(decl)?;
                Ok(StmtResult::Continue)
            }
            Statement::ExpressionStatement(expr_stmt) => {
                let value = self.eval_expression_iterative(&expr_stmt.expression)?;

                // Check for side effects that need preservation
                let must_preserve = match &expr_stmt.expression {
                    Expression::AssignmentExpression(assign) => {
                        match &assign.left {
                            AssignmentTarget::AssignmentTargetIdentifier(id) => {
                                let name = id.name.to_string();
                                !self.env.exists(&name)
                            }
                            AssignmentTarget::ComputedMemberExpression(_) => value.is_dynamic(),
                            _ => true,
                        }
                    }
                    Expression::CallExpression(call) => {
                        if let Expression::Identifier(id) = &call.callee {
                            let func_name = id.name.to_string();
                            // Check if it's our function OR a variable holding a closure to our function
                            let is_known_function = self.functions.contains_key(&func_name)
                                || matches!(
                                    self.env.get(&func_name),
                                    Some(Value::Closure { name: Some(ref n), .. }) if self.functions.contains_key(n.as_str())
                                );
                            !is_known_function
                        } else if let Expression::StaticMemberExpression(member) = &call.callee {
                            let method = member.property.name.to_string();
                            if matches!(method.as_str(), "push" | "pop" | "shift" | "unshift") {
                                // Use extract_member_chain to handle nested paths like myGlobal.listeners.push()
                                let (base_name, _path) = Self::extract_member_chain(&member.object);
                                match base_name {
                                    Some(name) => self.env.get(&name).is_none(),
                                    None => true,
                                }
                            } else {
                                true
                            }
                        } else {
                            true
                        }
                    }
                    Expression::UpdateExpression(update) => {
                        match &update.argument {
                            SimpleAssignmentTarget::AssignmentTargetIdentifier(id) => {
                                !self.env.exists(&id.name.to_string())
                            }
                            _ => true,
                        }
                    }
                    _ => false,
                };

                if must_preserve {
                    // Use the evaluated value's residual representation if dynamic,
                    // otherwise use the original statement
                    if let Value::Dynamic(residual_expr) = &value {
                        self.emit_residual(residual_expr.clone());
                    } else {
                        self.emit_residual(emit_stmt(stmt));
                    }
                }

                Ok(StmtResult::Continue)
            }
            Statement::FunctionDeclaration(func) => {
                self.register_function(func)?;
                Ok(StmtResult::Continue)
            }
            Statement::BlockStatement(block) => {
                self.env.push_scope();
                work.push(StmtWork::PopScope);
                if !block.body.is_empty() {
                    work.push(StmtWork::BlockContinue {
                        stmts: &block.body,
                        idx: 1,
                    });
                    work.push(StmtWork::Eval(&block.body[0]));
                }
                Ok(StmtResult::Continue)
            }
            Statement::ReturnStatement(ret) => {
                if let Some(arg) = &ret.argument {
                    let val = self.eval_expression_iterative(arg)?;
                    Ok(StmtResult::Return(val))
                } else {
                    Ok(StmtResult::Return(Value::Undefined))
                }
            }
            Statement::IfStatement(if_stmt) => {
                *expr_result = Some(self.eval_expression_iterative(&if_stmt.test)?);
                work.push(StmtWork::IfBranch {
                    consequent: &if_stmt.consequent,
                    alternate: if_stmt.alternate.as_ref(),
                    original_stmt: stmt,
                });
                Ok(StmtResult::Continue)
            }
            Statement::WhileStatement(while_stmt) => {
                self.depth += 1;

                // Use the configurable max_recursion_depth for depth limit
                if self.depth > self.max_recursion_depth {
                    self.depth -= 1;
                    let remaining = self.build_while_residual(while_stmt);
                    self.emit_residual(remaining);
                    return Ok(StmtResult::ResidualEmitted);
                }

                work.push(StmtWork::WhileCheck {
                    while_stmt,
                    iterations: 0,
                    original_stmt: stmt,
                });
                Ok(StmtResult::Continue)
            }
            Statement::SwitchStatement(switch_stmt) => {
                let discriminant = self.eval_expression_iterative(&switch_stmt.discriminant)?;

                if discriminant.is_dynamic() {
                    return Ok(StmtResult::ResidualEmitted);
                }

                work.push(StmtWork::SwitchCase {
                    switch_stmt,
                    discriminant,
                    case_idx: 0,
                    matched: false,
                    fell_through: false,
                });
                Ok(StmtResult::Continue)
            }
            Statement::ForStatement(for_stmt) => {
                // Execute init
                if let Some(init) = &for_stmt.init {
                    match init {
                        ForStatementInit::VariableDeclaration(decl) => {
                            for declarator in &decl.declarations {
                                if let BindingPatternKind::BindingIdentifier(id) = &declarator.id.kind {
                                    let name = id.name.to_string();
                                    let value = if let Some(init_expr) = &declarator.init {
                                        self.eval_expression_iterative(init_expr)?
                                    } else {
                                        Value::Undefined
                                    };
                                    self.env.define(&name, value);
                                }
                            }
                        }
                        _ => {
                            self.eval_expression_iterative(init.to_expression())?;
                        }
                    }
                }

                work.push(StmtWork::ForCheck {
                    for_stmt,
                    iterations: 0,
                    original_stmt: stmt,
                });
                Ok(StmtResult::Continue)
            }
            Statement::BreakStatement(_) => Ok(StmtResult::Break),
            Statement::ContinueStatement(_) => Ok(StmtResult::ContinueLoop),
            Statement::ThrowStatement(throw_stmt) => {
                let val = self.eval_expression_iterative(&throw_stmt.argument)?;
                Ok(StmtResult::Throw(val))
            }
            Statement::TryStatement(try_stmt) => {
                // Delegate to recursive version for try-catch
                self.eval_try_statement(try_stmt, stmt)
            }
            _ => {
                self.emit_residual(emit_stmt(stmt));
                Ok(StmtResult::Continue)
            }
        }
    }

    fn eval_binary_expression(&mut self, bin: &'a BinaryExpression<'a>) -> Result<Value, String> {
        let left = self.eval_expression(&bin.left)?;
        let right = self.eval_expression(&bin.right)?;

        // Both static - compute
        if left.is_static() && right.is_static() {
            match bin.operator {
                BinaryOperator::Addition => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(a + b)),
                    (Value::String(a), Value::String(b)) => return Ok(Value::String(format!("{}{}", a, b))),
                    (Value::String(a), Value::Number(b)) => return Ok(Value::String(format!("{}{}", a, b))),
                    (Value::Number(a), Value::String(b)) => return Ok(Value::String(format!("{}{}", a, b))),
                    _ => {}
                },
                BinaryOperator::Subtraction => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(a - b)),
                    _ => {}
                },
                BinaryOperator::Multiplication => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(a * b)),
                    _ => {}
                },
                BinaryOperator::Division => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(a / b)),
                    _ => {}
                },
                BinaryOperator::LessThan => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Bool(a < b)),
                    _ => {}
                },
                BinaryOperator::GreaterThan => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Bool(a > b)),
                    _ => {}
                },
                BinaryOperator::LessEqualThan => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Bool(a <= b)),
                    _ => {}
                },
                BinaryOperator::GreaterEqualThan => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Bool(a >= b)),
                    _ => {}
                },
                BinaryOperator::Equality | BinaryOperator::StrictEquality => {
                    return Ok(Value::Bool(left == right));
                }
                BinaryOperator::Inequality | BinaryOperator::StrictInequality => {
                    return Ok(Value::Bool(left != right));
                }
                BinaryOperator::Remainder => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(*a as i64 as f64 % *b as i64 as f64)),
                    _ => {}
                },
                BinaryOperator::BitwiseAnd => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number((*a as i32 & *b as i32) as f64)),
                    _ => {}
                },
                BinaryOperator::BitwiseOR => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number((*a as i32 | *b as i32) as f64)),
                    _ => {}
                },
                BinaryOperator::BitwiseXOR => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number((*a as i32 ^ *b as i32) as f64)),
                    _ => {}
                },
                BinaryOperator::ShiftLeft => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(((*a as i32) << (*b as u32 & 0x1f)) as f64)),
                    _ => {}
                },
                BinaryOperator::ShiftRight => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(((*a as i32) >> (*b as u32 & 0x1f)) as f64)),
                    _ => {}
                },
                BinaryOperator::ShiftRightZeroFill => match (&left, &right) {
                    (Value::Number(a), Value::Number(b)) => return Ok(Value::Number(((*a as i32 as u32) >> (*b as u32 & 0x1f)) as f64)),
                    _ => {}
                },
                _ => {}
            }
        }

        // At least one side is dynamic - build residual expression
        let left_js = residual_of(&left)?;
        let right_js = residual_of(&right)?;

        let op_str = match bin.operator {
            BinaryOperator::Addition => "+",
            BinaryOperator::Subtraction => "-",
            BinaryOperator::Multiplication => "*",
            BinaryOperator::Division => "/",
            BinaryOperator::LessThan => "<",
            BinaryOperator::GreaterThan => ">",
            BinaryOperator::LessEqualThan => "<=",
            BinaryOperator::GreaterEqualThan => ">=",
            BinaryOperator::Equality => "==",
            BinaryOperator::StrictEquality => "===",
            BinaryOperator::Inequality => "!=",
            BinaryOperator::StrictInequality => "!==",
            BinaryOperator::Remainder => "%",
            BinaryOperator::BitwiseAnd => "&",
            BinaryOperator::BitwiseOR => "|",
            BinaryOperator::BitwiseXOR => "^",
            BinaryOperator::ShiftLeft => "<<",
            BinaryOperator::ShiftRight => ">>",
            BinaryOperator::ShiftRightZeroFill => ">>>",
            _ => {
                // Unknown operator - emit original expression
                return Ok(dynamic(emit_binary_expr(bin)));
            }
        };

        Ok(dynamic(format!("{} {} {}", left_js, op_str, right_js)))
    }

    fn eval_unary_expression(&mut self, unary: &'a UnaryExpression<'a>) -> Result<Value, String> {
        let arg = self.eval_expression(&unary.argument)?;

        // Static argument - compute
        if arg.is_static() {
            match unary.operator {
                UnaryOperator::UnaryNegation => match &arg {
                    Value::Number(n) => return Ok(Value::Number(-n)),
                    _ => {}
                },
                UnaryOperator::LogicalNot => match arg.is_truthy() {
                    Some(b) => return Ok(Value::Bool(!b)),
                    None => {}
                },
                UnaryOperator::UnaryPlus => match &arg {
                    Value::Number(n) => return Ok(Value::Number(*n)),
                    _ => {}
                },
                UnaryOperator::Typeof => {
                    let type_str = match &arg {
                        Value::Number(_) => "number",
                        Value::String(_) => "string",
                        Value::Bool(_) => "boolean",
                        Value::Undefined => "undefined",
                        Value::Null => "object", // typeof null === "object" in JS
                        Value::Array(_) => "object",
                        Value::Object(_) => "object",
                        Value::Closure { .. } => "function",
                        Value::ArrayBuffer { .. } => "object",
                        Value::TypedArray { .. } => "object",
                        Value::DataView { .. } => "object",
                        Value::TextDecoder { .. } => "object",
                        Value::Dynamic(_) => unreachable!(),
                    };
                    return Ok(Value::String(type_str.to_string()));
                }
                _ => {}
            }
        }

        // Dynamic - build residual expression
        let arg_js = residual_of(&arg)?;
        let op_str = match unary.operator {
            UnaryOperator::UnaryNegation => "-",
            UnaryOperator::UnaryPlus => "+",
            UnaryOperator::LogicalNot => "!",
            UnaryOperator::BitwiseNot => "~",
            UnaryOperator::Typeof => "typeof ",
            UnaryOperator::Void => "void ",
            UnaryOperator::Delete => "delete ",
        };

        Ok(dynamic(format!("{}{}", op_str, arg_js)))
    }

    fn eval_call_expression(&mut self, call: &'a CallExpression<'a>) -> Result<Value, String> {
        // Handle method calls like arr.push(x)
        if let Expression::StaticMemberExpression(member) = &call.callee {
            return self.eval_method_call(member, &call.arguments, call);
        }

        // Handle IIFE (Immediately Invoked Function Expression)
        // Unwrap parenthesized expressions to find the function
        let mut callee = &call.callee;
        while let Expression::ParenthesizedExpression(paren) = callee {
            callee = &paren.expression;
        }
        if let Expression::FunctionExpression(func_expr) = callee {
            // Try to evaluate IIFE; if it fails (e.g., spread args), preserve original
            match self.eval_iife(func_expr, &call.arguments) {
                Ok(val) => return Ok(val),
                Err(_) => return Ok(dynamic(emit_call_expr(call))),
            }
        }

        // Handle regular function calls
        if let Expression::Identifier(id) = &call.callee {
            let func_name = id.name.to_string();

            // Mark function as called in trace
            self.trace.mark_function_called(&func_name);

            // Evaluate arguments first (before borrowing function)
            let mut arg_values = Vec::new();
            let mut has_spread = false;
            for arg in &call.arguments {
                match arg {
                    Argument::SpreadElement(_) => {
                        // Spread arguments make the whole call dynamic
                        has_spread = true;
                        break;
                    }
                    _ => {
                        let val = self.eval_expression(arg.to_expression())?;
                        arg_values.push(val);
                    }
                }
            }

            if has_spread {
                // Can't inline call with spread - preserve it
                self.trace.mark_function_live(&func_name);
                return Ok(dynamic(emit_call_expr(call)));
            }

            // If function isn't a declaration, check if the variable holds a closure with a known function name
            let func = if let Some(func) = self.functions.get(&func_name) {
                *func
            } else {
                // Check if the variable holds a Closure value with a function name
                if let Some(Value::Closure { name: Some(ref closure_func_name), .. }) = self.env.get(&func_name) {
                    if let Some(func) = self.functions.get(closure_func_name) {
                        self.trace.mark_function_called(closure_func_name);
                        *func
                    } else {
                        // Closure name doesn't map to a known function
                        let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                        return Ok(dynamic(format!("{}({})", func_name, arg_strs?.join(", "))));
                    }
                } else {
                    // Build the call expression as residual
                    let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                    return Ok(dynamic(format!("{}({})", func_name, arg_strs?.join(", "))));
                }
            };

            // Check for recursion - allow up to max_recursion_depth recursive calls
            let recursion_count = self.call_stack.iter().filter(|n| n.as_str() == func_name).count();
            if recursion_count >= self.max_recursion_depth {
                // Too deep - emit as residual
                self.trace.mark_function_live(&func_name);
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                return Ok(dynamic(format!("{}({})", func_name, arg_strs?.join(", "))));
            }

            // Push function onto call stack before evaluating
            self.call_stack.push(func_name.clone());

            // Get params
            let params: Vec<String> = func
                .params
                .items
                .iter()
                .filter_map(|p| {
                    if let BindingPatternKind::BindingIdentifier(id) = &p.pattern.kind {
                        Some(id.name.to_string())
                    } else {
                        None
                    }
                })
                .collect();

            // Save current residual state
            let saved_residual = self.take_residual();

            // Create new scope for function execution
            self.env.push_scope();

            // Bind parameters to arguments
            for (param, arg) in params.iter().zip(arg_values.iter()) {
                self.env.define(param, arg.clone());
            }

            // IMPORTANT: Hoist var declarations from nested structures (while, switch, if, etc.)
            // JavaScript hoists ALL vars to the nearest function scope
            if let Some(body) = &func.body {
                let mut var_names = Vec::new();
                for stmt in body.statements.iter() {
                    Self::collect_var_names(stmt, &mut var_names);
                }
                for name in var_names {
                    // Only hoist if not already defined in current scope (e.g., as a parameter)
                    // We use exists_in_current_scope because var should create a new local binding
                    // even if a variable with the same name exists in an outer scope
                    if !self.env.exists_in_current_scope(&name) {
                        self.env.define(&name, dynamic(name.clone()));
                    }
                }
            }

            // Execute function body
            #[allow(unused_assignments)]
            let mut body_result = StmtResult::Continue;
            let mut residual_emitted_at: Option<usize> = None;
            let result = if let Some(body) = &func.body {
                let mut return_val = None;
                for (idx, stmt) in body.statements.iter().enumerate() {
                    body_result = self.eval_statement(stmt)?;
                    match &body_result {
                        StmtResult::Return(v) => {
                            return_val = Some(v.clone());
                            break;
                        }
                        StmtResult::Break => {
                            // Break in function body - shouldn't happen, treat as continue
                        }
                        StmtResult::ResidualEmitted => {
                            // Function body emitted residual - record position and continue
                            // to process remaining statements as residual
                            residual_emitted_at = Some(idx);
                            break;
                        }
                        _ => {}
                    }
                }
                return_val.unwrap_or(Value::Undefined)
            } else {
                Value::Undefined
            };

            // Check if function produced residual statements
            let mut func_residual = self.take_residual();

            // If residual was emitted, we need to also emit remaining statements
            // and extract the return value for the function result
            let mut inlined_return_expr: Option<String> = None;
            if let (Some(emit_idx), Some(body)) = (residual_emitted_at, &func.body) {
                // Specialize and add remaining statements after the one that emitted residual
                for stmt in body.statements.iter().skip(emit_idx + 1) {
                    match stmt {
                        Statement::ReturnStatement(ret) => {
                            // Don't emit return - extract the expression as the result
                            // Use emit_expr (not specialize_expression) because variables
                            // may be modified by the residual loop
                            if let Some(arg) = &ret.argument {
                                inlined_return_expr = Some(emit_expr(arg));
                            }
                        }
                        _ => {
                            let specialized = self.specialize_statement(stmt);
                            func_residual.push(specialized);
                        }
                    }
                }
            }

            // Capture current scope bindings BEFORE popping the scope
            let scope_bindings = self.env.current_scope_bindings();

            self.env.pop_scope();

            // Restore outer residual
            self.residual_stmts = saved_residual;

            // If the function produced residual statements, wrap in IIFE to preserve scope
            if !func_residual.is_empty() {
                // Build IIFE body with variable declarations and residual statements
                let mut iife_body = Vec::new();

                // Optimize array/index pairs: if we have an array like `program` and an
                // index like `pc`, truncate the array to start at pc and set pc = 0
                let mut bindings_map: std::collections::HashMap<String, Value> =
                    scope_bindings.into_iter().collect();

                // Check for program/pc optimization pattern
                if let (Some(Value::Array(arr)), Some(Value::Number(pc_val))) =
                    (bindings_map.get("program"), bindings_map.get("pc"))
                {
                    let pc_idx = *pc_val as usize;
                    if pc_idx > 0 && pc_idx < arr.len() {
                        // Truncate array to start at pc
                        let truncated: Vec<Value> = arr.to_vec()[pc_idx..].to_vec();
                        bindings_map.insert("program".to_string(), Value::Array(SharedArray::new(truncated)));
                        bindings_map.insert("pc".to_string(), Value::Number(0.0));
                    }
                }

                // Emit variable declarations for current state of function locals
                for (name, value) in bindings_map {
                    let value_str = residual_of(&value)?;
                    iife_body.push(format!("let {} = {};", name, value_str));
                }

                // Add the residual code from the function body
                for stmt in &func_residual {
                    iife_body.push(stmt.clone());
                }

                // Add return statement
                let return_expr = inlined_return_expr.unwrap_or_else(|| "undefined".to_string());
                iife_body.push(format!("return {};", return_expr));

                // Build the IIFE
                let iife = format!("(() => {{\n{}\n}})()", iife_body.join("\n"));

                // Pop function from call stack before returning
                self.call_stack.pop();
                return Ok(dynamic(iife));
            }

            // Pop function from call stack before returning
            self.call_stack.pop();

            // Function fully evaluated - return the result (even if it's dynamic)
            // The result carries its residual expression from inside the function
            return Ok(result);
        }

        // Unsupported call expression - emit as residual
        Ok(dynamic(emit_call_expr(call)))
    }

    /// Recursively collect all var declaration names from a statement and its nested statements.
    /// This is needed because JavaScript hoists `var` to the nearest function scope,
    /// regardless of where the var appears (inside while loops, switch cases, etc.).
    fn collect_var_names(stmt: &Statement<'_>, names: &mut Vec<String>) {
        match stmt {
            Statement::VariableDeclaration(decl) => {
                if decl.kind == oxc_ast::ast::VariableDeclarationKind::Var {
                    for declarator in &decl.declarations {
                        if let BindingPatternKind::BindingIdentifier(id) = &declarator.id.kind {
                            names.push(id.name.to_string());
                        }
                    }
                }
            }
            Statement::WhileStatement(while_stmt) => {
                Self::collect_var_names(&while_stmt.body, names);
            }
            Statement::ForStatement(for_stmt) => {
                // Handle var in init
                if let Some(ForStatementInit::VariableDeclaration(decl)) = &for_stmt.init {
                    if decl.kind == oxc_ast::ast::VariableDeclarationKind::Var {
                        for declarator in &decl.declarations {
                            if let BindingPatternKind::BindingIdentifier(id) = &declarator.id.kind {
                                names.push(id.name.to_string());
                            }
                        }
                    }
                }
                Self::collect_var_names(&for_stmt.body, names);
            }
            Statement::ForInStatement(for_in_stmt) => {
                if let ForStatementLeft::VariableDeclaration(decl) = &for_in_stmt.left {
                    if decl.kind == oxc_ast::ast::VariableDeclarationKind::Var {
                        for declarator in &decl.declarations {
                            if let BindingPatternKind::BindingIdentifier(id) = &declarator.id.kind {
                                names.push(id.name.to_string());
                            }
                        }
                    }
                }
                Self::collect_var_names(&for_in_stmt.body, names);
            }
            Statement::IfStatement(if_stmt) => {
                Self::collect_var_names(&if_stmt.consequent, names);
                if let Some(alt) = &if_stmt.alternate {
                    Self::collect_var_names(alt, names);
                }
            }
            Statement::SwitchStatement(switch_stmt) => {
                for case in &switch_stmt.cases {
                    for case_stmt in &case.consequent {
                        Self::collect_var_names(case_stmt, names);
                    }
                }
            }
            Statement::TryStatement(try_stmt) => {
                for stmt in &try_stmt.block.body {
                    Self::collect_var_names(stmt, names);
                }
                if let Some(handler) = &try_stmt.handler {
                    for stmt in &handler.body.body {
                        Self::collect_var_names(stmt, names);
                    }
                }
                if let Some(finalizer) = &try_stmt.finalizer {
                    for stmt in &finalizer.body {
                        Self::collect_var_names(stmt, names);
                    }
                }
            }
            Statement::BlockStatement(block) => {
                for stmt in &block.body {
                    Self::collect_var_names(stmt, names);
                }
            }
            Statement::LabeledStatement(labeled) => {
                Self::collect_var_names(&labeled.body, names);
            }
            Statement::WithStatement(with_stmt) => {
                Self::collect_var_names(&with_stmt.body, names);
            }
            Statement::DoWhileStatement(do_while) => {
                Self::collect_var_names(&do_while.body, names);
            }
            _ => {}
        }
    }

    /// Evaluate an Immediately Invoked Function Expression (IIFE)
    fn eval_iife(
        &mut self,
        func_expr: &'a Function<'a>,
        arguments: &'a oxc_allocator::Vec<'a, Argument<'a>>,
    ) -> Result<Value, String> {
        // Evaluate arguments first
        let mut arg_values = Vec::new();
        for arg in arguments {
            match arg {
                Argument::SpreadElement(_) => {
                    // Spread in IIFE is rare - just return undefined as fallback
                    // (The caller will preserve the original expression)
                    return Err("Spread in IIFE not supported".to_string());
                }
                _ => {
                    let val = self.eval_expression(arg.to_expression())?;
                    arg_values.push(val);
                }
            }
        }

        // Get params from function expression
        let params: Vec<String> = func_expr
            .params
            .items
            .iter()
            .filter_map(|p| {
                if let BindingPatternKind::BindingIdentifier(id) = &p.pattern.kind {
                    Some(id.name.to_string())
                } else {
                    None
                }
            })
            .collect();

        // Save current residual state
        let saved_residual = self.take_residual();

        // Create new scope for function execution
        self.env.push_scope();

        // Bind parameters to arguments
        for (param, arg) in params.iter().zip(arg_values.iter()) {
            self.env.define(param, arg.clone());
        }

        // Execute function body with proper hoisting
        let result = if let Some(body) = &func_expr.body {
            // First pass: hoist function declarations and register them
            for stmt in body.statements.iter() {
                if let Statement::FunctionDeclaration(func) = stmt {
                    self.register_function(func)?;
                }
            }

            // Second pass: hoist var declarations (define as dynamic with their name)
            // Using Dynamic(name) instead of Undefined ensures proper residual emission
            // IMPORTANT: Recursively collect vars from nested structures (while, switch, if, etc.)
            // because JavaScript hoists ALL vars to the nearest function scope
            let mut var_names = Vec::new();
            for stmt in body.statements.iter() {
                Self::collect_var_names(stmt, &mut var_names);
            }
            for name in var_names {
                // Only hoist if not already defined in current scope (e.g., as a parameter)
                // We use exists_in_current_scope because var should create a new local binding
                // even if a variable with the same name exists in an outer scope
                if !self.env.exists_in_current_scope(&name) {
                    self.env.define(&name, dynamic(name.clone()));
                }
            }

            // Third pass: execute statements
            let mut return_val = None;
            for stmt in body.statements.iter() {
                let stmt_result = self.eval_statement(stmt)?;
                match stmt_result {
                    StmtResult::Return(v) => {
                        return_val = Some(v);
                        break;
                    }
                    _ => {}
                }
            }
            return_val.unwrap_or(Value::Undefined)
        } else {
            Value::Undefined
        };

        // Check if result contains any closures/functions that might reference scope variables
        if Self::value_contains_functions(&result) {
            // Result contains closures - we need to emit a specialized IIFE
            // that preserves the closure's captured scope

            // Capture scope bindings BEFORE popping
            let scope_bindings = self.env.current_scope_bindings();

            self.env.pop_scope();
            self.residual_stmts = saved_residual;

            // Build specialized IIFE body
            let mut iife_body = Vec::new();

            // Track which closure names we've emitted as function declarations
            let mut emitted_closures: std::collections::HashSet<String> = std::collections::HashSet::new();

            // Emit variable declarations for non-closure values (simplified/folded)
            // Emit function declarations for closure values
            for (name, value) in &scope_bindings {
                match value {
                    Value::Closure { source, .. } => {
                        // Emit the function declaration
                        iife_body.push(source.clone());
                        emitted_closures.insert(name.clone());
                    }
                    _ => {
                        // Emit simplified variable declaration
                        let value_str = residual_of(value)?;
                        iife_body.push(format!("var {} = {};", name, value_str));
                    }
                }
            }

            // Emit return statement with the result
            // If result is a closure we already emitted, just return its name
            let result_str = if let Value::Closure { source, .. } = &result {
                // Check if this closure matches one we emitted
                // Look for function name in source like "function foo("
                if let Some(name) = emitted_closures.iter().find(|name| {
                    source.contains(&format!("function {}(", name))
                }) {
                    name.clone()
                } else {
                    residual_of(&result)?
                }
            } else {
                residual_of(&result)?
            };
            iife_body.push(format!("return {};", result_str));

            // Build the specialized IIFE
            let iife = format!("(function() {{\n{}\n}})()", iife_body.join("\n"));

            return Ok(dynamic(iife));
        }

        // Pop the scope
        self.env.pop_scope();

        // Restore outer residual
        self.residual_stmts = saved_residual;

        Ok(result)
    }

    /// Check if a value contains any function expressions or closures
    fn value_contains_functions(value: &Value) -> bool {
        match value {
            Value::Closure { .. } => true,
            Value::Dynamic(s) => {
                // Check if the dynamic value looks like a function
                s.contains("function") || s.contains("=>")
            }
            Value::Array(arr) => arr.iter().any(|v| Self::value_contains_functions(&v)),
            Value::Object(props) => props.iter().any(|(_, v)| Self::value_contains_functions(&v)),
            _ => false,
        }
    }

    fn eval_method_call(
        &mut self,
        member: &'a StaticMemberExpression<'a>,
        args: &'a oxc_allocator::Vec<'a, Argument<'a>>,
        call: &'a CallExpression<'a>,
    ) -> Result<Value, String> {
        let method_name = member.property.name.to_string();

        // Get the object
        let obj_name = match &member.object {
            Expression::Identifier(id) => id.name.to_string(),
            // Method call on non-identifier - emit as residual
            _ => return Ok(dynamic(emit_call_expr(call))),
        };

        // Check for spread arguments first
        for arg in args {
            if matches!(arg, Argument::SpreadElement(_)) {
                // Spread makes the whole call dynamic
                return Ok(dynamic(emit_call_expr(call)));
            }
        }

        // Evaluate arguments
        let mut arg_values = Vec::new();
        for arg in args {
            let val = self.eval_expression(arg.to_expression())?;
            arg_values.push(val);
        }

        // Get current value of object, or treat as dynamic if undefined
        let obj_val = match self.env.get(&obj_name) {
            Some(v) => v,
            None => {
                // Unknown object - emit call as residual
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                return Ok(dynamic(format!("{}.{}({})", obj_name, method_name, arg_strs?.join(", "))));
            }
        };

        match (&obj_val, method_name.as_str()) {
            (Value::Array(arr), "push") => {
                // We can always push to an array, even if values are dynamic
                // The array can contain dynamic elements
                let mut new_arr = arr.clone();
                for val in arg_values {
                    new_arr.push(val);
                }
                let new_len = Value::Number(new_arr.len() as f64);
                self.env.set(&obj_name, Value::Array(new_arr));
                Ok(new_len)
            }
            (Value::Array(arr), "pop") => {
                let mut new_arr = arr.clone();
                let popped = new_arr.pop().unwrap_or(Value::Undefined);
                self.env.set(&obj_name, Value::Array(new_arr));
                Ok(popped)
            }
            (Value::Array(arr), "length") => Ok(Value::Number(arr.len() as f64)),
            (Value::Dynamic(obj_expr), _) => {
                // Method call on dynamic object
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                Ok(dynamic(format!("{}.{}({})", obj_expr, method_name, arg_strs?.join(", "))))
            }
            // Unsupported method on static object - emit as residual
            _ => {
                let obj_js = residual_of(&obj_val)?;
                let arg_strs: Result<Vec<_>, _> = arg_values.iter().map(residual_of).collect();
                Ok(dynamic(format!("{}.{}({})", obj_js, method_name, arg_strs?.join(", "))))
            }
        }
    }

    fn eval_assignment_expression(
        &mut self,
        assign: &'a AssignmentExpression<'a>,
    ) -> Result<Value, String> {
        let right_value = self.eval_expression(&assign.right)?;

        match &assign.left {
            AssignmentTarget::AssignmentTargetIdentifier(id) => {
                let name = id.name.to_string();

                // For compound operators, we need to combine with existing value
                let final_value = match assign.operator {
                    AssignmentOperator::Assign => right_value,
                    AssignmentOperator::Addition => {
                        let left = self.env.get(&name).unwrap_or_else(|| dynamic(name.clone()));
                        match (&left, &right_value) {
                            (Value::Number(a), Value::Number(b)) => Value::Number(a + b),
                            (Value::String(a), Value::String(b)) => Value::String(format!("{}{}", a, b)),
                            _ => {
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right_value)?;
                                dynamic(format!("{} + {}", left_js, right_js))
                            }
                        }
                    }
                    AssignmentOperator::Subtraction => {
                        let left = self.env.get(&name).unwrap_or_else(|| dynamic(name.clone()));
                        match (&left, &right_value) {
                            (Value::Number(a), Value::Number(b)) => Value::Number(a - b),
                            _ => {
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right_value)?;
                                dynamic(format!("{} - {}", left_js, right_js))
                            }
                        }
                    }
                    AssignmentOperator::Multiplication => {
                        let left = self.env.get(&name).unwrap_or_else(|| dynamic(name.clone()));
                        match (&left, &right_value) {
                            (Value::Number(a), Value::Number(b)) => Value::Number(a * b),
                            _ => {
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right_value)?;
                                dynamic(format!("{} * {}", left_js, right_js))
                            }
                        }
                    }
                    AssignmentOperator::Division => {
                        let left = self.env.get(&name).unwrap_or_else(|| dynamic(name.clone()));
                        match (&left, &right_value) {
                            (Value::Number(a), Value::Number(b)) => Value::Number(a / b),
                            _ => {
                                let left_js = residual_of(&left)?;
                                let right_js = residual_of(&right_value)?;
                                dynamic(format!("{} / {}", left_js, right_js))
                            }
                        }
                    }
                    // Other operators - emit as residual
                    _ => {
                        dynamic(emit_assign_expr(assign))
                    }
                };

                if !self.env.set(&name, final_value.clone()) {
                    // Variable not defined in our env - could be a global, emit as residual
                    return Ok(dynamic(emit_assign_expr(assign)));
                }
                Ok(final_value)
            }
            // Handle computed member expression like x[0] = val or x[0][1] = val
            AssignmentTarget::ComputedMemberExpression(member) => {
                // Only handle simple assignment for now
                if assign.operator != AssignmentOperator::Assign {
                    return Ok(dynamic(emit_assign_expr(assign)));
                }

                // Try to trace back to base variable and indices
                match self.trace_member_chain(&member.object, &member.expression) {
                    Some((base_name, indices)) => {
                        // Get the base value from environment
                        if let Some(base_value) = self.env.get(&base_name) {
                            // Try to update the nested value
                            if let Some(updated) = Self::update_nested_value(&base_value, &indices, &right_value) {
                                self.env.set(&base_name, updated);
                                return Ok(right_value);
                            }
                        }
                        // Fall through to emit as residual
                        Ok(dynamic(emit_assign_expr(assign)))
                    }
                    None => Ok(dynamic(emit_assign_expr(assign))),
                }
            }
            // Handle static member expression like x.foo = val
            AssignmentTarget::StaticMemberExpression(member) => {
                // Only handle simple assignment for now
                if assign.operator != AssignmentOperator::Assign {
                    return Ok(dynamic(emit_assign_expr(assign)));
                }

                // Get the object name
                let obj_name = match &member.object {
                    Expression::Identifier(id) => id.name.to_string(),
                    _ => return Ok(dynamic(emit_assign_expr(assign))),
                };
                let property = member.property.name.to_string();

                // Get the object and update the property
                // With SharedObject, the mutation is visible to all aliases
                if let Some(obj_val) = self.env.get(&obj_name) {
                    if let Value::Object(props) = obj_val {
                        props.set(property, right_value.clone());
                        return Ok(right_value);
                    }
                }
                Ok(dynamic(emit_assign_expr(assign)))
            }
            // Unsupported assignment target - emit as residual
            _ => Ok(dynamic(emit_assign_expr(assign))),
        }
    }

    /// Trace a member expression chain back to find the base variable and indices
    /// Returns (base_name, vec of indices) if successful
    fn trace_member_chain(&mut self, object: &'a Expression<'a>, index_expr: &'a Expression<'a>) -> Option<(String, Vec<Value>)> {
        // Evaluate the index
        let index = self.eval_expression(index_expr).ok()?;

        match object {
            Expression::Identifier(id) => {
                // Base case: we've reached the variable
                Some((id.name.to_string(), vec![index]))
            }
            Expression::ComputedMemberExpression(inner) => {
                // Recursive case: another computed member
                let mut result = self.trace_member_chain(&inner.object, &inner.expression)?;
                result.1.push(index);
                Some(result)
            }
            _ => None, // Can't trace other expression types
        }
    }

    /// Update a nested value in an array/object structure
    /// indices is a list of indices to traverse (in order from outer to inner)
    fn update_nested_value(value: &Value, indices: &[Value], new_value: &Value) -> Option<Value> {
        if indices.is_empty() {
            return Some(new_value.clone());
        }

        match (value, &indices[0]) {
            (Value::Array(arr), Value::Number(n)) => {
                let idx = *n as usize;
                if indices.len() == 1 {
                    // Last index - directly update in place using SharedArray
                    arr.set(idx, new_value.clone());
                    Some(value.clone())
                } else {
                    // More indices to traverse - get the inner value and recurse
                    let inner = arr.get(idx)?;
                    Self::update_nested_value(&inner, &indices[1..], new_value)?;
                    Some(value.clone())
                }
            }
            (Value::TypedArray { kind, buffer, byte_offset, length }, Value::Number(n)) => {
                // TypedArray indexed assignment - update the underlying buffer directly
                // This mutates the shared buffer, so other views will see the change
                let idx = *n as usize;
                if idx >= *length {
                    return None; // Out of bounds
                }
                if indices.len() != 1 {
                    return None; // TypedArray elements are primitive, can't traverse further
                }

                // Get the value to write
                let byte_val = match new_value {
                    Value::Number(v) => *v as u8, // For now, just handle Uint8Array style
                    _ => return None, // Can't assign non-number to TypedArray
                };

                // Calculate the actual byte offset in the buffer
                let element_size = kind.element_size();
                let actual_offset = *byte_offset + idx * element_size;

                // Update the shared buffer
                buffer.set(actual_offset, byte_val);

                // Return the same TypedArray (buffer is already mutated)
                Some(value.clone())
            }
            (Value::Object(props), Value::String(key)) => {
                if indices.len() == 1 {
                    // Last index - directly update in place using SharedObject
                    props.set(key.clone(), new_value.clone());
                    Some(value.clone())
                } else {
                    // More indices to traverse - get the inner value and recurse
                    let inner = props.get(key)?;
                    Self::update_nested_value(&inner, &indices[1..], new_value)?;
                    Some(value.clone())
                }
            }
            _ => None,
        }
    }

    fn eval_update_expression(
        &mut self,
        update: &'a UpdateExpression<'a>,
    ) -> Result<Value, String> {
        let name = match &update.argument {
            SimpleAssignmentTarget::AssignmentTargetIdentifier(id) => id.name.to_string(),
            // Unsupported update target - emit as residual
            _ => return Ok(dynamic(emit_update_expr(update))),
        };

        let current = match self.env.get(&name) {
            Some(v) => v,
            // Undefined variable - emit as residual
            None => return Ok(dynamic(emit_update_expr(update))),
        };

        match &current {
            Value::Number(n) => {
                let new_val = match update.operator {
                    UpdateOperator::Increment => Value::Number(n + 1.0),
                    UpdateOperator::Decrement => Value::Number(n - 1.0),
                };

                self.env.set(&name, new_val.clone());

                if update.prefix {
                    Ok(new_val)
                } else {
                    Ok(Value::Number(*n))
                }
            }
            Value::Dynamic(_) => {
                // Dynamic variable - can't statically update, emit residual
                let op = match update.operator {
                    UpdateOperator::Increment => "++",
                    UpdateOperator::Decrement => "--",
                };
                let residual = if update.prefix {
                    format!("{}{}", op, name)
                } else {
                    format!("{}{}", name, op)
                };
                // The value becomes the new dynamic
                self.env.set(&name, dynamic(name.clone()));
                Ok(dynamic(residual))
            }
            _ => {
                // Non-number static - emit as residual
                Ok(dynamic(emit_update_expr(update)))
            }
        }
    }
}

/// Emit an expression using oxc_codegen
fn emit_expr(expr: &Expression<'_>) -> String {
    let mut codegen = Codegen::new();
    codegen.print_expression(expr);
    codegen.into_source_text()
}

/// Emit a binary expression using the codegen's print_expression
fn emit_binary_expr(bin: &BinaryExpression<'_>) -> String {
    // Build the expression manually since we can't easily construct Expression variant
    let left = emit_expr(&bin.left);
    let right = emit_expr(&bin.right);
    let op = match bin.operator {
        BinaryOperator::Addition => "+",
        BinaryOperator::Subtraction => "-",
        BinaryOperator::Multiplication => "*",
        BinaryOperator::Division => "/",
        BinaryOperator::Remainder => "%",
        BinaryOperator::LessThan => "<",
        BinaryOperator::GreaterThan => ">",
        BinaryOperator::LessEqualThan => "<=",
        BinaryOperator::GreaterEqualThan => ">=",
        BinaryOperator::Equality => "==",
        BinaryOperator::StrictEquality => "===",
        BinaryOperator::Inequality => "!=",
        BinaryOperator::StrictInequality => "!==",
        BinaryOperator::BitwiseAnd => "&",
        BinaryOperator::BitwiseOR => "|",
        BinaryOperator::BitwiseXOR => "^",
        BinaryOperator::ShiftLeft => "<<",
        BinaryOperator::ShiftRight => ">>",
        BinaryOperator::ShiftRightZeroFill => ">>>",
        BinaryOperator::Instanceof => "instanceof",
        BinaryOperator::In => "in",
        BinaryOperator::Exponential => "**",
    };
    format!("{} {} {}", left, op, right)
}

/// Emit a call expression
fn emit_call_expr(call: &CallExpression<'_>) -> String {
    let callee = emit_expr(&call.callee);
    let args: Vec<String> = call.arguments.iter().map(|arg| {
        emit_expr(arg.to_expression())
    }).collect();
    format!("{}({})", callee, args.join(", "))
}

/// Emit a new expression
fn emit_new_expr(new_expr: &NewExpression<'_>) -> String {
    let callee = emit_expr(&new_expr.callee);
    let args: Vec<String> = new_expr.arguments.iter().map(|arg| {
        emit_expr(arg.to_expression())
    }).collect();
    format!("new {}({})", callee, args.join(", "))
}

/// Emit an assignment expression
fn emit_assign_expr(assign: &AssignmentExpression<'_>) -> String {
    let right = emit_expr(&assign.right);
    let left = match &assign.left {
        AssignmentTarget::AssignmentTargetIdentifier(id) => id.name.to_string(),
        AssignmentTarget::StaticMemberExpression(member) => {
            format!("{}.{}", emit_expr(&member.object), member.property.name)
        }
        AssignmentTarget::ComputedMemberExpression(member) => {
            format!("{}[{}]", emit_expr(&member.object), emit_expr(&member.expression))
        }
        _ => "<unsupported target>".to_string(),
    };
    let op = match assign.operator {
        AssignmentOperator::Assign => "=",
        AssignmentOperator::Addition => "+=",
        AssignmentOperator::Subtraction => "-=",
        AssignmentOperator::Multiplication => "*=",
        AssignmentOperator::Division => "/=",
        AssignmentOperator::Remainder => "%=",
        AssignmentOperator::BitwiseAnd => "&=",
        AssignmentOperator::BitwiseOR => "|=",
        AssignmentOperator::BitwiseXOR => "^=",
        AssignmentOperator::ShiftLeft => "<<=",
        AssignmentOperator::ShiftRight => ">>=",
        AssignmentOperator::ShiftRightZeroFill => ">>>=",
        AssignmentOperator::LogicalAnd => "&&=",
        AssignmentOperator::LogicalOr => "||=",
        AssignmentOperator::LogicalNullish => "??=",
        AssignmentOperator::Exponential => "**=",
    };
    format!("{} {} {}", left, op, right)
}

/// Emit an update expression
fn emit_update_expr(update: &UpdateExpression<'_>) -> String {
    let arg = match &update.argument {
        SimpleAssignmentTarget::AssignmentTargetIdentifier(id) => id.name.to_string(),
        _ => "<complex target>".to_string(),
    };
    let op = match update.operator {
        UpdateOperator::Increment => "++",
        UpdateOperator::Decrement => "--",
    };
    if update.prefix {
        format!("{}{}", op, arg)
    } else {
        format!("{}{}", arg, op)
    }
}

/// Emit a function declaration using oxc_codegen
fn emit_function(func: &Function<'_>) -> String {
    let mut codegen = Codegen::new();
    func.print(&mut codegen, Context::empty());
    codegen.into_source_text()
}

/// Emit a statement using oxc_codegen
fn emit_stmt(stmt: &Statement<'_>) -> String {
    let mut codegen = Codegen::new();
    stmt.print(&mut codegen, Context::empty());
    codegen.into_source_text()
}

impl Default for Evaluator<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;
    use oxc_allocator::Allocator;

    fn eval_and_get(source: &str, var_name: &str) -> Value {
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        evaluator.env.get(var_name).unwrap()
    }

    #[test]
    fn test_simple_assignment() {
        let val = eval_and_get("let x = 42;", "x");
        assert_eq!(val, Value::Number(42.0));
    }

    #[test]
    fn test_arithmetic() {
        let val = eval_and_get("let x = 2 + 3 * 4;", "x");
        assert_eq!(val, Value::Number(14.0));
    }

    #[test]
    fn test_array_literal() {
        let val = eval_and_get("let arr = [1, 2, 3];", "arr");
        assert_eq!(
            val,
            Value::Array(SharedArray::new(vec![
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(3.0)
            ]))
        );
    }

    #[test]
    fn test_array_push() {
        let val = eval_and_get(
            r#"
            let arr = [];
            arr.push(1);
            arr.push(2);
            arr.push(3);
        "#,
            "arr",
        );
        assert_eq!(
            val,
            Value::Array(SharedArray::new(vec![
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(3.0)
            ]))
        );
    }

    #[test]
    fn test_array_pop() {
        let val = eval_and_get(
            r#"
            let arr = [1, 2, 3];
            arr.pop();
        "#,
            "arr",
        );
        assert_eq!(
            val,
            Value::Array(SharedArray::new(vec![Value::Number(1.0), Value::Number(2.0)]))
        );
    }

    #[test]
    fn test_array_length() {
        let val = eval_and_get(
            r#"
            let arr = [1, 2, 3];
            let len = arr.length;
        "#,
            "len",
        );
        assert_eq!(val, Value::Number(3.0));
    }

    #[test]
    fn test_while_loop() {
        let val = eval_and_get(
            r#"
            let x = 0;
            while (x < 5) {
                x = x + 1;
            }
        "#,
            "x",
        );
        assert_eq!(val, Value::Number(5.0));
    }

    #[test]
    fn test_if_statement() {
        let val = eval_and_get(
            r#"
            let x = 10;
            let result = 0;
            if (x > 5) {
                result = 1;
            } else {
                result = 2;
            }
        "#,
            "result",
        );
        assert_eq!(val, Value::Number(1.0));
    }

    #[test]
    fn test_function_call_simple() {
        let val = eval_and_get(
            r#"
            function add(a, b) {
                return a + b;
            }
            let result = add(2, 3);
        "#,
            "result",
        );
        assert_eq!(val, Value::Number(5.0));
    }

    #[test]
    fn test_function_modifies_global() {
        let val = eval_and_get(
            r#"
            let arr = [];
            function push3() {
                arr.push(1);
                arr.push(2);
                arr.push(3);
            }
            push3();
        "#,
            "arr",
        );
        assert_eq!(
            val,
            Value::Array(SharedArray::new(vec![
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(3.0)
            ]))
        );
    }

    #[test]
    fn test_residual_push3() {
        use crate::residual::emit_residual;
        use std::process::Command;

        let source = r#"
            let arr = [];
            function push3() {
                arr.push(1);
                arr.push(2);
                arr.push(3);
            }
            push3();
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        // Generate residual code
        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The residual should contain `let arr = [1, 2, 3];`
        assert!(residual.contains("let arr = [1, 2, 3];"), "Residual was: {}", residual);

        // Verify the residual runs correctly in node
        let test_code = format!("{}\nconsole.log(JSON.stringify(arr));", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(), "Node failed: {:?}", String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "[1,2,3]", "Output was: {}", stdout);
    }

    #[test]
    fn test_residual_matches_original_semantics() {
        use crate::residual::emit_residual;
        use std::process::Command;

        let source = r#"
            let x = 10;
            let y = x + 5;
            let arr = [x, y, x + y];
        "#;

        // Run original in node
        let original_test = format!("{}\nconsole.log(JSON.stringify({{x, y, arr}}));", source);
        let original_output = Command::new("node")
            .arg("-e")
            .arg(&original_test)
            .output()
            .expect("Failed to run node");
        let original_result = String::from_utf8_lossy(&original_output.stdout);

        // Generate and run residual
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();
        let residual_test = format!("{}\nconsole.log(JSON.stringify({{x, y, arr}}));", residual);

        let residual_output = Command::new("node")
            .arg("-e")
            .arg(&residual_test)
            .output()
            .expect("Failed to run node");

        assert!(residual_output.status.success(),
            "Residual failed to run:\nCode: {}\nError: {}",
            residual,
            String::from_utf8_lossy(&residual_output.stderr));

        let residual_result = String::from_utf8_lossy(&residual_output.stdout);

        assert_eq!(
            original_result.trim(),
            residual_result.trim(),
            "Semantics mismatch!\nOriginal output: {}\nResidual output: {}\nResidual code:\n{}",
            original_result,
            residual_result,
            residual
        );
    }

    #[test]
    fn test_dynamic_value_preserved() {
        use crate::residual::emit_residual;

        // Simulate a dynamic input by using a variable that's not defined
        // For this test, we'll mark 'input' as dynamic in the environment
        let source = "let x = 5;";

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();

        // Pre-define 'input' as dynamic
        evaluator.env.define("input", Value::Dynamic("input".to_string()));

        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();
        assert!(residual.contains("let x = 5;"), "Residual was: {}", residual);
    }

    #[test]
    fn test_dynamic_expression_preserved() {
        use crate::residual::emit_residual;

        let source = "let y = input + 1;";

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();

        // Pre-define 'input' as dynamic
        evaluator.env.define("input", Value::Dynamic("input".to_string()));

        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();
        // The expression should be preserved (not folded)
        assert!(residual.contains("let y = input + 1"), "Residual was: {}", residual);
    }

    #[test]
    fn test_mixed_static_dynamic_with_node() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // Original code with a mix of static and dynamic values
        let source = r#"
            let a = 5;
            let b = 3;
            let c = a + b;
            let d = input * 2;
            let e = c + input;
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();

        // Pre-define 'input' as dynamic
        evaluator.env.define("input", Value::Dynamic("input".to_string()));

        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // Static values should be folded
        assert!(residual.contains("let a = 5;"), "Expected 'let a = 5;' in: {}", residual);
        assert!(residual.contains("let b = 3;"), "Expected 'let b = 3;' in: {}", residual);
        assert!(residual.contains("let c = 8;"), "Expected 'let c = 8;' (folded) in: {}", residual);

        // Dynamic values should preserve expressions
        assert!(residual.contains("let d = input * 2"), "Expected dynamic 'd' in: {}", residual);
        assert!(residual.contains("let e = c + input") || residual.contains("let e = 8 + input"),
            "Expected dynamic 'e' in: {}", residual);

        // Verify the residual runs correctly in Node with a concrete input
        let test_code = format!("let input = 10;\n{}\nconsole.log(JSON.stringify({{a, b, c, d, e}}));", residual);

        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        // a=5, b=3, c=8, d=20, e=18
        assert!(stdout.contains("\"a\":5"), "Output: {}", stdout);
        assert!(stdout.contains("\"b\":3"), "Output: {}", stdout);
        assert!(stdout.contains("\"c\":8"), "Output: {}", stdout);
        assert!(stdout.contains("\"d\":20"), "Output: {}", stdout);
        assert!(stdout.contains("\"e\":18"), "Output: {}", stdout);
    }

    #[test]
    fn test_preserved_function_not_called() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // A function that is never called should be preserved in residual
        let source = r#"
            function greet(name) {
                return "Hello, " + name + "!";
            }
            let x = 42;
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The function should be preserved
        assert!(residual.contains("function greet"), "Function should be preserved. Residual: {}", residual);
        assert!(residual.contains("return"), "Function body should be preserved. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(greet(\"World\"));", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "Hello, World!", "Output was: {}", stdout);
    }

    #[test]
    fn test_consumed_function_not_preserved() {
        use crate::residual::emit_residual;

        // A function that is called and consumed should NOT be preserved
        let source = r#"
            function add(a, b) {
                return a + b;
            }
            let result = add(2, 3);
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The function was called and is dead, so it should NOT be preserved
        assert!(!residual.contains("function add"), "Dead function should not be preserved. Residual: {}", residual);

        // But the result should be the computed value
        assert!(residual.contains("let result = 5;"), "Result should be folded. Residual: {}", residual);
    }

    #[test]
    fn test_closure_capturing_outer_scope() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // A function that captures a variable from outer scope
        let source = r#"
            let multiplier = 3;
            function multiply(x) {
                return x * multiplier;
            }
            let result = multiply(4);
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        // The function was called, so it should be dead
        // The result should be folded
        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // Result should be 12 (3 * 4)
        assert!(residual.contains("let result = 12;"), "Result should be folded. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(result);", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "12", "Output was: {}", stdout);
    }

    #[test]
    fn test_closure_preserved_when_not_called() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // A closure that captures outer scope but is never called
        let source = r#"
            let factor = 5;
            function scale(x) {
                return x * factor;
            }
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The function should be preserved since it wasn't called
        assert!(residual.contains("function scale"), "Function should be preserved. Residual: {}", residual);

        // The captured variable should also be preserved
        assert!(residual.contains("let factor = 5;") || residual.contains("factor"), "Captured variable should be preserved. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(scale(10));", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "50", "Output was: {}", stdout);
    }

    #[test]
    fn test_bitwise_and() {
        let source = "let x = 7 & 3;";  // 0111 & 0011 = 0011 = 3
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(3.0)));
    }

    #[test]
    fn test_bitwise_or() {
        let source = "let x = 5 | 3;";  // 0101 | 0011 = 0111 = 7
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(7.0)));
    }

    #[test]
    fn test_bitwise_xor() {
        let source = "let x = 5 ^ 3;";  // 0101 ^ 0011 = 0110 = 6
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(6.0)));
    }

    #[test]
    fn test_shift_left() {
        let source = "let x = 5 << 2;";  // 5 * 4 = 20
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(20.0)));
    }

    #[test]
    fn test_shift_right() {
        let source = "let x = 20 >> 2;";  // 20 / 4 = 5
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(5.0)));
    }

    #[test]
    fn test_shift_right_zero_fill() {
        let source = "let x = -1 >>> 28;";  // unsigned right shift
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(15.0)));
    }

    #[test]
    fn test_remainder() {
        let source = "let x = 17 % 5;";  // 17 mod 5 = 2
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(2.0)));
    }

    #[test]
    fn test_ternary_true() {
        let source = "let x = true ? 42 : 0;";
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(42.0)));
    }

    #[test]
    fn test_ternary_false() {
        let source = "let x = false ? 42 : 99;";
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(99.0)));
    }

    #[test]
    fn test_ternary_with_expression() {
        let source = "let x = 5 > 3 ? 1 : 0;";
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(1.0)));
    }

    #[test]
    fn test_logical_and_short_circuit_false() {
        let source = "let x = false && true;";
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Bool(false)));
    }

    #[test]
    fn test_logical_and_both_true() {
        let source = "let x = true && 42;";
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(42.0)));
    }

    #[test]
    fn test_logical_or_short_circuit_true() {
        let source = "let x = true || false;";
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Bool(true)));
    }

    #[test]
    fn test_logical_or_first_false() {
        let source = "let x = false || 42;";
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(42.0)));
    }

    #[test]
    fn test_nullish_coalescing_null() {
        let source = "let x = null ?? 42;";
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(42.0)));
    }

    #[test]
    fn test_nullish_coalescing_value() {
        let source = "let x = 5 ?? 42;";
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(5.0)));
    }

    #[test]
    fn test_parenthesized_expression() {
        let source = "let a = 5; let x = (a + 3) & 255;";
        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();
        assert_eq!(evaluator.env.get("x"), Some(Value::Number(8.0)));
    }

    // =========================================================================
    // IIFE + Closure Tests
    // =========================================================================

    #[test]
    fn test_iife_no_closure_fully_evaluates() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // IIFE that returns a simple value should be fully evaluated
        let source = r#"
            var result = (function() {
                var x = 10;
                var y = 20;
                return x + y;
            })();
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The IIFE should be fully evaluated to 30
        assert!(residual.contains("let result = 30;") || residual.contains("var result = 30;"),
            "IIFE should be fully evaluated to 30. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(result);", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "30", "Output was: {}", stdout);
    }

    #[test]
    fn test_iife_returning_closure_preserves_scope() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // IIFE that returns a closure should preserve scope
        let source = r#"
            var result = (function() {
                var x = 10;
                var y = 20;
                function inner() {
                    return x + y;
                }
                return inner;
            })();
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The residual should be a specialized IIFE with folded values
        // and the closure properly preserved
        assert!(residual.contains("x = 10") || residual.contains("var x = 10"),
            "x should be preserved in scope. Residual: {}", residual);
        assert!(residual.contains("y = 20") || residual.contains("var y = 20"),
            "y should be preserved in scope. Residual: {}", residual);
        assert!(residual.contains("function inner"),
            "inner function should be preserved. Residual: {}", residual);
        assert!(residual.contains("return inner"),
            "Should return inner by name. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(result());", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "30", "Output was: {}", stdout);
    }

    #[test]
    fn test_iife_with_constant_folding_in_closure() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // IIFE where we can fold constants even with a closure
        let source = r#"
            var result = (function() {
                var base = 2 + 3;
                var scale = base * 2;
                function multiply(x) {
                    return x * scale;
                }
                return multiply;
            })();
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // Constants should be folded: base = 5, scale = 10
        assert!(residual.contains("base = 5") || residual.contains("var base = 5"),
            "base should be folded to 5. Residual: {}", residual);
        assert!(residual.contains("scale = 10") || residual.contains("var scale = 10"),
            "scale should be folded to 10. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(result(7));", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "70", "7 * 10 should be 70. Output was: {}", stdout);
    }

    #[test]
    fn test_iife_returning_anonymous_function() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // IIFE that returns an anonymous function
        let source = r#"
            var result = (function() {
                return function() { return 42; };
            })();
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The function should be preserved
        assert!(residual.contains("function"),
            "Anonymous function should be preserved. Residual: {}", residual);
        assert!(residual.contains("42"),
            "Return value 42 should be preserved. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(result());", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "42", "Output was: {}", stdout);
    }

    #[test]
    fn test_iife_with_multiple_closures() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // IIFE that returns an object with multiple closures
        let source = r#"
            var counter = (function() {
                var count = 0;
                function inc() { count = count + 1; return count; }
                function get() { return count; }
                return { inc: inc, get: get };
            })();
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // Both functions should be preserved
        assert!(residual.contains("function inc"),
            "inc function should be preserved. Residual: {}", residual);
        assert!(residual.contains("function get"),
            "get function should be preserved. Residual: {}", residual);
        // count should be preserved
        assert!(residual.contains("count = 0") || residual.contains("var count = 0"),
            "count should be preserved. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\ncounter.inc(); counter.inc(); console.log(counter.get());", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "2", "After 2 increments, count should be 2. Output was: {}", stdout);
    }

    #[test]
    fn test_iife_with_parameters() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // IIFE with parameters
        let source = r#"
            var result = (function(a, b) {
                return a + b;
            })(10, 20);
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The IIFE should be fully evaluated to 30
        assert!(residual.contains("result = 30"),
            "IIFE with params should be fully evaluated to 30. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(result);", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "30", "Output was: {}", stdout);
    }

    #[test]
    fn test_nested_iife() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // Nested IIFEs
        let source = r#"
            var result = (function() {
                var outer = 10;
                var inner_result = (function() {
                    return outer * 2;
                })();
                return inner_result + 5;
            })();
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The nested IIFEs should be fully evaluated: outer=10, inner_result=20, result=25
        assert!(residual.contains("result = 25"),
            "Nested IIFE should evaluate to 25. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(result);", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "25", "Output was: {}", stdout);
    }

    #[test]
    fn test_iife_closure_with_loop_variable() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // IIFE that returns a closure capturing a computed loop result
        let source = r#"
            var result = (function() {
                var sum = 0;
                var i = 0;
                while (i < 5) {
                    sum = sum + i;
                    i = i + 1;
                }
                function getSum() { return sum; }
                return getSum;
            })();
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // sum should be computed: 0+1+2+3+4 = 10
        assert!(residual.contains("sum = 10") || residual.contains("var sum = 10"),
            "sum should be folded to 10 after loop. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(result());", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "10", "Sum should be 10. Output was: {}", stdout);
    }

    #[test]
    fn test_function_declaration_inside_iife() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // Function declarations inside IIFE are properly hoisted and accessible
        let source = r#"
            var result = (function() {
                function helper(n) {
                    return n * 2;
                }
                return helper(21);
            })();
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The function call should be evaluated: helper(21) = 42
        assert!(residual.contains("result = 42"),
            "Function inside IIFE should be called. Residual: {}", residual);

        // Verify the residual runs correctly in Node
        let test_code = format!("{}\nconsole.log(result);", residual);
        let output = Command::new("node")
            .arg("-e")
            .arg(&test_code)
            .output()
            .expect("Failed to run node");

        assert!(output.status.success(),
            "Node failed:\nCode: {}\nError: {}",
            test_code,
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "42", "Output was: {}", stdout);
    }

    #[test]
    fn test_textdecoder_uint8array() {
        use crate::residual::emit_residual;
        use std::process::Command;

        // Test that TextDecoder with Uint8Array decodes correctly
        let source = r#"
            let v6 = [72, 101, 108, 108, 111];
            let result = new TextDecoder().decode(new Uint8Array(v6));
        "#;

        let allocator = Allocator::default();
        let program = parse(&allocator, source);
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&program).unwrap();

        let residual = emit_residual(&evaluator.trace, true).unwrap();

        // The result should be the decoded string
        assert!(residual.contains(r#"let result = "Hello""#),
            "Result should be decoded string. Residual was:\n{}", residual);

        // Verify original and specialized produce the same output
        let original_code = format!("{}\nconsole.log(result);", source);
        let specialized_code = format!("{}\nconsole.log(result);", residual);

        let original_output = Command::new("node")
            .arg("-e")
            .arg(&original_code)
            .output()
            .expect("Failed to run original code");

        let specialized_output = Command::new("node")
            .arg("-e")
            .arg(&specialized_code)
            .output()
            .expect("Failed to run specialized code");

        let original_stdout = String::from_utf8_lossy(&original_output.stdout);
        let specialized_stdout = String::from_utf8_lossy(&specialized_output.stdout);

        assert_eq!(original_stdout.trim(), specialized_stdout.trim(),
            "Outputs should match. Original: {}, Specialized: {}",
            original_stdout.trim(), specialized_stdout.trim());
        assert_eq!(specialized_stdout.trim(), "Hello");
    }

    // ============ Reference Aliasing Tests ============

    #[test]
    fn test_array_alias_mutation() {
        // When sub = arr[0], sub should be a reference to arr[0]
        // Modifying sub[0] should also modify arr[0][0]
        let val = eval_and_get(
            r#"
            var arr = [[1, 2, 3], [4, 5, 6]];
            var sub = arr[0];
            sub[0] = 99;
        "#,
            "arr",
        );
        // arr[0][0] should be 99, not 1
        if let Value::Array(outer) = val {
            if let Some(Value::Array(inner)) = outer.get(0) {
                assert_eq!(inner.get(0), Some(Value::Number(99.0)),
                    "arr[0][0] should be 99 after alias mutation");
            } else {
                panic!("Expected inner array");
            }
        } else {
            panic!("Expected outer array");
        }
    }

    #[test]
    fn test_array_alias_mutation_in_function() {
        // Same test but mutation happens inside a function
        let val = eval_and_get(
            r#"
            var arr = [[1, 2, 3], [4, 5, 6]];
            function mutate() {
                var sub = arr[0];
                sub[0] = 99;
            }
            mutate();
        "#,
            "arr",
        );
        if let Value::Array(outer) = val {
            if let Some(Value::Array(inner)) = outer.get(0) {
                assert_eq!(inner.get(0), Some(Value::Number(99.0)),
                    "arr[0][0] should be 99 after alias mutation in function");
            } else {
                panic!("Expected inner array");
            }
        } else {
            panic!("Expected outer array");
        }
    }

    #[test]
    fn test_array_alias_mutation_in_iife() {
        // Test with IIFE like in simple-full.js
        let val = eval_and_get(
            r#"
            var arr = [[1, 2, 3], [4, 5, 6]];
            (function() {
                var sub = arr[0];
                sub[0] = 99;
            })();
        "#,
            "arr",
        );
        if let Value::Array(outer) = val {
            if let Some(Value::Array(inner)) = outer.get(0) {
                assert_eq!(inner.get(0), Some(Value::Number(99.0)),
                    "arr[0][0] should be 99 after alias mutation in IIFE");
            } else {
                panic!("Expected inner array");
            }
        } else {
            panic!("Expected outer array");
        }
    }

    #[test]
    fn test_array_alias_xor_mutation() {
        // XOR mutation pattern from simple-full.js decryption
        let val = eval_and_get(
            r#"
            var arr = [[100, 200], [50, 60]];
            (function() {
                var i = 0;
                while (i < 2) {
                    var sub = arr[i];
                    var j = 0;
                    while (j < 2) {
                        sub[j] = sub[j] ^ 0xFF;
                        j++;
                    }
                    i++;
                }
            })();
        "#,
            "arr",
        );
        // arr[0][0] should be 100 ^ 0xFF = 155
        // arr[0][1] should be 200 ^ 0xFF = 55
        if let Value::Array(outer) = val {
            if let Some(Value::Array(inner)) = outer.get(0) {
                assert_eq!(inner.get(0), Some(Value::Number(155.0)),
                    "arr[0][0] should be 100 ^ 255 = 155");
                assert_eq!(inner.get(1), Some(Value::Number(55.0)),
                    "arr[0][1] should be 200 ^ 255 = 55");
            } else {
                panic!("Expected inner array");
            }
        } else {
            panic!("Expected outer array");
        }
    }

    #[test]
    fn test_object_alias_mutation() {
        // Object aliasing should also work
        let val = eval_and_get(
            r#"
            var obj = { inner: { x: 1 } };
            var alias = obj.inner;
            alias.x = 99;
        "#,
            "obj",
        );
        if let Value::Object(outer) = val {
            if let Some(Value::Object(inner)) = outer.get("inner") {
                assert_eq!(inner.get("x"), Some(Value::Number(99.0)),
                    "obj.inner.x should be 99 after alias mutation");
            } else {
                panic!("Expected inner object");
            }
        } else {
            panic!("Expected outer object");
        }
    }

}
