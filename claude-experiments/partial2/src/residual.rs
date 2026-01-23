use crate::env::Value;
use std::collections::HashSet;

/// Check if an identifier is actually used in the source code.
/// Uses word boundary checking to avoid false positives (e.g., "v1" matching "v10").
fn is_identifier_used_in_source(source: &str, identifier: &str) -> bool {
    // Find all occurrences and check if they're at word boundaries
    let id_bytes = identifier.as_bytes();
    let src_bytes = source.as_bytes();

    if id_bytes.len() > src_bytes.len() {
        return false;
    }

    let mut i = 0;
    while i <= src_bytes.len() - id_bytes.len() {
        if &src_bytes[i..i + id_bytes.len()] == id_bytes {
            // Check that it's at a word boundary
            let before_ok = i == 0 || !is_identifier_char(src_bytes[i - 1]);
            let after_ok = i + id_bytes.len() == src_bytes.len()
                || !is_identifier_char(src_bytes[i + id_bytes.len()]);

            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Check if a byte is a valid identifier character (alphanumeric or underscore)
fn is_identifier_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

/// Represents a binding in the residual program
#[derive(Debug, Clone)]
pub enum Binding {
    /// A fully static value that can be emitted directly
    Static(Value),
    /// A dynamic expression that must be preserved (stored as source code)
    Dynamic(String),
}

/// An item in the residual program, in order
#[derive(Debug, Clone)]
pub enum ResidualItem {
    /// A variable binding (let x = ...)
    Binding { name: String, binding: Binding },
    /// A preserved statement (original source code)
    Statement(String),
    /// A preserved function definition
    Function(String),
}

/// What to do with a top-level statement
#[derive(Debug, Clone)]
pub enum StmtAction {
    /// Statement was fully consumed, emit nothing
    Consumed,
    /// Emit the statement with these bindings (for variable declarations)
    EmitBindings(Vec<(String, Binding)>),
    /// Preserve the original statement source with dependency info
    Preserve {
        source: String,
        defines: HashSet<String>,
        uses: HashSet<String>,
    },
    /// Preserve a function if it's not dead
    PreserveFunction { name: String, source: String },
}

/// Tracks what happened during evaluation to inform residual generation.
#[derive(Debug, Default)]
pub struct EvalTrace {
    /// What to do with each top-level statement, by index
    pub stmt_actions: Vec<(usize, StmtAction)>,
    /// Functions that were called (and thus their call sites can be removed)
    pub called_functions: HashSet<String>,
    /// Functions that are still "live" - might be called later or exported
    pub live_functions: HashSet<String>,
    /// Statements that were fully evaluated (indices into original program)
    pub consumed_statements: HashSet<usize>,
    /// Residual statements generated during evaluation (for specialization)
    pub residual_stmts: Vec<String>,
    /// Specialized function bodies: maps variable name to specialized arrow function source
    /// When a function stored in a variable is called, we save its specialized body here
    pub specialized_functions: std::collections::HashMap<String, String>,

    // Legacy fields for compatibility
    pub items: Vec<ResidualItem>,
    pub bindings: Vec<(String, Binding)>,
    pub preserved_functions: Vec<(String, String)>,
}

impl EvalTrace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_function_called(&mut self, name: &str) {
        self.called_functions.insert(name.to_string());
    }

    pub fn mark_function_live(&mut self, name: &str) {
        self.live_functions.insert(name.to_string());
    }

    pub fn mark_statement_consumed(&mut self, idx: usize) {
        self.consumed_statements.insert(idx);
    }

    pub fn add_static_binding(&mut self, name: String, value: Value) {
        self.bindings.push((name.clone(), Binding::Static(value.clone())));
        self.items.push(ResidualItem::Binding {
            name,
            binding: Binding::Static(value),
        });
    }

    pub fn add_dynamic_binding(&mut self, name: String, expr_source: String) {
        self.bindings.push((name.clone(), Binding::Dynamic(expr_source.clone())));
        self.items.push(ResidualItem::Binding {
            name,
            binding: Binding::Dynamic(expr_source),
        });
    }

    pub fn add_preserved_function(&mut self, name: String, source: String) {
        self.preserved_functions.push((name, source.clone()));
        self.items.push(ResidualItem::Function(source));
    }

    pub fn add_preserved_statement(&mut self, source: String) {
        self.items.push(ResidualItem::Statement(source));
    }

    /// Record what to do with a statement at a given index
    pub fn set_stmt_action(&mut self, idx: usize, action: StmtAction) {
        self.stmt_actions.push((idx, action));
    }

    /// Check if a function is dead (called and not needed anymore)
    pub fn is_function_dead(&self, name: &str) -> bool {
        self.called_functions.contains(name) && !self.live_functions.contains(name)
    }

    /// Save a specialized function body for a variable
    /// This is used when a function stored in a variable is called and specialized
    pub fn set_specialized_function(&mut self, var_name: &str, specialized_source: String) {
        self.specialized_functions.insert(var_name.to_string(), specialized_source);
    }

    /// Get the specialized function body for a variable, if any
    pub fn get_specialized_function(&self, var_name: &str) -> Option<&String> {
        self.specialized_functions.get(var_name)
    }
}

/// Convert a Value to JavaScript source code
/// For top-level bindings, closures are wrapped in IIFEs to preserve their captured scope
pub fn value_to_js(value: &Value) -> Result<String, String> {
    value_to_js_impl(value, true)
}

/// Convert a Value to JavaScript source code
/// `toplevel` indicates whether this is a top-level binding (true) or nested inside an object/array (false)
/// Closures only get IIFE-wrapped at the top level; nested closures just emit their source
fn value_to_js_impl(value: &Value, toplevel: bool) -> Result<String, String> {
    match value {
        Value::Number(n) => {
            if n.is_nan() {
                Ok("NaN".to_string())
            } else if n.is_infinite() {
                if *n > 0.0 {
                    Ok("Infinity".to_string())
                } else {
                    Ok("-Infinity".to_string())
                }
            } else if *n == n.trunc() && n.abs() < 1e15 {
                // Integer-like, emit without decimal
                Ok(format!("{}", *n as i64))
            } else {
                Ok(format!("{}", n))
            }
        }
        Value::String(s) => {
            // Escape the string properly
            let escaped = s
                .replace('\\', "\\\\")
                .replace('\"', "\\\"")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            Ok(format!("\"{}\"", escaped))
        }
        Value::Bool(b) => Ok(format!("{}", b)),
        Value::Null => Ok("null".to_string()),
        Value::Undefined => Ok("undefined".to_string()),
        Value::Array(elements) => {
            let mut parts = Vec::new();
            for elem in elements.iter() {
                parts.push(value_to_js_impl(&elem, false)?);
            }
            Ok(format!("[{}]", parts.join(", ")))
        }
        Value::Object(props) => {
            // For nested objects (inside another Object/Array), just emit object literal
            // without IIFE analysis to avoid infinite recursion
            if !toplevel {
                let mut obj_parts = Vec::new();
                for (key, val) in props.iter() {
                    let val_js = value_to_js_impl(&val, false)?;
                    obj_parts.push(format!("{}: {}", key, val_js));
                }
                return Ok(format!("{{{}}}", obj_parts.join(", ")));
            }

            // Collect all captured non-closure variables from closures in this object
            let mut all_captured: Vec<(String, Value)> = Vec::new();
            let mut closure_names: HashSet<String> = HashSet::new();

            // First pass: collect closure names
            for (_, val) in props.iter() {
                if let Value::Closure { name, .. } = val {
                    if let Some(n) = name {
                        closure_names.insert(n.clone());
                    }
                }
            }

            // Second pass: collect captured variables from closures
            for (_, val) in props.iter() {
                if let Value::Closure { env, name, source, .. } = val {
                    let closure_name = name.as_ref().map(|s| s.as_str());
                    for (scope_idx, var_name, captured_val) in env.all_bindings() {
                        // Skip global scope bindings - they're already available
                        if scope_idx == 0 {
                            continue;
                        }
                        // Skip variables not actually referenced in the closure source
                        if !is_identifier_used_in_source(&source, &var_name) {
                            continue;
                        }
                        // Skip the closure itself
                        if closure_name == Some(var_name.as_str()) {
                            continue;
                        }
                        // Skip other closures (sibling functions in the object)
                        if matches!(captured_val, Value::Closure { .. }) {
                            continue;
                        }
                        // Skip if it's a closure name from this object
                        if closure_names.contains(&var_name) {
                            continue;
                        }
                        // Skip dynamic values that are just variable references to themselves
                        if let Value::Dynamic(s) = &captured_val {
                            if s == &var_name {
                                continue;
                            }
                        }
                        // Skip the `arguments` object - it's a JavaScript built-in
                        if var_name == "arguments" {
                            continue;
                        }
                        // Add if not already captured
                        if !all_captured.iter().any(|(n, _)| n == &var_name) {
                            all_captured.push((var_name.clone(), captured_val.clone()));
                        }
                    }
                }
            }

            // Build object literal
            let mut obj_parts = Vec::new();
            for (key, val) in props.iter() {
                let val_js = value_to_js_impl(&val, false)?;
                obj_parts.push(format!("{}: {}", key, val_js));
            }
            let obj_literal = format!("{{{}}}", obj_parts.join(", "));

            // If no captured variables, just return the object literal
            if all_captured.is_empty() {
                return Ok(obj_literal);
            }

            // Wrap in IIFE with captured variables
            let mut iife_body = Vec::new();
            for (var_name, val) in &all_captured {
                let val_js = value_to_js_impl(val, false)?;
                if val_js != *var_name {
                    iife_body.push(format!("var {} = {};", var_name, val_js));
                }
            }
            iife_body.push(format!("return {};", obj_literal));

            Ok(format!("(function() {{\n{}\n}})()", iife_body.join("\n")))
        }
        Value::Closure { source, env, name, .. } => {
            // For nested closures (inside Object/Array), just emit the function source
            // without IIFE wrapping - the captured scope is handled at the top level
            if !toplevel {
                return Ok(source.clone());
            }

            // Get captured bindings from the environment
            let all_bindings = env.all_bindings();

            // Filter out:
            // 1. The closure itself (to avoid infinite recursion)
            // 2. Built-in/global bindings (scope 0 is the global scope)
            // 3. Closures that reference the same function (sibling functions)
            let closure_name = name.as_ref().map(|s| s.as_str());
            let captured: Vec<_> = all_bindings.iter()
                .filter(|(scope_idx, var_name, val)| {
                    // Skip global scope bindings - they're already available in the residual
                    if *scope_idx == 0 {
                        return false;
                    }
                    // Skip variables not actually referenced in the closure source
                    // Use word boundary check to avoid matching substrings (e.g., "v1" in "v10")
                    if !is_identifier_used_in_source(source, var_name) {
                        return false;
                    }
                    // Skip the closure itself
                    if closure_name == Some(var_name.as_str()) {
                        return false;
                    }
                    // Skip other closures (they'll be emitted separately or are siblings)
                    if matches!(val, Value::Closure { .. }) {
                        return false;
                    }
                    // Skip dynamic values that reference the closure we're emitting (avoids circular refs)
                    if let Value::Dynamic(s) = val {
                        // Skip self-referential declarations
                        if s == var_name {
                            return false;
                        }
                        // Skip if the dynamic value contains a reference to this closure
                        if let Some(cname) = closure_name {
                            if s.contains(cname) {
                                return false;
                            }
                        }
                    }
                    // Skip the `arguments` object - it's a JavaScript built-in available in regular functions
                    if var_name == "arguments" {
                        return false;
                    }
                    true
                })
                .collect();

            // If there are no captured variables, just return the function source
            if captured.is_empty() {
                return Ok(source.clone());
            }

            // Build an IIFE that wraps the closure with its captured scope
            let mut iife_body = Vec::new();

            // Emit captured variables (use toplevel=false to avoid nested IIFEs)
            for (_, var_name, val) in &captured {
                let val_js = value_to_js_impl(val, false)?;
                // Skip self-referential declarations
                if val_js != *var_name {
                    iife_body.push(format!("var {} = {};", var_name, val_js));
                }
            }

            // Check if name is an internal generated name (__anon_N or __arrow_N)
            // These are only used for internal tracking and shouldn't be used in output
            let is_internal_name = name.as_ref().map_or(false, |n| {
                n.starts_with("__anon_") || n.starts_with("__arrow_")
            });

            // For internal names or no name, just return the function source directly
            // For real function names, we need to emit the function and return by name
            if is_internal_name || name.is_none() {
                // Just return the function expression directly
                iife_body.push(format!("return {};", source));
            } else {
                // Named function - emit it and return by name
                iife_body.push(source.clone());
                iife_body.push(format!("return {};", name.as_ref().unwrap()));
            }

            Ok(format!("(function() {{\n{}\n}})()", iife_body.join("\n")))
        }
        Value::ArrayBuffer { buffer } => {
            // Emit as new ArrayBuffer with Uint8Array initialization
            // Since we can't directly emit ArrayBuffer contents, we create it and fill it
            let bytes = buffer.to_vec();
            if bytes.iter().all(|&b| b == 0) {
                // All zeros - just emit the size
                Ok(format!("new ArrayBuffer({})", bytes.len()))
            } else {
                // Has data - need to create and fill via Uint8Array view
                let elements: Vec<String> = bytes.iter().map(|b| b.to_string()).collect();
                Ok(format!(
                    "(function() {{ var b = new ArrayBuffer({}); new Uint8Array(b).set([{}]); return b; }})()",
                    bytes.len(),
                    elements.join(", ")
                ))
            }
        }
        Value::TypedArray { kind, buffer, byte_offset, length } => {
            // Get the bytes for this view
            let element_size = kind.element_size();
            let byte_length = length * element_size;
            let bytes = buffer.get_bytes(*byte_offset, byte_length);
            let elements: Vec<String> = bytes.iter().map(|b| b.to_string()).collect();
            Ok(format!("new {}([{}])", kind.name(), elements.join(", ")))
        }
        Value::DataView { buffer, byte_offset, byte_length } => {
            // Emit DataView - for now emit the underlying buffer data
            let bytes = buffer.get_bytes(*byte_offset, *byte_length);
            if bytes.iter().all(|&b| b == 0) {
                Ok(format!("new DataView(new ArrayBuffer({}))", byte_length))
            } else {
                let elements: Vec<String> = bytes.iter().map(|b| b.to_string()).collect();
                Ok(format!(
                    "(function() {{ var b = new ArrayBuffer({}); new Uint8Array(b).set([{}]); return new DataView(b); }})()",
                    byte_length,
                    elements.join(", ")
                ))
            }
        }
        Value::TextDecoder { encoding } => {
            // Emit as new TextDecoder(encoding)
            if encoding == "utf-8" {
                Ok("new TextDecoder()".to_string())
            } else {
                Ok(format!("new TextDecoder(\"{}\")", encoding))
            }
        }
        Value::Dynamic(expr) => {
            // Return the residual expression
            Ok(expr.clone())
        }
    }
}

/// A residual statement with dependency information for reordering
struct ResidualStmtWithDeps {
    source: String,
    defines: HashSet<String>,
    uses: HashSet<String>,
}

/// Reorder statements so that definitions come before uses
fn reorder_by_dependencies(stmts: Vec<ResidualStmtWithDeps>) -> Vec<String> {
    use std::collections::HashMap;

    if stmts.is_empty() {
        return Vec::new();
    }

    // Build def_index: variable -> statement index that defines it
    let mut def_index: HashMap<String, usize> = HashMap::new();
    for (i, stmt) in stmts.iter().enumerate() {
        for def in &stmt.defines {
            def_index.insert(def.clone(), i);
        }
    }

    // Build dependency graph: for each statement, which statement indices must come before it
    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); stmts.len()];
    for (i, stmt) in stmts.iter().enumerate() {
        for var_used in &stmt.uses {
            if let Some(&def_i) = def_index.get(var_used) {
                if def_i != i {
                    deps[i].insert(def_i);
                }
            }
        }
    }

    // Topological sort using Kahn's algorithm
    // in_degree[i] = number of statements that must come before statement i
    let mut in_degree: Vec<usize> = vec![0; stmts.len()];
    for i in 0..stmts.len() {
        in_degree[i] = deps[i].len();
    }

    // Start with statements that have no dependencies
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for (i, &degree) in in_degree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(i);
        }
    }

    let mut result_order: Vec<usize> = Vec::new();
    while let Some(i) = queue.pop_front() {
        result_order.push(i);

        // For each statement j that depends on statement i, decrement its in-degree
        for j in 0..stmts.len() {
            if deps[j].contains(&i) {
                in_degree[j] -= 1;
                if in_degree[j] == 0 {
                    queue.push_back(j);
                }
            }
        }
    }

    // If we couldn't order all statements, there's a cycle - fall back to original order
    if result_order.len() != stmts.len() {
        return stmts.into_iter().map(|s| s.source).collect();
    }

    // Return statements in dependency order
    result_order.into_iter().map(|i| stmts[i].source.clone()).collect()
}

/// Extract identifier references from a JavaScript expression string
/// This is a simple scan that looks for valid JS identifier patterns
pub fn extract_identifiers(expr: &str) -> HashSet<String> {
    use std::collections::HashSet;
    let mut result = HashSet::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Skip string literals
        if chars[i] == '"' || chars[i] == '\'' {
            let quote = chars[i];
            i += 1;
            while i < chars.len() && chars[i] != quote {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            i += 1;
            continue;
        }

        // Check for identifier start
        if chars[i].is_alphabetic() || chars[i] == '_' || chars[i] == '$' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '$') {
                i += 1;
            }
            let ident: String = chars[start..i].iter().collect();
            // Skip JavaScript keywords
            let keywords = ["let", "var", "const", "function", "return", "if", "else", "while",
                          "for", "switch", "case", "break", "continue", "true", "false", "null",
                          "undefined", "new", "this", "typeof", "instanceof", "in", "of", "try",
                          "catch", "finally", "throw", "class", "extends", "super", "import",
                          "export", "default", "async", "await", "yield", "void", "delete"];
            if !keywords.contains(&ident.as_str()) {
                result.insert(ident);
            }
        } else {
            i += 1;
        }
    }
    result
}

/// Extract the variable name being defined from a declaration statement
/// Handles: "var x = ...", "let x = ...", "const x = ..."
fn extract_defined_var(stmt: &str) -> Option<String> {
    let trimmed = stmt.trim();
    for prefix in &["var ", "let ", "const "] {
        if trimmed.starts_with(prefix) {
            let rest = &trimmed[prefix.len()..];
            // Find the = sign
            if let Some(eq_pos) = rest.find('=') {
                let var_part = rest[..eq_pos].trim();
                // Handle simple identifier (not destructuring)
                if var_part.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '$') {
                    return Some(var_part.to_string());
                }
            }
        }
    }
    None
}

/// Reorder a list of IIFE body statements based on variable dependencies
/// This parses each statement to find what it defines and uses
pub fn reorder_iife_body(statements: Vec<String>) -> Vec<String> {
    if statements.len() <= 1 {
        return statements;
    }

    // Build dependency info for each statement
    let mut stmts_with_deps: Vec<ResidualStmtWithDeps> = Vec::new();
    for stmt in statements {
        let mut defines = HashSet::new();
        let mut uses = HashSet::new();

        // Check if this is a variable declaration
        if let Some(var_name) = extract_defined_var(&stmt) {
            defines.insert(var_name.clone());
            // Extract identifiers from the RHS
            if let Some(eq_pos) = stmt.find('=') {
                let rhs = &stmt[eq_pos + 1..];
                uses = extract_identifiers(rhs);
                // Don't count self-reference
                uses.remove(&var_name);
            }
        } else if stmt.trim().starts_with("function ") {
            // Function declaration - extract name
            let trimmed = stmt.trim();
            let after_fn = &trimmed[9..]; // Skip "function "
            if let Some(paren_pos) = after_fn.find('(') {
                let func_name = after_fn[..paren_pos].trim();
                if !func_name.is_empty() {
                    defines.insert(func_name.to_string());
                }
            }
            // Functions can use variables from their body, but for ordering
            // purposes we only care about the declaration, not the body execution
        } else {
            // Other statement - extract all identifiers as uses
            uses = extract_identifiers(&stmt);
        }

        stmts_with_deps.push(ResidualStmtWithDeps { source: stmt, defines, uses });
    }

    reorder_by_dependencies(stmts_with_deps)
}

/// Extract all variable declarations from a JavaScript statement string,
/// including nested ones (e.g., inside try-catch, for loops, etc.)
/// Returns a set of variable names that are declared with var/let/const
fn extract_all_declarations(stmt: &str) -> HashSet<String> {
    let mut result = HashSet::new();
    let chars: Vec<char> = stmt.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Skip string literals
        if chars[i] == '"' || chars[i] == '\'' {
            let quote = chars[i];
            i += 1;
            while i < chars.len() && chars[i] != quote {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            i += 1;
            continue;
        }

        // Check for var/let/const keywords
        let remaining: String = chars[i..].iter().collect();
        for keyword in &["var ", "let ", "const "] {
            if remaining.starts_with(keyword) {
                // Check that it's not part of a larger identifier
                if i == 0 || !chars[i - 1].is_alphanumeric() && chars[i - 1] != '_' && chars[i - 1] != '$' {
                    i += keyword.len();
                    // Skip whitespace
                    while i < chars.len() && chars[i].is_whitespace() {
                        i += 1;
                    }
                    // Extract variable name
                    let start = i;
                    while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '$') {
                        i += 1;
                    }
                    if i > start {
                        let var_name: String = chars[start..i].iter().collect();
                        result.insert(var_name);
                    }
                }
                break;
            }
        }

        i += 1;
    }

    result
}

/// Filter IIFE body to remove dead code:
/// - Remove variable declarations for variables not used in residual statements
/// - Remove variable declarations for variables already declared in residual statements
/// - Remove function declarations that have been fully inlined
/// Returns the filtered body and a boolean indicating if a return statement is needed
pub fn filter_dead_code(
    bindings: Vec<(String, String)>,  // (name, value_js)
    residual_stmts: &[String],
    return_expr: &str,
) -> (Vec<String>, bool) {
    // First, collect all identifiers used in residual statements and return expr
    let mut used: HashSet<String> = HashSet::new();
    for stmt in residual_stmts {
        for ident in extract_identifiers(stmt) {
            used.insert(ident);
        }
    }
    for ident in extract_identifiers(return_expr) {
        used.insert(ident);
    }

    // IMPORTANT: Also collect variables that are DECLARED inside residual statements
    // These should NOT be emitted as separate bindings (would cause duplicate declarations)
    let mut declared_in_residual: HashSet<String> = HashSet::new();
    for stmt in residual_stmts {
        for var_name in extract_all_declarations(stmt) {
            declared_in_residual.insert(var_name);
        }
    }

    // Build the output, only including bindings that are used
    // We need to iterate until fixpoint since bindings can reference each other
    let mut needed: HashSet<String> = used.clone();
    loop {
        let mut changed = false;
        for (name, value_js) in &bindings {
            if needed.contains(name) {
                // This binding is needed, so we also need any variables it references
                for ident in extract_identifiers(value_js) {
                    if !needed.contains(&ident) {
                        needed.insert(ident);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Now filter the bindings
    let mut result = Vec::new();
    for (name, value_js) in bindings {
        if needed.contains(&name) {
            // Don't emit self-referential declarations (e.g., let v5 = v5)
            if value_js != name {
                // Don't emit bindings for variables already declared in residual statements
                // This prevents duplicate declarations when a var/let/const is inside the residual
                if !declared_in_residual.contains(&name) {
                    result.push(format!("let {} = {};", name, value_js));
                }
            }
        }
    }

    // Check if return is needed - it's not needed if:
    // 1. The return expression is "undefined" AND
    // 2. There are actual residual statements (otherwise we need to return something)
    let needs_return = return_expr != "undefined" || residual_stmts.is_empty();

    (result, needs_return)
}

/// Generate residual JavaScript from evaluation trace
pub fn emit_residual(
    trace: &EvalTrace,
    _skip_dead_functions: bool,
) -> Result<String, String> {
    let mut stmts_with_deps: Vec<ResidualStmtWithDeps> = Vec::new();

    // Use the statement actions approach if available
    if !trace.stmt_actions.is_empty() {
        // Sort by index to maintain original order
        let mut actions: Vec<_> = trace.stmt_actions.clone();
        actions.sort_by_key(|(idx, _)| *idx);

        for (_idx, action) in actions {
            match action {
                StmtAction::Consumed => {
                    // Nothing to emit
                }
                StmtAction::EmitBindings(bindings) => {
                    for (name, binding) in bindings {
                        match binding {
                            Binding::Static(value) => {
                                // Check if this closure has a specialized version
                                if matches!(value, Value::Closure { .. }) {
                                    if let Some(specialized) = trace.get_specialized_function(&name) {
                                        // Emit the specialized function
                                        let mut defines = HashSet::new();
                                        defines.insert(name.clone());
                                        stmts_with_deps.push(ResidualStmtWithDeps {
                                            source: format!("var {} = {};", name, specialized),
                                            defines,
                                            uses: HashSet::new(),
                                        });
                                        continue;
                                    }
                                    // No specialized version - emit the closure with its captured scope
                                    // value_to_js will wrap it in an IIFE with captured variables
                                }
                                let js_value = value_to_js(&value)?;
                                let mut defines = HashSet::new();
                                defines.insert(name.clone());
                                stmts_with_deps.push(ResidualStmtWithDeps {
                                    source: format!("let {} = {};", name, js_value),
                                    defines,
                                    uses: HashSet::new(), // Static values have no dynamic dependencies
                                });
                            }
                            Binding::Dynamic(expr_source) => {
                                let mut defines = HashSet::new();
                                defines.insert(name.clone());
                                stmts_with_deps.push(ResidualStmtWithDeps {
                                    source: format!("let {} = {};", name, expr_source),
                                    defines,
                                    uses: HashSet::new(), // TODO: Could parse expr_source for refs
                                });
                            }
                        }
                    }
                }
                StmtAction::Preserve { source, defines, uses } => {
                    stmts_with_deps.push(ResidualStmtWithDeps {
                        source,
                        defines,
                        uses,
                    });
                }
                StmtAction::PreserveFunction { name, source } => {
                    // Only emit if function is not dead
                    if !trace.is_function_dead(&name) {
                        let mut defines = HashSet::new();
                        defines.insert(name);
                        stmts_with_deps.push(ResidualStmtWithDeps {
                            source,
                            defines,
                            uses: HashSet::new(), // Function declarations don't use variables at declaration time
                        });
                    }
                }
            }
        }

        // Reorder statements based on dependencies
        let reordered = reorder_by_dependencies(stmts_with_deps);
        return Ok(reordered.join("\n"));
    }

    // Fallback: use the items-based approach
    let mut lines = Vec::new();
    if !trace.items.is_empty() {
        for item in &trace.items {
            match item {
                ResidualItem::Binding { name, binding } => {
                    match binding {
                        Binding::Static(value) => {
                            // Closures are now handled by value_to_js with IIFE wrapping
                            let js_value = value_to_js(value)?;
                            lines.push(format!("let {} = {};", name, js_value));
                        }
                        Binding::Dynamic(expr_source) => {
                            lines.push(format!("let {} = {};", name, expr_source));
                        }
                    }
                }
                ResidualItem::Statement(source) => {
                    lines.push(source.clone());
                }
                ResidualItem::Function(source) => {
                    lines.push(source.clone());
                }
            }
        }
        return Ok(lines.join("\n"));
    }

    // Legacy path - emit preserved functions first
    for (_name, source) in &trace.preserved_functions {
        lines.push(source.clone());
    }

    // Emit bindings in order
    for (name, binding) in &trace.bindings {
        match binding {
            Binding::Static(value) => {
                // Closures are now handled by value_to_js with IIFE wrapping
                let js_value = value_to_js(value)?;
                lines.push(format!("let {} = {};", name, js_value));
            }
            Binding::Dynamic(expr_source) => {
                lines.push(format!("let {} = {};", name, expr_source));
            }
        }
    }

    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::SharedArray;

    #[test]
    fn test_value_to_js_number() {
        assert_eq!(value_to_js(&Value::Number(42.0)).unwrap(), "42");
        assert_eq!(value_to_js(&Value::Number(3.14)).unwrap(), "3.14");
        assert_eq!(value_to_js(&Value::Number(-5.0)).unwrap(), "-5");
    }

    #[test]
    fn test_value_to_js_string() {
        assert_eq!(value_to_js(&Value::String("hello".to_string())).unwrap(), "\"hello\"");
        assert_eq!(value_to_js(&Value::String("say \"hi\"".to_string())).unwrap(), "\"say \\\"hi\\\"\"");
        assert_eq!(value_to_js(&Value::String("line\nbreak".to_string())).unwrap(), "\"line\\nbreak\"");
    }

    #[test]
    fn test_value_to_js_bool() {
        assert_eq!(value_to_js(&Value::Bool(true)).unwrap(), "true");
        assert_eq!(value_to_js(&Value::Bool(false)).unwrap(), "false");
    }

    #[test]
    fn test_value_to_js_null_undefined() {
        assert_eq!(value_to_js(&Value::Null).unwrap(), "null");
        assert_eq!(value_to_js(&Value::Undefined).unwrap(), "undefined");
    }

    #[test]
    fn test_value_to_js_array() {
        let arr = Value::Array(SharedArray::new(vec![
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]));
        assert_eq!(value_to_js(&arr).unwrap(), "[1, 2, 3]");
    }

    #[test]
    fn test_value_to_js_nested_array() {
        let arr = Value::Array(SharedArray::new(vec![
            Value::Array(SharedArray::new(vec![Value::Number(1.0), Value::Number(2.0)])),
            Value::Array(SharedArray::new(vec![Value::Number(3.0), Value::Number(4.0)])),
        ]));
        assert_eq!(value_to_js(&arr).unwrap(), "[[1, 2], [3, 4]]");
    }

    #[test]
    fn test_value_to_js_dynamic_returns_residual() {
        // Dynamic values now carry their residual expression
        assert_eq!(value_to_js(&Value::Dynamic("x + 1".to_string())).unwrap(), "x + 1");
        assert_eq!(value_to_js(&Value::Dynamic("foo.bar()".to_string())).unwrap(), "foo.bar()");
    }

    #[test]
    fn test_emit_residual_static() {
        let mut trace = EvalTrace::new();
        trace.add_static_binding("x".to_string(), Value::Number(42.0));
        trace.add_static_binding("arr".to_string(), Value::Array(SharedArray::new(vec![
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ])));

        let result = emit_residual(&trace, true).unwrap();
        assert!(result.contains("let x = 42;"));
        assert!(result.contains("let arr = [1, 2, 3];"));
    }

    #[test]
    fn test_emit_residual_dynamic() {
        let mut trace = EvalTrace::new();
        trace.add_static_binding("a".to_string(), Value::Number(5.0));
        trace.add_dynamic_binding("b".to_string(), "input".to_string());
        trace.add_dynamic_binding("c".to_string(), "5 + input".to_string());

        let result = emit_residual(&trace, true).unwrap();
        assert!(result.contains("let a = 5;"));
        assert!(result.contains("let b = input;"));
        assert!(result.contains("let c = 5 + input;"));
    }

    #[test]
    fn test_emit_residual_preserves_order() {
        let mut trace = EvalTrace::new();
        trace.add_dynamic_binding("first".to_string(), "input".to_string());
        trace.add_static_binding("second".to_string(), Value::Number(42.0));
        trace.add_dynamic_binding("third".to_string(), "input + 1".to_string());

        let result = emit_residual(&trace, true).unwrap();
        let first_pos = result.find("let first").unwrap();
        let second_pos = result.find("let second").unwrap();
        let third_pos = result.find("let third").unwrap();

        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }

    #[test]
    fn test_extract_all_declarations() {
        // Simple var declaration
        let result = extract_all_declarations("var x = 5;");
        assert!(result.contains("x"));
        assert_eq!(result.len(), 1);

        // Nested in try-catch
        let result = extract_all_declarations("try { var v22 = v4(); } catch (e) {}");
        assert!(result.contains("v22"));
        // Note: catch parameter (e) is not detected - it uses different syntax than var/let/const
        // This is OK because catch parameters aren't the issue we're fixing

        // Multiple declarations
        let result = extract_all_declarations("var a = 1; let b = 2; const c = 3;");
        assert!(result.contains("a"));
        assert!(result.contains("b"));
        assert!(result.contains("c"));
        assert_eq!(result.len(), 3);

        // Nested in for loop
        let result = extract_all_declarations("for (var i = 0; i < 10; i++) { let j = i * 2; }");
        assert!(result.contains("i"));
        assert!(result.contains("j"));

        // Should not match var in string literal
        let result = extract_all_declarations("console.log(\"var x = 5\");");
        assert!(!result.contains("x"));

        // Declaration inside function body
        let result = extract_all_declarations("function foo() { var inner = 42; }");
        assert!(result.contains("inner"));
    }
}
