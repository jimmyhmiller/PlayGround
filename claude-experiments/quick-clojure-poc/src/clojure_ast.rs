use crate::value::Value;

/// Clojure Abstract Syntax Tree
///
/// This represents Clojure code after parsing but before compilation.
/// Much simpler than Beagle's AST since Clojure has uniform S-expression syntax.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    Literal(Value),

    // Variable reference
    Var {
        namespace: Option<String>,  // None = unqualified, Some("user") = qualified
        name: String,
    },

    // Namespace special forms
    Ns {
        name: String,
    },

    Use {
        namespace: String,
    },

    // Special forms
    Def {
        name: String,
        value: Box<Expr>,
        metadata: Option<im::HashMap<String, Value>>,
    },

    /// (set! var value)
    /// Modifies a thread-local binding (only works within binding context)
    Set {
        var: Box<Expr>,
        value: Box<Expr>,
    },

    If {
        test: Box<Expr>,
        then: Box<Expr>,
        else_: Option<Box<Expr>>,
    },

    Do {
        exprs: Vec<Expr>,
    },

    /// (binding [var1 val1 var2 val2 ...] body)
    /// Establishes thread-local bindings for dynamic vars
    Binding {
        bindings: Vec<(String, Box<Expr>)>,  // [(var-name, value-expr), ...]
        body: Vec<Expr>,  // Multiple expressions in body (like do)
    },

    /// (let [x 10 y 20] (+ x y))
    /// Establishes lexical (stack-allocated) local bindings
    /// Bindings are sequential - each can see prior bindings
    Let {
        bindings: Vec<(String, Box<Expr>)>,  // [(name, value-expr), ...]
        body: Vec<Expr>,  // Body expressions (returns last)
    },

    /// (loop [x 0 sum 0] body...)
    /// Establishes bindings (like let) and a recursion point for recur
    Loop {
        bindings: Vec<(String, Box<Expr>)>,
        body: Vec<Expr>,
    },

    /// (recur expr1 expr2 ...)
    /// Jumps back to nearest loop/fn with new values for bindings
    Recur {
        args: Vec<Expr>,
    },

    /// (fn name? [params*] exprs*)
    /// (fn name? ([params*] exprs*)+)
    /// Function definition - first-class values
    /// Supports multi-arity dispatch and closures
    Fn {
        name: Option<String>,                      // Optional self-recursion name
        arities: Vec<crate::value::FnArity>,       // One or more arity overloads
    },

    // Function call (for now, all calls are the same)
    // Later we'll distinguish between special forms at parse time
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },

    // Quote - return value unevaluated
    Quote(Value),

    /// (var symbol) or #'symbol
    /// Returns the Var object itself, not its value
    VarRef {
        namespace: Option<String>,
        name: String,
    },

    /// (deftype* TypeName [field1 ^:mutable field2 ...])
    /// Defines a new type with named fields
    /// Fields can have ^:mutable metadata for mutable fields
    DefType {
        name: String,
        fields: Vec<FieldDef>,
    },

    /// (TypeName. arg1 arg2 ...) - constructor call
    /// Creates an instance of a deftype
    TypeConstruct {
        type_name: String,
        args: Vec<Expr>,
    },

    /// (.-field obj) - field access
    /// Reads a field from a deftype instance
    FieldAccess {
        field: String,
        object: Box<Expr>,
    },

    /// (set! (.-field obj) value) - field assignment
    /// Writes a value to a mutable field in a deftype instance
    /// Requires the field to be declared as ^:mutable
    FieldSet {
        field: String,
        object: Box<Expr>,
        value: Box<Expr>,
    },

    /// (throw expr)
    /// Throws an exception value
    Throw {
        exception: Box<Expr>,
    },

    /// (try expr* catch-clause* finally-clause?)
    /// Exception handling
    Try {
        body: Vec<Expr>,
        catches: Vec<CatchClause>,
        finally: Option<Vec<Expr>>,
    },

    // ========== Protocol System ==========

    /// (defprotocol ProtocolName
    ///   (method1 [this arg1])
    ///   (method2 [this arg1 arg2]))
    /// Defines a protocol with method signatures
    DefProtocol {
        name: String,
        methods: Vec<ProtocolMethodSig>,
    },

    /// (extend-type TypeName
    ///   ProtocolName
    ///   (method1 [this] body...)
    ///   ProtocolName2
    ///   (method2 [this x] body...))
    /// Extends protocols to an existing type
    ExtendType {
        type_name: String,
        implementations: Vec<ProtocolImpl>,
    },

    /// Protocol method call: (-first coll)
    /// Protocol methods are called like regular functions but dispatch on first arg's type
    ProtocolCall {
        method_name: String,  // e.g., "-first"
        args: Vec<Expr>,      // First arg is the dispatch target
    },

    /// (debugger expr)
    /// Inserts a BRK instruction before evaluating expr, returns expr's value
    /// Useful for debugging JIT code with lldb
    Debugger {
        expr: Box<Expr>,
    },
}

/// A catch clause in a try expression
/// (catch ExceptionType binding body*)
#[derive(Debug, Clone, PartialEq)]
pub struct CatchClause {
    pub exception_type: String,  // e.g., "Exception" (ignored for now - catch all)
    pub binding: String,         // Local binding for caught exception
    pub body: Vec<Expr>,
}

// ========== Protocol-related Structs ==========

/// Method signature in a defprotocol
/// (method-name [this arg1 arg2] [this arg1])  ; multiple arities
#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolMethodSig {
    pub name: String,
    /// Each arity is a list of parameter names (including 'this')
    pub arities: Vec<Vec<String>>,
}

/// Protocol implementation block in extend-type or deftype
#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolImpl {
    pub protocol_name: String,
    pub methods: Vec<ProtocolMethodImpl>,
}

/// Method implementation in a protocol block
#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolMethodImpl {
    pub name: String,
    pub params: Vec<String>,  // Includes 'this'
    pub body: Vec<Expr>,
}

/// Field definition in deftype
/// Supports ^:mutable metadata for mutable fields
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    pub name: String,
    pub mutable: bool,
}

/// Convert parsed Value to AST
///
/// This is the analyzer phase - we recognize special forms and
/// build an AST suitable for compilation.
pub fn analyze(value: &Value) -> Result<Expr, String> {
    match value {
        // Literals pass through
        Value::Nil | Value::Bool(_) | Value::Int(_) | Value::Float(_) |
        Value::String(_) | Value::Keyword(_) => {
            Ok(Expr::Literal(value.clone()))
        }

        // Vectors and other data structures are literals for now
        // (until we implement vector/map construction)
        Value::Vector(_) | Value::Map(_) | Value::Set(_) => {
            Ok(Expr::Literal(value.clone()))
        }

        // WithMeta wraps a value with metadata - analyze the inner value
        Value::WithMeta(_, inner) => {
            analyze(inner)
        }

        // Symbols become variable references
        // Parse qualified symbols: "user/foo" → Var { namespace: Some("user"), name: "foo" }
        // Special case: "/" is the division operator, not a qualified symbol
        Value::Symbol(s) => {
            if s == "/" {
                // Division operator - treat as unqualified symbol
                Ok(Expr::Var {
                    namespace: None,
                    name: s.clone(),
                })
            } else if let Some(idx) = s.find('/') {
                let namespace = s[..idx].to_string();
                let name = s[idx+1..].to_string();
                Ok(Expr::Var {
                    namespace: Some(namespace),
                    name,
                })
            } else {
                Ok(Expr::Var {
                    namespace: None,
                    name: s.clone(),
                })
            }
        }

        // Lists are either special forms or function calls
        Value::List(items) if !items.is_empty() => {
            // Check if first element is a special form or builtin macro
            if let Some(Value::Symbol(name)) = items.get(0) {
                match name.as_str() {
                    // ========== TRUE SPECIAL FORMS ==========
                    // These are actual Clojure special forms that cannot be implemented as macros
                    "def" => analyze_def(items),
                    "set!" => analyze_set(items),
                    "if" => analyze_if(items),
                    "do" => analyze_do(items),
                    "let" => analyze_let(items),
                    "loop" => analyze_loop(items),
                    "recur" => analyze_recur(items),
                    "fn" => analyze_fn(items),
                    "quote" => analyze_quote(items),
                    "var" => analyze_var_ref(items),
                    "throw" => analyze_throw(items),
                    "try" => analyze_try(items),

                    // ========== BUILTIN MACROS ==========
                    // These are macros in real Clojure but implemented here as special forms.
                    // TODO: Replace with actual macro implementations once we have macros.
                    "defn" => analyze_defn(items),           // (defn name [args] body) -> (def name (fn [args] body))
                    "declare" => analyze_declare(items),     // (declare x y z) -> (def x) (def y) (def z)
                    "if-not" => analyze_if_not(items),       // (if-not test then else) -> (if (not test) then else)
                    "when" => analyze_when(items),           // (when test body...) -> (if test (do body...) nil)
                    "when-not" => analyze_when_not(items),   // (when-not test body...) -> (if test nil (do body...))
                    "and" => analyze_and(items),             // (and x y) -> (if x y false)
                    "or" => analyze_or(items),               // (or x y) -> (if x x y)
                    "cond" => analyze_cond(items),           // (cond test1 expr1 ...) -> nested ifs
                    "dotimes" => analyze_dotimes(items),     // (dotimes [i n] body) -> loop
                    "ns" => analyze_ns(items),               // namespace declaration
                    "use" => analyze_use(items),             // (use 'ns) - deprecated, use require
                    "binding" => analyze_binding(items),     // dynamic binding
                    "deftype*" | "deftype" => analyze_deftype(items),  // type definition
                    "defprotocol" => analyze_defprotocol(items),       // protocol definition
                    "extend-type" => analyze_extend_type(items),       // extend protocol to type
                    "debugger" => analyze_debugger(items),   // debugging helper

                    _ => analyze_call(items),
                }
            } else {
                analyze_call(items)
            }
        }

        Value::List(_) => {
            Err("Cannot evaluate empty list".to_string())
        }

        Value::Function { .. } => {
            Err("Functions not yet implemented".to_string())
        }

        Value::Namespace { .. } => {
            Err("Namespace values not yet implemented".to_string())
        }
    }
}

fn analyze_def(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 3 {
        return Err(format!("def requires 2 arguments, got {}", items.len() - 1));
    }

    // Extract name and metadata from the symbol (which might have metadata attached)
    let (name, metadata) = match &items[1] {
        Value::WithMeta(meta, inner) => {
            match **inner {
                Value::Symbol(ref s) => (s.clone(), Some(meta.clone())),
                _ => return Err("def requires a symbol".to_string()),
            }
        }
        Value::Symbol(s) => (s.clone(), None),
        _ => return Err("def requires a symbol as first argument".to_string()),
    };

    let value = analyze(&items[2])?;

    Ok(Expr::Def {
        name,
        value: Box::new(value),
        metadata,
    })
}

/// Analyze (declare name1 name2 ...) - creates forward declarations
/// Each name becomes (def name nil) with a special flag to mark it as declared but unbound
fn analyze_declare(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() < 2 {
        return Err("declare requires at least one symbol".to_string());
    }

    // Create a series of def expressions wrapped in do
    let mut defs = Vec::new();
    for i in 1..items.len() {
        let name = match &items[i] {
            Value::Symbol(s) => s.clone(),
            _ => return Err("declare requires symbols".to_string()),
        };
        defs.push(Expr::Def {
            name,
            value: Box::new(Expr::Literal(Value::Nil)),
            metadata: None,
        });
    }

    if defs.len() == 1 {
        Ok(defs.pop().unwrap())
    } else {
        Ok(Expr::Do { exprs: defs })
    }
}

fn analyze_defn(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (defn name [params] body...)
    // (defn name "docstring" [params] body...)
    // (defn name ([params] body) ([params] body)...)
    if items.len() < 3 {
        return Err("defn requires at least a name and parameter vector".to_string());
    }

    // Get name
    let name = match &items[1] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("defn requires a symbol as first argument".to_string()),
    };

    // Build the fn form by removing "defn" and the name
    // (defn name [params] body...) -> (fn [params] body...)
    let mut fn_items = im::Vector::new();
    fn_items.push_back(Value::Symbol("fn".to_string()));

    // Skip optional docstring
    let mut start_idx = 2;
    if let Some(Value::String(_)) = items.get(2) {
        start_idx = 3;
    }

    // Copy remaining items (params and body)
    for i in start_idx..items.len() {
        fn_items.push_back(items[i].clone());
    }

    // Analyze as fn
    let fn_expr = analyze_fn(&fn_items)?;

    Ok(Expr::Def {
        name,
        value: Box::new(fn_expr),
        metadata: None,
    })
}

fn analyze_set(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 3 {
        return Err(format!("set! requires 2 arguments, got {}", items.len() - 1));
    }

    // First, analyze the target to see what it is
    let target = analyze(&items[1])?;
    let value = analyze(&items[2])?;

    // Check if target is a field access - if so, this is a field set
    match target {
        Expr::FieldAccess { field, object } => {
            Ok(Expr::FieldSet {
                field,
                object,
                value: Box::new(value),
            })
        }
        _ => {
            // Original behavior for var set!
            Ok(Expr::Set {
                var: Box::new(target),
                value: Box::new(value),
            })
        }
    }
}

fn analyze_if(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() < 3 || items.len() > 4 {
        return Err(format!("if requires 2 or 3 arguments, got {}", items.len() - 1));
    }

    let test = analyze(&items[1])?;
    let then = analyze(&items[2])?;
    let else_ = if items.len() == 4 {
        Some(Box::new(analyze(&items[3])?))
    } else {
        None
    };

    Ok(Expr::If {
        test: Box::new(test),
        then: Box::new(then),
        else_,
    })
}

/// Transforms (if-not test then else) into (if (not test) then else)
fn analyze_if_not(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() < 3 || items.len() > 4 {
        return Err(format!("if-not requires 2 or 3 arguments, got {}", items.len() - 1));
    }

    // Wrap the test in (not ...)
    let test = analyze(&items[1])?;
    let not_test = Expr::Call {
        func: Box::new(Expr::Var {
            namespace: None,
            name: "not".to_string(),
        }),
        args: vec![test],
    };

    let then = analyze(&items[2])?;
    let else_ = if items.len() == 4 {
        Some(Box::new(analyze(&items[3])?))
    } else {
        None
    };

    Ok(Expr::If {
        test: Box::new(not_test),
        then: Box::new(then),
        else_,
    })
}

/// Transforms (when test body...) into (if test (do body...) nil)
fn analyze_when(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() < 2 {
        return Err("when requires at least a test".to_string());
    }

    let test = analyze(&items[1])?;

    // Collect body expressions
    let mut body_exprs = Vec::new();
    for i in 2..items.len() {
        body_exprs.push(analyze(&items[i])?);
    }

    let then = if body_exprs.is_empty() {
        Expr::Literal(Value::Nil)
    } else if body_exprs.len() == 1 {
        body_exprs.pop().unwrap()
    } else {
        Expr::Do { exprs: body_exprs }
    };

    Ok(Expr::If {
        test: Box::new(test),
        then: Box::new(then),
        else_: None,  // when returns nil if test is false
    })
}

/// Transforms (when-not test body...) into (if (not test) (do body...) nil)
fn analyze_when_not(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() < 2 {
        return Err("when-not requires at least a test".to_string());
    }

    let test = analyze(&items[1])?;
    let not_test = Expr::Call {
        func: Box::new(Expr::Var {
            namespace: None,
            name: "not".to_string(),
        }),
        args: vec![test],
    };

    // Collect body expressions
    let mut body_exprs = Vec::new();
    for i in 2..items.len() {
        body_exprs.push(analyze(&items[i])?);
    }

    let then = if body_exprs.is_empty() {
        Expr::Literal(Value::Nil)
    } else if body_exprs.len() == 1 {
        body_exprs.pop().unwrap()
    } else {
        Expr::Do { exprs: body_exprs }
    };

    Ok(Expr::If {
        test: Box::new(not_test),
        then: Box::new(then),
        else_: None,
    })
}

/// Transforms (and) -> true, (and x) -> x, (and x y ...) -> (if x (and y ...) x)
fn analyze_and(items: &im::Vector<Value>) -> Result<Expr, String> {
    match items.len() {
        1 => {
            // (and) -> true
            Ok(Expr::Literal(Value::Bool(true)))
        }
        2 => {
            // (and x) -> x
            analyze(&items[1])
        }
        _ => {
            // (and x y ...) -> (let [temp x] (if temp (and y ...) temp))
            // For simplicity, expand as (if x (and y ...) false)
            // Note: This doesn't preserve the exact semantics (should return x if falsy)
            // but is good enough for most uses
            let test = analyze(&items[1])?;

            // Build (and y ...)
            let mut rest = im::Vector::new();
            rest.push_back(Value::Symbol("and".to_string()));
            for i in 2..items.len() {
                rest.push_back(items[i].clone());
            }
            let rest_and = analyze_and(&rest)?;

            Ok(Expr::If {
                test: Box::new(test),
                then: Box::new(rest_and),
                else_: Some(Box::new(Expr::Literal(Value::Bool(false)))),
            })
        }
    }
}

/// Transforms (or) -> nil, (or x) -> x, (or x y ...) -> (if x x (or y ...))
fn analyze_or(items: &im::Vector<Value>) -> Result<Expr, String> {
    match items.len() {
        1 => {
            // (or) -> nil
            Ok(Expr::Literal(Value::Nil))
        }
        2 => {
            // (or x) -> x
            analyze(&items[1])
        }
        _ => {
            // (or x y ...) -> (let [temp x] (if temp temp (or y ...)))
            // For simplicity: (if x true (or y ...))
            // Note: This doesn't preserve exact semantics (should return first truthy value)
            let test = analyze(&items[1])?;

            // Build (or y ...)
            let mut rest = im::Vector::new();
            rest.push_back(Value::Symbol("or".to_string()));
            for i in 2..items.len() {
                rest.push_back(items[i].clone());
            }
            let rest_or = analyze_or(&rest)?;

            Ok(Expr::If {
                test: Box::new(test.clone()),
                then: Box::new(test),
                else_: Some(Box::new(rest_or)),
            })
        }
    }
}

/// Transforms (cond test1 expr1 test2 expr2 ...) into nested if statements
/// (cond) -> nil
/// (cond :else expr) -> expr
/// (cond test expr rest...) -> (if test expr (cond rest...))
fn analyze_cond(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() == 1 {
        // (cond) -> nil
        return Ok(Expr::Literal(Value::Nil));
    }

    if (items.len() - 1) % 2 != 0 {
        return Err("cond requires an even number of forms".to_string());
    }

    // Process pairs
    fn build_cond(items: &im::Vector<Value>, idx: usize) -> Result<Expr, String> {
        if idx >= items.len() {
            // No more clauses, return nil
            return Ok(Expr::Literal(Value::Nil));
        }

        let test = &items[idx];
        let expr = &items[idx + 1];

        // Check for :else clause
        if let Value::Keyword(kw) = test {
            if kw == "else" {
                return analyze(expr);
            }
        }

        let test_expr = analyze(test)?;
        let then_expr = analyze(expr)?;
        let else_expr = build_cond(items, idx + 2)?;

        Ok(Expr::If {
            test: Box::new(test_expr),
            then: Box::new(then_expr),
            else_: Some(Box::new(else_expr)),
        })
    }

    build_cond(items, 1)
}

fn analyze_do(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() < 2 {
        return Err("do requires at least 1 argument".to_string());
    }

    let mut exprs = Vec::new();
    for i in 1..items.len() {
        exprs.push(analyze(&items[i])?);
    }

    Ok(Expr::Do { exprs })
}

fn analyze_debugger(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (debugger expr) - inserts breakpoint before evaluating expr
    if items.len() != 2 {
        return Err(format!("debugger requires 1 argument, got {}", items.len() - 1));
    }

    let expr = analyze(&items[1])?;
    Ok(Expr::Debugger { expr: Box::new(expr) })
}

fn analyze_quote(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 2 {
        return Err(format!("quote requires 1 argument, got {}", items.len() - 1));
    }

    Ok(Expr::Quote(items[1].clone()))
}

fn analyze_var_ref(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (var symbol) - returns the Var object itself
    if items.len() != 2 {
        return Err(format!("var requires 1 argument, got {}", items.len() - 1));
    }

    // Parse the symbol (may be qualified: ns/name or unqualified: name)
    match &items[1] {
        Value::Symbol(s) => {
            if s == "/" {
                Ok(Expr::VarRef {
                    namespace: None,
                    name: s.clone(),
                })
            } else if let Some(idx) = s.find('/') {
                let namespace = s[..idx].to_string();
                let name = s[idx+1..].to_string();
                Ok(Expr::VarRef {
                    namespace: Some(namespace),
                    name,
                })
            } else {
                Ok(Expr::VarRef {
                    namespace: None,
                    name: s.clone(),
                })
            }
        }
        _ => Err("var requires a symbol".to_string()),
    }
}

fn analyze_deftype(items: &im::Vector<Value>) -> Result<Expr, String> {
    // Two forms supported:
    // 1. Simple: (deftype* TypeName [field1 ^:mutable field2 ...])
    // 2. With protocols: (deftype TypeName [fields...] Protocol1 (method1 [this] body) ...)
    if items.len() < 3 {
        return Err("deftype requires at least a name and fields vector".to_string());
    }

    // Get type name
    let name = match &items[1] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("deftype requires a symbol as type name".to_string()),
    };

    // Get field names (for binding in method bodies) and field definitions
    let (field_names, fields) = match &items[2] {
        Value::Vector(v) => {
            let mut field_defs = Vec::new();
            let mut field_names = Vec::new();
            for field in v.iter() {
                let field_def = match field {
                    // Field with metadata (e.g., ^:mutable field-name)
                    Value::WithMeta(meta, inner) => {
                        let field_name = match &**inner {
                            Value::Symbol(s) => s.clone(),
                            _ => return Err(format!("deftype field must be a symbol, got {:?}", inner)),
                        };
                        // Check if :mutable is in the metadata
                        let is_mutable = meta.contains_key(":mutable") || meta.contains_key("mutable");
                        field_names.push(field_name.clone());
                        FieldDef {
                            name: field_name,
                            mutable: is_mutable,
                        }
                    }
                    // Plain field symbol (immutable)
                    Value::Symbol(s) => {
                        field_names.push(s.clone());
                        FieldDef {
                            name: s.clone(),
                            mutable: false,
                        }
                    }
                    _ => return Err(format!("deftype field must be a symbol, got {:?}", field)),
                };
                field_defs.push(field_def);
            }
            (field_names, field_defs)
        }
        _ => return Err("deftype requires a vector of field names".to_string()),
    };

    // If there are protocol implementations (items beyond name and fields)
    if items.len() > 3 {
        // Parse protocol implementations starting from index 3, with field names for binding
        let implementations = parse_protocol_implementations_with_fields(items, 3, &field_names)?;

        // Generate: (do (deftype* name fields) (extend-type name protocols...))
        let deftype_expr = Expr::DefType { name: name.clone(), fields };
        let extend_expr = Expr::ExtendType { type_name: name, implementations };

        Ok(Expr::Do { exprs: vec![deftype_expr, extend_expr] })
    } else {
        // Simple deftype without protocol implementations
        Ok(Expr::DefType { name, fields })
    }
}

fn analyze_ns(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 2 {
        return Err(format!("ns requires 1 argument, got {}", items.len() - 1));
    }

    let ns_name = match &items[1] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("ns requires a symbol as namespace name".to_string()),
    };

    Ok(Expr::Ns { name: ns_name })
}

fn analyze_use(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 2 {
        return Err(format!("use requires 1 argument, got {}", items.len() - 1));
    }

    let ns_name = match &items[1] {
        // Accept both symbols and quoted symbols: (use clojure.core) or (use 'clojure.core)
        Value::Symbol(s) => s.clone(),
        Value::List(quoted) if quoted.len() == 2 => {
            if let (Some(Value::Symbol(q)), Some(Value::Symbol(ns))) = (quoted.get(0), quoted.get(1)) {
                if q == "quote" {
                    ns.clone()
                } else {
                    return Err("use requires a symbol or quoted symbol".to_string());
                }
            } else {
                return Err("use requires a symbol or quoted symbol".to_string());
            }
        }
        _ => return Err("use requires a symbol or quoted symbol".to_string()),
    };

    Ok(Expr::Use { namespace: ns_name })
}

fn analyze_binding(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (binding [var1 val1 var2 val2 ...] body...)
    if items.len() < 3 {
        return Err("binding requires at least 2 arguments: bindings vector and body".to_string());
    }

    // Parse bindings vector
    let bindings_vec = match &items[1] {
        Value::Vector(v) => v,
        _ => return Err("binding requires a vector of bindings as first argument".to_string()),
    };

    // Bindings must be even (pairs of var/value)
    if bindings_vec.len() % 2 != 0 {
        return Err("binding vector must contain an even number of forms (var/value pairs)".to_string());
    }

    // Parse each binding pair
    let mut bindings = Vec::new();
    for i in (0..bindings_vec.len()).step_by(2) {
        let var_name = match &bindings_vec[i] {
            Value::Symbol(s) => {
                // Handle both qualified and unqualified symbols
                // For now, we'll store the full symbol name
                s.clone()
            }
            _ => return Err(format!("binding requires symbols as variable names, got {:?}", bindings_vec[i])),
        };

        let value_expr = analyze(&bindings_vec[i + 1])?;
        bindings.push((var_name, Box::new(value_expr)));
    }

    // Parse body expressions (like do - all expressions evaluated, last one returned)
    let mut body = Vec::new();
    for i in 2..items.len() {
        body.push(analyze(&items[i])?);
    }

    Ok(Expr::Binding { bindings, body })
}

fn analyze_let(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (let [x 10 y 20] (+ x y))
    //      ^^^^^^^^^  ^^^^^^^^^^
    //      bindings   body
    // Body is optional: (let [x 10]) returns nil

    if items.len() < 2 {
        return Err("let requires at least 1 argument: bindings vector".to_string());
    }

    // Parse bindings vector
    let bindings_vec = match &items[1] {
        Value::Vector(v) => v,
        _ => return Err("let requires a vector of bindings as first argument".to_string()),
    };

    // Bindings must be even (pairs of name/value)
    if bindings_vec.len() % 2 != 0 {
        return Err("let bindings vector must contain an even number of forms (name/value pairs)".to_string());
    }

    // Parse each binding pair
    let mut bindings = Vec::new();
    for i in (0..bindings_vec.len()).step_by(2) {
        let name = match &bindings_vec[i] {
            Value::Symbol(s) => s.clone(),
            _ => return Err(format!("let binding names must be symbols, got {:?}", bindings_vec[i])),
        };

        let value_expr = analyze(&bindings_vec[i + 1])?;
        bindings.push((name, Box::new(value_expr)));
    }

    // Parse body expressions (all evaluated, last one returned)
    // If empty body, return nil
    let mut body = Vec::new();
    for i in 2..items.len() {
        body.push(analyze(&items[i])?);
    }

    if body.is_empty() {
        // Empty body returns nil
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(Expr::Let { bindings, body })
}

fn analyze_loop(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (loop [x 0 sum 0] body...)
    if items.len() < 2 {
        return Err("loop requires at least 1 argument: bindings vector".to_string());
    }

    let bindings_vec = match &items[1] {
        Value::Vector(v) => v,
        _ => return Err("loop requires a vector of bindings as first argument".to_string()),
    };

    if bindings_vec.len() % 2 != 0 {
        return Err("loop bindings must contain an even number of forms".to_string());
    }

    let mut bindings = Vec::new();
    for i in (0..bindings_vec.len()).step_by(2) {
        let name = match &bindings_vec[i] {
            Value::Symbol(s) => s.clone(),
            _ => return Err(format!("loop binding names must be symbols, got {:?}", bindings_vec[i])),
        };
        let value_expr = analyze(&bindings_vec[i + 1])?;
        bindings.push((name, Box::new(value_expr)));
    }

    let mut body = Vec::new();
    for i in 2..items.len() {
        body.push(analyze(&items[i])?);
    }

    if body.is_empty() {
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(Expr::Loop { bindings, body })
}

/// Transforms (dotimes [i n] body...) into (loop [i 0] (when (< i n) body... (recur (inc i))))
fn analyze_dotimes(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() < 2 {
        return Err("dotimes requires at least a binding vector".to_string());
    }

    // Parse bindings [i n]
    let bindings_vec = match &items[1] {
        Value::Vector(v) => v,
        _ => return Err("dotimes requires a vector binding [i n]".to_string()),
    };

    if bindings_vec.len() != 2 {
        return Err("dotimes binding must be [i n]".to_string());
    }

    let var_name = match &bindings_vec[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("dotimes binding variable must be a symbol".to_string()),
    };

    let count_expr = analyze(&bindings_vec[1])?;

    // Parse body
    let mut body_exprs = Vec::new();
    for i in 2..items.len() {
        body_exprs.push(analyze(&items[i])?);
    }

    // Build: (loop [i 0] (when (< i n) body... (recur (inc i))))
    // The inc call
    let inc_call = Expr::Call {
        func: Box::new(Expr::Var { namespace: None, name: "inc".to_string() }),
        args: vec![Expr::Var { namespace: None, name: var_name.clone() }],
    };

    // The recur
    let recur_expr = Expr::Recur { args: vec![inc_call] };

    // Add recur to body
    body_exprs.push(recur_expr);

    // The when body (do body... (recur (inc i)))
    let when_body = if body_exprs.len() == 1 {
        body_exprs.pop().unwrap()
    } else {
        Expr::Do { exprs: body_exprs }
    };

    // The < comparison
    let lt_call = Expr::Call {
        func: Box::new(Expr::Var { namespace: None, name: "<".to_string() }),
        args: vec![
            Expr::Var { namespace: None, name: var_name.clone() },
            count_expr,
        ],
    };

    // The when (if test then nil)
    let when_expr = Expr::If {
        test: Box::new(lt_call),
        then: Box::new(when_body),
        else_: None,
    };

    // The loop bindings [i 0]
    let loop_bindings = vec![(var_name, Box::new(Expr::Literal(Value::Int(0))))];

    Ok(Expr::Loop {
        bindings: loop_bindings,
        body: vec![when_expr],
    })
}

fn analyze_recur(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (recur expr1 expr2 ...)
    let mut args = Vec::new();
    for i in 1..items.len() {
        args.push(analyze(&items[i])?);
    }
    Ok(Expr::Recur { args })
}

fn analyze_fn(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (fn name? [params] body*)
    // (fn name? ([params] body*)+)
    if items.len() < 2 {
        return Err("fn requires at least a parameter vector".to_string());
    }

    let mut idx = 1;

    // Check for optional name
    let name = if let Some(Value::Symbol(s)) = items.get(idx) {
        // Could be name or could be param vector
        // Name is a symbol, params is a vector
        // Peek ahead to disambiguate
        if idx + 1 < items.len() {
            match items.get(idx + 1) {
                Some(Value::Vector(_)) => {
                    // This is a name, next is params
                    idx += 1;
                    Some(s.clone())
                }
                Some(Value::List(_)) => {
                    // Multi-arity form starting with (params) list, no name
                    None
                }
                _ => {
                    // Ambiguous - assume no name if current is a vector
                    if matches!(items.get(idx), Some(Value::Vector(_))) {
                        None
                    } else {
                        return Err("fn requires parameter vector or arity forms".to_string());
                    }
                }
            }
        } else {
            // Only one element after fn - must be params vector
            None
        }
    } else {
        None
    };

    // Parse arities
    let mut arities = Vec::new();

    // Check if single-arity or multi-arity form
    let is_multi_arity = matches!(items.get(idx), Some(Value::List(_)));

    if is_multi_arity {
        // Multi-arity: (fn name? ([params] body)+ )
        for i in idx..items.len() {
            let arity_form = match &items[i] {
                Value::List(arity_items) => arity_items,
                _ => return Err("Multi-arity fn requires lists for each arity".to_string()),
            };

            let arity = parse_fn_arity(arity_form)?;
            arities.push(arity);
        }
    } else {
        // Single-arity: (fn name? [params] body*)
        let params_vec = match &items[idx] {
            Value::Vector(v) => v,
            _ => return Err("fn requires parameter vector".to_string()),
        };

        // Collect body expressions (everything after params)
        let body_start = idx + 1;
        let mut body_values = Vec::new();
        for i in body_start..items.len() {
            body_values.push(items[i].clone());
        }

        let arity = parse_fn_params_and_body(params_vec, &body_values)?;
        arities.push(arity);
    }

    // Validate arities
    validate_fn_arities(&arities)?;

    Ok(Expr::Fn { name, arities })
}

fn parse_fn_arity(arity_items: &im::Vector<Value>) -> Result<crate::value::FnArity, String> {
    // ([params] condition-map? body*)
    if arity_items.is_empty() {
        return Err("fn arity form cannot be empty".to_string());
    }

    let params_vec = match &arity_items[0] {
        Value::Vector(v) => v,
        _ => return Err("fn arity form must start with parameter vector".to_string()),
    };

    let mut body_values = Vec::new();
    for i in 1..arity_items.len() {
        body_values.push(arity_items[i].clone());
    }
    parse_fn_params_and_body(params_vec, &body_values)
}

fn parse_fn_params_and_body(
    params_vec: &im::Vector<Value>,
    body_items: &[Value],
) -> Result<crate::value::FnArity, String> {
    // Parse parameters: [x y] or [x y & rest]
    let mut params = Vec::new();
    let mut rest_param = None;
    let mut found_ampersand = false;

    for (i, param) in params_vec.iter().enumerate() {
        match param {
            Value::Symbol(s) if s == "&" => {
                if found_ampersand {
                    return Err("Only one & allowed in parameter list".to_string());
                }
                if i == params_vec.len() - 1 {
                    return Err("& must be followed by rest parameter".to_string());
                }
                found_ampersand = true;
            }
            Value::Symbol(s) => {
                if found_ampersand {
                    if rest_param.is_some() {
                        return Err("Only one rest parameter allowed after &".to_string());
                    }
                    rest_param = Some(s.clone());
                } else {
                    params.push(s.clone());
                }
            }
            _ => return Err(format!("fn parameters must be symbols, got {:?}", param)),
        }
    }

    // Parse condition map and body
    let (pre_conditions, post_conditions, body_start_idx) =
        if !body_items.is_empty() {
            if let Some(Value::Map(cond_map)) = body_items.first() {
                // Check if this is truly a condition map or just the body
                // Condition map has :pre and/or :post keys
                let has_pre = cond_map.contains_key(&Value::Keyword("pre".to_string()));
                let has_post = cond_map.contains_key(&Value::Keyword("post".to_string()));

                if has_pre || has_post {
                    // Parse condition map
                    let pre = if let Some(Value::Vector(pre_vec)) = cond_map.get(&Value::Keyword("pre".to_string())) {
                        pre_vec.iter()
                            .map(|v| analyze(v))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        Vec::new()
                    };

                    let post = if let Some(Value::Vector(post_vec)) = cond_map.get(&Value::Keyword("post".to_string())) {
                        post_vec.iter()
                            .map(|v| analyze(v))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        Vec::new()
                    };

                    (pre, post, 1)
                } else {
                    // Just a map in the body, not a condition map
                    (Vec::new(), Vec::new(), 0)
                }
            } else {
                (Vec::new(), Vec::new(), 0)
            }
        } else {
            (Vec::new(), Vec::new(), 0)
        };

    // Parse body expressions
    let mut body = Vec::new();
    for i in body_start_idx..body_items.len() {
        body.push(analyze(&body_items[i])?);
    }

    if body.is_empty() {
        // Empty body returns nil
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(crate::value::FnArity {
        params,
        rest_param,
        body,
        pre_conditions,
        post_conditions,
    })
}

fn validate_fn_arities(arities: &[crate::value::FnArity]) -> Result<(), String> {
    if arities.is_empty() {
        return Err("fn requires at least one arity".to_string());
    }

    let mut seen_arities = std::collections::HashSet::new();
    let mut variadic_count = 0;

    for arity in arities {
        let arity_num = arity.params.len();

        if arity.rest_param.is_some() {
            variadic_count += 1;
            if variadic_count > 1 {
                return Err("fn can have at most one variadic arity".to_string());
            }
        }

        if !seen_arities.insert(arity_num) {
            return Err(format!("Duplicate arity {} in fn", arity_num));
        }
    }

    Ok(())
}

fn analyze_throw(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (throw expr)
    if items.len() != 2 {
        return Err(format!("throw requires 1 argument, got {}", items.len() - 1));
    }

    let exception = analyze(&items[1])?;
    Ok(Expr::Throw {
        exception: Box::new(exception),
    })
}

fn analyze_try(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (try body... (catch ExType e handler...) ... (finally cleanup...))
    // Body is everything before catch/finally clauses
    // Can have multiple catch clauses
    // Finally is optional, must be last

    if items.len() < 2 {
        return Err("try requires at least 1 body expression".to_string());
    }

    let mut body = Vec::new();
    let mut catches = Vec::new();
    let mut finally = None;

    // Iterate through items[1..] and categorize them
    for i in 1..items.len() {
        let item = &items[i];

        // Check if this is a catch or finally clause
        if let Value::List(list_items) = item {
            if let Some(Value::Symbol(sym)) = list_items.get(0) {
                if sym == "catch" {
                    // (catch ExType binding body*)
                    if list_items.len() < 3 {
                        return Err("catch requires at least exception type and binding".to_string());
                    }

                    let exception_type = match &list_items[1] {
                        Value::Symbol(s) => s.clone(),
                        _ => return Err("catch exception type must be a symbol".to_string()),
                    };

                    let binding = match &list_items[2] {
                        Value::Symbol(s) => s.clone(),
                        _ => return Err("catch binding must be a symbol".to_string()),
                    };

                    let mut catch_body = Vec::new();
                    for j in 3..list_items.len() {
                        catch_body.push(analyze(&list_items[j])?);
                    }

                    if catch_body.is_empty() {
                        catch_body.push(Expr::Literal(Value::Nil));
                    }

                    catches.push(CatchClause {
                        exception_type,
                        binding,
                        body: catch_body,
                    });
                    continue;
                } else if sym == "finally" {
                    // (finally body*)
                    if finally.is_some() {
                        return Err("try can have at most one finally clause".to_string());
                    }

                    let mut finally_body = Vec::new();
                    for j in 1..list_items.len() {
                        finally_body.push(analyze(&list_items[j])?);
                    }

                    finally = Some(finally_body);
                    continue;
                }
            }
        }

        // Not a catch or finally - must be body
        if !catches.is_empty() || finally.is_some() {
            return Err("try body expressions must come before catch/finally".to_string());
        }

        body.push(analyze(item)?);
    }

    if body.is_empty() {
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(Expr::Try {
        body,
        catches,
        finally,
    })
}

// ========== Protocol Analyzer Functions ==========

fn analyze_defprotocol(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (defprotocol ProtocolName
    //   "Optional docstring"
    //   (method1 [this arg1])
    //   (method2 [this] [this arg1 arg2]))  ; multiple arities
    if items.len() < 2 {
        return Err("defprotocol requires at least a name".to_string());
    }

    // Get protocol name
    let name = match &items[1] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("defprotocol name must be a symbol".to_string()),
    };

    // Parse method signatures, skipping any protocol-level docstrings
    let mut methods = Vec::new();
    for i in 2..items.len() {
        match &items[i] {
            Value::String(_) => {
                // Protocol docstring - skip it
            }
            _ => {
                let method_sig = parse_protocol_method_sig(&items[i])?;
                methods.push(method_sig);
            }
        }
    }

    Ok(Expr::DefProtocol { name, methods })
}

fn parse_protocol_method_sig(value: &Value) -> Result<ProtocolMethodSig, String> {
    // (method-name [this arg1] [this arg1 arg2])  ; or just (method-name [this])
    let items = match value {
        Value::List(items) => items,
        _ => return Err("Protocol method signature must be a list".to_string()),
    };

    if items.is_empty() {
        return Err("Protocol method signature cannot be empty".to_string());
    }

    // Get method name
    let name = match &items[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("Protocol method name must be a symbol".to_string()),
    };

    // Parse arities - can be multiple vectors
    let mut arities = Vec::new();
    for i in 1..items.len() {
        match &items[i] {
            Value::Vector(params) => {
                let mut param_names = Vec::new();
                for param in params.iter() {
                    match param {
                        Value::Symbol(s) => param_names.push(s.clone()),
                        _ => return Err(format!("Protocol method parameter must be a symbol, got {:?}", param)),
                    }
                }
                if param_names.is_empty() || param_names[0] != "this" {
                    // For flexibility, don't require 'this' but recommend it
                }
                arities.push(param_names);
            }
            Value::String(_) => {
                // Docstring - skip it
            }
            _ => {
                return Err(format!("Protocol method arity must be a vector, got {:?}", items[i]));
            }
        }
    }

    if arities.is_empty() {
        return Err(format!("Protocol method {} requires at least one arity", name));
    }

    Ok(ProtocolMethodSig { name, arities })
}

fn analyze_extend_type(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (extend-type TypeName
    //   ProtocolName
    //   (method1 [this] body...)
    //   (method2 [this x] body...)
    //   AnotherProtocol
    //   (method3 [this] body...))
    if items.len() < 2 {
        return Err("extend-type requires at least a type name".to_string());
    }

    // Get type name
    let type_name = match &items[1] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("extend-type type name must be a symbol".to_string()),
    };

    // Parse protocol implementations
    let implementations = parse_protocol_implementations(&items, 2)?;

    Ok(Expr::ExtendType { type_name, implementations })
}

fn parse_protocol_implementations(items: &im::Vector<Value>, start_idx: usize) -> Result<Vec<ProtocolImpl>, String> {
    // Protocol implementations are grouped by protocol name:
    // ProtocolName
    // (method1 [this] body...)
    // (method2 [this x] body...)
    // AnotherProtocol
    // (method3 [this] body...)

    let mut implementations = Vec::new();
    let mut current_protocol: Option<String> = None;
    let mut current_methods: Vec<ProtocolMethodImpl> = Vec::new();

    for i in start_idx..items.len() {
        match &items[i] {
            // A bare symbol is a protocol name
            Value::Symbol(s) => {
                // Save previous protocol if any
                if let Some(protocol_name) = current_protocol.take() {
                    implementations.push(ProtocolImpl {
                        protocol_name,
                        methods: std::mem::take(&mut current_methods),
                    });
                }
                current_protocol = Some(s.clone());
            }
            // A list is a method implementation
            Value::List(method_items) => {
                if current_protocol.is_none() {
                    return Err("Method implementation found before protocol name".to_string());
                }
                let method_impl = parse_protocol_method_impl(method_items, &[])?;
                current_methods.push(method_impl);
            }
            _ => {
                return Err(format!("Expected protocol name or method implementation, got {:?}", items[i]));
            }
        }
    }

    // Save last protocol
    if let Some(protocol_name) = current_protocol {
        implementations.push(ProtocolImpl {
            protocol_name,
            methods: current_methods,
        });
    }

    Ok(implementations)
}

/// Parse protocol implementations for deftype, wrapping method bodies with field bindings
fn parse_protocol_implementations_with_fields(
    items: &im::Vector<Value>,
    start_idx: usize,
    field_names: &[String],
) -> Result<Vec<ProtocolImpl>, String> {
    let mut implementations = Vec::new();
    let mut current_protocol: Option<String> = None;
    let mut current_methods: Vec<ProtocolMethodImpl> = Vec::new();

    for i in start_idx..items.len() {
        match &items[i] {
            Value::Symbol(s) => {
                if let Some(protocol_name) = current_protocol.take() {
                    implementations.push(ProtocolImpl {
                        protocol_name,
                        methods: std::mem::take(&mut current_methods),
                    });
                }
                current_protocol = Some(s.clone());
            }
            Value::List(method_items) => {
                if current_protocol.is_none() {
                    return Err("Method implementation found before protocol name".to_string());
                }
                let method_impl = parse_protocol_method_impl(method_items, field_names)?;
                current_methods.push(method_impl);
            }
            _ => {
                return Err(format!("Expected protocol name or method implementation, got {:?}", items[i]));
            }
        }
    }

    if let Some(protocol_name) = current_protocol {
        implementations.push(ProtocolImpl {
            protocol_name,
            methods: current_methods,
        });
    }

    Ok(implementations)
}

fn parse_protocol_method_impl(items: &im::Vector<Value>, field_names: &[String]) -> Result<ProtocolMethodImpl, String> {
    // (method-name [this arg1 arg2] body...)
    if items.len() < 2 {
        return Err("Protocol method implementation requires at least name and params".to_string());
    }

    // Get method name
    let name = match &items[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("Protocol method name must be a symbol".to_string()),
    };

    // Get params
    let params = match &items[1] {
        Value::Vector(params) => {
            let mut param_names = Vec::new();
            for param in params.iter() {
                match param {
                    Value::Symbol(s) => param_names.push(s.clone()),
                    _ => return Err(format!("Method parameter must be a symbol, got {:?}", param)),
                }
            }
            param_names
        }
        _ => return Err("Method params must be a vector".to_string()),
    };

    // Parse body
    let mut body_exprs = Vec::new();
    for i in 2..items.len() {
        body_exprs.push(analyze(&items[i])?);
    }

    if body_exprs.is_empty() {
        body_exprs.push(Expr::Literal(Value::Nil));
    }

    // If we have field names (deftype context), wrap body in let that binds fields
    // This makes fields available as locals within method body
    // (-meta [coll] meta) becomes (-meta [coll] (let [meta (.-meta coll) ...] meta))
    let body = if !field_names.is_empty() && !params.is_empty() {
        let this_param = &params[0]; // First param is always 'this'/'self'

        // Build let bindings: [(field1 (.-field1 this)) (field2 (.-field2 this)) ...]
        let mut bindings: Vec<(String, Box<Expr>)> = Vec::new();
        for field_name in field_names {
            // Create field access expression: (.-field this)
            let field_access = Expr::FieldAccess {
                field: field_name.clone(),
                object: Box::new(Expr::Var {
                    namespace: None,
                    name: this_param.clone(),
                }),
            };
            bindings.push((field_name.clone(), Box::new(field_access)));
        }

        // Wrap body in let
        let inner_body = if body_exprs.len() == 1 {
            body_exprs.pop().unwrap()
        } else {
            Expr::Do { exprs: body_exprs }
        };

        vec![Expr::Let {
            bindings,
            body: vec![inner_body],
        }]
    } else {
        body_exprs
    };

    Ok(ProtocolMethodImpl { name, params, body })
}

fn analyze_call(items: &im::Vector<Value>) -> Result<Expr, String> {
    // Check for special call patterns based on the first symbol
    if let Some(Value::Symbol(sym)) = items.get(0) {
        // Check for constructor call: (TypeName. arg1 arg2 ...)
        if sym.ends_with('.') && sym.len() > 1 && !sym.starts_with('.') {
            let type_name = sym[..sym.len()-1].to_string();
            let mut args = Vec::new();
            for i in 1..items.len() {
                args.push(analyze(&items[i])?);
            }
            return Ok(Expr::TypeConstruct { type_name, args });
        }

        // Check for field access: (.-field obj)
        if sym.starts_with(".-") && sym.len() > 2 {
            if items.len() != 2 {
                return Err(format!("Field access {} requires exactly 1 argument", sym));
            }
            let field = sym[2..].to_string();
            let object = analyze(&items[1])?;
            return Ok(Expr::FieldAccess {
                field,
                object: Box::new(object),
            });
        }

        // Check for factory constructor call: (->TypeName arg1 arg2 ...)
        // This must come before protocol method check since both start with "-"
        if sym.starts_with("->") && sym.len() > 2 {
            let type_name = sym[2..].to_string();
            let mut args = Vec::new();
            for i in 1..items.len() {
                args.push(analyze(&items[i])?);
            }
            return Ok(Expr::TypeConstruct { type_name, args });
        }

        // Check for protocol method call: (-method-name obj args...)
        // Protocol methods start with "-" but not ".-" or "->"
        if sym.starts_with('-') && sym.len() > 1 && !sym.starts_with(".-") && !sym.starts_with("->") {
            if items.len() < 2 {
                return Err(format!("Protocol method {} requires at least 1 argument (the target)", sym));
            }
            let method_name = sym.clone();
            let mut args = Vec::new();
            for i in 1..items.len() {
                args.push(analyze(&items[i])?);
            }
            return Ok(Expr::ProtocolCall { method_name, args });
        }
    }

    // Regular function call
    let func = analyze(&items[0])?;
    let mut args = Vec::new();

    for i in 1..items.len() {
        args.push(analyze(&items[i])?);
    }

    Ok(Expr::Call {
        func: Box::new(func),
        args,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read;

    #[test]
    fn test_analyze_literals() {
        let val = read("42").unwrap();
        let expr = analyze(&val).unwrap();
        assert!(matches!(expr, Expr::Literal(Value::Int(42))));
    }

    #[test]
    fn test_analyze_var() {
        let val = read("x").unwrap();
        let expr = analyze(&val).unwrap();
        assert!(matches!(expr, Expr::Var { namespace: None, ref name } if name == "x"));
    }

    #[test]
    fn test_analyze_def() {
        let val = read("(def x 42)").unwrap();
        let expr = analyze(&val).unwrap();
        match expr {
            Expr::Def { name, value, .. } => {
                assert_eq!(name, "x");
                assert!(matches!(*value, Expr::Literal(Value::Int(42))));
            }
            _ => panic!("Expected Def"),
        }
    }

    #[test]
    fn test_analyze_if() {
        let val = read("(if true 1 2)").unwrap();
        let expr = analyze(&val).unwrap();
        match expr {
            Expr::If { test, then, else_ } => {
                assert!(matches!(*test, Expr::Literal(Value::Bool(true))));
                assert!(matches!(*then, Expr::Literal(Value::Int(1))));
                assert!(matches!(else_, Some(_)));
            }
            _ => panic!("Expected If"),
        }
    }

    #[test]
    fn test_analyze_call() {
        let val = read("(+ 1 2)").unwrap();
        let expr = analyze(&val).unwrap();
        match expr {
            Expr::Call { func, args } => {
                assert!(matches!(*func, Expr::Var { namespace: None, ref name } if name == "+"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call"),
        }
    }
}
