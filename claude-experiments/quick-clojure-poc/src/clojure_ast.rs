use crate::gc_runtime::{
    GCRuntime, TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TYPE_KEYWORD,
    TYPE_LIST, TYPE_NIL, TYPE_READER_LIST, TYPE_READER_MAP, TYPE_READER_SET, TYPE_READER_SYMBOL,
    TYPE_READER_VECTOR, TYPE_SET, TYPE_STRING, TYPE_SYMBOL, TYPE_VECTOR,
};
use crate::value::Value;

// Helper functions that work with both reader types and runtime types
// This is needed because macro expansion produces PersistentVector/Cons, not __ReaderVector/__ReaderList

fn is_vector(rt: &GCRuntime, value: usize) -> bool {
    rt.prim_is_vector(value)
}

fn is_list(rt: &GCRuntime, value: usize) -> bool {
    let type_id = rt.get_type_id_for_value(value);
    match type_id {
        TYPE_READER_LIST | TYPE_LIST => true,
        _ if type_id >= crate::gc_runtime::DEFTYPE_ID_OFFSET => {
            // Check for known list types (may be qualified or unqualified)
            if let Some(type_def) = rt.get_type_def(type_id - crate::gc_runtime::DEFTYPE_ID_OFFSET) {
                let name = type_def.name.as_str();
                matches!(name, "PList" | "Cons" | "EmptyList"
                    | "clojure.core/PList" | "clojure.core/Cons" | "clojure.core/EmptyList")
            } else {
                false
            }
        }
        _ => false,
    }
}

fn is_symbol(rt: &GCRuntime, value: usize) -> bool {
    rt.prim_is_symbol(value)
}

fn vector_count(rt: &mut GCRuntime, vec_ptr: usize) -> Result<usize, String> {
    rt.prim_count(vec_ptr)
}

fn vector_nth(rt: &mut GCRuntime, vec_ptr: usize, index: usize) -> Result<usize, String> {
    rt.prim_nth(vec_ptr, index)
}

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
        namespace: Option<String>, // None = unqualified, Some("user") = qualified
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

    /// Top-level do - forms are stored as tagged pointers and processed sequentially
    /// This allows macros defined in earlier forms to be used in later forms
    TopLevelDo {
        forms: Vec<usize>, // Tagged pointers to be analyzed/compiled/executed in sequence
    },

    /// (binding [var1 val1 var2 val2 ...] body)
    /// Establishes thread-local bindings for dynamic vars
    Binding {
        bindings: Vec<(String, Box<Expr>)>, // [(var-name, value-expr), ...]
        body: Vec<Expr>,                    // Multiple expressions in body (like do)
    },

    /// (let [x 10 y 20] (+ x y))
    /// Establishes lexical (stack-allocated) local bindings
    /// Bindings are sequential - each can see prior bindings
    Let {
        bindings: Vec<(String, Box<Expr>)>, // [(name, value-expr), ...]
        body: Vec<Expr>,                    // Body expressions (returns last)
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
        name: Option<String>,                // Optional self-recursion name
        arities: Vec<crate::value::FnArity>, // One or more arity overloads
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
        method_name: String, // e.g., "-first"
        args: Vec<Expr>,     // First arg is the dispatch target
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
    pub exception_type: String, // e.g., "Exception" (ignored for now - catch all)
    pub binding: String,        // Local binding for caught exception
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
    pub params: Vec<String>, // Includes 'this'
    pub body: Vec<Expr>,
}

/// Field definition in deftype
/// Supports ^:mutable metadata for mutable fields
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    pub name: String,
    pub mutable: bool,
}

// ========== Tagged Pointer Analyzer ==========
// Analyzes tagged pointers (from read_to_tagged) instead of Value enum

/// Get the type ID for a tagged pointer value
fn get_type_id(rt: &GCRuntime, tagged: usize) -> usize {
    // Delegate to GCRuntime which handles deftype type_data correctly
    rt.get_type_id_for_value(tagged)
}

/// Analyze a top-level tagged pointer value into an AST expression
/// This handles `do` specially by returning TopLevelDo with unevaluated forms
/// so the compiler can process them sequentially (allowing macros defined
/// in earlier forms to be used in later forms).
pub fn analyze_toplevel_tagged(rt: &mut GCRuntime, tagged: usize) -> Result<Expr, String> {
    let type_id = get_type_id(rt, tagged);

    // Check for top-level (do ...)
    if type_id == TYPE_READER_LIST {
        let count = rt.reader_list_count(tagged);
        if count > 0 {
            let first = rt.reader_list_first(tagged);
            let first_type = get_type_id(rt, first);
            if first_type == TYPE_READER_SYMBOL {
                let name = rt.reader_symbol_name(first);
                let ns = rt.reader_symbol_namespace(first);
                if ns.is_none() && name == "do" {
                    // Return TopLevelDo with the raw tagged pointers
                    let items = list_to_vec(rt, tagged);
                    let forms: Vec<usize> = items[1..].to_vec();
                    return Ok(Expr::TopLevelDo { forms });
                }
            }
        }
    }

    // Otherwise, use normal analysis
    analyze_tagged(rt, tagged)
}

/// Analyze a tagged pointer value into an AST expression
pub fn analyze_tagged(rt: &mut GCRuntime, tagged: usize) -> Result<Expr, String> {
    let type_id = get_type_id(rt, tagged);


    // ========== Primitive types (exact type matches) ==========
    match type_id {
        TYPE_NIL => return Ok(Expr::Literal(Value::Nil)),

        TYPE_BOOL => {
            let value = (tagged >> 3) != 0;
            return Ok(Expr::Literal(Value::Bool(value)));
        }

        TYPE_INT => {
            let value = ((tagged as isize) >> 3) as i64;
            return Ok(Expr::Literal(Value::Int(value)));
        }

        TYPE_FLOAT => {
            // Float is heap-allocated, get the value
            let untagged = tagged >> 3;
            let data_ptr = (untagged + 8) as *const u64;
            let bits = unsafe { *data_ptr };
            let value = f64::from_bits(bits);
            return Ok(Expr::Literal(Value::Float(value)));
        }

        TYPE_STRING => {
            // String is heap-allocated
            let s = rt.read_string(tagged);
            return Ok(Expr::Literal(Value::String(s)));
        }

        TYPE_KEYWORD => {
            let kw = rt.get_keyword_text(tagged).unwrap_or("").to_string();
            return Ok(Expr::Literal(Value::Keyword(kw)));
        }

        _ => {} // Fall through to dynamic dispatch
    }

    // ========== Dynamic dispatch for composite types ==========

    // Check for symbol (ReaderSymbol or user-defined Symbol type)
    if rt.prim_is_symbol(tagged) {
        let name = rt.prim_symbol_name(tagged)?;
        let ns = rt.prim_symbol_namespace(tagged)?;

        // Check for special "/" symbol (division operator)
        if name == "/" && ns.is_none() {
            return Ok(Expr::Var {
                namespace: None,
                name,
            });
        }

        return Ok(Expr::Var {
            namespace: ns,
            name,
        });
    }

    // Check for vector BEFORE seq (vectors are seqable but should be treated as literals, not calls)
    if rt.prim_is_vector(tagged) {
        // Vectors are literals - convert to Value::Vector
        let vec = tagged_to_value(rt, tagged)?;
        return Ok(Expr::Literal(vec));
    }

    // Check for map BEFORE seq (maps are seqable but should be treated as literals, not calls)
    if rt.prim_is_map(tagged) {
        // Maps are literals - convert to Value::Map
        let map = tagged_to_value(rt, tagged)?;
        return Ok(Expr::Literal(map));
    }

    // Check for set BEFORE seq (sets are seqable but should be treated as literals, not calls)
    if rt.prim_is_set(tagged) {
        // Sets are literals - convert to Value::Set
        let set = tagged_to_value(rt, tagged)?;
        return Ok(Expr::Literal(set));
    }

    // Check for seq (ReaderList, PersistentList, or any ISeq impl)
    if rt.prim_is_seq(tagged) {
        let count = rt.prim_count(tagged)?;
        if count == 0 {
            return Err("Cannot evaluate empty list".to_string());
        }

        // Get first element to check for special forms
        let first = rt.prim_first(tagged)?;

        // Check if first element is a symbol for special form dispatch
        if rt.prim_is_symbol(first) {
            let name = rt.prim_symbol_name(first)?;
            let ns = rt.prim_symbol_namespace(first)?;

            // Only dispatch on special forms if unqualified
            if ns.is_none() {
                match name.as_str() {
                    // ========== TRUE SPECIAL FORMS ==========
                    "def" => return analyze_def_tagged(rt, tagged),
                    "set!" => return analyze_set_tagged(rt, tagged),
                    "if" => return analyze_if_tagged(rt, tagged),
                    "do" => return analyze_do_tagged(rt, tagged),
                    "let" => return analyze_let_tagged(rt, tagged),
                    "loop" => return analyze_loop_tagged(rt, tagged),
                    "recur" => return analyze_recur_tagged(rt, tagged),
                    "fn" => return analyze_fn_tagged(rt, tagged),
                    "quote" => return analyze_quote_tagged(rt, tagged),
                    "var" => return analyze_var_ref_tagged(rt, tagged),
                    "throw" => return analyze_throw_tagged(rt, tagged),
                    "try" => return analyze_try_tagged(rt, tagged),

                    // ========== BUILTIN MACROS ==========
                    "defn" => return analyze_defn_tagged(rt, tagged),
                    "declare" => return analyze_declare_tagged(rt, tagged),
                    "if-not" => return analyze_if_not_tagged(rt, tagged),
                    "when" => return analyze_when_tagged(rt, tagged),
                    "when-not" => return analyze_when_not_tagged(rt, tagged),
                    "and" => return analyze_and_tagged(rt, tagged),
                    "or" => return analyze_or_tagged(rt, tagged),
                    "cond" => return analyze_cond_tagged(rt, tagged),
                    "dotimes" => return analyze_dotimes_tagged(rt, tagged),
                    "ns" => return analyze_ns_tagged(rt, tagged),
                    "use" => return analyze_use_tagged(rt, tagged),
                    "binding" => return analyze_binding_tagged(rt, tagged),
                    "deftype*" | "deftype" => return analyze_deftype_tagged(rt, tagged),
                    "defprotocol" => return analyze_defprotocol_tagged(rt, tagged),
                    "extend-type" => return analyze_extend_type_tagged(rt, tagged),
                    "debugger" => return analyze_debugger_tagged(rt, tagged),

                    _ => {}
                }
            }
        }

        // Regular function call
        return analyze_call_tagged(rt, tagged);
    }

    Err(format!("Cannot analyze type_id {}", type_id))
}

/// Convert a tagged pointer back to a Value (for literals and quote)
fn tagged_to_value(rt: &mut GCRuntime, tagged: usize) -> Result<Value, String> {
    let type_id = get_type_id(rt, tagged);

    match type_id {
        TYPE_NIL => Ok(Value::Nil),
        TYPE_BOOL => {
            let value = (tagged >> 3) != 0;
            Ok(Value::Bool(value))
        }
        TYPE_INT => {
            let value = ((tagged as isize) >> 3) as i64;
            Ok(Value::Int(value))
        }
        TYPE_FLOAT => {
            let untagged = tagged >> 3;
            let data_ptr = (untagged + 8) as *const u64;
            let bits = unsafe { *data_ptr };
            let value = f64::from_bits(bits);
            Ok(Value::Float(value))
        }
        TYPE_STRING => {
            let s = rt.read_string(tagged);
            Ok(Value::String(s))
        }
        TYPE_KEYWORD => {
            let kw = rt.get_keyword_text(tagged).unwrap_or("").to_string();
            Ok(Value::Keyword(kw))
        }
        TYPE_READER_SYMBOL | TYPE_SYMBOL => {
            // Use primitive dispatch to support both ReaderSymbol and runtime Symbol types
            let name = rt.prim_symbol_name(tagged)?;
            let ns = rt.prim_symbol_namespace(tagged)?;
            if let Some(ns) = ns {
                Ok(Value::Symbol(format!("{}/{}", ns, name)))
            } else {
                Ok(Value::Symbol(name))
            }
        }
        TYPE_READER_LIST | TYPE_LIST => {
            // Use primitive dispatch to support both ReaderList and PersistentList
            let count = rt.prim_count(tagged).unwrap_or(0);
            let mut items = im::Vector::new();
            let mut current = tagged;
            for _ in 0..count {
                let first = rt.prim_first(current)?;
                items.push_back(tagged_to_value(rt, first)?);
                current = rt.prim_rest(current)?;
            }
            Ok(Value::List(items))
        }
        TYPE_READER_VECTOR | TYPE_VECTOR => {
            // Use primitive dispatch to support both ReaderVector and PersistentVector
            let count = rt.prim_count(tagged)?;
            let mut items = im::Vector::new();
            for i in 0..count {
                let elem = rt.prim_nth(tagged, i)?;
                items.push_back(tagged_to_value(rt, elem)?);
            }
            Ok(Value::Vector(items))
        }
        TYPE_READER_MAP => {
            let count = rt.reader_map_count(tagged);
            let mut map = im::HashMap::new();
            // Get keys and values
            let keys_ptr = rt.reader_map_keys(tagged)?;
            let vals_ptr = rt.reader_map_vals(tagged)?;
            for i in 0..count {
                let key = rt.reader_vector_nth(keys_ptr, i)?;
                let val = rt.reader_vector_nth(vals_ptr, i)?;
                map.insert(tagged_to_value(rt, key)?, tagged_to_value(rt, val)?);
            }
            Ok(Value::Map(map))
        }
        TYPE_READER_SET | TYPE_SET => {
            let count = rt.reader_set_count(tagged);
            let mut set = im::HashSet::new();
            for i in 0..count {
                let elem = rt.reader_set_get(tagged, i)?;
                set.insert(tagged_to_value(rt, elem)?);
            }
            Ok(Value::Set(set))
        }
        _ if type_id >= crate::gc_runtime::DEFTYPE_ID_OFFSET => {
            // Deftype - check if it's seqable (e.g., Cons, PList, etc.)
            if rt.prim_is_seqable(tagged) {
                // Treat as a list
                let count = rt.prim_count(tagged).unwrap_or(0);
                let mut items = im::Vector::new();
                let mut current = tagged;
                for _ in 0..count {
                    let first = rt.prim_first(current)?;
                    items.push_back(tagged_to_value(rt, first)?);
                    current = rt.prim_rest(current)?;
                }
                Ok(Value::List(items))
            } else if rt.prim_is_symbol(tagged) {
                // It's a symbol type
                let name = rt.prim_symbol_name(tagged)?;
                let ns = rt.prim_symbol_namespace(tagged)?;
                if let Some(ns) = ns {
                    Ok(Value::Symbol(format!("{}/{}", ns, name)))
                } else {
                    Ok(Value::Symbol(name))
                }
            } else {
                Err(format!(
                    "Cannot convert deftype {} to Value",
                    crate::gc_runtime::GCRuntime::builtin_type_name(type_id)
                ))
            }
        }
        _ => Err(format!("Cannot convert type_id {} to Value", type_id)),
    }
}

/// Helper to collect list elements into a Vec of tagged pointers
fn list_to_vec(rt: &mut GCRuntime, list_ptr: usize) -> Vec<usize> {
    // Use primitive dispatch to support both ReaderList and runtime list types (Cons, PList)
    let count = rt.prim_count(list_ptr).unwrap_or(0);
    let mut result = Vec::with_capacity(count);
    let mut current = list_ptr;
    for _ in 0..count {
        result.push(rt.prim_first(current).unwrap_or(7));
        current = rt.prim_rest(current).unwrap_or(7);
    }
    result
}

/// Helper to get symbol name from tagged pointer, returns None if not a symbol
fn get_symbol_name(rt: &mut GCRuntime, tagged: usize) -> Option<String> {
    if rt.prim_is_symbol(tagged) {
        rt.prim_symbol_name(tagged).ok()
    } else {
        None
    }
}

/// Helper to get symbol namespace from tagged pointer
fn get_symbol_namespace(rt: &mut GCRuntime, tagged: usize) -> Option<String> {
    if rt.prim_is_symbol(tagged) {
        rt.prim_symbol_namespace(tagged).ok().flatten()
    } else {
        None
    }
}

/// Helper to get symbol metadata (as ReaderMap pointer, or nil)
fn get_symbol_metadata(rt: &GCRuntime, tagged: usize) -> usize {
    if get_type_id(rt, tagged) == TYPE_READER_SYMBOL {
        rt.reader_symbol_metadata(tagged)
    } else {
        7 // nil
    }
}

/// Convert a ReaderMap to im::HashMap<String, Value> for metadata
fn reader_map_to_metadata(rt: &mut GCRuntime, map_ptr: usize) -> Option<im::HashMap<String, Value>> {
    if map_ptr == 7 {
        // nil
        return None;
    }
    if get_type_id(rt, map_ptr) != TYPE_READER_MAP {
        return None;
    }

    let mut result = im::HashMap::new();

    // Get the entry count from the reader map
    let entry_count = rt.reader_map_count(map_ptr);

    for i in 0..entry_count {
        let (key_ptr, value_ptr) = rt.reader_map_entry(map_ptr, i);

        // Convert key to string - only support keywords and symbols as keys
        let key_type = get_type_id(rt, key_ptr);
        let key_str = if key_type == TYPE_KEYWORD {
            rt.get_keyword_text(key_ptr).ok().map(|s| s.to_string())
        } else if key_type == TYPE_READER_SYMBOL {
            Some(rt.reader_symbol_name(key_ptr))
        } else if key_type == TYPE_STRING {
            Some(rt.read_string(key_ptr))
        } else {
            None
        };

        // Skip entries with non-string keys
        let key_str = match key_str {
            Some(k) => k,
            None => continue,
        };

        // Convert value to Value - use the existing tagged_to_value function
        let value = tagged_to_value(rt, value_ptr).unwrap_or(Value::Nil);

        result.insert(key_str, value);
    }

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Look up a var by namespace/name for macro checking.
/// This is used during analysis to determine if a symbol refers to a macro.
/// Returns Some(var_ptr) if the var exists, None otherwise.
fn lookup_var_for_macro_check(rt: &GCRuntime, ns: Option<&str>, name: &str) -> Option<usize> {
    // If namespace is specified, look up in that namespace
    if let Some(ns_name) = ns {
        // Look up the namespace
        if let Some(&ns_ptr) = rt.get_namespace_pointers().get(ns_name) {
            return rt.namespace_lookup(ns_ptr, name);
        }
        return None;
    }

    // No namespace specified - look in current namespace first, then clojure.core
    // Get the current namespace (we use "user" as default during analysis)
    // First try to find in any namespace - check all namespaces for the var
    for &ns_ptr in rt.get_namespace_pointers().values() {
        if let Some(var_ptr) = rt.namespace_lookup(ns_ptr, name) {
            return Some(var_ptr);
        }
    }

    None
}

// ========== Special Form Analyzers (Tagged) ==========

fn analyze_def_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() != 3 {
        return Err(format!("def requires 2 arguments, got {}", items.len() - 1));
    }

    // Get symbol name
    let name = get_symbol_name(rt, items[1])
        .ok_or_else(|| "def requires a symbol as first argument".to_string())?;

    // Extract metadata from the symbol (if any)
    let meta_ptr = get_symbol_metadata(rt, items[1]);
    let metadata = reader_map_to_metadata(rt, meta_ptr);

    let value = analyze_tagged(rt, items[2])?;

    Ok(Expr::Def {
        name,
        value: Box::new(value),
        metadata,
    })
}

fn analyze_set_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() != 3 {
        return Err(format!("set! requires 2 arguments, got {}", items.len() - 1));
    }

    let target = analyze_tagged(rt, items[1])?;
    let value = analyze_tagged(rt, items[2])?;

    match target {
        Expr::FieldAccess { field, object } => Ok(Expr::FieldSet {
            field,
            object,
            value: Box::new(value),
        }),
        _ => Ok(Expr::Set {
            var: Box::new(target),
            value: Box::new(value),
        }),
    }
}

fn analyze_if_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 3 || items.len() > 4 {
        return Err(format!("if requires 2 or 3 arguments, got {}", items.len() - 1));
    }

    let test = analyze_tagged(rt, items[1])?;
    let then = analyze_tagged(rt, items[2])?;
    let else_ = if items.len() == 4 {
        Some(Box::new(analyze_tagged(rt, items[3])?))
    } else {
        None
    };

    Ok(Expr::If {
        test: Box::new(test),
        then: Box::new(then),
        else_,
    })
}

fn analyze_if_not_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 3 || items.len() > 4 {
        return Err(format!("if-not requires 2 or 3 arguments, got {}", items.len() - 1));
    }

    let test = analyze_tagged(rt, items[1])?;
    let not_test = Expr::Call {
        func: Box::new(Expr::Var {
            namespace: None,
            name: "not".to_string(),
        }),
        args: vec![test],
    };

    let then = analyze_tagged(rt, items[2])?;
    let else_ = if items.len() == 4 {
        Some(Box::new(analyze_tagged(rt, items[3])?))
    } else {
        None
    };

    Ok(Expr::If {
        test: Box::new(not_test),
        then: Box::new(then),
        else_,
    })
}

fn analyze_when_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("when requires at least a test".to_string());
    }

    let test = analyze_tagged(rt, items[1])?;

    let mut body_exprs = Vec::new();
    for item in items.iter().skip(2) {
        body_exprs.push(analyze_tagged(rt, *item)?);
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
        else_: None,
    })
}

fn analyze_when_not_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("when-not requires at least a test".to_string());
    }

    let test = analyze_tagged(rt, items[1])?;
    let not_test = Expr::Call {
        func: Box::new(Expr::Var {
            namespace: None,
            name: "not".to_string(),
        }),
        args: vec![test],
    };

    let mut body_exprs = Vec::new();
    for item in items.iter().skip(2) {
        body_exprs.push(analyze_tagged(rt, *item)?);
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

fn analyze_and_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    match items.len() {
        1 => Ok(Expr::Literal(Value::Bool(true))),
        2 => analyze_tagged(rt, items[1]),
        _ => {
            let test = analyze_tagged(rt, items[1])?;

            // Build rest of and recursively
            let rest_list = rt.reader_list_rest(list_ptr)?;
            let rest_and = analyze_and_tagged(rt, rest_list)?;

            // Use let binding to evaluate test once, then return it if falsey
            // This matches Clojure's behavior: (and nil 1) => nil, (and false 1) => false
            Ok(Expr::If {
                test: Box::new(test.clone()),
                then: Box::new(rest_and),
                else_: Some(Box::new(test)),
            })
        }
    }
}

fn analyze_or_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    match items.len() {
        1 => Ok(Expr::Literal(Value::Nil)),
        2 => analyze_tagged(rt, items[1]),
        _ => {
            let test = analyze_tagged(rt, items[1])?;

            // Build rest of or recursively
            let rest_list = rt.reader_list_rest(list_ptr)?;
            let rest_or = analyze_or_tagged(rt, rest_list)?;

            Ok(Expr::If {
                test: Box::new(test.clone()),
                then: Box::new(test),
                else_: Some(Box::new(rest_or)),
            })
        }
    }
}

fn analyze_cond_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);

    if items.len() == 1 {
        return Ok(Expr::Literal(Value::Nil));
    }

    if (items.len() - 1) % 2 != 0 {
        return Err("cond requires an even number of forms".to_string());
    }

    fn build_cond(rt: &mut GCRuntime, items: &[usize], idx: usize) -> Result<Expr, String> {
        if idx >= items.len() {
            return Ok(Expr::Literal(Value::Nil));
        }

        let test = items[idx];
        let expr = items[idx + 1];

        // Check for :else clause
        if get_type_id(rt, test) == TYPE_KEYWORD {
            let kw = rt.get_keyword_text(test).unwrap_or("");
            if kw == "else" {
                return analyze_tagged(rt, expr);
            }
        }

        let test_expr = analyze_tagged(rt, test)?;
        let then_expr = analyze_tagged(rt, expr)?;
        let else_expr = build_cond(rt, items, idx + 2)?;

        Ok(Expr::If {
            test: Box::new(test_expr),
            then: Box::new(then_expr),
            else_: Some(Box::new(else_expr)),
        })
    }

    build_cond(rt, &items, 1)
}

fn analyze_do_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("do requires at least 1 argument".to_string());
    }

    let mut exprs = Vec::new();
    for item in items.iter().skip(1) {
        exprs.push(analyze_tagged(rt, *item)?);
    }

    Ok(Expr::Do { exprs })
}

fn analyze_debugger_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() != 2 {
        return Err(format!("debugger requires 1 argument, got {}", items.len() - 1));
    }

    let expr = analyze_tagged(rt, items[1])?;
    Ok(Expr::Debugger {
        expr: Box::new(expr),
    })
}

fn analyze_quote_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() != 2 {
        return Err(format!("quote requires 1 argument, got {}", items.len() - 1));
    }

    let value = tagged_to_value(rt, items[1])?;
    Ok(Expr::Quote(value))
}

fn analyze_var_ref_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() != 2 {
        return Err(format!("var requires 1 argument, got {}", items.len() - 1));
    }

    let name = get_symbol_name(rt, items[1])
        .ok_or_else(|| "var requires a symbol".to_string())?;
    let namespace = get_symbol_namespace(rt, items[1]);

    Ok(Expr::VarRef { namespace, name })
}

fn analyze_let_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("let requires at least 1 argument: bindings vector".to_string());
    }

    // Parse bindings vector (works with both __ReaderVector and PersistentVector)
    let bindings_ptr = items[1];
    if !is_vector(rt, bindings_ptr) {
        return Err("let requires a vector of bindings as first argument".to_string());
    }

    let bindings_count = vector_count(rt, bindings_ptr)?;
    if bindings_count % 2 != 0 {
        return Err("let bindings vector must contain an even number of forms".to_string());
    }

    let mut bindings = Vec::new();
    for i in (0..bindings_count).step_by(2) {
        let name_ptr = vector_nth(rt, bindings_ptr, i)?;
        let value_ptr = vector_nth(rt, bindings_ptr, i + 1)?;

        let name = get_symbol_name(rt, name_ptr)
            .ok_or_else(|| "let binding names must be symbols".to_string())?;
        let value_expr = analyze_tagged(rt, value_ptr)?;
        bindings.push((name, Box::new(value_expr)));
    }

    // Parse body
    let mut body = Vec::new();
    for item in items.iter().skip(2) {
        body.push(analyze_tagged(rt, *item)?);
    }

    if body.is_empty() {
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(Expr::Let { bindings, body })
}

fn analyze_loop_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("loop requires at least 1 argument: bindings vector".to_string());
    }

    // Parse bindings vector (works with both __ReaderVector and PersistentVector)
    let bindings_ptr = items[1];
    if !is_vector(rt, bindings_ptr) {
        return Err("loop requires a vector of bindings as first argument".to_string());
    }

    let bindings_count = vector_count(rt, bindings_ptr)?;
    if bindings_count % 2 != 0 {
        return Err("loop bindings must contain an even number of forms".to_string());
    }

    let mut bindings = Vec::new();
    for i in (0..bindings_count).step_by(2) {
        let name_ptr = vector_nth(rt, bindings_ptr, i)?;
        let value_ptr = vector_nth(rt, bindings_ptr, i + 1)?;

        let name = get_symbol_name(rt, name_ptr)
            .ok_or_else(|| "loop binding names must be symbols".to_string())?;
        let value_expr = analyze_tagged(rt, value_ptr)?;
        bindings.push((name, Box::new(value_expr)));
    }

    let mut body = Vec::new();
    for item in items.iter().skip(2) {
        body.push(analyze_tagged(rt, *item)?);
    }

    if body.is_empty() {
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(Expr::Loop { bindings, body })
}

fn analyze_recur_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    let mut args = Vec::new();
    for item in items.iter().skip(1) {
        args.push(analyze_tagged(rt, *item)?);
    }
    Ok(Expr::Recur { args })
}

fn analyze_dotimes_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("dotimes requires at least a binding vector".to_string());
    }

    // Parse bindings vector (works with both __ReaderVector and PersistentVector)
    let bindings_ptr = items[1];
    if !is_vector(rt, bindings_ptr) {
        return Err("dotimes requires a vector binding [i n]".to_string());
    }

    let bindings_count = vector_count(rt, bindings_ptr)?;
    if bindings_count != 2 {
        return Err("dotimes binding must be [i n]".to_string());
    }

    let var_ptr = vector_nth(rt, bindings_ptr, 0)?;
    let count_ptr = vector_nth(rt, bindings_ptr, 1)?;

    let var_name = get_symbol_name(rt, var_ptr)
        .ok_or_else(|| "dotimes binding variable must be a symbol".to_string())?;
    let count_expr = analyze_tagged(rt, count_ptr)?;

    // Parse body
    let mut body_exprs = Vec::new();
    for item in items.iter().skip(2) {
        body_exprs.push(analyze_tagged(rt, *item)?);
    }

    // Build: (loop [i 0] (when (< i n) body... (recur (inc i))))
    let inc_call = Expr::Call {
        func: Box::new(Expr::Var {
            namespace: None,
            name: "inc".to_string(),
        }),
        args: vec![Expr::Var {
            namespace: None,
            name: var_name.clone(),
        }],
    };

    let recur_expr = Expr::Recur {
        args: vec![inc_call],
    };

    body_exprs.push(recur_expr);

    let when_body = if body_exprs.len() == 1 {
        body_exprs.pop().unwrap()
    } else {
        Expr::Do { exprs: body_exprs }
    };

    let lt_call = Expr::Call {
        func: Box::new(Expr::Var {
            namespace: None,
            name: "<".to_string(),
        }),
        args: vec![
            Expr::Var {
                namespace: None,
                name: var_name.clone(),
            },
            count_expr,
        ],
    };

    let when_expr = Expr::If {
        test: Box::new(lt_call),
        then: Box::new(when_body),
        else_: None,
    };

    let loop_bindings = vec![(var_name, Box::new(Expr::Literal(Value::Int(0))))];

    Ok(Expr::Loop {
        bindings: loop_bindings,
        body: vec![when_expr],
    })
}

fn validate_fn_arities(arities: &[crate::value::FnArity]) -> Result<(), String> {
    if arities.is_empty() {
        return Err("fn requires at least one arity".to_string());
    }

    let mut seen_fixed_arities = std::collections::HashSet::new();
    let mut variadic_count = 0;

    for arity in arities {
        let arity_num = arity.params.len();

        if arity.rest_param.is_some() {
            variadic_count += 1;
            if variadic_count > 1 {
                return Err("fn can have at most one variadic arity".to_string());
            }
            // Variadic arities don't conflict with fixed arities of the same param count.
            // E.g., ([x y] ...) and ([x y & rest] ...) can coexist:
            // - Fixed arity handles exactly 2 args
            // - Variadic handles 3+ args (or 2 args if no fixed arity exists)
        } else {
            // Only check for duplicates among fixed arities
            if !seen_fixed_arities.insert(arity_num) {
                return Err(format!("Duplicate arity {} in fn", arity_num));
            }
        }
    }

    Ok(())
}

fn analyze_fn_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("fn requires at least a parameter vector".to_string());
    }

    let mut idx = 1;

    // Check for optional name (works with both reader symbols and runtime symbols)
    let name = if is_symbol(rt, items[idx]) {
        // Could be name or could be params - check next item
        if idx + 1 < items.len() {
            if is_vector(rt, items[idx + 1]) {
                // This is a name, next is params
                let n = get_symbol_name(rt, items[idx]);
                idx += 1;
                n
            } else if is_list(rt, items[idx + 1]) {
                // Multi-arity form, no name
                None
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Parse arities (works with both reader and runtime types)
    let mut arities = Vec::new();

    let is_multi_arity_form = is_list(rt, items[idx]);

    if is_multi_arity_form {
        for arity_ptr in items.iter().skip(idx) {
            if !is_list(rt, *arity_ptr) {
                return Err("Multi-arity fn requires lists for each arity".to_string());
            }
            let arity = parse_fn_arity_tagged(rt, *arity_ptr)?;
            arities.push(arity);
        }
    } else {
        // Single-arity
        let params_ptr = items[idx];
        if !is_vector(rt, params_ptr) {
            return Err("fn requires parameter vector".to_string());
        }

        let body_start = idx + 1;
        let body_items: Vec<usize> = items[body_start..].to_vec();

        let arity = parse_fn_params_and_body_tagged(rt, params_ptr, &body_items)?;
        arities.push(arity);
    }

    validate_fn_arities(&arities)?;

    Ok(Expr::Fn { name, arities })
}

fn parse_fn_arity_tagged(rt: &mut GCRuntime, arity_ptr: usize) -> Result<crate::value::FnArity, String> {
    let arity_items = list_to_vec(rt, arity_ptr);
    if arity_items.is_empty() {
        return Err("fn arity form cannot be empty".to_string());
    }

    let params_ptr = arity_items[0];
    if !is_vector(rt, params_ptr) {
        return Err("fn arity form must start with parameter vector".to_string());
    }

    let body_items: Vec<usize> = arity_items[1..].to_vec();
    parse_fn_params_and_body_tagged(rt, params_ptr, &body_items)
}

fn parse_fn_params_and_body_tagged(
    rt: &mut GCRuntime,
    params_ptr: usize,
    body_items: &[usize],
) -> Result<crate::value::FnArity, String> {
    // Works with both __ReaderVector and PersistentVector
    let params_count = vector_count(rt, params_ptr)?;

    let mut params = Vec::new();
    let mut rest_param = None;
    let mut found_ampersand = false;

    for i in 0..params_count {
        let param_ptr = vector_nth(rt, params_ptr, i)?;
        let param_name = get_symbol_name(rt, param_ptr)
            .ok_or_else(|| "fn parameters must be symbols".to_string())?;

        if param_name == "&" {
            if found_ampersand {
                return Err("Only one & allowed in parameter list".to_string());
            }
            if i == params_count - 1 {
                return Err("& must be followed by rest parameter".to_string());
            }
            found_ampersand = true;
        } else if found_ampersand {
            if rest_param.is_some() {
                return Err("Only one rest parameter allowed after &".to_string());
            }
            rest_param = Some(param_name);
        } else {
            params.push(param_name);
        }
    }

    // Parse body (skip condition map for now - TODO)
    let mut body = Vec::new();
    for &item in body_items {
        body.push(analyze_tagged(rt, item)?);
    }

    if body.is_empty() {
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(crate::value::FnArity {
        params,
        rest_param,
        body,
        pre_conditions: Vec::new(),
        post_conditions: Vec::new(),
    })
}

fn analyze_defn_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 3 {
        return Err("defn requires at least a name and parameter vector".to_string());
    }

    let name = get_symbol_name(rt, items[1])
        .ok_or_else(|| "defn requires a symbol as first argument".to_string())?;

    // Build fn form - skip defn and name
    let mut start_idx = 2;

    // Skip optional docstring
    if get_type_id(rt, items[2]) == TYPE_STRING {
        start_idx = 3;
    }

    // Build fn list: (fn [params] body...)
    let mut fn_list_items = vec![items[0]]; // Reuse 'defn' position, we'll call analyze_fn_tagged
    fn_list_items.extend_from_slice(&items[start_idx..]);

    // Create a new list for fn
    let _fn_list = rt.reader_list_from_vec(&fn_list_items)?;

    // Analyze as fn (but we need to make it look like fn, not defn)
    // Actually, let's just inline the fn analysis here

    let fn_items = &items[start_idx..];
    let mut arities = Vec::new();

    let is_multi_arity = !fn_items.is_empty() && is_list(rt, fn_items[0]);

    if is_multi_arity {
        for &arity_ptr in fn_items {
            if !is_list(rt, arity_ptr) {
                return Err("Multi-arity defn requires lists for each arity".to_string());
            }
            let arity = parse_fn_arity_tagged(rt, arity_ptr)?;
            arities.push(arity);
        }
    } else if !fn_items.is_empty() {
        let params_ptr = fn_items[0];
        if !is_vector(rt, params_ptr) {
            return Err("defn requires parameter vector".to_string());
        }

        let body_items: Vec<usize> = fn_items[1..].to_vec();
        let arity = parse_fn_params_and_body_tagged(rt, params_ptr, &body_items)?;
        arities.push(arity);
    } else {
        return Err("defn requires at least a parameter vector".to_string());
    }

    validate_fn_arities(&arities)?;

    let fn_expr = Expr::Fn { name: None, arities };

    Ok(Expr::Def {
        name,
        value: Box::new(fn_expr),
        metadata: None,
    })
}

fn analyze_declare_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("declare requires at least one symbol".to_string());
    }

    let mut defs = Vec::new();
    for item in items.iter().skip(1) {
        let name = get_symbol_name(rt, *item)
            .ok_or_else(|| "declare requires symbols".to_string())?;
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

fn analyze_throw_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() != 2 {
        return Err(format!("throw requires 1 argument, got {}", items.len() - 1));
    }

    let exception = analyze_tagged(rt, items[1])?;
    Ok(Expr::Throw {
        exception: Box::new(exception),
    })
}

fn analyze_try_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("try requires at least 1 body expression".to_string());
    }

    let mut body = Vec::new();
    let mut catches = Vec::new();
    let mut finally = None;

    for item in items.iter().skip(1) {
        let item = *item;

        if get_type_id(rt, item) == TYPE_READER_LIST {
            let list_items = list_to_vec(rt, item);
            if !list_items.is_empty()
                && let Some(sym) = get_symbol_name(rt, list_items[0]) {
                    if sym == "catch" {
                        if list_items.len() < 3 {
                            return Err("catch requires at least exception type and binding".to_string());
                        }

                        let exception_type = get_symbol_name(rt, list_items[1])
                            .ok_or_else(|| "catch exception type must be a symbol".to_string())?;
                        let binding = get_symbol_name(rt, list_items[2])
                            .ok_or_else(|| "catch binding must be a symbol".to_string())?;

                        let mut catch_body = Vec::new();
                        for list_item in list_items.iter().skip(3) {
                            catch_body.push(analyze_tagged(rt, *list_item)?);
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
                        if finally.is_some() {
                            return Err("try can have at most one finally clause".to_string());
                        }

                        let mut finally_body = Vec::new();
                        for list_item in list_items.iter().skip(1) {
                            finally_body.push(analyze_tagged(rt, *list_item)?);
                        }

                        finally = Some(finally_body);
                        continue;
                    }
                }
        }

        // Not a catch or finally
        if !catches.is_empty() || finally.is_some() {
            return Err("try body expressions must come before catch/finally".to_string());
        }

        body.push(analyze_tagged(rt, item)?);
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

fn analyze_ns_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() != 2 {
        return Err(format!("ns requires 1 argument, got {}", items.len() - 1));
    }

    let ns_name = get_symbol_name(rt, items[1])
        .ok_or_else(|| "ns requires a symbol as namespace name".to_string())?;

    Ok(Expr::Ns { name: ns_name })
}

fn analyze_use_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() != 2 {
        return Err(format!("use requires 1 argument, got {}", items.len() - 1));
    }

    let ns_name = if get_type_id(rt, items[1]) == TYPE_READER_SYMBOL {
        get_symbol_name(rt, items[1]).unwrap()
    } else if get_type_id(rt, items[1]) == TYPE_READER_LIST {
        // Quoted symbol: (use 'clojure.core)
        let quoted = list_to_vec(rt, items[1]);
        if quoted.len() == 2 {
            if let Some(q) = get_symbol_name(rt, quoted[0]) {
                if q == "quote" {
                    if let Some(ns) = get_symbol_name(rt, quoted[1]) {
                        ns
                    } else {
                        return Err("use requires a symbol or quoted symbol".to_string());
                    }
                } else {
                    return Err("use requires a symbol or quoted symbol".to_string());
                }
            } else {
                return Err("use requires a symbol or quoted symbol".to_string());
            }
        } else {
            return Err("use requires a symbol or quoted symbol".to_string());
        }
    } else {
        return Err("use requires a symbol or quoted symbol".to_string());
    };

    Ok(Expr::Use { namespace: ns_name })
}

fn analyze_binding_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 3 {
        return Err("binding requires at least 2 arguments: bindings vector and body".to_string());
    }

    let bindings_ptr = items[1];
    if !is_vector(rt, bindings_ptr) {
        return Err("binding requires a vector of bindings as first argument".to_string());
    }

    let bindings_count = vector_count(rt, bindings_ptr)?;
    if bindings_count % 2 != 0 {
        return Err("binding vector must contain an even number of forms".to_string());
    }

    let mut bindings = Vec::new();
    for i in (0..bindings_count).step_by(2) {
        let var_ptr = vector_nth(rt, bindings_ptr, i)?;
        let val_ptr = vector_nth(rt, bindings_ptr, i + 1)?;

        let var_name = get_symbol_name(rt, var_ptr)
            .ok_or_else(|| "binding requires symbols as variable names".to_string())?;
        let value_expr = analyze_tagged(rt, val_ptr)?;
        bindings.push((var_name, Box::new(value_expr)));
    }

    let mut body = Vec::new();
    for item in items.iter().skip(2) {
        body.push(analyze_tagged(rt, *item)?);
    }

    Ok(Expr::Binding { bindings, body })
}

fn analyze_deftype_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 3 {
        return Err("deftype requires at least a name and fields vector".to_string());
    }

    let name = get_symbol_name(rt, items[1])
        .ok_or_else(|| "deftype requires a symbol as type name".to_string())?;

    let fields_ptr = items[2];
    if !is_vector(rt, fields_ptr) {
        return Err("deftype requires a vector of field names".to_string());
    }

    let fields_count = vector_count(rt, fields_ptr)?;
    let mut field_names = Vec::new();
    let mut fields = Vec::new();

    for i in 0..fields_count {
        let field_ptr = vector_nth(rt, fields_ptr, i)?;
        // TODO: Handle ^:mutable metadata
        let field_name = get_symbol_name(rt, field_ptr)
            .ok_or_else(|| "deftype field must be a symbol".to_string())?;
        field_names.push(field_name.clone());
        fields.push(FieldDef {
            name: field_name,
            mutable: false,
        });
    }

    if items.len() > 3 {
        // Has protocol implementations
        let implementations = parse_protocol_implementations_tagged(rt, &items, 3, &field_names)?;

        let deftype_expr = Expr::DefType {
            name: name.clone(),
            fields,
        };
        let extend_expr = Expr::ExtendType {
            type_name: name,
            implementations,
        };

        Ok(Expr::Do {
            exprs: vec![deftype_expr, extend_expr],
        })
    } else {
        Ok(Expr::DefType { name, fields })
    }
}

fn analyze_defprotocol_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("defprotocol requires at least a name".to_string());
    }

    let name = get_symbol_name(rt, items[1])
        .ok_or_else(|| "defprotocol name must be a symbol".to_string())?;

    let mut methods = Vec::new();
    for item in items.iter().skip(2) {
        if get_type_id(rt, *item) == TYPE_STRING {
            // Skip docstring
            continue;
        }
        let method_sig = parse_protocol_method_sig_tagged(rt, *item)?;
        methods.push(method_sig);
    }

    Ok(Expr::DefProtocol { name, methods })
}

fn parse_protocol_method_sig_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<ProtocolMethodSig, String> {
    if get_type_id(rt, list_ptr) != TYPE_READER_LIST {
        return Err("Protocol method signature must be a list".to_string());
    }

    let items = list_to_vec(rt, list_ptr);
    if items.is_empty() {
        return Err("Protocol method signature cannot be empty".to_string());
    }

    let name = get_symbol_name(rt, items[0])
        .ok_or_else(|| "Protocol method name must be a symbol".to_string())?;

    let mut arities = Vec::new();
    for item in items.iter().skip(1) {
        if get_type_id(rt, *item) == TYPE_STRING {
            // Skip docstring
            continue;
        }
        if !is_vector(rt, *item) {
            return Err("Protocol method arity must be a vector".to_string());
        }

        let params_count = vector_count(rt, *item)?;
        let mut param_names = Vec::new();
        for j in 0..params_count {
            let param_ptr = vector_nth(rt, *item, j)?;
            let param_name = get_symbol_name(rt, param_ptr)
                .ok_or_else(|| "Protocol method parameter must be a symbol".to_string())?;
            param_names.push(param_name);
        }
        arities.push(param_names);
    }

    if arities.is_empty() {
        return Err(format!("Protocol method {} requires at least one arity", name));
    }

    Ok(ProtocolMethodSig { name, arities })
}

fn analyze_extend_type_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("extend-type requires at least a type name".to_string());
    }

    let type_name = get_symbol_name(rt, items[1])
        .ok_or_else(|| "extend-type type name must be a symbol".to_string())?;

    let implementations = parse_protocol_implementations_tagged(rt, &items, 2, &[])?;

    Ok(Expr::ExtendType {
        type_name,
        implementations,
    })
}

fn parse_protocol_implementations_tagged(
    rt: &mut GCRuntime,
    items: &[usize],
    start_idx: usize,
    field_names: &[String],
) -> Result<Vec<ProtocolImpl>, String> {
    let mut implementations = Vec::new();
    let mut current_protocol: Option<String> = None;
    let mut current_methods: Vec<ProtocolMethodImpl> = Vec::new();

    for item in items.iter().skip(start_idx) {
        let item = *item;

        if get_type_id(rt, item) == TYPE_READER_SYMBOL {
            // Save previous protocol if any
            if let Some(protocol_name) = current_protocol.take() {
                implementations.push(ProtocolImpl {
                    protocol_name,
                    methods: std::mem::take(&mut current_methods),
                });
            }
            current_protocol = Some(get_symbol_name(rt, item).unwrap());
        } else if get_type_id(rt, item) == TYPE_READER_LIST {
            if current_protocol.is_none() {
                return Err("Method implementation found before protocol name".to_string());
            }
            let method_impl = parse_protocol_method_impl_tagged(rt, item, field_names)?;
            current_methods.push(method_impl);
        } else {
            return Err("Expected protocol name or method implementation".to_string());
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

fn parse_protocol_method_impl_tagged(
    rt: &mut GCRuntime,
    list_ptr: usize,
    field_names: &[String],
) -> Result<ProtocolMethodImpl, String> {
    let items = list_to_vec(rt, list_ptr);
    if items.len() < 2 {
        return Err("Protocol method implementation requires at least name and params".to_string());
    }

    let name = get_symbol_name(rt, items[0])
        .ok_or_else(|| "Protocol method name must be a symbol".to_string())?;

    let params_ptr = items[1];
    if !is_vector(rt, params_ptr) {
        return Err("Method params must be a vector".to_string());
    }

    let params_count = vector_count(rt, params_ptr)?;
    let mut params = Vec::new();
    for i in 0..params_count {
        let param_ptr = vector_nth(rt, params_ptr, i)?;
        let param_name = get_symbol_name(rt, param_ptr)
            .ok_or_else(|| "Method parameter must be a symbol".to_string())?;
        params.push(param_name);
    }

    // Parse body
    let mut body_exprs = Vec::new();
    for item in items.iter().skip(2) {
        body_exprs.push(analyze_tagged(rt, *item)?);
    }

    if body_exprs.is_empty() {
        body_exprs.push(Expr::Literal(Value::Nil));
    }

    // If we have field names, wrap body in let that binds fields
    let body = if !field_names.is_empty() && !params.is_empty() {
        let this_param = &params[0];

        let mut bindings: Vec<(String, Box<Expr>)> = Vec::new();
        for field_name in field_names {
            let field_access = Expr::FieldAccess {
                field: field_name.clone(),
                object: Box::new(Expr::Var {
                    namespace: None,
                    name: this_param.clone(),
                }),
            };
            bindings.push((field_name.clone(), Box::new(field_access)));
        }

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

fn analyze_call_tagged(rt: &mut GCRuntime, list_ptr: usize) -> Result<Expr, String> {
    let items = list_to_vec(rt, list_ptr);

    if get_type_id(rt, items[0]) == TYPE_READER_SYMBOL {
        let name = get_symbol_name(rt, items[0]).unwrap();
        let ns = get_symbol_namespace(rt, items[0]);

        // Reconstruct full symbol for patterns that need it
        let _full_sym = if let Some(ref ns) = ns {
            format!("{}/{}", ns, name)
        } else {
            name.clone()
        };

        // ===== MACRO EXPANSION =====
        // Check if this is a call to a macro var
        // Look up the var by namespace/name
        if let Some(var_ptr) = lookup_var_for_macro_check(rt, ns.as_deref(), &name)
            && rt.is_var_macro(var_ptr) {
                // This is a macro! Expand it.
                // Get the macro function
                let macro_fn = rt.var_get_value(var_ptr);

                // In Clojure, macros receive two implicit arguments before the explicit ones:
                // - &form: the original macro call form (the entire list being analyzed)
                // - &env: a map of local bindings in scope at the call site
                //
                // defmacro transforms (defmacro foo [x] body) into:
                // (defn foo [&form &env x] body)
                //
                // So we prepend &form and &env to the argument list.

                // &form is the original macro call form (list_ptr)
                let form = list_ptr;

                // &env is a map of local bindings. For now, pass an empty map.
                // TODO: Track lexical bindings during analysis and pass them here.
                let env = rt
                    .allocate_reader_map(&[])
                    .map_err(|e| format!("Failed to allocate &env map: {}", e))?;

                // Build args: &form, &env, then the explicit args (items[1..])
                let mut macro_args: Vec<usize> = vec![form, env];
                macro_args.extend(items[1..].iter().copied());

                // Invoke the macro
                let expanded = crate::trampoline::invoke_macro(rt, macro_fn, &macro_args)?;

                // Recursively analyze the result
                return analyze_tagged(rt, expanded);
            }

        // Constructor: (TypeName. arg1 arg2 ...) or (ns/TypeName. arg1 arg2 ...)
        if name.ends_with('.') && name.len() > 1 && !name.starts_with('.') {
            // For qualified constructors like myns/Widget., we need the full type name
            let type_name = if let Some(ref ns) = ns {
                format!("{}/{}", ns, &name[..name.len() - 1])
            } else {
                name[..name.len() - 1].to_string()
            };
            let mut args = Vec::new();
            for item in items.iter().skip(1) {
                args.push(analyze_tagged(rt, *item)?);
            }
            return Ok(Expr::TypeConstruct { type_name, args });
        }

        // Field access: (.-field obj)
        if name.starts_with(".-") && name.len() > 2 {
            if items.len() != 2 {
                return Err(format!("Field access {} requires exactly 1 argument", name));
            }
            let field = name[2..].to_string();
            let object = analyze_tagged(rt, items[1])?;
            return Ok(Expr::FieldAccess {
                field,
                object: Box::new(object),
            });
        }

        // Factory constructor: (->TypeName arg1 arg2 ...)
        if name.starts_with("->") && name.len() > 2 {
            // For qualified factory constructors like myns/->Widget
            let type_name = if let Some(ref ns) = ns {
                format!("{}/{}", ns, &name[2..])
            } else {
                name[2..].to_string()
            };
            let mut args = Vec::new();
            for item in items.iter().skip(1) {
                args.push(analyze_tagged(rt, *item)?);
            }
            return Ok(Expr::TypeConstruct { type_name, args });
        }
    }

    // Regular function call
    let func = analyze_tagged(rt, items[0])?;
    let mut args = Vec::new();

    for item in items.iter().skip(1) {
        args.push(analyze_tagged(rt, *item)?);
    }

    Ok(Expr::Call {
        func: Box::new(func),
        args,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read_to_tagged;

    #[test]
    fn test_analyze_tagged_int() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("42", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        assert!(matches!(expr, Expr::Literal(Value::Int(42))));
    }

    #[test]
    fn test_analyze_tagged_bool() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("true", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        assert!(matches!(expr, Expr::Literal(Value::Bool(true))));
    }

    #[test]
    fn test_analyze_tagged_nil() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("nil", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        assert!(matches!(expr, Expr::Literal(Value::Nil)));
    }

    #[test]
    fn test_analyze_tagged_symbol() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("foo", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        match expr {
            Expr::Var { namespace, name } => {
                assert!(namespace.is_none());
                assert_eq!(name, "foo");
            }
            _ => panic!("Expected Var"),
        }
    }

    #[test]
    fn test_analyze_tagged_qualified_symbol() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("clojure.core/first", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        match expr {
            Expr::Var { namespace, name } => {
                assert_eq!(namespace, Some("clojure.core".to_string()));
                assert_eq!(name, "first");
            }
            _ => panic!("Expected Var"),
        }
    }

    #[test]
    fn test_analyze_tagged_def() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("(def x 42)", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        match expr {
            Expr::Def { name, value, .. } => {
                assert_eq!(name, "x");
                assert!(matches!(*value, Expr::Literal(Value::Int(42))));
            }
            _ => panic!("Expected Def"),
        }
    }

    #[test]
    fn test_analyze_tagged_if() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("(if true 1 2)", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        match expr {
            Expr::If { test, then, else_ } => {
                assert!(matches!(*test, Expr::Literal(Value::Bool(true))));
                assert!(matches!(*then, Expr::Literal(Value::Int(1))));
                assert!(matches!(else_, Some(ref e) if matches!(**e, Expr::Literal(Value::Int(2)))));
            }
            _ => panic!("Expected If"),
        }
    }

    #[test]
    fn test_analyze_tagged_let() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("(let [x 10] x)", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        match expr {
            Expr::Let { bindings, body } => {
                assert_eq!(bindings.len(), 1);
                assert_eq!(bindings[0].0, "x");
                assert_eq!(body.len(), 1);
            }
            _ => panic!("Expected Let"),
        }
    }

    #[test]
    fn test_analyze_tagged_fn() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("(fn [x] x)", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        match expr {
            Expr::Fn { name, arities } => {
                assert!(name.is_none());
                assert_eq!(arities.len(), 1);
                assert_eq!(arities[0].params, vec!["x".to_string()]);
            }
            _ => panic!("Expected Fn"),
        }
    }

    #[test]
    fn test_analyze_tagged_call() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("(+ 1 2)", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        match expr {
            Expr::Call { func, args } => {
                assert!(matches!(*func, Expr::Var { namespace: None, ref name } if name == "+"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_analyze_tagged_vector_literal() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("[1 2 3]", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        match expr {
            Expr::Literal(Value::Vector(v)) => {
                assert_eq!(v.len(), 3);
            }
            other => panic!("Expected Vector literal, got {:?}", other),
        }
    }

    #[test]
    fn test_analyze_tagged_defn() {
        let mut rt = GCRuntime::new();
        let tagged = read_to_tagged("(defn add [a b] (+ a b))", &mut rt).unwrap();
        let expr = analyze_tagged(&mut rt, tagged).unwrap();
        match expr {
            Expr::Def { name, value, .. } => {
                assert_eq!(name, "add");
                match *value {
                    Expr::Fn { ref arities, .. } => {
                        assert_eq!(arities.len(), 1);
                        assert_eq!(arities[0].params, vec!["a".to_string(), "b".to_string()]);
                    }
                    _ => panic!("Expected Fn inside Def"),
                }
            }
            _ => panic!("Expected Def"),
        }
    }
}
