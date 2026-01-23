use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

/// Shared buffer storage for ArrayBuffer/TypedArray/DataView
/// Views share the same underlying buffer, so modifications through one view
/// are visible through other views of the same buffer.
#[derive(Debug, Clone)]
pub struct SharedBuffer {
    data: Rc<RefCell<Vec<u8>>>,
}

impl SharedBuffer {
    pub fn new(size: usize) -> Self {
        SharedBuffer {
            data: Rc::new(RefCell::new(vec![0; size])),
        }
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        SharedBuffer {
            data: Rc::new(RefCell::new(bytes)),
        }
    }

    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }

    pub fn get(&self, index: usize) -> Option<u8> {
        self.data.borrow().get(index).copied()
    }

    pub fn set(&self, index: usize, value: u8) {
        let mut data = self.data.borrow_mut();
        if index < data.len() {
            data[index] = value;
        }
    }

    pub fn get_bytes(&self, offset: usize, len: usize) -> Vec<u8> {
        let data = self.data.borrow();
        if offset + len <= data.len() {
            data[offset..offset + len].to_vec()
        } else {
            Vec::new()
        }
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.data.borrow().clone()
    }
}

impl PartialEq for SharedBuffer {
    fn eq(&self, other: &Self) -> bool {
        // Two buffers are equal if they point to the same data (same identity)
        Rc::ptr_eq(&self.data, &other.data)
    }
}

/// Shared array storage for reference semantics
/// When you do `sub = arr[0]` where arr[0] is an array, both `sub` and `arr[0]`
/// should reference the same underlying array, so mutations through one are
/// visible through the other.
#[derive(Debug, Clone)]
pub struct SharedArray {
    data: Rc<RefCell<Vec<Value>>>,
}

/// Helper to get a short, non-recursive description of a value for tracing
fn value_short_desc(v: &Value) -> String {
    match v {
        Value::Number(n) => format!("Number({})", n),
        Value::String(s) => format!("String({:?})", s.chars().take(30).collect::<String>()),
        Value::Bool(b) => format!("Bool({})", b),
        Value::Array(arr) => format!("Array(len={})", arr.len()),
        Value::Object(obj) => format!("Object(len={})", obj.len()),
        Value::Undefined => "Undefined".to_string(),
        Value::Null => "Null".to_string(),
        Value::Closure { name, params, .. } => {
            format!("Closure({}, params={:?})",
                    name.as_deref().unwrap_or("anonymous"),
                    params)
        }
        Value::ArrayBuffer { buffer } => format!("ArrayBuffer(len={})", buffer.len()),
        Value::TypedArray { kind, length, .. } => {
            format!("TypedArray({}, len={})", kind.name(), length)
        }
        Value::DataView { byte_length, .. } => format!("DataView(len={})", byte_length),
        Value::TextDecoder { encoding } => format!("TextDecoder({})", encoding),
        Value::Dynamic(s) => format!("Dynamic({:?})", s.chars().take(50).collect::<String>()),
    }
}

impl SharedArray {
    pub fn new(values: Vec<Value>) -> Self {
        SharedArray {
            data: Rc::new(RefCell::new(values)),
        }
    }

    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.borrow().is_empty()
    }

    pub fn get(&self, index: usize) -> Option<Value> {
        self.data.borrow().get(index).cloned()
    }

    /// Get with tracing - logs the operation to stderr
    pub fn get_traced(&self, index: usize, array_name: &str) -> Option<Value> {
        let result = self.data.borrow().get(index).cloned();
        eprintln!("[ARRAY_GET] {}[{}] = {}", array_name, index,
                  result.as_ref().map(|v| value_short_desc(v)).unwrap_or_else(|| "None".to_string()));
        result
    }

    pub fn set(&self, index: usize, value: Value) {
        let mut data = self.data.borrow_mut();
        if index < data.len() {
            data[index] = value;
        } else {
            // Extend the array to accommodate the new index (JS behavior)
            data.resize(index + 1, Value::Undefined);
            data[index] = value;
        }
    }

    /// Set with tracing - logs the operation to stderr
    pub fn set_traced(&self, index: usize, value: Value, array_name: &str) {
        eprintln!("[ARRAY_SET] {}[{}] = {}", array_name, index, value_short_desc(&value));
        let mut data = self.data.borrow_mut();
        if index < data.len() {
            data[index] = value;
        } else {
            // Extend the array to accommodate the new index (JS behavior)
            data.resize(index + 1, Value::Undefined);
            data[index] = value;
        }
    }

    pub fn push(&self, value: Value) {
        self.data.borrow_mut().push(value);
    }

    /// Push with tracing - logs the operation to stderr
    pub fn push_traced(&self, value: Value, array_name: &str) {
        let len = self.data.borrow().len();
        eprintln!("[ARRAY_PUSH] {}.push({}) at index {}", array_name, value_short_desc(&value), len);
        self.data.borrow_mut().push(value);
    }

    pub fn pop(&self) -> Option<Value> {
        self.data.borrow_mut().pop()
    }

    /// Pop with tracing - logs the operation to stderr
    pub fn pop_traced(&self, array_name: &str) -> Option<Value> {
        let len = self.data.borrow().len();
        let result = self.data.borrow_mut().pop();
        eprintln!("[ARRAY_POP] {}.pop() from index {} = {}", array_name, len.saturating_sub(1),
                  result.as_ref().map(|v| value_short_desc(v)).unwrap_or_else(|| "None".to_string()));
        result
    }

    pub fn to_vec(&self) -> Vec<Value> {
        self.data.borrow().clone()
    }

    /// Check if all elements are static
    pub fn all_static(&self) -> bool {
        self.data.borrow().iter().all(|v| v.is_static())
    }

    /// Iterate over elements (by cloning - use to_vec for owned iteration)
    pub fn iter(&self) -> impl Iterator<Item = Value> {
        self.data.borrow().clone().into_iter()
    }
}

impl PartialEq for SharedArray {
    fn eq(&self, other: &Self) -> bool {
        // Compare by contents for structural equality
        // (Identity comparison can be done with Rc::ptr_eq if needed)
        *self.data.borrow() == *other.data.borrow()
    }
}

/// Shared object storage for reference semantics
/// When you do `sub = obj.field` where obj.field is an object, both `sub` and `obj.field`
/// should reference the same underlying object, so mutations through one are
/// visible through the other.
#[derive(Debug, Clone)]
pub struct SharedObject {
    data: Rc<RefCell<HashMap<String, Value>>>,
}

impl SharedObject {
    pub fn new(properties: HashMap<String, Value>) -> Self {
        SharedObject {
            data: Rc::new(RefCell::new(properties)),
        }
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        self.data.borrow().get(key).cloned()
    }

    pub fn set(&self, key: String, value: Value) {
        self.data.borrow_mut().insert(key, value);
    }

    pub fn keys(&self) -> Vec<String> {
        self.data.borrow().keys().cloned().collect()
    }

    pub fn to_map(&self) -> HashMap<String, Value> {
        self.data.borrow().clone()
    }

    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.borrow().is_empty()
    }

    /// Check if all values are static
    pub fn all_static(&self) -> bool {
        self.data.borrow().values().all(|v| v.is_static())
    }

    /// Iterate over key-value pairs (by cloning)
    pub fn iter(&self) -> impl Iterator<Item = (String, Value)> {
        self.data.borrow().clone().into_iter()
    }
}

impl PartialEq for SharedObject {
    fn eq(&self, other: &Self) -> bool {
        // Compare by contents for structural equality
        // (Identity comparison can be done with Rc::ptr_eq if needed)
        *self.data.borrow() == *other.data.borrow()
    }
}

/// Kind of typed array
#[derive(Debug, Clone, PartialEq)]
pub enum TypedArrayKind {
    Int8Array,
    Uint8Array,
    Uint8ClampedArray,
    Int16Array,
    Uint16Array,
    Int32Array,
    Uint32Array,
    Float32Array,
    Float64Array,
    BigInt64Array,
    BigUint64Array,
}

impl TypedArrayKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Int8Array" => Some(TypedArrayKind::Int8Array),
            "Uint8Array" => Some(TypedArrayKind::Uint8Array),
            "Uint8ClampedArray" => Some(TypedArrayKind::Uint8ClampedArray),
            "Int16Array" => Some(TypedArrayKind::Int16Array),
            "Uint16Array" => Some(TypedArrayKind::Uint16Array),
            "Int32Array" => Some(TypedArrayKind::Int32Array),
            "Uint32Array" => Some(TypedArrayKind::Uint32Array),
            "Float32Array" => Some(TypedArrayKind::Float32Array),
            "Float64Array" => Some(TypedArrayKind::Float64Array),
            "BigInt64Array" => Some(TypedArrayKind::BigInt64Array),
            "BigUint64Array" => Some(TypedArrayKind::BigUint64Array),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TypedArrayKind::Int8Array => "Int8Array",
            TypedArrayKind::Uint8Array => "Uint8Array",
            TypedArrayKind::Uint8ClampedArray => "Uint8ClampedArray",
            TypedArrayKind::Int16Array => "Int16Array",
            TypedArrayKind::Uint16Array => "Uint16Array",
            TypedArrayKind::Int32Array => "Int32Array",
            TypedArrayKind::Uint32Array => "Uint32Array",
            TypedArrayKind::Float32Array => "Float32Array",
            TypedArrayKind::Float64Array => "Float64Array",
            TypedArrayKind::BigInt64Array => "BigInt64Array",
            TypedArrayKind::BigUint64Array => "BigUint64Array",
        }
    }

    /// Returns the size in bytes of each element
    pub fn element_size(&self) -> usize {
        match self {
            TypedArrayKind::Int8Array => 1,
            TypedArrayKind::Uint8Array => 1,
            TypedArrayKind::Uint8ClampedArray => 1,
            TypedArrayKind::Int16Array => 2,
            TypedArrayKind::Uint16Array => 2,
            TypedArrayKind::Int32Array => 4,
            TypedArrayKind::Uint32Array => 4,
            TypedArrayKind::Float32Array => 4,
            TypedArrayKind::Float64Array => 8,
            TypedArrayKind::BigInt64Array => 8,
            TypedArrayKind::BigUint64Array => 8,
        }
    }
}

/// A value in our partial evaluator.
/// Values are either fully known (static) or unknown (dynamic).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A statically known number
    Number(f64),
    /// A statically known string
    String(String),
    /// A statically known boolean
    Bool(bool),
    /// A statically known array (elements may be static or dynamic)
    /// Uses SharedArray for reference semantics - aliased arrays share the same data
    Array(SharedArray),
    /// A statically known object (properties may be static or dynamic)
    /// Uses SharedObject for reference semantics - aliased objects share the same data
    Object(SharedObject),
    /// Undefined
    Undefined,
    /// Null
    Null,
    /// A closure - captures its defining environment
    Closure {
        /// Parameter names
        params: Vec<String>,
        /// The function body (we'll store AST reference later, for now just an id)
        body_id: usize,
        /// The environment at definition time
        env: Env,
        /// The original source code of the function (for residual emission)
        source: String,
        /// The function name (if from a named function declaration)
        name: Option<String>,
    },
    /// An ArrayBuffer - raw binary data buffer
    ArrayBuffer {
        buffer: SharedBuffer,
    },
    /// A typed array (Uint8Array, Int32Array, etc.)
    /// Can be either standalone (owns its own data) or a view over an ArrayBuffer
    TypedArray {
        kind: TypedArrayKind,
        /// The underlying buffer (shared with ArrayBuffer if this is a view)
        buffer: SharedBuffer,
        /// Byte offset into the buffer
        byte_offset: usize,
        /// Length in elements (not bytes)
        length: usize,
    },
    /// A DataView - provides a low-level interface for reading/writing to an ArrayBuffer
    DataView {
        buffer: SharedBuffer,
        byte_offset: usize,
        byte_length: usize,
    },
    /// A TextDecoder instance
    TextDecoder {
        encoding: String,
    },
    /// A dynamic value - we don't know what it is at partial eval time
    /// The String is the residual expression that produces this value
    Dynamic(String),
}

impl Value {
    pub fn is_static(&self) -> bool {
        !matches!(self, Value::Dynamic(_))
    }

    pub fn is_dynamic(&self) -> bool {
        matches!(self, Value::Dynamic(_))
    }

    /// Get the residual expression for a dynamic value
    pub fn residual(&self) -> Option<&str> {
        match self {
            Value::Dynamic(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_truthy(&self) -> Option<bool> {
        match self {
            Value::Number(n) => Some(*n != 0.0 && !n.is_nan()),
            Value::String(s) => Some(!s.is_empty()),
            Value::Bool(b) => Some(*b),
            Value::Array(_) => Some(true),
            Value::Object(_) => Some(true),
            Value::Undefined => Some(false),
            Value::Null => Some(false),
            Value::Closure { .. } => Some(true),
            Value::ArrayBuffer { .. } => Some(true),
            Value::TypedArray { .. } => Some(true),
            Value::DataView { .. } => Some(true),
            Value::TextDecoder { .. } => Some(true),
            Value::Dynamic(_) => None, // Can't know at partial eval time
        }
    }

    /// Coerce a value to i32 for bitwise operations.
    /// In JavaScript, bitwise operators convert their operands to 32-bit integers.
    /// Returns None for dynamic values or values that can't be coerced.
    pub fn to_i32_for_bitwise(&self) -> Option<i32> {
        match self {
            Value::Number(n) => {
                // JavaScript ToInt32: truncate to 32-bit signed integer
                // Handle NaN and infinities -> 0
                if n.is_nan() || n.is_infinite() {
                    Some(0)
                } else {
                    Some(*n as i32)
                }
            }
            Value::Undefined => Some(0), // undefined -> NaN -> 0
            Value::Null => Some(0),      // null -> 0
            Value::Bool(true) => Some(1),
            Value::Bool(false) => Some(0),
            Value::String(s) => {
                // Empty string -> 0, numeric strings -> their value, else NaN -> 0
                if s.is_empty() {
                    Some(0)
                } else if let Ok(n) = s.trim().parse::<f64>() {
                    if n.is_nan() || n.is_infinite() {
                        Some(0)
                    } else {
                        Some(n as i32)
                    }
                } else {
                    Some(0) // Non-numeric string -> NaN -> 0
                }
            }
            // Arrays, objects, closures etc. would need toString/valueOf which is complex
            // For now, return None to fall back to dynamic
            Value::Dynamic(_) => None,
            _ => None,
        }
    }

    /// Returns true if this value is DEFINITELY not undefined.
    /// Used for optimizing comparisons like `x === undefined`.
    pub fn is_definitely_not_undefined(&self) -> bool {
        match self {
            Value::Number(_) => true,
            Value::String(_) => true,
            Value::Bool(_) => true,
            Value::Array(_) => true,
            Value::Object(_) => true,
            Value::Closure { .. } => true,
            Value::ArrayBuffer { .. } => true,
            Value::TypedArray { .. } => true,
            Value::DataView { .. } => true,
            Value::TextDecoder { .. } => true,
            Value::Null => true, // null !== undefined
            Value::Undefined => false,
            Value::Dynamic(s) => {
                // A Dynamic that looks like a function/object literal is definitely not undefined
                let trimmed = s.trim();
                trimmed.starts_with("(function")
                    || trimmed.starts_with("function")
                    || trimmed.starts_with("(() =>")
                    || trimmed.starts_with("(()=>")
                    || trimmed.starts_with("{")
                    || trimmed.starts_with("[")
            }
        }
    }

    /// Returns true if this value is DEFINITELY not null.
    /// Used for optimizing comparisons like `x === null`.
    pub fn is_definitely_not_null(&self) -> bool {
        match self {
            Value::Number(_) => true,
            Value::String(_) => true,
            Value::Bool(_) => true,
            Value::Array(_) => true,
            Value::Object(_) => true,
            Value::Closure { .. } => true,
            Value::ArrayBuffer { .. } => true,
            Value::TypedArray { .. } => true,
            Value::DataView { .. } => true,
            Value::TextDecoder { .. } => true,
            Value::Undefined => true, // undefined !== null
            Value::Null => false,
            Value::Dynamic(s) => {
                // A Dynamic that looks like a function/object literal is definitely not null
                let trimmed = s.trim();
                trimmed.starts_with("(function")
                    || trimmed.starts_with("function")
                    || trimmed.starts_with("(() =>")
                    || trimmed.starts_with("(()=>")
                    || trimmed.starts_with("{")
                    || trimmed.starts_with("[")
            }
        }
    }
}

/// A shared binding cell - allows multiple closures to share the same variable
pub type Binding = Rc<RefCell<Value>>;

/// Shared bindings map - allows captured environments to see new bindings added to a scope
pub type SharedBindings = Rc<RefCell<HashMap<String, Binding>>>;

/// A single scope frame - holds bindings for one lexical scope
/// The bindings HashMap is wrapped in Rc<RefCell<>> so that captured environments
/// see new bindings added after capture time (important for hoisted functions)
#[derive(Debug, Clone)]
struct Scope {
    bindings: SharedBindings,
}

impl Scope {
    fn new() -> Self {
        Scope {
            bindings: Rc::new(RefCell::new(HashMap::new())),
        }
    }
}

/// The environment - a chain of scopes representing lexical scope.
/// Uses Rc<RefCell<>> to allow closures to capture and share environments.
#[derive(Debug, Clone)]
pub struct Env {
    /// Stack of scopes, innermost last
    scopes: Rc<RefCell<Vec<Scope>>>,
}

impl PartialEq for Env {
    fn eq(&self, other: &Self) -> bool {
        // Two envs are equal if they point to the same scope chain
        Rc::ptr_eq(&self.scopes, &other.scopes)
    }
}

impl Env {
    /// Create a new empty environment with a global scope
    pub fn new() -> Self {
        Env {
            scopes: Rc::new(RefCell::new(vec![Scope::new()])),
        }
    }

    /// Create a child environment that shares the parent's scopes
    /// but can have new scopes pushed onto it
    pub fn child(&self) -> Self {
        // Clone the Rc to share the scope chain
        Env {
            scopes: Rc::clone(&self.scopes),
        }
    }

    /// Create a snapshot of the current environment for closure capture.
    /// The scope vector is cloned, but each scope's bindings HashMap (SharedBindings)
    /// is Rc::clone'd. This means:
    /// 1. Closures share variable bindings with their defining scope
    /// 2. New bindings added to a scope after capture are visible to captured envs
    /// 3. New scopes pushed after capture are NOT visible
    pub fn capture(&self) -> Self {
        let scopes = self.scopes.borrow();
        let captured_scopes: Vec<Scope> = scopes.iter().map(|scope| {
            Scope {
                // Rc::clone shares the entire bindings HashMap
                // New bindings added to this scope will be visible to captured env
                bindings: Rc::clone(&scope.bindings),
            }
        }).collect();
        Env {
            scopes: Rc::new(RefCell::new(captured_scopes)),
        }
    }

    /// Push a new scope (entering a block, function, etc.)
    pub fn push_scope(&self) {
        self.scopes.borrow_mut().push(Scope::new());
    }

    /// Pop a scope (leaving a block, function, etc.)
    pub fn pop_scope(&self) {
        let mut scopes = self.scopes.borrow_mut();
        if scopes.len() > 1 {
            scopes.pop();
        }
    }

    /// Define a new variable in the current (innermost) scope
    pub fn define(&self, name: &str, value: Value) {
        let scopes = self.scopes.borrow();
        if let Some(scope) = scopes.last() {
            scope.bindings.borrow_mut().insert(name.to_string(), Rc::new(RefCell::new(value)));
        }
    }

    /// Look up a variable, searching from innermost to outermost scope
    pub fn get(&self, name: &str) -> Option<Value> {
        let scopes = self.scopes.borrow();
        for scope in scopes.iter().rev() {
            let bindings = scope.bindings.borrow();
            if let Some(binding) = bindings.get(name) {
                return Some(binding.borrow().clone());
            }
        }
        None
    }

    /// Update an existing variable (finds it in scope chain and updates)
    /// Returns false if variable doesn't exist
    pub fn set(&self, name: &str, value: Value) -> bool {
        let scopes = self.scopes.borrow();
        for scope in scopes.iter().rev() {
            let bindings = scope.bindings.borrow();
            if let Some(binding) = bindings.get(name) {
                *binding.borrow_mut() = value;
                return true;
            }
        }
        false
    }

    /// Get the binding cell itself (for when you need to check identity or share)
    pub fn get_binding(&self, name: &str) -> Option<Binding> {
        let scopes = self.scopes.borrow();
        for scope in scopes.iter().rev() {
            let bindings = scope.bindings.borrow();
            if let Some(binding) = bindings.get(name) {
                return Some(Rc::clone(binding));
            }
        }
        None
    }

    /// Check if a variable exists in any scope
    pub fn exists(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Check if a variable exists in the current (innermost) scope only
    /// Does not search outer scopes
    pub fn exists_in_current_scope(&self, name: &str) -> bool {
        let scopes = self.scopes.borrow();
        if let Some(scope) = scopes.last() {
            scope.bindings.borrow().contains_key(name)
        } else {
            false
        }
    }

    /// Get the current scope depth (useful for debugging)
    pub fn depth(&self) -> usize {
        self.scopes.borrow().len()
    }

    /// Get all bindings in the current (innermost) scope
    /// Returns a vector of (name, value) pairs
    pub fn current_scope_bindings(&self) -> Vec<(String, Value)> {
        let scopes = self.scopes.borrow();
        if let Some(scope) = scopes.last() {
            let bindings = scope.bindings.borrow();
            bindings.iter()
                .map(|(k, v)| (k.clone(), v.borrow().clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all bindings across all scopes for debugging
    /// Returns a vector of (scope_index, name, value) tuples
    pub fn all_bindings(&self) -> Vec<(usize, String, Value)> {
        let scopes = self.scopes.borrow();
        let mut result = Vec::new();
        for (scope_idx, scope) in scopes.iter().enumerate() {
            let bindings = scope.bindings.borrow();
            for (name, binding) in bindings.iter() {
                result.push((scope_idx, name.clone(), binding.borrow().clone()));
            }
        }
        result
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_define_and_get() {
        let env = Env::new();
        env.define("x", Value::Number(42.0));
        assert_eq!(env.get("x"), Some(Value::Number(42.0)));
    }

    #[test]
    fn test_undefined_variable() {
        let env = Env::new();
        assert_eq!(env.get("x"), None);
    }

    #[test]
    fn test_nested_scopes() {
        let env = Env::new();
        env.define("x", Value::Number(1.0));

        env.push_scope();
        env.define("y", Value::Number(2.0));

        // Can see both x and y
        assert_eq!(env.get("x"), Some(Value::Number(1.0)));
        assert_eq!(env.get("y"), Some(Value::Number(2.0)));

        env.pop_scope();

        // Can still see x, but not y
        assert_eq!(env.get("x"), Some(Value::Number(1.0)));
        assert_eq!(env.get("y"), None);
    }

    #[test]
    fn test_shadowing() {
        let env = Env::new();
        env.define("x", Value::Number(1.0));

        env.push_scope();
        env.define("x", Value::Number(2.0));

        // Inner x shadows outer x
        assert_eq!(env.get("x"), Some(Value::Number(2.0)));

        env.pop_scope();

        // Outer x is restored
        assert_eq!(env.get("x"), Some(Value::Number(1.0)));
    }

    #[test]
    fn test_set_updates_correct_scope() {
        let env = Env::new();
        env.define("x", Value::Number(1.0));

        env.push_scope();
        // No new x defined here, so set should update outer x
        env.set("x", Value::Number(99.0));
        assert_eq!(env.get("x"), Some(Value::Number(99.0)));

        env.pop_scope();
        // Outer x was modified
        assert_eq!(env.get("x"), Some(Value::Number(99.0)));
    }

    #[test]
    fn test_set_with_shadowing() {
        let env = Env::new();
        env.define("x", Value::Number(1.0));

        env.push_scope();
        env.define("x", Value::Number(2.0));

        // Set updates the inner x
        env.set("x", Value::Number(99.0));
        assert_eq!(env.get("x"), Some(Value::Number(99.0)));

        env.pop_scope();
        // Outer x is unchanged
        assert_eq!(env.get("x"), Some(Value::Number(1.0)));
    }

    #[test]
    fn test_closure_capture() {
        let env = Env::new();
        env.define("x", Value::Number(1.0));

        // Simulate creating a closure - capture the environment
        let captured = env.capture();

        // Modify the original environment
        env.set("x", Value::Number(999.0));

        // With shared bindings, the captured environment SEES the change
        // This is correct JavaScript semantics for closures
        assert_eq!(captured.get("x"), Some(Value::Number(999.0)));

        // The original also has the new value
        assert_eq!(env.get("x"), Some(Value::Number(999.0)));

        // Modifications through the captured env are also visible in original
        captured.set("x", Value::Number(42.0));
        assert_eq!(env.get("x"), Some(Value::Number(42.0)));
    }

    #[test]
    fn test_closure_nested_capture() {
        let env = Env::new();
        env.define("a", Value::Number(1.0));

        env.push_scope();
        env.define("b", Value::Number(2.0));

        // Capture at this point sees both a and b
        let captured = env.capture();

        env.push_scope();
        env.define("c", Value::Number(3.0));

        // Captured env doesn't see c
        assert_eq!(captured.get("a"), Some(Value::Number(1.0)));
        assert_eq!(captured.get("b"), Some(Value::Number(2.0)));
        assert_eq!(captured.get("c"), None);
    }

    #[test]
    fn test_value_is_truthy() {
        assert_eq!(Value::Number(1.0).is_truthy(), Some(true));
        assert_eq!(Value::Number(0.0).is_truthy(), Some(false));
        assert_eq!(Value::String("hello".to_string()).is_truthy(), Some(true));
        assert_eq!(Value::String("".to_string()).is_truthy(), Some(false));
        assert_eq!(Value::Bool(true).is_truthy(), Some(true));
        assert_eq!(Value::Bool(false).is_truthy(), Some(false));
        assert_eq!(Value::Undefined.is_truthy(), Some(false));
        assert_eq!(Value::Null.is_truthy(), Some(false));
        assert_eq!(Value::Dynamic("x".to_string()).is_truthy(), None);
    }

    #[test]
    fn test_dynamic_values() {
        let env = Env::new();
        env.define("x", Value::Dynamic("x".to_string()));

        assert!(env.get("x").unwrap().is_dynamic());
        assert!(!env.get("x").unwrap().is_static());
    }
}
