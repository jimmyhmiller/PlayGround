//! Pluggable handlers for opaque/built-in constructs
//!
//! This module provides a way to extend the partial evaluator to handle
//! specific "opaque" constructs like `new TextDecoder()`, `new ArrayBuffer(n)`, etc.
//!
//! Each handler can match on specific patterns and return partially evaluated results.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::ast::Expr;
use crate::partial::{PEnv, PValue};
use crate::value::Value;

// ============================================================================
// Shared Buffer Types
// ============================================================================

/// Shared buffer storage - Rc enables shared ownership, RefCell enables mutation
#[derive(Debug, Clone)]
pub struct SharedBuffer(pub Rc<RefCell<Vec<u8>>>);

impl SharedBuffer {
    pub fn new(size: usize) -> Self {
        SharedBuffer(Rc::new(RefCell::new(vec![0u8; size])))
    }

    pub fn len(&self) -> usize {
        self.0.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }

    pub fn get(&self, i: usize) -> Option<u8> {
        self.0.borrow().get(i).copied()
    }

    pub fn set(&self, i: usize, v: u8) {
        self.0.borrow_mut()[i] = v;
    }

    #[allow(dead_code)]
    pub fn is_zero_filled(&self) -> bool {
        self.0.borrow().iter().all(|&b| b == 0)
    }

    #[allow(dead_code)]
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.borrow().clone()
    }
}

// ============================================================================
// State Types for Typed Arrays
// ============================================================================

/// State for ArrayBuffer
#[derive(Debug, Clone)]
pub struct ArrayBufferState {
    pub buffer: SharedBuffer,
}

/// State for DataView
#[derive(Debug, Clone)]
pub struct DataViewState {
    pub buffer: SharedBuffer,
    pub byte_offset: usize,
    pub byte_length: usize,
}

/// State for Uint8Array
#[derive(Debug, Clone)]
pub struct Uint8ArrayState {
    pub buffer: SharedBuffer,
    pub byte_offset: usize,
    pub length: usize,
}

/// State for TextDecoder
#[derive(Debug, Clone)]
pub struct TextDecoderState {
    pub encoding: String, // "utf-8", "utf-16le", etc.
}

/// Trait for handling opaque constructs during partial evaluation
pub trait OpaqueHandler: Send + Sync {
    /// A unique name for this handler (for debugging/logging)
    fn name(&self) -> &'static str;

    /// Check if this handler can handle the given `new` expression
    /// Returns Some with the constructor name if it matches, None otherwise
    fn matches_new(&self, ctor: &Expr, args: &[Expr]) -> bool {
        let _ = (ctor, args);
        false
    }

    /// Check if this handler can handle the given method call
    /// Returns true if it matches
    fn matches_call(&self, callee: &Expr, args: &[Expr]) -> bool {
        let _ = (callee, args);
        false
    }

    /// Handle a `new` expression. Called only if matches_new returned true.
    /// Returns None to fall back to default behavior (emit as residual)
    fn handle_new(
        &self,
        ctor: &Expr,
        args: &[PValue],
        env: &PEnv,
    ) -> Option<PValue> {
        let _ = (ctor, args, env);
        None
    }

    /// Handle a call expression. Called only if matches_call returned true.
    /// Returns None to fall back to default behavior
    fn handle_call(
        &self,
        callee: &Expr,
        args: &[PValue],
        env: &PEnv,
    ) -> Option<PValue> {
        let _ = (callee, args, env);
        None
    }
}

/// Registry of opaque handlers
pub struct OpaqueRegistry {
    handlers: Vec<Box<dyn OpaqueHandler>>,
}

impl Default for OpaqueRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl OpaqueRegistry {
    pub fn new() -> Self {
        OpaqueRegistry {
            handlers: Vec::new(),
        }
    }

    /// Create a registry with the standard built-in handlers
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(TextDecoderHandler));
        registry.register(Box::new(ArrayBufferHandler));
        registry.register(Box::new(DataViewHandler));
        registry.register(Box::new(Uint8ArrayHandler));
        registry
    }

    /// Register a new handler
    pub fn register(&mut self, handler: Box<dyn OpaqueHandler>) {
        self.handlers.push(handler);
    }

    /// Try to handle a `new` expression with registered handlers
    pub fn try_handle_new(
        &self,
        ctor: &Expr,
        args_exprs: &[Expr],
        args_pvalues: &[PValue],
        env: &PEnv,
    ) -> Option<PValue> {
        for handler in &self.handlers {
            if handler.matches_new(ctor, args_exprs) {
                if let Some(result) = handler.handle_new(ctor, args_pvalues, env) {
                    return Some(result);
                }
            }
        }
        None
    }

    /// Try to handle a call expression with registered handlers
    pub fn try_handle_call(
        &self,
        callee: &Expr,
        args_exprs: &[Expr],
        args_pvalues: &[PValue],
        env: &PEnv,
    ) -> Option<PValue> {
        for handler in &self.handlers {
            if handler.matches_call(callee, args_exprs) {
                if let Some(result) = handler.handle_call(callee, args_pvalues, env) {
                    return Some(result);
                }
            }
        }
        None
    }

    /// List all registered handler names
    pub fn handler_names(&self) -> Vec<&'static str> {
        self.handlers.iter().map(|h| h.name()).collect()
    }
}

// ============================================================================
// Built-in Handlers
// ============================================================================

/// Handler for TextDecoder - creates an opaque decoder value
pub struct TextDecoderHandler;

impl OpaqueHandler for TextDecoderHandler {
    fn name(&self) -> &'static str {
        "TextDecoder"
    }

    fn matches_new(&self, ctor: &Expr, args: &[Expr]) -> bool {
        // TextDecoder can be created with no args or with an encoding string
        if !matches!(ctor, Expr::Var(name) if name == "TextDecoder") {
            return false;
        }
        match args.len() {
            0 => true,
            1 => matches!(&args[0], Expr::String(_)),
            _ => false,
        }
    }

    fn handle_new(
        &self,
        _ctor: &Expr,
        args: &[PValue],
        _env: &PEnv,
    ) -> Option<PValue> {
        // Get encoding from first arg, default to "utf-8"
        let encoding = match args.first() {
            Some(PValue::Static(Value::String(s))) => s.clone(),
            Some(PValue::StaticNamed { value: Value::String(s), .. }) => s.clone(),
            None => "utf-8".to_string(),
            _ => return None, // Dynamic encoding, can't handle
        };

        let state = TextDecoderState { encoding: encoding.clone() };

        // Build the expression for residualization
        let expr = if encoding == "utf-8" {
            Expr::New(Box::new(Expr::Var("TextDecoder".into())), vec![])
        } else {
            Expr::New(
                Box::new(Expr::Var("TextDecoder".into())),
                vec![Expr::String(encoding)],
            )
        };

        Some(PValue::Static(Value::Opaque {
            label: "TextDecoder".to_string(),
            expr,
            state: Some(Rc::new(state) as Rc<dyn Any>),
        }))
    }
}

/// Handler for ArrayBuffer - tracks buffer size
pub struct ArrayBufferHandler;

impl OpaqueHandler for ArrayBufferHandler {
    fn name(&self) -> &'static str {
        "ArrayBuffer"
    }

    fn matches_new(&self, ctor: &Expr, args: &[Expr]) -> bool {
        matches!(ctor, Expr::Var(name) if name == "ArrayBuffer") && args.len() == 1
    }

    fn handle_new(
        &self,
        _ctor: &Expr,
        args: &[PValue],
        _env: &PEnv,
    ) -> Option<PValue> {
        // If the size argument is static, we can create an opaque buffer with known size
        if let Some(PValue::Static(Value::Int(size))) = args.first() {
            let state = ArrayBufferState {
                buffer: SharedBuffer::new(*size as usize),
            };
            Some(PValue::Static(Value::Opaque {
                label: format!("ArrayBuffer({})", size),
                expr: Expr::New(
                    Box::new(Expr::Var("ArrayBuffer".into())),
                    vec![Expr::Int(*size)],
                ),
                state: Some(Rc::new(state) as Rc<dyn Any>),
            }))
        } else {
            None // Fall back to default
        }
    }
}

/// Handler for DataView - wraps a buffer
pub struct DataViewHandler;

impl OpaqueHandler for DataViewHandler {
    fn name(&self) -> &'static str {
        "DataView"
    }

    fn matches_new(&self, ctor: &Expr, args: &[Expr]) -> bool {
        matches!(ctor, Expr::Var(name) if name == "DataView") && args.len() == 1
    }

    fn handle_new(
        &self,
        _ctor: &Expr,
        args: &[PValue],
        _env: &PEnv,
    ) -> Option<PValue> {
        // Extract the opaque value from either Static or StaticNamed
        let (label, state_opt, buf_expr) = match args.first() {
            Some(PValue::Static(Value::Opaque { label, state, expr })) => {
                (label, state.as_ref(), expr.clone())
            }
            Some(PValue::StaticNamed { name, value: Value::Opaque { label, state, .. } }) => {
                // Use the variable name for the expression instead of the opaque's expr
                (label, state.as_ref(), Expr::Var(name.clone()))
            }
            _ => return None,
        };

        // If the buffer argument is a known ArrayBuffer with state, wrap it
        if let Some(s) = state_opt {
            if let Some(ab_state) = s.downcast_ref::<ArrayBufferState>() {
                let state = DataViewState {
                    buffer: ab_state.buffer.clone(), // Rc clone - shares bytes!
                    byte_offset: 0,
                    byte_length: ab_state.buffer.len(),
                };
                return Some(PValue::Static(Value::Opaque {
                    label: format!("DataView({})", label),
                    expr: Expr::New(
                        Box::new(Expr::Var("DataView".into())),
                        vec![buf_expr],
                    ),
                    state: Some(Rc::new(state) as Rc<dyn Any>),
                }));
            }
        }
        None
    }
}

/// Handler for Uint8Array - typed array view
pub struct Uint8ArrayHandler;

impl OpaqueHandler for Uint8ArrayHandler {
    fn name(&self) -> &'static str {
        "Uint8Array"
    }

    fn matches_new(&self, ctor: &Expr, _args: &[Expr]) -> bool {
        matches!(ctor, Expr::Var(name) if name == "Uint8Array")
    }

    fn handle_new(
        &self,
        _ctor: &Expr,
        args: &[PValue],
        _env: &PEnv,
    ) -> Option<PValue> {
        // Handle Uint8Array with ArrayBuffer argument - check both Static and StaticNamed
        let opaque_info: Option<(&String, &Rc<dyn Any>, Expr)> = match args.first() {
            Some(PValue::Static(Value::Opaque { label, state: Some(ref s), expr })) => {
                Some((label, s, expr.clone()))
            }
            Some(PValue::StaticNamed { name, value: Value::Opaque { label, state: Some(ref s), .. } }) => {
                Some((label, s, Expr::Var(name.clone())))
            }
            _ => None,
        };

        if let Some((label, s, buf_expr)) = opaque_info {
            if let Some(ab_state) = s.downcast_ref::<ArrayBufferState>() {
                let len = ab_state.buffer.len();
                let state = Uint8ArrayState {
                    buffer: ab_state.buffer.clone(), // Rc clone - shares bytes!
                    byte_offset: 0,
                    length: len,
                };
                return Some(PValue::Static(Value::Opaque {
                    label: format!("Uint8Array({})", label),
                    expr: Expr::New(
                        Box::new(Expr::Var("Uint8Array".into())),
                        vec![buf_expr],
                    ),
                    state: Some(Rc::new(state) as Rc<dyn Any>),
                }));
            }
        }

        // Handle Uint8Array with static array argument
        if let Some(PValue::Static(Value::Array(elements))) = args.first() {
            let borrowed = elements.borrow();
            let static_elements: Vec<_> = borrowed
                .iter()
                .filter_map(|v| {
                    if let Value::Int(n) = v {
                        Some(*n as u8)
                    } else {
                        None
                    }
                })
                .collect();

            if static_elements.len() == borrowed.len() {
                // Create a SharedBuffer with the static elements
                let buffer = SharedBuffer::new(static_elements.len());
                for (i, &v) in static_elements.iter().enumerate() {
                    buffer.set(i, v);
                }
                let state = Uint8ArrayState {
                    buffer,
                    byte_offset: 0,
                    length: static_elements.len(),
                };
                return Some(PValue::Static(Value::Opaque {
                    label: format!("Uint8Array[{}]", borrowed.len()),
                    expr: Expr::New(
                        Box::new(Expr::Var("Uint8Array".into())),
                        vec![Expr::Array(
                            static_elements.iter().map(|&v| Expr::Int(v as i64)).collect(),
                        )],
                    ),
                    state: Some(Rc::new(state) as Rc<dyn Any>),
                }));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = OpaqueRegistry::with_builtins();
        let names = registry.handler_names();
        assert!(names.contains(&"TextDecoder"));
        assert!(names.contains(&"ArrayBuffer"));
        assert!(names.contains(&"DataView"));
        assert!(names.contains(&"Uint8Array"));
    }

    #[test]
    fn test_text_decoder_matches() {
        let handler = TextDecoderHandler;
        let ctor = Expr::Var("TextDecoder".to_string());
        assert!(handler.matches_new(&ctor, &[]));
        assert!(!handler.matches_new(&ctor, &[Expr::Int(1)]));

        let wrong_ctor = Expr::Var("ArrayBuffer".to_string());
        assert!(!handler.matches_new(&wrong_ctor, &[]));
    }

    #[test]
    fn test_array_buffer_matches() {
        let handler = ArrayBufferHandler;
        let ctor = Expr::Var("ArrayBuffer".to_string());
        assert!(handler.matches_new(&ctor, &[Expr::Int(8)]));
        assert!(!handler.matches_new(&ctor, &[]));
    }
}
