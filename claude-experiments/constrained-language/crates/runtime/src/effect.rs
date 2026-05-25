//! Effect adapters. The runtime fulfills each emit by dispatching to the
//! adapter registered for that effect type.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::value::Value;

#[derive(Debug, Clone)]
pub enum AdapterResult {
    Ok(Value),
    Failed { reason: String },
}

pub trait Adapter {
    fn fulfill(&mut self, request: Value, emit_id: u64) -> AdapterResult;
}

#[derive(Default)]
pub struct AdapterRegistry {
    adapters: HashMap<String, Box<dyn Adapter>>,
}

impl AdapterRegistry {
    pub fn register(&mut self, effect: impl Into<String>, adapter: Box<dyn Adapter>) {
        self.adapters.insert(effect.into(), adapter);
    }

    pub fn fulfill(
        &mut self,
        effect: &str,
        request: Value,
        emit_id: u64,
    ) -> Option<AdapterResult> {
        self.adapters
            .get_mut(effect)
            .map(|a| a.fulfill(request, emit_id))
    }
}

/// Shared, inspectable mock adapter for tests. Clone holds a second handle
/// to the same inner state, so tests can register one clone with the
/// runtime and keep the other for assertions.
#[derive(Default, Clone)]
pub struct MockAdapter {
    inner: Rc<RefCell<MockInner>>,
}

#[derive(Default)]
struct MockInner {
    calls: Vec<Value>,
    response: Value,
    fail_with: Option<String>,
}

impl MockAdapter {
    pub fn with_response(response: Value) -> Self {
        Self {
            inner: Rc::new(RefCell::new(MockInner {
                calls: vec![],
                response,
                fail_with: None,
            })),
        }
    }

    pub fn failing(reason: impl Into<String>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(MockInner {
                calls: vec![],
                response: Value::Null,
                fail_with: Some(reason.into()),
            })),
        }
    }

    pub fn calls(&self) -> Vec<Value> {
        self.inner.borrow().calls.clone()
    }
}

impl Adapter for MockAdapter {
    fn fulfill(&mut self, request: Value, _emit_id: u64) -> AdapterResult {
        let mut inner = self.inner.borrow_mut();
        inner.calls.push(request.clone());
        match &inner.fail_with {
            Some(r) => AdapterResult::Failed { reason: r.clone() },
            None => AdapterResult::Ok(inner.response.clone()),
        }
    }
}

/// An adapter that returns a pre-scripted sequence of responses, one per
/// call, popping them in order. Useful for testing flows where the same
/// effect is invoked repeatedly with different expected outcomes (e.g. an
/// LLM that asks for a tool, then gives a final answer).
///
/// Panics on `fulfill` if the script has been exhausted — tests should
/// supply exactly the right number of responses.
#[derive(Default, Clone)]
pub struct ScriptedAdapter {
    inner: Rc<RefCell<ScriptedInner>>,
}

#[derive(Default)]
struct ScriptedInner {
    calls: Vec<Value>,
    responses: std::collections::VecDeque<AdapterResult>,
}

impl ScriptedAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience: build from a list of `Ok` responses.
    pub fn from_ok(responses: impl IntoIterator<Item = Value>) -> Self {
        let me = Self::new();
        for r in responses {
            me.push_ok(r);
        }
        me
    }

    pub fn push_ok(&self, response: Value) {
        self.inner
            .borrow_mut()
            .responses
            .push_back(AdapterResult::Ok(response));
    }

    pub fn push_failed(&self, reason: impl Into<String>) {
        self.inner
            .borrow_mut()
            .responses
            .push_back(AdapterResult::Failed {
                reason: reason.into(),
            });
    }

    pub fn calls(&self) -> Vec<Value> {
        self.inner.borrow().calls.clone()
    }

    pub fn remaining(&self) -> usize {
        self.inner.borrow().responses.len()
    }
}

impl Adapter for ScriptedAdapter {
    fn fulfill(&mut self, request: Value, _emit_id: u64) -> AdapterResult {
        let mut inner = self.inner.borrow_mut();
        inner.calls.push(request);
        inner
            .responses
            .pop_front()
            .expect("ScriptedAdapter: response script exhausted")
    }
}
