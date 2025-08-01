use std::collections::HashMap;
use std::sync::Arc;
use bumpalo::Bump;
use crate::{ScopeId, ScopeSet, BindingId, BindingInfo, MacroTransformer};

#[derive(Debug)]
pub struct BindingEnv<'arena> {
    pub arena: &'arena Bump,
    pub bindings: HashMap<String, Vec<(ScopeSet, BindingInfo<'arena>)>>,
    pub current_scope: ScopeId,
    pub scope_counter: u32,
    pub binding_counter: u32,
}

impl<'arena> BindingEnv<'arena> {
    pub fn new(arena: &'arena Bump) -> Self {
        Self {
            arena,
            bindings: HashMap::new(),
            current_scope: ScopeId(0),
            scope_counter: 1,
            binding_counter: 0,
        }
    }

    pub fn new_scope(&mut self) -> ScopeId {
        let id = ScopeId(self.scope_counter);
        self.scope_counter += 1;
        id
    }

    pub fn new_binding(&mut self) -> BindingId {
        let id = BindingId(self.binding_counter);
        self.binding_counter += 1;
        id
    }

    pub fn lookup(&self, name: &str, scopes: &ScopeSet) -> Option<&BindingInfo<'arena>> {
        self.bindings.get(name)?
            .iter()
            .filter(|(binding_scopes, _)| binding_scopes.subset_of(scopes))
            .max_by_key(|(binding_scopes, _)| binding_scopes.scopes.len())
            .map(|(_, info)| info)
    }

    pub fn lookup_binary_op(&self, name: &str, scopes: &ScopeSet) -> Option<&BindingInfo<'arena>> {
        self.bindings.get(name)?
            .iter()
            .filter(|(binding_scopes, _)| binding_scopes.subset_of(scopes))
            .find(|(_, info)| matches!(info, BindingInfo::BinaryOp { .. }))
            .map(|(_, info)| info)
    }

    pub fn bind(&mut self, name: String, scopes: ScopeSet, info: BindingInfo<'arena>) {
        self.bindings.entry(name)
            .or_insert_with(Vec::new)
            .push((scopes, info));
    }

    pub fn current_scopes(&self) -> ScopeSet {
        ScopeSet::new().with_scope(self.current_scope)
    }
}

impl<'arena> Clone for BindingEnv<'arena> {
    fn clone(&self) -> Self {
        Self {
            arena: self.arena,
            bindings: self.bindings.clone(),
            current_scope: self.current_scope,
            scope_counter: self.scope_counter,
            binding_counter: self.binding_counter,
        }
    }
}