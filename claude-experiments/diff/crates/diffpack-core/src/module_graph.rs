//! Dense, stable module identity and record storage.
//!
//! Loaders decide what a record contains; core owns the invariant that a module
//! id is interned once, keeps one dense index for the graph's lifetime, and has
//! a record slot that may be cleared and repopulated during incremental builds.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

pub type DenseModuleId = usize;
pub type SharedModuleId = Arc<str>;

pub struct ModuleGraph<T> {
    pub entry: DenseModuleId,
    pub ids: Vec<SharedModuleId>,
    pub indices: HashMap<SharedModuleId, DenseModuleId>,
    pub modules: Vec<Option<T>>,
}

impl<T> Default for ModuleGraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ModuleGraph<T> {
    pub fn new() -> Self {
        Self {
            entry: 0,
            ids: Vec::new(),
            indices: HashMap::new(),
            modules: Vec::new(),
        }
    }

    /// Returns the stable dense id, allocating its empty record slot once.
    pub fn intern(&mut self, id: SharedModuleId) -> DenseModuleId {
        if let Some(&index) = self.indices.get(id.as_ref()) {
            return index;
        }
        let index = self.ids.len();
        self.ids.push(id.clone());
        self.indices.insert(id, index);
        self.modules.push(None);
        index
    }

    pub fn entry_id(&self) -> &str {
        &self.ids[self.entry]
    }
}

/// Borrowed framework-neutral static dependency graph used by linking and
/// rendering order derivations.
pub struct StaticGraph<'a> {
    pub module_ids: &'a [SharedModuleId],
    pub present: &'a [bool],
    pub edges: &'a [Vec<DenseModuleId>],
}

pub trait StaticGraphView {
    type Dependencies<'a>: Iterator<Item = DenseModuleId>
    where
        Self: 'a;

    fn module_id(&self, module: DenseModuleId) -> &str;
    fn present(&self, module: DenseModuleId) -> bool;
    fn dependencies(&self, module: DenseModuleId) -> Self::Dependencies<'_>;
}

impl StaticGraphView for StaticGraph<'_> {
    type Dependencies<'a>
        = std::iter::Copied<std::slice::Iter<'a, DenseModuleId>>
    where
        Self: 'a;

    fn module_id(&self, module: DenseModuleId) -> &str {
        &self.module_ids[module]
    }

    fn present(&self, module: DenseModuleId) -> bool {
        self.present.get(module).copied().unwrap_or(false)
    }

    fn dependencies(&self, module: DenseModuleId) -> Self::Dependencies<'_> {
        self.edges[module].iter().copied()
    }
}

impl StaticGraph<'_> {
    pub fn closure(
        &self,
        root: DenseModuleId,
        allowed: &HashSet<DenseModuleId>,
    ) -> Vec<DenseModuleId> {
        static_closure(self, root, allowed)
    }

    /// Returns dependency-first execution order, or `None` for a cycle or a
    /// missing record. The root's source edge order wins; disconnected members
    /// follow in stable module-id order.
    pub fn execution_order(
        &self,
        root: DenseModuleId,
        allowed: &HashSet<DenseModuleId>,
    ) -> Option<Vec<DenseModuleId>> {
        static_execution_order(self, root, allowed)
    }
}

pub fn static_closure(
    graph: &impl StaticGraphView,
    root: DenseModuleId,
    allowed: &HashSet<DenseModuleId>,
) -> Vec<DenseModuleId> {
    let mut seen = HashSet::new();
    let mut pending = vec![root];
    while let Some(source) = pending.pop() {
        if !allowed.contains(&source) || !graph.present(source) || !seen.insert(source) {
            continue;
        }
        pending.extend(graph.dependencies(source));
    }
    let mut modules = seen.into_iter().collect::<Vec<_>>();
    modules.sort_by(|left, right| graph.module_id(*left).cmp(graph.module_id(*right)));
    modules
}

pub fn static_execution_order(
    graph: &impl StaticGraphView,
    root: DenseModuleId,
    allowed: &HashSet<DenseModuleId>,
) -> Option<Vec<DenseModuleId>> {
    fn visit<G: StaticGraphView>(
        graph: &G,
        source: DenseModuleId,
        allowed: &HashSet<DenseModuleId>,
        states: &mut HashMap<DenseModuleId, u8>,
        order: &mut Vec<DenseModuleId>,
    ) -> Option<()> {
        match states.get(&source) {
            Some(1) => return None,
            Some(2) => return Some(()),
            _ => {}
        }
        if !graph.present(source) {
            return None;
        }
        states.insert(source, 1);
        for target in graph.dependencies(source) {
            if allowed.contains(&target) {
                visit(graph, target, allowed, states, order)?;
            }
        }
        states.insert(source, 2);
        order.push(source);
        Some(())
    }

    let mut roots = Vec::with_capacity(allowed.len());
    if allowed.contains(&root) {
        roots.push(root);
    }
    let mut rest = allowed
        .iter()
        .copied()
        .filter(|module| *module != root)
        .collect::<Vec<_>>();
    rest.sort_by(|left, right| graph.module_id(*left).cmp(graph.module_id(*right)));
    roots.extend(rest);
    let mut order = Vec::with_capacity(allowed.len());
    let mut states = HashMap::new();
    for root in roots {
        visit(graph, root, allowed, &mut states, &mut order)?;
    }
    (order.len() == allowed.len()).then_some(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_stable_and_allocates_one_record_slot() {
        let mut graph = ModuleGraph::<String>::new();
        let first = graph.intern(Arc::from("/app/entry.js"));
        let again = graph.intern(Arc::from("/app/entry.js"));
        assert_eq!(first, again);
        assert_eq!(graph.ids.len(), 1);
        assert_eq!(graph.modules.len(), 1);
        graph.entry = first;
        graph.modules[first] = Some("compiled".to_string());
        assert_eq!(graph.entry_id(), "/app/entry.js");
    }

    #[test]
    fn static_order_preserves_root_edge_order_and_rejects_cycles() {
        let ids = ["entry", "a", "b"].map(|id| SharedModuleId::from(id));
        let present = [true, true, true];
        let edges = vec![vec![2, 1], vec![], vec![]];
        let graph = StaticGraph {
            module_ids: &ids,
            present: &present,
            edges: &edges,
        };
        let allowed = HashSet::from([0, 1, 2]);
        assert_eq!(graph.execution_order(0, &allowed), Some(vec![2, 1, 0]));

        let cyclic = vec![vec![1], vec![0], vec![]];
        let graph = StaticGraph {
            edges: &cyclic,
            ..graph
        };
        assert_eq!(graph.execution_order(0, &allowed), None);
    }
}
