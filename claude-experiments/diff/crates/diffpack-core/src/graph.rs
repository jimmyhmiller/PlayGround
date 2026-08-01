//! Framework-independent incremental graph records and reachability repair.

use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::{Diagnostic, ModuleId};

/// Result of initial discovery or one incremental module reload.
#[derive(Debug)]
pub struct BuildUpdate {
    pub delta: GraphDelta,
    pub transformed_modules: usize,
    pub diagnostics: Vec<Diagnostic>,
}

/// The observable changes made to a module graph by one rebuild.
#[derive(Debug, Clone, Default)]
pub struct GraphDelta {
    pub edge_updates: Vec<((ModuleId, ModuleId), isize)>,
    pub changed: BTreeSet<ModuleId>,
}

#[derive(Debug, Default)]
pub struct DirectReachabilityUpdate {
    pub added: BTreeSet<ModuleId>,
    pub removed: BTreeSet<ModuleId>,
    pub used_full_recompute: bool,
}

/// A compact persistent single-entry reachability index.
///
/// Its selected parent edges form a spanning tree. Removing a non-tree edge is
/// constant-time; removing a tree edge repairs only the detached subtree unless
/// that subtree is large enough that a dense full traversal is cheaper.
pub struct DirectReachability {
    ids: Vec<ModuleId>,
    indices: HashMap<ModuleId, usize>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    reachable: Vec<bool>,
    parent: Vec<Option<usize>>,
    tree_children: Vec<Vec<usize>>,
    subtree_marks: Vec<u32>,
    mark_epoch: u32,
    entry: usize,
    reachable_count: usize,
}

impl DirectReachability {
    const RECOMPUTE_NUMERATOR: usize = 1;
    const RECOMPUTE_DENOMINATOR: usize = 4;

    /// Constructs the dense index from graph records without depending on the
    /// bundler's storage representation.
    pub fn new(
        entry: impl Into<ModuleId>,
        modules: impl IntoIterator<Item = ModuleId>,
        edges: impl IntoIterator<Item = (ModuleId, ModuleId)>,
    ) -> Self {
        let entry = entry.into();
        let mut ids = Vec::new();
        let mut indices = HashMap::new();
        for id in std::iter::once(entry.clone()).chain(modules) {
            if !indices.contains_key(&id) {
                indices.insert(id.clone(), ids.len());
                ids.push(id);
            }
        }
        let entry_index = indices[&entry];
        let node_count = ids.len();
        let mut graph = Self {
            ids,
            indices,
            outgoing: vec![Vec::new(); node_count],
            incoming: vec![Vec::new(); node_count],
            reachable: vec![false; node_count],
            parent: vec![None; node_count],
            tree_children: vec![Vec::new(); node_count],
            subtree_marks: vec![0; node_count],
            mark_epoch: 0,
            entry: entry_index,
            reachable_count: 0,
        };
        for (source, target) in edges {
            let source = graph.intern(&source);
            let target = graph.intern(&target);
            graph.insert_edge(source, target);
        }
        graph.recompute();
        graph
    }

    pub fn reachable_modules(&self) -> BTreeSet<ModuleId> {
        self.reachable
            .iter()
            .enumerate()
            .filter(|(_, reachable)| **reachable)
            .map(|(index, _)| self.ids[index].clone())
            .collect()
    }

    pub fn apply(&mut self, revision: &GraphDelta) -> DirectReachabilityUpdate {
        let mut update = DirectReachabilityUpdate::default();
        for ((source, target), diff) in &revision.edge_updates {
            if *diff > 0 {
                let source = self.intern(source);
                let target = self.intern(target);
                if self.insert_edge(source, target)
                    && self.reachable[source]
                    && !self.reachable[target]
                {
                    self.activate_from(target, source, &mut update);
                }
            }
        }
        for ((source, target), diff) in &revision.edge_updates {
            if *diff < 0 {
                let Some(&source) = self.indices.get(source) else {
                    continue;
                };
                let Some(&target) = self.indices.get(target) else {
                    continue;
                };
                if self.remove_edge(source, target) && self.parent[target] == Some(source) {
                    self.repair_detached_subtree(source, target, &mut update);
                }
            }
        }
        update
    }

    fn intern(&mut self, id: &str) -> usize {
        if let Some(&index) = self.indices.get(id) {
            return index;
        }
        let index = self.ids.len();
        self.ids.push(id.to_owned());
        self.indices.insert(id.to_owned(), index);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        self.reachable.push(false);
        self.parent.push(None);
        self.tree_children.push(Vec::new());
        self.subtree_marks.push(0);
        index
    }

    fn insert_edge(&mut self, source: usize, target: usize) -> bool {
        if self.outgoing[source].contains(&target) {
            return false;
        }
        self.outgoing[source].push(target);
        self.incoming[target].push(source);
        true
    }

    fn remove_edge(&mut self, source: usize, target: usize) -> bool {
        let Some(position) = self.outgoing[source]
            .iter()
            .position(|item| *item == target)
        else {
            return false;
        };
        self.outgoing[source].swap_remove(position);
        if let Some(position) = self.incoming[target]
            .iter()
            .position(|item| *item == source)
        {
            self.incoming[target].swap_remove(position);
        }
        true
    }

    fn recompute(&mut self) {
        self.reachable.fill(false);
        self.parent.fill(None);
        for children in &mut self.tree_children {
            children.clear();
        }
        self.reachable_count = 1;
        self.reachable[self.entry] = true;
        let mut queue = VecDeque::from([self.entry]);
        while let Some(source) = queue.pop_front() {
            for &target in &self.outgoing[source] {
                if self.reachable[target] {
                    continue;
                }
                self.reachable[target] = true;
                self.reachable_count += 1;
                self.parent[target] = Some(source);
                self.tree_children[source].push(target);
                queue.push_back(target);
            }
        }
    }

    fn activate_from(
        &mut self,
        target: usize,
        parent: usize,
        update: &mut DirectReachabilityUpdate,
    ) {
        self.set_reachable(target, true, update);
        self.parent[target] = Some(parent);
        self.tree_children[parent].push(target);
        let mut queue = VecDeque::from([target]);
        while let Some(source) = queue.pop_front() {
            for edge_index in 0..self.outgoing[source].len() {
                let target = self.outgoing[source][edge_index];
                if !self.reachable[target] {
                    self.set_reachable(target, true, update);
                    self.parent[target] = Some(source);
                    self.tree_children[source].push(target);
                    queue.push_back(target);
                }
            }
        }
    }

    fn repair_detached_subtree(
        &mut self,
        old_parent: usize,
        root: usize,
        update: &mut DirectReachabilityUpdate,
    ) {
        if let Some(position) = self.tree_children[old_parent]
            .iter()
            .position(|child| *child == root)
        {
            self.tree_children[old_parent].swap_remove(position);
        }
        let mut subtree = Vec::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            subtree.push(node);
            stack.extend(self.tree_children[node].iter().copied());
        }
        if subtree.len() * Self::RECOMPUTE_DENOMINATOR
            >= self.reachable_count * Self::RECOMPUTE_NUMERATOR
        {
            let before = self.reachable.clone();
            self.recompute();
            for (node, was_reachable) in before.into_iter().enumerate() {
                if was_reachable != self.reachable[node] {
                    self.record_change(node, self.reachable[node], update);
                }
            }
            update.used_full_recompute = true;
            return;
        }
        self.mark_epoch = self.mark_epoch.wrapping_add(1);
        if self.mark_epoch == 0 {
            self.subtree_marks.fill(0);
            self.mark_epoch = 1;
        }
        for &node in &subtree {
            self.subtree_marks[node] = self.mark_epoch;
            self.set_reachable(node, false, update);
            self.parent[node] = None;
            self.tree_children[node].clear();
        }
        let mut queue = VecDeque::new();
        for &node in &subtree {
            if let Some(parent) = self.incoming[node]
                .iter()
                .copied()
                .find(|predecessor| self.reachable[*predecessor])
            {
                self.set_reachable(node, true, update);
                self.parent[node] = Some(parent);
                self.tree_children[parent].push(node);
                queue.push_back(node);
            }
        }
        while let Some(source) = queue.pop_front() {
            for edge_index in 0..self.outgoing[source].len() {
                let target = self.outgoing[source][edge_index];
                if self.subtree_marks[target] == self.mark_epoch && !self.reachable[target] {
                    self.set_reachable(target, true, update);
                    self.parent[target] = Some(source);
                    self.tree_children[source].push(target);
                    queue.push_back(target);
                }
            }
        }
    }

    fn set_reachable(
        &mut self,
        node: usize,
        reachable: bool,
        update: &mut DirectReachabilityUpdate,
    ) {
        if self.reachable[node] == reachable {
            return;
        }
        self.reachable[node] = reachable;
        if reachable {
            self.reachable_count += 1;
        } else {
            self.reachable_count -= 1;
        }
        self.record_change(node, reachable, update);
    }

    fn record_change(&self, node: usize, reachable: bool, update: &mut DirectReachabilityUpdate) {
        let id = &self.ids[node];
        if reachable {
            if !update.removed.remove(id) {
                update.added.insert(id.clone());
            }
        } else if !update.added.remove(id) {
            update.removed.insert(id.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_a_detached_subtree() {
        let mut graph = DirectReachability::new(
            "entry".to_string(),
            ["a", "b", "unused"].map(str::to_string),
            [("entry", "a"), ("a", "b")].map(|(a, b)| (a.into(), b.into())),
        );
        let update = graph.apply(&GraphDelta {
            edge_updates: vec![(("entry".into(), "a".into()), -1)],
            changed: BTreeSet::new(),
        });
        assert_eq!(update.removed, ["a".into(), "b".into()].into());
        assert_eq!(graph.reachable_modules(), ["entry".into()].into());
    }

    #[test]
    fn an_alternate_parent_repairs_a_detached_branch() {
        let mut graph = DirectReachability::new(
            "entry".to_string(),
            ["a", "b", "shared"].map(str::to_string),
            [
                ("entry", "a"),
                ("entry", "b"),
                ("a", "shared"),
                ("b", "shared"),
            ]
            .map(|(a, b)| (a.into(), b.into())),
        );
        let update = graph.apply(&GraphDelta {
            edge_updates: vec![(("a".into(), "shared".into()), -1)],
            changed: BTreeSet::new(),
        });
        assert!(update.removed.is_empty());
        assert!(graph.reachable_modules().contains("shared"));
    }
}
