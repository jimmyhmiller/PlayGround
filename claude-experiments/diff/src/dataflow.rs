use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use differential_dataflow::input::InputSession;
use differential_dataflow::operators::*;
use timely::dataflow::operators::probe::Handle as ProbeHandle;

use crate::graph::{GraphSnapshot, ModuleId};

#[derive(Debug, Clone)]
pub struct Revision {
    pub label: String,
    pub snapshot: GraphSnapshot,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RevisionResult {
    pub label: String,
    pub reachable: BTreeSet<ModuleId>,
    pub added: BTreeSet<ModuleId>,
    pub removed: BTreeSet<ModuleId>,
    pub changed: BTreeSet<ModuleId>,
    pub module_facts: usize,
    pub edge_facts: usize,
    pub reachable_facts: usize,
    pub diagnostics: Vec<String>,
}

type ReachUpdate = (ModuleId, u64, isize);
type ArtifactUpdate = ((ModuleId, u64), u64, isize);

pub fn run_revisions(revisions: Vec<Revision>) -> Vec<RevisionResult> {
    timely::execute_directly(move |worker| {
        let mut entries = InputSession::<u64, ModuleId, isize>::new();
        let mut edges = InputSession::<u64, (ModuleId, ModuleId), isize>::new();
        let mut modules = InputSession::<u64, (ModuleId, u64), isize>::new();
        let probe = ProbeHandle::new();
        let reach_updates = Rc::new(RefCell::new(Vec::<ReachUpdate>::new()));
        let artifact_updates = Rc::new(RefCell::new(Vec::<ArtifactUpdate>::new()));

        worker.dataflow::<u64, _, _>(|scope| {
            let entry_collection = entries.to_collection(scope);
            let edge_collection = edges.to_collection(scope);
            let module_collection = modules.to_collection(scope);

            let reachable = entry_collection.clone().iterate(|inner_scope, reached| {
                let loop_entries = entry_collection.enter(inner_scope);
                let loop_edges = edge_collection.enter(inner_scope);

                reached
                    .map(|module| (module, ()))
                    .join_map(loop_edges, |_source, &(), target| target.clone())
                    .concat(loop_entries)
                    .distinct()
            });

            let artifacts = module_collection
                .map(|(module, hash)| (module, hash))
                .join_map(
                    reachable.clone().map(|module| (module, ())),
                    |module, hash, &()| (module.clone(), *hash),
                )
                .distinct();

            let reach_output = Rc::clone(&reach_updates);
            reachable
                .consolidate()
                .inspect(move |(module, time, diff)| {
                    reach_output
                        .borrow_mut()
                        .push((module.clone(), *time, *diff));
                })
                .probe_with(&probe);

            let artifact_output = Rc::clone(&artifact_updates);
            artifacts
                .consolidate()
                .inspect(move |(artifact, time, diff)| {
                    artifact_output
                        .borrow_mut()
                        .push((artifact.clone(), *time, *diff));
                })
                .probe_with(&probe);
        });

        let mut previous = GraphSnapshot::default();
        let mut current_reachable = BTreeMap::<ModuleId, isize>::new();
        let mut current_artifacts = BTreeMap::<(ModuleId, u64), isize>::new();
        let mut results = Vec::with_capacity(revisions.len());

        for (time, revision) in revisions.into_iter().enumerate() {
            apply_set_diff(
                &mut entries,
                entry_set(&previous),
                entry_set(&revision.snapshot),
            );
            apply_set_diff(
                &mut edges,
                previous.edges.clone(),
                revision.snapshot.edges.clone(),
            );
            apply_set_diff(
                &mut modules,
                module_set(&previous),
                module_set(&revision.snapshot),
            );

            let next_time = time as u64 + 1;
            entries.advance_to(next_time);
            edges.advance_to(next_time);
            modules.advance_to(next_time);
            entries.flush();
            edges.flush();
            modules.flush();
            while probe.less_than(&next_time) {
                worker.step();
            }

            let old_reachable = positive_keys(&current_reachable);
            for (module, _update_time, diff) in reach_updates.borrow_mut().drain(..) {
                update_count(&mut current_reachable, module, diff);
            }
            for (artifact, _update_time, diff) in artifact_updates.borrow_mut().drain(..) {
                update_count(&mut current_artifacts, artifact, diff);
            }
            let reachable = positive_keys(&current_reachable);
            let added = reachable.difference(&old_reachable).cloned().collect();
            let removed = old_reachable.difference(&reachable).cloned().collect();
            let changed = changed_modules(&previous, &revision.snapshot);

            debug_assert_eq!(
                current_artifacts
                    .values()
                    .filter(|weight| **weight > 0)
                    .count(),
                reachable.len(),
                "each reachable module should have one live content artifact"
            );

            results.push(RevisionResult {
                label: revision.label,
                reachable_facts: reachable.len(),
                reachable,
                added,
                removed,
                changed,
                module_facts: revision.snapshot.modules.len(),
                edge_facts: revision.snapshot.edges.len(),
                diagnostics: revision.snapshot.diagnostics.clone(),
            });
            previous = revision.snapshot;
        }

        results
    })
}

fn entry_set(snapshot: &GraphSnapshot) -> BTreeSet<ModuleId> {
    if snapshot.entry.is_empty() {
        BTreeSet::new()
    } else {
        BTreeSet::from([snapshot.entry.clone()])
    }
}

fn module_set(snapshot: &GraphSnapshot) -> BTreeSet<(ModuleId, u64)> {
    snapshot
        .modules
        .iter()
        .map(|(module, hash)| (module.clone(), *hash))
        .collect()
}

fn apply_set_diff<T: Ord + Clone + std::fmt::Debug + 'static>(
    input: &mut InputSession<u64, T, isize>,
    old: BTreeSet<T>,
    new: BTreeSet<T>,
) {
    for value in old.difference(&new) {
        input.remove(value.clone());
    }
    for value in new.difference(&old) {
        input.insert(value.clone());
    }
}

fn update_count<T: Ord>(counts: &mut BTreeMap<T, isize>, value: T, diff: isize) {
    let new_count = counts.get(&value).copied().unwrap_or(0) + diff;
    if new_count == 0 {
        counts.remove(&value);
    } else {
        counts.insert(value, new_count);
    }
}

fn positive_keys<T: Ord + Clone>(counts: &BTreeMap<T, isize>) -> BTreeSet<T> {
    counts
        .iter()
        .filter(|(_, weight)| **weight > 0)
        .map(|(value, _)| value.clone())
        .collect()
}

fn changed_modules(old: &GraphSnapshot, new: &GraphSnapshot) -> BTreeSet<ModuleId> {
    new.modules
        .iter()
        .filter(|(module, hash)| {
            old.modules
                .get(*module)
                .is_some_and(|old_hash| old_hash != *hash)
        })
        .map(|(module, _)| module.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(entry_source_hash: u64, include_leaf: bool) -> GraphSnapshot {
        let mut modules = BTreeMap::from([
            ("entry.js".into(), entry_source_hash),
            ("shared.js".into(), 2),
        ]);
        let mut edges = BTreeSet::from([("entry.js".into(), "shared.js".into())]);
        if include_leaf {
            modules.insert("leaf.js".into(), 3);
            edges.insert(("entry.js".into(), "leaf.js".into()));
        }
        GraphSnapshot {
            entry: "entry.js".into(),
            modules,
            edges,
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn incrementally_adds_changes_and_retracts_modules() {
        let results = run_revisions(vec![
            Revision {
                label: "initial".into(),
                snapshot: graph(1, false),
            },
            Revision {
                label: "add leaf".into(),
                snapshot: graph(4, true),
            },
            Revision {
                label: "remove leaf".into(),
                snapshot: graph(5, false),
            },
        ]);

        assert_eq!(
            results[0].added,
            BTreeSet::from(["entry.js".into(), "shared.js".into()])
        );
        assert_eq!(results[1].added, BTreeSet::from(["leaf.js".into()]));
        assert_eq!(results[1].changed, BTreeSet::from(["entry.js".into()]));
        assert_eq!(results[2].removed, BTreeSet::from(["leaf.js".into()]));
        assert_eq!(results[2].reachable_facts, 2);
    }
}
