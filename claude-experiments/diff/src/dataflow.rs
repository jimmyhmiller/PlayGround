use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use differential_dataflow::input::InputSession;
use differential_dataflow::operators::*;
use timely::dataflow::operators::probe::Handle as ProbeHandle;

use crate::graph::{GraphSnapshot, ModuleId};

type Edge = (ModuleId, ModuleId);
type ModuleVersion = (ModuleId, u64);
type ReachUpdate = (ModuleId, u64, isize);
type ArtifactUpdate = (ModuleVersion, u64, isize);

#[derive(Debug, Clone)]
pub struct Revision {
    pub label: String,
    pub snapshot: GraphSnapshot,
}

/// A revision expressed as weighted facts rather than a complete graph snapshot.
#[derive(Debug, Clone, Default)]
pub struct DeltaRevision {
    pub label: String,
    pub entry_updates: Vec<(ModuleId, isize)>,
    pub edge_updates: Vec<(Edge, isize)>,
    pub module_updates: Vec<(ModuleVersion, isize)>,
    pub changed: BTreeSet<ModuleId>,
    pub diagnostics: Vec<String>,
}

impl DeltaRevision {
    pub fn initial(label: impl Into<String>, snapshot: &GraphSnapshot) -> Self {
        Self {
            label: label.into(),
            entry_updates: vec![(snapshot.entry.clone(), 1)],
            edge_updates: snapshot
                .edges
                .iter()
                .cloned()
                .map(|edge| (edge, 1))
                .collect(),
            module_updates: snapshot
                .modules
                .iter()
                .map(|(module, hash)| ((module.clone(), *hash), 1))
                .collect(),
            diagnostics: snapshot.diagnostics.clone(),
            changed: BTreeSet::new(),
        }
    }

    pub fn between(label: impl Into<String>, old: &GraphSnapshot, new: &GraphSnapshot) -> Self {
        let mut delta = Self {
            label: label.into(),
            diagnostics: new.diagnostics.clone(),
            ..Self::default()
        };

        if old.entry != new.entry {
            if !old.entry.is_empty() {
                delta.entry_updates.push((old.entry.clone(), -1));
            }
            if !new.entry.is_empty() {
                delta.entry_updates.push((new.entry.clone(), 1));
            }
        }
        delta.edge_updates.extend(
            old.edges
                .difference(&new.edges)
                .cloned()
                .map(|edge| (edge, -1)),
        );
        delta.edge_updates.extend(
            new.edges
                .difference(&old.edges)
                .cloned()
                .map(|edge| (edge, 1)),
        );

        let module_ids = old
            .modules
            .keys()
            .chain(new.modules.keys())
            .cloned()
            .collect::<BTreeSet<_>>();
        for module in module_ids {
            let old_hash = old.modules.get(&module).copied();
            let new_hash = new.modules.get(&module).copied();
            if old_hash == new_hash {
                continue;
            }
            if let Some(hash) = old_hash {
                delta.module_updates.push(((module.clone(), hash), -1));
            }
            if let Some(hash) = new_hash {
                delta.module_updates.push(((module.clone(), hash), 1));
            }
            if old_hash.is_some() && new_hash.is_some() {
                delta.changed.insert(module);
            }
        }
        delta
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RevisionResult {
    pub label: String,
    /// Populated by the snapshot compatibility API; omitted by the scalable API.
    pub reachable: BTreeSet<ModuleId>,
    /// Populated by the snapshot compatibility API; use `added_facts` otherwise.
    pub added: BTreeSet<ModuleId>,
    /// Populated by the snapshot compatibility API; use `removed_facts` otherwise.
    pub removed: BTreeSet<ModuleId>,
    pub changed: BTreeSet<ModuleId>,
    pub added_facts: usize,
    pub removed_facts: usize,
    pub changed_facts: usize,
    pub module_facts: usize,
    pub edge_facts: usize,
    pub reachable_facts: usize,
    pub diagnostics: Vec<String>,
    pub input_update_micros: u128,
    pub dataflow_micros: u128,
    pub output_micros: u128,
}

/// Compatibility path for the scanner/demo. It still discovers deltas by comparing snapshots.
pub fn run_revisions(revisions: Vec<Revision>) -> Vec<RevisionResult> {
    let mut previous = GraphSnapshot::default();
    let mut deltas = Vec::with_capacity(revisions.len());
    for revision in revisions {
        let delta = if previous.entry.is_empty() {
            DeltaRevision::initial(revision.label, &revision.snapshot)
        } else {
            DeltaRevision::between(revision.label, &previous, &revision.snapshot)
        };
        previous = revision.snapshot;
        deltas.push(delta);
    }
    run_delta_revisions_inner(deltas, true, true)
}

/// Scalable path: callers provide only changed facts and receive only output deltas/counts.
pub fn run_delta_revisions(revisions: Vec<DeltaRevision>) -> Vec<RevisionResult> {
    run_delta_revisions_inner(revisions, false, false)
}

struct SessionRequest {
    revision: Arc<DeltaRevision>,
    response: mpsc::Sender<WorkerResult>,
}

struct WorkerResult {
    worker_index: usize,
    reach_updates: Vec<(ModuleId, isize)>,
    artifact_delta: isize,
    input_update_micros: u128,
    dataflow_micros: u128,
}

#[derive(Default)]
struct SessionCounts {
    module_facts: isize,
    edge_facts: isize,
    reachable_facts: isize,
    artifact_facts: isize,
}

/// A long-lived Differential Dataflow worker for filesystem-driven revisions.
pub struct DeltaSession {
    senders: Vec<mpsc::Sender<SessionRequest>>,
    counts: Mutex<SessionCounts>,
    thread: Option<JoinHandle<()>>,
}

impl DeltaSession {
    pub fn new() -> Self {
        let workers = std::env::var("DIFFPACK_DATAFLOW_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|workers| *workers > 0)
            .unwrap_or(1);
        Self::with_workers(workers)
    }

    pub fn with_workers(workers: usize) -> Self {
        assert!(workers > 0, "a dataflow session needs at least one worker");
        let mut senders = Vec::with_capacity(workers);
        let mut receivers = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (sender, receiver) = mpsc::channel::<SessionRequest>();
            senders.push(sender);
            receivers.push(Mutex::new(receiver));
        }
        let receivers = Arc::new(receivers);
        let thread = thread::spawn(move || run_session_workers(receivers, workers));
        Self {
            senders,
            counts: Mutex::new(SessionCounts::default()),
            thread: Some(thread),
        }
    }

    pub fn apply(&self, revision: DeltaRevision) -> Result<RevisionResult, String> {
        let mut counts = self
            .counts
            .lock()
            .map_err(|_| "dataflow session state is poisoned".to_string())?;
        let (response, results) = mpsc::channel();
        let revision = Arc::new(revision);
        for sender in &self.senders {
            sender
                .send(SessionRequest {
                    revision: Arc::clone(&revision),
                    response: response.clone(),
                })
                .map_err(|_| "dataflow worker stopped".to_string())?;
        }
        drop(response);

        let output_started = Instant::now();
        let mut reach_updates = Vec::new();
        let mut artifact_delta = 0;
        let mut input_update_micros = 0;
        let mut dataflow_micros = 0;
        for _ in 0..self.senders.len() {
            let result = results
                .recv()
                .map_err(|_| "dataflow worker did not return a result".to_string())?;
            reach_updates.extend(result.reach_updates);
            artifact_delta += result.artifact_delta;
            input_update_micros = input_update_micros.max(result.input_update_micros);
            dataflow_micros = dataflow_micros.max(result.dataflow_micros);
            debug_assert!(result.worker_index < self.senders.len());
        }

        counts.edge_facts += revision
            .edge_updates
            .iter()
            .map(|(_, diff)| diff)
            .sum::<isize>();
        counts.module_facts += revision
            .module_updates
            .iter()
            .map(|(_, diff)| diff)
            .sum::<isize>();
        let reach_delta = consolidate_updates(reach_updates);
        let mut added = BTreeSet::new();
        let mut removed = BTreeSet::new();
        for (module, diff) in reach_delta {
            counts.reachable_facts += diff;
            if diff > 0 {
                added.insert(module);
            } else if diff < 0 {
                removed.insert(module);
            }
        }
        counts.artifact_facts += artifact_delta;
        debug_assert_eq!(counts.artifact_facts, counts.reachable_facts);

        let changed_facts = revision.changed.len();
        Ok(RevisionResult {
            label: revision.label.clone(),
            reachable: BTreeSet::new(),
            added_facts: added.len(),
            removed_facts: removed.len(),
            changed_facts,
            added,
            removed,
            changed: revision.changed.clone(),
            module_facts: usize::try_from(counts.module_facts).expect("negative module fact count"),
            edge_facts: usize::try_from(counts.edge_facts).expect("negative edge fact count"),
            reachable_facts: usize::try_from(counts.reachable_facts)
                .expect("negative reachable fact count"),
            diagnostics: revision.diagnostics.clone(),
            input_update_micros,
            dataflow_micros,
            output_micros: output_started.elapsed().as_micros(),
        })
    }

    pub fn worker_count(&self) -> usize {
        self.senders.len()
    }
}

impl Default for DeltaSession {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DeltaSession {
    fn drop(&mut self) {
        self.senders.clear();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run_session_workers(receivers: Arc<Vec<Mutex<mpsc::Receiver<SessionRequest>>>>, workers: usize) {
    let guards = timely::execute(timely::Config::process(workers), move |worker| {
        let worker_index = worker.index();
        let mut entries = InputSession::<u64, ModuleId, isize>::new();
        let mut edges = InputSession::<u64, Edge, isize>::new();
        let mut modules = InputSession::<u64, ModuleVersion, isize>::new();
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

        let mut time = 0_u64;
        let receiver = receivers[worker_index]
            .lock()
            .expect("session receiver poisoned");
        while let Ok(request) = receiver.recv() {
            let revision = request.revision;
            let input_started = Instant::now();
            if worker_index == 0 {
                apply_update_refs(&mut entries, &revision.entry_updates);
                apply_update_refs(&mut edges, &revision.edge_updates);
                apply_update_refs(&mut modules, &revision.module_updates);
            }
            let input_update_micros = input_started.elapsed().as_micros();

            let dataflow_started = Instant::now();
            time += 1;
            entries.advance_to(time);
            edges.advance_to(time);
            modules.advance_to(time);
            entries.flush();
            edges.flush();
            modules.flush();
            while probe.less_than(&time) {
                worker.step();
            }
            let dataflow_micros = dataflow_started.elapsed().as_micros();

            let reach_delta = consolidate_updates(
                reach_updates
                    .borrow_mut()
                    .drain(..)
                    .map(|(module, _, diff)| (module, diff)),
            );
            let artifact_delta = consolidate_updates(
                artifact_updates
                    .borrow_mut()
                    .drain(..)
                    .map(|(artifact, _, diff)| (artifact, diff)),
            );
            let result = WorkerResult {
                worker_index,
                reach_updates: reach_delta.into_iter().collect(),
                artifact_delta: artifact_delta.values().sum(),
                input_update_micros,
                dataflow_micros,
            };
            if request.response.send(result).is_err() {
                break;
            }
        }
    })
    .expect("cannot start Timely dataflow workers");
    for result in guards.join() {
        result.expect("Timely dataflow worker panicked");
    }
}

fn run_delta_revisions_inner(
    revisions: Vec<DeltaRevision>,
    materialize_manifest: bool,
    capture_output_facts: bool,
) -> Vec<RevisionResult> {
    timely::execute_directly(move |worker| {
        let mut entries = InputSession::<u64, ModuleId, isize>::new();
        let mut edges = InputSession::<u64, Edge, isize>::new();
        let mut modules = InputSession::<u64, ModuleVersion, isize>::new();
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

        let mut module_facts = 0_isize;
        let mut edge_facts = 0_isize;
        let mut reachable_facts = 0_isize;
        let mut artifact_facts = 0_isize;
        let mut manifest = materialize_manifest.then(BTreeSet::new);
        let mut results = Vec::with_capacity(revisions.len());

        for (time, revision) in revisions.into_iter().enumerate() {
            let input_started = Instant::now();
            apply_updates(&mut entries, revision.entry_updates);
            edge_facts += apply_updates(&mut edges, revision.edge_updates);
            module_facts += apply_updates(&mut modules, revision.module_updates);
            let input_update_micros = input_started.elapsed().as_micros();

            let dataflow_started = Instant::now();
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
            let dataflow_micros = dataflow_started.elapsed().as_micros();

            let output_started = Instant::now();
            let reach_delta = consolidate_updates(
                reach_updates
                    .borrow_mut()
                    .drain(..)
                    .map(|(module, _time, diff)| (module, diff)),
            );
            let artifact_delta = consolidate_updates(
                artifact_updates
                    .borrow_mut()
                    .drain(..)
                    .map(|(artifact, _time, diff)| (artifact, diff)),
            );

            let mut added = BTreeSet::new();
            let mut removed = BTreeSet::new();
            let mut added_facts = 0;
            let mut removed_facts = 0;
            for (module, diff) in reach_delta {
                reachable_facts += diff;
                if diff > 0 {
                    added_facts += diff as usize;
                    if capture_output_facts {
                        added.insert(module.clone());
                    }
                    if let Some(manifest) = &mut manifest {
                        manifest.insert(module);
                    }
                } else if diff < 0 {
                    removed_facts += (-diff) as usize;
                    if capture_output_facts {
                        removed.insert(module.clone());
                    }
                    if let Some(manifest) = &mut manifest {
                        manifest.remove(&module);
                    }
                }
            }
            artifact_facts += artifact_delta.values().sum::<isize>();
            debug_assert_eq!(artifact_facts, reachable_facts);

            let reachable = manifest.clone().unwrap_or_default();
            let output_micros = output_started.elapsed().as_micros();
            let changed_facts = revision.changed.len();
            results.push(RevisionResult {
                label: revision.label,
                reachable,
                added,
                removed,
                changed: revision.changed,
                added_facts,
                removed_facts,
                changed_facts,
                module_facts: usize::try_from(module_facts).expect("negative module fact count"),
                edge_facts: usize::try_from(edge_facts).expect("negative edge fact count"),
                reachable_facts: usize::try_from(reachable_facts)
                    .expect("negative reachable fact count"),
                diagnostics: revision.diagnostics,
                input_update_micros,
                dataflow_micros,
                output_micros,
            });
        }
        results
    })
}

fn apply_updates<T: differential_dataflow::Data>(
    input: &mut InputSession<u64, T, isize>,
    updates: Vec<(T, isize)>,
) -> isize {
    let mut fact_delta = 0;
    for (fact, diff) in updates {
        input.update(fact, diff);
        fact_delta += diff;
    }
    fact_delta
}

fn apply_update_refs<T: differential_dataflow::Data>(
    input: &mut InputSession<u64, T, isize>,
    updates: &[(T, isize)],
) {
    for (fact, diff) in updates {
        input.update(fact.clone(), *diff);
    }
}

fn consolidate_updates<T: Ord>(
    updates: impl IntoIterator<Item = (T, isize)>,
) -> BTreeMap<T, isize> {
    let mut consolidated = BTreeMap::new();
    for (fact, diff) in updates {
        *consolidated.entry(fact).or_default() += diff;
    }
    consolidated.retain(|_, diff| *diff != 0);
    consolidated
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

    #[test]
    fn direct_deltas_do_not_materialize_the_manifest() {
        let initial = graph(1, false);
        let results = run_delta_revisions(vec![DeltaRevision::initial("initial", &initial)]);
        assert_eq!(results[0].reachable_facts, 2);
        assert_eq!(results[0].added_facts, 2);
        assert!(results[0].reachable.is_empty());
        assert!(results[0].added.is_empty());
    }

    #[test]
    fn persistent_session_keeps_state_between_calls() {
        let session = DeltaSession::with_workers(4);
        let initial = session
            .apply(DeltaRevision::initial("initial", &graph(1, false)))
            .unwrap();
        assert_eq!(initial.added.len(), 2);

        let next = session
            .apply(DeltaRevision {
                label: "add leaf".into(),
                edge_updates: vec![(("entry.js".into(), "leaf.js".into()), 1)],
                module_updates: vec![(("leaf.js".to_string(), 3), 1)],
                ..DeltaRevision::default()
            })
            .unwrap();
        assert_eq!(next.added, BTreeSet::from(["leaf.js".into()]));
        assert_eq!(next.reachable_facts, 3);
    }
}
