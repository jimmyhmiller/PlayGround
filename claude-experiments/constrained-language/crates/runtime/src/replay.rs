//! Replay tools.
//!
//! The event log captures every write produced by every handler invocation.
//! That alone is enough to reconstruct state from scratch — no bodies need
//! to be run, no effects re-fulfilled. This module implements that and
//! (below) a body-driven replay that re-runs handlers against recorded
//! effect responses for the determinism check.

use std::collections::HashMap;

use crate::effect::{Adapter, AdapterResult};
use crate::log::{EffectOutcome, EventLog, LogEntryKind, WriteRecord};
use crate::state::StateStore;
use crate::value::Value;

/// Re-apply every write recorded in the log to `state`, in order. Returns
/// the number of write operations applied.
///
/// This is the fastest form of recovery: starting from an empty state
/// store seeded with the manifest's cells, calling this brings it to the
/// final state of the original run.
pub fn apply_writes_from_log(state: &mut StateStore, log: &EventLog) -> usize {
    let mut applied = 0;
    for entry in log.entries() {
        if let LogEntryKind::HandlerInvoked { writes, .. } = &entry.kind {
            for w in writes {
                match w {
                    WriteRecord::SetAtom { cell, value } => {
                        state.set_atom(cell, value.clone());
                    }
                    WriteRecord::PutMap { cell, key, value } => {
                        state.put_map_entry(cell, key.clone(), value.clone());
                    }
                    WriteRecord::DeleteMap { cell, key } => {
                        state.delete_map_entry(cell, key);
                    }
                }
                applied += 1;
            }
        }
    }
    applied
}

/// An adapter that replays recorded effect outcomes from a log, keyed by
/// `emit_id`. Used to drive a fresh runtime through the same sequence of
/// effect responses without touching the outside world.
///
/// Construct one with [`ReplayAdapter::from_log`] (which builds the index
/// for every effect), or directly with the per-effect index.
pub struct ReplayAdapter {
    /// Pre-recorded outcomes keyed by emit_id. The adapter pops the matching
    /// outcome as it sees emits; if a body is non-deterministic and emits in
    /// a different order, replay will surface a missing-id error.
    outcomes: HashMap<u64, EffectOutcome>,
}

impl ReplayAdapter {
    /// Build an adapter that knows the recorded outcomes for one effect type
    /// (matches by `effect` name).
    pub fn from_log(log: &EventLog, effect_name: &str) -> Self {
        let mut outcomes = HashMap::new();
        for entry in log.entries() {
            if let LogEntryKind::EffectFulfilled {
                emit_id,
                effect,
                result,
            } = &entry.kind
            {
                if effect == effect_name {
                    outcomes.insert(*emit_id, result.clone());
                }
            }
        }
        Self { outcomes }
    }

    /// Number of recorded outcomes available for replay.
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }
}

impl Adapter for ReplayAdapter {
    fn fulfill(&mut self, _request: Value, emit_id: u64) -> AdapterResult {
        match self.outcomes.remove(&emit_id) {
            Some(EffectOutcome::Ok { response }) => AdapterResult::Ok(response),
            Some(EffectOutcome::Failed { reason }) => AdapterResult::Failed { reason },
            None => AdapterResult::Failed {
                reason: format!(
                    "replay: no recorded outcome for emit_id {emit_id} \
                     (body emitted in a different order than the original run?)"
                ),
            },
        }
    }
}
