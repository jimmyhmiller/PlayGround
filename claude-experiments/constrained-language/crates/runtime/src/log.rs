//! Append-only event log. Every event, handler invocation, and effect
//! fulfillment is a row. The log is the source of truth for replay,
//! inspector views, and (eventually) counterfactual forks.

use serde::{Deserialize, Serialize};

use crate::value::Value;

pub type LogicalClock = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub t: LogicalClock,
    pub kind: LogEntryKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogEntryKind {
    EventEnqueued {
        event_id: u64,
        event: String,
        payload: Value,
    },
    HandlerInvoked {
        event_id: u64,
        handler: String,
        reads: Value,
        writes: Vec<WriteRecord>,
        emits: Vec<EmitRecord>,
    },
    EffectFulfilled {
        emit_id: u64,
        effect: String,
        result: EffectOutcome,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WriteRecord {
    SetAtom { cell: String, value: Value },
    PutMap { cell: String, key: Value, value: Value },
    DeleteMap { cell: String, key: Value },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmitRecord {
    pub emit_id: u64,
    pub effect: String,
    pub request: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EffectOutcome {
    Ok { response: Value },
    Failed { reason: String },
}

#[derive(Debug, Default, Clone)]
pub struct EventLog {
    entries: Vec<LogEntry>,
    clock: LogicalClock,
}

impl EventLog {
    pub fn append(&mut self, kind: LogEntryKind) -> LogicalClock {
        let t = self.clock;
        self.clock += 1;
        self.entries.push(LogEntry { t, kind });
        t
    }

    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    pub fn now(&self) -> LogicalClock {
        self.clock
    }
}
