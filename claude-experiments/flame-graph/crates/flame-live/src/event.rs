//! Schema mirror of samply's `LiveEvent`. Must stay byte-compatible with
//! `samply/src/shared/live_profile.rs` — postcard's externally-tagged enum
//! encoding is sensitive to variant order, so do not reorder.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum LiveEvent {
    Thread {
        pid: u32,
        tid: u32,
        name: Option<String>,
        is_main: bool,
        timestamp_ns: u64,
    },
    ThreadEnd {
        pid: u32,
        tid: u32,
        timestamp_ns: u64,
    },
    LibMapping {
        pid: u32,
        base_avma: u64,
        end_avma: u64,
        name: String,
        path: String,
        debug_id: Option<String>,
        code_id: Option<String>,
        timestamp_ns: u64,
    },
    LibUnmap {
        pid: u32,
        base_avma: u64,
        timestamp_ns: u64,
    },
    StackDef {
        id: u32,
        parent_id: Option<u32>,
        frame: LiveFrame,
    },
    Sample {
        pid: u32,
        tid: u32,
        timestamp_ns: u64,
        cpu_delta_us: u64,
        stack_id: Option<u32>,
    },
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LiveFrame {
    pub addr: u64,
    pub kind: LiveFrameKind,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LiveFrameKind {
    InstructionPointer,
    ReturnAddress,
}
