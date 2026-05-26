//! State accumulator + `Profile` snapshot builder.
//!
//! Lifetime model:
//! - One `LiveAggregator` per live session.
//! - Reader thread holds a `&mut` (via a mutex on the outside) and calls
//!   `apply` for every decoded event. Cheap: a few hash inserts.
//! - Render side calls `snapshot()` to materialize a fresh `flame_core::Profile`
//!   built from the current state. Snapshot is read-only on `&self` so the
//!   reader can keep ingesting while we build.

use std::collections::HashMap;
use std::sync::Arc;

use ahash::AHashMap;
use flame_core::{
    Profile, ProfileBuilder, TrackKind,
};

use crate::event::{LiveEvent, LiveFrame, LiveFrameKind};
use crate::symbols::SymbolStore;

#[derive(Clone, Debug)]
struct LibRange {
    base: u64,
    end: u64,
    name: String,
    /// On-disk path samply gave us. Used as the key into `SymbolStore` so the
    /// loader thread can mmap + parse the binary's symbol table.
    path: String,
}

#[derive(Clone, Debug)]
struct ThreadInfo {
    pid: u32,
    tid: u32,
    name: Option<String>,
    is_main: bool,
    alive: bool,
}

#[derive(Clone, Debug)]
struct StackNode {
    parent: Option<u32>,
    frame: LiveFrame,
}

#[derive(Clone, Debug)]
struct LiveSample {
    pid: u32,
    tid: u32,
    stack_id: u32,
    /// Mach-absolute-time-derived nanoseconds, as samply emits. Used to lay
    /// out slices on the real wall-clock X axis — without this, new threads
    /// that arrived mid-recording would render at t=0 of the trace.
    timestamp_ns: u64,
}

pub struct LiveAggregator {
    /// Optional symbol store. When present, `frame_label` looks up function
    /// names here before falling back to `lib_name+0xRVA`. Constructed via
    /// `LiveAggregator::with_symbols`; `LiveAggregator::default()` runs
    /// label-only.
    symbols: Option<Arc<SymbolStore>>,
    /// Per-pid lib mappings. Sorted ascending by `base` (we insert by binary
    /// search and shift). Tiny per process (usually 100–500 entries), so the
    /// shift cost is fine.
    libs: HashMap<u32, Vec<LibRange>>,
    /// Process display name. We don't get one from the stream — derive from
    /// the main-thread name if available, else `pid <n>`.
    process_name: HashMap<u32, String>,
    threads: HashMap<(u32, u32), ThreadInfo>,
    /// Tree node id → (parent, frame). StackDef events monotonically grow this.
    stacks: HashMap<u32, StackNode>,
    /// Append-only samples. Drives the flame-graph aggregation in `snapshot`.
    samples: Vec<LiveSample>,
    time_start_ns: Option<u64>,
    time_end_ns: u64,
    /// Bumped on any state-changing event so the render side can avoid
    /// rebuilding when nothing has changed. Combine with `symbols_version()`
    /// to also catch background symbol arrivals.
    pub version: u64,
    /// Wall-clock instant of the most recently applied event. Used for
    /// end-to-end latency measurement: subtract it from `Instant::now()` at
    /// snapshot-build time to see how stale the freshest sample is.
    pub last_apply_instant: Option<std::time::Instant>,
}

impl LiveAggregator {
    /// Combined version that reflects both event ingestion and background
    /// symbol loading. The builder thread uses this so a snapshot rebuilds
    /// when new function names land — not just when new samples do.
    pub fn combined_version(&self) -> u64 {
        let sym_v = self
            .symbols
            .as_ref()
            .map(|s| s.version())
            .unwrap_or(0);
        self.version.wrapping_add(sym_v)
    }
}

impl Default for LiveAggregator {
    fn default() -> Self {
        Self {
            symbols: None,
            libs: HashMap::new(),
            process_name: HashMap::new(),
            threads: HashMap::new(),
            stacks: HashMap::new(),
            samples: Vec::new(),
            time_start_ns: None,
            time_end_ns: 0,
            version: 0,
            last_apply_instant: None,
        }
    }
}

impl LiveAggregator {
    pub fn with_symbols(symbols: Arc<SymbolStore>) -> Self {
        let mut a = Self::default();
        a.symbols = Some(symbols);
        a
    }

    pub fn apply(&mut self, event: LiveEvent) {
        self.version = self.version.wrapping_add(1);
        self.last_apply_instant = Some(std::time::Instant::now());
        match event {
            LiveEvent::Thread {
                pid,
                tid,
                name,
                is_main,
                timestamp_ns,
            } => {
                self.observe_ts(timestamp_ns);
                let key = (pid, tid);
                let info = self.threads.entry(key).or_insert_with(|| ThreadInfo {
                    pid,
                    tid,
                    name: None,
                    is_main: false,
                    alive: true,
                });
                let _ = timestamp_ns; // observed above; not stored per-thread
                if let Some(n) = name.clone() {
                    info.name = Some(n.clone());
                    if is_main {
                        self.process_name.entry(pid).or_insert(n);
                    }
                }
                info.is_main = info.is_main || is_main;
                info.alive = true;
            }
            LiveEvent::ThreadEnd {
                pid,
                tid,
                timestamp_ns,
            } => {
                self.observe_ts(timestamp_ns);
                if let Some(info) = self.threads.get_mut(&(pid, tid)) {
                    info.alive = false;
                }
            }
            LiveEvent::LibMapping {
                pid,
                base_avma,
                end_avma,
                name,
                path,
                timestamp_ns,
                ..
            } => {
                self.observe_ts(timestamp_ns);
                if let Some(syms) = &self.symbols {
                    syms.request(&path);
                }
                let v = self.libs.entry(pid).or_default();
                let range = LibRange {
                    base: base_avma,
                    end: end_avma,
                    name,
                    path,
                };
                let pos = v.partition_point(|r| r.base < range.base);
                v.insert(pos, range);
            }
            LiveEvent::LibUnmap {
                pid,
                base_avma,
                timestamp_ns,
            } => {
                self.observe_ts(timestamp_ns);
                if let Some(v) = self.libs.get_mut(&pid) {
                    v.retain(|r| r.base != base_avma);
                }
            }
            LiveEvent::StackDef {
                id,
                parent_id,
                frame,
            } => {
                self.stacks.insert(
                    id,
                    StackNode {
                        parent: parent_id,
                        frame,
                    },
                );
            }
            LiveEvent::Sample {
                pid,
                tid,
                timestamp_ns,
                stack_id,
                ..
            } => {
                self.observe_ts(timestamp_ns);
                if let Some(stack_id) = stack_id {
                    self.samples.push(LiveSample {
                        pid,
                        tid,
                        stack_id,
                        timestamp_ns,
                    });
                }
            }
        }
    }

    fn observe_ts(&mut self, ts: u64) {
        if ts == 0 {
            return;
        }
        match self.time_start_ns {
            None => self.time_start_ns = Some(ts),
            Some(s) if ts < s => self.time_start_ns = Some(ts),
            _ => {}
        }
        if ts > self.time_end_ns {
            self.time_end_ns = ts;
        }
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn thread_count(&self) -> usize {
        self.threads.len()
    }

    /// Resolve an address to a human label using the lib map for `pid`. Frames
    /// of kind `ReturnAddress` get the documented -1 byte adjustment before
    /// lookup so they land back inside the calling instruction. The returned
    /// label is `"lib_name+0xRVA"` if the address is inside a known mapping,
    /// else the raw hex address.
    fn frame_label(&self, pid: u32, frame: &LiveFrame) -> String {
        let lookup = match frame.kind {
            LiveFrameKind::InstructionPointer => frame.addr,
            LiveFrameKind::ReturnAddress => frame.addr.saturating_sub(1),
        };
        if let Some(libs) = self.libs.get(&pid) {
            // Largest base ≤ lookup.
            let idx = libs.partition_point(|r| r.base <= lookup);
            if idx > 0 {
                let r = &libs[idx - 1];
                if lookup < r.end {
                    let rva = lookup - r.base;
                    // Try symbol table first (background-loaded). Falls back
                    // to lib+0xRVA if symbols haven't arrived yet for this
                    // lib, or there's no symbol covering this RVA.
                    if let Some(syms) = &self.symbols {
                        if let Some(name) = syms.lookup(&r.path, rva as u32) {
                            return name;
                        }
                    }
                    return format!("{}+0x{:x}", r.name, rva);
                }
            }
        }
        format!("0x{:x}", frame.addr)
    }

    /// Build a `Profile` reflecting current state. Each thread gets its own
    /// track; samples for that thread are aggregated into a flame-graph tree.
    pub fn snapshot(&self) -> Profile {
        let mut b = ProfileBuilder::new();
        let cat = b.intern_category("samply-live");

        // Stable process iteration order.
        let mut pids: Vec<u32> = self
            .threads
            .keys()
            .map(|(p, _)| *p)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        pids.sort_unstable();

        // pid → ProcessId in the builder
        let mut pid_to_proc = AHashMap::new();
        for &pid in &pids {
            let name = self
                .process_name
                .get(&pid)
                .cloned()
                .unwrap_or_else(|| format!("pid {pid}"));
            let pidv = b.add_process(pid as i64, &name);
            pid_to_proc.insert(pid, pidv);
        }

        // (pid, tid) → TrackId
        let mut thread_keys: Vec<(u32, u32)> = self.threads.keys().copied().collect();
        thread_keys.sort_unstable();
        let mut thread_tracks: AHashMap<(u32, u32), flame_core::TrackId> = AHashMap::new();
        for &key in &thread_keys {
            let info = &self.threads[&key];
            let proc_id = pid_to_proc.get(&info.pid).copied();
            let label = match (info.is_main, info.name.as_deref()) {
                (true, Some(n)) => format!("{n} (main, tid {})", info.tid),
                (true, None) => format!("main (tid {})", info.tid),
                (false, Some(n)) => format!("{n} (tid {})", info.tid),
                (false, None) => format!("tid {}", info.tid),
            };
            let thread_id = b.add_thread(proc_id, info.tid as i64, &label);
            let track = b.add_track(TrackKind::Thread(thread_id), &label, None);
            thread_tracks.insert(key, track);
        }

        // Partition samples by thread, sorted by wall-clock timestamp.
        // This is what makes the X axis represent real time: a thread that
        // appeared mid-recording has its slices start at the timestamp of
        // its first sample, not at t=0 of the trace.
        let mut per_thread: AHashMap<(u32, u32), Vec<&LiveSample>> = AHashMap::new();
        for s in &self.samples {
            per_thread.entry((s.pid, s.tid)).or_default().push(s);
        }
        for samples in per_thread.values_mut() {
            samples.sort_unstable_by_key(|s| s.timestamp_ns);
        }

        // Cache frame labels: (pid, stack_node_id) → interned StringId. Stack
        // nodes are pid-agnostic in the wire (one global tree), but the same
        // address resolves differently per pid against its libs, so we key
        // by (pid, node_id).
        let mut label_cache: AHashMap<(u32, u32), flame_core::StringId> = AHashMap::new();
        let mut frame_cache: AHashMap<flame_core::StringId, flame_core::FrameId> = AHashMap::new();
        let empty_file = b.intern_string("");

        // Time-ordered slice emission via the longest-common-prefix
        // merging algorithm. Adjacent samples on the same thread that
        // share a stack prefix become one wide slice for that prefix
        // (rather than N skinny ones), and stack frames close out at the
        // moment the next sample's stack diverges from them.
        //
        // We extend the very last slice on each thread by
        // `LIVE_SAMPLE_TAIL_NS` so even a single-sample thread has a
        // visible width on screen.
        const LIVE_SAMPLE_TAIL_NS: u64 = 1_000_000; // samply default 1000Hz

        for ((pid, tid), samples) in per_thread {
            let track = match thread_tracks.get(&(pid, tid)) {
                Some(t) => *t,
                None => continue,
            };
            let thread_id = match &b.tracks[track.0 as usize].kind {
                TrackKind::Thread(t) => *t,
                _ => continue,
            };

            // Open frames: index = depth. (start_ns, name_StringId, FrameId).
            type Open = (u64, flame_core::StringId, flame_core::FrameId);
            let mut open: Vec<Open> = Vec::new();
            let mut last_ts: u64 = 0;

            for sample in samples {
                let cur_ts = sample.timestamp_ns;
                last_ts = cur_ts;

                // Walk innermost → root, then reverse so chain is root-first.
                let mut chain: Vec<(flame_core::StringId, flame_core::FrameId)> = Vec::new();
                let mut cur = Some(sample.stack_id);
                while let Some(id) = cur {
                    let Some(node) = self.stacks.get(&id) else {
                        break;
                    };
                    let sid = *label_cache
                        .entry((pid, id))
                        .or_insert_with(|| {
                            let label = self.frame_label(pid, &node.frame);
                            b.intern_string(&label)
                        });
                    let fid = *frame_cache.entry(sid).or_insert_with(|| {
                        b.stacks.intern_frame(sid, empty_file, 0, node.frame.addr)
                    });
                    chain.push((sid, fid));
                    cur = node.parent;
                }
                chain.reverse();

                // Build the StackId chain for `add_sample` (used by the
                // renderer's call-tree aggregation; without this the
                // CallTree tab stays empty for live profiles).
                let mut stack_id: Option<flame_core::StackId> = None;
                for &(_, fid) in &chain {
                    stack_id = Some(b.intern_stack(fid, stack_id));
                }
                if let Some(sid) = stack_id {
                    b.add_sample(thread_id, cur_ts, sid, 1);
                }

                // Longest common prefix between the currently-open stack
                // and the new sample's stack.
                let mut lcp = 0;
                while lcp < open.len() && lcp < chain.len() && open[lcp].1 == chain[lcp].0 {
                    lcp += 1;
                }
                // Close frames above LCP — their lifetime ends at cur_ts.
                while open.len() > lcp {
                    let (start_ts, name, _fid) = open.pop().unwrap();
                    let depth = open.len() as u16;
                    let dur = cur_ts.saturating_sub(start_ts);
                    b.add_complete_slice(track, depth, start_ts, dur, name, cat, None);
                }
                // Open new frames from LCP downward.
                for &(name, fid) in &chain[lcp..] {
                    open.push((cur_ts, name, fid));
                    let _ = fid;
                }
            }

            // Whatever is still open at the last sample: render it as
            // extending one sampling interval past, so even a thread with
            // a single sample shows a visible slice.
            let end_ts = last_ts.saturating_add(LIVE_SAMPLE_TAIL_NS);
            while !open.is_empty() {
                let (start_ts, name, _fid) = open.pop().unwrap();
                let depth = open.len() as u16;
                let dur = end_ts.saturating_sub(start_ts);
                b.add_complete_slice(track, depth, start_ts, dur, name, cat, None);
            }
        }

        b.finish()
    }
}
