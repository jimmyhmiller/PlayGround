use ahash::AHashMap;

use crate::profile::{
    AttrTable, Category, CategoryId, Process, ProcessId, Profile, Sample, SliceTable, Thread,
    ThreadId, Track, TrackId, TrackKind,
};
use crate::stacks::{FrameId, StackId, StackTable};
use crate::strings::{StringId, StringInterner};

/// Mutable builder a `TraceSource` writes into. Call `finish()` to get a sorted,
/// indexed `Profile` ready for rendering.
pub struct ProfileBuilder {
    pub strings: StringInterner,
    pub categories: Vec<Category>,
    pub processes: Vec<Process>,
    pub threads: Vec<Thread>,
    pub tracks: Vec<Track>,
    pub stacks: StackTable,
    pub samples: Vec<Sample>,

    // Unsorted slice records — finish() sorts and partitions them into the SoA SliceTable.
    pending_slices: Vec<PendingSlice>,
    /// One entry per pending_slice, parallel index. Inner Vec is the (key_idx,
    /// value_string_id) list for that slice, or empty if it has no attrs.
    /// The permutation from finish()'s sort is applied to both vectors together.
    pending_attrs: Vec<Vec<(u16, StringId)>>,
    /// Stack of open B-events keyed by track. Used by `begin_slice` / `end_slice`.
    open_stacks: AHashMap<TrackId, Vec<OpenSlice>>,
    /// Min/max timestamp seen, used to populate Profile::time_range.
    min_ts: Option<u64>,
    max_ts: u64,

    /// Dedup of (process, thread) -> ThreadId, so format crates can call add_thread idempotently.
    thread_dedup: AHashMap<(Option<ProcessId>, i64), ThreadId>,
    process_dedup: AHashMap<i64, ProcessId>,
    category_dedup: AHashMap<StringId, CategoryId>,

    /// Attribute keys observed so far. attr_idx is the position in this Vec.
    attr_keys: Vec<StringId>,
    attr_key_lookup: AHashMap<StringId, u16>,
}

#[derive(Clone, Debug)]
struct PendingSlice {
    track: TrackId,
    depth: u16,
    start_ns: u64,
    dur_ns: u64,
    name: StringId,
    category: CategoryId,
    stack: Option<StackId>,
}

#[derive(Clone, Debug)]
struct OpenSlice {
    start_ns: u64,
    name: StringId,
    category: CategoryId,
}

impl Default for ProfileBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileBuilder {
    pub fn new() -> Self {
        let mut strings = StringInterner::new();
        let default_cat_name = strings.intern("default");
        let categories = vec![Category {
            name: default_cat_name,
            color_idx: u16::MAX,
        }];

        Self {
            strings,
            categories,
            processes: Vec::new(),
            threads: Vec::new(),
            tracks: Vec::new(),
            stacks: StackTable::new(),
            samples: Vec::new(),
            pending_slices: Vec::new(),
            pending_attrs: Vec::new(),
            open_stacks: AHashMap::new(),
            min_ts: None,
            max_ts: 0,
            thread_dedup: AHashMap::new(),
            process_dedup: AHashMap::new(),
            category_dedup: AHashMap::new(),
            attr_keys: Vec::new(),
            attr_key_lookup: AHashMap::new(),
        }
    }

    /// Intern an attribute key. Returns a compact `u16` index. Caller passes
    /// this index to `add_complete_slice_with_attrs` instead of re-hashing the
    /// key string per slice.
    pub fn intern_attr_key(&mut self, key: &str) -> u16 {
        let sid = self.strings.intern(key);
        if let Some(&i) = self.attr_key_lookup.get(&sid) {
            return i;
        }
        let i = self.attr_keys.len() as u16;
        self.attr_keys.push(sid);
        self.attr_key_lookup.insert(sid, i);
        i
    }

    pub fn intern_string(&mut self, s: &str) -> StringId {
        self.strings.intern(s)
    }

    pub fn intern_category(&mut self, name: &str) -> CategoryId {
        let id = self.strings.intern(name);
        if let Some(&c) = self.category_dedup.get(&id) {
            return c;
        }
        let cat = CategoryId(self.categories.len() as u32);
        self.categories.push(Category { name: id, color_idx: u16::MAX });
        self.category_dedup.insert(id, cat);
        cat
    }

    pub fn add_process(&mut self, pid: i64, name: &str) -> ProcessId {
        if let Some(&existing) = self.process_dedup.get(&pid) {
            // Update name if it was empty.
            if self.processes[existing.0 as usize].name == StringId::EMPTY && !name.is_empty() {
                self.processes[existing.0 as usize].name = self.strings.intern(name);
            }
            return existing;
        }
        let name_id = self.strings.intern(name);
        let id = ProcessId(self.processes.len() as u32);
        self.processes.push(Process { pid, name: name_id });
        self.process_dedup.insert(pid, id);
        id
    }

    pub fn add_thread(&mut self, process: Option<ProcessId>, tid: i64, name: &str) -> ThreadId {
        let key = (process, tid);
        if let Some(&existing) = self.thread_dedup.get(&key) {
            if self.threads[existing.0 as usize].name == StringId::EMPTY && !name.is_empty() {
                self.threads[existing.0 as usize].name = self.strings.intern(name);
            }
            return existing;
        }
        let name_id = self.strings.intern(name);
        let id = ThreadId(self.threads.len() as u32);
        self.threads.push(Thread { tid, process, name: name_id });
        self.thread_dedup.insert(key, id);
        id
    }

    pub fn add_track(&mut self, kind: TrackKind, name: &str, parent: Option<TrackId>) -> TrackId {
        let name_id = self.strings.intern(name);
        let id = TrackId(self.tracks.len() as u32);
        self.tracks.push(Track { kind, name: name_id, parent, row_count: 0 });
        id
    }

    pub fn intern_frame(&mut self, name: &str, file: &str, line: u32, addr: u64) -> FrameId {
        let n = self.strings.intern(name);
        let f = self.strings.intern(file);
        self.stacks.intern_frame(n, f, line, addr)
    }

    /// The interned name of a previously-interned frame. Used by format crates
    /// to feed `begin_slice` / `add_complete_slice` without a fresh `intern_string`.
    pub fn stacks_frame_name(&self, frame: FrameId) -> StringId {
        self.stacks.frame(frame).name
    }

    pub fn intern_stack(&mut self, frame: FrameId, parent: Option<StackId>) -> StackId {
        self.stacks.intern_stack(frame, parent)
    }

    /// Track-level pair: pushes onto an open-slice stack. Returns the depth the slice will be at.
    pub fn begin_slice(
        &mut self,
        track: TrackId,
        start_ns: u64,
        name: StringId,
        category: CategoryId,
    ) -> u16 {
        self.observe_ts(start_ns);
        let stack = self.open_stacks.entry(track).or_default();
        let depth = stack.len() as u16;
        stack.push(OpenSlice { start_ns, name, category });
        depth
    }

    /// Pops the most recent open slice on the track and emits a finished slice. Mismatched
    /// `end_slice` calls (no open slice) are tolerated and dropped with a warning.
    pub fn end_slice(&mut self, track: TrackId, end_ns: u64) {
        self.observe_ts(end_ns);
        let Some(stack) = self.open_stacks.get_mut(&track) else { return };
        let Some(open) = stack.pop() else { return };
        let depth = stack.len() as u16; // depth after pop = position the closing slice occupied
        let dur = end_ns.saturating_sub(open.start_ns);
        self.pending_slices.push(PendingSlice {
            track,
            depth,
            start_ns: open.start_ns,
            dur_ns: dur,
            name: open.name,
            category: open.category,
            stack: None,
        });
        self.pending_attrs.push(Vec::new());
    }

    /// Direct emit of a complete slice (Chrome 'X' phase, synthesized samples, etc.).
    pub fn add_complete_slice(
        &mut self,
        track: TrackId,
        depth: u16,
        start_ns: u64,
        dur_ns: u64,
        name: StringId,
        category: CategoryId,
        stack: Option<StackId>,
    ) {
        self.add_complete_slice_with_attrs(
            track, depth, start_ns, dur_ns, name, category, stack, Vec::new(),
        );
    }

    /// Same as `add_complete_slice` but with a per-span attribute list. Each
    /// entry is `(attr_key_idx, value_string_id)`; the key index must come from
    /// `intern_attr_key`. Pass an empty Vec to skip attrs for this slice.
    pub fn add_complete_slice_with_attrs(
        &mut self,
        track: TrackId,
        depth: u16,
        start_ns: u64,
        dur_ns: u64,
        name: StringId,
        category: CategoryId,
        stack: Option<StackId>,
        attrs: Vec<(u16, StringId)>,
    ) {
        self.observe_ts(start_ns);
        self.observe_ts(start_ns + dur_ns);
        self.pending_slices.push(PendingSlice {
            track, depth, start_ns, dur_ns, name, category, stack,
        });
        self.pending_attrs.push(attrs);
    }

    pub fn add_sample(&mut self, thread: ThreadId, ts_ns: u64, stack: StackId, weight: u32) {
        self.observe_ts(ts_ns);
        self.samples.push(Sample { thread, ts_ns, stack, weight });
    }

    /// Number of currently-open B-slices on `track`. Used by Chrome `X`/`i` events
    /// to emit at the right depth without disturbing the open stack.
    pub fn open_stack_depth(&self, track: TrackId) -> u16 {
        self.open_stacks
            .get(&track)
            .map(|v| v.len() as u16)
            .unwrap_or(0)
    }

    /// Close any still-open B slices at the largest timestamp seen so far. Useful
    /// for malformed inputs that don't emit a closing E for every B.
    pub fn close_open_slices_at_max(&mut self) {
        let end = self.max_ts;
        let tracks: Vec<TrackId> = self.open_stacks.keys().copied().collect();
        for t in tracks {
            while self.open_stacks.get(&t).map(|s| !s.is_empty()).unwrap_or(false) {
                self.end_slice(t, end);
            }
        }
    }

    /// For sample-only formats (folded, pprof). Lays out all samples on a single track
    /// as a flame graph: each (thread, stack-prefix) becomes a slice whose width is the
    /// summed sample weight underneath it. The x-axis is synthetic (one unit per weight).
    pub fn synthesize_slices_from_samples(&mut self, track: TrackId, category: CategoryId) {
        if self.samples.is_empty() {
            return;
        }

        struct Node {
            frame: FrameId,
            weight: u64,
            children: Vec<u32>, // indices into `nodes`
        }
        let mut nodes: Vec<Node> = Vec::new();
        let mut by_parent_frame: AHashMap<(Option<u32>, FrameId), u32> = AHashMap::new();

        for s in &self.samples {
            let frames = self.stacks.frames_root_first(s.stack);
            let mut parent: Option<u32> = None;
            for &fid in &frames {
                let key = (parent, fid);
                let idx = match by_parent_frame.get(&key) {
                    Some(&i) => i,
                    None => {
                        let i = nodes.len() as u32;
                        nodes.push(Node { frame: fid, weight: 0, children: Vec::new() });
                        if let Some(p) = parent {
                            nodes[p as usize].children.push(i);
                        }
                        by_parent_frame.insert(key, i);
                        i
                    }
                };
                nodes[idx as usize].weight += s.weight as u64;
                parent = Some(idx);
            }
        }

        // Roots = nodes whose key has parent == None.
        let mut roots: Vec<u32> = by_parent_frame
            .iter()
            .filter(|((p, _), _)| p.is_none())
            .map(|(_, &v)| v)
            .collect();
        let weight_of = |i: u32| -> u64 { nodes[i as usize].weight };
        let frame_of = |i: u32| -> FrameId { nodes[i as usize].frame };
        roots.sort_by(|&a, &b| weight_of(b).cmp(&weight_of(a)).then(frame_of(a).0.cmp(&frame_of(b).0)));

        // Iterative DFS: stack of (node_idx, depth, x_offset).
        let mut x: u64 = 0;
        let mut stack: Vec<(u32, u16, u64)> = roots.iter().rev().map(|&r| (r, 0, 0)).collect();
        // Pre-position roots left-to-right.
        let mut cur_root_x: u64 = 0;
        stack.clear();
        for &r in roots.iter().rev() {
            let w = nodes[r as usize].weight;
            stack.push((r, 0, cur_root_x));
            cur_root_x += w;
        }

        while let Some((idx, depth, sx)) = stack.pop() {
            let n = &nodes[idx as usize];
            let name_id = self.stacks.frame(n.frame).name;
            self.pending_slices.push(PendingSlice {
                track,
                depth,
                start_ns: sx,
                dur_ns: n.weight,
                name: name_id,
                category,
                stack: None,
            });
            self.pending_attrs.push(Vec::new());
            // Sort children for deterministic layout, then push in reverse so leftmost
            // is processed first (preserves left-to-right visual order regardless of
            // pop order).
            let mut kids: Vec<u32> = n.children.clone();
            kids.sort_by(|&a, &b| {
                weight_of(b)
                    .cmp(&weight_of(a))
                    .then(frame_of(a).0.cmp(&frame_of(b).0))
            });
            let mut cx = sx;
            // Compute child x positions in forward order, then push in reverse.
            let positions: Vec<(u32, u64)> = kids
                .iter()
                .map(|&k| {
                    let pos = cx;
                    cx += weight_of(k);
                    (k, pos)
                })
                .collect();
            for &(k, pos) in positions.iter().rev() {
                stack.push((k, depth + 1, pos));
            }
            x = x.max(sx + n.weight);
        }

        if x > 0 {
            self.observe_ts(0);
            self.observe_ts(x);
        }
    }

    fn observe_ts(&mut self, ts: u64) {
        match self.min_ts {
            None => self.min_ts = Some(ts),
            Some(m) if ts < m => self.min_ts = Some(ts),
            _ => {}
        }
        if ts > self.max_ts {
            self.max_ts = ts;
        }
    }

    pub fn finish(mut self) -> Profile {
        let n = self.pending_slices.len();
        debug_assert_eq!(self.pending_attrs.len(), n);

        // Sort an index vector by (track, depth, start_ns) so the same
        // permutation can reorder both pending_slices and pending_attrs.
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&i| {
            let s = &self.pending_slices[i as usize];
            (s.track.0, s.depth, s.start_ns)
        });

        let mut slices = SliceTable {
            track:    Vec::with_capacity(n),
            depth:    Vec::with_capacity(n),
            start_ns: Vec::with_capacity(n),
            dur_ns:   Vec::with_capacity(n),
            name:     Vec::with_capacity(n),
            category: Vec::with_capacity(n),
            stack:    Vec::with_capacity(n),
            rows:     AHashMap::new(),
        };
        let mut attrs_per_slice: Vec<Vec<(u16, StringId)>> = Vec::with_capacity(n);

        // Compute per-(track, depth) row ranges and per-track max depth.
        let mut row_start: u32 = 0;
        let mut cur_key: Option<(TrackId, u16)> = None;
        for (out_i, &src) in order.iter().enumerate() {
            let src = src as usize;
            let s = &self.pending_slices[src];
            let key = (s.track, s.depth);
            if Some(key) != cur_key {
                if let Some(prev) = cur_key {
                    slices.rows.insert(prev, row_start..out_i as u32);
                }
                row_start = out_i as u32;
                cur_key = Some(key);
            }
            slices.track.push(s.track);
            slices.depth.push(s.depth);
            slices.start_ns.push(s.start_ns);
            slices.dur_ns.push(s.dur_ns);
            slices.name.push(s.name);
            slices.category.push(s.category);
            slices.stack.push(s.stack);
            attrs_per_slice.push(std::mem::take(&mut self.pending_attrs[src]));
        }
        if let Some(prev) = cur_key {
            slices.rows.insert(prev, row_start..n as u32);
        }

        // Update each track's row_count.
        for (i, t) in self.tracks.iter_mut().enumerate() {
            let tid = TrackId(i as u32);
            let max_depth = slices
                .rows
                .keys()
                .filter(|(t, _)| *t == tid)
                .map(|(_, d)| *d)
                .max();
            t.row_count = max_depth.map(|d| d + 1).unwrap_or(0);
        }

        let attrs = AttrTable {
            keys: self.attr_keys,
            key_lookup: self.attr_key_lookup,
            per_slice: attrs_per_slice,
        };

        Profile {
            strings: self.strings,
            categories: self.categories,
            processes: self.processes,
            threads: self.threads,
            tracks: self.tracks,
            stacks: self.stacks,
            slices,
            samples: self.samples,
            attrs,
            time_range: (self.min_ts.unwrap_or(0), self.max_ts),
        }
    }
}
