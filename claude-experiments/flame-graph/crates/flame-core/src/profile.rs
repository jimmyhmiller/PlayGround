use crate::stacks::{StackId, StackTable};
use crate::strings::{StringId, StringInterner};
use ahash::AHashMap;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ProcessId(pub u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ThreadId(pub u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TrackId(pub u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct CategoryId(pub u32);

impl CategoryId {
    pub const DEFAULT: CategoryId = CategoryId(0);
}

#[derive(Clone, Debug)]
pub struct Process {
    pub pid: i64,
    pub name: StringId,
}

#[derive(Clone, Debug)]
pub struct Thread {
    pub tid: i64,
    pub process: Option<ProcessId>,
    pub name: StringId,
}

#[derive(Clone, Debug)]
pub enum TrackKind {
    /// A regular thread timeline (slices on it represent stack frames).
    Thread(ThreadId),
    /// An async / virtual track grouped under a parent.
    Async,
    /// A counter track (single-row line/area chart).
    Counter,
    /// A global track not bound to any thread.
    Global,
}

#[derive(Clone, Debug)]
pub struct Track {
    pub kind: TrackKind,
    pub name: StringId,
    pub parent: Option<TrackId>,
    pub row_count: u16,
}

#[derive(Clone, Debug)]
pub struct Category {
    pub name: StringId,
    /// Index into the renderer's color palette. `u16::MAX` means "auto from name hash".
    pub color_idx: u16,
}

/// Slices stored as struct-of-arrays. After `ProfileBuilder::finish`, slices are
/// sorted by (track, depth, start_ns) and `rows` indexes contiguous spans for fast
/// viewport culling via two `partition_point` calls.
#[derive(Default, Debug, Clone)]
pub struct SliceTable {
    pub track:    Vec<TrackId>,
    pub depth:    Vec<u16>,
    pub start_ns: Vec<u64>,
    pub dur_ns:   Vec<u64>,
    pub name:     Vec<StringId>,
    pub category: Vec<CategoryId>,
    pub stack:    Vec<Option<StackId>>,
    /// (track, depth) -> contiguous range `[lo, hi)` into the SoA arrays.
    pub rows: AHashMap<(TrackId, u16), std::ops::Range<u32>>,
}

impl SliceTable {
    pub fn len(&self) -> usize {
        self.start_ns.len()
    }
    pub fn is_empty(&self) -> bool {
        self.start_ns.is_empty()
    }

    /// Return the range of slice indices in this row whose interval intersects
    /// `[lo_ns, hi_ns)`. Slices are sorted by `start_ns` within a row.
    pub fn visible_in_row(
        &self,
        track: TrackId,
        depth: u16,
        lo_ns: u64,
        hi_ns: u64,
    ) -> std::ops::Range<u32> {
        let row = match self.rows.get(&(track, depth)) {
            Some(r) => r.clone(),
            None => return 0..0,
        };
        if row.is_empty() {
            return row;
        }
        // Find first slice whose end > lo_ns. End times are not separately sorted, but
        // we can binary-search start_ns for a lower bound and back up — practical and
        // correct since within a parent at the same depth, slices don't overlap (flame
        // graph invariant). For safety we walk back at most a small constant.
        let starts = &self.start_ns[row.start as usize..row.end as usize];
        // First index whose start >= lo_ns.
        let mut first = starts.partition_point(|&s| s < lo_ns);
        // Back up by one if the previous slice still covers lo_ns.
        if first > 0 {
            let prev = first - 1;
            let prev_idx = row.start as usize + prev;
            if self.start_ns[prev_idx] + self.dur_ns[prev_idx] > lo_ns {
                first = prev;
            }
        }
        // First index whose start >= hi_ns.
        let last = starts.partition_point(|&s| s < hi_ns);
        let lo = row.start + first as u32;
        let hi = row.start + last as u32;
        lo..hi
    }
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub thread: ThreadId,
    pub ts_ns: u64,
    pub stack: StackId,
    pub weight: u32,
}

/// Per-slice attribute side-table. Populated by loaders that have free-form
/// span-level metadata (OpenTelemetry `attributes`, Chrome `args`, …); empty
/// otherwise. Indexed by SoA slice index (so it stays valid through the
/// builder's sort).
///
/// Storage shape: keys are interned once into `keys`; each slice's row is a
/// tiny Vec of `(key_index, value_string_id)`. Typical span carries 0–10
/// attrs, so linear lookup is fine.
#[derive(Default, Clone, Debug)]
pub struct AttrTable {
    /// Ordered list of distinct attribute keys observed across the profile.
    pub keys: Vec<StringId>,
    /// `keys.len() == key_lookup.len()`. Maps StringId of key name → index in `keys`.
    pub key_lookup: AHashMap<StringId, u16>,
    /// One entry per slice (same indexing as `SliceTable.start_ns` etc.). Inner
    /// vec is `(key_idx, value_string_id)`. Empty when the slice has no attrs.
    pub per_slice: Vec<Vec<(u16, StringId)>>,
}

impl AttrTable {
    /// Look up an attribute on a slice by key StringId. Returns the value's
    /// interned StringId, or None if the slice doesn't carry that key.
    pub fn get(&self, slice_idx: u32, key: StringId) -> Option<StringId> {
        let key_idx = *self.key_lookup.get(&key)?;
        let row = self.per_slice.get(slice_idx as usize)?;
        row.iter().find(|(k, _)| *k == key_idx).map(|(_, v)| *v)
    }
}

#[derive(Default, Clone)]
pub struct Profile {
    pub strings: StringInterner,
    pub categories: Vec<Category>,
    pub processes: Vec<Process>,
    pub threads: Vec<Thread>,
    pub tracks: Vec<Track>,
    pub stacks: StackTable,
    pub slices: SliceTable,
    pub samples: Vec<Sample>,
    pub attrs: AttrTable,
    pub time_range: (u64, u64),
}

impl Profile {
    pub fn track(&self, id: TrackId) -> &Track {
        &self.tracks[id.0 as usize]
    }
    pub fn thread(&self, id: ThreadId) -> &Thread {
        &self.threads[id.0 as usize]
    }
    pub fn process(&self, id: ProcessId) -> &Process {
        &self.processes[id.0 as usize]
    }
    pub fn category(&self, id: CategoryId) -> &Category {
        &self.categories[id.0 as usize]
    }
    pub fn duration_ns(&self) -> u64 {
        self.time_range.1.saturating_sub(self.time_range.0)
    }
}
