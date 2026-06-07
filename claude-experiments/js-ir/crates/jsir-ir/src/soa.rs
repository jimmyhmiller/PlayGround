//! Struct-of-arrays storage for JSIR — the data-oriented backend behind
//! [`IrRead`]/[`IrBuild`]. See `docs/IR_STORAGE.md` for the design.
//!
//! The whole module is a handful of parallel columns indexed by [`OpId`] plus a
//! few flat pools; all structure is integer ranges into those pools, never
//! nested `Vec`s. Opcodes are interned to a 4-byte [`OpKind`]; locations/trivia
//! live in cold columns the printer never touches.

use crate::traits::{BlockSlot, IrBuild, IrRead, OpId, OpKind, RegionId};
use crate::{Attr, Block, BlockId, Op, Region, Successor, Trivia, ValueId};

/// A half-open slice `[start, start+len)` into a pool. 8 bytes, no heap.
#[derive(Clone, Copy, Default)]
struct Range32 {
    start: u32,
    len: u32,
}

impl Range32 {
    fn as_range(self) -> std::ops::Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }
}

/// A `(offset, len)` string handle. The high bit of `off` selects the backing
/// buffer: set ⇒ a span into the owned **source** copy (zero-copy: the parser
/// stores byte offsets, never copying the substring); clear ⇒ a span into the
/// **arena** (`str_arena`, used for attr keys and `from_op`'s owned strings).
#[derive(Clone, Copy, Default)]
struct Span {
    off: u32,
    len: u32,
}

/// High bit of `Span::off` marking a source-buffer span.
const SRC_FLAG: u32 = 1 << 31;

/// A columnar attribute value. Strings are arena spans (zero per-attr alloc);
/// the rare/complex attribute shapes fall back to the owned [`Attr`] enum (only
/// produced by `from_op` for member/object/import/etc., never on the hot parse
/// path).
#[derive(Clone)]
enum AVal {
    Str(Span),
    Bool(bool),
    F64(f64),
    I64(i64),
    NumExtra { raw: Span, value: f64 },
    StrExtra { raw: Span, raw_value: Span },
    BigExtra { raw: Span, raw_value: Span },
    RegExpExtra { raw: Span },
    Other(Box<Attr>),
}

/// One stored attribute: an arena key span + a columnar value.
#[derive(Clone)]
struct MAttr {
    key: Span,
    val: AVal,
}

const NO_IDX: u32 = u32::MAX;

/// Max operands stored inline in an [`OpRow`] before spilling to `value_pool`.
const INLINE_OPERANDS: usize = 2;

/// One operation's row. Operands (≤2, the overwhelming common case) and the
/// single result live **inline** — so emitting a typical op writes nothing to
/// `value_pool` (Cranelift's instruction-with-inline-operands trick). Only ops
/// with >2 operands spill to the pool via `op_overflow`. ~64 bytes, one cache
/// line.
#[derive(Clone, Copy)]
struct OpRow {
    kind: OpKind,
    op_inline: [ValueId; INLINE_OPERANDS],
    op_count: u16,        // operand count; if > INLINE_OPERANDS they're in op_overflow
    has_result: bool,
    result: ValueId,      // the single result value (valid iff has_result)
    op_overflow: Range32, // operands → value_pool, only when op_count > INLINE_OPERANDS
    attrs: Range32,       // → attr_pool
    regions: Range32,     // → region_pool
    succs: Range32,       // → succ_pool
    trivia: u32,          // → trivia_pool, NO_IDX if none
    node_id: u32,         // raw node id, NO_IDX if none
}

// Layout guard (oxc's `assert_layouts` discipline): an op row must stay within
// one cache line so a build writes one line per op. Bump deliberately if needed.
const _: () = assert!(
    std::mem::size_of::<OpRow>() <= 64,
    "OpRow grew past a cache line — re-check the inline-operand layout"
);

impl Default for OpRow {
    fn default() -> Self {
        OpRow {
            kind: OpKind(0),
            op_inline: [ValueId(0); INLINE_OPERANDS],
            op_count: 0,
            has_result: false,
            result: ValueId(0),
            op_overflow: Range32::default(),
            attrs: Range32::default(),
            regions: Range32::default(),
            succs: Range32::default(),
            trivia: NO_IDX,
            node_id: NO_IDX,
        }
    }
}

/// A zero-allocation, zero-copy attribute spec for the build/parse path. Keys are
/// `&'static str` (interned once, deduped); string values are `(start, end)`
/// **byte offsets into the source** — the parser stores offsets, never copying
/// the substring. `Module::set_source` must have installed the matching source.
pub enum AttrSpec {
    /// key, value `(start, end)` in source
    Str(&'static str, u32, u32),
    Bool(&'static str, bool),
    F64(&'static str, f64),
    /// key, raw `(start, end)`, value
    NumExtra(&'static str, u32, u32, f64),
    /// key, raw `(start, end)`, raw_value `(start, end)`
    StrExtra(&'static str, u32, u32, u32, u32),
    /// key, raw `(start, end)`, raw_value `(start, end)`
    BigExtra(&'static str, u32, u32, u32, u32),
}

/// The columnar IR store.
#[derive(Default)]
pub struct Module {
    // ── opcode interner (tiny set → linear scan) ──
    names: Vec<String>,

    // ── op rows (indexed by OpId) ──
    // One contiguous row per op (kind + the five payload ranges + cold trivia/
    // node-id indices). Consolidating these into a single struct means a build
    // writes ONE cache line per op instead of touching eight separate column
    // arrays — the dominant cost when constructing the IR. The fields are still
    // accessed together by the printer, so this costs reads nothing.
    ops: Vec<OpRow>,
    trivia_pool: Vec<Trivia>,

    // ── region table (indexed by RegionId) ──
    region_blocks: Vec<Range32>, // → block_pool

    // ── block table (indexed by BlockSlot) ──
    block_label: Vec<BlockId>,
    block_args: Vec<Range32>, // → value_pool
    block_ops: Vec<Range32>,  // → op_pool

    // ── shared flat pools ──
    value_pool: Vec<ValueId>,
    region_pool: Vec<RegionId>,
    block_pool: Vec<BlockSlot>,
    op_pool: Vec<OpId>,
    attr_pool: Vec<MAttr>,
    succ_pool: Vec<Successor>,

    // ── string buffers ──
    str_arena: String,  // attr keys + from_op's owned string values
    source: String,     // one owned copy of the input; source-span attr values
                        // (the hot parse path) index into this — no per-attr copy
    // Dedup cache for attribute keys (a tiny fixed set): `&'static str` pointer →
    // its interned arena span, so a repeated key is interned once, not per attr.
    key_cache: Vec<(usize, Span)>,

    root: OpId,
}

impl Module {
    pub fn new() -> Module {
        Module::default()
    }

    /// Number of ops in the store.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Pre-size the columns/pools for ~`ops` operations over `src_bytes` of
    /// source, so a build does no reallocation. Estimates are fine (over-reserve
    /// is harmless).
    pub fn reserve(&mut self, ops: usize, src_bytes: usize) {
        self.ops.reserve(ops);
        self.value_pool.reserve(ops * 2);
        self.op_pool.reserve(ops);
        self.attr_pool.reserve(ops * 2);
        self.str_arena.reserve(src_bytes);
        let blocks = ops / 4 + 4;
        self.block_label.reserve(blocks);
        self.block_args.reserve(blocks);
        self.block_ops.reserve(blocks);
        self.region_blocks.reserve(blocks);
    }

    /// Lower an existing AoS [`Op`] tree into the columnar store. Byte-exact
    /// round-trip through the printer is the oracle (`Module::from_op(&op)
    /// .print() == op.print()`).
    pub fn from_op(op: &Op) -> Module {
        let mut m = Module::new();
        let root = m.lower_op(op);
        m.set_root(root);
        m
    }

    fn lower_op(&mut self, op: &Op) -> OpId {
        let id = self.add_op(&op.name);
        self.set_operands(id, &op.operands);
        self.set_results(id, &op.results);
        self.set_attrs(id, &op.attrs);
        self.set_successors(id, &op.successors);
        self.set_trivia(id, op.trivia.clone());
        self.set_node_id(id, op.node_id);

        // Children first, then record the handle lists (keeps pools contiguous).
        let region_ids: Vec<RegionId> = op.regions.iter().map(|r| self.lower_region(r)).collect();
        self.set_regions(id, &region_ids);
        id
    }

    fn lower_region(&mut self, region: &Region) -> RegionId {
        let rid = self.add_region();
        let block_slots: Vec<BlockSlot> = region.blocks.iter().map(|b| self.lower_block(b)).collect();
        self.set_region_blocks(rid, &block_slots);
        rid
    }

    fn lower_block(&mut self, block: &Block) -> BlockSlot {
        let slot = self.add_block(block.id);
        self.set_block_args(slot, &block.args);
        let op_ids: Vec<OpId> = block.ops.iter().map(|o| self.lower_op(o)).collect();
        self.set_block_ops(slot, &op_ids);
        slot
    }

    /// Print the store as MLIR generic text. One printer, shared with the AoS
    /// path via [`IrRead`] (see `print.rs`).
    pub fn print(&self) -> String {
        crate::print::print_root_via(self)
    }

    /// Resident heap footprint of the store: `(bytes, blocks)` where `bytes` is
    /// the sum of all column/pool capacities (× element size) plus the interned
    /// name strings and any boxed trivia, and `blocks` is the number of distinct
    /// heap allocations. Attribute *payload* internals are excluded (identical
    /// content in both layouts — a wash); see the bench for methodology.
    pub fn footprint(&self) -> (usize, usize) {
        let mut bytes = 0usize;
        let mut blocks = 0usize;
        let mut col = |cap: usize, elem: usize| {
            if cap > 0 {
                bytes += cap * elem;
                blocks += 1;
            }
        };
        let r32 = std::mem::size_of::<Range32>();
        // op rows (one consolidated array)
        col(self.ops.capacity(), std::mem::size_of::<OpRow>());
        col(self.trivia_pool.capacity(), std::mem::size_of::<Trivia>());
        // region/block tables
        col(self.region_blocks.capacity(), r32);
        col(self.block_label.capacity(), std::mem::size_of::<BlockId>());
        col(self.block_args.capacity(), r32);
        col(self.block_ops.capacity(), r32);
        // shared pools
        col(self.value_pool.capacity(), std::mem::size_of::<ValueId>());
        col(self.region_pool.capacity(), std::mem::size_of::<RegionId>());
        col(self.block_pool.capacity(), std::mem::size_of::<BlockSlot>());
        col(self.op_pool.capacity(), std::mem::size_of::<OpId>());
        col(self.attr_pool.capacity(), std::mem::size_of::<MAttr>());
        col(self.str_arena.capacity(), 1); // the one string arena buffer
        col(self.source.capacity(), 1); // the one owned source copy
        col(self.succ_pool.capacity(), std::mem::size_of::<Successor>());
        // interned opcode names
        col(self.names.capacity(), std::mem::size_of::<String>());
        for n in &self.names {
            if n.capacity() > 0 {
                bytes += n.capacity();
                blocks += 1;
            }
        }
        (bytes, blocks)
    }

    /// Append a string to the arena, returning its span (amortized O(1), no
    /// per-call heap allocation).
    #[inline]
    fn intern_str(&mut self, s: &str) -> Span {
        let off = self.str_arena.len() as u32;
        self.str_arena.push_str(s);
        Span { off, len: s.len() as u32 }
    }
    fn resolve(&self, sp: Span) -> &str {
        let len = sp.len as usize;
        if sp.off & SRC_FLAG != 0 {
            let off = (sp.off & !SRC_FLAG) as usize;
            &self.source[off..off + len]
        } else {
            let off = sp.off as usize;
            &self.str_arena[off..off + len]
        }
    }

    /// Overwrite a block's op list with `kept` **in place** — `kept` must be a
    /// same-order subsequence of the block's current ops (DCE filtering). Writes
    /// into the existing `op_pool` slots and shortens the range; removed ops'
    /// column/pool data is left as unreachable garbage (never printed), the way
    /// oxc/swc leave removed AST nodes in their arena. No rebuild, no allocation.
    pub fn compact_block_ops(&mut self, block: BlockSlot, kept: &[OpId]) {
        let r = self.block_ops[block.0 as usize];
        debug_assert!(kept.len() <= r.len as usize, "compact must not grow the block");
        let start = r.start as usize;
        for (i, &op) in kept.iter().enumerate() {
            self.op_pool[start + i] = op;
        }
        self.block_ops[block.0 as usize] = Range32 { start: r.start, len: kept.len() as u32 };
    }

    /// The string value of an op's attribute `key`, if present and string-typed
    /// (no allocation — borrows the backing buffer). Used by analysis passes.
    pub fn str_attr(&self, op: OpId, key: &str) -> Option<&str> {
        let r = self.ops[op.0 as usize].attrs;
        for m in &self.attr_pool[r.as_range()] {
            if self.resolve(m.key) == key {
                if let AVal::Str(sp) = m.val {
                    return Some(self.resolve(sp));
                }
            }
        }
        None
    }

    /// A zero-copy span into the owned source copy (`[start, end)` byte offsets).
    fn src_span(start: u32, end: u32) -> Span {
        debug_assert!(start & SRC_FLAG == 0, "source offset too large for flag bit");
        Span { off: start | SRC_FLAG, len: end - start }
    }

    /// Install the owned copy of the source that source-span attr values index
    /// into. Call once before building (the parser does this).
    pub fn set_source(&mut self, src: &str) {
        self.source.clear();
        self.source.push_str(src);
    }

    /// Intern an attribute key, deduplicated by its `&'static str` pointer (the
    /// key set is tiny and repeats on nearly every op) — so a key's bytes are
    /// copied into the arena once, not once per attribute.
    #[inline]
    fn key_span(&mut self, key: &'static str) -> Span {
        let p = key.as_ptr() as usize;
        if let Some(&(_, span)) = self.key_cache.iter().find(|(cp, _)| *cp == p) {
            return span;
        }
        let span = self.intern_str(key);
        self.key_cache.push((p, span));
        span
    }

    /// Set an op's attrs from zero-allocation [`AttrSpec`]s — the fast build/parse
    /// path. Keys and string values are interned into the arena; no `String` is
    /// allocated per attribute.
    #[inline]
    pub fn set_attrs_spec(&mut self, op: OpId, specs: &[AttrSpec]) {
        let start = self.attr_pool.len() as u32;
        for spec in specs {
            let m = match *spec {
                AttrSpec::Str(k, s, e) => {
                    MAttr { key: self.key_span(k), val: AVal::Str(Module::src_span(s, e)) }
                }
                AttrSpec::Bool(k, b) => MAttr { key: self.key_span(k), val: AVal::Bool(b) },
                AttrSpec::F64(k, f) => MAttr { key: self.key_span(k), val: AVal::F64(f) },
                AttrSpec::NumExtra(k, s, e, value) => MAttr {
                    key: self.key_span(k),
                    val: AVal::NumExtra { raw: Module::src_span(s, e), value },
                },
                AttrSpec::StrExtra(k, rs, re, vs, ve) => MAttr {
                    key: self.key_span(k),
                    val: AVal::StrExtra {
                        raw: Module::src_span(rs, re),
                        raw_value: Module::src_span(vs, ve),
                    },
                },
                AttrSpec::BigExtra(k, rs, re, vs, ve) => MAttr {
                    key: self.key_span(k),
                    val: AVal::BigExtra {
                        raw: Module::src_span(rs, re),
                        raw_value: Module::src_span(vs, ve),
                    },
                },
            };
            self.attr_pool.push(m);
        }
        self.ops[op.0 as usize].attrs = Range32 { start, len: specs.len() as u32 };
    }

    /// Convert an owned [`Attr`] into a columnar [`AVal`], arena-interning its
    /// strings. Simple shapes go columnar; complex shapes fall back to `Other`.
    fn attr_to_aval(&mut self, a: &Attr) -> AVal {
        match a {
            Attr::Str(s) => AVal::Str(self.intern_str(s)),
            Attr::Bool(b) => AVal::Bool(*b),
            Attr::F64(f) => AVal::F64(*f),
            Attr::I64(i) => AVal::I64(*i),
            Attr::NumericLiteralExtra { raw, value } => {
                let raw = self.intern_str(raw);
                AVal::NumExtra { raw, value: *value }
            }
            Attr::StringLiteralExtra { raw, raw_value } => {
                let raw = self.intern_str(raw);
                let raw_value = self.intern_str(raw_value);
                AVal::StrExtra { raw, raw_value }
            }
            Attr::BigIntLiteralExtra { raw, raw_value } => {
                let raw = self.intern_str(raw);
                let raw_value = self.intern_str(raw_value);
                AVal::BigExtra { raw, raw_value }
            }
            Attr::RegExpLiteralExtra { raw } => AVal::RegExpExtra { raw: self.intern_str(raw) },
            other => AVal::Other(Box::new(other.clone())),
        }
    }

    /// Render one columnar attribute value byte-exactly (mirrors `Attr::render`).
    fn render_val(&self, v: &AVal, out: &mut String) {
        use crate::attr::{format_mlir_f64, quote_mlir_string};
        match v {
            AVal::Str(sp) => out.push_str(&quote_mlir_string(self.resolve(*sp))),
            AVal::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            AVal::F64(f) => {
                out.push_str(&format_mlir_f64(*f));
                out.push_str(" : f64");
            }
            AVal::I64(i) => out.push_str(&i.to_string()),
            AVal::NumExtra { raw, value } => {
                out.push_str("#jsir<numeric_literal_extra ");
                out.push_str(&quote_mlir_string(self.resolve(*raw)));
                out.push_str(", ");
                out.push_str(&format_mlir_f64(*value));
                out.push_str(" : f64>");
            }
            AVal::StrExtra { raw, raw_value } => {
                out.push_str("#jsir<string_literal_extra ");
                out.push_str(&quote_mlir_string(self.resolve(*raw)));
                out.push_str(", ");
                out.push_str(&quote_mlir_string(self.resolve(*raw_value)));
                out.push('>');
            }
            AVal::BigExtra { raw, raw_value } => {
                out.push_str("#jsir<big_int_literal_extra ");
                out.push_str(&quote_mlir_string(self.resolve(*raw)));
                out.push_str(", ");
                out.push_str(&quote_mlir_string(self.resolve(*raw_value)));
                out.push('>');
            }
            AVal::RegExpExtra { raw } => {
                out.push_str("#jsir<reg_exp_literal_extra ");
                out.push_str(&quote_mlir_string(self.resolve(*raw)));
                out.push('>');
            }
            AVal::Other(a) => out.push_str(&a.render()),
        }
    }

    /// Materialize one columnar value back to an owned [`Attr`] (for the generic
    /// rebuild/copy path; not used by the printer).
    fn aval_to_attr(&self, v: &AVal) -> Attr {
        match v {
            AVal::Str(sp) => Attr::Str(self.resolve(*sp).to_string()),
            AVal::Bool(b) => Attr::Bool(*b),
            AVal::F64(f) => Attr::F64(*f),
            AVal::I64(i) => Attr::I64(*i),
            AVal::NumExtra { raw, value } => {
                Attr::NumericLiteralExtra { raw: self.resolve(*raw).to_string(), value: *value }
            }
            AVal::StrExtra { raw, raw_value } => Attr::StringLiteralExtra {
                raw: self.resolve(*raw).to_string(),
                raw_value: self.resolve(*raw_value).to_string(),
            },
            AVal::BigExtra { raw, raw_value } => Attr::BigIntLiteralExtra {
                raw: self.resolve(*raw).to_string(),
                raw_value: self.resolve(*raw_value).to_string(),
            },
            AVal::RegExpExtra { raw } => {
                Attr::RegExpLiteralExtra { raw: self.resolve(*raw).to_string() }
            }
            AVal::Other(a) => (**a).clone(),
        }
    }

    /// Per-op bytes held in the *hot* columns — what the printer and any
    /// traversal touch (excludes cold trivia/node-id and all pools).
    pub fn per_op_hot_bytes() -> usize {
        let r32 = std::mem::size_of::<Range32>();
        4 + r32 * 5 // op_kind + {operands,results,attrs,regions,succs} ranges
    }

    // Pool append helpers (set-once).
    fn push_values(&mut self, vs: &[ValueId]) -> Range32 {
        let start = self.value_pool.len() as u32;
        self.value_pool.extend_from_slice(vs);
        Range32 { start, len: vs.len() as u32 }
    }
}

impl IrRead for Module {
    fn root(&self) -> OpId {
        self.root
    }
    fn op_name(&self, op: OpId) -> &str {
        &self.names[self.ops[op.0 as usize].kind.0 as usize]
    }
    fn op_kind(&self, op: OpId) -> OpKind {
        self.ops[op.0 as usize].kind
    }
    fn operands(&self, op: OpId) -> &[ValueId] {
        let row = &self.ops[op.0 as usize];
        let n = row.op_count as usize;
        if n <= INLINE_OPERANDS {
            &row.op_inline[..n]
        } else {
            &self.value_pool[row.op_overflow.as_range()]
        }
    }
    fn results(&self, op: OpId) -> &[ValueId] {
        let row = &self.ops[op.0 as usize];
        if row.has_result {
            std::slice::from_ref(&row.result)
        } else {
            &[]
        }
    }
    fn attr_dict(&self, op: OpId, out: &mut String) {
        let entries = &self.attr_pool[self.ops[op.0 as usize].attrs.as_range()];
        if entries.is_empty() {
            return;
        }
        // MLIR DictionaryAttr ordering: sort by key string.
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by(|&a, &b| self.resolve(entries[a].key).cmp(self.resolve(entries[b].key)));
        out.push_str(" <{");
        for (n, &i) in order.iter().enumerate() {
            if n > 0 {
                out.push_str(", ");
            }
            out.push_str(self.resolve(entries[i].key));
            out.push_str(" = ");
            self.render_val(&entries[i].val, out);
        }
        out.push_str("}>");
    }
    fn attrs_owned(&self, op: OpId) -> Vec<(String, Attr)> {
        self.attr_pool[self.ops[op.0 as usize].attrs.as_range()]
            .iter()
            .map(|m| (self.resolve(m.key).to_string(), self.aval_to_attr(&m.val)))
            .collect()
    }
    fn regions(&self, op: OpId) -> &[RegionId] {
        &self.region_pool[self.ops[op.0 as usize].regions.as_range()]
    }
    fn successors(&self, op: OpId) -> &[Successor] {
        &self.succ_pool[self.ops[op.0 as usize].succs.as_range()]
    }
    fn region_blocks(&self, region: RegionId) -> &[BlockSlot] {
        &self.block_pool[self.region_blocks[region.0 as usize].as_range()]
    }
    fn block_label(&self, block: BlockSlot) -> BlockId {
        self.block_label[block.0 as usize]
    }
    fn block_args(&self, block: BlockSlot) -> &[ValueId] {
        &self.value_pool[self.block_args[block.0 as usize].as_range()]
    }
    fn block_ops(&self, block: BlockSlot) -> &[OpId] {
        &self.op_pool[self.block_ops[block.0 as usize].as_range()]
    }
    fn trivia(&self, op: OpId) -> Option<&Trivia> {
        let t = self.ops[op.0 as usize].trivia;
        (t != NO_IDX).then(|| &self.trivia_pool[t as usize])
    }
    fn node_id(&self, op: OpId) -> Option<u32> {
        let n = self.ops[op.0 as usize].node_id;
        (n != NO_IDX).then_some(n)
    }
}

impl IrBuild for Module {
    #[inline]
    fn intern(&mut self, name: &str) -> OpKind {
        // The opcode set is tiny (~tens of names), so a linear scan beats hashing
        // a string on every op.
        if let Some(i) = self.names.iter().position(|n| n == name) {
            return OpKind(i as u32);
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        OpKind(id)
    }

    #[inline]
    fn add_op(&mut self, name: &str) -> OpId {
        let kind = self.intern(name);
        let id = OpId(self.ops.len() as u32);
        self.ops.push(OpRow { kind, ..OpRow::default() }); // one row write per op
        id
    }
    #[inline]
    fn set_operands(&mut self, op: OpId, vs: &[ValueId]) {
        let i = op.0 as usize;
        if vs.len() <= INLINE_OPERANDS {
            let row = &mut self.ops[i];
            row.op_count = vs.len() as u16;
            row.op_inline[..vs.len()].copy_from_slice(vs); // inline — no pool write
        } else {
            let r = self.push_values(vs);
            let row = &mut self.ops[i];
            row.op_count = vs.len() as u16;
            row.op_overflow = r;
        }
    }
    #[inline]
    fn set_results(&mut self, op: OpId, vs: &[ValueId]) {
        let row = &mut self.ops[op.0 as usize];
        match vs.first() {
            Some(&v) => {
                row.result = v;
                row.has_result = true;
            }
            None => row.has_result = false,
        }
        debug_assert!(vs.len() <= 1, "jsir ops have at most one result");
    }
    fn set_attrs(&mut self, op: OpId, attrs: &[(String, Attr)]) {
        let start = self.attr_pool.len() as u32;
        for (k, a) in attrs {
            let key = self.intern_str(k); // owned path: keys aren't &'static, no dedup
            let val = self.attr_to_aval(a);
            self.attr_pool.push(MAttr { key, val });
        }
        self.ops[op.0 as usize].attrs = Range32 { start, len: attrs.len() as u32 };
    }
    #[inline]
    fn set_regions(&mut self, op: OpId, regions: &[RegionId]) {
        let start = self.region_pool.len() as u32;
        self.region_pool.extend_from_slice(regions);
        self.ops[op.0 as usize].regions = Range32 { start, len: regions.len() as u32 };
    }
    fn set_successors(&mut self, op: OpId, succs: &[Successor]) {
        let start = self.succ_pool.len() as u32;
        self.succ_pool.extend_from_slice(succs);
        self.ops[op.0 as usize].succs = Range32 { start, len: succs.len() as u32 };
    }
    fn set_trivia(&mut self, op: OpId, trivia: Option<Trivia>) {
        let slot = match trivia {
            Some(t) => {
                let idx = self.trivia_pool.len() as u32;
                self.trivia_pool.push(t);
                idx
            }
            None => NO_IDX,
        };
        self.ops[op.0 as usize].trivia = slot;
    }
    fn set_node_id(&mut self, op: OpId, node_id: Option<u32>) {
        self.ops[op.0 as usize].node_id = node_id.unwrap_or(NO_IDX);
    }

    #[inline]
    fn add_region(&mut self) -> RegionId {
        let id = RegionId(self.region_blocks.len() as u32);
        self.region_blocks.push(Range32::default());
        id
    }
    #[inline]
    fn set_region_blocks(&mut self, region: RegionId, blocks: &[BlockSlot]) {
        let start = self.block_pool.len() as u32;
        self.block_pool.extend_from_slice(blocks);
        self.region_blocks[region.0 as usize] = Range32 { start, len: blocks.len() as u32 };
    }

    #[inline]
    fn add_block(&mut self, label: BlockId) -> BlockSlot {
        let id = BlockSlot(self.block_label.len() as u32);
        self.block_label.push(label);
        self.block_args.push(Range32::default());
        self.block_ops.push(Range32::default());
        id
    }
    #[inline]
    fn set_block_args(&mut self, block: BlockSlot, vs: &[ValueId]) {
        let r = self.push_values(vs);
        self.block_args[block.0 as usize] = r;
    }
    #[inline]
    fn set_block_ops(&mut self, block: BlockSlot, ops: &[OpId]) {
        let start = self.op_pool.len() as u32;
        self.op_pool.extend_from_slice(ops);
        self.block_ops[block.0 as usize] = Range32 { start, len: ops.len() as u32 };
    }

    fn set_root(&mut self, op: OpId) {
        self.root = op;
    }
}
