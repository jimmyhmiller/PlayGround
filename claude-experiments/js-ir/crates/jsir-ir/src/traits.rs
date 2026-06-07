//! The data-oriented access vocabulary for JSIR.
//!
//! These two traits are the *stable boundary* every consumer (the printer,
//! analyses, transforms) is written against, so the underlying storage can be
//! swapped (today: the SoA [`crate::soa::Module`]) without touching call sites.
//!
//! Design rules that keep the abstraction zero-cost over a columnar backend
//! (see `docs/IR_STORAGE.md`):
//!
//! 1. **Field accessors keyed by id — never a whole-node reference.** No method
//!    returns `&Op`. If one did, a struct-of-arrays backend would be forced to
//!    materialize an `Op` to borrow from, defeating the layout. Every getter
//!    returns a scalar or a slice into a column/pool.
//! 2. **Static dispatch.** Passes take `impl IrRead` / `impl IrBuild`, never
//!    `&dyn`, so the accessors monomorphize and inline to the raw slice indexing
//!    a hand-written columnar pass would use.
//! 3. **Mutation is method-based, set-once.** There is no `&mut Op`. Building is
//!    "allocate a slot, then set each field with the full slice"; structural
//!    edits are expressed as a *rebuild* (read old via [`IrRead`], emit new via
//!    [`IrBuild`]) — the idiomatic compacting-pass shape.

use crate::{Attr, BlockId, Successor, Trivia, ValueId};

/// Storage handle for an operation: a dense index into the op columns. This is
/// the op's identity — there is no pointer. `NonMax`/niche packing is a later
/// refinement; a plain `u32` is enough to prove the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct OpId(pub u32);

/// Storage handle for a region (index into the region table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionId(pub u32);

/// Storage handle for a block (index into the block table). Distinct from
/// [`BlockId`], which is the *printed* region-scoped `^bbN` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockSlot(pub u32);

/// An interned opcode. Replaces the per-op `String name`: ~80 distinct opcode
/// strings are deduplicated into one table, and an op carries a 4-byte index.
/// Comparisons are only meaningful within the module that minted them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpKind(pub u32);

/// Read side of the IR: pure field accessors keyed by storage handle.
pub trait IrRead {
    /// The top-level op (the `jsir.file`), entry point for traversal/printing.
    fn root(&self) -> OpId;

    /// The op's textual name (e.g. `"jsir.binary_expression"`). Backed by the
    /// opcode interner — no per-op allocation.
    fn op_name(&self, op: OpId) -> &str;
    /// The op's interned opcode, for integer-fast dispatch in passes.
    fn op_kind(&self, op: OpId) -> OpKind;

    fn operands(&self, op: OpId) -> &[ValueId];
    fn results(&self, op: OpId) -> &[ValueId];
    /// Render this op's attribute dictionary (the ` <{k = v, ...}>` section,
    /// sorted by key) into `out`, or nothing if it has no attributes. The
    /// backend owns rendering so a columnar/arena store never materializes
    /// `String`s on the hot path.
    fn attr_dict(&self, op: OpId, out: &mut String);
    /// Materialize this op's attributes as owned pairs (for the generic
    /// rebuild/copy path; not used by the printer).
    fn attrs_owned(&self, op: OpId) -> Vec<(String, Attr)>;
    fn regions(&self, op: OpId) -> &[RegionId];
    fn successors(&self, op: OpId) -> &[Successor];

    fn region_blocks(&self, region: RegionId) -> &[BlockSlot];
    /// The printed `^bbN` label of a block.
    fn block_label(&self, block: BlockSlot) -> BlockId;
    fn block_args(&self, block: BlockSlot) -> &[ValueId];
    fn block_ops(&self, block: BlockSlot) -> &[OpId];

    /// Cold metadata — ignored by the printer, carried for `hir2ast`.
    fn trivia(&self, op: OpId) -> Option<&Trivia>;
    fn node_id(&self, op: OpId) -> Option<u32>;
}

/// Write side of the IR: set-once construction. A field is filled exactly once,
/// with its full slice; child-list setters ([`set_regions`](IrBuild::set_regions),
/// [`set_region_blocks`](IrBuild::set_region_blocks),
/// [`set_block_ops`](IrBuild::set_block_ops)) take handles minted by the
/// corresponding `add_*` and must be called *after* the children are built, so
/// each column/pool slice stays contiguous.
pub trait IrBuild: IrRead {
    /// Intern an opcode name (idempotent).
    fn intern(&mut self, name: &str) -> OpKind;

    /// Allocate a fresh op slot with the given opcode and empty fields.
    fn add_op(&mut self, name: &str) -> OpId;
    fn set_operands(&mut self, op: OpId, vs: &[ValueId]);
    fn set_results(&mut self, op: OpId, vs: &[ValueId]);
    fn set_attrs(&mut self, op: OpId, attrs: &[(String, Attr)]);
    fn set_regions(&mut self, op: OpId, regions: &[RegionId]);
    fn set_successors(&mut self, op: OpId, succs: &[Successor]);
    fn set_trivia(&mut self, op: OpId, trivia: Option<Trivia>);
    fn set_node_id(&mut self, op: OpId, node_id: Option<u32>);

    /// Allocate a fresh region slot with no blocks.
    fn add_region(&mut self) -> RegionId;
    fn set_region_blocks(&mut self, region: RegionId, blocks: &[BlockSlot]);

    /// Allocate a fresh block slot with the given printed label.
    fn add_block(&mut self, label: BlockId) -> BlockSlot;
    fn set_block_args(&mut self, block: BlockSlot, vs: &[ValueId]);
    fn set_block_ops(&mut self, block: BlockSlot, ops: &[OpId]);

    fn set_root(&mut self, op: OpId);
}
