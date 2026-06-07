//! The reference shape for *building* a [`Module`]: read a source IR via
//! [`IrRead`] and reconstruct it via the [`IrBuild`] API into a fresh store.
//!
//! [`rebuild`] is the identity version — it exercises the entire `IrBuild`
//! surface and is proven to print byte-exact by the corpus test, so it doubles
//! as the worked example of how to construct the columnar IR by hand.

use crate::soa::Module;
use crate::traits::{IrBuild, IrRead, OpId, RegionId};

/// Identity rebuild: read the whole IR via [`IrRead`] and reconstruct it via
/// [`IrBuild`] into a fresh store. Generic over both ends, so it documents the
/// portable build shape exactly. Proven byte-exact by the corpus test.
pub fn rebuild_into<R: IrRead, B: IrBuild>(src: &R, dst: &mut B) {
    let root = copy_op(src, src.root(), dst);
    dst.set_root(root);
}

/// Convenience wrapper that rebuilds into a fresh [`Module`].
pub fn rebuild<R: IrRead>(src: &R) -> Module {
    let mut dst = Module::new();
    rebuild_into(src, &mut dst);
    dst
}

fn copy_op<R: IrRead, B: IrBuild>(src: &R, op: OpId, dst: &mut B) -> OpId {
    let id = dst.add_op(src.op_name(op));
    dst.set_operands(id, src.operands(op));
    dst.set_results(id, src.results(op));
    dst.set_attrs(id, &src.attrs_owned(op));
    dst.set_successors(id, src.successors(op));
    dst.set_trivia(id, src.trivia(op).cloned());
    dst.set_node_id(id, src.node_id(op));
    // Children first, then record the handle list (keeps pools contiguous).
    let regions: Vec<RegionId> =
        src.regions(op).iter().map(|r| copy_region(src, *r, dst)).collect();
    dst.set_regions(id, &regions);
    id
}

fn copy_region<R: IrRead, B: IrBuild>(src: &R, region: RegionId, dst: &mut B) -> RegionId {
    let rid = dst.add_region();
    let blocks: Vec<_> = src
        .region_blocks(region)
        .iter()
        .map(|b| {
            let slot = dst.add_block(src.block_label(*b));
            dst.set_block_args(slot, src.block_args(*b));
            let ops: Vec<OpId> = src.block_ops(*b).iter().map(|o| copy_op(src, *o, dst)).collect();
            dst.set_block_ops(slot, &ops);
            slot
        })
        .collect();
    dst.set_region_blocks(rid, &blocks);
    rid
}
