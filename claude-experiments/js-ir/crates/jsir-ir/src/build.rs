//! The reference shape for *building* a [`Module`]: read a source IR via
//! [`IrRead`] and reconstruct it via the [`IrBuild`] API into a fresh store.
//!
//! [`rebuild`] is the identity version — it exercises the entire `IrBuild`
//! surface and is proven to print byte-exact by the corpus test, so it doubles
//! as the worked example of how to construct the columnar IR by hand.

use std::collections::HashSet;

use crate::soa::Module;
use crate::traits::{IrBuild, IrRead, OpId, RegionId};
use crate::BlockId;

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

/// Dead-code elimination on the columnar IR: drops top-level `var`/`let`/`const`
/// declarations whose bound name is never **read** (and whose initializer is
/// pure — true for this subset). A read is a `jsir.identifier` op; a binding is a
/// `jsir.identifier_ref`, which does not count as a use. Returns `(new module,
/// removed count)`.
///
/// This is the SoA IR doing its actual job — an analysis+transform pass, the
/// thing the columnar layout exists for.
pub fn dce(src: &Module) -> (Module, usize) {
    // 1. Liveness: every name that is read somewhere.
    let used = collect_used(src);

    // 2. Navigate to the program's top-level statement list.
    let file = src.root();
    let prog = src.block_ops(src.region_blocks(src.regions(file)[0])[0])[0];
    let prog_regions = src.regions(prog);
    let body_block = src.region_blocks(prog_regions[0])[0];

    // 3. Rebuild, dropping dead variable declarations.
    let mut dst = Module::new();
    let mut removed = 0usize;
    let mut body_ops = Vec::new();
    for &st in src.block_ops(body_block) {
        if src.op_name(st) == "jsir.variable_declaration" && is_dead_var_decl(src, st, &used) {
            removed += 1;
            continue;
        }
        body_ops.push(copy_op(src, st, &mut dst));
    }

    // 4. Re-wrap: program { <live body> }, { <directives, copied> } inside file.
    let bb = dst.add_block(BlockId(0));
    dst.set_block_args(bb, &[]);
    dst.set_block_ops(bb, &body_ops);
    let body_region = dst.add_region();
    dst.set_region_blocks(body_region, &[bb]);
    let dir_region = copy_region(src, prog_regions[1], &mut dst); // directives verbatim

    let p = dst.add_op("jsir.program");
    dst.set_attrs(p, &src.attrs_owned(prog));
    dst.set_results(p, &[]);
    dst.set_regions(p, &[body_region, dir_region]);

    let fb = dst.add_block(BlockId(0));
    dst.set_block_args(fb, &[]);
    dst.set_block_ops(fb, &[p]);
    let file_region = dst.add_region();
    dst.set_region_blocks(file_region, &[fb]);
    let f = dst.add_op("jsir.file");
    dst.set_attrs(f, &src.attrs_owned(file));
    dst.set_results(f, &[]);
    dst.set_regions(f, &[file_region]);
    dst.set_root(f);

    (dst, removed)
}

/// **In-place** DCE: like [`dce`] but mutates the module instead of rebuilding
/// it — it just drops dead top-level `var`-decl OpIds from the body block's op
/// list (the dead ops' columns become unreachable garbage, never printed), the
/// same shape oxc/swc use (in-place AST mutation). Returns the removed count.
pub fn dce_in_place(m: &mut Module) -> usize {
    // Phase 1 (immutable): compute the kept statement list + dead count.
    let (body_block, kept, removed) = {
        let used = collect_used(m);
        let file = m.root();
        let prog = m.block_ops(m.region_blocks(m.regions(file)[0])[0])[0];
        let body_block = m.region_blocks(m.regions(prog)[0])[0];
        let mut kept = Vec::new();
        let mut removed = 0usize;
        for &st in m.block_ops(body_block) {
            if m.op_name(st) == "jsir.variable_declaration" && is_dead_var_decl(m, st, &used) {
                removed += 1;
            } else {
                kept.push(st);
            }
        }
        (body_block, kept, removed)
    };
    // Phase 2 (mutable): compact the body block's op list in place.
    m.compact_block_ops(body_block, &kept);
    removed
}

/// Every name read somewhere (via a `jsir.identifier` op). Bindings
/// (`jsir.identifier_ref`) don't count as reads.
fn collect_used(m: &Module) -> HashSet<&str> {
    let mut used = HashSet::new();
    for i in 0..m.op_count() {
        let op = OpId(i as u32);
        if m.op_name(op) == "jsir.identifier" {
            if let Some(name) = m.str_attr(op, "name") {
                used.insert(name);
            }
        }
    }
    used
}

/// A `jsir.variable_declaration` is dead if every declarator's bound name (the
/// `jsir.identifier_ref` inside its region) is unread.
fn is_dead_var_decl(src: &Module, decl: OpId, used: &HashSet<&str>) -> bool {
    for r in src.regions(decl) {
        for b in src.region_blocks(*r) {
            for &op in src.block_ops(*b) {
                if src.op_name(op) == "jsir.identifier_ref" {
                    if let Some(name) = src.str_attr(op, "name") {
                        if used.contains(name) {
                            return false; // a bound name is read → keep
                        }
                    }
                }
            }
        }
    }
    true
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
