//! The generic-format printer + MLIR SSA value numbering.

use crate::traits::{IrRead, OpId, RegionId};
use crate::{Op, Region, ValueId};
use std::collections::HashMap;

pub type Numbering = HashMap<ValueId, u32>;

/// Number the values defined in `region` and its descendants, reproducing
/// MLIR's scheme: within a region, number all block-level values first (block
/// args + op results, in order), THEN descend into each op's child regions.
/// Each child region starts numbering at `base + (values defined in this
/// region)`, and sibling regions share that start (so they reuse numbers).
pub fn number_region(region: &Region, base: u32, numbering: &mut Numbering) {
    let mut counter = base;
    for block in &region.blocks {
        for arg in &block.args {
            numbering.insert(*arg, counter);
            counter += 1;
        }
        for op in &block.ops {
            for res in &op.results {
                numbering.insert(*res, counter);
                counter += 1;
            }
        }
    }
    // `counter` now = base + number of values defined directly in this region.
    for block in &region.blocks {
        for op in &block.ops {
            for child in &op.regions {
                number_region(child, counter, numbering);
            }
        }
    }
}

/// Print an op at the given indentation (number of leading spaces).
pub fn print_op(op: &Op, indent: usize, numbering: &Numbering, out: &mut String) {
    push_spaces(out, indent);

    // Results: `%a, %b = `
    if !op.results.is_empty() {
        let names: Vec<String> = op
            .results
            .iter()
            .map(|v| format!("%{}", numbering[v]))
            .collect();
        out.push_str(&names.join(", "));
        out.push_str(" = ");
    }

    // Name + operands.
    out.push('"');
    out.push_str(&op.name);
    out.push('"');
    out.push('(');
    let operands: Vec<String> = op
        .operands
        .iter()
        .map(|v| format!("%{}", numbering[v]))
        .collect();
    out.push_str(&operands.join(", "));
    out.push(')');

    // CFG successor list (MLIR `[^bb1(%a), ^bb2]`). Empty for non-terminators and
    // the whole AST dialect, so existing output is unaffected.
    if !op.successors.is_empty() {
        let succs: Vec<String> = op
            .successors
            .iter()
            .map(|s| {
                if s.args.is_empty() {
                    format!("^bb{}", s.block.0)
                } else {
                    let args: Vec<String> =
                        s.args.iter().map(|v| format!("%{}", numbering[v])).collect();
                    format!("^bb{}({})", s.block.0, args.join(", "))
                }
            })
            .collect();
        out.push('[');
        out.push_str(&succs.join(", "));
        out.push(']');
    }

    // Attribute dictionary, sorted by key (MLIR DictionaryAttr ordering).
    if !op.attrs.is_empty() {
        let mut attrs = op.attrs.clone();
        attrs.sort_by(|a, b| a.0.cmp(&b.0));
        let rendered: Vec<String> = attrs
            .iter()
            .map(|(k, v)| format!("{} = {}", k, v.render()))
            .collect();
        out.push_str(" <{");
        out.push_str(&rendered.join(", "));
        out.push_str("}>");
    }

    // Regions.
    if !op.regions.is_empty() {
        out.push_str(" (");
        let region_strs: Vec<String> = op
            .regions
            .iter()
            .map(|r| print_region(r, indent, numbering))
            .collect();
        out.push_str(&region_strs.join(", "));
        out.push(')');
    }

    // Type signature: ` : (operandTypes) -> resultTypes`.
    out.push_str(" : (");
    out.push_str(&vec!["!jsir.any"; op.operands.len()].join(", "));
    out.push_str(") -> ");
    match op.results.len() {
        0 => out.push_str("()"),
        1 => out.push_str("!jsir.any"),
        n => {
            out.push('(');
            out.push_str(&vec!["!jsir.any"; n].join(", "));
            out.push(')');
        }
    }
}

/// Render a single region as `{\n ...content... \n<indent>}` (the op-level `(`
/// and `)` wrapping is added by the caller; regions are joined with `, `).
///
/// Dispatches on shape: the AST dialect is always a single block with no args and
/// no successors and takes the legacy path (byte-identical to upstream jsir).
/// CFG-dialect regions (multiple blocks, block arguments, or terminator
/// successors) print MLIR-style block labels `^bbN(%args):`.
fn print_region(region: &Region, indent: usize, numbering: &Numbering) -> String {
    let is_cfg = region.blocks.len() > 1
        || region.blocks.iter().any(|b| !b.args.is_empty())
        || region
            .blocks
            .iter()
            .any(|b| b.ops.iter().any(|op| !op.successors.is_empty()));
    if is_cfg {
        print_cfg_region(region, indent, numbering)
    } else {
        print_legacy_region(region, indent, numbering)
    }
}

/// The original single-block AST-dialect printer. MUST stay byte-identical to
/// upstream jsir output (fixtures depend on it).
fn print_legacy_region(region: &Region, indent: usize, numbering: &Numbering) -> String {
    let mut s = String::from("{\n");
    for block in &region.blocks {
        if block.ops.is_empty() && block.args.is_empty() {
            // An empty block prints just its label `^bb0:` at the op indent.
            push_spaces(&mut s, indent);
            s.push_str("^bb0:\n");
        } else {
            // Non-empty entry block: ops at indent+2, label elided.
            for op in &block.ops {
                print_op(op, indent + 2, numbering, &mut s);
                s.push('\n');
            }
        }
    }
    push_spaces(&mut s, indent);
    s.push('}');
    s
}

/// CFG-dialect region printer (MLIR `^bbN(%args):` blocks). The entry block's
/// label is elided when it takes no arguments (MLIR's rule); every other block
/// prints `^bb<id>` (its stable [`crate::BlockId`]) and its argument list.
fn print_cfg_region(region: &Region, indent: usize, numbering: &Numbering) -> String {
    let mut s = String::from("{\n");
    for (i, block) in region.blocks.iter().enumerate() {
        let elide_label = i == 0 && block.args.is_empty();
        if !elide_label {
            push_spaces(&mut s, indent);
            s.push_str(&format!("^bb{}", block.id.0));
            if !block.args.is_empty() {
                let args: Vec<String> = block
                    .args
                    .iter()
                    .map(|a| format!("%{}: !jsir.any", numbering[a]))
                    .collect();
                s.push('(');
                s.push_str(&args.join(", "));
                s.push(')');
            }
            s.push_str(":\n");
        }
        for op in &block.ops {
            print_op(op, indent + 2, numbering, &mut s);
            s.push('\n');
        }
    }
    push_spaces(&mut s, indent);
    s.push('}');
    s
}

fn push_spaces(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push(' ');
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Generic printer over `IrRead`.
//
// Byte-for-byte the same algorithm as the AoS functions above, but reading the
// IR through the trait so it works over any backend (today: the SoA `Module`).
// The cross-fixture test asserts `Module::from_op(&op).print() == op.print()`,
// so this and the AoS path cannot silently diverge.
// ───────────────────────────────────────────────────────────────────────────

/// Print the whole IR rooted at [`IrRead::root`].
pub fn print_root_via<R: IrRead>(ir: &R) -> String {
    let root = ir.root();
    let mut numbering = Numbering::new();
    let mut counter = 0u32;
    for r in ir.results(root) {
        numbering.insert(*r, counter);
        counter += 1;
    }
    for region in ir.regions(root) {
        number_region_via(ir, *region, counter, &mut numbering);
    }
    let mut out = String::new();
    print_op_via(ir, root, 0, &numbering, &mut out);
    out
}

fn number_region_via<R: IrRead>(ir: &R, region: RegionId, base: u32, numbering: &mut Numbering) {
    let mut counter = base;
    for block in ir.region_blocks(region) {
        for arg in ir.block_args(*block) {
            numbering.insert(*arg, counter);
            counter += 1;
        }
        for op in ir.block_ops(*block) {
            for res in ir.results(*op) {
                numbering.insert(*res, counter);
                counter += 1;
            }
        }
    }
    for block in ir.region_blocks(region) {
        for op in ir.block_ops(*block) {
            for child in ir.regions(*op) {
                number_region_via(ir, *child, counter, numbering);
            }
        }
    }
}

fn print_op_via<R: IrRead>(
    ir: &R,
    op: OpId,
    indent: usize,
    numbering: &Numbering,
    out: &mut String,
) {
    push_spaces(out, indent);

    let results = ir.results(op);
    if !results.is_empty() {
        let names: Vec<String> = results.iter().map(|v| format!("%{}", numbering[v])).collect();
        out.push_str(&names.join(", "));
        out.push_str(" = ");
    }

    out.push('"');
    out.push_str(ir.op_name(op));
    out.push('"');
    out.push('(');
    let operands = ir.operands(op);
    let operand_strs: Vec<String> = operands.iter().map(|v| format!("%{}", numbering[v])).collect();
    out.push_str(&operand_strs.join(", "));
    out.push(')');

    let successors = ir.successors(op);
    if !successors.is_empty() {
        let succs: Vec<String> = successors
            .iter()
            .map(|s| {
                if s.args.is_empty() {
                    format!("^bb{}", s.block.0)
                } else {
                    let args: Vec<String> =
                        s.args.iter().map(|v| format!("%{}", numbering[v])).collect();
                    format!("^bb{}({})", s.block.0, args.join(", "))
                }
            })
            .collect();
        out.push('[');
        out.push_str(&succs.join(", "));
        out.push(']');
    }

    let attrs = ir.attrs(op);
    if !attrs.is_empty() {
        let mut attrs = attrs.to_vec();
        attrs.sort_by(|a, b| a.0.cmp(&b.0));
        let rendered: Vec<String> =
            attrs.iter().map(|(k, v)| format!("{} = {}", k, v.render())).collect();
        out.push_str(" <{");
        out.push_str(&rendered.join(", "));
        out.push_str("}>");
    }

    let regions = ir.regions(op);
    if !regions.is_empty() {
        out.push_str(" (");
        let region_strs: Vec<String> =
            regions.iter().map(|r| print_region_via(ir, *r, indent, numbering)).collect();
        out.push_str(&region_strs.join(", "));
        out.push(')');
    }

    out.push_str(" : (");
    out.push_str(&vec!["!jsir.any"; operands.len()].join(", "));
    out.push_str(") -> ");
    match results.len() {
        0 => out.push_str("()"),
        1 => out.push_str("!jsir.any"),
        n => {
            out.push('(');
            out.push_str(&vec!["!jsir.any"; n].join(", "));
            out.push(')');
        }
    }
}

fn print_region_via<R: IrRead>(
    ir: &R,
    region: RegionId,
    indent: usize,
    numbering: &Numbering,
) -> String {
    let blocks = ir.region_blocks(region);
    let is_cfg = blocks.len() > 1
        || blocks.iter().any(|b| !ir.block_args(*b).is_empty())
        || blocks
            .iter()
            .any(|b| ir.block_ops(*b).iter().any(|op| !ir.successors(*op).is_empty()));
    if is_cfg {
        print_cfg_region_via(ir, region, indent, numbering)
    } else {
        print_legacy_region_via(ir, region, indent, numbering)
    }
}

fn print_legacy_region_via<R: IrRead>(
    ir: &R,
    region: RegionId,
    indent: usize,
    numbering: &Numbering,
) -> String {
    let mut s = String::from("{\n");
    for block in ir.region_blocks(region) {
        let ops = ir.block_ops(*block);
        if ops.is_empty() && ir.block_args(*block).is_empty() {
            push_spaces(&mut s, indent);
            s.push_str("^bb0:\n");
        } else {
            for op in ops {
                print_op_via(ir, *op, indent + 2, numbering, &mut s);
                s.push('\n');
            }
        }
    }
    push_spaces(&mut s, indent);
    s.push('}');
    s
}

fn print_cfg_region_via<R: IrRead>(
    ir: &R,
    region: RegionId,
    indent: usize,
    numbering: &Numbering,
) -> String {
    let mut s = String::from("{\n");
    for (i, block) in ir.region_blocks(region).iter().enumerate() {
        let args = ir.block_args(*block);
        let elide_label = i == 0 && args.is_empty();
        if !elide_label {
            push_spaces(&mut s, indent);
            s.push_str(&format!("^bb{}", ir.block_label(*block).0));
            if !args.is_empty() {
                let arg_strs: Vec<String> =
                    args.iter().map(|a| format!("%{}: !jsir.any", numbering[a])).collect();
                s.push('(');
                s.push_str(&arg_strs.join(", "));
                s.push(')');
            }
            s.push_str(":\n");
        }
        for op in ir.block_ops(*block) {
            print_op_via(ir, *op, indent + 2, numbering, &mut s);
            s.push('\n');
        }
    }
    push_spaces(&mut s, indent);
    s.push('}');
    s
}
