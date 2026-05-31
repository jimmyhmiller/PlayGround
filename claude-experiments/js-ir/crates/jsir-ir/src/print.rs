//! The generic-format printer + MLIR SSA value numbering.

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
fn print_region(region: &Region, indent: usize, numbering: &Numbering) -> String {
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

fn push_spaces(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push(' ');
    }
}
