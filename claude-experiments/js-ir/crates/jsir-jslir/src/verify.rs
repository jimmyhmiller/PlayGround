//! JSLIR CFG well-formedness verifier.
//!
//! A safety net over `build_jslir`: a JSLIR function-body region must be a proper
//! CFG. Run on every lowered function so a builder bug surfaces as a loud error
//! rather than malformed IR a later pass would silently misanalyze.

use std::collections::HashSet;

use jsir_ir::Region;

use crate::dialect;

/// Verify a JSLIR CFG region. Returns a list of problems (empty = well-formed).
pub fn verify_cfg(region: &Region) -> Vec<String> {
    let mut errs = Vec::new();
    let ids: HashSet<u32> = region.blocks.iter().map(|b| b.id.0).collect();
    if ids.len() != region.blocks.len() {
        errs.push("duplicate block ids in region".into());
    }

    for block in &region.blocks {
        let label = format!("^bb{}", block.id.0);

        // Exactly one terminator, and it must be last.
        let term_positions: Vec<usize> = block
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| dialect::is_terminator(&op.name))
            .map(|(i, _)| i)
            .collect();
        match term_positions.as_slice() {
            [] => errs.push(format!("{label}: block has no terminator")),
            [pos] => {
                if *pos != block.ops.len() - 1 {
                    errs.push(format!("{label}: terminator is not the last op"));
                }
            }
            many => errs.push(format!("{label}: {} terminators (expected 1)", many.len())),
        }

        // Per-terminator shape + successor validity.
        if let Some(term) = block.ops.last() {
            let (want_succ, want_operands): (Option<usize>, Option<usize>) = match term.name.as_str()
            {
                dialect::RETURN => (Some(0), None),
                dialect::BR => (Some(1), Some(0)),
                dialect::COND_BR => (Some(2), Some(1)),
                _ => (None, None),
            };
            if let Some(n) = want_succ {
                if term.successors.len() != n {
                    errs.push(format!(
                        "{label}: {} has {} successors (expected {n})",
                        term.name,
                        term.successors.len()
                    ));
                }
            }
            if let Some(n) = want_operands {
                if term.operands.len() != n {
                    errs.push(format!(
                        "{label}: {} has {} operands (expected {n})",
                        term.name,
                        term.operands.len()
                    ));
                }
            }
            for s in &term.successors {
                if !ids.contains(&s.block.0) {
                    errs.push(format!("{label}: successor ^bb{} does not exist", s.block.0));
                }
            }
        }
    }
    errs
}
