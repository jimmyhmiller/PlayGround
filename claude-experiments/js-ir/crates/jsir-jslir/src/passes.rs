//! Ported React Compiler passes that operate on JSLIR + its analyses.
//!
//! These mirror the upstream Rust port (`react-rust-pr36173`) pass-for-pass, but
//! consume our [`crate::ssa::SsaInfo`] / JSLIR instead of the React HIR. Each is
//! covered by hand-written equivalence tests reflecting the upstream behavior.

use jsir_ir::ValueId;

use crate::ssa::SsaInfo;

/// `EliminateRedundantPhi` — the SSA cleanup that runs right after `enter_ssa`.
///
/// A phi is *trivial* when, ignoring self-references, all of its operands are the
/// same value `v`; it then carries no information and is replaced by `v`
/// everywhere (def-use edges and other phis' operands). Removing one phi can make
/// another trivial, so this iterates to a fixpoint. Returns the number removed.
///
/// (Our `enter_ssa` already avoids emitting most trivial merge-phis, so the common
/// case this catches is a loop-header phi for a variable that is rewritten to
/// itself, e.g. `while (c) { x = x; }`.)
pub fn eliminate_redundant_phi(info: &mut SsaInfo) -> usize {
    let mut removed = 0;
    loop {
        // Find a trivial phi: its operands, excluding any self-reference, reduce to
        // a single distinct value.
        let trivial = info.phis.iter().position(|phi| {
            let distinct: std::collections::HashSet<ValueId> = phi
                .operands
                .iter()
                .map(|(_, v)| *v)
                .filter(|v| *v != phi.value)
                .collect();
            distinct.len() == 1
        });
        let Some(idx) = trivial else { break };

        let phi = info.phis.remove(idx);
        let from = phi.value;
        let to = phi
            .operands
            .iter()
            .map(|(_, v)| *v)
            .find(|v| *v != from)
            .expect("trivial phi has one non-self operand");

        // Replace `from` with `to` in every def-use edge and remaining phi operand.
        for def in info.reaching.values_mut() {
            if *def == from {
                *def = to;
            }
        }
        for p in &mut info.phis {
            for (_, v) in &mut p.operands {
                if *v == from {
                    *v = to;
                }
            }
        }
        removed += 1;
    }
    removed
}
