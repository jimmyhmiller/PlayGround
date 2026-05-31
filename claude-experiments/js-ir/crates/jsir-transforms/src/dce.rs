//! Dead code elimination, driven by constant propagation.
//!
//! Three sound rewrites:
//!  1. **Constant-condition `if`** — when the test propagates to a constant, the
//!     `jshir.if_statement` is replaced by the taken branch's statements (the
//!     dead branch is dropped). More powerful than upstream, which only matches
//!     a literal `true`/`false` test; we fold `2 > 1`, `!0`, a const variable…
//!  2. **Constant-false `while`** — a loop whose test is constantly false is
//!     removed entirely. (A constant-true test is an infinite loop; left alone,
//!     exactly like upstream.)
//!  3. **Unreachable-after-terminator** — statements following a `return` /
//!     `throw` / `break` / `continue` in the same block can never execute, so
//!     they are dropped. To stay sound w.r.t. hoisting, a block is left intact
//!     if its unreachable tail contains any hoisting declaration.
//!
//! A final cleanup removes the now-dead *constant* expression ops that fed the
//! eliminated conditions (e.g. the `2`, `1`, `2 > 1` behind a folded `if`). We
//! only ever remove ops that are provably pure: literals, and arithmetic whose
//! operands are all known constants. Never a call, member load, or bare
//! identifier read (which could throw a `ReferenceError`).

use std::collections::HashSet;

use jsir_analyses::dataflow::run;
use jsir_analyses::{Analysis, ConstProp, Lattice};
use jsir_ir::{Block, Op, Region, ValueId};

/// What the pass eliminated, for the capability study / reporting.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stats {
    /// `if (const-truthy)` collapsed to its consequent.
    pub if_taken_consequent: usize,
    /// `if (const-falsy)` collapsed to its alternate (or nothing).
    pub if_taken_alternate: usize,
    /// `while (const-falsy)` removed.
    pub while_removed: usize,
    /// Statements dropped because they followed a terminator.
    pub unreachable_statements: usize,
    /// Dead constant-expression ops removed during cleanup.
    pub dead_values_removed: usize,
    /// `var`/`let`/`const` declarations removed because the binding is never
    /// read and never reassigned, and its initializer is side-effect-free.
    pub unused_vars_removed: usize,
    /// `function f(){…}` declarations removed because `f` is never referenced.
    pub unused_fns_removed: usize,
}

impl Stats {
    /// Total number of top-level edits (excludes the value-cleanup bookkeeping).
    pub fn total_eliminations(&self) -> usize {
        self.if_taken_consequent
            + self.if_taken_alternate
            + self.while_removed
            + self.unreachable_statements
    }
}

/// Statement ops after which control cannot fall through to the next statement.
const TERMINATORS: &[&str] = &[
    "jsir.return_statement",
    "jsir.throw_statement",
    "jshir.break_statement",
    "jshir.continue_statement",
];

/// Declaration ops whose binding is hoisted (or creates a TDZ binding) and thus
/// may be observable even when textually unreachable. If an unreachable tail
/// contains one of these, we conservatively keep the whole tail.
const HOISTING_DECLS: &[&str] = &[
    "jsir.variable_declaration",
    "jsir.function_declaration",
    "jsir.class_declaration",
    "jsir.import_declaration",
    "jsir.export_named_declaration",
    "jsir.export_default_declaration",
    "jsir.export_all_declaration",
];

/// Run DCE over a `jsir.file` op, returning the rewritten IR and a report of
/// what was eliminated. The input is not mutated.
pub fn eliminate_dead_code(file: &Op) -> (Op, Stats) {
    eliminate_dead_code_with_roots(file, &HashSet::new())
}

/// Like [`eliminate_dead_code`], but `pinned` names are treated as live roots
/// and never removed even if they look unreferenced. Used by the cross-module
/// tree-shaker to keep bindings that are exported-and-imported elsewhere.
pub fn eliminate_dead_code_with_roots(file: &Op, pinned: &HashSet<String>) -> (Op, Stats) {
    let values = run(&ConstProp, file).values;
    let mut cx = Dce { values, stats: Stats::default() };

    let mut out = cx.transform_op(file);

    // Two cleanups, alternated to a combined fixpoint:
    //  - prune: drop dead pure value ops (e.g. the now-unused condition read
    //    left behind by a folded `if`), which frees their source symbols;
    //  - remove_unused_decls: drop declarations whose bindings are now never
    //    read or reassigned and whose initializer is pure.
    // Each enables the other (folding `if (DEBUG)` -> dead `DEBUG` read ->
    // `var DEBUG` becomes unused), so we loop until neither fires.
    loop {
        let mut changed = false;

        let mut used = HashSet::new();
        collect_used(&out, &mut used);
        let pruned = cx.prune(&mut out, &used);
        cx.stats.dead_values_removed += pruned;
        changed |= pruned != 0;

        let counts = count_symbol_refs(&out);
        let decls = remove_unused_decls(&mut out, &counts, pinned);
        cx.stats.unused_vars_removed += decls;
        changed |= decls != 0;

        let fns = remove_unused_fns(&mut out, &counts, pinned);
        cx.stats.unused_fns_removed += fns;
        changed |= fns != 0;

        if !changed {
            break;
        }
    }

    (out, cx.stats)
}

struct Dce {
    values: std::collections::HashMap<ValueId, Lattice>,
    stats: Stats,
}

impl Dce {
    /// Rebuild `op` with each region's block ops transformed.
    fn transform_op(&mut self, op: &Op) -> Op {
        let mut new = op.clone();
        for region in &mut new.regions {
            for block in &mut region.blocks {
                block.ops = self.transform_block_ops(&block.ops);
            }
        }
        new
    }

    /// Transform a block's op list: fold constant `if`/`while`, drop code after
    /// a statement that always completes abruptly, and recurse into survivors.
    fn transform_block_ops(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut out: Vec<Op> = Vec::with_capacity(ops.len());

        for (i, op) in ops.iter().enumerate() {
            // Statements emitted for this source op (folding can splice >1, or 0).
            let emitted: Vec<Op> = match op.name.as_str() {
                "jshir.if_statement" if self.values.get(op.operands.first().unwrap_or(&ValueId(u32::MAX))).is_some() => {
                    match self.fold_if(op) {
                        Some(taken) => taken,
                        None => vec![self.transform_op(op)],
                    }
                }
                "jshir.while_statement" if self.while_is_dead(op) => {
                    self.stats.while_removed += 1;
                    vec![]
                }
                _ => vec![self.transform_op(op)],
            };

            // Does control fall out the bottom of what we just emitted? If any
            // emitted statement always completes abruptly, the rest of this
            // block is unreachable.
            let abrupt = emitted.iter().any(completes_abruptly);
            out.extend(emitted);

            if abrupt {
                let tail = &ops[i + 1..];
                // Hoisted `var`/`function` bindings stay observable even when
                // textually unreachable, so keep the tail if it has any.
                if !tail.is_empty() && !tail.iter().any(|t| HOISTING_DECLS.contains(&t.name.as_str())) {
                    self.stats.unreachable_statements += tail.len();
                    break;
                }
            }
        }

        out
    }

    /// If `op` is a constant-condition `if`, return the transformed statements of
    /// the taken branch (empty vec for a taken-but-empty branch). `None` means
    /// the condition is not a known constant — keep the `if`.
    fn fold_if(&mut self, op: &Op) -> Option<Vec<Op>> {
        let test = op.operands.first()?;
        let v = self.values.get(test)?;
        let taken_region = if ConstProp.is_true(v) {
            self.stats.if_taken_consequent += 1;
            op.regions.first()
        } else if ConstProp.is_false(v) {
            self.stats.if_taken_alternate += 1;
            op.regions.get(1)
        } else {
            return None;
        };
        Some(self.transform_region_stmts(taken_region))
    }

    /// `while` whose test propagates to a constant-falsy value never runs.
    fn while_is_dead(&self, op: &Op) -> bool {
        match self.expr_region_value(op.regions.first()) {
            Some(v) => ConstProp.is_false(v),
            None => false,
        }
    }

    /// Transform the statement ops of a (single-block) statement region. A
    /// `None`/empty region yields no statements.
    fn transform_region_stmts(&mut self, region: Option<&Region>) -> Vec<Op> {
        match region.and_then(|r| r.blocks.first()) {
            Some(block) => self.transform_block_ops(&block.ops),
            None => Vec::new(),
        }
    }

    /// The propagated value produced by an expression region (the operand of its
    /// `jsir.expr_region_end` terminator).
    fn expr_region_value(&self, region: Option<&Region>) -> Option<&Lattice> {
        let block = region?.blocks.first()?;
        let end = block.ops.iter().find(|o| o.name == "jsir.expr_region_end")?;
        self.values.get(end.operands.first()?)
    }

    /// Recursively remove dead pure ops from `op`'s regions. Returns the count
    /// removed. An op is removable when all its results are unused (not in
    /// `used`) and it is provably side-effect-free.
    fn prune(&self, op: &mut Op, used: &HashSet<ValueId>) -> usize {
        let mut removed = 0;
        for region in &mut op.regions {
            for block in &mut region.blocks {
                // Recurse first so freed inner values count this round.
                for inner in &mut block.ops {
                    removed += self.prune(inner, used);
                }
                let before = block.ops.len();
                block.ops.retain(|o| !self.is_removable_dead(o, used));
                removed += before - block.ops.len();
            }
        }
        removed
    }

    /// True if `op` produces only unused results and is safe to delete.
    fn is_removable_dead(&self, op: &Op, used: &HashSet<ValueId>) -> bool {
        if op.results.is_empty() || op.results.iter().any(|r| used.contains(r)) {
            return false;
        }
        match op.name.as_str() {
            // Literals are always pure.
            "jsir.numeric_literal"
            | "jsir.string_literal"
            | "jsir.boolean_literal"
            | "jsir.null_literal"
            | "jsir.big_int_literal"
            | "jsir.reg_exp_literal" => true,
            // Arithmetic is pure only when every operand is a known constant
            // (so no `valueOf`/`toString` side effect, no throwing coercion).
            "jsir.binary_expression"
            | "jsir.unary_expression"
            | "jsir.parenthesized_expression" => op
                .operands
                .iter()
                .all(|v| matches!(self.values.get(v), Some(Lattice::Const(_)))),
            // A resolved variable read (e.g. a folded-away condition's operand)
            // is pure — removing it can't throw. An unresolved/global read could
            // throw `ReferenceError`, so it is never removed here.
            "jsir.identifier" => op
                .trivia
                .as_ref()
                .and_then(|t| t.referenced_symbol.as_ref())
                .is_some(),
            _ => false,
        }
    }
}

/// A binding's identity: its name plus the uid of the scope that defines it.
type SymKey = (String, Option<i64>);

/// Per-symbol reference counts gathered from the IR trivia.
#[derive(Default)]
struct SymCounts {
    /// r-value reads (`jsir.identifier`).
    reads: std::collections::HashMap<SymKey, usize>,
    /// l-value occurrences (`jsir.identifier_ref`): binding sites + reassignments.
    writes: std::collections::HashMap<SymKey, usize>,
    /// Every referenced name (read or write), ignoring scope. Used for function
    /// declarations, whose binding scope can't be read reliably off the op; a
    /// name-based check is conservative but always sound (we never remove a
    /// function whose name is referenced anywhere).
    referenced_names: HashSet<String>,
}

/// Count reads and writes of every resolved symbol across the whole tree.
fn count_symbol_refs(op: &Op) -> SymCounts {
    let mut c = SymCounts::default();
    count_symbol_refs_into(op, &mut c);
    c
}

fn count_symbol_refs_into(op: &Op, c: &mut SymCounts) {
    if let Some(sym) = op.trivia.as_ref().and_then(|t| t.referenced_symbol.as_ref()) {
        let key = (sym.name.clone(), sym.def_scope_uid);
        match op.name.as_str() {
            "jsir.identifier" => {
                *c.reads.entry(key).or_default() += 1;
                c.referenced_names.insert(sym.name.clone());
            }
            "jsir.identifier_ref" => {
                *c.writes.entry(key).or_default() += 1;
                c.referenced_names.insert(sym.name.clone());
            }
            _ => {}
        }
    }
    for region in &op.regions {
        for block in &region.blocks {
            for inner in &block.ops {
                count_symbol_refs_into(inner, c);
            }
        }
    }
}

/// Remove unused `jsir.variable_declaration` statements from every block.
/// Returns the number of declarations removed.
fn remove_unused_decls(op: &mut Op, counts: &SymCounts, pinned: &HashSet<String>) -> usize {
    let mut removed = 0;
    for region in &mut op.regions {
        for block in &mut region.blocks {
            for inner in &mut block.ops {
                removed += remove_unused_decls(inner, counts, pinned);
            }
            let before = block.ops.len();
            block.ops.retain(|o| !(o.name == "jsir.variable_declaration" && decl_is_unused(o, counts, pinned)));
            removed += before - block.ops.len();
        }
    }
    removed
}

/// Remove unused `function f(){…}` declarations from every block, where `f` is
/// never read or reassigned anywhere. Self-recursive functions that are never
/// called externally are conservatively kept (their internal self-reference
/// counts as a read). Returns the number removed.
fn remove_unused_fns(op: &mut Op, counts: &SymCounts, pinned: &HashSet<String>) -> usize {
    let mut removed = 0;
    for region in &mut op.regions {
        for block in &mut region.blocks {
            for inner in &mut block.ops {
                removed += remove_unused_fns(inner, counts, pinned);
            }
            let before = block.ops.len();
            block.ops.retain(|o| !fn_is_unused(o, counts, pinned));
            removed += before - block.ops.len();
        }
    }
    removed
}

/// A `jsir.function_declaration` whose bound name is referenced nowhere. The
/// check is name-based (not scope-precise) so it stays sound even though a
/// declaration op doesn't reliably carry its own binding scope: if any
/// reference uses this name, in any scope, we keep the function.
fn fn_is_unused(op: &Op, counts: &SymCounts, pinned: &HashSet<String>) -> bool {
    if op.name != "jsir.function_declaration" {
        return false;
    }
    let Some(name) = fn_decl_name(op) else { return false };
    !counts.referenced_names.contains(&name) && !pinned.contains(&name)
}

/// The declared name of a `function f(){…}` (its `id` attribute).
fn fn_decl_name(op: &Op) -> Option<String> {
    op.attrs.iter().find_map(|(k, v)| match v {
        jsir_ir::Attr::Identifier(i) if k == "id" => Some(i.name.clone()),
        _ => None,
    })
}

/// A whole `var`/`let`/`const` declaration is removable when every binding it
/// introduces is never read and never reassigned, and its initializer subtree
/// is side-effect-free.
fn decl_is_unused(decl: &Op, counts: &SymCounts, pinned: &HashSet<String>) -> bool {
    // Gather the bindings this declaration introduces, and how many binding
    // sites (l-value identifier_refs) each has *within* this declaration.
    let mut defined: Vec<SymKey> = Vec::new();
    let mut local_writes: std::collections::HashMap<SymKey, usize> = std::collections::HashMap::new();
    collect_decl_symbols(decl, &mut defined, &mut local_writes);
    if defined.is_empty() {
        return false;
    }
    // A pinned (exported-and-used) binding is a live root.
    if defined.iter().any(|(name, _)| pinned.contains(name)) {
        return false;
    }
    for key in &defined {
        if counts.reads.get(key).copied().unwrap_or(0) != 0 {
            return false; // read somewhere
        }
        // Every l-value occurrence of this symbol must be one of its binding
        // sites here; any extra means it's reassigned elsewhere — keep it.
        let total = counts.writes.get(key).copied().unwrap_or(0);
        if total != local_writes.get(key).copied().unwrap_or(0) {
            return false;
        }
    }
    is_pure_decl_subtree(decl)
}

/// Collect the symbols a declaration defines (from each `variable_declarator`)
/// and, per symbol, the number of l-value binding sites inside the declaration.
fn collect_decl_symbols(op: &Op, defined: &mut Vec<SymKey>, local_writes: &mut std::collections::HashMap<SymKey, usize>) {
    if op.name == "jsir.variable_declarator" {
        if let Some(syms) = op.trivia.as_ref().and_then(|t| t.defined_symbols.as_ref()) {
            for s in syms {
                defined.push((s.name.clone(), s.def_scope_uid));
            }
        }
    }
    if op.name == "jsir.identifier_ref" {
        if let Some(sym) = op.trivia.as_ref().and_then(|t| t.referenced_symbol.as_ref()) {
            *local_writes.entry((sym.name.clone(), sym.def_scope_uid)).or_default() += 1;
        }
    }
    for region in &op.regions {
        for block in &region.blocks {
            for inner in &block.ops {
                collect_decl_symbols(inner, defined, local_writes);
            }
        }
    }
}

/// Ops that may appear inside a removable declaration's initializer without
/// risking a side effect. A `jsir.identifier` read is allowed only when it
/// resolved to a binding (an unresolved/global read can throw `ReferenceError`).
/// Arithmetic is treated as pure under the standard DCE assumption that operand
/// coercion (`valueOf`/`toString`) has no observable side effect.
fn is_pure_decl_subtree(op: &Op) -> bool {
    let ok = match op.name.as_str() {
        "jsir.variable_declaration"
        | "jsir.variable_declarator"
        | "jsir.identifier_ref"
        | "jsir.expr_region_end"
        | "jsir.exprs_region_end"
        | "jsir.numeric_literal"
        | "jsir.string_literal"
        | "jsir.boolean_literal"
        | "jsir.null_literal"
        | "jsir.big_int_literal"
        | "jsir.reg_exp_literal"
        | "jsir.binary_expression"
        | "jsir.unary_expression"
        | "jsir.parenthesized_expression"
        | "jshir.logical_expression"
        | "jshir.conditional_expression" => true,
        "jsir.identifier" => op.trivia.as_ref().and_then(|t| t.referenced_symbol.as_ref()).is_some(),
        _ => false,
    };
    if !ok {
        return false;
    }
    for region in &op.regions {
        for block in &region.blocks {
            if !block.ops.iter().all(is_pure_decl_subtree) {
                return false;
            }
        }
    }
    true
}

/// Whether executing `op` *always* completes abruptly (returns/throws/breaks/
/// continues), so that a following statement in the same block is unreachable.
/// Conservative: anything not provably always-abrupt returns `false`.
fn completes_abruptly(op: &Op) -> bool {
    match op.name.as_str() {
        n if TERMINATORS.contains(&n) => true,
        // A block completes abruptly iff some statement in its body does (every
        // statement before it falls through, so that one is reached).
        "jshir.block_statement" => region_abrupt(op.regions.first()),
        // An `if` completes abruptly only if it has both branches and both do.
        "jshir.if_statement" => {
            region_abrupt(op.regions.first()) && region_abrupt(op.regions.get(1))
        }
        _ => false,
    }
}

/// Whether a (single-block) region's statement sequence completes abruptly.
fn region_abrupt(region: Option<&Region>) -> bool {
    match region.and_then(|r| r.blocks.first()) {
        Some(block) => block.ops.iter().any(completes_abruptly),
        None => false,
    }
}

/// Collect every value used as an operand anywhere in the tree.
fn collect_used(op: &Op, out: &mut HashSet<ValueId>) {
    out.extend(op.operands.iter().copied());
    for region in &op.regions {
        for block in &region.blocks {
            collect_used_block(block, out);
        }
    }
}

fn collect_used_block(block: &Block, out: &mut HashSet<ValueId>) {
    out.extend(block.args.iter().copied());
    for op in &block.ops {
        collect_used(op, out);
    }
}
