//! IR-rewrite memoization (the React-Compiler payoff, done as a JSIR -> JSIR
//! transform instead of string codegen).
//!
//! This is the *production* path that replaces the frozen string emitter
//! (`jsir-ssa::codegen::emit_memoized`, kept only as the parity reference). The
//! reactive-scope ANALYSIS + the value-level emit live in `jsir-ssa`; it hands
//! us a [`MemoLayout`] — a ready list of JSIR statement-run ops built from only
//! op names `hir2ast` already lifts — and we do the pure structural splice here:
//!
//!  * prepend the `react/compiler-runtime` import (switching the program to a
//!    module),
//!  * relabel the synthesized ops' value ids past everything already in the
//!    file (so the global `hir2ast` value index never collides),
//!  * replace the memoized function's body with the new statement list.
//!
//! NO relooper, NO string codegen, NO purity violation: every op is real JSIR
//! printed through the reversible IR path. Any malformed input is a hard
//! [`Err`], never a silent skip.

use std::collections::HashMap;

use jsir_ir::{
    Attr, IdentifierAttr, ImportSpecKind, ImportSpecifierAttr, Op, StringLiteralKeyAttr, ValueId,
};

/// The memoized function body, as a flat list of JSIR ops (operand value-ops
/// then statement roots, exactly the shape a statements region holds). Built by
/// the analysis side (`jsir-ssa::memoize_plan`); the transform splices it in.
///
/// Statement runs may be wrapped in transparent `jsir.__stmt_run` holders (a run
/// of operand ops + a statement root that must stay contiguous); the transform
/// flattens them. All value ids are local and get relabeled here.
#[derive(Debug, Clone)]
pub struct MemoLayout {
    /// `_c(N)` cache size (informational; the body already encodes it).
    pub cache_size: usize,
    /// The new function-body statement ops (may contain `jsir.__stmt_run`
    /// holders to flatten).
    pub body: Vec<Op>,
}

/// Result of [`memoize_file`].
pub struct MemoizeResult {
    /// The rewritten program op-tree, ready for `jsir_swc::ir_to_source`.
    pub file: Op,
}

/// Rewrite the memoized function in `file` to use `layout`'s body.
pub fn memoize_file(file: &Op, layout: &MemoLayout) -> Result<MemoizeResult, String> {
    let mut new_file = file.clone();

    // Relabel each statement run (independent local namespace) to fresh ids past
    // the file's max, THEN flatten the run holders into the real body op list.
    // Relabel-before-flatten keeps each run's within-run value references intact
    // (cross-run references are by name, never by value id).
    let mut runs: Vec<Op> = layout.body.clone();
    relabel_past_file(&new_file, &mut runs);
    let mut body: Vec<Op> = Vec::new();
    for op in runs {
        flatten_into(op, &mut body);
    }

    // Locate program op.
    let program = new_file
        .regions
        .first_mut()
        .and_then(|r| r.blocks.first_mut())
        .and_then(|blk| blk.ops.first_mut())
        .ok_or("memoize: no program op")?;
    if program.name != "jsir.program" {
        return Err(format!("memoize: expected jsir.program, got {}", program.name));
    }
    // Adding an `import` makes the program a module.
    if let Some((_, a)) = program.attrs.iter_mut().find(|(k, _)| k == "source_type") {
        *a = Attr::Str("module".into());
    } else {
        program.attrs.push(("source_type".into(), Attr::Str("module".into())));
    }

    // Prepend the runtime import as the program's first statement.
    let prog_block = program
        .regions
        .first_mut()
        .and_then(|r| r.blocks.first_mut())
        .ok_or("memoize: program has no body block")?;
    let import = import_c();
    prog_block.ops.insert(0, import);

    // Find the function declaration (descending export wrappers) and replace its
    // body block's op list.
    let func = prog_block
        .ops
        .iter_mut()
        .find_map(find_function_mut)
        .ok_or("memoize: no function declaration to memoize")?;
    let body_block = func
        .regions
        .get_mut(1)
        .and_then(|r| r.blocks.first_mut())
        .and_then(|blk| blk.ops.first_mut())
        .ok_or("memoize: function has no body block_statement")?;
    if body_block.name != "jshir.block_statement" {
        return Err(format!("memoize: expected body block_statement, got {}", body_block.name));
    }
    let stmts_block = body_block
        .regions
        .first_mut()
        .and_then(|r| r.blocks.first_mut())
        .ok_or("memoize: body block_statement has no statements block")?;
    stmts_block.ops = body;

    Ok(MemoizeResult { file: new_file })
}

/// Recursively expand `jsir.__stmt_run` holders into their flat op lists.
fn flatten_into(op: Op, out: &mut Vec<Op>) {
    if op.name == "jsir.__stmt_run" {
        if let Some(block) = op.regions.into_iter().next().and_then(|r| r.blocks.into_iter().next())
        {
            for inner in block.ops {
                flatten_into(inner, out);
            }
        }
        return;
    }
    out.push(op);
}

/// The largest `result`/`operand` value id in `file` + 1.
fn file_max_id(file: &Op) -> u32 {
    let mut max = 0u32;
    fn scan(op: &Op, max: &mut u32) {
        for r in &op.results {
            *max = (*max).max(r.0 + 1);
        }
        for o in &op.operands {
            *max = (*max).max(o.0 + 1);
        }
        for region in &op.regions {
            for block in &region.blocks {
                for inner in &block.ops {
                    scan(inner, max);
                }
            }
        }
    }
    scan(file, &mut max);
    max
}

/// Relabel synthesized value ids to fresh ids past the file's max.
///
/// Each top-level body op (a single statement, after `__stmt_run` flattening)
/// owns an INDEPENDENT local value namespace (the analysis emits each statement
/// with ids starting at 0; cross-statement references are by *name*, never by
/// value id). So we relabel one statement at a time with its own map but a
/// shared running counter, keeping every global id unique.
fn relabel_past_file(file: &Op, body: &mut [Op]) {
    let mut next = file_max_id(file);

    fn assign(op: &mut Op, next: &mut u32, map: &mut HashMap<u32, ValueId>) {
        for r in &mut op.results {
            let nv = ValueId(*next);
            *next += 1;
            map.insert(r.0, nv);
            *r = nv;
        }
        for region in &mut op.regions {
            for block in &mut region.blocks {
                for inner in &mut block.ops {
                    assign(inner, next, map);
                }
            }
        }
    }
    fn remap(op: &mut Op, map: &HashMap<u32, ValueId>) {
        for o in &mut op.operands {
            if let Some(nv) = map.get(&o.0) {
                *o = *nv;
            }
        }
        for region in &mut op.regions {
            for block in &mut region.blocks {
                for inner in &mut block.ops {
                    remap(inner, map);
                }
            }
        }
    }

    for op in body.iter_mut() {
        let mut map: HashMap<u32, ValueId> = HashMap::new();
        assign(op, &mut next, &mut map);
        remap(op, &map);
    }
}

/// The first `function_declaration` in `op`, descending one level into export
/// wrappers (mirrors `jsir-ssa::lower::find_function`).
fn find_function_mut(op: &mut Op) -> Option<&mut Op> {
    if op.name == "jsir.function_declaration" {
        return Some(op);
    }
    if op.name == "jsir.export_named_declaration" || op.name == "jsir.export_default_declaration" {
        return op
            .regions
            .first_mut()
            .and_then(|r| r.blocks.first_mut())
            .and_then(|b| b.ops.iter_mut().find(|o| o.name == "jsir.function_declaration"));
    }
    None
}

/// A `JsirIdentifierAttr` with dummy spans (the textual printer is never used on
/// this path; `to_swc` takes only the name).
fn ident_attr(name: &str) -> IdentifierAttr {
    IdentifierAttr {
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
        identifier_name: name.to_string(),
        start_index: 0,
        end_index: 0,
        scope_uid: 0,
        name: name.to_string(),
    }
}

/// `import { c as _c } from "react/compiler-runtime";`
fn import_c() -> Op {
    let mut op = Op::new("jsir.import_declaration");
    op.attrs.push((
        "source".into(),
        Attr::StringLiteralKey(Box::new(StringLiteralKeyAttr {
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            start_index: 0,
            end_index: 0,
            scope_uid: 0,
            value: "react/compiler-runtime".into(),
            raw: "\"react/compiler-runtime\"".into(),
            raw_value: "react/compiler-runtime".into(),
        })),
    ));
    op.attrs.push((
        "specifiers".into(),
        Attr::Array(vec![Attr::ImportSpecifier(Box::new(ImportSpecifierAttr {
            kind: ImportSpecKind::Named,
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            start_index: 0,
            end_index: 0,
            scope_uid: 0,
            sym_name: "_c".into(),
            sym_scope: 0,
            imported: Some(Attr::Identifier(Box::new(ident_attr("c")))),
            local: ident_attr("_c"),
        }))]),
    ));
    op
}
