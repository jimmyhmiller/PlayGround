//! The frontend: a Rust-flavored surface syntax that compiles to the runtime's
//! IR. `struct` → `Schema`, `fn` → `Function`; the body language has `let`,
//! assignment, `if`/`else`, `while`, `return`, `emit`, struct literals, field
//! access, calls, and `+ - < >` arithmetic.
//!
//! ```text
//! struct Account { balance: i64, fee: i64 = 0 }
//! fn charge(a: &Account, amt: i64) -> i64 {
//!     let b = a.balance;
//!     return b - amt;
//! }
//! ```
//!
//! Not yet: recursion (functions install in call-graph order, so cycles are a
//! compile error — use a `while` loop), tail-expression returns (`return` is
//! explicit), and `*` / `==` (only `+ - < >` are wired to IR ops so far).

mod ast;
mod lexer;
mod lower;
mod parser;

pub use ast::*;

use crate::{DefId, Runtime, Schema, Type, verify_function};
use std::collections::{BTreeMap, HashMap};

/// A compiled program: a runtime with every struct and function installed, plus
/// the name → id maps so callers can spawn `main` (or any function).
pub struct Compiled {
    pub runtime: Runtime,
    pub functions: HashMap<String, DefId>,
    pub structs: HashMap<String, DefId>,
}

/// Compile source text into a ready-to-run [`Runtime`].
pub fn compile(source: &str) -> Result<Compiled, String> {
    let tokens = lexer::lex(source)?;
    let program = parser::parse(tokens)?;
    let lowered = lower::lower(&program)?;

    let mut rt = Runtime::default();

    // Install structs in dependency order: a struct that references another
    // must be installed after it (self-references are fine).
    let schema_by_id: BTreeMap<DefId, Schema> =
        lowered.schemas.iter().map(|s| (s.type_id, s.clone())).collect();
    let struct_deps: HashMap<DefId, Vec<DefId>> = lowered
        .schemas
        .iter()
        .map(|s| {
            let deps = s
                .fields
                .iter()
                .filter_map(|f| match f.ty {
                    Type::Ref(t) if t != s.type_id => Some(t),
                    _ => None,
                })
                .collect();
            (s.type_id, deps)
        })
        .collect();
    for id in topo(schema_by_id.keys().copied().collect(), &struct_deps)
        .map_err(|_| "structs reference each other cyclically (not supported)".to_string())?
    {
        rt.install_schema(schema_by_id[&id].clone())
            .map_err(|e| format!("installing a struct: {e:?}"))?;
    }

    // Install functions in call-graph order: a callee before its caller (so it
    // is Ready when the caller is verified). A cycle is recursion, unsupported.
    let fn_by_id: BTreeMap<DefId, &crate::Function> =
        lowered.functions.iter().map(|f| (f.id, f)).collect();
    let fn_deps: HashMap<DefId, Vec<DefId>> = lowered
        .functions
        .iter()
        .map(|f| {
            let mut callees: Vec<DefId> = f
                .code
                .iter()
                .filter_map(|i| match i {
                    crate::Instruction::Call { function, .. } => Some(*function),
                    _ => None,
                })
                .collect();
            callees.sort_unstable();
            callees.dedup();
            (f.id, callees)
        })
        .collect();
    for id in topo(fn_by_id.keys().copied().collect(), &fn_deps)
        .map_err(|_| "recursion is not supported yet (use a `while` loop)".to_string())?
    {
        let func = fn_by_id[&id];
        // Verify first so a type error is a clean compile error rather than a
        // silently-Broken function version.
        verify_function(func, &rt.world)
            .map_err(|diags| format!("in `{}`: {}", func.name, diags.join("; ")))?;
        rt.install_function((*func).clone())
            .map_err(|e| format!("installing `{}`: {e:?}", func.name))?;
    }

    Ok(Compiled {
        runtime: rt,
        functions: lowered.fn_ids,
        structs: lowered.struct_ids,
    })
}

/// Topological sort: returns an order in which every node follows all of its
/// dependencies. `Err` on a cycle.
fn topo(nodes: Vec<DefId>, deps: &HashMap<DefId, Vec<DefId>>) -> Result<Vec<DefId>, ()> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Unseen,
        Active,
        Done,
    }
    let mut mark: HashMap<DefId, Mark> = nodes.iter().map(|n| (*n, Mark::Unseen)).collect();
    let mut order = Vec::new();
    // Iterative DFS to avoid recursion depth limits.
    for &start in &nodes {
        if mark[&start] != Mark::Unseen {
            continue;
        }
        let mut stack = vec![(start, 0usize)];
        while let Some((node, idx)) = stack.pop() {
            if idx == 0 {
                mark.insert(node, Mark::Active);
            }
            let empty = Vec::new();
            let children = deps.get(&node).unwrap_or(&empty);
            if idx < children.len() {
                stack.push((node, idx + 1));
                let child = children[idx];
                match mark.get(&child).copied().unwrap_or(Mark::Done) {
                    Mark::Active => return Err(()),
                    Mark::Unseen => stack.push((child, 0)),
                    Mark::Done => {}
                }
            } else {
                if mark[&node] != Mark::Done {
                    mark.insert(node, Mark::Done);
                    order.push(node);
                }
            }
        }
    }
    Ok(order)
}
