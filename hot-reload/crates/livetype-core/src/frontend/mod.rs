//! The frontend: a Rust-flavored surface syntax that compiles to the runtime's
//! IR. `struct` → `Schema`, `fn` → `Function`; the body language has `let`,
//! assignment, `if`/`else`, `while`, `return`, `emit`, struct literals, field
//! access, calls, and `+ - < >` arithmetic.
//!
//! ```text
//! struct Account { balance: i64, fee: i64 = 0 }
//! fn charge(a: Account, amt: i64) -> i64 {   // structs are GC references, no `&`
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

use crate::{
    ActorStatus, Condition, DefId, ForeignFn, Runtime, Schema, Type, Value, Version,
    verify_function_with,
};
use std::collections::{BTreeMap, HashMap};

/// A compiled program: a runtime with every struct and function installed, plus
/// the name → id maps so callers can spawn `main` (or any function).
#[derive(Debug)]
pub struct Compiled {
    pub runtime: Runtime,
    pub functions: HashMap<String, DefId>,
    pub structs: HashMap<String, DefId>,
}

/// A live-programming session: a running [`Runtime`] plus the symbol table and
/// version bookkeeping that let you `eval` more source *into it over time*. Each
/// `eval` installs new versions of the definitions it names — driving migration,
/// invalidation, and trap-and-repair on whatever is already running. This is the
/// object behind the live-edit POC.
pub struct Session {
    pub runtime: Runtime,
    ids: lower::IdEnv,
}

impl Default for Session {
    fn default() -> Self {
        Session::new()
    }
}

impl Session {
    pub fn new() -> Session {
        Session {
            runtime: Runtime::default(),
            ids: lower::IdEnv::new(),
        }
    }

    pub fn struct_id(&self, name: &str) -> Option<DefId> {
        self.ids.struct_of(name)
    }
    pub fn fn_id(&self, name: &str) -> Option<DefId> {
        self.ids.fn_of(name)
    }
    /// The opaque kind assigned to a declared `foreign type`, so a host's native
    /// constructor can tag the handle it returns to match the declared type.
    pub fn foreign_kind(&self, name: &str) -> Option<crate::ForeignKind> {
        self.ids.foreign_kind_of(name)
    }

    /// Compile `source` and install its definitions as the next versions of
    /// whatever they name — the live edit. Structs install first (in
    /// reference-dependency order) so the functions in the same edit verify
    /// against the new layouts; functions install as a batch so recursion and
    /// forward references work.
    /// Register (or replace) the native implementation of a `foreign fn` by
    /// name — the host binds the native layer before the code that uses it runs.
    /// The `foreign fn` must already have been declared (in this or a prior
    /// `eval`), so its id is known.
    pub fn register_foreign(&mut self, name: &str, f: ForeignFn) -> Result<(), String> {
        let id = self
            .ids
            .foreign_fn_id_of(name)
            .ok_or_else(|| format!("no foreign fn `{name}` has been declared"))?;
        self.runtime.register_foreign(id, f);
        Ok(())
    }

    /// The trampoline: call a managed function by name and run it to completion,
    /// resolving its *current version* at call time (late binding). This is how
    /// native code calls up into managed code — an edit to `name` is picked up
    /// by the next call, with no re-registration. A con-freeness trap surfaces
    /// as the returned [`Condition`] (a callback cannot freeze under a native
    /// frame); the host decides whether to supply a default and report.
    pub fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value, Condition> {
        let id = self.ids.fn_of(name).ok_or_else(|| Condition::BrokenFunction {
            function: 0,
            diagnostics: vec![format!("no function `{name}`")],
        })?;
        let actor = self.runtime.spawn(id, args)?;
        let outcome = loop {
            self.runtime.step(actor);
            match &self.runtime.actors[&actor].status {
                ActorStatus::Complete(v) => break Ok(v.clone()),
                ActorStatus::Paused(c) => break Err(c.clone()),
                ActorStatus::Runnable => {}
            }
        };
        // The trampoline owns this actor for the duration of the call; drop it so
        // repeated calls (one per frame) don't accumulate frames.
        self.runtime.actors.remove(&actor);
        outcome
    }

    pub fn eval(&mut self, source: &str) -> Result<(), String> {
        let tokens = lexer::lex(source)?;
        let program = parser::parse(tokens)?;
        let lowered = lower::lower(&program, &mut self.ids)?;

        // Publish the foreign signatures and global types this edit introduced
        // (accumulated in `ids`) into the world before verification, so a
        // `CallForeign`/`LoadGlobal` in any installed function type-checks.
        self.runtime.world.foreign_sigs =
            self.ids.foreign_sigs().into_iter().collect();
        self.runtime.world.global_types =
            self.ids.global_types().into_iter().collect();

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
            let mut schema = schema_by_id[&id].clone();
            // Next version follows the runtime's current one — which may already
            // have advanced (e.g. a prior edit's `invalidate` bumped it).
            let v = self.runtime.world.current_schemas.get(&id).map_or(1, |c| c.0 + 1);
            schema.version = Version(v);
            self.runtime
                .install_schema(schema)
                .map_err(|e| format!("installing a struct: {e:?}"))?;
        }

        let sigs: BTreeMap<DefId, (Vec<Type>, Type)> = lowered
            .functions
            .iter()
            .map(|f| (f.id, (f.params.clone(), f.result.clone())))
            .collect();
        for func in &lowered.functions {
            let mut func = func.clone();
            // Next version follows the runtime's current one (which invalidation
            // may have bumped to a Broken version behind our back).
            let v = self
                .runtime
                .world
                .current_functions
                .get(&func.id)
                .map_or(1, |c| c.0 + 1);
            func.version = Version(v);
            let deps = verify_function_with(&func, &self.runtime.world, &sigs)
                .map_err(|diags| format!("in `{}`: {}", func.name, diags.join("; ")))?;
            self.runtime
                .install_verified_function(func, deps)
                .map_err(|e| format!("installing a function: {e:?}"))?;
        }

        // Run each `letonce` initializer exactly once: only if the global is not
        // already set. A reload re-installs the init function but skips running
        // it, so a native resource created on the first load survives the edit.
        for init in &lowered.global_inits {
            if self.runtime.globals.contains_key(&init.global_id) {
                continue;
            }
            let value = self
                .run_to_value(init.init_fn)
                .map_err(|c| format!("initializing a `letonce`: {c:?}"))?;
            self.runtime.set_global(init.global_id, value);
        }
        Ok(())
    }

    /// Run a zero-argument function to completion and return its value — used to
    /// evaluate `letonce` initializers. Shares the trampoline's run loop.
    fn run_to_value(&mut self, id: DefId) -> Result<Value, Condition> {
        let actor = self.runtime.spawn(id, Vec::new())?;
        let outcome = loop {
            self.runtime.step(actor);
            match &self.runtime.actors[&actor].status {
                ActorStatus::Complete(v) => break Ok(v.clone()),
                ActorStatus::Paused(c) => break Err(c.clone()),
                ActorStatus::Runnable => {}
            }
        };
        self.runtime.actors.remove(&actor);
        outcome
    }

    fn maps(&self) -> (HashMap<String, DefId>, HashMap<String, DefId>) {
        (self.ids.fn_map(), self.ids.struct_map())
    }
}

/// Compile source text into a ready-to-run [`Runtime`] (a one-shot [`Session`]).
pub fn compile(source: &str) -> Result<Compiled, String> {
    let mut session = Session::new();
    session.eval(source)?;
    let (functions, structs) = session.maps();
    Ok(Compiled {
        runtime: session.runtime,
        functions,
        structs,
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
