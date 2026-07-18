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
    Condition, DefId, Engine, ForeignFn, Schema, Type, Value, Version, verify_function_with,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// A compiled program: an engine with every struct and function installed, plus
/// the name → id maps so callers can run `main` (or any function).
pub struct Compiled {
    pub engine: Arc<Engine>,
    pub functions: HashMap<String, DefId>,
    pub structs: HashMap<String, DefId>,
}

/// A live-programming session: a running [`Engine`] plus the symbol table and
/// version bookkeeping that let you `eval` more source *into it over time*. Each
/// `eval` installs new versions of the definitions it names — driving migration,
/// invalidation, and trap-and-repair on whatever is already running (on this
/// thread or any other; installs land through the engine's world lock).
pub struct Session {
    pub engine: Arc<Engine>,
    ids: lower::IdEnv,
}

impl Default for Session {
    fn default() -> Self {
        Session::new()
    }
}

impl Session {
    /// A session on an interpreter-only engine (the LLVM-free configuration).
    pub fn new() -> Session {
        Session::with_engine(Engine::interp())
    }

    /// A session on any engine — a compiling one auto-promotes hot functions.
    pub fn with_engine(engine: Arc<Engine>) -> Session {
        Session {
            engine,
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

    /// Register (or replace) the native implementation of a `foreign fn` by
    /// name — the host binds the native layer before the code that uses it runs.
    /// The `foreign fn` must already have been declared (in this or a prior
    /// `eval`), so its id is known.
    pub fn register_foreign(&mut self, name: &str, f: ForeignFn) -> Result<(), String> {
        let id = self
            .ids
            .foreign_fn_id_of(name)
            .ok_or_else(|| format!("no foreign fn `{name}` has been declared"))?;
        self.engine.register_foreign(id, f);
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
        match self.engine.run_call(id, args) {
            crate::Outcome::Complete(v) => Ok(v),
            crate::Outcome::Paused(c) => Err(c),
        }
    }

    pub fn eval(&mut self, source: &str) -> Result<(), String> {
        let tokens = lexer::lex(source)?;
        let program = parser::parse(tokens)?;
        let lowered = lower::lower(&program, &mut self.ids)?;

        // Publish the foreign signatures and global types this edit introduced
        // (accumulated in `ids`) into the world before verification, so a
        // `CallForeign`/`LoadGlobal` in any installed function type-checks.
        self.engine.shared().set_declared_interface(
            self.ids.foreign_sigs().into_iter().collect(),
            self.ids.global_types().into_iter().collect(),
        );

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
            // Next version follows the world's current one — which may already
            // have advanced (e.g. a prior edit's `invalidate` bumped it).
            let v = self
                .engine
                .with_world(|w| w.current_schemas.get(&id).map_or(1, |c| c.0 + 1));
            schema.version = Version(v);
            self.engine
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
            // Next version follows the world's current one (which invalidation
            // may have bumped to a Broken version behind our back), and
            // verification runs against the world as of this moment — the read
            // guard is released before installing (the write lock is not
            // reentrant with it).
            let deps = self.engine.with_world(|w| {
                let v = w.current_functions.get(&func.id).map_or(1, |c| c.0 + 1);
                func.version = Version(v);
                verify_function_with(&func, w, &sigs)
            });
            let deps =
                deps.map_err(|diags| format!("in `{}`: {}", func.name, diags.join("; ")))?;
            self.engine
                .install_verified_function(func, deps)
                .map_err(|e| format!("installing a function: {e:?}"))?;
        }

        // Run each `letonce` initializer exactly once: only if the global is not
        // already set. A reload re-installs the init function but skips running
        // it, so a native resource created on the first load survives the edit.
        for init in &lowered.global_inits {
            if self.engine.global(init.global_id).is_some() {
                continue;
            }
            let value = match self.engine.run_call(init.init_fn, Vec::new()) {
                crate::Outcome::Complete(v) => v,
                crate::Outcome::Paused(c) => {
                    return Err(format!("initializing a `letonce`: {c:?}"));
                }
            };
            self.engine.set_global(init.global_id, value);
        }
        Ok(())
    }

    fn maps(&self) -> (HashMap<String, DefId>, HashMap<String, DefId>) {
        (self.ids.fn_map(), self.ids.struct_map())
    }
}

/// Compile source text into a ready-to-run [`Engine`] (a one-shot [`Session`]
/// on the interpreter-only configuration).
pub fn compile(source: &str) -> Result<Compiled, String> {
    compile_on(source, Engine::interp())
}

/// Compile source text onto a specific engine.
pub fn compile_on(source: &str, engine: Arc<Engine>) -> Result<Compiled, String> {
    let mut session = Session::with_engine(engine);
    session.eval(source)?;
    let (functions, structs) = session.maps();
    Ok(Compiled {
        engine: session.engine,
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
