use crate::*;
use std::collections::{BTreeMap, BTreeSet};

/// Reconstruct the condition for an operand-tag trap from the trapping
/// instruction — free-standing so the concurrent JIT driver (in the `livetype`
/// crate, with no `Runtime`) can build the same condition the interpreter does.
pub fn operand_type_error(function: DefId, pc: usize, instruction: &Instruction) -> Condition {
    let message = match instruction {
        Instruction::AddI64 { .. } => ERR_ADD_NON_I64,
        Instruction::SubI64 { .. } => ERR_SUB_NON_I64,
        Instruction::MulI64 { .. } => ERR_MUL_NON_I64,
        Instruction::LtI64 { .. } => ERR_LT_NON_I64,
        Instruction::EqI64 { .. } => ERR_EQ_NON_I64,
        Instruction::Not { .. } => ERR_NOT_NON_BOOL,
        Instruction::Branch { .. } => ERR_BRANCH_NON_BOOL,
        _ => "operand type mismatch",
    };
    Condition::RuntimeTypeError {
        function,
        pc,
        message: message.into(),
    }
}

/// Does `function` contain a direct call to `callee`?
fn calls(function: &Function, callee: DefId) -> bool {
    function
        .code
        .iter()
        .any(|i| matches!(i, Instruction::Call { function, .. } if *function == callee))
}

/// How supplying a value resumes a con-freeness trap, treating the paused frame
/// as a one-shot delimited continuation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResumePlan {
    /// Install the value as the trapping instruction's result, advance past it.
    SetAdvance(usize),
    /// Use the (Bool) value to pick a branch target.
    Branch(usize, usize),
    /// Make the value the frame's return value.
    ReturnValue,
}

/// Given the instruction a con-freeness trap fired on, the type of value that
/// resumes it and how resuming installs that value. This is the continuation
/// view of repair: the frozen instruction "yields", and supplying a well-typed
/// value is resuming it with that result. `result_ty` is the trapped function's
/// declared result type (needed for a `Return`).
pub fn resume_shape(
    instruction: &Instruction,
    result_ty: &Type,
    world: &World,
) -> Result<(Type, ResumePlan), String> {
    Ok(match instruction {
        Instruction::AddI64 { dst, .. } | Instruction::SubI64 { dst, .. } | Instruction::MulI64 { dst, .. } => {
            (Type::I64, ResumePlan::SetAdvance(*dst))
        }
        Instruction::LtI64 { dst, .. } | Instruction::EqI64 { dst, .. } | Instruction::Not { dst, .. } => {
            (Type::Bool, ResumePlan::SetAdvance(*dst))
        }
        Instruction::Branch {
            then_pc, else_pc, ..
        } => (Type::Bool, ResumePlan::Branch(*then_pc, *else_pc)),
        Instruction::New { dst, type_id, .. } => (Type::Ref(*type_id), ResumePlan::SetAdvance(*dst)),
        Instruction::Call { dst, function, .. } => {
            let version = *world
                .current_functions
                .get(function)
                .ok_or("callee has no current version")?;
            let FunctionState::Ready(f) = &world.functions[&(*function, version)] else {
                return Err("callee is not ready".into());
            };
            (f.result.clone(), ResumePlan::SetAdvance(*dst))
        }
        Instruction::Return { .. } => (result_ty.clone(), ResumePlan::ReturnValue),
        other => return Err(format!("cannot resume a {other:?} trap by supplying a value")),
    })
}

// Operand-tag error messages, shared so the interpreter arms and the native
// driver's `OUT_TYPE_ERROR` path produce byte-identical conditions.
pub(crate) const ERR_ADD_NON_I64: &str = "addition on non-i64";
pub(crate) const ERR_SUB_NON_I64: &str = "subtraction on non-i64";
pub(crate) const ERR_MUL_NON_I64: &str = "multiplication on non-i64";
pub(crate) const ERR_LT_NON_I64: &str = "comparison on non-i64";
pub(crate) const ERR_EQ_NON_I64: &str = "equality on non-i64";
pub(crate) const ERR_NOT_NON_BOOL: &str = "negation on non-bool";
pub(crate) const ERR_BRANCH_NON_BOOL: &str = "branch on non-bool";

/// A registered native function: the implementation behind a `foreign fn`.
/// `Send` so a runtime carrying foreign functions can still move to the driver
/// thread of a `:live` session. It takes its arguments as values and returns
/// one; native side effects (opening a window, drawing) happen in its body.
pub type ForeignFn = Box<dyn FnMut(&[Value]) -> Value + Send>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallError {
    BadVersion,
    Invalid(Vec<String>),
}

/// Installing definitions is a pure operation on the [`World`] — it never
/// touches running frames. Keeping it here (rather than on `Runtime`) is what
/// lets *both* the single-threaded runtime and the concurrent tier apply hot
/// edits through one code path: `Runtime` wraps these and then resumes any of
/// its paused actors; the concurrent tier applies them to its locked world
/// while worker threads run (a running frame keeps its pinned version; the next
/// call re-resolves the current one).
impl World {
    pub fn install_schema(&mut self, schema: Schema) -> Result<(), InstallError> {
        let expected = self
            .current_schemas
            .get(&schema.type_id)
            .map_or(Version(1), |v| Version(v.0 + 1));
        if schema.version != expected {
            return Err(InstallError::BadVersion);
        }
        verify_schema(&schema, self).map_err(InstallError::Invalid)?;
        let type_id = schema.type_id;
        let version = schema.version;
        self.current_schemas.insert(type_id, version);
        self.schemas.insert((type_id, version), schema);
        // Auto-derive the migration from the previous version where the change
        // is trivial (D6): copy a surviving field, default a new/retyped one; a
        // field that is neither is a gap that abandons derivation (a developer
        // supplies a transformer). Derived migrations are copy/default only, so
        // they are type-sound by construction; an explicit `install_migration`
        // overrides them.
        if version.0 > 1 {
            let from = Version(version.0 - 1);
            if let Some(migration) = self.derive_migration(type_id, from, version) {
                self.migrations.insert((type_id, from), migration);
            }
        }
        self.invalidate_functions(type_id);
        self.epoch += 1;
        Ok(())
    }

    /// Try to build the `from → to` migration mechanically. `None` on a gap.
    fn derive_migration(&self, type_id: DefId, from: Version, to: Version) -> Option<Migration> {
        let old = self.schemas.get(&(type_id, from))?;
        let new = self.schemas.get(&(type_id, to))?;
        let mut fields = BTreeMap::new();
        for field in &new.fields {
            let source = match old.field(field.id) {
                Some(old_field) if old_field.ty == field.ty => MigrationSource::Copy(field.id),
                _ => MigrationSource::Value(field.default.clone()?),
            };
            fields.insert(field.id, source);
        }
        Some(Migration {
            type_id,
            from,
            to,
            fields,
        })
    }

    pub fn install_migration(
        &mut self,
        migration: Migration,
        heap: &Heap,
    ) -> Result<(), InstallError> {
        if migration.to.0 != migration.from.0 + 1 {
            return Err(InstallError::BadVersion);
        }
        let old = self
            .schemas
            .get(&(migration.type_id, migration.from))
            .ok_or(InstallError::BadVersion)?;
        let new = self
            .schemas
            .get(&(migration.type_id, migration.to))
            .ok_or(InstallError::BadVersion)?;
        let mut errors = Vec::new();
        for field in &new.fields {
            let Some(source) = migration.fields.get(&field.id) else {
                errors.push(format!("missing output field '{}'", field.name));
                continue;
            };
            let source_ty = match source {
                MigrationSource::Copy(id) => old.field(*id).map(|f| f.ty.clone()),
                MigrationSource::Value(value) => heap.shallow_type(value),
                MigrationSource::Wrap {
                    type_id,
                    field: inner,
                    source,
                } => {
                    let input = old.field(*source).map(|f| &f.ty);
                    let version = self.current_schemas.get(type_id);
                    let target = version
                        .and_then(|v| self.schemas.get(&(*type_id, *v)))
                        .and_then(|s| s.field(*inner));
                    if input == target.map(|f| &f.ty) {
                        Some(Type::Ref(*type_id))
                    } else {
                        None
                    }
                }
            };
            if source_ty.as_ref() != Some(&field.ty) {
                errors.push(format!("migration for '{}' has the wrong type", field.name));
            }
        }
        if !errors.is_empty() {
            return Err(InstallError::Invalid(errors));
        }
        self.migrations
            .insert((migration.type_id, migration.from), migration);
        Ok(())
    }

    pub fn install_function(&mut self, function: Function) -> Result<(), InstallError> {
        let expected = self
            .current_functions
            .get(&function.id)
            .map_or(Version(1), |v| Version(v.0 + 1));
        if function.version != expected {
            return Err(InstallError::BadVersion);
        }
        let state = match verify_function(&function, self) {
            Ok(deps) => {
                self.function_deps.insert(function.id, deps);
                FunctionState::Ready(function)
            }
            Err(diagnostics) => FunctionState::Broken {
                id: function.id,
                version: function.version,
                name: function.name,
                diagnostics,
            },
        };
        let id = state.id();
        let version = state.version();
        self.functions.insert((id, version), state);
        self.current_functions.insert(id, version);
        self.epoch += 1;
        Ok(())
    }

    pub fn install_verified_function(
        &mut self,
        function: Function,
        deps: BTreeSet<DefId>,
    ) -> Result<(), InstallError> {
        let expected = self
            .current_functions
            .get(&function.id)
            .map_or(Version(1), |v| Version(v.0 + 1));
        if function.version != expected {
            return Err(InstallError::BadVersion);
        }
        let id = function.id;
        let version = function.version;
        self.function_deps.insert(id, deps);
        self.functions
            .insert((id, version), FunctionState::Ready(function));
        self.current_functions.insert(id, version);
        self.epoch += 1;
        Ok(())
    }

    /// Re-verify only the functions a schema change can reach (D7), propagating
    /// brokenness through the call graph to a fixpoint. See the long note that
    /// used to live on `Runtime::invalidate_functions`.
    fn invalidate_functions(&mut self, changed: DefId) {
        let mut work: Vec<DefId> = self
            .current_functions
            .keys()
            .filter(|id| {
                self.function_deps
                    .get(id)
                    .is_some_and(|deps| deps.contains(&changed))
            })
            .copied()
            .collect();
        let mut seen: BTreeSet<DefId> = work.iter().copied().collect();

        while let Some(id) = work.pop() {
            let version = self.current_functions[&id];
            let Some(FunctionState::Ready(function)) = self.functions.get(&(id, version)) else {
                continue; // already broken (or gone): nothing to re-verify
            };
            let function = function.clone();
            let Err(diagnostics) = verify_function(&function, self) else {
                continue; // still well-typed against the new definitions
            };
            let broken = Version(version.0 + 1);
            self.functions.insert(
                (id, broken),
                FunctionState::Broken {
                    id,
                    version: broken,
                    name: function.name,
                    diagnostics,
                },
            );
            self.current_functions.insert(id, broken);
            let callers: Vec<DefId> = self
                .current_functions
                .iter()
                .filter(|(caller, _)| !seen.contains(caller))
                .filter_map(|(caller, cver)| match self.functions.get(&(*caller, *cver)) {
                    Some(FunctionState::Ready(f)) if calls(f, id) => Some(*caller),
                    _ => None,
                })
                .collect();
            for caller in callers {
                seen.insert(caller);
                work.push(caller);
            }
        }
    }
}
