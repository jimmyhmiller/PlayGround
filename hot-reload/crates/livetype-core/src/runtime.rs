use crate::*;
use std::collections::{BTreeMap, BTreeSet};

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

#[derive(Debug, Default)]
pub struct Runtime {
    pub world: World,
    pub heap: BTreeMap<ObjectId, Object>,
    pub actors: BTreeMap<ActorId, Actor>,
    /// Committed external observations. `Emit` advances the frame only after
    /// appending, so pausing and resuming cannot replay an earlier effect.
    pub output: Vec<Value>,
    /// Set by [`Runtime::jit_get_field`] when a migration barrier trips: the
    /// native `step` cannot carry a rich condition through its integer return,
    /// so it stashes the condition here and returns the `CONDITION` outcome for
    /// the JIT driver to pick up. Unused by the interpreter path.
    pub pending_condition: Option<Condition>,
    next_object: ObjectId,
    next_actor: ActorId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallError {
    BadVersion,
    Invalid(Vec<String>),
}

impl Runtime {
    /// Hand off this runtime's world, heap, and object-id counter to the
    /// thread-safe [`crate::Shared`] tier. All setup — schema/function installs,
    /// auto-derived migrations, verification — is done through the ordinary
    /// single-threaded API and then frozen for concurrent execution.
    pub fn into_parts(self) -> (World, BTreeMap<ObjectId, Object>, ObjectId) {
        (self.world, self.heap, self.next_object)
    }

    pub fn install_schema(&mut self, schema: Schema) -> Result<(), InstallError> {
        let expected = self
            .world
            .current_schemas
            .get(&schema.type_id)
            .map_or(Version(1), |v| Version(v.0 + 1));
        if schema.version != expected {
            return Err(InstallError::BadVersion);
        }
        verify_schema(&schema, &self.world).map_err(InstallError::Invalid)?;
        let type_id = schema.type_id;
        let version = schema.version;
        self.world.current_schemas.insert(type_id, version);
        self.world.schemas.insert((type_id, version), schema);
        // Auto-derive the migration from the previous version where the change
        // is trivial (D6). A field is copied when it survives unchanged, or
        // default-initialized when new/retyped-with-a-default; a field that is
        // neither is a *gap* that abandons derivation, leaving a developer to
        // supply a transformer (a `MissingMigration` trap on first cross). A
        // derived migration is copy/default only, so it is type-sound by
        // construction; an explicit `install_migration` for the same step
        // overrides it.
        if version.0 > 1 {
            let from = Version(version.0 - 1);
            if let Some(migration) = self.derive_migration(type_id, from, version) {
                self.world.migrations.insert((type_id, from), migration);
            }
        }
        self.invalidate_functions(type_id);
        self.world.epoch += 1;
        Ok(())
    }

    /// Try to build the `from → to` migration mechanically (see
    /// [`Runtime::install_schema`]). Returns `None` when any field is a gap.
    fn derive_migration(&self, type_id: DefId, from: Version, to: Version) -> Option<Migration> {
        let old = self.world.schemas.get(&(type_id, from))?;
        let new = self.world.schemas.get(&(type_id, to))?;
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

    pub fn install_migration(&mut self, migration: Migration) -> Result<(), InstallError> {
        if migration.to.0 != migration.from.0 + 1 {
            return Err(InstallError::BadVersion);
        }
        let old = self
            .world
            .schemas
            .get(&(migration.type_id, migration.from))
            .ok_or(InstallError::BadVersion)?;
        let new = self
            .world
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
                MigrationSource::Value(value) => value.shallow_type(&self.heap),
                MigrationSource::Wrap {
                    type_id,
                    field: inner,
                    source,
                } => {
                    let input = old.field(*source).map(|f| &f.ty);
                    let version = self.world.current_schemas.get(type_id);
                    let target = version
                        .and_then(|v| self.world.schemas.get(&(*type_id, *v)))
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
        self.world
            .migrations
            .insert((migration.type_id, migration.from), migration);
        self.resume_repaired();
        Ok(())
    }

    pub fn install_function(&mut self, function: Function) -> Result<(), InstallError> {
        let expected = self
            .world
            .current_functions
            .get(&function.id)
            .map_or(Version(1), |v| Version(v.0 + 1));
        if function.version != expected {
            return Err(InstallError::BadVersion);
        }
        let state = match verify_function(&function, &self.world) {
            Ok(deps) => {
                self.world.function_deps.insert(function.id, deps);
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
        self.world.functions.insert((id, version), state);
        self.world.current_functions.insert(id, version);
        self.world.epoch += 1;
        self.resume_repaired();
        Ok(())
    }

    /// Install a function that has already been verified elsewhere, with its
    /// dependency set (`deps`) supplied — used by the frontend's batch install,
    /// where a group of functions (possibly mutually recursive) are verified
    /// together against each other's signatures and cannot pass the per-function
    /// callee-is-Ready check of [`Runtime::install_function`]. Trusts the caller
    /// verified it; installs it Ready.
    pub fn install_verified_function(
        &mut self,
        function: Function,
        deps: std::collections::BTreeSet<DefId>,
    ) -> Result<(), InstallError> {
        let expected = self
            .world
            .current_functions
            .get(&function.id)
            .map_or(Version(1), |v| Version(v.0 + 1));
        if function.version != expected {
            return Err(InstallError::BadVersion);
        }
        let id = function.id;
        let version = function.version;
        self.world.function_deps.insert(id, deps);
        self.world
            .functions
            .insert((id, version), FunctionState::Ready(function));
        self.world.current_functions.insert(id, version);
        self.world.epoch += 1;
        self.resume_repaired();
        Ok(())
    }

    pub fn spawn(&mut self, function: DefId, args: Vec<Value>) -> Result<ActorId, Condition> {
        let version = *self.world.current_functions.get(&function).ok_or_else(|| {
            Condition::BrokenFunction {
                function,
                diagnostics: vec!["unknown function".into()],
            }
        })?;
        let FunctionState::Ready(code) = &self.world.functions[&(function, version)] else {
            let FunctionState::Broken { diagnostics, .. } =
                &self.world.functions[&(function, version)]
            else {
                unreachable!()
            };
            return Err(Condition::BrokenFunction {
                function,
                diagnostics: diagnostics.clone(),
            });
        };
        let mut registers = vec![None; code.registers];
        for (slot, value) in args.into_iter().enumerate() {
            registers[slot] = Some(value);
        }
        self.next_actor += 1;
        let id = self.next_actor;
        self.actors.insert(
            id,
            Actor {
                id,
                frames: vec![Frame {
                    function: (function, version),
                    pc: 0,
                    registers,
                    return_to: None,
                }],
                status: ActorStatus::Runnable,
            },
        );
        Ok(id)
    }

    pub fn run(&mut self) {
        loop {
            let runnable: Vec<_> = self
                .actors
                .iter()
                .filter_map(|(id, actor)| {
                    matches!(actor.status, ActorStatus::Runnable).then_some(*id)
                })
                .collect();
            if runnable.is_empty() {
                break;
            }
            for actor in runnable {
                self.step(actor);
            }
        }
    }

    pub fn step(&mut self, actor_id: ActorId) {
        if !matches!(self.actors[&actor_id].status, ActorStatus::Runnable) {
            return;
        }
        let (function_key, pc, instruction) = {
            let actor = &self.actors[&actor_id];
            let frame = actor.frames.last().expect("runnable actor has a frame");
            let FunctionState::Ready(function) = &self.world.functions[&frame.function] else {
                unreachable!("frames only pin ready code");
            };
            (frame.function, frame.pc, function.code[frame.pc].clone())
        };
        if let Err(condition) = self.execute(actor_id, function_key.0, pc, instruction) {
            self.actors.get_mut(&actor_id).unwrap().status = ActorStatus::Paused(condition);
        }
    }

    fn reg(&self, actor: ActorId, index: usize) -> Result<Value, Condition> {
        self.actors[&actor].frames.last().unwrap().registers[index]
            .clone()
            .ok_or_else(|| Condition::RuntimeTypeError {
                function: self.actors[&actor].frames.last().unwrap().function.0,
                pc: self.actors[&actor].frames.last().unwrap().pc,
                message: format!("empty r{index}"),
            })
    }

    fn write_and_advance(&mut self, actor: ActorId, dst: usize, value: Value) {
        let frame = self.frame_mut(actor);
        frame.registers[dst] = Some(value);
        frame.pc += 1;
    }

    fn frame_mut(&mut self, actor: ActorId) -> &mut Frame {
        self.actors
            .get_mut(&actor)
            .unwrap()
            .frames
            .last_mut()
            .unwrap()
    }

    fn execute(
        &mut self,
        actor: ActorId,
        function: DefId,
        pc: usize,
        instruction: Instruction,
    ) -> Result<(), Condition> {
        match instruction {
            Instruction::Const { dst, value } => self.write_and_advance(actor, dst, value),
            Instruction::New {
                dst,
                type_id,
                fields,
            } => {
                let supplied: Vec<(FieldId, Value)> = fields
                    .iter()
                    .map(|(id, reg)| Ok((*id, self.reg(actor, *reg)?)))
                    .collect::<Result<_, Condition>>()?;
                let id = self.jit_new(type_id, &supplied)?;
                self.write_and_advance(actor, dst, Value::Ref(id));
            }
            Instruction::GetField { dst, object, field } => {
                let Value::Ref(id) = self.reg(actor, object)? else {
                    return Err(self.type_error(function, pc, "field access on non-reference"));
                };
                let value = self.jit_get_field(id, field)?;
                self.write_and_advance(actor, dst, value);
            }
            Instruction::Copy { dst, src } => {
                let value = self.reg(actor, src)?;
                self.write_and_advance(actor, dst, value);
            }
            Instruction::AddI64 { dst, left, right } => {
                let (Value::I64(a), Value::I64(b)) =
                    (self.reg(actor, left)?, self.reg(actor, right)?)
                else {
                    return Err(self.type_error(function, pc, ERR_ADD_NON_I64));
                };
                self.write_and_advance(actor, dst, Value::I64(a + b));
            }
            Instruction::SubI64 { dst, left, right } => {
                let (Value::I64(a), Value::I64(b)) =
                    (self.reg(actor, left)?, self.reg(actor, right)?)
                else {
                    return Err(self.type_error(function, pc, ERR_SUB_NON_I64));
                };
                self.write_and_advance(actor, dst, Value::I64(a - b));
            }
            Instruction::MulI64 { dst, left, right } => {
                let (Value::I64(a), Value::I64(b)) =
                    (self.reg(actor, left)?, self.reg(actor, right)?)
                else {
                    return Err(self.type_error(function, pc, ERR_MUL_NON_I64));
                };
                self.write_and_advance(actor, dst, Value::I64(a * b));
            }
            Instruction::LtI64 { dst, left, right } => {
                let (Value::I64(a), Value::I64(b)) =
                    (self.reg(actor, left)?, self.reg(actor, right)?)
                else {
                    return Err(self.type_error(function, pc, ERR_LT_NON_I64));
                };
                self.write_and_advance(actor, dst, Value::Bool(a < b));
            }
            Instruction::EqI64 { dst, left, right } => {
                let (Value::I64(a), Value::I64(b)) =
                    (self.reg(actor, left)?, self.reg(actor, right)?)
                else {
                    return Err(self.type_error(function, pc, ERR_EQ_NON_I64));
                };
                self.write_and_advance(actor, dst, Value::Bool(a == b));
            }
            Instruction::Not { dst, src } => {
                let Value::Bool(b) = self.reg(actor, src)? else {
                    return Err(self.type_error(function, pc, ERR_NOT_NON_BOOL));
                };
                self.write_and_advance(actor, dst, Value::Bool(!b));
            }
            Instruction::Jump { target } => {
                self.frame_mut(actor).pc = target;
            }
            Instruction::Branch {
                cond,
                then_pc,
                else_pc,
            } => {
                let Value::Bool(taken) = self.reg(actor, cond)? else {
                    return Err(self.type_error(function, pc, ERR_BRANCH_NON_BOOL));
                };
                self.frame_mut(actor).pc = if taken { then_pc } else { else_pc };
            }
            Instruction::Yield => {
                // A recurring safe point. Observationally a no-op in the
                // interpreter; the native path uses it to hand control back so
                // a pending update can land between iterations (DESIGN.md T5).
                self.frame_mut(actor).pc += 1;
            }
            Instruction::Call {
                dst,
                function: callee,
                args,
            } => {
                let version = self.world.current_functions[&callee];
                let state = &self.world.functions[&(callee, version)];
                let FunctionState::Ready(code) = state else {
                    let FunctionState::Broken { diagnostics, .. } = state else {
                        unreachable!()
                    };
                    return Err(Condition::BrokenFunction {
                        function: callee,
                        diagnostics: diagnostics.clone(),
                    });
                };
                let params = code.params.clone();
                let registers_len = code.registers;
                let values: Vec<_> = args
                    .into_iter()
                    .map(|r| self.reg(actor, r))
                    .collect::<Result<_, _>>()?;
                for (value, expected) in values.iter().zip(&params) {
                    self.expect_value(value, expected, callee, pc, "call argument")?;
                }
                let mut registers = vec![None; registers_len];
                for (slot, value) in values.into_iter().enumerate() {
                    registers[slot] = Some(value);
                }
                let owner = self.actors.get_mut(&actor).unwrap();
                owner.frames.last_mut().unwrap().pc += 1;
                owner.frames.push(Frame {
                    function: (callee, version),
                    pc: 0,
                    registers,
                    return_to: Some(ReturnTo { register: dst }),
                });
            }
            Instruction::Emit { value } => {
                let value = self.reg(actor, value)?;
                self.jit_emit(value);
                self.frame_mut(actor).pc += 1;
            }
            Instruction::Send { .. } | Instruction::Recv { .. } => {
                return Err(self.type_error(
                    function,
                    pc,
                    "message passing is only available in the concurrent runtime",
                ));
            }
            Instruction::Return { value } => {
                let result = self.reg(actor, value)?;
                // Check the result against this function version's declared type
                // before it leaves the frame: a pinned old function returning a
                // since-migrated value traps here instead of handing a lie to
                // its caller (or completing the actor with one).
                let key = self.actors[&actor].frames.last().unwrap().function;
                if let FunctionState::Ready(f) = &self.world.functions[&key] {
                    let result_ty = f.result.clone();
                    self.expect_value(&result, &result_ty, function, pc, "return value")?;
                }
                let owner = self.actors.get_mut(&actor).unwrap();
                let frame = owner.frames.pop().unwrap();
                match frame.return_to {
                    Some(target) => {
                        owner.frames.last_mut().unwrap().registers[target.register] = Some(result)
                    }
                    None => owner.status = ActorStatus::Complete(result),
                }
            }
        }
        Ok(())
    }

    /// Construct an object at the type's current schema. Supplied fields
    /// override defaults; missing fields fall back to their schema default.
    /// Shared by the interpreter's `New` arm and the native `lt_new` extern so
    /// the two executors allocate identically. `expect` is safe because both
    /// callers only reach here for verified constructors. Each field value is
    /// checked against its declared type (`value_ok`) so a pinned old function
    /// constructing with a since-migrated value traps instead of publishing an
    /// ill-typed object.
    pub fn jit_new(
        &mut self,
        type_id: DefId,
        supplied: &[(FieldId, Value)],
    ) -> Result<ObjectId, Condition> {
        let version = self.world.current_schemas[&type_id];
        let schema = self.world.schemas[&(type_id, version)].clone();
        let mut values = BTreeMap::new();
        for field in &schema.fields {
            let value = supplied
                .iter()
                .find(|(id, _)| *id == field.id)
                .map(|(_, v)| v.clone())
                .or_else(|| field.default.clone())
                .expect("verified constructor");
            self.expect_value(&value, &field.ty, 0, 0, &format!("field '{}'", field.name))?;
            values.insert(field.id, value);
        }
        Ok(self.alloc(type_id, version, values))
    }

    /// The migration barrier: migrate the object up to its type's current
    /// schema (lazily, identity-preserving) and read one field. A migration gap
    /// returns [`Condition::MissingMigration`]. Shared by the interpreter's
    /// `GetField` arm and the native `lt_get_field` extern.
    pub fn jit_get_field(&mut self, id: ObjectId, field: FieldId) -> Result<Value, Condition> {
        self.migrate(id)?;
        self.heap[&id]
            .fields
            .get(&field)
            .cloned()
            .ok_or_else(|| self.type_error(0, 0, "field is absent after migration"))
    }

    /// Commit one external observation. `Emit` advances the frame only after
    /// this appends, so a later pause/resume cannot replay an earlier effect.
    pub fn jit_emit(&mut self, value: Value) {
        self.output.push(value);
    }

    fn alloc(
        &mut self,
        type_id: DefId,
        schema: Version,
        fields: BTreeMap<FieldId, Value>,
    ) -> ObjectId {
        self.next_object += 1;
        let id = self.next_object;
        self.heap.insert(
            id,
            Object {
                id,
                body: std::sync::Arc::new(Body {
                    type_id,
                    schema,
                    fields,
                }),
            },
        );
        id
    }

    /// Transactional migrate-on-read: all replacement bodies and nested wrapper
    /// objects are staged first; the stable handle is changed only after success.
    fn migrate(&mut self, id: ObjectId) -> Result<(), Condition> {
        loop {
            let old = self.heap[&id].clone();
            let current = self.world.current_schemas[&old.type_id];
            if old.schema == current {
                return Ok(());
            }
            let Some(plan) = self
                .world
                .migrations
                .get(&(old.type_id, old.schema))
                .cloned()
            else {
                return Err(Condition::MissingMigration {
                    object: id,
                    type_id: old.type_id,
                    from: old.schema,
                    to: Version(old.schema.0 + 1),
                });
            };
            let mut fields = BTreeMap::new();
            for (target, source) in plan.fields {
                let value = match source {
                    MigrationSource::Copy(source) => old.fields[&source].clone(),
                    MigrationSource::Value(value) => value,
                    MigrationSource::Wrap {
                        type_id,
                        field,
                        source,
                    } => {
                        let version = self.world.current_schemas[&type_id];
                        let wrapped = self.alloc(
                            type_id,
                            version,
                            BTreeMap::from([(field, old.fields[&source].clone())]),
                        );
                        Value::Ref(wrapped)
                    }
                };
                fields.insert(target, value);
            }
            // value_ok on the freshly built body before it is published: the
            // migration plan is validated at install, so this is defense in
            // depth — a body that is not structurally well-typed against the
            // target schema never becomes the object's observable state.
            let target_schema = self.world.schemas[&(old.type_id, plan.to)].clone();
            for field in &target_schema.fields {
                let value = &fields[&field.id];
                if !self.value_ok(value, &field.ty) {
                    return Err(self.type_error(
                        0,
                        0,
                        &format!(
                            "migration to {} v{} produced an ill-typed '{}'",
                            target_schema.name, plan.to.0, field.name
                        ),
                    ));
                }
            }
            self.heap.insert(
                id,
                Object {
                    id,
                    body: std::sync::Arc::new(Body {
                        type_id: old.type_id,
                        schema: plan.to,
                        fields,
                    }),
                },
            );
        }
    }

    fn type_error(&self, function: DefId, pc: usize, message: &str) -> Condition {
        Condition::RuntimeTypeError {
            function,
            pc,
            message: message.into(),
        }
    }

    /// Reconstruct the condition for a native operand-tag trap (`OUT_TYPE_ERROR`)
    /// from the trapping instruction, so it matches the interpreter's exactly.
    pub fn operand_type_error(
        &self,
        function: DefId,
        pc: usize,
        instruction: &Instruction,
    ) -> Condition {
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
        self.type_error(function, pc, message)
    }

    /// The soundness predicate (`value_ok`): does a live value's actual nominal
    /// type match what a definition says it should be? References match at the
    /// nominal type, not the schema version, so a migrated object still matches
    /// a field/param/result typed `Ref(T)` — the check only fires on a genuine
    /// representation confusion (e.g. an `Int` field read as a `Ref` by pinned
    /// old code after a migration changed its type).
    pub fn value_ok(&self, value: &Value, expected: &Type) -> bool {
        value.shallow_type(&self.heap).as_ref() == Some(expected)
    }

    /// Enforce [`Runtime::value_ok`] at a value-use boundary. This is the
    /// tested form of the soundness invariant: rather than trust that every
    /// running frame stays well-typed, we check the value the code is about to
    /// observe and trap (trap-and-repair, quarantining the frame) if a hot
    /// update made it lie. Shared by both executors so they trap identically.
    pub fn expect_value(
        &self,
        value: &Value,
        expected: &Type,
        function: DefId,
        pc: usize,
        what: &str,
    ) -> Result<(), Condition> {
        if self.value_ok(value, expected) {
            Ok(())
        } else {
            Err(self.type_error(
                function,
                pc,
                &format!("{what}: expected {expected:?}, found a value of another type"),
            ))
        }
    }

    /// The type of value a paused actor's con-freeness trap expects, so a
    /// developer knows what to hand back. `None` if the actor isn't paused on a
    /// type trap (or the trap isn't value-resumable, e.g. a migration gap).
    pub fn pause_expected(&self, actor: ActorId) -> Option<Type> {
        let a = self.actors.get(&actor)?;
        if !matches!(a.status, ActorStatus::Paused(Condition::RuntimeTypeError { .. })) {
            return None;
        }
        let frame = a.frames.last()?;
        let FunctionState::Ready(f) = &self.world.functions[&frame.function] else {
            return None;
        };
        resume_shape(&f.code[frame.pc], &f.result, &self.world)
            .ok()
            .map(|(ty, _)| ty)
    }

    /// Resume a con-freeness trap by supplying a value — the delimited-
    /// continuation repair. The frozen instruction "produces" `value` and the
    /// frame continues. The value must satisfy [`Runtime::value_ok`] for the
    /// trap's expected type, so a repair can never reintroduce an ill-typed
    /// value; a wrong-typed offering is rejected and the actor stays paused.
    pub fn resume_with(&mut self, actor: ActorId, value: Value) -> Result<(), String> {
        let a = &self.actors[&actor];
        if !matches!(a.status, ActorStatus::Paused(Condition::RuntimeTypeError { .. })) {
            return Err("actor is not paused on a resumable type trap".into());
        }
        let key = a.frames.last().unwrap().function;
        let pc = a.frames.last().unwrap().pc;
        let FunctionState::Ready(f) = &self.world.functions[&key] else {
            return Err("frame pins non-ready code".into());
        };
        let result_ty = f.result.clone();
        let instruction = f.code[pc].clone();
        let (expected, plan) = resume_shape(&instruction, &result_ty, &self.world)?;
        if !self.value_ok(&value, &expected) {
            return Err(format!(
                "supplied value does not have the expected type {expected:?}"
            ));
        }
        match plan {
            ResumePlan::SetAdvance(dst) => {
                let frame = self.frame_mut(actor);
                frame.registers[dst] = Some(value);
                frame.pc += 1;
                self.actors.get_mut(&actor).unwrap().status = ActorStatus::Runnable;
            }
            ResumePlan::Branch(then_pc, else_pc) => {
                let take = matches!(value, Value::Bool(true));
                self.frame_mut(actor).pc = if take { then_pc } else { else_pc };
                self.actors.get_mut(&actor).unwrap().status = ActorStatus::Runnable;
            }
            ResumePlan::ReturnValue => {
                let owner = self.actors.get_mut(&actor).unwrap();
                let frame = owner.frames.pop().unwrap();
                match frame.return_to {
                    Some(target) => {
                        owner.frames.last_mut().unwrap().registers[target.register] = Some(value);
                        owner.status = ActorStatus::Runnable;
                    }
                    None => owner.status = ActorStatus::Complete(value),
                }
            }
        }
        Ok(())
    }

    fn resume_repaired(&mut self) {
        for actor in self.actors.values_mut() {
            let repaired = match &actor.status {
                ActorStatus::Paused(Condition::BrokenFunction { function, .. }) => {
                    self.world.current_functions.get(function).is_some_and(|v| {
                        matches!(
                            self.world.functions.get(&(*function, *v)),
                            Some(FunctionState::Ready(_))
                        )
                    })
                }
                ActorStatus::Paused(Condition::MissingMigration { type_id, from, .. }) => {
                    self.world.migrations.contains_key(&(*type_id, *from))
                }
                _ => false,
            };
            if repaired {
                actor.status = ActorStatus::Runnable;
            }
        }
    }

    /// A schema update is allowed to land even when it makes callers invalid.
    /// Ready artifacts remain available to already-running pinned frames, while
    /// a new broken version becomes the call target for future entries.
    ///
    /// Demand-driven (D7): re-verify only the functions this schema change can
    /// reach, not every current function. The worklist is seeded with functions
    /// whose type-dependency set contains the `changed` type; whenever one is
    /// newly broken, its callers are enqueued, since a call to a broken function
    /// no longer verifies. This propagates brokenness through the call graph to
    /// a fixpoint (a caller re-checked before its callee broke used to be missed
    /// by the old single map-order pass). A function's pinned registers may
    /// still hold values of the changed type; that is the use-site soundness
    /// checks' job, not this pass's.
    fn invalidate_functions(&mut self, changed: DefId) {
        let mut work: Vec<DefId> = self
            .world
            .current_functions
            .keys()
            .filter(|id| {
                self.world
                    .function_deps
                    .get(id)
                    .is_some_and(|deps| deps.contains(&changed))
            })
            .copied()
            .collect();
        let mut seen: BTreeSet<DefId> = work.iter().copied().collect();

        while let Some(id) = work.pop() {
            let version = self.world.current_functions[&id];
            let Some(FunctionState::Ready(function)) = self.world.functions.get(&(id, version))
            else {
                continue; // already broken (or gone): nothing to re-verify
            };
            let function = function.clone();
            let Err(diagnostics) = verify_function(&function, &self.world) else {
                continue; // still well-typed against the new definitions
            };
            let broken = Version(version.0 + 1);
            self.world.functions.insert(
                (id, broken),
                FunctionState::Broken {
                    id,
                    version: broken,
                    name: function.name,
                    diagnostics,
                },
            );
            self.world.current_functions.insert(id, broken);
            // Enqueue callers of the now-broken function.
            let callers: Vec<DefId> = self
                .world
                .current_functions
                .iter()
                .filter(|(caller, _)| !seen.contains(caller))
                .filter_map(|(caller, cver)| match self.world.functions.get(&(*caller, *cver)) {
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

    /// Precise collection using explicit frame registers and object fields as
    /// the complete root graph. This is the same root contract LLVM lowering
    /// must preserve at every step boundary.
    pub fn collect_garbage(&mut self) -> usize {
        self.collect_garbage_with_roots(&[])
    }

    /// Precise collection with additional roots supplied by the caller — used
    /// by the JIT driver to hand over the [`ObjectId`]s it reads out of native
    /// [`crate::RawSlot`] frame slots. This is the same complete-root-map
    /// contract the interpreter relies on, proven for the native path: every
    /// live reference is a typed frame slot, nothing hides in a register the GC
    /// cannot see.
    pub fn collect_garbage_with_roots(&mut self, extra_roots: &[ObjectId]) -> usize {
        let mut work: Vec<ObjectId> = extra_roots.to_vec();
        for actor in self.actors.values() {
            for frame in &actor.frames {
                for value in frame.registers.iter().flatten() {
                    if let Value::Ref(id) = value {
                        work.push(*id);
                    }
                }
            }
        }
        let mut live = std::collections::BTreeSet::new();
        while let Some(id) = work.pop() {
            if !live.insert(id) {
                continue;
            }
            if let Some(object) = self.heap.get(&id) {
                for value in object.fields.values() {
                    if let Value::Ref(child) = value {
                        work.push(*child);
                    }
                }
            }
        }
        let before = self.heap.len();
        self.heap.retain(|id, _| live.contains(id));
        before - self.heap.len()
    }
}
