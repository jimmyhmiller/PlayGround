use crate::*;
use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct Runtime {
    pub world: World,
    pub heap: BTreeMap<ObjectId, Object>,
    pub actors: BTreeMap<ActorId, Actor>,
    /// Committed external observations. `Emit` advances the frame only after
    /// appending, so pausing and resuming cannot replay an earlier effect.
    pub output: Vec<Value>,
    next_object: ObjectId,
    next_actor: ActorId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallError {
    BadVersion,
    Invalid(Vec<String>),
}

impl Runtime {
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
        self.world
            .current_schemas
            .insert(schema.type_id, schema.version);
        self.world
            .schemas
            .insert((schema.type_id, schema.version), schema);
        self.invalidate_functions();
        self.world.epoch += 1;
        Ok(())
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
            Ok(()) => FunctionState::Ready(function),
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
        let frame = self
            .actors
            .get_mut(&actor)
            .unwrap()
            .frames
            .last_mut()
            .unwrap();
        frame.registers[dst] = Some(value);
        frame.pc += 1;
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
                let version = self.world.current_schemas[&type_id];
                let schema = &self.world.schemas[&(type_id, version)];
                let mut values = BTreeMap::new();
                for field in &schema.fields {
                    let value = fields
                        .iter()
                        .find(|(id, _)| *id == field.id)
                        .map(|(_, reg)| self.reg(actor, *reg))
                        .transpose()?
                        .or_else(|| field.default.clone())
                        .expect("verified constructor");
                    values.insert(field.id, value);
                }
                let id = self.alloc(type_id, version, values);
                self.write_and_advance(actor, dst, Value::Ref(id));
            }
            Instruction::GetField { dst, object, field } => {
                let Value::Ref(id) = self.reg(actor, object)? else {
                    return Err(self.type_error(function, pc, "field access on non-reference"));
                };
                self.migrate(id)?;
                let value = self.heap[&id]
                    .fields
                    .get(&field)
                    .cloned()
                    .ok_or_else(|| self.type_error(function, pc, "field is absent"))?;
                self.write_and_advance(actor, dst, value);
            }
            Instruction::SubI64 { dst, left, right } => {
                let (Value::I64(a), Value::I64(b)) =
                    (self.reg(actor, left)?, self.reg(actor, right)?)
                else {
                    return Err(self.type_error(function, pc, "subtraction on non-i64"));
                };
                self.write_and_advance(actor, dst, Value::I64(a - b));
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
                let values: Vec<_> = args
                    .into_iter()
                    .map(|r| self.reg(actor, r))
                    .collect::<Result<_, _>>()?;
                let mut registers = vec![None; code.registers];
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
                self.output.push(value);
                self.actors
                    .get_mut(&actor)
                    .unwrap()
                    .frames
                    .last_mut()
                    .unwrap()
                    .pc += 1;
            }
            Instruction::Return { value } => {
                let result = self.reg(actor, value)?;
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
                type_id,
                schema,
                fields,
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
            self.heap.insert(
                id,
                Object {
                    id,
                    type_id: old.type_id,
                    schema: plan.to,
                    fields,
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
    fn invalidate_functions(&mut self) {
        let current: Vec<_> = self
            .world
            .current_functions
            .iter()
            .filter_map(
                |(id, version)| match self.world.functions.get(&(*id, *version)) {
                    Some(FunctionState::Ready(function)) => Some(function.clone()),
                    _ => None,
                },
            )
            .collect();
        for function in current {
            if let Err(diagnostics) = verify_function(&function, &self.world) {
                let version = Version(function.version.0 + 1);
                self.world.functions.insert(
                    (function.id, version),
                    FunctionState::Broken {
                        id: function.id,
                        version,
                        name: function.name,
                        diagnostics,
                    },
                );
                self.world.current_functions.insert(function.id, version);
            }
        }
    }

    /// Precise collection using explicit frame registers and object fields as
    /// the complete root graph. This is the same root contract LLVM lowering
    /// must preserve at every step boundary.
    pub fn collect_garbage(&mut self) -> usize {
        let mut work = Vec::new();
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
