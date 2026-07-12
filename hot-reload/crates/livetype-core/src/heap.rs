//! The one heap. A single thread-safe object store shared by every executor:
//! the single-threaded interpreter, the JIT (through its externs), and the
//! concurrent tier. The object model is the design's "moving bodies behind
//! non-moving handles": a stable [`ObjectId`] names an [`ObjCell`] whose
//! `Arc<Body>` can be atomically swapped by a migration, so a reader always sees
//! a consistent whole body and old bodies are reclaimed by refcount.
//!
//! Migration, allocation, the soundness predicate (`value_ok`), and the sweep
//! all live here exactly once — the interpreter and the concurrent tier can no
//! longer drift on them. The single-threaded caller simply runs uncontended
//! against the same locks the concurrent tier relies on (an accepted cost we can
//! optimize to an atomic `Arc` swap later; see `UNIFICATION.md`).

use crate::{
    Body, Condition, DefId, FieldId, MigrationSource, ObjectId, Type, Value, Version, World,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// One heap slot: a stable handle whose body can be atomically swapped. The
/// `Mutex<Arc<Body>>` gives a reader a consistent whole body (clone the `Arc`,
/// then read) and a migrator an atomic swap; the old `Arc` lives until its last
/// reader drops it.
struct ObjCell {
    body: Mutex<Arc<Body>>,
}

/// The shared object store. Cheap to move (single-threaded callers own it
/// directly); safe to share behind an `Arc` (the concurrent tier).
#[derive(Default)]
pub struct Heap {
    objects: Mutex<BTreeMap<ObjectId, Arc<ObjCell>>>,
    next_object: AtomicU64,
}

fn type_error(message: &str) -> Condition {
    Condition::RuntimeTypeError {
        function: 0,
        pc: 0,
        message: message.into(),
    }
}

impl Heap {
    pub fn new() -> Heap {
        Heap::default()
    }

    /// Rebuild a heap with its id counter seeded to `seed` — used when handing a
    /// set-up world to the concurrent tier so new allocations don't collide with
    /// existing ids.
    pub fn with_seed(seed: ObjectId) -> Heap {
        Heap {
            objects: Mutex::new(BTreeMap::new()),
            next_object: AtomicU64::new(seed),
        }
    }

    fn cell(&self, id: ObjectId) -> Option<Arc<ObjCell>> {
        self.objects.lock().unwrap().get(&id).cloned()
    }

    /// Allocate an object with the given body. Non-moving: the returned handle
    /// never changes even as the body is later migrated.
    pub fn alloc(
        &self,
        type_id: DefId,
        schema: Version,
        fields: BTreeMap<FieldId, Value>,
    ) -> ObjectId {
        let id = self.next_object.fetch_add(1, Ordering::Relaxed) + 1;
        let cell = Arc::new(ObjCell {
            body: Mutex::new(Arc::new(Body {
                type_id,
                schema,
                fields,
            })),
        });
        self.objects.lock().unwrap().insert(id, cell);
        id
    }

    /// A consistent snapshot of one object's body (the `Arc` is cloned, so it
    /// stays valid even if the object migrates immediately after).
    pub fn body(&self, id: ObjectId) -> Option<Arc<Body>> {
        Some(self.cell(id)?.body.lock().unwrap().clone())
    }

    pub fn type_id(&self, id: ObjectId) -> Option<DefId> {
        Some(self.body(id)?.type_id)
    }

    pub fn schema_version(&self, id: ObjectId) -> Option<Version> {
        Some(self.body(id)?.schema)
    }

    pub fn contains(&self, id: ObjectId) -> bool {
        self.objects.lock().unwrap().contains_key(&id)
    }

    pub fn len(&self) -> usize {
        self.objects.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The nominal (shallow) type of a value — a reference resolves to its
    /// object's current `type_id`. This is the one place references are typed.
    pub fn shallow_type(&self, value: &Value) -> Option<Type> {
        match value {
            Value::Ref(id) => Some(Type::Ref(self.type_id(*id)?)),
            other => other.scalar_type(),
        }
    }

    /// The soundness predicate: does a live value's actual nominal type match
    /// what a definition says it should be? References match at the nominal type,
    /// not the schema version, so a migrated object still matches `Ref(T)`.
    pub fn value_ok(&self, value: &Value, expected: &Type) -> bool {
        self.shallow_type(value).as_ref() == Some(expected)
    }

    /// The reference children of an object, for GC tracing.
    pub fn child_refs(&self, id: ObjectId) -> Vec<ObjectId> {
        match self.body(id) {
            Some(body) => body
                .fields
                .values()
                .filter_map(|v| match v {
                    Value::Ref(child) => Some(*child),
                    _ => None,
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// Sweep: keep only the objects in `live`; return how many were reclaimed.
    pub fn retain(&self, live: &BTreeSet<ObjectId>) -> usize {
        let mut objects = self.objects.lock().unwrap();
        let before = objects.len();
        objects.retain(|id, _| live.contains(id));
        before - objects.len()
    }

    /// Construct an object at the type's current schema. Supplied fields override
    /// defaults; each field value is checked against its declared type so a
    /// pinned old constructor with a since-migrated value traps rather than
    /// publishing an ill-typed object. Shared by both executors and the JIT's
    /// `lt_new` extern, so allocation is identical everywhere.
    pub fn new_object(
        &self,
        type_id: DefId,
        supplied: &[(FieldId, Value)],
        world: &World,
    ) -> Result<ObjectId, Condition> {
        let version = world.current_schemas[&type_id];
        let schema = world.schemas[&(type_id, version)].clone();
        let mut values = BTreeMap::new();
        for field in &schema.fields {
            let value = supplied
                .iter()
                .find(|(id, _)| *id == field.id)
                .map(|(_, v)| v.clone())
                .or_else(|| field.default.clone())
                .expect("verified constructor");
            if !self.value_ok(&value, &field.ty) {
                return Err(type_error(&format!(
                    "field '{}': expected {:?}, found a value of another type",
                    field.name, field.ty
                )));
            }
            values.insert(field.id, value);
        }
        Ok(self.alloc(type_id, version, values))
    }

    /// The migration barrier: migrate an object up to its type's current schema
    /// (lazily, identity-preserving) — a chain of versions is crossed in one
    /// call. Concurrency-safe: each replacement body is built *without* holding
    /// the object's lock (so allocating wrapper objects can't deadlock against
    /// the heap-table lock), then swapped in under the lock with a double-check
    /// — if another thread already advanced this object, the freshly built body
    /// is discarded and its wrapper allocations become garbage the GC reclaims.
    /// A missing step is a [`Condition::MissingMigration`] gap.
    pub fn migrate(&self, id: ObjectId, world: &World) -> Result<(), Condition> {
        let cell = self.cell(id).ok_or_else(|| type_error("migrate: unknown object"))?;
        loop {
            let body = cell.body.lock().unwrap().clone();
            let current = world.current_schemas[&body.type_id];
            if body.schema == current {
                return Ok(());
            }
            let Some(plan) = world.migrations.get(&(body.type_id, body.schema)).cloned() else {
                return Err(Condition::MissingMigration {
                    object: id,
                    type_id: body.type_id,
                    from: body.schema,
                    to: Version(body.schema.0 + 1),
                });
            };
            let mut fields = BTreeMap::new();
            for (target, source) in &plan.fields {
                let value = match source {
                    MigrationSource::Copy(s) => body.fields[s].clone(),
                    MigrationSource::Value(v) => v.clone(),
                    MigrationSource::Wrap {
                        type_id,
                        field,
                        source,
                    } => {
                        let v = world.current_schemas[type_id];
                        let wid = self.alloc(
                            *type_id,
                            v,
                            BTreeMap::from([(*field, body.fields[source].clone())]),
                        );
                        Value::Ref(wid)
                    }
                };
                fields.insert(*target, value);
            }
            // Defense in depth: the migration plan is validated at install, so a
            // structurally ill-typed body should be impossible — but check it
            // against the target schema before it ever becomes observable.
            let target_schema = &world.schemas[&(body.type_id, plan.to)];
            for field in &target_schema.fields {
                if !self.value_ok(&fields[&field.id], &field.ty) {
                    return Err(type_error(&format!(
                        "migration to {} v{} produced an ill-typed '{}'",
                        target_schema.name, plan.to.0, field.name
                    )));
                }
            }
            let next = Arc::new(Body {
                type_id: body.type_id,
                schema: plan.to,
                fields,
            });
            {
                let mut slot = cell.body.lock().unwrap();
                if slot.schema == body.schema {
                    *slot = next;
                }
            }
            // Re-read: continue the chain or observe the now-current object.
        }
    }

    /// Migrate to current then read one field. Shared by the interpreter's
    /// `GetField` arm, the JIT's `lt_get_field` extern, and the concurrent tier.
    pub fn get_field(&self, id: ObjectId, field: FieldId, world: &World) -> Result<Value, Condition> {
        self.migrate(id, world)?;
        self.body(id)
            .and_then(|b| b.fields.get(&field).cloned())
            .ok_or_else(|| type_error("field is absent after migration"))
    }

    /// A by-value snapshot of the whole heap — for equality and inspection in
    /// tests and diagnostics (never on a hot path).
    pub fn snapshot(&self) -> BTreeMap<ObjectId, Body> {
        self.objects
            .lock()
            .unwrap()
            .iter()
            .map(|(id, cell)| (*id, (**cell.body.lock().unwrap()).clone()))
            .collect()
    }

    /// The next object id that will be allocated — used to seed the concurrent
    /// tier's counter when a set-up world is frozen.
    pub fn next_seed(&self) -> ObjectId {
        self.next_object.load(Ordering::Relaxed)
    }
}

impl PartialEq for Heap {
    fn eq(&self, other: &Self) -> bool {
        self.snapshot() == other.snapshot()
    }
}
impl Eq for Heap {}

impl std::fmt::Debug for Heap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.snapshot().fmt(f)
    }
}
