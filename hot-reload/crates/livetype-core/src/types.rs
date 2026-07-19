//! The type-intern table: a process-global registry giving structural
//! [`Type`]s a stable integer identity, so compiled native code can name a
//! type across the C ABI (e.g. `lt_new_array` carries its element type as a
//! [`TypeId`]). Interning dedups, so equal types have equal ids; entries are
//! immortal, exactly like the string table — an id embedded in compiled code
//! can never dangle.

use crate::Type;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub type TypeId = u64;

#[derive(Default)]
struct Table {
    by_type: HashMap<Type, TypeId>,
    by_id: Vec<Type>,
}

fn table() -> &'static Mutex<Table> {
    static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(Table::default()))
}

/// Intern `ty`, returning its stable id (equal types ⇒ equal ids).
pub fn intern(ty: &Type) -> TypeId {
    let mut t = table().lock().unwrap();
    if let Some(id) = t.by_type.get(ty) {
        return *id;
    }
    let id = t.by_id.len() as TypeId;
    t.by_id.push(ty.clone());
    t.by_type.insert(ty.clone(), id);
    id
}

/// The type behind an interned id. Ids only come from [`intern`], so an
/// unknown id is a runtime-integrity bug and a hard error.
pub fn get(id: TypeId) -> Type {
    let t = table().lock().unwrap();
    t.by_id
        .get(id as usize)
        .cloned()
        .unwrap_or_else(|| panic!("unknown TypeId {id} — not produced by intern()"))
}
