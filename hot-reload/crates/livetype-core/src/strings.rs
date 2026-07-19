//! The string table. Strings are immutable and **interned process-wide**: a
//! [`StrId`] names a unique text, so equality is id equality and a string value
//! stays a plain `Copy` scalar (`Value::Str`) — one machine word in a JIT slot,
//! nothing for the GC to trace, nothing a migration can tear.
//!
//! Interned strings are immortal (the table only grows). That is a deliberate
//! research-prototype tradeoff: it keeps every tier trivially sound (a `StrId`
//! in a pinned native frame can never dangle) at the cost of leaking unique
//! string contents. A collected string heap can replace this table behind the
//! same two functions if it ever matters.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// The identity of an interned string. Ids are dense and start at 0; equal
/// text always has an equal id (interning dedups), so `==` on strings is `==`
/// on ids — in every tier.
pub type StrId = u64;

#[derive(Default)]
struct Table {
    by_text: HashMap<Arc<str>, StrId>,
    by_id: Vec<Arc<str>>,
}

fn table() -> &'static Mutex<Table> {
    static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(Table::default()))
}

/// Intern `text`, returning its (stable, content-unique) id.
pub fn intern(text: &str) -> StrId {
    let mut t = table().lock().unwrap();
    if let Some(id) = t.by_text.get(text) {
        return *id;
    }
    let id = t.by_id.len() as StrId;
    let arc: Arc<str> = Arc::from(text);
    t.by_id.push(Arc::clone(&arc));
    t.by_text.insert(arc, id);
    id
}

/// The text behind an interned id. A [`StrId`] can only come from
/// [`intern`]/[`concat`], so an unknown id is a runtime-integrity bug and a
/// hard error, never a silent empty string.
pub fn text(id: StrId) -> Arc<str> {
    let t = table().lock().unwrap();
    t.by_id
        .get(id as usize)
        .cloned()
        .unwrap_or_else(|| panic!("unknown StrId {id} — not produced by intern()"))
}

/// Concatenate two interned strings, interning (and deduping) the result.
pub fn concat(left: StrId, right: StrId) -> StrId {
    let joined = {
        let t = table().lock().unwrap();
        let l = t
            .by_id
            .get(left as usize)
            .unwrap_or_else(|| panic!("unknown StrId {left} — not produced by intern()"));
        let r = t
            .by_id
            .get(right as usize)
            .unwrap_or_else(|| panic!("unknown StrId {right} — not produced by intern()"));
        format!("{l}{r}")
    };
    intern(&joined)
}
