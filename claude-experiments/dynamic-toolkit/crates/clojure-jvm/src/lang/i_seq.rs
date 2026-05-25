//! Port of `clojure.lang.ISeq`.
//!
//! Source: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/ISeq.java`
//!
//! ```text
//! public interface ISeq extends IPersistentCollection {
//!     Object first();
//!     ISeq next();
//!     ISeq more();
//!     ISeq cons(Object o);
//! }
//! ```
//!
//! In Java a "consumed" seq returns `null` from `next()`; here `next()` returns
//! `None`. `more()` returns the empty seq rather than null (matches Java's
//! behavior where `more` is `next` coerced to empty-rather-than-null).

use std::sync::Arc;

use super::object::Object;

/// `clojure.lang.ISeq`. The persistent sequence interface.
pub trait ISeq: std::fmt::Debug + Send + Sync {
    /// Java: `Object first()`.
    fn first(&self) -> Object;

    /// Java: `ISeq next()`. Returns `None` once exhausted (Java's `null`).
    fn next(&self) -> Option<Arc<dyn ISeq>>;

    /// Java: `ISeq more()`. Like `next()` but returns the empty seq when
    /// exhausted instead of `null`.
    fn more(&self) -> Arc<dyn ISeq>;

    /// Java: `ISeq cons(Object o)`. Prepend an element; returns a new seq.
    fn cons(&self, o: Object) -> Arc<dyn ISeq>;

    /// Java: `int count()` from `IPersistentCollection`. Convenience method
    /// (concrete types may cache this).
    fn count(&self) -> i32;
}
