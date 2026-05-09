//! `tracing_subscriber::Layer` that captures spans + events and posts
//! them into cf-runtime's `EventLog`. Each span event also carries the
//! tokio task id that was polling when the span was entered (read from
//! our hooks' thread-local). That id is what bridges the runtime layer
//! (poll spans on worker lanes) with the application layer (semantic
//! operations like ResolveModule / ParseJs / Transform).

use cf_runtime::EventKind;
use std::fmt::Write;
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

/// Layer producing cf-runtime events for every span enter/exit/close
/// and every `tracing::event!()` invocation. Stateless — span metadata
/// is read from `tracing`'s registry via the `Context`.
pub struct CfLayer;

impl CfLayer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CfLayer {
    fn default() -> Self {
        Self
    }
}

impl<S> Layer<S> for CfLayer
where
    S: Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        // Stash the recorded fields on the span itself so on_enter can
        // include them in the event. We use the registry's extensions.
        let mut visitor = FieldVisitor::new();
        attrs.record(&mut visitor);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert::<RecordedFields>(RecordedFields(visitor.0));
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut ext = span.extensions_mut();
            let entry = ext.get_mut::<RecordedFields>();
            let buf = match entry {
                Some(r) => &mut r.0,
                None => {
                    ext.insert::<RecordedFields>(RecordedFields(String::new()));
                    &mut ext.get_mut::<RecordedFields>().unwrap().0
                }
            };
            let mut visitor = FieldVisitor::with_buf(std::mem::take(buf));
            values.record(&mut visitor);
            *buf = visitor.0;
        }
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(observer) = cf_runtime::hooks::current() else {
            return;
        };
        let Some(span) = ctx.span(id) else { return };
        let metadata = span.metadata();
        if is_noise(metadata.name(), metadata.target()) {
            return;
        }
        // Stash an allocation snapshot so on_exit can compute the delta.
        // Spans can be entered multiple times (turbopack uses
        // task_local::scope which re-enters); use remove+insert so we
        // don't trip tracing's "duplicate type" assertion.
        if let Some(snap) = cf_runtime::hooks::alloc_snapshot() {
            let mut ext = span.extensions_mut();
            ext.remove::<AllocSnap>();
            ext.insert::<AllocSnap>(AllocSnap(snap));
        }
        let parent_id = span.parent().map(|p| p.id().into_u64());
        let fields = span
            .extensions()
            .get::<RecordedFields>()
            .map(|r| r.0.clone())
            .unwrap_or_default();
        let task = cf_runtime::hooks::polling_tokio_id().and_then(|tid| {
            // Map the tokio id to our cf TaskId via the same mapping
            // hooks use; if unavailable, still record without a task
            // attribution.
            map_tokio_to_cf(tid)
        });
        observer.log.push(
            task,
            cf_runtime::hooks::current_worker(),
            EventKind::SpanEnter {
                span_id: id.into_u64(),
                name: metadata.name(),
                target: metadata.target(),
                parent_id,
                fields,
            },
        );
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(observer) = cf_runtime::hooks::current() else {
            return;
        };
        // Match the on_enter filter: don't emit Exit for spans we
        // dropped at Enter time, otherwise the log fills with orphan
        // exits.
        let span_meta = ctx.span(id);
        if let Some(span) = &span_meta {
            let m = span.metadata();
            if is_noise(m.name(), m.target()) {
                return;
            }
        }
        // Compute allocation delta and emit a SpanAllocs event before
        // the SpanExit. Skip if we have no enter snapshot (means alloc
        // tracking wasn't installed when we entered).
        if let Some(span) = &span_meta {
            if let Some(snap) = cf_runtime::hooks::alloc_snapshot() {
                let mut ext = span.extensions_mut();
                if let Some(start) = ext.remove::<AllocSnap>() {
                    let (a0, d0, ac0, dc0) = start.0;
                    let (a1, d1, ac1, dc1) = snap;
                    let bytes_delta = (a1 as i64 - d1 as i64) - (a0 as i64 - d0 as i64);
                    let count_delta = (ac1 as i64 - dc1 as i64) - (ac0 as i64 - dc0 as i64);
                    if bytes_delta != 0 || count_delta != 0 {
                        observer.log.push(
                            cf_runtime::hooks::polling_tokio_id()
                                .and_then(map_tokio_to_cf),
                            cf_runtime::hooks::current_worker(),
                            cf_runtime::EventKind::SpanAllocs {
                                span_id: id.into_u64(),
                                bytes_delta,
                                count_delta,
                            },
                        );
                    }
                }
            }
        }
        let task = cf_runtime::hooks::polling_tokio_id().and_then(map_tokio_to_cf);
        observer.log.push(
            task,
            cf_runtime::hooks::current_worker(),
            EventKind::SpanExit {
                span_id: id.into_u64(),
            },
        );
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        let Some(observer) = cf_runtime::hooks::current() else {
            return;
        };
        observer.log.push(
            None,
            None,
            EventKind::SpanClose {
                span_id: id.into_u64(),
            },
        );
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let Some(observer) = cf_runtime::hooks::current() else {
            return;
        };
        let metadata = event.metadata();
        let mut visitor = FieldVisitor::new();
        event.record(&mut visitor);
        let in_span = ctx.event_span(event).map(|s| s.id().into_u64());
        let task = cf_runtime::hooks::polling_tokio_id().and_then(map_tokio_to_cf);
        observer.log.push(
            task,
            cf_runtime::hooks::current_worker(),
            EventKind::SpanEvent {
                target: metadata.target(),
                level: metadata.level().as_str(),
                message: visitor.0,
                in_span,
            },
        );
    }
}

/// Per-span buffer of the textual representation of recorded fields.
struct RecordedFields(String);

/// Per-span captured allocation snapshot at on_enter, consumed at
/// on_exit to compute the delta.
struct AllocSnap((u64, u64, u64, u64));

/// Field visitor that flattens `tracing` field values into a "k=v"
/// string. Cheap; allocates one String per span enter event.
struct FieldVisitor(String);
impl FieldVisitor {
    fn new() -> Self {
        Self(String::new())
    }
    fn with_buf(s: String) -> Self {
        Self(s)
    }
}
impl tracing::field::Visit for FieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if !self.0.is_empty() {
            self.0.push(' ');
        }
        let _ = write!(self.0, "{}={:?}", field.name(), value);
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if !self.0.is_empty() {
            self.0.push(' ');
        }
        let _ = write!(self.0, "{}={:?}", field.name(), value);
    }
}

/// Look up a tokio task id in our hooks' map (which records every
/// `Spawned` event). Hooks expose a `current()` for the observer but
/// not the map directly — we expose `recently_completed_lookup` for
/// post-completion lookup, and add a parallel `lookup_tokio_id` for
/// in-flight tasks.
fn map_tokio_to_cf(tokio_id: u64) -> Option<cf_runtime::TaskId> {
    cf_runtime::hooks::task_id_for_tokio_id(tokio_id)
}

/// Spans we don't want clogging the timeline. Most are SWC AST visitors
/// that fire per-node — they're useful for SWC profiling but drown out
/// turbopack's semantic spans. Override with CF_TRACING_KEEP_ALL=1 if
/// somebody actually wants them.
fn is_noise(name: &str, target: &str) -> bool {
    if std::env::var("CF_TRACING_KEEP_ALL").ok().as_deref() == Some("1") {
        return false;
    }
    matches!(
        name,
        "visit_mut_expr"
            | "visit_mut_stmt"
            | "visit_mut_stmts"
            | "visit_mut_block_stmt"
            | "visit_mut_block_stmt_or_expr"
            | "visit_mut_prop_name"
            | "visit_mut_pat"
            | "visit_mut_module_item"
            | "visit_mut_module_decl"
            | "visit_mut_module"
    ) || target.starts_with("swc_ecma_visit")
}
