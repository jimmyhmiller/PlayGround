//! The runtime drives event → handler invocation → write apply → effect
//! fulfillment → log appends, in a sequential loop.

use thiserror::Error;

use ir::manifest::Manifest;

use crate::body::{fresh_emit_id_counter, BodyCtx, BodyError, EmitIdCounter, NativeBodyRegistry};
use crate::effect::{AdapterRegistry, AdapterResult};
use crate::generator::GeneratorRegistry;
use crate::log::{EffectOutcome, EventLog, LogEntryKind, WriteRecord};
use crate::scheduler::{EventQueue, InboundEvent};
use crate::state::StateStore;
use crate::value::Value;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("manifest validation failed: {0}")]
    Validation(#[from] ir::validate::ValidationError),
    #[error("body error in handler `{handler}`: {source}")]
    Body {
        handler: String,
        #[source]
        source: BodyError,
    },
    #[error("no body registered for uri `{0}`")]
    BodyMissing(String),
    #[error("no adapter registered for effect `{0}`")]
    AdapterMissing(String),
    #[error("event `{0}` is not declared in the manifest")]
    UnknownEvent(String),
    #[error("no generator registered for `{0}`")]
    GeneratorMissing(String),
}

pub struct Runtime {
    pub manifest: Manifest,
    pub state: StateStore,
    pub log: EventLog,
    pub bodies: NativeBodyRegistry,
    pub adapters: AdapterRegistry,
    pub generators: GeneratorRegistry,
    queue: EventQueue,
    next_event_id: u64,
    emit_id_counter: EmitIdCounter,
    arrival_ts: u64,
}

impl Runtime {
    pub fn new(manifest: Manifest) -> Result<Self, RuntimeError> {
        ir::validate::validate(&manifest)?;
        let state = StateStore::from_manifest(&manifest);
        Ok(Self {
            manifest,
            state,
            log: EventLog::default(),
            bodies: NativeBodyRegistry::default(),
            adapters: AdapterRegistry::default(),
            generators: GeneratorRegistry::default(),
            queue: EventQueue::default(),
            next_event_id: 0,
            emit_id_counter: fresh_emit_id_counter(),
            arrival_ts: 0,
        })
    }

    /// Spawn one thread per declared generator. Each thread pulls
    /// `Some(payload)` from its generator and pushes events onto the queue;
    /// when `None` is returned it marks the generator finished and exits.
    /// `run_until_idle` then returns once the queue is empty AND all
    /// generators have finished.
    pub fn start_generators(&mut self) -> Result<(), RuntimeError> {
        let names: Vec<String> = self.manifest.generators.keys().cloned().collect();
        for name in names {
            let decl = self.manifest.generators.get(&name).cloned().ok_or_else(|| {
                RuntimeError::GeneratorMissing(name.clone())
            })?;
            let mut gen = self
                .generators
                .take(&name)
                .ok_or_else(|| RuntimeError::GeneratorMissing(name.clone()))?;
            if !self.manifest.events.contains_key(&decl.event) {
                return Err(RuntimeError::UnknownEvent(decl.event));
            }
            let queue = self.queue.clone();
            let event_name = decl.event.clone();
            queue.generator_started();
            std::thread::spawn(move || {
                while let Some(payload) = gen.next() {
                    queue.push_back(InboundEvent::new(event_name.clone(), payload));
                }
                queue.generator_finished();
            });
        }
        Ok(())
    }

    pub fn enqueue(&mut self, event: InboundEvent) -> Result<(), RuntimeError> {
        if !self.manifest.events.contains_key(&event.event) {
            return Err(RuntimeError::UnknownEvent(event.event));
        }
        self.queue.push_back(event);
        Ok(())
    }

    pub fn run_to_quiescence(&mut self) -> Result<(), RuntimeError> {
        while let Some(ev) = self.queue.pop_front() {
            self.step(ev)?;
        }
        Ok(())
    }

    /// Block-and-process until the queue is empty AND every registered
    /// generator has finished. Use this when generators are pushing events
    /// from background threads.
    pub fn run_until_idle(&mut self) -> Result<(), RuntimeError> {
        while let Some(ev) = self.queue.pop_blocking() {
            self.step(ev)?;
        }
        Ok(())
    }

    /// Shared handle to the event queue. Generators (or any other event
    /// producer) clone this to push events onto the same queue the scheduler
    /// pops from.
    pub fn queue_handle(&self) -> EventQueue {
        self.queue.clone()
    }

    fn step(&mut self, ev: InboundEvent) -> Result<(), RuntimeError> {
        let event_id = self.next_event_id;
        self.next_event_id += 1;
        self.arrival_ts += 1;
        let arrival_ts = self.arrival_ts;

        self.log.append(LogEntryKind::EventEnqueued {
            event_id,
            event: ev.event.clone(),
            payload: ev.payload.clone(),
        });

        // Snapshot handler indices first to release the immutable borrow before
        // we start invoking bodies (which need mutable access to bodies/state).
        let handler_indices: Vec<usize> = self
            .manifest
            .handlers
            .iter()
            .enumerate()
            .filter_map(|(i, h)| if h.on == ev.event { Some(i) } else { None })
            .collect();

        for idx in handler_indices {
            // Clone the handler descriptor (small) so we don't hold a borrow on manifest.
            let handler = self.manifest.handlers[idx].clone();
            let body_uri = handler.body.uri.clone();

            // Run the body. We move the body fn out, run it, then put it back —
            // avoids the awkward simultaneous &mut bodies + &mut other-fields dance.
            let mut body_fn = self
                .bodies
                .bodies_take(&body_uri)
                .ok_or_else(|| RuntimeError::BodyMissing(body_uri.clone()))?;

            let result = {
                let mut ctx = BodyCtx::new(
                    &ev.event,
                    &ev.payload,
                    event_id,
                    arrival_ts,
                    &handler,
                    &self.manifest,
                    &self.state,
                    self.emit_id_counter.clone(),
                );
                let outcome = body_fn(&mut ctx);
                outcome.map(|()| (ctx.writes, ctx.emits, ctx.reads))
            };

            self.bodies.bodies_put(body_uri.clone(), body_fn);

            let (writes, emits, reads) = match result {
                Ok(t) => t,
                Err(source) => {
                    return Err(RuntimeError::Body {
                        handler: handler.name,
                        source,
                    });
                }
            };

            // Apply writes atomically (single-threaded, so just in order).
            for w in &writes {
                match w {
                    WriteRecord::SetAtom { cell, value } => {
                        self.state.set_atom(cell, value.clone());
                    }
                    WriteRecord::PutMap { cell, key, value } => {
                        self.state.put_map_entry(cell, key.clone(), value.clone());
                    }
                    WriteRecord::DeleteMap { cell, key } => {
                        self.state.delete_map_entry(cell, key);
                    }
                }
            }

            self.log.append(LogEntryKind::HandlerInvoked {
                event_id,
                handler: handler.name.clone(),
                reads: serde_json::to_value(&reads).unwrap_or(Value::Null),
                writes: writes.clone(),
                emits: emits.clone(),
            });

            // Fulfill emits.
            for emit in emits {
                let result = self
                    .adapters
                    .fulfill(&emit.effect, emit.request.clone(), emit.emit_id)
                    .ok_or_else(|| RuntimeError::AdapterMissing(emit.effect.clone()))?;
                let outcome = match result {
                    AdapterResult::Ok(response) => EffectOutcome::Ok { response },
                    AdapterResult::Failed { reason } => EffectOutcome::Failed { reason },
                };

                // If the effect declares a response_event, auto-enqueue it
                // with a synthesized payload. The body's correlation id is
                // the emit_id. `outcome` is shaped as a sum variant
                // ({tag, value}) so manifests can declare it as
                // `sum { ok: <ResponseSchema>, failed: string }` and the
                // WASM loader's typed Val conversion lines up.
                if let Some(effect_def) = self.manifest.effects.get(&emit.effect) {
                    if let Some(response_event) = &effect_def.response_event {
                        let outcome_variant = match &outcome {
                            EffectOutcome::Ok { response } => serde_json::json!({
                                "tag": "ok",
                                "value": response,
                            }),
                            EffectOutcome::Failed { reason } => serde_json::json!({
                                "tag": "failed",
                                "value": reason,
                            }),
                        };
                        let payload = serde_json::json!({
                            "emit_id": emit.emit_id,
                            "outcome": outcome_variant,
                        });
                        let follow_up = InboundEvent {
                            event: response_event.clone(),
                            payload,
                            correlates_to: Some(emit.emit_id),
                        };
                        self.queue.push_back(follow_up);
                    }
                }

                self.log.append(LogEntryKind::EffectFulfilled {
                    emit_id: emit.emit_id,
                    effect: emit.effect,
                    result: outcome,
                });
            }
        }

        Ok(())
    }
}
