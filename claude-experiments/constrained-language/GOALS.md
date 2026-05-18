# Constrained Language — Goals

## Why this exists

AI can produce a lot of code quickly. The problem isn't volume — it's that the
resulting programs are hard for humans to understand, audit, or constrain. You
read a 500-line file and have no clear sense of what state it touches, what
side effects it can produce, or what happens when a particular event arrives.

This project is a runtime and program model designed so that the property "the
program is understandable" holds *by construction* rather than by discipline.
Every program is visualizable and auditable at multiple levels of abstraction,
and the AI authoring it cannot do anything outside what its declarations allow.

## Vision

A program is a set of pure **handlers** reacting to typed **events**. Each
handler declares its full footprint up front: the **state slices** it reads,
the slices it writes, and the **effects** it may emit. The runtime owns all
state, schedules handlers in parallel when their footprints don't conflict,
fulfills effects externally, and records everything that happens to a single
append-only log. An **inspector** reads the same IR and log the runtime does,
presenting the program from three angles — events, state, effects — at any
zoom level.

## Non-goals

- **Not a general-purpose programming language.** Wide swaths of "normal" code
  are deliberately unexpressible.
- **Not a research project on novel type systems or effect calculi.** Use the
  simplest formulation that does the job.
- **Not optimizing for human-written code ergonomics.** The primary author is
  an AI. Humans should be able to read, audit, and edit, but not necessarily
  write from scratch as fast as in a general language.
- **Not a single-language ecosystem.** Bodies are WASM components; authoring
  can happen in any language that compiles to components.
- **Not a UI framework or a workflow engine.** It's the substrate one of those
  could be built on.

## Core principles

1. **Declaration is contract.** A handler's declared `on / read / write / emit`
   is its complete causal envelope. The runtime enforces it; the inspector
   visualizes it; the AI cannot violate it.

2. **Pure handlers, owned state, external effects.** Bodies are pure functions
   `(event, reads) -> (writes, effects)`. State lives in the runtime. Effects
   are *requests* fulfilled by the host, never performed by handlers directly.

3. **Schemas are explicit and structural.** Wiring is checked by shape. Named
   aliases exist for readability but don't gate compatibility.

4. **One log, one clock.** Every event, invocation, read snapshot, write,
   effect request, and effect response is recorded in an append-only log keyed
   by a logical clock. Inspection, replay, and forking all fall out of this.

5. **Deterministic by default.** Given the same input log, the runtime produces
   the same outcome. Non-determinism is a contract violation, not a feature.

6. **Inspector first, not last.** The inspector is the primary operator UI,
   built alongside the runtime — not bolted on. The runtime exposes state and
   log through an API; the inspector is the canonical consumer.

7. **No escape hatches.** No `raw_eval`, no `host_call`, no `any` type, no
   "just for now" backdoor. Every leaf operation lives in the typed registry.
   The moment an escape hatch ships, the AI will reach for it.

8. **The IR is the source of truth.** Text syntax, graphical editor, and AI
   emission all converge on the same typed manifest. Programs are *data*:
   diffable, lintable, transformable.

## Properties we want to hold

**A reader** can answer all of the following from the inspector with one click
and one picture:

- What does this program do? *(program map)*
- What happens when event X arrives? *(event view)*
- What handlers touch this data, in what mode? *(state cell view)*
- What can this program do to the outside world? *(effects view)*
- Why was this run slow / why did these handlers serialize?
  *(timeline view with conflict edges)*
- What exactly does this one handler do, top to bottom, with its full
  footprint? *(handler card)*

**An author** — human or AI — can violate none of the following:

- Touch state not declared in `read` or `write`.
- Emit an effect not declared in `emit`.
- Receive any input other than the declared event shape and read snapshot.
- Mutate state observed by a peer handler in a way the scheduler didn't
  authorize.

**A host operator** can:

- Replay any session deterministically.
- Fork the timeline at any event for what-if analysis.
- Mock any effect type for testing.
- Snapshot and restore full runtime state.

## Architectural shape

- **IR.** A typed manifest: schemas, events, state cells, effect types,
  handlers (with declared footprints), body references. Serializable; the
  single artifact every other component consumes.
- **Runtime.** Implemented in Rust. Owns state (persistent data structures for
  cheap snapshots), routes events, schedules handlers by footprint conflict,
  fulfills effects via pluggable host adapters, writes the canonical event log.
- **Bodies.** WASM components instantiated by the runtime. Each handler's
  component is given imports keyed to its declared footprint, so a body
  literally cannot call what it didn't declare.
- **Inspector.** A web app reading IR + event log via the runtime's API.
  Provides the program map, handler cards, state cell views, timelines,
  replay, and forking.
- **Authoring.** A text syntax compiles to IR; the inspector doubles as a
  graphical editor; AI can emit IR directly.

## Why WASM components specifically

- Bodies can be authored in any language with a component-model toolchain
  (Rust, TypeScript, Python via componentize-py, Go via TinyGo, and so on).
- The component model's typed imports and exports map cleanly onto the
  declared footprint: a handler's `read / write / emit` becomes a component's
  imports, generated from the IR.
- Sandboxing is strong by default — bodies have no ambient capabilities, only
  what is wired through imports.
- WIT (WebAssembly Interface Types) gives us a serializable, language-neutral
  way to describe schemas, doubling as a publication format for the IR.

## Build phasing

Each phase produces something runnable end-to-end; later phases swap out
implementations but don't change the IR or the inspector contract.

1. **IR.** Types, schemas, manifest format, validator.
2. **Reference runtime.** Single-threaded, in-memory state, a trivial built-in
   body language sufficient to write the first toy program.
3. **Event log and replay.** Append-only log keyed by logical clock;
   deterministic replay from log.
4. **Inspector.** Web UI with the three views (map, handler card, state cell)
   and the timeline; reads runtime over HTTP/SSE.
5. **Parallel scheduler.** Footprint-based conflict detection; versioned state
   reads so non-conflicting handlers run concurrently.
6. **WASM component bodies.** wasmtime-backed; per-handler imports generated
   from declared footprint.
7. **Effect host adapters.** Pluggable (HTTP, timer, log, filesystem, etc.)
   with mock implementations for replay and testing.

## Open questions parked for later

- **Persistence.** In-memory only, or durable? An append-only log makes
  "replay from boot" cheap; that may be enough for v1.
- **Time and causality.** Total global event order, or per-slice order? Total
  is easier to draw; per-slice scales further.
- **Failure semantics.** How do effect failures, body traps, and schema
  violations surface? Probably as events on the same inbox.
- **Versioning.** How do schema changes interact with stored event logs and
  live state?
- **Distribution.** Single-node only initially. The model is friendly to
  distribution but it's out of scope for v1.
