// Core types for the constrained state-machine language.
//
// A Machine is a *value*. It contains no runtime behavior — only:
//   - a set of named states
//   - a context shape
//   - the events it accepts
//   - the effects it can emit
//   - a pure transition function per (state, event) pair
//
// The Runtime (see runtime.ts) is the only thing that executes effects
// or holds mutable state. Reducers are pure: (context, event, meta) -> result.

export type EventBase = { type: string };
export type EffectBase = { kind: string };

// Runtime-injected metadata. Reducers MAY read this (e.g. `meta.receivedAt`)
// but should treat it as ambient — they must not call Date.now / Math.random
// themselves.
export interface EventMeta {
  readonly receivedAt: number; // ms since epoch, runtime-stamped
  readonly seq: number;        // monotonic event sequence
}

// What a transition returns. All fields optional:
//   - omit `state`   -> stay in current state
//   - omit `context` -> context unchanged
//   - omit `effects` -> no effects emitted
// Returning `null` means "this event is ignored in this state" — distinct from
// "handled but no change", which is `{}`.
export type TransitionResult<S extends string, C, F extends EffectBase> = {
  readonly state?: S;
  readonly context?: C;
  readonly effects?: ReadonlyArray<F>;
} | null;

// A per-(state, event-type) handler. Receives current context + the event
// (narrowed to its specific shape) and returns a TransitionResult.
export type TransitionFn<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
> = (context: C, event: E, meta: EventMeta) => TransitionResult<S, C, F>;

// State configuration: a map from event-type to its handler.
// We use `any` for the event narrowing here because TS can't express
// "narrow E to the variant whose `type` matches the key" in a
// single-record type position. The `defineMachine` builder provides the
// type-safe surface; this is the internal representation.
export interface StateConfig<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly on: Readonly<Record<string, TransitionFn<S, C, any, F>>>;
}

// A complete machine definition.
export interface MachineDef<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
> {
  readonly name: string;
  readonly initial: S;
  readonly initialContext: C;
  readonly states: Readonly<Record<S, StateConfig<S, C, E, F>>>;
  // Catalog of effect kinds this machine is allowed to emit. The runtime
  // validates every emitted effect against this set. AI-written reducers
  // cannot smuggle in new effects.
  readonly effectKinds: ReadonlySet<F['kind']>;
  // Optional invariants checked after every transition. Violation freezes
  // the machine and surfaces the offending (prev, event, next).
  readonly invariants: ReadonlyArray<Invariant<S, C>>;
}

export interface Invariant<S extends string, C> {
  readonly name: string;
  readonly check: (state: S, context: C) => boolean;
}

// One entry in the recorded timeline. Pure data — serializable.
export interface TimelineEntry<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
> {
  readonly seq: number;
  readonly at: number;
  readonly event: E;
  readonly prev: { state: S; context: C };
  readonly next: { state: S; context: C };
  readonly effects: ReadonlyArray<F>;
  // If an invariant failed on this transition, its name lands here.
  readonly invariantFailure?: string;
}
