// Type-safe builder for MachineDef.
//
// Usage:
//   const machine = defineMachine<State, Context, Event, Effect>({
//     name: 'fetch-user',
//     initial: 'Idle',
//     initialContext: {},
//     effectKinds: ['http.get', 'log'],
//     states: {
//       Idle:    { on: { FETCH: (ctx, ev) => ({ state: 'Loading', ... }) } },
//       Loading: { on: { OK:    (ctx, ev) => ({ state: 'Ready',   ... }) } },
//       ...
//     },
//     invariants: [...],
//   })

import type {
  EffectBase,
  EventBase,
  Invariant,
  MachineDef,
  StateConfig,
  TransitionFn,
} from './types.ts';

// Per-state config in the *user-facing* shape: each event handler is narrowed
// to the variant of E whose `type` matches the key.
type UserStateConfig<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
> = {
  readonly on: {
    readonly [K in E['type']]?: TransitionFn<S, C, Extract<E, { type: K }>, F>;
  };
};

export interface DefineMachineInput<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
> {
  readonly name: string;
  readonly initial: S;
  readonly initialContext: C;
  readonly effectKinds: ReadonlyArray<F['kind']>;
  readonly states: { readonly [K in S]: UserStateConfig<S, C, E, F> };
  readonly invariants?: ReadonlyArray<Invariant<S, C>>;
}

export function defineMachine<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
>(input: DefineMachineInput<S, C, E, F>): MachineDef<S, C, E, F> {
  // The user-facing per-state shape is structurally identical to the internal
  // shape (a map from event-type string to a transition fn) once we erase
  // the per-key narrowing, so this cast is sound.
  const states = input.states as unknown as Readonly<
    Record<S, StateConfig<S, C, E, F>>
  >;
  return {
    name: input.name,
    initial: input.initial,
    initialContext: input.initialContext,
    states,
    effectKinds: new Set(input.effectKinds),
    invariants: input.invariants ?? [],
  };
}
