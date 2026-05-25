// Runtime that drives a MachineDef.
//
// The runtime is the *only* place mutable state and side effects live.
// Reducers are pure values-in / values-out. The runtime:
//   - holds (state, context) and applies transitions
//   - stamps event metadata (receivedAt, seq)
//   - validates every emitted effect against the machine's declared catalog
//   - dispatches effects to user-registered handlers; results come back as
//     ordinary events via `send`
//   - records a TimelineEntry for every transition (full event-sourced log)
//   - runs invariants after each transition; on failure, freezes the machine
//   - emits a subscription stream so a debugger can watch transitions live

import type {
  EffectBase,
  EventBase,
  EventMeta,
  MachineDef,
  TimelineEntry,
} from './types.ts';

// Handler for one effect kind. Receives the full effect value and a `send`
// callback to push reply events back into the machine. May be async.
export type EffectHandler<F extends EffectBase, E extends EventBase> = (
  effect: F,
  send: (event: E) => void,
) => void | Promise<void>;

// Map from effect kind to handler. Built via the helper below for type safety.
export type EffectHandlers<F extends EffectBase, E extends EventBase> = {
  readonly [K in F['kind']]?: EffectHandler<Extract<F, { kind: K }>, E>;
};

export interface RuntimeOptions<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
> {
  readonly handlers: EffectHandlers<F, E>;
  // Override for tests; defaults to Date.now.
  readonly now?: () => number;
  // Called for every transition (after invariants run). The debugger
  // subscribes here.
  readonly onTransition?: (entry: TimelineEntry<S, C, E, F>) => void;
}

// Distinct failure modes the runtime can produce. We throw these as named
// errors so the host can react instead of dealing with generic Error strings.
export class UnknownEffectKindError extends Error {
  constructor(public readonly kind: string) {
    super(`Effect kind "${kind}" is not declared in machine.effectKinds`);
  }
}
export class InvariantViolation extends Error {
  constructor(
    public readonly invariantName: string,
    public readonly entry: TimelineEntry<string, unknown, EventBase, EffectBase>,
  ) {
    super(`Invariant "${invariantName}" violated`);
  }
}
export class MachineFrozenError extends Error {
  constructor() {
    super('Machine is frozen due to an earlier invariant violation');
  }
}

export class Runtime<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
> {
  private state: S;
  private context: C;
  private seq = 0;
  private frozen = false;
  private readonly timeline: TimelineEntry<S, C, E, F>[] = [];
  private readonly now: () => number;
  private readonly handlers: EffectHandlers<F, E>;
  private readonly subscribers = new Set<
    (entry: TimelineEntry<S, C, E, F>) => void
  >();

  constructor(
    public readonly def: MachineDef<S, C, E, F>,
    opts: RuntimeOptions<S, C, E, F>,
  ) {
    this.state = def.initial;
    this.context = def.initialContext;
    this.handlers = opts.handlers;
    this.now = opts.now ?? Date.now;
    if (opts.onTransition !== undefined) {
      this.subscribers.add(opts.onTransition);
    }
  }

  // Register a transition listener. Returns an unsubscribe fn. Listeners are
  // invoked synchronously in registration order after the transition is
  // applied and recorded, including for entries with invariantFailure set.
  subscribe(fn: (entry: TimelineEntry<S, C, E, F>) => void): () => void {
    this.subscribers.add(fn);
    return () => {
      this.subscribers.delete(fn);
    };
  }

  private notify(entry: TimelineEntry<S, C, E, F>): void {
    for (const sub of this.subscribers) {
      try {
        sub(entry);
      } catch (err: unknown) {
        // A buggy subscriber must never break the machine. Log and continue.
        // eslint-disable-next-line no-console
        console.error('[runtime] subscriber threw:', err);
      }
    }
  }

  getState(): S {
    return this.state;
  }
  getContext(): C {
    return this.context;
  }
  getTimeline(): ReadonlyArray<TimelineEntry<S, C, E, F>> {
    return this.timeline;
  }
  isFrozen(): boolean {
    return this.frozen;
  }

  // Push an event into the machine. Returns the recorded TimelineEntry if the
  // event was handled, or `null` if the current state ignored this event type
  // (no handler registered for it).
  send(event: E): TimelineEntry<S, C, E, F> | null {
    if (this.frozen) throw new MachineFrozenError();

    const stateConfig = this.def.states[this.state];
    const handler = stateConfig.on[event.type];
    if (handler === undefined) {
      // Ignored — no handler in this state. We deliberately do not record
      // these in the timeline; they're noise and not transitions.
      return null;
    }

    const seq = this.seq++;
    const at = this.now();
    const meta: EventMeta = { receivedAt: at, seq };
    const prev = { state: this.state, context: this.context };

    const result = handler(this.context, event, meta);
    if (result === null) {
      // Handler explicitly chose to ignore this event in this state. Same
      // treatment as missing handler — no timeline entry.
      return null;
    }

    const nextState = result.state ?? this.state;
    const nextContext = result.context ?? this.context;
    const effects = result.effects ?? [];

    // Validate every effect against the declared catalog. This is the key
    // constraint: AI-written reducers cannot smuggle in new effect kinds.
    for (const eff of effects) {
      if (!this.def.effectKinds.has(eff.kind as F['kind'])) {
        throw new UnknownEffectKindError(eff.kind);
      }
    }

    // Apply transition.
    this.state = nextState;
    this.context = nextContext;

    // Build timeline entry first so invariants can be reported against it.
    let entry: TimelineEntry<S, C, E, F> = {
      seq,
      at,
      event,
      prev,
      next: { state: nextState, context: nextContext },
      effects,
    };

    // Invariants.
    for (const inv of this.def.invariants) {
      if (!inv.check(nextState, nextContext)) {
        entry = { ...entry, invariantFailure: inv.name };
        this.timeline.push(entry);
        this.notify(entry);
        this.frozen = true;
        throw new InvariantViolation(
          inv.name,
          entry as TimelineEntry<string, unknown, EventBase, EffectBase>,
        );
      }
    }

    this.timeline.push(entry);
    this.notify(entry);

    // Dispatch effects. Handlers can `send` follow-up events synchronously
    // (handled inline) or asynchronously (handled later). Order: we run
    // handlers in the order effects were emitted, but do not await async
    // ones — they are background work by construction.
    for (const eff of effects) {
      const h = this.handlers[eff.kind as F['kind']] as
        | EffectHandler<F, E>
        | undefined;
      if (h === undefined) {
        // No handler registered for a declared kind. This is a host-side
        // gap, not a reducer bug — surface it as a console warning rather
        // than freezing the machine.
        // eslint-disable-next-line no-console
        console.warn(
          `[runtime] no handler registered for effect kind "${eff.kind}"`,
        );
        continue;
      }
      const result = h(eff, (ev) => this.send(ev));
      // If async, attach a rejection logger so failures aren't silent.
      if (result instanceof Promise) {
        result.catch((err: unknown) => {
          // eslint-disable-next-line no-console
          console.error(`[runtime] effect handler "${eff.kind}" threw:`, err);
        });
      }
    }

    return entry;
  }
}
