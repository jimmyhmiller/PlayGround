// Request/reply correlation on top of a Runtime.
//
// The pattern: a host wants to ask the machine a question and get one answer.
// State machines are fire-and-forget — `runtime.send(event)` returns nothing.
// To bridge that gap we adopt a convention:
//
//   1. The machine's effect union MUST include a `reply` variant carrying
//      `{ kind: 'reply', correlationId: string, body: unknown }`.
//   2. When the machine wants to respond to a specific caller, its reducer
//      emits a `reply` effect with the caller's correlationId.
//   3. The framework intercepts those effects and resolves any pending
//      `request(builder)` promise that minted that id.
//
// The machine itself never knows whether the caller is a CLI, an HTTP
// request, or a queue worker — it just emits replies addressed by id.

import { randomUUID } from 'node:crypto';

import { Runtime, type EffectHandlers } from '../core/runtime.ts';
import type { EffectBase, EventBase, MachineDef } from '../core/types.ts';

// The reply effect shape. Machines must include a variant compatible with
// this in their F union. We don't constrain F at the type level (would be
// invasive across the codebase); instead we runtime-check effectKinds.
export interface ReplyEffect {
  readonly kind: 'reply';
  readonly correlationId: string;
  readonly body: unknown;
}

export interface ReplyHostOptions {
  readonly defaultTimeoutMs?: number;
  readonly now?: () => number;
}

export class ReplyTimeoutError extends Error {
  constructor(public readonly correlationId: string, ms: number) {
    super(`No reply for correlationId=${correlationId} within ${ms}ms`);
  }
}

export class ReplyHost<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
> {
  readonly runtime: Runtime<S, C, E, F>;
  private readonly defaultTimeoutMs: number;
  private readonly pending = new Map<
    string,
    {
      resolve: (body: unknown) => void;
      reject: (err: Error) => void;
      timer: ReturnType<typeof setTimeout>;
    }
  >();

  constructor(
    machine: MachineDef<S, C, E, F>,
    userHandlers: EffectHandlers<F, E>,
    opts: ReplyHostOptions = {},
  ) {
    if (!machine.effectKinds.has('reply' as F['kind'])) {
      throw new Error(
        `ReplyHost requires the machine to declare 'reply' in effectKinds. ` +
          `Machine "${machine.name}" did not.`,
      );
    }
    this.defaultTimeoutMs = opts.defaultTimeoutMs ?? 30_000;

    // Wrap the user's handlers with our reply interceptor. If the user
    // also registered a 'reply' handler we call theirs after ours — useful
    // for logging — but ours owns the promise resolution.
    const userReply = (userHandlers as Record<string, unknown>)['reply'];
    const handlers: Record<string, unknown> = { ...userHandlers };
    handlers['reply'] = (effect: ReplyEffect, _send: (e: E) => void) => {
      this.dispatchReply(effect);
      if (typeof userReply === 'function') {
        (userReply as (eff: ReplyEffect, s: (e: E) => void) => void)(effect, _send);
      }
    };

    const runtimeOpts: { handlers: EffectHandlers<F, E>; now?: () => number } = {
      handlers: handlers as EffectHandlers<F, E>,
    };
    if (opts.now !== undefined) {
      runtimeOpts.now = opts.now;
    }
    this.runtime = new Runtime<S, C, E, F>(machine, runtimeOpts);
  }

  // Mint a correlation id, build an event with it, push it into the machine,
  // and return a promise that resolves with the reply body or rejects on
  // timeout. The caller is responsible for ensuring the machine will
  // eventually emit a `reply` effect with this id — otherwise it times out.
  async request<Body = unknown>(
    eventBuilder: (correlationId: string) => E | Promise<E>,
    timeoutMs?: number,
  ): Promise<Body> {
    const id = randomUUID();
    const ms = timeoutMs ?? this.defaultTimeoutMs;
    // Build the event first (may throw / reject — that surfaces as a normal
    // rejection rather than a timeout, which is what callers want).
    const event = await eventBuilder(id);
    return new Promise<Body>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new ReplyTimeoutError(id, ms));
      }, ms);
      this.pending.set(id, {
        resolve: (body: unknown) => resolve(body as Body),
        reject,
        timer,
      });
      try {
        this.runtime.send(event);
      } catch (err: unknown) {
        // Roll back the pending entry if send itself failed.
        const entry = this.pending.get(id);
        if (entry !== undefined) {
          clearTimeout(entry.timer);
          this.pending.delete(id);
        }
        reject(err as Error);
      }
    });
  }

  private dispatchReply(effect: ReplyEffect): void {
    const entry = this.pending.get(effect.correlationId);
    if (entry === undefined) {
      // No one is waiting — could be a duplicate reply, or a reply for a
      // request that already timed out. Drop silently; timeouts already
      // rejected the caller.
      return;
    }
    clearTimeout(entry.timer);
    this.pending.delete(effect.correlationId);
    entry.resolve(effect.body);
  }

  // Cancel all pending requests with a shutdown error. Use on host shutdown.
  shutdown(): void {
    const err = new Error('host shutting down');
    for (const entry of this.pending.values()) {
      clearTimeout(entry.timer);
      entry.reject(err);
    }
    this.pending.clear();
  }
}
