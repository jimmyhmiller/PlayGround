// The fetch-user machine + its effect handlers. Shared between the CLI demo
// and the inspector demo so both drive the same logic.

import { defineMachine } from '../core/machine.ts';
import type { EffectHandlers } from '../core/runtime.ts';

export type State = 'Idle' | 'Loading' | 'Ready' | 'Error';

export interface User {
  id: string;
  name: string;
}

export interface Context {
  userId: string | null;
  user: User | null;
  error: string | null;
  startedAt: number | null;
}

export type Event =
  | { type: 'FETCH'; userId: string }
  | { type: 'HTTP_OK'; body: User }
  | { type: 'HTTP_ERR'; message: string }
  | { type: 'RESET' };

export type Effect =
  | { kind: 'http.get'; url: string; replyOk: 'HTTP_OK'; replyErr: 'HTTP_ERR' }
  | { kind: 'log'; level: 'info' | 'warn' | 'error'; message: string };

export const fetchUserMachine = defineMachine<State, Context, Event, Effect>({
  name: 'fetch-user',
  initial: 'Idle',
  initialContext: {
    userId: null,
    user: null,
    error: null,
    startedAt: null,
  },
  effectKinds: ['http.get', 'log'],

  states: {
    Idle: {
      on: {
        FETCH: (ctx, ev, meta) => ({
          state: 'Loading',
          context: {
            ...ctx,
            userId: ev.userId,
            error: null,
            startedAt: meta.receivedAt,
          },
          effects: [
            {
              kind: 'http.get',
              url: `/users/${ev.userId}`,
              replyOk: 'HTTP_OK',
              replyErr: 'HTTP_ERR',
            },
            { kind: 'log', level: 'info', message: `fetching ${ev.userId}` },
          ],
        }),
      },
    },
    Loading: {
      on: {
        HTTP_OK: (ctx, ev) => ({
          state: 'Ready',
          context: { ...ctx, user: ev.body, error: null },
        }),
        HTTP_ERR: (ctx, ev) => ({
          state: 'Error',
          context: { ...ctx, user: null, error: ev.message },
          effects: [{ kind: 'log', level: 'error', message: ev.message }],
        }),
      },
    },
    Ready: {
      on: {
        RESET: (ctx) => ({
          state: 'Idle',
          context: { ...ctx, user: null, userId: null, startedAt: null },
        }),
      },
    },
    Error: {
      on: {
        RESET: (ctx) => ({
          state: 'Idle',
          context: {
            ...ctx,
            user: null,
            userId: null,
            error: null,
            startedAt: null,
          },
        }),
      },
    },
  },

  invariants: [
    {
      name: 'ready-implies-user',
      check: (state, ctx) => state !== 'Ready' || ctx.user !== null,
    },
    {
      name: 'error-implies-message',
      check: (state, ctx) => state !== 'Error' || ctx.error !== null,
    },
  ],
});

// Fake user database so the demo runs without a network. Swap for real
// `fetch(effect.url)` to hit a real backend; the machine doesn't change.
const fakeDb: Record<string, User> = {
  u1: { id: 'u1', name: 'Ada' },
  u2: { id: 'u2', name: 'Grace' },
};

export const fetchUserHandlers: EffectHandlers<Effect, Event> = {
  'http.get': (effect, send) => {
    setTimeout(() => {
      const id = effect.url.split('/').pop() ?? '';
      const user = fakeDb[id];
      if (user !== undefined) {
        send({ type: effect.replyOk, body: user });
      } else {
        send({ type: effect.replyErr, message: `no such user: ${id}` });
      }
    }, 100);
  },
  log: (effect) => {
    // eslint-disable-next-line no-console
    console.log(`  [log:${effect.level}] ${effect.message}`);
  },
};
