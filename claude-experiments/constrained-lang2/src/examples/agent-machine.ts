// Toy "agent" machine — proves the substrate. Receives an Ask event, kicks
// off a fake LLM call effect, and emits a reply when the LLM result comes
// back. Real LLM-calling logic lives in the effect handler, NOT the machine.
//
// The same machine + handlers power the CLI, HTTP, and queue examples.

import { defineMachine } from '../core/machine.ts';
import type { EffectHandlers } from '../core/runtime.ts';

export type State = 'Idle' | 'Working';

export interface Context {
  // Track in-flight questions so a reply can address the right caller.
  // Keyed by correlationId.
  inflight: Record<string, string>; // correlationId -> original question
}

export type Event =
  | { type: 'Ask'; question: string; correlationId: string }
  | { type: 'LlmReply'; correlationId: string; answer: string }
  | { type: 'LlmError'; correlationId: string; message: string };

export type Effect =
  | { kind: 'call_llm'; correlationId: string; prompt: string }
  | { kind: 'reply'; correlationId: string; body: unknown }
  | { kind: 'log'; level: 'info' | 'warn' | 'error'; message: string };

// Notice: a single Ask transitions Idle -> Working, but subsequent Asks while
// Working ALSO get handled (multiple in-flight). The machine isn't strictly
// single-threaded — each request is tracked by correlationId in context.
export const agentMachine = defineMachine<State, Context, Event, Effect>({
  name: 'agent',
  initial: 'Idle',
  initialContext: { inflight: {} },
  effectKinds: ['call_llm', 'reply', 'log'],

  states: {
    Idle: {
      on: {
        Ask: handleAsk,
      },
    },
    Working: {
      on: {
        Ask: handleAsk,
        LlmReply: (ctx, ev) => {
          const { [ev.correlationId]: _gone, ...rest } = ctx.inflight;
          const nextInflight = rest;
          return {
            state: Object.keys(nextInflight).length === 0 ? 'Idle' : 'Working',
            context: { inflight: nextInflight },
            effects: [
              {
                kind: 'reply',
                correlationId: ev.correlationId,
                body: { answer: ev.answer },
              },
            ],
          };
        },
        LlmError: (ctx, ev) => {
          const { [ev.correlationId]: _gone, ...rest } = ctx.inflight;
          const nextInflight = rest;
          return {
            state: Object.keys(nextInflight).length === 0 ? 'Idle' : 'Working',
            context: { inflight: nextInflight },
            effects: [
              {
                kind: 'reply',
                correlationId: ev.correlationId,
                body: { error: ev.message },
              },
              { kind: 'log', level: 'error', message: ev.message },
            ],
          };
        },
      },
    },
  },

  invariants: [
    {
      name: 'working-iff-inflight',
      check: (state, ctx) => {
        const has = Object.keys(ctx.inflight).length > 0;
        return state === 'Working' ? has : !has;
      },
    },
  ],
});

function handleAsk(ctx: Context, ev: { question: string; correlationId: string }): {
  state: State;
  context: Context;
  effects: Effect[];
} {
  return {
    state: 'Working',
    context: {
      inflight: { ...ctx.inflight, [ev.correlationId]: ev.question },
    },
    effects: [
      {
        kind: 'call_llm',
        correlationId: ev.correlationId,
        prompt: ev.question,
      },
      { kind: 'log', level: 'info', message: `ask: ${ev.question}` },
    ],
  };
}

// --- Effect handlers ----------------------------------------------------

// Fake LLM: echoes with a short delay. Replace with a real client.
export const agentHandlers: EffectHandlers<Effect, Event> = {
  call_llm: (effect, send) => {
    setTimeout(() => {
      // Pretend errors happen for the magic word "fail".
      if (effect.prompt.toLowerCase().includes('fail')) {
        send({
          type: 'LlmError',
          correlationId: effect.correlationId,
          message: `simulated LLM failure for prompt: ${effect.prompt}`,
        });
        return;
      }
      send({
        type: 'LlmReply',
        correlationId: effect.correlationId,
        answer: `you said: "${effect.prompt}" (${effect.prompt.length} chars)`,
      });
    }, 80);
  },
  log: (effect) => {
    // eslint-disable-next-line no-console
    console.log(`  [log:${effect.level}] ${effect.message}`);
  },
  // reply: handled by ReplyHost itself; no user handler needed.
};
