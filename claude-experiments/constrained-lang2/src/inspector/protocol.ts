// Wire protocol between Inspector (Node) and the browser UI.
//
// All messages are JSON. Server -> Client uses the `Server*` types,
// Client -> Server uses `Client*`. Both directions are tagged unions on
// `type`. Keep this file dependency-free so it could be imported from a
// future bundled UI build too.

import type { EffectBase, EventBase, TimelineEntry } from '../core/types.ts';

// Stripped, serializable view of a machine. Reducer fns are not sent;
// instead we send, per state, the list of event-type strings it accepts.
// That's enough for the UI to highlight valid events and show structure.
export interface MachineSnapshot {
  readonly name: string;
  readonly initial: string;
  readonly states: ReadonlyArray<{
    readonly name: string;
    readonly accepts: ReadonlyArray<string>;
  }>;
  readonly effectKinds: ReadonlyArray<string>;
  readonly invariants: ReadonlyArray<string>;
}

export type ServerMessage =
  | {
      type: 'init';
      machine: MachineSnapshot;
      state: string;
      context: unknown;
      timeline: ReadonlyArray<
        TimelineEntry<string, unknown, EventBase, EffectBase>
      >;
      frozen: boolean;
    }
  | {
      type: 'transition';
      entry: TimelineEntry<string, unknown, EventBase, EffectBase>;
      state: string;
      context: unknown;
      frozen: boolean;
    }
  | {
      type: 'error';
      message: string;
    };

export type ClientMessage = {
  type: 'send';
  event: EventBase;
};
