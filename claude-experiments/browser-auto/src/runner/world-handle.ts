import type { DoctorReport } from "../world/adapter.js";
import { applyWorld, capabilityLevel, doctor } from "../world/adapter.js";
import { WorldError } from "../world/algebra.js";
import type { SessionState, VerificationReport, WorldAdapter, WorldDescription } from "../world/types.js";

/** Uniform interface over the two transports (in-process module / HTTP). */
export interface WorldHandle {
  capabilities(): Promise<{ level: number; hasSession: boolean; hasSnapshot: boolean; hasFingerprint: boolean }>;
  apply(description: WorldDescription): Promise<{ verification: VerificationReport }>;
  session(userKey: string): Promise<SessionState>;
  doctor(): Promise<DoctorReport>;
  snapshot(): Promise<string | null>;
  restore(id: string): Promise<void>;
  fingerprint(): Promise<string | null>;
}

export function localWorldHandle(adapter: WorldAdapter): WorldHandle {
  let lastIds: Record<string, Record<string, unknown>> = {};
  return {
    async capabilities() {
      const { level } = capabilityLevel(adapter);
      return {
        level,
        hasSession: !!adapter.session,
        hasSnapshot: !!adapter.snapshot,
        hasFingerprint: !!adapter.fingerprint,
      };
    },
    async apply(description) {
      const applied = await applyWorld(adapter, description);
      lastIds = applied.ids as Record<string, Record<string, unknown>>;
      return { verification: applied.verification };
    },
    async session(userKey) {
      if (!adapter.session) {
        throw new WorldError([
          `flow uses 'given user "${userKey}" signed-in' but the world adapter has no session() — ` +
            `add session(userKey, ctx) to defineWorld to mint real sessions`,
        ]);
      }
      const ctx = {
        id: (refOrType: unknown, key?: string) => {
          const [t, k] =
            typeof refOrType === "string"
              ? [refOrType, key!]
              : ((refOrType as { $ref: [string, string] }).$ref ?? [undefined, undefined]);
          const v = t !== undefined ? lastIds[t]?.[k!] : undefined;
          if (v === undefined) throw new WorldError([`session(): cannot resolve id for ${String(t)}/"${String(k)}"`]);
          return v;
        },
      };
      return adapter.session(userKey, ctx as never);
    },
    async doctor() {
      return doctor(adapter);
    },
    async snapshot() {
      return adapter.snapshot ? await adapter.snapshot() : null;
    },
    async restore(id) {
      if (!adapter.restore) throw new WorldError(["adapter has no restore() — snapshot/restore is L4"]);
      await adapter.restore(id);
    },
    async fingerprint() {
      return adapter.fingerprint ? await adapter.fingerprint() : null;
    },
  };
}

export function httpWorldHandle(endpoint: string): WorldHandle {
  async function call<T>(op: string, payload: Record<string, unknown> = {}): Promise<T> {
    let res: Response;
    try {
      res = await fetch(endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ op, ...payload }),
      });
    } catch (e) {
      throw new WorldError([
        `world endpoint ${endpoint} unreachable (${e instanceof Error ? e.message : String(e)}) — ` +
          `is the app running with BAT_TEST=1 and the handler mounted?`,
      ]);
    }
    const body = (await res.json().catch(() => null)) as { ok?: boolean; error?: string; problems?: string[]; result?: T } | null;
    if (!res.ok || !body?.ok) {
      if (body?.problems) throw new WorldError(body.problems);
      throw new WorldError([`world endpoint ${op} failed: HTTP ${res.status} ${body?.error ?? ""}`]);
    }
    return body.result as T;
  }
  return {
    capabilities: () => call("capabilities"),
    apply: (description) => call("apply", { description }),
    session: (userKey) => call("session", { userKey }),
    doctor: () => call("doctor"),
    snapshot: () => call("snapshot"),
    restore: (id) => call("restore", { id }).then(() => undefined),
    fingerprint: () => call("fingerprint"),
  };
}
