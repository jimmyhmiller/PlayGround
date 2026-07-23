import type { Page } from "playwright";
import type { ObservedRequest } from "./settle.js";

/**
 * Simulated bad conditions: seeded latency and failure injection, applied via
 * route interception to tracked requests (fetch/xhr/document).
 *
 * Conditions are runtime physics, not test semantics — they live in config /
 * CLI flags, never in flow files. Everything injected is recorded and
 * attributed in traces and reports, so "what is happening" is always explicit:
 * a chaos-induced failure says so, and names the seed that reproduces it.
 *
 * Ordering note: `given stub` routes are registered AFTER the condition route,
 * so stubs win (Playwright dispatches the last-registered handler first) —
 * stubbed traffic is hermetic and immune to chaos by design.
 */
export interface ConditionProfile {
  /** [min, max] added latency per request, in ms */
  latencyMs?: [number, number];
  /** 0..1 — probability a request is failed at the network level */
  failRate?: number;
  /** PRNG seed; same seed + same request order = same injections */
  seed: number;
}

export function describeConditions(p: ConditionProfile): string {
  const parts: string[] = [];
  if (p.latencyMs) parts.push(`latency +${p.latencyMs[0]}–${p.latencyMs[1]}ms`);
  if (p.failRate) parts.push(`${Math.round(p.failRate * 100)}% request failures`);
  parts.push(`seed ${p.seed}`);
  return parts.join(", ");
}

/** Deterministic PRNG (mulberry32). */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TRACKED_TYPES = new Set(["fetch", "xhr", "document"]);

export class ConditionEngine {
  private rng: () => number;
  /** FIFO of injections per "METHOD url" key, consumed by annotate() */
  private injections = new Map<string, string[]>();

  constructor(public profile: ConditionProfile) {
    this.rng = mulberry32(profile.seed);
  }

  async install(page: Page): Promise<void> {
    await page.route(
      (url) => url.protocol === "http:" || url.protocol === "https:",
      async (route) => {
        const req = route.request();
        if (!TRACKED_TYPES.has(req.resourceType())) return route.fallback();
        const key = `${req.method()} ${req.url()}`;

        if (this.profile.failRate && this.rng() < this.profile.failRate) {
          this.record(key, "injected failure (conditions)");
          return route.abort("failed");
        }
        if (this.profile.latencyMs) {
          const [lo, hi] = this.profile.latencyMs;
          const delay = Math.round(lo + this.rng() * (hi - lo));
          this.record(key, `+${delay}ms injected latency`);
          await new Promise((r) => setTimeout(r, delay));
        }
        return route.fallback();
      },
    );
  }

  private record(key: string, what: string): void {
    const q = this.injections.get(key) ?? [];
    q.push(what);
    this.injections.set(key, q);
  }

  /** Attach injection attribution to an observed request (FIFO per method+url). */
  annotate(rec: ObservedRequest): void {
    const q = this.injections.get(`${rec.method} ${rec.url}`);
    const injected = q?.shift();
    if (injected !== undefined) rec.injected = injected;
  }
}
