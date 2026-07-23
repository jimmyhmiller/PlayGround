import type { Page, Request, Response } from "playwright";
import { matchPath, pathOf } from "./patterns.js";

/**
 * The settlement engine. The DSL has no way to express time; this is where
 * waiting actually happens, and it waits on EVENTS:
 *   - declared `expect request` matchers resolving,
 *   - the in-flight tracked request count draining to zero,
 *   - no navigation mid-flight,
 *   - a drained JS task queue (double-rAF, or a plain roundtrip under a fake clock).
 * The per-step budget is runtime physics (config), never test semantics.
 * Exceeding it produces a stuck-state description, not a bare timeout.
 */

const TRACKED_TYPES = new Set(["fetch", "xhr", "document"]);

export interface ObservedRequest {
  method: string;
  url: string;
  resourceType: string;
  status: number | null;
  failure: string | null;
  finished: boolean;
}

export class NetworkTracker {
  private inflight = new Map<Request, ObservedRequest>();
  /** everything observed since the last step boundary, for traces */
  observed: ObservedRequest[] = [];
  private drainWaiters: Array<() => void> = [];

  constructor(page: Page) {
    page.on("request", (req) => {
      if (!TRACKED_TYPES.has(req.resourceType())) return;
      const rec: ObservedRequest = {
        method: req.method(),
        url: req.url(),
        resourceType: req.resourceType(),
        status: null,
        failure: null,
        finished: false,
      };
      this.inflight.set(req, rec);
      this.observed.push(rec);
    });
    const complete = (req: Request, failure: string | null) => {
      const rec = this.inflight.get(req);
      if (!rec) return;
      rec.finished = true;
      rec.failure = failure;
      this.inflight.delete(req);
      if (this.inflight.size === 0) {
        const waiters = this.drainWaiters;
        this.drainWaiters = [];
        for (const w of waiters) w();
      }
    };
    // Status comes from the synchronous `response` EVENT, never from awaiting
    // req.response() — that promise can stall, and completion bookkeeping
    // (which settlement's drain depends on) must be event-driven and sync.
    page.on("response", (resp) => {
      const rec = this.inflight.get(resp.request());
      if (rec) rec.status = resp.status();
    });
    page.on("requestfinished", (req) => complete(req, null));
    page.on("requestfailed", (req) => complete(req, req.failure()?.errorText ?? "failed"));
  }

  get inflightCount(): number {
    return this.inflight.size;
  }

  pendingDescriptions(): string[] {
    return [...this.inflight.values()].map((r) => `${r.method} ${r.url} (still pending)`);
  }

  /** true when every in-flight request has at least received response headers */
  inflightAllHaveStatus(): boolean {
    return [...this.inflight.values()].every((r) => r.status !== null);
  }

  /** Evict in-flight requests whose response headers arrived but whose
   * finished/failed event never came (lost CDP events, SSE/long-poll bodies).
   * Returns trace notes describing what was evicted. */
  forceCompleteStragglers(): string[] {
    const notes: string[] = [];
    for (const [req, rec] of [...this.inflight.entries()]) {
      if (rec.status === null) continue;
      rec.finished = true;
      this.inflight.delete(req);
      notes.push(
        `${rec.method} ${rec.url}: response ${rec.status} received but the finish event never arrived ` +
          `(lost browser event, or a streaming body) — counted as complete after two quiet passes`,
      );
    }
    if (this.inflight.size === 0) {
      const waiters = this.drainWaiters;
      this.drainWaiters = [];
      for (const w of waiters) w();
    }
    return notes;
  }

  /** Resolves the moment in-flight drains to zero (immediately if already drained). */
  waitForDrain(signal: AbortSignal): Promise<void> {
    if (this.inflight.size === 0) return Promise.resolve();
    return new Promise((resolve) => {
      const onAbort = () => resolve(); // settle loop re-checks state; abort just unblocks
      signal.addEventListener("abort", onAbort, { once: true });
      this.drainWaiters.push(() => {
        signal.removeEventListener("abort", onAbort);
        resolve();
      });
    });
  }

  stepBoundary(): void {
    this.observed = [];
  }
}

export interface RequestExpectation {
  method: string;
  pathPattern: string;
  status: "ok" | number;
}

export interface RequestMatchResult {
  expectation: RequestExpectation;
  matched: ObservedRequest | null;
  statusOk: boolean;
}

/**
 * Armed BEFORE the action (that is the whole point): watches for a response
 * matching method+path, then judges its status. Never expresses a timeout —
 * the settle loop owns the budget and asks us for a verdict at the end.
 */
export class RequestMatcher {
  result: RequestMatchResult;
  private resolveMatched!: () => void;
  /** resolves when a matching request has finished (any status) */
  matchedAndFinished: Promise<void>;
  private page: Page;
  private onResponse: (resp: Response) => void;
  private onReqFailed: (req: Request) => void;

  constructor(public expectation: RequestExpectation, tracker: NetworkTracker, page: Page) {
    this.page = page;
    this.result = { expectation, matched: null, statusOk: false };
    this.matchedAndFinished = new Promise((resolve) => {
      this.resolveMatched = resolve;
    });
    const onFinished = (rec: ObservedRequest) => {
      if (this.result.matched) return;
      if (rec.method !== expectation.method) return;
      const { path, query } = pathOf(rec.url);
      if (!matchPath(expectation.pathPattern, path, query)) return;
      this.result.matched = rec;
      this.result.statusOk =
        expectation.status === "ok"
          ? rec.status !== null && rec.status >= 200 && rec.status < 300
          : rec.status === expectation.status;
      this.resolveMatched();
    };
    // Status comes from the synchronous `response` event (headers received);
    // never await req.response() — see NetworkTracker.
    this.onResponse = (resp: Response) => {
      const req = resp.request();
      if (!TRACKED_TYPES.has(req.resourceType())) return;
      onFinished({
        method: req.method(),
        url: req.url(),
        resourceType: req.resourceType(),
        status: resp.status(),
        failure: null,
        finished: true,
      });
    };
    this.onReqFailed = (req: Request) => {
      if (!TRACKED_TYPES.has(req.resourceType())) return;
      onFinished({
        method: req.method(),
        url: req.url(),
        resourceType: req.resourceType(),
        status: null,
        failure: req.failure()?.errorText ?? "failed",
        finished: true,
      });
    };
    page.on("response", this.onResponse);
    page.on("requestfailed", this.onReqFailed);
  }

  /** step boundary: stop listening */
  dispose(): void {
    this.page.off("response", this.onResponse);
    this.page.off("requestfailed", this.onReqFailed);
  }
}

export interface SettleOutcome {
  settled: boolean;
  iterations: number;
  /** why settlement stopped, when it did not settle */
  stuck: string[];
  /** virtual ms advanced on the installed clock during settlement */
  clockAdvanced: number;
  /** oddities worth surfacing even on success (e.g. straggler evictions) */
  notes: string[];
}

export interface SettleOptions {
  budgetMs: number;
  /** a `given clock` is installed — fake timers need explicit advancing */
  clockInstalled: boolean;
  /** matchers armed for this step; settlement requires them all matched */
  matchers: RequestMatcher[];
}

export async function settle(page: Page, tracker: NetworkTracker, opts: SettleOptions): Promise<SettleOutcome> {
  const deadline = Date.now() + opts.budgetMs;
  const controller = new AbortController();
  const budgetLeft = () => deadline - Date.now();
  let iterations = 0;
  let clockAdvanced = 0;
  const notes: string[] = [];
  let stragglerPasses = 0;

  // One "quiet pulse": a real roundtrip through both event loops. Under an
  // installed (fake) clock, advance virtual time a deterministic quantum so
  // 0-delay timers and debounces run; otherwise a double-rAF with a macrotask
  // fallback (headless pages may produce no frames while idle).
  const quietPulse = async (): Promise<boolean> => {
    try {
      if (opts.clockInstalled) {
        await tick(page, 16);
        clockAdvanced += 16;
      } else {
        // NOTE: no named bindings inside this closure — esbuild-based dev
        // runners (tsx/vitest) inject a __name helper for them, which does
        // not exist inside the page and makes the evaluate throw.
        await page.evaluate(
          () =>
            new Promise<void>((resolve) => {
              requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(resolve, 0)));
              setTimeout(resolve, 50); // fallback: idle headless pages may produce no frames
            }),
        );
      }
      return true;
    } catch {
      // evaluate fails mid-navigation; wait for the nav to land, then re-check
      await page.waitForLoadState("domcontentloaded").catch(() => {});
      return false;
    }
  };

  const timeout = setTimeout(() => controller.abort(), opts.budgetMs);
  try {
    while (budgetLeft() > 0) {
      iterations++;

      // 1. declared request expectations first — they anchor this step's traffic
      const unmatched = opts.matchers.filter((m) => !m.result.matched);
      if (unmatched.length > 0) {
        await Promise.race([
          Promise.all(unmatched.map((m) => m.matchedAndFinished)),
          abortPromise(controller.signal),
          quietPulse(),
        ]);
        if (controller.signal.aborted) break;
        continue;
      }

      // 2. drain tracked in-flight traffic. Pulse while waiting so lost
      //    finish events can't wedge the loop.
      if (tracker.inflightCount > 0) {
        const drained = await Promise.race([
          tracker.waitForDrain(controller.signal).then(() => true),
          quietPulse().then(() => false),
        ]);
        if (controller.signal.aborted) break;
        if (!drained && tracker.inflightCount > 0) {
          // Straggler policy: every in-flight request has response headers,
          // and two consecutive quiet pulses saw no finish event. Chromium
          // sometimes never delivers requestfinished (lost event, or a
          // deliberately open body like SSE). Evict, loudly.
          if (tracker.inflightAllHaveStatus()) {
            stragglerPasses++;
            if (stragglerPasses >= 2) {
              notes.push(...tracker.forceCompleteStragglers());
              stragglerPasses = 0;
            }
          } else {
            stragglerPasses = 0;
          }
        }
        continue;
      }
      stragglerPasses = 0;

      // 3. quiet check: JS task queue drained; navigation not mid-flight.
      if (!(await quietPulse())) continue;

      // 4. anything new appear during the quiet check?
      if (tracker.inflightCount > 0) continue;
      if (opts.matchers.some((m) => !m.result.matched)) continue;

      return { settled: true, iterations, stuck: [], clockAdvanced, notes };
    }
  } finally {
    clearTimeout(timeout);
  }

  const stuck: string[] = [];
  for (const m of opts.matchers) {
    if (!m.result.matched) {
      stuck.push(`declared 'expect request ${m.expectation.method} ${m.expectation.pathPattern}' never saw a matching request`);
    }
  }
  stuck.push(...tracker.pendingDescriptions());
  if (stuck.length === 0) stuck.push("the page kept scheduling work and never went quiet (a poller? consider 'given stub' for it)");
  return { settled: false, iterations, stuck, clockAdvanced, notes };
}

async function tick(page: Page, virtualMs: number): Promise<void> {
  // runFor fires due timers deterministically on the virtual timeline
  await page.clock.runFor(virtualMs).catch(() => {});
}

function abortPromise(signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
}
