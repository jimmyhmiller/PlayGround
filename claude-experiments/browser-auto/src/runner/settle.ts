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
  /** order the request STARTED within this step (1-based) */
  startSeq: number;
  /** order the request FINISHED within this step (1-based); null while pending */
  finishSeq: number | null;
  /** set when a condition profile injected latency or failure into this request */
  injected?: string;
  /** response has no Content-Length (chunked/streaming: RSC, SSE, …) —
   * its body is genuinely still in flight until requestfinished fires */
  streaming?: boolean;
  /** deliberately long-lived stream (text/event-stream) — EXEMPT from drain:
   * an SSE app would otherwise never settle */
  liveStream?: boolean;
}

export interface WsFrame {
  dir: "sent" | "received";
  url: string;
  data: string;
}

/** the contract settle needs from any armed matcher (request or ws) */
export interface SettleMatcher {
  result: { matched: unknown | null };
  matchedAndFinished: Promise<void>;
  describe(): string;
}

export class NetworkTracker {
  private inflight = new Map<Request, ObservedRequest>();
  /** everything observed since the last step boundary, for traces */
  observed: ObservedRequest[] = [];
  private drainWaiters: Array<() => void> = [];
  private startCounter = 0;
  private finishCounter = 0;

  /**
   * @param page the page to observe
   * @param appCreated true only for pages the APP opened (popups/new tabs),
   *   which navigate the instant they exist — bat attaches as early as it can
   *   (the `page` event) and cannot precede their navigation. For pages bat
   *   creates itself (the main page), the tracker MUST be attached before the
   *   first navigation, and the guard below enforces exactly that.
   */
  constructor(page: Page, appCreated = false) {
    // SOUNDNESS INVARIANT: a tracker for a bat-created page must observe it
    // from BEFORE it navigates. Attached after an http(s) navigation, the
    // page's initial requests may already have fired and will escape
    // observation — settle would return early and leak an unfinished render
    // into the next step (the intermittent "--fast replay" flake this guard
    // exists to prevent). bat-created pages are still blank at construction.
    const url = page.url();
    if (!appCreated && /^https?:/i.test(url)) {
      throw new Error(
        `NetworkTracker attached to an already-navigated page (${url}). Trackers for bat-created pages must be built ` +
          `before the page navigates, or in-flight requests can escape and settlement becomes unsound. Create it before the goto/click.`,
      );
    }
    page.on("request", (req) => {
      if (!TRACKED_TYPES.has(req.resourceType())) return;
      const rec: ObservedRequest = {
        method: req.method(),
        url: req.url(),
        resourceType: req.resourceType(),
        status: null,
        failure: null,
        finished: false,
        startSeq: ++this.startCounter,
        finishSeq: null,
      };
      this.inflight.set(req, rec);
      this.observed.push(rec);
    });
    const complete = (req: Request, failure: string | null) => {
      const rec = this.inflight.get(req);
      if (!rec) return;
      rec.finished = true;
      rec.failure = failure;
      rec.finishSeq = ++this.finishCounter;
      this.inflight.delete(req);
      this.flushIfDrained();
    };
    // Status comes from the synchronous `response` EVENT, never from awaiting
    // req.response() — that promise can stall, and completion bookkeeping
    // (which settlement's drain depends on) must be event-driven and sync.
    page.on("response", (resp) => {
      const rec = this.inflight.get(resp.request());
      if (rec) {
        rec.status = resp.status();
        rec.streaming = resp.headers()["content-length"] === undefined;
        rec.liveStream = (resp.headers()["content-type"] ?? "").includes("text/event-stream");
        if (rec.liveStream) this.flushIfDrained(); // exemption may complete the drain
      }
    });
    page.on("requestfinished", (req) => complete(req, null));
    page.on("requestfailed", (req) => complete(req, req.failure()?.errorText ?? "failed"));

    // websockets: record every frame per step; matchers subscribe via onWsFrame
    page.on("websocket", (ws) => {
      const url = ws.url();
      ws.on("framesent", (data) => this.wsEvent("sent", url, data.payload));
      ws.on("framereceived", (data) => this.wsEvent("received", url, data.payload));
    });
  }

  wsFrames: WsFrame[] = [];
  private wsListeners = new Set<(f: WsFrame) => void>();

  private wsEvent(dir: "sent" | "received", url: string, payload: string | Buffer): void {
    const frame: WsFrame = { dir, url, data: String(payload).slice(0, 500) };
    this.wsFrames.push(frame);
    for (const l of this.wsListeners) l(frame);
  }

  onWsFrame(listener: (f: WsFrame) => void): () => void {
    this.wsListeners.add(listener);
    return () => this.wsListeners.delete(listener);
  }

  private flushIfDrained(): void {
    if (this.drainableCount === 0) {
      const waiters = this.drainWaiters;
      this.drainWaiters = [];
      for (const w of waiters) w();
    }
  }

  /** in-flight requests that GATE settlement (live streams are exempt) */
  get drainableCount(): number {
    return [...this.inflight.values()].filter((r) => !r.liveStream).length;
  }

  get inflightCount(): number {
    return this.inflight.size;
  }

  pendingDescriptions(): string[] {
    return [...this.inflight.values()]
      .filter((r) => !r.liveStream) // live streams are exempt by design, never "stuck"
      .map((r) =>
        r.streaming
          ? `${r.method} ${r.url} (response ${r.status} still streaming)`
          : `${r.method} ${r.url} (still pending)`,
      );
  }

  /** true when every in-flight request has at least received response headers */
  inflightAllHaveStatus(): boolean {
    return [...this.inflight.values()].every((r) => r.status !== null);
  }

  /** Evict in-flight requests whose complete (Content-Length-known) response
   * arrived but whose finished event never came — a lost browser event.
   * Streaming responses (no Content-Length: RSC, SSE) are NEVER evicted:
   * their bodies are genuinely still in flight, and settling early would
   * assert against Suspense fallbacks. Returns trace notes. */
  forceCompleteStragglers(): string[] {
    const notes: string[] = [];
    for (const [req, rec] of [...this.inflight.entries()]) {
      if (rec.status === null || rec.streaming || rec.liveStream) continue;
      rec.finished = true;
      rec.finishSeq = ++this.finishCounter;
      this.inflight.delete(req);
      notes.push(
        `${rec.method} ${rec.url}: complete response ${rec.status} received but the finish event never arrived ` +
          `(lost browser event) — counted as complete after two quiet passes`,
      );
    }
    this.flushIfDrained();
    return notes;
  }

  /** Resolves the moment in-flight drains to zero (immediately if already drained). */
  waitForDrain(signal: AbortSignal): Promise<void> {
    if (this.drainableCount === 0) return Promise.resolve();
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
    this.wsFrames = [];
    this.startCounter = 0;
    this.finishCounter = 0;
  }
}

export interface RequestExpectation {
  method: string;
  pathPattern: string;
  status: "ok" | number;
  /** substring the request BODY must contain (GraphQL operations, payload fields) */
  bodyContains?: string;
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
    const onFinished = (rec: ObservedRequest, postData: string | null) => {
      if (this.result.matched) return;
      if (rec.method !== expectation.method) return;
      const { path, query } = pathOf(rec.url);
      if (!matchPath(expectation.pathPattern, path, query)) return;
      if (expectation.bodyContains !== undefined && !(postData ?? "").includes(expectation.bodyContains)) return;
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
      onFinished(
        {
          method: req.method(),
          url: req.url(),
          resourceType: req.resourceType(),
          status: resp.status(),
          failure: null,
          finished: true,
          startSeq: 0, // synthetic matcher record; ordering lives in NetworkTracker.observed
          finishSeq: null,
        },
        req.postData(),
      );
    };
    this.onReqFailed = (req: Request) => {
      if (!TRACKED_TYPES.has(req.resourceType())) return;
      onFinished(
        {
          method: req.method(),
          url: req.url(),
          resourceType: req.resourceType(),
          status: null,
          failure: req.failure()?.errorText ?? "failed",
          finished: true,
          startSeq: 0,
          finishSeq: null,
        },
        req.postData(),
      );
    };
    page.on("response", this.onResponse);
    page.on("requestfailed", this.onReqFailed);
  }

  /** step boundary: stop listening */
  dispose(): void {
    this.page.off("response", this.onResponse);
    this.page.off("requestfailed", this.onReqFailed);
  }

  describe(): string {
    const e = this.expectation;
    return (
      `declared 'expect request ${e.method} ${e.pathPattern}` +
      (e.bodyContains !== undefined ? ` containing "${e.bodyContains}"` : "") +
      `' never saw a matching request`
    );
  }
}

export interface WsExpectation {
  dir: "sent" | "received";
  text: string;
  pathPattern?: string;
}

/** Armed BEFORE the action: matches a websocket frame by direction, substring,
 * and (optionally) socket path. Subscribes through the tracker so frames on
 * sockets opened in earlier steps are seen too. */
export class WsMatcher implements SettleMatcher {
  result: { matched: WsFrame | null } = { matched: null };
  matchedAndFinished: Promise<void>;
  private unsubscribe: () => void;

  constructor(public expectation: WsExpectation, tracker: NetworkTracker) {
    let resolveMatched!: () => void;
    this.matchedAndFinished = new Promise((resolve) => {
      resolveMatched = resolve;
    });
    this.unsubscribe = tracker.onWsFrame((frame) => {
      if (this.result.matched) return;
      if (frame.dir !== expectation.dir) return;
      if (expectation.pathPattern !== undefined) {
        const { path, query } = pathOf(frame.url);
        if (!matchPath(expectation.pathPattern, path, query)) return;
      }
      if (!frame.data.includes(expectation.text)) return;
      this.result.matched = frame;
      resolveMatched();
    });
  }

  dispose(): void {
    this.unsubscribe();
  }

  describe(): string {
    const e = this.expectation;
    return `declared 'expect ws ${e.dir} "${e.text}"${e.pathPattern ? ` on ${e.pathPattern}` : ""}' never saw a matching frame`;
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
  matchers: SettleMatcher[];
}

export async function settle(page: Page, tracker: NetworkTracker, opts: SettleOptions): Promise<SettleOutcome> {
  const deadline = Date.now() + opts.budgetMs;
  const controller = new AbortController();
  const budgetLeft = () => deadline - Date.now();
  let iterations = 0;
  let clockAdvanced = 0;
  const notes: string[] = [];
  let stragglerPasses = 0;

  // One "quiet pulse": a real roundtrip through both event loops that also
  // measures, PAGE-SIDE (no cross-process race), whether the DOM mutated
  // during the window and whether React streaming content is still hidden
  // awaiting reveal (`<div hidden id="S:...">` / `<template id="B:...">` —
  // React's stable streaming format; Next.js reveals Suspense content long
  // after the network went quiet, gated only on its own scheduler).
  // Under an installed (fake) clock, advance virtual time a deterministic
  // quantum instead so 0-delay timers and debounces run.
  let lastPulse = { mutated: false, pendingBoundaries: 0 };
  const quietPulse = async (): Promise<boolean> => {
    try {
      let mutated = false;
      let pendingBoundaries = 0;
      if (opts.clockInstalled) {
        const before = (await page.evaluate("window.__batMutationCount ?? 0")) as number;
        await tick(page, 16);
        clockAdvanced += 16;
        const after = (await page.evaluate(
          '({ count: window.__batMutationCount ?? 0, boundaries: document.querySelectorAll(\'div[hidden][id^="S:"], template[id^="B:"]\').length })',
        )) as { count: number; boundaries: number };
        mutated = after.count !== before;
        pendingBoundaries = after.boundaries;
      } else {
        // NOTE: no named bindings inside this closure — esbuild-based dev
        // runners (tsx/vitest) inject a __name helper for them, which does
        // not exist inside the page and makes the evaluate throw.
        const result = (await page.evaluate(
          () =>
            new Promise<{ mutated: boolean; boundaries: number }>((resolve) => {
              const w = window as unknown as { __batMutationCount?: number };
              const before = w.__batMutationCount ?? 0;
              requestAnimationFrame(() =>
                requestAnimationFrame(() =>
                  setTimeout(
                    () =>
                      resolve({
                        mutated: (w.__batMutationCount ?? 0) !== before,
                        boundaries: document.querySelectorAll('div[hidden][id^="S:"], template[id^="B:"]').length,
                      }),
                    0,
                  ),
                ),
              );
              // fallback: idle headless pages may produce no frames
              setTimeout(
                () =>
                  resolve({
                    mutated: (w.__batMutationCount ?? 0) !== before,
                    boundaries: document.querySelectorAll('div[hidden][id^="S:"], template[id^="B:"]').length,
                  }),
                50,
              );
            }),
        )) as { mutated: boolean; boundaries: number };
        mutated = result.mutated;
        pendingBoundaries = result.boundaries;
      }
      lastPulse = { mutated, pendingBoundaries };
      return true;
    } catch {
      // evaluate fails mid-navigation; wait for the nav to land, then re-check
      await page.waitForLoadState("domcontentloaded").catch(() => {});
      lastPulse = { mutated: true, pendingBoundaries: 0 };
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
      if (tracker.drainableCount > 0) {
        const drained = await Promise.race([
          tracker.waitForDrain(controller.signal).then(() => true),
          quietPulse().then(() => false),
        ]);
        if (controller.signal.aborted) break;
        if (!drained && tracker.drainableCount > 0) {
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

      // 3. quiet check: JS task queue drained; navigation not mid-flight;
      //    no DOM mutations during the window; no streaming content still
      //    hidden awaiting a framework reveal.
      if (!(await quietPulse())) continue;
      if (lastPulse.mutated || lastPulse.pendingBoundaries > 0) continue;

      // 4. anything new appear during the quiet check?
      if (tracker.drainableCount > 0) continue;
      if (opts.matchers.some((m) => !m.result.matched)) continue;

      return { settled: true, iterations, stuck: [], clockAdvanced, notes };
    }
  } finally {
    clearTimeout(timeout);
  }

  const stuck: string[] = [];
  for (const m of opts.matchers) {
    if (!m.result.matched) {
      stuck.push(m.describe());
    }
  }
  stuck.push(...tracker.pendingDescriptions());
  if (lastPulse.pendingBoundaries > 0) {
    stuck.push(
      `${lastPulse.pendingBoundaries} streamed content boundar${lastPulse.pendingBoundaries === 1 ? "y" : "ies"} received but never revealed by the framework`,
    );
  }
  if (stuck.length === 0 && lastPulse.mutated) {
    stuck.push("the page's DOM never stopped changing (an animation loop or poller? consider 'given stub' or a testid on stable content)");
  }
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
