import type { BrowserContext, Page } from "playwright";
import type { Effect, Flow, Step, Target } from "../dsl/ir.js";
import { formatEffect, formatTarget } from "../dsl/ir.js";
import type { Seed, SessionState, WorldDescription } from "../world/types.js";
import { composeWorld, WorldError } from "../world/algebra.js";
import { ConditionEngine, type ConditionProfile } from "./conditions.js";
import { matchPath, pathOf } from "./patterns.js";
import { NetworkTracker, RequestMatcher, WsMatcher, settle } from "./settle.js";
import { DialogRouter, DownloadWatcher, TabMatcher, waitForTab } from "./interactions.js";
import { TransientHub } from "./transients.js";
import { ariaSnapshotSafe, buildLocator, interpolate, LOOP_MARKER, resolveUnique, TargetError, type Captures, type LoopPins } from "./targets.js";
import {
  newStepTrace,
  type Checkpoint,
  type EffectVerdict,
  type FlowTrace,
  type StepTrace,
} from "./trace.js";
import type { WorldHandle } from "./world-handle.js";

export interface RunOptions {
  baseUrl: string;
  stepBudgetMs: number;
  seedRegistry: Map<string, Seed>;
  world: WorldHandle;
  /** simulated bad conditions (seeded latency / failure injection) */
  conditions?: ConditionProfile;
  /** simulate a slow CPU (chromium only): JS runs N× slower */
  cpuThrottle?: number;
  /** when to capture screenshots (default "on-failure") */
  screenshotMode?: "on-failure" | "steps" | "off";
  /** persist a step screenshot; returns the stored filename (or null) */
  onScreenshot?: (stepIndex: number, png: Buffer) => Promise<string | null> | string | null;
  /** called after each step with the checkpoint (for persistence) */
  onCheckpoint?: (cp: Checkpoint) => Promise<void> | void;
  /** replay support: skip full trace verbosity for fast-forwarded steps */
  quietSteps?: Set<number>;
  /** replay support: stop after executing this DISPLAY-index step (0-based),
   * which can be a single `for each` iteration */
  stopAtDisplay?: number;
}

export class FlowSetupError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FlowSetupError";
  }
}

export function composeFlowWorld(flow: Flow, registry: Map<string, Seed>): WorldDescription | null {
  const seedGivens = flow.givens.filter((g) => g.type === "seed");
  const patchGivens = flow.givens.filter((g) => g.type === "patch");
  if (seedGivens.length === 0 && patchGivens.length === 0) return null;
  const seeds = seedGivens.map((g) => {
    const s = registry.get(g.name);
    if (!s) {
      const known = [...registry.keys()].map((k) => `"${k}"`).join(", ") || "(none found)";
      throw new FlowSetupError(`flow "${flow.name}": unknown seed "${g.name}" — known seeds: ${known}`);
    }
    return s;
  });
  return composeWorld(
    seeds,
    patchGivens.map((g) => ({ type: g.entity, key: g.key, field: g.field, value: g.value })),
  );
}

interface StepContext {
  session: PageSession;
  captures: Captures;
  /** active `for each` element pins (empty outside a loop body) */
  pins: LoopPins;
  baseUrl: string;
  budgetMs: number;
  clockInstalled: boolean;
  allowConsoleErrors: boolean;
  hub: TransientHub;
  engine: ConditionEngine | null;
}

/** The active browsing session: which page is current, plus the per-context
 * routers for tabs/dialogs/downloads. `switch tab` swaps `page` here, and all
 * subsequent effects and settlement run against the new active page. */
export interface PageSession {
  context: BrowserContext;
  page: Page;
  tracker: NetworkTracker;
  dialogs: DialogRouter;
  downloads: DownloadWatcher;
  /** every page bat has an event-wired tracker for, keyed by Page */
  trackers: Map<Page, NetworkTracker>;
  ensureTracked(p: Page): NetworkTracker;
}

export interface FlowEnv {
  clockInstalled: boolean;
  allowConsoleErrors: boolean;
  allowDialogs: boolean;
  hub: TransientHub;
  engine: ConditionEngine | null;
  session: PageSession;
}

export async function prepareContext(
  context: BrowserContext,
  page: Page,
  flow: Flow,
  opts: RunOptions,
): Promise<FlowEnv> {
  const hub = await TransientHub.install(context);
  // slow-CPU perturbation (chromium only): apply BEFORE any navigation so the
  // initial load runs throttled too — maximizes exposure of timing races.
  if (opts.cpuThrottle && opts.cpuThrottle > 1) {
    try {
      const cdp = await context.newCDPSession(page);
      await (cdp.send as (m: string, p: unknown) => Promise<unknown>)("Emulate.setCPUThrottlingRate", { rate: opts.cpuThrottle });
    } catch {
      // non-chromium engines have no CDP; the perturbation is simply unavailable
    }
  }
  // conditions first: stubs are registered after and therefore win —
  // stubbed traffic is hermetic and immune to chaos
  let engine: ConditionEngine | null = null;
  if (opts.conditions) {
    engine = new ConditionEngine(opts.conditions);
    await engine.install(page);
  }
  let clockInstalled = false;
  const allowConsoleErrors = flow.givens.some((g) => g.type === "allow" && g.what === "console-errors");
  const allowDialogs = flow.givens.some((g) => g.type === "allow" && g.what === "dialogs");

  // Tabs, dialogs, and downloads are first-class. New pages are tracked (not
  // punished); dialogs route to declared responses; downloads are collected.
  const dialogs = new DialogRouter(allowDialogs);
  const downloads = new DownloadWatcher();
  const trackers = new Map<Page, NetworkTracker>();
  const session: PageSession = {
    context,
    page,
    tracker: new NetworkTracker(page),
    dialogs,
    downloads,
    trackers,
    ensureTracked(p: Page): NetworkTracker {
      let t = trackers.get(p);
      if (!t) {
        // a page reaching ensureTracked without an existing tracker is one the
        // APP opened (a popup/new tab); bat cannot precede its navigation.
        t = new NetworkTracker(p, /* appCreated */ true);
        trackers.set(p, t);
        dialogs.attach(p);
        downloads.attach(p);
      }
      return t;
    },
  };
  trackers.set(page, session.tracker);
  dialogs.attach(page);
  downloads.attach(page);
  // any new tab/popup gets its own tracker + routers the moment it opens
  context.on("page", (p) => session.ensureTracked(p));

  for (const g of flow.givens) {
    if (g.type === "clock") {
      await page.clock.install({ time: new Date(g.iso) });
      clockInstalled = true;
    }
  }
  for (const g of flow.givens) {
    if (g.type === "stub") {
      await page.route(
        (url) => {
          const { path, query } = pathOf(url.href);
          return matchPath(g.pathPattern, path, query);
        },
        async (route) => {
          if (route.request().method() !== g.method) return route.fallback();
          await route.fulfill({
            status: g.status,
            contentType: "application/json",
            body: JSON.stringify(g.body ?? null),
          });
        },
      );
    }
  }
  for (const g of flow.givens) {
    if (g.type === "user") {
      const session = await opts.world.session(g.key);
      await applySession(context, session, opts.baseUrl);
    }
  }
  return { clockInstalled, allowConsoleErrors, allowDialogs, hub, engine, session };
}

async function applySession(context: BrowserContext, session: SessionState, baseUrl: string): Promise<void> {
  if (session.cookies?.length) {
    await context.addCookies(
      session.cookies.map((c) => {
        if (c.domain) return { path: "/", ...c };
        const { path: _path, ...rest } = c;
        return { ...rest, url: baseUrl };
      }),
    );
  }
  if (session.localStorage && Object.keys(session.localStorage).length) {
    const entries = session.localStorage;
    await context.addInitScript((kv: Record<string, string>) => {
      for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
    }, entries);
  }
}

export async function runSteps(
  flow: Flow,
  context: BrowserContext,
  page: Page,
  opts: RunOptions,
  env: FlowEnv,
  worldMeta: { fingerprint: string | null; verification: FlowTrace["worldVerification"] },
  startAt = 0,
  stopAfter = flow.steps.length - 1,
  captures: Captures = new Map(),
): Promise<FlowTrace> {
  const session = env.session;
  const trace: FlowTrace = {
    flow: flow.name,
    file: flow.file,
    startedAt: new Date().toISOString(),
    worldFingerprint: worldMeta.fingerprint,
    worldVerification: worldMeta.verification,
    conditions: opts.conditions ?? null,
    status: "pass",
    steps: [],
  };

  // `stopAtDisplay` addresses EXECUTED (display) steps, so it can target a
  // single `for each` iteration; `executed` counts display positions.
  const stopAtDisplay = opts.stopAtDisplay;
  let executed = 0;
  let done = false;
  const markRestNotRun = (fromParsed: number) => {
    for (let j = fromParsed; j < flow.steps.length; j++) trace.steps.push(newStepTrace(j, flow.steps[j]!, ""));
  };

  for (let i = 0; i < flow.steps.length && !done; i++) {
    const step = flow.steps[i]!;
    if (i > stopAfter) {
      trace.steps.push(newStepTrace(i, step, ""));
      continue;
    }
    if (i < startAt) {
      const skipped = newStepTrace(i, step, "");
      skipped.status = "pass";
      trace.steps.push(skipped);
      executed++;
      continue;
    }
    const ctx: StepContext = {
      session,
      captures,
      pins: new Map(),
      baseUrl: opts.baseUrl,
      budgetMs: opts.stepBudgetMs,
      clockInstalled: env.clockInstalled,
      allowConsoleErrors: env.allowConsoleErrors,
      hub: env.hub,
      engine: env.engine,
    };

    // `for each` is a runtime loop: it produces MANY executed steps from one
    // parsed step. Expand it in place, pushing each body iteration's trace.
    if (step.action.type === "forEach") {
      const produced = await runForEach(step, ctx, i);
      for (const bt of produced) {
        bt.index = trace.steps.length;
        trace.steps.push(bt);
        executed++;
        if (bt.status === "fail") {
          trace.status = "fail";
          done = true;
          break;
        }
        if (stopAtDisplay !== undefined && executed - 1 >= stopAtDisplay) {
          done = true; // replay reached its target iteration
          break;
        }
      }
      if (done) { markRestNotRun(i + 1); break; }
      continue;
    }

    const stepTrace = await runStep(step, i, ctx);
    const mode = opts.screenshotMode ?? "on-failure";
    if (opts.onScreenshot && mode !== "off" && (stepTrace.status === "fail" || mode === "steps")) {
      const png = await session.page.screenshot({ timeout: 5000 }).catch(() => null);
      if (png) {
        const name = await opts.onScreenshot(i, png);
        if (name) stepTrace.screenshot = name;
      }
    }
    trace.steps.push(stepTrace);
    executed++;
    if (stepTrace.status === "fail") {
      trace.status = "fail";
      markRestNotRun(i + 1);
      break;
    }
    if (stopAtDisplay !== undefined && executed - 1 >= stopAtDisplay) {
      markRestNotRun(i + 1); // replay reached its target step
      break;
    }
    if (opts.onCheckpoint) {
      const [storageState, snapshotId] = await Promise.all([
        context.storageState(),
        opts.world.snapshot().catch(() => null),
      ]);
      await opts.onCheckpoint({
        step: i,
        url: session.page.url(),
        storageState,
        worldFingerprint: worldMeta.fingerprint,
        worldSnapshotId: snapshotId,
      });
    }
  }
  // `for each` produces more executed steps than the parsed flow has, so give
  // every step a contiguous display index (no-op for loop-free flows).
  trace.steps.forEach((s, idx) => { s.index = idx; });
  return trace;
}

async function runStep(step: Step, index: number, ctx: StepContext): Promise<StepTrace> {
  const { session } = ctx;
  const actPage = session.page; // the page the action runs on (arm observers here)
  const tracker = session.ensureTracked(actPage);
  const started = Date.now();
  const deadline = started + ctx.budgetMs;
  const remaining = () => Math.max(1, deadline - Date.now());
  tracker.stepBoundary();
  session.dialogs.stepBoundary();
  session.downloads.stepBoundary();
  const trace = newStepTrace(index, step, actPage.url());

  // ---- collectors for this step
  const consoleHandler = (msg: { type(): string; text(): string }) => {
    if (msg.type() === "error") trace.consoleErrors.push({ kind: "console-error", text: msg.text() });
  };
  const pageErrorHandler = (err: Error) => {
    trace.consoleErrors.push({ kind: "pageerror", text: err.message });
  };
  const navHandler = (frame: { parentFrame(): unknown; url(): string }) => {
    if (frame.parentFrame() === null) trace.navigations.push(frame.url());
  };
  actPage.on("console", consoleHandler);
  actPage.on("pageerror", pageErrorHandler);
  actPage.on("framenavigated", navHandler);

  const armed: ArmedWatchers = {
    requests: new Map(),
    ws: new Map(),
    appear: new Map(),
    gone: new Map(),
    tabs: new Map(),
    dialogs: new Map(),
    downloads: new Map(),
  };
  const allMatchers = () => [
    ...armed.requests.values(),
    ...armed.ws.values(),
    ...armed.tabs.values(),
    ...armed.dialogs.values(),
    ...armed.downloads.values(),
  ];
  ctx.hub.clear();

  try {
    // ---- arm phase: observers exist BEFORE the action (the whole point)
    for (const eff of step.effects) {
      if (eff.type === "request") {
        armed.requests.set(
          eff,
          new RequestMatcher(
            {
              method: eff.method,
              pathPattern: interpolate(eff.pathPattern, ctx.captures),
              status: eff.status,
              ...(eff.bodyContains !== undefined ? { bodyContains: interpolate(eff.bodyContains, ctx.captures) } : {}),
            },
            tracker,
            actPage,
          ),
        );
      } else if (eff.type === "ws") {
        armed.ws.set(
          eff,
          new WsMatcher(
            {
              dir: eff.dir,
              text: interpolate(eff.text, ctx.captures),
              ...(eff.pathPattern !== undefined ? { pathPattern: interpolate(eff.pathPattern, ctx.captures) } : {}),
            },
            tracker,
          ),
        );
      } else if (eff.type === "tab") {
        armed.tabs.set(eff, new TabMatcher(interpolate(eff.path, ctx.captures), session.context));
      } else if (eff.type === "dialog") {
        armed.dialogs.set(
          eff,
          session.dialogs.arm(eff, interpolate(eff.message, ctx.captures), eff.text !== undefined ? interpolate(eff.text, ctx.captures) : undefined),
        );
      } else if (eff.type === "download") {
        armed.downloads.set(eff, session.downloads.arm(interpolate(eff.name, ctx.captures)));
      } else if (eff.type === "appear") {
        const loc = buildLocator(actPage, eff.target, ctx.captures, false, ctx.pins);
        armed.appear.set(eff, await ctx.hub.arm("appear", loc));
      } else if (eff.type === "gone") {
        const loc = buildLocator(actPage, eff.target, ctx.captures, false, ctx.pins);
        const presentAtAct = await loc.first().isVisible().catch(() => false);
        armed.gone.set(eff, { presentAtAct, watcher: await ctx.hub.arm("gone", loc) });
      }
    }

    // ---- act (may switch the active page, e.g. `switch tab`)
    await performAction(step, ctx, remaining);

    // ---- settle against whatever page is now active
    const settlePage = session.page;
    const settleTracker = session.ensureTracked(settlePage);
    const outcome = await settle(settlePage, settleTracker, {
      budgetMs: remaining(),
      clockInstalled: ctx.clockInstalled,
      matchers: allMatchers(),
    });
    trace.settle = outcome;

    // ---- evaluate effects against the settled page.
    // State effects are EVENTUALLY-HOLDS assertions: frameworks (e.g. React
    // transitions) can commit DOM changes after all network and task-queue
    // signals have gone quiet, so a failing check re-evaluates on each DOM
    // mutation tick until it passes or the budget ends.
    // Instant reads first, armed watchers last: a transient that never shows
    // must not starve the other verdicts' budget (and their reports).
    const verdicts = new Map<Effect, EffectVerdict>();
    const instant = step.effects.filter((e) => e.type !== "appear" && e.type !== "gone");
    const armedEffects = step.effects.filter((e) => e.type === "appear" || e.type === "gone");
    for (const eff of [...instant, ...armedEffects]) {
      let verdict = await evaluateEffect(eff, ctx, armed, remaining);
      // Keep the most INFORMATIVE failing verdict: a re-evaluation running as
      // the budget expires can have its read starved ("element not found"
      // with a ~1ms timeout) — that must never overwrite a concrete earlier
      // observation. The explanation's quality may not depend on load.
      let informative = verdict;
      while (!verdict.pass && eff.type !== "request" && eff.type !== "ws" && remaining() > 500) {
        await ctx.hub.waitForNextTick(Math.min(250, remaining()));
        verdict = await evaluateEffect(eff, ctx, armed, remaining);
        if (verdict.pass || !isStarvedRead(verdict)) informative = verdict;
      }
      verdicts.set(eff, verdict.pass ? verdict : informative);
    }
    for (const eff of step.effects) {
      const verdict = verdicts.get(eff)!;
      trace.effects.push(verdict);
      if (verdict.effect.type === "let" && verdict.pass) {
        trace.captures[(verdict.effect as { name: string }).name] =
          ctx.captures.get((verdict.effect as { name: string }).name) ?? "";
      }
    }

    const effectsFailed = trace.effects.filter((e) => !e.pass);
    const consoleFailed = !ctx.allowConsoleErrors && trace.consoleErrors.length > 0;
    const settleFailed = !outcome.settled;
    // an undeclared native dialog during this step is a real failure
    const dialogFailed = session.dialogs.unmodeled.slice();

    if (effectsFailed.length || consoleFailed || settleFailed || dialogFailed.length) {
      trace.status = "fail";
      const reasons: string[] = [];
      if (effectsFailed.length) reasons.push(`${effectsFailed.length} expectation(s) not met`);
      if (settleFailed) reasons.push("the page never settled within the step budget");
      if (consoleFailed) reasons.push(`the page emitted ${trace.consoleErrors.length} error(s) (use 'allow console-errors' to opt out)`);
      for (const u of dialogFailed) reasons.push(u);
      trace.failure = `after '${step.source}': ${reasons.join("; ")}`;
      trace.ariaSnapshot = await ariaSnapshotSafe(session.page);
    } else {
      trace.status = "pass";
    }
  } catch (e) {
    trace.status = "fail";
    trace.failure = e instanceof Error ? e.message : String(e);
    if (e instanceof TargetError && e.target) trace.failedTarget = e.target;
    if (e instanceof TargetError && e.ariaSnapshot) trace.ariaSnapshot = e.ariaSnapshot;
    else trace.ariaSnapshot = await ariaSnapshotSafe(session.page);
  } finally {
    actPage.off("console", consoleHandler);
    actPage.off("pageerror", pageErrorHandler);
    actPage.off("framenavigated", navHandler);
    for (const m of armed.requests.values()) m.dispose();
    for (const m of armed.ws.values()) m.dispose();
    for (const m of armed.tabs.values()) m.dispose();
    for (const m of armed.downloads.values()) m.dispose();
  }

  const finalPage = session.page;
  const finalTracker = session.ensureTracked(finalPage);
  trace.postUrl = finalPage.url();
  trace.requests = [...finalTracker.observed];
  if (session.dialogs.records.length) trace.dialogs = [...session.dialogs.records];
  if (session.downloads.records.length) trace.downloads = session.downloads.records.map((d) => ({ filename: d.filename, savedAs: d.savedAs }));
  if (finalTracker.wsFrames.length) trace.wsFrames = [...finalTracker.wsFrames];
  if (ctx.engine) for (const rec of trace.requests) ctx.engine.annotate(rec);
  if (trace.status === "fail") {
    trace.testids = await finalPage
      .$$eval("[data-testid]", (els) => els.map((el) => el.getAttribute("data-testid")).filter((x): x is string => x !== null))
      .catch(() => []);
  }
  trace.durationMs = Date.now() - started;
  return trace;
}

let loopCounter = 0;

/**
 * Runtime expansion of `for each`. Resolves the collection against the settled
 * page, PINS each element with an injected marker attribute (so an iteration
 * survives the DOM mutating underneath it — removing rows, re-rendering), then
 * runs the body once per element with the loop var bound as a scope. Each
 * iteration's body steps are ordinary settled/explained steps.
 *
 * Determinism (and thus replay) is preserved: the collection is read from a
 * settled point, which is reproducible given the seeded world + prior steps —
 * exactly what the fallback replay tier already reconstructs.
 */
async function runForEach(step: Step, ctx: StepContext, parsedIndex: number): Promise<StepTrace[]> {
  const action = step.action;
  if (action.type !== "forEach") return [];
  const page = ctx.session.page;
  const container = newStepTrace(parsedIndex, step, page.url());
  container.status = "pass";

  const collection = buildLocator(page, action.collection, ctx.captures, false, ctx.pins);
  let n = 0;
  try {
    n = await collection.count();
  } catch (e) {
    container.status = "fail";
    container.failure = `for each: could not resolve ${formatTarget(action.collection)}: ${e instanceof Error ? e.message : String(e)}`;
    return [container];
  }
  container.source = `${step.source}  (${n} match${n === 1 ? "" : "es"})`;
  if (n === 0) return [container]; // empty collection: body runs zero times (not a failure)

  const loopId = `batloop-${parsedIndex}-${++loopCounter}`;
  // pin every matching element with a stable marker (survives sibling mutation)
  await collection.evaluateAll(
    (els, args) => {
      const [attr, id] = args as [string, string];
      els.forEach((el, k) => el.setAttribute(attr, `${id}:${k}`));
    },
    [LOOP_MARKER, loopId] as [string, string],
  );

  const results: StepTrace[] = [container];
  for (let k = 0; k < n; k++) {
    const key = `${loopId}:${k}`;
    let label = `iteration ${k + 1}/${n}`;
    try {
      const txt = (await page.locator(`[${LOOP_MARKER}="${key}"]`).innerText())
        .trim()
        .replace(/\s+/g, " ")
        .slice(0, 48);
      if (txt) label += `: $${action.loopVar}="${txt}"`;
    } catch {
      // element may have been removed by a prior iteration's body; label stays generic
    }
    const iterPins: LoopPins = new Map(ctx.pins);
    iterPins.set(action.loopVar, key);
    const iterCtx: StepContext = { ...ctx, captures: new Map(ctx.captures), pins: iterPins };

    for (const bodyStep of action.body) {
      if (bodyStep.action.type === "forEach") {
        const nested = await runForEach(bodyStep, iterCtx, parsedIndex);
        for (const nt of nested) {
          if (nt.iteration === undefined) nt.iteration = label;
          results.push(nt);
        }
        if (nested.some((t) => t.status === "fail")) return results;
      } else {
        const bt = await runStep(bodyStep, parsedIndex, iterCtx);
        bt.iteration = label;
        results.push(bt);
        if (bt.status === "fail") return results;
      }
    }
  }
  return results;
}

async function performAction(step: Step, ctx: StepContext, remaining: () => number): Promise<void> {
  const a = step.action;
  const { session, captures, pins } = ctx;
  const page = session.page;
  switch (a.type) {
    case "go": {
      const url = new URL(interpolate(a.path, captures), ctx.baseUrl).href;
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: remaining() });
      return;
    }
    case "click":
    case "dblclick":
    case "hover": {
      const loc = await resolveUnique(page, a.target, captures, remaining(), pins);
      if (a.type === "click") await loc.click({ timeout: remaining() });
      else if (a.type === "dblclick") await loc.dblclick({ timeout: remaining() });
      else await loc.hover({ timeout: remaining() });
      return;
    }
    case "fill": {
      const loc = await resolveUnique(page, a.target, captures, remaining(), pins);
      await loc.fill(interpolate(a.value, captures), { timeout: remaining() });
      return;
    }
    case "select": {
      const loc = await resolveUnique(page, a.target, captures, remaining(), pins);
      await loc.selectOption({ label: interpolate(a.value, captures) }, { timeout: remaining() });
      return;
    }
    case "check":
    case "uncheck": {
      const loc = await resolveUnique(page, a.target, captures, remaining(), pins);
      if (a.type === "check") await loc.check({ timeout: remaining() });
      else await loc.uncheck({ timeout: remaining() });
      return;
    }
    case "press": {
      if (a.target) {
        const loc = await resolveUnique(page, a.target, captures, remaining(), pins);
        await loc.press(a.key, { timeout: remaining() });
      } else {
        await page.keyboard.press(a.key);
      }
      return;
    }
    case "upload": {
      const loc = await resolveUnique(page, a.target, captures, remaining(), pins);
      await loc.setInputFiles(interpolate(a.file, captures), { timeout: remaining() });
      return;
    }
    case "drag": {
      const source = await resolveUnique(page, a.target, captures, remaining(), pins);
      const dest = await resolveUnique(page, a.to, captures, remaining(), pins);
      await source.dragTo(dest, { timeout: remaining() });
      return;
    }
    case "switchTab": {
      const target = await waitForTab(session.context, interpolate(a.path, captures), remaining());
      session.page = target;
      session.tracker = session.ensureTracked(target);
      await target.bringToFront().catch(() => {});
      return;
    }
    case "closeTab": {
      const closing = session.page;
      const others = session.context.pages().filter((p) => p !== closing && !p.isClosed());
      if (others.length === 0) {
        throw new Error("close tab: this is the only open tab — nothing to return to");
      }
      await closing.close().catch(() => {});
      const next = others[others.length - 1]!;
      session.page = next;
      session.tracker = session.ensureTracked(next);
      await next.bringToFront().catch(() => {});
      return;
    }
    case "forEach":
      throw new Error("internal: forEach is expanded by runForEach, not performAction");
  }
}

interface ArmedWatchers {
  requests: Map<Effect, RequestMatcher>;
  ws: Map<Effect, WsMatcher>;
  tabs: Map<Effect, TabMatcher>;
  dialogs: Map<Effect, ReturnType<DialogRouter["arm"]>>;
  downloads: Map<Effect, ReturnType<DownloadWatcher["arm"]>>;
  appear: Map<Effect, Awaited<ReturnType<TransientHub["arm"]>>>;
  gone: Map<Effect, { presentAtAct: boolean; watcher: Awaited<ReturnType<TransientHub["arm"]>> }>;
}

async function evaluateEffect(
  eff: Effect,
  ctx: StepContext,
  armed: ArmedWatchers,
  remaining: () => number,
): Promise<EffectVerdict> {
  const { captures, pins } = ctx;
  const page = ctx.session.page;
  const rendered = formatEffect(eff);
  const v = (pass: boolean, observed?: string): EffectVerdict =>
    observed === undefined ? { effect: eff, rendered, pass } : { effect: eff, rendered, pass, observed };

  try {
    switch (eff.type) {
      case "visible": {
        const loc = buildLocator(page, eff.target, captures, false, pins).first();
        const ok = await loc.waitFor({ state: "visible", timeout: remaining() }).then(() => true, () => false);
        return ok ? v(true) : v(false, `no visible ${formatTarget(eff.target)} on ${page.url()}`);
      }
      case "absent": {
        const loc = buildLocator(page, eff.target, captures, false, pins).first();
        const ok = await loc.waitFor({ state: "hidden", timeout: remaining() }).then(() => true, () => false);
        return ok ? v(true) : v(false, `${formatTarget(eff.target)} is still visible`);
      }
      case "appear": {
        const ok = await ctx.hub.waitFor(armed.appear.get(eff)!, remaining());
        return ok ? v(true) : v(false, `${formatTarget(eff.target)} never appeared (watcher was armed before the action)`);
      }
      case "gone": {
        const w = armed.gone.get(eff)!;
        if (!w.presentAtAct) {
          return v(false, `${formatTarget(eff.target)} was not present when the action ran — 'gone' asserts a disappearance`);
        }
        const ok = await ctx.hub.waitFor(w.watcher, remaining());
        return ok ? v(true) : v(false, `${formatTarget(eff.target)} is still visible`);
      }
      case "text": {
        const want = interpolate(eff.value, captures);
        const scope = eff.target ? buildLocator(page, eff.target, captures, false, pins).first() : page.locator("body");
        await scope.waitFor({ state: "visible", timeout: remaining() }).catch(() => {});
        const got = (await scope.innerText({ timeout: remaining() }).catch(() => null)) ?? "(element not found)";
        const m = matchString(got, want, eff.mode);
        return m.ok
          ? v(true)
          : v(false, m.error ?? `${eff.target ? formatTarget(eff.target) : "page"} text is ${JSON.stringify(truncate(got, 200))}`);
      }
      case "value": {
        const want = interpolate(eff.value, captures);
        const loc = buildLocator(page, eff.target, captures, false, pins).first();
        const got = await loc.inputValue({ timeout: remaining() }).catch(() => null);
        if (got === null) return v(false, "(no input found)");
        const m = matchString(got, want, eff.mode);
        return m.ok ? v(true) : v(false, m.error ?? `value is ${JSON.stringify(got)}`);
      }
      case "title": {
        const want = interpolate(eff.value, captures);
        const got = await page.title().catch(() => "");
        const m = matchString(got, want, eff.mode);
        return m.ok ? v(true) : v(false, m.error ?? `title is ${JSON.stringify(got)}`);
      }
      case "attribute": {
        const want = interpolate(eff.value, captures);
        const loc = buildLocator(page, eff.target, captures, false, pins).first();
        const got = await loc.getAttribute(eff.attr, { timeout: remaining() }).catch(() => null);
        if (got === null) return v(false, `${formatTarget(eff.target)} has no "${eff.attr}" attribute`);
        const m = matchString(got, want, eff.mode);
        return m.ok ? v(true) : v(false, m.error ?? `"${eff.attr}" is ${JSON.stringify(got)}`);
      }
      case "enabled":
      case "disabled": {
        const loc = buildLocator(page, eff.target, captures, false, pins).first();
        const got = await loc.isEnabled({ timeout: remaining() }).catch(() => null);
        if (got === null) return v(false, `${formatTarget(eff.target)} not found`);
        const pass = eff.type === "enabled" ? got : !got;
        return pass ? v(true) : v(false, `${formatTarget(eff.target)} is ${got ? "enabled" : "disabled"}`);
      }
      case "checked":
      case "unchecked": {
        const loc = buildLocator(page, eff.target, captures, false, pins).first();
        const got = await loc.isChecked({ timeout: remaining() }).catch(() => null);
        if (got === null) return v(false, `${formatTarget(eff.target)} is not a checkable element (or not found)`);
        const pass = eff.type === "checked" ? got : !got;
        return pass ? v(true) : v(false, `${formatTarget(eff.target)} is ${got ? "checked" : "not checked"}`);
      }
      case "selected": {
        const want = interpolate(eff.value, captures);
        const loc = buildLocator(page, eff.target, captures, false, pins).first();
        const got = (await loc
          .evaluate((el) => (el instanceof HTMLSelectElement ? (el.selectedOptions[0]?.label ?? null) : null))
          .catch(() => null)) as string | null;
        if (got === null) return v(false, `${formatTarget(eff.target)} is not a <select> (or nothing is selected)`);
        return got === want ? v(true) : v(false, `selected option is ${JSON.stringify(got)}`);
      }
      case "count": {
        const target: Target = eff.name !== undefined ? { kind: eff.kind, name: eff.name } : { kind: eff.kind };
        if (eff.within) target.within = eff.within;
        const loc = buildLocator(page, target, captures, false, pins);
        const got = await loc.count();
        const ok =
          eff.op === ">=" ? got >= eff.n :
          eff.op === "<=" ? got <= eff.n :
          eff.op === ">" ? got > eff.n :
          eff.op === "<" ? got < eff.n :
          got === eff.n;
        return ok ? v(true) : v(false, `found ${got} (needed ${eff.op === "=" ? "" : eff.op}${eff.n})`);
      }
      case "url": {
        const want = interpolate(eff.path, captures);
        const { path, query } = pathOf(page.url());
        const pass = matchPath(want, path, query);
        return pass ? v(true) : v(false, `url is ${path}${query ? `?${query}` : ""}`);
      }
      case "request": {
        const m = armed.requests.get(eff);
        if (!m) return v(false, "internal: no matcher was armed for this expectation");
        if (!m.result.matched) {
          return v(
            false,
            `no ${eff.method} request matching ${eff.pathPattern}` +
              (eff.bodyContains !== undefined ? ` with a body containing ${JSON.stringify(eff.bodyContains)}` : "") +
              ` was made during this step`,
          );
        }
        if (!m.result.statusOk) {
          const r = m.result.matched;
          return v(false, `${r.method} ${r.url} responded ${r.failure ? `FAILED (${r.failure})` : r.status}`);
        }
        return v(true);
      }
      case "ws": {
        const m = armed.ws.get(eff);
        if (!m) return v(false, "internal: no ws matcher was armed for this expectation");
        if (!m.result.matched) {
          return v(
            false,
            `no ${eff.dir} websocket frame containing ${JSON.stringify(eff.text)}` +
              (eff.pathPattern !== undefined ? ` on ${eff.pathPattern}` : "") +
              ` was observed during this step (watcher was armed before the action)`,
          );
        }
        return v(true);
      }
      case "tab": {
        const m = armed.tabs.get(eff);
        if (!m) return v(false, "internal: no tab matcher was armed for this expectation");
        if (!m.result.matched) {
          const open = ctx.session.context.pages().map((p) => p.url() || "about:blank").join(", ");
          return v(false, `no tab/popup with a url matching ${eff.path} opened (open tabs: ${open})`);
        }
        return v(true);
      }
      case "dialog": {
        const m = armed.dialogs.get(eff);
        if (!m) return v(false, "internal: no dialog matcher was armed for this expectation");
        if (!m.result.matched) {
          return v(false, `no dialog whose message contains ${JSON.stringify(eff.message)} appeared during this step`);
        }
        return v(true);
      }
      case "download": {
        const m = armed.downloads.get(eff);
        if (!m) return v(false, "internal: no download matcher was armed for this expectation");
        if (!(m.result.matched as { filename: string } | null)) {
          const seen = ctx.session.downloads.records.map((d) => d.filename).join(", ") || "none";
          return v(false, `no download whose filename contains ${JSON.stringify(eff.name)} started (downloads this step: ${seen})`);
        }
        return v(true);
      }
      case "let": {
        const src = eff.from;
        let captured: string;
        if (src.kind === "query") {
          const u = new URL(page.url());
          const q = u.searchParams.get(interpolate(src.param, captures));
          if (q === null) return v(false, `no query parameter "${src.param}" in ${page.url()}`);
          captured = q;
        } else if (src.kind === "count") {
          const target: Target = src.name !== undefined ? { kind: src.countKind, name: src.name } : { kind: src.countKind };
          if (src.within) target.within = src.within;
          captured = String(await buildLocator(page, target, captures, false, pins).count());
        } else {
          const loc = await resolveUnique(page, src.target, captures, remaining(), pins);
          if (src.kind === "text") captured = (await loc.innerText()).trim();
          else if (src.kind === "value") captured = await loc.inputValue();
          else {
            const a = await loc.getAttribute(interpolate(src.attr, captures));
            if (a === null) return v(false, `element has no "${src.attr}" attribute`);
            captured = a;
          }
        }
        captures.set(eff.name, captured);
        return v(true, `captured $${eff.name} = ${JSON.stringify(captured)}`);
      }
    }
  } catch (e) {
    return v(false, e instanceof Error ? e.message : String(e));
  }
}

/** Compare an observed string against a wanted value in one of three modes. */
function matchString(got: string, want: string, mode: "contains" | "exact" | "matches"): { ok: boolean; error?: string } {
  if (mode === "exact") return { ok: got.trim() === want };
  if (mode === "contains") return { ok: got.includes(want) };
  // matches: `want` is a JS regex. Accept both a bare source and a /…/flags
  // literal (`/Today is \d{4}/i`). Invalid patterns are a flow authoring error.
  try {
    const lit = /^\/(.+)\/([a-z]*)$/s.exec(want);
    const re = lit ? new RegExp(lit[1]!, lit[2]) : new RegExp(want);
    return { ok: re.test(got) };
  } catch (e) {
    return { ok: false, error: `invalid regex ${JSON.stringify(want)}: ${e instanceof Error ? e.message : String(e)}` };
  }
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}…` : s;
}

/** a failure whose read itself was cut off by an exhausted budget — carries
 * no information about the page, only about the clock */
function isStarvedRead(v: EffectVerdict): boolean {
  return v.observed !== undefined && (v.observed.includes("(element not found)") || v.observed.includes("(no input found)"));
}
