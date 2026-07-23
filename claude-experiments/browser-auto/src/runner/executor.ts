import type { BrowserContext, Page } from "playwright";
import type { Effect, Flow, Step, Target } from "../dsl/ir.js";
import { formatEffect, formatTarget } from "../dsl/ir.js";
import type { Seed, SessionState, WorldDescription } from "../world/types.js";
import { composeWorld, WorldError } from "../world/algebra.js";
import { ConditionEngine, type ConditionProfile } from "./conditions.js";
import { matchPath, pathOf } from "./patterns.js";
import { NetworkTracker, RequestMatcher, settle } from "./settle.js";
import { TransientHub } from "./transients.js";
import { ariaSnapshotSafe, buildLocator, interpolate, resolveUnique, TargetError, type Captures } from "./targets.js";
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
  /** called after each step with the checkpoint (for persistence) */
  onCheckpoint?: (cp: Checkpoint) => Promise<void> | void;
  /** replay support: skip full trace verbosity for fast-forwarded steps */
  quietSteps?: Set<number>;
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
  page: Page;
  tracker: NetworkTracker;
  captures: Captures;
  baseUrl: string;
  budgetMs: number;
  clockInstalled: boolean;
  allowConsoleErrors: boolean;
  hub: TransientHub;
  engine: ConditionEngine | null;
}

export interface FlowEnv {
  clockInstalled: boolean;
  allowConsoleErrors: boolean;
  hub: TransientHub;
  engine: ConditionEngine | null;
}

export async function prepareContext(
  context: BrowserContext,
  page: Page,
  flow: Flow,
  opts: RunOptions,
): Promise<FlowEnv> {
  const hub = await TransientHub.install(context);
  // conditions first: stubs are registered after and therefore win —
  // stubbed traffic is hermetic and immune to chaos
  let engine: ConditionEngine | null = null;
  if (opts.conditions) {
    engine = new ConditionEngine(opts.conditions);
    await engine.install(page);
  }
  let clockInstalled = false;
  const allowConsoleErrors = flow.givens.some((g) => g.type === "allow" && g.what === "console-errors");

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
  return { clockInstalled, allowConsoleErrors, hub, engine };
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
  const tracker = new NetworkTracker(page);
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

  for (let i = 0; i < flow.steps.length; i++) {
    const step = flow.steps[i]!;
    if (i > stopAfter) {
      trace.steps.push(newStepTrace(i, step, ""));
      continue;
    }
    if (i < startAt) {
      const skipped = newStepTrace(i, step, "");
      skipped.status = "pass";
      trace.steps.push(skipped);
      continue;
    }
    const ctx: StepContext = {
      page,
      tracker,
      captures,
      baseUrl: opts.baseUrl,
      budgetMs: opts.stepBudgetMs,
      clockInstalled: env.clockInstalled,
      allowConsoleErrors: env.allowConsoleErrors,
      hub: env.hub,
      engine: env.engine,
    };
    const stepTrace = await runStep(step, i, ctx);
    trace.steps.push(stepTrace);
    if (stepTrace.status === "fail") {
      trace.status = "fail";
      for (let j = i + 1; j < flow.steps.length; j++) {
        trace.steps.push(newStepTrace(j, flow.steps[j]!, ""));
      }
      break;
    }
    if (opts.onCheckpoint) {
      const [storageState, snapshotId] = await Promise.all([
        context.storageState(),
        opts.world.snapshot().catch(() => null),
      ]);
      await opts.onCheckpoint({
        step: i,
        url: page.url(),
        storageState,
        worldFingerprint: worldMeta.fingerprint,
        worldSnapshotId: snapshotId,
      });
    }
  }
  return trace;
}

async function runStep(step: Step, index: number, ctx: StepContext): Promise<StepTrace> {
  const { page, tracker } = ctx;
  const started = Date.now();
  const deadline = started + ctx.budgetMs;
  const remaining = () => Math.max(1, deadline - Date.now());
  tracker.stepBoundary();
  const trace = newStepTrace(index, step, page.url());

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
  page.on("console", consoleHandler);
  page.on("pageerror", pageErrorHandler);
  page.on("framenavigated", navHandler);

  const matchers: RequestMatcher[] = [];
  const armed: ArmedWatchers = { matchers, appear: new Map(), gone: new Map() };
  ctx.hub.clear();

  try {
    // ---- arm phase: observers exist BEFORE the action (the whole point)
    for (const eff of step.effects) {
      if (eff.type === "request") {
        matchers.push(
          new RequestMatcher(
            { method: eff.method, pathPattern: interpolate(eff.pathPattern, ctx.captures), status: eff.status },
            tracker,
            page,
          ),
        );
      } else if (eff.type === "appear") {
        const loc = buildLocator(page, eff.target, ctx.captures);
        armed.appear.set(eff, await ctx.hub.arm("appear", loc));
      } else if (eff.type === "gone") {
        const loc = buildLocator(page, eff.target, ctx.captures);
        const presentAtAct = await loc.first().isVisible().catch(() => false);
        armed.gone.set(eff, { presentAtAct, watcher: await ctx.hub.arm("gone", loc) });
      }
    }

    // ---- act
    await performAction(step, ctx, remaining);

    // ---- settle
    const outcome = await settle(page, tracker, {
      budgetMs: remaining(),
      clockInstalled: ctx.clockInstalled,
      matchers,
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
      while (!verdict.pass && eff.type !== "request" && remaining() > 100) {
        await ctx.hub.waitForNextTick(Math.min(250, remaining()));
        verdict = await evaluateEffect(eff, ctx, armed, remaining);
      }
      verdicts.set(eff, verdict);
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

    if (effectsFailed.length || consoleFailed || settleFailed) {
      trace.status = "fail";
      const reasons: string[] = [];
      if (effectsFailed.length) reasons.push(`${effectsFailed.length} expectation(s) not met`);
      if (settleFailed) reasons.push("the page never settled within the step budget");
      if (consoleFailed) reasons.push(`the page emitted ${trace.consoleErrors.length} error(s) (use 'allow console-errors' to opt out)`);
      trace.failure = `after '${step.source}': ${reasons.join("; ")}`;
      trace.ariaSnapshot = await ariaSnapshotSafe(page);
    } else {
      trace.status = "pass";
    }
  } catch (e) {
    trace.status = "fail";
    trace.failure = e instanceof Error ? e.message : String(e);
    if (e instanceof TargetError && e.target) trace.failedTarget = e.target;
    if (e instanceof TargetError && e.ariaSnapshot) trace.ariaSnapshot = e.ariaSnapshot;
    else trace.ariaSnapshot = await ariaSnapshotSafe(page);
  } finally {
    page.off("console", consoleHandler);
    page.off("pageerror", pageErrorHandler);
    page.off("framenavigated", navHandler);
    for (const m of matchers) m.dispose();
  }

  trace.postUrl = page.url();
  trace.requests = [...tracker.observed];
  if (ctx.engine) for (const rec of trace.requests) ctx.engine.annotate(rec);
  if (trace.status === "fail") {
    trace.testids = await page
      .$$eval("[data-testid]", (els) => els.map((el) => el.getAttribute("data-testid")).filter((x): x is string => x !== null))
      .catch(() => []);
  }
  trace.durationMs = Date.now() - started;
  return trace;
}

async function performAction(step: Step, ctx: StepContext, remaining: () => number): Promise<void> {
  const a = step.action;
  const { page, captures } = ctx;
  switch (a.type) {
    case "go": {
      const url = new URL(interpolate(a.path, captures), ctx.baseUrl).href;
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: remaining() });
      return;
    }
    case "click":
    case "dblclick":
    case "hover": {
      const loc = await resolveUnique(page, a.target, captures, remaining());
      if (a.type === "click") await loc.click({ timeout: remaining() });
      else if (a.type === "dblclick") await loc.dblclick({ timeout: remaining() });
      else await loc.hover({ timeout: remaining() });
      return;
    }
    case "fill": {
      const loc = await resolveUnique(page, a.target, captures, remaining());
      await loc.fill(interpolate(a.value, captures), { timeout: remaining() });
      return;
    }
    case "select": {
      const loc = await resolveUnique(page, a.target, captures, remaining());
      await loc.selectOption({ label: interpolate(a.value, captures) }, { timeout: remaining() });
      return;
    }
    case "check":
    case "uncheck": {
      const loc = await resolveUnique(page, a.target, captures, remaining());
      if (a.type === "check") await loc.check({ timeout: remaining() });
      else await loc.uncheck({ timeout: remaining() });
      return;
    }
    case "press": {
      if (a.target) {
        const loc = await resolveUnique(page, a.target, captures, remaining());
        await loc.press(a.key, { timeout: remaining() });
      } else {
        await page.keyboard.press(a.key);
      }
      return;
    }
    case "upload": {
      const loc = await resolveUnique(page, a.target, captures, remaining());
      await loc.setInputFiles(interpolate(a.file, captures), { timeout: remaining() });
      return;
    }
  }
}

interface ArmedWatchers {
  matchers: RequestMatcher[];
  appear: Map<Effect, Awaited<ReturnType<TransientHub["arm"]>>>;
  gone: Map<Effect, { presentAtAct: boolean; watcher: Awaited<ReturnType<TransientHub["arm"]>> }>;
}

async function evaluateEffect(
  eff: Effect,
  ctx: StepContext,
  armed: ArmedWatchers,
  remaining: () => number,
): Promise<EffectVerdict> {
  const { page, captures } = ctx;
  const rendered = formatEffect(eff);
  const v = (pass: boolean, observed?: string): EffectVerdict =>
    observed === undefined ? { effect: eff, rendered, pass } : { effect: eff, rendered, pass, observed };

  try {
    switch (eff.type) {
      case "visible": {
        const loc = buildLocator(page, eff.target, captures).first();
        const ok = await loc.waitFor({ state: "visible", timeout: remaining() }).then(() => true, () => false);
        return ok ? v(true) : v(false, `no visible ${formatTarget(eff.target)} on ${page.url()}`);
      }
      case "absent": {
        const loc = buildLocator(page, eff.target, captures).first();
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
        const scope = eff.target ? buildLocator(page, eff.target, captures).first() : page.locator("body");
        await scope.waitFor({ state: "visible", timeout: remaining() }).catch(() => {});
        const got = (await scope.innerText({ timeout: remaining() }).catch(() => null)) ?? "(element not found)";
        const pass = eff.exact ? got.trim() === want : got.includes(want);
        return pass
          ? v(true)
          : v(false, `${eff.target ? formatTarget(eff.target) : "page"} text is ${JSON.stringify(truncate(got, 200))}`);
      }
      case "value": {
        const want = interpolate(eff.value, captures);
        const loc = buildLocator(page, eff.target, captures).first();
        const got = await loc.inputValue({ timeout: remaining() }).catch(() => null);
        return got === want ? v(true) : v(false, `value is ${got === null ? "(no input found)" : JSON.stringify(got)}`);
      }
      case "enabled":
      case "disabled": {
        const loc = buildLocator(page, eff.target, captures).first();
        const got = await loc.isEnabled({ timeout: remaining() }).catch(() => null);
        if (got === null) return v(false, `${formatTarget(eff.target)} not found`);
        const pass = eff.type === "enabled" ? got : !got;
        return pass ? v(true) : v(false, `${formatTarget(eff.target)} is ${got ? "enabled" : "disabled"}`);
      }
      case "checked":
      case "unchecked": {
        const loc = buildLocator(page, eff.target, captures).first();
        const got = await loc.isChecked({ timeout: remaining() }).catch(() => null);
        if (got === null) return v(false, `${formatTarget(eff.target)} is not a checkable element (or not found)`);
        const pass = eff.type === "checked" ? got : !got;
        return pass ? v(true) : v(false, `${formatTarget(eff.target)} is ${got ? "checked" : "not checked"}`);
      }
      case "selected": {
        const want = interpolate(eff.value, captures);
        const loc = buildLocator(page, eff.target, captures).first();
        const got = (await loc
          .evaluate((el) => (el instanceof HTMLSelectElement ? (el.selectedOptions[0]?.label ?? null) : null))
          .catch(() => null)) as string | null;
        if (got === null) return v(false, `${formatTarget(eff.target)} is not a <select> (or nothing is selected)`);
        return got === want ? v(true) : v(false, `selected option is ${JSON.stringify(got)}`);
      }
      case "count": {
        const target: Target = eff.name !== undefined ? { kind: eff.kind, name: eff.name } : { kind: eff.kind };
        if (eff.within) target.within = eff.within;
        const loc = buildLocator(page, target, captures);
        const got = await loc.count();
        return got === eff.n ? v(true) : v(false, `found ${got}`);
      }
      case "url": {
        const want = interpolate(eff.path, captures);
        const { path, query } = pathOf(page.url());
        const pass = matchPath(want, path, query);
        return pass ? v(true) : v(false, `url is ${path}${query ? `?${query}` : ""}`);
      }
      case "request": {
        const m = armed.matchers.find((mm) => mm.expectation.pathPattern === interpolate(eff.pathPattern, captures) && mm.expectation.method === eff.method);
        if (!m) return v(false, "internal: no matcher was armed for this expectation");
        if (!m.result.matched) {
          return v(false, `no ${eff.method} request matching ${eff.pathPattern} was made during this step`);
        }
        if (!m.result.statusOk) {
          const r = m.result.matched;
          return v(false, `${r.method} ${r.url} responded ${r.failure ? `FAILED (${r.failure})` : r.status}`);
        }
        return v(true);
      }
      case "let": {
        const loc = await resolveUnique(page, eff.from, captures, remaining());
        const text = (await loc.innerText()).trim();
        captures.set(eff.name, text);
        return v(true, `captured $${eff.name} = ${JSON.stringify(text)}`);
      }
    }
  } catch (e) {
    return v(false, e instanceof Error ? e.message : String(e));
  }
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}…` : s;
}
