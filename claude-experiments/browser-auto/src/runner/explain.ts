import type { Effect, Flow, Target } from "../dsl/ir.js";
import { formatTarget } from "../dsl/ir.js";
import type { RunDeps } from "./run.js";
import { completionOrder, type FlowTrace, type StepTrace } from "./trace.js";

/**
 * Automatic failure explanation. When a run fails, bat does not hand down a
 * verdict — it cannot: an app's timing may legitimately vary, and whether an
 * observed state is "broken" depends on user expectations, which only the
 * flow's author knows. What bat CAN do is replace "couldn't find X after n
 * seconds" with a causal account:
 *
 *   - exactly what was expected and what the settled page showed instead,
 *   - what the page did during the step (requests, ordering, errors),
 *   - whether the failure reproduces under identical reruns (same seeded
 *     world, same steps) — and when it doesn't, what the outcome tracks
 *     (e.g. response completion order),
 *   - and, where interpretation is required, both readings — so deciding
 *     whether the app or the expectation should change takes a minute,
 *     not an afternoon.
 */

export interface OrderOutcome {
  order: string;
  /** runs with this ordering that reached the expected state */
  reached: number;
  /** runs with this ordering that did not */
  missed: number;
}

export interface Explanation {
  /** terse lines naming what failed */
  failed: string[];
  /** causal account of the step: requests (with completion order), errors, navigation */
  whatHappened: string[];
  /** findings from automatic identical reruns */
  reproducibility: string[];
  /** neutral interpretation help — states facts and both readings, never a verdict */
  meaning: string[];
  reruns: { total: number; reachedExpected: number; failedSame: number; failedOther: number };
  orderEvidence?: OrderOutcome[];
}

export interface ExplainDeps extends RunDeps {
  /** rerun the flow once, without explanation or persistence; collects the trace */
  rerun: (deps: RunDeps) => Promise<FlowTrace>;
}

export async function explainFailure(flow: Flow, original: FlowTrace, deps: ExplainDeps): Promise<Explanation> {
  const failedStep = original.steps.find((s) => s.status === "fail");
  const none = { total: 0, reachedExpected: 0, failedSame: 0, failedOther: 0 };
  if (!failedStep) {
    return {
      failed: ["no failing step found in the trace (runner bug — please report)"],
      whatHappened: [],
      reproducibility: [],
      meaning: [],
      reruns: none,
    };
  }

  const failed = describeFailed(failedStep);
  const whatHappened = describeStep(failedStep);

  // ---- ambiguity: the flow names more than one element; reruns add nothing
  if (failedStep.failure?.includes("is ambiguous:")) {
    return {
      failed,
      whatHappened,
      reproducibility: [],
      meaning: [
        "the flow must identify exactly one element — bat never guesses between matches.",
        "scope the target ('in <container>') or use a testid; the matches are listed above.",
      ],
      reruns: none,
    };
  }

  const suggestions = nearMissSuggestions(failedStep);

  // ---- reruns: does this reproduce under identical conditions?
  const rerunCount = deps.config.rerunsOnFailure ?? 4;
  const runs: FlowTrace[] = [original];
  for (let i = 0; i < rerunCount; i++) {
    runs.push(await deps.rerun(deps));
  }
  const sigOf = (t: FlowTrace): string | null => {
    const s = t.steps.find((st) => st.status === "fail");
    if (!s) return null;
    return `${s.index}|${s.effects.filter((e) => !e.pass).map((e) => e.rendered).join("|")}|${s.failure?.split("\n")[0] ?? ""}`;
  };
  const originalSig = sigOf(original);
  const rerunTraces = runs.slice(1);
  const reachedExpected = rerunTraces.filter((t) => t.status === "pass").length;
  const failedSame = rerunTraces.filter((t) => t.status === "fail" && sigOf(t) === originalSig).length;
  const failedOther = rerunTraces.filter((t) => t.status === "fail" && sigOf(t) !== originalSig).length;
  const reruns = { total: rerunCount, reachedExpected, failedSame, failedOther };

  const reproducibility: string[] = [];
  const meaning: string[] = [];
  let orderEvidence: OrderOutcome[] | undefined;

  const worldNote = original.worldFingerprint ? `world ${original.worldFingerprint}, ` : "";

  // ---- conditions: does the failure exist without the injected chaos?
  if (deps.config.conditions && reachedExpected === 0) {
    const { conditions: _drop, ...cleanConfig } = deps.config;
    const clean = await deps.rerun({ ...deps, config: cleanConfig });
    if (clean.status === "pass") {
      reproducibility.push(
        `every rerun under the injected conditions failed; a rerun WITHOUT them reached the expected state.`,
      );
      meaning.push(
        `this failure only occurs under the injected conditions (${conditionText(deps.config.conditions)}).`,
        `injected latency/failures are marked [injected …] in the network listing above.`,
        `→ if the app should tolerate these conditions, what's missing is app resilience;`,
        `→ if not, relax the condition profile (or raise the step budget for pure latency).`,
      );
      return { failed, whatHappened, reproducibility, meaning, reruns };
    }
  }

  if (reachedExpected > 0) {
    reproducibility.push(
      `NOT deterministic: ${rerunCount} automatic rerun(s) of the identical flow (${worldNote}same steps) — ` +
        `${reachedExpected} reached the expected state, ${failedSame + failedOther} did not.`,
    );
    orderEvidence = orderCrossTab(runs, failedStep.index);
    const determined = orderEvidence.length > 1 && orderEvidence.every((o) => o.reached === 0 || o.missed === 0);
    if (determined) {
      reproducibility.push("the outcome tracks response completion order exactly:");
      for (const o of orderEvidence) {
        reproducibility.push(`  ${o.order}  →  expected state in ${o.reached} of ${o.reached + o.missed} run(s)`);
      }
      const good = orderEvidence.filter((o) => o.missed === 0).map((o) => o.order);
      meaning.push(
        `the expected state occurs only under ${good.length === 1 ? `this response ordering: ${good[0]}` : "some response orderings"}.`,
        `response order varying between runs is normal app timing; the question is what a user should see afterwards:`,
        `→ if a user should ALWAYS end up in the expected state here, the app doesn't guarantee it under every ordering;`,
        `→ if every ordering's end state is acceptable, the flow's expectation is stricter than the app's actual contract.`,
      );
    } else if (orderEvidence.length > 1) {
      reproducibility.push(
        "response completion order varied between runs but does not fully explain the difference.",
      );
      meaning.push(
        "something else differs between passing and failing runs — rerun traces are saved next to this run's trace for comparison.",
      );
    } else {
      meaning.push(
        "the runs' network traffic was identical; the difference is elsewhere (rendering, app state). Rerun traces are saved next to this run's trace.",
      );
    }
  } else if (failedOther > 0) {
    reproducibility.push(
      `NOT deterministic: every run failed, but not always the same way — ${failedSame + 1} like this one, ${failedOther} differently (rerun traces saved next to this run's trace).`,
    );
  } else {
    reproducibility.push(
      `fully reproducible: the original and all ${rerunCount} automatic rerun(s) (${worldNote}same steps) failed identically, observing the same values every time.`,
    );
    meaning.push(
      "this is stable behavior, not a timing variation — rerunning will never change it.",
      "the app reliably reaches a state different from the one the flow expects; which of the two should change depends on what a user is supposed to see here.",
    );
  }

  if (suggestions.length > 0) {
    meaning.push(...suggestions);
  }

  const result: Explanation = { failed, whatHappened, reproducibility, meaning, reruns };
  if (orderEvidence) result.orderEvidence = orderEvidence;
  return result;
}

// ---------------------------------------------------------------------------

function describeFailed(step: StepTrace): string[] {
  const out: string[] = [];
  for (const v of step.effects) {
    if (v.pass) continue;
    out.push(v.observed ? `${v.rendered} — the settled page showed: ${v.observed}` : v.rendered);
  }
  if (out.length === 0 && step.failure) out.push(step.failure.split("\n")[0]!);
  if (step.settle && !step.settle.settled) {
    out.push(`the page never settled: ${step.settle.stuck.join("; ")}`);
  }
  if (step.consoleErrors.length) {
    out.push(`the page emitted ${step.consoleErrors.length} error(s): ${step.consoleErrors[0]!.text.slice(0, 120)}`);
  }
  return out;
}

function describeStep(step: StepTrace): string[] {
  const out: string[] = [];
  const api = step.requests.filter((r) => r.resourceType !== "document");
  if (api.length) {
    out.push(
      `during '${step.source}' the page issued: ` +
        api
          .slice(0, 8)
          .map(
            (r) =>
              `${r.method} ${new URL(r.url).pathname} -> ${r.failure ? `FAILED (${r.failure})` : r.status}` +
              (r.finishSeq !== null ? ` (finished #${r.finishSeq})` : " (never finished)") +
              (r.injected ? ` [${r.injected}]` : ""),
          )
          .join("; ") +
        (api.length > 8 ? ` … and ${api.length - 8} more` : ""),
    );
  } else {
    out.push(`during '${step.source}' the page issued no tracked requests.`);
  }
  if (step.postUrl && step.postUrl !== step.preUrl) {
    out.push(`the page navigated: ${step.preUrl} → ${step.postUrl}`);
  }
  return out;
}

function conditionText(c: { latencyMs?: [number, number]; failRate?: number; seed: number }): string {
  const parts: string[] = [];
  if (c.latencyMs) parts.push(`latency +${c.latencyMs[0]}–${c.latencyMs[1]}ms`);
  if (c.failRate) parts.push(`${Math.round(c.failRate * 100)}% request failures`);
  parts.push(`seed ${c.seed}`);
  return parts.join(", ");
}

// ---------------------------------------------------------------------------
// near-miss detection: the target doesn't exist, but something similar does

function nearMissSuggestions(step: StepTrace): string[] {
  const wanted: Target[] = [];
  if (step.failedTarget) wanted.push(step.failedTarget);
  for (const v of step.effects) {
    if (v.pass) continue;
    const missing =
      v.observed !== undefined &&
      (v.observed.startsWith("no visible") || v.observed.includes("(element not found)") || v.observed.includes("no visible match"));
    if (!missing) continue;
    const eff = v.effect as Effect;
    const target = "target" in eff && eff.target ? eff.target : eff.type === "let" ? eff.from : undefined;
    if (target) wanted.push(target);
  }

  const candidates = snapshotCandidates(step.ariaSnapshot ?? "");
  const suggestions: string[] = [];
  for (const t of wanted) {
    if (t.name === undefined) continue;
    if (t.kind === "testid") {
      const best = bestMatch(t.name, (step.testids ?? []).map((id) => ({ kind: "testid", name: id })));
      if (best) {
        suggestions.push(
          `nothing on the page matches ${formatTarget(t)}; the closest present testid is "${best.name}" — if that is the element the flow means, the flow's name for it is wrong.`,
        );
      }
      continue;
    }
    const sameKind = candidates.filter((c) => c.kind === t.kind);
    const best = bestMatch(t.name, sameKind) ?? bestMatch(t.name, candidates);
    if (best) {
      suggestions.push(
        `nothing on the page matches ${formatTarget(t)}; closest present: ${best.kind} "${best.name}" — if that is the element the flow means, the flow's name for it is wrong.`,
      );
    }
  }
  return [...new Set(suggestions)];
}

function bestMatch(name: string, candidates: Array<{ kind: string; name: string }>): { kind: string; name: string } | null {
  let best: { kind: string; name: string } | null = null;
  let bestScore = 0.55; // below this, suggesting would mislead
  for (const c of candidates) {
    if (c.name === name) continue; // an exact match means the name isn't the problem
    const s = similarity(name, c.name);
    if (s > bestScore) {
      best = c;
      bestScore = s;
    }
  }
  return best;
}

export function snapshotCandidates(snapshot: string): Array<{ kind: string; name: string }> {
  const out: Array<{ kind: string; name: string }> = [];
  for (const m of snapshot.matchAll(/-\s+([a-z]+)\s+"((?:[^"\\]|\\.)*)"/g)) {
    out.push({ kind: m[1]!, name: m[2]! });
  }
  return out;
}

export function similarity(a: string, b: string): number {
  const x = a.toLowerCase().trim();
  const y = b.toLowerCase().trim();
  if (x === y) return 1;
  if (x.includes(y) || y.includes(x)) return 0.85;
  const d = levenshtein(x, y);
  return 1 - d / Math.max(x.length, y.length);
}

function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  let prev = Array.from({ length: n + 1 }, (_, j) => j);
  for (let i = 1; i <= m; i++) {
    const cur = [i, ...new Array<number>(n).fill(0)];
    for (let j = 1; j <= n; j++) {
      cur[j] = Math.min(prev[j]! + 1, cur[j - 1]! + 1, prev[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[n]!;
}

// ---------------------------------------------------------------------------

function orderCrossTab(runs: FlowTrace[], stepIndex: number): OrderOutcome[] {
  const tab = new Map<string, { reached: number; missed: number }>();
  for (const t of runs) {
    const step = t.steps[stepIndex];
    if (!step || step.status === "not-run") continue;
    const order = completionOrder(step.requests).join(" → ") || "(no api traffic)";
    const cell = tab.get(order) ?? { reached: 0, missed: 0 };
    if (t.status === "pass") cell.reached++;
    else cell.missed++;
    tab.set(order, cell);
  }
  return [...tab.entries()]
    .map(([order, cell]) => ({ order, ...cell }))
    .sort((a, b) => b.reached + b.missed - (a.reached + a.missed));
}

export function renderExplanation(e: Explanation): string {
  const out: string[] = [];
  out.push("─".repeat(72));
  out.push("why this failed:");
  for (const line of e.failed) out.push(`  ${line}`);
  for (const line of e.whatHappened) out.push(`  ${line}`);
  if (e.reproducibility.length) {
    out.push(`  reproducibility: ${e.reproducibility[0]}`);
    for (const line of e.reproducibility.slice(1)) out.push(`  ${line}`);
  }
  if (e.meaning.length) {
    out.push("  what this means:");
    for (const line of e.meaning) out.push(`    ${line}`);
  }
  return out.join("\n");
}
