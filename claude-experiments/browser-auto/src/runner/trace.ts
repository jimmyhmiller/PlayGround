import type { Effect, Step, Target } from "../dsl/ir.js";
import { formatEffect } from "../dsl/ir.js";
import type { Explanation } from "./explain.js";
import { renderExplanation } from "./explain.js";
import type { ObservedRequest } from "./settle.js";

export interface EffectVerdict {
  effect: Effect;
  rendered: string;
  pass: boolean;
  /** what we observed instead, when it failed */
  observed?: string;
}

export interface ConsoleEntry {
  kind: "console-error" | "pageerror";
  text: string;
}

export interface StepTrace {
  index: number;
  line: number;
  source: string;
  status: "pass" | "fail" | "not-run";
  preUrl: string;
  postUrl?: string;
  requests: ObservedRequest[];
  consoleErrors: ConsoleEntry[];
  navigations: string[];
  effects: EffectVerdict[];
  captures: Record<string, string>;
  settle?: {
    settled: boolean;
    iterations: number;
    stuck: string[];
    clockAdvanced: number;
    notes: string[];
  };
  /** populated on failure */
  failure?: string;
  ariaSnapshot?: string;
  /** the unresolved action target, when the action itself failed */
  failedTarget?: Target;
  /** data-testids present on the page at failure time */
  testids?: string[];
  durationMs: number;
}

export interface FlowTrace {
  flow: string;
  file: string;
  startedAt: string;
  worldFingerprint: string | null;
  worldVerification: { level: number; proven: string[]; asserted: string[] } | null;
  /** active simulated-conditions profile, if any */
  conditions: { latencyMs?: [number, number]; failRate?: number; seed: number } | null;
  status: "pass" | "fail";
  steps: StepTrace[];
  /** automatic causal explanation of the failure (no verdicts — evidence) */
  explanation?: Explanation;
}

export interface Checkpoint {
  step: number;
  url: string;
  storageState: unknown;
  worldFingerprint: string | null;
  worldSnapshotId: string | null;
}

export function newStepTrace(index: number, step: Step, preUrl: string): StepTrace {
  return {
    index,
    line: step.line,
    source: step.source,
    status: "not-run",
    preUrl,
    requests: [],
    consoleErrors: [],
    navigations: [],
    effects: [],
    captures: {},
    durationMs: 0,
  };
}

/** Render the human/agent-readable story of a failed flow. */
export function renderReport(trace: FlowTrace): string {
  const out: string[] = [];
  out.push(`flow "${trace.flow}" — ${trace.status.toUpperCase()}`);
  out.push(`file: ${trace.file}`);
  if (trace.conditions) {
    const c = trace.conditions;
    const parts: string[] = [];
    if (c.latencyMs) parts.push(`latency +${c.latencyMs[0]}–${c.latencyMs[1]}ms`);
    if (c.failRate) parts.push(`${Math.round(c.failRate * 100)}% request failures`);
    parts.push(`seed ${c.seed}`);
    out.push(`conditions: SIMULATED BAD CONDITIONS ACTIVE — ${parts.join(", ")} (rerun with the same seed to reproduce)`);
  }
  if (trace.worldFingerprint) out.push(`world: ${trace.worldFingerprint}`);
  if (trace.worldVerification) {
    out.push(`world verification: L${trace.worldVerification.level}`);
    for (const p of trace.worldVerification.proven) out.push(`  proven:   ${p}`);
    for (const a of trace.worldVerification.asserted) out.push(`  asserted: ${a} (unverified — see 'bat doctor')`);
  }
  out.push("");

  for (const s of trace.steps) {
    const mark = s.status === "pass" ? "✓" : s.status === "fail" ? "✗" : "·";
    out.push(`${mark} step ${s.index + 1} (line ${s.line}): ${s.source}`);
    if (s.status === "not-run") continue;
    if (s.settle?.notes.length) {
      for (const n of s.settle.notes) out.push(`  note: ${n}`);
    }

    if (s.status === "fail") {
      out.push("");
      if (s.failure) out.push(indent(s.failure, 2));
      for (const e of s.effects) {
        const em = e.pass ? "✓" : "✗";
        out.push(`  ${em} ${e.rendered}`);
        if (!e.pass && e.observed) out.push(`      observed: ${e.observed}`);
      }
      if (s.settle && !s.settle.settled) {
        out.push(`  the step never settled (${s.settle.iterations} convergence passes). Still outstanding:`);
        for (const p of s.settle.stuck) out.push(`    - ${p}`);
      }
      if (s.consoleErrors.length) {
        out.push(`  errors from the page during this step:`);
        for (const c of s.consoleErrors) out.push(`    [${c.kind}] ${c.text.slice(0, 300)}`);
      }
      if (s.requests.length) {
        out.push(`  network during this step (in start order):`);
        for (const r of s.requests.slice(0, 20)) {
          const outcome = r.failure ? `FAILED (${r.failure})` : (r.status ?? "pending");
          const finished = r.finishSeq !== null ? ` (finished #${r.finishSeq})` : "";
          const injected = r.injected ? ` [${r.injected}]` : "";
          out.push(`    ${r.method} ${r.url} -> ${outcome}${finished}${injected}`);
        }
        const order = completionOrder(s.requests);
        if (order.length > 1) out.push(`  response completion order: ${order.join(" → ")}`);
      }
      out.push(`  url: ${s.preUrl}${s.postUrl && s.postUrl !== s.preUrl ? ` -> ${s.postUrl}` : ""}`);
      if (s.ariaSnapshot && !(s.failure && s.failure.includes("semantic tree"))) {
        out.push(`  the page's semantic tree at failure:`);
        out.push(indent(s.ariaSnapshot, 4));
      }
      out.push("");
      out.push(`  replay just this step: bat replay ${trace.file}:${s.index + 1}`);
    }
  }
  if (trace.explanation) {
    out.push("");
    out.push(renderExplanation(trace.explanation));
  }
  return out.join("\n");
}

export function renderEffect(e: Effect): string {
  return formatEffect(e);
}

/** "METHOD /path" list in the order responses completed (api traffic only). */
export function completionOrder(requests: ObservedRequest[]): string[] {
  return requests
    .filter((r) => r.finishSeq !== null && r.resourceType !== "document")
    .sort((a, b) => a.finishSeq! - b.finishSeq!)
    .map((r) => `${r.method} ${new URL(r.url).pathname}${r.failure ? " (failed)" : ""}`);
}

function indent(s: string, n: number): string {
  const pad = " ".repeat(n);
  return s
    .split("\n")
    .map((l) => pad + l)
    .join("\n");
}
