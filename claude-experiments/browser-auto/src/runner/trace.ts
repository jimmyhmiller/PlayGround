import type { Effect, Step } from "../dsl/ir.js";
import { formatEffect } from "../dsl/ir.js";
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
  durationMs: number;
}

export interface FlowTrace {
  flow: string;
  file: string;
  startedAt: string;
  worldFingerprint: string | null;
  worldVerification: { level: number; proven: string[]; asserted: string[] } | null;
  status: "pass" | "fail";
  steps: StepTrace[];
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
        out.push(`  network during this step:`);
        for (const r of s.requests.slice(0, 20)) {
          out.push(`    ${r.method} ${r.url} -> ${r.failure ? `FAILED (${r.failure})` : (r.status ?? "pending")}`);
        }
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
  return out.join("\n");
}

export function renderEffect(e: Effect): string {
  return formatEffect(e);
}

function indent(s: string, n: number): string {
  const pad = " ".repeat(n);
  return s
    .split("\n")
    .map((l) => pad + l)
    .join("\n");
}
