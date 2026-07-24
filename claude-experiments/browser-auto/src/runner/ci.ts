import { appendFile } from "node:fs/promises";
import type { FlowTrace } from "./trace.js";
import { renderReport } from "./trace.js";

/** CI-facing outputs: JUnit XML, GitHub Actions annotations + job summary. */

export interface CiRun {
  trace: FlowTrace;
  runDir: string;
}

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

export function renderJUnit(runs: CiRun[]): string {
  const cases = runs.map(({ trace, runDir }) => {
    const time = (trace.steps.reduce((n, s) => n + s.durationMs, 0) / 1000).toFixed(3);
    if (trace.status === "pass") {
      return `    <testcase name="${esc(trace.flow)}" classname="${esc(trace.file)}" time="${time}"/>`;
    }
    const failed = trace.steps.find((s) => s.status === "fail");
    const headline = failed
      ? `step ${failed.index + 1} '${failed.source}': ${failed.effects.filter((e) => !e.pass).map((e) => e.rendered).join("; ") || failed.failure?.split("\n")[0] || "failed"}`
      : "failed";
    return [
      `    <testcase name="${esc(trace.flow)}" classname="${esc(trace.file)}" time="${time}">`,
      `      <failure message="${esc(headline)}"><![CDATA[${renderReport(trace).replace(/\]\]>/g, "]]]]><![CDATA[>")}`,
      "",
      `artifacts: ${runDir}]]></failure>`,
      `    </testcase>`,
    ].join("\n");
  });
  const failures = runs.filter((r) => r.trace.status === "fail").length;
  return [
    `<?xml version="1.0" encoding="UTF-8"?>`,
    `<testsuites name="bat" tests="${runs.length}" failures="${failures}">`,
    `  <testsuite name="bat" tests="${runs.length}" failures="${failures}">`,
    ...cases,
    `  </testsuite>`,
    `</testsuites>`,
    ``,
  ].join("\n");
}

/** `::error file=…` annotations — GitHub shows them inline on the flow file. */
export function githubAnnotations(runs: CiRun[]): string[] {
  const out: string[] = [];
  for (const { trace } of runs) {
    if (trace.status !== "fail") continue;
    const failed = trace.steps.find((s) => s.status === "fail");
    if (!failed) continue;
    const what =
      failed.effects.filter((e) => !e.pass).map((e) => e.rendered).join("; ") ||
      failed.failure?.split("\n")[0] ||
      "failed";
    const hint = trace.explanation?.reproducibility[0] ?? "";
    out.push(
      `::error file=${trace.file},line=${failed.line},title=bat: ${trace.flow}::${what}${hint ? ` — ${hint}` : ""}`.replace(/\n/g, " "),
    );
  }
  return out;
}

/** Markdown for $GITHUB_STEP_SUMMARY — the at-a-glance CI report. */
export function githubSummary(runs: CiRun[]): string {
  const out: string[] = [];
  const failures = runs.filter((r) => r.trace.status === "fail");
  out.push(`## bat — ${runs.length - failures.length}/${runs.length} flows passed`);
  out.push("");
  out.push("| flow | result | steps | world |");
  out.push("|---|---|---|---|");
  for (const { trace } of runs) {
    const steps = `${trace.steps.filter((s) => s.status === "pass").length}/${trace.steps.length}`;
    out.push(`| ${trace.flow} | ${trace.status === "pass" ? "✅" : "❌"} | ${steps} | \`${trace.worldFingerprint ?? "—"}\` |`);
  }
  for (const { trace, runDir } of failures) {
    out.push("");
    out.push(`<details><summary>❌ ${trace.flow} — why this failed</summary>`);
    out.push("");
    out.push("```");
    out.push(renderReport(trace));
    out.push("```");
    out.push(`artifacts (trace, report, screenshots, rerun traces): \`${runDir}\``);
    out.push("</details>");
  }
  out.push("");
  return out.join("\n");
}

export async function emitCi(runs: CiRun[]): Promise<void> {
  if (process.env.GITHUB_ACTIONS) {
    for (const line of githubAnnotations(runs)) console.log(line);
  }
  if (process.env.GITHUB_STEP_SUMMARY) {
    await appendFile(process.env.GITHUB_STEP_SUMMARY, githubSummary(runs), "utf8");
  }
}
