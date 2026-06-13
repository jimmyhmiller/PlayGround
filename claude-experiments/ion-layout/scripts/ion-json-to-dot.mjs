#!/usr/bin/env node
import { readFileSync, writeFileSync } from "node:fs";

const [jsonFile, outFile, functionIndexArg, passIndexArg] = process.argv.slice(2);

if (!jsonFile || !outFile) {
  console.error("Usage: ion-json-to-dot.mjs <ion-json> <out.dot> [function-index] [pass-index]");
  process.exit(1);
}

const data = JSON.parse(readFileSync(jsonFile, "utf8"));

function passScore(fn, pass) {
  const blocks = pass?.mir?.blocks ?? [];
  const edges = blocks.reduce((sum, block) => sum + (block.successors?.length ?? 0), 0);
  const instructions = blocks.reduce((sum, block) => sum + (block.instructions?.length ?? 0), 0);
  return { blocks: blocks.length, edges, instructions };
}

let functionIndex = functionIndexArg == null ? null : Number(functionIndexArg);
let passIndex = passIndexArg == null ? null : Number(passIndexArg);

if (functionIndex == null || Number.isNaN(functionIndex) || passIndex == null || Number.isNaN(passIndex)) {
  let best = null;
  for (const [fi, fn] of (data.functions ?? []).entries()) {
    for (const [pi, pass] of (fn.passes ?? []).entries()) {
      const score = passScore(fn, pass);
      const key = [score.blocks, score.edges, score.instructions];
      if (!best || compareKey(key, best.key) > 0) best = { fi, pi, fn, pass, score, key };
    }
  }
  if (!best) throw new Error("No functions/passes found");
  functionIndex = best.fi;
  passIndex = best.pi;
}

const fn = data.functions[functionIndex];
const pass = fn?.passes?.[passIndex];
if (!fn || !pass) throw new Error(`Function/pass not found: ${functionIndex}/${passIndex}`);

const blocks = pass.mir?.blocks ?? [];
const blockIds = new Set(blocks.map(block => block.id));
const out = [];

out.push("digraph IonMega {");
out.push(`  graph [rankdir=TB, label=${q(`${fn.name ?? `function ${functionIndex}`} / ${pass.name ?? `pass ${passIndex}`} (${blocks.length} blocks)`)}, labelloc=t]`);
out.push("  node [shape=box, style=rounded, fontname=\"Menlo\", fontsize=13, margin=\"0.18,0.12\"]");
out.push("  edge [fontname=\"Menlo\", fontsize=10]");
out.push("");

for (const block of blocks) {
  const attrs = block.attributes?.length ? ` [${block.attributes.join(", ")}]` : "";
  const depth = block.loopDepth ? ` loopDepth=${block.loopDepth}` : "";
  const lines = [`Block ${block.id}${attrs}${depth}`, ""];
  for (const ins of block.instructions ?? []) {
    lines.push(`v${ins.id} = ${ins.opcode}`);
  }
  const dotAttrs = [
    `label=${leftLabel(lines)}`,
    `ion_loop_depth=${Number(block.loopDepth ?? 0)}`,
  ];
  if (block.attributes?.includes("loopheader")) dotAttrs.push("ion_loop_header=true");
  if (block.attributes?.includes("backedge")) dotAttrs.push("ion_backedge=true");
  out.push(`  B${block.id} [${dotAttrs.join(", ")}]`);
}

out.push("");
for (const block of blocks) {
  for (const succ of block.successors ?? []) {
    if (!blockIds.has(succ)) continue;
    out.push(`  B${block.id} -> B${succ}`);
  }
}

out.push("}");
out.push("");

writeFileSync(outFile, out.join("\n"));

const score = passScore(fn, pass);
console.error(`Wrote ${outFile}`);
console.error(`Selected function ${functionIndex} (${fn.name ?? "<unnamed>"}), pass ${passIndex} (${pass.name ?? "<pass>"})`);
console.error(`${score.blocks} blocks, ${score.edges} edges, ${score.instructions} instructions`);

function compareKey(a, b) {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return 0;
}

function leftLabel(lines) {
  return `"${lines.map(escapeLabelLine).join("\\l")}\\l"`;
}

function q(value) {
  return `"${String(value).replace(/\\/g, "\\\\").replace(/"/g, "\\\"").replace(/\n/g, "\\n")}"`;
}

function escapeLabelLine(value) {
  return String(value).replace(/\\/g, "\\\\").replace(/"/g, "\\\"");
}
