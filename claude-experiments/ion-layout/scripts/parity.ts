// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Parity oracle: run the ORIGINAL iongraph layout (generic-layout/layout.ts,
// the real viewer algorithm) and the Rust port on the same graphs and diff
// the resulting node geometry byte-exactly.
//
// Known deliberate deviations of the Rust port (auto-tagged, not failures):
//   D1 port compression/widening when a node is too narrow for its ports
//   D2 whole-drawing right-shift when anything would land left of the margin
//   D3 robustness fixes for degraded inputs (the original throws or hangs;
//      those cases show up as TSERR here)
//
// Usage: ./scripts/parity.sh

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { layoutGraph, DEFAULT_CONFIG, type InputNode } from "/Users/jimmyhmiller/Documents/Code/open-source/iongraph/generic-layout/layout.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ION_DUMP = join(ROOT, "target", "release", "ion-dump");
const IONGRAPH = "/Users/jimmyhmiller/Documents/Code/open-source/iongraph";

interface Case {
  name: string;
  nodes: InputNode[];
}

const mk = (
  id: number,
  w: number,
  h: number,
  succs: number[],
  opts: Partial<InputNode> = {},
): InputNode => ({
  id,
  size: { x: w, y: h },
  predecessors: [], // filled below
  successors: succs,
  loopDepth: 0,
  isLoopHeader: false,
  isBackedge: false,
  ...opts,
});

function fillPreds(nodes: InputNode[]): InputNode[] {
  const byId = new Map(nodes.map(n => [n.id, n]));
  for (const n of nodes) n.predecessors = [];
  for (const n of nodes) {
    for (const s of n.successors) byId.get(s)!.predecessors.push(n.id);
  }
  return nodes;
}

class Rng {
  state: number;
  constructor(seed: number) {
    this.state = seed;
  }
  next() {
    this.state = (this.state * 1664525 + 1013904223) & 0x7fffffff;
    return this.state / 0x7fffffff;
  }
  int(min: number, max: number) {
    return Math.floor(this.next() * (max - min + 1)) + min;
  }
}

const cases: Case[] = [];

cases.push({ name: "single", nodes: fillPreds([mk(0, 100, 40, [])]) });
cases.push({
  name: "chain5",
  nodes: fillPreds(Array.from({ length: 5 }, (_, i) => mk(i, 100, 40, i < 4 ? [i + 1] : []))),
});
cases.push({
  name: "diamond",
  nodes: fillPreds([mk(0, 100, 40, [1, 2]), mk(1, 100, 40, [3]), mk(2, 100, 40, [3]), mk(3, 100, 40, [])]),
});
cases.push({
  name: "skip",
  nodes: fillPreds([mk(0, 100, 40, [1, 3]), mk(1, 100, 40, [2]), mk(2, 100, 40, [3]), mk(3, 100, 40, [])]),
});
cases.push({
  name: "simple-loop",
  nodes: fillPreds([
    mk(0, 100, 40, [1]),
    mk(1, 100, 40, [2, 4], { loopDepth: 1, isLoopHeader: true }),
    mk(2, 100, 40, [3], { loopDepth: 1 }),
    mk(3, 80, 30, [1], { loopDepth: 1, isBackedge: true }),
    mk(4, 100, 40, []),
  ]),
});
cases.push({
  name: "nested-loop",
  nodes: fillPreds([
    mk(0, 100, 40, [1]),
    mk(1, 100, 40, [2, 7], { loopDepth: 1, isLoopHeader: true }),
    mk(2, 100, 40, [3, 5], { loopDepth: 2, isLoopHeader: true }),
    mk(3, 100, 40, [4], { loopDepth: 2 }),
    mk(4, 80, 30, [2], { loopDepth: 2, isBackedge: true }),
    mk(5, 100, 40, [6], { loopDepth: 1 }),
    mk(6, 80, 30, [1], { loopDepth: 1, isBackedge: true }),
    mk(7, 100, 40, []),
  ]),
});

// graph.json: the demo CFG shipped with iongraph (nested loops, bailouts).
{
  const g = JSON.parse(readFileSync(join(IONGRAPH, "graph.json"), "utf8"));
  const ids = new Map<string, number>(g.nodes.map((n: any, i: number) => [n.id, i]));
  const nodes: InputNode[] = g.nodes.map((n: any, i: number) => {
    const lines = [n.label, ...(n.instructions ?? [])];
    const titleLen = (lines[0].length + (n.loopHeader ? " (loop header)".length : n.backedge ? " (backedge)".length : 0)) * 7.5;
    const bodyLen = Math.max(0, ...lines.map((l: string) => l.length)) * 6.5;
    const w = Math.max(140, Math.max(titleLen, bodyLen) + 24);
    const h = 20 + lines.length * 14 + 8;
    return mk(i, w, h, (n.succs ?? []).map((s: string) => ids.get(s)!), {
      loopDepth: n.loopDepth ?? 0,
      isLoopHeader: !!n.loopHeader,
      isBackedge: !!n.backedge,
    });
  });
  cases.push({ name: "graph.json", nodes: fillPreds(nodes) });
}

// mega-complex.json: a real 9.7MB SpiderMonkey ion dump — every pass of
// every function becomes a parity case (~526 real compiler graphs).
{
  const MEGA = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/iongraph-rust/mega-complex.json";
  try {
    const g = JSON.parse(readFileSync(MEGA, "utf8"));
    for (const [fi, fn] of (g.functions ?? []).entries()) {
      for (const [pi, pass] of (fn.passes ?? []).entries()) {
        const blocks = pass.mir?.blocks ?? [];
        if (blocks.length < 2) continue;
        const present = new Set(blocks.map((b: any) => b.id));
        const nodes: InputNode[] = blocks.map((b: any) => {
          const lines = [`Block ${b.id}`, ...(b.instructions ?? []).map((ins: any) => `v${ins.id} = ${ins.opcode}`)];
          const w = Math.max(140, Math.max(...lines.map((l: string) => l.length)) * 6.5 + 24);
          const h = 20 + lines.length * 14 + 8;
          return mk(b.id, w, h, (b.successors ?? []).filter((s: number) => present.has(s)), {
            loopDepth: b.loopDepth ?? 0,
            isLoopHeader: (b.attributes ?? []).includes("loopheader"),
            isBackedge: (b.attributes ?? []).includes("backedge"),
          });
        });
        cases.push({ name: `mega:f${fi}p${pi}:${pass.name}`, nodes: fillPreds(nodes) });
      }
    }
  } catch (e: any) {
    console.log(`(skipping mega-complex.json: ${String(e.message).slice(0, 60)})`);
  }
}

// Random DAGs (no loops).
for (let seed = 0; seed < 60; seed++) {
  const rng = new Rng(seed);
  const n = 4 + (seed % 20);
  const nodes: InputNode[] = [];
  for (let i = 0; i < n; i++) nodes.push(mk(i, rng.int(160, 260), rng.int(30, 100), []));
  const hasPred = new Array(n).fill(false);
  for (let i = 0; i < n; i++) {
    const numSuccs = rng.int(0, Math.min(3, n - i - 1));
    const cands = Array.from({ length: n - i - 1 }, (_, j) => i + j + 1);
    for (let s = 0; s < numSuccs && cands.length; s++) {
      const idx = rng.int(0, cands.length - 1);
      const succ = cands.splice(idx, 1)[0];
      nodes[i].successors.push(succ);
      hasPred[succ] = true;
    }
  }
  for (let i = 1; i < n; i++) {
    if (!hasPred[i]) nodes[rng.int(0, i - 1)].successors.push(i);
  }
  cases.push({ name: `random-dag:${seed}`, nodes: fillPreds(nodes) });
}

// Random single-loop graphs with explicit metadata.
for (let seed = 0; seed < 40; seed++) {
  const rng = new Rng(1000 + seed);
  const bodySize = rng.int(1, 4);
  const nodes: InputNode[] = [];
  let id = 0;
  const entry = id++;
  nodes.push(mk(entry, rng.int(160, 240), rng.int(30, 80), []));
  const header = id++;
  nodes.push(mk(header, rng.int(160, 240), rng.int(30, 80), [], { loopDepth: 1, isLoopHeader: true }));
  nodes[entry].successors.push(header);
  let prev = header;
  for (let i = 0; i < bodySize; i++) {
    const b = id++;
    nodes.push(mk(b, rng.int(160, 240), rng.int(30, 80), [], { loopDepth: 1 }));
    nodes[prev].successors.push(b);
    prev = b;
  }
  const back = id++;
  nodes.push(mk(back, rng.int(40, 80), rng.int(20, 40), [header], { loopDepth: 1, isBackedge: true }));
  nodes[prev].successors.push(back);
  const exit = id++;
  nodes.push(mk(exit, rng.int(160, 240), rng.int(30, 80), []));
  nodes[header].successors.push(exit);
  cases.push({ name: `random-loop:${seed}`, nodes: fillPreds(nodes) });
}

/** The Rust port widens/compresses ports for nodes too narrow to hold them. */
function hasPortDeviation(nodes: InputNode[]): boolean {
  return nodes.some(n => {
    const ports = n.successors.length;
    return ports > 1 && n.size.x < 2 * DEFAULT_CONFIG.portStart + DEFAULT_CONFIG.portSpacing * (ports - 1);
  });
}

let pass = 0;
let deviations = 0;
let fail = 0;
let tsErrors = 0;

for (const c of cases) {
  let tsResult;
  try {
    tsResult = layoutGraph(c.nodes);
  } catch (e: any) {
    tsErrors++;
    console.log(`TSERR ${c.name}: original layout.ts failed (${String(e.message).slice(0, 60)})`);
    continue;
  }

  // D2 detection: the port shifts everything right when ANY layout node
  // (including dummies) would sit left of the padding.
  let minX = Infinity;
  for (const layer of tsResult.layoutNodesByLayer) for (const n of layer) minX = Math.min(minX, n.pos.x);
  const shiftDeviation = minX < DEFAULT_CONFIG.contentPadding - 1e-9;

  const lines: string[] = [];
  const index = new Map(c.nodes.map((n, i) => [n.id, i]));
  for (const n of c.nodes) {
    lines.push(`node ${n.size.x} ${n.size.y} ${n.loopDepth} ${n.isLoopHeader ? 1 : 0} ${n.isBackedge ? 1 : 0}`);
  }
  for (const n of c.nodes) {
    for (const s of n.successors) lines.push(`edge ${index.get(n.id)} ${index.get(s)}`);
  }
  const out = execFileSync(ION_DUMP, [], { input: lines.join("\n") + "\n" }).toString();
  const rsCells = new Map<number, { l: number; t: number; w: number; h: number; layer: number }>();
  for (const line of out.trim().split("\n")) {
    const p = line.split(" ");
    if (p[0] === "cell") {
      rsCells.set(Number(p[1]), { l: Number(p[2]), t: Number(p[3]), w: Number(p[4]), h: Number(p[5]), layer: Number(p[6]) });
    }
  }

  const expectedDeviation = hasPortDeviation(c.nodes) || shiftDeviation;
  const diffs: string[] = [];
  for (const n of tsResult.nodes) {
    const rs = rsCells.get(index.get(n.id)!);
    if (!rs) {
      diffs.push(`${n.id}: missing in rust output`);
      continue;
    }
    const eps = 1e-6;
    if (
      Math.abs(n.pos.x - rs.l) > eps ||
      Math.abs(n.pos.y - rs.t) > eps ||
      Math.abs(n.size.x - rs.w) > eps ||
      Math.abs(n.size.y - rs.h) > eps ||
      n.layer !== rs.layer
    ) {
      diffs.push(
        `${n.id}: ts(${n.pos.x.toFixed(1)},${n.pos.y.toFixed(1)} ${n.size.x}x${n.size.y} L${n.layer}) vs rs(${rs.l.toFixed(1)},${rs.t.toFixed(1)} ${rs.w}x${rs.h} L${rs.layer})`
      );
    }
  }

  if (diffs.length === 0) {
    pass++;
  } else if (expectedDeviation) {
    deviations++;
    console.log(`DEV  ${c.name} (${diffs.length} diffs, expected: ${hasPortDeviation(c.nodes) ? "ports" : ""}${shiftDeviation ? " shift" : ""})`);
  } else {
    fail++;
    console.log(`FAIL ${c.name} (${diffs.length} diffs)`);
    for (const d of diffs.slice(0, 6)) console.log(`     ${d}`);
  }
}

console.log(`\n${pass} exact, ${deviations} expected deviations, ${tsErrors} original-side errors, ${fail} FAILURES out of ${cases.length} cases`);
process.exit(fail > 0 ? 1 : 0);
