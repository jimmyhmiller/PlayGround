#!/usr/bin/env node
// Verify rendered layouts at the Graphviz level: run `dot -Kion -Tjson` on
// every corpus file and machine-check the geometry Graphviz will render.
//
// Checks:
//   C1  no two node boxes overlap
//   C2  every node's label text fits inside its node box (horizontally,
//       with a small tolerance; vertical is checked loosely vs font size)
//   C3  no edge spline passes through the interior of a node that is not
//       one of its endpoints
//   C4  every node box and spline point is inside the graph bounding box
//   C5  every edge has a drawable spline and an arrowhead
//
// Usage: node scripts/verify.mjs [file.dot ...]   (default: corpus/*.dot)

import { execFileSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const GVBINDIR = join(ROOT, "target", "graphviz");

const args = process.argv.slice(2);
const files = args.length
  ? args
  : readdirSync(join(ROOT, "corpus"))
      .filter(f => f.endsWith(".dot"))
      .sort()
      .map(f => join(ROOT, "corpus", f));

let totalViolations = 0;
let failedFiles = 0;

for (const file of files) {
  const violations = verifyFile(file);
  const name = file.replace(`${ROOT}/`, "");
  if (violations.length === 0) {
    console.log(`\x1b[32mPASS\x1b[0m ${name}`);
  } else {
    failedFiles++;
    totalViolations += violations.length;
    console.log(`\x1b[31mFAIL\x1b[0m ${name} (${violations.length} violation${violations.length > 1 ? "s" : ""})`);
    for (const v of violations.slice(0, 8)) console.log(`     - ${v}`);
    if (violations.length > 8) console.log(`     … and ${violations.length - 8} more`);
  }
}

console.log(`\n${files.length - failedFiles}/${files.length} files clean, ${totalViolations} total violations`);
process.exit(failedFiles > 0 ? 1 : 0);

function verifyFile(file) {
  let json;
  try {
    const out = execFileSync("dot", ["-Kion", "-Tjson", file], {
      env: { ...process.env, GVBINDIR },
      maxBuffer: 256 * 1024 * 1024,
    });
    json = JSON.parse(out.toString());
  } catch (e) {
    return [`dot -Kion failed: ${String(e.message).split("\n")[0]}`];
  }

  const violations = [];
  const objects = json.objects ?? [];
  const edges = json.edges ?? [];

  // Node boxes (pos = center in points, width/height in inches).
  const nodes = new Map(); // _gvid -> {name, l, r, t, b}
  for (const o of objects) {
    if (!o.pos || o.width == null) continue; // subgraphs/clusters
    const [cx, cy] = o.pos.split(",").map(Number);
    const w = Number(o.width) * 72;
    const h = Number(o.height) * 72;
    nodes.set(o._gvid, { name: o.name, l: cx - w / 2, r: cx + w / 2, b: cy - h / 2, t: cy + h / 2, obj: o });
  }

  // C6: the polygon Graphviz will actually DRAW must match the node's
  // declared geometry (a stale shape polygon renders detached from ports).
  for (const n of nodes.values()) {
    const draw = (n.obj._draw_ ?? []).filter(op => (op.op === "p" || op.op === "P") && Array.isArray(op.points));
    for (const op of draw) {
      const xs = op.points.map(p => p[0]);
      const ys = op.points.map(p => p[1]);
      const pw = Math.max(...xs) - Math.min(...xs);
      const ph = Math.max(...ys) - Math.min(...ys);
      const w = n.r - n.l;
      const h = n.t - n.b;
      if (Math.abs(pw - w) > 2 || Math.abs(ph - h) > 2) {
        violations.push(
          `C6 node ${n.name}: drawn polygon ${pw.toFixed(0)}x${ph.toFixed(0)} != declared size ${w.toFixed(0)}x${h.toFixed(0)}`
        );
      }
    }
  }

  // C1: node-node overlap
  const list = [...nodes.values()];
  for (let i = 0; i < list.length; i++) {
    for (let j = i + 1; j < list.length; j++) {
      const a = list[i];
      const b = list[j];
      const ox = Math.min(a.r, b.r) - Math.max(a.l, b.l);
      const oy = Math.min(a.t, b.t) - Math.max(a.b, b.b);
      if (ox > 0.5 && oy > 0.5) {
        violations.push(`C1 nodes ${a.name} and ${b.name} overlap by ${ox.toFixed(1)}x${oy.toFixed(1)}pt`);
      }
    }
  }

  // C2: label text inside node box
  for (const n of nodes.values()) {
    let fontSize = 14;
    for (const op of n.obj._ldraw_ ?? []) {
      if (op.op === "F") fontSize = op.size ?? fontSize;
      if (op.op !== "T" && op.op !== "t") continue;
      const [x, y] = op.pt;
      const w = op.width ?? 0;
      let left = x;
      let right = x + w;
      if (op.align === "c") {
        left = x - w / 2;
        right = x + w / 2;
      } else if (op.align === "r") {
        left = x - w;
        right = x;
      }
      const tol = 2.0;
      if (left < n.l - tol || right > n.r + tol) {
        violations.push(
          `C2 label of ${n.name} overflows horizontally: text [${left.toFixed(1)}..${right.toFixed(1)}] box [${n.l.toFixed(1)}..${n.r.toFixed(1)}] ("${(op.text ?? "").slice(0, 30)}")`
        );
      }
      if (y - fontSize > n.t + tol || y + fontSize < n.b - tol) {
        violations.push(`C2 label of ${n.name} outside vertically: baseline ${y.toFixed(1)} box [${n.b.toFixed(1)}..${n.t.toFixed(1)}]`);
      }
    }
  }

  // bb
  let bb = null;
  if (json.bb) {
    const [llx, lly, urx, ury] = json.bb.split(",").map(Number);
    bb = { llx, lly, urx, ury };
  }
  // C4 for nodes
  if (bb) {
    for (const n of nodes.values()) {
      if (n.l < bb.llx - 1 || n.r > bb.urx + 1 || n.b < bb.lly - 1 || n.t > bb.ury + 1) {
        violations.push(`C4 node ${n.name} outside graph bb`);
      }
    }
  }

  // C3 + C4 + C5 for edges
  for (const e of edges) {
    const draws = (e._draw_ ?? []).filter(op => op.op === "b" || op.op === "B");
    if (draws.length === 0) {
      violations.push(`C5 edge ${nodes.get(e.tail)?.name}->${nodes.get(e.head)?.name} has no spline`);
      continue;
    }
    const wantsArrow = json.directed && e.arrowhead !== "none" && e.dir !== "none" && e.dir !== "back";
    if (wantsArrow && !(e._hdraw_ ?? []).length) {
      violations.push(`C5 edge ${nodes.get(e.tail)?.name}->${nodes.get(e.head)?.name} has no arrowhead`);
    }
    for (const d of draws) {
      const pts = d.points;
      if ((pts.length - 1) % 3 !== 0) {
        violations.push(`C5 edge ${nodes.get(e.tail)?.name}->${nodes.get(e.head)?.name} spline has ${pts.length} points (not 3k+1)`);
        continue;
      }
      const samples = [];
      for (let i = 0; i + 3 < pts.length; i += 3) {
        for (let s = 0; s <= 12; s++) {
          const t = s / 12;
          const u = 1 - t;
          samples.push([
            u * u * u * pts[i][0] + 3 * u * u * t * pts[i + 1][0] + 3 * u * t * t * pts[i + 2][0] + t * t * t * pts[i + 3][0],
            u * u * u * pts[i][1] + 3 * u * u * t * pts[i + 1][1] + 3 * u * t * t * pts[i + 2][1] + t * t * t * pts[i + 3][1],
          ]);
        }
      }
      const reported = new Set();
      for (const [gvid, n] of nodes) {
        if (gvid === e.tail || gvid === e.head || reported.has(gvid)) continue;
        const shrink = 2;
        for (const [px, py] of samples) {
          if (px > n.l + shrink && px < n.r - shrink && py > n.b + shrink && py < n.t - shrink) {
            violations.push(
              `C3 edge ${nodes.get(e.tail)?.name}->${nodes.get(e.head)?.name} passes through node ${n.name} at (${px.toFixed(1)},${py.toFixed(1)})`
            );
            reported.add(gvid);
            break;
          }
        }
      }
      if (bb) {
        for (const [px, py] of samples) {
          if (px < bb.llx - 1 || px > bb.urx + 1 || py < bb.lly - 1 || py > bb.ury + 1) {
            violations.push(`C4 edge ${nodes.get(e.tail)?.name}->${nodes.get(e.head)?.name} spline outside bb at (${px.toFixed(1)},${py.toFixed(1)})`);
            break;
          }
        }
      }
    }
  }

  return violations;
}
