// The claim `scripts/tree-rss.mjs` rests on, tested against a real process tree.
//
//   node scripts/tree-rss.test.mjs
//
// The cal.com harnesses stopped reporting `/usr/bin/time -l`'s `maximum resident set
// size` as peak build memory because `ru_maxrss` is the largest SINGLE process in a
// tree, not the tree's footprint. That is the whole argument for sampling, so it is
// worth being sure of rather than asserting in a comment: this spawns a parent that
// holds ~120 MiB and two children that hold ~240 MiB each, under `/usr/bin/time -l`,
// and checks that the sampler sees the sum while ru_maxrss sees roughly one child.
import { spawn } from "node:child_process";
import { sampleTreeRss, treeRssKb } from "./tree-rss.mjs";

const PARENT_MB = 120;
const CHILD_MB = 240;
const CHILDREN = 2;
const TREE_MB = PARENT_MB + CHILD_MB * CHILDREN;

let failures = 0;
function check(what, cond, detail) {
  if (cond) console.log(`  ok   ${what}`);
  else {
    failures++;
    console.log(`  FAIL ${what}${detail ? `\n       ${detail}` : ""}`);
  }
}

// Hold N MiB resident (a filled Buffer is real, untouchable-by-the-GC RSS) and stay
// alive long enough to be sampled a few times. `--eval` keeps this to one file.
function holder(mb, children) {
  return [
    "-e",
    `const b = Buffer.alloc(${mb} * 1024 * 1024, 7);` +
      `let kids = [];` +
      `for (let i = 0; i < ${children}; i++) {` +
      `  kids.push(require("node:child_process").spawn(process.execPath, ["-e",` +
      `    "const b = Buffer.alloc(${CHILD_MB} * 1024 * 1024, 9); setTimeout(() => { if (b[0]) process.exit(0); }, 4000);"` +
      `  ], { stdio: "ignore" }));` +
      `}` +
      `setTimeout(() => { if (b[0]) process.exit(0); }, 4000);`,
  ];
}

const proc = spawn("/usr/bin/time", ["-l", process.execPath, ...holder(PARENT_MB, CHILDREN)], {
  stdio: ["ignore", "ignore", "pipe"],
});
let stderr = "";
proc.stderr.on("data", (d) => (stderr += d));

const sampler = sampleTreeRss(proc.pid, 100);
const code = await new Promise((resolve) => proc.on("close", resolve));
const treePeakMb = sampler.stop();
const single = /([0-9]+)\s+maximum resident set size/.exec(stderr);
const singleMb = single ? Number(single[1]) / (1024 * 1024) : null;

console.log(`tree-rss: parent ${PARENT_MB} MiB + ${CHILDREN} x ${CHILD_MB} MiB = ${TREE_MB} MiB expected`);
console.log(`  sampled tree peak   ${treePeakMb?.toFixed(0)} MiB over ${sampler.count()} samples`);
console.log(`  ru_maxrss (time -l) ${singleMb?.toFixed(0)} MiB`);

check("the tree exited cleanly", code === 0, `exit ${code}\n${stderr.slice(0, 400)}`);
check("the sampler took several samples", sampler.count() >= 3, `${sampler.count()} samples`);
// Generous floor: RSS is pages actually resident, and the runtimes' own footprint is on
// top, so this asserts the shape of the answer (the tree, not one process) rather than
// an exact figure.
check(
  "the sampled peak covers the WHOLE tree, not one process",
  treePeakMb !== null && treePeakMb > TREE_MB * 0.8,
  `${treePeakMb?.toFixed(0)} MiB, expected > ${(TREE_MB * 0.8).toFixed(0)}`,
);
check(
  "ru_maxrss reports roughly one process, which is why it is not the headline",
  singleMb !== null && singleMb < TREE_MB * 0.75,
  `ru_maxrss ${singleMb?.toFixed(0)} MiB vs tree ${TREE_MB} MiB — if this fails, /usr/bin/time changed semantics`,
);
check(
  "so the sampled peak is materially larger than ru_maxrss",
  treePeakMb !== null && singleMb !== null && treePeakMb > singleMb * 1.5,
  `${treePeakMb?.toFixed(0)} MiB vs ${singleMb?.toFixed(0)} MiB`,
);
check("a dead pid samples as no data rather than as a zero peak", treeRssKb(999999) === 0);

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
