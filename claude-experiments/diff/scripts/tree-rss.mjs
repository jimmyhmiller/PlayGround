// Peak memory of a PROCESS TREE, sampled.
//
// Used by both cal.com harnesses (`demo/server.mjs`, `scripts/bench-calcom.mjs`) so
// the demo and the reference benchmark report the same axis the same way.
//
// Why this exists at all: `/usr/bin/time -l` (and `-v`) report `ru_maxrss`, which is
// the maximum over the direct child and whatever descendants that child actually
// WAITED for — never a sum, and blind to concurrent or unreaped processes entirely.
// Both sides of the cal.com comparison are process trees: diffpack's production build
// runs the client, react-server and ssr graphs concurrently and then spawns a node
// prerenderer, and `next build` spawns worker processes. So ru_maxrss under-reports
// both sides by whatever their process layout happens to be, not by the same factor,
// and it flatters the side that spreads work across more processes.
//
// `scripts/tree-rss.test.mjs` measures the gap on a tree built for the purpose: a
// parent holding 120 MiB with two 240 MiB children reports ~730 MiB sampled here and
// ~164 MiB from ru_maxrss. That is the size of the mistake, on this platform, today.
//
// So: sum RSS over the root and every descendant, every `intervalMs`, and keep the
// largest total. One `ps` call per sample (cheaper than walking `pgrep -P` per level).
// A spike that starts and ends between two samples is missed, which is why `count()`
// is reported next to the number instead of presenting it as an exact peak.
import { execFileSync } from "node:child_process";

export const RSS_SAMPLE_MS = 250;

export function sampleTreeRss(rootPid, intervalMs = RSS_SAMPLE_MS) {
  let peakKb = 0;
  let samples = 0;
  const tick = () => {
    const kb = treeRssKb(rootPid);
    if (kb > 0) {
      samples++;
      if (kb > peakKb) peakKb = kb;
    }
  };
  // No synchronous first sample: `ps -ax` is a fork/exec that blocks the caller's event
  // loop for tens of milliseconds. The demo spawns both sides back to back, so an
  // immediate sample would delay the second spawn by exactly that much, which is the
  // handicap the alternating spawn order exists to remove. RSS at t=0 is a process that
  // has not allocated anything yet, so nothing is lost by waiting one interval.
  const timer = setInterval(tick, intervalMs);
  return {
    /// Stop sampling and return the peak in MiB, or null if nothing was ever sampled
    /// (a process that exited inside the first interval).
    stop() {
      clearInterval(timer);
      return peakKb ? peakKb / 1024 : null;
    },
    count: () => samples,
  };
}

/// RSS of `rootPid` plus every descendant, in KiB (macOS and Linux `ps` both report
/// KiB). 0 means the tree was already gone, which callers treat as "no sample" rather
/// than as a peak of zero.
export function treeRssKb(rootPid) {
  let out = "";
  try {
    out = execFileSync("ps", ["-axo", "pid=,ppid=,rss="], { encoding: "utf8" });
  } catch {
    return 0;
  }
  const children = new Map();
  const rss = new Map();
  for (const line of out.split("\n")) {
    const m = /^\s*(\d+)\s+(\d+)\s+(\d+)\s*$/.exec(line);
    if (!m) continue;
    const pid = Number(m[1]);
    const ppid = Number(m[2]);
    rss.set(pid, Number(m[3]));
    if (!children.has(ppid)) children.set(ppid, []);
    children.get(ppid).push(pid);
  }
  let total = 0;
  const stack = [rootPid];
  const seen = new Set();
  while (stack.length) {
    const pid = stack.pop();
    if (seen.has(pid)) continue; // a pid cannot be its own ancestor, but never loop
    seen.add(pid);
    total += rss.get(pid) ?? 0;
    for (const kid of children.get(pid) ?? []) stack.push(kid);
  }
  return total;
}
