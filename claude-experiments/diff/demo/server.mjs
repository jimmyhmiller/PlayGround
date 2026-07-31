// The side-by-side demo: one cal.com checkout, two dev servers, one dashboard.
//
//   node demo/server.mjs                       # then open http://localhost:4321
//   node demo/server.mjs --app /tmp/dpe2e/calcom --dp-port 3000 --tp-port 3001
//   node demo/server.mjs --no-boot             # attach to dev servers you started yourself
//
// This process owns three things:
//
//   1. Both dev servers. `diffpack dev` on --dp-port and `next dev --turbopack` on
//      --tp-port, over the SAME source tree, each with its own self-consistent
//      NEXT_PUBLIC_WEBAPP_URL / NEXTAUTH_URL so neither side's browser code calls
//      the other side's API.
//   2. The scenario edits. Every scenario is one write to one real cal.com source
//      file. One write, both watchers: the two sides are reacting to the identical
//      event, which is what makes the two clocks comparable.
//   3. The dashboard at `/`, plus an SSE stream of both servers' logs and status.
//
// Timing lives in the browser, not here: the dashboard starts one clock per side
// when it sends the edit request and stops each side's clock when that side's frame
// reports the new token (see demo/probe.js). The number on screen is therefore
// edit -> visibly updated page, in a real visible browser frame.
//
// Honest caveats, stated in the UI too:
//   * Both dev servers run at once, so they contend for CPU. That is the price of a
//     live side-by-side; the isolated, interleaved, multi-sample numbers come from
//     `node scripts/bench-calcom.mjs`.
//   * The probe polls every 8 ms, so each reading carries up to 8 ms of quantisation
//     — identical on both sides.
//   * Which side is SPAWNED FIRST alternates every race (`racingOrder`), because the
//     first process spawned gets a moment of an uncontended machine.
//   * `next build` runs with this checkout's `typescript.ignoreBuildErrors` and
//     `eslint.ignoreDuringBuilds` set (see the benchmark `next.config.ts` wrapper):
//     diffpack does no type checking, so leaving `tsc` inside `next build` would time
//     a compiler only one side runs. Both sides therefore skip it.
//   * Peak memory is SAMPLED over each side's whole process tree, not read from
//     `/usr/bin/time -l`, whose `ru_maxrss` is the largest single process and would
//     under-report both trees by different amounts (see `sampleTreeRss`).
//   * A build that exits non-zero is reported as FAILED, never as a timeout: a crash
//     on one side must not read as a win for the other.
//   * Navigation and editing are separate phases: route chips only replace iframe
//     documents; scenario buttons navigate if needed, settle, then edit exactly once.
//
// Every file this touches is snapshotted at startup and restored on exit, including
// SIGINT/SIGTERM: the injected probe tag, the demo-only X-Frame-Options strip, and
// the four scenario source files.

import { spawn, execFileSync } from "node:child_process";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
  createWriteStream,
} from "node:fs";
import { createServer } from "node:http";
import { sampleTreeRss } from "../scripts/tree-rss.mjs";
import { racingOrder } from "./racing-order.mjs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const demoDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = dirname(demoDir);

const args = parseArgs(process.argv.slice(2));
const corpusApp = join(repoRoot, "integration", "e2e", ".cache", "calcom");
const APP = args.app ?? (existsSync(corpusApp) ? corpusApp : "/tmp/dpe2e/calcom");
const WEB = join(APP, "apps", "web");
const DP_PORT = args.dpPort ?? 3000;
const TP_PORT = args.tpPort ?? 3001;
const PORT = args.port ?? 4321;
const BOOT = !args.noBoot;
const BOOT_TIMEOUT_MS = args.bootTimeout ?? 240000;
const READY_PATH = args.readyPath ?? "/auth/login";
// A 200 is not proof the page rendered. A dev server that answers an error shell, or
// that streams a document it never finishes, would be called ready early and would be
// credited with a boot time it did not earn. Ready therefore means: 200, AND the
// response body carries a marker only the real page has, AND the document is closed.
const READY_MARKER = args.readyMarker ?? "Cal.diy";

const diffpackBin = join(repoRoot, "target", "release", "diffpack");
const nextBin = join(APP, "node_modules", ".bin", "next");
const logDir = join(demoDir, "logs");

// ---------------------------------------------------------------------------
// The two sides. Written once, run for both.

const SIDES = [
  {
    key: "dp",
    label: "diffpack",
    port: DP_PORT,
    // Everything a cold start has to remove for this side. `.diffpack-next/` is the
    // adapter's GENERATED GLUE, not compiled output: it is rewritten on every boot,
    // and the adapter is written so a tree without it behaves identically. It still
    // goes, because a cold start that deletes all of one side's generated files and
    // keeps some of the other's invites exactly the question a demo should not leave
    // open.
    outDirs: [join(WEB, ".diffpack-output"), join(WEB, ".diffpack-next")],
    dev: () => ({ cmd: diffpackBin, argv: ["dev", WEB, String(DP_PORT)], cwd: repoRoot }),
    build: () => ({ cmd: diffpackBin, argv: ["build-app", ".", "production"], cwd: WEB }),
  },
  {
    key: "tp",
    label: "Turbopack",
    port: TP_PORT,
    outDirs: [join(WEB, ".next")],
    dev: () => ({ cmd: nextBin, argv: ["dev", "--turbopack", "--port", String(TP_PORT)], cwd: WEB }),
    // `--turbopack` is passed EXPLICITLY even though Next 16 already defaults to it
    // for `build`. The default is both version- and config-dependent: Next 16 picks
    // Turbopack only when `process.env.TURBOPACK === "auto"` resolves that way, and
    // it hard-exits (code 1) when it auto-selects Turbopack for a project that has a
    // webpack config and no turbopack config. A demo whose headline is "vs Turbopack"
    // must not be one Next release or one config change away from quietly measuring
    // webpack instead — or from scoring that exit-1 as a win for the other side.
    build: () => ({ cmd: nextBin, argv: ["build", "--turbopack"], cwd: WEB }),
  },
];
const sideByKey = new Map(SIDES.map((s) => [s.key, s]));


// ---------------------------------------------------------------------------
// The scenario edits.
//
// Each one plants a VISIBLE badge carrying its token, so the viewer sees the change
// land in the frame and the probe can detect it with an indexed attribute lookup
// instead of scanning body text. `position: fixed` and a per-kind corner keep the
// badges visible and non-overlapping whatever page is showing.

const CORNERS = {
  island: "top: 8, left: 8",
  server: "top: 8, right: 8",
  shared: "bottom: 8, left: 8",
};
const SWATCH = {
  island: ["#7cf6c0", "#0b1220"],
  server: ["#ffd166", "#1a1206"],
  shared: ["#8ab4ff", "#080f22"],
};
const BADGE_LABEL = {
  island: "ISLAND EDIT",
  server: "SERVER COMPONENT EDIT",
  shared: "SHARED CLIENT EDIT",
};

function badge(kind, n) {
  const [fg, bg] = SWATCH[kind];
  const style =
    `{ position: "fixed", ${CORNERS[kind]}, zIndex: 2147483000, background: "${bg}", ` +
    `color: "${fg}", border: "3px solid ${fg}", borderRadius: 10, padding: "10px 16px", ` +
    `boxShadow: "0 8px 28px rgba(0,0,0,.45)", ` +
    `font: "800 18px/1.2 ui-monospace, SFMono-Regular, monospace", pointerEvents: "none" }`;
  return `<span data-dpmark="${kind}-${n}" style={${style}}>${BADGE_LABEL[kind]} #${n}</span>`;
}

function badgeRe(kind) {
  return new RegExp(`<span data-dpmark="${kind}-[^"]*"[^>]*>[^<]*</span>`);
}

// The stylesheet class of edit has no DOM footprint at all, so its token rides on a
// custom property and its visible effect is a ring around the whole page. Detecting
// the property proves the COMPILED SHEET reached the browser, not just that a file
// changed on disk.
const CSS_RING = ["#a855f7", "#22d3ee", "#f43f5e", "#facc15", "#34d399", "#fb923c"];
const CSS_BLOCK_RE = /\n\/\* dpmark \*\/[\s\S]*$/;
function cssBlock(n) {
  const color = CSS_RING[n % CSS_RING.length];
  return (
    `\n/* dpmark */\n` +
    `:root { --dpmark: css-${n}; }\n` +
    `body { box-shadow: inset 0 0 0 10px ${color}; }\n` +
    `body::before { content: "GLOBAL CSS EDIT #${n}"; position: fixed; right: 8px; bottom: 8px; ` +
    `z-index: 2147483000; padding: 10px 16px; border: 3px solid ${color}; border-radius: 10px; ` +
    `background: #111827; color: ${color}; box-shadow: 0 8px 28px rgba(0,0,0,.45); ` +
    `font: 800 18px/1.2 ui-monospace, SFMono-Regular, monospace; pointer-events: none; }\n`
  );
}

const KINDS = [
  {
    key: "island",
    name: "island edit",
    detail: "leaf client component, modules/auth/login-view.tsx",
    url: "/auth/login",
    file: join(WEB, "modules", "auth", "login-view.tsx"),
    anchor: ">Cal.diy<",
    plant: (s) => s.replace(">Cal.diy<", `>Cal.diy${badge("island", 1)}<`),
    step: (s, n) => s.replace(badgeRe("island"), badge("island", n)),
  },
  {
    key: "server",
    name: "server component edit",
    detail: "app/(use-page-wrapper)/auth/login/page.tsx",
    url: "/auth/login",
    file: join(WEB, "app", "(use-page-wrapper)", "auth", "login", "page.tsx"),
    anchor: "return <Login {...props} />;",
    plant: (s) =>
      s.replace(
        "return <Login {...props} />;",
        `return (<>${badge("server", 1)}<Login {...props} /></>);`,
      ),
    step: (s, n) => s.replace(badgeRe("server"), badge("server", n)),
  },
  {
    key: "shared",
    name: "shared client component edit",
    detail: "components/PageWrapperAppDir.tsx, seen on the booker",
    url: "/pro/30min",
    file: join(WEB, "components", "PageWrapperAppDir.tsx"),
    anchor: "return (\n    <>",
    plant: (s) => s.replace("return (\n    <>", `return (\n    <>${badge("shared", 1)}`),
    step: (s, n) => s.replace(badgeRe("shared"), badge("shared", n)),
  },
  {
    key: "css",
    name: "global stylesheet edit",
    detail: "styles/globals.css, recompiles Tailwind",
    url: "/auth/login",
    file: join(WEB, "styles", "globals.css"),
    anchor: null, // appended
    plant: (s) => s.replace(CSS_BLOCK_RE, "") + cssBlock(1),
    step: (s, n) => s.replace(CSS_BLOCK_RE, "") + cssBlock(n),
  },
];
const kindByKey = new Map(KINDS.map((k) => [k.key, k]));

// Routes offered as cold-compile races. All public on the seeded checkout — these
// are the ones the reference audit already compares byte for byte.
//
// `pattern` is the Next route a path belongs to, and it is what decides whether a
// chip says cold or warm: /pro/60min after /pro/30min is a NEW URL but the SAME
// compiled route, so calling it cold would credit both bundlers with work neither
// of them does.
const ROUTES = [
  { path: "/auth/login", label: "login", pattern: "/auth/login" },
  { path: "/pro", label: "pro (profile)", pattern: "/[user]" },
  { path: "/pro/30min", label: "booker 30min", pattern: "/[user]/[type]" },
  { path: "/pro/60min", label: "booker 60min", pattern: "/[user]/[type]" },
  { path: "/apps", label: "app store", pattern: "/apps" },
  { path: "/event-types", label: "event types", pattern: "/event-types" },
  { path: "/settings/my-account/profile", label: "settings", pattern: "/settings/my-account/profile" },
  { path: "/does-not-exist", label: "404 through the user route", pattern: "/[user]" },
];

// ---------------------------------------------------------------------------
// Managed files: the probe tag, the demo-only frame-header strip, every edit target.

const layoutFile = join(WEB, "app", "layout.tsx");
const wrapperFile = join(WEB, "next.config.ts");
const demoBaseConfig = join(WEB, "next.config.__diffpack_demo_base__.ts");
const probeDest = join(WEB, "public", "diffpack-demo-probe.js");

// `async` so the probe never blocks body parsing in the page being measured. It
// posts a snapshot as soon as it runs, so arriving late costs nothing: whatever
// tokens are already on screen are reported immediately.
const PROBE_TAG = `        <script src="/diffpack-demo-probe.js" data-diffpack-demo="1" async />\n`;
const LAYOUT_ANCHOR = "        <IconSprites />\n";

const DEMO_CONFIG_MARKER = "// Generated temporarily by diffpack's side-by-side demo.";
function demoConfigModule() {
  return `${DEMO_CONFIG_MARKER}
import base from "./next.config.__diffpack_demo_base__.ts";

export default async (...args) => {
  const original = typeof base === "function" ? await base(...args) : await base;
  const config = { ...(original ?? {}) };
  if (process.env.DIFFPACK_DEMO === "1") {
    // cal.com sends X-Frame-Options: DENY on /auth/*; remove only that header so the
    // two demo frames are visible. Empty header entries are invalid in Next.
    const originalHeaders = config.headers;
    config.headers = async () => {
      const list = originalHeaders ? await originalHeaders() : [];
      return list
        .map((entry) => ({
          ...entry,
          headers: (entry.headers ?? []).filter((header) => header.key !== "X-Frame-Options"),
        }))
        .filter((entry) => entry.headers.length > 0);
    };
  }
  // A production-build race compares bundlers, not diffpack against tsc. diffpack
  // performs no type/lint pass, so the reference build skips those passes too.
  config.typescript = { ...(config.typescript ?? {}), ignoreBuildErrors: true };
  config.eslint = { ...(config.eslint ?? {}), ignoreDuringBuilds: true };
  return config;
};
`;
}

const pristine = new Map(); // path -> original text
const state = {
  planted: new Set(),
  counters: new Map(),
  procs: new Map(), // side key -> { proc, ready, startedAt, bootMs, status }
  busy: null, // a label while a restart/build is in flight
};

preflight();
installDemoFiles();
let shuttingDown = false;
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => shutdown(sig));

// ---------------------------------------------------------------------------
// HTTP: dashboard, state, SSE, scenario actions.

const sseClients = new Set();

const http = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  try {
    if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
      const html = readFileSync(join(demoDir, "dashboard.html"));
      res.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
      res.end(html);
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/state") return json(res, publicState());
    if (req.method === "GET" && url.pathname === "/api/events") return sse(req, res);
    if (req.method === "POST" && url.pathname === "/api/edit") return await apiEdit(req, res);
    if (req.method === "POST" && url.pathname === "/api/burst") return await apiBurst(req, res);
    if (req.method === "POST" && url.pathname === "/api/reset") return await apiReset(req, res);
    if (req.method === "POST" && url.pathname === "/api/restart") return await apiRestart(req, res);
    if (req.method === "POST" && url.pathname === "/api/build") return await apiBuild(req, res);
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("not found\n");
  } catch (err) {
    console.error(err);
    json(res, { error: String(err && err.message ? err.message : err) }, 500);
  }
});

http.listen(PORT, () => {
  console.log(`\ndiffpack vs Turbopack, side by side`);
  console.log(`  dashboard   http://localhost:${PORT}`);
  console.log(`  diffpack    http://localhost:${DP_PORT}`);
  console.log(`  Turbopack   http://localhost:${TP_PORT}`);
  console.log(`  app         ${WEB}`);
  console.log(`  dev logs    ${logDir}/{dp,tp}.log\n`);
  if (BOOT) for (const side of racingOrder("boot", SIDES)) bootSide(side);
  else for (const side of SIDES) attachSide(side);
});

// ---------------------------------------------------------------------------
// Scenario actions

async function apiEdit(req, res) {
  const body = await readJson(req);
  const kind = kindByKey.get(body.kind);
  if (!kind) return json(res, { error: `unknown scenario ${body.kind}` }, 400);
  const result = applyEdit(kind);
  broadcast({ type: "edit", ...result });
  json(res, result);
}

// Sustained edits with no settle gap: the contended path, where a bundler that
// queues work behind a deferred pass shows it. The dashboard times each edit from
// the SSE frame that announces it.
async function apiBurst(req, res) {
  const body = await readJson(req);
  const kind = kindByKey.get(body.kind);
  if (!kind) return json(res, { error: `unknown scenario ${body.kind}` }, 400);
  const count = clamp(Number(body.count) || 5, 1, 20);
  const gapMs = clamp(Number(body.gapMs) || 1000, 200, 10000);
  json(res, { started: true, kind: kind.key, count, gapMs, url: kind.url });
  for (let i = 0; i < count; i++) {
    if (i) await sleep(gapMs);
    const result = applyEdit(kind);
    broadcast({ type: "edit", ...result, burst: { index: i, count } });
  }
  broadcast({ type: "burst-done", kind: kind.key, count });
}

function applyEdit(kind) {
  const before = readFileSync(kind.file, "utf8");
  const first = !state.planted.has(kind.key);
  const n = first ? 1 : (state.counters.get(kind.key) ?? 0) + 1;
  const after = first ? kind.plant(before) : kind.step(before, n);
  if (after === before) {
    // Never report a measurement for an edit that changed nothing: that is the one
    // failure mode that would silently make either side look infinitely fast.
    throw new Error(
      `the ${kind.key} edit changed nothing in ${kind.file} — the badge or anchor is gone; POST /api/reset`,
    );
  }
  const t0 = Date.now();
  writeFileSync(kind.file, after);
  state.planted.add(kind.key);
  state.counters.set(kind.key, n);
  return {
    kind: kind.key,
    name: kind.name,
    token: `${kind.key}-${n}`,
    n,
    warmup: first,
    url: kind.url,
    writeMs: Date.now() - t0,
    at: t0,
  };
}

async function apiReset(req, res) {
  restoreEdits();
  state.planted.clear();
  state.counters.clear();
  broadcast({ type: "reset" });
  json(res, { ok: true });
}

async function apiRestart(req, res) {
  if (state.busy) return json(res, { error: `busy: ${state.busy}` }, 409);
  const body = await readJson(req).catch(() => ({}));
  const wipe = body.wipe !== false;
  const keys = Array.isArray(body.sides) && body.sides.length ? body.sides : ["dp", "tp"];
  const sides = keys.map((k) => sideByKey.get(k)).filter(Boolean);
  if (!sides.length) return json(res, { error: "no such sides" }, 400);
  state.busy = "restart";
  const order = racingOrder("boot", sides);
  json(res, { restarting: order.map((s) => s.key), wipe });
  broadcast({
    type: "restart-begin",
    sides: sides.map((s) => s.key),
    order: order.map((s) => s.key),
    wipe,
  });
  try {
    await Promise.all(sides.map((s) => stopSide(s)));
    // A cold start means cold: with the output tree left in place a restart measures
    // a warm boot, which is a different and much smaller number. Every side's wipe
    // finishes before any side boots, so no side pays for the other's deletion.
    if (wipe) {
      for (const s of sides) for (const dir of s.outDirs) rmSync(dir, { recursive: true, force: true });
    }
    for (const s of order) bootSide(s);
  } finally {
    state.busy = null;
  }
}

// The headline production-build number. Both builds run at once so the two clocks
// race on screen; that contention is called out in the UI, and the isolated
// interleaved measurement stays in scripts/bench-calcom.mjs.
async function apiBuild(req, res) {
  if (state.busy) return json(res, { error: `busy: ${state.busy}` }, 409);
  state.busy = "build";
  const order = racingOrder("build", SIDES);
  json(res, { started: true, order: order.map((s) => s.key) });
  broadcast({ type: "build-begin", sides: SIDES.map((s) => s.key), order: order.map((s) => s.key) });
  try {
    await Promise.all(SIDES.map((s) => stopSide(s)));
    for (const s of SIDES) for (const dir of s.outDirs) rmSync(dir, { recursive: true, force: true });
    // `order` decides who is spawned first; it alternates per race, because the first
    // process spawned gets a moment of the machine to itself before the other starts.
    await Promise.all(order.map((side) => runBuild(side)));
    broadcast({ type: "build-done" });
    for (const s of racingOrder("boot", SIDES)) bootSide(s);
  } finally {
    state.busy = null;
  }
}

function runBuild(side) {
  return new Promise((resolve) => {
    const { cmd, argv, cwd } = side.build();
    const t0 = Date.now();
    broadcast({ type: "build-start", side: side.key, at: t0 });
    const proc = spawn("/usr/bin/time", ["-l", cmd, ...argv], {
      cwd,
      env: sideEnv(side),
      stdio: ["ignore", "pipe", "pipe"],
    });
    const rss = sampleTreeRss(proc.pid);
    let out = "";
    const onData = (d) => {
      out += d;
      for (const line of String(d).split("\n")) {
        const t = line.trim();
        if (t) broadcast({ type: "log", side: side.key, line: plain(t), from: "build" });
      }
    };
    proc.stdout.on("data", onData);
    proc.stderr.on("data", onData);
    proc.on("close", (code) => {
      const peakTreeMb = rss.stop();
      const user = /([0-9.]+)\s+user/.exec(out);
      // `maximum resident set size` from `/usr/bin/time -l` is `ru_maxrss`, which for
      // children is the max over any ONE waited-for process, never the sum. It is
      // reported alongside the sampled tree peak, clearly labelled, and it is NOT the
      // headline: see `sampleTreeRss`.
      const single = /([0-9]+)\s+maximum resident set size/.exec(out);
      broadcast({
        type: "build-end",
        side: side.key,
        ms: Date.now() - t0,
        code,
        cpuUserS: user ? Number(user[1]) : null,
        peakRssMb: peakTreeMb,
        peakRssSingleMb: single ? Number(single[1]) / (1024 * 1024) : null,
        rssSamples: rss.count(),
      });
      resolve();
    });
  });
}

// ---------------------------------------------------------------------------
// Dev server lifecycle

function bootSide(side) {
  const owners = portOwners(side.port);
  if (owners.length) {
    // A leaked dev server answering from a previous run once made Turbopack look
    // 2.3x faster than it is. Refuse rather than measure someone else's process.
    setStatus(side, "error", `port ${side.port} is held by pid(s) ${owners.join(", ")}`);
    return;
  }
  const { cmd, argv, cwd } = side.dev();
  const logPath = join(logDir, `${side.key}.log`);
  const logStream = createWriteStream(logPath, { flags: "a" });
  logStream.write(`\n===== boot ${new Date().toISOString()} : ${cmd} ${argv.join(" ")} =====\n`);
  const proc = spawn(cmd, argv, { cwd, env: sideEnv(side), stdio: ["ignore", "pipe", "pipe"] });
  const entry = { proc, startedAt: Date.now(), bootMs: null, status: "booting", note: null, logStream };
  state.procs.set(side.key, entry);
  broadcast({ type: "status", side: side.key, status: "booting", at: entry.startedAt });

  const onData = (d) => {
    logStream.write(d);
    for (const line of String(d).split("\n")) {
      const t = line.trim();
      if (t) broadcast({ type: "log", side: side.key, line: plain(t), from: "dev" });
    }
  };
  proc.stdout.on("data", onData);
  proc.stderr.on("data", onData);
  proc.on("exit", (code) => {
    logStream.write(`\n[dev process exited with ${code}]\n`);
    if (state.procs.get(side.key) === entry && !shuttingDown && entry.status !== "stopping") {
      setStatus(side, "error", `dev process exited with ${code}; see ${logPath}`);
    }
  });

  waitForReady(side, entry);
}

function attachSide(side) {
  const entry = { proc: null, startedAt: Date.now(), bootMs: null, status: "booting", note: "attached" };
  state.procs.set(side.key, entry);
  waitForReady(side, entry);
}

async function waitForReady(side, entry) {
  const deadline = Date.now() + BOOT_TIMEOUT_MS;
  // How many times this side answered 200 with a body that was not the real page.
  // Counted rather than ignored: a side that does that is being handed free boot time
  // by any harness whose gate is the status code alone, and the count is the evidence.
  entry.shellHits = 0;
  while (Date.now() < deadline) {
    if (state.procs.get(side.key) !== entry) return; // superseded by a newer boot
    if (entry.status === "error") return;
    try {
      const r = await fetch(`http://127.0.0.1:${side.port}${READY_PATH}`, {
        signal: AbortSignal.timeout(30000),
        redirect: "manual",
      });
      const body = await r.text().catch(() => "");
      if (r.status === 200) {
        if (body.includes(READY_MARKER) && body.includes("</html>")) {
          entry.bootMs = Date.now() - entry.startedAt;
          if (entry.shellHits) {
            // Both channels: the SSE log pane is empty when no dashboard is attached,
            // and this is evidence about how a boot time was earned, not a debug aid.
            const line =
              `answered 200 without ${JSON.stringify(READY_MARKER)}, or an unclosed document, ` +
              `${entry.shellHits}x before it was really ready; those did not count as ready`;
            console.log(`[${side.key}] ${line}`);
            broadcast({ type: "log", side: side.key, from: "demo", line });
          }
          setStatus(side, "ready");
          broadcast({
            type: "ready",
            side: side.key,
            ms: entry.bootMs,
            at: Date.now(),
            shellHits: entry.shellHits,
          });
          return;
        }
        entry.shellHits++;
      }
    } catch {}
    await sleep(25);
  }
  setStatus(
    side,
    "error",
    `no 200 carrying ${JSON.stringify(READY_MARKER)} from ${READY_PATH} within ` +
      `${BOOT_TIMEOUT_MS} ms (${entry.shellHits} bare 200s); wrong --ready-marker looks exactly like this`,
  );
}

function setStatus(side, status, note) {
  const entry = state.procs.get(side.key);
  if (!entry) return;
  entry.status = status;
  if (note !== undefined) entry.note = note;
  broadcast({ type: "status", side: side.key, status, note: entry.note ?? null, bootMs: entry.bootMs });
}

async function stopSide(side) {
  const entry = state.procs.get(side.key);
  if (entry) {
    entry.status = "stopping";
    broadcast({ type: "status", side: side.key, status: "stopping" });
    if (entry.proc) killTree(entry.proc);
    if (entry.logStream) entry.logStream.end();
  }
  state.procs.delete(side.key);
  await waitPortFree(side.port, 60000);
}

// Kill the whole descendant tree: both dev servers outlive a bare kill of the
// parent (diffpack keeps its react-server worker, next keeps next-server).
function killTree(proc) {
  const pid = proc.pid;
  if (!pid) return;
  const all = [];
  const walk = (root) => {
    all.push(root);
    let kids = "";
    try {
      kids = execFileSync("pgrep", ["-P", String(root)], { encoding: "utf8" });
    } catch {}
    for (const line of kids.split("\n")) {
      const k = Number(line.trim());
      if (k) walk(k);
    }
  };
  walk(pid);
  for (const p of all.reverse()) {
    try {
      process.kill(p, "SIGTERM");
    } catch {}
  }
  setTimeout(() => {
    for (const p of all) {
      try {
        process.kill(p, "SIGKILL");
      } catch {}
    }
  }, 1500);
}

function portOwners(port) {
  try {
    return execFileSync("lsof", ["-tnP", `-iTCP:${port}`, "-sTCP:LISTEN"], { encoding: "utf8" })
      .trim()
      .split("\n")
      .filter(Boolean);
  } catch {
    return [];
  }
}

async function waitPortFree(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!portOwners(port).length) return;
    await sleep(150);
  }
  throw new Error(`port ${port} still held after teardown by pid(s) ${portOwners(port).join(", ")}`);
}

// Each side gets its own absolute URLs. cal.com bakes NEXT_PUBLIC_WEBAPP_URL into
// the browser bundle; leaving both at :3000 would have the Turbopack page calling
// the diffpack server's API, so the frame would not be showing its own bundler.
function sideEnv(side) {
  const origin = `http://localhost:${side.port}`;
  return {
    ...process.env,
    FORCE_COLOR: "0",
    NEXT_TELEMETRY_DISABLED: "1",
    DIFFPACK_DEMO: "1",
    NEXT_PUBLIC_WEBAPP_URL: origin,
    NEXT_PUBLIC_WEBSITE_URL: origin,
    NEXTAUTH_URL: origin,
  };
}

// ---------------------------------------------------------------------------
// Managed source files

function preflight() {
  if (!existsSync(diffpackBin)) throw new Error(`missing ${diffpackBin}; run \`cargo build --release\``);
  if (!existsSync(WEB)) throw new Error(`missing ${WEB}; pass --app <cal.com checkout>`);
  if (!existsSync(nextBin)) throw new Error(`missing ${nextBin}; install that checkout's node_modules`);
  if (!existsSync(wrapperFile)) throw new Error(`missing ${wrapperFile}; the demo expects cal.com's Next config`);
  if (DP_PORT === TP_PORT) throw new Error(`--dp-port and --tp-port must differ`);

  // Normally the base file does not exist. If a previous demo was killed with SIGKILL,
  // recover the pristine config from it instead of snapshotting our generated wrapper.
  const wrapperOnDisk = readFileSync(wrapperFile, "utf8");
  if (existsSync(demoBaseConfig)) {
    if (!wrapperOnDisk.startsWith(DEMO_CONFIG_MARKER)) {
      throw new Error(
        `${demoBaseConfig} already exists but ${wrapperFile} is not the demo wrapper; ` +
          `refusing to overwrite an unrelated file`,
      );
    }
    pristine.set(wrapperFile, readFileSync(demoBaseConfig, "utf8"));
  } else {
    pristine.set(wrapperFile, wrapperOnDisk);
  }

  for (const f of [layoutFile, ...KINDS.map((k) => k.file)]) {
    if (!existsSync(f)) throw new Error(`missing ${f}`);
    pristine.set(f, readFileSync(f, "utf8"));
  }
  for (const k of KINDS) {
    if (k.anchor && !pristine.get(k.file).includes(k.anchor)) {
      throw new Error(
        `edit anchor ${JSON.stringify(k.anchor)} is gone from ${k.file} — refusing to start: ` +
          `the ${k.key} scenario would change nothing`,
      );
    }
  }
  if (!pristine.get(layoutFile).includes(LAYOUT_ANCHOR)) {
    throw new Error(`cannot find the probe insertion point ${JSON.stringify(LAYOUT_ANCHOR)} in ${layoutFile}`);
  }
  mkdirSync(logDir, { recursive: true });
}

function installDemoFiles() {
  mkdirSync(join(WEB, "public"), { recursive: true });
  copyFileSync(join(demoDir, "probe.js"), probeDest);
  writeFileSync(layoutFile, pristine.get(layoutFile).replace(LAYOUT_ANCHOR, PROBE_TAG + LAYOUT_ANCHOR));
  writeFileSync(demoBaseConfig, pristine.get(wrapperFile));
  writeFileSync(wrapperFile, demoConfigModule());
}

function restoreEdits() {
  for (const k of KINDS) {
    const text = pristine.get(k.file);
    try {
      if (readFileSync(k.file, "utf8") !== text) writeFileSync(k.file, text);
    } catch {}
  }
}

function restoreAll() {
  restoreEdits();
  for (const f of [layoutFile, wrapperFile]) {
    const text = pristine.get(f);
    try {
      if (readFileSync(f, "utf8") !== text) writeFileSync(f, text);
    } catch {}
  }
  try {
    rmSync(probeDest, { force: true });
  } catch {}
  try {
    rmSync(demoBaseConfig, { force: true });
  } catch {}
}

function shutdown(sig) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`\n${sig}: restoring sources and stopping both dev servers`);
  restoreAll();
  for (const side of SIDES) {
    const entry = state.procs.get(side.key);
    if (entry?.proc) killTree(entry.proc);
  }
  setTimeout(() => process.exit(0), 2000);
}

// ---------------------------------------------------------------------------
// HTTP plumbing

function publicState() {
  return {
    dashboardPort: PORT,
    app: WEB,
    appCommit: gitHead(APP),
    diffpackCommit: gitHead(repoRoot),
    nextVersion: nextVersion(),
    readyPath: READY_PATH,
    readyMarker: READY_MARKER,
    busy: state.busy,
    sides: SIDES.map((s) => {
      const e = state.procs.get(s.key);
      return {
        key: s.key,
        label: s.label,
        port: s.port,
        origin: `http://localhost:${s.port}`,
        status: e?.status ?? "stopped",
        note: e?.note ?? null,
        bootMs: e?.bootMs ?? null,
      };
    }),
    scenarios: KINDS.map((k) => ({ key: k.key, name: k.name, detail: k.detail, url: k.url })),
    routes: ROUTES,
    planted: [...state.planted],
  };
}

function sse(req, res) {
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-store",
    connection: "keep-alive",
  });
  res.write(`retry: 1000\n\n`);
  const client = { res };
  sseClients.add(client);
  send(client, { type: "state", state: publicState() });
  const keepalive = setInterval(() => res.write(`: ping\n\n`), 15000);
  req.on("close", () => {
    clearInterval(keepalive);
    sseClients.delete(client);
  });
}

function send(client, obj) {
  try {
    client.res.write(`data: ${JSON.stringify(obj)}\n\n`);
  } catch {}
}

function broadcast(obj) {
  for (const c of sseClients) send(c, obj);
}

function json(res, obj, code = 200) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { "content-type": "application/json", "cache-control": "no-store" });
  res.end(body);
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      if (!body) return resolve({});
      try {
        resolve(JSON.parse(body));
      } catch (err) {
        reject(new Error(`bad JSON body: ${err.message}`));
      }
    });
    req.on("error", reject);
  });
}

// ---------------------------------------------------------------------------

function gitHead(dir) {
  try {
    return execFileSync("git", ["-C", dir, "rev-parse", "--short", "HEAD"], { encoding: "utf8" }).trim();
  } catch {
    return "unknown";
  }
}

function nextVersion() {
  try {
    return JSON.parse(readFileSync(join(APP, "node_modules", "next", "package.json"), "utf8")).version;
  } catch {
    return "unknown";
  }
}

// Next colourises its dev output even under FORCE_COLOR=0; the dashboard shows
// plain text, so the escapes come off on the way out. The on-disk log keeps the
// raw bytes.
const ANSI_RE = /\[[0-9;?]*[A-Za-z]/g;
function plain(line) {
  return line.replace(ANSI_RE, "");
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function parseArgs(a) {
  const o = {};
  for (let i = 0; i < a.length; i++) {
    const k = a[i];
    if (k === "--app") o.app = a[++i];
    else if (k === "--port") o.port = Number(a[++i]);
    else if (k === "--dp-port") o.dpPort = Number(a[++i]);
    else if (k === "--tp-port") o.tpPort = Number(a[++i]);
    else if (k === "--boot-timeout") o.bootTimeout = Number(a[++i]);
    else if (k === "--ready-path") o.readyPath = a[++i];
    else if (k === "--ready-marker") o.readyMarker = a[++i];
    else if (k === "--no-boot") o.noBoot = true;
    else throw new Error(`unknown arg ${k}`);
  }
  return o;
}
