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
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const demoDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = dirname(demoDir);

const args = parseArgs(process.argv.slice(2));
const APP = args.app ?? "/tmp/dpe2e/calcom";
const WEB = join(APP, "apps", "web");
const DP_PORT = args.dpPort ?? 3000;
const TP_PORT = args.tpPort ?? 3001;
const PORT = args.port ?? 4321;
const BOOT = !args.noBoot;
const BOOT_TIMEOUT_MS = args.bootTimeout ?? 240000;
const READY_PATH = args.readyPath ?? "/auth/login";

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
    outDir: join(WEB, ".diffpack-output"),
    dev: () => ({ cmd: diffpackBin, argv: ["dev", WEB, String(DP_PORT)], cwd: repoRoot }),
    build: () => ({ cmd: diffpackBin, argv: ["build-app", ".", "production"], cwd: WEB }),
  },
  {
    key: "tp",
    label: "Turbopack",
    port: TP_PORT,
    outDir: join(WEB, ".next"),
    dev: () => ({ cmd: nextBin, argv: ["dev", "--turbopack", "--port", String(TP_PORT)], cwd: WEB }),
    build: () => ({ cmd: nextBin, argv: ["build"], cwd: WEB }),
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

function badge(kind, n) {
  const [fg, bg] = SWATCH[kind];
  const style =
    `{ position: "fixed", ${CORNERS[kind]}, zIndex: 2147483000, background: "${bg}", ` +
    `color: "${fg}", border: "2px solid ${fg}", borderRadius: 8, padding: "5px 11px", ` +
    `font: "700 13px/1.2 ui-monospace, SFMono-Regular, monospace", pointerEvents: "none" }`;
  return `<span data-dpmark="${kind}-${n}" style={${style}}>${kind} #${n}</span>`;
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
    `body { box-shadow: inset 0 0 0 10px ${color}; }\n`
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
    plant: (s) => s.replace(">Cal.diy<", `>Cal.diy${badge("island", 0)}<`),
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
        `return (<>${badge("server", 0)}<Login {...props} /></>);`,
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
    plant: (s) => s.replace("return (\n    <>", `return (\n    <>${badge("shared", 0)}`),
    step: (s, n) => s.replace(badgeRe("shared"), badge("shared", n)),
  },
  {
    key: "css",
    name: "global stylesheet edit",
    detail: "styles/globals.css, recompiles Tailwind",
    url: "/auth/login",
    file: join(WEB, "styles", "globals.css"),
    anchor: null, // appended
    plant: (s) => s.replace(CSS_BLOCK_RE, "") + cssBlock(0),
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
const probeDest = join(WEB, "public", "diffpack-demo-probe.js");

// `async` so the probe never blocks body parsing in the page being measured. It
// posts a snapshot as soon as it runs, so arriving late costs nothing: whatever
// tokens are already on screen are reported immediately.
const PROBE_TAG = `        <script src="/diffpack-demo-probe.js" data-diffpack-demo="1" async />\n`;
const LAYOUT_ANCHOR = "        <IconSprites />\n";

const WRAPPER_ANCHOR = "  const config = orig(phase);\n";
const WRAPPER_PATCH = `  if (process.env.DIFFPACK_DEMO === "1") {
    // The demo dashboard frames this app side by side. cal.com sends
    // X-Frame-Options: DENY on /auth/*, which would blank the frame, so the demo
    // flag — and only the demo flag — drops that one header.
    const origHeaders = config.headers;
    config.headers = async () => {
      const list = origHeaders ? await origHeaders() : [];
      // An entry left with no headers at all is rejected by next ("\`headers\`
      // field cannot be empty for route"), so those entries go away entirely.
      return list
        .map((entry) => ({
          ...entry,
          headers: (entry.headers ?? []).filter((h) => h.key !== "X-Frame-Options"),
        }))
        .filter((entry) => entry.headers.length > 0);
    };
  }
`;

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
  if (BOOT) for (const side of SIDES) bootSide(side);
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
  const n = first ? 0 : (state.counters.get(kind.key) ?? 0) + 1;
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
  json(res, { restarting: sides.map((s) => s.key), wipe });
  broadcast({ type: "restart-begin", sides: sides.map((s) => s.key), wipe });
  try {
    await Promise.all(sides.map((s) => stopSide(s)));
    // A cold start means cold: with the output tree left in place a restart measures
    // a warm boot, which is a different and much smaller number.
    if (wipe) for (const s of sides) rmSync(s.outDir, { recursive: true, force: true });
    for (const s of sides) bootSide(s);
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
  json(res, { started: true });
  broadcast({ type: "build-begin", sides: SIDES.map((s) => s.key) });
  try {
    await Promise.all(SIDES.map((s) => stopSide(s)));
    for (const s of SIDES) rmSync(s.outDir, { recursive: true, force: true });
    await Promise.all(SIDES.map((side) => runBuild(side)));
    broadcast({ type: "build-done" });
    for (const s of SIDES) bootSide(s);
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
      const user = /([0-9.]+)\s+user/.exec(out);
      const rss = /([0-9]+)\s+maximum resident set size/.exec(out);
      broadcast({
        type: "build-end",
        side: side.key,
        ms: Date.now() - t0,
        code,
        cpuUserS: user ? Number(user[1]) : null,
        peakRssMb: rss ? Number(rss[1]) / (1024 * 1024) : null,
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
  while (Date.now() < deadline) {
    if (state.procs.get(side.key) !== entry) return; // superseded by a newer boot
    if (entry.status === "error") return;
    try {
      const r = await fetch(`http://127.0.0.1:${side.port}${READY_PATH}`, {
        signal: AbortSignal.timeout(30000),
        redirect: "manual",
      });
      await r.text().catch(() => {});
      if (r.status === 200) {
        entry.bootMs = Date.now() - entry.startedAt;
        setStatus(side, "ready");
        broadcast({ type: "ready", side: side.key, ms: entry.bootMs, at: Date.now() });
        return;
      }
    } catch {}
    await sleep(25);
  }
  setStatus(side, "error", `no 200 from ${READY_PATH} within ${BOOT_TIMEOUT_MS} ms`);
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
  if (DP_PORT === TP_PORT) throw new Error(`--dp-port and --tp-port must differ`);

  for (const f of [layoutFile, wrapperFile, ...KINDS.map((k) => k.file)]) {
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
  if (!pristine.get(wrapperFile).includes(WRAPPER_ANCHOR)) {
    throw new Error(
      `cannot find ${JSON.stringify(WRAPPER_ANCHOR)} in ${wrapperFile} — the demo needs to drop ` +
        `X-Frame-Options for the framed pages; is this the benchmark next.config wrapper?`,
    );
  }
  mkdirSync(logDir, { recursive: true });
}

function installDemoFiles() {
  mkdirSync(join(WEB, "public"), { recursive: true });
  copyFileSync(join(demoDir, "probe.js"), probeDest);
  writeFileSync(layoutFile, pristine.get(layoutFile).replace(LAYOUT_ANCHOR, PROBE_TAG + LAYOUT_ANCHOR));
  writeFileSync(
    wrapperFile,
    pristine.get(wrapperFile).replace(WRAPPER_ANCHOR, WRAPPER_ANCHOR + WRAPPER_PATCH),
  );
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
    else if (k === "--no-boot") o.noBoot = true;
    else throw new Error(`unknown arg ${k}`);
  }
  return o;
}
