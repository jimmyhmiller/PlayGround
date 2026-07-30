// The dashboard's fairness rules, tested offline.
//
//   node demo/dashboard.test.mjs [--app /tmp/dpe2e/calcom]
//
// The rules this file exists to hold in place are the ones that decide whether a
// number on screen is comparable at all:
//
//   1. Navigation and editing are separate phases: loading or revisiting a route
//      performs no source edit, while a scenario button may navigate, settle, and then
//      performs exactly one source edit.
//   2. Every landing says HOW it arrived: `hot` (the running page was patched) or
//      `reload` (that side replaced the document). A fast reload is not plainly better
//      than a slower in-place patch, so the table may not hide the difference.
//   3. A non-zero exit is FAILED, never a timeout: a crash on one side must not read
//      as the other side being fast.
//   4. Nothing is dropped: superseded, timed-out and failed runs all keep their row.
//
// It drives the real `demo/dashboard.html` in jsdom with the server stubbed, so it
// needs no dev server, no browser and no cal.com build. jsdom is resolved from the
// cal.com checkout (it is a transitive dependency there); nothing is installed here.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const demoDir = dirname(fileURLToPath(import.meta.url));
const APP = argValue("--app") ?? "/tmp/dpe2e/calcom";

const DP_ORIGIN = "http://localhost:3000";
const TP_ORIGIN = "http://localhost:3001";

let failures = 0;
let checks = 0;

function check(what, cond, detail) {
  checks++;
  if (cond) {
    console.log(`  ok   ${what}`);
  } else {
    failures++;
    console.log(`  FAIL ${what}${detail ? `\n       ${detail}` : ""}`);
  }
}

// ---------------------------------------------------------------------------
// Harness

async function loadJsdom() {
  const require_ = createRequire(join(APP, "package.json"));
  let entry;
  try {
    entry = require_.resolve("jsdom");
  } catch {
    throw new Error(`jsdom is not resolvable from ${APP}; pass --app <a checkout that has it>`);
  }
  return import(pathToFileURL(entry).href);
}

function state(overrides = {}) {
  return {
    dashboardPort: 4321,
    app: `${APP}/apps/web`,
    appCommit: "testcommit",
    diffpackCommit: "testcommit",
    nextVersion: "16.2.3",
    readyPath: "/auth/login",
    readyMarker: "Cal.diy",
    busy: null,
    sides: [
      { key: "dp", label: "diffpack", port: 3000, origin: DP_ORIGIN, status: "ready", note: null, bootMs: 6605 },
      { key: "tp", label: "Turbopack", port: 3001, origin: TP_ORIGIN, status: "ready", note: null, bootMs: 13151 },
    ],
    scenarios: [
      { key: "island", name: "island edit", detail: "leaf client component", url: "/auth/login" },
      { key: "css", name: "global stylesheet edit", detail: "styles/globals.css", url: "/auth/login" },
      { key: "shared", name: "shared client component edit", detail: "components/PageWrapperAppDir.tsx", url: "/pro/30min" },
    ],
    routes: [
      { path: "/auth/login", label: "login", pattern: "/auth/login" },
      { path: "/pro/30min", label: "booker", pattern: "/[user]/[type]" },
    ],
    planted: [],
    ...overrides,
  };
}

async function openDashboard(jsdom) {
  const html = readFileSync(join(demoDir, "dashboard.html"), "utf8");
  const posts = [];
  let editSeq = 0;
  // What /api/edit answers with. Overridable per test.
  let editReply = (body) => ({
    token: `${body.kind}-${++editSeq}`,
    n: editSeq,
    warmup: false,
    kind: body.kind,
    url: "/auth/login",
  });

  // Any uncaught error in the dashboard's own script fails the test that provoked it.
  // After `retire()` the window is being torn down and jsdom can still fire a queued
  // iframe load into it, which is noise rather than a defect.
  let retired = false;
  const scriptErrors = [];
  const virtualConsole = new jsdom.VirtualConsole();
  virtualConsole.on("jsdomError", (err) => {
    if (!retired) scriptErrors.push(String(err?.message ?? err));
  });

  const dom = new jsdom.JSDOM(html, {
    url: "http://localhost:4321/",
    runScripts: "dangerously",
    pretendToBeVisual: true, // requestAnimationFrame, which drives the clocks and timeouts
    virtualConsole,
    beforeParse(win) {
      // The server, stubbed: the dashboard only ever POSTs JSON and reads an SSE stream.
      win.fetch = async (path, init) => {
        const body = init?.body ? JSON.parse(init.body) : {};
        posts.push({ path, body });
        const data = path === "/api/edit" ? editReply(body) : { ok: true };
        return { ok: true, status: 200, json: async () => data };
      };
      win.EventSource = class {
        constructor() {
          win.__es = this;
        }
        close() {}
      };
    },
  });
  const win = dom.window;
  await new Promise((resolve) => {
    if (win.document.readyState === "complete") resolve();
    else win.addEventListener("load", resolve);
  });
  const push = (msg) => win.__es.onmessage({ data: JSON.stringify(msg) });
  return {
    win,
    posts,
    push,
    scriptErrors,
    retire() {
      retired = true;
      win.close();
    },
    setEditReply: (fn) => (editReply = fn),
    // A probe report from one side. `doc` is the document instance: reusing the id the
    // side reported before means the page was patched, a new id means it reloaded.
    probe: (origin, tokens, doc) =>
      win.dispatchEvent(
        new win.MessageEvent("message", {
          data: { source: "diffpack-demo-probe", kind: "tokens", tokens, doc, path: "/auth/login" },
          origin,
        }),
      ),
    // jsdom does not fetch iframe src, so the load both frames would fire is explicit.
    loadFrames: () => {
      for (const frame of win.document.querySelectorAll(".pane iframe")) {
        frame.dispatchEvent(new win.Event("load"));
      }
    },
    rows: () =>
      [...win.document.querySelectorAll("#results tr")].map((tr) =>
        [...tr.children].map((td) => ({ text: td.textContent.trim(), html: td.innerHTML })),
      ),
    // The token the dashboard is currently waiting for, read off the stage line it
    // writes once a race is armed. Tests wait for this rather than assuming a token:
    // an edit is POSTed before the reply arrives, so a probe sent the moment the POST
    // is recorded would reach a race that is not listening for anything yet — and it
    // would look exactly like the side missing the update.
    armedToken: () => {
      const stage = win.document.querySelector(".pane .stage")?.textContent ?? "";
      const m = /token (\S+)/.exec(stage);
      return m ? m[1] : null;
    },
    button: (label) =>
      [...win.document.querySelectorAll(".bar button")].find((b) => b.textContent.trim() === label),
  };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Wait for a condition the dashboard reaches on its own timers, rather than sleeping a
// guessed amount: a flaky test about fairness would be worse than no test.
async function until(what, cond, timeoutMs = 20000, context) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (cond()) return true;
    await sleep(20);
  }
  // A bare "timed out" tells you nothing about a dashboard whose whole job is to
  // report state, so whatever the caller knows goes into the message.
  throw new Error(
    `timed out after ${timeoutMs} ms waiting for: ${what}` +
      (context ? `\n  state at timeout: ${JSON.stringify(context(), null, 2)}` : ""),
  );
}

const editPosts = (posts) => posts.filter((p) => p.path === "/api/edit");
const findRow = (rows, needle) => rows.find((r) => r[0].text.includes(needle));
const findExactRow = (rows, label) => rows.find((r) => r[0].text === label);

// Wait until the dashboard is armed and waiting for a token it was not waiting for
// before, and return it.
async function nextArmedToken(d, previous, what, timeoutMs = 15000) {
  await until(what, () => d.armedToken() && d.armedToken() !== previous, timeoutMs, () => ({
    armed: d.armedToken(),
    previous,
    posts: d.posts.map((p) => p.path),
    rows: d.rows().map((r) => r.map((c) => c.text)),
  }));
  return d.armedToken();
}

// ---------------------------------------------------------------------------
// 1 + 2: navigation performs no edits, one press performs one edit, and every landing
// says whether the running page was patched or replaced.

async function testNavigationAndEditSeparation(jsdom) {
  console.log("\nnavigation never edits; one scenario press edits once; hot vs reload labelled");
  const d = await openDashboard(jsdom);
  d.push({ type: "state", state: state() });
  await until("panes built", () => d.win.document.querySelectorAll(".pane").length === 2);

  // Initial navigation to the default route is navigation only.
  await until("both frames navigated", () => {
    const frames = [...d.win.document.querySelectorAll(".pane iframe")];
    return frames.length === 2 && frames.every((f) => (f.getAttribute("src") ?? "").includes("/auth/login"));
  });
  d.loadFrames();
  await sleep(300);
  check(
    "arriving on a route performs no source edit",
    editPosts(d.posts).length === 0,
    JSON.stringify(d.posts.map((p) => p.path)),
  );

  // Re-visiting the same route forces a new iframe document via ?dpnav, but still
  // must not mutate source code.
  const route = d.win.document.querySelector('#routes button[data-path="/auth/login"]');
  route.click();
  d.loadFrames();
  // The probe reports the identity of each newly loaded document before an edit.
  d.probe(DP_ORIGIN, [], "dp-doc-1");
  d.probe(TP_ORIGIN, [], "tp-doc-1");
  await sleep(300);
  check(
    "clicking a route chip performs no source edit",
    editPosts(d.posts).length === 0,
    JSON.stringify(d.posts.map((p) => p.path)),
  );

  // A scenario press performs one edit, with no hidden edit before it.
  d.button("island").click();
  await until("the measured edit was written", () => editPosts(d.posts).length === 1, 15000);
  const measuredToken = await nextArmedToken(d, null, "the measured edit was armed");
  await sleep(300); // long enough for a second write to show up, if there were one
  check(
    "one press of a scenario writes exactly one edit",
    editPosts(d.posts).length === 1,
    `${editPosts(d.posts).length} edits written for one press`,
  );

  // diffpack patches the running page (same document id); Turbopack replaces the
  // document (new id). Same token, same clock, different cost to the user.
  d.probe(DP_ORIGIN, [measuredToken], "dp-doc-1");
  d.probe(TP_ORIGIN, [measuredToken], "tp-doc-2");
  await until("the measured row appeared", () => findExactRow(d.rows(), "island edit") !== undefined, 10000);

  const measured = findExactRow(d.rows(), "island edit");
  check(
    "the measured row is the requested scenario",
    measured !== undefined && measured[0].text === "island edit",
    JSON.stringify(measured?.[0].text),
  );
  check(
    "a side that patched the running page is labelled hot",
    /class="how hot"/.test(measured[1].html),
    measured[1].html,
  );
  check(
    "a side that replaced the document is labelled reload",
    /class="how reload"/.test(measured[2].html),
    measured[2].html,
  );
  check(
    "both sides still produced a time",
    /\ds\b|\d\.\d+ s/.test(measured[1].text) && /\ds\b|\d\.\d+ s/.test(measured[2].text),
    `${measured[1].text} | ${measured[2].text}`,
  );

  // A different scenario on the same route is also exactly one edit.
  const before = editPosts(d.posts).length;
  d.button("global stylesheet").click();
  await until("the second measured edit was written", () => editPosts(d.posts).length === before + 1, 10000);
  await sleep(300);
  check(
    "a second scenario press also writes exactly one edit",
    editPosts(d.posts).length === before + 1,
    `${editPosts(d.posts).length - before} edits written for one press`,
  );

  // An edit whose marker belongs to another route navigates first, but may not write
  // until both iframe loads have completed and the settle window has elapsed.
  const beforeShared = editPosts(d.posts).length;
  d.button("shared client component").click();
  await until("shared edit navigated both frames", () =>
    [...d.win.document.querySelectorAll(".pane iframe")].every((f) => f.dataset.path === "/pro/30min"),
  );
  const pathsAfterShared = [...d.win.document.querySelectorAll(".pane iframe")].map((f) => f.dataset.path);
  check(
    "an edit button automatically navigates to its target route",
    pathsAfterShared.every((path) => path === "/pro/30min"),
    JSON.stringify(pathsAfterShared),
  );
  check(
    "automatic navigation performs no source edit before the frames load",
    editPosts(d.posts).length === beforeShared,
    `${editPosts(d.posts).length - beforeShared} edits written`,
  );
  d.loadFrames();
  await sleep(300);
  check(
    "the settle window still performs no source edit",
    editPosts(d.posts).length === beforeShared,
    JSON.stringify(d.posts.map((p) => p.path)),
  );
  await until("the shared edit was written after settling", () => editPosts(d.posts).length === beforeShared + 1, 10000);
  await sleep(300);
  check(
    "shared client component performs exactly one edit after automatic navigation",
    editPosts(d.posts).length === beforeShared + 1,
    `${editPosts(d.posts).length - beforeShared} edits written`,
  );
  check("no uncaught error in the dashboard's own script", d.scriptErrors.length === 0, d.scriptErrors.join("\n       "));
  d.retire();
}

// The page can update before the edit endpoint's response has armed the token. The
// probe reports only when the visible token set changes, so dropping that early report
// would leave the clock running until an unrelated reload reported the DOM again.
async function testEarlyProbeIsRemembered(jsdom) {
  console.log("\na visible edit reported before /api/edit returns is still counted");
  const d = await openDashboard(jsdom);
  d.push({ type: "state", state: state() });
  await until("panes built", () => d.win.document.querySelectorAll(".pane").length === 2);
  await until("both frames navigated", () =>
    [...d.win.document.querySelectorAll(".pane iframe")].every((f) => f.dataset.path === "/auth/login"),
  );
  d.loadFrames();
  d.probe(DP_ORIGIN, [], "dp-doc-1");
  d.probe(TP_ORIGIN, [], "tp-doc-1");

  let releaseReply;
  d.setEditReply(
    () =>
      new Promise((resolve) => {
        releaseReply = () =>
          resolve({ token: "island-1", n: 1, warmup: false, kind: "island", url: "/auth/login" });
      }),
  );
  d.button("island").click();
  await until("the edit request is pending", () => editPosts(d.posts).length === 1 && releaseReply);

  // Both frames show the update while the dashboard still does not know which token
  // the pending request will return.
  d.probe(DP_ORIGIN, ["island-1"], "dp-doc-1");
  d.probe(TP_ORIGIN, ["island-1"], "tp-doc-1");
  releaseReply();

  await until("the early report closed the edit race", () => findExactRow(d.rows(), "island edit") !== undefined);
  const row = findExactRow(d.rows(), "island edit");
  check(
    "an early probe report stops both clocks without waiting for another DOM change",
    row && /\d\.\d+ s/.test(row[1].text) && /\d\.\d+ s/.test(row[2].text),
    JSON.stringify(row?.map((c) => c.text)),
  );
  check(
    "the remembered reports retain hot/reload classification",
    /class="how hot"/.test(row?.[1].html ?? "") && /class="how hot"/.test(row?.[2].html ?? ""),
    JSON.stringify(row?.map((c) => c.html)),
  );
  check("no uncaught error in the dashboard's own script", d.scriptErrors.length === 0, d.scriptErrors.join("\n       "));
  d.retire();
}

// ---------------------------------------------------------------------------
// 4: a crash is a failure, not a slow build.

async function testFailedBuildIsNotATimeout(jsdom) {
  console.log("\na non-zero build exit is FAILED, never a timeout");
  const d = await openDashboard(jsdom);
  d.push({ type: "state", state: state() });
  await until("panes built", () => d.win.document.querySelectorAll(".pane").length === 2);

  d.win.document.getElementById("build").click();
  await until("the build race started", () => d.posts.some((p) => p.path === "/api/build"));
  d.push({ type: "build-begin", sides: ["dp", "tp"], order: ["tp", "dp"] });
  d.push({ type: "build-end", side: "dp", ms: 41000, code: 0, cpuUserS: 300, peakRssMb: 4096, peakRssSingleMb: 2048, rssSamples: 164 });
  d.push({ type: "build-end", side: "tp", ms: 0, code: 1, cpuUserS: null, peakRssMb: null, peakRssSingleMb: null, rssSamples: 3 });
  await until("the build row appeared", () => findRow(d.rows(), "production build") !== undefined);

  const row = findRow(d.rows(), "production build");
  check("the crashed side reads FAILED", row[2].text === "FAILED", JSON.stringify(row[2].text));
  check(
    "the advantage column says it failed rather than implying slowness",
    row[3].text.includes("FAILED") && !/timed out/.test(row[3].text),
    JSON.stringify(row[3].text),
  );
  check(
    "the surviving side is not credited with a ratio against a crash",
    !/x diffpack/.test(row[3].text),
    JSON.stringify(row[3].text),
  );
  check("no uncaught error in the dashboard's own script", d.scriptErrors.length === 0, d.scriptErrors.join("\n       "));
  d.retire();
}

// ---------------------------------------------------------------------------
// The memory axis: the tree number is the headline, ru_maxrss is labelled as partial.

async function testMemoryRowsAreDistinguished(jsdom) {
  console.log("\nbuild memory: the whole-tree number is the comparable one");
  const d = await openDashboard(jsdom);
  d.push({ type: "state", state: state() });
  await until("panes built", () => d.win.document.querySelectorAll(".pane").length === 2);

  d.win.document.getElementById("build").click();
  await until("the build race started", () => d.posts.some((p) => p.path === "/api/build"));
  d.push({ type: "build-begin", sides: ["dp", "tp"], order: ["dp", "tp"] });
  d.push({ type: "build-end", side: "dp", ms: 41000, code: 0, cpuUserS: 300, peakRssMb: 6144, peakRssSingleMb: 2048, rssSamples: 164 });
  d.push({ type: "build-end", side: "tp", ms: 155000, code: 0, cpuUserS: 900, peakRssMb: 8192, peakRssSingleMb: 5120, rssSamples: 620 });
  await until("the memory rows appeared", () => findRow(d.rows(), "peak RSS, whole tree") !== undefined);

  const rows = d.rows();
  const tree = findRow(rows, "peak RSS, whole tree");
  const single = findRow(rows, "largest single process");
  check("the tree row reports the summed tree peak", tree[1].text === "6.00 GiB", JSON.stringify(tree[1].text));
  check("the tree row shows how many samples it is based on", /164\/620 samples/.test(tree[0].text), tree[0].text);
  check("ru_maxrss is reported too", single !== undefined && single[1].text === "2.00 GiB", JSON.stringify(single?.[1].text));
  check(
    "ru_maxrss is labelled as under-reporting rather than presented as the peak",
    /under-reports/.test(single[0].text),
    single[0].text,
  );
  check(
    "CPU is labelled as a whole-tree total",
    /whole process tree/.test(findRow(rows, "CPU")[0].text),
    findRow(rows, "CPU")[0].text,
  );
  check("no uncaught error in the dashboard's own script", d.scriptErrors.length === 0, d.scriptErrors.join("\n       "));
  d.retire();
}

// ---------------------------------------------------------------------------
// 5: a superseded edit keeps its row and is not called a timeout.

async function testSupersededKeepsItsRow(jsdom) {
  console.log("\nan edit a side never displayed reads `not shown`, and is kept");
  const d = await openDashboard(jsdom);
  d.push({ type: "state", state: state() });
  await until("panes built", () => d.win.document.querySelectorAll(".pane").length === 2);

  // The real burst path starts only from its explicit button; merely loading the route
  // performs no source write.
  await until("both frames navigated", () => {
    const frames = [...d.win.document.querySelectorAll(".pane iframe")];
    return frames.length === 2 && frames.every((f) => (f.getAttribute("src") ?? "").includes("/auth/login"));
  });
  d.loadFrames();
  await sleep(300);
  check(
    "loading the burst route performs no source edit",
    editPosts(d.posts).length === 0,
    JSON.stringify(d.posts.map((p) => p.path)),
  );
  d.win.document.getElementById("burst").click();
  await until("the burst started", () => d.posts.some((p) => p.path === "/api/burst"), 15000, () => ({
    posts: d.posts.map((p) => p.path),
    rows: d.rows().map((r) => r.map((c) => c.text)),
  }));

  // Five edits at the server's cadence; the source only ever holds one token, so a
  // side still building edit 2 when edit 3 arrives never displays edit 2 at all.
  const COUNT = 5;
  for (let i = 0; i < COUNT; i++) {
    d.push({
      type: "edit",
      kind: "island",
      token: `burst-${i + 1}`,
      n: i + 1,
      warmup: false,
      url: "/auth/login",
      burst: { index: i, count: COUNT },
    });
    await sleep(30);
  }
  // Both sides only ever show the LAST state.
  d.probe(DP_ORIGIN, [`burst-${COUNT}`], "dp-doc-1");
  d.probe(TP_ORIGIN, [`burst-${COUNT}`], "tp-doc-1");
  d.push({ type: "burst-done", kind: "island", count: COUNT });
  await until("the first burst row appeared", () => findRow(d.rows(), "sustained island edit 1/5") !== undefined);

  const missed = findRow(d.rows(), "sustained island edit 1/5");
  check("the state neither side displayed is kept as a row", missed !== undefined);
  check("and reads `not shown`, not TIMEOUT", missed[1].text === "not shown" && missed[2].text === "not shown", JSON.stringify([missed[1].text, missed[2].text]));
  check(
    "the advantage column names both sides' misses",
    /neither side showed it/.test(missed[3].text),
    JSON.stringify(missed[3].text),
  );
  const summary = findRow(d.rows(), "how many of the 5 states");
  check(
    "the burst summary counts what each side actually displayed",
    summary !== undefined && summary[1].text === "1/5 shown" && summary[2].text === "1/5 shown",
    JSON.stringify(summary?.map((c) => c.text)),
  );
  check("no uncaught error in the dashboard's own script", d.scriptErrors.length === 0, d.scriptErrors.join("\n       "));
  d.retire();
}

// ---------------------------------------------------------------------------

function argValue(flag) {
  const i = process.argv.indexOf(flag);
  return i > 0 ? process.argv[i + 1] : undefined;
}

const jsdom = await loadJsdom();
console.log(`dashboard fairness rules (jsdom from ${APP})`);
await testNavigationAndEditSeparation(jsdom);
await testEarlyProbeIsRemembered(jsdom);
await testFailedBuildIsNotATimeout(jsdom);
await testMemoryRowsAreDistinguished(jsdom);
await testSupersededKeepsItsRow(jsdom);
console.log(`\n${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
