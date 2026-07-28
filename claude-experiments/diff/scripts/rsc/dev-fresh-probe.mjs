// The freshness probe behind `next-dev-fresh-check.sh`.
//
// Edits one string in a source file and then polls a FRESH full document load until
// the new string appears in the server-rendered HTML — the assertion that a test on
// "the HMR push happened" cannot make. Runs both edit classes (a "use client" island,
// which the SSR-of-flight graph renders, and a Server Component, which the
// react-server graph renders) because they take different branches in the dev
// server's watch loop and used to fail differently: the island edit went stale
// forever, the server edit "recovered" by crashing the react-server worker.
//
// Also asserts the dev server logged no error while doing it, so freshness bought by
// a crash-and-respawn does not pass.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [, , portArg, fixture, logPath] = process.argv;
const port = Number(portArg);
if (!port || !fixture || !logPath) {
  throw new Error("usage: dev-fresh-probe.mjs <port> <fixture-dir> <dev-log-path>");
}
const base = `http://127.0.0.1:${port}`;
// Polling budget per edit. Generous: this gate is a correctness assertion, not a
// latency one (the numbers live in the dev-hmr bench), and CI machines are slow.
const BUDGET_MS = Number(process.env.DEV_FRESH_BUDGET_MS || 60_000);

// Errors the fixture is EXPECTED to log, so the "no new error" assertion below stays
// sharp instead of being disabled by known noise. Anything else counts.
const KNOWN_LOG_NOISE = [/DeprecationWarning/, /ExperimentalWarning/, /^\s*\(Use `node /];

function serverErrors() {
  const text = readFileSync(logPath, "utf8");
  return text
    .split("\n")
    .filter((line) => /error|Error|not a function|is not loaded|onError/.test(line))
    .filter((line) => !KNOWN_LOG_NOISE.some((re) => re.test(line)));
}

async function documentText(path) {
  const res = await fetch(base + path, { headers: { "cache-control": "no-cache" }, cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} -> HTTP ${res.status}`);
  return await res.text();
}

/// Rewrite `find` to `token` in `file`, then poll a fresh document until it shows up.
async function editAndProbe(label, file, find, path) {
  const token = `DEV-FRESH-${label}-${Date.now()}`;
  const before = readFileSync(file, "utf8");
  if (!before.includes(find)) throw new Error(`${file} does not contain ${JSON.stringify(find)}`);
  const errorsBefore = serverErrors().length;

  const started = performance.now();
  writeFileSync(file, before.split(find).join(token));
  let html = "";
  let elapsed = -1;
  while (performance.now() - started < BUDGET_MS) {
    html = await documentText(path);
    if (html.includes(token)) {
      elapsed = performance.now() - started;
      break;
    }
  }
  if (elapsed < 0) {
    throw new Error(
      `${label}: after editing ${file}, a freshly fetched ${path} STILL did not contain ${token} after ${BUDGET_MS}ms — the dev server is serving stale server-rendered HTML`,
    );
  }
  console.log(`OK: ${label} edit reached a freshly fetched ${path} in ${Math.round(elapsed)}ms`);

  // The edit must land through a hot update, not through a crashed-and-respawned
  // process. Give the server a beat to flush anything it was going to log.
  await new Promise((resolve) => setTimeout(resolve, 500));
  const errorsAfter = serverErrors();
  if (errorsAfter.length > errorsBefore) {
    throw new Error(
      `${label}: the dev server logged ${errorsAfter.length - errorsBefore} new error line(s) while applying the edit:\n  ${errorsAfter
        .slice(errorsBefore)
        .join("\n  ")}`,
    );
  }
  // Restore, and wait for the ORIGINAL text to come back, so the next probe starts
  // from a settled server rather than racing this rebuild.
  writeFileSync(file, before);
  const restoreStarted = performance.now();
  while (performance.now() - restoreStarted < BUDGET_MS) {
    if ((await documentText(path)).includes(find)) return;
  }
  throw new Error(`${label}: restoring ${file} did not come back through a fresh ${path}`);
}

// A "use client" island: its text is rendered into the document by the SSR-of-flight
// graph, so a stale SSR bundle shows up here and nowhere else.
await editAndProbe("island", join(fixture, "app/Counter.tsx"), "count: ", "/");
// A Server Component: its text is rendered into the flight by the react-server graph.
await editAndProbe("server-component", join(fixture, "app/page.tsx"), "from-server", "/");
console.log("OK: both edit classes reached a freshly fetched document with no new server error");
