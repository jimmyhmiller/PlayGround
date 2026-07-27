// Regression test for "flight scripts are injected mid-HTML-token".
//
// react-dom fills a 2048-byte view and calls `destination.write()` when it is full —
// that boundary lands anywhere, including inside an attribute value. Anything written
// to `res` between two of React's own writes therefore corrupts the document. This
// renders a tree deliberately built to cross a view boundary inside `src="..."` while
// inline `__DF_FLIGHT` scripts are queued, and asserts the SHIPPED destination
// (src/next_runtime/flight_sink.js) never splits a tag — plus a control proving the old
// PassThrough + `for await` shape does.
//
// Run it from integration/next-app-router/ so react + react-dom resolve to the pinned
// copies:  node ../../scripts/rsc/ssr-stream-integrity.mjs
import { PassThrough } from "node:stream";
import { createRequire } from "node:module";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { scriptsInsideTags } from "./html-integrity.mjs";

// react/react-dom come from the APP the test is run in (the pinned copies), not from
// this script's own directory — the script lives outside any node_modules tree.
const appRequire = createRequire(join(process.cwd(), "package.json"));
const load = (id) => import(pathToFileURL(appRequire.resolve(id)).href);
const { createElement, Suspense } = await load("react");
const { renderToPipeableStream } = await load("react-dom/server");
const { createFlightSink } = await import(
  new URL("../../src/next_runtime/flight_sink.js", import.meta.url).href
);

let failures = 0;
function check(ok, message) {
  console.log(`${ok ? "OK" : "FAIL"}: ${message}`);
  if (!ok) failures++;
}

// A collecting stand-in for node's ServerResponse.
function fakeRes() {
  const parts = [];
  let ended = false;
  return {
    parts,
    write(chunk) {
      if (ended) throw new Error("ssr-stream-integrity: res.write() after res.end()");
      parts.push(Buffer.from(chunk));
      return true;
    },
    end() {
      ended = true;
    },
    get html() {
      return Buffer.concat(parts).toString("utf8");
    },
  };
}

// Wide enough that React's 2048-byte view boundaries fall inside `src="/vercel-N.svg"`,
// with staggered Suspense boundaries so several flush cycles happen (as in a real app,
// where the flight chunks that resolve them keep arriving).
const BOUNDARIES = 3;
function slowBoundary(index) {
  let resolved = false;
  const later = {
    then(cb) {
      if (resolved) return cb();
      setTimeout(() => {
        resolved = true;
        cb();
      }, 20 * (index + 1));
    },
  };
  const Slow = () => {
    if (!resolved) throw later;
    return createElement("p", null, `boundary ${index}`);
  };
  return createElement(
    Suspense,
    { key: `b${index}`, fallback: createElement("p", null, "loading") },
    createElement(Slow),
  );
}

function tree() {
  const children = [];
  for (let i = 0; i < 400; i++) {
    children.push(
      createElement("img", { key: i, src: `/vercel-${i}.svg`, alt: `Vercel logomark ${i}` }),
    );
  }
  for (let i = 0; i < BOUNDARIES; i++) children.push(slowBoundary(i));
  return createElement("div", null, ...children);
}

// The flight chunks that race React's HTML. Some are already queued when React starts
// writing (the shell resolves from the first flight rows, exactly as in the real app);
// the rest arrive over the same window as the boundaries.
function startPump(scriptQueue, notify) {
  const script = (n) =>
    `<script>(self.__DF_FLIGHT=self.__DF_FLIGHT||[]).push([1,"chunk-${n}"])</script>`;
  for (let n = 0; n < 5; n++) scriptQueue.push(script(n));
  return new Promise((resolve) => {
    let n = 5;
    const tick = setInterval(() => {
      scriptQueue.push(script(n));
      notify();
      if (++n === 60) {
        clearInterval(tick);
        scriptQueue.push('<script>(self.__DF_FLIGHT=self.__DF_FLIGHT||[]).push([0])</script>');
        notify();
        resolve();
      }
    }, 1);
  });
}

// --- the shipped destination -------------------------------------------------------
async function renderWithSink() {
  const res = fakeRes();
  const scriptQueue = [];
  let sink = null;
  let flushes = 0;
  const pump = startPump(scriptQueue, () => sink && sink.scheduleDrain());
  await new Promise((resolve, reject) => {
    sink = createFlightSink({
      res,
      scriptQueue,
      renderInserted: () => [],
      onFirstWrite: () => {},
      beforeEnd: () => pump,
    });
    const realFlush = sink.flush;
    sink.flush = () => {
      flushes++;
      realFlush();
    };
    sink.on("finish", resolve);
    sink.on("error", reject);
    const { pipe } = renderToPipeableStream(tree(), {
      onShellReady() {
        pipe(sink);
      },
      onShellError: reject,
    });
  });
  return { html: res.html, flushes };
}

// --- the OLD shape, as a control (must be able to fail) -----------------------------
async function renderWithPassThrough() {
  const res = fakeRes();
  const scriptQueue = [];
  const pump = startPump(scriptQueue, () => {});
  await new Promise((resolve, reject) => {
    const html = new PassThrough();
    const { pipe } = renderToPipeableStream(tree(), {
      onShellReady() {
        pipe(html);
      },
      onShellError: reject,
    });
    (async () => {
      try {
        for await (const chunk of html) {
          res.write(chunk);
          while (scriptQueue.length) res.write(scriptQueue.shift());
        }
        await pump;
        while (scriptQueue.length) res.write(scriptQueue.shift());
        res.end();
        resolve();
      } catch (error) {
        reject(error);
      }
    })();
  });
  return res.html;
}

const control = await renderWithPassThrough();
check(
  scriptsInsideTags(control).length > 0,
  `control: the old PassThrough + for-await shape splits a tag (${scriptsInsideTags(control).length} hit(s)) — the test can fail`,
);

const { html, flushes } = await renderWithSink();
const bad = scriptsInsideTags(html);
if (bad.length) console.log(bad.slice(0, 3).map((s) => JSON.stringify(s)).join("\n"));
check(bad.length === 0, "createFlightSink never writes a <script> inside an open tag");
check(flushes > 1, `react-dom drove more than one flush cycle (${flushes}) — the boundary hook is live`);

const first = html.indexOf("__DF_FLIGHT=self.__DF_FLIGHT||[]).push([1,");
check(first >= 0, "the inline flight scripts were written at all");
// The fix must not degrade into "dump everything once the document is done": scripts
// queued early have to reach the client before the last boundary's HTML does.
const lastBoundary = html.indexOf(`boundary ${BOUNDARIES - 1}`);
check(
  first >= 0 && lastBoundary >= 0 && first < lastBoundary,
  `the flight stays interleaved (first script at ${first}, last boundary at ${lastBoundary}, of ${html.length})`,
);
for (let i = 0; i < BOUNDARIES; i++) {
  check(html.includes(`boundary ${i}`), `the late Suspense boundary ${i} still streamed in`);
}

if (failures) {
  console.error(`ssr-stream-integrity: ${failures} check(s) failed`);
  process.exit(1);
}
console.log("ssr-stream-integrity: all checks passed");
