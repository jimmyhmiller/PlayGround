// react-dom's ready callbacks are NOT fire-once — the executable proof, and the reason
// diffpack's generated SSR entries guard `pipe` behind a once-flag.
//
// The bug this locks: `renderFlightToDocument` piped straight out of `onAllReady`, and
// cal.com logged
//
//     next-ssr onError: React currently only supports piping to one writable stream.
//
// about once per request, for months. Every content assertion kept passing, because
// React classifies the throw as a RECOVERABLE error: the second `pipe` throws inside
// `completeAll`, which runs inside the enclosing task's try/catch, so it comes back out
// as "this task errored" — the document still arrives, and the only trace is one log
// line (plus, had the destination not already closed, an already-completed boundary
// marked client-rendered, which discards its SSR markup on the client).
//
// The trigger, from the captured stack: when the last work to finish is a Suspense
// boundary that STILL HOLDS abortable fallback tasks, `finishedTask` decrements
// `allPendingTasks` first, then aborts those fallback tasks — and each abort re-enters
// `finishedTask`, whose tail sees the counter already at 0 and calls `completeAll`.
// The outer frame's tail then calls it AGAIN. A boundary holds a pending fallback task
// exactly when the FALLBACK ITSELF suspends, which is the tree below: fallback resolves
// late, content resolves early.
//
// Two tests, and the first one matters as much as the second: the control proves this
// tree really does reproduce the upstream double-call against the react-dom the fixture
// ships. If React ever fixes it, the control fails and someone re-reads the guard
// rather than cargo-culting it forever.
//
// The react-dom under test is the one diffpack's entries actually bind to: NOT the app's
// own dependency, but the copy Next vendors (see rsc_runtime_resolve's `react_aliases` —
// Next aliases react/react-dom onto its vendored copies in every layer, and so does
// diffpack).
import { test, before } from "node:test";
import assert from "node:assert/strict";
import { Writable } from "node:stream";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const repo = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const fixture = process.env.DIFFPACK_NEXT_FIXTURE || join(repo, "integration", "next-app-router");
const vendored = join(fixture, "node_modules", "next", "dist", "compiled");

let React;
let renderToPipeableStream;

before(async (t) => {
  if (!existsSync(join(vendored, "react-dom", "server.node.js"))) {
    // Same rule as the shell gates: an uninstalled fixture skips, it does not fake a pass.
    t.skip(`no vendored react-dom under ${vendored} (npm install in ${fixture})`);
    return;
  }
  const require = createRequire(join(vendored, "react-dom", "package.json"));
  React = require(join(vendored, "react", "index.js"));
  ({ renderToPipeableStream } = require(join(vendored, "react-dom", "server.node.js")));
});

/// A component that throws a promise until `ms` has elapsed — one suspend, one resolve.
function suspender(label, ms) {
  let resolved = false;
  let pending = null;
  return function Suspending() {
    if (resolved) return React.createElement("p", null, label);
    if (!pending) {
      pending = new Promise((resolve) =>
        setTimeout(() => {
          resolved = true;
          resolve();
        }, ms),
      );
    }
    throw pending;
  };
}

/// One Suspense boundary whose FALLBACK suspends for longer than its content, so the
/// content finishes last while the boundary still holds a pending fallback task.
function doubleReadyTree() {
  return React.createElement(
    "div",
    null,
    React.createElement(
      React.Suspense,
      { fallback: React.createElement(suspender("fallback", 120), null) },
      React.createElement(suspender("content", 5), null),
    ),
  );
}

/// Render `tree`, invoking `onAllReady` through `handler`, and report what happened.
function render(tree, handler) {
  return new Promise((resolve, reject) => {
    const parts = [];
    const errors = [];
    let readyCalls = 0;
    const sink = new Writable({
      write(chunk, _encoding, callback) {
        parts.push(Buffer.from(chunk));
        callback();
      },
    });
    sink.on("finish", () =>
      resolve({ html: Buffer.concat(parts).toString("utf8"), readyCalls, errors }),
    );
    sink.on("error", reject);
    const { pipe } = renderToPipeableStream(tree, {
      onAllReady() {
        readyCalls++;
        handler(pipe, sink);
      },
      onShellError: reject,
      onError(error) {
        errors.push(error && error.message ? error.message : String(error));
      },
    });
  });
}

test("react-dom calls onAllReady more than once for a boundary whose fallback suspends", async (t) => {
  if (!React) return t.skip("fixture not installed");
  // The OLD shape, kept as a control: pipe unconditionally, exactly as the generated
  // entry did. It must still reproduce, message and all.
  const { readyCalls, errors, html } = await render(doubleReadyTree(), (pipe, sink) =>
    pipe(sink),
  );
  assert.ok(
    readyCalls > 1,
    `expected react-dom to call onAllReady more than once for this tree; got ${readyCalls}. ` +
      "If upstream fixed the double call, re-read the once-guard in next_adapter.rs's " +
      "renderFlightToDocument/renderFlightToStream before deleting anything.",
  );
  assert.ok(
    errors.some((message) => message.includes("only supports piping to one writable stream")),
    `expected the second pipe to throw the cal.com message; got ${JSON.stringify(errors)}`,
  );
  // And the damning part: the document still arrives, so no content assertion sees this.
  assert.match(html, /<p>content<\/p>/);
});

test("the once-guard the generated entries use pipes exactly once and renders the document", async (t) => {
  if (!React) return t.skip("fixture not installed");
  // The shape asserted on in next_adapter.rs's tests: a `piped` flag, checked and set
  // before `pipe`. Same tree, same double call, no throw.
  let piped = false;
  let pipeCalls = 0;
  const { readyCalls, errors, html } = await render(doubleReadyTree(), (pipe, sink) => {
    if (piped) return;
    piped = true;
    pipeCalls++;
    pipe(sink);
  });
  assert.ok(readyCalls > 1, "the tree must still exercise the double call");
  assert.equal(pipeCalls, 1, "the guard must let exactly one pipe through");
  assert.deepEqual(errors, [], `the guarded render must log nothing; got ${JSON.stringify(errors)}`);
  assert.match(html, /<p>content<\/p>/);
});
