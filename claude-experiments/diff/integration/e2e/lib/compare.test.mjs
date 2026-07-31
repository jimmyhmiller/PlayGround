// Harness regression tests for how findings are SCORED:
//   node --test integration/e2e/lib/compare.test.mjs
//
// FINDINGS #32. The suite has three severities: `fail` (an observed
// difference), `info` (a difference in diffpack's favour — an error only the
// reference produces) and `error` (the comparison could not be made at all;
// today only "probe missing", i.e. one side never produced a record for the
// route). `run.mjs` counted `fail` alone, everywhere: the per-route line, the
// printed differences, the pass/differs verdict, the summary table's failing
// count, and its channel list. So an app whose routes diffpack could not serve
// scored a clean `pass`.
//
// That is not hypothetical. `results/next-i18n-routing/probe-_en_US-diffpack.json`
// records `"record": null` — diffpack redirected `/en-US` to
// `http://localhost/en/en-US` (wrong host, doubled locale prefix), the document
// fetch failed, and BOTH of that app's two routes were unmeasurable. The app was
// reported as passing, and appears in FINDINGS.md's "What genuinely works".
// An unmeasurable route is the loudest thing the suite can see, not the quietest.
import { test } from "node:test";
import assert from "node:assert/strict";

import { compareRecords, isFailure } from "./compare.mjs";

/** A minimal record both sides can produce, so an equal pair yields no findings. */
const record = () => ({
  title: "t",
  lang: "en",
  bodyText: "hello",
  headings: ["h1:hello"],
  outline: "0:div",
  elementCount: 1,
  elements: [],
  links: [],
  stylesheetCount: 1,
  clickable: [],
  hydrationHints: { hasReactFiber: true },
});

test("a route only one side could probe is scored as a failure", () => {
  const findings = compareRecords(record(), null, { label: "/en-US" });
  assert.equal(findings.length, 1);
  assert.equal(findings[0].channel, "probe");
  assert.equal(findings[0].summary, "/en-US: probe missing on diffpack");
  assert.equal(
    isFailure(findings[0]),
    true,
    "an unmeasurable route must count against the app, not be filed as a footnote"
  );
});

test("a missing REFERENCE record is a failure too — the oracle is gone", () => {
  const findings = compareRecords(null, record(), { label: "/" });
  assert.equal(findings[0].summary, "/: probe missing on reference");
  assert.equal(isFailure(findings[0]), true);
});

test("an error only the reference produces stays informational", () => {
  // `info` is the one severity that is recorded and not charged: diffpack is
  // doing BETTER than the reference there, and charging it would invert the
  // suite's meaning.
  assert.equal(isFailure({ severity: "info", channel: "errors" }), false);
  assert.equal(isFailure({ severity: "fail", channel: "text" }), true);
});

test("two identical records still produce no findings at all", () => {
  assert.deepEqual(compareRecords(record(), record(), { label: "/" }), []);
});
