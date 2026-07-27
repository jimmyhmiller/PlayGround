// Harness regression tests for the observation probe:
//   node --test integration/e2e/lib/probe.test.mjs
//
// `next-strict-csp`'s only failing channel was `hydration`, and it was a
// HARNESS defect. The probe answered "did React hydrate?" by scanning
// `document.body.querySelectorAll("*")` — the body's element DESCENDANTS —
// which is a different question. That app's page renders a single `next/script
// strategy="afterInteractive"`, so the component tree contributes no DOM at
// all; React hydrated `<html>` and `<body>` on BOTH builds, but only the
// reference happened to have a fibre-bearing descendant, React's own streaming
// slot (`<div hidden>`), which `isScaffolding` excludes from every other
// channel. The suite reported a difference that did not exist.
//
// Verified in a real browser against the real builds before the widening
// landed (`agent-browser`, both servers up):
//
//   diffpack   htmlKeys ["__reactFiber$qkdwuwqiljo","__reactProps$qkdwuwqiljo"]
//              bodyKeys ["__reactFiber$qkdwuwqiljo","__reactProps$qkdwuwqiljo"]
//              descendantWithFiber []            narrow=false  wide=true
//   reference  htmlKeys ["__reactFiber$quh01hd7evq", …]
//              descendantWithFiber ["DIV"]       narrow=true   wide=true
//
// and the negative controls, both derived from that same diffpack build:
//
//   the served document with its `<script type="module" src="/client.js">`
//   removed                                       narrow=false  wide=false
//   the whole bundle shipped, client.js throwing
//   on its first statement (React loaded, never
//   hydrates)                                     narrow=false  wide=false
//
// These cases pin the rule itself, deterministically and with no browser: they
// evaluate the SHIPPED predicate source over synthetic documents.
import { test } from "node:test";
import assert from "node:assert/strict";

import { REACT_FIBER_SOURCE } from "./probe.mjs";

/** The exact source the page will run, made callable here. */
const hasReactFiber = new Function(`return (${REACT_FIBER_SOURCE});`)();

/** A synthetic element; `fiber: true` stamps the key React would attach. */
const el = (tag, { fiber = false, children = [] } = {}) => {
  const node = { tagName: tag, children };
  if (fiber) {
    node["__reactFiber$abc123"] = {};
    node["__reactProps$abc123"] = {};
  }
  return node;
};

const flatten = (nodes) => nodes.flatMap((n) => [n, ...flatten(n.children ?? [])]);

/** A synthetic document exposing only what the predicate touches. */
const doc = (documentElement, body) => ({
  documentElement,
  body: body && { ...body, querySelectorAll: () => flatten(body.children ?? []) },
});

test("a page whose React tree renders no elements is still hydrated", () => {
  // next-strict-csp: fibres on <html> and <body>, nothing below them.
  const page = doc(
    el("HTML", { fiber: true }),
    el("BODY", { fiber: true, children: [el("SCRIPT"), el("SCRIPT")] }),
  );
  assert.equal(hasReactFiber(page), true);
});

test("a genuinely unhydrated page is still reported unhydrated", () => {
  // No client bundle, or an entry that threw before hydrateRoot: React stamped
  // nothing, anywhere. Nothing but React writes these keys.
  const page = doc(
    el("HTML"),
    el("BODY", { children: [el("DIV", { children: [el("P")] }), el("SCRIPT")] }),
  );
  assert.equal(hasReactFiber(page), false);
});

test("the ordinary case — React owns nodes below the body — still passes", () => {
  const page = doc(
    el("HTML"),
    el("BODY", { children: [el("DIV", { fiber: true, children: [el("P", { fiber: true })] })] }),
  );
  assert.equal(hasReactFiber(page), true);
});

test("a fibre on <html> alone is enough, and on <body> alone is enough", () => {
  assert.equal(hasReactFiber(doc(el("HTML", { fiber: true }), el("BODY"))), true);
  assert.equal(hasReactFiber(doc(el("HTML"), el("BODY", { fiber: true }))), true);
});

test("a document with no body does not throw", () => {
  // The probe runs after settle, but a crashed navigation can leave no body;
  // the hydration channel must not take the whole run down with it.
  assert.equal(hasReactFiber({ documentElement: el("HTML"), body: null }), false);
});

test("only __reactFiber keys count, not any expando", () => {
  const impostor = el("BODY");
  impostor.__reactive$x = {};
  impostor.__vueParentComponent = {};
  assert.equal(hasReactFiber(doc(el("HTML"), impostor)), false);
});
