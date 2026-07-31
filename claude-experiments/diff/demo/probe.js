// The dashboard's only view inside a cross-origin frame.
//
// `demo/server.mjs` copies this file to `<app>/public/diffpack-demo-probe.js` and
// adds one `<script src>` to the app's root layout, so BOTH dev servers serve the
// identical probe. Every scenario edit plants a token the probe can see; when the
// set of visible tokens changes, the probe posts it to the parent, which stops that
// side's clock. Detection is a token-membership test, so "was it already showing
// that?" can never pass.
//
// Three things are deliberate:
//   * Tokens live in `data-dpmark` attributes (and one CSS custom property), never
//     in body text: `document.body.textContent` on the booker allocates a large
//     string, and doing that every few milliseconds would tax the page being
//     measured. `querySelectorAll` over an attribute is indexed and cheap.
//   * Not framed -> the probe returns immediately and does nothing, so leaving the
//     script tag in place cannot affect an ordinary dev session or a benchmark run.
//   * Every message carries a `doc` id unique to THIS document instance. Both
//     bundlers fall back to a full page reload when a hot update cannot be applied,
//     and a reload loses everything the user had typed — so "the badge appeared" is
//     not the whole truth about an edit. The dashboard compares the `doc` id at the
//     start of a race with the one that reported the change: same id means the
//     running page was patched in place, a new id means the document was replaced.
(function () {
  if (window.parent === window) return;

  var POLL_MS = 8;
  var last = "";
  // Unique per document instance: a reload runs this file again and gets a new one.
  // `performance.timeOrigin` alone is not enough (two loads can round to the same
  // millisecond), so a random component comes along.
  var DOC =
    String(Math.round(window.performance && performance.timeOrigin ? performance.timeOrigin : Date.now())) +
    "-" +
    Math.random().toString(36).slice(2, 10);

  // The booker opens a timezone dialog on a fresh visit, which covers the page the
  // demo is showing. The real benchmark sets the same cookie through Playwright.
  try {
    document.cookie = "calcom-timezone-dialog=1; path=/; max-age=31536000";
  } catch (e) {}

  function tokens() {
    var out = [];
    var nodes = document.querySelectorAll("[data-dpmark]");
    for (var i = 0; i < nodes.length; i++) {
      var v = nodes[i].getAttribute("data-dpmark");
      if (v) out.push(v);
    }
    // The stylesheet class of edit has no DOM footprint; its token rides on a
    // custom property, which is the only signal that proves the *compiled sheet*
    // reached the browser rather than just the file changing on disk.
    try {
      var css = getComputedStyle(document.documentElement)
        .getPropertyValue("--dpmark")
        .trim();
      if (css) out.push(css);
    } catch (e) {}
    out.sort();
    return out;
  }

  function post(kind, extra) {
    var msg = {
      source: "diffpack-demo-probe",
      kind: kind,
      path: location.pathname,
      doc: DOC,
      ts: Date.now(),
    };
    if (extra) for (var k in extra) msg[k] = extra[k];
    try {
      window.parent.postMessage(msg, "*");
    } catch (e) {}
  }

  function tick() {
    var t = tokens();
    var key = t.join("|");
    if (key !== last) {
      last = key;
      post("tokens", { tokens: t });
    }
  }

  post("hello", { tokens: [] });
  window.addEventListener("load", function () {
    post("load", { tokens: tokens() });
  });
  setInterval(tick, POLL_MS);
})();
