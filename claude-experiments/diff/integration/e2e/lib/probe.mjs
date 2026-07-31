// The in-page observation probe and the determinism init script.
//
// Both sides of a comparison run byte-identical JavaScript against their own
// build, so any difference in the returned record is a difference in what the
// two bundlers produced — not in how they were measured.

/** Injected before any page script runs, on both sides, to remove sources of drift. */
export const DETERMINISM_INIT = `
(() => {
  let seed = 0x9e3779b9;
  const next = () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  Math.random = next;
  if (globalThis.crypto) {
    try {
      globalThis.crypto.randomUUID = () => {
        const hex = "0123456789abcdef";
        let out = "";
        for (let i = 0; i < 32; i++) out += hex[Math.floor(next() * 16)];
        return out.slice(0, 8) + "-" + out.slice(8, 12) + "-4" + out.slice(13, 16) +
               "-a" + out.slice(17, 20) + "-" + out.slice(20, 32);
      };
      const fill = globalThis.crypto.getRandomValues?.bind(globalThis.crypto);
      if (fill) {
        globalThis.crypto.getRandomValues = (array) => {
          for (let i = 0; i < array.length; i++) array[i] = Math.floor(next() * 256);
          return array;
        };
      }
    } catch {}
  }
  // A fixed wall clock keeps date rendering stable. performance.now() is left
  // alone: React's scheduler depends on it actually advancing.
  const FIXED = Date.parse("2025-06-15T12:00:00.000Z");
  const RealDate = Date;
  const PatchedDate = new Proxy(RealDate, {
    construct: (target, args) => (args.length ? new target(...args) : new target(FIXED)),
    apply: () => new RealDate(FIXED).toString(),
  });
  PatchedDate.now = () => FIXED;
  globalThis.Date = PatchedDate;
  // CSS animations are sampled MID-FLIGHT by the style channel, so an animated
  // element's \`transform\` drifts every run on BOTH sides — a false difference
  // that no bundler change can remove (next-with-redux's logo runs
  // \`animation: logo-float infinite 3s\`). Pausing at a FIXED negative delay puts
  // both deployments at the same phase instead of two arbitrary ones.
  //
  // This does not weaken the channel: an animation diffpack failed to emit at
  // all still differs here, because the reference sits at the 1ms phase while
  // the unstyled side sits at the identity transform. JS-driven animation
  // (framer-motion) is untouched — it does not go through CSS animation at all.
  const freezeAnimations = () => {
    const style = document.createElement("style");
    style.setAttribute("data-df-scaffold", "");
    style.textContent =
      "*, *::before, *::after {" +
      "animation-play-state: paused !important;" +
      "animation-delay: -1ms !important;" +
      "transition: none !important;" +
      "}";
    (document.head || document.documentElement).appendChild(style);
  };
  if (document.documentElement) freezeAnimations();
  else document.addEventListener("readystatechange", freezeAnimations, { once: true });
  globalThis.__DP_E2E__ = { seeded: true };
})();
`;

/**
 * Source of the "did React take this document over?" predicate, as a function
 * expression taking a `document`. Spliced into `PROBE_SOURCE` below and
 * exercised directly by `probe.test.mjs` — the rule is small and it decides a
 * whole channel, so it is a unit rather than a fragment buried in a string.
 *
 * React stamps `__reactFiber$<n>` on every host node it OWNS, including the
 * container it hydrated. Looking only at `document.body`'s element DESCENDANTS
 * therefore answers "did the app render any elements into the body?", not "did
 * React hydrate?" — and those are different questions for a page whose
 * component tree renders no elements. `next-strict-csp`'s page is exactly that:
 * it renders a single `next/script` with `strategy="afterInteractive"`, which
 * contributes no DOM. React hydrated `<html>` and `<body>` on both builds
 * (verified in a real browser: both carry `__reactFiber$…`/`__reactProps$…`),
 * yet the descendant-only check called the diffpack build unhydrated, because
 * the only fibre-bearing descendant on the reference side was React's own
 * streaming slot (`<div hidden>`) — a node `isScaffolding`, in the probe below,
 * deliberately excludes from every other channel.
 *
 * Widening it to the two root elements does not make an unhydrated page look
 * hydrated: nothing stamps those keys but React, so a document that ships no
 * client bundle, or whose entry throws before `hydrateRoot`, still answers
 * false. Both were checked in a real browser against this very build.
 */
export const REACT_FIBER_SOURCE = String.raw`
(function hasReactFiber(doc) {
  const owns = (el) => Boolean(el) && Object.keys(el).some((k) => k.startsWith("__reactFiber"));
  if (owns(doc.documentElement) || owns(doc.body)) return true;
  if (!doc.body) return false;
  return [...doc.body.querySelectorAll("*")].slice(0, 200).some(owns);
})
`;

/**
 * Runs in the page and returns the full observation record. Everything that a
 * user could perceive — text, structure, layout, applied styles, image
 * loading, form state — and nothing that is inherently per-bundler (chunk
 * names, module ids, content hashes).
 */
export const PROBE_SOURCE = String.raw`
(() => {
  const STYLE_PROPS = [
    "display","position","visibility","opacity","overflow","z-index",
    "color","background-color","background-image",
    "font-family","font-size","font-weight","font-style","line-height","letter-spacing",
    "text-align","text-decoration-line","text-transform","white-space",
    "margin-top","margin-right","margin-bottom","margin-left",
    "padding-top","padding-right","padding-bottom","padding-left",
    "border-top-width","border-top-style","border-top-color","border-radius",
    "width","height","max-width","min-height",
    "flex-direction","flex-wrap","justify-content","align-items","gap",
    "grid-template-columns","box-shadow","transform","object-fit","cursor",
  ];
  const ATTRS = ["href","src","alt","title","type","name","value","placeholder","role",
                 "aria-label","aria-current","aria-expanded","disabled","checked","hidden",
                 "target","rel","width","height","lang","dir","data-testid"];
  const SKIP_TAGS = new Set(["SCRIPT","STYLE","LINK","META","TEMPLATE","NOSCRIPT","TITLE","BASE"]);

  const collapse = (s) => (s || "").replace(/\s+/g, " ").trim();

  // Bundler-specific URL shapes are normalized to what a user actually
  // depends on: same origin or not, the path without content hashes, no query.
  const normUrl = (raw) => {
    if (!raw) return raw;
    if (/^(data|blob|javascript|mailto|tel):/i.test(raw)) return raw.split(",")[0] + ",…";
    let u;
    try { u = new URL(raw, location.href); } catch { return raw; }
    const external = u.origin !== location.origin;
    // Content hashes are dropped entirely rather than normalized to a token:
    // the two bundlers use different hash alphabets and lengths (Vite emits
    // base64url, diffpack hex), so keeping any residue would report a
    // difference where a user perceives none. "hero-CLDdwZDr.png" and
    // "hero.9f2c1a77b3.png" both become "hero.png".
    let path = u.pathname
      .replace(/\/_next\/static\/[^/]+\//g, "/_next/static/*/")
      .replace(/[-.][A-Za-z0-9_-]{8,}(\.[A-Za-z0-9]+)$/, "$1");
    return (external ? u.origin : "") + path + (u.search ? "?*" : "");
  };

  // Framework scaffolding that legitimately differs (React streaming slots,
  // Next's route announcer / dev overlays) is not part of the app's UI.
  const isScaffolding = (el) => {
    const tag = el.tagName;
    if (tag === "NEXT-ROUTE-ANNOUNCER" || tag === "NEXTJS-PORTAL") return true;
    const id = el.id || "";
    if (el.hasAttribute("hidden") && /^[SBPL]:/.test(id)) return true;
    if (id === "__next-build-watcher" || id.startsWith("__NEXT_DATA")) return true;
    if (el.hasAttribute("data-nextjs-dialog-overlay")) return true;
    if (el.hasAttribute("data-df-scaffold")) return true;
    return false;
  };

  const ownText = (el) => {
    let out = "";
    for (const node of el.childNodes) if (node.nodeType === 3) out += node.nodeValue;
    return collapse(out);
  };

  const elements = [];
  const walk = (el, depth) => {
    for (const child of el.children) {
      if (SKIP_TAGS.has(child.tagName)) continue;
      if (isScaffolding(child)) continue;
      const cs = getComputedStyle(child);
      if (cs.display === "none") { walk(child, depth + 1); continue; }
      const rect = child.getBoundingClientRect();
      const styles = {};
      for (const p of STYLE_PROPS) styles[p] = cs.getPropertyValue(p);
      if (styles["background-image"] && styles["background-image"] !== "none") {
        styles["background-image"] = styles["background-image"].replace(
          /url\(["']?([^"')]+)["']?\)/g,
          (_, u) => 'url(' + normUrl(u) + ')'
        );
      }
      const attrs = {};
      for (const a of ATTRS) {
        if (!child.hasAttribute(a)) continue;
        const v = child.getAttribute(a);
        attrs[a] = (a === "href" || a === "src") ? normUrl(v) : v;
      }
      const record = {
        tag: child.tagName.toLowerCase(),
        depth,
        text: ownText(child),
        classCount: child.classList.length,
        attrs,
        styles,
        box: { w: Math.round(rect.width), h: Math.round(rect.height) },
      };
      if (child.tagName === "IMG") {
        record.image = {
          complete: child.complete,
          loaded: child.complete && child.naturalWidth > 0,
          natural: child.naturalWidth > 0 ? "nonzero" : "zero",
          alt: child.alt,
        };
      }
      if (child.tagName === "CANVAS") record.canvasSize = child.width + "x" + child.height;
      if (child.tagName === "INPUT" || child.tagName === "TEXTAREA" || child.tagName === "SELECT") {
        record.field = { value: child.value, checked: child.checked ?? null, type: child.type };
      }
      elements.push(record);
      walk(child, depth + 1);
    }
  };
  walk(document.body, 0);

  const clickable = [];
  const clickableSelector = "button, [role='button'], summary, input[type='checkbox'], input[type='radio'], input[type='submit']";
  for (const el of document.querySelectorAll(clickableSelector)) {
    if (isScaffolding(el)) continue;
    const cs = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    if (cs.display === "none" || cs.visibility === "hidden") continue;
    if (rect.width === 0 && rect.height === 0) continue;
    if (el.disabled) continue;
    clickable.push({ tag: el.tagName.toLowerCase(), label: collapse(el.textContent) || el.getAttribute("aria-label") || el.value || "" });
  }

  const links = [];
  for (const a of document.querySelectorAll("a[href]")) {
    if (isScaffolding(a)) continue;
    const raw = a.getAttribute("href");
    if (!raw || raw.startsWith("#") || /^(mailto|tel|javascript):/i.test(raw)) continue;
    let u; try { u = new URL(raw, location.href); } catch { continue; }
    links.push({ href: normUrl(raw), internal: u.origin === location.origin, path: u.origin === location.origin ? u.pathname : null, text: collapse(a.textContent) });
  }

  const stylesheetCount = document.querySelectorAll('link[rel="stylesheet"], style').length;

  return JSON.stringify({
    location: { pathname: location.pathname, search: location.search, hash: location.hash },
    title: document.title,
    lang: document.documentElement.lang,
    bodyText: collapse(document.body.innerText),
    headings: [...document.querySelectorAll("h1,h2,h3,h4,h5,h6")].map((h) => h.tagName + ":" + collapse(h.textContent)),
    elementCount: elements.length,
    tagOutline: elements.map((e) => e.depth + ":" + e.tag).join("|"),
    elements,
    clickable,
    links,
    stylesheetCount,
    hydrationHints: {
      reactRoot: Boolean(document.querySelector("#__next, #root, [data-reactroot]")) ||
                 Boolean(Object.keys(document.body).some((k) => k.startsWith("__react"))),
      hasReactFiber: (${REACT_FIBER_SOURCE})(document),
    },
  });
})()
`;

/** Waits until the page has settled: loaded, two frames painted, network quiet. */
export const SETTLE_SOURCE = String.raw`
(async () => {
  const deadline = Date.now() + 20000;
  while (document.readyState !== "complete" && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  // Give hydration + any first-paint effects a bounded window to finish.
  await new Promise((r) => setTimeout(r, 350));
  if (document.fonts && document.fonts.ready) { try { await document.fonts.ready; } catch {} }
  await new Promise((r) => requestAnimationFrame(r));
  return JSON.stringify({ readyState: document.readyState });
})()
`;

/** Clicks the Nth element of the same clickable set the probe reports. */
export const clickSource = (index) => String.raw`
(async () => {
  const clickableSelector = "button, [role='button'], summary, input[type='checkbox'], input[type='radio'], input[type='submit']";
  const isScaffolding = (el) => el.tagName === "NEXT-ROUTE-ANNOUNCER" || el.tagName === "NEXTJS-PORTAL" ||
    (el.hasAttribute("hidden") && /^[SBPL]:/.test(el.id || "")) || el.id === "__next-build-watcher";
  const list = [...document.querySelectorAll(clickableSelector)].filter((el) => {
    if (isScaffolding(el)) return false;
    const cs = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    if (cs.display === "none" || cs.visibility === "hidden") return false;
    if (rect.width === 0 && rect.height === 0) return false;
    return !el.disabled;
  });
  const el = list[${index}];
  if (!el) return JSON.stringify({ clicked: false, reason: "no such clickable", count: list.length });
  const label = (el.textContent || el.getAttribute("aria-label") || el.value || "").replace(/\s+/g, " ").trim();
  el.scrollIntoView({ block: "center" });
  el.click();
  await new Promise((r) => setTimeout(r, 300));
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return JSON.stringify({ clicked: true, label, count: list.length });
})()
`;

/**
 * Schedules a click on the Nth internal link and returns IMMEDIATELY.
 *
 * The click must not be awaited in-page: following a link can tear down the
 * JavaScript execution context, and an evaluation that is still suspended when
 * that happens never resolves — it hangs the driver rather than failing. The
 * caller polls `LOCATION_SOURCE` afterwards instead.
 */
export const navigateSource = (index) => String.raw`
(() => {
  const anchors = [...document.querySelectorAll("a[href]")].filter((a) => {
    const raw = a.getAttribute("href");
    if (!raw || raw.startsWith("#") || /^(mailto|tel|javascript):/i.test(raw)) return false;
    try { return new URL(raw, location.href).origin === location.origin; } catch { return false; }
  });
  const a = anchors[${index}];
  if (!a) return JSON.stringify({ scheduled: false, reason: "no such internal link", count: anchors.length });
  const from = location.pathname;
  const target = new URL(a.getAttribute("href"), location.href).pathname;
  a.scrollIntoView({ block: "center" });
  setTimeout(() => a.click(), 0);
  return JSON.stringify({ scheduled: true, from, target });
})()
`;

/** Cheap, synchronous, navigation-safe: where is the page now? */
export const LOCATION_SOURCE = `JSON.stringify({ pathname: location.pathname, readyState: document.readyState })`;
