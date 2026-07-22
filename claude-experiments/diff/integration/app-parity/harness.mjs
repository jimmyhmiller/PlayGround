// Behavioral differential-testing harness: proves a diffpack-built app behaves
// identically to its reference Vite build by driving the SAME scenario against
// both dists in the same headless Chromium and comparing, per step:
//   1. normalized DOM        2. full computed styles (every element)
//   3. full-page screenshot  4. console + network   5. local/session storage
//
// Test-only. Never part of any build. See README.md for the normalization
// rules and the justification for each one.
import { createRequire } from "node:module";
import {
  mkdirSync,
  writeFileSync,
  rmSync,
  symlinkSync,
  existsSync,
  mkdtempSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REF_PKG = resolve(HERE, "../vite-react-reference/package.json");
const requireRef = createRequire(REF_PKG);
const puppeteer = requireRef("puppeteer-core");
const { startStaticServer } = await import(
  resolve(HERE, "../vite-react-reference/static-server.mjs")
);

export const CHROME = [
  process.env.CHROME,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1194/chrome-linux/chrome`,
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
]
  .filter(Boolean)
  .find((p) => existsSync(p));

export const VIEWPORT = { width: 1280, height: 800 };
const PIXEL_TOLERANCE = 3; // per channel, 0-255
const PIXEL_DIFF_ARTIFACT_PCT = 0.1; // write image artifacts above this
const NUM_TOLERANCE = 0.5; // px-ish numeric tolerance for computed styles
const COLOR_CHANNEL_TOLERANCE = 1; // 0-255 channels
const COLOR_ALPHA_TOLERANCE = 1 / 255 + 1e-9;

// ---------------------------------------------------------------------------
// Normalizers (each is a claim that a difference is benign — see README.md)
// ---------------------------------------------------------------------------

// [N1] Content-hashed asset names. Vite emits `name-B4x9Zk2p.ext`, diffpack
// `name-7375e725.ext` (and un-hashed `index.js`). Strip the dash-hash segment
// (8+ chars of [A-Za-z0-9_-]) before a known asset extension. The match is
// leftmost-greedy, so multi-dash names like `inter-v13-latin-700-HASH.woff2`
// collapse further (to `inter.woff2`) — aggressive but applied identically to
// both sides, and request *counts* are still compared.
const ASSET_EXT =
  "(?:worker\\.)?(?:js|css|map|json|wasm|png|svg|jpe?g|gif|webp|ico|ttf|otf|woff2?)";
const HASHED_NAME_RE = new RegExp(
  `-[A-Za-z0-9_-]{8,}(\\.${ASSET_EXT})\\b`,
  "g"
);
export function stripHashedNames(s) {
  return s.replace(HASHED_NAME_RE, "$1");
}

// [N2] The two dists are served on different ephemeral ports; the origin is
// harness noise, not app behavior.
const ORIGIN_RE = /https?:\/\/127\.0\.0\.1:\d+/g;
export function stripOrigin(s) {
  return s.replace(ORIGIN_RE, "ORIGIN");
}

// [N3] Source positions inside a bundle (file:line:col) are a property of the
// bundle layout, not of app behavior; the message identity is what we compare.
const POSITION_RE = /(ORIGIN[^\s)]*?):\d+:\d+/g;
export function stripBundlePositions(s) {
  return s.replace(POSITION_RE, "$1");
}

// [N8] v4 UUIDs are crypto-random runtime identifiers (e.g. MSW's service
// worker client id); a bundler cannot influence them and no two runs share
// them, so they are normalized to "UUID" wherever they appear in compared
// text. Caveat (documented): an app that *renders* a UUID it derived from
// seeded randomness would lose sensitivity here.
const UUID_RE =
  /\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b/g;
export function stripUuids(s) {
  return s.replace(UUID_RE, "UUID");
}

export function normalizeText(s) {
  return stripUuids(stripBundlePositions(stripOrigin(stripHashedNames(s))))
    .replace(/\s+/g, " ")
    .trim();
}

// [N9] Emitted asset *paths* are bundler packaging choices (vite:
// /assets/index-HASH.js, diffpack: /index.js). First-party request URLs are
// therefore compared by hash-stripped basename. A wrong path that fails to
// resolve still shows up in the network `failures` channel (404s), so
// base-path bugs are not masked.
// The `.worker.js` suffix is likewise a chunk-naming choice (vite emits
// `AIWorker-HASH.js`, diffpack `AIWorker-HASH.worker.js`); fold it into the
// same key.
export function requestKey(pathname) {
  const noQuery = pathname.split(/[?#]/)[0];
  const base = noQuery.split("/").filter(Boolean).pop() || "/";
  return normalizeText(base).replace(/\.worker\.js$/, ".js");
}

// ---------------------------------------------------------------------------
// In-page snapshot functions (run identically inside both pages)
// ---------------------------------------------------------------------------

// [N4] CSS-module scoped class names differ by construction between bundlers
// (vite: `_local_1lpfs_154`, diffpack: `_local_7375e725`). Any class token of
// the scoped shape is reduced to its stable local-name stem by iteratively
// stripping trailing `_<hash-or-line>` segments. Symmetric: applied to both
// sides; can only lose sensitivity (merge snake_case names), never invent a
// diff.
// [N5] class lists are compared as sets (sorted) because emission order of
// class tokens is bundler/toolchain dependent, not observable behavior.
// [N6] style="" attributes are re-serialized per-declaration and sorted, so
// declaration order and vendor serialization quirks don't count as diffs;
// values themselves are preserved.
const domSnapshotFn = () => {
  // Scoped shape: local name + one-or-more trailing `_<hash>` / `_<line>`
  // segments (vite: `_name_1gg8h_178`, diffpack: `_name_7375e725`).
  const MOD = /^_?[A-Za-z][A-Za-z0-9_-]*_(?:\d{1,4}|[A-Za-z0-9-]{5,})$/;
  const stripModuleToken = (t) => {
    if (!MOD.test(t)) return t;
    let s = t.replace(/^_/, "");
    let prev;
    do {
      prev = s;
      s = s.replace(/_(?:\d{1,4}|[A-Za-z0-9-]{5,})$/, "");
    } while (s !== prev && s.includes("_"));
    return s || t;
  };
  const clone = document.body.cloneNode(true);
  const walk = (el) => {
    if (el.nodeType !== 1) return;
    if (el.hasAttribute && el.hasAttribute("class")) {
      const tokens = (el.getAttribute("class") || "")
        .split(/\s+/)
        .filter(Boolean)
        .map(stripModuleToken)
        .sort();
      el.setAttribute("class", tokens.join(" "));
    }
    if (el.hasAttribute && el.hasAttribute("style")) {
      // Parse through the live CSSOM for canonical per-declaration form.
      const st = el.style;
      const decls = [];
      for (let i = 0; i < st.length; i++) {
        const p = st[i];
        decls.push(
          `${p}:${st.getPropertyValue(p)}${st.getPropertyPriority(p) ? " !important" : ""}`
        );
      }
      el.setAttribute("style", decls.sort().join(";"));
    }
    for (const child of el.children || []) walk(child);
  };
  walk(clone);
  // Remove the harness's own determinism <style> tag if it got cloned (it is
  // injected into <head>, not <body>, so normally absent here).
  return clone.outerHTML;
};

// Full computed styles for every element under (and including) body, keyed by
// a structural DOM path (tag + child index), which is stable after DOM
// normalization.
const computedStylesFn = () => {
  const pathOf = (el) => {
    const parts = [];
    let cur = el;
    while (cur && cur !== document.body) {
      const parent = cur.parentElement;
      if (!parent) break;
      const idx = Array.prototype.indexOf.call(parent.children, cur);
      parts.unshift(`${cur.tagName.toLowerCase()}[${idx}]`);
      cur = parent;
    }
    return "body" + (parts.length ? "/" + parts.join("/") : "");
  };
  const els = [document.body, ...document.body.querySelectorAll("*")];
  return els.map((el) => {
    const cs = getComputedStyle(el);
    const s = {};
    for (let i = 0; i < cs.length; i++) {
      const p = cs[i];
      s[p] = cs.getPropertyValue(p);
    }
    return { path: pathOf(el), s };
  });
};

const storageFn = () => ({
  local: Object.fromEntries(
    Object.keys(localStorage).map((k) => [k, localStorage.getItem(k)])
  ),
  session: Object.fromEntries(
    Object.keys(sessionStorage).map((k) => [k, sessionStorage.getItem(k)])
  ),
});

// ---------------------------------------------------------------------------
// Determinism init scripts (identical on both sides)
// ---------------------------------------------------------------------------

// Deterministic Math.random (mulberry32, fixed seed) so seeded app randomness
// aligns across the two sides. NOTE: does not reach Web Workers — worker
// randomness must be handled with `mode: "invariant"` steps.
const SEED_MATH_RANDOM = `
  (() => {
    let s = 0xC0FFEE ^ 0x9E3779B9;
    Math.random = function () {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  })();
`;

// Kill CSS animations/transitions and the text caret so screenshots compare a
// settled UI, not animation phase.
const KILL_MOTION = `
  (() => {
    const css = "*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}";
    const add = () => {
      if (!document.head) return;
      const st = document.createElement("style");
      st.setAttribute("data-parity-harness", "1");
      st.textContent = css;
      document.head.appendChild(st);
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", add);
    } else {
      add();
    }
  })();
`;

// ---------------------------------------------------------------------------
// Value comparison with tolerances
// ---------------------------------------------------------------------------

const NUM_RE = /-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/gi;

// [N10] Custom-property (--*) computed values are the *as-authored token
// text*, so CSS minifier serialization choices leak through verbatim
// (`#fff` vs `#ffffff`, `.4` vs `0.4`). These are identical CSS values by
// spec; normalize the token serialization before comparing. Real value
// changes still differ after normalization.
export function normalizeCssTokenText(s) {
  return s
    .replace(/\s+/g, " ")
    .replace(/'([^']*)'/g, '"$1"') // CSS string quote style is serializer choice
    .replace(/#([0-9a-fA-F]{3,4})\b/g, (_, h) =>
      "#" + [...h].map((c) => c + c).join("")
    )
    .replace(/(^|[\s(,:])\.(\d)/g, "$10.$2")
    .trim()
    .toLowerCase();
}
const COLOR_FN_RE = /\b(?:rgba?|oklch|oklab|lab|lch|hsla?|color)\(/g;

function colorSpans(s) {
  const spans = [];
  COLOR_FN_RE.lastIndex = 0;
  let m;
  while ((m = COLOR_FN_RE.exec(s))) {
    let depth = 0;
    let i = m.index + m[0].length - 1;
    for (; i < s.length; i++) {
      if (s[i] === "(") depth++;
      else if (s[i] === ")") {
        depth--;
        if (depth === 0) break;
      }
    }
    spans.push([m.index, i]);
  }
  return spans;
}

// Tolerant compare of two computed-style values. Structure (non-numeric
// skeleton) must match exactly; numbers compare with 0.5 tolerance, except
// inside color functions where channels (0-255) get tolerance 1 and
// alpha/normalized components (<=1) get 1/255.
export function styleValuesMatch(a, b) {
  if (a === b) return true;
  a = stripHashedNames(stripOrigin(a));
  b = stripHashedNames(stripOrigin(b));
  if (a === b) return true;
  const skelA = a.replace(NUM_RE, "#");
  const skelB = b.replace(NUM_RE, "#");
  if (skelA !== skelB) return false;
  const spans = colorSpans(a);
  const inColor = (idx) => spans.some(([s, e]) => idx >= s && idx <= e);
  NUM_RE.lastIndex = 0;
  const numsA = [...a.matchAll(NUM_RE)];
  const numsB = [...b.matchAll(NUM_RE)];
  if (numsA.length !== numsB.length) return false;
  for (let i = 0; i < numsA.length; i++) {
    const x = parseFloat(numsA[i][0]);
    const y = parseFloat(numsB[i][0]);
    const d = Math.abs(x - y);
    if (inColor(numsA[i].index)) {
      const tol =
        Math.abs(x) <= 1 && Math.abs(y) <= 1
          ? COLOR_ALPHA_TOLERANCE
          : COLOR_CHANNEL_TOLERANCE + 1e-9;
      if (d > tol) return false;
    } else if (d > NUM_TOLERANCE + 1e-9) {
      return false;
    }
  }
  return true;
}

// [N10b] Minifiers also rewrite colors between hex and rgb() forms (with
// 8-bit alpha quantization: rgb(0 0 0 / 0.045) -> #0000000b) and seconds to
// milliseconds (.15s -> 150ms). Canonicalize both forms before comparing;
// values that differ beyond quantization still DIFF.
export function canonicalizeCssValue(s) {
  let out = normalizeCssTokenText(s);
  out = out.replace(/#([0-9a-f]{6}|[0-9a-f]{8})\b/g, (_, h) => {
    const n = (i) => parseInt(h.slice(i, i + 2), 16);
    const a = h.length === 8 ? n(6) / 255 : 1;
    return `rgb(${n(0)} ${n(2)} ${n(4)} / ${a.toFixed(4)})`;
  });
  out = out.replace(
    /rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)/g,
    (_, r, g, b, a) => {
      const alpha =
        a === undefined
          ? 1
          : a.endsWith("%")
            ? parseFloat(a) / 100
            : parseFloat(a);
      return `rgb(${Math.round(r)} ${Math.round(g)} ${Math.round(b)} / ${alpha.toFixed(4)})`;
    }
  );
  out = out.replace(
    /(^|[\s(,:])((?:\d+\.?\d*|\.\d+))s\b/g,
    (_, p, n) => `${p}${parseFloat(n) * 1000}ms`
  );
  return out;
}

export function cssValuesEquivalent(a, b) {
  const ca = canonicalizeCssValue(a);
  const cb = canonicalizeCssValue(b);
  return ca === cb || styleValuesMatch(ca, cb);
}

// ---------------------------------------------------------------------------
// Channel comparators
// ---------------------------------------------------------------------------

function compareDom(refDom, diffDom) {
  const a = normalizeText(refDom);
  const b = normalizeText(diffDom);
  if (a === b) return { status: "PASS" };
  // find first divergence for the report
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return {
    status: "DIFF",
    detail: `normalized DOM differs at char ${i}`,
    refExcerpt: a.slice(Math.max(0, i - 120), i + 200),
    diffExcerpt: b.slice(Math.max(0, i - 120), i + 200),
  };
}

function compareStyles(refStyles, diffStyles) {
  const refMap = new Map(refStyles.map((e) => [e.path, e.s]));
  const diffMap = new Map(diffStyles.map((e) => [e.path, e.s]));
  const diffs = [];
  // [N11] Custom properties present on only ONE side are recorded as a note,
  // not a DIFF: unused theme variables are emitted differently by the two CSS
  // pipelines (observed: Tailwind content scanning breadth). A custom prop
  // that is actually *consumed* cannot hide here — its consumption shows up
  // in some resolved (non-custom) property, and those are all compared.
  const customOnlyRef = new Set();
  const customOnlyDiff = new Set();
  for (const [path, rs] of refMap) {
    const ds = diffMap.get(path);
    if (!ds) {
      diffs.push({ path, missing: "diffpack" });
      continue;
    }
    const props = new Set([...Object.keys(rs), ...Object.keys(ds)]);
    for (const prop of props) {
      const rv = rs[prop];
      const dv = ds[prop];
      if (prop.startsWith("--")) {
        if (rv === undefined) customOnlyDiff.add(prop);
        else if (dv === undefined) customOnlyRef.add(prop);
        else if (!cssValuesEquivalent(rv, dv)) {
          diffs.push({ path, prop, ref: rv, diffpack: dv });
        }
        continue;
      }
      if (!styleValuesMatch(rv ?? "", dv ?? "")) {
        diffs.push({ path, prop, ref: rv, diffpack: dv });
      }
    }
  }
  for (const path of diffMap.keys()) {
    if (!refMap.has(path)) diffs.push({ path, missing: "reference" });
  }
  const propsCompared = refStyles.reduce(
    (n, e) => n + Object.keys(e.s).length,
    0
  );
  const customNote =
    customOnlyRef.size || customOnlyDiff.size
      ? `custom props present on one side only (noted per [N11], not a diff): ref-only=${customOnlyRef.size} diffpack-only=${customOnlyDiff.size}`
      : "";
  const base = {
    customPropsOnlyRef: [...customOnlyRef].sort(),
    customPropsOnlyDiffpack: [...customOnlyDiff].sort(),
  };
  // Dedupe: inherited custom props repeat the same diff on every descendant;
  // group identical (prop, ref, diffpack) triples with a count + sample path.
  const grouped = new Map();
  for (const d of diffs) {
    const key = JSON.stringify([d.prop, d.ref, d.diffpack, d.missing]);
    const g = grouped.get(key);
    if (g) {
      g.elements++;
    } else {
      grouped.set(key, { ...d, samplePath: d.path, elements: 1 });
    }
  }
  const groupedDiffs = [...grouped.values()].map(({ path, ...rest }) => rest);
  return diffs.length
    ? {
        status: "DIFF",
        detail: `${diffs.length} style diffs, ${groupedDiffs.length} distinct (of ${propsCompared} props on ${refMap.size} elements)${customNote ? "; " + customNote : ""}`,
        diffs: groupedDiffs.slice(0, 200),
        ...base,
      }
    : {
        status: "PASS",
        detail: `${propsCompared} props on ${refMap.size} elements${customNote ? "; " + customNote : ""}`,
        ...base,
      };
}

function multisetDiff(aList, bList) {
  const count = (list) => {
    const m = new Map();
    for (const x of list) m.set(x, (m.get(x) || 0) + 1);
    return m;
  };
  const ca = count(aList);
  const cb = count(bList);
  const onlyRef = [];
  const onlyDiff = [];
  for (const [k, n] of ca) {
    const d = n - (cb.get(k) || 0);
    if (d > 0) onlyRef.push(`${k} (x${d})`);
  }
  for (const [k, n] of cb) {
    const d = n - (ca.get(k) || 0);
    if (d > 0) onlyDiff.push(`${k} (x${d})`);
  }
  return { onlyRef, onlyDiff };
}

// [N7] Console messages and requests are compared as multisets, not ordered
// lists: interleaving of async logs/fetches is scheduler timing, not app
// behavior. Counts and contents still must match exactly.
function compareConsole(refMsgs, diffMsgs) {
  const norm = (msgs) => msgs.map((m) => `[${m.type}] ${normalizeText(m.text)}`);
  const { onlyRef, onlyDiff } = multisetDiff(norm(refMsgs), norm(diffMsgs));
  if (!onlyRef.length && !onlyDiff.length)
    return { status: "PASS", detail: `${refMsgs.length} messages` };
  return { status: "DIFF", onlyRef, onlyDiff };
}

function compareNetwork(refNet, diffNet) {
  const { onlyRef, onlyDiff } = multisetDiff(refNet.urls, diffNet.urls);
  const failures = multisetDiff(refNet.failures, diffNet.failures);
  const dropped = multisetDiff(refNet.dropped, diffNet.dropped);
  const parts = [];
  if (onlyRef.length || onlyDiff.length)
    parts.push({ channel: "requests", onlyRef, onlyDiff });
  if (failures.onlyRef.length || failures.onlyDiff.length)
    parts.push({
      channel: "failures",
      onlyRef: failures.onlyRef,
      onlyDiff: failures.onlyDiff,
    });
  if (dropped.onlyRef.length || dropped.onlyDiff.length)
    parts.push({
      channel: "dropped-third-party-hosts",
      onlyRef: dropped.onlyRef,
      onlyDiff: dropped.onlyDiff,
    });
  if (!parts.length)
    return {
      status: "PASS",
      detail: `${refNet.urls.length} requests${
        refNet.dropped.length
          ? `, dropped third-party: ${[...new Set(refNet.dropped)].join(",")}`
          : ""
      }`,
    };
  return { status: "DIFF", parts };
}

function compareStorage(refSt, diffSt, normalizeValue) {
  const norm = (st) => {
    const out = {};
    for (const scope of ["local", "session"]) {
      out[scope] = {};
      for (const [k, v] of Object.entries(st[scope])) {
        out[scope][k] = normalizeText(normalizeValue ? normalizeValue(k, v) : v);
      }
    }
    return JSON.stringify(out, Object.keys(out).sort());
  };
  const a = JSON.stringify(sortDeep(JSON.parse(norm(refSt))));
  const b = JSON.stringify(sortDeep(JSON.parse(norm(diffSt))));
  if (a === b) return { status: "PASS" };
  return { status: "DIFF", ref: a, diffpack: b };
}

function sortDeep(o) {
  if (Array.isArray(o)) return o.map(sortDeep);
  if (o && typeof o === "object") {
    const out = {};
    for (const k of Object.keys(o).sort()) out[k] = sortDeep(o[k]);
    return out;
  }
  return o;
}

// Screenshot pixel diff, computed inside a Chromium utility page (native PNG
// decode; no extra npm deps). Per-channel tolerance; produces a diff PNG.
async function compareScreenshots(browser, refPng, diffPng) {
  const page = await browser.newPage();
  try {
    const res = await page.evaluate(
      async (a64, b64, tol) => {
        const load = (b64) =>
          new Promise((res, rej) => {
            const img = new Image();
            img.onload = () => res(img);
            img.onerror = () => rej(new Error("png decode failed"));
            img.src = "data:image/png;base64," + b64;
          });
        const [ia, ib] = await Promise.all([load(a64), load(b64)]);
        const w = Math.max(ia.width, ib.width);
        const h = Math.max(ia.height, ib.height);
        const draw = (img) => {
          const c = document.createElement("canvas");
          c.width = w;
          c.height = h;
          const ctx = c.getContext("2d", { willReadFrequently: true });
          ctx.drawImage(img, 0, 0);
          return ctx.getImageData(0, 0, w, h).data;
        };
        const da = draw(ia);
        const db = draw(ib);
        const out = new Uint8ClampedArray(w * h * 4);
        let ndiff = 0;
        for (let i = 0; i < w * h; i++) {
          const o = i * 4;
          const d = Math.max(
            Math.abs(da[o] - db[o]),
            Math.abs(da[o + 1] - db[o + 1]),
            Math.abs(da[o + 2] - db[o + 2]),
            Math.abs(da[o + 3] - db[o + 3])
          );
          if (d > tol) {
            ndiff++;
            out[o] = 255;
            out[o + 3] = 255;
          } else {
            out[o] = da[o];
            out[o + 1] = da[o + 1];
            out[o + 2] = da[o + 2];
            out[o + 3] = 40;
          }
        }
        const c = document.createElement("canvas");
        c.width = w;
        c.height = h;
        c.getContext("2d").putImageData(new ImageData(out, w, h), 0, 0);
        return {
          w,
          h,
          refDims: [ia.width, ia.height],
          diffDims: [ib.width, ib.height],
          ndiff,
          total: w * h,
          diffPng: c.toDataURL("image/png"),
        };
      },
      Buffer.from(refPng).toString("base64"),
      Buffer.from(diffPng).toString("base64"),
      PIXEL_TOLERANCE
    );
    const pct = (res.ndiff / res.total) * 100;
    const dimsMatch =
      res.refDims[0] === res.diffDims[0] && res.refDims[1] === res.diffDims[1];
    return {
      status: pct > 0 || !dimsMatch ? "DIFF" : "PASS",
      pctDiff: +pct.toFixed(4),
      dims: { ref: res.refDims, diffpack: res.diffDims },
      detail: dimsMatch
        ? `${res.ndiff}/${res.total} px differ (${pct.toFixed(4)}%)`
        : `page dimensions differ ref=${res.refDims} diffpack=${res.diffDims}; ${pct.toFixed(4)}% px differ`,
      diffPngDataUrl: res.diffPng,
    };
  } finally {
    await page.close();
  }
}

// ---------------------------------------------------------------------------
// Serving
// ---------------------------------------------------------------------------

async function serveDist(dist, base) {
  if (base === "/" || !base) {
    const server = await startStaticServer(dist, 0);
    return { server, root: null };
  }
  const root = mkdtempSync(join(tmpdir(), "app-parity-"));
  const nest = join(root, base.replace(/^\/|\/$/g, ""));
  mkdirSync(dirname(nest), { recursive: true });
  symlinkSync(resolve(dist), nest);
  const server = await startStaticServer(root, 0);
  return { server, root };
}

// ---------------------------------------------------------------------------
// Page side driver
// ---------------------------------------------------------------------------

async function openSide(browser, config, dist) {
  const { server, root } = await serveDist(dist, config.base);
  const origin = `http://127.0.0.1:${server.port}`;
  const url = origin + (config.base || "/");
  const page = await browser.newPage();
  await page.emulateMediaFeatures([
    { name: "prefers-reduced-motion", value: "reduce" },
  ]);
  await page.evaluateOnNewDocument(SEED_MATH_RANDOM);
  await page.evaluateOnNewDocument(KILL_MOTION);
  if (config.initStorage) {
    await page.evaluateOnNewDocument(
      `(() => { try { const kv = ${JSON.stringify(config.initStorage)};
          for (const [k, v] of Object.entries(kv)) if (localStorage.getItem(k) === null) localStorage.setItem(k, v);
        } catch (e) {} })();`
    );
  }
  if (config.initScript) await page.evaluateOnNewDocument(config.initScript);

  const sink = { console: [], urls: [], dropped: [], failures: [] };
  page.on("console", (m) => sink.console.push({ type: m.type(), text: m.text() }));
  page.on("pageerror", (e) => sink.console.push({ type: "pageerror", text: String(e) }));
  page.on("request", (r) => {
    const u = r.url();
    if (u.startsWith("data:") || u.startsWith("blob:")) return;
    if (u.startsWith(origin)) {
      sink.urls.push(requestKey(u.slice(origin.length)));
    } else {
      // Third-party requests are dropped from the URL multiset (timing/beacon
      // noise) but the *hostnames* are recorded and compared on both sides.
      try {
        sink.dropped.push(new URL(u).host);
      } catch {
        sink.dropped.push(u.slice(0, 64));
      }
    }
  });
  // Failures of third-party requests are environment noise (this sandbox has
  // no outbound network; beacon timing varies) — the attempted third-party
  // *hosts* are already recorded + compared via the dropped list above. Only
  // first-party failures are compared.
  page.on("requestfailed", (r) => {
    if (!r.url().startsWith(origin)) return;
    sink.failures.push(
      normalizeText(`${r.url()} -> ${r.failure()?.errorText ?? "failed"}`)
    );
  });
  page.on("response", (r) => {
    if (r.status() >= 400 && r.url().startsWith(origin))
      sink.failures.push(normalizeText(`${r.url()} -> HTTP ${r.status()}`));
  });

  const marks = { console: 0, urls: 0, dropped: 0, failures: 0 };
  const drain = () => {
    const out = {
      console: sink.console.slice(marks.console),
      urls: sink.urls.slice(marks.urls),
      dropped: sink.dropped.slice(marks.dropped),
      failures: sink.failures.slice(marks.failures),
    };
    marks.console = sink.console.length;
    marks.urls = sink.urls.length;
    marks.dropped = sink.dropped.length;
    marks.failures = sink.failures.length;
    return out;
  };

  return {
    page,
    url,
    origin,
    drain,
    close: async () => {
      await page.close().catch(() => {});
      server.close();
      if (root) rmSync(root, { recursive: true, force: true });
    },
  };
}

async function settle(page, ms) {
  // rAF only fires on the foreground tab in headless Chromium; time-bound it
  // regardless so a hidden tab can never hang the run.
  await Promise.race([
    page
      .evaluate(
        () =>
          new Promise((r) =>
            requestAnimationFrame(() => requestAnimationFrame(() => r()))
          )
      )
      .catch(() => {}),
    new Promise((r) => setTimeout(r, 2000)),
  ]);
  await page.evaluate(() => document.fonts.ready.then(() => {})).catch(() => {});
  await new Promise((r) => setTimeout(r, ms));
}

async function capture(side, step) {
  const page = side.page;
  await page.bringToFront(); // background tabs throttle rAF/timers/rendering
  await settle(page, step.settle ?? 400);
  const dom = await page.evaluate(domSnapshotFn);
  const styles = await page.evaluate(computedStylesFn);
  const storage = await page.evaluate(storageFn);
  const shot = await page.screenshot({ fullPage: true });
  const probe = step.probe ? await step.probe(page) : undefined;
  const net = side.drain();
  return { dom, styles, storage, shot, probe, net };
}

// ---------------------------------------------------------------------------
// Public entry: run one app config
// ---------------------------------------------------------------------------

export async function runApp(config, { artifactsDir }) {
  if (!CHROME) throw new Error("no Chromium executable found");
  const appArtifacts = join(artifactsDir, config.name);
  rmSync(appArtifacts, { recursive: true, force: true }); // no stale artifacts
  mkdirSync(appArtifacts, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu", "--force-device-scale-factor=1"],
    defaultViewport: VIEWPORT,
  });

  const results = { app: config.name, steps: [], notes: config.notes || [] };
  let ref, diff;
  try {
    ref = await openSide(browser, config, join(config.appDir, config.refDist ?? "dist"));
    diff = await openSide(
      browser,
      config,
      join(config.appDir, config.diffDist ?? "dist-diffpack")
    );

    for (const step of config.steps) {
      const stepResult = { name: step.name, mode: step.mode || "exact" };
      const t0 = Date.now();
      try {
        await ref.page.bringToFront();
        await step.run(ref.page, { url: ref.url, origin: ref.origin });
        await diff.page.bringToFront();
        await step.run(diff.page, { url: diff.url, origin: diff.origin });
        const refSnap = await capture(ref, step);
        const diffSnap = await capture(diff, step);

        if (step.mode === "invariant") {
          if (!step.why)
            throw new Error(
              `invariant step "${step.name}" must declare WHY it is not exact`
            );
          stepResult.why = step.why;
          const checks = await step.check({
            refPage: ref.page,
            diffPage: diff.page,
            refSnap,
            diffSnap,
          });
          stepResult.checks = checks;
          stepResult.status = checks.every((c) => c.pass) ? "PASS" : "DIFF";
          // Always keep screenshots for invariant steps so the nondeterminism
          // claim can be audited.
          writeShot(appArtifacts, step.name, "ref", refSnap.shot);
          writeShot(appArtifacts, step.name, "diffpack", diffSnap.shot);
        } else {
          const channels = {};
          channels.dom = compareDom(refSnap.dom, diffSnap.dom);
          channels.styles = compareStyles(refSnap.styles, diffSnap.styles);
          channels.screenshot = await compareScreenshots(
            browser,
            refSnap.shot,
            diffSnap.shot
          );
          channels.console = compareConsole(
            refSnap.net.console,
            diffSnap.net.console
          );
          channels.network = compareNetwork(refSnap.net, diffSnap.net);
          channels.storage = compareStorage(
            refSnap.storage,
            diffSnap.storage,
            config.storageNormalize
          );
          if (step.probe) {
            const pa = JSON.stringify(refSnap.probe);
            const pb = JSON.stringify(diffSnap.probe);
            channels.probe =
              pa === pb
                ? { status: "PASS", detail: pa?.slice(0, 200) }
                : { status: "DIFF", ref: pa, diffpack: pb };
          }
          stepResult.channels = channels;
          stepResult.status = Object.values(channels).every(
            (c) => c.status === "PASS"
          )
            ? "PASS"
            : "DIFF";

          if (stepResult.status === "DIFF") {
            writeArtifacts(appArtifacts, step.name, refSnap, diffSnap, channels);
          }
          if (
            channels.screenshot.status === "DIFF" &&
            channels.screenshot.pctDiff > PIXEL_DIFF_ARTIFACT_PCT
          ) {
            writePngDataUrl(
              join(appArtifacts, `${slug(step.name)}-pixeldiff.png`),
              channels.screenshot.diffPngDataUrl
            );
          }
          // keep the data URL out of the JSON report
          delete channels.screenshot.diffPngDataUrl;
        }
      } catch (e) {
        stepResult.status = "ERROR";
        stepResult.error = String(e && e.stack ? e.stack : e);
      }
      stepResult.ms = Date.now() - t0;
      results.steps.push(stepResult);
      if (stepResult.status === "ERROR") break; // later steps depend on state
    }
  } finally {
    await ref?.close();
    await diff?.close();
    await browser.close();
  }
  return results;
}

function slug(s) {
  return s.replace(/[^A-Za-z0-9._-]+/g, "-");
}

function writeShot(dir, stepName, side, png) {
  writeFileSync(join(dir, `${slug(stepName)}-${side}.png`), Buffer.from(png));
}

function writePngDataUrl(path, dataUrl) {
  writeFileSync(
    path,
    Buffer.from(dataUrl.replace(/^data:image\/png;base64,/, ""), "base64")
  );
}

function writeArtifacts(dir, stepName, refSnap, diffSnap, channels) {
  const base = slug(stepName);
  writeShot(dir, stepName, "ref", refSnap.shot);
  writeShot(dir, stepName, "diffpack", diffSnap.shot);
  writeFileSync(join(dir, `${base}-ref.html`), normalizeText(refSnap.dom));
  writeFileSync(join(dir, `${base}-diffpack.html`), normalizeText(diffSnap.dom));
  const { screenshot, ...rest } = channels;
  writeFileSync(
    join(dir, `${base}-channels.json`),
    JSON.stringify(
      { ...rest, screenshot: { ...screenshot, diffPngDataUrl: undefined } },
      null,
      2
    )
  );
}

// ---------------------------------------------------------------------------
// Scenario helpers for app configs
// ---------------------------------------------------------------------------

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Click the first element whose trimmed textContent equals `text` (among
// leaf-ish clickables).
export async function clickText(page, text, { tags = "button, a, [role=button]" } = {}) {
  const ok = await page.evaluate(
    (text, tags) => {
      const els = [...document.querySelectorAll(tags)];
      const el = els.find((e) => e.textContent.trim() === text);
      if (!el) return false;
      el.click();
      return true;
    },
    text,
    tags
  );
  if (!ok) throw new Error(`clickText: no element with text ${JSON.stringify(text)}`);
}

export async function clickTextContains(page, text, { tags = "button, a, [role=button]" } = {}) {
  const ok = await page.evaluate(
    (text, tags) => {
      const els = [...document.querySelectorAll(tags)];
      const el = els.find((e) => e.textContent.includes(text));
      if (!el) return false;
      el.click();
      return true;
    },
    text,
    tags
  );
  if (!ok)
    throw new Error(`clickTextContains: no element containing ${JSON.stringify(text)}`);
}

export async function waitForFn(page, fn, arg, { timeout = 15000, poll = 100 } = {}) {
  const start = Date.now();
  for (;;) {
    const v = await page.evaluate(fn, arg);
    if (v) return v;
    if (Date.now() - start > timeout)
      throw new Error(`waitForFn timed out after ${timeout}ms`);
    await sleep(poll);
  }
}
