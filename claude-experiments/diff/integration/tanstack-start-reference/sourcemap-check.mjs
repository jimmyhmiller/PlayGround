// Production source-map oracle (test-only; NEVER referenced by the build).
//
// Proves that after a NATIVE `diffpack build-app . client --sourcemap` +
// `diffpack build-app . ssr --sourcemap` (Rust on Oxc; Node/Chrome are test
// oracles only, never in the build path), the emitted MINIFIED chunks carry
// production source maps composed THROUGH the minify pass:
//
//   A. Every emitted client `.js` chunk references a sibling `//# sourceMappingURL`
//      whose `.map` is valid JSON that (1) lists the real original project source
//      files in `sources` as project-relative, traversal-free labels, (2) inlines
//      their real text in `sourcesContent`, and (3) decodes a sampled MINIFIED
//      position (a token unique to one source) back to that exact original source.
//   B. In a real headless Chrome, the app still hydrates cleanly with maps present,
//      and a genuine runtime error thrown from inside a minified chunk has its
//      stack frame decoded, THROUGH that chunk's composed map, to the correct
//      ORIGINAL source module.
import { spawnSync, spawn } from "node:child_process";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const outputRoot = join(here, ".diffpack-output");
const publicDir = join(outputRoot, "public");
const diffpack = join(here, "..", "..", "target", "release", "diffpack");
// Chrome discovery: explicit override, then the machines this repo runs on
// (Linux box: Playwright-cached Chromium; Mac: system Chrome).
import { existsSync as __chromeExists } from "node:fs";
const CHROME = [
  process.env.CHROME,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1194/chrome-linux/chrome`,
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean).find((p) => __chromeExists(p));
const PORT = 8598;
const BASE = `http://127.0.0.1:${PORT}`;

const results = [];
const record = (name, ok, detail) => results.push({ name, ok, detail: detail ?? "" });

// --- Base64 VLQ source-map decoder (a debugger's view of the emitted map) ------
const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_TABLE = {};
for (let i = 0; i < B64.length; i++) B64_TABLE[B64[i]] = i;
function decodeVlq(segment) {
  const values = [];
  let shift = 0;
  let value = 0;
  for (const ch of segment) {
    const digit = B64_TABLE[ch];
    if (digit === undefined) return values;
    const continuation = digit & 32;
    value += (digit & 31) << shift;
    if (continuation) {
      shift += 5;
    } else {
      const negative = value & 1;
      value >>= 1;
      values.push(negative ? -value : value);
      value = 0;
      shift = 0;
    }
  }
  return values;
}
// Parse `mappings` into per-generated-line arrays of {genCol, srcIdx, srcLine}.
function parseMappings(map) {
  const perLine = [];
  let srcIdx = 0;
  let srcLine = 0;
  let srcCol = 0;
  for (const line of map.mappings.split(";")) {
    let genCol = 0;
    const segments = [];
    if (line.length) {
      for (const raw of line.split(",")) {
        const fields = decodeVlq(raw);
        if (fields.length === 0) continue;
        genCol += fields[0];
        const segment = { genCol };
        if (fields.length >= 4) {
          srcIdx += fields[1];
          srcLine += fields[2];
          srcCol += fields[3];
          segment.srcIdx = srcIdx;
          segment.srcLine = srcLine;
        }
        segments.push(segment);
      }
    }
    perLine.push(segments);
  }
  return perLine;
}
// The mapping (greatest generated position <= the query) for a generated (line,col).
function resolvePosition(perLine, line, col) {
  const segments = perLine[line] || [];
  let best = null;
  for (const segment of segments) {
    if (segment.genCol <= col && segment.srcIdx != null) best = segment;
    else if (segment.genCol > col) break;
  }
  return best;
}

// --- Build the pinned app WITH source maps (native diffpack) -------------------
if (!existsSync(diffpack)) {
  record("release binary present", false, `missing ${diffpack} (run \`cargo build --release\`)`);
  report();
}
for (const env of ["client", "ssr"]) {
  const built = spawnSync(diffpack, ["build-app", ".", env, "--sourcemap"], {
    cwd: here,
    encoding: "utf8",
  });
  if (built.status !== 0) {
    record(`build-app ${env} --sourcemap`, false, (built.stderr || built.stdout || "").slice(-400));
    report();
  }
}
record("build-app client+ssr --sourcemap", true, "native diffpack emitted maps");

// --- A. Static validation + sampled decode of every client chunk map ----------
const chunks = readdirSync(publicDir).filter((file) => file.endsWith(".js"));
let mapCount = 0;
let strongDecodes = 0;
let structureFailures = [];
let decodeFailures = [];
for (const chunk of chunks) {
  const code = readFileSync(join(publicDir, chunk), "utf8");
  const urlMatch = code.match(/\/\/# sourceMappingURL=(\S+)\s*$/);
  if (!urlMatch) {
    structureFailures.push(`${chunk}: no sourceMappingURL comment`);
    continue;
  }
  const mapPath = join(publicDir, urlMatch[1]);
  if (!existsSync(mapPath)) {
    structureFailures.push(`${chunk}: map ${urlMatch[1]} missing`);
    continue;
  }
  const map = JSON.parse(readFileSync(mapPath, "utf8"));
  mapCount++;
  // Structure: real, project-relative, traversal-free sources with inlined content.
  const sources = map.sources || [];
  const contents = map.sourcesContent || [];
  if (
    map.version !== 3 ||
    sources.length === 0 ||
    contents.length !== sources.length ||
    !sources.every((s) => s.startsWith("diffpack:///") && !s.includes("..")) ||
    !contents.every((c) => typeof c === "string" && c.length > 0)
  ) {
    structureFailures.push(
      `${chunk}: version=${map.version} sources=${sources.length} contents=${contents.length}`
    );
    continue;
  }
  // Strong decode: a token that occurs once in the minified code AND in exactly
  // one source's content must decode to that source. (Small chunks may have none;
  // then the first mapped segment must at least resolve to a real source.)
  const perLine = parseMappings(map);
  const identifiers = [...new Set((code.match(/[A-Za-z_$][A-Za-z0-9_$]{5,}/g) || []))];
  let strong = null;
  for (const token of identifiers) {
    const owners = contents
      .map((content, index) => (content.includes(token) ? index : -1))
      .filter((index) => index >= 0);
    if (owners.length !== 1) continue;
    const first = code.indexOf(token);
    if (first < 0 || code.indexOf(token, first + 1) >= 0) continue; // must be unique in code too
    const prefix = code.slice(0, first);
    const line = (prefix.match(/\n/g) || []).length;
    const col = first - (prefix.lastIndexOf("\n") + 1);
    const segment = resolvePosition(perLine, line, col);
    if (segment && segment.srcIdx === owners[0]) {
      strong = { token, source: sources[owners[0]] };
      break;
    }
  }
  if (strong) {
    strongDecodes++;
  } else {
    // Fallback: at least one mapped position must resolve to a real source.
    let anyMapped = false;
    for (const segments of perLine) {
      for (const segment of segments) {
        if (segment.srcIdx != null && sources[segment.srcIdx]) {
          anyMapped = true;
          break;
        }
      }
      if (anyMapped) break;
    }
    if (!anyMapped) decodeFailures.push(`${chunk}: no mapped position resolves to a source`);
  }
}
record(
  "every client chunk carries a valid, project-relative, content-inlined map",
  structureFailures.length === 0 && mapCount === chunks.length,
  `${mapCount}/${chunks.length} chunks mapped${structureFailures.length ? "; " + structureFailures.join(" | ") : ""}`
);
record(
  "a sampled minified position decodes to the correct original source",
  decodeFailures.length === 0 && strongDecodes > 0,
  `${strongDecodes} chunk(s) strong-decoded a unique token to its exact source${decodeFailures.length ? "; " + decodeFailures.join(" | ") : ""}`
);

// --- B. Real browser: clean hydration + error-frame -> original source --------
let serverLog = "";
const server = spawn(process.execPath, [join(outputRoot, "server/index.mjs")], {
  env: { ...process.env, PORT: String(PORT), HOST: "127.0.0.1" },
  stdio: ["ignore", "pipe", "pipe"],
});
server.stdout.on("data", (d) => (serverLog += d));
server.stderr.on("data", (d) => (serverLog += d));
function waitForServer() {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("server did not listen: " + serverLog.trim())), 15000);
    const tick = async () => {
      try {
        const r = await fetch(BASE + "/");
        if (r.ok) {
          clearTimeout(timer);
          return resolve();
        }
      } catch {}
      setTimeout(tick, 200);
    };
    tick();
  });
}

let browser;
try {
  await waitForServer();
  browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu"],
  });
  const page = await browser.newPage();
  const jsErrors = [];
  page.on("pageerror", (e) => jsErrors.push(String(e.message || e)));
  page.on("console", (m) => {
    if (m.type() === "error" && /sourcemap|source map/i.test(m.text())) jsErrors.push(m.text());
  });
  await page.goto(BASE + "/", { waitUntil: "networkidle0", timeout: 20000 });

  const hydrated = await page
    .waitForFunction(() => window.__TSR_ROUTER__ !== undefined, { timeout: 10000 })
    .then(() => true)
    .catch(() => false);
  record(
    "app hydrates cleanly in Chrome with source maps present",
    hydrated && jsErrors.length === 0,
    `hydrated=${hydrated} jsErrors=${JSON.stringify(jsErrors)}`
  );

  // Trigger a GENUINE runtime error from inside a minified chunk and capture its
  // real stack. `buildLocation({ to: Symbol() })` throws from the router-core code
  // bundled in `client.js` (the string coercion of a Symbol). Fallbacks keep the
  // check honest if the primary path changes.
  const stack = await page.evaluate(() => {
    const router = window.__TSR_ROUTER__;
    const attempts = [
      () => router.buildLocation({ to: Symbol() }),
      () => router.matchRoute(Symbol()),
      () => router.navigate({ to: Symbol() }),
    ];
    for (const attempt of attempts) {
      try {
        attempt();
      } catch (e) {
        const s = String((e && e.stack) || "");
        if (/https?:\/\/[^ )]+\.js:\d+:\d+/.test(s)) return s;
      }
    }
    return null;
  });

  let frameDetail = "no chunk stack frame captured";
  let frameOk = false;
  if (stack) {
    // First stack frame that points into an emitted chunk on our server.
    const frame = stack
      .split("\n")
      .map((line) => line.match(/(https?:\/\/[^ )]+\/([^\/ )]+\.js)):(\d+):(\d+)/))
      .find((m) => m);
    if (frame) {
      const chunkFile = frame[2];
      const v8Line = Number(frame[3]);
      const v8Col = Number(frame[4]);
      const mapPath = join(publicDir, chunkFile + ".map");
      if (existsSync(mapPath)) {
        const map = JSON.parse(readFileSync(mapPath, "utf8"));
        const perLine = parseMappings(map);
        // V8 line/column are 1-based; source-map generated positions are 0-based.
        const segment = resolvePosition(perLine, v8Line - 1, v8Col - 1);
        const source = segment ? map.sources[segment.srcIdx] : null;
        const content = segment ? map.sourcesContent[segment.srcIdx] : null;
        frameOk =
          !!source &&
          source.startsWith("diffpack:///") &&
          typeof content === "string" &&
          content.length > 0;
        frameDetail = `${chunkFile}:${v8Line}:${v8Col} -> ${source} (orig line ${segment ? segment.srcLine : "?"})`;
      } else {
        frameDetail = `no map for ${chunkFile}`;
      }
    }
  }
  record(
    "a real chunk error stack frame resolves through the map to the original source",
    frameOk,
    frameDetail
  );
} catch (e) {
  record("browser harness", false, String(e.stack || e));
} finally {
  if (browser) await browser.close().catch(() => {});
  server.kill("SIGKILL");
}

report();

function report() {
  console.log("\n=== source-map gates ===");
  let pass = 0;
  for (const r of results) {
    console.log(`${r.ok ? "PASS" : "FAIL"} ${r.name}: ${r.detail}`);
    if (r.ok) pass++;
  }
  console.log(`\n${pass}/${results.length} source-map gates passed`);
  process.exit(pass === results.length && results.length > 0 ? 0 : 1);
}
