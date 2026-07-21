// Acceptance gates for the pinned create-vite react-ts app (test-only; NEVER
// referenced by the build). Runs the same artifact + HTTP gates against either
// build output:
//
//   node acceptance.mjs reference [--strict]   -> dist/          (vite build)
//   node acceptance.mjs diffpack  [--strict]   -> dist-diffpack/ (diffpack build --vite)
//
// A gate failure exits nonzero under --strict; without it the run reports
// readiness without failing (mirroring the TanStack fixture's runner).
import { readFileSync, existsSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { startStaticServer } from "./static-server.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const which = process.argv[2];
const strict = process.argv.includes("--strict");
if (which !== "reference" && which !== "diffpack") {
  console.error("usage: node acceptance.mjs <reference|diffpack> [--strict]");
  process.exit(2);
}
const dist = join(here, which === "reference" ? "dist" : "dist-diffpack");

const results = [];
const gate = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};

// --- Artifact gates -------------------------------------------------------
gate("output directory", existsSync(dist), dist);
const htmlPath = join(dist, "index.html");
gate("index.html emitted", existsSync(htmlPath));
const html = existsSync(htmlPath) ? readFileSync(htmlPath, "utf8") : "";

const scriptMatches = [...html.matchAll(/<script[^>]*type="module"[^>]*src="([^"]+)"[^>]*>/g)];
gate("exactly one module script in head", scriptMatches.length === 1, scriptMatches.map((m) => m[1]).join(", "));
const scriptUrl = scriptMatches[0]?.[1] ?? "";
const cssMatches = [...html.matchAll(/<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g)];
gate("stylesheet link in head", cssMatches.length === 1, cssMatches.map((m) => m[1]).join(", "));
const cssUrl = cssMatches[0]?.[1] ?? "";
gate("original /src script removed", !html.includes("/src/main.tsx"));

const fileAt = (url) => join(dist, url.replace(/^\//, ""));
gate("module script file exists", scriptUrl !== "" && existsSync(fileAt(scriptUrl)), scriptUrl);
gate("stylesheet file exists", cssUrl !== "" && existsSync(fileAt(cssUrl)), cssUrl);
for (const name of ["favicon.svg", "icons.svg"]) {
  gate(`public passthrough: ${name}`, existsSync(join(dist, name)));
}
const jsBytes = scriptUrl && existsSync(fileAt(scriptUrl)) ? statSync(fileAt(scriptUrl)).size : 0;
gate("bundle is a real production build (>50KB, React included)", jsBytes > 50_000, `${jsBytes} bytes`);
const js = jsBytes ? readFileSync(fileAt(scriptUrl), "utf8") : "";
gate("no development React in the bundle", !js.includes("react_stack_bottom_frame"), "dev-only marker absent");

// --- HTTP gates -----------------------------------------------------------
const server = await startStaticServer(dist, 0);
try {
  const base = `http://127.0.0.1:${server.port}`;
  const page = await fetch(base + "/");
  gate("GET / -> 200 html", page.ok && (page.headers.get("content-type") || "").includes("text/html"));
  const script = await fetch(base + scriptUrl);
  gate("module script fetches", script.ok, `${scriptUrl} -> ${script.status}`);
  const css = await fetch(base + cssUrl);
  gate("stylesheet fetches", css.ok, `${cssUrl} -> ${css.status}`);
  // Every asset URL referenced by the bundle must resolve (hashed images).
  const assetRefs = [...js.matchAll(/["'`](\/assets\/[^"'`]+)["'`]/g)].map((m) => m[1]);
  let assetFailures = [];
  for (const ref of new Set(assetRefs)) {
    const res = await fetch(base + ref);
    if (!res.ok) assetFailures.push(`${ref} -> ${res.status}`);
  }
  gate(
    "every referenced /assets URL fetches",
    assetRefs.length > 0 && assetFailures.length === 0,
    assetFailures.join(", ") || `${new Set(assetRefs).size} refs`
  );
} finally {
  server.close();
}

const passed = results.filter((r) => r.ok).length;
console.log(`\n${which}: ${passed}/${results.length} vite-app gates passed`);
if (strict && passed !== results.length) process.exit(1);
