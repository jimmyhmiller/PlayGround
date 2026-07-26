// The SSG prerenderer — build-time static generation for a diffpack-built Next
// app-router app. It reads the native prerender PLAN (written by
// `diffpack build-app <root> static`, which classified every route), and for each
// STATIC / FORCE-STATIC / SSG route renders the SAME per-request pipeline the live
// orchestrator uses (react-server render child -> flight -> SSR-of-flight -> full HTML
// document) AHEAD OF TIME, writing `<static>/<route>.html` + `<static>/<route>.rsc`
// to disk. DYNAMIC routes are SKIPPED and recorded in the manifest with a reason —
// never silently dropped. The result is served by a DUMB static file server
// (next-static-serve.mjs) with zero per-request render and zero child processes.
//
//   usage: node next-prerender.mjs <outputDir> [--static-export]
//
// The node process runs the app's OWN React (the explicitly-allowed oracle, exactly
// as the orchestrator does); the bundling of all three graphs stays native Rust.

import { readFileSync, writeFileSync, mkdirSync, existsSync, cpSync } from "node:fs";
import { join, dirname } from "node:path";
import {
  loadManifests,
  getRenderFlightToDocument,
  makeRunReactServer,
  requireBuiltBundles,
} from "./next-render-core.mjs";

function die(message) {
  console.error(`next-prerender: ${message}`);
  process.exit(1);
}

const outputDir = process.argv[2];
const staticExport = process.argv.includes("--static-export");
if (!outputDir) die("usage: node next-prerender.mjs <outputDir> [--static-export]");

requireBuiltBundles(outputDir);

const staticDir = join(outputDir, "static");
const planPath = join(staticDir, "prerender-plan.json");
if (!existsSync(planPath)) {
  die(`prerender plan not found at ${planPath} — run \`diffpack build-app <root> static\` (it writes the plan before invoking this)`);
}
const plan = JSON.parse(readFileSync(planPath, "utf8"));

const { serverConsumerManifest, clientManifestPath } = loadManifests(outputDir);
const render = getRenderFlightToDocument(join(outputDir, "server", "server.mjs"), { dev: false });
const runReactServer = makeRunReactServer(join(outputDir, "rsc-render", "server.mjs"));

// Substitute one generateStaticParams combo into a route's parsed segments to build a
// concrete URL path + on-disk file stem (mirrored). Catch-all params are string[].
function buildConcrete(segments, combo) {
  const parts = [];
  for (const seg of segments) {
    if (seg.k === "static") {
      parts.push(seg.v);
    } else if (seg.k === "dynamic") {
      const val = combo[seg.v];
      if (val == null) throw new Error(`generateStaticParams combo missing dynamic param "${seg.v}": ${JSON.stringify(combo)}`);
      parts.push(encodeURIComponent(String(val)));
    } else if (seg.k === "catchall" || seg.k === "optcatchall") {
      const val = combo[seg.v];
      const arr = Array.isArray(val) ? val : val == null ? [] : [val];
      for (const p of arr) parts.push(encodeURIComponent(String(p)));
    } else {
      throw new Error(`unknown segment kind ${JSON.stringify(seg.k)}`);
    }
  }
  const urlPath = "/" + parts.join("/");
  const fileStem = parts.join("/") || "index";
  return { urlPath: urlPath === "/" ? "/" : urlPath, fileStem };
}

function bufToString(buf) {
  return Buffer.from(buf).toString("utf8");
}

async function writeRoute(urlPath, fileStem) {
  // Render exactly as the orchestrator does for this pathname under an EMPTY request
  // context (a prerender has no request). Any render error PROPAGATES (no try/catch
  // swallow) and fails the build naming the pathname.
  let result;
  try {
    result = await runReactServer(["render", urlPath, clientManifestPath], "");
  } catch (error) {
    const msg = String(error && error.stack ? error.stack : error);
    if (msg.includes("DIFFPACK_DYNAMIC_BAILOUT")) {
      die(
        `route ${urlPath} read request state (cookies/headers) but was classified static — ` +
          `mark it Dynamic (it needs per-request rendering).\n${msg}`,
      );
    }
    die(`prerender of ${urlPath} failed:\n${msg}`);
  }
  const { flight, params, redirect, notFound, tags } = result;
  if (redirect) die(`route ${urlPath} issued a server-side redirect during prerender — it is not statically prerenderable (mark it Dynamic)`);
  if (notFound) die(`route ${urlPath} rendered notFound() during prerender — it is not statically prerenderable (mark it Dynamic)`);

  const flightBuf = Buffer.from(flight);
  let html;
  try {
    html = await render(
      new Uint8Array(flightBuf),
      serverConsumerManifest,
      flightBuf.toString("base64"),
      params,
      { pathname: urlPath, search: "" },
    );
  } catch (error) {
    die(`SSR-of-flight for ${urlPath} failed:\n${String(error && error.stack ? error.stack : error)}`);
  }

  const htmlPath = join(staticDir, `${fileStem}.html`);
  const rscPath = join(staticDir, `${fileStem}.rsc`);
  mkdirSync(dirname(htmlPath), { recursive: true });
  writeFileSync(htmlPath, html);
  writeFileSync(rscPath, flightBuf); // RAW flight — the soft-nav (?__rsc=1) source.
  // next/cache: the cache tags this page read (unstable_cache / tagged fetch), recorded in
  // the manifest so the orchestrator can map a tag back to this pathname for revalidateTag.
  return { htmlPath, rscPath, tags: Array.isArray(tags) ? tags : [] };
}

async function main() {
  mkdirSync(staticDir, { recursive: true });
  const written = []; // concrete static pathnames
  const dynamic = []; // { path, reason }

  for (const route of plan) {
    if (route.kind === "dynamic") {
      dynamic.push({ path: route.path, reason: route.reason || "dynamic" });
      continue;
    }
    if (route.kind === "static" || route.kind === "forceStatic" || route.kind === "isr") {
      const { tags } = await writeRoute(route.path, route.file || "index");
      written.push({ path: route.path, file: route.file || "index", revalidate: route.revalidate ?? null, tags });
      console.log(
        `prerendered ${route.kind} ${route.path} -> ${route.file || "index"}.html + .rsc` +
          (route.revalidate ? ` (ISR: revalidate ${route.revalidate}s)` : ""),
      );
      continue;
    }
    if (route.kind === "ssg") {
      // Enumerate the concrete param sets in the app's own React runtime.
      const enumResult = await runReactServer(["staticparams", route.path, clientManifestPath]);
      let combos;
      try {
        combos = JSON.parse(bufToString(enumResult.flight));
      } catch (error) {
        die(`generateStaticParams for ${route.path} did not print valid JSON: ${bufToString(enumResult.flight)}`);
      }
      if (!Array.isArray(combos)) die(`generateStaticParams for ${route.path} did not return an array`);
      if (combos.length === 0) {
        console.log(`WARN next SSG: ${route.path} generateStaticParams returned [] — no pages prerendered for it`);
      }
      for (const combo of combos) {
        const { urlPath, fileStem } = buildConcrete(route.segments, combo);
        const { tags } = await writeRoute(urlPath, fileStem);
        written.push({ path: urlPath, file: fileStem, revalidate: route.revalidate ?? null, tags });
        console.log(
          `prerendered ssg ${route.path} [${JSON.stringify(combo)}] -> ${fileStem}.html + .rsc` +
            (route.revalidate ? ` (ISR: revalidate ${route.revalidate}s)` : ""),
        );
      }
      continue;
    }
    die(`unknown route kind ${JSON.stringify(route.kind)} for ${route.path}`);
  }

  const writtenPaths = written.map((w) => w.path);

  // `--static-export`: a pure export cannot serve any dynamic route — fail naming them.
  if (staticExport && dynamic.length > 0) {
    const names = dynamic.map((d) => `${d.path} (${d.reason})`).join(", ");
    die(`--static-export: the app has route(s) that cannot be statically prerendered: ${names}. A static export cannot serve them.`);
  }

  // Copy public/ into static/ so `/rsc.css`, `/_diffpack-image/*`, `/client.js`, and
  // every asset are colocated — the prerendered pages are self-contained under a dumb
  // file server.
  const publicDir = join(outputDir, "public");
  if (existsSync(publicDir)) {
    cpSync(publicDir, staticDir, { recursive: true });
    console.log(`copied public/ -> static/ (assets colocated)`);
  }

  const manifest = {
    // Back-compat: the dumb static file server keys off the path list.
    static: writtenPaths,
    // The orchestrator keys off this: path -> { file stem, revalidate TTL | null }.
    // A null revalidate is a pure static page (served from cache forever); a number is
    // an ISR page (served from cache, regenerated on demand once older than N seconds).
    entries: written,
    dynamic,
    generatedAt: new Date().toISOString(),
  };
  writeFileSync(join(staticDir, "prerender-manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(
    `next-prerender: wrote ${written.length} static page(s); skipped ${dynamic.length} dynamic route(s) -> ${join(staticDir, "prerender-manifest.json")}`,
  );
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
