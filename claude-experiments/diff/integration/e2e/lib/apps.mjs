// Build + serve adapters. Each app kind knows how to produce two independent
// deployments of the same untouched source: the `reference` one from the app's
// own toolchain, and the `diffpack` one from the bundler under test.
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync, rmSync } from "node:fs";
import { join, extname, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import net from "node:net";

const here = dirname(fileURLToPath(import.meta.url));
export const repoRoot = join(here, "..", "..", "..");
// `DIFFPACK_BIN` pins the binary under test. Without it a concurrent
// `cargo build` would swap the binary out from under a running comparison.
export const diffpackBin = process.env.DIFFPACK_BIN || join(repoRoot, "target", "release", "diffpack");

export const freePort = () =>
  new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export const runCommand = (cmd, args, { cwd, env, timeout = 900_000 } = {}) => {
  const started = Date.now();
  const r = spawnSync(cmd, args, {
    cwd,
    env: { ...process.env, ...env },
    encoding: "utf8",
    timeout,
    maxBuffer: 128 * 1024 * 1024,
  });
  return {
    ok: r.status === 0,
    status: r.status,
    ms: Date.now() - started,
    stdout: r.stdout ?? "",
    stderr: r.stderr ?? "",
    timedOut: r.error?.code === "ETIMEDOUT",
    error: r.error ? String(r.error) : null,
  };
};

// Generous by default: with several apps building in parallel plus two servers
// and two browsers per app, a cold Next server can take a while to answer. A
// readiness timeout that is too tight reports "serve failed" for an app that is
// merely slow, which is a false accusation against the bundler.
export const waitForHttp = async (url, { timeoutMs = 180_000, child } = {}) => {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    if (child?.exitCode !== null && child?.exitCode !== undefined) {
      return { ok: false, reason: `server exited with code ${child.exitCode}` };
    }
    try {
      const res = await fetch(url, { redirect: "manual" });
      if (res.status < 500) return { ok: true, status: res.status };
      lastError = `status ${res.status}`;
    } catch (error) {
      lastError = String(error);
    }
    await sleep(200);
  }
  return { ok: false, reason: lastError ?? "timeout" };
};

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".ico": "image/x-icon",
  ".webp": "image/webp",
  ".avif": "image/avif",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".xml": "application/xml",
};

/**
 * Static file server with trailing-`/index.html` and `.html` extension fallback.
 *
 * `basePath` is the app's configured `basePath` (`output: 'export'` bakes it into
 * every emitted asset URL, exactly as the host — GitHub Pages — would serve the
 * directory under that prefix). Serving `out/` at `/` instead made every
 * `/<basePath>/_next/...` request miss, and the last-resort `index.html` fallback
 * then answered those misses with an HTML document: the reference page loaded no
 * stylesheet and no script, which read as "the app ships no CSS" and as a dead
 * client-side navigation. The fallback is now confined to extension-less paths
 * (SPA client routes); a missing asset 404s instead of silently becoming HTML.
 */
export const startStaticServer = async (root, port, basePath = "") => {
  const prefix = basePath.replace(/\/+$/, "");
  const server = createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    let pathname = decodeURIComponent(url.pathname);
    if (prefix && (pathname === prefix || pathname.startsWith(`${prefix}/`))) {
      pathname = pathname.slice(prefix.length) || "/";
    }
    const rel = pathname.replace(/^\/+/, "");
    const isAssetRequest = extname(rel) !== "";
    const candidates = [
      join(root, rel),
      join(root, rel, "index.html"),
      join(root, `${rel}.html`),
      ...(isAssetRequest ? [] : [join(root, "index.html")]),
    ];
    for (const candidate of candidates) {
      if (!candidate.startsWith(root)) continue;
      if (!existsSync(candidate)) continue;
      const stat = statSync(candidate);
      if (!stat.isFile()) continue;
      res.writeHead(200, {
        "content-type": MIME[extname(candidate).toLowerCase()] ?? "application/octet-stream",
        "cache-control": "no-store",
      });
      res.end(readFileSync(candidate));
      return;
    }
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("not found");
  });
  await new Promise((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    port,
    kill: () => new Promise((resolve) => server.close(resolve)),
  };
};

const spawnServer = (cmd, args, { cwd, env, logSink }) => {
  // `detached` puts the server in its own process group so the whole tree dies
  // with it: both `npx next start` and `diffpack start` fork grandchildren
  // (next-server, the react-server render worker) that survive a plain kill and
  // would otherwise pile up across the corpus.
  const child = spawn(cmd, args, {
    cwd,
    env: { ...process.env, ...env },
    stdio: ["ignore", "pipe", "pipe"],
    detached: true,
  });
  child.stdout.on("data", (d) => logSink.push(String(d)));
  child.stderr.on("data", (d) => logSink.push(String(d)));
  child.on("error", (e) => logSink.push(`spawn error: ${e}\n`));
  const killGroup = (signal) => {
    try {
      process.kill(-child.pid, signal);
    } catch {
      try {
        child.kill(signal);
      } catch {}
    }
  };
  return {
    child,
    kill: async () => {
      killGroup("SIGTERM");
      await sleep(400);
      if (child.exitCode === null) killGroup("SIGKILL");
      await sleep(100);
    },
  };
};

const readNextConfig = (appDir) => {
  for (const name of ["next.config.ts", "next.config.js", "next.config.mjs"]) {
    const p = join(appDir, name);
    if (existsSync(p)) return readFileSync(p, "utf8");
  }
  return "";
};

export const isStaticExport = (appDir) => /output\s*:\s*["']export["']/.test(readNextConfig(appDir));

export const basePathOf = (appDir) => {
  const m = readNextConfig(appDir).match(/basePath\s*:\s*["']([^"']+)["']/);
  return m ? m[1] : "";
};

// --- adapters -------------------------------------------------------------

const CONFIG_NAMES = ["next.config.ts", "next.config.js", "next.config.mjs", "next.config.cjs"];

/**
 * Runs `fn` with the app's type/lint gates disabled.
 *
 * This is a RETRY path only — the first reference build always runs the app
 * exactly as published. Several pinned examples are stale against current Next
 * types (their own `params` signatures, `@types/react` drift), and a type error
 * is not what this suite measures: the oracle has to be a *running* app.
 * Everything written here is removed again before diffpack ever sees the app.
 */
const withRelaxedChecks = (appDir, fn) => {
  const existing = CONFIG_NAMES.find((name) => existsSync(join(appDir, name)));
  const created = [];
  const restore = [];
  const RELAX = `typescript: { ignoreBuildErrors: true }, eslint: { ignoreDuringBuilds: true }`;

  if (!existing) {
    writeFileSync(join(appDir, "next.config.mjs"), `export default { ${RELAX} };\n`);
    created.push("next.config.mjs");
  } else {
    const original = readFileSync(join(appDir, existing), "utf8");
    restore.push([existing, original]);
    const ext = existing.slice("next.config".length);
    const shadow = `next.config.__e2e_base__${ext}`;
    writeFileSync(join(appDir, shadow), original);
    created.push(shadow);
    const cjs = /module\.exports/.test(original);
    const specifier = ext === ".ts" ? `./next.config.__e2e_base__` : `./${shadow}`;
    writeFileSync(
      join(appDir, existing),
      cjs
        ? `const base = require("${specifier}");\nmodule.exports = { ...base, ${RELAX} };\n`
        : `import base from "${specifier}";\nexport default { ...base, ${RELAX} };\n`
    );
  }
  try {
    return fn();
  } finally {
    for (const [name, content] of restore) writeFileSync(join(appDir, name), content);
    for (const name of created) rmSync(join(appDir, name), { force: true });
  }
};

/**
 * Folds a chain of reference-build attempts into the last one's result, keeping
 * every attempt's output in the record so a green build never hides the retries
 * it took to get there.
 */
const merged = (last, attempts, extra) => ({
  ...last,
  ...extra,
  stdout: attempts.map(([why, r]) => `=== ${why} ===\n${r.stdout}`).join("\n"),
  stderr: attempts.map(([why, r]) => `=== ${why} ===\n${r.stderr}`).join("\n"),
});

/**
 * Next 16 builds with Turbopack by default and hard-errors when the resolved
 * config carries a `webpack` function but no `turbopack` config.
 *
 * Whether the app needs `--webpack` is only readable from the config TEXT when
 * the app writes `webpack()` itself. A config *plugin* — `withMDX(...)`,
 * `withBundleAnalyzer(...)` — installs that function at runtime, so the text
 * says nothing and the textual probe misses it. Next names the case precisely
 * in its own error ("NOTE: your `webpack` config may have been added by a
 * configuration plugin"), so the retry reads the answer back off the failed
 * build instead of trying to out-guess it statically.
 */
export const TURBOPACK_WEBPACK_CONFLICT = /This build is using Turbopack, with a `webpack` config/;

/**
 * Turbopack's other refusal of a webpack-shaped config, with different wording:
 *
 *   Error: loader .../@next/mdx/mdx-js-loader.js for match "{*,next-mdx-rule}"
 *   does not have serializable options.
 *
 * Turbopack runs loaders in a separate process, so a loader rule may only carry
 * JSON. `createMDX({ options: { remarkPlugins: [remarkGfm] } })` puts live
 * FUNCTIONS there — the normal way every MDX app is configured — so such an app
 * is buildable only under webpack. webpack has no serializability requirement
 * and never emits this, which is why matching it is not over-broad; and this is
 * only ever consulted on a build that already failed, where the worst case of a
 * wrong guess is a second failure with both logs kept.
 */
export const TURBOPACK_UNSERIALIZABLE_LOADER = /loader .* does not have serializable options/;

/**
 * True when `next build`'s output is Turbopack refusing a webpack-shaped config —
 * either because the config carries a `webpack` function, or because a loader
 * rule's options are not serializable. Both mean: this app only builds with
 * `--webpack`.
 */
export const needsWebpackFlag = (result) => {
  const output = `${result.stdout || ""}\n${result.stderr || ""}`;
  return TURBOPACK_WEBPACK_CONFLICT.test(output) || TURBOPACK_UNSERIALIZABLE_LOADER.test(output);
};

const nextAdapter = {
  buildReference(app, appDir) {
    const env = { NODE_ENV: "production", NEXT_TELEMETRY_DISABLED: "1" };
    // Turbopack is Next 16's default builder and rejects a `webpack()` config
    // outright; such an app is only buildable with the explicit flag.
    const config = readNextConfig(appDir);
    let extra = /webpack\s*[(:]/.test(config) ? ["--webpack"] : [];
    const build = (wrap = (fn) => fn()) =>
      wrap(() => runCommand("npx", ["--no-install", "next", "build", ...extra], { cwd: appDir, env }));

    // Attempt 1: exactly as published.
    const first = build();
    if (first.ok) return first;
    const log = [["as published", first]];

    // Attempt 2: a config plugin added a webpack-shaped config Turbopack refuses
    // (a `webpack` function, or loader options it cannot serialize).
    if (!extra.includes("--webpack") && needsWebpackFlag(first)) {
      extra = ["--webpack"];
      const webpacked = build();
      log.push(["retry with --webpack (webpack-shaped config added by a config plugin)", webpacked]);
      if (webpacked.ok) return merged(webpacked, log, { webpack: true });
    }

    // Attempt 3: the app's own type/lint gates are stale against current Next.
    const relaxed = build((fn) => withRelaxedChecks(appDir, fn));
    log.push(["retry with type/lint checks disabled", relaxed]);
    return merged(relaxed, log, { relaxed: true, webpack: extra.includes("--webpack") });
  },
  async serveReference(app, appDir, logSink) {
    const port = await freePort();
    if (isStaticExport(appDir)) {
      const out = join(appDir, "out");
      if (!existsSync(out)) return { ok: false, reason: "next build produced no out/ for output:'export'" };
      const server = await startStaticServer(out, port, basePathOf(appDir));
      return { ok: true, port, kill: server.kill, static: true };
    }
    const { child, kill } = spawnServer("npx", ["--no-install", "next", "start", "-p", String(port)], {
      cwd: appDir,
      env: { NODE_ENV: "production", NEXT_TELEMETRY_DISABLED: "1" },
      logSink,
    });
    const up = await waitForHttp(`http://127.0.0.1:${port}${basePathOf(appDir) || "/"}`, { child });
    if (!up.ok) {
      await kill();
      return { ok: false, reason: up.reason };
    }
    return { ok: true, port, kill };
  },
  buildDiffpack(app, appDir) {
    return runCommand(diffpackBin, ["build-app", appDir, "production"], {
      cwd: appDir,
      env: { NODE_ENV: "production" },
    });
  },
  async serveDiffpack(app, appDir, logSink) {
    const port = await freePort();
    const out = join(appDir, ".diffpack-output");
    if (!existsSync(out)) return { ok: false, reason: ".diffpack-output missing" };
    const { child, kill } = spawnServer(diffpackBin, ["start", out, String(port)], {
      cwd: appDir,
      env: { NODE_ENV: "production" },
      logSink,
    });
    const up = await waitForHttp(`http://127.0.0.1:${port}${basePathOf(appDir) || "/"}`, { child });
    if (!up.ok) {
      await kill();
      return { ok: false, reason: up.reason };
    }
    return { ok: true, port, kill };
  },
};

const viteAdapter = {
  buildReference(app, appDir) {
    return runCommand("npx", ["--no-install", "vite", "build"], { cwd: appDir, env: { NODE_ENV: "production" } });
  },
  async serveReference(app, appDir, logSink) {
    const port = await freePort();
    const dist = join(appDir, app.distDir ?? "dist");
    if (!existsSync(dist)) return { ok: false, reason: `${dist} missing` };
    const server = await startStaticServer(dist, port);
    return { ok: true, port, kill: server.kill, static: true };
  },
  buildDiffpack(app, appDir) {
    // `--out-dir` is relative, so this also exercises the Vite semantic that a
    // relative out-dir resolves against the project root, not the process CWD.
    return runCommand(diffpackBin, ["build", appDir, "--vite", "--out-dir", "dist-diffpack"], {
      cwd: appDir,
      env: { NODE_ENV: "production" },
    });
  },
  async serveDiffpack(app, appDir, logSink) {
    const port = await freePort();
    const dist = join(appDir, "dist-diffpack");
    if (!existsSync(dist)) return { ok: false, reason: `${dist} missing` };
    const server = await startStaticServer(dist, port);
    return { ok: true, port, kill: server.kill, static: true };
  },
};

const tanstackAdapter = {
  buildReference(app, appDir) {
    return runCommand("npx", ["--no-install", "vite", "build"], { cwd: appDir, env: { NODE_ENV: "production" } });
  },
  async serveReference(app, appDir, logSink) {
    const port = await freePort();
    // TanStack Start's build output layout changed across the versions these
    // examples pin: older ones emit a self-contained `.output/server/index.mjs`,
    // newer ones emit `dist/server/server.js` (a fetch handler) plus
    // `dist/client`, served by srvx. Some examples' own `start` script is stale
    // relative to the version they install, so the layout is detected, not read
    // from package.json.
    const nitro = join(appDir, ".output", "server", "index.mjs");
    const srvxEntry = join(appDir, "dist", "server", "server.js");
    let cmd;
    let args;
    if (existsSync(nitro)) {
      cmd = process.execPath;
      args = [nitro];
    } else if (existsSync(srvxEntry)) {
      cmd = "npx";
      args = ["--yes", "srvx", "--prod", "-s", "../client", "dist/server/server.js"];
    } else {
      return { ok: false, reason: "neither .output/server/index.mjs nor dist/server/server.js was produced" };
    }
    const { child, kill } = spawnServer(cmd, args, {
      cwd: appDir,
      env: { PORT: String(port), NODE_ENV: "production" },
      logSink,
    });
    const up = await waitForHttp(`http://127.0.0.1:${port}/`, { child });
    if (!up.ok) {
      await kill();
      return { ok: false, reason: up.reason };
    }
    return { ok: true, port, kill };
  },
  buildDiffpack(app, appDir) {
    const client = runCommand(diffpackBin, ["build-app", appDir, "client"], { cwd: appDir });
    if (!client.ok) return client;
    const ssr = runCommand(diffpackBin, ["build-app", appDir, "ssr"], { cwd: appDir });
    return { ...ssr, stdout: client.stdout + ssr.stdout, stderr: client.stderr + ssr.stderr, ms: client.ms + ssr.ms };
  },
  async serveDiffpack(app, appDir, logSink) {
    const port = await freePort();
    const entry = join(appDir, ".diffpack-output", "server", "index.mjs");
    if (!existsSync(entry)) return { ok: false, reason: ".diffpack-output/server/index.mjs missing" };
    const { child, kill } = spawnServer(process.execPath, [entry], {
      cwd: appDir,
      env: { PORT: String(port), NODE_ENV: "production" },
      logSink,
    });
    const up = await waitForHttp(`http://127.0.0.1:${port}/`, { child });
    if (!up.ok) {
      await kill();
      return { ok: false, reason: up.reason };
    }
    return { ok: true, port, kill };
  },
};

export const adapters = {
  "next-app": nextAdapter,
  "next-pages": nextAdapter,
  vite: viteAdapter,
  tanstack: tanstackAdapter,
};

export const cleanBuildOutput = (appDir) => {
  // `.diffpack-next-pages` is the pages-router adapter's scaffold, and was missing
  // here: a stale scaffold from a previous run would have been reused instead of
  // regenerated, so a build could pass on last week's generated entry.
  for (const dir of [
    ".next",
    "out",
    ".output",
    "dist",
    "dist-diffpack",
    ".diffpack-output",
    ".diffpack-next",
    ".diffpack-next-pages",
  ]) {
    spawnSync("rm", ["-rf", join(appDir, dir)]);
  }
};

export const listRoutesFromDisk = (appDir) => {
  const roots = [join(appDir, "app"), join(appDir, "src", "app"), join(appDir, "pages"), join(appDir, "src", "pages")];
  const routes = new Set();
  const walk = (dir, prefix) => {
    if (!existsSync(dir)) return;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith("_") || entry.name === "api") continue;
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name.startsWith("[") || entry.name.startsWith("(") || entry.name.startsWith("@")) continue;
        walk(full, `${prefix}/${entry.name}`);
      } else if (/^(page|index)\.(t|j)sx?$|^(page|index)\.mdx$/.test(entry.name)) {
        routes.add(prefix || "/");
      } else if (/\.(t|j)sx?$|\.mdx$/.test(entry.name) && dir.includes("pages")) {
        const base = entry.name.replace(/\.(t|j)sx?$|\.mdx$/, "");
        if (base !== "index" && !base.startsWith("[")) routes.add(`${prefix}/${base}`);
      }
    }
  };
  for (const root of roots) walk(root, "");
  return [...routes].sort();
};
