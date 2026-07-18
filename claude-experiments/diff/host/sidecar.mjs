// Diffpack plugin-host sidecar.
//
// Diffpack (Rust) owns the module graph, linking, chunking, and emit. This Node
// process answers the questions that require the project's own JavaScript build
// plugins. It runs *inside the target project* (module resolution is anchored at
// the project root via createRequire) but does NOT run a Vite/Rollup build; it
// reports config and runs individual framework plugin hooks on demand.
//
// Two commands:
//
//   node sidecar.mjs resolve-config <projectRoot> <environment>
//     One-shot. Prints one JSON object:
//       { aliases: [[find, replacement], ...], environments: [...], ... }
//
//   node sidecar.mjs serve <projectRoot>
//     Long-lived. Newline-delimited JSON request/response over stdin/stdout, one
//     response object per request line. Diffpack calls this only for ids its own
//     resolver/loader cannot handle (virtual and plugin-generated modules), so
//     the common path stays native. Requests:
//       { id, op: "resolveId", environment, specifier, importer }
//         -> { id, resolved: string|null }
//       { id, op: "load", environment, moduleId }
//         -> { id, code: string|null }
//       { id, op: "shutdown" } -> { id, ok: true } then exit
//     Every response echoes the request `id`. Errors become
//       { id, error: string }.
//
// Only *framework* plugins run; Vite/Rollup internals (vite:/builtin:/fullstack:/
// alias/commonjs/nitro/rollup/_) are denied so their behavior stays Diffpack-
// native.

import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { createInterface } from "node:readline";

const DENY = /^(vite:|builtin:|fullstack:|alias$|commonjs$|nitro|rollup|_|@rollup)/;

function requireFrom(projectRoot) {
  return createRequire(`${projectRoot}/package.json`);
}

async function importVite(projectRoot) {
  const require = requireFrom(projectRoot);
  return import(pathToFileURL(require.resolve("vite")).href);
}

// ---- resolve-config (one-shot) --------------------------------------------

async function resolveConfigCommand(projectRoot, environment) {
  const { resolveConfig } = await importVite(projectRoot);
  const config = await resolveConfig({ root: projectRoot }, "build", "production", "production");

  const environments = Object.keys(config.environments ?? {});
  const selected = environment ?? "client";
  const envConfig = config.environments?.[selected];
  if (environment && !envConfig) {
    throw new Error(
      `environment ${JSON.stringify(environment)} not found; have: ${environments.join(", ")}`,
    );
  }

  const rawAliases = envConfig?.resolve?.alias ?? config.resolve?.alias ?? [];
  const list = Array.isArray(rawAliases)
    ? rawAliases
    : Object.entries(rawAliases).map(([find, replacement]) => ({ find, replacement }));

  const aliases = [];
  let skipped = 0;
  for (const entry of list) {
    if (typeof entry.find === "string" && typeof entry.replacement === "string") {
      aliases.push([entry.find, entry.replacement]);
    } else {
      skipped += 1;
    }
  }

  // Vite writes `development|production` as one condition entry it expands per
  // mode; this is a production build, so resolve it to `production`.
  const rawConditions = envConfig?.resolve?.conditions ?? config.resolve?.conditions ?? [];
  const conditions = rawConditions.map((condition) =>
    condition === "development|production" ? "production" : condition,
  );

  process.stdout.write(
    JSON.stringify({
      environment: selected,
      environments,
      aliases,
      conditions,
      skippedAliases: skipped,
    }),
  );
}

// ---- serve (long-lived hook host) -----------------------------------------

async function serveCommand(projectRoot) {
  const { createServer } = await importVite(projectRoot);
  const server = await createServer({
    root: projectRoot,
    configFile: `${projectRoot}/vite.config.ts`,
    server: { middlewareMode: true },
    logLevel: "silent",
  });

  // Framework plugins per environment, computed lazily and cached.
  const environmentPlugins = new Map();
  function pluginsFor(environmentName) {
    let cached = environmentPlugins.get(environmentName);
    if (cached) return cached;
    const env = server.environments[environmentName];
    if (!env) throw new Error(`no such environment: ${environmentName}`);
    cached = env.plugins.filter((plugin) => plugin.name && !DENY.test(plugin.name));
    environmentPlugins.set(environmentName, { env, plugins: cached });
    return environmentPlugins.get(environmentName);
  }

  function contextFor(env) {
    return {
      environment: env,
      meta: { watchMode: false },
      async resolve(source) {
        return { id: source, external: false };
      },
      emitFile() {
        return "ref";
      },
      getModuleInfo() {
        return null;
      },
      addWatchFile() {},
      warn() {},
      error(message) {
        throw new Error(typeof message === "string" ? message : message.message);
      },
      debug() {},
      info() {},
    };
  }

  function handler(hook) {
    if (!hook) return null;
    return typeof hook === "function" ? hook : hook.handler;
  }

  async function resolveId(environmentName, specifier, importer) {
    const { env, plugins } = pluginsFor(environmentName);
    const ctx = contextFor(env);
    for (const plugin of plugins) {
      const fn = handler(plugin.resolveId);
      if (!fn) continue;
      const result = await fn.call(ctx, specifier, importer ?? undefined, {});
      const id = typeof result === "string" ? result : result?.id;
      if (id) return id;
    }
    return null;
  }

  async function load(environmentName, moduleId) {
    const { env, plugins } = pluginsFor(environmentName);
    const ctx = contextFor(env);
    for (const plugin of plugins) {
      const fn = handler(plugin.load);
      if (!fn) continue;
      const result = await fn.call(ctx, moduleId, {});
      const code = typeof result === "string" ? result : result?.code;
      if (code != null) return code;
    }
    return null;
  }

  const readline = createInterface({ input: process.stdin });
  for await (const line of readline) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let request;
    try {
      request = JSON.parse(trimmed);
    } catch (error) {
      process.stdout.write(`${JSON.stringify({ id: null, error: `bad request: ${error}` })}\n`);
      continue;
    }
    const respond = (payload) =>
      process.stdout.write(`${JSON.stringify({ id: request.id, ...payload })}\n`);
    try {
      switch (request.op) {
        case "resolveId":
          respond({ resolved: await resolveId(request.environment, request.specifier, request.importer) });
          break;
        case "load":
          respond({ code: await load(request.environment, request.moduleId) });
          break;
        case "shutdown":
          respond({ ok: true });
          await server.close();
          process.exit(0);
          break;
        default:
          respond({ error: `unknown op: ${request.op}` });
      }
    } catch (error) {
      respond({ error: String(error?.message ?? error) });
    }
  }
  await server.close();
}

// ---- entry ----------------------------------------------------------------

async function main() {
  const [command, projectRoot, environment] = process.argv.slice(2);
  if (!projectRoot) {
    throw new Error(`${command ?? "<none>"} requires a project root`);
  }
  if (command === "resolve-config") {
    await resolveConfigCommand(projectRoot, environment);
  } else if (command === "serve") {
    await serveCommand(projectRoot);
  } else {
    throw new Error(`unknown sidecar command: ${command ?? "<none>"}`);
  }
}

main().catch((error) => {
  process.stderr.write(String(error?.stack ?? error));
  process.exit(1);
});
