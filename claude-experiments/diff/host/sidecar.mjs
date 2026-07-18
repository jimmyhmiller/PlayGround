// Diffpack plugin-host sidecar.
//
// Diffpack (Rust) owns the module graph, linking, chunking, and emit. This Node
// process answers the questions that require the project's own JavaScript build
// plugins. It runs *inside the target project* (module resolution is anchored at
// the project root via createRequire) but does NOT run a Vite/Rollup build; it
// only reports config and, later, runs individual framework plugin hooks.
//
// Protocol: one command per invocation, arguments on argv, a single JSON object
// printed to stdout. Errors go to stderr with a non-zero exit. This one-shot
// shape is enough for build-time config, which Diffpack fetches once per build;
// per-module hooks (resolveId/load/transform) will move to a long-lived
// newline-delimited request/response loop when they are needed.
//
//   node sidecar.mjs resolve-config <projectRoot> <environment>
//     -> { "aliases": [[find, replacement], ...], "environments": [...] }
//
// Only string->string aliases are reported (the entry aliases such as
// `#tanstack-router-entry`); regex/function aliases are skipped and counted so a
// silent drop is visible rather than mistaken for "none".

import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

async function main() {
  const [command, projectRoot, environment] = process.argv.slice(2);
  if (command !== "resolve-config") {
    throw new Error(`unknown sidecar command: ${command ?? "<none>"}`);
  }
  if (!projectRoot) {
    throw new Error("resolve-config requires a project root");
  }

  const require = createRequire(`${projectRoot}/package.json`);
  const { resolveConfig } = await import(pathToFileURL(require.resolve("vite")).href);
  const config = await resolveConfig(
    { root: projectRoot },
    "build",
    "production",
    "production",
  );

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

  process.stdout.write(
    JSON.stringify({ environment: selected, environments, aliases, skippedAliases: skipped }),
  );
}

main().catch((error) => {
  process.stderr.write(String(error?.stack ?? error));
  process.exit(1);
});
