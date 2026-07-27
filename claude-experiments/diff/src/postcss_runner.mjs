// Runs the APP's own PostCSS over one stylesheet, mirroring what Vite does when
// a project has a `postcss.config.*` (or a `package.json` "postcss" key).
//
// Contract (all values supplied by the native Rust caller, see `crate::postcss`):
//   env DIFFPACK_POSTCSS_CONFIG      absolute path to the config file, or to the
//                                    package.json when the config lives there
//   env DIFFPACK_POSTCSS_CONFIG_KIND "file" | "package"
//   env DIFFPACK_POSTCSS_FROM        the stylesheet's own path (postcss `from`)
//   stdin                            the CSS to transform
//   stdout                           the transformed CSS (nothing else)
//
// `postcss` and every configured plugin are resolved from the PROJECT's own
// node_modules (the process runs with cwd = project root, and a stdin ES module
// resolves bare specifiers against the cwd), so the app's exact toolchain and
// versions are used — never a bundled copy.

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

async function main() {
  const configPath = process.env.DIFFPACK_POSTCSS_CONFIG;
  const configKind = process.env.DIFFPACK_POSTCSS_CONFIG_KIND;
  const from = process.env.DIFFPACK_POSTCSS_FROM || undefined;
  if (!configPath || !configKind) {
    throw new Error("postcss runner: DIFFPACK_POSTCSS_CONFIG(_KIND) not set");
  }

  const css = readFileSync(0, "utf8");

  // Load the raw config object exactly like postcss-load-config would.
  let raw;
  if (configKind === "package") {
    const pkg = JSON.parse(readFileSync(configPath, "utf8"));
    raw = pkg.postcss;
    if (!raw) {
      throw new Error(`postcss runner: no "postcss" key in ${configPath}`);
    }
  } else {
    const mod = await import(pathToFileURL(configPath).href);
    raw = mod && mod.default !== undefined ? mod.default : mod;
  }
  // A config file may export a function of the build context.
  if (typeof raw === "function") {
    raw = raw({
      env: process.env.NODE_ENV || "production",
      mode: process.env.NODE_ENV || "production",
      cwd: process.cwd(),
      from,
    });
  }
  raw = await raw;

  const plugins = await resolvePlugins(raw && raw.plugins);

  const postcssMod = await import("postcss");
  const postcss = postcssMod.default || postcssMod;

  const result = await postcss(plugins).process(css, { from, map: false });
  // Surface plugin warnings on stderr; they must not corrupt the CSS on stdout.
  for (const warning of result.warnings()) {
    process.stderr.write(`postcss warning: ${warning.toString()}\n`);
  }
  process.stdout.write(result.css);
}

// Normalizes the many shapes `plugins` can take into an array PostCSS accepts:
//   - an array of plugin instances / factory calls (used verbatim)
//   - an object map { "plugin-name": options } (postcss.config.js style): each
//     name is imported from the project and invoked with its options
async function resolvePlugins(plugins) {
  if (plugins == null) {
    return [];
  }
  if (Array.isArray(plugins)) {
    // Entries may be plugin objects/functions already, or bare names (strings).
    const out = [];
    for (const entry of plugins) {
      if (typeof entry === "string") {
        out.push(await instantiate(entry, {}));
      } else {
        out.push(entry);
      }
    }
    return out;
  }
  if (typeof plugins === "object") {
    const out = [];
    for (const [name, options] of Object.entries(plugins)) {
      if (options === false) {
        continue; // disabled plugin
      }
      out.push(await instantiate(name, options === true ? {} : options));
    }
    return out;
  }
  throw new Error(
    `postcss runner: unsupported "plugins" shape (${typeof plugins})`,
  );
}

// Imports a plugin package from the project and applies its options. A plugin
// package's default export is a factory (`autoprefixer(opts)`); some are already
// plain plugin objects, which are used as-is.
async function instantiate(name, options) {
  let mod;
  try {
    mod = await import(name);
  } catch (error) {
    throw new Error(
      `postcss runner: cannot load plugin "${name}" from the project (${error && error.message}); ` +
        `is it installed in node_modules?`,
    );
  }
  const factory = mod && mod.default !== undefined ? mod.default : mod;
  if (typeof factory === "function") {
    return factory(options);
  }
  // Already a plugin object (has a postcss/postcssPlugin marker).
  return factory;
}

main().catch((error) => {
  process.stderr.write(String((error && error.stack) || error) + "\n");
  process.exit(1);
});
