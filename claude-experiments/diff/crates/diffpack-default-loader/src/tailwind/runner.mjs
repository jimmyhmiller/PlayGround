// Compiles ONE Tailwind CSS entry with the APP's OWN installed Tailwind, for the
// sheets diffpack's native engine cannot serve — the ones that load a JavaScript
// `@plugin` (`@tailwindcss/typography`, `tailwind-scrollbar`, `daisyui`, …), use an
// at-rule the native engine has no meaning for, or `@apply` a utility only a plugin
// registers.
//
// Why a Node process at all: a Tailwind plugin is an arbitrary JavaScript function
// that registers utilities, variants and base rules at runtime. No CSS-level engine
// can know what it produces, and reimplementing the plugins one by one is an
// unbounded tail that drifts from every new release. Running the app's own compiler
// is the faithful answer and covers every plugin that exists. This mirrors
// `src/mdx_runner.mjs` (the app's own `@mdx-js/mdx`), `src/less_stylus_runner.mjs`
// and `src/postcss_runner.mjs`.
//
// Contract (values set by `crate::tailwind_delegate`):
//   argv[2]   absolute path of the Tailwind ENTRY stylesheet. Two jobs: it is the
//             module-resolution anchor (the app's Tailwind is resolved FROM it, never
//             from this script — this file is executed out of a temp directory, so a
//             bare `import("tailwindcss")` would resolve against THAT directory), and
//             its parent directory is the compile's `base`, against which `@plugin`,
//             `@config`, `@source` and any remaining `@import` resolve.
//   stdin     {"css": "<entry source>", "candidates": ["flex", …]}
//             `css` is the entry with diffpack's `@import` splicing, `url()`
//             rewriting and `@source` absolutization already applied, so the
//             delegated sheet references the same content-hashed assets the native
//             one does. `candidates` is diffpack's own class scan — one source of
//             truth for both engines.
//   stdout    {"css": "<compiled stylesheet>", "engine": "...", "version": "..."}
//
// Any failure exits non-zero with a message naming the entry; `crate::tailwind_delegate`
// turns that into a hard build error. There is no fall-back to the native engine:
// silently shipping a sheet without the plugin's utilities is the defect this file
// exists to remove.
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";

// stdout carries the JSON protocol and nothing else. Tailwind and its plugins are
// third-party code that may `console.log`; route that to stderr so a chatty plugin
// cannot corrupt the result.
console.log = (...parts) => process.stderr.write(parts.map(String).join(" ") + "\n");
console.info = console.log;

/** Resolves `id` the way the ENTRY stylesheet's own package would. */
function resolveFromApp(entry, id) {
  try {
    return createRequire(entry).resolve(id);
  } catch {
    return null;
  }
}

// The version of the package a resolved file belongs to. Read by walking up from
// the resolved file rather than by resolving "<id>/package.json": most Tailwind
// packages (`@tailwindcss/node` among them) do not list `./package.json` in their
// `exports`, so that specifier does not resolve at all.
function packageVersion(file) {
  let dir = dirname(file);
  for (;;) {
    try {
      const manifest = JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
      if (manifest.name && manifest.version) return manifest.version;
    } catch {
      /* keep walking */
    }
    const parent = dirname(dir);
    if (parent === dir) return null;
    dir = parent;
  }
}

// The compiler diffpack drives is `@tailwindcss/node`'s `compile()`. That is the
// exact entry point `@tailwindcss/postcss`, `@tailwindcss/vite` and
// `@tailwindcss/cli` all call, and it pins its own `tailwindcss` core — so a sheet
// delegated here is produced by the same code, and the same Tailwind version, as the
// app's own build. It also supplies the `@plugin`/`@config` module loader (jiti, so a
// TypeScript plugin works) and the `style`-condition stylesheet resolver.
async function loadTailwindNode(entry) {
  const resolved = resolveFromApp(entry, "@tailwindcss/node");
  if (!resolved) return null;
  const mod = await import(pathToFileURL(resolved).href);
  if (typeof mod.compile !== "function") return null;
  // The reported version is the `tailwindcss` CORE this adapter will load, not the
  // adapter's own: the adapters pin a core copy, and that copy is what the app's own
  // build compiles with, so it is the number that describes the emitted sheet.
  const core = resolveFromApp(resolved, "tailwindcss") || resolved;
  return { compile: mod.compile, engine: "@tailwindcss/node", version: packageVersion(core) };
}

// Fallback for an app that installs `tailwindcss` but none of the packages that
// bundle `@tailwindcss/node` (its PostCSS/Vite/CLI adapters). The core `compile()`
// takes the same CSS-text + `base` shape but leaves module and stylesheet loading to
// the caller, so both are supplied here. Everything they are asked for is named
// explicitly; an unsupported shape throws rather than resolving to nothing.
async function loadTailwindCore(entry) {
  const resolved = resolveFromApp(entry, "tailwindcss");
  if (!resolved) return null;
  const mod = await import(pathToFileURL(resolved).href);
  const compile = mod.compile || (mod.default && mod.default.compile);
  if (typeof compile !== "function") return null;

  // `@plugin`/`@config` targets. A TypeScript or ESM-in-CJS plugin needs a loader;
  // the app's own jiti is used when it has one (that is what @tailwindcss/node does),
  // otherwise a plain dynamic import.
  let jiti = null;
  const jitiPath = resolveFromApp(entry, "jiti");
  if (jitiPath) {
    try {
      const { createJiti } = await import(pathToFileURL(jitiPath).href);
      jiti = createJiti(entry, { interopDefault: true });
    } catch {
      jiti = null;
    }
  }
  const loadModule = async (id, base) => {
    const anchor = join(base, "__diffpack_tailwind__.js");
    let file;
    if (id.startsWith("./") || id.startsWith("../") || isAbsolute(id)) {
      file = resolve(base, id);
    } else {
      try {
        file = createRequire(anchor).resolve(id);
      } catch (error) {
        throw new Error(
          `cannot resolve the Tailwind plugin/config "${id}" from ${base}: ` +
            `${(error && error.message) || error}`,
        );
      }
    }
    const loaded = jiti
      ? await jiti.import(file)
      : await import(pathToFileURL(file).href);
    return { path: file, base: dirname(file), module: loaded?.default ?? loaded };
  };

  // Stylesheet targets. diffpack has already inlined every `@import` it can resolve
  // itself, so what reaches here is the framework's own (`tailwindcss`,
  // `tailwindcss/theme`, …) plus the relative imports inside those files.
  const loadStylesheet = async (id, base) => {
    let file;
    if (id.startsWith("./") || id.startsWith("../") || isAbsolute(id)) {
      file = resolve(base, id);
    } else {
      const [scope, ...rest] = id.split("/");
      const pkg = scope.startsWith("@") ? `${scope}/${rest.shift()}` : scope;
      const sub = rest.join("/");
      const manifest = resolveFromApp(join(base, "__diffpack_tailwind__.css"), `${pkg}/package.json`);
      if (!manifest) {
        throw new Error(
          `cannot resolve the stylesheet "${id}" from ${base}: no such package. ` +
            `Install "@tailwindcss/node" (a dependency of @tailwindcss/postcss, ` +
            `@tailwindcss/vite and @tailwindcss/cli) so the app's own stylesheet ` +
            `resolver is used instead of this fallback.`,
        );
      }
      const root = dirname(manifest);
      file = sub
        ? [join(root, sub), join(root, `${sub}.css`)].find(exists)
        : join(root, "index.css");
      if (!file) {
        throw new Error(`cannot resolve the stylesheet "${id}" inside ${root}`);
      }
    }
    return { path: file, base: dirname(file), content: readFileSync(file, "utf8") };
  };

  return {
    engine: "tailwindcss",
    version: packageVersion(resolved),
    compile: (css, options) =>
      compile(css, { base: options.base, loadModule, loadStylesheet }),
  };
}

function exists(path) {
  try {
    readFileSync(path);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  const entry = process.argv[2];
  if (!entry) {
    throw new Error("tailwind runner: entry stylesheet path not given");
  }
  const request = JSON.parse(readFileSync(0, "utf8"));
  const engine = (await loadTailwindNode(entry)) || (await loadTailwindCore(entry));
  if (!engine) {
    throw new Error(
      `Tailwind ${entry}: this stylesheet needs the app's own Tailwind compiler, but no ` +
        `usable "tailwindcss" could be loaded from the project (neither ` +
        `"@tailwindcss/node" nor "tailwindcss" resolves from ${entry}, or the copy found ` +
        `exports no compile() — that API is Tailwind v4). Install tailwindcss v4 in the ` +
        `workspace that owns this stylesheet.`,
    );
  }
  const compiler = await engine.compile(request.css, {
    base: dirname(entry),
    // No `from`: diffpack does not consume Tailwind's source map for this sheet, and
    // asking for one makes the compile track positions for no consumer.
    onDependency: () => {},
  });
  const css = compiler.build(request.candidates || []);
  process.stdout.write(
    JSON.stringify({ css, engine: engine.engine, version: engine.version || "unknown" }),
  );
}

main().catch((error) => {
  const message = (error && (error.message || error.stack)) || String(error);
  process.stderr.write(message + "\n");
  process.exit(1);
});
