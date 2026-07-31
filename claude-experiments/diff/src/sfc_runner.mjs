// Compiles ONE single-file component to JavaScript (+ CSS) using the APP's OWN
// installed compiler — `@vue/compiler-sfc` for `.vue`, `svelte/compiler` for
// `.svelte` — then hands both back to the native pipeline, exactly as
// `less_stylus_runner.mjs` hands back plain CSS. diffpack reimplements neither
// compiler: the app's pinned version is what runs, so a component compiles the
// way that app's own toolchain compiles it.
//
// The emitted JavaScript is NOT final: it still contains the component's own
// `import`s (Vue/Svelte runtime, sibling components, asset URLs) and, for a
// TypeScript `<script>`, TypeScript annotations. The caller feeds it through the
// ordinary module pipeline, so those imports become real graph edges and the
// types are stripped by the same transform every `.ts` module takes. This
// mirrors @vitejs/plugin-vue, whose own output is handed to Vite's TS transform
// (`transformWithOxc(..., { lang: "ts" })`) for exactly the same reason.
//
// Contract (values from `crate::sfc`):
//   env DIFFPACK_SFC_FRAMEWORK  "vue" | "svelte"
//   env DIFFPACK_SFC_FILE       the component's absolute path
//   env DIFFPACK_SFC_ROOT       project root — the base for the component id
//                               hash (Vue) and for locating svelte.config.js
//   stdin                       the component source
//   stdout                      one JSON object:
//                                 { "code": string,      // JavaScript (maybe TS)
//                                   "css": string,       // "" when the component has no styles
//                                   "language": "ts"|"js" }
//
// Anything that cannot be compiled throws; the caller turns a non-zero exit into
// a build error naming the file. Nothing is ever passed through uncompiled.

import { readFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { pathToFileURL } from "node:url";

const slash = (value) => value.replace(/\\/g, "/");
const shortHash = (text) => createHash("sha256").update(text).digest("hex").slice(0, 8);

/// Imports a package from the PROJECT's node_modules. The runner is passed to
/// `node` via `--eval` with cwd = the project root, and a bare specifier in an
/// `--input-type=module` eval resolves against the working directory — so this
/// is the project's own copy of the compiler, never diffpack's.
async function importFromProject(root, specifier, humanName) {
  try {
    return await import(specifier);
  } catch (error) {
    const detail = (error && (error.message || error.stack)) || String(error);
    throw new Error(
      `cannot load ${humanName} ("${specifier}") from ${root}: ${detail}\n` +
        `  diffpack compiles this component with the project's OWN compiler; ` +
        `install it (it is normally a dependency of the framework's Vite plugin)`,
    );
  }
}

// --- Vue ------------------------------------------------------------------
//
// Mirrors @vitejs/plugin-vue's `transformMain` for a production client build,
// with one deliberate difference: the script/template/style blocks are emitted
// into ONE module instead of separate `?vue&type=...` sub-requests. The
// sub-request split exists so Vite can route each block through its own plugin
// chain (TS through esbuild, styles through the CSS pipeline); here the caller
// already does both to the combined output, and plugin-vue itself inlines the
// script this way whenever it can (`canInlineMain`).
async function compileVue(source, filename, root) {
  const module = await importFromProject(root, "@vue/compiler-sfc", "the Vue SFC compiler");
  const compiler = module.default ?? module;

  const { descriptor, errors } = compiler.parse(source, { filename, sourceMap: false });
  if (errors.length) {
    throw new Error(errors.map((error) => error.message ?? String(error)).join("\n"));
  }
  // plugin-vue's production component id: hash(root-relative path + source).
  // It is the `data-v-xxxxxxxx` scope attribute, so it must be derived the same
  // way or scoped styles land on a different attribute than the app expects.
  const id = shortHash(slash(path.relative(root, filename)) + source);
  const hasScoped = descriptor.styles.some((style) => style.scoped);
  const lang = descriptor.scriptSetup?.lang ?? descriptor.script?.lang;
  const isTypescript = !!lang && /tsx?$/.test(lang);

  const templateOptions = {
    id,
    ast: descriptor.template?.ast,
    filename,
    scoped: hasScoped,
    slotted: descriptor.slotted,
    isProd: true,
    ssr: false,
    ssrCssVars: descriptor.cssVars,
    // A production build resolves absolute asset URLs too (plugin-vue passes
    // `includeAbsolute: true` whenever there is no dev server).
    transformAssetUrls: { includeAbsolute: true },
    compilerOptions: {
      scopeId: hasScoped ? `data-v-${id}` : undefined,
      expressionPlugins: isTypescript ? ["typescript"] : [],
      sourceMap: false,
    },
  };

  // `<script setup>` compiles the template INTO the setup function; a plain
  // `<script>` (or a `<template src>`) keeps a separate render function.
  const inlineTemplate = !!descriptor.scriptSetup && !descriptor.template?.src;
  const attachedProps = [];
  const output = [];

  let resolvedScript = null;
  if (descriptor.script || descriptor.scriptSetup) {
    if (descriptor.script?.src || descriptor.scriptSetup?.src) {
      throw new Error(
        "`<script src=\"...\">` in a Vue SFC is not supported by diffpack yet; " +
          "move the script into the component or import the file from it",
      );
    }
    resolvedScript = compiler.compileScript(descriptor, {
      id,
      isProd: true,
      inlineTemplate,
      templateOptions,
      sourceMap: false,
      genDefaultAs: "_sfc_main",
    });
    output.push(resolvedScript.content);
  } else {
    output.push("const _sfc_main = {}");
  }

  if (descriptor.template && !inlineTemplate) {
    if (descriptor.template.src) {
      throw new Error(
        "`<template src=\"...\">` in a Vue SFC is not supported by diffpack yet; " +
          "inline the template in the component",
      );
    }
    if (descriptor.template.lang && descriptor.template.lang !== "html") {
      throw new Error(
        `<template lang="${descriptor.template.lang}"> needs a template preprocessor diffpack does not run`,
      );
    }
    const compiled = compiler.compileTemplate({
      ...templateOptions,
      source: descriptor.template.content,
      compilerOptions: {
        ...templateOptions.compilerOptions,
        bindingMetadata: resolvedScript?.bindings,
      },
    });
    if (compiled.errors.length) {
      throw new Error(compiled.errors.map((error) => error.message ?? String(error)).join("\n"));
    }
    output.push(compiled.code.replace(/\nexport (function|const) (render|ssrRender)/, "\n$1 _sfc_$2"));
    attachedProps.push(["render", "_sfc_render"]);
  }

  let css = "";
  for (let index = 0; index < descriptor.styles.length; index += 1) {
    const style = descriptor.styles[index];
    if (style.src) {
      throw new Error(
        "`<style src=\"...\">` in a Vue SFC is not supported by diffpack yet; " +
          "import the stylesheet from the component's script instead",
      );
    }
    if (style.module) {
      throw new Error(
        "`<style module>` (CSS Modules inside an SFC) is not supported by diffpack yet",
      );
    }
    const compiled = await compiler.compileStyleAsync({
      filename,
      id: `data-v-${id}`,
      isProd: true,
      source: style.content,
      scoped: style.scoped,
      preprocessLang: style.lang,
      trim: true,
    });
    if (compiled.errors.length) {
      throw new Error(compiled.errors.map((error) => error.message ?? String(error)).join("\n"));
    }
    css += `${compiled.code}\n`;
  }

  if (descriptor.customBlocks.length) {
    const kinds = [...new Set(descriptor.customBlocks.map((block) => block.type))].join(", ");
    throw new Error(
      `custom SFC block(s) <${kinds}> need a Vite plugin to give them meaning; diffpack hosts none`,
    );
  }

  if (hasScoped) attachedProps.push(["__scopeId", JSON.stringify(`data-v-${id}`)]);

  if (attachedProps.length === 0) {
    output.push("export default _sfc_main");
  } else {
    // plugin-vue imports this two-line helper from a virtual module; inlining it
    // keeps the component a single self-contained module.
    output.push(
      "const __diffpack_export_sfc = (sfc, props) => {\n" +
        "  const target = sfc.__vccOpts || sfc\n" +
        "  for (const [key, val] of props) { target[key] = val }\n" +
        "  return target\n" +
        "}",
    );
    const pairs = attachedProps.map(([key, value]) => `['${key}',${value}]`).join(",");
    output.push(`export default /*#__PURE__*/__diffpack_export_sfc(_sfc_main, [${pairs}])`);
  }

  return { code: output.join("\n"), css, language: isTypescript ? "ts" : "js" };
}

// --- Svelte ---------------------------------------------------------------
//
// Mirrors @sveltejs/vite-plugin-svelte for a production client build: the
// project's `svelte.config.js` supplies `preprocess` and `compilerOptions`, and
// production forces `dev: false`, `hmr: false`, `css: "external"` (styles are
// returned separately instead of injected by the component at runtime).
// Svelte 5's compiler strips TypeScript annotations itself, which is why the
// create-vite `svelte-ts` template ships an empty `svelte.config.js`.
async function loadSvelteConfig(root) {
  for (const name of ["svelte.config.js", "svelte.config.mjs", "svelte.config.cjs"]) {
    const candidate = path.join(root, name);
    if (existsSync(candidate)) {
      const loaded = await import(pathToFileURL(candidate).href);
      return loaded.default ?? loaded;
    }
  }
  return {};
}

async function compileSvelte(source, filename, root) {
  const compiler = await importFromProject(root, "svelte/compiler", "the Svelte compiler");
  const config = await loadSvelteConfig(root);

  let code = source;
  if (config.preprocess) {
    const processed = await compiler.preprocess(source, config.preprocess, { filename });
    code = processed.code;
  }

  const compiled = compiler.compile(code, {
    ...(config.compilerOptions ?? {}),
    filename,
    generate: "client",
    dev: false,
    hmr: false,
    css: "external",
  });

  return {
    code: compiled.js.code,
    css: compiled.css?.code ?? "",
    // The Svelte compiler emits plain JavaScript; TypeScript in `<script
    // lang="ts">` is already stripped by the compiler itself.
    language: "js",
  };
}

async function main() {
  const framework = process.env.DIFFPACK_SFC_FRAMEWORK;
  const filename = process.env.DIFFPACK_SFC_FILE;
  const root = process.env.DIFFPACK_SFC_ROOT;
  if (!framework || !filename || !root) {
    throw new Error(
      "SFC runner: DIFFPACK_SFC_FRAMEWORK / DIFFPACK_SFC_FILE / DIFFPACK_SFC_ROOT not set",
    );
  }
  const source = readFileSync(0, "utf8");
  let result;
  if (framework === "vue") {
    result = await compileVue(source, filename, root);
  } else if (framework === "svelte") {
    result = await compileSvelte(source, filename, root);
  } else {
    throw new Error(`SFC runner: unknown framework "${framework}"`);
  }
  process.stdout.write(JSON.stringify(result));
}

main().catch((error) => {
  const message = (error && (error.message || error.stack)) || String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
});
