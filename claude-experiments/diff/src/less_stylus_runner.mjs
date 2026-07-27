// Compiles one Less or Stylus source to plain CSS using the APP's OWN `less` /
// `stylus` package (resolved from the project's node_modules, cwd = project
// root), then hands the result back to the native CSS pipeline. This mirrors
// how `crate::sass` produces plain CSS before the shared CSS-Modules / @import /
// url() rebasing stages run.
//
// Contract (values from `crate::less_stylus`):
//   env DIFFPACK_CSS_TOOL   "less" | "stylus"
//   env DIFFPACK_CSS_FILE   the source file's absolute path (for @import roots
//                           and error messages)
//   stdin                   the Less/Stylus source
//   stdout                  a single JSON object { "css": string, "deps": [..] }
//                           where deps are the other files pulled in via @import
//                           (recorded by the caller so edits invalidate).

import { readFileSync } from "node:fs";
import { dirname } from "node:path";

async function compileLess(source, file) {
  const mod = await import("less");
  const less = mod.default || mod;
  const output = await less.render(source, {
    filename: file,
    paths: [dirname(file)],
    // Keep url()s untouched here; the native pipeline rebases them relative to
    // the source file, exactly as it does for a hand-written stylesheet.
    rewriteUrls: "off",
  });
  return { css: output.css, deps: output.imports || [] };
}

async function compileStylus(source, file) {
  const mod = await import("stylus");
  const stylus = mod.default || mod;
  const style = stylus(source).set("filename", file).set("paths", [dirname(file)]);
  const css = await new Promise((resolve, reject) => {
    style.render((error, css) => (error ? reject(error) : resolve(css)));
  });
  let deps = [];
  if (typeof style.deps === "function") {
    deps = style.deps();
  }
  return { css, deps };
}

async function main() {
  const tool = process.env.DIFFPACK_CSS_TOOL;
  const file = process.env.DIFFPACK_CSS_FILE;
  if (!tool || !file) {
    throw new Error("less/stylus runner: DIFFPACK_CSS_TOOL / DIFFPACK_CSS_FILE not set");
  }
  const source = readFileSync(0, "utf8");
  let result;
  if (tool === "less") {
    result = await compileLess(source, file);
  } else if (tool === "stylus") {
    result = await compileStylus(source, file);
  } else {
    throw new Error(`less/stylus runner: unknown tool "${tool}"`);
  }
  process.stdout.write(JSON.stringify(result));
}

main().catch((error) => {
  const message = (error && (error.message || error.stack)) || String(error);
  process.stderr.write(message + "\n");
  process.exit(1);
});
