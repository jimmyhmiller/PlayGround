// Evaluate a legacy (v3-style) `tailwind.config.js` referenced by a Tailwind v4
// `@config` directive, and print the equivalent v4 `@theme { … }` + `@keyframes`
// CSS on stdout — so diffpack's native v4 compiler can merge the config's custom
// design tokens (fonts, colors, animations, keyframes, spacing, …) into its theme.
//
// Loaded via jiti (from the app's own node_modules) so a config that mixes ESM
// `import` with `module.exports` (Tailwind allows this) evaluates correctly.
//
//   node tailwind-config-eval.mjs <abs path to tailwind.config.js>
//
// Unmapped theme categories are reported on stderr (never silently dropped).
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const configPath = process.argv[2];
if (!configPath) {
  console.error('usage: tailwind-config-eval.mjs <config-path>');
  process.exit(1);
}

// jiti (and the config's own deps like `tailwindcss/defaultTheme`) live in the APP's
// node_modules, not this script's — resolve them relative to the config file.
const appRequire = createRequire(configPath);
const { createJiti } = await import(pathToFileURL(appRequire.resolve('jiti')).href);
const jiti = createJiti(pathToFileURL(configPath).href, { interopDefault: true });
const loaded = await jiti.import(configPath);
const config = (loaded && loaded.default) || loaded || {};
const theme = config.theme || {};

// v3 config theme key -> v4 theme-variable namespace (the ones Tailwind's compat
// layer maps). `keyframes` and `fontFamily` are handled specially below.
const NS = {
  colors: 'color',
  fontFamily: 'font',
  fontSize: 'text',
  fontWeight: 'font-weight',
  letterSpacing: 'tracking',
  lineHeight: 'leading',
  spacing: 'spacing',
  screens: 'breakpoint',
  borderRadius: 'radius',
  boxShadow: 'shadow',
  animation: 'animate',
  transitionTimingFunction: 'ease',
  transitionDuration: 'duration',
  aspectRatio: 'aspect',
  blur: 'blur',
  perspective: 'perspective',
  columns: 'container',
};

// Merge base `theme.<cat>` (overrides) and `theme.extend.<cat>` (additions) — the
// extend wins on key collisions, matching Tailwind's resolution for our purposes.
const cats = {};
for (const [cat, val] of Object.entries(theme)) {
  if (cat === 'extend') continue;
  cats[cat] = { ...(cats[cat] || {}), ...(isPlainObject(val) ? val : { DEFAULT: val }) };
}
for (const [cat, val] of Object.entries(theme.extend || {})) {
  cats[cat] = { ...(cats[cat] || {}), ...(isPlainObject(val) ? val : { DEFAULT: val }) };
}

const themeLines = [];
let keyframesCss = '';
const unmapped = [];

for (const [cat, obj] of Object.entries(cats)) {
  if (cat === 'keyframes') {
    for (const [name, steps] of Object.entries(obj)) {
      keyframesCss += `@keyframes ${name} {\n`;
      for (const [stop, decls] of Object.entries(steps)) {
        const body = Object.entries(decls)
          .map(([prop, v]) => `${kebab(prop)}: ${v}`)
          .join('; ');
        keyframesCss += `  ${stop} { ${body} }\n`;
      }
      keyframesCss += `}\n`;
    }
    continue;
  }
  const ns = NS[cat];
  if (!ns) { unmapped.push(cat); continue; }
  flatten(obj, [], (path, value) => {
    // `DEFAULT` collapses to the bare namespace (`--radius: …`); nested keys join
    // with `-` (`colors.brand.500` -> `--color-brand-500`).
    const suffix = path.filter((p) => p !== 'DEFAULT').join('-');
    const varName = suffix ? `--${ns}-${suffix}` : `--${ns}`;
    themeLines.push(`  ${varName}: ${value};`);
  });
}

let out = '';
if (themeLines.length) out += `@theme {\n${themeLines.join('\n')}\n}\n`;
out += keyframesCss;
process.stdout.write(out);
if (unmapped.length) {
  console.error(`tailwind @config: theme categories not mapped to v4 (ignored): ${unmapped.join(', ')}`);
}

function isPlainObject(v) {
  return v && typeof v === 'object' && !Array.isArray(v);
}
function kebab(s) {
  return s.replace(/[A-Z]/g, (m) => '-' + m.toLowerCase());
}
// Walk a theme category object to its leaves. An array leaf (font stacks) joins
// with `, `; a function value (rare, references other tokens) is skipped.
function flatten(obj, path, emit) {
  for (const [key, val] of Object.entries(obj)) {
    const next = [...path, key];
    if (Array.isArray(val)) emit(next, val.join(', '));
    else if (isPlainObject(val)) flatten(val, next, emit);
    else if (typeof val !== 'function') emit(next, String(val));
  }
}
