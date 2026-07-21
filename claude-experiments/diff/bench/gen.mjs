// Corpus generator, adapted from oracle/benchmark.mjs (liveModuleSource /
// liveDependencyIndices). The dependency graph and the runtime-observable
// value are IDENTICAL to the oracle's live-output graph: module `i` imports
// modules i*imports+1 .. i*imports+imports (a k-ary tree, so every module has
// exactly one importer), and exports
//   value_i = base_i + sum(value_child ...)
// with base_i = i. The entry (module 0) prints value_0, so a fresh corpus of
// n modules prints n*(n-1)/2. Every bundler's output is executed under node
// and must print this independently computed number before it may be timed.
//
// Two module-size profiles share that exact graph and value semantics:
//  - "tiny":      the oracle's original module bodies (a few import lines and
//                 one exported const) — parse/link overhead dominated.
//  - "realistic": ~2-4 KB of plausible TypeScript per module: an interface,
//                 string/object/array literals, exported functions (some live,
//                 some unused and tree-shakeable), typed helpers, comments.
import { mkdirSync, writeFileSync, openSync, closeSync, renameSync } from "node:fs";
import { join } from "node:path";

export function moduleName(index) {
  return `module-${String(index).padStart(8, "0")}.ts`;
}

export function dependencyIndices(index, moduleCount, importsPerModule) {
  const dependencies = [];
  for (let offset = 1; offset <= importsPerModule; offset += 1) {
    const child = index * importsPerModule + offset;
    if (child < moduleCount) dependencies.push(child);
  }
  return dependencies;
}

export function expectedFreshValue(moduleCount) {
  // Each module contributes its index exactly once (k-ary tree).
  return (moduleCount * (moduleCount - 1)) / 2;
}

export function editedModuleIndex(moduleCount) {
  return Math.floor(moduleCount / 2);
}

// Same content edit as oracle/benchmark.mjs: the edited module's base value
// changes from `index` to `moduleCount + index`, raising the printed value by
// exactly `moduleCount`.
export function expectedEditedValue(moduleCount) {
  return expectedFreshValue(moduleCount) + moduleCount;
}

export function moduleSource(profile, index, moduleCount, importsPerModule, baseValue) {
  if (profile === "tiny") {
    return tinyModuleSource(index, moduleCount, importsPerModule, baseValue);
  }
  if (profile === "realistic") {
    return realisticModuleSource(index, moduleCount, importsPerModule, baseValue);
  }
  throw new Error(`unknown profile: ${profile}`);
}

// Verbatim shape of oracle/benchmark.mjs liveModuleSource().
function tinyModuleSource(index, moduleCount, importsPerModule, baseValue) {
  const dependencies = dependencyIndices(index, moduleCount, importsPerModule);
  let source = "";
  for (const dependency of dependencies) {
    source += `import { value_${dependency} } from "./${moduleName(dependency)}";\n`;
  }
  const expression = [String(baseValue), ...dependencies.map((d) => `value_${d}`)].join(" + ");
  source += `export const value_${index}: number = ${expression};\n`;
  if (index === 0) source += "console.log(value_0);\n";
  return source;
}

const TAG_POOL = [
  "ingest", "billing", "search", "profile", "catalog", "checkout",
  "session", "metrics", "gateway", "cache", "router", "audit",
];

function realisticModuleSource(index, moduleCount, importsPerModule, baseValue) {
  const dependencies = dependencyIndices(index, moduleCount, importsPerModule);
  const tagA = TAG_POOL[index % TAG_POOL.length];
  const tagB = TAG_POOL[(index * 7 + 3) % TAG_POOL.length];
  const lines = [];
  lines.push(`// Module ${index}: synthetic ${tagA}/${tagB} feature slice.`);
  lines.push(`// Part of the diffpack competitive-benchmark realistic corpus; the`);
  lines.push(`// exported value_${index} participates in the runtime-verified sum.`);
  for (const dependency of dependencies) {
    lines.push(`import { value_${dependency} } from "./${moduleName(dependency)}";`);
  }
  lines.push("");
  lines.push(`interface Record_${index} {`);
  lines.push(`  id: number;`);
  lines.push(`  label: string;`);
  lines.push(`  score: number;`);
  lines.push(`  tags: readonly string[];`);
  lines.push(`  createdAt: number;`);
  lines.push(`}`);
  lines.push("");
  lines.push(`const LABEL_${index} = "corpus/${tagA}/${tagB}/module-${index}";`);
  lines.push(`const SETTINGS_${index} = {`);
  lines.push(`  id: ${index},`);
  lines.push(`  channel: "${tagA}-${tagB}",`);
  lines.push(`  retries: ${(index % 5) + 1},`);
  lines.push(`  timeoutMs: ${250 + (index % 17) * 50},`);
  lines.push(`  thresholds: { low: ${index % 10}, high: ${90 + (index % 10)}, step: ${1 + (index % 4)} },`);
  lines.push(`  tags: ["${tagA}", "${tagB}", "generated", "profile-${index % 8}"],`);
  lines.push(`  enabled: ${index % 3 === 0 ? "true" : "false"},`);
  lines.push(`};`);
  lines.push("");
  lines.push(`function clampScore_${index}(value: number): number {`);
  lines.push(`  if (value < SETTINGS_${index}.thresholds.low) return SETTINGS_${index}.thresholds.low;`);
  lines.push(`  if (value > SETTINGS_${index}.thresholds.high) return SETTINGS_${index}.thresholds.high;`);
  lines.push(`  return value;`);
  lines.push(`}`);
  lines.push("");
  lines.push(`export function makeRecord_${index}(seed: number): Record_${index} {`);
  lines.push(`  const score = clampScore_${index}(seed * ${(index % 7) + 2} + ${index % 13});`);
  lines.push(`  return {`);
  lines.push(`    id: seed,`);
  lines.push(`    label: \`\${LABEL_${index}}#\${seed}\`,`);
  lines.push(`    score,`);
  lines.push(`    tags: SETTINGS_${index}.tags,`);
  lines.push(`    createdAt: 1700000000000 + seed * 1000,`);
  lines.push(`  };`);
  lines.push(`}`);
  lines.push("");
  lines.push(`export function summarize_${index}(records: readonly Record_${index}[]): string {`);
  lines.push(`  const parts: string[] = [];`);
  lines.push(`  for (const record of records) {`);
  lines.push(`    parts.push(\`\${record.label}:\${record.score.toFixed(2)}\`);`);
  lines.push(`  }`);
  lines.push(`  return parts.join(" | ") || "empty-${tagA}-${index}";`);
  lines.push(`}`);
  lines.push("");
  lines.push(`export function selectTopRecords_${index}(records: readonly Record_${index}[], limit: number): Record_${index}[] {`);
  lines.push(`  const sorted = [...records].sort((left, right) => right.score - left.score);`);
  lines.push(`  return sorted.slice(0, Math.max(0, Math.min(limit, sorted.length)));`);
  lines.push(`}`);
  lines.push("");
  lines.push(`const KEYWORDS_${index} = [`);
  lines.push(`  "${tagA}-primary", "${tagB}-secondary", "region-${index % 12}",`);
  lines.push(`  "tier-${index % 4}", "cohort-${(index * 3) % 29}", "release-${2020 + (index % 6)}",`);
  lines.push(`];`);
  lines.push("");
  lines.push(`export function matchKeywords_${index}(query: string): string[] {`);
  lines.push(`  const needle = query.trim().toLowerCase();`);
  lines.push(`  if (needle.length === 0) return [];`);
  lines.push(`  return KEYWORDS_${index}.filter((keyword) => keyword.includes(needle));`);
  lines.push(`}`);
  lines.push("");
  lines.push(`export function mergeSettings_${index}(overrides: Partial<typeof SETTINGS_${index}>): typeof SETTINGS_${index} {`);
  lines.push(`  return { ...SETTINGS_${index}, ...overrides, thresholds: { ...SETTINGS_${index}.thresholds } };`);
  lines.push(`}`);
  lines.push("");
  lines.push(`function accumulate_${index}(seed: number, inputs: readonly number[]): number {`);
  lines.push(`  let total = seed;`);
  lines.push(`  for (const input of inputs) {`);
  lines.push(`    total += input;`);
  lines.push(`  }`);
  lines.push(`  return total;`);
  lines.push(`}`);
  lines.push("");
  const inputs = dependencies.map((d) => `value_${d}`).join(", ");
  lines.push(`export const value_${index}: number = accumulate_${index}(${baseValue}, [${inputs}]);`);
  if (index === 0) lines.push("console.log(value_0);");
  lines.push("");
  return lines.join("\n");
}

export function generateCorpus(directory, { profile, moduleCount, importsPerModule }) {
  mkdirSync(directory, { recursive: true });
  let bytes = 0;
  for (let index = 0; index < moduleCount; index += 1) {
    const source = moduleSource(profile, index, moduleCount, importsPerModule, index);
    bytes += Buffer.byteLength(source);
    writeFileSync(join(directory, moduleName(index)), source);
  }
  // Match oracle/benchmark.mjs: publish directory metadata before timing readers.
  const descriptor = openSync(directory, "r");
  closeSync(descriptor);
  return { sourceBytes: bytes };
}

// Content edit used by all incremental benchmarks. `edited` toggles between
// the edited state (base = index + moduleCount) and the original state.
// The write is atomic (temp file + rename, like an editor's atomic save) so a
// filesystem watcher can never observe a truncated/half-written module.
export function writeContentEdit(directory, { profile, moduleCount, importsPerModule }, edited) {
  const index = editedModuleIndex(moduleCount);
  const base = edited ? moduleCount + index : index;
  const temporary = join(directory, `.edit-${index}.tmp`);
  writeFileSync(temporary, moduleSource(profile, index, moduleCount, importsPerModule, base));
  renameSync(temporary, join(directory, moduleName(index)));
  return edited ? expectedEditedValue(moduleCount) : expectedFreshValue(moduleCount);
}
