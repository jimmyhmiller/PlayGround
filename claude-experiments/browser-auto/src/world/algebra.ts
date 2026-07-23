import { createHash } from "node:crypto";
import type { Facts, FactRow, Patch, Ref, Seed, WorldDescription } from "./types.js";

export class WorldError extends Error {
  constructor(public problems: string[]) {
    super(problems.length === 1 ? problems[0]! : `${problems.length} world problems:\n  - ${problems.join("\n  - ")}`);
    this.name = "WorldError";
  }
}

export function ref(type: string, key: string): Ref {
  return { $ref: [type, key] };
}

export function isRef(v: unknown): v is Ref {
  return (
    typeof v === "object" &&
    v !== null &&
    "$ref" in v &&
    Array.isArray((v as Ref).$ref) &&
    (v as Ref).$ref.length === 2 &&
    (v as Ref).$ref.every((x) => typeof x === "string")
  );
}

export function seed(name: string, facts: Facts): Seed {
  if (!name) throw new WorldError(["seed() requires a non-empty name"]);
  for (const [type, rows] of Object.entries(facts)) {
    if (typeof rows !== "object" || rows === null || Array.isArray(rows)) {
      throw new WorldError([`seed "${name}": entity type "${type}" must map keys to fact rows (got ${describe(rows)})`]);
    }
    for (const [key, row] of Object.entries(rows)) {
      if (typeof row !== "object" || row === null || Array.isArray(row)) {
        throw new WorldError([`seed "${name}": fact ${type}/"${key}" must be an object of fields (got ${describe(row)})`]);
      }
    }
  }
  return { kind: "bat.seed", name, facts };
}

function describe(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "an array";
  return typeof v;
}

export function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (typeof a !== "object" || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a as object).sort();
  const kb = Object.keys(b as object).sort();
  if (ka.length !== kb.length) return false;
  for (let i = 0; i < ka.length; i++) {
    if (ka[i] !== kb[i]) return false;
    if (!deepEqual((a as Record<string, unknown>)[ka[i]!], (b as Record<string, unknown>)[ka[i]!])) return false;
  }
  return true;
}

/**
 * W₁ ⊕ W₂ ⊕ … : commutative, associative, idempotent merge of seeds.
 * Same (type, key) with deep-equal rows dedupes; different rows is a hard error
 * naming both seeds. Never last-writer-wins.
 */
export function mergeSeeds(seeds: Seed[]): { facts: Facts; problems: string[] } {
  const facts: Facts = {};
  const provenance = new Map<string, string>(); // "type/key" -> seed name
  const problems: string[] = [];
  for (const s of seeds) {
    for (const [type, rows] of Object.entries(s.facts)) {
      const into = (facts[type] ??= {});
      for (const [key, row] of Object.entries(rows)) {
        const slot = `${type}/${key}`;
        const existing = into[key];
        if (existing === undefined) {
          into[key] = row;
          provenance.set(slot, s.name);
        } else if (!deepEqual(existing, row)) {
          problems.push(
            `merge conflict on ${type}/"${key}": seed "${provenance.get(slot)}" and seed "${s.name}" ` +
              `define it with different values (${canonicalJson(existing)} vs ${canonicalJson(row)})`,
          );
        }
      }
    }
  }
  return { facts, problems };
}

/** Ordered, explicit overrides applied after merge. Patching a missing fact/field-path is an error. */
export function applyPatches(facts: Facts, patches: Patch[]): { facts: Facts; problems: string[] } {
  const problems: string[] = [];
  const out: Facts = structuredCloneFacts(facts);
  for (const p of patches) {
    const rows = out[p.type];
    if (!rows) {
      problems.push(`patch targets ${p.type}/"${p.key}" but no seed defines any "${p.type}" facts`);
      continue;
    }
    const row = rows[p.key];
    if (!row) {
      const known = Object.keys(rows).slice(0, 8).map((k) => `"${k}"`).join(", ");
      problems.push(`patch targets ${p.type}/"${p.key}" which no seed defines (known keys: ${known})`);
      continue;
    }
    row[p.field] = p.value;
  }
  return { facts: out, problems };
}

function structuredCloneFacts(facts: Facts): Facts {
  return JSON.parse(JSON.stringify(facts)) as Facts;
}

/** Closure check: every ref inside every fact resolves to a merged fact. */
export function checkClosure(facts: Facts): string[] {
  const problems: string[] = [];
  for (const [type, rows] of Object.entries(facts)) {
    for (const [key, row] of Object.entries(rows)) {
      walkRefs(row, (r, path) => {
        const [rt, rk] = r.$ref;
        if (!facts[rt]?.[rk]) {
          problems.push(
            `${type}/"${key}" field "${path}" references ${rt}/"${rk}", which no seed defines`,
          );
        }
      });
    }
  }
  return problems;
}

export function walkRefs(value: unknown, visit: (r: Ref, path: string) => void, path = ""): void {
  if (isRef(value)) {
    visit(value, path);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((v, i) => walkRefs(v, visit, path ? `${path}[${i}]` : `[${i}]`));
  } else if (typeof value === "object" && value !== null) {
    for (const [k, v] of Object.entries(value)) {
      walkRefs(v, visit, path ? `${path}.${k}` : k);
    }
  }
}

/** Canonical JSON: sorted keys, stable across runs → hashable. */
export function canonicalJson(value: unknown): string {
  return JSON.stringify(sortValue(value));
}

function sortValue(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sortValue);
  if (typeof v === "object" && v !== null) {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(v).sort()) {
      out[k] = sortValue((v as Record<string, unknown>)[k]);
    }
    return out;
  }
  return v;
}

export function fingerprintOf(facts: Facts, patches: Patch[]): string {
  const hash = createHash("sha256")
    .update(canonicalJson({ facts, patches }))
    .digest("hex");
  return `sha256:${hash.slice(0, 12)}`;
}

/**
 * Merge seeds, apply patches, check closure. Throws WorldError with every
 * problem found (not just the first) so an agent can fix all at once.
 */
export function composeWorld(seeds: Seed[], patches: Patch[] = []): WorldDescription {
  const problems: string[] = [];
  const merged = mergeSeeds(seeds);
  problems.push(...merged.problems);
  const patched = applyPatches(merged.facts, patches);
  problems.push(...patched.problems);
  problems.push(...checkClosure(patched.facts));
  if (problems.length > 0) throw new WorldError(problems);
  return {
    facts: patched.facts,
    patches,
    sources: [...new Set(seeds.map((s) => s.name))].sort(),
    fingerprint: fingerprintOf(patched.facts, patches),
  };
}

/** L1: run adapter schemas over every fact. Returns problems (empty = valid). */
export function checkSchemas(
  facts: Facts,
  schemas: Record<string, ((row: FactRow, key: string) => string | null) | undefined>,
): string[] {
  const problems: string[] = [];
  for (const [type, rows] of Object.entries(facts)) {
    const schema = schemas[type];
    if (!schema) continue;
    for (const [key, row] of Object.entries(rows)) {
      let msg: string | null;
      try {
        msg = schema(row, key);
      } catch (e) {
        msg = `schema threw: ${e instanceof Error ? e.message : String(e)}`;
      }
      if (msg) problems.push(`${type}/"${key}" fails schema: ${msg}`);
    }
  }
  return problems;
}
