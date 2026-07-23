import { WorldError, canonicalJson, checkSchemas, deepEqual, isRef } from "./algebra.js";
import type {
  AppliedWorld,
  CapabilityLevel,
  EntityDef,
  FactRow,
  IdMap,
  InstallCtx,
  Ref,
  VerificationReport,
  WorldAdapter,
  WorldDescription,
} from "./types.js";

export function defineWorld(spec: Omit<WorldAdapter, "kind">): WorldAdapter {
  if (typeof spec.reset !== "function") {
    throw new WorldError(["defineWorld: `reset` is required — bat rebuilds every world from empty"]);
  }
  if (!spec.entities || typeof spec.entities !== "object") {
    throw new WorldError(["defineWorld: `entities` is required (may be empty for a stateless app)"]);
  }
  for (const [type, def] of Object.entries(spec.entities)) {
    if (typeof def.install !== "function") {
      throw new WorldError([`defineWorld: entity "${type}" needs an install(rows, ctx) function`]);
    }
  }
  if ((spec.snapshot && !spec.restore) || (!spec.snapshot && spec.restore)) {
    throw new WorldError(["defineWorld: `snapshot` and `restore` must be provided together (L4)"]);
  }
  return { kind: "bat.world", ...spec };
}

/** Topological order over `needs`, restricted to types present in the description. */
export function installOrder(facts: Record<string, unknown>, entities: Record<string, EntityDef>): string[] {
  const present = Object.keys(facts);
  const problems: string[] = [];
  for (const type of present) {
    if (!entities[type]) {
      const known = Object.keys(entities).map((t) => `"${t}"`).join(", ") || "(none)";
      problems.push(`no installer for entity type "${type}" — defineWorld knows: ${known}`);
    }
  }
  if (problems.length) throw new WorldError(problems);

  const order: string[] = [];
  const state = new Map<string, "visiting" | "done">();
  const visit = (type: string, chain: string[]) => {
    const s = state.get(type);
    if (s === "done") return;
    if (s === "visiting") {
      throw new WorldError([`entity dependency cycle: ${[...chain, type].join(" -> ")}`]);
    }
    state.set(type, "visiting");
    for (const dep of entities[type]?.needs ?? []) {
      if (!entities[dep]) {
        throw new WorldError([`entity "${type}" declares needs: ["${dep}"] but no such entity is defined`]);
      }
      // Only order over types actually present in this world.
      if (facts[dep] !== undefined) visit(dep, [...chain, type]);
    }
    state.set(type, "done");
    order.push(type);
  };
  for (const type of present.sort()) visit(type, []);
  return order;
}

function makeCtx(ids: Record<string, IdMap>): InstallCtx {
  function id(refOrType: Ref | string, key?: string): unknown {
    let type: string, k: string;
    if (typeof refOrType === "string") {
      type = refOrType;
      k = key!;
    } else if (isRef(refOrType)) {
      [type, k] = refOrType.$ref;
    } else {
      throw new WorldError([`ctx.id() called with ${canonicalJson(refOrType)} — pass a ref(...) or (type, key)`]);
    }
    const map = ids[type];
    if (!map || !(k in map)) {
      throw new WorldError([
        `ctx.id(${type}, "${k}") cannot resolve: the "${type}" installer ` +
          (map ? `returned no id for "${k}"` : `has not run or returned no ids`) +
          ` — return { "${k}": <id> } from install() to make refs to it resolvable`,
      ]);
    }
    return map[k];
  }
  return { id: id as InstallCtx["id"] };
}

export function capabilityLevel(adapter: WorldAdapter): { level: CapabilityLevel; missing: string[] } {
  const entityTypes = Object.keys(adapter.entities);
  const withSchema = entityTypes.filter((t) => adapter.entities[t]!.schema);
  const withRead = entityTypes.filter((t) => adapter.entities[t]!.read);
  const missing: string[] = [];
  let level: CapabilityLevel = 0;
  const allSchema = entityTypes.length > 0 && withSchema.length === entityTypes.length;
  const allRead = entityTypes.length > 0 && withRead.length === entityTypes.length;
  if (allSchema) level = 1;
  else missing.push(`L1: add schema() to ${entityTypes.filter((t) => !adapter.entities[t]!.schema).map((t) => `"${t}"`).join(", ") || "your entities"}`);
  if (allSchema && allRead) level = 2;
  else if (!allRead) missing.push(`L2: add read() to ${entityTypes.filter((t) => !adapter.entities[t]!.read).map((t) => `"${t}"`).join(", ") || "your entities"}`);
  if (level === 2 && adapter.fingerprint) level = 3;
  else if (!adapter.fingerprint) missing.push("L3: add fingerprint() — bat will detect world drift at flow boundaries");
  if (level === 3 && adapter.snapshot && adapter.restore) level = 4;
  else if (!adapter.snapshot) missing.push("L4: add snapshot()/restore() — content-addressed world cache + single-step replay");
  return { level, missing };
}

/**
 * apply(W) = reset to empty, install facts in dependency order, then verify
 * as hard as the adapter allows. Every guarantee is recorded as proven or asserted.
 */
export async function applyWorld(adapter: WorldAdapter, description: WorldDescription): Promise<AppliedWorld> {
  const proven: string[] = [];
  const asserted: string[] = [];

  // L1 — validate shapes before touching anything.
  const schemas = Object.fromEntries(
    Object.entries(adapter.entities).map(([t, d]) => [t, d.schema?.bind(d)]),
  );
  const schemaProblems = checkSchemas(description.facts, schemas);
  if (schemaProblems.length) throw new WorldError(schemaProblems.map((p) => `[pre-install] ${p}`));
  const schemaTypes = Object.keys(description.facts).filter((t) => adapter.entities[t]?.schema);
  if (schemaTypes.length) proven.push(`fact shapes validated for: ${schemaTypes.join(", ")}`);
  const unschemad = Object.keys(description.facts).filter((t) => !adapter.entities[t]?.schema);
  if (unschemad.length) asserted.push(`fact shapes NOT validated (no schema): ${unschemad.join(", ")}`);

  const order = installOrder(description.facts, adapter.entities);

  try {
    await adapter.reset();
  } catch (e) {
    throw new WorldError([`world reset() threw: ${e instanceof Error ? e.message : String(e)}`]);
  }

  const ids: Record<string, IdMap> = {};
  const ctx = makeCtx(ids);
  for (const type of order) {
    const rows = description.facts[type]!;
    try {
      const returned = await adapter.entities[type]!.install(rows, ctx);
      if (returned) ids[type] = returned;
    } catch (e) {
      if (e instanceof WorldError) throw e;
      throw new WorldError([
        `installer for "${type}" threw: ${e instanceof Error ? e.message : String(e)} ` +
          `(installing keys: ${Object.keys(rows).map((k) => `"${k}"`).join(", ")})`,
      ]);
    }
  }

  // L2 — read back and diff.
  const diffProblems: string[] = [];
  for (const type of order) {
    const def = adapter.entities[type]!;
    if (!def.read) {
      asserted.push(`"${type}" install NOT verified (no read())`);
      continue;
    }
    const keys = Object.keys(description.facts[type]!);
    let got: Record<string, FactRow>;
    try {
      got = await def.read(keys);
    } catch (e) {
      throw new WorldError([`read() for "${type}" threw: ${e instanceof Error ? e.message : String(e)}`]);
    }
    for (const key of keys) {
      const described = description.facts[type]![key]!;
      const actual = got[key];
      if (actual === undefined) {
        diffProblems.push(`${type}/"${key}": described but absent from the world after install`);
        continue;
      }
      for (const [field, value] of Object.entries(described)) {
        if (isRef(value)) continue; // refs live in id-space; skipped in read-back diff (documented)
        if (!(field in actual)) {
          diffProblems.push(`${type}/"${key}".${field}: described as ${canonicalJson(value)} but read() returned no such field`);
        } else if (!deepEqual(actual[field], value)) {
          diffProblems.push(
            `${type}/"${key}".${field}: described ${canonicalJson(value)}, world contains ${canonicalJson(actual[field])}`,
          );
        }
      }
    }
    if (!diffProblems.length) proven.push(`"${type}" verified by read-back (${keys.length} facts)`);
  }
  if (diffProblems.length) {
    throw new WorldError(diffProblems.map((p) => `[read-back] ${p}`));
  }

  const { level } = capabilityLevel(adapter);
  if (adapter.fingerprint) proven.push("world fingerprint available for drift detection");
  else asserted.push("no fingerprint(): drift between flows cannot be detected");

  return { description, ids, verification: { level, proven, asserted } };
}

export interface DoctorReport {
  level: CapabilityLevel;
  levelName: string;
  proven: string[];
  nextRungs: string[];
}

const LEVEL_NAMES: Record<CapabilityLevel, string> = {
  0: "L0 trust-me",
  1: "L1 validated",
  2: "L2 verified",
  3: "L3 drift-guarded",
  4: "L4 time-travel",
};

export function doctor(adapter: WorldAdapter): DoctorReport {
  const { level, missing } = capabilityLevel(adapter);
  const proven: string[] = ["algebra: merge conflicts, dangling refs, patch targets (always on)"];
  if (level >= 1) proven.push("L1: fact shapes validated before install");
  if (level >= 2) proven.push("L2: installs verified by read-back diff");
  if (level >= 3) proven.push("L3: world drift detected at flow boundaries");
  if (level >= 4) proven.push("L4: world snapshots — content-addressed cache + single-step replay");
  return { level, levelName: LEVEL_NAMES[level], proven, nextRungs: missing };
}
