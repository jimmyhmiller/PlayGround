import { readFile, readdir, stat } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { Seed, WorldAdapter } from "./world/types.js";
import { WorldError } from "./world/algebra.js";
import { httpWorldHandle, localWorldHandle, type WorldHandle } from "./runner/world-handle.js";

export interface BatConfig {
  baseUrl: string;
  world: { module?: string; http?: string };
  seeds: string;
  flows: string;
  stepBudgetMs: number;
  headless: boolean;
  /** simulated bad conditions (seeded latency / failure injection) */
  conditions?: { latencyMs?: [number, number]; failRate?: number; seed: number };
  /** automatic reruns used to triage a failure (default 4; 0 = fast-path checks only) */
  rerunsOnFailure?: number;
  /** parallel workers (default 1). >1 requires an `app` spec — isolation is structural. */
  workers?: number;
  /** how bat launches an isolated app instance per worker; {port}/{index} substitute */
  app?: { command: string; readyUrl?: string; env?: Record<string, string>; startupTimeoutMs?: number };
  /** resolved project root (dir containing bat.config.json) */
  root: string;
}

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigError";
  }
}

const DEFAULTS = {
  seeds: "./e2e/world/*.seed.{ts,js,mjs}",
  flows: "./e2e/flows/**/*.flow",
  stepBudgetMs: 15000,
  headless: true,
};

export async function loadConfig(cwd: string, overrides: Partial<BatConfig> = {}): Promise<BatConfig> {
  const path = join(cwd, "bat.config.json");
  let raw: string;
  try {
    raw = await readFile(path, "utf8");
  } catch {
    throw new ConfigError(
      `no bat.config.json found in ${cwd} — create one with at least: {"baseUrl": "http://localhost:3000", "world": {"module": "./e2e/world/world.ts"}}`,
    );
  }
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(stripJsonComments(raw)) as Record<string, unknown>;
  } catch (e) {
    throw new ConfigError(`bat.config.json is not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
  }
  if (typeof parsed.baseUrl !== "string") throw new ConfigError(`bat.config.json needs a "baseUrl" string`);
  const world = (parsed.world ?? {}) as { module?: string; http?: string };
  if (!world.module && !world.http) {
    throw new ConfigError(`bat.config.json needs "world": {"module": "./path/to/world.ts"} or {"http": "http://..."}`);
  }
  let conditions: BatConfig["conditions"];
  if (parsed.conditions !== undefined) {
    const c = parsed.conditions as { latencyMs?: unknown; failRate?: unknown; seed?: unknown };
    if (typeof c !== "object" || c === null || typeof c.seed !== "number") {
      throw new ConfigError(`"conditions" needs at least a numeric "seed" (runs must be reproducible)`);
    }
    conditions = c as BatConfig["conditions"];
  }
  return {
    baseUrl: parsed.baseUrl,
    world,
    seeds: typeof parsed.seeds === "string" ? parsed.seeds : DEFAULTS.seeds,
    flows: typeof parsed.flows === "string" ? parsed.flows : DEFAULTS.flows,
    stepBudgetMs: typeof parsed.stepBudgetMs === "number" ? parsed.stepBudgetMs : DEFAULTS.stepBudgetMs,
    headless: typeof parsed.headless === "boolean" ? parsed.headless : DEFAULTS.headless,
    ...(conditions ? { conditions } : {}),
    ...(typeof parsed.rerunsOnFailure === "number" ? { rerunsOnFailure: parsed.rerunsOnFailure } : {}),
    ...(typeof parsed.workers === "number" ? { workers: parsed.workers } : {}),
    ...(parsed.app && typeof (parsed.app as { command?: unknown }).command === "string"
      ? { app: parsed.app as NonNullable<BatConfig["app"]> }
      : {}),
    root: cwd,
    ...overrides,
  };
}

/** Per-worker view of the config: {port}/{index} resolved in baseUrl and world.http. */
export function resolveWorkerConfig(config: BatConfig, vars: { port: number; index: number }): BatConfig {
  const sub = (s: string) => s.replace(/\{port\}/g, String(vars.port)).replace(/\{index\}/g, String(vars.index));
  return {
    ...config,
    baseUrl: sub(config.baseUrl),
    world: {
      ...config.world,
      ...(config.world.http ? { http: sub(config.world.http) } : {}),
    },
  };
}

function stripJsonComments(s: string): string {
  return s.replace(/^\s*\/\/.*$/gm, "");
}

export interface WorkerEnvInfo {
  index: number;
  baseUrl: string;
  port: number | null;
}

export async function loadWorldHandle(config: BatConfig, workerEnv?: WorkerEnvInfo): Promise<WorldHandle> {
  if (config.world.http) return httpWorldHandle(config.world.http);
  const modPath = isAbsolute(config.world.module!) ? config.world.module! : resolve(config.root, config.world.module!);
  let mod: Record<string, unknown>;
  try {
    mod = (await import(pathToFileURL(modPath).href)) as Record<string, unknown>;
  } catch (e) {
    throw new ConfigError(`could not import world module ${modPath}: ${e instanceof Error ? e.message : String(e)}`);
  }
  // per-worker isolation: a module may export createWorld(env) to bind each
  // worker slot to its own isolated state (db name, server instance, …)
  const factory = mod.createWorld;
  let adapter: WorldAdapter | undefined;
  if (typeof factory === "function") {
    adapter = (await factory(workerEnv ?? { index: 0, baseUrl: config.baseUrl, port: null })) as WorldAdapter;
  } else {
    adapter = (mod.default ?? mod.world) as WorldAdapter | undefined;
  }
  if (!adapter || adapter.kind !== "bat.world") {
    throw new ConfigError(`${modPath} must default-export defineWorld(...) or export createWorld(env) (got ${typeof adapter})`);
  }
  return localWorldHandle(adapter);
}

export async function loadSeeds(config: BatConfig): Promise<Map<string, Seed>> {
  const files = await globFiles(config.root, config.seeds);
  const registry = new Map<string, Seed>();
  for (const file of files) {
    let mod: Record<string, unknown>;
    try {
      mod = (await import(pathToFileURL(file).href)) as Record<string, unknown>;
    } catch (e) {
      throw new WorldError([`could not import seed file ${file}: ${e instanceof Error ? e.message : String(e)}`]);
    }
    const s = mod.default as Seed | undefined;
    if (!s || s.kind !== "bat.seed") {
      throw new WorldError([`${file} must default-export seed("name", {...})`]);
    }
    if (registry.has(s.name)) {
      throw new WorldError([`duplicate seed name "${s.name}" (${file})`]);
    }
    registry.set(s.name, s);
  }
  return registry;
}

/** Minimal glob: `*` within a segment, `**` any directories, `{a,b}` alternation. */
export async function globFiles(root: string, pattern: string): Promise<string[]> {
  const abs = isAbsolute(pattern) ? pattern : resolve(root, pattern);
  const regex = globToRegex(abs);
  // walk from the deepest non-wildcard prefix
  const parts = abs.split("/");
  const firstWild = parts.findIndex((p) => /[*{]/.test(p));
  const base = firstWild === -1 ? abs : parts.slice(0, firstWild).join("/") || "/";
  const out: string[] = [];
  await walk(base, regex, out);
  return out.sort();
}

async function walk(dir: string, regex: RegExp, out: string[]): Promise<void> {
  let entries: string[];
  try {
    entries = await readdir(dir);
  } catch {
    return;
  }
  for (const name of entries) {
    if (name === "node_modules" || name.startsWith(".")) continue;
    const full = join(dir, name);
    const st = await stat(full).catch(() => null);
    if (!st) continue;
    if (st.isDirectory()) await walk(full, regex, out);
    else if (regex.test(full)) out.push(full);
  }
}

function globToRegex(pattern: string): RegExp {
  let re = "";
  let i = 0;
  while (i < pattern.length) {
    const c = pattern[i]!;
    if (c === "*") {
      if (pattern[i + 1] === "*") {
        re += ".*";
        i += 2;
        if (pattern[i] === "/") i++;
      } else {
        re += "[^/]*";
        i++;
      }
    } else if (c === "{") {
      const end = pattern.indexOf("}", i);
      if (end === -1) {
        re += "\\{";
        i++;
      } else {
        const alts = pattern.slice(i + 1, end).split(",");
        re += `(${alts.map(escapeRe).join("|")})`;
        i = end + 1;
      }
    } else {
      re += escapeRe(c);
      i++;
    }
  }
  return new RegExp(`^${re}$`);
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
