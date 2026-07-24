import { watch } from "node:fs";
import { relative, resolve, sep } from "node:path";
import type { Browser } from "playwright";
import { globFiles, loadSeeds, loadWorldHandle, type BatConfig } from "./config.js";
import { runFlowFile } from "./runner/run.js";
import { renderReport } from "./runner/trace.js";

/**
 * `bat watch` — rerun affected flows when e2e files change.
 * - a .flow file changed  -> rerun that flow
 * - a seed / world module / bat.config.json changed -> rerun everything
 * Changes arriving mid-run are queued and coalesced; .bat/, node_modules/ and
 * build dirs are ignored (runs write into .bat — reacting to that would loop).
 */

const IGNORED = [`${sep}.bat${sep}`, `${sep}node_modules${sep}`, `${sep}.next${sep}`, `${sep}.svelte-kit${sep}`, `${sep}dist${sep}`, `${sep}.git${sep}`];

export async function watchFlows(config: BatConfig, browser: Browser): Promise<never> {
  const seedsRef = { current: await loadSeeds(config) };
  const world = await loadWorldHandle(config);

  const dirty = new Set<string>(); // absolute flow paths, or "*" for everything
  let running = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const runDirty = async () => {
    if (running) return;
    running = true;
    try {
      while (dirty.size > 0) {
        const wantAll = dirty.has("*");
        const files = wantAll ? await globFiles(config.root, config.flows) : [...dirty];
        dirty.clear();
        if (wantAll) {
          // seeds/world may have changed — reload the registry
          seedsRef.current = await loadSeeds(config).catch((e) => {
            console.error(String(e instanceof Error ? e.message : e));
            return seedsRef.current;
          });
        }
        for (const file of files.sort()) {
          const started = Date.now();
          try {
            const { trace } = await runFlowFile(file, { config, world, seeds: seedsRef.current, browser });
            if (trace.status === "pass") {
              console.log(`✓ ${trace.flow} (${Date.now() - started}ms)`);
            } else {
              console.log(renderReport(trace));
            }
          } catch (e) {
            console.error(`✗ ${relative(config.root, file)}: ${e instanceof Error ? e.message : String(e)}`);
          }
        }
        console.log(`\nwatching ${relative(process.cwd(), config.root) || "."} — edit a .flow/.seed/world file to rerun`);
      }
    } finally {
      running = false;
      if (dirty.size > 0) void runDirty();
    }
  };

  const schedule = () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => void runDirty(), 300);
  };

  watch(config.root, { recursive: true }, (_event, filename) => {
    if (!filename) return;
    const abs = resolve(config.root, filename.toString());
    if (IGNORED.some((seg) => abs.includes(seg))) return;
    if (abs.endsWith(".flow")) {
      dirty.add(abs);
      schedule();
    } else if (abs.endsWith(".seed.ts") || abs.endsWith("bat.config.json") || abs === resolve(config.root, config.world.module ?? "")) {
      dirty.add("*");
      schedule();
    }
  });

  console.log(`bat watch: initial run of all flows…`);
  dirty.add("*");
  await runDirty();
  return new Promise<never>(() => {});
}
