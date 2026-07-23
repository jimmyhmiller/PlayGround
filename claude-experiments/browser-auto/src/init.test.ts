import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { initProject } from "./init.js";

describe("bat init", () => {
  it("scaffolds config, world, seed, and flow", async () => {
    const root = await mkdtemp(join(tmpdir(), "bat-init-"));
    const result = await initProject(root);
    expect(result.created).toEqual(["bat.config.json", "e2e/world/world.ts", "e2e/world/basic.seed.ts", "e2e/flows/home.flow"]);
    const world = await readFile(join(root, "e2e/world/world.ts"), "utf8");
    // stubs must throw hard errors with clear messages, never return silently
    expect(world).toContain("throw new Error");
    expect(world).toContain("reset() not implemented");
    const config = JSON.parse(await readFile(join(root, "bat.config.json"), "utf8")) as { baseUrl: string };
    expect(config.baseUrl).toContain("http://");
  });

  it("never overwrites existing files", async () => {
    const root = await mkdtemp(join(tmpdir(), "bat-init-"));
    await writeFile(join(root, "bat.config.json"), '{"custom": true}', "utf8");
    const result = await initProject(root);
    expect(result.skipped).toEqual(["bat.config.json"]);
    expect(await readFile(join(root, "bat.config.json"), "utf8")).toBe('{"custom": true}');
  });
});
