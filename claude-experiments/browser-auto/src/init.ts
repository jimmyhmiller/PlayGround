import { mkdir, writeFile, access } from "node:fs/promises";
import { join } from "node:path";

/**
 * `bat init` — scaffold a bat integration for an app directory:
 * bat.config.json, an L0 world adapter (whose operators THROW with clear
 * messages until implemented — never silent stubs), an example seed, and an
 * example flow. Prints doctor-style next steps.
 */

const CONFIG = `{
  "baseUrl": "http://localhost:3000",
  "world": { "module": "./e2e/world/world.ts" },
  "seeds": "./e2e/world/*.seed.ts",
  "flows": "./e2e/flows/**/*.flow",
  "stepBudgetMs": 15000
}
`;

function worldTemplate(batImport: string): string {
  return `import { defineWorld } from "${batImport}";

/**
 * bat world adapter for this app. The contract: bat rebuilds the world from
 * empty for every flow — \`reset\` makes it empty, installers realize facts.
 * Every operator you add buys a stronger CHECKED guarantee (run \`bat doctor\`):
 *   L1 schema() · L2 read() · L3 fingerprint() · L4 snapshot()/restore()
 *
 * If your app is stateless (no server-side world), an empty \`entities\` and a
 * no-op reset are legitimate — delete the throws and say so in a comment.
 */
export default defineWorld({
  reset: async () => {
    throw new Error(
      "bat world: reset() not implemented — make the app's world empty here (e.g. TRUNCATE your tables). " +
        "If this app has no server-side state, replace this throw with a comment saying so.",
    );
  },
  entities: {
    // one entry per entity type your seeds describe, e.g.:
    // users: {
    //   install: async (rows, ctx) => {
    //     const ids: Record<string, string> = {};
    //     for (const [key, row] of Object.entries(rows)) {
    //       ids[key] = await db.user.create(row);
    //     }
    //     return ids; // lets other facts ref("users", key)
    //   },
    //   schema: (row) => (typeof row.email === "string" ? null : "email must be a string"),
    //   read: async (keys) => ({ /* key -> description-shaped row */ }),
    // },
  },
  // session: async (userKey) => ({ cookies: [...] }),  // for \`given user X signed-in\`
});
`;
}

const SEED = `import { seed } from "browser-auto/world";

// Seeds are pure data describing a world — no code, no ordering, no mutation.
// Same-key facts across seeds must be identical (merge is checked); reference
// other facts with ref("type", "key").
export default seed("basic", {
  // users: {
  //   admin: { email: "admin@example.dev", password: "secret123" },
  // },
});
`;

const FLOW = `# Every step is an action plus the observable effects it must cause.
# There is no way to wait in this language — that's the point.
flow "home page loads"

# given seed "basic"
# given user "admin" signed-in

go /
  expect heading "Welcome"
`;

export interface InitResult {
  created: string[];
  skipped: string[];
  nextSteps: string[];
}

export async function initProject(root: string, opts: { batImport?: string } = {}): Promise<InitResult> {
  const created: string[] = [];
  const skipped: string[] = [];
  const batImport = opts.batImport ?? "browser-auto/world";

  const files: Array<[string, string]> = [
    ["bat.config.json", CONFIG],
    ["e2e/world/world.ts", worldTemplate(batImport)],
    ["e2e/world/basic.seed.ts", SEED.replace("browser-auto/world", batImport)],
    ["e2e/flows/home.flow", FLOW],
  ];

  for (const [rel, content] of files) {
    const path = join(root, rel);
    const exists = await access(path).then(() => true, () => false);
    if (exists) {
      skipped.push(rel);
      continue;
    }
    await mkdir(join(path, ".."), { recursive: true });
    await writeFile(path, content, "utf8");
    created.push(rel);
  }

  return {
    created,
    skipped,
    nextSteps: [
      "1. set baseUrl in bat.config.json to where the app runs",
      "2. implement reset() (and installers) in e2e/world/world.ts — it throws until you do",
      "3. describe a world in e2e/world/basic.seed.ts",
      "4. run `bat inspect <url>` on a page and write flows from its semantic tree",
      "5. `bat check` (static) then `bat run`; `bat doctor` names the next verification rung",
    ],
  };
}
