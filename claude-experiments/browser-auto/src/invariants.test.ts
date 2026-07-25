import { readFile, readdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const SRC = join(dirname(fileURLToPath(import.meta.url)));

async function tsFiles(dir: string): Promise<string[]> {
  const out: string[] = [];
  for (const name of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, name.name);
    if (name.isDirectory()) out.push(...(await tsFiles(full)));
    else if (name.name.endsWith(".ts") && !name.name.endsWith(".test.ts")) out.push(full);
  }
  return out;
}

/**
 * Structural invariants that protect bat's no-flake guarantee — the kind of
 * rule a run-once behavioral test cannot see, but a code-shape test can.
 */
describe("runtime invariants (structural)", () => {
  it("the NetworkTracker soundness guard (tracker-before-navigation) is present", async () => {
    // WHY: settlement is only sound if the tracker observes a page's requests
    // from BEFORE it navigates. A tracker attached after a goto can miss the
    // page's initial fetch, return early, and leak an unfinished render into
    // the next step — a real intermittent flake the "--fast replay" path hid
    // for many commits. The invariant is enforced at RUNTIME in the tracker's
    // constructor (throws if the page already navigated); this test guards the
    // guard, so it can't be silently deleted. The behavioral proof that it
    // actually fires lives in e2e.test.ts.
    const settle = await readFile(join(SRC, "runner/settle.ts"), "utf8");
    expect(settle).toMatch(/SOUNDNESS INVARIANT/);
    expect(settle).toMatch(/already-navigated page/);
    // the guard must be at the very top of the constructor, before listeners attach
    const ctor = /constructor\(page: Page[^)]*\) \{([\s\S]*?)page\.on\("request"/.exec(settle)?.[1] ?? "";
    expect(ctor, "the guard must run before any page.on(...) listener").toMatch(/throw new Error/);
  });

  it("the DSL grammar has no duration/wait tokens (the core no-timing axiom)", async () => {
    // WHY: the whole thesis is that a flow cannot encode time. If a wait/sleep/
    // timeout/retry keyword ever leaked into the parser it would reopen the door.
    const parser = await readFile(join(SRC, "dsl/parser.ts"), "utf8");
    // these must never appear as accepted keywords in ACTION_WORDS / effect verbs
    const actionWords = /const ACTION_WORDS = \[([^\]]*)\]/.exec(parser)?.[1] ?? "";
    for (const banned of ["wait", "sleep", "timeout", "retry", "delay"]) {
      expect(actionWords, `'${banned}' must not be an action keyword`).not.toContain(`"${banned}"`);
    }
  });
});
