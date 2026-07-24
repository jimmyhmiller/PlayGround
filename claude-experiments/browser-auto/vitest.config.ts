import { defineConfig } from "vitest/config";

// Evidence retention: every vitest run writes a full machine-readable result
// file (per-test status, errors, durations) to .test-logs/, timestamped.
// A failure must NEVER be unidentifiable after the fact — that is this
// project's entire thesis, and it applies to bat's own suite too.
const stamp = new Date().toISOString().replace(/[:.]/g, "-");

export default defineConfig({
  test: {
    reporters: [
      "default",
      ["json", { outputFile: `.test-logs/vitest-${stamp}.json` }],
    ],
  },
});
