import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { composeWorld, mergeSeeds, seed, WorldError } from "./algebra.js";
import type { Facts, Seed } from "./types.js";

/**
 * Property-based verification of the world algebra's laws. These are not
 * examples — the laws hold for ALL fact sets, which is what makes
 * `given seed A/B/C` order-independent and therefore un-flaky by construction.
 */

const key = fc.stringMatching(/^[a-z][a-z0-9-]{0,8}$/);
const scalar = fc.oneof(
  fc.stringMatching(/^[a-zA-Z0-9 ]{0,12}$/),
  fc.integer({ min: -1000, max: 1000 }),
  fc.boolean(),
  fc.constant(null),
);
const row = fc.dictionary(key, scalar, { minKeys: 1, maxKeys: 4 });
const facts: fc.Arbitrary<Facts> = fc.dictionary(key, fc.dictionary(key, row, { minKeys: 1, maxKeys: 4 }), {
  minKeys: 1,
  maxKeys: 3,
});

let n = 0;
const arbSeed: fc.Arbitrary<Seed> = facts.map((f) => seed(`s${n++}`, f));

/** merge that may legitimately conflict — normalize to a comparable outcome */
function outcome(seeds: Seed[]): string {
  try {
    return composeWorld(seeds).fingerprint;
  } catch (e) {
    if (e instanceof WorldError) return "CONFLICT";
    throw e;
  }
}

describe("world algebra laws (property-based)", () => {
  it("⊕ is commutative: A ⊕ B ≡ B ⊕ A for ALL fact sets", () => {
    fc.assert(
      fc.property(arbSeed, arbSeed, (a, b) => {
        expect(outcome([a, b])).toBe(outcome([b, a]));
      }),
      { numRuns: 300 },
    );
  });

  it("⊕ is associative: (A ⊕ B) ⊕ C ≡ A ⊕ (B ⊕ C)", () => {
    fc.assert(
      fc.property(arbSeed, arbSeed, arbSeed, (a, b, c) => {
        expect(outcome([a, b, c])).toBe(outcome([c, b, a]));
      }),
      { numRuns: 300 },
    );
  });

  it("⊕ is idempotent: A ⊕ A ≡ A", () => {
    fc.assert(
      fc.property(arbSeed, (a) => {
        expect(outcome([a, a])).toBe(outcome([a]));
      }),
      { numRuns: 300 },
    );
  });

  it("fingerprints are canonical: key insertion order never matters", () => {
    fc.assert(
      fc.property(facts, (f) => {
        const reversed: Facts = Object.fromEntries(
          Object.entries(f)
            .reverse()
            .map(([t, rows]) => [
              t,
              Object.fromEntries(
                Object.entries(rows)
                  .reverse()
                  .map(([k, r]) => [k, Object.fromEntries(Object.entries(r).reverse())]),
              ),
            ]),
        );
        expect(outcome([seed("x", f)])).toBe(outcome([seed("x", reversed)]));
      }),
      { numRuns: 200 },
    );
  });

  it("conflicts are symmetric and always name both seeds", () => {
    fc.assert(
      fc.property(facts, key, key, scalar, scalar, (f, t, k, v1, v2) => {
        fc.pre(!Object.is(v1, v2));
        const a = seed("alpha", { ...f, [t]: { [k]: { field: v1 } } });
        const b = seed("beta", { ...f, [t]: { [k]: { field: v2 } } });
        const ab = mergeSeeds([a, b]);
        const ba = mergeSeeds([b, a]);
        expect(ab.problems.length).toBe(ba.problems.length);
        expect(ab.problems.length).toBeGreaterThan(0);
        expect(ab.problems[0]).toContain("alpha");
        expect(ab.problems[0]).toContain("beta");
      }),
      { numRuns: 200 },
    );
  });
});
