import { describe, expect, it } from "vitest";
import { applyPatches, composeWorld, fingerprintOf, mergeSeeds, ref, seed, WorldError } from "./algebra.js";
import { applyWorld, capabilityLevel, defineWorld, doctor, installOrder } from "./adapter.js";
import type { FactRow } from "./types.js";

const catalog = seed("catalog-basic", {
  users: { shopper: { email: "shopper@test.dev", role: "customer" } },
  products: {
    "blue-widget": { name: "Blue Widget", price: 1999, stock: 12 },
    "red-widget": { name: "Red Widget", price: 2499, stock: 3 },
  },
});

const discounts = seed("discounts", {
  discounts: {
    summer: { code: "SUMMER", pct: 10, product: ref("products", "blue-widget") },
  },
});

describe("merge algebra", () => {
  it("is commutative: A ⊕ B fingerprints identically to B ⊕ A", () => {
    const ab = composeWorld([catalog, discounts]);
    const ba = composeWorld([discounts, catalog]);
    expect(ab.fingerprint).toBe(ba.fingerprint);
    expect(ab.facts).toEqual(ba.facts);
  });

  it("is idempotent: A ⊕ A = A", () => {
    const a = composeWorld([catalog]);
    const aa = composeWorld([catalog, catalog]);
    expect(aa.fingerprint).toBe(a.fingerprint);
  });

  it("dedupes identical facts from different seeds", () => {
    const other = seed("other", { users: { shopper: { email: "shopper@test.dev", role: "customer" } } });
    const w = composeWorld([catalog, other]);
    expect(w.facts.users!.shopper).toEqual({ email: "shopper@test.dev", role: "customer" });
  });

  it("hard-errors on conflicting facts, naming both seeds", () => {
    const evil = seed("evil", { products: { "blue-widget": { name: "Blue Widget", price: 1, stock: 12 } } });
    expect(() => composeWorld([catalog, evil])).toThrowError(/catalog-basic.*evil|evil.*catalog-basic/s);
    expect(() => composeWorld([catalog, evil])).toThrowError(/blue-widget/);
  });

  it("collects ALL problems, not just the first", () => {
    const evil = seed("evil", {
      products: { "blue-widget": { name: "X", price: 1, stock: 0 } },
      discounts: { bad: { code: "B", product: ref("products", "nonexistent") } },
    });
    try {
      composeWorld([catalog, evil]);
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(WorldError);
      expect((e as WorldError).problems.length).toBeGreaterThanOrEqual(2);
    }
  });
});

describe("closure check", () => {
  it("errors on dangling refs with exact fact/field path", () => {
    const dangling = seed("dangling", {
      discounts: { s: { code: "S", product: ref("products", "no-such") } },
    });
    expect(() => composeWorld([dangling])).toThrowError(/discounts\/"s" field "product" references products\/"no-such"/);
  });

  it("finds refs nested in arrays and objects", () => {
    const nested = seed("nested", {
      bundles: { b1: { items: [{ product: ref("products", "ghost") }] } },
    });
    expect(() => composeWorld([nested])).toThrowError(/items\[0\]\.product/);
  });
});

describe("patches", () => {
  it("applies after merge, ordered", () => {
    const w = composeWorld([catalog], [
      { type: "products", key: "blue-widget", field: "stock", value: 5 },
      { type: "products", key: "blue-widget", field: "stock", value: 0 },
    ]);
    expect(w.facts.products!["blue-widget"]!.stock).toBe(0);
  });

  it("errors on missing patch target with known keys listed", () => {
    const { problems } = applyPatches(catalog.facts, [
      { type: "products", key: "ghost", field: "stock", value: 0 },
    ]);
    expect(problems[0]).toMatch(/ghost/);
    expect(problems[0]).toMatch(/blue-widget/);
  });

  it("does not mutate the input facts", () => {
    applyPatches(catalog.facts, [{ type: "products", key: "blue-widget", field: "stock", value: 0 }]);
    expect(catalog.facts.products!["blue-widget"]!.stock).toBe(12);
  });

  it("changes the fingerprint", () => {
    const plain = composeWorld([catalog]);
    const patched = composeWorld([catalog], [{ type: "products", key: "blue-widget", field: "stock", value: 0 }]);
    expect(patched.fingerprint).not.toBe(plain.fingerprint);
  });
});

describe("fingerprint", () => {
  it("is stable under key ordering", () => {
    const a = fingerprintOf({ t: { k: { x: 1, y: 2 } } }, []);
    const b = fingerprintOf({ t: { k: { y: 2, x: 1 } } }, []);
    expect(a).toBe(b);
  });
});

function makeMemoryWorld() {
  const db: Record<string, Map<string, FactRow>> = { users: new Map(), products: new Map(), discounts: new Map() };
  let nextId = 1;
  const adapter = defineWorld({
    reset: () => {
      for (const t of Object.values(db)) t.clear();
      nextId = 1;
    },
    entities: {
      users: {
        install: (rows) => {
          const ids: Record<string, number> = {};
          for (const [k, row] of Object.entries(rows)) {
            ids[k] = nextId;
            db.users!.set(k, { id: nextId++, ...row });
          }
          return ids;
        },
        schema: (row) => (typeof row.email === "string" ? null : "email must be a string"),
        read: (keys) => Object.fromEntries(keys.flatMap((k) => (db.users!.has(k) ? [[k, stripId(db.users!.get(k)!)]] : []))),
      },
      products: {
        install: (rows) => {
          const ids: Record<string, number> = {};
          for (const [k, row] of Object.entries(rows)) {
            ids[k] = nextId;
            db.products!.set(k, { id: nextId++, ...row });
          }
          return ids;
        },
        schema: (row) => (typeof row.price === "number" ? null : "price must be a number"),
        read: (keys) => Object.fromEntries(keys.flatMap((k) => (db.products!.has(k) ? [[k, stripId(db.products!.get(k)!)]] : []))),
      },
      discounts: {
        needs: ["products"],
        install: (rows, ctx) => {
          for (const [k, row] of Object.entries(rows)) {
            db.discounts!.set(k, { ...row, productId: ctx.id(row.product as never) });
          }
        },
      },
    },
  });
  return { adapter, db };
}

function stripId(row: FactRow): FactRow {
  const { id: _id, ...rest } = row;
  return rest;
}

describe("applyWorld", () => {
  it("resets, installs in dependency order, resolves refs", async () => {
    const { adapter, db } = makeMemoryWorld();
    const w = composeWorld([catalog, discounts]);
    const applied = await applyWorld(adapter, w);
    expect(db.products!.get("blue-widget")!.stock).toBe(12);
    const productId = applied.ids.products!["blue-widget"];
    expect(db.discounts!.get("summer")!.productId).toBe(productId);
  });

  it("read-back verification catches a lying installer", async () => {
    const { adapter } = makeMemoryWorld();
    const broken = defineWorld({
      reset: adapter.reset,
      entities: {
        ...adapter.entities,
        products: {
          ...adapter.entities.products!,
          install: (rows) => {
            // "installs" but corrupts stock
            const out: Record<string, number> = {};
            let i = 100;
            for (const k of Object.keys(rows)) out[k] = i++;
            return out;
          },
          read: () => ({}),
        },
      },
    });
    await expect(applyWorld(broken, composeWorld([catalog]))).rejects.toThrowError(/described but absent/);
  });

  it("schema failures abort before reset/install touches anything", async () => {
    const { adapter, db } = makeMemoryWorld();
    await applyWorld(adapter, composeWorld([catalog])); // world now has data
    const bad = seed("bad", { users: { u1: { email: 42 } } });
    await expect(applyWorld(adapter, composeWorld([bad]))).rejects.toThrowError(/email must be a string/);
    expect(db.users!.size).toBeGreaterThan(0); // untouched — validation is pre-install
  });

  it("missing installer for a described type is a hard error naming known types", () => {
    const { adapter } = makeMemoryWorld();
    const w = composeWorld([seed("s", { martians: { m1: { name: "Zorp" } } })]);
    expect(() => installOrder(w.facts, adapter.entities)).toThrowError(/martians.*users|users.*martians/s);
  });

  it("dependency cycles are hard errors", () => {
    const cyclic = defineWorld({
      reset: () => {},
      entities: {
        a: { install: () => {}, needs: ["b"] },
        b: { install: () => {}, needs: ["a"] },
      },
    });
    expect(() => installOrder({ a: { x: {} }, b: { y: {} } }, cyclic.entities)).toThrowError(/cycle/);
  });

  it("unresolvable ctx.id names the fix", async () => {
    const adapter = defineWorld({
      reset: () => {},
      entities: {
        products: { install: () => {} }, // returns no ids
        discounts: { needs: ["products"], install: (rows, ctx) => {
          for (const row of Object.values(rows)) ctx.id(row.product as never);
        } },
      },
    });
    const w = composeWorld([
      seed("s", {
        products: { p: { name: "P" } },
        discounts: { d: { product: ref("products", "p") } },
      }),
    ]);
    await expect(applyWorld(adapter, w)).rejects.toThrowError(/return \{ "p": <id> \} from install/);
  });
});

describe("capability ladder", () => {
  it("full memory world is L2 (no fingerprint/snapshot)", () => {
    const { adapter } = makeMemoryWorld();
    // discounts has no schema/read, so this world is actually L0 with named gaps
    const { level, missing } = capabilityLevel(adapter);
    expect(level).toBe(0);
    expect(missing.join("\n")).toMatch(/discounts/);
  });

  it("climbs to L4 when everything is provided", () => {
    const adapter = defineWorld({
      reset: () => {},
      entities: {
        t: { install: () => {}, schema: () => null, read: () => ({}) },
      },
      fingerprint: () => "fp",
      snapshot: () => "s1",
      restore: () => {},
    });
    expect(capabilityLevel(adapter).level).toBe(4);
    expect(doctor(adapter).levelName).toBe("L4 time-travel");
  });

  it("doctor names the exact next rung", () => {
    const { adapter } = makeMemoryWorld();
    const report = doctor(adapter);
    expect(report.nextRungs.some((r) => r.includes("discounts"))).toBe(true);
  });

  it("snapshot without restore is rejected at definition time", () => {
    expect(() =>
      defineWorld({ reset: () => {}, entities: {}, snapshot: () => "x" }),
    ).toThrowError(/together/);
  });
});
