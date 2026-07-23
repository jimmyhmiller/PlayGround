import { createHash } from "node:crypto";
import { defineWorld } from "../../src/world/index.js";
import type { FactRow } from "../../src/world/types.js";
import { db, resetDb, restoreDb, serializeDb, type Product, type User } from "./db.js";

/** Full-ladder (L4) world adapter for the fixture shop. */

const snapshots = new Map<string, string>();
let snapshotCounter = 0;

export const world = defineWorld({
  reset: () => resetDb(),
  entities: {
    users: {
      install: (rows) => {
        const ids: Record<string, number> = {};
        for (const [key, row] of Object.entries(rows)) {
          const user: User = { id: db.nextId++, email: String(row.email), role: String(row.role ?? "customer") };
          db.users.set(key, user);
          ids[key] = user.id;
        }
        return ids;
      },
      schema: (row) => {
        if (typeof row.email !== "string") return "email must be a string";
        return null;
      },
      read: (keys) =>
        Object.fromEntries(
          keys.flatMap((k) => {
            const u = db.users.get(k);
            return u ? [[k, { email: u.email, role: u.role } satisfies FactRow]] : [];
          }),
        ),
    },
    products: {
      install: (rows) => {
        const ids: Record<string, number> = {};
        for (const [key, row] of Object.entries(rows)) {
          const product: Product = {
            id: db.nextId++,
            name: String(row.name),
            price: Number(row.price),
            stock: Number(row.stock ?? 0),
          };
          db.products.set(key, product);
          ids[key] = product.id;
        }
        return ids;
      },
      schema: (row) => {
        if (typeof row.name !== "string") return "name must be a string";
        if (typeof row.price !== "number") return "price must be a number (cents)";
        if (row.stock !== undefined && typeof row.stock !== "number") return "stock must be a number";
        return null;
      },
      read: (keys) =>
        Object.fromEntries(
          keys.flatMap((k) => {
            const p = db.products.get(k);
            return p ? [[k, { name: p.name, price: p.price, stock: p.stock } satisfies FactRow]] : [];
          }),
        ),
    },
  },
  session: (userKey) => {
    if (!db.users.has(userKey)) {
      throw new Error(`session: no user fact "${userKey}" installed in this world`);
    }
    return { cookies: [{ name: "batsession", value: userKey, path: "/" }] };
  },
  fingerprint: () => `sha256:${createHash("sha256").update(serializeDb()).digest("hex").slice(0, 12)}`,
  snapshot: () => {
    const id = `snap-${++snapshotCounter}`;
    snapshots.set(id, serializeDb());
    return id;
  },
  restore: (id) => {
    const s = snapshots.get(id);
    if (!s) throw new Error(`no snapshot "${id}"`);
    restoreDb(s);
  },
});

export default world;
