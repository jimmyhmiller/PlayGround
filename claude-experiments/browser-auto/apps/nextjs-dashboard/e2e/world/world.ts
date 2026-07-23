import { createHash } from "node:crypto";
import bcrypt from "bcrypt";
import postgres from "postgres";
import { defineWorld } from "../../../../src/world/index.js";
import type { FactRow } from "../../../../src/world/types.js";

/**
 * bat world adapter for the Next.js Learn dashboard app. Runs in bat's
 * process (module transport) and talks straight to the app's Postgres.
 * Sessions are minted through the app's real NextAuth credentials endpoints.
 */

const POSTGRES_URL = process.env.POSTGRES_URL ?? "postgres://jimmyhmiller@localhost:5432/bat_dashboard";
const APP_URL = process.env.BAT_APP_URL ?? "http://localhost:3000";

const sql = postgres(POSTGRES_URL, { max: 2 });

/** deterministic uuid from a fact key — same world description, same ids */
function keyUuid(kind: string, key: string): string {
  const h = createHash("sha256").update(`${kind}/${key}`).digest("hex");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-4${h.slice(13, 16)}-8${h.slice(17, 20)}-${h.slice(20, 32)}`;
}

/** credentials captured at install time so session() can log in for real */
const installedUsers = new Map<string, { email: string; password: string }>();

const TABLES = ["invoices", "customers", "users", "revenue"] as const;

async function dumpAll(): Promise<Record<string, unknown[]>> {
  const out: Record<string, unknown[]> = {};
  for (const t of TABLES) {
    out[t] = await sql`SELECT * FROM ${sql(t)} ORDER BY 1`;
  }
  return out;
}

const snapshots = new Map<string, Record<string, unknown[]>>();
let snapshotCounter = 0;

export default defineWorld({
  reset: async () => {
    await sql`TRUNCATE invoices, customers, users, revenue`;
    installedUsers.clear();
  },
  entities: {
    users: {
      install: async (rows) => {
        const ids: Record<string, string> = {};
        for (const [key, row] of Object.entries(rows)) {
          const id = keyUuid("users", key);
          const hashed = await bcrypt.hash(String(row.password), 10);
          await sql`INSERT INTO users (id, name, email, password) VALUES (${id}, ${String(row.name)}, ${String(row.email)}, ${hashed})`;
          installedUsers.set(key, { email: String(row.email), password: String(row.password) });
          ids[key] = id;
        }
        return ids;
      },
      schema: (row) => {
        if (typeof row.email !== "string" || !row.email.includes("@")) return "email must be an email string";
        if (typeof row.password !== "string" || row.password.length < 6) return "password must be a string of 6+ chars (NextAuth zod schema requires it)";
        if (typeof row.name !== "string") return "name must be a string";
        return null;
      },
      read: async (keys) => {
        const out: Record<string, FactRow> = {};
        for (const key of keys) {
          const r = await sql`SELECT name, email FROM users WHERE id = ${keyUuid("users", key)}`;
          if (r[0]) out[key] = { name: r[0].name, email: r[0].email, password: installedUsers.get(key)?.password };
        }
        return out;
      },
    },
    customers: {
      install: async (rows) => {
        const ids: Record<string, string> = {};
        for (const [key, row] of Object.entries(rows)) {
          const id = keyUuid("customers", key);
          await sql`INSERT INTO customers (id, name, email, image_url) VALUES (${id}, ${String(row.name)}, ${String(row.email)}, ${String(row.image_url)})`;
          ids[key] = id;
        }
        return ids;
      },
      schema: (row) => {
        if (typeof row.name !== "string") return "name must be a string";
        if (typeof row.email !== "string") return "email must be a string";
        if (typeof row.image_url !== "string") return "image_url must be a string (the app renders it in tables)";
        return null;
      },
      read: async (keys) => {
        const out: Record<string, FactRow> = {};
        for (const key of keys) {
          const r = await sql`SELECT name, email, image_url FROM customers WHERE id = ${keyUuid("customers", key)}`;
          if (r[0]) out[key] = { name: r[0].name, email: r[0].email, image_url: r[0].image_url };
        }
        return out;
      },
    },
    invoices: {
      needs: ["customers"],
      install: async (rows, ctx) => {
        const ids: Record<string, string> = {};
        for (const [key, row] of Object.entries(rows)) {
          const id = keyUuid("invoices", key);
          const customerId = ctx.id(row.customer as never);
          await sql`INSERT INTO invoices (id, customer_id, amount, status, date) VALUES (${id}, ${String(customerId)}, ${Number(row.amount)}, ${String(row.status)}, ${String(row.date)})`;
          ids[key] = id;
        }
        return ids;
      },
      schema: (row) => {
        if (typeof row.amount !== "number") return "amount must be a number (cents)";
        if (row.status !== "pending" && row.status !== "paid") return 'status must be "pending" or "paid"';
        if (typeof row.date !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(row.date)) return "date must be YYYY-MM-DD";
        return null;
      },
      read: async (keys) => {
        const out: Record<string, FactRow> = {};
        for (const key of keys) {
          const r = await sql`SELECT amount, status, to_char(date, 'YYYY-MM-DD') AS date, customer_id FROM invoices WHERE id = ${keyUuid("invoices", key)}`;
          if (r[0]) out[key] = { amount: r[0].amount, status: r[0].status, date: r[0].date };
        }
        return out;
      },
    },
    revenue: {
      install: async (rows) => {
        for (const [month, row] of Object.entries(rows)) {
          await sql`INSERT INTO revenue (month, revenue) VALUES (${month}, ${Number(row.revenue)})`;
        }
      },
      schema: (row) => (typeof row.revenue === "number" ? null : "revenue must be a number"),
      read: async (keys) => {
        const out: Record<string, FactRow> = {};
        for (const key of keys) {
          const r = await sql`SELECT revenue FROM revenue WHERE month = ${key}`;
          if (r[0]) out[key] = { revenue: r[0].revenue };
        }
        return out;
      },
    },
  },

  /** real session, minted through the app's own NextAuth credentials flow */
  session: async (userKey) => {
    const creds = installedUsers.get(userKey);
    if (!creds) throw new Error(`session: user fact "${userKey}" was not installed in this world`);

    const csrfRes = await fetch(`${APP_URL}/api/auth/csrf`);
    const { csrfToken } = (await csrfRes.json()) as { csrfToken: string };
    const csrfCookies = csrfRes.headers.getSetCookie().map((c) => c.split(";")[0]!);

    const loginRes = await fetch(`${APP_URL}/api/auth/callback/credentials`, {
      method: "POST",
      redirect: "manual",
      headers: {
        "content-type": "application/x-www-form-urlencoded",
        cookie: csrfCookies.join("; "),
      },
      body: new URLSearchParams({ csrfToken, email: creds.email, password: creds.password }).toString(),
    });
    const sessionCookies = loginRes.headers
      .getSetCookie()
      .map((c) => c.split(";")[0]!)
      .filter((c) => c.includes("session-token"));
    if (sessionCookies.length === 0) {
      throw new Error(
        `session: the app did not issue a session token for "${creds.email}" (HTTP ${loginRes.status}) — wrong credentials or the app is not running at ${APP_URL}`,
      );
    }
    return {
      cookies: sessionCookies.map((c) => {
        const eq = c.indexOf("=");
        return { name: c.slice(0, eq), value: c.slice(eq + 1), path: "/" };
      }),
    };
  },

  fingerprint: async () => {
    const dump = await dumpAll();
    return `sha256:${createHash("sha256").update(JSON.stringify(dump)).digest("hex").slice(0, 12)}`;
  },
  snapshot: async () => {
    const id = `snap-${++snapshotCounter}`;
    snapshots.set(id, await dumpAll());
    return id;
  },
  restore: async (id) => {
    const dump = snapshots.get(id);
    if (!dump) throw new Error(`no snapshot "${id}"`);
    await sql`TRUNCATE invoices, customers, users, revenue`;
    for (const t of TABLES) {
      for (const row of dump[t] ?? []) {
        await sql`INSERT INTO ${sql(t)} ${sql(row as Record<string, unknown>)}`;
      }
    }
  },
});
