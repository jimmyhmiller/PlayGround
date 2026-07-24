import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createWorldHandler } from "../../src/server/index.js";
import { db } from "./db.js";
import { world } from "./world.js";

const PUBLIC_DIR = join(dirname(fileURLToPath(import.meta.url)), "public");

/** Server-side latency — the whole point of the fixture. Configurable and
 * seedable so property tests can quantify over timing profiles. */
export interface ShopTiming {
  /** [min, max] added ms per API request */
  apiLatencyMs: [number, number];
  /** how long the client-side toast lives (injected into the page) */
  toastMs: number;
  /** PRNG seed; same profile = same latency sequence */
  seed: number;
}

let timing: ShopTiming = { apiLatencyMs: [50, 400], toastMs: 150, seed: 0 };
let rng = () => Math.random();

export function setShopTiming(profile: ShopTiming): void {
  timing = profile;
  let s = profile.seed | 0;
  rng =
    profile.seed === 0
      ? () => Math.random()
      : () => {
          s = (s + 0x6d2b79f5) | 0;
          let t = Math.imul(s ^ (s >>> 15), 1 | s);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
}

function jitter(): Promise<void> {
  const [lo, hi] = timing.apiLatencyMs;
  const ms = lo + rng() * (hi - lo);
  return new Promise((r) => setTimeout(r, ms));
}

const worldHandler = createWorldHandler(world);

async function toWebRequest(req: IncomingMessage): Promise<Request> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  const init: RequestInit = { method: req.method ?? "GET", headers: req.headers as Record<string, string> };
  if (chunks.length) init.body = Buffer.concat(chunks);
  return new Request(`http://localhost${req.url ?? "/"}`, init);
}

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(payload));
}

function cookieValue(req: IncomingMessage, name: string): string | null {
  const raw = req.headers.cookie ?? "";
  for (const part of raw.split(";")) {
    const [k, ...rest] = part.trim().split("=");
    if (k === name) return rest.join("=");
  }
  return null;
}

function cartPayload() {
  const lines = db.cart.map((line) => {
    const product = [...db.products.values()].find((p) => p.id === line.productId)!;
    return { name: product.name, price: product.price, qty: line.qty };
  });
  return { lines, count: db.cart.reduce((n, l) => n + l.qty, 0) };
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://localhost");
  const path = url.pathname;

  if (path === "/api/__bat") {
    const response = await worldHandler(await toWebRequest(req));
    res.writeHead(response.status, Object.fromEntries(response.headers.entries()));
    res.end(Buffer.from(await response.arrayBuffer()));
    return;
  }

  if (path === "/api/cart" && req.method === "GET") {
    // Read-then-jitter: the payload is read from the db when the request
    // ARRIVES, but the response leaves after a random delay. Correct for a
    // settled world — but it gives concurrent writers a stale-read window,
    // which is exactly the race the /flaky-cart page's refetch bug trips over.
    const payload = cartPayload();
    await jitter();
    return sendJson(res, 200, payload);
  }

  if (path.startsWith("/api/")) {
    await jitter();
    if (path === "/api/products" && req.method === "GET") {
      const q = url.searchParams.get("q")?.toLowerCase() ?? "";
      const all = [...db.products.values()];
      return sendJson(res, 200, q ? all.filter((p) => p.name.toLowerCase().includes(q)) : all);
    }
    if (path === "/api/cart" && req.method === "POST") {
      const chunks: Buffer[] = [];
      for await (const c of req) chunks.push(c as Buffer);
      const body = JSON.parse(Buffer.concat(chunks).toString() || "{}") as { productId?: number };
      const product = [...db.products.values()].find((p) => p.id === body.productId);
      if (!product) return sendJson(res, 404, { error: "no such product" });
      if (product.stock <= 0) return sendJson(res, 409, { error: "out of stock" });
      const line = db.cart.find((l) => l.productId === product.id);
      if (line) line.qty++;
      else db.cart.push({ productId: product.id, qty: 1 });
      return sendJson(res, 200, cartPayload());
    }
    if (path === "/api/me" && req.method === "GET") {
      const key = cookieValue(req, "batsession");
      const user = key ? db.users.get(key) : undefined;
      if (!user) return sendJson(res, 401, { error: "not signed in" });
      return sendJson(res, 200, { email: user.email, role: user.role });
    }
    return sendJson(res, 404, { error: `no route ${req.method} ${path}` });
  }

  if (path === "/app.js") {
    res.writeHead(200, { "content-type": "text/javascript" });
    res.end(await readFile(join(PUBLIC_DIR, "app.js")));
    return;
  }
  // SPA: every page path serves the shell (timing config injected as data)
  res.writeHead(200, { "content-type": "text/html" });
  const html = (await readFile(join(PUBLIC_DIR, "index.html"), "utf8")).replace(
    "</head>",
    `<script>window.__batTiming = ${JSON.stringify({ toastMs: timing.toastMs })}</script></head>`,
  );
  res.end(html);
}

export async function startShopServer(port = 0): Promise<{ url: string; server: Server; close: () => Promise<void> }> {
  const server = createServer((req, res) => {
    handle(req, res).catch((e) => {
      res.writeHead(500, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: e instanceof Error ? e.message : String(e) }));
    });
  });
  await new Promise<void>((resolve) => server.listen(port, resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("could not determine server port");
  return {
    url: `http://localhost:${address.port}`,
    server,
    close: () => new Promise((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}
