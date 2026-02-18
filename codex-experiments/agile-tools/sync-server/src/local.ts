import { createServer } from "node:http";
import { mkdir, readFile, writeFile, readdir } from "node:fs/promises";
import { join, dirname } from "node:path";
import Database from "better-sqlite3";
import { handleRequest } from "./router.js";
import { signBlobUrl, verifyBlobToken, initBlobSecret } from "./core/blob-token.js";
import { createSqliteMetadataStore } from "./core/sqlite-metadata-store.js";
import type { BlobStore, HandlerRequest } from "./core/types.js";

const PORT = parseInt(process.env.PORT ?? "4899", 10);
const DATA_DIR = process.env.DATA_DIR ?? join(process.cwd(), ".data");
const BLOB_DIR = join(DATA_DIR, "blobs");
const DB_PATH = join(DATA_DIR, "meta.db");

// Initialize blob signing secret
initBlobSecret();

const baseUrl = `http://localhost:${PORT}`;

function localBlobStore(blobRoot: string): BlobStore {
  return {
    async createUploadUrl(key: string): Promise<string> {
      return signBlobUrl(baseUrl, key);
    },
    async createDownloadUrl(key: string): Promise<string> {
      return signBlobUrl(baseUrl, key);
    },
    async listKeys(prefix: string): Promise<string[]> {
      const dir = join(blobRoot, prefix);
      const results: string[] = [];
      try {
        await collectKeys(dir, blobRoot, results);
      } catch (err: any) {
        if (err.code === "ENOENT") return [];
        throw err;
      }
      return results.sort();
    },
  };
}

async function collectKeys(dir: string, root: string, out: string[]) {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      await collectKeys(full, root, out);
    } else {
      out.push(full.slice(root.length + 1));
    }
  }
}

// Ensure data directory exists, then open SQLite
await mkdir(DATA_DIR, { recursive: true });
const db = new Database(DB_PATH);
const store = createSqliteMetadataStore(db);
const blobs = localBlobStore(BLOB_DIR);

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", baseUrl);
  const path = url.pathname;

  // Blob endpoints — verify signed token
  if (path.startsWith("/blobs/")) {
    const blobKey = path.slice("/blobs/".length);
    const expires = url.searchParams.get("expires");
    const token = url.searchParams.get("token");

    if (!verifyBlobToken(blobKey, expires, token)) {
      res.writeHead(403, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "invalid or expired blob token" }));
      return;
    }

    const filePath = join(BLOB_DIR, blobKey);

    if (req.method === "PUT") {
      const chunks: Buffer[] = [];
      for await (const chunk of req) {
        chunks.push(chunk as Buffer);
      }
      const body = Buffer.concat(chunks).toString("utf-8");
      await mkdir(dirname(filePath), { recursive: true });
      await writeFile(filePath, body, "utf-8");
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      return;
    }

    if (req.method === "GET") {
      try {
        const content = await readFile(filePath, "utf-8");
        res.writeHead(200, { "Content-Type": "application/x-ndjson" });
        res.end(content);
      } catch (err: any) {
        if (err.code === "ENOENT") {
          res.writeHead(404);
          res.end("not found");
        } else {
          throw err;
        }
      }
      return;
    }

    res.writeHead(405);
    res.end("method not allowed");
    return;
  }

  // API endpoints
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(chunk as Buffer);
  }
  const body = Buffer.concat(chunks).toString("utf-8");

  const headers: Record<string, string> = {};
  for (const [key, value] of Object.entries(req.headers)) {
    if (typeof value === "string") {
      headers[key.toLowerCase()] = value;
    }
  }

  const handlerReq: HandlerRequest = {
    method: req.method ?? "GET",
    path: path + url.search,
    headers,
    body,
  };

  const response = await handleRequest(handlerReq, store, blobs);

  res.writeHead(response.status, response.headers);
  res.end(response.body);
});

server.listen(PORT, () => {
  console.log(`Scope sync server listening on ${baseUrl}`);
  console.log(`Data directory: ${DATA_DIR}`);
  console.log(`Database: ${DB_PATH}`);
  console.log(`\nAll endpoints require Ed25519 authentication.`);
  console.log(`Blob URLs are HMAC-signed with 1h expiry.\n`);
  console.log(`Endpoints:`);
  console.log(`  POST /auth/signup              Register (requires signed request)`);
  console.log(`  POST /projects                 Create project`);
  console.log(`  GET  /projects                 List your projects`);
  console.log(`  POST /projects/members/add     Add member (owner only)`);
  console.log(`  POST /projects/members/remove  Remove member (owner only)`);
  console.log(`  POST /projects/members         List members`);
  console.log(`  POST /sync/push                Get upload URL for batch`);
  console.log(`  POST /sync/pull                Get download URLs for batches`);
  console.log(`  GET  /sync/status?project=     Check sync status`);
});
