import { handleRequest } from "../../../src/router.js";
import type { HandlerRequest, BlobStore } from "../../../src/core/types.js";
import { createD1MetadataStore } from "./d1-metadata-store.js";

interface Env {
  BUCKET: R2Bucket;
  DB: D1Database;
}

function r2BlobStore(bucket: R2Bucket, baseUrl: string): BlobStore {
  // In production, this would generate presigned R2 URLs.
  // For now, proxy through the worker.
  return {
    async createUploadUrl(key: string): Promise<string> {
      return `${baseUrl}/blobs/${key}`;
    },
    async createDownloadUrl(key: string): Promise<string> {
      return `${baseUrl}/blobs/${key}`;
    },
    async listKeys(prefix: string): Promise<string[]> {
      const listed = await bucket.list({ prefix });
      return listed.objects.map((o) => o.key);
    },
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;
    const baseUrl = url.origin;

    // Blob proxy endpoints
    if (path.startsWith("/blobs/")) {
      const blobKey = path.slice("/blobs/".length);

      if (request.method === "PUT") {
        const body = await request.text();
        await env.BUCKET.put(blobKey, body);
        return new Response(JSON.stringify({ ok: true }), {
          headers: { "Content-Type": "application/json" },
        });
      }

      if (request.method === "GET") {
        const obj = await env.BUCKET.get(blobKey);
        if (!obj) return new Response("not found", { status: 404 });
        return new Response(obj.body, {
          headers: { "Content-Type": "application/x-ndjson" },
        });
      }

      return new Response("method not allowed", { status: 405 });
    }

    // API endpoints
    const body = request.method !== "GET" ? await request.text() : "";
    const headers: Record<string, string> = {};
    request.headers.forEach((value, key) => {
      headers[key.toLowerCase()] = value;
    });

    const handlerReq: HandlerRequest = {
      method: request.method,
      path: path + url.search,
      headers,
      body,
    };

    const store = await createD1MetadataStore(env.DB);
    const blobs = r2BlobStore(env.BUCKET, baseUrl);
    const response = await handleRequest(handlerReq, store, blobs);

    return new Response(response.body, {
      status: response.status,
      headers: response.headers,
    });
  },
};
