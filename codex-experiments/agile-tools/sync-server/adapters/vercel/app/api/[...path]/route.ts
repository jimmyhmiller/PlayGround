import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@libsql/client";
import { handleRequest } from "../../../../../src/router.js";
import type { HandlerRequest, BlobStore } from "../../../../../src/core/types.js";
import { createTursoMetadataStore } from "../../../lib/turso-metadata-store.js";
import {
  S3Client,
  GetObjectCommand,
  PutObjectCommand,
  ListObjectsV2Command,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const s3 = new S3Client({
  region: process.env.AWS_REGION ?? "us-east-1",
});
const BLOB_BUCKET = process.env.S3_BLOB_BUCKET ?? "scope-sync-data";

const tursoClient = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN,
});

function s3BlobStore(): BlobStore {
  return {
    async createUploadUrl(key: string): Promise<string> {
      const command = new PutObjectCommand({
        Bucket: BLOB_BUCKET,
        Key: key,
        ContentType: "application/x-ndjson",
      });
      return getSignedUrl(s3, command, { expiresIn: 3600 });
    },
    async createDownloadUrl(key: string): Promise<string> {
      const command = new GetObjectCommand({
        Bucket: BLOB_BUCKET,
        Key: key,
      });
      return getSignedUrl(s3, command, { expiresIn: 3600 });
    },
    async listKeys(prefix: string): Promise<string[]> {
      const resp = await s3.send(
        new ListObjectsV2Command({ Bucket: BLOB_BUCKET, Prefix: prefix }),
      );
      return (resp.Contents ?? []).map((obj) => obj.Key!).filter(Boolean);
    },
  };
}

async function handler(request: NextRequest) {
  const url = new URL(request.url);
  const body = request.method !== "GET" ? await request.text() : "";

  const headers: Record<string, string> = {};
  request.headers.forEach((value, key) => {
    headers[key.toLowerCase()] = value;
  });

  const handlerReq: HandlerRequest = {
    method: request.method,
    path: url.pathname.replace(/^\/api/, "") + url.search,
    headers,
    body,
  };

  const store = await createTursoMetadataStore(tursoClient);
  const blobs = s3BlobStore();
  const response = await handleRequest(handlerReq, store, blobs);

  return new NextResponse(response.body, {
    status: response.status,
    headers: response.headers,
  });
}

export const GET = handler;
export const POST = handler;
export const PUT = handler;
export const DELETE = handler;
