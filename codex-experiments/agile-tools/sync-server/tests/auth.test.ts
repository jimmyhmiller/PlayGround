import { describe, it, expect, beforeEach } from "vitest";
import * as ed from "@noble/ed25519";
import { sha512 } from "@noble/hashes/sha512";
import { sha256 } from "@noble/hashes/sha256";
import { verifySignature, deriveAccountId, sha256Hex } from "../src/core/auth.js";
import { handleRequest, setRateLimiter } from "../src/router.js";
import { authenticate } from "../src/middleware/authenticate.js";
import { createTestMetadataStore, createMockBlobStore } from "./helpers/mock-storage.js";
import type { HandlerRequest, MetadataStore, BlobStore } from "../src/core/types.js";

// Disable rate limiting for these tests
setRateLimiter(null);

ed.etc.sha512Sync = (...m) => sha512(ed.etc.concatBytes(...m));

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

interface TestKeypair {
  privateKey: Uint8Array;
  publicKey: Uint8Array;
  publicKeyB64: string;
  accountId: string;
}

async function generateTestKeypair(): Promise<TestKeypair> {
  const privKey = ed.utils.randomPrivateKey();
  const pubKey = await ed.getPublicKeyAsync(privKey);
  const pubB64 = bytesToBase64(pubKey);
  return {
    privateKey: privKey,
    publicKey: pubKey,
    publicKeyB64: pubB64,
    accountId: deriveAccountId(pubB64),
  };
}

async function signedHeaders(
  method: string,
  path: string,
  body: string,
  keys: TestKeypair,
): Promise<Record<string, string>> {
  const timestamp = new Date().toISOString();
  const bodyHash = bytesToHex(sha256(new TextEncoder().encode(body)));
  const payload = `${method}:${path}:${timestamp}:${bodyHash}`;
  const msgBytes = new TextEncoder().encode(payload);
  const sig = await ed.signAsync(msgBytes, keys.privateKey);
  return {
    "x-scope-publickey": keys.publicKeyB64,
    "x-scope-timestamp": timestamp,
    "x-scope-signature": bytesToBase64(sig),
    "content-type": "application/json",
  };
}

async function signedReq(
  method: string,
  path: string,
  body: string,
  keys: TestKeypair,
): Promise<HandlerRequest> {
  return {
    method,
    path,
    headers: await signedHeaders(method, path, body, keys),
    body,
  };
}

// Helper: signup + create project for a keypair
async function setupAccount(
  store: MetadataStore,
  blobs: BlobStore,
  keys: TestKeypair,
  projectName?: string,
) {
  // Signup
  const signupReq = await signedReq(
    "POST",
    "/auth/signup",
    JSON.stringify({ public_key: keys.publicKeyB64, display_name: "user" }),
    keys,
  );
  const signupResp = await handleRequest(signupReq, store, blobs);
  expect(signupResp.status).toBe(201);

  // Create project if requested
  if (projectName) {
    const body = JSON.stringify({ name: projectName });
    const req = await signedReq("POST", "/projects", body, keys);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(201);
  }
}

// ────────────────────────────────────────────────────────────────────────

describe("crypto primitives", () => {
  it("deriveAccountId is deterministic", () => {
    const id1 = deriveAccountId("dGVzdA==");
    const id2 = deriveAccountId("dGVzdA==");
    expect(id1).toBe(id2);
    expect(id1).toMatch(/^acct_[0-9a-f]{16}$/);
  });

  it("different keys produce different account IDs", () => {
    expect(deriveAccountId("dGVzdDE=")).not.toBe(deriveAccountId("dGVzdDI="));
  });

  it("sha256Hex produces 64-char hex", () => {
    expect(sha256Hex("")).toMatch(/^[0-9a-f]{64}$/);
  });

  it("verifySignature accepts valid signature", async () => {
    const keys = await generateTestKeypair();
    const msg = "hello";
    const sig = await ed.signAsync(new TextEncoder().encode(msg), keys.privateKey);
    expect(verifySignature(keys.publicKeyB64, bytesToBase64(sig), msg)).toBe(true);
  });

  it("verifySignature rejects garbage", async () => {
    const keys = await generateTestKeypair();
    expect(verifySignature(keys.publicKeyB64, "aW52YWxpZA==", "test")).toBe(false);
  });

  it("verifySignature rejects wrong key", async () => {
    const alice = await generateTestKeypair();
    const bob = await generateTestKeypair();
    const msg = "hello";
    const sig = await ed.signAsync(new TextEncoder().encode(msg), alice.privateKey);
    // Verify with Bob's key should fail
    expect(verifySignature(bob.publicKeyB64, bytesToBase64(sig), msg)).toBe(false);
  });
});

describe("authenticate middleware", () => {
  it("rejects missing headers", () => {
    expect(() =>
      authenticate({ method: "GET", path: "/", headers: {}, body: "" }),
    ).toThrow("missing auth headers");
  });

  it("rejects partial headers (missing signature)", async () => {
    const keys = await generateTestKeypair();
    expect(() =>
      authenticate({
        method: "GET",
        path: "/",
        headers: {
          "x-scope-publickey": keys.publicKeyB64,
          "x-scope-timestamp": new Date().toISOString(),
        },
        body: "",
      }),
    ).toThrow("missing auth headers");
  });

  it("rejects unparseable timestamp", async () => {
    const keys = await generateTestKeypair();
    expect(() =>
      authenticate({
        method: "GET",
        path: "/",
        headers: {
          "x-scope-publickey": keys.publicKeyB64,
          "x-scope-timestamp": "not-a-date",
          "x-scope-signature": "aW52YWxpZA==",
        },
        body: "",
      }),
    ).toThrow("invalid timestamp");
  });

  it("rejects timestamp >5 min in the past", async () => {
    const keys = await generateTestKeypair();
    const old = new Date(Date.now() - 6 * 60 * 1000).toISOString();
    expect(() =>
      authenticate({
        method: "GET",
        path: "/",
        headers: {
          "x-scope-publickey": keys.publicKeyB64,
          "x-scope-timestamp": old,
          "x-scope-signature": "aW52YWxpZA==",
        },
        body: "",
      }),
    ).toThrow("timestamp too old");
  });

  it("rejects timestamp >5 min in the future", async () => {
    const keys = await generateTestKeypair();
    const future = new Date(Date.now() + 6 * 60 * 1000).toISOString();
    expect(() =>
      authenticate({
        method: "GET",
        path: "/",
        headers: {
          "x-scope-publickey": keys.publicKeyB64,
          "x-scope-timestamp": future,
          "x-scope-signature": "aW52YWxpZA==",
        },
        body: "",
      }),
    ).toThrow("timestamp too old");
  });

  it("rejects valid format but wrong signature", async () => {
    const alice = await generateTestKeypair();
    const bob = await generateTestKeypair();
    const body = '{"test":true}';
    // Sign with Alice's key
    const headers = await signedHeaders("POST", "/sync/push", body, alice);
    // But swap in Bob's public key
    headers["x-scope-publickey"] = bob.publicKeyB64;
    expect(() =>
      authenticate({ method: "POST", path: "/sync/push", headers, body }),
    ).toThrow("invalid signature");
  });

  it("rejects tampered body", async () => {
    const keys = await generateTestKeypair();
    const body = '{"project":"demo"}';
    const headers = await signedHeaders("POST", "/sync/push", body, keys);
    // Tamper with the body after signing
    expect(() =>
      authenticate({
        method: "POST",
        path: "/sync/push",
        headers,
        body: '{"project":"HACKED"}',
      }),
    ).toThrow("invalid signature");
  });

  it("rejects tampered path", async () => {
    const keys = await generateTestKeypair();
    const body = '{"project":"demo"}';
    const headers = await signedHeaders("POST", "/sync/push", body, keys);
    // Verify against a different path
    expect(() =>
      authenticate({
        method: "POST",
        path: "/sync/pull",
        headers,
        body,
      }),
    ).toThrow("invalid signature");
  });

  it("rejects tampered method", async () => {
    const keys = await generateTestKeypair();
    const body = "";
    const headers = await signedHeaders("GET", "/projects", body, keys);
    expect(() =>
      authenticate({ method: "POST", path: "/projects", headers, body }),
    ).toThrow("invalid signature");
  });

  it("accepts valid signed request and returns correct account ID", async () => {
    const keys = await generateTestKeypair();
    const body = '{"test":true}';
    const req = await signedReq("POST", "/sync/push", body, keys);
    const accountId = authenticate(req);
    expect(accountId).toBe(keys.accountId);
  });
});

describe("router auth enforcement", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;

  beforeEach(() => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
  });

  it("unauthenticated requests to protected endpoints return 401", async () => {
    const resp = await handleRequest(
      { method: "POST", path: "/sync/push", headers: {}, body: '{"project":"x"}' },
      store,
      blobs,
    );
    expect(resp.status).toBe(401);
    expect(JSON.parse(resp.body).error).toContain("missing auth headers");
  });

  it("bad signature to protected endpoints returns 401", async () => {
    const keys = await generateTestKeypair();
    const body = '{"project":"x"}';
    const headers = await signedHeaders("POST", "/sync/push", body, keys);
    // Corrupt the signature
    headers["x-scope-signature"] = "aW52YWxpZA==";

    const resp = await handleRequest(
      { method: "POST", path: "/sync/push", headers, body },
      store,
      blobs,
    );
    expect(resp.status).toBe(401);
    expect(JSON.parse(resp.body).error).toContain("invalid signature");
  });

  it("valid signature passes auth and reaches handler", async () => {
    const keys = await generateTestKeypair();
    await setupAccount(store, blobs, keys);

    const body = JSON.stringify({ name: "my-project" });
    const req = await signedReq("POST", "/projects", body, keys);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(201);
  });

  it("signup endpoint requires auth (rejects unsigned request)", async () => {
    const keys = await generateTestKeypair();
    const resp = await handleRequest(
      {
        method: "POST",
        path: "/auth/signup",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          public_key: keys.publicKeyB64,
          display_name: "test",
        }),
      },
      store,
      blobs,
    );
    expect(resp.status).toBe(401);
  });

  it("signup endpoint works with valid auth", async () => {
    const keys = await generateTestKeypair();
    const body = JSON.stringify({
      public_key: keys.publicKeyB64,
      display_name: "test",
    });
    const req = await signedReq("POST", "/auth/signup", body, keys);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(201);
  });
});

describe("account isolation", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;
  let alice: TestKeypair;
  let bob: TestKeypair;

  beforeEach(async () => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
    alice = await generateTestKeypair();
    bob = await generateTestKeypair();
    await setupAccount(store, blobs, alice, "shared-name");
    await setupAccount(store, blobs, bob, "shared-name");
  });

  it("alice cannot push to bob's project", async () => {
    // Both have a project called "shared-name", but they're isolated by account ID.
    // Alice pushes to her project — gets URL scoped to her account.
    const body = JSON.stringify({ project: "shared-name" });
    const req = await signedReq("POST", "/sync/push", body, alice);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(200);
    const result = JSON.parse(resp.body);
    // The upload URL contains Alice's account ID, not Bob's
    expect(result.upload_url).toContain(alice.accountId);
    expect(result.upload_url).not.toContain(bob.accountId);
  });

  it("alice cannot pull from bob's project", async () => {
    // Put a batch in Bob's namespace
    const bobBatchKey = `projects/${bob.accountId}/shared-name/batches/000000001_aaa.jsonl`;
    blobs.putBlob(bobBatchKey, '{"id":"evt_001","type":"issue.create"}\n');

    // Alice pulls her project — should get zero batches (Bob's data is invisible)
    const body = JSON.stringify({ project: "shared-name", since_batch: null });
    const req = await signedReq("POST", "/sync/pull", body, alice);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(200);
    const result = JSON.parse(resp.body);
    expect(result.batches).toHaveLength(0);
  });

  it("alice's batches are invisible to bob", async () => {
    // Put a batch in Alice's namespace
    const aliceBatchKey = `projects/${alice.accountId}/shared-name/batches/000000001_aaa.jsonl`;
    blobs.putBlob(aliceBatchKey, '{"id":"evt_001","type":"issue.create"}\n');

    // Bob pulls — should not see Alice's batch
    const body = JSON.stringify({ project: "shared-name", since_batch: null });
    const req = await signedReq("POST", "/sync/pull", body, bob);
    const resp = await handleRequest(req, store, blobs);
    const result = JSON.parse(resp.body);
    expect(result.batches).toHaveLength(0);

    // Alice pulls — should see her own batch
    const aliceReq = await signedReq("POST", "/sync/pull", body, alice);
    const aliceResp = await handleRequest(aliceReq, store, blobs);
    const aliceResult = JSON.parse(aliceResp.body);
    expect(aliceResult.batches).toHaveLength(1);
  });

  it("alice cannot see bob's projects in list", async () => {
    const req = await signedReq("GET", "/projects", "", alice);
    const resp = await handleRequest(req, store, blobs);
    const result = JSON.parse(resp.body);
    // Alice sees only her own project
    expect(result.projects).toHaveLength(1);
    expect(result.projects[0].account_id).toBe(alice.accountId);
  });

  it("alice cannot get status of bob's project", async () => {
    // Bob has batches
    const bobBatchKey = `projects/${bob.accountId}/shared-name/batches/000000001_aaa.jsonl`;
    blobs.putBlob(bobBatchKey, "data\n");

    // Alice checks status — sees 0 batches (her namespace is empty)
    const req = await signedReq("GET", "/sync/status?project=shared-name", "", alice);
    const resp = await handleRequest(req, store, blobs);
    const result = JSON.parse(resp.body);
    expect(result.total_batches).toBe(0);
  });

  it("push URL cannot be reused across accounts", async () => {
    // Alice gets an upload URL
    const body = JSON.stringify({ project: "shared-name" });
    const req = await signedReq("POST", "/sync/push", body, alice);
    const resp = await handleRequest(req, store, blobs);
    const result = JSON.parse(resp.body);

    // The URL path contains Alice's account ID
    expect(result.upload_url).toContain(alice.accountId);

    // Even if Bob somehow obtained this URL and uploaded to it,
    // the data would be in Alice's namespace, not Bob's.
    // Bob's pull would never see it.
    blobs.putBlob(
      `projects/${alice.accountId}/shared-name/batches/${result.batch_id}.jsonl`,
      '{"id":"evt_injected"}\n',
    );

    // Bob's pull returns nothing (his namespace is empty)
    const pullBody = JSON.stringify({ project: "shared-name", since_batch: null });
    const bobPullReq = await signedReq("POST", "/sync/pull", pullBody, bob);
    const bobPullResp = await handleRequest(bobPullReq, store, blobs);
    expect(JSON.parse(bobPullResp.body).batches).toHaveLength(0);
  });
});

describe("signature replay protection", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;

  beforeEach(() => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
  });

  it("replaying a request with old timestamp is rejected", async () => {
    const keys = await generateTestKeypair();
    await setupAccount(store, blobs, keys, "demo");

    // Create a request signed 6 minutes ago
    const timestamp = new Date(Date.now() - 6 * 60 * 1000).toISOString();
    const body = JSON.stringify({ project: "demo" });
    const bodyHash = bytesToHex(sha256(new TextEncoder().encode(body)));
    const payload = `POST:/sync/push:${timestamp}:${bodyHash}`;
    const sig = await ed.signAsync(new TextEncoder().encode(payload), keys.privateKey);

    const resp = await handleRequest(
      {
        method: "POST",
        path: "/sync/push",
        headers: {
          "x-scope-publickey": keys.publicKeyB64,
          "x-scope-timestamp": timestamp,
          "x-scope-signature": bytesToBase64(sig),
          "content-type": "application/json",
        },
        body,
      },
      store,
      blobs,
    );
    expect(resp.status).toBe(401);
    expect(JSON.parse(resp.body).error).toContain("timestamp too old");
  });
});
