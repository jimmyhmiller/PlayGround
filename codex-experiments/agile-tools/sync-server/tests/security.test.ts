import { describe, it, expect, beforeEach, afterEach } from "vitest";
import * as ed from "@noble/ed25519";
import { sha512 } from "@noble/hashes/sha512";
import { sha256 } from "@noble/hashes/sha256";
import { deriveAccountId } from "../src/core/auth.js";
import { handleRequest, setRateLimiter } from "../src/router.js";
import { RateLimiter } from "../src/core/rate-limit.js";
import { createTestMetadataStore, createMockBlobStore } from "./helpers/mock-storage.js";
import {
  initBlobSecret,
  signBlobUrl,
  verifyBlobToken,
  getSecret,
} from "../src/core/blob-token.js";
import type { MetadataStore, BlobStore } from "../src/core/types.js";

ed.etc.sha512Sync = (...m) => sha512(ed.etc.concatBytes(...m));

// Disable rate limiting for most tests (re-enabled in rate-limit tests)
setRateLimiter(null);

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
) {
  return {
    method,
    path,
    headers: await signedHeaders(method, path, body, keys),
    body,
  };
}

async function signup(
  store: MetadataStore,
  blobs: BlobStore,
  keys: TestKeypair,
) {
  const body = JSON.stringify({
    public_key: keys.publicKeyB64,
    display_name: "user",
  });
  const req = await signedReq("POST", "/auth/signup", body, keys);
  const resp = await handleRequest(req, store, blobs);
  expect(resp.status).toBe(201);
}

async function createProject(
  store: MetadataStore,
  blobs: BlobStore,
  keys: TestKeypair,
  name: string,
) {
  const body = JSON.stringify({ name });
  const req = await signedReq("POST", "/projects", body, keys);
  const resp = await handleRequest(req, store, blobs);
  expect(resp.status).toBe(201);
}

// ────────────────────────────────────────────────────────────────────────
// Signup security
// ────────────────────────────────────────────────────────────────────────

describe("signup security", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;

  beforeEach(() => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
  });

  it("rejects signup signed with a different key than the one being registered", async () => {
    const alice = await generateTestKeypair();
    const bob = await generateTestKeypair();

    // Alice signs the request, but tries to register Bob's public key
    const body = JSON.stringify({
      public_key: bob.publicKeyB64,
      display_name: "impersonator",
    });
    const req = await signedReq("POST", "/auth/signup", body, alice);
    const resp = await handleRequest(req, store, blobs);

    expect(resp.status).toBe(401);
    expect(JSON.parse(resp.body).error).toContain(
      "signing key does not match",
    );
  });

  it("rejects duplicate signup", async () => {
    const keys = await generateTestKeypair();
    await signup(store, blobs, keys);

    // Try again
    const body = JSON.stringify({
      public_key: keys.publicKeyB64,
      display_name: "dup",
    });
    const req = await signedReq("POST", "/auth/signup", body, keys);
    const resp = await handleRequest(req, store, blobs);

    expect(resp.status).toBe(409);
    expect(JSON.parse(resp.body).error).toContain("already exists");
  });

  it("signup returns correct account ID derived from public key", async () => {
    const keys = await generateTestKeypair();
    const body = JSON.stringify({
      public_key: keys.publicKeyB64,
      display_name: "test",
    });
    const req = await signedReq("POST", "/auth/signup", body, keys);
    const resp = await handleRequest(req, store, blobs);

    expect(resp.status).toBe(201);
    const result = JSON.parse(resp.body);
    expect(result.account_id).toBe(keys.accountId);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Membership model
// ────────────────────────────────────────────────────────────────────────

describe("membership model", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;
  let alice: TestKeypair;
  let bob: TestKeypair;
  let charlie: TestKeypair;

  beforeEach(async () => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
    alice = await generateTestKeypair();
    bob = await generateTestKeypair();
    charlie = await generateTestKeypair();
    await signup(store, blobs, alice);
    await signup(store, blobs, bob);
    await signup(store, blobs, charlie);
    await createProject(store, blobs, alice, "team-project");
  });

  it("owner can add a member", async () => {
    const body = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const req = await signedReq(
      "POST",
      "/projects/members/add",
      body,
      alice,
    );
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(200);
  });

  it("non-owner cannot add a member", async () => {
    const body = JSON.stringify({
      project: "team-project",
      member_account_id: charlie.accountId,
      owner: alice.accountId,
    });
    const req = await signedReq(
      "POST",
      "/projects/members/add",
      body,
      bob,
    );
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(401);
    expect(JSON.parse(resp.body).error).toContain("only the project owner");
  });

  it("owner can remove a member", async () => {
    // Add Bob first
    const addBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const addReq = await signedReq(
      "POST",
      "/projects/members/add",
      addBody,
      alice,
    );
    await handleRequest(addReq, store, blobs);

    // Now remove Bob
    const removeBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const removeReq = await signedReq(
      "POST",
      "/projects/members/remove",
      removeBody,
      alice,
    );
    const resp = await handleRequest(removeReq, store, blobs);
    expect(resp.status).toBe(200);
  });

  it("non-owner cannot remove a member", async () => {
    // Add Bob first
    const addBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const addReq = await signedReq(
      "POST",
      "/projects/members/add",
      addBody,
      alice,
    );
    await handleRequest(addReq, store, blobs);

    // Bob tries to remove himself — only owner can
    const removeBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
      owner: alice.accountId,
    });
    const removeReq = await signedReq(
      "POST",
      "/projects/members/remove",
      removeBody,
      bob,
    );
    const resp = await handleRequest(removeReq, store, blobs);
    expect(resp.status).toBe(401);
  });

  it("member can list members", async () => {
    // Add Bob
    const addBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const addReq = await signedReq(
      "POST",
      "/projects/members/add",
      addBody,
      alice,
    );
    await handleRequest(addReq, store, blobs);

    // Bob lists members
    const listBody = JSON.stringify({
      project: "team-project",
      owner: alice.accountId,
    });
    const listReq = await signedReq(
      "POST",
      "/projects/members",
      listBody,
      bob,
    );
    const resp = await handleRequest(listReq, store, blobs);
    expect(resp.status).toBe(200);
    const result = JSON.parse(resp.body);
    expect(result.members).toContain(alice.accountId);
    expect(result.members).toContain(bob.accountId);
  });

  it("non-member cannot list members", async () => {
    const listBody = JSON.stringify({
      project: "team-project",
      owner: alice.accountId,
    });
    const listReq = await signedReq(
      "POST",
      "/projects/members",
      listBody,
      charlie,
    );
    const resp = await handleRequest(listReq, store, blobs);
    expect(resp.status).toBe(401);
    expect(JSON.parse(resp.body).error).toContain("do not have access");
  });

  it("member can push to owner's project", async () => {
    // Add Bob as member
    const addBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const addReq = await signedReq(
      "POST",
      "/projects/members/add",
      addBody,
      alice,
    );
    await handleRequest(addReq, store, blobs);

    // Bob pushes to Alice's project
    const pushBody = JSON.stringify({
      project: "team-project",
      owner: alice.accountId,
    });
    const pushReq = await signedReq("POST", "/sync/push", pushBody, bob);
    const resp = await handleRequest(pushReq, store, blobs);
    expect(resp.status).toBe(200);
    const result = JSON.parse(resp.body);
    expect(result.upload_url).toBeTruthy();
    expect(result.batch_id).toBeTruthy();
  });

  it("non-member cannot push to owner's project", async () => {
    const pushBody = JSON.stringify({
      project: "team-project",
      owner: alice.accountId,
    });
    const pushReq = await signedReq("POST", "/sync/push", pushBody, charlie);
    const resp = await handleRequest(pushReq, store, blobs);
    expect(resp.status).toBe(401);
    expect(JSON.parse(resp.body).error).toContain("do not have access");
  });

  it("member can pull from owner's project", async () => {
    // Add Bob as member
    const addBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const addReq = await signedReq(
      "POST",
      "/projects/members/add",
      addBody,
      alice,
    );
    await handleRequest(addReq, store, blobs);

    // Bob pulls from Alice's project
    const pullBody = JSON.stringify({
      project: "team-project",
      since_batch: null,
      owner: alice.accountId,
    });
    const pullReq = await signedReq("POST", "/sync/pull", pullBody, bob);
    const resp = await handleRequest(pullReq, store, blobs);
    expect(resp.status).toBe(200);
  });

  it("non-member cannot pull from owner's project", async () => {
    const pullBody = JSON.stringify({
      project: "team-project",
      since_batch: null,
      owner: alice.accountId,
    });
    const pullReq = await signedReq("POST", "/sync/pull", pullBody, charlie);
    const resp = await handleRequest(pullReq, store, blobs);
    expect(resp.status).toBe(401);
  });

  it("member can check status of owner's project", async () => {
    // Add Bob
    const addBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const addReq = await signedReq(
      "POST",
      "/projects/members/add",
      addBody,
      alice,
    );
    await handleRequest(addReq, store, blobs);

    // Bob checks status
    const statusReq = await signedReq(
      "GET",
      `/sync/status?project=team-project&owner=${alice.accountId}`,
      "",
      bob,
    );
    const resp = await handleRequest(statusReq, store, blobs);
    expect(resp.status).toBe(200);
  });

  it("non-member cannot check status of owner's project", async () => {
    const statusReq = await signedReq(
      "GET",
      `/sync/status?project=team-project&owner=${alice.accountId}`,
      "",
      charlie,
    );
    const resp = await handleRequest(statusReq, store, blobs);
    expect(resp.status).toBe(401);
  });

  it("removed member loses access", async () => {
    // Add Bob, then remove him
    const addBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const addReq = await signedReq(
      "POST",
      "/projects/members/add",
      addBody,
      alice,
    );
    await handleRequest(addReq, store, blobs);

    // Confirm Bob can push
    const pushBody = JSON.stringify({
      project: "team-project",
      owner: alice.accountId,
    });
    let pushReq = await signedReq("POST", "/sync/push", pushBody, bob);
    let resp = await handleRequest(pushReq, store, blobs);
    expect(resp.status).toBe(200);

    // Remove Bob
    const removeBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const removeReq = await signedReq(
      "POST",
      "/projects/members/remove",
      removeBody,
      alice,
    );
    await handleRequest(removeReq, store, blobs);

    // Now Bob can't push
    pushReq = await signedReq("POST", "/sync/push", pushBody, bob);
    resp = await handleRequest(pushReq, store, blobs);
    expect(resp.status).toBe(401);
  });

  it("rejects adding a nonexistent account as member", async () => {
    const body = JSON.stringify({
      project: "team-project",
      member_account_id: "acct_does_not_exist",
    });
    const req = await signedReq(
      "POST",
      "/projects/members/add",
      body,
      alice,
    );
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(400);
    expect(JSON.parse(resp.body).error).toContain("does not exist");
  });

  it("adding same member twice is idempotent", async () => {
    const addBody = JSON.stringify({
      project: "team-project",
      member_account_id: bob.accountId,
    });
    const req1 = await signedReq(
      "POST",
      "/projects/members/add",
      addBody,
      alice,
    );
    await handleRequest(req1, store, blobs);

    const req2 = await signedReq(
      "POST",
      "/projects/members/add",
      addBody,
      alice,
    );
    const resp = await handleRequest(req2, store, blobs);
    expect(resp.status).toBe(200);

    // Check members — Bob should appear only once
    const listBody = JSON.stringify({
      project: "team-project",
    });
    const listReq = await signedReq(
      "POST",
      "/projects/members",
      listBody,
      alice,
    );
    const listResp = await handleRequest(listReq, store, blobs);
    const result = JSON.parse(listResp.body);
    const bobCount = result.members.filter(
      (m: string) => m === bob.accountId,
    ).length;
    expect(bobCount).toBe(1);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Blob token security
// ────────────────────────────────────────────────────────────────────────

describe("blob token security", () => {
  beforeEach(() => {
    initBlobSecret(new Uint8Array(32).fill(42));
  });

  it("valid token verifies correctly", () => {
    const url = signBlobUrl("http://localhost:4899", "some/blob/key.jsonl");
    const parsed = new URL(url);
    const key = parsed.pathname.slice("/blobs/".length);
    const expires = parsed.searchParams.get("expires");
    const token = parsed.searchParams.get("token");

    expect(verifyBlobToken(key, expires, token)).toBe(true);
  });

  it("rejects missing expires param", () => {
    expect(verifyBlobToken("key", null, "some-token")).toBe(false);
  });

  it("rejects missing token param", () => {
    expect(verifyBlobToken("key", "99999999999999", null)).toBe(false);
  });

  it("rejects expired token", () => {
    const pastExpiry = (Date.now() - 1000).toString();
    expect(verifyBlobToken("key", pastExpiry, "some-token")).toBe(false);
  });

  it("rejects tampered key", () => {
    const url = signBlobUrl("http://localhost:4899", "legit-key.jsonl");
    const parsed = new URL(url);
    const expires = parsed.searchParams.get("expires");
    const token = parsed.searchParams.get("token");

    // Try to use the token for a different key
    expect(verifyBlobToken("evil-key.jsonl", expires, token)).toBe(false);
  });

  it("rejects tampered expires", () => {
    const url = signBlobUrl("http://localhost:4899", "mykey.jsonl");
    const parsed = new URL(url);
    const token = parsed.searchParams.get("token");

    // Extend expiry far into the future
    const farFuture = (Date.now() + 999999999).toString();
    expect(verifyBlobToken("mykey.jsonl", farFuture, token)).toBe(false);
  });

  it("rejects tampered token", () => {
    const url = signBlobUrl("http://localhost:4899", "mykey.jsonl");
    const parsed = new URL(url);
    const expires = parsed.searchParams.get("expires");

    expect(
      verifyBlobToken("mykey.jsonl", expires, "0000000000000000"),
    ).toBe(false);
  });

  it("different secrets produce different tokens", () => {
    initBlobSecret(new Uint8Array(32).fill(1));
    const url1 = signBlobUrl("http://localhost:4899", "key.jsonl");
    const token1 = new URL(url1).searchParams.get("token");

    initBlobSecret(new Uint8Array(32).fill(2));
    const url2 = signBlobUrl("http://localhost:4899", "key.jsonl");
    const token2 = new URL(url2).searchParams.get("token");

    expect(token1).not.toBe(token2);
  });

  it("token from one secret does not verify with another", () => {
    initBlobSecret(new Uint8Array(32).fill(1));
    const url = signBlobUrl("http://localhost:4899", "key.jsonl");
    const parsed = new URL(url);
    const expires = parsed.searchParams.get("expires");
    const token = parsed.searchParams.get("token");

    // Switch to different secret
    initBlobSecret(new Uint8Array(32).fill(2));
    expect(verifyBlobToken("key.jsonl", expires, token)).toBe(false);
  });

  it("rejects non-numeric expires", () => {
    expect(verifyBlobToken("key", "not-a-number", "token")).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Cross-account access via owner param
// ────────────────────────────────────────────────────────────────────────

describe("cross-account access control via owner param", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;
  let alice: TestKeypair;
  let bob: TestKeypair;

  beforeEach(async () => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
    alice = await generateTestKeypair();
    bob = await generateTestKeypair();
    await signup(store, blobs, alice);
    await signup(store, blobs, bob);
    await createProject(store, blobs, alice, "secret-project");
  });

  it("stranger cannot push via owner param", async () => {
    const body = JSON.stringify({
      project: "secret-project",
      owner: alice.accountId,
    });
    const req = await signedReq("POST", "/sync/push", body, bob);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(401);
  });

  it("stranger cannot pull via owner param", async () => {
    const body = JSON.stringify({
      project: "secret-project",
      since_batch: null,
      owner: alice.accountId,
    });
    const req = await signedReq("POST", "/sync/pull", body, bob);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(401);
  });

  it("stranger cannot check status via owner param", async () => {
    const req = await signedReq(
      "GET",
      `/sync/status?project=secret-project&owner=${alice.accountId}`,
      "",
      bob,
    );
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(401);
  });

  it("stranger cannot add themselves as member", async () => {
    const body = JSON.stringify({
      project: "secret-project",
      member_account_id: bob.accountId,
      owner: alice.accountId,
    });
    const req = await signedReq("POST", "/projects/members/add", body, bob);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(401);
  });

  it("stranger cannot remove members", async () => {
    const body = JSON.stringify({
      project: "secret-project",
      member_account_id: alice.accountId,
      owner: alice.accountId,
    });
    const req = await signedReq(
      "POST",
      "/projects/members/remove",
      body,
      bob,
    );
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(401);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Rate limiting
// ────────────────────────────────────────────────────────────────────────

describe("rate limiting", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;
  let keys: TestKeypair;

  beforeEach(async () => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
    keys = await generateTestKeypair();

    // Use a tight limiter for testing: 3 requests per 60s window
    const limiter = new RateLimiter({
      auth: { maxRequests: 3, windowMs: 60_000 },
      sync: { maxRequests: 3, windowMs: 60_000 },
      projects: { maxRequests: 3, windowMs: 60_000 },
    });
    setRateLimiter(limiter);

    // Signup (uses 1 of 3 auth requests)
    await signup(store, blobs, keys);
  });

  afterEach(() => {
    setRateLimiter(null);
  });

  it("allows requests under the limit", async () => {
    const body = JSON.stringify({ name: "proj1" });
    const req = await signedReq("POST", "/projects", body, keys);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(201);
  });

  it("returns 429 when limit is exceeded", async () => {
    // Create 3 projects (uses all 3 project requests)
    for (let i = 0; i < 3; i++) {
      const body = JSON.stringify({ name: `proj${i}` });
      const req = await signedReq("POST", "/projects", body, keys);
      await handleRequest(req, store, blobs);
    }

    // 4th should be rate limited
    const body = JSON.stringify({ name: "proj-overflow" });
    const req = await signedReq("POST", "/projects", body, keys);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(429);
    const result = JSON.parse(resp.body);
    expect(result.error).toContain("rate limit");
    expect(result.retry_after_ms).toBeGreaterThan(0);
  });

  it("rate limits are per-account", async () => {
    // Exhaust keys' project limit
    for (let i = 0; i < 3; i++) {
      const body = JSON.stringify({ name: `proj${i}` });
      const req = await signedReq("POST", "/projects", body, keys);
      await handleRequest(req, store, blobs);
    }

    // Different user should still work
    const other = await generateTestKeypair();
    // Signup uses auth category, not projects
    await signup(store, blobs, other);

    const body = JSON.stringify({ name: "other-proj" });
    const req = await signedReq("POST", "/projects", body, other);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(201);
  });

  it("rate limits are per-category", async () => {
    // Create a project (uses 1 of 3 project requests)
    const createBody = JSON.stringify({ name: "test-proj" });
    const createReq = await signedReq("POST", "/projects", createBody, keys);
    await handleRequest(createReq, store, blobs);

    // Sync requests are a separate category — should have full budget
    for (let i = 0; i < 3; i++) {
      const body = JSON.stringify({ project: "test-proj" });
      const req = await signedReq("POST", "/sync/push", body, keys);
      const resp = await handleRequest(req, store, blobs);
      expect(resp.status).toBe(200);
    }

    // 4th sync request should be rate limited
    const body = JSON.stringify({ project: "test-proj" });
    const req = await signedReq("POST", "/sync/push", body, keys);
    const resp = await handleRequest(req, store, blobs);
    expect(resp.status).toBe(429);
  });
});
