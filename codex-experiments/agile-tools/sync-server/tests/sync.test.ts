import { describe, it, expect, beforeEach } from "vitest";
import { createTestMetadataStore, createMockBlobStore } from "./helpers/mock-storage.js";
import { handleSignup } from "../src/handlers/auth-signup.js";
import { handleProjectsCreate } from "../src/handlers/projects-create.js";
import { handleProjectsList } from "../src/handlers/projects-list.js";
import { handleSyncPush } from "../src/handlers/sync-push.js";
import { handleSyncPull } from "../src/handlers/sync-pull.js";
import { handleSyncStatus } from "../src/handlers/sync-status.js";
import { deriveAccountId } from "../src/core/auth.js";
import type { HandlerRequest, MetadataStore, BlobStore } from "../src/core/types.js";

const TEST_PUBLIC_KEY = "dGVzdHB1YmtleQ==";
const TEST_ACCOUNT_ID = deriveAccountId(TEST_PUBLIC_KEY);

function authedReq(overrides: Partial<HandlerRequest>): HandlerRequest {
  return {
    method: "POST",
    path: "/",
    headers: {},
    body: "",
    accountId: TEST_ACCOUNT_ID,
    ...overrides,
  };
}

async function setupAccountAndProject(
  store: MetadataStore,
  projectName: string = "test-project",
) {
  await handleSignup(
    {
      method: "POST",
      path: "/auth/signup",
      headers: {},
      body: JSON.stringify({
        public_key: TEST_PUBLIC_KEY,
        display_name: "tester",
      }),
      accountId: TEST_ACCOUNT_ID,
    },
    store,
  );

  await handleProjectsCreate(
    authedReq({
      path: "/projects",
      body: JSON.stringify({ name: projectName }),
    }),
    store,
  );
}

describe("projects", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;

  beforeEach(() => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
  });

  it("should create a project", async () => {
    await setupAccountAndProject(store);

    const resp = await handleProjectsList(
      authedReq({ method: "GET", path: "/projects" }),
      store,
    );
    expect(resp.status).toBe(200);
    const body = JSON.parse(resp.body);
    expect(body.projects).toHaveLength(1);
    expect(body.projects[0].name).toBe("test-project");
  });

  it("should reject duplicate project", async () => {
    await setupAccountAndProject(store);
    await expect(
      handleProjectsCreate(
        authedReq({
          path: "/projects",
          body: JSON.stringify({ name: "test-project" }),
        }),
        store,
      ),
    ).rejects.toThrow("already exists");
  });
});

describe("sync push", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;

  beforeEach(async () => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
    await setupAccountAndProject(store);
  });

  it("should return upload URL and batch ID", async () => {
    const resp = await handleSyncPush(
      authedReq({
        path: "/sync/push",
        body: JSON.stringify({ project: "test-project" }),
      }),
      store,
      blobs,
    );

    expect(resp.status).toBe(200);
    const body = JSON.parse(resp.body);
    expect(body.batch_id).toBeTruthy();
    expect(body.upload_url).toContain("mock://blob/");
    expect(body.upload_url).toContain("test-project");
    expect(body.upload_url).toContain("batches");
  });

  it("should reject push to nonexistent project", async () => {
    await expect(
      handleSyncPush(
        authedReq({
          path: "/sync/push",
          body: JSON.stringify({ project: "no-such-project" }),
        }),
        store,
        blobs,
      ),
    ).rejects.toThrow("not found");
  });

  it("should generate unique batch IDs", async () => {
    const resp1 = await handleSyncPush(
      authedReq({
        path: "/sync/push",
        body: JSON.stringify({ project: "test-project" }),
      }),
      store,
      blobs,
    );
    const resp2 = await handleSyncPush(
      authedReq({
        path: "/sync/push",
        body: JSON.stringify({ project: "test-project" }),
      }),
      store,
      blobs,
    );

    const id1 = JSON.parse(resp1.body).batch_id;
    const id2 = JSON.parse(resp2.body).batch_id;
    expect(id1).not.toBe(id2);
  });
});

describe("sync pull", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;

  beforeEach(async () => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
    await setupAccountAndProject(store);
  });

  it("should return empty batches when nothing pushed", async () => {
    const resp = await handleSyncPull(
      authedReq({
        path: "/sync/pull",
        body: JSON.stringify({ project: "test-project", since_batch: null }),
      }),
      store,
      blobs,
    );

    expect(resp.status).toBe(200);
    const body = JSON.parse(resp.body);
    expect(body.batches).toHaveLength(0);
  });

  it("should return download URLs for uploaded batches", async () => {
    // Simulate client uploading batches by writing to blob store
    const batchKey = `projects/${TEST_ACCOUNT_ID}/test-project/batches/000000001_abc.jsonl`;
    blobs.putBlob(batchKey, '{"id":"evt_001","type":"issue.create"}\n');

    const resp = await handleSyncPull(
      authedReq({
        path: "/sync/pull",
        body: JSON.stringify({ project: "test-project", since_batch: null }),
      }),
      store,
      blobs,
    );

    const body = JSON.parse(resp.body);
    expect(body.batches).toHaveLength(1);
    expect(body.batches[0].id).toBe("000000001_abc");
    expect(body.batches[0].url).toContain("mock://blob/");
  });

  it("should filter batches by cursor", async () => {
    const prefix = `projects/${TEST_ACCOUNT_ID}/test-project/batches/`;
    blobs.putBlob(`${prefix}000000001_aaa.jsonl`, "event1\n");
    blobs.putBlob(`${prefix}000000002_bbb.jsonl`, "event2\n");
    blobs.putBlob(`${prefix}000000003_ccc.jsonl`, "event3\n");

    const resp = await handleSyncPull(
      authedReq({
        path: "/sync/pull",
        body: JSON.stringify({
          project: "test-project",
          since_batch: "000000001_aaa",
        }),
      }),
      store,
      blobs,
    );

    const body = JSON.parse(resp.body);
    expect(body.batches).toHaveLength(2);
    expect(body.batches[0].id).toBe("000000002_bbb");
    expect(body.batches[1].id).toBe("000000003_ccc");
  });

  it("should reject pull from nonexistent project", async () => {
    await expect(
      handleSyncPull(
        authedReq({
          path: "/sync/pull",
          body: JSON.stringify({ project: "nope", since_batch: null }),
        }),
        store,
        blobs,
      ),
    ).rejects.toThrow("not found");
  });
});

describe("sync status", () => {
  let store: MetadataStore;
  let blobs: ReturnType<typeof createMockBlobStore>;

  beforeEach(async () => {
    store = createTestMetadataStore();
    blobs = createMockBlobStore();
    await setupAccountAndProject(store);
  });

  it("should return zero batches for empty project", async () => {
    const resp = await handleSyncStatus(
      authedReq({
        method: "GET",
        path: "/sync/status?project=test-project",
      }),
      store,
      blobs,
    );

    expect(resp.status).toBe(200);
    const body = JSON.parse(resp.body);
    expect(body.total_batches).toBe(0);
    expect(body.latest_batch).toBeNull();
  });

  it("should count batches", async () => {
    const prefix = `projects/${TEST_ACCOUNT_ID}/test-project/batches/`;
    blobs.putBlob(`${prefix}000000001_aaa.jsonl`, "event1\n");
    blobs.putBlob(`${prefix}000000002_bbb.jsonl`, "event2\n");

    const resp = await handleSyncStatus(
      authedReq({
        method: "GET",
        path: "/sync/status?project=test-project",
      }),
      store,
      blobs,
    );

    const body = JSON.parse(resp.body);
    expect(body.total_batches).toBe(2);
    expect(body.latest_batch).toBe("000000002_bbb");
  });

  it("should reject missing project param", async () => {
    await expect(
      handleSyncStatus(
        authedReq({ method: "GET", path: "/sync/status" }),
        store,
        blobs,
      ),
    ).rejects.toThrow("project");
  });
});
