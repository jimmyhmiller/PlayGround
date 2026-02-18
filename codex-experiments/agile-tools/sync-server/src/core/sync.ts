import type { MetadataStore, BlobStore } from "./types.js";
import { batchPrefix, batchKey } from "./storage.js";
import { assertAccess } from "./projects.js";
import { NotFoundError } from "./errors.js";

function generateBatchId(): string {
  const ts = Date.now().toString(36).padStart(9, "0");
  const rand = Math.random().toString(36).slice(2, 10);
  return `${ts}_${rand}`;
}

// Resolve the owner account ID. If `owner` is provided, use it (for shared projects).
// Otherwise default to the caller's own account.
function resolveOwner(callerAccountId: string, owner?: string | null): string {
  return owner || callerAccountId;
}

export async function createPushUrl(
  store: MetadataStore,
  blobs: BlobStore,
  callerAccountId: string,
  projectName: string,
  owner?: string | null,
): Promise<{ batch_id: string; upload_url: string }> {
  const ownerAccountId = resolveOwner(callerAccountId, owner);
  const project = await store.getProject(ownerAccountId, projectName);
  if (!project) {
    throw new NotFoundError(
      `project '${projectName}' not found. Create it first with POST /projects.`,
    );
  }
  await assertAccess(store, project, callerAccountId);

  const batchId = generateBatchId();
  const key = batchKey(ownerAccountId, projectName, batchId);
  const url = await blobs.createUploadUrl(key);

  return { batch_id: batchId, upload_url: url };
}

export async function createPullUrls(
  store: MetadataStore,
  blobs: BlobStore,
  callerAccountId: string,
  projectName: string,
  sinceBatch: string | null | undefined,
  owner?: string | null,
): Promise<{ batches: { id: string; url: string }[] }> {
  const ownerAccountId = resolveOwner(callerAccountId, owner);
  const project = await store.getProject(ownerAccountId, projectName);
  if (!project) {
    throw new NotFoundError(`project '${projectName}' not found.`);
  }
  await assertAccess(store, project, callerAccountId);

  const prefix = batchPrefix(ownerAccountId, projectName);
  const allKeys = await blobs.listKeys(prefix);

  const batches: { id: string; url: string }[] = [];
  for (const key of allKeys) {
    const filename = key.split("/").pop() ?? "";
    const id = filename.replace(".jsonl", "");
    if (!id) continue;
    if (sinceBatch && id <= sinceBatch) continue;

    const url = await blobs.createDownloadUrl(key);
    batches.push({ id, url });
  }

  batches.sort((a, b) => a.id.localeCompare(b.id));
  return { batches };
}

export async function getBatchCount(
  store: MetadataStore,
  blobs: BlobStore,
  callerAccountId: string,
  projectName: string,
  owner?: string | null,
): Promise<{ total_batches: number; latest_batch: string | null }> {
  const ownerAccountId = resolveOwner(callerAccountId, owner);
  const project = await store.getProject(ownerAccountId, projectName);
  if (!project) {
    throw new NotFoundError(`project '${projectName}' not found.`);
  }
  await assertAccess(store, project, callerAccountId);

  const prefix = batchPrefix(ownerAccountId, projectName);
  const allKeys = await blobs.listKeys(prefix);

  let latest: string | null = null;
  for (const key of allKeys) {
    const filename = key.split("/").pop() ?? "";
    const id = filename.replace(".jsonl", "");
    if (id && (!latest || id > latest)) {
      latest = id;
    }
  }

  return { total_batches: allKeys.length, latest_batch: latest };
}
