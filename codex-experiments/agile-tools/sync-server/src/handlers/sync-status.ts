import type { HandlerRequest, HandlerResponse, MetadataStore, BlobStore } from "../core/types.js";
import { getBatchCount } from "../core/sync.js";
import { ValidationError } from "../core/errors.js";

export async function handleSyncStatus(
  req: HandlerRequest,
  store: MetadataStore,
  blobs: BlobStore,
): Promise<HandlerResponse> {
  const url = new URL(req.path, "http://localhost");
  const project = url.searchParams.get("project");
  if (!project) {
    throw new ValidationError("'project' query parameter is required");
  }
  const owner = url.searchParams.get("owner");

  const result = await getBatchCount(store, blobs, req.accountId!, project, owner);

  return {
    status: 200,
    body: JSON.stringify(result),
  };
}
