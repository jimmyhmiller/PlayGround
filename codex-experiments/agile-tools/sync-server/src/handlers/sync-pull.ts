import type { HandlerRequest, HandlerResponse, MetadataStore, BlobStore } from "../core/types.js";
import { PullRequestSchema } from "../core/validate.js";
import { createPullUrls } from "../core/sync.js";
import { ValidationError } from "../core/errors.js";

export async function handleSyncPull(
  req: HandlerRequest,
  store: MetadataStore,
  blobs: BlobStore,
): Promise<HandlerResponse> {
  const parsed = PullRequestSchema.safeParse(JSON.parse(req.body));
  if (!parsed.success) {
    throw new ValidationError(parsed.error.message);
  }

  const { project, since_batch, owner } = parsed.data;
  const result = await createPullUrls(
    store,
    blobs,
    req.accountId!,
    project,
    since_batch ?? null,
    owner,
  );

  return {
    status: 200,
    body: JSON.stringify(result),
  };
}
