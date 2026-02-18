import type { HandlerRequest, HandlerResponse, MetadataStore, BlobStore } from "../core/types.js";
import { PushRequestSchema } from "../core/validate.js";
import { createPushUrl } from "../core/sync.js";
import { ValidationError } from "../core/errors.js";

export async function handleSyncPush(
  req: HandlerRequest,
  store: MetadataStore,
  blobs: BlobStore,
): Promise<HandlerResponse> {
  const parsed = PushRequestSchema.safeParse(JSON.parse(req.body));
  if (!parsed.success) {
    throw new ValidationError(parsed.error.message);
  }

  const { project, owner } = parsed.data;
  const result = await createPushUrl(store, blobs, req.accountId!, project, owner);

  return {
    status: 200,
    body: JSON.stringify(result),
  };
}
