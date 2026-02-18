import type { HandlerRequest, HandlerResponse, MetadataStore } from "../core/types.js";
import { deriveAccountId } from "../core/auth.js";
import { SignupSchema } from "../core/validate.js";
import { AuthError, ValidationError, ConflictError } from "../core/errors.js";

export async function handleSignup(
  req: HandlerRequest,
  store: MetadataStore,
): Promise<HandlerResponse> {
  const parsed = SignupSchema.safeParse(JSON.parse(req.body));
  if (!parsed.success) {
    throw new ValidationError(parsed.error.message);
  }

  const { public_key, display_name } = parsed.data;

  // Verify the signing key matches the public key being registered.
  // req.accountId is set by the auth middleware from the request signature.
  const claimedAccountId = deriveAccountId(public_key);
  if (req.accountId !== claimedAccountId) {
    throw new AuthError(
      "signing key does not match the public key being registered",
    );
  }

  const existing = await store.getAccount(claimedAccountId);
  if (existing) {
    throw new ConflictError("account already exists");
  }

  const account = {
    account_id: claimedAccountId,
    public_key,
    display_name,
    created_at: new Date().toISOString(),
  };

  await store.createAccount(account);

  return {
    status: 201,
    body: JSON.stringify({ account_id: claimedAccountId }),
  };
}
