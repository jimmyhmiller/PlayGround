import type { HandlerRequest, HandlerResponse } from "../core/types.js";
import { verifySignature, deriveAccountId, sha256Hex } from "../core/auth.js";
import { AuthError } from "../core/errors.js";

const MAX_TIMESTAMP_DRIFT_MS = 5 * 60 * 1000; // 5 minutes

export function authenticate(
  req: HandlerRequest,
): string {
  const publicKey = req.headers["x-scope-publickey"];
  const timestamp = req.headers["x-scope-timestamp"];
  const signature = req.headers["x-scope-signature"];

  if (!publicKey || !timestamp || !signature) {
    throw new AuthError("missing auth headers (X-Scope-PublicKey, X-Scope-Timestamp, X-Scope-Signature)");
  }

  const requestTime = new Date(timestamp).getTime();
  if (isNaN(requestTime)) {
    throw new AuthError("invalid timestamp format");
  }

  const drift = Math.abs(Date.now() - requestTime);
  if (drift > MAX_TIMESTAMP_DRIFT_MS) {
    throw new AuthError("request timestamp too old or too far in the future");
  }

  const bodyHash = sha256Hex(req.body || "");
  const payload = `${req.method}:${req.path}:${timestamp}:${bodyHash}`;

  if (!verifySignature(publicKey, signature, payload)) {
    throw new AuthError("invalid signature");
  }

  return deriveAccountId(publicKey);
}
