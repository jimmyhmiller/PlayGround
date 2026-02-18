import type { HandlerRequest, HandlerResponse, MetadataStore, BlobStore } from "./core/types.js";
import { AppError } from "./core/errors.js";
import { authenticate } from "./middleware/authenticate.js";
import { RateLimiter, RateLimitError } from "./core/rate-limit.js";
import { handleSignup } from "./handlers/auth-signup.js";
import { handleSyncPush } from "./handlers/sync-push.js";
import { handleSyncPull } from "./handlers/sync-pull.js";
import { handleSyncStatus } from "./handlers/sync-status.js";
import { handleProjectsCreate } from "./handlers/projects-create.js";
import { handleProjectsList } from "./handlers/projects-list.js";
import { handleMemberAdd, handleMemberRemove, handleMemberList } from "./handlers/members.js";

type HandlerFn = (
  req: HandlerRequest,
  store: MetadataStore,
  blobs: BlobStore,
) => Promise<HandlerResponse>;

interface Route {
  method: string;
  pattern: string;
  handler: HandlerFn;
}

// All routes require auth.
const routes: Route[] = [
  { method: "POST", pattern: "/auth/signup", handler: handleSignup as HandlerFn },
  { method: "POST", pattern: "/sync/push", handler: handleSyncPush },
  { method: "POST", pattern: "/sync/pull", handler: handleSyncPull },
  { method: "GET", pattern: "/sync/status", handler: handleSyncStatus },
  { method: "POST", pattern: "/projects", handler: handleProjectsCreate as HandlerFn },
  { method: "GET", pattern: "/projects", handler: handleProjectsList as HandlerFn },
  { method: "POST", pattern: "/projects/members/add", handler: handleMemberAdd as HandlerFn },
  { method: "POST", pattern: "/projects/members/remove", handler: handleMemberRemove as HandlerFn },
  { method: "POST", pattern: "/projects/members", handler: handleMemberList as HandlerFn },
];

function matchRoute(method: string, path: string): Route | null {
  const pathWithoutQuery = path.split("?")[0];
  for (const route of routes) {
    if (route.method === method && route.pattern === pathWithoutQuery) {
      return route;
    }
  }
  return null;
}

// Shared rate limiter instance. Tests can replace via setRateLimiter().
let rateLimiter: RateLimiter | null = new RateLimiter();

export function setRateLimiter(limiter: RateLimiter | null): void {
  rateLimiter = limiter;
}

export async function handleRequest(
  req: HandlerRequest,
  store: MetadataStore,
  blobs: BlobStore,
): Promise<HandlerResponse> {
  try {
    const route = matchRoute(req.method, req.path);
    if (!route) {
      return jsonResponse(404, { error: "not found" });
    }

    req.accountId = authenticate(req);

    // Rate limit after auth so we have the account ID
    if (rateLimiter) {
      const category = RateLimiter.categorize(req.path);
      rateLimiter.check(req.accountId, category);
    }

    const response = await route.handler(req, store, blobs);
    return {
      ...response,
      headers: {
        "Content-Type": "application/json",
        ...response.headers,
      },
    };
  } catch (err) {
    if (err instanceof RateLimitError) {
      return jsonResponse(429, { error: err.message, retry_after_ms: err.retryAfterMs });
    }
    if (err instanceof AppError) {
      return jsonResponse(err.statusCode, { error: err.message });
    }
    console.error("Unhandled error:", err);
    return jsonResponse(500, { error: "internal server error" });
  }
}

function jsonResponse(status: number, body: unknown): HandlerResponse {
  return {
    status,
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
  };
}
