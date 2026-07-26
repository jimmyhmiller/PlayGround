"use server";

// A real RSC server action in an app-router app. Its body MUST NOT reach the
// browser: in the client graph diffpack rewrites this module into
// `createServerReference(id, callServer)` stubs (body dropped); in the react-server
// graph it keeps the body and calls `registerServerReference`. The action the
// Server Component passes into the client island serializes into the flight as a
// server reference the browser invokes over `/_action/`.
export async function increment(n: number): Promise<number> {
  return n + 1;
}

// On-demand revalidation from a Server Action (next/cache). Calling
// `revalidateTag("products")` here busts every prerendered page that read the `products`
// tag (the /products page). diffpack collects the tag off the action's per-request store
// and the orchestrator marks the matching cache entries stale — the next request to
// /products serves STALE and regenerates in the background. `next build` accepts this.
import { revalidateTag } from "next/cache";

export async function revalidateProducts(): Promise<void> {
  revalidateTag("products");
}
