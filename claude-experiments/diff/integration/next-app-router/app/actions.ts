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
// `revalidateTag("products", "max")` here busts every prerendered page that read the `products`
// tag (the /products page). diffpack collects the tag off the action's per-request store
// and the orchestrator marks the matching cache entries stale — the next request to
// /products serves STALE and regenerates in the background. `next build` accepts this.
import { revalidateTag } from "next/cache";

export async function revalidateProducts(): Promise<void> {
  revalidateTag("products", "max");
}

// Server-side cookie writes from a Server Action (next/headers cookies()). Invoked over
// `/_action/`, this sets `pref` and clears `old_pref`; diffpack collects both onto the
// action's per-request store and the orchestrator merges them into the action's 200
// response as Set-Cookie. `next build` accepts cookies() writes in a Server Action.
import { cookies } from "next/headers";

export async function setPrefCookie(): Promise<string> {
  const store = await cookies();
  store.set("pref", "dark", { path: "/", sameSite: "lax" });
  store.delete("old_pref");
  return "ok";
}
