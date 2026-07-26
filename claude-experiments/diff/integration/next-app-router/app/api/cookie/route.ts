// A Route Handler that writes cookies via next/headers `cookies()` (server-side cookie
// writes, the cluster this fixture exercises). `GET /api/cookie` sets `route_cookie` and
// deletes `stale_cookie`; diffpack collects both onto the handler's per-request store and
// the orchestrator emits them as Set-Cookie on the 200 response. `next build`/`next start`
// accept this unchanged: cookies() is a real next/headers export usable in a Route Handler.
import { cookies } from "next/headers";

export async function GET(): Promise<Response> {
  const store = await cookies();
  store.set("route_cookie", "from-route", { path: "/", httpOnly: true, sameSite: "lax" });
  store.delete("stale_cookie");
  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}
