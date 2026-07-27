// An EDGE route handler: `runtime = "edge"` runs it in diffpack's lean WinterCG context
// (globalThis.EdgeRuntime advertised; Node built-ins rejected at build) instead of the
// Node.js runtime. It uses only Web APIs — Request/URL/Response and Web Crypto — exactly
// what a real Vercel/Cloudflare edge deployment provides.
export const runtime = "edge";

export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const name = url.searchParams.get("name") ?? "world";
  // Web Crypto (WinterCG) — a random request id, no Node `crypto` import needed.
  const id = crypto.randomUUID();
  return new Response(
    JSON.stringify({
      hello: name,
      id,
      runtime: typeof EdgeRuntime !== "undefined" ? String(EdgeRuntime) : "node",
      edge: typeof EdgeRuntime !== "undefined",
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}
