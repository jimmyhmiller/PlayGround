// A route handler (`route.ts`) exposing `GET /api/health` — serves an HTTP endpoint
// (not a React page). Local data only.
export function GET() {
  return Response.json({ status: "ok" });
}
