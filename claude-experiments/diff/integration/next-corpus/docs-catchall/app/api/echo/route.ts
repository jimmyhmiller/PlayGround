// A route handler exposing BOTH `GET` and `POST` at `/api/echo`. GET echoes the query
// string; POST echoes the request body. Local only (no network).
import { NextRequest } from "next/server";

export function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.get("q") ?? "";
  return Response.json({ method: "GET", q });
}

export async function POST(req: NextRequest) {
  const body = await req.text();
  return Response.json({ method: "POST", body });
}
