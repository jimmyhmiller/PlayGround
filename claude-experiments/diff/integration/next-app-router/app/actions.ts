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
