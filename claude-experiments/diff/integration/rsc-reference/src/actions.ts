"use server";

// A real RSC server action. Its body MUST NOT reach the browser: in the CLIENT
// graph diffpack rewrites this module into `createServerReference(id, callServer)`
// stubs (the body dropped); in the REACT-SERVER graph it keeps the body and calls
// `registerServerReference(increment, "<moduleId>", "increment")`, so the action
// the Server Component passes into the client island serializes into the flight as
// a server reference the browser invokes over `/_action/`.
export async function increment(n: number): Promise<number> {
  return n + 1;
}
