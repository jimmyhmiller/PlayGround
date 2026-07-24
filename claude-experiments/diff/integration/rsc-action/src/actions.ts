"use server";

// A module-level `"use server"` boundary: every export is a server action. The
// real body runs ONLY on the server; the client build ships a thin RPC stub.
export async function increment(n) {
  return n + 1;
}

export async function add(a, b) {
  return a + b;
}
