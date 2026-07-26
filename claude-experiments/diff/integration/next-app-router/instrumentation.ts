// The `instrumentation.ts` convention: `register()` runs ONCE at server boot (the
// OpenTelemetry/Sentry-style hook), before the server accepts connections. diffpack
// bundles this natively at build and the orchestrator dynamic-imports it once before
// listen, so it never touches request latency. This one logs a boot marker the smoke
// test asserts appears exactly once, before "next-server listening".
export async function register() {
  console.log("INSTRUMENTATION_REGISTERED runtime=" + (process.env.NEXT_RUNTIME || "nodejs"));
}
