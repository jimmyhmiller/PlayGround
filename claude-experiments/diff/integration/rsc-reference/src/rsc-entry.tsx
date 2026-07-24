// The REACT-SERVER render entry (Target::ReactServer). diffpack bundles this graph
// under the `react-server` export condition, so `react-server-dom-webpack/server`
// resolves to the real flight writer and `react` to its react-server build — both
// inlined, so this bundle runs as plain Node with no `--conditions` flag. It never
// shares React with the SSR/browser graphs; the two only exchange flight BYTES,
// which is why the orchestrator runs this bundle in its own child process.
//
// It is a small CLI the orchestrator (scripts/rsc/rsc-server.mjs) spawns per
// request:
//   • `render <manifest1Path>`  — render <Page/> to a flight stream using the
//     client-references manifest (Manifest #1) as `bundlerConfig`, write the flight
//     bytes to stdout.
//   • `action <id> <manifest1Path>` — read the `encodeReply`d arguments from stdin,
//     dispatch through `handleServerAction` (decodeReply -> resolver -> apply ->
//     renderToReadableStream), write the result flight to stdout.
import { renderToReadableStream } from "react-server-dom-webpack/server";
import { createElement } from "react";
import { readFileSync } from "node:fs";
import { Page } from "./Page";
import { handleServerAction } from "#diffpack-rsc-action-handler";

async function streamToStdout(stream: ReadableStream<Uint8Array>): Promise<void> {
  const reader = stream.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    process.stdout.write(Buffer.from(value));
  }
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function main() {
  const [op, ...rest] = process.argv.slice(2);
  if (op === "render") {
    const manifestPath = rest[0];
    if (!manifestPath) throw new Error("rsc-entry render: missing manifest path argument");
    const bundlerConfig = JSON.parse(readFileSync(manifestPath, "utf8"));
    const stream = renderToReadableStream(createElement(Page), bundlerConfig);
    await streamToStdout(stream);
    return;
  }
  if (op === "action") {
    const id = rest[0];
    const manifestPath = rest[1];
    if (!id) throw new Error("rsc-entry action: missing action id argument");
    if (!manifestPath) throw new Error("rsc-entry action: missing manifest path argument");
    const bundlerConfig = JSON.parse(readFileSync(manifestPath, "utf8"));
    const body = await readStdin();
    const request = new Request("http://diffpack.local/_action/", {
      method: "POST",
      headers: { "x-diffpack-action-id": id, "content-type": "application/json" },
      body,
    });
    const response = await handleServerAction(request, bundlerConfig);
    if (!response.body) throw new Error("rsc-entry action: handler produced no response body");
    await streamToStdout(response.body as ReadableStream<Uint8Array>);
    return;
  }
  throw new Error(`rsc-entry: unknown op ${JSON.stringify(op)}; expected "render" or "action"`);
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
