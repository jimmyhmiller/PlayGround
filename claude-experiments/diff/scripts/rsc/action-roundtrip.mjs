// RSC Slice C / R2 oracle — prove a `"use server"` exported function, called
// through diffpack's CLIENT stub id, round-trips to the SERVER implementation and
// returns its real result, using the REAL `react-server-dom-webpack` runtime.
//
// It exercises the exact Rust transform outputs (no hand-written stand-ins):
//
//   • `diffpack rsc-transform actions.ts client`  → createServerReference stubs
//   • `diffpack rsc-transform actions.ts server`  → registerServerReference module
//   • `diffpack rsc-resolver <fixture>`           → getServerActionById resolver
//   • src/rsc_runtime/action_handler.js           → handleServerAction dispatcher
//   • src/rsc_runtime/call_server.js              → callServer transport
//
// Two assertions (per docs/RSC_SPEC.md Slice R2):
//   (a) the client stub id === the server `$$id` prefix#name === the resolver key.
//   (b) end-to-end: encodeReply(args)  [client, default condition]
//         → handleServerAction: decodeReply → getServerActionById → apply
//           → renderToReadableStream  [server, --conditions=react-server]
//         → createFromReadableStream  [client, default condition] === real result.
//
// The two React conditions cannot coexist in one process (the react-server build
// is a different React), so the server half runs in a child spawned with
// `--conditions=react-server`; the flight bytes cross the process boundary via a
// file. Fails loudly (never skips) if the pinned RSC deps are absent.

import { spawnSync } from "node:child_process";
import {
  mkdirSync,
  writeFileSync,
  readFileSync,
  realpathSync,
  existsSync,
  rmSync,
  createReadStream,
} from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Readable } from "node:stream";

const here = dirname(fileURLToPath(import.meta.url));
const repo = realpathSync(join(here, "..", ".."));
const fixture = process.argv[2]
  ? realpathSync(process.argv[2])
  : join(repo, "integration", "rsc-action");
const diffpack = join(repo, "target", "release", "diffpack");

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exit(1);
}

if (!existsSync(diffpack)) fail(`diffpack binary not found at ${diffpack} — run cargo build --release`);
const actions = join(fixture, "src", "actions.ts");
if (!existsSync(actions)) fail(`fixture action module not found at ${actions}`);
if (!existsSync(join(fixture, "node_modules", "react-server-dom-webpack"))) {
  fail(
    `react-server-dom-webpack not installed in ${fixture}; ` +
      `run \`npm install\` in the fixture (the pinned experimental RSC deps are required, never skipped)`,
  );
}

function run(args) {
  const result = spawnSync(diffpack, args, { encoding: "utf8" });
  if (result.status !== 0) {
    fail(`diffpack ${args.join(" ")} failed (exit ${result.status}):\n${result.stderr || result.stdout}`);
  }
  return result.stdout;
}

// --- Produce the real transform outputs -------------------------------------
const clientStub = run(["rsc-transform", actions, "client"]);
const serverReg = run(["rsc-transform", actions, "server"]);
const resolver = run(["rsc-resolver", fixture]);
const actionsReal = realpathSync(actions);

// --- Assertion (a): id agreement across the three graphs --------------------
// The canonical module id every graph shares.
const moduleId = actionsReal;
for (const name of ["increment", "add"]) {
  const id = `${moduleId}#${name}`;
  if (!clientStub.includes(`createServerReference(${JSON.stringify(id)}, callServer)`)) {
    fail(`client stub is missing createServerReference for id ${id}:\n${clientStub}`);
  }
  if (!serverReg.includes(`__rsr(${name}, ${JSON.stringify(moduleId)}, ${JSON.stringify(name)});`)) {
    fail(`server registration is missing registerServerReference for ${name}:\n${serverReg}`);
  }
  if (!resolver.includes(`${JSON.stringify(id)}: { importer:`)) {
    fail(`resolver is missing the manifest key ${id}:\n${resolver}`);
  }
}
// The server body must be present on the server, absent from the client.
if (!serverReg.includes("return n + 1")) fail("server registration dropped the real body");
if (clientStub.includes("return n + 1")) fail("client stub leaked the server body");
console.log("OK (a): client stub id === server $$id === resolver key, for every export");

// --- Assemble the oracle sandbox (inside the fixture, so node_modules resolves)
const sandbox = join(fixture, ".diffpack-rsc-oracle");
rmSync(sandbox, { recursive: true, force: true });
mkdirSync(sandbox, { recursive: true });

// The server-transformed action module (self-registers on load), and the resolver
// pointed at IT (so registerServerReference is exercised in the dispatch path).
// Only the importer literal is repointed — the manifest KEY (which carries the id)
// is left intact so id resolution still matches the client stub.
writeFileSync(join(sandbox, "actions.server.mjs"), serverReg);
const importerLiteral = `import(${JSON.stringify(actionsReal)})`;
const patchedResolver = resolver.split(importerLiteral).join('import("./actions.server.mjs")');
if (patchedResolver === resolver) fail(`could not repoint the resolver importer (${importerLiteral} not found)`);
writeFileSync(join(sandbox, "resolver.mjs"), patchedResolver);

// The real embedded runtime files, verbatim.
writeFileSync(join(sandbox, "action_handler.js"), readFileSync(join(repo, "src", "rsc_runtime", "action_handler.js")));
writeFileSync(join(sandbox, "call_server.js"), readFileSync(join(repo, "src", "rsc_runtime", "call_server.js")));
writeFileSync(join(sandbox, "client-stub.mjs"), clientStub);

// The client API, re-exported from inside the fixture so it resolves against the
// fixture's node_modules (this parent script lives outside the fixture tree).
writeFileSync(
  join(sandbox, "client-api.mjs"),
  'export { encodeReply, createFromReadableStream } from "react-server-dom-webpack/client";\n',
);

// The subpath import map the transforms + runtime rely on.
writeFileSync(
  join(sandbox, "package.json"),
  JSON.stringify(
    {
      name: "diffpack-rsc-oracle",
      private: true,
      type: "module",
      imports: {
        "#diffpack-rsc-action-resolver": "./resolver.mjs",
        "#diffpack-call-server": "./call_server.js",
      },
    },
    null,
    2,
  ),
);

// The SERVER half — runs in the react-server condition. Reads {id, bodyPath} from
// argv, dispatches through the REAL handleServerAction, writes flight bytes out.
writeFileSync(
  join(sandbox, "server-half.mjs"),
  `import { readFileSync, writeFileSync } from "node:fs";
import { handleServerAction } from "./action_handler.js";
const [id, bodyPath, outPath] = process.argv.slice(2);
const body = readFileSync(bodyPath, "utf8");
const request = new Request("http://oracle.invalid/_action/", {
  method: "POST",
  headers: { "x-diffpack-action-id": id, "content-type": "application/json" },
  body,
});
const response = await handleServerAction(request, {});
const buffer = Buffer.from(await response.arrayBuffer());
writeFileSync(outPath, buffer);
`,
);

// --- Prove the CLIENT stub module actually loads + is callable ---------------
// (default condition: react-server-dom-webpack/client + #diffpack-call-server).
const stubModule = await import(join(sandbox, "client-stub.mjs"));
if (typeof stubModule.increment !== "function") {
  fail("client stub export `increment` is not a callable server reference");
}
console.log("OK: client stub module loads and its exports are callable server references");

// --- Assertion (b): the end-to-end round-trip -------------------------------
const { encodeReply, createFromReadableStream } =
  await import(join(sandbox, "client-api.mjs"));

async function roundTrip(name, args) {
  const id = `${moduleId}#${name}`;
  const body = await encodeReply(args);
  if (typeof body !== "string") fail(`expected a JSON string body for ${name}, got ${typeof body}`);
  const bodyPath = join(sandbox, `body-${name}.txt`);
  const outPath = join(sandbox, `flight-${name}.bin`);
  writeFileSync(bodyPath, body);

  const child = spawnSync(process.execPath, ["--conditions=react-server", join(sandbox, "server-half.mjs"), id, bodyPath, outPath], {
    encoding: "utf8",
  });
  if (child.status !== 0) {
    fail(`server half failed for ${name} (exit ${child.status}):\n${child.stderr || child.stdout}`);
  }

  const stream = Readable.toWeb(createReadStream(outPath));
  // The node client (client.node) reconstructs through the SSR consumer manifest
  // (Manifest #2). The action's result is a plain value with no client references,
  // so the maps are empty — but `moduleMap`/`moduleLoading` must be present
  // (reading them off `undefined` crashes), exactly as docs/RSC_SPEC.md §1 notes.
  const result = await createFromReadableStream(stream, {
    callServer: () => fail("no nested server call expected"),
    serverConsumerManifest: {
      moduleMap: {},
      serverModuleMap: null,
      moduleLoading: { prefix: "", crossOrigin: null },
    },
  });
  return result;
}

const inc = await roundTrip("increment", [5]);
if (inc !== 6) fail(`increment(5) round-trip returned ${JSON.stringify(inc)}, expected 6`);
const sum = await roundTrip("add", [3, 4]);
if (sum !== 7) fail(`add(3,4) round-trip returned ${JSON.stringify(sum)}, expected 7`);

console.log(`OK (b): increment(5) => ${inc}, add(3,4) => ${sum} (client encodeReply → handleServerAction → createFromReadableStream)`);

rmSync(sandbox, { recursive: true, force: true });
console.log("PASS: 'use server' client stub round-trips to the server implementation via the real react-server-dom runtime");
