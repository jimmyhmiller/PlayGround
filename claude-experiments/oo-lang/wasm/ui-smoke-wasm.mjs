// ui-smoke-wasm.mjs — real headless-browser test of the IN-BROWSER viewer.
//
// Serves the repo over http, opens wasm/index.html in system Chrome, and verifies that the
// Scry VM boots inside the page and the unmodified React viewer drives it with NO server:
// every pane's data comes from globalThis.__scryWasm.eval(), not fetch("/eval").
//
// Zero npm deps — drives Chrome over the DevTools Protocol via Node's built-in WebSocket,
// same approach as tests/ui-smoke.mjs. SKIPPED (exit 0) if no Chrome is present.
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { join, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function findChrome() {
  const cands = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
  ];
  for (const c of cands) if (spawnSync("test", ["-x", c]).status === 0) return c;
  return null;
}

const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
               ".css": "text/css", ".wasm": "application/wasm", ".scry": "text/plain" };

async function main() {
  const chrome = findChrome();
  if (!chrome) { console.log("SKIPPED ui-smoke-wasm: no Chrome/Chromium binary found"); process.exit(0); }

  // 1. static server over the repo root (so ../viewer/* resolves from /wasm/index.html)
  const server = createServer(async (req, res) => {
    try {
      const path = join(ROOT, decodeURIComponent(req.url.split("?")[0]));
      const body = await readFile(path);
      res.writeHead(200, { "Content-Type": MIME[extname(path)] || "application/octet-stream" });
      res.end(body);
    } catch { res.writeHead(404); res.end("not found"); }
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const port = server.address().port;

  const profile = mkdtempSync(join(tmpdir(), "scry-wasm-ui-"));
  const dport = 9333 + Math.floor(Math.random() * 400);
  const cproc = spawn(chrome, ["--headless=new", "--disable-gpu", "--no-first-run",
    "--no-default-browser-check", `--remote-debugging-port=${dport}`,
    `--user-data-dir=${profile}`, "about:blank"], { stdio: ["ignore", "pipe", "pipe"] });

  const cleanup = () => { try { cproc.kill("SIGKILL"); } catch {} server.close(); };
  const fail = (m) => { console.log("FAIL " + m); cleanup(); process.exit(1); };

  try {
    let wsURL = null;
    for (let i = 0; i < 100 && !wsURL; i++) {
      try {
        const targets = await (await fetch(`http://127.0.0.1:${dport}/json/list`)).json();
        wsURL = targets.find((t) => t.type === "page")?.webSocketDebuggerUrl || null;
      } catch {}
      if (!wsURL) await sleep(150);
    }
    if (!wsURL) fail("Chrome DevTools endpoint never came up");

    const ws = new WebSocket(wsURL);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error("ws error")); });
    let msgId = 0; const pending = new Map();
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id);
        m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result); }
    };
    const send = (method, params = {}) => new Promise((res, rej) => {
      const id = ++msgId; pending.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params })); });
    const evalPage = async (expr, awaitPromise = false) => {
      const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise });
      if (r.exceptionDetails) throw new Error("page exception: " + JSON.stringify(r.exceptionDetails.exception || r.exceptionDetails));
      return r.result.value;
    };

    // collect page errors so a silent boot failure can't masquerade as success
    const pageErrors = [];
    await send("Runtime.enable");
    ws.addEventListener("message", (ev) => {
      const m = JSON.parse(ev.data);
      if (m.method === "Runtime.exceptionThrown")
        pageErrors.push(m.params?.exceptionDetails?.exception?.description || "exception");
    });

    // 2. open the in-browser viewer
    await send("Page.enable");
    await send("Page.navigate", { url: `http://127.0.0.1:${port}/wasm/index.html` });

    // 3. wait for the VM to boot inside the page
    let status = "";
    for (let i = 0; i < 120; i++) {
      status = await evalPage(`document.getElementById("scry-boot-status")?.textContent || ""`);
      if (status.startsWith("live") || status.startsWith("boot failed")) break;
      await sleep(250);
    }
    if (!status.startsWith("live")) fail(`VM never booted in the page (status="${status}") ${pageErrors.join(" | ")}`);
    console.log("  ok  VM booted in-page:", status);

    // 4. the transport really is the wasm module, not a server
    const hasBridge = await evalPage(`!!globalThis.__scryWasm`);
    if (!hasBridge) fail("globalThis.__scryWasm missing — viewer would fall back to fetch");
    const direct = await evalPage(`JSON.stringify(globalThis.__scryWasm.eval("Task.instances().len()"))`);
    if (!/"value"/.test(direct)) fail("direct bridge eval failed: " + direct);
    console.log("  ok  bridge eval Task.instances().len() ->", direct);

    // 5. the React viewer actually rendered the program's types
    let text = "";
    for (let i = 0; i < 60; i++) {
      text = await evalPage(`document.getElementById("app")?.innerText || ""`);
      if (/Task/.test(text) && /Project/.test(text)) break;
      await sleep(250);
    }
    if (!/Task/.test(text)) fail("viewer never rendered type 'Task'. app text: " + JSON.stringify(text.slice(0, 300)));
    if (!/Project/.test(text)) fail("viewer never rendered type 'Project'. app text: " + JSON.stringify(text.slice(0, 300)));
    console.log("  ok  viewer rendered live types (Task, Project) from the in-page VM");

    // 6. a hard eval panic must NOT take the page down (uncrashable eval, in the browser)
    const panic = await evalPage(`JSON.stringify(globalThis.__scryWasm.eval("Task.instances().get(99)"))`);
    if (!/"error"/.test(panic)) fail("expected a typed error from an out-of-bounds eval, got: " + panic);
    const after = await evalPage(`JSON.stringify(globalThis.__scryWasm.eval("1 + 1"))`);
    if (!/"value":2|"value":\{"type":"Int","value":2\}/.test(after)) fail("VM did not survive the panic: " + after);
    console.log("  ok  eval panic returned typed error; VM still live afterwards");

    // ---- phase 2: the agent demo page (xterm terminal + viewer, all in-page) ----
    await send("Page.navigate", { url: `http://127.0.0.1:${port}/wasm/demo.html` });
    let dstatus = "";
    for (let i = 0; i < 120; i++) {
      dstatus = await evalPage(`document.getElementById("status")?.textContent || ""`);
      if (dstatus === "live" || dstatus.startsWith("boot failed")) break;
      await sleep(250);
    }
    if (dstatus !== "live") fail(`agent demo never booted (status="${dstatus}")`);
    console.log("  ok  agent demo booted (terminal + viewer in one page)");

    const termText = () => evalPage(`document.querySelector("#term .xterm-screen")?.innerText || document.getElementById("term")?.innerText || ""`);
    let banner = "";
    for (let i = 0; i < 40 && !/WebAssembly/.test(banner); i++) { banner = await termText(); await sleep(200); }
    if (!/WebAssembly/.test(banner)) fail("terminal never showed the program banner: " + JSON.stringify(banner.slice(0,200)));
    console.log("  ok  xterm shows the program's stdout");

    // type a real agent turn into the terminal, one keystroke at a time
    for (const ch of "weather in Tokyo") await send("Input.dispatchKeyEvent", { type: "char", text: ch });
    await send("Input.dispatchKeyEvent", { type: "rawKeyDown", windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13 });
    await send("Input.dispatchKeyEvent", { type: "char", text: "\r" });
    let out = "";
    for (let i = 0; i < 60 && !/Tokyo:/.test(out); i++) { out = await termText(); await sleep(200); }
    if (!/tool_use|get_weather|Tokyo:/.test(out)) fail("agent turn produced no tool use / reply: " + JSON.stringify(out.slice(-300)));
    console.log("  ok  typed a turn -> agent ran a REAL tool call in the terminal");

    // the viewer must see the live agent state the turn just created
    const msgs = await evalPage(`JSON.stringify(globalThis.__scryWasm.eval("Message.instances().len()"))`);
    if (!/"value"/.test(msgs)) fail("viewer-side eval of Message.instances() failed: " + msgs);
    console.log("  ok  live agent state visible through the viewer transport ->", msgs);

    // ---- phase 3: a BACKGROUND worker advances while the page stays interactive ----
    const msgCount = async () => {
      const r = await evalPage(`JSON.stringify(globalThis.__scryWasm.eval("Message.instances().len()"))`);
      return JSON.parse(r).value?.value ?? -1;
    };
    const before = await msgCount();
    for (const ch of "research wasm") await send("Input.dispatchKeyEvent", { type: "char", text: ch });
    await send("Input.dispatchKeyEvent", { type: "rawKeyDown", windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13 });
    await send("Input.dispatchKeyEvent", { type: "char", text: "\r" });

    // the spawned Researcher has NO OS thread — it only advances because the page pumps
    // scry_tick(). Messages must climb on their own, with no further input.
    let grew = before, seen = [];
    for (let i = 0; i < 60; i++) {
      await sleep(250);
      grew = await msgCount();
      seen.push(grew);
      if (grew >= before + 6) break;   // 5 steps + "done"
    }
    if (grew < before + 6) fail(`background worker never progressed: ${before} -> ${grew} (samples ${seen.slice(-8)})`);
    console.log(`  ok  green thread ran in the background: Messages ${before} -> ${grew} with no further input`);

    const stillLive = await evalPage(`JSON.stringify(globalThis.__scryWasm.eval("1 + 1"))`);
    if (!/"value"/.test(stillLive)) fail("page not interactive after background work: " + stillLive);
    console.log("  ok  page stayed interactive throughout");

    if (pageErrors.length) fail("page threw: " + pageErrors.join(" | "));
    console.log("PASS ui-smoke-wasm");
    cleanup();
    process.exit(0);
  } catch (e) {
    fail(String(e && e.message || e));
  }
}
main();
