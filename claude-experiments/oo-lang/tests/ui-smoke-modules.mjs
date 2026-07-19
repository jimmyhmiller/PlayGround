// Real headless-browser smoke test for the Phase 3 MODULE SYSTEM viewer surface — a lean sibling
// to tests/ui-smoke.mjs (07-modules.md §7). Deliberately narrow: this does not re-verify every
// beat ui-smoke.mjs already covers (rail/table/detail/method-invoke click-through); it asserts
// only the module-specific behavior that phase added:
//   1. the type rail groups types by module (nesting interface groups inside), with modules()
//      counts on each group header;
//   2. clicking a module group header FOCUSES the viewer — the URL becomes addressable
//      (#m=<dotted.module>) and the Map's census/instance tree scope to that subtree;
//   3. clearing focus restores the whole-program view.
// Boots `scry run examples/assistant.scry` (a real multi-module program: assistant + agent.core +
// std.json — see docs/07-modules.md's viewer bullets). Exit 0 = pass, nonzero = fail. Prints
// SKIPPED and exits 0 if no Chrome is found (matches ui-smoke.mjs's gating).
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

function findChrome() {
  const cands = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
  ];
  for (const c of cands) { if (spawnSync("test", ["-x", c]).status === 0) return c; }
  return null;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const chrome = findChrome();
  if (!chrome) { console.log("SKIPPED ui-smoke-modules: no Chrome/Chromium binary found"); process.exit(0); }

  const scry = spawn(join(ROOT, "scry"), ["run", join(ROOT, "examples", "assistant.scry")],
    { stdio: ["pipe", "pipe", "pipe"] });
  try { scry.stdin.write("hello\n"); } catch {}
  let port = null;
  const waitPort = new Promise((res) => {
    scry.stdout.on("data", (b) => {
      const m = /localhost:(\d+)/.exec(b.toString());
      if (m && !port) { port = +m[1]; res(); }
    });
  });
  await Promise.race([waitPort, sleep(15000)]);
  if (!port) { console.error("FAIL ui-smoke-modules: scry never printed viewer URL"); scry.kill(); process.exit(1); }
  const viewerURL = `http://127.0.0.1:${port}/`;

  const profile = mkdtempSync(join(tmpdir(), "scry-ui-mod-"));
  const dport = 9722 + Math.floor(Math.random() * 500);
  const cproc = spawn(chrome, [
    "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    `--remote-debugging-port=${dport}`, `--user-data-dir=${profile}`, "about:blank",
  ], { stdio: ["ignore", "pipe", "pipe"] });

  const cleanup = () => { try { cproc.kill("SIGKILL"); } catch {} try { scry.kill("SIGKILL"); } catch {} };

  try {
    let wsURL = null;
    for (let i = 0; i < 100; i++) {
      try {
        const r = await fetch(`http://127.0.0.1:${dport}/json/list`);
        const targets = await r.json();
        const page = targets.find((t) => t.type === "page");
        if (page && page.webSocketDebuggerUrl) { wsURL = page.webSocketDebuggerUrl; break; }
      } catch {}
      await sleep(150);
    }
    if (!wsURL) throw new Error("Chrome DevTools endpoint never came up");

    const ws = new WebSocket(wsURL);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error("ws error")); });
    let msgId = 0;
    const pending = new Map();
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && pending.has(msg.id)) {
        const { res, rej } = pending.get(msg.id); pending.delete(msg.id);
        if (msg.error) rej(new Error(JSON.stringify(msg.error))); else res(msg.result);
      }
    };
    const send = (method, params = {}) =>
      new Promise((res, rej) => { const id = ++msgId; pending.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params })); });
    const evalPage = async (expr, awaitPromise = false) => {
      const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise, allowUnsafeEvalBlocklistBypass: true });
      if (r.exceptionDetails) throw new Error("page exception: " + JSON.stringify(r.exceptionDetails.exception || r.exceptionDetails));
      return r.result.value;
    };
    const waitFor = (sel, timeout = 12000) => evalPage(
      `new Promise((res,rej)=>{const t0=Date.now();(function l(){const n=document.querySelectorAll(${JSON.stringify(sel)}).length;if(n)return res(n);if(Date.now()-t0>${timeout})return rej('timeout: '+${JSON.stringify(sel)});setTimeout(l,100);})();})`,
      true);

    await send("Page.enable");
    await send("Runtime.enable");
    await send("Page.navigate", { url: viewerURL });

    const fails = [];
    const ok = (label) => console.log("  ok  " + label);

    // (1) type rail groups by module, nesting the existing interface groups inside.
    await waitFor(".vt-btn");
    await evalPage(`[...document.querySelectorAll('.vt-btn')].find(b=>b.textContent.trim()==='List').click()`);
    await waitFor(".mod-group", 12000);
    const groups = await evalPage(`[...document.querySelectorAll('.type-row.mod .mname')].map(e=>e.getAttribute('title'))`);
    const hasAssistant = groups.includes("focus assistant");
    const hasAgentCore = groups.includes("focus agent.core");
    if (!hasAssistant || !hasAgentCore) fails.push(`type rail did not group by module as expected (got ${JSON.stringify(groups)})`);
    else ok(`type rail groups types by module (${JSON.stringify(groups)})`);
    // an interface group (Tool) nests INSIDE the agent.core module group, not at the top level.
    const nestedIface = await evalPage(`(()=>{
      const core=[...document.querySelectorAll('.mod-group')].find(g=>{
        const h=g.querySelector(':scope > .type-row.mod .mname'); return h && h.getAttribute('title')==='focus agent.core'; });
      if(!core) return {noCore:true};
      const iface=[...core.querySelectorAll('.iface-group .tname')].some(e=>/Tool/.test(e.textContent));
      return {iface};
    })()`);
    if (nestedIface.noCore || !nestedIface.iface) fails.push(`the 'Tool' interface group did not nest inside the agent.core module group (${JSON.stringify(nestedIface)})`);
    else ok("interface groups nest INSIDE their module group (Tool under agent.core)");

    // (2) clicking a module group's header FOCUSES the viewer: URL becomes #m=<path>, and the
    // Map's census scopes to that subtree only (module tags outside it, e.g. "assistant", vanish).
    await evalPage(`document.querySelector('.type-row.mod .mname[title="focus agent.core"]').click()`);
    await sleep(400);
    const hash = await evalPage(`location.hash`);
    if (hash !== "#m=agent.core") fails.push(`focusing agent.core did not set the URL hash (got ${JSON.stringify(hash)})`);
    else ok(`focus is URL-addressable (location.hash === "${hash}")`);

    await evalPage(`[...document.querySelectorAll('.vt-btn')].find(b=>b.textContent.trim()==='Map').click()`);
    await waitFor(".focus-badge", 8000);
    await waitFor("#nested .census .cx-mod", 8000);
    const scoped = await evalPage(`(()=>{
      const mods=[...document.querySelectorAll('#nested .census .cx-mod')].map(e=>e.textContent.trim());
      const everywhere=!!document.querySelector('.everywhere-toggle');
      return {mods, everywhere, unique:[...new Set(mods)]};
    })()`);
    if (scoped.unique.length !== 1 || scoped.unique[0] !== "agent.core")
      fails.push(`focused census is not scoped to agent.core alone (module tags: ${JSON.stringify(scoped.unique)})`);
    else ok(`focusing agent.core scopes the census to that module alone (${scoped.mods.length} rows, all "agent.core")`);
    if (!scoped.everywhere) fails.push("the 'everywhere' scope-override toggle did not appear while focused");
    else ok("the 'everywhere' toggle appears while a focus is active");

    // (3) clearing focus (the focus-badge's close button) restores the whole-program view.
    await evalPage(`document.querySelector('.focus-badge .ghost-btn').click()`);
    await sleep(400);
    const afterClear = await evalPage(`(()=>{
      const mods=new Set([...document.querySelectorAll('#nested .census .cx-mod')].map(e=>e.textContent.trim()));
      return {hash: location.hash, hasBadge: !!document.querySelector('.focus-badge'), moduleCount: mods.size};
    })()`);
    if (afterClear.hasBadge || afterClear.hash) fails.push(`clearing focus left residue (hash=${JSON.stringify(afterClear.hash)}, badge=${afterClear.hasBadge})`);
    else if (afterClear.moduleCount < 2) fails.push(`clearing focus did not restore the whole-program census (only ${afterClear.moduleCount} distinct module(s))`);
    else ok(`clearing focus restores the whole-program view (${afterClear.moduleCount} modules visible again, no hash)`);

    if (fails.length) {
      console.error("FAIL ui-smoke-modules");
      for (const f of fails) console.error("     " + f);
      cleanup(); process.exit(1);
    }
    console.log("PASS ui-smoke-modules (module rail grouping + focus mode verified)");
    cleanup(); process.exit(0);
  } catch (e) {
    console.error("FAIL ui-smoke-modules: " + (e && e.stack || e));
    cleanup(); process.exit(1);
  }
}
main();
