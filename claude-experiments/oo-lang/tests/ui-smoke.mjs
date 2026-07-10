// Real headless-browser smoke test for the React viewer — zero npm deps.
// Drives system Chrome over the DevTools Protocol using Node's built-in global WebSocket
// (Node 18+/24). Boots `scry run examples/assistant.scry`, opens the viewer headless, and
// asserts the whole click-through works AND that interactive state survives the poll refresh
// (the entire point of the React rewrite).
//
// Exit 0 = pass, nonzero = fail. Prints SKIPPED and exits 0 if no Chrome is found.
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
  if (!chrome) { console.log("SKIPPED ui-smoke: no Chrome/Chromium binary found"); process.exit(0); }

  // 1. boot scry with the viewer server
  const scry = spawn(join(ROOT, "scry"), ["run", join(ROOT, "examples", "assistant.scry")],
    { stdio: ["ignore", "pipe", "pipe"] });
  let port = null;
  const waitPort = new Promise((res) => {
    scry.stdout.on("data", (b) => {
      const m = /localhost:(\d+)/.exec(b.toString());
      if (m && !port) { port = +m[1]; res(); }
    });
  });
  await Promise.race([waitPort, sleep(15000)]);
  if (!port) { console.error("FAIL ui-smoke: scry never printed viewer URL"); scry.kill(); process.exit(1); }
  const viewerURL = `http://127.0.0.1:${port}/`;

  // 2. launch headless Chrome with a debugging port
  const profile = mkdtempSync(join(tmpdir(), "scry-ui-"));
  const dport = 9222 + Math.floor(Math.random() * 500);
  const cproc = spawn(chrome, [
    "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    `--remote-debugging-port=${dport}`, `--user-data-dir=${profile}`, "about:blank",
  ], { stdio: ["ignore", "pipe", "pipe"] });

  const cleanup = () => { try { cproc.kill("SIGKILL"); } catch {} try { scry.kill("SIGKILL"); } catch {} };

  try {
    // 3. find the page target's WebSocket URL
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

    // 4. minimal CDP client over the page-level WebSocket
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

    // evaluate JS in the page; supports async (awaitPromise) and returns the value
    const evalPage = async (expr, awaitPromise = false) => {
      const r = await send("Runtime.evaluate", {
        expression: expr, returnByValue: true, awaitPromise, allowUnsafeEvalBlocklistBypass: true,
      });
      if (r.exceptionDetails) throw new Error("page exception: " + JSON.stringify(r.exceptionDetails.exception || r.exceptionDetails));
      return r.result.value;
    };
    // wait until selector appears (returns count matched)
    const waitFor = (sel, timeout = 10000) => evalPage(
      `new Promise((res,rej)=>{const t0=Date.now();(function l(){const n=document.querySelectorAll(${JSON.stringify(sel)}).length;if(n)return res(n);if(Date.now()-t0>${timeout})return rej('timeout: '+${JSON.stringify(sel)});setTimeout(l,100);})();})`,
      true);

    await send("Page.enable");
    await send("Runtime.enable");
    await send("Page.navigate", { url: viewerURL });

    const fails = [];
    const ok = (label) => console.log("  ok  " + label);

    // (g0) the class-relationship GRAPH is the landing view — nodes for classes, an interface
    // and an enum, plus a resolved Agent->Conversation field edge; clicking a live class node
    // navigates to its instance table (the static/live unification).
    await waitFor("#graph .gnode");
    const gnodes = await evalPage(`[...document.querySelectorAll('.gnode')].map(n=>n.dataset.name)`);
    for (const want of ["Agent", "Conversation"]) {
      if (!gnodes.includes(want)) fails.push(`graph missing node ${want}: ${JSON.stringify(gnodes)}`);
    }
    if (fails.length === 0) ok(`graph renders ${gnodes.length} nodes (Agent, Conversation present)`);
    const kinds = await evalPage(`(()=>{const s={};for(const n of document.querySelectorAll('.gnode'))s[n.dataset.kind]=(s[n.dataset.kind]||0)+1;return s;})()`);
    if (!kinds.interface) fails.push("graph has no interface node (e.g. Tool)"); else ok(`graph has ${kinds.interface} interface node(s)`);
    if (!kinds.enum) fails.push("graph has no enum node (e.g. AgentStatus)"); else ok(`graph has ${kinds.enum} enum node(s)`);
    const hasEdge = await evalPage(
      `[...document.querySelectorAll('.gedge')].some(l=>l.dataset.from==='Agent'&&l.dataset.to==='Conversation')`);
    if (!hasEdge) fails.push("graph missing Agent->Conversation field edge"); else ok("graph has Agent->Conversation edge");
    // click the Agent node (live count >= 1 while running) -> drills to its instance table
    await evalPage(`document.querySelector('.gnode[data-name="Agent"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))`);
    try { await waitFor(".itable tbody tr", 8000); ok("clicking a live graph node navigates to its table"); }
    catch { fails.push("clicking Agent graph node did not open its instance table"); }

    // (a) the List view's type rail renders types, incl. Agent (we are now in browse mode)
    await waitFor(".type-row");
    const hasAgent = await evalPage(
      `[...document.querySelectorAll('.type-row .tname')].some(e=>e.textContent.trim()==='Agent')`);
    if (!hasAgent) fails.push("type rail missing Agent"); else ok("type rail renders types (Agent present)");

    // (b) click Agent -> table renders rows
    await evalPage(
      `(()=>{const r=[...document.querySelectorAll('.type-row')].find(e=>{const n=e.querySelector('.tname');return n&&n.textContent.trim()==='Agent';});r.click();return true;})()`);
    const rows = await waitFor(".itable tbody tr");
    if (rows < 1) fails.push("table rendered no rows"); else ok(`table renders ${rows} rows`);

    // (c) click a row -> detail renders fields
    await evalPage(`document.querySelector('.itable tbody tr').click()`);
    await waitFor(".field-grid");
    const fieldCount = await evalPage(`document.querySelectorAll('.field-grid .fname').length`);
    if (fieldCount < 1) fails.push("detail rendered no fields"); else ok(`detail renders ${fieldCount} fields`);

    // (d) open a method card + type into an arg input, then survive two polls
    await waitFor(".method");
    // open the first method card that HAS a param input, and type into it
    const typed = await evalPage(`(()=>{
      const cards=[...document.querySelectorAll('.method')];
      const card=cards.find(c=>c.querySelector('.method-body input'));
      if(!card) return null;
      card.querySelector('.method-head').click();          // open
      const inp=card.querySelector('.method-body input');
      const setter=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
      setter.call(inp,'SMOKE_TEST_TEXT');
      inp.dispatchEvent(new Event('input',{bubbles:true}));
      card.dataset.smoke='1';
      return card.className;
    })()`);
    if (typed === null) { fails.push("no method card with an input param to test"); }
    else {
      if (!/\bopen\b/.test(typed)) fails.push("method card did not open");
      await sleep(2100); // two+ detail polls (750ms) go by
      const state = await evalPage(`(()=>{
        const card=document.querySelector('.method[data-smoke="1"]');
        if(!card) return {gone:true};
        const inp=card.querySelector('.method-body input');
        return {open:/\\bopen\\b/.test(card.className), val: inp?inp.value:null};
      })()`);
      if (state.gone) fails.push("method card disappeared across polls");
      else {
        if (!state.open) fails.push("method card CLOSED across polls (jank!)");
        else ok("method card STILL open after 2 polls");
        if (state.val !== "SMOKE_TEST_TEXT") fails.push(`arg input text lost across polls: got ${JSON.stringify(state.val)}`);
        else ok("arg input text SURVIVED 2 polls");
      }
    }

    // (e) invoke a 0-arg method -> a result appears
    const invoked = await evalPage(`(()=>{
      const cards=[...document.querySelectorAll('.method')];
      const card=cards.find(c=>!c.querySelector('.method-body input')); // 0-arg
      if(!card) return false;
      card.querySelector('.invoke-btn').click();
      return true;
    })()`);
    if (!invoked) { console.log("  --  no 0-arg method available to invoke (skipped that assertion)"); }
    else {
      try { await waitFor(".invoke-result", 8000); ok("0-arg invoke produced a result"); }
      catch { fails.push("0-arg invoke produced no result"); }
    }

    if (fails.length) {
      console.error("FAIL ui-smoke");
      for (const f of fails) console.error("     " + f);
      cleanup(); process.exit(1);
    }
    console.log("PASS ui-smoke (React viewer: click-through + poll-survival verified)");
    cleanup(); process.exit(0);
  } catch (e) {
    console.error("FAIL ui-smoke: " + (e && e.stack || e));
    cleanup(); process.exit(1);
  }
}
main();
