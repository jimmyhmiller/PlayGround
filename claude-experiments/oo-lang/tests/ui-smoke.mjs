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

  // 1. boot scry with the viewer server. Feed it two prompts (stdin kept OPEN so the process
  // stays interactive + alive for the viewer) so the agent loop actually runs and the main
  // Agent's Conversation accumulates Messages — the mass the nested view visualizes. Keys are
  // scrubbed by the harness, so this drives the deterministic ScriptedModel (real tool_use turns).
  const scry = spawn(join(ROOT, "scry"), ["run", join(ROOT, "examples", "assistant.scry")],
    { stdio: ["pipe", "pipe", "pipe"] });
  try { scry.stdin.write("what is 17 times 23?\nweather in Tokyo?\n"); } catch {}
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

  const extraProcs = [];
  const cleanup = () => {
    for (const p of extraProcs) { try { p.kill("SIGKILL"); } catch {} }
    try { cproc.kill("SIGKILL"); } catch {} try { scry.kill("SIGKILL"); } catch {}
  };

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

    // (V) the bespoke NESTED-CONTAINMENT view is the default landing mode (Phase V1). Ownership
    // is nesting: a root region (Orchestrator) contains an Agent region that contains a
    // Conversation with a dense message stack (mass). Shared tools render as identity-colored
    // chips that hover-highlight across every appearance. Utility types recede to a faded,
    // collapsible infrastructure strip. Driven by the graph() + schema() evals — no force graph.
    await waitFor("#nested .census");
    ok("nested view is the default landing (census / mass ribbon renders)");
    await waitFor("#nested .orch .region");
    // a helper (in-page) that finds a region whose OWN header names a given type
    const findRegionJS = (kind) => `(()=>{
      for(const r of document.querySelectorAll('#nested .region')){
        const head=r.querySelector('.region-head');
        const k=head&&head.querySelector('.node-kind');
        if(k&&k.textContent.trim()===${JSON.stringify(kind)}) return r;
      }
      return null;
    })()`;
    // an Agent region contains a Conversation which contains message rows
    const nesting = await evalPage(`(()=>{
      const agent=${findRegionJS("Agent")};
      if(!agent) return {noagent:true};
      const mstack=agent.querySelector('.conv .mstack');
      return {rows: mstack?mstack.querySelectorAll('.mrow').length:0, hasConv: !!agent.querySelector('.conv')};
    })()`);
    if (nesting.noagent) fails.push("nested view has no Agent region");
    else {
      if (!nesting.hasConv) fails.push("Agent region does not nest a Conversation"); else ok("Agent region nests a Conversation (ownership = nesting)");
      if (nesting.rows < 1) fails.push("nested Conversation renders no message rows"); else ok(`nested Conversation renders ${nesting.rows} message rows (mass)`);
    }
    // a shared tool chip appears in >=2 owners with the SAME identity color slot
    const chipInfo = await evalPage(`(()=>{
      const chips=[...document.querySelectorAll('#nested .stage-wrap .chip[data-identity]')];
      const byId={};
      for(const c of chips){const id=c.dataset.identity;(byId[id]=byId[id]||[]).push(c.dataset.slot);}
      const shared=Object.entries(byId).filter(([,slots])=>slots.length>=2);
      const sameColor=shared.every(([,slots])=>slots.every(s=>s===slots[0]));
      return {chipCount:chips.length, sharedCount:shared.length, sameColor, firstShared: shared[0]?shared[0][0]:null};
    })()`);
    if (chipInfo.sharedCount < 1) fails.push(`no shared tool chip appears in >=2 owner regions (chips=${chipInfo.chipCount})`);
    else {
      ok(`a shared instance chip appears across ${chipInfo.sharedCount} owner region(s)`);
      if (!chipInfo.sameColor) fails.push("shared chip appearances have inconsistent identity colors"); else ok("shared chip keeps ONE stable identity color everywhere");
    }
    // hover a shared chip -> id-active + EVERY appearance of that exact instance gets id-hi
    if (chipInfo.firstShared) {
      const hi = await evalPage(`(()=>{
        const id=${JSON.stringify(chipInfo.firstShared)};
        const c=document.querySelector('#nested .chip[data-identity="'+CSS.escape(id)+'"]');
        c.dispatchEvent(new MouseEvent('mouseover',{bubbles:true}));  // React onMouseEnter fires on native mouseover
        return new Promise(res=>setTimeout(()=>res({
          active: document.querySelector('#nested').classList.contains('id-active'),
          n: document.querySelectorAll('#nested .chip[data-identity="'+CSS.escape(id)+'"].id-hi').length,
        }),100));
      })()`, true);
      if (!hi.active) fails.push("hovering a chip did not activate identity highlighting (id-active)");
      else if (hi.n < 2) fails.push(`hovering a shared chip highlighted only ${hi.n} appearance(s), expected >=2`);
      else ok(`hovering a shared chip lights up all ${hi.n} appearances (id-active + id-hi)`);
    }
    // the infrastructure strip renders and expands to reveal utility-type rows
    if (!await evalPage(`!!document.querySelector('#nested .infra')`)) fails.push("infrastructure strip did not render");
    else {
      ok("infrastructure strip renders (utility types recede)");
      await evalPage(`document.querySelector('#nested .infra-head').click()`);
      const rows = await evalPage(`document.querySelector('#nested .infra').classList.contains('open') ? document.querySelectorAll('#nested .util').length : 0`);
      if (!rows) fails.push("infrastructure strip did not expand with utility rows"); else ok(`infrastructure strip expands (${rows} utility rows)`);
    }
    // clicking an Agent region drills into its detail (switches to browse) -> sets up the rail beats
    await evalPage(`(()=>{const agent=${findRegionJS("Agent")};if(agent)agent.querySelector('.region-head').click();return !!agent;})()`);
    try { await waitFor(".field-grid", 8000); ok("clicking a region drills into that instance's detail"); }
    catch { fails.push("clicking an Agent region did not open its detail"); }
    // switch to the List rail for the remaining browse beats
    await evalPage(`[...document.querySelectorAll('.vt-btn')].find(b=>b.textContent.trim()==='List').click()`);

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

    // (P) Phase 10 PORTAL dual-mode: boot a portal + a program, load the portal `/`, assert a
    // program CARD renders, click it, and assert the inspector graph loads THROUGH the proxy.
    // Skipped (not failed) if :7357 is already in use by a developer's own portal.
    let portalBusy = false;
    try {
      const probe = await fetch("http://127.0.0.1:7357/api/programs").then(() => true).catch(() => false);
      portalBusy = probe;
    } catch { portalBusy = false; }
    if (portalBusy) {
      console.log("  --  portal scenario skipped (:7357 already in use)");
    } else {
      const portal = spawn(join(ROOT, "scry"), ["portal"], { stdio: ["ignore", "pipe", "pipe"] });
      extraProcs.push(portal);
      let portalUp = false, portalFailed = false;
      portal.stdout.on("data", (b) => {
        const s = b.toString();
        if (s.includes("portal: http://localhost:7357")) portalUp = true;
        if (s.includes("could not bind")) portalFailed = true;
      });
      for (let i = 0; i < 100 && !portalUp && !portalFailed; i++) await sleep(50);
      if (portalFailed || !portalUp) {
        console.log("  --  portal scenario skipped (portal did not start)");
      } else {
        // a program registers with the portal (its own viewer port is ephemeral, >=7400)
        const prog = spawn(join(ROOT, "scry"), ["run", join(ROOT, "examples", "demo-mini.scry")],
          { stdio: ["ignore", "ignore", "ignore"] });
        extraProcs.push(prog);
        let regd = null;
        for (let i = 0; i < 60; i++) {
          await sleep(150);
          try {
            const progs = await fetch("http://127.0.0.1:7357/api/programs").then((r) => r.json());
            regd = progs.find((p) => p.mode === "run" && p.status === "running");
            if (regd) break;
          } catch {}
        }
        if (!regd) fails.push("portal: program never registered / appeared in /api/programs");
        else {
          ok(`portal registry shows program card '${regd.name}'`);
          await send("Page.navigate", { url: "http://127.0.0.1:7357/" });
          try {
            await waitFor(".pcard");
            const cards = await evalPage(`[...document.querySelectorAll('.pcard-name')].map(n=>n.textContent)`);
            ok(`portal landing renders ${cards.length} card(s): ${JSON.stringify(cards)}`);
            // click the first (running) card -> drills into that program's bespoke view, proxied
            await evalPage(`document.querySelector('.pcard:not(.exited)').click()`);
            await waitFor("#nested .cx-name", 12000);
            const pTypes = await evalPage(`[...document.querySelectorAll('#nested .cx-name')].map(n=>n.textContent.trim())`);
            if (!pTypes.includes("Agent")) fails.push(`portal proxied nested view missing Agent in census: ${JSON.stringify(pTypes)}`);
            else ok(`clicking a portal card loads the nested view through the proxy (${pTypes.length} census types)`);
            // the back affordance returns to the landing grid
            await evalPage(`(()=>{const b=document.querySelector('.back-btn');if(b)b.click();return !!b;})()`);
            await waitFor(".pcard", 6000);
            ok("portal back button returns to the landing grid");
          } catch (e) {
            fails.push("portal dual-mode scenario failed: " + (e && e.message || e));
          }
        }
      }
    }

    // (I) INSPECT-MODE regression: `scry inspect` never calls vm-run, so main() never
    // executes — every arena exists but is EMPTY. The graph's own node click is gated on
    // liveCount (GraphPane.clickNode only drills into the live TablePane when `live > 0`;
    // a zero-count node just opens the static NodeCard, which hides the "browse ->" button
    // for the same reason) — so a zero-count node can never reach TablePane THROUGH the
    // graph, by design. The List rail has no such gate: TypeRow always calls
    // `nav.openTable(name)` regardless of live count, and TablePane polls
    // `<Class>.instances()` every 750ms — exactly the opcode path (OP_ARENA_INSTANCES) that
    // used to dereference a null `(vm).program` and crash the WHOLE server in inspect mode
    // (only `vm-run` bound that pointer, and `scry inspect` intentionally never calls it).
    // Assert browsing a zero-count type here renders the empty state, never a crash, and the
    // server is still answering afterward.
    const inspectProc = spawn(join(ROOT, "scry"), ["inspect", join(ROOT, "examples", "assistant.scry")],
      { stdio: ["ignore", "pipe", "pipe"] });
    extraProcs.push(inspectProc);
    let inspectPort = null;
    inspectProc.stdout.on("data", (b) => {
      const m = /localhost:(\d+)/.exec(b.toString());
      if (m && !inspectPort) inspectPort = +m[1];
    });
    for (let i = 0; i < 100 && !inspectPort; i++) await sleep(50);
    if (!inspectPort) {
      fails.push("inspect: `scry inspect` never printed a viewer URL");
    } else {
      try {
        await send("Page.navigate", { url: `http://127.0.0.1:${inspectPort}/` });
        // the bespoke view loads; with main() never run, no instances exist so the stage shows
        // its empty state (nothing to nest) — never a crash.
        await waitFor("#nested", 12000);
        await waitFor("#nested .stage-empty", 8000);
        ok("inspect mode: bespoke view renders its empty stage (no live instances to nest)");
        // switch to the List rail (no liveCount gate) and browse Agent's (empty) table
        await evalPage(`[...document.querySelectorAll('.vt-btn')].find(b=>b.textContent.trim()==='List').click()`);
        await waitFor(".type-row");
        await evalPage(`(()=>{const r=[...document.querySelectorAll('.type-row')].find(e=>{const n=e.querySelector('.tname');return n&&n.textContent.trim()==='Agent';});r.click();return true;})()`);
        await waitFor(".empty, .itable tbody tr", 8000);
        const emptyState = await evalPage(`!!document.querySelector('.empty')`);
        if (!emptyState) fails.push("inspect: browsing the zero-count Agent table did not render the empty state (expected 0 live instances, since main() never ran)");
        else ok("inspect mode: browsing a zero-count type renders the empty state, no crash");
        // the whole point of this beat: the server process must survive that instances() poll
        await sleep(300);
        if (inspectProc.exitCode !== null) {
          fails.push(`inspect: \`scry inspect\` process died (exit ${inspectProc.exitCode}) after browsing Agent`);
        } else {
          const stillUp = await fetch(`http://127.0.0.1:${inspectPort}/`).then(() => true).catch(() => false);
          if (!stillUp) fails.push("inspect: server unreachable after browsing Agent");
          else ok("inspect mode: server still alive and serving after the instances() poll");
        }
      } catch (e) {
        fails.push("inspect-mode browse scenario failed: " + (e && e.message || e));
      }
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
