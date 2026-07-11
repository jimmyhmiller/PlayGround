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
    // the infrastructure strip renders and expands to reveal utility-type rows. The infra types
    // (ModelResponse/ToolCall/…) only appear once the agent loop has run a tool turn, so wait for
    // the strip rather than sampling a single poll (deterministic, not timing-flaky).
    try { await waitFor("#nested .infra", 15000); } catch (e) {}
    if (!await evalPage(`!!document.querySelector('#nested .infra')`)) fails.push("infrastructure strip did not render");
    else {
      ok("infrastructure strip renders (utility types recede)");
      await evalPage(`document.querySelector('#nested .infra-head').click()`);
      const rows = await evalPage(`document.querySelector('#nested .infra').classList.contains('open') ? document.querySelectorAll('#nested .util').length : 0`);
      if (!rows) fails.push("infrastructure strip did not expand with utility rows"); else ok(`infrastructure strip expands (${rows} utility rows)`);
    }
    // (V2) program-declared view: an Agent declares `view AgentBoard for Agent` -> a
    // ▤ cell / ▧ board toggle. Toggling renders the bespoke board (timeline rows + tool chips);
    // toggling back returns to the default nested cell. This is DECISIONS #15b landing in place.
    const hasToggle = await evalPage(`(()=>{const a=${findRegionJS("Agent")};return !!(a&&a.querySelector('.vtoggle'));})()`);
    if (!hasToggle) fails.push("Agent region has no declared-view toggle (AgentBoard not surfaced)");
    else {
      ok("Agent region shows a default<->custom view toggle (AgentBoard declared)");
      await evalPage(`(()=>{const a=${findRegionJS("Agent")};const b=[...a.querySelectorAll('.vtoggle button')].find(x=>/board/.test(x.textContent));b.click();return true;})()`);
      try { await waitFor('#nested .board[data-view="AgentBoard"] .timeline', 8000); } catch (e) { fails.push("toggling to board never rendered an AgentBoard timeline"); }
      const board = await evalPage(`(()=>{
        const b=document.querySelector('#nested .board[data-view="AgentBoard"]');
        if(!b) return {noboard:true};
        return {rows:b.querySelectorAll('.timeline .tl').length, chips:b.querySelectorAll('.vchips .chip').length, hasTitle:!!b.querySelector('.board-title')};
      })()`);
      if (board.noboard) fails.push("toggling to board did not render an AgentBoard");
      else {
        if (!board.hasTitle) fails.push("AgentBoard has no title header");
        if (board.rows < 1) fails.push("AgentBoard timeline rendered no message rows"); else ok(`toggling to board renders the AgentBoard timeline (${board.rows} rows)`);
        if (board.chips < 1) fails.push("AgentBoard tools section rendered no chips"); else ok(`AgentBoard tools section renders ${board.chips} identity chips`);
      }
      // toggle back to the default nested cell
      await evalPage(`(()=>{const b=document.querySelector('#nested .board[data-view="AgentBoard"] .vtoggle button');b.click();return true;})()`);
      try { await waitFor("#nested .region .conv .mstack", 8000); ok("toggling back returns to the default nested cell (message stack)"); }
      catch (e) { fails.push("toggling back did not restore the default nested cell"); }
    }
    // (V4) IN-MAP INSPECTOR: clicking an Agent region opens a docked inspector WITHOUT leaving the
    // Map. The map stays visible; the inspector shows the SAME full detail (fields + methods) the
    // browse DetailPane does; ref fields navigate WITHIN the inspector (a growing breadcrumb);
    // message rows are individually drillable; a 0-arg invoke works; closing restores full width.
    await evalPage(`(()=>{const agent=${findRegionJS("Agent")};if(agent)agent.querySelector('.region-head').click();return !!agent;})()`);
    try { await waitFor("#inspector .field-grid", 8000); }
    catch { fails.push("V4: clicking an Agent region did not open the in-map inspector"); }
    const inMap = await evalPage(`!!document.querySelector('#nested') && !!document.querySelector('.nested-layout.has-inspector #inspector')`);
    if (!inMap) fails.push("V4: inspector did not open IN the map (mode left map / no docked inspector)");
    else ok("V4: clicking an Agent region opens the docked inspector, staying in the Map");
    const inspFields = await evalPage(`document.querySelectorAll('#inspector .field-grid .fname').length`);
    const inspMethods = await evalPage(`document.querySelectorAll('#inspector .method').length`);
    if (inspFields < 1) fails.push("V4: inspector shows no fields"); else ok(`V4: inspector shows the Agent's ${inspFields} fields`);
    if (inspMethods < 1) fails.push("V4: inspector shows no methods"); else ok(`V4: inspector shows the Agent's ${inspMethods} methods`);

    // (V6) DECLARED ACTIONS: the program declares `action "Pause"/"Resume"/"Ask" for Agent`, which
    // the inspector surfaces as prominent BUTTONS in an ACTIONS section ABOVE the raw method list.
    // Assert the buttons render (Pause/Resume/Ask), a param action (Ask) exposes an inline arg form,
    // and clicking a 0-arg mutating action (Pause) flips the Agent's `paused` field LIVE in-inspector.
    try { await waitFor("#inspector .actions-section .action-card", 8000); } catch { fails.push("V6: no ACTIONS section rendered in the inspector"); }
    const actLabels = await evalPage(`[...document.querySelectorAll('#inspector .actions-section .action-btn .action-label')].map(e=>e.textContent.trim())`);
    for (const need of ["Pause", "Resume", "Ask"]) {
      if (!actLabels.includes(need)) fails.push(`V6: inspector ACTIONS missing '${need}' button (got ${JSON.stringify(actLabels)})`);
    }
    if (["Pause", "Resume", "Ask"].every((n) => actLabels.includes(n))) ok(`V6: inspector renders declared action buttons ${JSON.stringify(actLabels)}`);
    // the ACTIONS section is ABOVE the methods section (curated affordances are primary)
    const orderOk = await evalPage(`(()=>{
      const secs=[...document.querySelectorAll('#inspector .detail-section h3')].map(h=>h.textContent.trim());
      return secs.indexOf('actions') >= 0 && secs.indexOf('actions') < secs.indexOf('methods');
    })()`);
    if (!orderOk) fails.push("V6: ACTIONS section is not placed above the methods section"); else ok("V6: ACTIONS section sits above the method list (visually primary)");
    // Ask has a param -> clicking it opens an inline arg form with an input (React state is async,
    // so click first, then poll the card's class until it flips to open).
    const askClicked = await evalPage(`(()=>{
      const card=[...document.querySelectorAll('#inspector .action-card')].find(c=>{
        const l=c.querySelector('.action-label'); return l&&l.textContent.trim()==='Ask'; });
      if(!card) return false; card.dataset.smokeAsk='1'; card.querySelector('.action-btn').click(); return true;
    })()`);
    if (!askClicked) fails.push("V6: no 'Ask' action card found");
    else {
      let askForm = null;
      for (let i = 0; i < 20; i++) {
        await sleep(100);
        askForm = await evalPage(`(()=>{const card=document.querySelector('#inspector .action-card[data-smoke-ask="1"]');
          if(!card) return {gone:true};
          const input=card.querySelector('.action-form input');
          const shown=input && getComputedStyle(card.querySelector('.action-form')).display!=='none';
          return {open:/\\bopen\\b/.test(card.className), input:!!input, shown:!!shown};})()`);
        if (askForm.open && askForm.shown) break;
      }
      if (!askForm || askForm.gone) fails.push("V6: 'Ask' action card disappeared");
      else if (!askForm.open || !askForm.input || !askForm.shown)
        fails.push(`V6: 'Ask' (param action) did not reveal an inline arg form (open=${askForm.open}, input=${askForm.input}, shown=${askForm.shown})`);
      else ok("V6: a param action (Ask) reveals an inline arg-input form");
    }
    // read the paused field, click the 0-arg Pause action, assert paused flips LIVE in the inspector
    const readPaused = `(()=>{
      const cells=[...document.querySelectorAll('#inspector .field-grid .fcell')];
      for(let i=0;i<cells.length;i++){
        if(cells[i].classList.contains('fname') && cells[i].textContent.trim().startsWith('paused')){
          const v=cells[i+1]; return v?v.textContent.trim():null;
        }
      }
      return null;
    })()`;
    const pausedBefore = await evalPage(readPaused);
    const clickedPause = await evalPage(`(()=>{
      const card=[...document.querySelectorAll('#inspector .action-card')].find(c=>{
        const l=c.querySelector('.action-label'); return l&&l.textContent.trim()==='Pause'; });
      if(!card) return false; card.querySelector('.action-btn').click(); return true;
    })()`);
    if (!clickedPause) fails.push("V6: no 'Pause' action button to click");
    else {
      // wait for the invoke result + the detailBus read-back poll to flip the field
      let pausedAfter = null;
      for (let i = 0; i < 40; i++) { await sleep(150); pausedAfter = await evalPage(readPaused); if (pausedAfter && /true/i.test(pausedAfter)) break; }
      if (!(pausedAfter && /true/i.test(pausedAfter)))
        fails.push(`V6: clicking Pause did not flip the Agent's paused field live (before=${JSON.stringify(pausedBefore)}, after=${JSON.stringify(pausedAfter)})`);
      else ok(`V6: clicking the Pause action mutated the instance LIVE in the inspector (paused ${JSON.stringify(pausedBefore)} -> ${JSON.stringify(pausedAfter)})`);
    }
    // the inspected instance is ringed in the map
    if (!await evalPage(`!!document.querySelector('#nested .sel-inspect')`)) fails.push("V4: inspected instance not highlighted in the map");
    else ok("V4: inspected instance is highlighted in the map");
    // invoke a 0-arg method FROM the inspector -> a result appears
    const inspInvoked = await evalPage(`(()=>{
      const card=[...document.querySelectorAll('#inspector .method')].find(c=>!c.querySelector('.method-body input'));
      if(!card) return false; card.querySelector('.invoke-btn').click(); return true;
    })()`);
    if (!inspInvoked) console.log("  --  no 0-arg method in inspector to invoke (skipped)");
    else { try { await waitFor("#inspector .invoke-result", 8000); ok("V4: invoking a 0-arg method from the inspector produced a result"); }
           catch { fails.push("V4: inspector 0-arg invoke produced no result"); } }
    // click a reference field in the inspector -> navigate to that instance IN the inspector (breadcrumb grows)
    const refNav = await evalPage(`(()=>{
      const before=document.querySelectorAll('#inspector .insp-crumbs .icrumb').length;
      const link=document.querySelector('#inspector .field-grid .fval .reflink');
      if(!link) return {nolink:true, before};
      link.click(); return {before};
    })()`);
    if (refNav.nolink) console.log("  --  no reference field in Agent detail to navigate (skipped)");
    else {
      await sleep(400);
      const after = await evalPage(`document.querySelectorAll('#inspector .insp-crumbs .icrumb').length`);
      if (after <= refNav.before) fails.push(`V4: clicking a ref field did not grow the inspector breadcrumb (${refNav.before}->${after})`);
      else ok(`V4: a reference field navigates inside the inspector (breadcrumb ${refNav.before}->${after})`);
    }
    // click a Message row in a Conversation stack -> that specific Message opens in the inspector
    const msgDrill = await evalPage(`(()=>{
      const row=document.querySelector('#nested .conv .mstack .mrow.drill');
      if(!row) return {norow:true}; row.click(); return {ok:true};
    })()`);
    if (msgDrill.norow) fails.push("V4: no drillable message row found in a Conversation stack");
    else {
      await sleep(500);
      const cur = await evalPage(`(()=>{const c=document.querySelector('#inspector .insp-crumbs .icrumb.cur');return c?c.textContent.trim():null;})()`);
      const mf = await evalPage(`document.querySelectorAll('#inspector .field-grid .fname').length`);
      if (!cur || mf < 1) fails.push("V4: clicking a message row did not open that Message's detail in the inspector");
      else ok(`V4: clicking a message row opens that Message in the inspector (${cur}, ${mf} fields)`);
    }
    // close the inspector -> the map returns to full width
    await evalPage(`document.querySelector('#inspector .insp-btn[title="close inspector"]').click()`);
    await sleep(300);
    if (await evalPage(`!!document.querySelector('#inspector') || !!document.querySelector('.nested-layout.has-inspector')`))
      fails.push("V4: closing the inspector did not restore the full-width map");
    else ok("V4: closing the inspector restores the full-width map");

    // (F1/F2) Functions are FIRST-CLASS MAP CITIZENS (no separate mode). The Map (NestedView) shows
    // a `functions` section listing top-level functions with signatures. Clicking one opens a TRACE
    // view in the SAME in-map inspector, pre-filled from the signature; tracing renders the call tree
    // + stats strip. Runs against assistant.scry's `contains(s, needle)`.
    await waitFor("#nested .fnsec .fn-item", 10000);
    const hasContainsFn = await evalPage(`[...document.querySelectorAll('#nested .fnsec .fn-item .fn-name')].some(e=>e.textContent.trim()==='contains')`);
    if (!hasContainsFn) fails.push("Functions section: 'contains' top-level function not listed in the Map");
    else ok("Functions section: the Map lists top-level functions with signatures");
    await evalPage(`(()=>{const it=[...document.querySelectorAll('#nested .fnsec .fn-item')].find(c=>{const n=c.querySelector('.fn-name');return n&&n.textContent.trim()==='contains';});if(it)it.click();return !!it;})()`);
    try {
      await waitFor("#inspector .trace-panel.fn-trace", 8000);
      const prefill = await evalPage(`(()=>{const i=document.querySelector('#inspector .trace-input');return i?i.value:null;})()`);
      if (!prefill || !/^contains\(/.test(prefill)) fails.push(`Functions: clicking a function did not pre-fill the trace input (got ${JSON.stringify(prefill)})`);
      else ok(`Functions: clicking a function opens the in-map trace inspector, pre-filled (${JSON.stringify(prefill)})`);
      // type real args + trace -> a call tree renders IN the inspector
      await evalPage(`(()=>{
        const inp=document.querySelector('#inspector .trace-input');
        const setter=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
        setter.call(inp,'contains("hello world", "wor")');
        inp.dispatchEvent(new Event('input',{bubbles:true}));
      })()`);
      await sleep(250);  // let React flush the input state before the run button reads it
      await evalPage(`document.querySelector('#inspector .trace-run').click()`);
      const tn = await waitFor("#inspector .tnode", 8000);
      const strip = await evalPage(`(()=>{const s=document.querySelector('#inspector .ts-strip');return s?s.textContent:'';})()`);
      const hasFn = await evalPage(`[...document.querySelectorAll('#inspector .tnode-fn')].some(e=>e.textContent.trim()==='contains')`);
      if (tn >= 1 && /calls/.test(strip) && hasFn) ok(`Functions: tracing a function from the Map renders a call tree (${tn} node(s)) + stats strip`);
      else fails.push(`Functions: incomplete trace render (nodes=${tn}, strip=${JSON.stringify(strip)}, hasFn=${hasFn})`);
      await evalPage(`(()=>{const b=document.querySelector('#inspector .insp-btn[title="close inspector"]');if(b)b.click();return true;})()`);
    } catch (e) { fails.push("Functions: no trace inspector / call tree after clicking a function (" + e.message + ")"); }

    // switch to the List rail for the remaining browse beats (List is unchanged by V4)
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

    // (T) TYPED ARG ENTRY (ArgInput): the proper fix for the bare-word "unknown identifier" bug.
    // A non-primitive param used to render a plain <input>; typing `test` for a User/Account/enum
    // param compiled to an unbound variable. Now each param renders a TYPE-AWARE widget: an enum
    // param is a <select> of variants (emits `Type.Variant`), an entity param is a <select> of
    // live instances (emits `Type#slot` -> Type.at(slot,0)), numbers validate + block invoke.
    // Boot kanban ("Set status" enum param + "Reassign" entity param) and bank ("Transfer" entity
    // + Int params): assert the picker renders (not a bare input), select an option, invoke, and
    // assert a real, non-"unknown identifier" result.
    const bootProgram = async (example) => {
      const p = spawn(join(ROOT, "scry"), ["run", join(ROOT, "examples", example)], { stdio: ["pipe", "pipe", "pipe"] });
      extraProcs.push(p);
      let pport = null;
      p.stdout.on("data", (b) => { const m = /localhost:(\d+)/.exec(b.toString()); if (m && !pport) pport = +m[1]; });
      for (let i = 0; i < 160 && !pport; i++) await sleep(50);
      return pport;
    };
    // browse to the first instance of a type via the List rail -> its DetailPane (fields+actions)
    const openFirstInstance = async (typeName) => {
      await waitFor(".vt-btn", 12000);   // wait for the freshly-navigated app to boot
      await evalPage(`[...document.querySelectorAll('.vt-btn')].find(b=>b.textContent.trim()==='List').click()`);
      await waitFor(".type-row", 12000);
      await evalPage(`(()=>{const r=[...document.querySelectorAll('.type-row')].find(e=>{const n=e.querySelector('.tname');return n&&n.textContent.trim()===${JSON.stringify(typeName)};});if(r)r.click();return !!r;})()`);
      await waitFor(".itable tbody tr", 12000);
      await evalPage(`document.querySelector('.itable tbody tr').click()`);
      await waitFor(".field-grid", 12000);
    };
    // open an action card by label (marks it for later lookup); returns whether it was found
    const openAction = (label) => `(()=>{
      const card=[...document.querySelectorAll('.actions-section .action-card')].find(c=>{const l=c.querySelector('.action-label');return l&&l.textContent.trim()===${JSON.stringify(label)};});
      if(!card) return false; card.dataset.smokeAct=${JSON.stringify(label)}; card.querySelector('.action-btn').click(); return true;
    })()`;
    const actSel = (label) => `document.querySelector('.action-card[data-smoke-act=${JSON.stringify(label)}]')`;
    // fire a React-controlled <select>/<input> change
    const setControl = (elExpr, val, proto) => `(()=>{
      const el=${elExpr}; if(!el) return false;
      const setter=Object.getOwnPropertyDescriptor(window.${proto}.prototype,'value').set;
      setter.call(el, ${JSON.stringify(val)});
      el.dispatchEvent(new Event(${proto === "HTMLSelectElement" ? "'change'" : "'input'"},{bubbles:true}));
      return el.value;
    })()`;

    const kport = await bootProgram("kanban.scry");
    if (!kport) fails.push("T: kanban never printed a viewer URL");
    else {
      await send("Page.navigate", { url: `http://127.0.0.1:${kport}/` });
      try {
        await openFirstInstance("Card");
        // --- ENUM param: "Set status" (s: Status) is a <select> of variants, never a bare input
        if (!await evalPage(openAction("Set status"))) fails.push("T: kanban Card has no 'Set status' action");
        else {
          let form = null;
          for (let i = 0; i < 30; i++) { await sleep(100);
            form = await evalPage(`(()=>{const c=${actSel("Set status")};if(!c)return{gone:true};
              const f=c.querySelector('.action-form'); const sel=f&&f.querySelector('select.arg-select');
              return{open:/\\bopen\\b/.test(c.className), hasSelect:!!sel, hasBareInput:!!(f&&f.querySelector('input')),
                     opts: sel?[...sel.options].map(o=>o.value):[]};})()`);
            if (form.open && form.hasSelect) break;
          }
          if (!form || form.gone) fails.push("T: 'Set status' action card disappeared");
          else if (!form.hasSelect) fails.push("T: enum param 'Set status' did not render a <select> picker (got a bare input)");
          else {
            ok(`T: enum param renders a <select> variant picker (options: ${JSON.stringify(form.opts)})`);
            if (form.hasBareInput) fails.push("T: enum param 'Set status' still shows a bare text input alongside the select");
            if (!form.opts.includes("Status.Done")) fails.push(`T: enum picker options are not fully-qualified Type.Variant (${JSON.stringify(form.opts)})`);
            await evalPage(setControl(`${actSel("Set status")}.querySelector('select.arg-select')`, "Status.Done", "HTMLSelectElement"));
            await sleep(100);
            await evalPage(`${actSel("Set status")}.querySelector('.action-run').click()`);
            try { await waitFor(`.action-card[data-smoke-act="Set status"] .invoke-result`, 8000); } catch { fails.push("T: enum-param invoke produced no result"); }
            const res = await evalPage(`(()=>{const c=${actSel("Set status")};const r=c&&c.querySelector('.invoke-result');
              return r?{err:r.classList.contains('invoke-error'), text:r.textContent}:{none:true};})()`);
            if (res.none) fails.push("T: enum-param invoke produced no result element");
            else if (res.err || /unknown identifier/i.test(res.text)) fails.push(`T: enum-param invoke errored (bare-word bug NOT fixed): ${JSON.stringify(res.text)}`);
            else ok("T: enum-param action invokes cleanly (no 'unknown identifier')");
          }
        }
        // --- ENTITY param: "Reassign" (to: User) is a <select> populated from User.instances()
        if (!await evalPage(openAction("Reassign"))) fails.push("T: kanban Card has no 'Reassign' action");
        else {
          let ent = null;
          for (let i = 0; i < 40; i++) { await sleep(100);
            ent = await evalPage(`(()=>{const c=${actSel("Reassign")};if(!c)return{gone:true};
              const sel=c.querySelector('.action-form select.arg-select');
              return{hasSelect:!!sel, opts: sel?[...sel.options].map(o=>o.value).filter(Boolean):[]};})()`);
            if (ent.hasSelect && ent.opts.length) break;
          }
          if (!ent || ent.gone) fails.push("T: 'Reassign' action card disappeared");
          else if (!ent.hasSelect) fails.push("T: entity param 'Reassign' did not render a <select> picker");
          else if (!ent.opts.some((o) => /^User#\d+$/.test(o))) fails.push(`T: entity picker not populated with live User#slot options (${JSON.stringify(ent.opts)})`);
          else {
            ok(`T: entity param renders a <select> of live instances (${JSON.stringify(ent.opts)})`);
            await evalPage(setControl(`${actSel("Reassign")}.querySelector('select.arg-select')`, ent.opts[0], "HTMLSelectElement"));
            await sleep(100);
            await evalPage(`${actSel("Reassign")}.querySelector('.action-run').click()`);
            try { await waitFor(`.action-card[data-smoke-act="Reassign"] .invoke-result`, 8000); } catch { fails.push("T: entity-param invoke produced no result"); }
            const res = await evalPage(`(()=>{const c=${actSel("Reassign")};const r=c&&c.querySelector('.invoke-result');return r?{err:r.classList.contains('invoke-error'),text:r.textContent}:{none:true};})()`);
            if (res.none || res.err || /unknown identifier/i.test(res.text || "")) fails.push(`T: entity-param invoke errored: ${JSON.stringify(res)}`);
            else ok("T: entity-param action invokes cleanly (Type#slot -> Type.at(slot,0))");
          }
        }
      } catch (e) { fails.push("T: kanban typed-arg scenario failed: " + (e && e.message || e)); }
    }

    // bank "Transfer" (to: Account, amount: Int): the entity param is a picker; the Int param
    // validates + BLOCKS invoke (the run button is disabled) until it is a real number.
    const bport = await bootProgram("bank.scry");
    if (!bport) fails.push("T: bank never printed a viewer URL");
    else {
      await send("Page.navigate", { url: `http://127.0.0.1:${bport}/` });
      try {
        await openFirstInstance("Account");
        if (!await evalPage(openAction("Transfer"))) fails.push("T: bank Account has no 'Transfer' action");
        else {
          let tf = null;
          for (let i = 0; i < 40; i++) { await sleep(100);
            tf = await evalPage(`(()=>{const c=${actSel("Transfer")};if(!c)return{gone:true};
              const f=c.querySelector('.action-form'); const sel=f&&f.querySelector('select.arg-select');
              return{hasSelect:!!sel, opts: sel?[...sel.options].map(o=>o.value).filter(Boolean):[], hasNum:!!(f&&f.querySelector('input')),
                     runDisabled: !!(c.querySelector('.action-run')&&c.querySelector('.action-run').disabled)};})()`);
            if (tf.hasSelect && tf.opts.length) break;
          }
          if (!tf || tf.gone) fails.push("T: 'Transfer' action card disappeared");
          else if (!tf.hasSelect || !tf.opts.some((o) => /^Account#\d+$/.test(o))) fails.push(`T: Transfer 'to' param is not an Account picker (${JSON.stringify(tf)})`);
          else {
            ok(`T: bank Transfer 'to' param is an Account picker (${JSON.stringify(tf.opts)})`);
            if (!tf.hasNum) fails.push("T: Transfer 'amount' Int param has no text input");
            // block-on-invalid: with amount empty (and/or 'to' unset) the run button is disabled
            if (!tf.runDisabled) fails.push("T: Transfer run button was NOT disabled while required params were empty (no blocking)");
            else ok("T: Transfer run button is disabled while a required param is empty (invoke blocked)");
            // fill both: pick an account + a valid amount -> run enables, invoke is clean
            await evalPage(setControl(`${actSel("Transfer")}.querySelector('select.arg-select')`, tf.opts[0], "HTMLSelectElement"));
            await evalPage(setControl(`${actSel("Transfer")}.querySelector('.action-form input')`, "10", "HTMLInputElement"));
            await sleep(150);
            const enabled = await evalPage(`!${actSel("Transfer")}.querySelector('.action-run').disabled`);
            if (!enabled) fails.push("T: Transfer run button stayed disabled after valid entity + number were entered");
            else {
              ok("T: Transfer run button enables once entity + number are valid");
              await evalPage(`${actSel("Transfer")}.querySelector('.action-run').click()`);
              try { await waitFor(`.action-card[data-smoke-act="Transfer"] .invoke-result`, 8000); } catch { fails.push("T: Transfer invoke produced no result"); }
              const res = await evalPage(`(()=>{const c=${actSel("Transfer")};const r=c&&c.querySelector('.invoke-result');return r?{err:r.classList.contains('invoke-error'),text:r.textContent}:{none:true};})()`);
              if (res.none || res.err || /unknown identifier/i.test(res.text || "")) fails.push(`T: Transfer invoke errored: ${JSON.stringify(res)}`);
              else ok("T: Transfer (entity + Int) invokes cleanly with well-typed source");
            }
          }
        }
      } catch (e) { fails.push("T: bank typed-arg scenario failed: " + (e && e.message || e)); }
    }
    // return the page to the assistant viewer so the remaining portal/inspect beats are unaffected
    await send("Page.navigate", { url: viewerURL });

    // (T-safety) SAFETY NET for the reported viewer white-screen: a bad arg that reaches eval
    // (here the List<Tool> free-text fallback given a bare word `test`) returns a clean server
    // TypeError — but rendering that error must NEVER throw and unmount the whole app. Invoke
    // ScriptedModel.respond(convo, tools=`test`): assert the error renders INLINE, the form stays
    // usable, and the app root (#app) is still mounted (no white-screen).
    try {
      // whichever Model brain is live depends on whether LLM keys are present (scrubbed in the
      // harness -> ScriptedModel; present -> AnthropicModel). Both expose respond(convo, tools:
      // List<Tool>) with a List free-text fallback, so try each until one has a live instance.
      await waitFor(".vt-btn", 12000);
      await evalPage(`[...document.querySelectorAll('.vt-btn')].find(b=>b.textContent.trim()==='List').click()`);
      await waitFor(".type-row", 12000);
      // Model brains are interface implementors -> under a collapsed `Model ‹interface›` group;
      // expand every interface group so the implementor rows are in the DOM to click.
      await evalPage(`(()=>{document.querySelectorAll('#rail .type-row.iface').forEach(r=>{const c=r.querySelector('.caret');if(c&&!c.classList.contains('open'))r.click();});})()`);
      await sleep(300);
      let modelOpened = false;
      for (const modelType of ["ScriptedModel", "AnthropicModel"]) {
        const clicked = await evalPage(`(()=>{const r=[...document.querySelectorAll('.type-row')].find(e=>{const n=e.querySelector('.tname');return n&&n.textContent.trim()===${JSON.stringify(modelType)};});if(r)r.click();return !!r;})()`);
        if (!clicked) continue;
        let hasRow = false;
        for (let i = 0; i < 30; i++) { await sleep(100);
          hasRow = await evalPage(`!!document.querySelector('.itable tbody tr')`);
          if (hasRow || await evalPage(`!!document.querySelector('.empty')`)) break;
        }
        if (hasRow) { await evalPage(`document.querySelector('.itable tbody tr').click()`); await waitFor(".field-grid", 10000); modelOpened = true; break; }
      }
      if (!modelOpened) { console.log("  --  no live Model instance to test the safety net (skipped)"); throw { skip: true }; }
      await waitFor(".method", 10000);
      const opened = await evalPage(`(()=>{
        const card=[...document.querySelectorAll('.method')].find(c=>{const s=c.querySelector('.method-sig');return s&&s.textContent.trim().startsWith('respond(');});
        if(!card) return false; card.dataset.smokeBad='1'; card.querySelector('.method-head').click(); return true;
      })()`);
      if (!opened) console.log("  --  no ScriptedModel.respond method to test the safety net (skipped)");
      else {
        // wait for the convo entity picker to populate a Conversation option
        let ready = null;
        for (let i = 0; i < 40; i++) { await sleep(100);
          ready = await evalPage(`(()=>{const c=document.querySelector('.method[data-smoke-bad="1"]');if(!c)return{gone:true};
            const sel=c.querySelector('.method-body select.arg-select'); const inp=c.querySelector('.method-body input');
            return{sel:!!sel, opts: sel?[...sel.options].map(o=>o.value).filter(Boolean):[], inp:!!inp};})()`);
          if (ready.sel && ready.opts.length && ready.inp) break;
        }
        if (!ready || ready.gone || !ready.sel || !ready.inp) fails.push(`T-safety: respond card did not expose convo picker + tools input (${JSON.stringify(ready)})`);
        else {
          // select a real Conversation, and give the List<Tool> param a BARE WORD -> server errors
          await evalPage(`(()=>{const el=document.querySelector('.method[data-smoke-bad="1"] select.arg-select');
            const s=Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype,'value').set;s.call(el,${JSON.stringify("")});
            s.call(el, [...el.options].map(o=>o.value).filter(Boolean)[0]); el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
          await evalPage(`(()=>{const el=document.querySelector('.method[data-smoke-bad="1"] .method-body input');
            const s=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;s.call(el,'test');el.dispatchEvent(new Event('input',{bubbles:true}));})()`);
          await sleep(150);
          await evalPage(`document.querySelector('.method[data-smoke-bad="1"] .method-body .invoke-btn').click()`);
          try { await waitFor('.method[data-smoke-bad="1"] .invoke-result.invoke-error', 8000); }
          catch { fails.push("T-safety: bad-arg invoke did not render an inline error"); }
          const after = await evalPage(`(()=>{
            const app=document.querySelector('#app'); const err=document.querySelector('.method[data-smoke-bad="1"] .invoke-result.invoke-error');
            return { rootMounted: !!(app && app.childElementCount > 0), topbar: !!document.querySelector('#topbar'),
                     errText: err?err.textContent:null, formUsable: !!document.querySelector('.method[data-smoke-bad="1"] .method-body input') };
          })()`);
          if (!after.rootMounted || !after.topbar) fails.push("T-safety: the app root white-screened (unmounted) while rendering the invoke error");
          else if (!after.errText) fails.push("T-safety: no inline error text after a bad-arg invoke");
          else if (!after.formUsable) fails.push("T-safety: the invoke form was destroyed after the error");
          else ok(`T-safety: a bad-arg invoke renders the error INLINE with the app still mounted (${JSON.stringify(after.errText.slice(0,60))})`);
        }
      }
    } catch (e) { if (!(e && e.skip)) fails.push("T-safety: bad-arg render-safety scenario failed: " + (e && e.message || e)); }
    await send("Page.navigate", { url: viewerURL });

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
      const portal = spawn(join(ROOT, "scry"), ["portal"], { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] });
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
            // Phase V5: a STATIC project card (discovered from the working tree — no process)
            // opens the V3 type-skeleton served from /proj, with the "schema · not running" note.
            await waitFor(".pcard.project", 6000);
            const projNames = await evalPage(`[...document.querySelectorAll('.pcard.project .pcard-name')].map(n=>n.textContent)`);
            ok(`portal landing shows ${projNames.length} project card(s): ${JSON.stringify(projNames.slice(0,6))}`);
            // click a project card whose name contains 'assistant' (deterministic type set)
            const clicked = await evalPage(`(()=>{const c=[...document.querySelectorAll('.pcard.project')].find(x=>/assistant/.test(x.textContent));if(c){c.click();return true;}return false;})()`);
            if (!clicked) fails.push("portal: no 'assistant' project card to click");
            else {
              await waitFor("#nested.skeleton .cx-name", 12000);
              const stTypes = await evalPage(`[...document.querySelectorAll('#nested .cx-name')].map(n=>n.textContent.trim())`);
              const hasNote = await evalPage(`!!document.querySelector('.schema-affordance')`);
              if (!stTypes.includes("Agent")) fails.push(`static project view missing Agent: ${JSON.stringify(stTypes)}`);
              else if (!hasNote) fails.push("static project view missing 'schema · not running' affordance");
              else ok(`static project card opens the V3 type-skeleton from /proj (${stTypes.length} types, no process)`);
              // (bug fix) clicking a TYPE in the static-portal Map must show its fields/methods, NOT
              // "type not in schema". The /proj route serves schema()/views()/actions() but NOT
              // types(), so the inspector must resolve the type against the schema() payload
              // (fullSchema), which the InspectorPanel now receives. Click the Agent type cell.
              const findTypeCellJS2 = (name) => `(()=>{
                for(const r of document.querySelectorAll('#nested .region')){
                  const nm=r.querySelector('.region-head .region-name');
                  if(nm&&nm.textContent.trim()===${JSON.stringify(name)}) return r;
                }
                return null;
              })()`;
              await evalPage(`(()=>{const a=${findTypeCellJS2("Agent")};if(a)a.querySelector('.region-head').click();return !!a;})()`);
              try {
                await waitFor("#inspector .field-grid", 6000);
                const st = await evalPage(`(()=>{const insp=document.querySelector('#inspector');
                  return {notInSchema:/type not in schema/i.test(insp.textContent), fields:insp.querySelectorAll('.field-grid .fname').length};})()`);
                if (st.notInSchema) fails.push("bug: static-portal type inspector shows 'type not in schema' (schema prop not resolved to schema())");
                else if (st.fields < 1) fails.push("bug: static-portal type inspector shows no fields");
                else ok(`bug fix: clicking a type in the static-portal Map shows its ${st.fields} fields (no 'type not in schema')`);
                await evalPage(`(()=>{const b=document.querySelector('#inspector .insp-btn[title="close inspector"]');if(b)b.click();return true;})()`);
              } catch (e) { fails.push("bug fix: static-portal type-cell inspector scenario failed: " + (e && e.message || e)); }
              await evalPage(`(()=>{const b=document.querySelector('.back-btn');if(b)b.click();return !!b;})()`);
              await waitFor(".pcard", 6000);
            }
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
        // (V3) with main() never run, no instances exist — so the Map view renders the TYPE-LEVEL
        // SKELETON: the same bespoke nested view drawn from schema() alone. Assert the skeleton
        // structure (Agent type-cell nests a Conversation type-cell nests a Message placeholder
        // stack; a shared Tool type chip appears; the infra strip is present) and that a declared
        // view renders as a TEMPLATE (toggle the Agent type-cell to board -> AgentBoard timeline
        // placeholder + tool-type chips). Never the old "no live instances" dead-end, never a crash.
        await waitFor("#nested.skeleton", 12000);
        ok("inspect mode: Map view renders the type-level SKELETON (not the dead-end empty state)");
        // (F2 static) functions are first-class Map citizens STATICALLY too: the skeleton Map lists
        // the program's top-level functions with signatures (parallel to how classes show statically),
        // with a "run the program to trace" affordance — no error, no running process needed.
        try {
          await waitFor("#nested .fnsec .fn-item", 8000);
          const statFns = await evalPage(`[...document.querySelectorAll('#nested .fnsec .fn-item .fn-name')].map(e=>e.textContent.trim())`);
          const hasSig = await evalPage(`!!document.querySelector('#nested .fnsec .fn-item .fn-sig')`);
          const runNote = await evalPage(`(()=>{const s=document.querySelector('#nested .fnsec .fnsec-head .sub');return s?s.textContent:'';})()`);
          if (!statFns.includes("contains") || !statFns.includes("main")) fails.push(`F2 static: functions section missing expected functions (got ${JSON.stringify(statFns.slice(0,8))})`);
          else if (!hasSig) fails.push("F2 static: function items render no signature");
          else if (!/run the program to trace/.test(runNote)) fails.push(`F2 static: functions section missing 'run the program to trace' affordance (got ${JSON.stringify(runNote)})`);
          else ok(`F2 static: inspect Map lists ${statFns.length} functions with signatures + a run-to-trace note (no process)`);
        } catch (e) { fails.push("F2 static: functions section did not render in inspect Map (" + (e && e.message || e) + ")"); }
        // a helper (in-page) that finds a TYPE cell whose header names a given type
        const findTypeCellJS = (name) => `(()=>{
          for(const r of document.querySelectorAll('#nested .region')){
            const head=r.querySelector('.region-head');
            const nm=head&&head.querySelector('.region-name');
            if(nm&&nm.textContent.trim()===${JSON.stringify(name)}) return r;
          }
          return null;
        })()`;
        // Agent type-cell nests a Conversation type-cell that owns a Message placeholder stack
        const nest = await evalPage(`(()=>{
          const agent=${findTypeCellJS("Agent")};
          if(!agent) return {noagent:true};
          const conv=[...agent.querySelectorAll('.region .region-name')].some(n=>n.textContent.trim()==='Conversation');
          const rows=agent.querySelectorAll('.conv .mstack .mrow').length;
          return {conv, rows};
        })()`);
        if (nest.noagent) fails.push("inspect skeleton: no Agent type-cell");
        else {
          if (!nest.conv) fails.push("inspect skeleton: Agent type-cell does not nest a Conversation type-cell");
          else ok("inspect skeleton: Agent type nests Conversation nests a Message placeholder stack");
          if (nest.rows < 1) fails.push("inspect skeleton: nested Conversation renders no Message placeholder rows"); else ok(`inspect skeleton: placeholder message stack renders (${nest.rows} rows)`);
        }
        // a shared Tool TYPE chip appears (identity-colored, referenced by >=2 owner types)
        const tchip = await evalPage(`(()=>{
          const chips=[...document.querySelectorAll('#nested .stage-wrap .chip[data-identity]')];
          const names=[...new Set(chips.map(c=>c.dataset.identity))];
          return {chipCount:chips.length, names};
        })()`);
        if (tchip.chipCount < 1) fails.push("inspect skeleton: no shared type chip appears");
        else ok(`inspect skeleton: shared type chips render (${tchip.names.length} distinct: ${JSON.stringify(tchip.names.slice(0,4))})`);
        // the infrastructure strip renders in skeleton mode
        if (!await evalPage(`!!document.querySelector('#nested .infra')`)) fails.push("inspect skeleton: infrastructure strip did not render");
        else ok("inspect skeleton: infrastructure strip renders (utility types recede)");
        // (V3 template) toggle the Agent type-cell to its declared-view BOARD -> AgentBoard renders
        // as a TEMPLATE: a timeline placeholder + identity tool-type chips, titled by field wiring.
        const hasToggle = await evalPage(`(()=>{const a=${findTypeCellJS("Agent")};return !!(a&&a.querySelector('.vtoggle'));})()`);
        if (!hasToggle) fails.push("inspect skeleton: Agent type-cell has no declared-view toggle (AgentBoard not surfaced)");
        else {
          ok("inspect skeleton: Agent type-cell shows a cell<->board toggle (AgentBoard declared)");
          await evalPage(`(()=>{const a=${findTypeCellJS("Agent")};const b=[...a.querySelectorAll('.vtoggle button')].find(x=>/board/.test(x.textContent));b.click();return true;})()`);
          try { await waitFor('#nested .board[data-view="AgentBoard"] .timeline', 8000); } catch (e) { fails.push("inspect template: toggling to board never rendered an AgentBoard timeline"); }
          const board = await evalPage(`(()=>{
            const b=document.querySelector('#nested .board[data-view="AgentBoard"]');
            if(!b) return {noboard:true};
            return {rows:b.querySelectorAll('.timeline .tl').length, chips:b.querySelectorAll('.vchips .chip').length, hasTitle:!!b.querySelector('.board-title'), wired:!!b.querySelector('.tmpl-wire')};
          })()`);
          if (board.noboard) fails.push("inspect template: toggling to board did not render an AgentBoard");
          else {
            if (!board.hasTitle || !board.wired) fails.push("inspect template: AgentBoard missing title / field-name wiring"); else ok("inspect template: AgentBoard header shows title/badge FIELD-NAME wiring");
            if (board.rows < 1) fails.push("inspect template: AgentBoard timeline rendered no placeholder rows"); else ok(`inspect template: AgentBoard timeline placeholder renders (${board.rows} rows)`);
            if (board.chips < 1) fails.push("inspect template: AgentBoard tools section rendered no type chips"); else ok(`inspect template: AgentBoard tools section renders ${board.chips} tool-type chips`);
          }
        }
        // (V4 static) in inspect mode there are no instances — clicking a TYPE cell opens that
        // TYPE's static detail (fields/methods/implementors) in the inspector, NOT an error, and
        // the skeleton map stays visible. First toggle the Agent cell back from board -> cell.
        await evalPage(`(()=>{const b=document.querySelector('#nested .board[data-view="AgentBoard"] .vtoggle button');if(b)b.click();return true;})()`);
        try { await waitFor("#nested.skeleton .region .region-head", 6000); } catch {}
        await evalPage(`(()=>{const a=${findTypeCellJS("Agent")};if(a)a.querySelector('.region-head').click();return !!a;})()`);
        try {
          await waitFor("#inspector .field-grid", 6000);
          const stat = await evalPage(`(()=>{
            const insp=document.querySelector('#inspector'); if(!insp) return {noinsp:true};
            return { title:(insp.querySelector('.pane-title')||{}).textContent, fields:insp.querySelectorAll('.field-grid .fname').length,
                     stillSkeleton:!!document.querySelector('#nested.skeleton') };
          })()`);
          if (stat.noinsp) fails.push("V4 static: clicking a type cell did not open the inspector");
          else {
            if (!stat.stillSkeleton) fails.push("V4 static: opening a type detail left the skeleton map");
            else ok(`V4 static: clicking a type cell opens its static detail in the inspector (${stat.title}), map still skeleton`);
            if (stat.fields < 1) fails.push("V4 static: type static detail shows no fields"); else ok(`V4 static: type static detail shows ${stat.fields} fields`);
          }
          await evalPage(`(()=>{const b=document.querySelector('#inspector .insp-btn[title="close inspector"]');if(b)b.click();return true;})()`);
        } catch (e) { fails.push("V4 static: type-cell inspector scenario failed: " + (e && e.message || e)); }

        // switch to the List rail (no liveCount gate) and browse Agent's (empty) table. With 0 live
        // instances the TablePane renders the CLASS SCHEMA fallback (TablePaneSchema: "no live
        // instances — showing the class schema") rather than a bare empty div — so accept either the
        // schema fallback or a real table, and assert it is NOT a crash/error.
        await evalPage(`[...document.querySelectorAll('.vt-btn')].find(b=>b.textContent.trim()==='List').click()`);
        await waitFor(".type-row");
        await evalPage(`(()=>{const r=[...document.querySelectorAll('.type-row')].find(e=>{const n=e.querySelector('.tname');return n&&n.textContent.trim()==='Agent';});r.click();return true;})()`);
        await waitFor(".empty, .itable tbody tr, #pane .detail-section", 8000);
        const zeroState = await evalPage(`(()=>{
          const empty=!!document.querySelector('.empty');
          const schemaFallback=!!document.querySelector('#pane .detail-section');
          const rows=document.querySelectorAll('.itable tbody tr').length;
          const err=!!document.querySelector('#pane .invoke-error');
          return {empty, schemaFallback, rows, err};
        })()`);
        if (zeroState.err) fails.push("inspect: browsing the zero-count Agent table rendered an error (expected the class-schema fallback)");
        else if (zeroState.rows > 0) fails.push(`inspect: zero-count Agent table unexpectedly rendered ${zeroState.rows} rows (main() never ran)`);
        else if (!zeroState.empty && !zeroState.schemaFallback) fails.push("inspect: browsing the zero-count Agent table rendered neither the empty state nor the class-schema fallback");
        else ok("inspect mode: browsing a zero-count type renders the class-schema fallback, no crash");
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
