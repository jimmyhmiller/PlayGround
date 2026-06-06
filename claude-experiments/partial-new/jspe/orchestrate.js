#!/usr/bin/env node
// orchestrate.js — drive DeepSeek agents to fill the UNIMPLEMENTED stubs.
//
//   DEEPSEEK_KEY=...  node orchestrate.js [--only=SUBSTR] [--limit=N] [--dry] [--list]
//
// For every `UNIMPLEMENTED("MARKER")` stub it finds in the module files, it asks the
// model for {impl, test}, then UNDER A PER-FILE LOCK: snapshot -> replace the stub line
// -> write test/MARKER.test.js -> `node test/run.js MARKER`. Keep iff the gate passes;
// otherwise REVERT (so the shared file is always valid). Loops in passes until a pass
// resolves nothing (dependencies resolve themselves: a stub whose dep is still a stub
// just fails its gate and is retried next pass).
const fs = require("fs"), path = require("path"), cp = require("child_process");
const DIR = __dirname;

const KEY = process.env.DEEPSEEK_KEY || process.env.DEEPSEEK_API_KEY;
const BASE = process.env.DEEPSEEK_BASE || "https://api.deepseek.com";
const MODEL = process.env.DEEPSEEK_MODEL || "deepseek-chat";
const CONCURRENCY = +(process.env.CONCURRENCY || 5);
const ATTEMPTS = +(process.env.ATTEMPTS || 3);
const arg = (k) => (process.argv.find((a) => a.startsWith(k)) || "").split("=")[1];
const ONLY = arg("--only");
const LIMIT = +(arg("--limit") || 0);
const DRY = process.argv.includes("--dry");
const LIST = process.argv.includes("--list");

const MODULES = ["ir.js", "state.js", "lower.js", "whistle.js",
  "step/arith.js", "step/mem.js", "step/control.js", "step/call.js"];
const read = (f) => fs.readFileSync(path.join(DIR, f), "utf8");
const write = (f, s) => fs.writeFileSync(path.join(DIR, f), s);
const TASKS = JSON.parse(read("tasks.json")).tasks;
const taskFor = (m) => TASKS.find((t) => t.symbol === m) || TASKS.find((t) => t.symbol && (m.endsWith(t.symbol.split(".").pop()) || t.symbol.includes(m))) || null;

function scan() {
  const items = [];
  for (const f of MODULES) {
    let lines; try { lines = read(f).split("\n"); } catch { continue; }
    for (let i = 0; i < lines.length; i++) {
      const m = lines[i].match(/UNIMPLEMENTED\("([^"]+)"\)/);
      if (!m) continue;
      let c = i - 1, spec = [];
      while (c >= 0 && lines[c].trim().startsWith("//")) { spec.unshift(lines[c].trim().replace(/^\/\/\s?/, "")); c--; }
      items.push({ marker: m[1], file: f, specComment: spec.join("\n"), task: taskFor(m[1]) });
    }
  }
  return items.filter((i) => !ONLY || i.marker.includes(ONLY)).slice(0, LIMIT || undefined);
}

function buildMessages(item, feedback) {
  const fileSrc = read(item.file);
  const reqPath = "../" + item.file;
  const t = item.task || {};
  const sys = `You implement ONE function in a JavaScript port of a partial evaluator (porting Rust at src/js.rs and src/engine.rs).
Return ONLY a JSON object: {"impl": "<code>", "test": "<code>"}.
- impl: the COMPLETE replacement for the single stub line shown. Keep the EXACT left-hand side / declaration (same name and params); just implement it. May be multiple lines. No markdown fences.
- test: a Node CommonJS test file. It must be: module.exports = [ {name:"...", fn:()=>{ ... throw on failure ... }} , ... ] with >= 3 cases. require the module as require("${reqPath}") and shapes from require("../contracts.js"). No markdown fences.
Edit nothing else. Match the behavior of the Rust reference exactly.`;
  const usr = `## Conventions (AGENTS.md)\n${read("AGENTS.md")}

## Shared contracts (contracts.js) — data shapes, never edit\n\`\`\`js\n${read("contracts.js")}\n\`\`\`

## File you are editing: ${item.file}\n\`\`\`js\n${fileSrc}\n\`\`\`

## Your task — implement the stub with marker "${item.marker}"
Spec (from comments + manifest): ${item.specComment}
${t.spec ? "Manifest spec: " + t.spec : ""}
${t.rustRef ? "Rust reference (behavior is defined here): " + t.rustRef : ""}
Signature: ${t.signature || "(see the stub line)"}
${feedback ? "\nThe previous attempt FAILED its gate. Fix it. Failure output:\n" + feedback.slice(0, 1500) : ""}`;
  return [{ role: "system", content: sys }, { role: "user", content: usr }];
}

async function callLLM(messages) {
  const res = await fetch(BASE + "/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + KEY },
    body: JSON.stringify({ model: MODEL, messages, temperature: 0, response_format: { type: "json_object" }, max_tokens: 4096 }),
  });
  if (!res.ok) throw new Error("API " + res.status + ": " + (await res.text()).slice(0, 300));
  const j = await res.json();
  const content = j.choices[0].message.content;
  const obj = JSON.parse(content);
  if (typeof obj.impl !== "string" || typeof obj.test !== "string") throw new Error("model did not return {impl,test}");
  return obj;
}

// per-file mutex
const locks = {};
function withLock(file, fn) {
  const prev = locks[file] || Promise.resolve();
  let release; const gate = new Promise((r) => (release = r));
  locks[file] = prev.then(() => gate);
  return prev.then(() => Promise.resolve().then(fn)).finally(release);
}

function applyGate(item, impl, test) {
  return withLock(item.file, () => {
    const lines = read(item.file).split("\n");
    const idx = lines.findIndex((l) => l.includes(`UNIMPLEMENTED("${item.marker}")`));
    if (idx < 0) return { ok: true, note: "already done" };
    const snapshot = lines.join("\n");
    lines[idx] = impl;
    write(item.file, lines.join("\n"));
    const tf = path.join("test", item.marker + ".test.js");
    write(tf, test);
    let r;
    try { cp.execSync(`node test/run.js ${JSON.stringify(item.marker)}`, { cwd: DIR, stdio: "pipe" }); r = { ok: true }; }
    catch (e) { r = { ok: false, output: (e.stdout || "") + (e.stderr || "") + (e.message || "") }; }
    if (!r.ok) { write(item.file, snapshot); try { fs.unlinkSync(path.join(DIR, tf)); } catch {} }
    return r;
  });
}

async function doItem(item) {
  let feedback = null;
  for (let a = 0; a < ATTEMPTS; a++) {
    let resp;
    try { resp = await callLLM(buildMessages(item, feedback)); }
    catch (e) { feedback = "LLM error: " + e.message; continue; }
    const r = await applyGate(item, resp.impl, resp.test);
    if (r.ok) return { marker: item.marker, ok: true };
    feedback = r.output;
  }
  return { marker: item.marker, ok: false, feedback };
}

async function pool(items, n, fn) {
  const out = []; let i = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => {
    while (i < items.length) { const k = i++; out[k] = await fn(items[k]); }
  }));
  return out;
}

(async () => {
  if (!KEY && !DRY && !LIST) { console.error("set DEEPSEEK_KEY"); process.exit(2); }
  if (LIST) { for (const it of scan()) console.log(it.marker.padEnd(22), it.file); console.log("\n" + scan().length + " stubs"); return; }
  if (DRY) { const it = scan()[0]; console.log("model:", MODEL, "base:", BASE, "\n--- first prompt ---\n", buildMessages(it).map((m) => `[${m.role}]\n` + m.content).join("\n\n")); return; }

  console.log(`orchestrating with ${MODEL} @ ${BASE}, concurrency ${CONCURRENCY}`);
  const failed = new Map();
  for (let pass = 1; ; pass++) {
    const todo = scan();
    if (!todo.length) { console.log("\nALL STUBS DONE 🎉"); break; }
    console.log(`\n=== pass ${pass}: ${todo.length} stubs remaining ===`);
    const before = todo.length;
    const results = await pool(todo, CONCURRENCY, doItem);
    for (const r of results) {
      if (r.ok) console.log("  ✓ " + r.marker);
      else { console.log("  ✗ " + r.marker); failed.set(r.marker, r.feedback); }
    }
    if (scan().length === before) {
      console.log(`\nFIXPOINT: ${before} stubs could not be resolved this pass. Stuck:`);
      for (const it of scan()) console.log("  - " + it.marker + "  (" + it.file + ")");
      break;
    }
  }
  try { cp.execSync("node test/run.js", { cwd: DIR, stdio: "inherit" }); } catch {}
})();
