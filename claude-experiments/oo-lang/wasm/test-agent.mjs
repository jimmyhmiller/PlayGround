import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';
const base = new URL('./', import.meta.url);
const root = new URL('../', import.meta.url);
const rd = (u) => readFile(new URL(u, root), 'utf8');

// VFS laid out so module resolution works: root = the entry's dir = "/"
const vfs = {
  "/agentdemo.scry": await readFile(new URL('agentdemo.scry', base), 'utf8'),
  "/agent/core.scry": await rd('agent/core.scry'),
  "/std/json.scry":   await rd('std/json.scry'),
};
let out = "";
const scry = await ScryWasm.instantiate(await readFile(new URL('scry.wasm', base)),
  { onStdout: t => out += t, onStderr: t => process.stderr.write("[err] " + t), vfs });

console.log("boot rc:", scry.boot("/agentdemo.scry"));
console.log("banner:", JSON.stringify(out)); out = "";

const ask = (t) => { const r = scry.eval(`Session.instances().get(0).ask(${JSON.stringify(t)})`); return r.error ? JSON.stringify(r.error) : (out.trim() || r.value?.value); };
console.log("\n> hello           ->", ask("hello")); out = "";
console.log("> what is 6 * 7   ->", ask("what is 6 * 7")); out = "";
console.log("> weather in Tokyo->", ask("weather in Tokyo")); out = "";
console.log("\nlive Message count:", JSON.stringify(scry.eval("Message.instances().len()").value));
console.log("live Session turns:", JSON.stringify(scry.eval("Session.instances().get(0).turns").value));
