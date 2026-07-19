import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';
const R = new URL('../', import.meta.url).pathname;
const rd = (p) => readFile(R + p, 'utf8');
let out = "";
const scry = await ScryWasm.instantiate(await readFile(R + 'wasm/scry.wasm'), {
  onStdout: t => out += t, onStderr: t => process.stderr.write("[err] " + t.trim() + "\n"),
  env: { DEEPSEEK_API_KEY: "in-page-demo", ANTHROPIC_BASE_URL: "http://fake-api.in-page/v1", SCRY_MODEL: "fake-1 (in-page)" },
  vfs: { "/assistant.scry": await rd('examples/assistant.scry'),
         "/agent/core.scry": await rd('agent/core.scry'),
         "/std/json.scry":   await rd('std/json.scry') },
});
console.log("boot:", scry.boot("/assistant.scry"));
console.log("brain line:", out.split("\n").find(l => /brain:/.test(l)));
const pump = (n=80) => { for (let i=0;i<n;i++) scry.tick(); };
out=""; scry.feedLine("hello"); pump();
console.log("\n[hello]\n" + out.trim());
out=""; scry.feedLine("what is 17 times 23?"); pump(150);
const out0 = out;
console.log("\n[17 times 23 — full protocol: buildBody -> host_http -> parseAnthropic]\n" + out.trim());
out=""; scry.feedLine("weather in Tokyo"); pump(150);
console.log("\n[weather in Tokyo]\n" + out.trim());
const n = scry.eval("HttpResponse.instances().len()").value?.value;
console.log("\nlive HttpResponse instances:", n);
const ok = /391/.test(out0) && n > 0;
console.log(ok ? "PASS: full buildBody -> host_http -> parseAnthropic protocol" : "FAIL");
process.exit(ok ? 0 : 1);
