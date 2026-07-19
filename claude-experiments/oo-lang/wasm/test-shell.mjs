// The agent's ShellTool / SearchTool / File tools must do REAL work against the in-memory
// filesystem — not trap on popen, and not silently return "".
import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';
const base = new URL('./', import.meta.url), root = new URL('../', import.meta.url);
const rd = (p) => readFile(new URL(p, root), 'utf8');
const s = await ScryWasm.instantiate(await readFile(new URL('scry.wasm', base)), {
  onStdout(){}, onStderr(){},
  env: { DEEPSEEK_API_KEY: "in-page-demo", ANTHROPIC_BASE_URL: "http://fake-api.in-page/v1", SCRY_MODEL: "fake-1 (in-page)" },
  vfs: {
    "/assistant.scry": await rd('examples/assistant.scry'),
    "/agent/core.scry": await rd('agent/core.scry'),
    "/std/json.scry":   await rd('std/json.scry'),
    "/notes.txt":       "scry runs in the browser\nthe VM is WebAssembly\n",
  },
});
s.boot("/assistant.scry");
let ok = true;
const check = (n, c, extra="") => { console.log(`${c?"ok  ":"FAIL"} ${n}${extra?" — "+extra:""}`); if(!c) ok=false; };
const run = (cmd) => s.eval(`Process.run(${JSON.stringify(cmd)})`).value?.value ?? "";

check("ls lists the VFS", /notes\.txt/.test(run("ls")), JSON.stringify(run("ls").slice(0,60)));
check("ls -la has a long format", /-rw-r--r--.*notes\.txt/.test(run("ls -la")));
check("cat reads a file", /WebAssembly/.test(run("cat /notes.txt")));
check("cat reports a missing file", /No such file/.test(run("cat /nope.txt")));
check("grep finds matches with file:line", /notes\.txt:1:.*browser/.test(run("grep -rnI -- 'browser' .")));
check("grep | head truncates", run("grep -rnI -- 'e' . | head -3").split("\n").filter(Boolean).length <= 3);
check("echo works", run("echo hi there").trim() === "hi there");
check("unknown command says so (not a silent empty)", /not available in the in-page shell/.test(run("kubectl get pods")));
check("no popen trap", !/popen/.test(run("ls")));

// and through the AGENT's tools, the way the demo actually uses it
let out = "";
s.opts.onStdout = (t) => { out += t; };
const turn = (line) => { out = ""; s.feedLine(line); for (let i=0;i<250;i++) s.tick(); return out; };

const listed = turn("list files");
check("agent ShellTool ran a real shell tool call", /tool_use: shell/.test(listed) && /notes\.txt/.test(listed),
      JSON.stringify(listed.replace(/\u001b\[[0-9;]*m/g,"").slice(0,70)));
const catted = turn("run cat /notes.txt");
check("agent ran an arbitrary command", /scry runs in the browser/.test(catted));
const searched = turn("search browser");
check("agent SearchTool greps the VFS", /tool_use: search/.test(searched) && /notes\.txt/.test(searched),
      JSON.stringify(searched.replace(/\u001b\[[0-9;]*m/g,"").slice(0,70)));

console.log(ok ? "PASS test-shell" : "FAIL test-shell");
process.exit(ok ? 0 : 1);
