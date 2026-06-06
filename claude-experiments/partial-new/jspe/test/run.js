// Test runner.  node test/run.js [taskId]
//   - no arg: run every test/*.test.js
//   - taskId: run only test/<taskId>.test.js
// A test file exports an array of {name, fn}; fn throws on failure.
const fs = require("fs"), path = require("path");
const DIR = __dirname;
const only = process.argv[2];
const files = fs.readdirSync(DIR).filter((f) => f.endsWith(".test.js") && (!only || f === only + ".test.js"));
let pass = 0, fail = 0;
for (const f of files) {
  let cases;
  try { cases = require(path.join(DIR, f)); } catch (e) { console.log("LOAD FAIL " + f + ": " + e.message); fail++; continue; }
  for (const c of cases) {
    try { c.fn(); pass++; }
    catch (e) { fail++; console.log("FAIL " + f + " :: " + c.name + "\n   " + (e && e.message)); }
  }
}
console.log(`\n${pass} passed, ${fail} failed` + (only ? ` (task ${only})` : ` over ${files.length} files`));
process.exit(fail ? 1 : 0);
