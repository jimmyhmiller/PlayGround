import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';
const base = new URL('./', import.meta.url);

// A program written the NATIVE way: a blocking REPL loop. It must now work unmodified in
// the browser — readLine SUSPENDS when no input is buffered instead of seeing EOF.
const prog = `
class Log { lines: Int }
fn main() {
  let l = Log(lines: 0)
  var running = true
  while running {
    Console.print("> ")
    match Console.readLine() {
      None -> { running = false }
      Some(line) -> {
        if line == "quit" { running = false }
        else { l.lines = l.lines + 1
               Console.log("echo " + line) }
      }
    }
  }
  Console.log("bye")
}
`;
let out = "";
const scry = await ScryWasm.instantiate(await readFile(new URL('scry.wasm', base)),
  { onStdout: t => out += t, onStderr: t => process.stderr.write("[err] " + t), vfs: { "/repl.scry": prog } });

console.log("boot:", scry.boot("/repl.scry"), "| output so far:", JSON.stringify(out));
console.log("  ^ main is SUSPENDED at readLine (prompt printed, no EOF, no 'bye')");

const pump = (n = 40) => { for (let i = 0; i < n; i++) scry.tick(); };
out = ""; scry.feedLine("hello"); pump();
console.log("after feeding 'hello':", JSON.stringify(out));
out = ""; scry.feedLine("world"); pump();
console.log("after feeding 'world':", JSON.stringify(out));
console.log("lines counted (live heap):", JSON.stringify(scry.eval("Log.instances().get(0).lines").value));
out = ""; scry.feedLine("quit"); pump();
console.log("after 'quit':", JSON.stringify(out), "(main should have finished)");
