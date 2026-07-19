import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';
const base=new URL('./', import.meta.url);
// a program that echoes every line it reads, then exits at EOF
const prog = `fn main() {
  var running = true
  while running {
    match Console.readLine() {
      None -> { running = false }
      Some(line) -> { Console.log("you said: " + line) }
    }
  }
}
`;
let out=""; 
const s = await ScryWasm.instantiate(await readFile(new URL('scry.wasm', base)),
  { onStdout(t){out+=t}, onStderr(t){process.stderr.write("[err] "+t)}, vfs: { "/echo.scry": prog } });
s.feedLine("hello");
s.feedLine("world");
console.log("boot rc:", s.boot("/echo.scry"));
console.log("output:", JSON.stringify(out));
