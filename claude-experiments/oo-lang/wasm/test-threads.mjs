import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';
const base = new URL('./', import.meta.url);

// A background worker that counts up, pacing itself with Clock.sleep — exactly the shape of
// assistant.scry's `loop <task>` worker. On wasm there is no pthread: sched-tick drives it.
const prog = `
class Counter { n: Int }

class Worker implements Runnable {
  counter: Counter
  limit: Int
  fn run() {
    var i = 0
    while i < self.limit {
      self.counter.n = self.counter.n + 1
      i = i + 1
      Clock.sleep(1)
    }
  }
}

class App { handle: ThreadHandle }

fn main() {
  let c = Counter(n: 0)
  let h = Thread.spawn(Worker(counter: c, limit: 25))
  let a = App(handle: h)
  Console.log("spawned a background worker")
}
`;
let out = "";
const scry = await ScryWasm.instantiate(await readFile(new URL('scry.wasm', base)),
  { onStdout: t => out += t, onStderr: t => process.stderr.write("[err] " + t), vfs: { "/t.scry": prog } });

console.log("boot:", scry.boot("/t.scry"), JSON.stringify(out.trim()));
const n = () => scry.eval("Counter.instances().get(0).n").value?.value;
console.log("counter right after spawn (should be 0 — nothing has ticked):", n());

let ticks = 0;
for (let i = 0; i < 500; i++) {
  const runnable = Number(scry.exports.scry_tick());
  ticks++;
  if (i % 100 === 0) console.log(`  tick ${i}: runnable=${runnable} counter=${n()}`);
  if (runnable === 0) break;
}
console.log(`after ${ticks} ticks: counter =`, n(), "(expect 25)");
console.log("main thread still evaluable:", JSON.stringify(scry.eval("1+1").value));
