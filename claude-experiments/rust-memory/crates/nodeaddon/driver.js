// Load the Rust addon the way any Node app does, then hammer it.
//
//   MEMSCOPE_RECORD=/tmp/addon.mscope node driver.js <path-to.node> [iterations]
//
// `process.dlopen` is what `require('./foo.node')` does underneath; using it
// directly keeps the fixture free of a package.json / build toolchain.
const path = require('path');

const file = process.argv[2];
const iterations = Number(process.argv[3] || '100');
if (!file) {
  console.error('usage: node driver.js <path-to.node> [iterations]');
  process.exit(2);
}

const addon = { exports: {} };
process.dlopen(addon, path.resolve(file));

if (typeof addon.exports.work !== 'function') {
  console.error('addon did not export work()');
  process.exit(3);
}

for (let i = 0; i < iterations; i++) {
  addon.exports.work();
}

console.log(`ok ${iterations}`);

// `hold`: keep the process (and the addon's live heap) around so a live agent
// can be attached to from outside — `memscope monitor` / `dump` / `graph`, and
// `kill -USR1` for a heap dump. Exits on SIGTERM.
if (process.argv[4] === 'hold') {
  let n = iterations;
  setInterval(() => {
    addon.exports.work();
    n++;
  }, 50);
  process.on('SIGTERM', () => process.exit(0));
}
