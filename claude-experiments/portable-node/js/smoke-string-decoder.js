(function () {
const { StringDecoder } = require('string_decoder');
const log = [];
const ok = (name, val) => log.push(`OK   ${name}: ${val}`);
const fail = (name, e) => log.push(`FAIL ${name}: ${e.message || e}`);
const check = (name, fn) => { try { ok(name, JSON.stringify(fn())); } catch (e) { fail(name, e); } };

const euroBytes = Buffer.from('€', 'utf8');  // E2 82 AC
check('euro bytes', () => Array.from(euroBytes).map(b => b.toString(16)));

// Whole-buffer write
check('whole', () => {
  const d = new StringDecoder('utf8');
  return d.write(euroBytes) + d.end();
});

// Byte-by-byte
check('byte by byte', () => {
  const d = new StringDecoder('utf8');
  let out = '';
  out += d.write(euroBytes.slice(0, 1));
  out += d.write(euroBytes.slice(1, 2));
  out += d.write(euroBytes.slice(2, 3));
  out += d.end();
  return out;
});

// 2+1 split
check('2+1', () => {
  const d = new StringDecoder('utf8');
  return d.write(euroBytes.slice(0, 2)) + d.write(euroBytes.slice(2, 3)) + d.end();
});

// 1+2 split
check('1+2', () => {
  const d = new StringDecoder('utf8');
  return d.write(euroBytes.slice(0, 1)) + d.write(euroBytes.slice(1, 3)) + d.end();
});

// E2 41 — incomplete lead byte followed by ASCII. Expected '�A'.
const e241 = Buffer.from('E241', 'hex');
check('E241 whole',  () => { const d = new StringDecoder('utf8'); const o = d.write(e241); return JSON.stringify(o + d.end()); });
check('E241 split',  () => { const d = new StringDecoder('utf8'); const o = d.write(e241.slice(0,1)) + d.write(e241.slice(1,2)); return JSON.stringify(o + d.end()); });

return log.join('\n');
})()
