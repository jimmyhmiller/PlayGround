// Smoke test exercising every primitive in __internalBinding('buffer').
// We don't have lib/buffer.js wired up yet — these calls hit the Rust side
// directly via the binding object.

const buf = __internalBinding('buffer');
const log = [];
const ok = (name, val) => log.push(`OK   ${name}: ${JSON.stringify(val)}`);
const fail = (name, e) => log.push(`FAIL ${name}: ${e.message || e}`);

const check = (name, fn) => {
  try { ok(name, fn()); } catch (e) { fail(name, e); }
};

// Helper: make a Uint8Array from a hex string.
const u8 = (...vals) => new Uint8Array(vals);
const eq = (a, b) => {
  if (a === b) return true;
  if (a instanceof Uint8Array && b instanceof Uint8Array) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  return JSON.stringify(a) === JSON.stringify(b);
};
const assert = (cond, msg) => { if (!cond) throw new Error(msg || 'assertion failed'); };

// ---- byteLengthUtf8 ----
check('byteLengthUtf8("hello")',  () => buf.byteLengthUtf8('hello'));
check('byteLengthUtf8("héllo")',  () => buf.byteLengthUtf8('héllo'));
check('byteLengthUtf8("漢字")',    () => buf.byteLengthUtf8('漢字'));

// ---- *Slice (bytes -> string) ----
check('utf8Slice', () => {
  const b = new Uint8Array([0x68, 0x69]);                        // "hi"
  const s = buf.utf8Slice(b, 0, 2);
  assert(s === 'hi', `got ${s}`);
  return s;
});
check('utf8Slice multibyte', () => {
  const b = new Uint8Array([0xe6, 0xbc, 0xa2, 0xe5, 0xad, 0x97]); // "漢字"
  return buf.utf8Slice(b, 0, 6);
});
check('asciiSlice', () => buf.asciiSlice(new Uint8Array([0x41, 0xC1]), 0, 2));  // "AA" (high bit masked)
check('latin1Slice', () => buf.latin1Slice(new Uint8Array([0xC9]), 0, 1));       // "É"
check('base64Slice', () => buf.base64Slice(new Uint8Array([0x68, 0x69]), 0, 2)); // "aGk="
check('base64urlSlice', () => buf.base64urlSlice(new Uint8Array([0xff, 0xfe, 0xfd]), 0, 3));
check('hexSlice', () => buf.hexSlice(new Uint8Array([0xde, 0xad, 0xbe, 0xef]), 0, 4));
check('ucs2Slice', () => buf.ucs2Slice(new Uint8Array([0x48, 0x00, 0x69, 0x00]), 0, 4)); // "Hi"

// ---- *Write (string -> bytes) ----
check('base64Write', () => {
  const out = new Uint8Array(8);
  const n = buf.base64Write(out, 'aGVsbG8=', 0, out.length);
  return { n, bytes: Array.from(out.slice(0, n)) }; // -> "hello"
});
check('base64urlWrite', () => {
  const out = new Uint8Array(8);
  const n = buf.base64urlWrite(out, 'aGVsbG8', 0, out.length);
  return { n, bytes: Array.from(out.slice(0, n)) };
});
check('hexWrite', () => {
  const out = new Uint8Array(4);
  const n = buf.hexWrite(out, 'deadbeef', 0, 4);
  return { n, bytes: Array.from(out) };
});
check('ucs2Write', () => {
  const out = new Uint8Array(4);
  const n = buf.ucs2Write(out, 'Hi', 0, 4);
  return { n, bytes: Array.from(out) };
});

// ---- atob / btoa ----
check('btoa', () => buf.btoa('hello'));
check('atob', () => {
  const s = buf.atob('aGVsbG8=');
  return { len: s.length, codepoints: [...s].map(c => c.charCodeAt(0)) };
});

// ---- compare / compareOffset ----
check('compare equal', () => buf.compare(new Uint8Array([1,2,3]), new Uint8Array([1,2,3])));
check('compare less',  () => buf.compare(new Uint8Array([1,2,3]), new Uint8Array([1,2,4])));
check('compare more',  () => buf.compare(new Uint8Array([2]),     new Uint8Array([1])));
check('compareOffset', () => buf.compareOffset(
  new Uint8Array([0,0,1,2,3]),  // source
  new Uint8Array([9,9,1,2,3,9]), // target
  2, 2,   // targetStart, sourceStart
  5, 5,   // targetEnd, sourceEnd
));

// ---- copy ----
check('copy', () => {
  const src = new Uint8Array([1,2,3,4,5]);
  const dst = new Uint8Array(5);
  const n = buf.copy(src, dst, 0, 0, 5);
  return { n, dst: Array.from(dst) };
});
check('copy with offsets', () => {
  const src = new Uint8Array([1,2,3,4,5]);
  const dst = new Uint8Array([0,0,0,0,0]);
  const n = buf.copy(src, dst, 1, 2, 4);   // dst[1..] = src[2..4]
  return { n, dst: Array.from(dst) };       // [0,3,4,0,0]
});

// ---- fill ----
check('fill with number', () => {
  const b = new Uint8Array(5);
  buf.fill(b, 0xAB, 0, 5, 'utf8');
  return Array.from(b);
});
check('fill with bytes pattern', () => {
  const b = new Uint8Array(7);
  buf.fill(b, new Uint8Array([1,2,3]), 0, 7, 'utf8');
  return Array.from(b); // [1,2,3,1,2,3,1]
});

// ---- isAscii / isUtf8 ----
check('isAscii true',  () => buf.isAscii(new Uint8Array([0x41, 0x42])));
check('isAscii false', () => buf.isAscii(new Uint8Array([0x41, 0xC0])));
check('isUtf8 valid',  () => buf.isUtf8(new Uint8Array([0xe6, 0xbc, 0xa2])));
check('isUtf8 invalid',() => buf.isUtf8(new Uint8Array([0xff, 0xfe])));

// ---- indexOf* ----
check('indexOfNumber fwd', () => buf.indexOfNumber(new Uint8Array([1,2,3,2,1]), 2, 0, true));   // 1
check('indexOfNumber rev', () => buf.indexOfNumber(new Uint8Array([1,2,3,2,1]), 2, 4, false));  // 3
check('indexOfNumber miss',() => buf.indexOfNumber(new Uint8Array([1,2,3]), 9, 0, true));       // -1
check('indexOfBuffer fwd', () => buf.indexOfBuffer(
  new Uint8Array([1,2,3,4,5]), new Uint8Array([3,4]), 0, '', true,
));
check('indexOfString utf8', () => buf.indexOfString(
  new Uint8Array([0x68,0x65,0x6c,0x6c,0x6f]), 'llo', 0, 'utf8', true,
));
check('indexOfString hex',  () => buf.indexOfString(
  new Uint8Array([0xde,0xad,0xbe,0xef]), 'beef', 0, 'hex', true,
));

// ---- swap16/32/64 ----
check('swap16', () => {
  const b = new Uint8Array([1,2,3,4]);
  buf.swap16(b);
  return Array.from(b); // [2,1,4,3]
});
check('swap32', () => {
  const b = new Uint8Array([1,2,3,4,5,6,7,8]);
  buf.swap32(b);
  return Array.from(b); // [4,3,2,1,8,7,6,5]
});
check('swap64', () => {
  const b = new Uint8Array([1,2,3,4,5,6,7,8]);
  buf.swap64(b);
  return Array.from(b); // [8,7,6,5,4,3,2,1]
});
check('swap16 odd len throws', () => {
  try { buf.swap16(new Uint8Array(3)); return 'NO THROW'; }
  catch (e) { return 'threw: ' + e.message; }
});

// ---- constants ----
check('kMaxLength', () => buf.kMaxLength);
check('kStringMaxLength', () => buf.kStringMaxLength);

log.join('\n');
