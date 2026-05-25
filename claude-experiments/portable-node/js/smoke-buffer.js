// End-to-end smoke test for the lifted Node Buffer.
// Buffer comes from Node's lib/buffer.js, executed on QuickJS via our
// host primitives — no Node runtime involved.
//
// Wrapped in an IIFE so successive smoke files don't collide on `const log`.

(function () {
const log = [];
const ok = (name, val) => log.push(`OK   ${name}: ${val}`);
const fail = (name, e) => log.push(`FAIL ${name}: ${e.stack ? e.stack.split('\n')[0] : e.message || e}`);
const check = (name, fn) => { try { ok(name, fn()); } catch (e) { fail(name, e); } };

check('typeof Buffer',            () => typeof Buffer);
check('Buffer.from(string) utf8', () => Buffer.from('hello').toString());
check('Buffer.from(string) len',  () => Buffer.from('héllo').length);
check('Buffer.alloc',             () => Buffer.alloc(3).length);
check('Buffer.alloc fill',        () => Buffer.alloc(3, 0xAB).toString('hex'));
check('toString hex',             () => Buffer.from('hi').toString('hex'));
check('toString base64',          () => Buffer.from('hello').toString('base64'));
check('toString utf8 multibyte',  () => Buffer.from([0xe6,0xbc,0xa2,0xe5,0xad,0x97]).toString('utf8'));
check('Buffer.compare',           () => Buffer.compare(Buffer.from('a'), Buffer.from('b')));
check('Buffer.concat',            () => Buffer.concat([Buffer.from('he'), Buffer.from('llo')]).toString());
check('Buffer.byteLength',        () => Buffer.byteLength('漢字'));
check('write returns bytes',      () => { const b = Buffer.alloc(8); return b.write('hi'); });
check('write+toString roundtrip', () => {
  const b = Buffer.alloc(8);
  b.write('test');
  return b.toString('utf8', 0, 4);
});
check('Buffer.isBuffer(true)',    () => Buffer.isBuffer(Buffer.from('x')));
check('Buffer.isBuffer(false)',   () => Buffer.isBuffer(new Uint8Array(1)));
check('indexOf',                  () => Buffer.from('hello').indexOf('ll'));
check('slice',                    () => Buffer.from('hello').slice(1, 4).toString());
check('readUInt8',                () => Buffer.from([10, 20, 30]).readUInt8(1));
check('readUInt16LE',             () => Buffer.from([1, 2, 3, 4]).readUInt16LE(0));
check('writeUInt32LE',            () => {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(0xdeadbeef, 0);
  return b.toString('hex');
});
check('binding utf8WriteStatic type', () => typeof __internalBinding('buffer').utf8WriteStatic);
check('binding byteLengthUtf8 type',  () => typeof __internalBinding('buffer').byteLengthUtf8);
check('byteLengthUtf8 surrogate',     () => __internalBinding('buffer').byteLengthUtf8('😀')); // 4
check('byteLengthUtf8 lone',          () => __internalBinding('buffer').byteLengthUtf8('\uD800x'));      // 3+1=4
check('Buffer.from lone surrogate',   () => Buffer.from('\uD800x').toString('hex'));                     // efbfbd78
check('utf8WriteStatic source',       () => String(__internalBinding('buffer').utf8WriteStatic).slice(0, 50));
check('Buffer.from big lone',         () => {
  // 5000 lone surrogates — > 4096 so goes through createFromString path.
  const s = '\uD800'.repeat(5000);
  return Buffer.from(s).length;
});
check('indexOf negative number',     () => Buffer.from('this is a test').indexOf(-140));
check('indexOf large uint32',        () => Buffer.from('this is a test').indexOf(0x6973));
// Node guarantees val is uint32 (val >>> 0) before reaching the binding;
// passing -140 directly is invalid use of the binding so we don't test it.
check('binding indexOfNumber large', () => __internalBinding('buffer').indexOfNumber(Buffer.from('this is a test'), 4294967156, 0, true, undefined));

return log.join('\n');
})()
