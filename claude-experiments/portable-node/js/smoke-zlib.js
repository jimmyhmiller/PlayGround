// Smoke test for node:zlib (portable shim using __host.zlib.* primitives).
(function () {
const log = [];
const ok = (name, val) => log.push(`OK   ${name}: ${val}`);
const fail = (name, e) => log.push(`FAIL ${name}: ${e.message || e}`);
const check = (name, fn) => { try { ok(name, JSON.stringify(fn())); } catch (e) { fail(name, e); } };

const text = 'hello world '.repeat(20);

// Sync round-trips
check('gzip roundtrip', () => {
  const out = zlib.gunzipSync(zlib.gzipSync(text)).toString('utf8');
  return out === text;
});
check('deflate roundtrip', () => {
  const out = zlib.inflateSync(zlib.deflateSync(text)).toString('utf8');
  return out === text;
});
check('deflateRaw roundtrip', () => {
  const out = zlib.inflateRawSync(zlib.deflateRawSync(text)).toString('utf8');
  return out === text;
});
check('unzip auto-detect gzip', () => {
  const gz = zlib.gzipSync(text);
  return zlib.unzipSync(gz).toString('utf8') === text;
});
check('unzip auto-detect zlib', () => {
  const def = zlib.deflateSync(text);
  return zlib.unzipSync(def).toString('utf8') === text;
});

// Size reduction (sanity that it's actually compressing)
check('gzip reduces size', () => zlib.gzipSync(text).length < text.length);
check('compression level 9', () => {
  const small = zlib.gzipSync(text, { level: 9 });
  const big = zlib.gzipSync(text, { level: 0 });
  return big.length > small.length;
});

// Async callback form
check('async gzip callback', () => {
  let result;
  zlib.gzip(text, (err, buf) => { result = err ? err : zlib.gunzipSync(buf).toString('utf8'); });
  return result === text;
});

// Promises form
check('promises form is async', () => zlib.promises.gzip(text).then ? 'has .then' : 'no');

// CRC32
check('crc32',           () => zlib.crc32(Buffer.from('123456789')));  // expected: 0xCBF43926 = 3421780262

// Stream stubs throw clearly
check('createGzip throws clearly', () => {
  try { zlib.createGzip(); return 'NO THROW'; }
  catch (e) { return e.message.includes('not implemented') ? 'throws-clearly' : 'wrong-message'; }
});

// Constants
check('constants.Z_DEFAULT_COMPRESSION', () => zlib.constants.Z_DEFAULT_COMPRESSION);
check('exported Z_BEST_COMPRESSION',     () => zlib.Z_BEST_COMPRESSION);

return log.join('\n');
})()
