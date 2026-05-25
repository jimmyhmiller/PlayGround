// Direct smoke test for __host.file.* — bypasses Node's fs.js entirely.
// Proves the host-level primitive surface works before lifting fs.js.

(function () {
const log = [];
const ok = (name, val) => log.push(`OK   ${name}: ${val}`);
const fail = (name, e) => log.push(`FAIL ${name}: ${e.code ? '[' + e.code + '] ' : ''}${e.message || e}`);
const check = (name, fn) => { try { ok(name, JSON.stringify(fn())); } catch (e) { fail(name, e); } };

const F = __host.file;
const flags = F.flags;
const tmp = __host.os.tmpdir().replace(/\/$/, '') + '/portable-node-fs-test';

// Try cleanup leftover.
try { F.unlink(tmp); } catch (_e) {}

check('flags.O_RDONLY exists', () => typeof flags.O_RDONLY);
check('flags.O_WRONLY|O_CREAT', () => flags.O_WRONLY | flags.O_CREAT);

// --- write then read a temp file ---
check('open write', () => {
  const fd = F.open(tmp, flags.O_WRONLY | flags.O_CREAT | flags.O_TRUNC, 0o644);
  return fd > 0;
});

const data = new Uint8Array([0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x21]); // "hello!"
check('write', () => {
  const fd = F.open(tmp, flags.O_WRONLY | flags.O_CREAT | flags.O_TRUNC, 0o644);
  const n = F.write(fd, data, 0, data.length, -1);
  F.close(fd);
  return n;
});

check('stat size', () => F.stat(tmp).size);
check('stat mode is file', () => {
  const m = F.stat(tmp).mode;
  // S_IFREG = 0o100000 — high bits set means regular file.
  return (m & 0o170000) === 0o100000;
});

check('read', () => {
  const fd = F.open(tmp, flags.O_RDONLY, 0);
  const buf = new Uint8Array(16);
  const n = F.read(fd, buf, 0, buf.length, -1);
  F.close(fd);
  // Decode the bytes we actually read.
  return String.fromCharCode(...buf.slice(0, n));
});

check('access F_OK', () => { F.access(tmp, flags.F_OK); return 'ok'; });
check('access readable',  () => { F.access(tmp, flags.R_OK); return 'ok'; });

check('rename', () => {
  const dest = tmp + '.renamed';
  try { F.unlink(dest); } catch (_e) {}
  F.rename(tmp, dest);
  const s = F.stat(dest);
  F.rename(dest, tmp); // restore for cleanup below
  return s.size;
});

check('unlink', () => { F.unlink(tmp); try { F.stat(tmp); return 'NOT REMOVED'; } catch (e) { return e.code; } });

// --- mkdir/rmdir ---
const dir = tmp + '-dir';
try { F.rmdir(dir); } catch (_e) {}
check('mkdir',     () => { F.mkdir(dir, 0o755); return F.stat(dir).mode & 0o170000; }); // S_IFDIR=0o040000
check('readdir empty', () => F.readdir(dir).length);
check('rmdir',     () => { F.rmdir(dir); try { F.stat(dir); return 'NOT REMOVED'; } catch (e) { return e.code; } });

// --- realpath ---
check('realpath /', () => F.realpath('/'));
check('realpath relative', () => F.realpath('.').length > 0);

// --- error semantics ---
check('open(missing) throws ENOENT', () => {
  try { F.open('/__does_not_exist__/portable-node', flags.O_RDONLY, 0); return 'NO THROW'; }
  catch (e) { return e.code; }
});

return log.join('\n');
})()
