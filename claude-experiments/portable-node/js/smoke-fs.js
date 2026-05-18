// Smoke test for the user-facing node:fs sync API.
//
// Pipe under test:
//   user JS → fs.readFileSync(...) → __binding/fs → __host.file.* → libc
//
// Everything from `__binding/fs` upward is portable; only __host.file.*
// changes per host.

(function () {
const log = [];
const ok = (name, val) => log.push(`OK   ${name}: ${val}`);
const fail = (name, e) => log.push(`FAIL ${name}: ${e.code ? '[' + e.code + '] ' : ''}${e.message || e}`);
const check = (name, fn) => { try { ok(name, JSON.stringify(fn())); } catch (e) { fail(name, e); } };

const tmp = os.tmpdir().replace(/\/$/, '') + '/portable-node-fs-api-test';
try { fs.unlinkSync(tmp); } catch (_e) {}

check('typeof fs',           () => typeof fs);
check('typeof readFileSync', () => typeof fs.readFileSync);
check('typeof writeFileSync',() => typeof fs.writeFileSync);

check('writeFileSync(string)', () => { fs.writeFileSync(tmp, 'hello world'); return fs.statSync(tmp).size; });
check('readFileSync(utf8)',    () => fs.readFileSync(tmp, 'utf8'));
check('readFileSync(buffer)',  () => {
  const b = fs.readFileSync(tmp);
  return Buffer.isBuffer(b) ? b.toString('utf8') : 'NOT BUFFER';
});

check('writeFileSync(Buffer)', () => {
  fs.writeFileSync(tmp, Buffer.from([0x66, 0x6f, 0x6f])); // "foo"
  return fs.readFileSync(tmp, 'utf8');
});

check('appendFileSync',        () => {
  fs.appendFileSync(tmp, 'bar');
  return fs.readFileSync(tmp, 'utf8');
});

check('statSync isFile',       () => fs.statSync(tmp).isFile());
check('statSync isDir false',  () => fs.statSync(tmp).isDirectory());

check('existsSync(true)',  () => fs.existsSync(tmp));
check('existsSync(false)', () => fs.existsSync('/__nope__/never'));

// Directory ops.
const dir = tmp + '-dir';
try { fs.rmdirSync(dir); } catch (_e) {}
check('mkdirSync',         () => { fs.mkdirSync(dir); return fs.statSync(dir).isDirectory(); });
check('readdirSync (empty)', () => fs.readdirSync(dir).length);

// Multi-file readdir.
fs.writeFileSync(dir + '/a.txt', 'A');
fs.writeFileSync(dir + '/b.txt', 'B');
check('readdirSync (2 files)', () => fs.readdirSync(dir).sort().join(','));
check('readdirSync withFileTypes', () => {
  const ents = fs.readdirSync(dir, { withFileTypes: true });
  return ents.every(e => e.isFile());
});

// rename + cleanup.
const tmp2 = tmp + '.renamed';
try { fs.unlinkSync(tmp2); } catch (_e) {}
check('renameSync', () => { fs.renameSync(tmp, tmp2); return fs.existsSync(tmp2); });
fs.unlinkSync(dir + '/a.txt');
fs.unlinkSync(dir + '/b.txt');
fs.rmdirSync(dir);
fs.unlinkSync(tmp2);

check('readFileSync(missing) throws ENOENT', () => {
  try { fs.readFileSync('/__nope__/never'); return 'NO THROW'; }
  catch (e) { return e.code; }
});

// Promise wrappers — defer to sync.
const p = fs.promises.readFile(__host.process.cwd() + '/Cargo.toml', 'utf8');
check('fs.promises.readFile is thenable', () => typeof p.then);

return log.join('\n');
})()
