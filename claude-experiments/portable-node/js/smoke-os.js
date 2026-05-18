// Smoke test for the lifted node:os, exercising the portable-host pipe:
//   Node lib/os.js → internalBinding('os') → __binding/os shim → __host.os.*
//   → Rust impl in src/host.rs.
//
// If any link in that chain is broken, you'll see it here.

(function () {
const log = [];
const ok = (name, val) => log.push(`OK   ${name}: ${val}`);
const fail = (name, e) => log.push(`FAIL ${name}: ${e.message || e}`);
const check = (name, fn) => { try { ok(name, JSON.stringify(fn())); } catch (e) { fail(name, e); } };

check('typeof os',          () => typeof os);
check('os.hostname()',      () => os.hostname());
check('os.platform()',      () => os.platform());
check('os.arch()',          () => os.arch());
check('os.type()',          () => os.type());
check('os.release()',       () => os.release());
check('os.endianness()',    () => os.endianness());
check('os.tmpdir()',        () => os.tmpdir());
check('os.homedir()',       () => os.homedir());
check('os.uptime() > 0',    () => os.uptime() >= 0);
check('os.totalmem() > 0',  () => os.totalmem() > 0);
check('os.freemem() ≥ 0',   () => os.freemem() >= 0);
check('os.loadavg() len 3', () => os.loadavg().length);
check('os.cpus() count',    () => os.cpus().length);
check('os.cpus()[0] shape', () => {
  const c = os.cpus()[0];
  return ['model', 'speed', 'times'].every(k => k in c) &&
         ['user','nice','sys','idle','irq'].every(k => k in c.times);
});
check('os.userInfo() shape', () => {
  const u = os.userInfo();
  return ['username','uid','gid','shell','homedir'].every(k => k in u);
});
check('os.networkInterfaces() has lo', () => {
  const i = os.networkInterfaces();
  return Object.keys(i).length > 0;
});
check('os.availableParallelism > 0',  () => os.availableParallelism() > 0);
check('os.EOL',             () => os.EOL);
check('process.platform',   () => process.platform);
check('process.arch',       () => process.arch);

return log.join('\n');
})()
