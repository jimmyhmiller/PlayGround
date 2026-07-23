// Host harness to RUN the wasm64 Coil compiler (coilc.wasm) in Node.
// Provides the 44 env.* imports: bump allocator over the module's own linear
// memory (starting at exported __heap_base), real-filesystem I/O, and LOUD TRAPS
// for the Wall-1 comptime imports (mmap/dlopen/... ) so a comptime-needing compile
// fails with a clear message instead of silent garbage.
//
// Usage:  node --experimental-wasm-memory64 run-coil-wasm.mjs <wasm> <coil-args...>
//   e.g.  node ... run-coil-wasm.mjs /tmp/coilc.wasm check /tmp/wtest.coil
import fs from 'node:fs';

const [wasmPath, ...coilArgs] = process.argv.slice(2);
if (!wasmPath) { console.error('usage: run-coil-wasm.mjs <wasm> <coil-args...>'); process.exit(2); }

const bytes = fs.readFileSync(wasmPath);
let instance = null;                              // set after instantiate; closures capture it
const mem   = () => instance.exports.memory;
const buf   = () => mem().buffer;
const u8    = () => new Uint8Array(buf());
const dv    = () => new DataView(buf());
const PAGE  = 65536;

// ---- bump allocator over linear memory -------------------------------------
let heap = 0n;                                    // next free offset (BigInt)
const sizes = new Map();                          // ptr -> size, for realloc
const alignUp = (v, a) => (v + (a - 1n)) & ~(a - 1n);
function ensure(end) {                            // grow memory to cover byte `end`
  const have = BigInt(buf().byteLength);
  if (end <= have) return;
  const need = (end - have + BigInt(PAGE) - 1n) / BigInt(PAGE);
  mem().grow(need);                                // memory64: grow takes a BigInt page count
}
function malloc(size) {
  size = BigInt(size);
  if (size === 0n) size = 1n;
  const p = alignUp(heap, 16n);
  heap = p + size;
  ensure(heap);
  sizes.set(p, size);
  return p;                                       // BigInt (i64)
}
function realloc(ptr, size) {
  ptr = BigInt(ptr); size = BigInt(size);
  if (ptr === 0n) return malloc(size);
  const old = sizes.get(ptr) ?? 0n;
  const np = malloc(size);
  const n = Number(old < size ? old : size);
  u8().copyWithin(Number(np), Number(ptr), Number(ptr) + n);
  return np;
}

// ---- string / memory helpers -----------------------------------------------
function cstr(ptr) {                              // read NUL-terminated string
  const m = u8(); let e = Number(ptr);
  while (m[e] !== 0) e++;
  return Buffer.from(m.slice(Number(ptr), e)).toString('utf8');
}
function writeBytes(ptr, data) { u8().set(data, Number(ptr)); }

// ---- file descriptors ------------------------------------------------------
// Node fds are small ints; use them directly. 0/1/2 are std streams.
function doOpen(pathPtr, flags) {
  const path = cstr(pathPtr);
  // O_RDONLY=0, O_WRONLY=1, O_RDWR=2, O_CREAT=0x200(mac), O_TRUNC=0x400(mac)
  const acc = flags & 3;
  let mode = acc === 0 ? 'r' : acc === 1 ? 'w' : 'r+';
  try { return fs.openSync(path, mode); }
  catch { return -1; }
}
function doRead(fd, ptr, len) {
  len = Number(len);
  const b = Buffer.alloc(len);
  let n;
  try { n = fs.readSync(fd, b, 0, len, null); } catch { return -1n; }
  writeBytes(ptr, b.subarray(0, n));
  return BigInt(n);
}
function doWrite(fd, ptr, len) {
  len = Number(len);
  const b = Buffer.from(u8().slice(Number(ptr), Number(ptr) + len));
  if (fd === 1) { process.stdout.write(b); return BigInt(len); }
  if (fd === 2) { process.stderr.write(b); return BigInt(len); }
  try { return BigInt(fs.writeSync(fd, b, 0, len)); } catch { return -1n; }
}

const WALL1 = new Set();
function trap(name) { return (...a) => { throw new Error(`WALL1: env.${name} — native execution (JIT/dlopen/subprocess) is not available in the wasm sandbox`); }; }

// ---- minimal snprintf(buf,size,fmt,arg) — one conversion -------------------
function snprintf(bufPtr, size, fmtPtr, arg) {
  const fmt = cstr(fmtPtr);
  let out = fmt.replace(/%l?l?[dizu]/,(m)=>String(BigInt.asIntN(64,BigInt(arg))))
               .replace(/%l?l?[xX]/,(m)=>BigInt.asUintN(64,BigInt(arg)).toString(16))
               .replace(/%s/,()=>cstr(arg))
               .replace(/%c/,()=>String.fromCharCode(Number(arg)&0xff))
               .replace(/%p/,()=>'0x'+BigInt.asUintN(64,BigInt(arg)).toString(16));
  const b = Buffer.from(out + '\0','utf8');
  const n = Math.min(b.length, Number(size));
  writeBytes(bufPtr, b.subarray(0, n));
  return b.length - 1;                            // chars that would have been written
}

const env = {
  // allocation
  malloc, realloc, free: (p)=>{}, memset:(s,c,n)=>{u8().fill(Number(c)&0xff,Number(s),Number(s)+Number(n));return s;},
  memcmp:(a,b,n)=>{const m=u8();a=Number(a);b=Number(b);n=Number(n);for(let i=0;i<n;i++){const d=m[a+i]-m[b+i];if(d)return BigInt(Math.sign(d));}return 0n;},
  // file io
  open: doOpen, read: doRead, write: doWrite, close:(fd)=>{ if(fd>2){try{fs.closeSync(fd);}catch{}} return 0; },
  creat:(p,mode)=>{ try{return BigInt(fs.openSync(cstr(p),'w'));}catch{return -1n;} },
  access:(p,m)=>{ try{fs.accessSync(cstr(p));return 0;}catch{return -1;} },
  unlink:(p)=>{ try{fs.unlinkSync(cstr(p));return 0;}catch{return -1;} },
  rename:(a,b)=>{ try{fs.renameSync(cstr(a),cstr(b));return 0;}catch{return -1;} },
  realpath:(p,out)=>{ try{const r=Buffer.from(fs.realpathSync(cstr(p))+'\0');writeBytes(out,r);return out;}catch{return 0n;} },
  fopen:(p,mode)=>{ try{return BigInt(fs.openSync(cstr(p), cstr(mode).includes('w')?'w':'r'));}catch{return 0n;} },
  fclose:(f)=>{ if(Number(f)>2){try{fs.closeSync(Number(f));}catch{}} return 0; },
  fwrite:(ptr,sz,nm,f)=>{ const n=Number(sz)*Number(nm); doWrite(Number(f),ptr,BigInt(n)); return BigInt(nm); },
  opendir:(p)=>0n, closedir:(d)=>0,
  getcwd:(b,sz)=>{ const r=Buffer.from(process.cwd()+'\0'); writeBytes(b,r); return b; },
  getenv:(n)=>0n, getpid:()=>Number(process.pid), realpath_stub:()=>0n,
  // string
  strlen:(p)=>{ const m=u8(); let e=Number(p); while(m[e]!==0)e++; return BigInt(e-Number(p)); },
  snprintf, strtod:(nptr,endptr)=>{ const s=cstr(nptr); const v=parseFloat(s); if(endptr){/*approx*/ } return isNaN(v)?0:v; },
  // process
  abort:()=>{ throw new Error('env.abort() called'); }, exit:(c)=>{ throw new ExitSignal(Number(c)); },
  // threads — init/lock are noops (single-threaded); create spawns → WALL1
  pthread_mutex_init:()=>0, pthread_mutex_lock:()=>0, pthread_mutex_unlock:()=>0,
  pthread_cond_init:()=>0, pthread_cond_signal:()=>0, pthread_cond_wait:()=>0,
  pthread_attr_init:()=>0, pthread_attr_setstacksize:()=>0, pthread_join:()=>0, pthread_exit:()=>0n,
  pthread_create: trap('pthread_create'),
  // Wall 1: comptime JIT / dylib / subprocess
  mmap: trap('mmap'), munmap: trap('munmap'), mprotect: trap('mprotect'),
  dlopen: trap('dlopen'), dlsym: trap('dlsym'), system: trap('system'),
};

class ExitSignal extends Error { constructor(code){ super('exit'); this.code = code; } }

const { instance: inst } = await WebAssembly.instantiate(bytes, { env });
instance = inst;
heap = BigInt(instance.exports.__heap_base.value);          // start allocating after static+stack

// ---- set up argv = ["coil", ...coilArgs] -----------------------------------
const argvStrings = ['coil', ...coilArgs];
const ptrs = argvStrings.map(s => { const b = Buffer.from(s + '\0'); const p = malloc(BigInt(b.length)); writeBytes(p, b); return p; });
const argvPtr = malloc(BigInt(ptrs.length * 8));
for (let i = 0; i < ptrs.length; i++) dv().setBigUint64(Number(argvPtr) + i * 8, ptrs[i], true);

// ---- run main --------------------------------------------------------------
try {
  const rc = instance.exports.main(argvStrings.length, argvPtr);
  console.error(`\n[wasm compiler main returned ${rc}]`);
  process.exit(Number(rc) & 0xff);
} catch (e) {
  if (e instanceof ExitSignal) { process.exit(e.code & 0xff); }
  console.error(`\n[wasm compiler trapped] ${e.message}`);
  process.exit(70);
}
