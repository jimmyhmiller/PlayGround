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

// ---- meta_run_wasm: run a metaprogram side-module IN the sandbox ------------
// The compiler (metaengine.coil, wasm meta path) compiled the metaprogram to a wasm
// SIDE-MODULE (codegen_wasm --wasm-side-module) whose bytes live in the compiler's
// linear memory. It hands us those bytes, the coil_mp_<k> entry symbol, and up to 8
// boxed Sexp arg pointers. We instantiate the side-module SHARING the compiler's
// memory: its static data is placed at a fresh __memory_base region we carve from the
// compiler heap, its shadow stack in another, and every `env.mh_*` import is bridged
// to the COMPILER instance's exported mh_* — so the metaprogram's callbacks build
// Sexps in the same memory the compiler reads. No pthread, no dlopen: synchronous.
//
// Parse the side-module's single active data segment length so __memory_base gets a
// region exactly large enough for the module's static data (blob written at base+0).
function sideDataSize(bytes) {
  let p = 8, max = 0;                              // skip magic+version
  const uleb = () => { let x = 0n, s = 0n, b; do { b = bytes[p++]; x |= BigInt(b & 0x7f) << s; s += 7n; } while (b & 0x80); return Number(x); };
  while (p < bytes.length) {
    const id = bytes[p++]; const size = uleb(); const end = p + size;
    if (id === 11) {                               // data section
      const count = uleb();
      for (let i = 0; i < count; i++) {
        const flags = uleb();
        if ((flags & 1) === 0) { if (flags & 2) uleb(); while (bytes[p++] !== 0x0b); } // active: skip memidx?, offset expr to `end`
        const n = uleb(); p += n;                  // segment payload
        if (n > max) max = n;
      }
    }
    p = end;
  }
  return max;
}

let compilerExports = null;                        // set after instantiate
function meta_run_wasm(bytesPtr, len, symPtr, argc, ...args) {
  const modBytes = Uint8Array.prototype.slice.call(u8(), Number(bytesPtr), Number(bytesPtr) + Number(len));
  const sym = cstr(symPtr);
  const mod = new WebAssembly.Module(modBytes);

  // carve regions from the compiler heap for the side-module's data + shadow stack
  const dataSize = Math.max(sideDataSize(modBytes), 64);
  const memBase = malloc(BigInt(dataSize));
  const STACK = 1n << 23n;                          // 8 MiB shadow stack
  const stackLo = malloc(STACK);
  const stackTop = stackLo + STACK;

  // build the side-module's env from ITS declared imports
  const sideEnv = {
    memory: mem(),
    __memory_base: new WebAssembly.Global({ value: 'i64', mutable: false }, memBase),
    __stack_pointer: new WebAssembly.Global({ value: 'i64', mutable: true }, stackTop),
  };
  for (const imp of WebAssembly.Module.imports(mod)) {
    if (imp.module !== 'env' || imp.name in sideEnv) continue;
    if (imp.name.startsWith('mh_')) {
      const f = compilerExports[imp.name];
      if (typeof f !== 'function') throw new Error(`meta_run_wasm: compiler does not export ${imp.name}`);
      sideEnv[imp.name] = f;                        // bridge callback → compiler instance
    } else if (imp.name in env) {
      sideEnv[imp.name] = env[imp.name];             // reuse host libc (malloc/free/mem*)
    } else {
      sideEnv[imp.name] = trap(`side:${imp.name}`);  // loud on anything unexpected
    }
  }

  const side = new WebAssembly.Instance(mod, { env: sideEnv });
  const entry = side.exports[sym];
  if (typeof entry !== 'function') throw new Error(`meta_run_wasm: side-module has no export ${sym}`);
  const callArgs = args.slice(0, Number(argc)).map((x) => BigInt(x));
  const ret = entry(...callArgs);                    // synchronous; returns Sexp ptr (i64)
  if (process.env.COIL_WASM_META_TRACE) console.error(`[meta_run_wasm] ran ${sym}(argc=${argc}) → ${ret}`);
  return BigInt(ret);
}

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
  snprintf,
  // strtod MUST set *endptr = first-unparsed byte: the reader computes the consumed
  // length as (*endptr - nptr) and rejects the token as a symbol if it isn't the full
  // number. The old stub left endptr untouched, so every float literal (e.g. 0.0 in
  // fmt.coil) resolved as an unbound variable.
  strtod:(nptr,endptr)=>{
    const s=cstr(nptr);
    const m=s.match(/^[ \t\n\r]*[+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)/i);
    let v=0, consumed=0;
    if(m){ const p=parseFloat(m[0]); if(!isNaN(p)) v=p; consumed=Buffer.byteLength(m[0],'utf8'); }
    if(endptr && Number(endptr)!==0) dv().setBigUint64(Number(endptr), BigInt(Number(nptr)+consumed), true);
    return v;
  },
  // process
  abort:()=>{ throw new Error('env.abort() called'); }, exit:(c)=>{ throw new ExitSignal(Number(c)); },
  // threads — init/lock are noops (single-threaded); create spawns → WALL1
  pthread_mutex_init:()=>0, pthread_mutex_lock:()=>0, pthread_mutex_unlock:()=>0,
  pthread_cond_init:()=>0, pthread_cond_signal:()=>0, pthread_cond_wait:()=>0,
  pthread_attr_init:()=>0, pthread_attr_setstacksize:()=>0, pthread_join:()=>0, pthread_exit:()=>0n,
  pthread_create: trap('pthread_create'),
  // Wall 1: comptime JIT / dylib / subprocess — the wasm meta path replaces these
  // with meta_run_wasm (run a metaprogram as a shared-memory side-module in-sandbox).
  meta_run_wasm,
  mmap: trap('mmap'), munmap: trap('munmap'), mprotect: trap('mprotect'),
  dlopen: trap('dlopen'), dlsym: trap('dlsym'), system: trap('system'),
};

class ExitSignal extends Error { constructor(code){ super('exit'); this.code = code; } }

const { instance: inst } = await WebAssembly.instantiate(bytes, { env });
instance = inst;
compilerExports = instance.exports;                          // meta_run_wasm bridges mh_* to these
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
