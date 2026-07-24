// Host harness to RUN the wasm64 Coil compiler (coilc.wasm) in Node.
// Provides the 44 env.* imports: bump allocator over the module's own linear
// memory (starting at exported __heap_base), real-filesystem I/O, and LOUD TRAPS
// for the Wall-1 comptime imports (mmap/dlopen/... ) so a comptime-needing compile
// fails with a clear message instead of silent garbage.
//
// Usage:  node --experimental-wasm-memory64 run-coil-wasm.mjs <wasm> <coil-args...>
//   e.g.  node ... run-coil-wasm.mjs /tmp/coilc.wasm check /tmp/wtest.coil
import fs from 'node:fs';
import { execSync } from 'node:child_process';

const [wasmPath, ...coilArgs] = process.argv.slice(2);
if (!wasmPath) { console.error('usage: run-coil-wasm.mjs <wasm> <coil-args...>'); process.exit(2); }

const bytes = fs.readFileSync(wasmPath);
let instance = null;                              // set after instantiate; closures capture it
const mem   = () => instance.exports.memory;
const buf   = () => mem().buffer;
const u8    = () => new Uint8Array(buf());
const dv    = () => new DataView(buf());
const PAGE  = 65536;

// ---- reclaiming allocator over linear memory --------------------------------
// A plain bump allocator LEAKS catastrophically here: the in-process bytecode
// interpreter mallocs a 1 MiB frame buffer (+ operand stack) PER vm-exec call and
// frees it on return, so with thousands of macro expansions a no-op free exhausts
// memory. Reclaim by a per-size free list: sizes are rounded up so the interpreter's
// repeated fixed-size frame allocations reuse the same blocks, bounding memory to the
// peak concurrent live set rather than the cumulative churn.
let heap = 0n;                                    // next free offset (BigInt)
const sizes = new Map();                          // live ptr -> rounded size
const freeBins = new Map();                       // size(string) -> [ptr,…] reusable
const alignUp = (v, a) => (v + (a - 1n)) & ~(a - 1n);
function ensure(end) {                            // grow memory to cover byte `end`
  const have = BigInt(buf().byteLength);
  if (end <= have) return;
  const need = (end - have + BigInt(PAGE) - 1n) / BigInt(PAGE);
  mem().grow(need);                                // memory64: grow takes a BigInt page count
}
function malloc(size) {
  size = BigInt(size);
  if (size <= 0n) size = 1n;
  size = alignUp(size, 16n);                       // bin by 16-byte-rounded size
  const bin = freeBins.get(size.toString());
  if (bin && bin.length) { const p = bin.pop(); sizes.set(p, size); return p; }
  const p = alignUp(heap, 16n);
  heap = p + size;
  ensure(heap);
  sizes.set(p, size);
  return p;                                       // BigInt (i64)
}
function hostFree(ptr) {
  ptr = BigInt(ptr);
  if (ptr === 0n) return;
  const s = sizes.get(ptr);
  if (s === undefined) return;                     // unknown/double free: ignore
  sizes.delete(ptr);
  const k = s.toString();
  let bin = freeBins.get(k);
  if (!bin) { bin = []; freeBins.set(k, bin); }
  bin.push(ptr);
}
function realloc(ptr, size) {
  ptr = BigInt(ptr); size = BigInt(size);
  if (ptr === 0n) return malloc(size);
  const old = sizes.get(ptr) ?? 0n;
  if (old >= alignUp(size <= 0n ? 1n : size, 16n)) return ptr;   // fits in place
  const np = malloc(size);
  const n = Number(old < size ? old : size);
  u8().copyWithin(Number(np), Number(ptr), Number(ptr) + n);
  hostFree(ptr);
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

// instantiate a metaprogram/comptime side-module sharing the compiler's memory: its
// static data lands in a fresh __memory_base region, its shadow stack in another, and
// every env.mh_* / libc import is bridged to the compiler instance (or host libc).
function instantiateSide(bytesPtr, len) {
  const modBytes = Uint8Array.prototype.slice.call(u8(), Number(bytesPtr), Number(bytesPtr) + Number(len));
  const mod = new WebAssembly.Module(modBytes);
  const dataSize = Math.max(sideDataSize(modBytes), 64);
  const memBase = malloc(BigInt(dataSize));
  const STACK = 1n << 23n;                          // 8 MiB shadow stack
  const stackTop = malloc(STACK) + STACK;
  const sideEnv = {
    memory: mem(),
    __memory_base: new WebAssembly.Global({ value: 'i64', mutable: false }, memBase),
    __stack_pointer: new WebAssembly.Global({ value: 'i64', mutable: true }, stackTop),
  };
  for (const imp of WebAssembly.Module.imports(mod)) {
    if (imp.module !== 'env' || imp.name in sideEnv) continue;
    if (imp.name.startsWith('mh_')) {
      const f = compilerExports[imp.name];
      if (typeof f !== 'function') throw new Error(`side-module: compiler does not export ${imp.name}`);
      sideEnv[imp.name] = f;                        // bridge callback → compiler instance
    } else if (imp.name in env) {
      sideEnv[imp.name] = env[imp.name];             // reuse host libc (malloc/free/mem*)
    } else {
      sideEnv[imp.name] = trap(`side:${imp.name}`);  // loud on anything unexpected
    }
  }
  return new WebAssembly.Instance(mod, { env: sideEnv });
}

function meta_run_wasm(bytesPtr, len, symPtr, argc, ...args) {
  const sym = cstr(symPtr);
  const side = instantiateSide(bytesPtr, len);
  const entry = side.exports[sym];
  if (typeof entry !== 'function') throw new Error(`meta_run_wasm: side-module has no export ${sym}`);
  const callArgs = args.slice(0, Number(argc)).map((x) => BigInt(x));
  let ret;
  try { ret = entry(...callArgs); }                  // synchronous; returns Sexp ptr (i64)
  catch (e) {
    if (e instanceof MetaHalt) {                     // (error …)/bad op: diag already recorded
      if (process.env.COIL_WASM_META_TRACE) console.error(`[meta_run_wasm] ${sym} halted (metaprogram error)`);
      return 0n;                                      // → compiler reads meta-host-err and reports it
    }
    throw e;
  }
  if (process.env.COIL_WASM_META_TRACE) console.error(`[meta_run_wasm] ran ${sym}(argc=${argc}) → ${ret}`);
  return BigInt(ret);
}

// run a COMPTIME fold side-module: coil_ct_thunk yields the folded value; write its
// bits into `cell` (or, for an aggregate, kind 0, the thunk writes through cell as a
// buf pointer). coil_ct_status (if present) reports div-by-zero (1=div, 2=rem). We
// return that status; the compiler discards the value and errors when it is nonzero.
function meta_run_ct(bytesPtr, len, thunkSymPtr, statusSymPtr, kind, cell) {
  const side = instantiateSide(bytesPtr, len);
  const thunkSym = cstr(thunkSymPtr);
  const thunk = side.exports[thunkSym];
  if (typeof thunk !== 'function') throw new Error(`meta_run_ct: side-module has no export ${thunkSym}`);
  kind = Number(kind);
  if (kind === 0) {
    thunk(BigInt(cell));                             // aggregate write-through into shared memory
  } else if (kind === 3 || kind === 4) {
    dv().setFloat64(Number(cell), Number(thunk()), true);
  } else {
    dv().setBigInt64(Number(cell), BigInt(thunk()), true);
  }
  let st = 0n;
  const statusSym = statusSymPtr && Number(statusSymPtr) !== 0 ? cstr(statusSymPtr) : null;
  if (statusSym) { const s = side.exports[statusSym]; if (typeof s === 'function') st = BigInt(s()); }
  if (process.env.COIL_WASM_META_TRACE) console.error(`[meta_run_ct] ran ${thunkSym}(kind=${kind}) status=${st}`);
  return st;
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

// minimal printf-family formatter over an argument list (BigInt cells): handles the
// integer/string/char/hex/pointer conversions a metaprogram would use. %f-family is
// approximated from the raw i64 bit-pattern reinterpreted as a double.
function fmtc(fmt, args) {
  let i = 0;
  return fmt.replace(/%l?l?[dioux%csfgpX]/g, (m) => {
    if (m === '%%') return '%';
    const a = args[i++] ?? 0n;
    const c = m[m.length - 1];
    if (c === 'd' || c === 'i') return String(BigInt.asIntN(64, BigInt(a)));
    if (c === 'u' || c === 'o') return String(BigInt.asUintN(64, BigInt(a)));
    if (c === 'x') return BigInt.asUintN(64, BigInt(a)).toString(16);
    if (c === 'X') return BigInt.asUintN(64, BigInt(a)).toString(16).toUpperCase();
    if (c === 'p') return '0x' + BigInt.asUintN(64, BigInt(a)).toString(16);
    if (c === 'c') return String.fromCharCode(Number(a) & 0xff);
    if (c === 's') return cstr(a);
    if (c === 'f' || c === 'g') { const dv2 = new DataView(new ArrayBuffer(8)); dv2.setBigUint64(0, BigInt.asUintN(64, BigInt(a)), true); return String(dv2.getFloat64(0, true)); }
    return m;
  });
}

const env = {
  // allocation
  malloc, realloc, free: (p)=>{ hostFree(p); return 0n; }, memset:(s,c,n)=>{u8().fill(Number(c)&0xff,Number(s),Number(s)+Number(n));return s;},
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
  // libc used by the in-process bytecode interpreter's FFI table (interp.coil). During
  // a compile these are imported but only called if a metaprogram/comptime thunk uses
  // them (macros build Code, folds compute — so mostly unused); implemented for real so
  // a metaprogram that does print/math behaves correctly.
  putchar:(c)=>{ process.stdout.write(Buffer.from([Number(c)&0xff])); return Number(c)&0xff; },
  putc:(c,_s)=>{ process.stdout.write(Buffer.from([Number(c)&0xff])); return Number(c)&0xff; },
  puts:(p)=>{ process.stdout.write(cstr(p)+'\n'); return 1; },
  printf:(fmtPtr,...a)=>{ const s=fmtc(cstr(fmtPtr),a); process.stdout.write(s); return s.length; },
  dprintf:(fd,fmtPtr,...a)=>{ const s=fmtc(cstr(fmtPtr),a); doWrite(Number(fd),0n,0n); (Number(fd)===2?process.stderr:process.stdout).write(s); return s.length; },
  calloc:(n,sz)=>{ const total=BigInt(n)*BigInt(sz); const p=malloc(total===0n?1n:total); u8().fill(0,Number(p),Number(p)+Number(total)); return p; },
  atoi:(p)=>{ const v=parseInt(cstr(p),10); return isNaN(v)?0:(v|0); },
  strcmp:(a,b)=>{ const sa=cstr(a),sb=cstr(b); return sa<sb?-1:(sa>sb?1:0); },
  strtol:(p,endptr,base)=>{ const v=parseInt(cstr(p),Number(base)||10); if(endptr&&Number(endptr)!==0){} return BigInt(isNaN(v)?0:Math.trunc(v)); },
  sqrt:(x)=>Math.sqrt(x), pow:(x,y)=>Math.pow(x,y),
  fmod:(x,y)=>x%y, fmodf:(x,y)=>Math.fround(Math.fround(x)%Math.fround(y)),
  // process
  abort:()=>{ throw new Error('env.abort() called'); }, exit:(c)=>{ throw new ExitSignal(Number(c)); },
  // threads — init/lock are noops (single-threaded); create spawns → WALL1
  pthread_mutex_init:()=>0, pthread_mutex_lock:()=>0, pthread_mutex_unlock:()=>0,
  pthread_cond_init:()=>0, pthread_cond_signal:()=>0, pthread_cond_wait:()=>0,
  pthread_attr_init:()=>0, pthread_attr_setstacksize:()=>0, pthread_join:()=>0,
  // metahost's mh-halt records the metaprogram diagnostic in the compiler's
  // MetaHostBox and then calls pthread_exit to end the (native) metaprogram thread.
  // In the sandbox there is no thread; THROW instead so the exception unwinds out of
  // the side-module (through the mh_* bridge and the metalowered (loop 0) it would
  // otherwise spin on) back into meta_run_wasm's try/catch, which returns null. The
  // compiler then reads the recorded Diag via meta-host-err and reports it located.
  pthread_exit:()=>{ throw new MetaHalt(); },
  pthread_create: trap('pthread_create'),
  // Wall 1: comptime JIT / dylib / subprocess — the wasm meta path replaces these
  // with meta_run_wasm (run a metaprogram as a shared-memory side-module in-sandbox).
  meta_run_wasm, meta_run_ct,
  // JIT/dylib in-sandbox execution stays a hard trap. `system` is different: the
  // compiler uses it only to invoke the host TOOLCHAIN (cc) to LINK the final object
  // it just emitted — a host build service, not sandboxed code execution, exactly like
  // the fs I/O this harness already provides. Run the command on the host and return
  // its exit status so `build` completes end to end.
  mmap: trap('mmap'), munmap: trap('munmap'), mprotect: trap('mprotect'),
  dlopen: trap('dlopen'), dlsym: trap('dlsym'),
  system: (cmdPtr) => {                              // returns i32 (a JS Number, not BigInt)
    const cmd = cstr(cmdPtr);
    try { execSync(cmd, { stdio: 'inherit' }); return 0; }
    catch (e) { return ((e && e.status ? e.status : 1) & 0xff) << 8; }  // wait()-style status
  },
};

class ExitSignal extends Error { constructor(code){ super('exit'); this.code = code; } }
class MetaHalt extends Error { constructor(){ super('meta-halt'); } }   // metaprogram (error …)

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
  if (process.env.COIL_WASM_STACK) console.error(e.stack);
  process.exit(70);
}
