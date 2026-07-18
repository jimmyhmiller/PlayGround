// scry-wasm.js — the browser/node runtime bridge for the Scry VM compiled to wasm32.
//
// The wasm module DEFINES and EXPORTS its own linear memory. This bridge supplies the
// `env.*` imports the module needs: a libc shim (malloc/realloc/free/mem*/str*/snprintf)
// implemented in JS over that exported memory, plus the host surface (terminal I/O, clock,
// VFS, mock shell, env). No Coil-native allocator — JS owns the C heap above __heap_base.
//
// Usage:
//   const scry = await ScryWasm.instantiate(wasmBytes, { onStdout, onStderr, vfs, now });
//   scry.exports.<fn>(...)               // call exported wasm functions
//   scry.malloc(n) / scry.writeStr(s)    // helpers to marshal args into linear memory
//
// Status: libc + memory management validated in node against a purpose-built Coil module.
// snprintf reads the wasm variadic ABI (a pointer to the promoted-arg buffer passed as the
// hidden trailing arg); its full validation is gated on the C0 shadow-stack compiler change.

export class ScryExit extends Error {
  constructor(code) { super(`scry exit(${code})`); this.code = code; }
}

// Thrown by the `longjmp` import to unwind wasm frames back to the eval() call, standing in
// for the native setjmp/longjmp landing pad. Never escapes eval().
export class ScryLongjmp extends Error {
  constructor() { super("scry: longjmp (eval panic) — recovered by the host"); }
}

export class ScryWasm {
  static async instantiate(bytes, opts = {}) {
    const s = new ScryWasm(opts);
    const module = await WebAssembly.compile(bytes);
    s.instance = await WebAssembly.instantiate(module, { env: s.env });
    s.mem = s.instance.exports.memory;
    // Heap starts at the module's __heap_base if exported, else just above static data
    // (the initial memory size already covers all static data + the shadow stack).
    const hb = s.instance.exports.__heap_base;
    s.heapTop = (hb && hb.value) ? (hb.value | 0) : s.mem.buffer.byteLength;
    s.heapTop = (s.heapTop + 15) & ~15;
    return s;
  }

  constructor(opts) {
    this.opts = opts;
    this.instance = null;
    this.mem = null;
    this.heapTop = 0;
    this.freeList = [];              // { addr, size } blocks available for reuse
    this.dec = new TextDecoder();
    this.enc = new TextEncoder();
    // in-memory VFS: path -> Uint8Array. Seed with opts.vfs ({ "path": "text"|Uint8Array }).
    this.vfs = new Map();
    for (const [p, v] of Object.entries(opts.vfs || {})) {
      this.vfs.set(p, typeof v === "string" ? this.enc.encode(v) : v);
    }
    this.files = new Map();          // FILE* handle -> { bytes, pos, path, writable }
    this.nextFd = 3;
    this.env = this._buildEnv();
  }

  // ---- linear-memory views (re-fetched: the buffer detaches on memory.grow) ----
  get u8()  { return new Uint8Array(this.mem.buffer); }
  get dv()  { return new DataView(this.mem.buffer); }

  _grow(toByte) {
    const have = this.mem.buffer.byteLength;
    if (toByte > have) this.mem.grow(Math.ceil((toByte - have) / 65536));
  }

  // ---- libc allocator: first-fit free-list over the module's own linear memory ----
  _malloc(n) {
    n = Number(n);
    if (n <= 0) return 0;
    n = (n + 15) & ~15;
    for (let i = 0; i < this.freeList.length; i++) {
      if (this.freeList[i].size >= n) {
        const b = this.freeList.splice(i, 1)[0];
        return b.addr;                 // (whole block; no split — fine for a demo heap)
      }
    }
    const addr = this.heapTop;
    this.heapTop += n;
    this._grow(this.heapTop);
    this._sizes.set(addr, n);
    return addr;
  }
  _free(p) {
    p = Number(p);
    if (!p) return;
    const size = this._sizes.get(p) || 16;
    this.freeList.push({ addr: p, size });
  }
  _realloc(p, n) {
    p = Number(p); n = Number(n);
    if (!p) return this._malloc(n);
    if (n === 0) { this._free(p); return 0; }
    const old = this._sizes.get(p) || 0;
    if (old >= n) return p;
    const q = this._malloc(n);
    this.u8.copyWithin(q, p, p + old);
    this._free(p);
    return q;
  }

  _buildEnv() {
    const self = this;
    this._sizes = new Map();          // addr -> allocated size (for free/realloc)
    const rd = (p, l) => self.dec.decode(new Uint8Array(self.mem.buffer, Number(p), Number(l)));
    const cstr = (p) => { const u = self.u8; let e = Number(p); while (u[e]) e++; return self.dec.decode(u.subarray(Number(p), e)); };

    return {
      // ---- memory ----
      malloc:  (n) => self._malloc(n),
      calloc:  (n, sz) => { const total = Number(n) * Number(sz); const p = self._malloc(total);
                            self.u8.fill(0, p, p + total); return p; },
      realloc: (p, n) => self._realloc(p, n),
      free:    (p) => self._free(p),
      memcpy:  (d, s, n) => { self.u8.copyWithin(Number(d), Number(s), Number(s) + Number(n)); return Number(d); },
      memset:  (d, c, n) => { self.u8.fill(Number(c) & 0xff, Number(d), Number(d) + Number(n)); return Number(d); },
      memmove: (d, s, n) => { self.u8.copyWithin(Number(d), Number(s), Number(s) + Number(n)); return Number(d); },

      // ---- strings ----
      strlen: (p) => { const u = self.u8; let e = Number(p); while (u[e]) e++; return (e - Number(p)) | 0; },  // isize = i32 on wasm32
      strcmp: (a, b) => { const u = self.u8; let i = Number(a), j = Number(b); while (u[i] && u[i] === u[j]) { i++; j++; } return (u[i] - u[j]) | 0; },
      strncmp:(a, b, n) => { const u = self.u8; let i = Number(a), j = Number(b); n = Number(n); while (n-- && u[i] && u[i] === u[j]) { i++; j++; } return n < 0 ? 0 : ((u[i] - u[j]) | 0); },
      atoi:   (p) => { const v = parseInt(cstr(p).trim(), 10); return (Number.isNaN(v) ? 0 : v) | 0; },
      strtod: (p, endp) => { const s = cstr(p); const v = parseFloat(s); return Number.isNaN(v) ? 0 : v; },

      // ---- host surface ----
      host_write: (fd, ptr, len) => { const t = rd(ptr, len); (Number(fd) === 2 ? self.opts.onStderr : self.opts.onStdout)?.(t); return Number(len) | 0; },
      write:      (fd, ptr, len) => { const t = rd(ptr, len); (Number(fd) === 2 ? self.opts.onStderr : self.opts.onStdout)?.(t); return Number(len) | 0; },  // isize
      host_now_ms: () => (self.opts.now ? self.opts.now() : Date.now()),
      getenv: (_p) => 0,             // no env in the browser → fake model auto-selected

      // ---- snprintf (wasm variadic ABI: hidden trailing arg = ptr to promoted args) ----
      snprintf: (dst, cap, fmt, va) => self._snprintf(Number(dst), Number(cap), Number(fmt), Number(va)),

      // ---- in-memory VFS: fopen/fread/fseek/ftell/rewind/fclose over this.vfs ----
      fopen: (pathP, modeP) => {
        const path = cstr(pathP), mode = cstr(modeP);
        const writing = /[wa]/.test(mode);
        let bytes = self.vfs.get(path) || self.vfs.get(path.replace(/^\.\//, ""));
        if (!bytes && !writing) return 0;               // not found (read) → NULL
        if (!bytes) { bytes = new Uint8Array(0); self.vfs.set(path, bytes); }
        const fd = self.nextFd++;
        self.files.set(fd, { path, pos: mode.includes("a") ? bytes.length : 0, writable: writing });
        return fd;
      },
      fread: (ptr, size, nmemb, fd) => {
        const f = self.files.get(Number(fd)); if (!f) return 0n;
        const bytes = self.vfs.get(f.path); const want = Number(size) * Number(nmemb);
        const avail = Math.min(want, bytes.length - f.pos);
        self.u8.set(bytes.subarray(f.pos, f.pos + avail), Number(ptr));
        f.pos += avail;
        return BigInt(Number(size) ? Math.floor(avail / Number(size)) : 0);
      },
      fwrite: (ptr, size, nmemb, fd) => {
        const f = self.files.get(Number(fd)); if (!f) return 0n;
        const n = Number(size) * Number(nmemb);
        const chunk = self.u8.slice(Number(ptr), Number(ptr) + n);
        const old = self.vfs.get(f.path) || new Uint8Array(0);
        const merged = new Uint8Array(Math.max(old.length, f.pos + n));
        merged.set(old); merged.set(chunk, f.pos); f.pos += n;
        self.vfs.set(f.path, merged);
        return BigInt(Number(size) ? nmemb : 0);
      },
      fseek: (fd, off, whence) => {
        const f = self.files.get(Number(fd)); if (!f) return -1;
        const len = (self.vfs.get(f.path) || []).length;
        off = Number(off); whence = Number(whence);
        f.pos = whence === 2 ? len + off : whence === 1 ? f.pos + off : off;
        return 0;
      },
      ftell: (fd) => { const f = self.files.get(Number(fd)); return BigInt(f ? f.pos : -1); },
      rewind: (fd) => { const f = self.files.get(Number(fd)); if (f) f.pos = 0; },
      fclose: (fd) => { self.files.delete(Number(fd)); return 0; },
      fileno: (fd) => Number(fd),
      // stat: report existence + size. struct stat is opaque here; return 0=ok / -1=missing.
      stat: (pathP, _statbuf) => (self.vfs.has(cstr(pathP)) ? 0 : -1),
      realpath: (pathP, outP) => {                        // no symlinks: copy path through
        const s = cstr(pathP); const b = self.enc.encode(s);
        if (Number(outP)) { self.u8.set(b, Number(outP)); self.u8[Number(outP) + b.length] = 0; return Number(outP); }
        return Number(pathP);
      },
      opendir: () => 0, readdir: () => 0, closedir: () => 0,   // directory scan unused in the demo

      // ---- clock / misc ----
      gettimeofday: (tvP, _tz) => {                       // struct timeval { i64 sec; i64 usec; }
        const ms = self.opts.now ? self.opts.now() : Date.now();
        if (Number(tvP)) { self.dv.setBigInt64(Number(tvP), BigInt(Math.floor(ms/1000)), true);
                           self.dv.setBigInt64(Number(tvP)+8, BigInt((ms%1000)*1000), true); }
        return 0;
      },
      nanosleep: () => 0, usleep: () => 0, getpid: () => 1,

      // ---- non-local exit (the uncrashable-eval invariant on wasm) ----
      // wasm32 has no setjmp/longjmp, so the HOST unwinds: setjmp always returns 0 (the
      // "try" path), and longjmp throws a sentinel whose JS exception unwinds the wasm
      // frames. eval() below catches it, restores the shadow stack, and calls
      // scry_eval_recover() to emit the same {"error":{…}} the native landing pad would.
      setjmp: (_buf) => 0,
      longjmp: (_buf, _v) => { throw new ScryLongjmp(); },
      abort: () => { throw new Error("scry: abort()"); },
      exit: (code) => { throw new ScryExit(Number(code)); },

      // ---- curl: benign no-ops. The demo never makes an HTTP request (the fake
      // ScriptedModel is selected because getenv returns null), but http.coil's
      // init path is still reachable until the when-wasm guard (docs §5b) lands.
      ...Object.fromEntries(["curl_global_init","curl_easy_init","curl_easy_perform","curl_easy_setopt",
        "curl_easy_getinfo","curl_easy_cleanup","curl_easy_strerror","curl_slist_append","curl_slist_free_all",
        "curl_multi_init","curl_multi_add_handle","curl_multi_remove_handle","curl_multi_perform",
        "curl_multi_poll","curl_multi_info_read","curl_multi_cleanup"].map(n => [n, () => 0])),

      // read: no stdin in the browser. Returns 0 (EOF) until readLine is rewired to the
      // terminal via a host callback that pumps the scheduler (docs §4).
      read: (_fd, _p, _n) => 0,

      // ---- genuinely unavailable: trap loudly if the demo ever reaches them ----
      ...Object.fromEntries(["socket","bind","listen","accept","setsockopt","poll","close",
        "popen","pclose","pthread_create","pthread_join","pthread_detach"]
        .map(n => [n, (...a) => { throw new Error(`scry: ${n} unavailable in wasm (should be guarded/mocked)`); }])),
    };
  }

  // minimal printf covering the specifiers scry uses: %lld %d %s %g %f %c %% and
  // star width/precision (%.*s is how scry prints non-NUL-terminated slices — each `*`
  // consumes an int arg BEFORE the value, so it must be read in order).
  _snprintf(dst, cap, fmtPtr, vaPtr) {
    const u = this.u8, dv = this.dv;
    const cstr = (p) => { let e = p; while (u[e]) e++; return this.dec.decode(u.subarray(p, e)); };
    const cstrN = (p, n) => this.dec.decode(u.subarray(p, p + n));   // bounded (no NUL needed)
    const fmt = cstr(fmtPtr);
    let out = "", va = vaPtr;
    const i32 = () => { const v = dv.getInt32(va, true); va += 4; return v; };
    const i64 = () => { const v = dv.getBigInt64(va, true); va += 8; return v; };
    const f64 = () => { va = (va + 7) & ~7; const v = dv.getFloat64(va, true); va += 8; return v; };
    for (let i = 0; i < fmt.length; i++) {
      if (fmt[i] !== '%') { out += fmt[i]; continue; }
      // consume a (possibly-lengthed, possibly star-precision) conversion
      let spec = "%", j = i + 1, prec = -1;
      while (j < fmt.length && "-+ #0123456789.lh*".includes(fmt[j])) {
        // a `*` takes its value from the arg list, in order, before the conversion's value
        if (fmt[j] === '*') prec = i32();
        spec += fmt[j]; j++;
      }
      const conv = fmt[j]; spec += conv; i = j;
      if (conv === '%') { out += '%'; }
      else if (conv === 'd' || conv === 'i') { out += String(spec.includes('ll') ? i64() : i32()); }
      else if (conv === 'u') { out += String((spec.includes('ll') ? i64() : BigInt(i32() >>> 0))); }
      else if (conv === 'c') { out += String.fromCharCode(i32() & 0xff); }
      else if (conv === 's') {
        const p = i32();
        // %.Ns / %.*s — bounded read, so a non-NUL-terminated slice prints correctly
        let n = prec;
        if (n < 0) { const m = /\.(\d+)/.exec(spec); if (m) n = +m[1]; }
        out += n >= 0 ? cstrN(p, n) : cstr(p);
      }
      else if (conv === 'g' || conv === 'f' || conv === 'e') { out += String(f64()); }
      else { out += spec; }
    }
    const bytes = this.enc.encode(out);
    const n = Math.min(bytes.length, cap - 1);
    if (cap > 0) { u.set(bytes.subarray(0, n), dst); u[dst + n] = 0; }
    return bytes.length;             // C snprintf returns would-be length
  }

  // ---- the viewer wire op ----
  // boot(path): load+typecheck+compile+run a program already seeded into the VFS.
  boot(path) { return Number(this.exports.scry_boot(this.writeStr(path))); }

  // eval(src) -> parsed {value}|{error}. Uncrashable: a hard eval panic (stale ref,
  // arena-OOM, bad opcode, internal compiler invariant) longjmps, which on wasm means the
  // `longjmp` import throws and unwinds the wasm frames. We restore the shadow-stack
  // pointer (native's longjmp would have restored it for us — without this every panic
  // leaks the frames between eval-core and the panic site) and let scry_eval_recover()
  // finish the response, so the caller always gets typed JSON and the instance stays live.
  eval(src) { return JSON.parse(this.evalRaw(src)); }

  evalRaw(src) {
    const bytes = this.enc.encode(src);
    // +1 and NUL-terminate: the lexer reads past `len` (it expects a terminated buffer).
    // Without this it runs into whatever follows — harmless while the allocator is handing
    // out fresh zeroed pages, but it reads neighbouring garbage as soon as blocks are
    // recycled, producing bogus "unknown identifier" errors from unrelated source text.
    // server.coil documents the same hazard on the native path.
    const p = this._malloc(bytes.length + 1);
    this.u8.set(bytes, p);
    this.u8[p + bytes.length] = 0;
    const sp = this.instance.exports.__stack_pointer;   // exported mutable global
    const savedSp = sp ? sp.value : null;
    try {
      return this.readCstr(Number(this.exports.scry_eval(p, BigInt(bytes.length))));
    } catch (e) {
      if (!(e instanceof ScryLongjmp)) throw e;
      if (sp) sp.value = savedSp;                        // unwind the shadow stack
      else this._spLeaked = (this._spLeaked || 0) + 1;   // no export → frames leak; see README
      return this.readCstr(Number(this.exports.scry_eval_recover()));
    } finally {
      this._free(p);
    }
  }

  // ---- marshalling helpers for callers ----
  readCstr(p) { const u = this.u8; let e = p; while (u[e]) e++; return this.dec.decode(u.subarray(p, e)); }

  writeStr(s) {                       // JS string -> NUL-terminated cstring in linear memory
    const bytes = this.enc.encode(s);
    const p = this._malloc(bytes.length + 1);
    this.u8.set(bytes, p); this.u8[p + bytes.length] = 0;
    return p;
  }
  readStr(p, len) { return this.dec.decode(new Uint8Array(this.mem.buffer, Number(p), Number(len))); }
  get exports() { return this.instance.exports; }
}
