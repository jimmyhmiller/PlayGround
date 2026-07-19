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
    this.stdin = new Uint8Array(0);  // pending terminal input, drained by the `read` import
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
      // Real C strtod, INCLUDING the endptr out-parameter. Callers use it to check the whole
      // string was consumed — String.toFloat computes `*endp - nptr` and rejects a partial
      // parse — so ignoring it silently breaks every numeric parse (std.json reported
      // "invalid number '17'" while native parsed it fine). char** is i32 on wasm32.
      strtod: (p, endp) => {
        const u = self.u8;
        const nptr = Number(p);
        let i = nptr;
        while (u[i] === 32 || (u[i] >= 9 && u[i] <= 13)) i++;          // leading space
        const start = i;
        if (u[i] === 43 || u[i] === 45) i++;                            // sign
        while (u[i] >= 48 && u[i] <= 57) i++;                           // integer digits
        if (u[i] === 46) { i++; while (u[i] >= 48 && u[i] <= 57) i++; } // fraction
        if (u[i] === 101 || u[i] === 69) {                              // exponent (only if valid)
          let j = i + 1;
          if (u[j] === 43 || u[j] === 45) j++;
          if (u[j] >= 48 && u[j] <= 57) { while (u[j] >= 48 && u[j] <= 57) j++; i = j; }
        }
        const v = i > start ? parseFloat(self.dec.decode(u.subarray(start, i))) : NaN;
        const ok = i > start && !Number.isNaN(v);
        if (Number(endp)) self.dv.setUint32(Number(endp), ok ? i : nptr, true);  // C: nptr on failure
        return ok ? v : 0;
      },

      // ---- host surface ----
      host_write: (fd, ptr, len) => { const t = rd(ptr, len); (Number(fd) === 2 ? self.opts.onStderr : self.opts.onStdout)?.(t); return Number(len) | 0; },
      write:      (fd, ptr, len) => { const t = rd(ptr, len); (Number(fd) === 2 ? self.opts.onStderr : self.opts.onStdout)?.(t); return Number(len) | 0; },  // isize
      host_now_ms: () => (self.opts.now ? self.opts.now() : Date.now()),

      // getenv: the demo supplies a fake API key so chooseBrain() picks the REAL
      // AnthropicModel path (buildBody -> Http.request -> parseAnthropic) instead of the
      // shortcut ScriptedModel. Everything still runs offline — host_http answers.
      getenv: (p) => {
        const name = cstr(p);
        const v = (self.opts.env || {})[name];
        return v === undefined ? 0 : self.writeStr(v);
      },

      // host_http: the fake API server. It PARSES the request JSON the agent built and
      // RESPONDS with a real Anthropic-shaped JSON body, which the agent then parses with
      // std.json — so buildBody, Json.parse, Json.stringify, the content-block walk and the
      // tool-use protocol all execute for real, with no network.
      host_http: (mp, ml, up, ul, bp, bl, outStatus) => {
        const method = rd(mp, ml), url = rd(up, ul), body = rd(bp, bl);
        let status = 200, respBody;
        try {
          respBody = JSON.stringify(self.fakeApi(JSON.parse(body), { method, url }));
        } catch (e) {
          status = 400;
          respBody = JSON.stringify({ type: "error", error: { type: "invalid_request_error", message: String(e) } });
        }
        self.dv.setBigInt64(Number(outStatus), BigInt(status), true);
        return self.writeStr(respBody);
      },

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

      // ---- stdin: a non-blocking line buffer fed by the terminal ----
      // vm-readline polls fd 0 then read()s a byte at a time until '\n'. A browser can't
      // block waiting for a keystroke (the JS event loop would never run to deliver it), so
      // we make readLine NON-BLOCKING instead of blocking: poll always reports "ready", and
      // read returns the next buffered byte, or 0 (EOF) when the buffer is drained — which
      // makes Console.readLine() return None rather than spin. The host feeds a line with
      // feedLine() and then drives the next turn through the normal eval op.
      poll: (_fds, _nfds, _timeout) => 1,
      read: (_fd, ptr, n) => {
        const want = Number(n);
        if (!self.stdin.length || want <= 0) return 0;      // drained -> EOF -> readLine None
        const take = Math.min(want, self.stdin.length);
        self.u8.set(self.stdin.subarray(0, take), Number(ptr));
        self.stdin = self.stdin.subarray(take);
        return take;                                         // isize = i32 on wasm32
      },

      // ---- genuinely unavailable: trap loudly if the demo ever reaches them ----
      ...Object.fromEntries(["socket","bind","listen","accept","setsockopt","close",
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

  // ---- the fake Anthropic-compatible API ----
  // Receives the PARSED request the agent built (model, system, messages[], tools[]) and
  // returns a response object shaped exactly like /v1/messages: {stop_reason, content:[…]}
  // with `text` and `tool_use` blocks. The VM serialises the request and parses this reply
  // with std.json, so the real protocol path is exercised end to end — offline, and
  // deterministic enough to demo. Override with `opts.api` to script a different brain.
  fakeApi(req, meta) {
    if (this.opts.api) return this.opts.api(req, meta);
    const msgs = Array.isArray(req.messages) ? req.messages : [];
    const last = msgs[msgs.length - 1];
    const toolNames = new Set((req.tools || []).map((t) => t.name));

    // Flatten a content field (a bare string, or an array of blocks) to plain text.
    const textOf = (c) => typeof c === "string" ? c
      : Array.isArray(c) ? c.map((b) => b?.type === "text" ? (b.text || "")
                                      : b?.type === "tool_result" ? (typeof b.content === "string" ? b.content : "")
                                      : "").join(" ")
      : "";
    const isToolResult = (m) => Array.isArray(m?.content) && m.content.some((b) => b?.type === "tool_result");
    const reply = (text) => ({ id: "msg_fake", type: "message", role: "assistant",
                               model: req.model || "fake-1", stop_reason: "end_turn",
                               content: [{ type: "text", text }] });
    const useTool = (name, input) => ({ id: "msg_fake", type: "message", role: "assistant",
                               model: req.model || "fake-1", stop_reason: "tool_use",
                               content: [{ type: "tool_use", id: "call_" + (++this._callSeq || (this._callSeq = 1)),
                                           name, input }] });

    // A tool already answered -> fold its output into a final reply.
    if (isToolResult(last)) return reply("Here you go: " + textOf(last.content).trim());

    const q = textOf(last?.content).trim();
    const lower = q.toLowerCase();

    // arithmetic -> the calculate tool (mirrors what a real model would decide)
    const m = lower.match(/(-?\d+)\s*(?:\*|x|times)\s*(-?\d+)/) || lower.match(/(-?\d+)\s*(\+|plus|-|minus|\/|over)\s*(-?\d+)/);
    if (m && toolNames.has("calculate")) {
      const a = parseInt(m[1], 10), b = parseInt(m[m.length - 1], 10);
      const op = /\*|x|times/.test(m[0]) ? "mul" : /\+|plus/.test(m[0]) ? "add"
               : /\/|over/.test(m[0]) ? "div" : "sub";
      return useTool("calculate", { a, b, op });
    }
    const w = lower.match(/weather (?:in|for) ([a-z ]+)/);
    if (w && toolNames.has("get_weather")) return useTool("get_weather", { location: w[1].trim() });
    if (/^(list files|ls)\b/.test(lower) && toolNames.has("shell")) return useTool("shell", { cmd: "ls -la" });
    if (/^run /.test(lower) && toolNames.has("shell")) return useTool("shell", { cmd: q.slice(4) });
    if (/^read /.test(lower) && toolNames.has("read_file")) return useTool("read_file", { path: q.slice(5) });

    if (/^(hi|hello|hey)\b/.test(lower)) return reply("Hi there - I'm a fake model served by the page. Ask me to calculate something, or try 'weather in Tokyo'.");
    if (lower.includes("thank")) return reply("You're welcome!");
    if (lower.includes("?")) return reply("Good question - " + q);
    return reply("You said: " + q);
  }

  // ---- the viewer wire op ----
  // boot(path): load+typecheck+compile+run a program already seeded into the VFS.
  // 0 = ok, 74 = can't open, 65 = typecheck failed, 70 = no main().
  boot(path) {
    this.booted = Number(this.exports.scry_boot(this.writeStr(path)));
    return this.booted;
  }

  // eval(src) -> parsed {value}|{error}. Uncrashable: a hard eval panic (stale ref,
  // arena-OOM, bad opcode, internal compiler invariant) longjmps, which on wasm means the
  // `longjmp` import throws and unwinds the wasm frames. We restore the shadow-stack
  // pointer (native's longjmp would have restored it for us — without this every panic
  // leaks the frames between eval-core and the panic site) and let scry_eval_recover()
  // finish the response, so the caller always gets typed JSON and the instance stays live.
  eval(src) { return JSON.parse(this.evalRaw(src)); }

  evalRaw(src) {
    // Evaluating with no live program (boot failed / never ran) has no program table to
    // resolve against and faults inside the VM. Refuse it here with the same typed-error
    // shape the caller already handles, rather than taking the instance down.
    if (this.booted !== 0) {
      const why = this.booted === undefined ? "no program has been booted"
                : `the program failed to load (scry_boot rc=${this.booted})`;
      return JSON.stringify({ error: { kind: "NoProgram", message: `cannot eval: ${why}` } });
    }
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

  // feedLine(text): queue a line of terminal input for Console.readLine(). Appends a
  // newline so readLine() sees a complete line.
  feedLine(text) {
    const add = this.enc.encode(text.endsWith("\n") ? text : text + "\n");
    const merged = new Uint8Array(this.stdin.length + add.length);
    merged.set(this.stdin); merged.set(add, this.stdin.length);
    this.stdin = merged;
  }

  // tick(): advance the green-thread scheduler one pass. Returns how many spawned threads
  // are still runnable. On wasm there are no OS threads — background workers only make
  // progress when the host ticks, so a page must pump this (startScheduler).
  tick() { return Number(this.exports.scry_tick()); }

  // Pump the scheduler on a timer. onProgress fires only when work actually advanced, so a
  // page can refresh its views without polling in the idle case.
  startScheduler(intervalMs = 30, onProgress) {
    if (this._sched) return;
    this._sched = setInterval(() => {
      let runnable = 0;
      try { runnable = this.tick(); } catch (e) { console.error("scry scheduler:", e); this.stopScheduler(); return; }
      if (runnable > 0) onProgress?.(runnable);
    }, intervalMs);
  }
  stopScheduler() { if (this._sched) { clearInterval(this._sched); this._sched = null; } }

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
