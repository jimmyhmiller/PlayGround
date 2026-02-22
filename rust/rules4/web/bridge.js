// Rules4 WASM Bridge — wraps raw exports into a friendly JS API

export class Rules4 {
  constructor(instance) {
    this.wasm = instance.exports;
    this.memory = this.wasm.memory;
    this._symCache = new Map(); // name -> termId
    this._nameCache = new Map(); // termId -> name (for sym terms)
  }

  static async load(wasmUrl) {
    const resp = await fetch(wasmUrl);
    const bytes = await resp.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const r4 = new Rules4(instance);
    r4.wasm.engine_new();
    return r4;
  }

  // ── String helpers ──

  _writeString(str) {
    const encoded = new TextEncoder().encode(str);
    const ptr = this.wasm.alloc_string(encoded.length);
    const buf = new Uint8Array(this.memory.buffer, ptr, encoded.length);
    buf.set(encoded);
    return [ptr, encoded.length];
  }

  _readString(ptr, len) {
    const buf = new Uint8Array(this.memory.buffer, ptr, len);
    return new TextDecoder().decode(buf);
  }

  // ── Core API ──

  loadProgram(source) {
    const [ptr, len] = this._writeString(source);
    const termId = this.wasm.load_program(ptr, len);
    this.wasm.free_string(ptr, len);
    return termId;
  }

  eval(termId) {
    const result = this.wasm.eval(termId);
    if (this.wasm.eval_step_limit_exceeded && this.wasm.eval_step_limit_exceeded()) {
      throw new Error("Step limit exceeded (possible infinite loop or very deep recursion)");
    }
    return result;
  }

  // ── Term construction ──

  num(n) {
    return this.wasm.term_num_i32(n);
  }

  float(n) {
    return this.wasm.term_float(n);
  }

  sym(name) {
    if (this._symCache.has(name)) {
      return this._symCache.get(name);
    }
    const [ptr, len] = this._writeString(name);
    const id = this.wasm.term_sym(ptr, len);
    this.wasm.free_string(ptr, len);
    this._symCache.set(name, id);
    this._nameCache.set(id, name);
    return id;
  }

  call(head, args) {
    if (args.length === 0) {
      return this.wasm.term_call(head, 0, 0);
    }
    // Write args array to WASM memory
    const ptr = this.wasm.alloc_string(args.length * 4);
    const view = new Uint32Array(this.memory.buffer, ptr, args.length);
    for (let i = 0; i < args.length; i++) {
      view[i] = args[i];
    }
    const id = this.wasm.term_call(head, ptr, args.length);
    this.wasm.free_string(ptr, args.length * 4);
    return id;
  }

  // ── Term inspection ──

  termTag(id) {
    return this.wasm.term_tag(id); // 0=Num, 1=Sym, 2=Call, 3=Float
  }

  termNum(id) {
    return this.wasm.term_get_num_i32(id);
  }

  termFloat(id) {
    return this.wasm.term_get_float(id);
  }

  termSymName(id) {
    if (this._nameCache.has(id)) {
      return this._nameCache.get(id);
    }
    const ptr = this.wasm.term_get_sym_ptr(id);
    const len = this.wasm.term_get_sym_len(id);
    const name = this._readString(ptr, len);
    this._nameCache.set(id, name);
    return name;
  }

  termCallHead(id) {
    return this.wasm.term_call_head(id);
  }

  termCallArity(id) {
    return this.wasm.term_call_arity(id);
  }

  termCallArg(id, idx) {
    return this.wasm.term_call_arg(id, idx);
  }

  // ── Dynamic rules ──

  assertRule(lhs, rhs) {
    this.wasm.assert_rule(lhs, rhs);
  }

  retractRule(lhs) {
    this.wasm.retract_rule(lhs);
  }

  queryAll(tagId) {
    return this.wasm.query_all(tagId);
  }

  // ── Display ──

  display(id) {
    const ptr = this.wasm.display_term(id);
    const len = this.wasm.display_term_len();
    return this._readString(ptr, len);
  }

  // ── Generic scope pending buffer ──

  /**
   * Take all pending terms for a named scope and clear the buffer.
   * Returns an array of term IDs.
   */
  scopeTakePending(name) {
    const [ptr, len] = this._writeString(name);
    const count = this.wasm.scope_pending_count(ptr, len);
    const terms = [];
    for (let i = 0; i < count; i++) {
      terms.push(this.wasm.scope_pending_get(ptr, len, i));
    }
    this.wasm.scope_pending_clear(ptr, len);
    this.wasm.free_string(ptr, len);
    return terms;
  }

  // ── High-level helpers ──

  /**
   * Recursively convert a term to a JS value:
   * - Num -> number
   * - Sym "true"/"false" -> boolean
   * - Sym "nil" -> []
   * - cons(h, t) -> [h, ...t] (flattened)
   * - Sym -> string
   * - Call -> { head: string, args: [...] }
   */
  termToJS(id) {
    const tag = this.termTag(id);
    if (tag === 0) return this.termNum(id);
    if (tag === 3) return this.termFloat(id);
    if (tag === 1) {
      const name = this.termSymName(id);
      if (name === "true") return true;
      if (name === "false") return false;
      if (name === "nil") return [];
      return name;
    }
    // Call
    const headId = this.termCallHead(id);
    const headTag = this.termTag(headId);
    if (headTag === 1) {
      const headName = this.termSymName(headId);
      if (headName === "vec") {
        // vec(a, b, c) — flat list
        const arr = [];
        const arity = this.termCallArity(id);
        for (let i = 0; i < arity; i++) {
          arr.push(this.termToJS(this.termCallArg(id, i)));
        }
        return arr;
      }
      if (headName === "cons") {
        // Flatten cons list to array
        const arr = [];
        let cur = id;
        while (true) {
          const t = this.termTag(cur);
          if (t === 1 && this.termSymName(cur) === "nil") break;
          if (t === 2) {
            const h = this.termCallHead(cur);
            if (this.termTag(h) === 1 && this.termSymName(h) === "cons") {
              arr.push(this.termToJS(this.termCallArg(cur, 0)));
              cur = this.termCallArg(cur, 1);
              continue;
            }
          }
          // Not a proper cons cell
          arr.push(this.termToJS(cur));
          break;
        }
        return arr;
      }
    }
    // Generic call
    const arity = this.termCallArity(id);
    const args = [];
    for (let i = 0; i < arity; i++) {
      args.push(this.termToJS(this.termCallArg(id, i)));
    }
    return { head: this.termToJS(headId), args };
  }

  // ── Actor scope scheduling ──

  tick(budget = 1) {
    return this.wasm.tick(budget);
  }

  scopeQueueCount(name) {
    const [ptr, len] = this._writeString(name);
    const count = this.wasm.scope_queue_count(ptr, len);
    this.wasm.free_string(ptr, len);
    return count;
  }

  gc() {
    this.wasm.gc();
    // All TermIds are invalidated by compaction — clear caches
    this._symCache.clear();
    this._nameCache.clear();
  }

  termCount() {
    return this.wasm.term_count();
  }

  reset() {
    this.wasm.engine_reset();
    this._symCache.clear();
    this._nameCache.clear();
  }
}
