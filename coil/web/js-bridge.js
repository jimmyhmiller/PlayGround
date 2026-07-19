// js-bridge.js -- the entire, app-agnostic JS runtime for Coil-on-the-web.
//
// Every JS value that crosses to Coil is a wasm `externref`: the wasm runtime holds
// the real JS value directly (GC-managed), so there is NO handle table for the
// values flowing through get/set/call. Transient results (the object returned by a
// getter, the string from a method) are just wasm locals — when a Coil function
// returns, they become GC-eligible automatically. Nothing to free.
//
// The one exception is state a Coil program keeps ACROSS turns (e.g. a DOM node it
// will mutate on a later event). externref can't live in wasm linear memory, so such
// values are `js_retain`ed into a small table and referred to by an i32 index the
// Coil model can store; `js_unretain` releases them. This table holds only what you
// deliberately persist — never the transient flow.

export function makeBridge() {
  const retained = [];        // index -> persisted JS value
  const freeSlots = [];
  let mem, instance;
  const dec = new TextDecoder(), enc = new TextEncoder();
  const str = (p, l) => dec.decode(new Uint8Array(mem.buffer, p, l));

  const env = {
    // ---- generic JS operations; `o`, `a`, `b`, `v`, and returns are externref ----
    js_global:    (p, l)        => globalThis[str(p, l)],
    js_get:       (o, p, l)     => o[str(p, l)],
    js_get_index: (o, i)        => o[i],
    js_set:       (o, p, l, v)  => (o[str(p, l)] = v, 0),
    js_call0:     (o, p, l)     => o[str(p, l)](),
    js_call1:     (o, p, l, a)  => o[str(p, l)](a),
    js_call2:     (o, p, l, a, b) => o[str(p, l)](a, b),
    js_str:       (p, l)        => str(p, l),                 // Coil string -> JS string (externref)
    js_read:      (r, p, cap)   => enc.encodeInto(String(r), new Uint8Array(mem.buffer, p, cap)).written,
    js_i32:       (n)           => n,                          // number -> externref
    js_to_i32:    (r)           => (r | 0),                    // externref (number/bool) -> i32
    js_cb:        (p, l, userdata) => { const name = str(p, l); return (event) => instance.exports[name](userdata, event); },

    // ---- retain table: only for refs a Coil program persists across turns ----
    js_retain:    (v)           => { const i = freeSlots.length ? freeSlots.pop() : retained.length; retained[i] = v; return i; },
    js_deref:     (i)           => retained[i],
    js_unretain:  (i)           => (retained[i] = undefined, freeSlots.push(i), 0),
  };

  return {
    env,
    bind(inst) { instance = inst; mem = inst.exports.memory; },
    // for tests/introspection: how many refs are currently retained (persisted).
    // Transient values never appear here — they are wasm-GC'd externrefs.
    stats() { return { retained: retained.length - freeSlots.length }; },
  };
}

// Fetch/instantiate a Coil module wired to this bridge, then call main().
export async function runCoil(source) {
  const bridge = makeBridge();
  const bytes = typeof source === 'string' ? await (await fetch(source)).arrayBuffer() : source;
  const { instance } = await WebAssembly.instantiate(bytes, { env: bridge.env });
  bridge.bind(instance);
  if (typeof instance.exports.main === 'function') instance.exports.main();
  return instance;
}
