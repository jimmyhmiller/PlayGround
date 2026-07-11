// js-bridge.js -- the ENTIRE, app-agnostic JS runtime for Coil-on-the-web.
//
// It exposes a handful of GENERIC JavaScript operations (get/set a property, call a
// method, make a string/number, register a callback, …). It knows nothing about the
// DOM, about TodoMVC, or about any particular program — every DOM/app concept is
// expressed in Coil on top of these primitives (see js.coil / dom.coil). This file
// never grows as your app grows.
//
// Every JS value that crosses into Coil is an opaque integer HANDLE (a "jsref") into
// `heap`. Coil holds only handles; conversions (string<->handle, number<->handle) are
// primitives. Callbacks are wasm exports invoked by name with a caller-supplied int.

export function makeBridge() {
  const heap = [null, undefined, globalThis]; // 0=null, 1=undefined, 2=globalThis
  const free = [];
  let mem, instance;
  const dec = new TextDecoder(), enc = new TextEncoder();

  const ref = (v) => { const i = free.length ? free.pop() : heap.length; heap[i] = v; return i; };
  const str = (p, l) => dec.decode(new Uint8Array(mem.buffer, p, l));

  const env = {
    js_global:    (p, l)        => ref(globalThis[str(p, l)]),
    js_get:       (o, p, l)     => ref(heap[o][str(p, l)]),
    js_get_index: (o, i)        => ref(heap[o][i]),
    js_set:       (o, p, l, v)  => (heap[o][str(p, l)] = heap[v], 0),
    js_call0:     (o, p, l)     => ref(heap[o][str(p, l)]()),
    js_call1:     (o, p, l, a)  => ref(heap[o][str(p, l)](heap[a])),
    js_call2:     (o, p, l, a, b) => ref(heap[o][str(p, l)](heap[a], heap[b])),
    js_str:       (p, l)        => ref(str(p, l)),
    js_read:      (r, p, cap)   => enc.encodeInto(String(heap[r]), new Uint8Array(mem.buffer, p, cap)).written,
    js_i32:       (n)           => ref(n),
    js_to_i32:    (r)           => (heap[r] | 0),
    js_cb: (p, l, userdata) => {
      const name = str(p, l);
      return ref((event) => {
        const e = event === undefined ? 1 : ref(event);   // give the handler a jsref for the event
        try { instance.exports[name](userdata, e); }
        finally { heap[e] = undefined; free.push(e); }     // release the per-event handle
      });
    },
    js_release:   (r)           => (heap[r] = undefined, free.push(r), 0),
  };

  return {
    env,
    bind(inst) { instance = inst; mem = inst.exports.memory; },
  };
}

// Convenience: fetch/instantiate a Coil module wired to this bridge, then call main().
export async function runCoil(source, { document: _doc } = {}) {
  const bridge = makeBridge();
  const bytes = typeof source === 'string' ? await (await fetch(source)).arrayBuffer() : source;
  const { instance } = await WebAssembly.instantiate(bytes, { env: bridge.env });
  bridge.bind(instance);
  if (typeof instance.exports.main === 'function') instance.exports.main();
  return instance;
}
