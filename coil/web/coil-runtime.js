// coil-runtime.js -- host runtime for Coil programs compiled to wasm32.
//
// Instantiates a Coil wasm module and supplies the `env.dom_*` imports that
// web/dom.coil declares, bridging them to a real DOM `document`. Coil refers to
// DOM nodes by opaque integer HANDLES that this runtime hands out; strings cross
// as (ptr,len) into the wasm linear memory and are decoded here as UTF-8.
//
// Usage (browser):
//   import { runCoil } from './coil-runtime.js'
//   await runCoil('./counter.wasm')            // fetches, instantiates, calls main()
//
// Usage (custom document, e.g. tests):
//   await runCoil(bytesOrUrl, { document: myDoc })
//
// The module is self-contained (the compiler resolves __memory_base/GOT itself),
// so the only imports are the dom_* functions below.

export async function instantiateCoil(source, opts = {}) {
  const doc =
    opts.document ||
    (typeof document !== 'undefined' ? document : null);
  if (!doc) throw new Error('coil-runtime: no document available (pass opts.document)');

  const dec = new TextDecoder();
  let mem;                         // set after instantiation
  let instance;
  const nodes = [null];            // handle 0 == null; handles index into this
  const put = (node) => (nodes.push(node), nodes.length - 1);
  const str = (ptr, len) => dec.decode(new Uint8Array(mem.buffer, ptr, len));

  const env = {
    dom_body:           ()          => put(doc.body),
    dom_create_element: (p, l)      => put(doc.createElement(str(p, l))),
    dom_create_text:    (p, l)      => put(doc.createTextNode(str(p, l))),
    dom_append:         (par, ch)   => { nodes[par].appendChild(nodes[ch]); },
    dom_set_text:       (n, p, l)   => { nodes[n].textContent = str(p, l); },
    dom_set_int:        (n, v)      => { nodes[n].textContent = String(v); },
    dom_set_attr:       (n, kp, kl, vp, vl) => { nodes[n].setAttribute(str(kp, kl), str(vp, vl)); },
    dom_on: (n, ep, el, hp, hl) => {
      const evt = str(ep, el), handler = str(hp, hl);
      nodes[n].addEventListener(evt, () => {
        const fn = instance.exports[handler];
        if (typeof fn !== 'function')
          throw new Error(`coil-runtime: event handler '${handler}' is not an exported wasm function`);
        fn();
      });
    },
    dom_get_by_id:      (p, l)      => put(doc.getElementById(str(p, l))),
    console_log:        (p, l)      => console.log(str(p, l)),
  };

  const bytes =
    typeof source === 'string'
      ? await (await fetch(source)).arrayBuffer()
      : source;

  ({ instance } = await WebAssembly.instantiate(bytes, { env }));
  mem = instance.exports.memory;
  return instance;
}

// Instantiate and run `main` (the page's one-shot setup). Returns the instance so
// callers can invoke exported functions directly if they want.
export async function runCoil(source, opts = {}) {
  const instance = await instantiateCoil(source, opts);
  if (typeof instance.exports.main === 'function') instance.exports.main();
  return instance;
}
