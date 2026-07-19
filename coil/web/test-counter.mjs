// Headless end-to-end test of the Coil counter on the generic bridge.
import { readFileSync } from 'node:fs';
import { makeBridge } from './js-bridge.js';

class El {
  constructor(tag) { this.tag = tag; this.textContent = ''; this.children = []; this.attrs = {}; this._on = {}; }
  appendChild(c) { this.children.push(c); return c; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  addEventListener(t, fn) { (this._on[t] ||= []).push(fn); }
  dispatch(t, ev = {}) { for (const fn of this._on[t] || []) fn({ type: t, target: this, ...ev }); }
  byId(id) { if (this.attrs.id === id) return this; for (const c of this.children) { const r = c.byId?.(id); if (r) return r; } return null; }
}
const document = { body: new El('body'), createElement: (t) => new El(t) };
globalThis.document = document;

const bridge = makeBridge();
const { instance } = await WebAssembly.instantiate(readFileSync(new URL('./counter.wasm', import.meta.url)), { env: bridge.env });
bridge.bind(instance);
instance.exports.main();

const display = document.body.byId('count');
const [inc, dec] = document.body.children.filter((n) => n.tag === 'button');

let ok = true;
const check = (label, got, want) => { const p = String(got) === String(want); if (!p) ok = false; console.log(`${p ? 'ok  ' : 'FAIL'}  ${label}: ${got}`); };

check('initial', display.textContent, 0);
inc.dispatch('click'); check('after +', display.textContent, 1);
inc.dispatch('click'); inc.dispatch('click'); check('after +++', display.textContent, 3);
dec.dispatch('click'); check('after -', display.textContent, 2);

console.log(ok ? '\nPASS' : '\nFAILED');
process.exit(ok ? 0 : 1);
