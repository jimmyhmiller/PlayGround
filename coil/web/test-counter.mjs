// Headless end-to-end test of the Coil counter: builds a tiny DOM mock, runs the
// wasm module's main() to construct the page, then simulates button clicks and
// asserts the coil-side state drives the DOM. Run: node web/test-counter.mjs
import { readFileSync } from 'node:fs';
import { instantiateCoil } from './coil-runtime.js';

// --- minimal DOM mock (only what dom.coil touches) ---
class Node {
  constructor(tag) { this.tag = tag; this.textContent = ''; this.children = []; this.attrs = {}; this._on = {}; }
  appendChild(c) { this.children.push(c); return c; }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(evt, fn) { (this._on[evt] ||= []).push(fn); }
  click() { for (const fn of this._on.click || []) fn(); }
  byId(id) { if (this.attrs.id === id) return this; for (const c of this.children) { const r = c.byId?.(id); if (r) return r; } return null; }
}
const document = { body: new Node('body'), createElement: (t) => new Node(t), createTextNode: (t) => { const n = new Node('#text'); n.textContent = t; return n; }, getElementById: (id) => document.body.byId(id) };

const bytes = readFileSync(new URL('./counter.wasm', import.meta.url));
const instance = await instantiateCoil(bytes, { document });
instance.exports.main();

const display = document.body.byId('count');
const [inc, dec] = document.body.children.filter((n) => n.tag === 'button');

let pass = true;
const check = (label, got, want) => { const ok = got === want; if (!ok) pass = false; console.log(`${ok ? 'ok  ' : 'FAIL'}  ${label}: got ${got}, want ${want}`); };

check('initial count', display.textContent, '0');
inc.click();                     check('after + ', display.textContent, '1');
inc.click(); inc.click();        check('after +++', display.textContent, '3');
dec.click();                     check('after - ', display.textContent, '2');
check('button labels', inc.textContent + dec.textContent, '+-');
check('title', document.body.children[0].textContent, 'Coil counter');

console.log(pass ? '\nPASS — coil drives the DOM end to end' : '\nFAILED');
process.exit(pass ? 0 : 1);
