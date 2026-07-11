// Headless end-to-end test of the Coil TodoMVC: a minimal DOM mock + simulated user
// actions (type/Enter, toggle, filter, clear, delete), asserting the Coil app logic.
import { readFileSync } from 'node:fs';
import { makeBridge } from './js-bridge.js';

class El {
  constructor(tag) {
    this.tag = tag; this._text = ''; this._value = ''; this.children = []; this.attrs = {}; this._on = {};
    this._cls = new Set(); this.parent = null;
    const self = this;
    this.classList = {
      add: (c) => self._cls.add(c), remove: (c) => self._cls.delete(c),
      toggle: (c) => (self._cls.has(c) ? self._cls.delete(c) : self._cls.add(c)),
      contains: (c) => self._cls.has(c),
    };
  }
  get className() { return [...this._cls].join(' '); }
  set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); }
  get value() { return this._value; }
  set value(v) { this._value = String(v); }
  appendChild(c) { this.children.push(c); c.parent = this; return c; }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); return c; }
  remove() { if (this.parent) this.parent.removeChild(this); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  addEventListener(t, fn) { (this._on[t] ||= []).push(fn); }
  dispatch(t, ev = {}) { for (const fn of (this._on[t] || []).slice()) fn({ type: t, target: this, ...ev }); }
}
const document = { body: new El('body'), createElement: (t) => new El(t) };
globalThis.document = document;

// tree helpers
const walk = (n, f) => { f(n); for (const c of n.children) walk(c, f); };
const all = (pred) => { const r = []; walk(document.body, (n) => pred(n) && r.push(n)); return r; };
const one = (pred) => all(pred).find(Boolean);
const byClass = (c) => all((n) => n._cls.has(c));

const bridge = makeBridge();
const { instance } = await WebAssembly.instantiate(readFileSync(new URL('./todomvc.wasm', import.meta.url)), { env: bridge.env });
bridge.bind(instance);
instance.exports.main();

const input = one((n) => n._cls.has('new-todo'));
const list = one((n) => n._cls.has('todo-list'));
const counter = one((n) => n.tag === 'strong');
const items = () => list.children;
const labelOf = (li) => li.children.find((c) => c.tag === 'label').textContent;

let ok = true;
const check = (label, got, want) => { const p = String(got) === String(want); if (!p) ok = false; console.log(`${p ? 'ok  ' : 'FAIL'}  ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`); };

const addTodo = (text) => { input.value = text; input.dispatch('keydown', { keyCode: 13 }); };

// 1. add two todos
addTodo('  Buy milk  ');            // leading/trailing space → trimmed by Coil
addTodo('Write a compiler');
check('two items added', items().length, 2);
check('first label trimmed', labelOf(items()[0]), 'Buy milk');
check('count = 2', counter.textContent, 2);

// 2. blank input is ignored
addTodo('   ');
check('blank ignored', items().length, 2);

// 3. toggle first complete (change on its checkbox)
const firstToggle = items()[0].children.find((c) => c.tag === 'input');
firstToggle.dispatch('change');
check('first has completed class', items()[0]._cls.has('completed'), true);
check('count = 1 after toggle', counter.textContent, 1);

// 4. filter Active → ul className flips (CSS hides the rest)
const activeLink = byClass('filters')[0].children.map((li) => li.children[0]).find((a) => a.textContent === 'Active');
activeLink.dispatch('click');
check('ul filtered-active', list._cls.has('filtered-active'), true);

// 5. clear completed → the toggled item is removed
one((n) => n._cls.has('clear-completed')).dispatch('click');
check('one item after clear', items().length, 1);
check('remaining label', labelOf(items()[0]), 'Write a compiler');
check('count = 1 after clear', counter.textContent, 1);

// 6. delete the last one (click its destroy button)
items()[0].children.find((c) => c.tag === 'button').dispatch('click');
check('empty after delete', items().length, 0);
check('count = 0', counter.textContent, 0);

// 7. slot reuse: adding again works after deletes
addTodo('Third');
check('add after delete', items().length, 1);
check('third label', labelOf(items()[0]), 'Third');

console.log(ok ? '\nPASS — TodoMVC runs entirely in Coil over the generic bridge' : '\nFAILED');
process.exit(ok ? 0 : 1);
