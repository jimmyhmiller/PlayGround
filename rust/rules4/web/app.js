// app.js — Rules4 TodoMVC event loop
// Architecture: rules4 program writes DOM terms to @dom scope.
// JS reads the pending buffer, renders the last emitted tree, and diffs.

import { Rules4 } from './bridge.js';
import { renderTerm, patch } from './vdom.js';

export class App {
  constructor(r4, rootEl) {
    this.r4 = r4;
    this.rootEl = rootEl;
    this.currentTree = null;
    this.currentTermId = null;
    this.renderSym = null;
    this.handleEventSym = null;
    this.inputEventSym = null;
    this.keydownSym = null;
    this._raf = null;
    this._tick = false;
  }

  static async create(wasmUrl, rootEl, programSource, opts = {}) {
    const r4 = await Rules4.load(wasmUrl);
    const app = new App(r4, rootEl);
    if (opts.tick) app._tick = true;
    await app.init(programSource);
    return app;
  }

  async init(programSource) {
    const r4 = this.r4;

    // Load the program (which sets up rules and executes init)
    const initExpr = r4.loadProgram(programSource);
    r4.eval(initExpr);

    // Cache commonly used symbols
    this.renderSym = r4.sym("render");
    this.handleEventSym = r4.sym("handle_event");
    this.inputEventSym = r4.sym("input_event");
    this.keydownSym = r4.sym("keydown");

    // Initial render
    this.render();

    // Start game loop if requested
    if (this._tick) {
      this.startLoop();
    }
  }

  startLoop() {
    let frame = 0;
    const tickSym = this.r4.sym("tick");
    const tick = () => {
      try {
        frame++;
        const frameTerm = this.r4.num(frame);
        const call = this.r4.call(this.handleEventSym, [tickSym, frameTerm]);
        this.r4.eval(call);
        this.render();
      } catch (e) {
        console.error("[rules4] tick error frame=" + frame + ":", e);
      }
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  stopLoop() {
    if (this._raf != null) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  }

  sendEvent(eventSym, payload) {
    const call = this.r4.call(this.handleEventSym, [eventSym, payload]);
    this.r4.eval(call);
    this.render();
  }

  render() {
    const r4 = this.r4;

    // Call render() — this emits DOM term to @dom scope
    const renderCall = r4.call(this.renderSym, []);
    r4.eval(renderCall);

    // Read the pending DOM terms from the @dom buffer
    const pending = r4.scopeTakePending("dom");
    if (pending.length === 0) return;

    // Use the last emitted term as the DOM tree
    const newTermId = pending[pending.length - 1];

    if (this.currentTree === null) {
      // First render
      const dom = renderTerm(r4, newTermId, (handler, ev) => this.handleEvent(handler, ev));
      this.rootEl.innerHTML = '';
      this.rootEl.appendChild(dom);
      this.currentTree = dom;
    } else {
      // Diff and patch
      this.currentTree = patch(
        r4,
        this.currentTermId,
        newTermId,
        this.currentTree,
        this.rootEl,
        (handler, ev) => this.handleEvent(handler, ev)
      );
    }

    this.currentTermId = newTermId;
  }

  handleEvent(handlerTermId, domEvent) {
    const r4 = this.r4;

    // Build event term based on the DOM event type
    let eventTerm;
    if (domEvent.type === "keydown") {
      const keySym = r4.sym(domEvent.key);
      eventTerm = r4.call(this.keydownSym, [keySym]);
    } else if (domEvent.type === "input") {
      const val = domEvent.target.value || "";
      const valSym = r4.sym(val);
      eventTerm = r4.call(this.inputEventSym, [valSym]);
    } else {
      // Generic event — just pass the type as a symbol
      const typeSym = r4.sym(domEvent.type);
      eventTerm = typeSym;
    }

    // Call handle_event(handler, event_term)
    const call = r4.call(this.handleEventSym, [handlerTermId, eventTerm]);
    r4.eval(call);

    // Re-render (which reads from @dom scope)
    this.render();
  }
}
