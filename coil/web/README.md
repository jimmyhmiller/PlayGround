# Coil on the web

Interactive web pages written in Coil, compiled to WebAssembly — the Coil analogue
of [Guile Hoot](https://spritely.institute/news/building-interactive-web-pages-with-guile-hoot.html).
**The DOM vocabulary and all app logic live in Coil.** The only JavaScript is one
fixed, app-agnostic bridge that never grows.

## The idea: one generic bridge, everything else in Coil

`js-bridge.js` exposes ~11 *generic* JavaScript operations — get/set a property, call
a method, make a string/number, register a callback, release a handle — and knows
nothing about the DOM or any app. Every JS value that crosses into Coil is an opaque
integer **handle** (`jsref`). On top of these primitives:

- `js.coil` — a typed Coil layer over the primitives (`js-get`, `js-call1`, `js-str`, …).
- `dom.coil` — a DOM vocabulary written **entirely in Coil** (`create`, `append!`,
  `set-text!`, `add-class!`, `on!`, …). No DOM-specific JavaScript anywhere.

Add a new host capability by writing more Coil — the bridge stays the same.

## Files

| file | what |
|------|------|
| `js-bridge.js` | the entire JS runtime (generic, ~50 lines, never changes) |
| `js.coil` | typed Coil wrappers over the generic primitives |
| `dom.coil` | DOM vocabulary, pure Coil on `js.coil` |
| `counter.coil` / `counter.html` | minimal example |
| `todomvc.coil` / `todomvc.html` | full **TodoMVC**: Coil-side model + DOM view |
| `test-*.mjs` | headless end-to-end tests (DOM mock + simulated events) |

## Build & run

```sh
coil build todomvc.coil --target wasm32-unknown-unknown -o todomvc.wasm
python3 -m http.server            # file:// won't load wasm; serve over HTTP
# open http://localhost:8000/todomvc.html
node test-todomvc.mjs             # or run the headless test
```

## How TodoMVC works (all Coil)

- **Model**: a fixed-capacity `array` of todo structs in `alloc-static` storage — no
  malloc. `used`/`done` flags + the JS handles the slot owns.
- **View**: the real DOM, built incrementally through `dom.coil`. Deleting a todo
  releases its handles and tombstones the slot; the index is reused on the next add.
- **Events**: handlers are exported Coil functions; `(on! node "click" "on_x" data)`
  registers one, and `data` (the slot index) comes back as the handler's first arg.
- **Filtering**: Coil sets a class on the list; a couple of CSS rules hide the rest.
- **The count**: `set-text-ref! counter (js-num (active-count))` — Coil computes it,
  JS coerces the number to text.

## Handle lifetime (the one caveat)

`jsref` handles are freed manually (`js-release!`); the wrappers release obvious
temporaries and per-todo handles are released on delete, but transient results of
get/call chains still accumulate. The clean fix is wasm **`externref`** (GC-managed
JS references, no handle table) — a planned backend addition; the handle table is the
zero-new-features stepping stone.
