# Coil on the web

Interactive web pages written in Coil, compiled to WebAssembly — the Coil analogue
of [Guile Hoot](https://spritely.institute/news/building-interactive-web-pages-with-guile-hoot.html).
**The DOM vocabulary and all app logic live in Coil.** The only JavaScript is one
fixed, app-agnostic bridge that never grows.

## The idea: one generic bridge, everything else in Coil

`js-bridge.js` exposes a handful of *generic* JavaScript operations — get/set a
property, call a method, make a string/number, register a callback — and knows
nothing about the DOM or any app. Every JS value that crosses into Coil is a wasm
**`externref`**: the runtime holds the real JS value directly (GC-managed), so there
is no handle table for the values flowing through get/set/call. A transient (the
object a getter returns, the string a method produces) is just a wasm local and is
collected automatically when the Coil function returns — **nothing to free, no leak**.

The one exception is state a program keeps *across* turns (a DOM node it will mutate
on a later event): `externref` can't live in linear memory, so those few values are
`retain`ed to a small table and referred to by an `i32` index the Coil model stores
(`unretain` on delete). The table holds only what you deliberately persist.

On top of these primitives:

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

## Ref lifetime

Transient JS values are `externref`s held in wasm locals and GC-managed — they never
accumulate. `test-todomvc.mjs` proves it: after 200 add/toggle/delete cycles (each
hundreds of get/set/call operations) the retain table stays flat at 3 (list + counter
+ one live todo). Only values the Coil model persists are `retain`ed, and they are
`unretain`ed when the todo is deleted.
