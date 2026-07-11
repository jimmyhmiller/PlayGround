# Coil on the web

Interactive web pages written in Coil, compiled to WebAssembly — the Coil analogue
of [Guile Hoot](https://spritely.institute/news/building-interactive-web-pages-with-guile-hoot.html).
JS calls into Coil (exported functions), Coil calls into JS (host imports for the
DOM). No per-widget glue: the page is Coil.

## Files

- `dom.coil` — a general DOM vocabulary. `extern` host imports (`env.dom_*`) plus
  `(slice u8)`-friendly wrappers (`create-element`, `set-text`, `on`, …).
- `coil-runtime.js` — instantiates a module and implements `env.dom_*` against a
  real `document`. Exposes `runCoil(url)` and `instantiateCoil(bytes, {document})`.
- `counter.coil` / `counter.html` — a worked example: a live counter.
- `test-counter.mjs` — headless end-to-end test (DOM mock + simulated clicks).

## Build & run

```sh
# build the wasm (uses the self-hosted compiler with the wasm backend)
coil build counter.coil --target wasm32-unknown-unknown -o counter.wasm

# serve over HTTP (file:// will not load wasm)
python3 -m http.server
# open http://localhost:8000/counter.html

# or run the headless test
node test-counter.mjs
```

## How the interop works

- **JS → Coil.** Mark handlers with `(export-c [on-inc :as "on_inc"])`; they become
  wasm exports. `main` is exported automatically and builds the page once.
- **Coil → JS.** A declared-but-undefined `extern … :cc c` becomes an `env.<name>`
  wasm import that `coil-runtime.js` implements. DOM nodes are passed as opaque
  integer handles; strings as `(ptr,len)` into linear memory (UTF-8).
- **State.** `alloc-static` gives Coil-side global cells; the compiler resolves the
  wasm GOT/`__memory_base` itself, so state persists across events with no JS help.

## Extending

Add a host op by declaring an `extern` in `dom.coil` (+ a wrapper) and implementing
the matching `env.*` function in `coil-runtime.js`. Keep the two in sync.
