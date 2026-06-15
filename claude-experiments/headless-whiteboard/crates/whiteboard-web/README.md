# whiteboard-web

The web (HTML5 Canvas 2D) backend for `whiteboard-core`, plus a runnable browser
demo.

The crate lowers a backend-agnostic `RenderScene` (the exact same draw-command
list the tiny-skia and SVG backends consume) into a flat list of `CanvasOp`s in
the pure, host-tested `scene_to_ops` function, then — on `wasm32` only — replays
those ops onto a real `CanvasRenderingContext2d` via `wasm-bindgen`/`web-sys`.
Because every drawing decision lives in the pure layer, the whole translation is
exercised by `cargo test -p whiteboard-web` on the host.

## What the browser sees

Three `#[wasm_bindgen]` exports drive the demo:

| Export                         | Signature                                  | Purpose |
|--------------------------------|--------------------------------------------|---------|
| `sample_scene_json()`          | `() -> String`                             | Build a fixed demo scene with the headless `Editor` and return `editor.render()` serialized as JSON. Deterministic (no RNG, no clock). |
| `render_scene_json(canvas, json)` | `(HtmlCanvasElement, &str) -> Result<()>` | Parse a `RenderScene` from JSON and paint it onto `canvas`. |
| `WebBackend`                   | class                                      | Stateful backend with image support (`new(canvas)`, `setImage(id, img)`, `render(json)`). |

The demo page calls `render_scene_json(canvas, sample_scene_json())`, so the
browser paints precisely what the host tests assert on.

## Files in `www/`

| File         | Role |
|--------------|------|
| `index.html` | A page with a 540×320 `<canvas id="board">`; loads `main.js` as an ES module. |
| `main.js`    | ES-module glue: `import init, { sample_scene_json, render_scene_json } from "../pkg/whiteboard_web.js"`, `await init()`, grab the canvas, render. No drawing logic. |

`main.js` imports from `../pkg/whiteboard_web.js`, which is produced by the build
step below.

## Build the wasm module

Requires [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) (install with
`cargo install wasm-pack`, or via the official installer). From this crate
directory:

```sh
cd crates/whiteboard-web
wasm-pack build --target web --out-dir www/pkg
```

That emits `www/pkg/whiteboard_web.js` and `www/pkg/whiteboard_web_bg.wasm`
(plus a `package.json` and `.d.ts`). The `--target web` flag produces an ES
module with a default `init()` export — exactly what `main.js` imports.

> Note: `--out-dir www/pkg` places the artifact where `main.js`'s
> `../pkg/whiteboard_web.js` import (relative to `www/`) resolves. If you build
> to the default `pkg/` (crate root), serve from the crate root instead and
> adjust the import path, or copy `pkg/` into `www/`.

## Run it

`<canvas>` + ES modules + wasm `fetch` require an HTTP origin (opening
`index.html` over `file://` will not load the module). Serve the `www/`
directory with any static server, e.g.:

```sh
cd crates/whiteboard-web/www
python3 -m http.server 8080
# then open http://localhost:8080/
```

You should see a filled rounded rectangle, an ellipse, an arrow, and a text
label painted on the canvas, with a status line reporting the byte size of the
rendered `RenderScene` JSON.

### Alternative: trunk

If you prefer [`trunk`](https://trunkrs.dev/), it can build and serve in one
step, but the static-asset + `wasm-pack` flow above needs no extra config and
matches the import paths shipped in `www/`.

## Host build & test (no browser required)

The pure layer and the `sample_scene_json` helper build and test on the host:

```sh
cargo build -p whiteboard-web
cargo test  -p whiteboard-web
cargo clippy -p whiteboard-web --all-targets -- -D warnings
```

The host tests cover `css_color`, `path_data`, `font_string`, the full
`scene_to_ops` lowering, and that `sample_scene_json()` returns valid JSON that
round-trips to a non-empty `RenderScene` and lowers to real canvas ops.
