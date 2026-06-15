// ES-module glue for the headless-whiteboard web backend demo.
//
// `wasm-pack build --target web` (run from crates/whiteboard-web/, see README.md)
// emits `../pkg/whiteboard_web.js` plus the `.wasm`. The default export is the
// async `init()` that fetches and instantiates the module; the named exports are
// the `#[wasm_bindgen]` functions from `src/lib.rs` / `src/backend.rs`:
//
//   - sample_scene_json(): String   — builds the demo scene, returns it as JSON
//   - render_scene_json(canvas, json) — paints a RenderScene JSON onto a canvas
//
// This file makes no drawing decisions; it only wires the two Rust calls
// together: `render_scene_json(canvas, sample_scene_json())`.

import init, { sample_scene_json, render_scene_json } from "../pkg/whiteboard_web.js";

const statusEl = document.getElementById("status");

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

async function main() {
  // `init()` fetches and compiles the .wasm next to whiteboard_web.js.
  await init();

  const canvas = document.getElementById("board");
  if (!(canvas instanceof HTMLCanvasElement)) {
    setStatus("no <canvas id=\"board\"> found", true);
    return;
  }

  // Build the demo scene in Rust, then render it in Rust. The JSON crossing the
  // boundary is exactly the RenderScene serde shape both backends share.
  const json = sample_scene_json();
  render_scene_json(canvas, json);

  setStatus(`rendered sample scene (${json.length} bytes of RenderScene JSON)`);
}

main().catch((err) => {
  console.error(err);
  setStatus(`failed to render: ${err}`, true);
});
