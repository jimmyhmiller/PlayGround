# The browser demo

A running program you edit while it runs, in a tab. The animation is one
long-running computation; every edit lands between two of its instructions.

```sh
./wasm/build.sh      # compile the core + host to wasm32, generate the JS glue
./wasm/serve.sh      # http://localhost:8080  (serves with Cache-Control: no-store)
```

`build.sh` needs the `wasm32-unknown-unknown` target and a `wasm-bindgen` CLI
matching the crate version pinned in `crates/livetype-wasm/Cargo.toml` — they
share a schema version and refuse to work across a mismatch.

## What it is

`crates/livetype-wasm` is a host for `livetype-core`, in the same shape as the
native `livetype-ffi-gui` demo: declare a foreign interface, bind native
implementations, load a program whose `letonce` globals hold the native
resources, then run it. The only difference is that the "native toolkit" is a
command buffer JavaScript paints onto a `<canvas>` instead of a windowing
library.

The guest program and every scripted edit are real source files in `demo/`,
embedded with `include_str!`. `tests/demo_scene.rs` drives those same files
headlessly, so the behaviour the page shows is pinned by the test suite rather
than by eyeballing a canvas.

## The walkthrough

Six steps in order, under the source on the right. Each step shows what to
watch for and its own source, applies straight into the live world, and only
unlocks the next step once applied. Nothing here loads into the editor — that
holds your work, and the walkthrough has to stay usable after you have been
playing in it.

Each step reports what it did in the demo's own terms: what it *evolved* (a
struct or enum version), what it *installed*, and what it **broke** — being
published Broken is not an installation, and step 5 exists to show exactly that
difference.


| # | Edit | What it shows |
|---|---|---|
| 1 | `edit_1_radius.lt` | Redefining a function takes effect at the next call. The loop is never restarted. |
| 2 | `edit_2_migrate.lt` | A struct gains a defaulted field; the particles already on screen migrate lazily at their next field read. Same objects, no reseed. |
| 3 | `edit_3_rejected.lt` | An ill-typed edit is refused at install. The running program keeps executing the last good version. |
| 4 | `edit_4_enum.lt` | An enum is introduced and dispatched on. |
| 5 | `edit_5_break.lt` | Adding a variant makes a `match` non-exhaustive. It is published Broken *at install*, brokenness propagates to callers, and the program freezes at the boundary — before the frame performs a single effect. The last good frame stays on screen. |
| 6 | `edit_6_repair.lt` | Repairing only the *root cause* revives the whole broken chain, and the suspended computation resumes at the instruction that trapped. |

Throughout, the "canvas opened" counter stays at 1: `letonce` holds the native
resource, so edits change code and never the running world.

## Controls

- **Drag any number** to scrub it. The value is rewritten in the source and
  installed on every step, so the animation responds under the cursor. Shift
  gives finer steps; a click that does not move places the caret instead, so
  numbers stay editable by typing.
- **Apply as I type** — install on every keystroke, debounced.
- **Apply edit** (⌘/Ctrl+Enter) — install the editor's contents into the *running*
  world. This is the live path: the program is never restarted.
- **Pause / Resume** and **Step 1 frame** — the host declining to step the
  program, which is unrelated to the engine. A *paused* program is healthy and
  simply is not being asked to run; a *frozen* one trapped on its own. Edits
  install while paused either way.
- **Restart world with this source** — boot a fresh world from the editor's
  contents. Needs a whole scene, not a fragment.
- **Load full scene** / **Reset to original** — restore the shipped source.
  Both ask first if you have unsaved changes in the editor.

### The editor

A transparent `<textarea>` over a syntax-highlighted `<pre>`, with no editor
dependency: the language has no existing mode, so a grammar had to be written
either way, and this keeps the page self-contained and offline. It also makes
scrubbing straightforward — every number is a real element in the highlight
layer, so it can be hit-tested directly rather than measured out of a textarea.

The highlight layer sits on top so number tokens can be grabbed, but is
click-through everywhere else, so all other input reaches the textarea beneath.
The two layers must lay text out identically or the caret drifts from the
glyphs; that is what the shared padding/font/wrapping rules are for.

**Scrubbing and live typing install only the definition under the point being
edited**, not the whole buffer. Re-evaluating the entire scene per keystroke
would reinstall every definition in it — versions climbing 60×/second with a
struct-migration churn behind them.

### Editing `seed` (and why "Apply" looks like it does nothing)

`seed` is called once, by `letonce world = seed()`. Applying a new `seed` to the
running world installs it and changes nothing on screen — the initializer already
ran and never runs again, which is precisely what makes the particles survive
every other edit. The demo says so rather than reporting a bare success: after an
edit it reports which installed functions the running program can no longer
reach (walking direct calls from `main`; it stays quiet if anything on that path
calls through a function *value*, since the callee is only known at call time).

To actually see a different `seed`: **Load full scene**, change it there, then
**Restart world with this source** — the initializer re-runs with your version.

## What this configuration does not cover

The engine here is the interpreted (cold) tier, single-threaded:

- **No LLVM tier.** `inkwell` does not target wasm. It owns no live-editing
  semantics — pause, repair, resume, migration, and verification all live in
  the core — and `tests/differential_fuzz.rs` holds the two tiers to identical
  observable behaviour, but the tier-promotion story is not visible here.
- **No threads.** Multicore actors, `Send`/`Recv`, and stop-the-world GC need
  wasm atomics, `SharedArrayBuffer`, and COOP/COEP headers.
