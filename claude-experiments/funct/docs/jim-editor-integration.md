# Hosting `funct` widgets in jim-editor

How to run `.ft` widgets inside jim-editor, side by side with the existing Rhai
widgets — what to build on the editor side, and why most of it is already done.

> **The short version.** funct already speaks the editor's widget contract. The
> proof is `tests/common/mod.rs` in this repo: that harness registers *exactly*
> the host surface `crates/jim-widget/src/rhai_widget.rs` provides, drives a
> real widget (`examples/widgets/chess.ft`) through `on_init` / `render` /
> `on_click` / `on_frame`, and even snapshots and restores it. So the editor
> side is a thin **adapter**: a `FunctWorker` that mirrors the existing Rhai
> `Worker`, registers the same natives, and calls the same lifecycle functions.
> Nothing in the render/layout/protocol stack changes.

---

## 1. The two sides line up already

| Concern | Rhai today (`rhai_widget.rs`) | funct (`funct` crate) |
|---|---|---|
| Engine | `rhai::Engine` + `Scope` | `funct::Funct` |
| Run script | `engine.run_ast_with_scope` | `vm.eval(src)` |
| Call a handler | `engine.call_fn::<Dynamic>(scope, ast, name, args)` | `vm.call(name, args) -> Result<Value, Fault>` |
| Register a native | `engine.register_fn("foo", \|..\| ..)` | `vm.register_raw` / `vm.register0..5` |
| Inject a global | `scope.push("canvas_w", w)` | `vm.set_global("canvas_w", v)` |
| Frame value | `Dynamic` → `from_dynamic::<Element>` | `Value` → `.to_json()` → `serde_json::from_value::<Element>` |
| Persisted state | round-trip a `state` map to JSON | **whole-VM** `vm.save_state` / `vm.restore_state` |
| Hot reload | re-parse AST, re-run top level | `vm.reload_module(path)` / re-`eval` (fn hot-swap by name) |
| Thread-safety | `Engine`/`Scope` on a worker thread | `Funct`/`Value`/`VmState` are `Send + Sync` |

The lifecycle function names are identical because the chess port was written to
the editor's contract: `on_init()`, `render(canvas_w, canvas_h)`,
`on_click(x, y, shift, ctrl, key)`, `on_frame(dt)`, `on_resize(w, h)`, `on_key`,
plus the bus/input handlers. A widget pulls the host surface in with
`import "host"` (see `examples/widgets/host.ft`).

---

## 2. The host surface to register (the spec is the harness)

`tests/common/mod.rs::install_widget_host` is the authoritative list. Register
each of these on the `Funct` the same way `register_widget_host_fns` does on the
Rhai `Engine`. Wire them to the **same editor subsystems** the Rhai versions use
(`crate::subprocess`, `crate::request_main_loop_wakeup`, the msgbus, the
clipboard) — they're literally the same functions, only the binding API differs.

Globals (inject with `vm.set_global`, refresh on `Resize`):
- `canvas_w`, `canvas_h` — `Value::Float`

Render / animation:
- `request_render()` → set the worker's `render_dirty` flag
- `set_animating(on: bool)` → set the worker's `animating` flag

Subprocess bridge (UCI engines, LSPs — back it with `crate::subprocess`):
- `proc_spawn(cmd)` / `proc_spawn(cmd, [args])` → `Int` handle (`-1` = spawn failed)
- `proc_write(handle, line) -> Bool`
- `proc_read(handle) -> Str` (empty string when nothing buffered)
- `proc_alive(handle) -> Bool`
- `proc_kill(handle)`

Style / drawing surface (no-ops are fine for non-Glaze widgets, but register
them so snapshots stay portable — see §5):
- `uniform_set(name, value)`
- `mask_paint(name, x, y, radius, value)`
- `oklch(l, c, h) -> Str`

Misc:
- `emit(kind, payload)` → publish on the widget msgbus
- `host_log(msg...)` → `eprintln!` / editor log
- `host_env(name) -> Str`
- `widget_asset(rel) -> Str` (absolute path under the widget's asset dir)
- `clipboard_set(text) -> Bool` → `crate::subprocess::clipboard_set`
- `time() -> Float` (unix seconds)

**Placeholder semantics (important and useful):** in funct, an `extern fn`
declared in `host.ft` that the embedder does *not* register becomes a loud
placeholder that faults **only if called**. So a widget can `import "host"`
(the whole interface) and use only part of it, and you can register a subset
during bring-up. Unimplemented natives fail loudly at the call site, never
silently — which is exactly the contract this project wants.

`vm.register_raw(name, |vm, args| -> Result<Value, Fault>)` gives full engine
access (needed for `proc_spawn`'s variadic args and for anything that calls back
into script). The arity helpers `vm.register0..register5` with auto type
conversion cover the rest. See the README "Embedding" section and `src/interop.rs`.

---

## 3. The frame conversion — the one real gotcha

The Rhai path is `Dynamic → rhai::serde::from_dynamic::<Element>`. The funct path
is `Value → Value::to_json() → serde_json::from_value::<Element>`.

**`type` is a keyword in funct, so funct widgets use `kind:` as the element
discriminator** (see the header comment in `examples/widgets/chess.ft:12`). But
`protocol::Element` and `protocol::CanvasItem` are `#[serde(tag = "type")]`. So
the adapter must rename the discriminator `kind` → `type` on the JSON before
deserializing:

```rust
fn funct_frame_to_element(frame: funct::Value) -> Result<Option<Element>, String> {
    if matches!(frame, funct::Value::Unit) {
        return Ok(None); // no render fn / nothing to draw — valid
    }
    let mut json = frame.to_json().map_err(|e| format!("frame to_json: {e}"))?;
    rename_kind_to_type(&mut json); // recursive: every object's "kind" -> "type"
    serde_json::from_value::<Element>(json)
        .map(Some)
        .map_err(|e| format!("frame deserialize: {e}"))
}
```

`rename_kind_to_type` walks the `serde_json::Value` tree and, for every object,
moves the `kind` field to `type` (recursing into `children` and arrays). Values
are kebab-case strings already (`"canvas"`, `"sprite"`, `"rect"`, `"text"`), so
nothing else changes. After the rename, the existing `Element`/`CanvasItem`
deserializers and the entire `render.rs` pipeline work unchanged.

> Alternative: teach widgets to emit `"type"` via a string key
> (`{ "type": "canvas", .. }` is legal funct even though bare `type:` isn't),
> or add `#[serde(alias = "kind")]` to the protocol enums. The rename in the
> adapter is the least invasive and keeps `.ft` widgets idiomatic.

Everything else about the frame is identical: `Canvas { children: [CanvasItem] }`,
sprite `path` resolved through `widget_asset`, ids like `sq_0` / `btn_new` for
hit-testing. The chess test asserts all of this (`tests/chess_widget.rs::initial_render_and_fen`).

---

## 4. The `FunctWorker` (mirror of `Worker`)

Add a sibling to the Rhai `Worker` in `jim-widget`. Same worker-thread model,
same `WorkerSlots` (`latest_frame`, `snapshot`, `render_dirty`, `animating`,
`last_error`, `frame_gen`), same `HostToWorker` message enum. Only the engine
calls differ:

```rust
struct FunctWorker {
    vm: funct::Funct,
    slots: WorkerSlots,
    canvas_w: f32,
    canvas_h: f32,
    initialized: bool,
}

impl FunctWorker {
    fn load(&mut self, src: &str) -> bool {
        // register the §2 host surface FIRST, then set module root, then eval
        install_widget_host(&mut self.vm, &self.slots, /* subprocess, msgbus, .. */);
        self.vm.set_module_root(widget_dir);          // so `import "host"` resolves
        self.vm.set_global("canvas_w", Value::Float(self.canvas_w as f64));
        self.vm.set_global("canvas_h", Value::Float(self.canvas_h as f64));
        match self.vm.eval(src) {
            Ok(_) => true,
            Err(e) => { self.set_error(format!("load: {e}")); false }
        }
    }

    fn call_handler(&mut self, name: &str, args: Vec<Value>) {
        match self.vm.call(name, args) {
            Ok(_) => self.clear_error(),
            // a missing optional handler is NOT an error — check the Fault kind
            // (or pre-check existence) and swallow only "function not found".
            Err(e) if is_unknown_fn(&e, name) => {}
            Err(e) => self.set_error(format!("{name}: {e}")),
        }
    }

    fn render_if_dirty(&mut self) {
        if !self.slots.render_dirty.swap(false, Ordering::AcqRel) { return; }
        let frame = match self.vm.call("render",
            vec![Value::Float(self.canvas_w as f64), Value::Float(self.canvas_h as f64)]) {
            Ok(v) => v,
            Err(e) if is_unknown_fn(&e, "render") => return, // no visual: valid
            Err(e) => { self.set_error(format!("render: {e}")); return; }
        };
        match funct_frame_to_element(frame) {
            Ok(el) => { *self.slots.latest_frame.lock().unwrap() = el; }
            Err(msg) => { self.set_error(msg); return; }
        }
        self.slots.frame_gen.fetch_add(1, Ordering::Release);
        crate::request_main_loop_wakeup();
    }
}
```

Message dispatch is a one-to-one translation of `Worker::handle_message`:

| `HostToWorker` | funct call |
|---|---|
| `Resize { w, h }` | set `canvas_w/h`, `set_global` both, `call_handler("on_resize", [w,h])` |
| `Key { key }` | `call_handler("on_key", [str])` |
| Click (from hit-test) | `call_handler("on_click", [x, y, shift, ctrl, key])` |
| Tick (`animating`) | `call_handler("on_frame", [dt])` |
| `ClaudeEvent` / bus | `call_handler("on_bus" | "on_message", [..])` |
| input/select/slider | `call_handler("on_input_*"/"on_select_*"/..)` |
| `Rerender` | `render_dirty = true` |
| `Reload` | see §5 |
| `Shutdown` | break the loop |

Two funct conveniences worth using vs. the Rhai path:

- **No CoW state round-trip.** Rhai must re-`push` `state` after every handler to
  defeat copy-on-write (`rhai_widget.rs:443`). funct has no such issue — atoms
  and top-level `let` are real mutable cells. Just call the handler.
- **Optional-handler check.** Instead of pattern-matching an error kind, you can
  pre-check whether the widget defines a handler before calling it (cache the
  set of defined function names after `eval`).

---

## 5. Snapshots & hot reload — where funct is strictly better

The Rhai worker can only persist a single `state` map round-tripped to JSON
(`maybe_render_and_persist`). funct reifies the **entire VM** — code table,
globals, atoms (identity + cycles preserved), frames, operand stack — so a paused
widget is fully serializable.

**Persist on close:**
```rust
let st = funct::VmState { frames: vec![], stack: vec![],
                          status: funct::Status::Done(Value::Unit) };
let json = vm.save_state(&st)?;     // store in the snapshot slot / ~/.jim
```

**Restore (possibly a new process):** re-register the *same* host surface first,
set the same globals, then restore and resume:
```rust
let mut vm = Funct::new();
install_widget_host(&mut vm, &slots, ..);   // MUST match — see below
vm.set_global("canvas_w", Value::Float(w)); vm.set_global("canvas_h", Value::Float(h));
vm.restore_state(&json)?;
vm.call("on_init", vec![])?;                 // or just resume; on_init is idempotent here
```

`tests/chess_widget.rs::snapshot_round_trip_preserves_the_game` does exactly
this: play `1. e4 e5`, `save_state`, build a fresh `Funct` with the host
re-installed, `restore_state`, and keep playing — SAN history, FEN, and move
legality all survive.

**Serialization contract (loudly enforced, by design):**
- Native **values** (host objects) can't be serialized — `save_state` fails with
  a clear error rather than dropping them. Keep widget state in script values.
- Native **functions** serialize as *names* — the restoring process must register
  the identical natives first or `restore_state` fails loudly. This is why the
  host surface must be installed before restore, and why §2 says to register the
  *whole* `host.ft` interface (incl. the style no-ops): a saved game references
  every native the imported `host.ft` binds.

**Hot reload:** re-`eval` a changed `fn` hot-swaps it by name — all callers and
stored closures pick up the new code (resolution is by `FnId` through the table),
and in-flight frames finish on the code they started with. For a module-backed
widget, `vm.reload_module(path)` hot-swaps a module's functions for all importers.
Either way you keep current atom/VM state across the reload — no `last_state_json`
re-seed dance.

---

## 6. Routing `.ft` vs `.rhai`

`widgets_dir()` already scans `~/.jim/widgets`. Route by extension: `.ft` →
`FunctWorker`, `.rhai` → `RhaiWidget`. Make the public `Widget` handle an enum
(or a trait object) over the two worker kinds so the Bevy plugin
(`RhaiWidgetPlugin` and friends) drives them through one interface — frame slot,
animating flag, `send_*` event methods, `is_animating`, `latest_frame`. The
hit-testing, render, and layout code consume `protocol::Element` and don't care
which engine produced it.

Bring-up order:
1. `funct` dep on `jim-widget`; `FunctWorker` that loads a trivial `.ft` returning
   a one-rect Canvas; confirm it renders (validates §3 rename + slots).
2. Port the host surface (§2), pointing the natives at the same subsystems as the
   Rhai versions. `examples/widgets/chess.ft` is the end-to-end exerciser — drop
   it in `~/.jim/widgets`, click pieces, enable the bot (needs `stockfish` on
   PATH), hit "review".
3. Wire snapshot/reload (§5).
4. Unify the `Widget` handle so both engines share the plugin/event plumbing.

---

## 7. Reference

In this repo (funct):
- `tests/common/mod.rs` — the host harness = the surface to implement (§2).
- `tests/chess_widget.rs` — end-to-end lifecycle, frame asserts, snapshot test.
- `examples/widgets/host.ft` — the `import "host"` interface.
- `examples/widgets/chess.ft` — a real ~1,700-line widget (note line 12 on `kind:`).
- `README.md` — "Embedding", "The reified VM", "Atoms & hot reload", serde limits.
- `src/interop.rs` — `register*`, `ToValue`/`FromValue`, modules.
- `src/snapshot.rs` — what `save_state`/`restore_state` cover.

In jim-editor:
- `crates/jim-widget/src/rhai_widget.rs` — the `Worker` / `WorkerSlots` /
  `HostToWorker` model and `register_widget_host_fns` to mirror.
- `crates/jim-widget/src/protocol.rs` — `Element` / `CanvasItem` (`tag = "type"`).
- `crates/jim-widget/src/{render,layout,subprocess,msgbus}.rs` — unchanged
  consumers of the frame.
