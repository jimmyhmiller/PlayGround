# editor-idea

Experimental Bevy-based canvas of floating "panes" — each pane is a
draggable/resizable widget on an infinite-ish 2D surface. The canvas
hosts multiple widget kinds; right now: a **terminal emulator** (built
on `libghostty-vt`), a **text editor**, and a **run-button** widget.

When the user mentions "the terminal" in this directory, they almost
always mean `terminal-bevy` (the terminal emulator we're building),
**not** the macOS terminal application or Claude Code's terminal UI.
Same for "the editor" → `editor-bevy`.

## Workspace layout

- `crates/editor-core` — buffer/selection/transaction/history/commands.
  Pure logic, no Bevy. The model layer for the editor pane.
- `crates/pane-bevy` — shared chrome + lifecycle for floating panes
  (drag by title bar, corner resize, close button, focus, z-order,
  hit-testing, persistence, radial menu). New widget kinds register
  via `PaneRegistry` with a `PaneKindSpec`.
- `crates/editor-bevy` — text-editor pane: renders spans into a pane's
  content_root, owns caret/selection visuals, scroll, keyboard input,
  syntax highlight. Provides `EditorPlugin` (standalone) and
  `EditorEmbedPlugin` (for hosts that already own camera/font).
- `crates/widget-bevy` — retained-UI widget panes. Two hosting paths
  sharing one `Element` vocabulary (`src/protocol.rs`): **in-process
  Rhai** scripts (`src/rhai_widget.rs`, worker thread + named handlers
  like `on_click`/`on_toggle`/`on_input_change`/`on_bus`, hot reload from
  `~/.terminal-bevy/widgets/`) and **subprocess** widgets (`src/lib.rs`,
  NDJSON `HostEvent`/`WidgetMsg` over stdio). UI events and the Claude
  Code bus are SEPARATE channels — `on_bus` is the bus, not UI. See
  `crates/widget-bevy/AUTHORING.md` for the full handler/event model.
- `crates/terminal-bevy` — terminal-emulator pane on top of
  `libghostty-vt`. Each terminal is an Entity; the `!Send` VT runtime
  lives in a `NonSend<TerminalStore>` keyed by entity. Per-cell
  textured sprites sample a shared `GlyphAtlas`. v0 has direct key
  encoding (no Kitty kb), no wide-char, no mouse reporting, no
  scrollback panning. Also hosts the `tbopen` binary and the radial
  menu / projects / run-button infrastructure.

## libghostty-vt patch

`Cargo.toml` pins `libghostty-vt` / `libghostty-vt-sys` to a git rev of
`Uzaaft/libghostty-rs` that includes the zig optimize-mode fix (upstream
`3378f0b`). Without it, vendored ghostty builds default to zig Debug,
which makes `vt_write` 100x+ slower. Crates.io 0.1.1 predates the fix,
so the patch has to stay until upstream cuts a new release.
