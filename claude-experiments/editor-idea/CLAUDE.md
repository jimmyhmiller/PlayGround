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
- `crates/terminal-bevy` — terminal-emulator pane on top of
  `libghostty-vt`. Each terminal is an Entity; the `!Send` VT runtime
  lives in a `NonSend<TerminalStore>` keyed by entity. Per-cell
  textured sprites sample a shared `GlyphAtlas`. v0 has direct key
  encoding (no Kitty kb), no wide-char, no mouse reporting, no
  scrollback panning. Also hosts the `tbopen` binary and the radial
  menu / projects / run-button infrastructure.

## libghostty-vt patch

`Cargo.toml` patches `libghostty-vt` / `libghostty-vt-sys` to a local
vendored copy at `../libghostty-rs-vendored`. The vendored `build.rs`
forces `zig build -Doptimize=ReleaseFast` — upstream defaults to Debug,
which makes `vt_write` 100×+ slower. Don't drop the patch.
