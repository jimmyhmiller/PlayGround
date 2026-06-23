# gc-rust + raylib — a real game, proving the FFI

A playable **Falling-Blocks Dodge** game written in gc-rust, rendered with
[raylib](https://www.raylib.com/). Move the yellow player with the **mouse**;
dodge the blue blocks; **R** restarts after a hit.

![screenshot](screenshot.png)

## What it proves

- **Direct calls into a real third-party C library.** Every scalar-clean raylib
  function is called straight from gc-rust via `extern "C"` — `InitWindow`
  (with a `RawPtr` string built by `as_c_bytes`), `BeginDrawing`/`EndDrawing`,
  `SetTargetFPS`, `CloseWindow`, `GetMouseX`. No wrapper.
- **A thin C shim only where the ABI demands it.** raylib passes `Color` (and
  `Vector2`, `Rectangle`) **by value in registers**; gc-rust crosses value
  structs **by pointer** — an ABI mismatch. So `rayshim.c` exposes scalar-only
  wrappers (`rs_circle`, `rs_rect`, `rs_text`, `rs_clear`) and the `bool`-returning
  calls (`rs_should_close`, `rs_key_down`). This is the standard way to bind a
  by-value-struct C API; the gc-rust side stays pure scalars + `RawPtr`.
- **GC and FFI coexisting under load.** The game state is a `Vec<Block>` of
  **GC-heap structs**, rebuilt every frame (off-screen blocks become garbage), so
  the moving generational collector runs continuously while native rendering
  happens through FFI. They don't interfere because **only scalars ever cross the
  boundary** — GC pointers never do, so the collector is free to relocate objects
  between frames.

## Build & run

Requires `raylib` (`brew install raylib`) and a built compiler
(`cargo build --bin gcr` from the repo root).

```sh
./build.sh      # compiles rayshim.c, then `gcr build dodge.gcr` with link args
./dodge         # play it
```

`shot.gcr` is a deterministic screenshot harness (renders fixed frames →
`TakeScreenshot` → exits) used to produce `screenshot.png`.

## Files

| file | what |
|------|------|
| `dodge.gcr` | the game (gc-rust) |
| `rayshim.c` | scalar-only C wrappers for raylib's by-value-`Color` / `bool` calls |
| `shot.gcr` | deterministic screenshot harness |
| `build.sh` | compile shim + `gcr build` with raylib link flags |
