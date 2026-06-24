# gc-rust + raylib — a real game, proving the FFI (no shim)

A playable **Falling-Blocks Dodge** game written in gc-rust, rendered with
[raylib](https://www.raylib.com/). Move the yellow player with the **mouse**;
dodge the blue blocks; **R** restarts after a hit.

![screenshot](screenshot.png)

## What it proves

- **gc-rust calls raylib's C API directly — including `Color`-by-value.** There
  is **no C shim**. `ClearBackground(Color)`, `DrawCircle(int,int,float,Color)`,
  `DrawRectangle(...,Color)`, `DrawText(...,Color)` all take a `Color` struct
  **by value**, and gc-rust passes it per the platform C ABI (AAPCS64): a 4-byte
  `Color` is coerced into a general register, exactly as `clang` does. A
  homogeneous-float aggregate like `Vector2 {f32,f32}` goes in SIMD registers,
  and small structs are returned the same way. See `docs/ffi.md` and
  `abi_coerce` in `src/codegen.rs`.
- **Strings and bools cross too.** `InitWindow(..., const char*)` via
  `as_c_bytes`; `WindowShouldClose() -> bool` / `IsKeyDown(int) -> bool`.
- **GC and FFI coexisting under load.** Game state is a `Vec<Block>` of GC-heap
  structs, rebuilt every frame (off-screen blocks become garbage), so the moving
  generational collector runs continuously while native rendering happens. Safe
  because only scalars / by-value PODs cross the boundary — GC pointers never do,
  so the collector is free to relocate objects between frames.

## Build & run

Requires `raylib` (`brew install raylib`) and a built compiler
(`cargo build --bin gcr` from the repo root).

```sh
./build.sh      # gcr build dodge.gcr, linking libraylib (no shim to compile)
./dodge         # play it
```

`shot.gcr` is a deterministic screenshot harness (renders fixed frames →
`TakeScreenshot` → exits) used to produce `screenshot.png`.

## Files

| file | what |
|------|------|
| `dodge.gcr` | the game (gc-rust) — declares raylib's C functions and calls them directly |
| `shot.gcr` | deterministic screenshot harness |
| `build.sh` | `gcr build` with raylib link flags |
