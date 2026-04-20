# beagle-zelda

A port of a Bevy-based Zelda-like prototype (`~/Documents/Code/PlayGround/claude-experiments/2d-experiment`) to Beagle + raylib + raygui.

## Status
All planned features implemented. Verified via headless tests that run
the engine for 60 frames and serialize/deserialize a level round-trip.

## Features
- Player with WASD / arrow movement, Shift=run, Ctrl=sneak
- AABB wall collision (per-axis resolution)
- Cone-light fragment shader in GLSL 330 (player flashlight + up to 8 sentinel lights)
- Dynamic shadow mesh via CPU raycasts to wall corners (`rlBegin`/`rlVertex2f` triangles)
- Y-sorted walls/enemies/player with top + front faces (fake 2.5D depth)
- Monster enemy: detection range, line-of-sight check, notice accumulator, wander
- Sentinel enemy: sweeping searchlight cone, alert + chase when in cone + LOS
- Monster attack with invulnerability frames + respawn on 0 HP
- Fixed 60 Hz tick; camera follow with exponential smoothing
- Level editor (Tab to toggle):
  - Tool selector (Wall / Monster / Sentinel) via raygui toggle group
  - Left-click to place (drag rectangle for walls, click for enemies)
  - Right-click to select, Delete key to remove
  - WASD to pan a free camera
  - raygui sliders for Tuning: speed, accel, friction, light range/angle/intensity,
    ambient, camera smoothing, run/sneak multipliers
  - Save / Load buttons persist to `levels/level.txt`
  - Auto-loads `levels/level.txt` on startup if present

## Dependencies
- macOS (ARM64)
- raylib 5.x installed via Homebrew: `brew install raylib`
- raygui (vendored as `vendor/raygui.h`, compiled into `vendor/libraygui.dylib`)

## Project layout
```
beagle-zelda/
├── src/
│   ├── beagle_zelda.bg       # main game (player, AI, editor, render)
│   ├── ray.bg                # raylib + raygui FFI bindings
│   ├── vec.bg                # 2D vector helpers
│   ├── headless_test.bg      # 60-frame smoke test (play mode)
│   ├── editor_test.bg        # 60-frame smoke test (editor mode + save/load)
│   ├── import_test.bg        # FFI module sanity check
│   └── smoke_test.bg         # raygui smoke test
├── vendor/
│   ├── raygui.h              # upstream raygui 5.0 single-header
│   ├── build_raygui.sh       # one-shot build script
│   └── libraygui.dylib       # produced by build_raygui.sh
├── shaders/cone_light.fs     # reference copy of the GLSL shader (also embedded)
├── levels/                   # level.txt is written here by the editor
└── README.md
```

## Building
```bash
# one-time: compile libraygui.dylib + C helpers
./vendor/build_raygui.sh

# run the game
~/Documents/Code/beagle/target/release/beag src/beagle_zelda.bg

# smoke-test headlessly (60 frames then exits)
~/Documents/Code/beagle/target/release/beag src/headless_test.bg

# smoke-test editor mode + serialization round-trip
~/Documents/Code/beagle/target/release/beag src/editor_test.bg
```

## Implementation notes
- Rectangle-by-value FFI (`GuiButton`'s `Rectangle bounds`) is passed as 4 × F32
  to match the AArch64 HFA calling convention (v0..v3).
- Shader (`{ u32 id, void* locs }`) is passed as 2 × U64, same pattern as
  `resources/examples/breakout.bg`.
- Uniform uploads use Beagle's native `ffi/set-f32` / `ffi/get-f32` buffer
  accessors.
- Mutable globals aren't allowed, so the RNG state lives in an `atom`.
- No `Camera2D` FFI: camera math is done in Beagle and every draw call receives
  screen coordinates. The shader works in screen-space via `gl_FragCoord`.
