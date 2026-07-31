# Space Invaders / Intel 8080 — a direct C→Coil port

A working [Space Invaders](https://en.wikipedia.org/wiki/Space_Invaders) arcade
emulator, ported directly from [emulator101](http://emulator101.com)'s public-domain
C (`8080emu.c` + `SpaceInvadersMachine.m`) to Coil.

![attract screen](.) <!-- run sitest to regenerate /tmp/si.png -->

## Layout

| file | what |
|------|------|
| `i8080.coil`   | the Intel 8080 CPU — a direct port of `8080emu.c` |
| `machine.coil` | the arcade hardware: bit-shift register (ports 2/3/4), input port, 2 video interrupts/frame, 256×224 framebuffer decode |
| `main.coil`    | native macOS window front-end (AppKit/CoreGraphics via `objc.coil`) |
| `sitest.coil`  | headless: boot the ROM, run N frames, dump `/tmp/si.pgm` (no GUI) |
| `cputest.coil` | CP/M diagnostic harness (the CPU oracle) |
| `ref/`         | the original C, kept as the oracle generator |
| `golden/`      | captured C output for the diagnostic ROMs |

## CPU implementation

`i8080.coil` is faithful to emulator101's reference **including its quirks** (no
auxiliary-carry flag; `CMC` clears rather than complements carry; `RST` pushes
`pc+2`). So the oracle isn't "pass the 8080 test suite" — it's "produce exactly
what the C produces". The old ROM-driven validation wrapper was removed with the
ROM corpus; the CPU implementation and source-level test harness remain.

The C's flat 256-case `switch` collapses in Coil into pattern handlers keyed off
the opcode bits (`MOV`/`ALU`/`INR`/`DCR`/`MVI`/`LXI`/`INX`/`DCX`/`DAD`/rotates/
`Jcc`/`Ccc`/`Rcc`/`RST`/`PUSH`/`POP`) plus the irregular opcodes explicit.

## Supplying ROMs

No ROM images are included in this repository. Supply the copyrighted Space
Invaders ROM yourself from a source you are legally permitted to use. The standard
standard MAME `invaders` set — `invaders.h`, `invaders.g`, `invaders.f`,
`invaders.e` (2 KB each) — and concatenate them in memory order into an 8 KB image:

```sh
cat invaders.h invaders.g invaders.f invaders.e > /tmp/invaders.rom
# sha1: 2c6e7301635fcb5c9b845a97fcb2632eb7fbcbf8
```

## Run

```sh
# headless sanity check → writes /tmp/si.pgm (attract screen)
coil run sitest.coil -- /tmp/invaders.rom 500
coil run sitest.coil -- /tmp/invaders.rom play   # coin+start, in-game frame

# native window
coil build && ./invaders /tmp/invaders.rom
```

Keys: **1 / C** insert coin · **2 / Return** 1P start · **← → (or A / D)** move ·
**Space** fire.
