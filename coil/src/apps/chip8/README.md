# CHIP-8 in Coil

This directory contains terminal and native macOS front ends over the shared
CHIP-8 VM in `chip8.coil`. ROM images are intentionally not included.

Supply a CHIP-8 program you are legally permitted to use:

```sh
cd src/apps/chip8
coil run terminal.coil -- /path/to/game.ch8
coil build
./chip8-gui /path/to/game.ch8
```
