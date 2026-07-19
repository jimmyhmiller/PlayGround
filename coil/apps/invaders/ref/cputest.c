// Golden CP/M test harness for the 8080 core (reference behaviour we port to coil).
// Loads a CP/M .COM at 0x100, installs a minimal BDOS shim, runs to completion.
//   cc -O2 cputest.c 8080emu_plain.c -o cputest && ./cputest ../../roms/CPUTEST.COM
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "8080emu.h"

static long load(State8080 *s, const char *path, uint16_t at) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    fread(&s->memory[at], 1, n, f); fclose(f);
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: cputest <rom.COM>\n"); return 1; }
    State8080 s; memset(&s, 0, sizeof s);
    s.memory = calloc(0x10000, 1);
    load(&s, argv[1], 0x100);
    s.pc = 0x100;

    // CP/M programs CALL 5 for BDOS and expect a stack; the .COM's final RET goes to 0.
    // Emulate101-style interception: pc==5 => BDOS, pc==0 => done.
    for (;;) {
        if (s.pc == 5) {
            if (s.c == 9) {                       // C_WRITESTR: print $-terminated string at DE
                uint16_t de = (s.d << 8) | s.e;
                for (uint16_t a = de; s.memory[a] != '$'; a++) putchar(s.memory[a]);
            } else if (s.c == 2) {                // C_WRITE: print char in E
                putchar(s.e);
            }
            // manual RET
            s.pc = s.memory[s.sp] | (s.memory[s.sp + 1] << 8);
            s.sp += 2;
            continue;
        }
        if (s.pc == 0) break;                     // returned to CP/M warm boot
        Emulate8080Op(&s);
    }
    putchar('\n');
    return 0;
}
