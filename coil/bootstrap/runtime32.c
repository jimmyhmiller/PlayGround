// runtime32.c — the C runtime + driver for the wasm2c-translated Coil compiler,
// wasm32 (MVP linear memory) variant.
//
// This is the sibling of runtime.c: runtime.c services a memory64 module (all
// linear-memory offsets are i64 → uint64_t); this file services a memory32
// module (offsets are i32 → uint32_t). The env.* import SIGNATURES therefore
// differ — most pointer/size params narrow to uint32_t — and are copied
// verbatim from the prototypes wasm2c emits at the top of the generated
// coilc32.c (a size_t/ssize_t/isize param is i32 on wasm32, an i64 param stays
// uint64_t). Everything else — the bump+free-list allocator, the printf
// family, the FS/thread shims — is identical logic, since MEM+off is a valid
// host pointer regardless of the offset's width.
//
// Built by build32.sh:  cc coilc32.c runtime32.c -o coil-bootstrap32 -lm

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

// ---- module interface (defined in the generated coilc32.c) -----------------
// NB: for a memory32 module wasm2c emits `wasm___heap_base` as uint32_t and
// `wasm_main(argc, argv_off)` with a uint32_t argv_off.
extern uint8_t **const wasm_memory;          // &m0 : the linear-memory buffer
extern const uint32_t *const wasm___heap_base;      // heap start offset (i32)
extern void wasm_init(void);                 // allocate memory + data segments
extern uint64_t wasm_main(uint32_t argc, uint32_t argv_off);

#define MEM (*wasm_memory)                    // current host base of linear mem
static char *hoststr(uint32_t off) { return (char *)(MEM + off); }

static void die(const char *msg) { fputs(msg, stderr); fputc('\n', stderr); abort(); }

// ---------------------------------------------------------------------------
// Allocator over linear memory (identical to runtime.c; offsets fit in 32 bits
// on wasm32 so internal bookkeeping stays uint64_t and the public offsets are
// truncated to uint32_t at the import boundary — always < 4 GiB of memory).
// ---------------------------------------------------------------------------
#define HDR 16u
#define ALIGN 16u
#define MAGIC UINT64_C(0xC011C0DEC0FFEE01)

static uint64_t g_brk;        // next unused offset (bump pointer)
static uint64_t g_cap;        // bytes currently backed by the host buffer

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + (a - 1)) & ~(a - 1); }

static uint64_t hdr_size_get(uint64_t h) { uint64_t v; memcpy(&v, MEM + h, 8); return v; }
static void     hdr_size_set(uint64_t h, uint64_t v) { memcpy(MEM + h, &v, 8); }
static uint64_t hdr_tag_get(uint64_t h) { uint64_t v; memcpy(&v, MEM + h + 8, 8); return v; }
static void     hdr_tag_set(uint64_t h, uint64_t v) { memcpy(MEM + h + 8, &v, 8); }

#define NBINS (1u << 17)
static uint64_t bin_key[NBINS];
static uint64_t bin_head[NBINS];

static uint64_t *bin_slot(uint64_t size) {
    uint64_t i = (size * UINT64_C(0x9E3779B97F4A7C15)) >> 47;
    for (uint32_t probe = 0; probe < NBINS; probe++) {
        uint64_t j = (i + probe) & (NBINS - 1);
        if (bin_key[j] == 0 || bin_key[j] == size) { bin_key[j] = size; return &bin_head[j]; }
    }
    die("bootstrap allocator: free-bin table full");
    return NULL;
}

// grow the linear-memory buffer so byte offset `end` is backed. On wasm32 the
// module addresses memory with 32-bit offsets, so the buffer must never exceed
// 4 GiB — a genuine 32-bit-overflow guard, unlike the memory64 runtime.
static void ensure(uint64_t end) {
    if (end <= g_cap) return;
    if (end > UINT64_C(0xFFFFFFFF)) die("bootstrap allocator: wasm32 linear memory exceeded 4 GiB");
    uint64_t ncap = g_cap + g_cap / 2;
    if (ncap < end) ncap = end;
    ncap = align_up(ncap, 65536);               // whole wasm pages
    if (ncap > UINT64_C(0xFFFFFFFF)) ncap = UINT64_C(0xFFFFFFFF);
    uint8_t *nm = realloc(MEM, (size_t)ncap);
    if (nm == NULL) die("bootstrap allocator: out of memory (realloc failed)");
    memset(nm + g_cap, 0, (size_t)(ncap - g_cap));
    MEM = nm;
    g_cap = ncap;
}

static uint64_t rt_malloc(uint64_t size) {
    uint64_t rounded = align_up(size == 0 ? 1 : size, ALIGN);
    uint64_t *slot = bin_slot(rounded);
    if (*slot != 0) {
        uint64_t h = *slot;
        *slot = hdr_tag_get(h);
        hdr_tag_set(h, MAGIC);
        return h + HDR;
    }
    uint64_t h = align_up(g_brk, ALIGN);
    g_brk = h + HDR + rounded;
    ensure(g_brk);
    hdr_size_set(h, rounded);
    hdr_tag_set(h, MAGIC);
    return h + HDR;
}

static void rt_free(uint64_t ptr) {
    if (ptr == 0) return;
    uint64_t h = ptr - HDR;
    if (hdr_tag_get(h) != MAGIC) return;
    uint64_t size = hdr_size_get(h);
    uint64_t *slot = bin_slot(size);
    hdr_tag_set(h, *slot);
    *slot = h;
}

static uint64_t rt_realloc(uint64_t ptr, uint64_t size) {
    if (ptr == 0) return rt_malloc(size);
    uint64_t h = ptr - HDR;
    uint64_t old = (hdr_tag_get(h) == MAGIC) ? hdr_size_get(h) : 0;
    uint64_t rounded = align_up(size == 0 ? 1 : size, ALIGN);
    if (old >= rounded) return ptr;
    uint64_t np = rt_malloc(size);
    uint64_t n = old < size ? old : size;
    if (n) memmove(MEM + np, MEM + ptr, (size_t)n);
    rt_free(ptr);
    return np;
}

// ---------------------------------------------------------------------------
// printf family (one variadic arg; identical to runtime.c).
// ---------------------------------------------------------------------------
static size_t fmt_one(char *out, size_t cap, const char *fmt, uint64_t arg) {
    size_t o = 0;
    int used = 0;
    #define PUT(ch) do { if (o + 1 < cap) out[o] = (ch); o++; } while (0)
    for (const char *p = fmt; *p; ) {
        if (*p != '%') { PUT(*p); p++; continue; }
        const char *start = p++;
        if (*p == '%') { PUT('%'); p++; continue; }
        while (*p == 'l' || *p == 'h' || *p == 'z' || *p == 'j' || *p == 't') p++;
        char conv = *p ? *p++ : 0;
        char tmp[64];
        int n = 0;
        uint64_t a = used ? 0 : arg;
        used = 1;
        switch (conv) {
            case 'd': case 'i':
                n = snprintf(tmp, sizeof tmp, "%lld", (long long)(int64_t)a); break;
            case 'u':
                n = snprintf(tmp, sizeof tmp, "%llu", (unsigned long long)a); break;
            case 'x':
                n = snprintf(tmp, sizeof tmp, "%llx", (unsigned long long)a); break;
            case 'X':
                n = snprintf(tmp, sizeof tmp, "%llX", (unsigned long long)a); break;
            case 'o':
                n = snprintf(tmp, sizeof tmp, "%llo", (unsigned long long)a); break;
            case 'c':
                n = snprintf(tmp, sizeof tmp, "%c", (int)a); break;
            case 'p':
                n = snprintf(tmp, sizeof tmp, "%p", (void *)(uintptr_t)a); break;
            case 'f': case 'F': case 'g': case 'G': case 'e': case 'E': {
                double d; memcpy(&d, &a, 8);
                char f2[3] = { '%', conv, 0 };
                n = snprintf(tmp, sizeof tmp, f2, d); break;
            }
            case 's': {
                const char *s = hoststr((uint32_t)a);
                for (const char *q = s; *q; q++) PUT(*q);
                continue;
            }
            default:
                for (const char *q = start; q < p; q++) PUT(*q);
                continue;
        }
        for (int i = 0; i < n; i++) PUT(tmp[i]);
    }
    if (cap) out[o < cap ? o : cap - 1] = 0;
    #undef PUT
    return o;
}

// ===========================================================================
// env.* imports — signatures MUST match the prototypes in coilc32.c.
// ===========================================================================

// ---- allocation ----
uint32_t env_malloc(uint32_t size) { return (uint32_t)rt_malloc(size); }
uint32_t env_realloc(uint32_t p, uint32_t size) { return (uint32_t)rt_realloc(p, size); }
uint64_t env_free(uint32_t p) { rt_free(p); return 0; }
uint32_t env_calloc(uint32_t n, uint32_t sz) {
    uint64_t total = (uint64_t)n * sz; if (total == 0) total = 1;
    uint32_t p = (uint32_t)rt_malloc(total);
    memset(MEM + p, 0, (size_t)total);
    return p;
}
uint32_t env_memset(uint32_t s, uint64_t c, uint32_t n) { memset(MEM + s, (int)c, (size_t)n); return s; }
uint64_t env_memcmp(uint32_t a, uint32_t b, uint32_t n) {
    int r = memcmp(MEM + a, MEM + b, (size_t)n);
    return (uint64_t)(int64_t)(r < 0 ? -1 : (r > 0 ? 1 : 0));
}
uint32_t env_strlen(uint32_t p) { return (uint32_t)strlen(hoststr(p)); }

// ---- file / directory I/O ----
uint32_t env_open(uint32_t path, uint32_t flags) { return (uint32_t)open(hoststr(path), (int)flags, 0666); }
uint64_t env_creat(uint32_t path, uint64_t mode) {
    return (uint64_t)(int64_t)open(hoststr(path), O_CREAT | O_WRONLY | O_TRUNC, (mode_t)mode);
}
uint32_t env_read(uint32_t fd, uint32_t ptr, uint32_t len) {
    return (uint32_t)(int32_t)read((int)fd, MEM + ptr, (size_t)len);
}
uint32_t env_write(uint32_t fd, uint32_t ptr, uint32_t len) {
    return (uint32_t)(int32_t)write((int)fd, MEM + ptr, (size_t)len);
}
uint32_t env_close(uint32_t fd) { if (fd > 2) close((int)fd); return 0; }
uint32_t env_access(uint32_t path, uint64_t mode) { return (uint32_t)access(hoststr(path), (int)mode); }
uint32_t env_unlink(uint32_t path) { return (uint32_t)unlink(hoststr(path)); }
uint32_t env_rename(uint32_t a, uint32_t b) { return (uint32_t)rename(hoststr(a), hoststr(b)); }
uint32_t env_realpath(uint32_t path, uint32_t out) {
    char buf[4096];
    if (realpath(hoststr(path), buf) == NULL) return 0;
    size_t n = strlen(buf) + 1;
    memcpy(MEM + out, buf, n);
    return out;
}
uint32_t env_getcwd(uint32_t buf, uint64_t size) {
    if (getcwd((char *)(MEM + buf), (size_t)size) == NULL) return 0;
    return buf;
}
uint32_t env_getenv(uint32_t name) { (void)name; return 0; }
uint32_t env_getpid(void) { return (uint32_t)getpid(); }

uint32_t env_fopen(uint32_t path, uint32_t mode) {
    const char *m = hoststr(mode);
    int fl = strchr(m, 'w') ? (O_CREAT | O_WRONLY | O_TRUNC)
           : strchr(m, 'a') ? (O_CREAT | O_WRONLY | O_APPEND) : O_RDONLY;
    int fd = open(hoststr(path), fl, 0666);
    return fd < 0 ? 0 : (uint32_t)fd;
}
uint32_t env_fclose(uint32_t f) { if ((int)f > 2) close((int)f); return 0; }
uint64_t env_fwrite(uint32_t ptr, uint64_t sz, uint64_t nm, uint32_t f) {
    size_t bytes = (size_t)(sz * nm);
    if (bytes) write((int)f, MEM + ptr, bytes);
    return nm;
}
uint32_t env_opendir(uint32_t path) { (void)path; return 0; }
uint32_t env_closedir(uint32_t d) { (void)d; return 0; }

// ---- string / number parsing (endptr slots hold a 4-byte pointer on wasm32) ----
uint32_t env_atoi(uint32_t p) { return (uint32_t)(int32_t)atoi(hoststr(p)); }
uint32_t env_strcmp(uint32_t a, uint32_t b) {
    int r = strcmp(hoststr(a), hoststr(b));
    return (uint32_t)(int32_t)(r < 0 ? -1 : (r > 0 ? 1 : 0));
}
uint64_t env_strtol(uint32_t nptr, uint32_t endptr, uint32_t base) {
    char *s = hoststr(nptr), *end = s;
    long v = strtol(s, &end, (int)base);
    if (endptr) { uint32_t off = nptr + (uint32_t)(end - s); memcpy(MEM + endptr, &off, 4); }
    return (uint64_t)(int64_t)v;
}
double env_strtod(uint32_t nptr, uint32_t endptr) {
    char *s = hoststr(nptr), *end = s;
    double v = strtod(s, &end);
    if (endptr) { uint32_t off = nptr + (uint32_t)(end - s); memcpy(MEM + endptr, &off, 4); }
    return v;
}

// ---- printf family ----
uint32_t env_printf(uint32_t fmt, uint32_t arg) {
    char buf[8192];
    size_t n = fmt_one(buf, sizeof buf, hoststr(fmt), arg);
    size_t w = n < sizeof buf - 1 ? n : sizeof buf - 1;
    fwrite(buf, 1, w, stdout);
    return (uint32_t)n;
}
uint32_t env_dprintf(uint32_t fd, uint32_t fmt, uint32_t arg) {
    char buf[8192];
    size_t n = fmt_one(buf, sizeof buf, hoststr(fmt), arg);
    size_t w = n < sizeof buf - 1 ? n : sizeof buf - 1;
    write((int)fd, buf, w);
    return (uint32_t)n;
}
uint32_t env_snprintf(uint32_t buf, uint64_t size, uint32_t fmt, uint32_t arg) {
    char tmp[8192];
    size_t n = fmt_one(tmp, sizeof tmp, hoststr(fmt), arg);
    if (size) {
        size_t w = n < (size_t)size - 1 ? n : (size_t)size - 1;
        memcpy(MEM + buf, tmp, w);
        MEM[buf + w] = 0;
    }
    return (uint32_t)n;
}
uint64_t env_putchar(uint64_t c) { putchar((int)(c & 0xff)); return c & 0xff; }
uint64_t env_puts(uint32_t p) { fputs(hoststr(p), stdout); putchar('\n'); return 1; }

// ---- math ----
double env_sqrt(double x) { return sqrt(x); }
double env_pow(double x, double y) { return pow(x, y); }
double env_fmod(double x, double y) { return fmod(x, y); }
float  env_fmodf(float x, float y) { return fmodf(x, y); }

// ---- process ----
uint64_t env_abort(void) { die("env.abort() called"); return 0; }
uint64_t env_exit(uint32_t code) { exit((int)(code & 0xff)); }
uint32_t env_system(uint32_t cmd) { return (uint32_t)system(hoststr(cmd)); }

// ---- threads: single-threaded no-ops ----
uint32_t env_pthread_mutex_init(uint32_t a, uint32_t b) { (void)a; (void)b; return 0; }
uint32_t env_pthread_mutex_lock(uint32_t a) { (void)a; return 0; }
uint32_t env_pthread_mutex_unlock(uint32_t a) { (void)a; return 0; }
uint32_t env_pthread_cond_init(uint32_t a, uint32_t b) { (void)a; (void)b; return 0; }
uint32_t env_pthread_cond_signal(uint32_t a) { (void)a; return 0; }
uint32_t env_pthread_cond_wait(uint32_t a, uint32_t b) { (void)a; (void)b; return 0; }
uint32_t env_pthread_attr_init(uint32_t a) { (void)a; return 0; }
uint32_t env_pthread_attr_setstacksize(uint32_t a, uint64_t b) { (void)a; (void)b; return 0; }

// ---- DEAD imports: abort LOUDLY (comptime is pure interpretation) ----
uint32_t env_pthread_create(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { (void)a;(void)b;(void)c;(void)d; die("unreachable: env.pthread_create"); return 0; }
uint32_t env_pthread_join(uint32_t a, uint32_t b) { (void)a;(void)b; die("unreachable: env.pthread_join"); return 0; }
uint64_t env_pthread_exit(uint32_t a) { (void)a; die("unreachable: env.pthread_exit"); return 0; }
uint32_t env_mmap(uint32_t a, uint64_t b, uint32_t c, uint32_t d, uint32_t e, uint64_t f) { (void)a;(void)b;(void)c;(void)d;(void)e;(void)f; die("unreachable: env.mmap"); return 0; }
uint32_t env_munmap(uint32_t a, uint64_t b) { (void)a;(void)b; die("unreachable: env.munmap"); return 0; }
uint32_t env_mprotect(uint32_t a, uint64_t b, uint32_t c) { (void)a;(void)b;(void)c; die("unreachable: env.mprotect"); return 0; }
uint32_t env_dlopen(uint32_t a, uint64_t b) { (void)a;(void)b; die("unreachable: env.dlopen"); return 0; }
uint32_t env_dlsym(uint32_t a, uint32_t b) { (void)a;(void)b; die("unreachable: env.dlsym"); return 0; }

// ===========================================================================
// driver
// ===========================================================================
int main(int argc, char **argv) {
    wasm_init();

    g_cap = *wasm___heap_base;                   // initial memory ends at heap_base
    g_brk = *wasm___heap_base;

    // argv = ["coil", user args...] as an array of i32 offsets to NUL strings.
    uint32_t *offs = malloc(sizeof(uint32_t) * (size_t)argc);
    for (int i = 0; i < argc; i++) {
        size_t n = strlen(argv[i]) + 1;
        uint32_t p = (uint32_t)rt_malloc(n);
        memcpy(MEM + p, argv[i], n);
        offs[i] = p;
    }
    uint32_t argv_off = (uint32_t)rt_malloc(sizeof(uint32_t) * (size_t)argc);
    for (int i = 0; i < argc; i++) memcpy(MEM + argv_off + (uint32_t)i * 4, &offs[i], 4);
    free(offs);

    uint64_t rc = wasm_main((uint32_t)argc, argv_off);
    return (int)(rc & 0xff);
}
