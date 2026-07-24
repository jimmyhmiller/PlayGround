// runtime.c — the C runtime + driver for the wasm2c-translated Coil compiler.
//
// coilc.wasm (a memory64 module) is translated to coilc.c by our extended
// wasm2c; this file provides the ~56 `env.*` imports it needs and the main()
// that sets up argv and calls the module's exported `main`. It is the C
// analogue of wasm-host/run-coil-wasm.mjs.
//
// LINEAR MEMORY. The module owns one linear-memory buffer, exported as
// `wasm_memory` (a `uint8_t**`). Every pointer the module hands us is an
// OFFSET into that buffer, not a host pointer; `MEM` converts. Because the
// buffer is ordinary host memory, `MEM + off` is a valid host `char*`, so we
// can pass it straight to real libc string/FS calls.
//
// ALLOCATOR. The module imports malloc/free/realloc/calloc; we implement a
// bump allocator with a per-size free list over the linear-memory buffer,
// starting at the exported `__heap_base`, growing the buffer (realloc) as
// needed. This mirrors run-coil-wasm.mjs: a plain bump allocator leaks
// catastrophically because the bytecode interpreter mallocs and frees a large
// frame buffer per call, so a reclaiming free list is required for a
// self-compile to fit in memory. All bookkeeping uses OFFSETS, which survive
// the realloc that moves the host buffer.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

// ---- module interface (defined in the generated coilc.c) -------------------
extern uint8_t **const wasm_memory;          // &m0 : the linear-memory buffer
extern const uint64_t *const wasm___heap_base;      // heap start offset
extern void wasm_init(void);                 // allocate memory + data segments
extern uint64_t wasm_main(uint32_t argc, uint64_t argv_off);

#define MEM (*wasm_memory)                    // current host base of linear mem
static uint8_t *hostp(uint64_t off) { return MEM + off; }
static char *hoststr(uint64_t off) { return (char *)(MEM + off); }

static void die(const char *msg) { fputs(msg, stderr); fputc('\n', stderr); abort(); }

// ---------------------------------------------------------------------------
// Allocator over linear memory
// ---------------------------------------------------------------------------
// Each block carries a 16-byte header immediately before the returned offset:
//   header[0..8)  = rounded payload size
//   header[8..16) = MAGIC while live / free-list `next` offset while free
// Returned pointer = header_off + 16 (16-byte aligned when header is).
#define HDR 16u
#define ALIGN 16u
#define MAGIC UINT64_C(0xC011C0DEC0FFEE01)

static uint64_t g_brk;        // next unused offset (bump pointer)
static uint64_t g_cap;        // bytes currently backed by the host buffer

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + (a - 1)) & ~(a - 1); }

// header field accessors (offset-based; recompute MEM each time)
static uint64_t hdr_size_get(uint64_t h) { uint64_t v; memcpy(&v, MEM + h, 8); return v; }
static void     hdr_size_set(uint64_t h, uint64_t v) { memcpy(MEM + h, &v, 8); }
static uint64_t hdr_tag_get(uint64_t h) { uint64_t v; memcpy(&v, MEM + h + 8, 8); return v; }
static void     hdr_tag_set(uint64_t h, uint64_t v) { memcpy(MEM + h + 8, &v, 8); }

// free-list bins: a host-side open-addressing map  rounded_size -> head offset.
// Heads are offsets into linear memory (stable across buffer realloc).
#define NBINS (1u << 17)
static uint64_t bin_key[NBINS];   // rounded size (0 = empty slot)
static uint64_t bin_head[NBINS];  // head block-header offset (0 = empty list)

static uint64_t *bin_slot(uint64_t size) {
    uint64_t i = (size * UINT64_C(0x9E3779B97F4A7C15)) >> 47;
    for (uint32_t probe = 0; probe < NBINS; probe++) {
        uint64_t j = (i + probe) & (NBINS - 1);
        if (bin_key[j] == 0 || bin_key[j] == size) { bin_key[j] = size; return &bin_head[j]; }
    }
    die("bootstrap allocator: free-bin table full");
    return NULL;
}

// grow the linear-memory buffer so that byte offset `end` is backed.
static void ensure(uint64_t end) {
    if (end <= g_cap) return;
    uint64_t ncap = g_cap + g_cap / 2;
    if (ncap < end) ncap = end;
    ncap = align_up(ncap, 65536);               // whole wasm pages
    uint8_t *nm = realloc(MEM, (size_t)ncap);
    if (nm == NULL) die("bootstrap allocator: out of memory (realloc failed)");
    memset(nm + g_cap, 0, (size_t)(ncap - g_cap));  // zero the fresh region
    MEM = nm;                                    // publish new base to module
    g_cap = ncap;
}

static uint64_t rt_malloc(uint64_t size) {
    uint64_t rounded = align_up(size == 0 ? 1 : size, ALIGN);
    uint64_t *slot = bin_slot(rounded);
    if (*slot != 0) {                            // reuse a freed block
        uint64_t h = *slot;
        *slot = hdr_tag_get(h);                  // pop: head = next
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
    if (hdr_tag_get(h) != MAGIC) return;         // unknown / double free: leak (as JS host)
    uint64_t size = hdr_size_get(h);
    uint64_t *slot = bin_slot(size);
    hdr_tag_set(h, *slot);                        // push: block.next = head
    *slot = h;
}

static uint64_t rt_realloc(uint64_t ptr, uint64_t size) {
    if (ptr == 0) return rt_malloc(size);
    uint64_t h = ptr - HDR;
    uint64_t old = (hdr_tag_get(h) == MAGIC) ? hdr_size_get(h) : 0;
    uint64_t rounded = align_up(size == 0 ? 1 : size, ALIGN);
    if (old >= rounded) return ptr;              // fits in place
    uint64_t np = rt_malloc(size);
    uint64_t n = old < size ? old : size;
    if (n) memmove(MEM + np, MEM + ptr, (size_t)n);
    rt_free(ptr);
    return np;
}

// ---------------------------------------------------------------------------
// printf-family: the module's printf/dprintf/snprintf imports take EXACTLY one
// variadic argument (fixed wasm signature), so a format has at most one
// value-consuming conversion. We format literals + %% ourselves and delegate
// each real conversion to host snprintf (byte-identical to native libc); %s
// reads its string straight out of linear memory.
// ---------------------------------------------------------------------------
static size_t fmt_one(char *out, size_t cap, const char *fmt, uint64_t arg) {
    size_t o = 0;
    int used = 0;
    #define PUT(ch) do { if (o + 1 < cap) out[o] = (ch); o++; } while (0)
    for (const char *p = fmt; *p; ) {
        if (*p != '%') { PUT(*p); p++; continue; }
        const char *start = p++;                 // at conversion after '%'
        if (*p == '%') { PUT('%'); p++; continue; }
        while (*p == 'l' || *p == 'h' || *p == 'z' || *p == 'j' || *p == 't') p++;
        char conv = *p ? *p++ : 0;
        char tmp[64];
        int n = 0;
        uint64_t a = used ? 0 : arg;             // only the first conversion has the arg
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
            case 's': {                          // string lives in linear memory
                const char *s = hoststr(a);
                for (const char *q = s; *q; q++) PUT(*q);
                p = p; continue;                 // handled inline
            }
            default:                             // unknown: echo verbatim
                for (const char *q = start; q < p; q++) PUT(*q);
                continue;
        }
        for (int i = 0; i < n; i++) PUT(tmp[i]);
    }
    if (cap) out[o < cap ? o : cap - 1] = 0;
    #undef PUT
    return o;                                    // length excluding NUL
}

// ===========================================================================
// env.* imports
// ===========================================================================

// ---- allocation ----
uint64_t env_malloc(uint64_t size) { return rt_malloc(size); }
uint64_t env_realloc(uint64_t p, uint64_t size) { return rt_realloc(p, size); }
uint64_t env_free(uint64_t p) { rt_free(p); return 0; }
uint64_t env_calloc(uint64_t n, uint64_t sz) {
    uint64_t total = n * sz; if (total == 0) total = 1;
    uint64_t p = rt_malloc(total);
    memset(MEM + p, 0, (size_t)total);
    return p;
}
uint64_t env_memset(uint64_t s, uint64_t c, uint64_t n) { memset(MEM + s, (int)c, (size_t)n); return s; }
uint64_t env_memcmp(uint64_t a, uint64_t b, uint64_t n) {
    int r = memcmp(MEM + a, MEM + b, (size_t)n);
    return (uint64_t)(int64_t)(r < 0 ? -1 : (r > 0 ? 1 : 0));
}
uint64_t env_strlen(uint64_t p) { return (uint64_t)strlen(hoststr(p)); }

// ---- file / directory I/O (guest O_* constants == host's; same OS) ----
uint32_t env_open(uint64_t path, uint32_t flags) { return (uint32_t)open(hoststr(path), (int)flags, 0666); }
uint64_t env_creat(uint64_t path, uint64_t mode) {
    return (uint64_t)(int64_t)open(hoststr(path), O_CREAT | O_WRONLY | O_TRUNC, (mode_t)mode);
}
uint64_t env_read(uint32_t fd, uint64_t ptr, uint64_t len) {
    return (uint64_t)(int64_t)read((int)fd, MEM + ptr, (size_t)len);
}
uint64_t env_write(uint32_t fd, uint64_t ptr, uint64_t len) {
    return (uint64_t)(int64_t)write((int)fd, MEM + ptr, (size_t)len);
}
uint32_t env_close(uint32_t fd) { if (fd > 2) close((int)fd); return 0; }
uint32_t env_access(uint64_t path, uint64_t mode) { return (uint32_t)access(hoststr(path), (int)mode); }
uint32_t env_unlink(uint64_t path) { return (uint32_t)unlink(hoststr(path)); }
uint32_t env_rename(uint64_t a, uint64_t b) { return (uint32_t)rename(hoststr(a), hoststr(b)); }
uint64_t env_realpath(uint64_t path, uint64_t out) {
    char buf[4096];
    if (realpath(hoststr(path), buf) == NULL) return 0;
    size_t n = strlen(buf) + 1;
    memcpy(MEM + out, buf, n);
    return out;
}
uint64_t env_getcwd(uint64_t buf, uint64_t size) {
    if (getcwd((char *)(MEM + buf), (size_t)size) == NULL) return 0;
    return buf;
}
// getenv: the reference JS host returns NULL for every variable and still
// produces a byte-identical self-build, so mirror that exactly.
uint64_t env_getenv(uint64_t name) { (void)name; return 0; }
uint32_t env_getpid(void) { return (uint32_t)getpid(); }

// stdio FILE* is modeled as the underlying fd (opaque handle round-trips).
uint64_t env_fopen(uint64_t path, uint64_t mode) {
    const char *m = hoststr(mode);
    int fl = strchr(m, 'w') ? (O_CREAT | O_WRONLY | O_TRUNC)
           : strchr(m, 'a') ? (O_CREAT | O_WRONLY | O_APPEND) : O_RDONLY;
    int fd = open(hoststr(path), fl, 0666);
    return fd < 0 ? 0 : (uint64_t)fd;
}
uint32_t env_fclose(uint64_t f) { if ((int)f > 2) close((int)f); return 0; }
uint64_t env_fwrite(uint64_t ptr, uint64_t sz, uint64_t nm, uint64_t f) {
    size_t bytes = (size_t)(sz * nm);
    if (bytes) write((int)f, MEM + ptr, bytes);
    return nm;
}
uint64_t env_opendir(uint64_t path) { (void)path; return 0; }
uint32_t env_closedir(uint64_t d) { (void)d; return 0; }

// ---- string / number parsing (linear memory is host-addressable) ----
uint32_t env_atoi(uint64_t p) { return (uint32_t)(int32_t)atoi(hoststr(p)); }
uint32_t env_strcmp(uint64_t a, uint64_t b) {
    int r = strcmp(hoststr(a), hoststr(b));
    return (uint32_t)(int32_t)(r < 0 ? -1 : (r > 0 ? 1 : 0));
}
uint64_t env_strtol(uint64_t nptr, uint64_t endptr, uint32_t base) {
    char *s = hoststr(nptr), *end = s;
    long v = strtol(s, &end, (int)base);
    if (endptr) { uint64_t off = nptr + (uint64_t)(end - s); memcpy(MEM + endptr, &off, 8); }
    return (uint64_t)(int64_t)v;
}
// strtod MUST set *endptr (the reader rejects a token whose consumed length
// != its length); linear memory is a valid host char*, so use real strtod.
double env_strtod(uint64_t nptr, uint64_t endptr) {
    char *s = hoststr(nptr), *end = s;
    double v = strtod(s, &end);
    if (endptr) { uint64_t off = nptr + (uint64_t)(end - s); memcpy(MEM + endptr, &off, 8); }
    return v;
}

// ---- printf family ----
uint32_t env_printf(uint64_t fmt, uint64_t arg) {
    char buf[8192];
    size_t n = fmt_one(buf, sizeof buf, hoststr(fmt), arg);
    size_t w = n < sizeof buf - 1 ? n : sizeof buf - 1;
    fwrite(buf, 1, w, stdout);
    return (uint32_t)n;
}
uint32_t env_dprintf(uint32_t fd, uint64_t fmt, uint64_t arg) {
    char buf[8192];
    size_t n = fmt_one(buf, sizeof buf, hoststr(fmt), arg);
    size_t w = n < sizeof buf - 1 ? n : sizeof buf - 1;
    write((int)fd, buf, w);
    return (uint32_t)n;
}
uint32_t env_snprintf(uint64_t buf, uint64_t size, uint64_t fmt, uint64_t arg) {
    char tmp[8192];
    size_t n = fmt_one(tmp, sizeof tmp, hoststr(fmt), arg);   // full length (excl NUL)
    if (size) {
        size_t w = n < (size_t)size - 1 ? n : (size_t)size - 1;
        memcpy(MEM + buf, tmp, w);
        MEM[buf + w] = 0;
    }
    return (uint32_t)n;
}
uint64_t env_putchar(uint64_t c) { putchar((int)(c & 0xff)); return c & 0xff; }
uint64_t env_puts(uint64_t p) { fputs(hoststr(p), stdout); putchar('\n'); return 1; }

// ---- math ----
double env_sqrt(double x) { return sqrt(x); }
double env_pow(double x, double y) { return pow(x, y); }
double env_fmod(double x, double y) { return fmod(x, y); }
float  env_fmodf(float x, float y) { return fmodf(x, y); }

// ---- process ----
uint64_t env_abort(void) { die("env.abort() called"); return 0; }
uint64_t env_exit(uint32_t code) { exit((int)(code & 0xff)); }
// system runs the host toolchain (the final `cc` link) — a build service, the
// same role the FS imports play. Real wait()-style status is returned.
uint32_t env_system(uint64_t cmd) { return (uint32_t)system(hoststr(cmd)); }

// ---- threads: single-threaded no-ops (called during metaengine setup) ----
uint32_t env_pthread_mutex_init(uint64_t a, uint64_t b) { (void)a; (void)b; return 0; }
uint32_t env_pthread_mutex_lock(uint64_t a) { (void)a; return 0; }
uint32_t env_pthread_mutex_unlock(uint64_t a) { (void)a; return 0; }
uint32_t env_pthread_cond_init(uint64_t a, uint64_t b) { (void)a; (void)b; return 0; }
uint32_t env_pthread_cond_signal(uint64_t a) { (void)a; return 0; }
uint32_t env_pthread_cond_wait(uint64_t a, uint64_t b) { (void)a; (void)b; return 0; }
uint32_t env_pthread_attr_init(uint64_t a) { (void)a; return 0; }
uint32_t env_pthread_attr_setstacksize(uint64_t a, uint64_t b) { (void)a; (void)b; return 0; }

// ---- DEAD imports: comptime is pure interpretation, so real threads, native
// JIT/dylib and raw mmap are never reached. Abort LOUDLY (not a silent no-op)
// so a regression that starts calling them is caught immediately.
uint32_t env_pthread_create(uint64_t a, uint64_t b, uint64_t c, uint64_t d) { (void)a;(void)b;(void)c;(void)d; die("unreachable: env.pthread_create"); return 0; }
uint32_t env_pthread_join(uint64_t a, uint64_t b) { (void)a;(void)b; die("unreachable: env.pthread_join"); return 0; }
uint64_t env_pthread_exit(uint64_t a) { (void)a; die("unreachable: env.pthread_exit"); return 0; }
uint64_t env_mmap(uint64_t a, uint64_t b, uint32_t c, uint32_t d, uint32_t e, uint64_t f) { (void)a;(void)b;(void)c;(void)d;(void)e;(void)f; die("unreachable: env.mmap"); return 0; }
uint32_t env_munmap(uint64_t a, uint64_t b) { (void)a;(void)b; die("unreachable: env.munmap"); return 0; }
uint32_t env_mprotect(uint64_t a, uint64_t b, uint32_t c) { (void)a;(void)b;(void)c; die("unreachable: env.mprotect"); return 0; }
uint64_t env_dlopen(uint64_t a, uint64_t b) { (void)a;(void)b; die("unreachable: env.dlopen"); return 0; }
uint64_t env_dlsym(uint64_t a, uint64_t b) { (void)a;(void)b; die("unreachable: env.dlsym"); return 0; }

// ===========================================================================
// driver
// ===========================================================================
int main(int argc, char **argv) {
    wasm_init();                                 // allocate linear memory + data

    g_cap = *wasm___heap_base;                   // initial memory ends at heap_base
    g_brk = *wasm___heap_base;                   // heap grows upward from there

    // argv = ["coil", user args...] laid out in linear memory as an array of
    // i64 offsets to NUL-terminated strings.
    uint64_t *offs = malloc(sizeof(uint64_t) * (size_t)argc);
    for (int i = 0; i < argc; i++) {
        size_t n = strlen(argv[i]) + 1;
        uint64_t p = rt_malloc(n);
        memcpy(MEM + p, argv[i], n);
        offs[i] = p;
    }
    uint64_t argv_off = rt_malloc(sizeof(uint64_t) * (size_t)argc);
    for (int i = 0; i < argc; i++) memcpy(MEM + argv_off + (uint64_t)i * 8, &offs[i], 8);
    free(offs);

    uint64_t rc = wasm_main((uint32_t)argc, argv_off);
    return (int)(rc & 0xff);
}
