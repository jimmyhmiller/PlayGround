#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

// =============================================================================
// Pointer Intrinsics
// =============================================================================

void* ptr_load_ptr(void* p, int64_t off) {
    void* result;
    memcpy(&result, (char*)p + off, 8);
    return result;
}

int64_t ptr_store_ptr(void* p, int64_t off, void* val) {
    memcpy((char*)p + off, &val, 8);
    return 0;
}

int64_t ptr_load_i64(void* p, int64_t off) {
    int64_t result;
    memcpy(&result, (char*)p + off, 8);
    return result;
}

int64_t ptr_store_i64(void* p, int64_t off, int64_t val) {
    memcpy((char*)p + off, &val, 8);
    return 0;
}

int64_t ptr_load_i32(void* p, int64_t off) {
    int32_t result;
    memcpy(&result, (char*)p + off, 4);
    return (int64_t)result;
}

int64_t ptr_store_i32(void* p, int64_t off, int64_t val) {
    int32_t v = (int32_t)val;
    memcpy((char*)p + off, &v, 4);
    return 0;
}

int64_t ptr_load_i8(void* p, int64_t off) {
    return (int64_t)((unsigned char*)p)[off];
}

int64_t ptr_store_i8(void* p, int64_t off, int64_t val) {
    ((unsigned char*)p)[off] = (unsigned char)val;
    return 0;
}

void* ptr_offset(void* p, int64_t off) {
    return (char*)p + off;
}

// =============================================================================
// Process Arguments
// =============================================================================

static int g_argc = 0;
static char** g_argv = NULL;

void rt_init_args(int argc, char** argv) {
    g_argc = argc;
    g_argv = argv;
}

int64_t rt_get_argc(void) {
    return (int64_t)g_argc;
}

char** rt_get_argv(void) {
    return g_argv;
}

int64_t arg_i64(int64_t index) {
    if (index < 0) return 0;
    int64_t arg_index = index + 1;
    if (g_argv == NULL || arg_index >= (int64_t)g_argc) return 0;
    char* s = g_argv[arg_index];
    if (!s) return 0;
    char* end;
    long long val = strtoll(s, &end, 10);
    if (*end != '\0') return 0;
    return (int64_t)val;
}

int64_t arg_str_impl(int64_t index, const char** out) {
    if (index < 0) return 0;
    int64_t arg_index = index + 1;
    if (g_argv == NULL || arg_index >= (int64_t)g_argc) return 0;
    *out = g_argv[arg_index];
    return 1;
}

const char* arg_str(int64_t index) {
    const char* s = NULL;
    if (arg_str_impl(index, &s)) return s;
    return NULL;
}

int64_t arg_is_i64(int64_t index) {
    if (index < 0) return 0;
    int64_t arg_index = index + 1;
    if (g_argv == NULL || arg_index >= (int64_t)g_argc) return 0;
    char* s = g_argv[arg_index];
    if (!s) return 0;
    char* end;
    strtoll(s, &end, 10);
    return (*end == '\0') ? 1 : 0;
}

int64_t arg_len(void) {
    if (g_argv == NULL || g_argc <= 1) return 0;
    return (int64_t)(g_argc - 1);
}

// =============================================================================
// I/O
// =============================================================================

int64_t print_int(int64_t v) {
    printf("%lld\n", (long long)v);
    return 0;
}

int64_t print_str(const char* ptr) {
    if (!ptr) {
        printf("\n");
        return 0;
    }
    printf("%s\n", ptr);
    return 0;
}

int64_t print_str_stderr(const char* ptr) {
    if (!ptr) {
        fprintf(stderr, "\n");
        return 0;
    }
    fprintf(stderr, "%s\n", ptr);
    return 0;
}

void print_stretch(int64_t depth, int64_t check) {
    printf("stretch tree of depth %lld\t check: %lld\n", (long long)depth, (long long)check);
}

void print_trees(int64_t iterations, int64_t depth, int64_t check) {
    printf("%lld\t trees of depth %lld\t check: %lld\n", (long long)iterations, (long long)depth, (long long)check);
}

void print_long_lived(int64_t depth, int64_t check) {
    printf("long lived tree of depth %lld\t check: %lld\n", (long long)depth, (long long)check);
}

// =============================================================================
// Misc
// =============================================================================

int64_t add_i64(int64_t a, int64_t b) { return a + b; }
void* null_ptr(void) { return NULL; }
int64_t ptr_is_null(void* p) { return p == NULL ? 1 : 0; }

void exit_process(int64_t code) {
    exit((int)code);
}

int64_t system_cmd(const char* cmd) {
    if (!cmd) return 1;
    int status = system(cmd);
    if (status == -1) return 1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

int64_t create_dir(const char* path) {
    if (!path) return 1;
    /* Use mkdir -p style: try to create, ignore EEXIST */
    mkdir(path, 0755);
    return 0;
}

int64_t write_file(const char* path, const char* data, int64_t len) {
    if (!path || !data) return 1;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;
    write(fd, data, (size_t)len);
    close(fd);
    return 0;
}

// =============================================================================
// String Operations
// =============================================================================

int64_t string_len(const char* ptr) {
    if (!ptr) return 0;
    return (int64_t)strlen(ptr);
}

int64_t string_eq(const char* a, const char* b) {
    if (!a || !b) return (a == b) ? 1 : 0;
    return strcmp(a, b) == 0 ? 1 : 0;
}

const char* string_concat(const char* a, const char* b) {
    if (!a && !b) return NULL;
    size_t la = a ? strlen(a) : 0;
    size_t lb = b ? strlen(b) : 0;
    char* r = (char*)malloc(la + lb + 1);
    if (a) memcpy(r, a, la);
    if (b) memcpy(r + la, b, lb);
    r[la + lb] = '\0';
    return r;
}

const char* string_slice(const char* ptr, int64_t start, int64_t end_pos) {
    if (!ptr || start < 0 || end_pos < start) return NULL;
    int64_t len = (int64_t)strlen(ptr);
    if (start > len || end_pos > len) return NULL;
    int64_t n = end_pos - start;
    char* r = (char*)malloc((size_t)n + 1);
    memcpy(r, ptr + start, (size_t)n);
    r[n] = '\0';
    return r;
}

int64_t string_byte_at(const char* ptr, int64_t index) {
    if (!ptr || index < 0) return 0;
    int64_t len = (int64_t)strlen(ptr);
    if (index >= len) return 0;
    return (int64_t)(unsigned char)ptr[index];
}

const char* string_from_i64(int64_t val) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%lld", (long long)val);
    size_t len = strlen(buf);
    char* r = (char*)malloc(len + 1);
    memcpy(r, buf, len + 1);
    return r;
}

int64_t string_parse_i64(const char* ptr) {
    if (!ptr) return -1;
    char* end;
    long long val = strtoll(ptr, &end, 10);
    if (*end != '\0') return -1;
    return (int64_t)val;
}

// =============================================================================
// File I/O
// =============================================================================

const char* read_file(const char* path) {
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';
    return buf;
}

// =============================================================================
// File existence check
// =============================================================================

int64_t file_exists(const char* path) {
    if (!path) return 0;
    return access(path, F_OK) == 0 ? 1 : 0;
}

// =============================================================================
// Enum Tag (legacy — reads tag from field at given byte offset)
// With new GC layout, enum tags are regular fields accessed via gc_read_field_i64.
// This function is kept for JIT compatibility.
// =============================================================================

int64_t enum_tag(void* obj, int64_t raw_base) {
    if (!obj) return -1;
    int64_t tag;
    memcpy(&tag, (char*)obj + raw_base, 8);
    return tag;
}

// =============================================================================
// Stdlib Path (baked in at compile time)
// =============================================================================

#ifndef LANG_STDLIB_PATH
#define LANG_STDLIB_PATH "stdlib"
#endif

const char* get_stdlib_path(void) {
    return LANG_STDLIB_PATH;
}
