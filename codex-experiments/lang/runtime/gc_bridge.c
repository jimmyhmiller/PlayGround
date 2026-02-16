#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "gc_library.h"

/* =========================================================================
 * gc_bridge.c — Connects the lang runtime to gc-library's custom callback API
 *
 * Object header layout (8 bytes):
 *   offset 0: u8  mark_flag
 *   offset 1: u8  forwarded_flag  (NEW - for generational GC)
 *   offset 2: u16 type_id
 *   offset 4: u16 num_ptr_fields
 *   offset 6: u16 num_total_fields
 *
 * After header: num_total_fields * 8 bytes of field data.
 * Fields 0..num_ptr_fields-1 are GC pointers (tracer visits these).
 * Fields num_ptr_fields..num_total_fields-1 are raw values (not traced).
 *
 * FORWARDING: When forwarded_flag==1, the first field contains the forwarding
 * pointer instead of normal data. The object has been moved by generational GC.
 *
 * Frame chain layout:
 *   offset 0:  parent (ptr)
 *   offset 8:  origin (ptr)
 *   Roots start at offset 16 (one ptr per root)
 *
 * FrameOrigin layout:
 *   offset 0:  num_roots (i32)
 *   offset 4:  name (ptr)
 *
 * Root slots only contain GC pointers (never LLVM handles or C strings).
 * This is ensured by is_gc_ref_type returning false for RawPointer.
 * ========================================================================= */

#define HEADER_SIZE 8

static GcCustomHandle* gc = NULL;
static void* g_thread = NULL;

void gc_set_thread(void* thread) {
    g_thread = thread;
}

/* --- Root enumerator: walk Thread frame chain --- */

void lang_roots(void* user_ctx, GcCustomVisitor visit, void* gc_ctx) {
    if (!user_ctx) return;
    void* frame = *(void**)user_ctx;  /* thread->top_frame */
    while (frame) {
        void* origin = *(void**)((char*)frame + 8);
        int32_t num_roots = 0;
        if (origin) {
            memcpy(&num_roots, origin, 4);
        }
        void** roots = (void**)((char*)frame + 16);
        for (int32_t i = 0; i < num_roots; i++) {
            if (roots[i]) {
                visit(&roots[i], gc_ctx);
            }
        }
        frame = *(void**)frame;
    }
}

/* --- Object tracer: visit pointer fields using num_ptr_fields from header --- */

void lang_trace(void* obj, GcCustomVisitor visit, void* gc_ctx) {
    if (!obj) return;
    uint16_t num_ptr = *(uint16_t*)((uint8_t*)obj + 4);
    void** fields = (void**)((uint8_t*)obj + HEADER_SIZE);
    for (uint16_t i = 0; i < num_ptr; i++) {
        if (fields[i]) {
            visit(&fields[i], gc_ctx);
        }
    }
}

/* --- Mark bit: byte at offset 0 --- */

int lang_is_marked(void* obj) {
    return ((uint8_t*)obj)[0];
}

void lang_set_mark(void* obj, int m) {
    ((uint8_t*)obj)[0] = (uint8_t)(m ? 1 : 0);
}

/* --- Size: computed from num_total_fields at offset 6 --- */

void lang_get_size(void* obj, size_t* out) {
    uint16_t total = *(uint16_t*)((uint8_t*)obj + 6);
    *out = HEADER_SIZE + (size_t)total * 8;
}

/* --- Forwarding pointer support (for generational GC) --- */

int lang_is_forwarded(void* obj) {
    if (!obj) return 0;
    return ((uint8_t*)obj)[1];  /* forwarded_flag at offset 1 */
}

void* lang_get_forwarding(void* obj) {
    if (!obj) return NULL;
    /* When forwarded, first field contains new location */
    void** fields = (void**)((uint8_t*)obj + HEADER_SIZE);
    return fields[0];
}

void lang_set_forwarding(void* obj, void* new_location) {
    if (!obj) return;
    /* Set forwarded_flag */
    ((uint8_t*)obj)[1] = 1;
    /* Store new location in first field */
    void** fields = (void**)((uint8_t*)obj + HEADER_SIZE);
    fields[0] = new_location;
}

/* =========================================================================
 * Public API
 * ========================================================================= */

void gc_init(void) {
    GcCustomConfig cfg;
    cfg.root_enumerator = lang_roots;
    cfg.tracer = lang_trace;
    cfg.is_marked = lang_is_marked;
    cfg.set_mark = lang_set_mark;
    cfg.get_size = lang_get_size;
    /* NEW - forwarding pointer support for generational GC */
    cfg.is_forwarded = lang_is_forwarded;
    cfg.get_forwarding = lang_get_forwarding;
    cfg.set_forwarding = lang_set_forwarding;
    cfg.initial_heap = 32 * 1024 * 1024;  /* 32 MB */
    cfg.strategy = GC_STRATEGY_GENERATIONAL;  /* Now using generational GC! */
    gc = gc_lib_custom_create(cfg);
}

/* Allocate a GC object.
 * total_fields: number of 8-byte fields
 * ptr_fields: number of GC pointer fields (must come first in layout)
 * type_id: runtime type identifier */
void* gc_alloc(int64_t total_fields, int64_t ptr_fields, int64_t type_id) {
    size_t size = HEADER_SIZE + (size_t)total_fields * 8;
    void* obj = gc_lib_custom_allocate(gc, size, g_thread);
    if (!obj) return NULL;
    /* Write header */
    ((uint8_t*)obj)[0] = 0;                                    /* mark = 0 */
    ((uint8_t*)obj)[1] = 0;                                    /* pad */
    *(uint16_t*)((uint8_t*)obj + 2) = (uint16_t)type_id;       /* type_id */
    *(uint16_t*)((uint8_t*)obj + 4) = (uint16_t)ptr_fields;    /* num_ptr_fields */
    *(uint16_t*)((uint8_t*)obj + 6) = (uint16_t)total_fields;  /* num_total_fields */
    return obj;
}

/* Read a pointer field */
void* gc_read_field(void* obj, int64_t index) {
    return ((void**)((uint8_t*)obj + HEADER_SIZE))[index];
}

/* Write a pointer field */
int64_t gc_write_field(void* obj, int64_t index, void* val) {
    ((void**)((uint8_t*)obj + HEADER_SIZE))[index] = val;
    return 0;
}

/* Read an i64 field */
int64_t gc_read_field_i64(void* obj, int64_t index) {
    int64_t val;
    memcpy(&val, (uint8_t*)obj + HEADER_SIZE + index * 8, 8);
    return val;
}

/* Write an i64 field */
int64_t gc_write_field_i64(void* obj, int64_t index, int64_t val) {
    memcpy((uint8_t*)obj + HEADER_SIZE + index * 8, &val, 8);
    return 0;
}

void gc_write_barrier(void* thread, void* obj, void* val) {
    (void)thread;
    gc_lib_custom_write_barrier(gc, obj, val);
}

void gc_pollcheck_slow(void* thread, void* origin) {
    (void)origin;
    if (thread) {
        int32_t zero = 0;
        memcpy((char*)thread + 8, &zero, 4);
    }
    if (gc_lib_custom_should_collect(gc)) {
        gc_lib_custom_collect(gc, thread);
    }
}

/* =========================================================================
 * Vec Operations — GC-allocated vector objects
 *
 * Vec layout (3 fields, type_id=0):
 *   field 0: data ptr (GC ref) — pointer to data array GC object
 *   field 1: len (raw i64)
 *   field 2: cap (raw i64)
 *   ptr_field_count = 1, total_field_count = 3
 * ========================================================================= */

#define VEC_DATA_FIELD 0
#define VEC_LEN_FIELD  1
#define VEC_CAP_FIELD  2

void* vec_new(void) {
    return gc_alloc(3, 1, 0);
}

int64_t vec_len(void* v) {
    return gc_read_field_i64(v, VEC_LEN_FIELD);
}

int64_t vec_cap(void* v) {
    return gc_read_field_i64(v, VEC_CAP_FIELD);
}

void* vec_data(void* v) {
    void* data_obj = gc_read_field(v, VEC_DATA_FIELD);
    return (uint8_t*)data_obj + HEADER_SIZE;
}

/* Grow data array for GC-ref elements (all fields are ptr fields) */
static void vec_grow_ptr(void* v, int64_t len, int64_t cap) {
    int64_t new_cap = (cap == 0) ? 8 : cap * 2;
    void* new_data = gc_alloc(new_cap, new_cap, 0);
    void* old_data = gc_read_field(v, VEC_DATA_FIELD);
    for (int64_t i = 0; i < len; i++) {
        void* elem = gc_read_field(old_data, i);
        gc_write_field(new_data, i, elem);
    }
    gc_write_field(v, VEC_DATA_FIELD, new_data);
    gc_write_field_i64(v, VEC_CAP_FIELD, new_cap);
}

/* Grow data array for raw elements (no ptr fields) */
static void vec_grow_raw(void* v, int64_t len, int64_t cap) {
    int64_t new_cap = (cap == 0) ? 8 : cap * 2;
    void* new_data = gc_alloc(new_cap, 0, 0);
    void* old_data = gc_read_field(v, VEC_DATA_FIELD);
    for (int64_t i = 0; i < len; i++) {
        void* elem = gc_read_field(old_data, i);
        gc_write_field(new_data, i, elem);
    }
    gc_write_field(v, VEC_DATA_FIELD, new_data);
    gc_write_field_i64(v, VEC_CAP_FIELD, new_cap);
}

/* Push a GC-ref element */
int64_t vec_push(void* v, void* item) {
    int64_t len = gc_read_field_i64(v, VEC_LEN_FIELD);
    int64_t cap = gc_read_field_i64(v, VEC_CAP_FIELD);
    if (len == cap) {
        vec_grow_ptr(v, len, cap);
    }
    void* data = gc_read_field(v, VEC_DATA_FIELD);
    gc_write_field(data, len, item);
    int64_t new_len = len + 1;
    gc_write_field_i64(v, VEC_LEN_FIELD, new_len);
    return new_len;
}

/* Push a raw (non-GC) element (String, RawPointer, etc.) */
int64_t vec_push_raw_val(void* v, void* item) {
    int64_t len = gc_read_field_i64(v, VEC_LEN_FIELD);
    int64_t cap = gc_read_field_i64(v, VEC_CAP_FIELD);
    if (len == cap) {
        vec_grow_raw(v, len, cap);
    }
    void* data = gc_read_field(v, VEC_DATA_FIELD);
    gc_write_field(data, len, item);
    int64_t new_len = len + 1;
    gc_write_field_i64(v, VEC_LEN_FIELD, new_len);
    return new_len;
}

void* vec_get(void* v, int64_t index) {
    void* data = gc_read_field(v, VEC_DATA_FIELD);
    return gc_read_field(data, index);
}

int64_t vec_clear(void* v) {
    gc_write_field_i64(v, VEC_LEN_FIELD, 0);
    return 0;
}

int64_t vec_set_len(void* v, int64_t new_len) {
    int64_t len = gc_read_field_i64(v, VEC_LEN_FIELD);
    if (new_len < 0) {
        gc_write_field_i64(v, VEC_LEN_FIELD, 0);
    } else if (new_len < len) {
        gc_write_field_i64(v, VEC_LEN_FIELD, new_len);
    }
    return 0;
}

int64_t vec_is_empty(void* v) {
    return gc_read_field_i64(v, VEC_LEN_FIELD) == 0 ? 1 : 0;
}
