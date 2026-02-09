/**
 * gc_library.h - C Interface for the Rust GC Library
 *
 * This header provides a C-compatible API for the garbage collector.
 * The library supports multiple GC algorithms:
 * - Mark-and-sweep (simple, non-moving)
 * - Generational (with write barriers for better performance)
 *
 * Thread-safe variants are also available.
 *
 * Usage:
 *   1. Create a GC handle with gc_lib_create_*()
 *   2. Allocate objects with gc_lib_allocate() or gc_lib_try_allocate()
 *   3. Access fields with gc_lib_read_field() and gc_lib_write_field()
 *   4. For generational GC, call gc_lib_write_barrier() after pointer writes
 *   5. Clean up with gc_lib_destroy()
 */

#ifndef GC_LIBRARY_H
#define GC_LIBRARY_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * Opaque Types
 * ========================================================================== */

/**
 * Opaque handle to a GC instance.
 * Create with gc_lib_create_* functions.
 */
typedef struct GcHandle GcHandle;

/* ==========================================================================
 * Callback Types
 * ========================================================================== */

/**
 * Callback function for reporting a single root.
 *
 * @param slot_addr  Address of the memory location containing the root.
 *                   The GC may update this location if the object moves.
 * @param value      The tagged pointer value at that slot.
 */
typedef void (*GcRootCallback)(uintptr_t slot_addr, uintptr_t value);

/**
 * User-provided function to enumerate all GC roots.
 *
 * Called by the GC during collection. The user must call the callback
 * for each root slot that may contain a heap pointer.
 *
 * @param context   User-provided context pointer.
 * @param callback  Function to call for each root.
 */
typedef void (*GcRootEnumerator)(void* context, GcRootCallback callback);

/* ==========================================================================
 * GC Creation and Destruction
 * ========================================================================== */

/**
 * Create a new mark-and-sweep garbage collector.
 *
 * Mark-and-sweep is simple and reliable, but may cause fragmentation
 * over time.
 *
 * @return  Pointer to a new GC handle, or NULL on failure.
 */
GcHandle* gc_lib_create_mark_sweep(void);

/**
 * Create a new mark-and-sweep GC with custom options.
 *
 * @param gc_enabled   If false, GC never runs (useful for debugging).
 * @param print_stats  If true, print timing statistics after each GC.
 * @return  Pointer to a new GC handle, or NULL on failure.
 */
GcHandle* gc_lib_create_mark_sweep_with_options(bool gc_enabled, bool print_stats);

/**
 * Create a new generational garbage collector.
 *
 * Generational GC is more efficient for typical workloads where most
 * objects die young. Requires write barriers after pointer stores.
 *
 * @return  Pointer to a new GC handle, or NULL on failure.
 */
GcHandle* gc_lib_create_generational(void);

/**
 * Create a new generational GC with custom options.
 *
 * @param gc_enabled   If false, GC never runs (useful for debugging).
 * @param print_stats  If true, print timing statistics after each GC.
 * @return  Pointer to a new GC handle, or NULL on failure.
 */
GcHandle* gc_lib_create_generational_with_options(bool gc_enabled, bool print_stats);

/**
 * Create a thread-safe mark-and-sweep garbage collector.
 *
 * Uses mutex synchronization for safe concurrent access.
 *
 * @return  Pointer to a new GC handle, or NULL on failure.
 */
GcHandle* gc_lib_create_mark_sweep_threadsafe(void);

/**
 * Create a thread-safe generational garbage collector.
 *
 * Uses mutex synchronization for safe concurrent access.
 *
 * @return  Pointer to a new GC handle, or NULL on failure.
 */
GcHandle* gc_lib_create_generational_threadsafe(void);

/**
 * Destroy a GC instance and free its resources.
 *
 * @param gc  GC handle to destroy. May be NULL (no-op).
 */
void gc_lib_destroy(GcHandle* gc);

/* ==========================================================================
 * Allocation
 * ========================================================================== */

/**
 * Try to allocate a new object.
 *
 * This function does not run GC automatically. If it returns NULL,
 * you should call gc_lib_collect() and retry.
 *
 * @param gc           GC handle.
 * @param field_count  Number of pointer-sized fields in the object.
 * @return  Pointer to the allocated object, or NULL if GC is needed.
 */
const uint8_t* gc_lib_try_allocate(GcHandle* gc, size_t field_count);

/**
 * Allocate a new object, running GC if needed.
 *
 * This is a convenience function that handles the GC loop internally.
 *
 * @param gc               GC handle.
 * @param field_count      Number of pointer-sized fields in the object.
 * @param root_enumerator  Function to enumerate all roots (called during GC).
 * @param context          User context passed to root_enumerator.
 * @return  Pointer to the allocated object, or NULL on fatal error.
 */
const uint8_t* gc_lib_allocate(
    GcHandle* gc,
    size_t field_count,
    GcRootEnumerator root_enumerator,
    void* context
);

/**
 * Initialize an allocated object's fields to zero.
 *
 * @param obj          Pointer to the object.
 * @param field_count  Number of fields to zero.
 */
void gc_lib_zero_fields(uint8_t* obj, size_t field_count);

/* ==========================================================================
 * Object Field Access
 * ========================================================================== */

/**
 * Read a field from an object.
 *
 * @param obj          Pointer to the object.
 * @param field_index  Index of the field (0-based).
 * @return  The field value (a tagged pointer).
 */
uintptr_t gc_lib_read_field(const uint8_t* obj, size_t field_index);

/**
 * Write a field in an object.
 *
 * For generational GC, call gc_lib_write_barrier() after this.
 *
 * @param obj          Pointer to the object.
 * @param field_index  Index of the field (0-based).
 * @param value        The value to write (a tagged pointer).
 */
void gc_lib_write_field(uint8_t* obj, size_t field_index, uintptr_t value);

/* ==========================================================================
 * Garbage Collection
 * ========================================================================== */

/**
 * Run garbage collection.
 *
 * @param gc               GC handle.
 * @param root_enumerator  Function to enumerate all roots.
 * @param context          User context passed to root_enumerator.
 */
void gc_lib_collect(GcHandle* gc, GcRootEnumerator root_enumerator, void* context);

/**
 * Grow the heap to accommodate more objects.
 *
 * Call this if allocation still fails after GC.
 *
 * @param gc  GC handle.
 */
void gc_lib_grow(GcHandle* gc);

/* ==========================================================================
 * Write Barriers (for Generational GC)
 * ========================================================================== */

/**
 * Write barrier for generational GC.
 *
 * Call this after writing a pointer into a heap object's field.
 * No-op for mark-and-sweep GC.
 *
 * @param gc          GC handle.
 * @param object_ptr  Tagged pointer to the object being written to.
 * @param new_value   The new value being written (tagged pointer).
 */
void gc_lib_write_barrier(GcHandle* gc, uintptr_t object_ptr, uintptr_t new_value);

/**
 * Mark a card unconditionally.
 *
 * Used by JIT-generated write barriers. No-op for mark-and-sweep GC.
 *
 * @param gc          GC handle.
 * @param object_ptr  Address of the object being written to.
 */
void gc_lib_mark_card(GcHandle* gc, uintptr_t object_ptr);

/**
 * Get young generation bounds for inline write barrier checks.
 *
 * For generational GC, returns the memory range of the young generation.
 * Objects in young gen don't need write barriers.
 *
 * For mark-and-sweep, both values will be 0.
 *
 * @param gc         GC handle.
 * @param start_out  Output: start address of young generation.
 * @param end_out    Output: end address of young generation.
 */
void gc_lib_get_young_gen_bounds(GcHandle* gc, uintptr_t* start_out, uintptr_t* end_out);

/**
 * Get the card table biased pointer for JIT write barriers.
 *
 * @param gc  GC handle.
 * @return  Biased pointer for card marking, or NULL for non-generational GCs.
 */
uint8_t* gc_lib_get_card_table_ptr(GcHandle* gc);

/* ==========================================================================
 * Tagged Pointer Utilities
 * ========================================================================== */

/**
 * Create a tagged heap pointer from a raw object pointer.
 *
 * @param raw_ptr  Raw pointer to an object (as returned by allocate).
 * @return  Tagged pointer value for storing in object fields.
 */
uintptr_t gc_lib_tag_heap_ptr(const uint8_t* raw_ptr);

/**
 * Create a tagged integer value.
 *
 * @param value  Integer value to tag.
 * @return  Tagged integer value.
 */
uintptr_t gc_lib_tag_int(int64_t value);

/**
 * Create a null tagged value.
 *
 * @return  Tagged null value.
 */
uintptr_t gc_lib_tag_null(void);

/**
 * Extract the raw pointer from a tagged heap pointer.
 *
 * @param tagged  Tagged pointer value.
 * @return  Raw pointer, or NULL if not a heap pointer.
 */
const uint8_t* gc_lib_untag_ptr(uintptr_t tagged);

/**
 * Extract an integer from a tagged integer value.
 *
 * @param tagged    Tagged pointer value.
 * @param value_out Output: the extracted integer value.
 * @return  true if the value was an integer, false otherwise.
 */
bool gc_lib_untag_int(uintptr_t tagged, int64_t* value_out);

/**
 * Check if a tagged value is a heap pointer.
 *
 * @param tagged  Tagged pointer value.
 * @return  true if heap pointer, false otherwise.
 */
bool gc_lib_is_heap_ptr(uintptr_t tagged);

/**
 * Check if a tagged value is null.
 *
 * @param tagged  Tagged pointer value.
 * @return  true if null, false otherwise.
 */
bool gc_lib_is_null(uintptr_t tagged);

/**
 * Check if a tagged value is an integer.
 *
 * @param tagged  Tagged pointer value.
 * @return  true if integer, false otherwise.
 */
bool gc_lib_is_int(uintptr_t tagged);

/* ==========================================================================
 * Constants
 * ========================================================================== */

/** Get the header size in bytes. */
size_t gc_lib_header_size(void);

/** Get the word size in bytes. */
size_t gc_lib_word_size(void);

/** Get the number of tag bits used. */
size_t gc_lib_tag_bits(void);

/* Constant values (also available as functions above) */
#define GC_LIB_HEADER_SIZE 8
#define GC_LIB_WORD_SIZE   8
#define GC_LIB_TAG_BITS    3

/* ==========================================================================
 * Callback-Based GC API (Custom Object Model)
 *
 * This API makes no assumptions about object layout, pointer tagging,
 * or header format. The GC manages memory blocks and triggers collection;
 * the caller provides callbacks for all object-model-specific operations.
 *
 * The GC is a pure memory manager + collector. The caller controls:
 *   - Marking: is_marked/set_mark callbacks own mark bits
 *   - Sizing: get_size callback returns object sizes from caller headers
 *   - Tracing: tracer callback knows which fields are pointers
 *   - Roots: root_enumerator callback knows where roots live
 *
 * Contract:
 *   - get_size(obj) must return the byte count passed to gc_lib_custom_allocate
 *   - All objects must be initialized before GC runs (headers written)
 *   - tracer/root_enumerator must only visit slots with valid GC pointers
 *
 * Usage:
 *   1. Define callbacks for root enumeration, tracing, marking, and sizing
 *   2. Create a GC with gc_lib_custom_create()
 *   3. Allocate with gc_lib_custom_allocate() (may trigger collection)
 *   4. Destroy with gc_lib_custom_destroy()
 * ========================================================================== */

/**
 * Opaque handle to a callback-based GC instance.
 */
typedef struct GcCustomHandle GcCustomHandle;

/**
 * Visit callback: called for each pointer slot during root enumeration
 * or object tracing.
 *
 * @param slot    Address where the pointer is stored (void**).
 *                The GC reads *slot to find the pointed-to object.
 *                For a moving GC, the GC may update *slot.
 * @param gc_ctx  Opaque GC context (pass through unchanged).
 */
typedef void (*GcCustomVisitor)(void** slot, void* gc_ctx);

/**
 * Root enumerator: called during collection to discover all GC roots.
 *
 * Must call `visit` for every root slot containing a GC-managed pointer.
 *
 * @param user_ctx  The value passed to gc_lib_custom_allocate/gc_lib_custom_collect
 *                  (typically a thread pointer or VM state).
 * @param visit     Visitor function to call for each root slot.
 * @param gc_ctx    Opaque GC context (pass to visit unchanged).
 */
typedef void (*GcCustomRootEnumerator)(void* user_ctx, GcCustomVisitor visit, void* gc_ctx);

/**
 * Object tracer: called during marking for each reachable object.
 *
 * Must call `visit` for every pointer field in the object.
 *
 * @param object  Pointer to the object being traced.
 * @param visit   Visitor function to call for each pointer field.
 * @param gc_ctx  Opaque GC context (pass to visit unchanged).
 */
typedef void (*GcCustomTracer)(void* object, GcCustomVisitor visit, void* gc_ctx);

/**
 * Check if an object is marked.
 *
 * @param object  Pointer to the object.
 * @return  Non-zero if marked, zero if unmarked.
 */
typedef int (*GcCustomIsMarked)(void* object);

/**
 * Set or clear the mark on an object.
 *
 * @param object  Pointer to the object.
 * @param marked  Non-zero to mark, zero to unmark.
 */
typedef void (*GcCustomSetMark)(void* object, int marked);

/**
 * Get the total byte size of an object (including caller-managed headers).
 *
 * Must return the same size that was passed to gc_lib_custom_allocate.
 *
 * @param object    Pointer to the object.
 * @param size_out  Output: the total size in bytes.
 */
typedef void (*GcCustomGetSize)(void* object, size_t* size_out);

/**
 * Configuration for creating a callback-based GC instance.
 */
typedef struct {
    GcCustomRootEnumerator root_enumerator;
    GcCustomTracer         tracer;
    GcCustomIsMarked       is_marked;
    GcCustomSetMark        set_mark;
    GcCustomGetSize        get_size;
    size_t                 initial_heap;  /**< Initial heap size in bytes. */
} GcCustomConfig;

/**
 * Create a callback-based GC instance.
 *
 * @param config  Configuration with callbacks and initial heap size.
 * @return  GC handle, or NULL on failure.
 */
GcCustomHandle* gc_lib_custom_create(GcCustomConfig config);

/**
 * Destroy a callback-based GC instance and free its resources.
 *
 * @param gc  GC handle. May be NULL (no-op).
 */
void gc_lib_custom_destroy(GcCustomHandle* gc);

/**
 * Allocate `size` bytes of zeroed memory, managed by the GC.
 *
 * May trigger collection (calling root_enumerator + tracer).
 * The caller must initialize the object (write headers) before the next
 * allocation or collection, so that get_size returns the correct value.
 *
 * @param gc        GC handle.
 * @param size      Number of bytes to allocate.
 * @param user_ctx  Passed to root_enumerator (typically a thread pointer).
 * @return  Pointer to zeroed memory, or NULL on out-of-memory.
 */
void* gc_lib_custom_allocate(GcCustomHandle* gc, size_t size, void* user_ctx);

/**
 * Run an explicit garbage collection cycle.
 *
 * @param gc        GC handle.
 * @param user_ctx  Passed to root_enumerator.
 */
void gc_lib_custom_collect(GcCustomHandle* gc, void* user_ctx);

/**
 * Write barrier (for future generational support).
 *
 * Currently a no-op for the mark-and-sweep collector.
 *
 * @param gc         GC handle.
 * @param object     Pointer to the object being written to.
 * @param new_value  The new pointer value being stored.
 */
void gc_lib_custom_write_barrier(GcCustomHandle* gc, void* object, void* new_value);

/**
 * Check if collection is recommended (for safepoints).
 *
 * @param gc  GC handle.
 * @return  Non-zero if collection is recommended, zero otherwise.
 */
int gc_lib_custom_should_collect(GcCustomHandle* gc);

#ifdef __cplusplus
}
#endif

#endif /* GC_LIBRARY_H */
