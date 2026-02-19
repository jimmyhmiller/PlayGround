/*  rules4.h — C API for the rules4 term-rewriting engine.
 *
 *  Link against librules4.a (static) or librules4.dylib / librules4.so (dynamic).
 *
 *  Terms are identified by opaque uint32_t handles (TermId).
 *  Tag values: 0 = Num, 1 = Sym, 2 = Call, 3 = Float.
 */

#ifndef RULES4_H
#define RULES4_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque engine handle. */
typedef struct Rules4Engine Rules4Engine;

/* ── Engine lifecycle ── */

Rules4Engine *r4_engine_new(void);
void          r4_engine_free(Rules4Engine *engine);

/* ── Program loading ── */

/* Parse + install rules from source text.  Returns the top-level expression id. */
uint32_t r4_load_program(Rules4Engine *engine, const uint8_t *src, size_t len);

/* ── Evaluation ── */

uint32_t r4_eval(Rules4Engine *engine, uint32_t term);
uint32_t r4_eval_step_limit_exceeded(Rules4Engine *engine);

/* ── Term construction ── */

uint32_t r4_term_num  (Rules4Engine *engine, int64_t n);
uint32_t r4_term_float(Rules4Engine *engine, double f);
uint32_t r4_term_sym  (Rules4Engine *engine, const uint8_t *name, size_t len);
uint32_t r4_term_call (Rules4Engine *engine, uint32_t head,
                       const uint32_t *args, size_t args_len);

/* ── Term inspection ── */

uint8_t  r4_term_tag       (Rules4Engine *engine, uint32_t id);
int64_t  r4_term_get_num   (Rules4Engine *engine, uint32_t id);
double   r4_term_get_float (Rules4Engine *engine, uint32_t id);

/* Returns a pointer into internal storage — valid until next mutation.
   *out_len receives the byte length (not NUL-terminated). */
const uint8_t *r4_term_get_sym_name(Rules4Engine *engine, uint32_t id,
                                    size_t *out_len);

uint32_t r4_term_call_head (Rules4Engine *engine, uint32_t id);
uint32_t r4_term_call_arity(Rules4Engine *engine, uint32_t id);
uint32_t r4_term_call_arg  (Rules4Engine *engine, uint32_t id, uint32_t idx);

/* ── Display ── */

/* Returns a NUL-terminated string; *out_len receives byte count (excl NUL).
   The pointer is valid until the next call to r4_display_term. */
const uint8_t *r4_display_term(Rules4Engine *engine, uint32_t id,
                               size_t *out_len);

/* ── Generic scope access ── */

/* Scopes are auto-created on first @scope_name emit.
   Use these to poll pending terms from any scope by name. */

uint32_t r4_scope_pending_count(Rules4Engine *engine,
                                const uint8_t *scope, size_t len);
uint32_t r4_scope_pending_get  (Rules4Engine *engine,
                                const uint8_t *scope, size_t len,
                                uint32_t idx);
void     r4_scope_pending_clear(Rules4Engine *engine,
                                const uint8_t *scope, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* RULES4_H */
