# Fully-total language migration (in progress)

Jimmy's decision (2026-06-11): the language has NO divergence construct.

- `abort()` builtin: REMOVED from the surface language. (Runtime-internal
  Rust-side traps for compiler/runtime bugs — type confusion, heap
  exhaustion, non-exhaustive-match trap — remain; they are not API.)
- `*_trusted` accessors: REMOVED. One indexing API: checked, Result.
- Provably-impossible `Err` arms THREAD OUT: public stdlib signatures
  change (e.g. `smap_get -> Result<Option<T>, IndexError>`). No
  sentinels, ever.
- Uninitialized non-scalar slot reads: `Err(IndexError::Uninitialized(OobInfo))`.
  Prim-repr (scalar) arrays are zero-filled, so scalar reads return 0/0.0;
  boxed-repr uninit reads are the Err. SETs need no init check.
- Bytes are zero-filled: no Uninitialized case for bytes.

## Steps

1. [x] stdlib types: `Uninitialized(OobInfo)` variant added; `ix<T>`
       String-error adapter added next to IndexError.
2. [x] codegen: inline `core/array.is_init` emitter; typecheck errors now
       carry the def name (`TypeError::InDef`).
3. [x] resolve: GET expansion has the is_init branch (ok-first nesting);
       `abort` + `*_trusted` fully removed (surface, typecheck, codegen
       case, printer inverses); svc_* are Result/Option-shaped.
4. [x] stdlib migration COMPLETE: all 7 slices converted + reassembled +
       spliced; whole stdlib (390 defs) typechecks fully total. Residual
       fixes applied: http_request_many double-wraps, lambda_run_once
       (http_header now Result), lambda_serve (env_get now Result),
       net_buf_to_bytes/net_recv_exact (IndexError→NetErr::ConnClosed at
       the boundary), D-slice duplicate sha/hmac defs stripped.
       (was IN FLIGHT:) source extracted to /tmp/stdlib.lang,
       split into /tmp/slice_{PREFIX,A_http,B_json,C_codec,D_aws,E_data,
       F_maps,G_net}.lang (cut points in /tmp/slice_cuts.json; reassemble
       = PREFIX + A..G in that order, splice into stdlib.rs SOURCE between
       `r#"` and `"#;`). Conversion contract: /tmp/contract.md. Strings +
       StringMap + cstr/ptr_to_string converted by hand (in PREFIX/early
       slices); 7 parallel agents converting A..G. After splice: rebuild,
       probe via `AI_LANG_CODEBASE=... add /tmp/ail-bench/probe.ail`,
       iterate residual type errors (now named via InDef).
       (~120 accessor sites; signature cascade):
       HAMT/smap/imap, base64/crc32/zip, wire/http/json, svc_* (abort →
       Result), string/bytes helpers. Compile-error-driven via
       `add`-ing the stdlib.
5. [x] Test snippets: evalrun (24/24) + effects (8/8, EffectSet::PANIC
       bit REMOVED — no producers exist; knowledge.rs + CLI display
       updated) done by hand; stdlib(82)/net(20)/codegen+typecheck(20)
       being fixed by 3 agents per /tmp/test-contract.md.
6. [x] Examples: all 20 add-clean; crypto/float/map verified running
       with correct outputs; kvstore.ail migrated (net tests fixture).
7. [x] DONE 2026-06-11: full suite 594/594, every example + benchmark
       compiles, benchmarks verified (nbody 373ms checked+init),
       GC stress green. The language is fully total.

## Compiler fixes that fell out
- `TypeError::InDef` names the failing def (typecheck_module).
- `merge_branch_types`: if/match joins ground free TypeVars + Never.
- infer_type recovers EnumNew partial instantiations from payloads.
- `is_boxed_scalar` includes Ptr (generic payloads instantiated to Ptr
  box/unbox like Int); decode::<T>'s non-scalar stand-in is the baked
  TypeRef(expected) — Builtin("Ptr") there caused mis-unboxing + segv.
- `core/array.is_init` inline emitter; `core/array.len` fully inline.
- EffectSet::PANIC removed (no producers can exist).

Perf note: the is_init branch must be INLINE or nbody regresses ~170ms.
