# The primitive surface — what earns a prim, and what never will

A step-back after building the JVM layer, bencode, sockets, and `*out*`:
what should this dialect's primitive set BE, now that "add a `%prim`" and
"add a `defclass`" are both cheap?

## The rule

A `%prim` earns its place only as one of:

1. **A host capability** — something the language *cannot express*: syscalls
   (I/O, sockets, clocks, entropy, process control), OS threads, atomic CAS,
   UTF-8 byte conversion (strings are host objects).
2. **A performance substrate** — mutable arrays, string concat, hashing:
   things in-language code could express but only ruinously (the persistent
   collections are built ON these, in the language).
3. **A reflection hook** — `type-of`, `%method-types`, `%current-ns`, the
   eval bridge: mirrors of engine/frontend state the language needs to see.

Everything else — **every Java class name, method, exception hierarchy,
stream type, charset** — is *vocabulary*, and vocabulary lives in the layer
(`defclass` in `host_jvm.clj` / `host_io.clj`) as data and library code. The
test: if deleting it would change what programs *can* do, it's a prim; if it
would only change what they can *say*, it's the layer.

Two corollaries we now enforce:

* Prims are named for the CAPABILITY (`%tcp-read`, `%str->bytes`), never for
  Java (`%socket-input-stream` would be wrong — `java.io` is layer policy).
* A prim's contract uses host-neutral values (ints, strings, raw arrays,
  handles-as-ints). Java's quirks (signed bytes vs unsigned reads) are encoded
  ONCE, at the prim boundary, and documented there.

## Current inventory (~86 prims, by family)

| family | prims | notes |
| --- | --- | --- |
| arithmetic/compare | `%add %sub %mul %div %quot %rem %mod %lt %num-eq` | numeric tower (auto-promote, Ratio) lives behind these |
| numerics | `%numerator %denominator %bigint? %to-long`, bit ops ×6 | |
| records | `record field type-of nfields %make-record %register-fields %field-by-name %field-names %hash` | THE data substrate; deftype/defclass ride it |
| lists | `list %cons %first %rest nil?` | |
| arrays/cells | `%make-array %aget %anew %apush %ashift %aclear %aclone %alength %cell*` | perf substrate for tries/HAMTs/buffers |
| strings/chars | `%str-cat %str-of %str-len %str->chars %char-code %char-of` | |
| bytes | `%str->bytes %bytes->str` | UTF-8, SIGNED (JVM) bytes; wire code's base |
| control | `throw %apply %callec gc` | |
| threads/atoms | `%spawn %await %atom-new/get/set/cas` | real OS threads, shared heap |
| dynamic vars | `%dyn-mark %dyn-bind %dyn-unwind` (+ DynGet/DynSet) | `binding` |
| vars/ns reflection | `%global-* %var-* %sym-name %sym-ns %symbol %ns-interns %all-ns %method-types %current-ns` | |
| eval bridge | `%read-string %eval %macroexpand-1` | re-enters reader+compiler |
| stdio | `%print %println %err-print` | `*out*`/`*err*`'s DEFAULT writers only; print fns are library code over `.write` |
| sockets | `%tcp-listen %tcp-accept %tcp-read %tcp-write %tcp-close %tcp-local-port` | blocking, thread-safe (Arc out of lock) |

## The gaps (planned capabilities, in rough order)

1. **Files** — `slurp`/`spit`/`load-file`/`file-seq` need: open/read/write/
   close/exists?/list-dir/delete. Design: UNIFY with the tcp handles — one
   handle registry, one `%io-read`/`%io-write`/`%io-close`; only the open
   calls differ (`%file-open`, `%tcp-listen`…). The current `%tcp-*` read/
   write/close would become aliases and eventually fold in.
2. **Time** — `%now-millis`, `%nanos` (System/currentTimeMillis, `time`).
3. **Process/env** — `%getenv`, `%exit`, `*command-line-args*` is already CLI-fed.
4. **Entropy** — `%rand-bytes` (→ `rand`, `random-uuid`). Note the toolkit
   deliberately keeps `Math.random`-free determinism elsewhere; entropy is
   opt-in via the prim.
5. **Channels/queues** — `%chan-new/%chan-put/%chan-take(+timeout-ms)`:
   the substrate for `java.util.concurrent` queues (LinkedBlockingQueue,
   SynchronousQueue), which is most of what running nrepl's `transport.clj`
   VERBATIM requires. Timeouts belong in the prim (host clocks + parking).
6. **Thread identity + interruption** — `%thread-id`, an interrupt FLAG per
   mutator checked by blocking prims (`%chan-take`, `%tcp-read` returning an
   interrupted signal). This is the rest of the transport.clj/executor story,
   and the honest model for `future-cancel`.
7. **Buffered socket reads** — `%tcp-read-into h arr off len` (today's
   1-byte-per-syscall read is correct but slow for bulk).
8. **Jar/zip reading** — a FRONTEND feature, not a prim: `ensure_loaded`
   learning `jar!/path/inside.clj` entries so `:mvn/version` deps (from
   `~/.m2`) can join the load path. Until then deps.edn supports `:paths` +
   `:local/root` and ERRORS on `:mvn`/`:git` (never silently skips).

## Non-goals

* No FFI-shaped prims for specific Java APIs. If a library needs
  `java.util.UUID`, that's a `defclass` over `%rand-bytes` — not a prim.
* No printing/formatting prims beyond the three stdio writers: `pr-str`,
  `format`, printers are library code.
* No prim may consult the JVM-layer registry — dependencies point one way
  (layer → prims), or bootstrap ordering dies.
