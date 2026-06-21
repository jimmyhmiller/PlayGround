# Phase-2 dogfood #3 — friction report (alloc/IO interfaces)

Written after `examples/nl.coil` — a `nl` (number-lines) tool chosen to battle-test
the part of the goal the JSON parser did NOT exercise: **allocation + IO as explicit
capability VALUES**. Rather than just *using* the interfaces, it stresses their
DESIGN by implementing custom backends and composing them:

- a **custom `Reader` backend** over an in-memory `(slice u8)` (`SliceReader` — proves
  a user can add a Reader with no compiler/runtime support: ctx pointer + a read-fn);
- a **custom `Writer` that WRAPS another `Writer`** and tallies bytes (`CountWriter` —
  proves the Writer capability *composes*, decorator-style, over any inner Writer);
- the read-loop/EOF protocol, an allocator-backed `StrBuf` grown from reads, and
  `fmt` over the Writer — with the allocator + Reader + Writer all threaded explicitly.

## Headline: the capability-as-a-value design HOLDS — no interface friction

Everything worked on the first real run. The design battle-tested cleanly:

- **Adding a backend is trivial + uniform.** A new Reader/Writer is a struct for the
  state + a plain function for the fn-pointer + a constructor that fills the vtable —
  identical to the built-in `fd-`/`null-`/`FixBuf` backends. No special casing.
- **Capabilities COMPOSE.** `CountWriter` wraps `(stdout)` (or any Writer) by holding
  it as `inner` and delegating in its write-fn — exactly the Zig "writer over a writer"
  pattern, and it Just Worked. The reference model (#4) carried the threading: `(mut
  src)`/`(mut cw)` places satisfied the `(ptr …)` constructor params via auto-borrow
  (#4e); no `alloc-stack` dance.
- **No ambient state leaked in.** The allocator + both IO capabilities are passed as
  values; nothing reached for a global. The goal's "allocation and IO as explicit
  values" is real and ergonomic under genuine use.

This is the validation a dogfood is for: the *least-tested* part of the goal (the
alloc/IO interface design) stands up to a real program that extends and composes it.

## Minor (not new, already covered)

- **Result match-pyramids in IO code.** Every `read-some`/`write-some`/`print-*`
  returns `(Result _ IoError)`, so call sites nest `match (Err …)(Ok …)`. This is the
  Result ergonomics already addressed by the `?`/`try` macro (lib/try.coil) — the IO
  loops here could use it to flatten; left explicit so the dogfood shows the raw
  capability calls. No new friction.

## Verdict
No alloc/IO-interface friction surfaced — the capability-as-value design is sound and
extensible. With the JSON dogfood (data/compute + macros/reflection) and this one
(alloc/IO interfaces), both halves of the goal are now battle-tested by real programs.
