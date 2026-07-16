# Stage M — survivor space + aging (fix over-promotion)

Decision (Jimmy, 2026-07-16): build a survivor space + aging. This is the
measured fix for the one remaining generational cost.

## Why (the bisect that scoped it)

The apply/interleave slowdown vs pre-generational Stage H is NOT the write
barrier or the rooting — a normalized-config bisect showed I1 ≈ I2 ≈ HEAD
(~226/730 ns), so both cost essentially nothing. And HEAD with a 4 GiB
nursery that never collects (like Stage H's 4 GiB space) gives apply 189 /
interleave 570 = Stage H exactly. So the whole delta is **minor-collection
cost**, and the profile shows it is COPYING-dominated (memmove 35 vs zeroing
7): it is *promotion*.

The inefficiency: **promote-on-first-survival OVER-PROMOTES.** A minor that
fires mid-iteration catches the current iteration's in-progress transients
alive and tenures them to the old gen, where they die immediately — filling
the old gen with garbage and driving majors. Real generational GCs promote
only after an object survives N minors; the ones that die in their first
few minors never touch the old gen.

## Design

Three young regions (HotSpot's eden + two survivor spaces, simplified):

- **Eden** = today's nursery (`MICROLANG_NURSERY_MB`, default 64). ALL
  allocation goes here; the AllocWindow points here, unchanged.
- **Survivor FROM / survivor TO**: two equal bump spaces, one active at a
  time. Size default `MICROLANG_SURVIVOR_MB` (start ~1/8 of eden, tune).
- **Old gen**: the existing semi-space pair, unchanged (major = its Cheney).

### Object age
Stored in the header's `spare` u16 (bits 16-31 of the header word) — it is
FREE today (type_id is bits 0-15, aux is 32-63, forwarding is bit 63). Use
the low bits (age 0..=TENURE_THRESHOLD). `TENURE_THRESHOLD` default 3
(`MICROLANG_TENURE`). NB: the age must be COPIED and incremented when the
object is evacuated — verify `copy_or_forward`'s header handling preserves
spare and the age bump writes the NEW header, not the forwarding word.

### Minor collection (rewritten)
The "from" set is now **eden ∪ survivor-FROM** (both are young; a dirty old
card may point into either). For each live young object reached from roots +
dirty cards:
- `age < TENURE_THRESHOLD` → copy to survivor-TO, `age += 1`.
- `age >= TENURE_THRESHOLD` → promote to old gen (age irrelevant thereafter).
- **survivor-TO full** → promote the overflow to the old gen instead (a minor
  must always be able to complete; there is no safe bail-out from a
  half-forwarded graph). Account for this in the fit check.
Cheney-scan the copied objects (in BOTH survivor-TO and the promoted old-gen
range — their fields may reference more young objects). Afterwards: eden
cursor = 0 (empty), survivor-FROM cursor = 0 (empty), swap FROM/TO survivor
roles, clear all cards.

INVARIANT (replaces "nursery empty after minor"): **eden and survivor-FROM
are empty after a minor** — an exact evacuation still. A live young object is
now in survivor-TO (or promoted), never left behind.

### The missed-barrier detector (preserved, re-aimed)
Old check: "no old object points into the nursery." New check: **no old
object points into the just-EVACUATED regions (eden ∪ survivor-FROM)**, which
are empty post-minor — still exact. An old→survivor-TO pointer is now LEGAL
(an old object holding a young survivor) and MUST be card-marked like any
old→young store; the barrier already does this (it marks the old storing
object regardless of the target's region). Verify the detector walks the old
gen AND asserts nothing points into eden or survivor-FROM.

### The write barrier (UNCHANGED)
The barrier marks the OLD storing object when it stores any pointer; the next
minor scans dirty cards and finds young targets in eden OR survivor. So the
barrier's single unsigned-compare "is the storing object old?" is unchanged —
survivor and eden are both "not old", so a store INTO a survivor/eden object
still needs no barrier (it is young). A store into an OLD object of a
survivor/eden pointer marks the card; the minor's card scan must treat BOTH
eden and survivor-FROM as from-space (it already will if `young_contains`
covers both).

Survivor→eden and survivor→survivor edges need NO card: the survivor space is
fully scanned every minor (it is part of the "from" set), so those edges are
found directly.

### Fit / fallback
`minor_will_fit` must now check: survivors-that-stay fit survivor-TO OR spill
to old gen, AND the aged+spill promotions fit the old gen. If the old gen
can't take the promotions, defer to a MAJOR (as today). A major evacuates
eden ∪ survivor ∪ old-gen into the fresh old space (all three are "from"),
so it always completes with a non-empty young gen.

## Phases (each gated: cargo test, --features jit, gc_stress --ignored,
##   gc_stress_library, scheme, clojure-stub oracle; then measure)

- M1 (`heap.rs`): survivor spaces, age in spare, minor rewrite (age/tenure/
  spill), detector re-aim, fit/fallback, major over three from-regions. Unit
  tests in the D1/I1 style: an object promoted only after TENURE minors; a
  short-lived object that dies in survivor without ever reaching old gen; the
  detector fires on an un-barriered old→eden AND old→survivor-FROM store;
  survivor-TO overflow spills to old gen; age is preserved+incremented across
  a minor; a major over eden+survivor+old. Mutation-test the new asserts.
- M2 (`gc.rs`/`runtime.rs`): wire the rewritten minor; counters (survived vs
  promoted bytes). `MICROLANG_GC_STRESS` must exercise survivor aging (run
  enough minors that objects actually age + tenure). Extend the gc_stress
  battery's oldyoung cases so an object is written old→survivor (not just
  old→eden).
- M3: measure. Targets: apply/interleave move toward the 189/570 "no-collect"
  floor (promotion should now be rare — most transients die in survivor);
  vecbuild/group-by/assoc no worse; the gc_stress_library minor/major counts
  should show FEWER majors (less over-promotion). Report honestly.

## Non-goals
More than 2 survivor spaces / dynamic tenuring thresholds (start fixed).
Concurrent collection. Changing the old gen or the major.
