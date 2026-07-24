# Tailwind conformance harness

Measures diffpack's native Tailwind v4 compiler (`src/tailwind.rs`) against **Tailwind's
own test suite**. `candidates.txt` is every unique candidate class extracted from
`tailwindcss@4.3.3`'s `utilities.test.ts` + `variants.test.ts` (`run([...])` arrays) —
3,363 classes, including the intentionally-invalid ones Tailwind must reject.

For each class the harness compiles it with **real Tailwind** (full framework via
`@import "tailwindcss"`, the oracle) and with **diffpack** (`diffpack tailwind`, one
NDJSON line per class), isolates the per-class utility rule(s) from both, normalizes
declarations (sorted, whitespace-stripped, escape-insensitive selectors, media/supports
context preserved), and compares. Theme `:root`, preflight and `@property`
infrastructure are excluded — this measures whether diffpack generates the correct
UTILITY rule, which is the core of a Tailwind reimplementation.

## Run

```
cargo build --release                       # builds the `diffpack tailwind` subcommand
cd scripts/tailwind-conformance
npm install tailwindcss@4.3.3 @tailwindcss/postcss@4.3.3 postcss@8
node conformance.mjs                        # ~2 min; writes failures.json
```

## Categories

- **correctly generated** — Tailwind emits a rule, diffpack emits the byte-equal rule.
- **correctly rejected** — both emit nothing (an invalid class).
- **unsupported** — Tailwind emits a rule, diffpack emits nothing.
- **hard error** — diffpack's compiler returns an error (unimplemented utility).
- **mismatch** — both emit a rule but they differ (a real gap, or a format difference
  like `calc(10 * -1)` vs `-10`, or Tailwind's `@supports` srgb color fallback which
  diffpack omits).
- **overgenerated** — Tailwind emits nothing but diffpack emits a rule (over-acceptance;
  a few are harness extraction artifacts, e.g. digit-led variants like `2xl:flex`).

Passing the FULL suite means byte-exact output across every category. diffpack today is
a faithful reimplementation of the common subset that real apps use, not the whole
surface; `failures.json` is the precise, prioritized worklist toward full conformance.
