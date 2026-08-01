# Diffpack bindings for Next

This experimental integration implements the raw project/endpoint contract that
Next normally receives from Turbopack through `next-swc`. It deliberately proxies
all compiler transforms to Next's native SWC addon and replaces only project
operations.

The first milestone is production-only. Development subscriptions and HMR fail
with explicit errors.

Build and test the bridge:

```sh
cargo build -p diffpack-next-bindings
DIFFPACK_NEXT_REPO=/absolute/path/to/next.js npm test --prefix integrations/next-bindings
```

Select it from a Next invocation:

```sh
DIFFPACK_NEXT_REPO=/absolute/path/to/next.js \
DIFFPACK_NEXT_BRIDGE=/absolute/path/to/diffpack/target/debug/diffpack-next-bindings \
DIFFPACK_BINARY=/absolute/path/to/diffpack/target/debug/diffpack \
__INTERNAL_CUSTOM_TURBOPACK_BINDINGS=/absolute/path/to/diffpack/integrations/next-bindings/binding.cjs \
next build
```

The binding protocol is versioned independently of Next's unstable internal
escape hatch. Native `.next` artifact emission is the next implementation slice.
