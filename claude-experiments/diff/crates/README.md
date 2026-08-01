# Diffpack crate boundaries

This workspace is the incremental extraction of the original `diffpack` crate.
The repository must continue to compile after every migration slice; compatibility
re-exports are therefore intentional and temporary.

Dependency direction:

```text
diffpack-core
    ↑
diffpack-default-loader
    ↑
diffpack-web ── diffpack-vite-compat
    ↑
diffpack-next  diffpack-tanstack
```

- `diffpack-core`: module identity, JavaScript dependency parsing, AST
  reachability, provider contracts, graph, linker, chunk plan, rendering, and
  structured diagnostics. It must not depend on framework policy, Node
  processes, or filesystem conventions.
- `diffpack-default-loader`: filesystem/Node resolution, package side-effect
  metadata, and built-in JS, JSON, CSS, and asset providers.
- `diffpack-web`: HTML entry, browser emission, and web development behavior.
- `diffpack-vite-compat`: opt-in Vite config/env/glob/define compatibility.
- `diffpack-next`: Next routes, RSC, images/fonts, manifests, and runtime adapters.
- `diffpack-tanstack`: TanStack route splitting, server functions, and manifests.

The current root `diffpack` package remains the executable and compatibility
facade. New core code should not be added there. During extraction an integration
crate may re-export a root module, but the end state reverses that relationship:
the executable composes the crates and no integration crate depends on it.
