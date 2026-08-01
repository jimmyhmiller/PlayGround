# Embedding Diffpack

The supported embedding entry point is `diffpack_default_loader::BuildEngine`.
External tools should depend on `diffpack-core` for provider contracts and on
`diffpack-default-loader` for the filesystem engine. They should not depend on
the root `diffpack` CLI package or construct driver internals directly.

```rust,ignore
use diffpack_core::{BuildMode, Environment, Platform};
use diffpack_default_loader::BuildEngine;

let engine = BuildEngine::builder(project_root)
    .environment(Environment {
        name: "my-tool".into(),
        platform: Platform::Node,
        mode: BuildMode::Production,
    })
    .provider(my_provider)
    .build()?;

let (mut build, initial_update) = engine.discover("src/entry.ts")?;
let incremental_update = build.rebuild_path(changed_file)?;
```

Providers run in registration order. Resolution and loading stop at the first
provider that answers; all applicable transforms run in registration order.
The built-in filesystem/query loaders run after external resolution and loading,
so a provider can claim a custom scheme without modifying Diffpack. Cold and
incremental compilation share the same immutable provider pipeline.

A provider may resolve a specifier, mark it external, load JavaScript or
TypeScript, transform ordinary modules, emit assets, and add watch files. Watch
files should be canonical absolute paths so filesystem notifications and graph
ownership compare identically. Provider source maps are currently rejected with
a specific error because end-to-end map composition is not implemented; they are
never silently discarded.

See the compiled workspace example in
[`examples/external-provider`](../examples/external-provider/src/lib.rs).

## Stability

The `BuildEngine`, `ModuleProvider`, request/result records, `Environment`, and
incremental `BuildUpdate` path are the intended embedding facade. Framework and
driver implementation modules remain unstable during the current architecture
iteration. Semver stability is not promised until the public-API snapshot gate
is introduced and the remaining migration-era exports are reduced.
