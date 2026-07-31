# Runtime helpers

The JavaScript helper functions oxc's transforms call at runtime. oxc emits
`import _decorate from "<package>/helpers/decorate"` rather than inlining a copy per
module (its "inline" helper-loader mode is unimplemented), so the helper has to come
from somewhere resolvable. diffpack serves these files from inside its own binary as
virtual modules (see `src/runtime_helpers.rs`), which is why a build never needs
`npm install @oxc-project/runtime` and cannot be broken by the app happening to have a
different version of it installed.

Each file is one helper, named exactly as oxc names it in the specifier it emits, and
is an ES module whose DEFAULT export is the helper function — the shape oxc's import
expects.

## Provenance

Vendored verbatim (module shape included) from `@oxc-project/runtime` 0.140.0, the
runtime package matching the pinned `oxc_transformer` version, which in turn copied
them from the TypeScript compiler's own `emitHelpers.ts`. Keeping them byte-identical
to what the transform was written against is the point: these are the semantics of
TypeScript's `experimentalDecorators`, not diffpack's interpretation of them.

`@oxc-project/runtime` is MIT licensed:

```
MIT License

Copyright (c) 2024-present VoidZero Inc. & Contributors
Copyright (c) 2023 Boshen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Adding one

A transform diffpack enables may call a helper that is not here yet. That is a hard
error naming the missing helper, never a silent resolve failure: copy the matching
`src/helpers/esm/<name>.js` out of `@oxc-project/runtime` at the pinned
`oxc_transformer` version and register it in `src/runtime_helpers.rs`.
