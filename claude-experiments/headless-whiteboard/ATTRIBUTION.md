# Attribution

`headless-whiteboard` is an independent Rust reimplementation. It is **not** a
fork or vendored copy of any existing project. However, its element model,
geometry/hit-test math, sketch-rendering algorithms, and `.excalidraw` file
format compatibility are derived from the behavior and published algorithms of
the projects below. We gratefully credit them and reproduce their license
notices as required.

Where a source file reimplements an algorithm from one of these projects, its
header notes the upstream file it derives from.

---

## Excalidraw

The element model, file format (`.excalidraw`), geometry/bounds/hit-test math,
binding behavior, and interaction semantics are reimplemented from Excalidraw.

> MIT License
>
> Copyright (c) 2020 Excalidraw
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Source: <https://github.com/excalidraw/excalidraw>

---

## Rough.js

The hand-drawn / "sketchy" shape generation (seeded RNG, hachure and other
fills, double-stroke roughening) is reimplemented from Rough.js.

> MIT License
>
> Copyright (c) 2019 Preet Shihn
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Source: <https://github.com/rough-stuff/rough>

---

## DejaVu Fonts

The `whiteboard-tiny-skia` backend bundles the DejaVu Sans and DejaVu Sans Mono
fonts (in `crates/whiteboard-tiny-skia/assets/`) to rasterize text deterministically
without system font discovery.

DejaVu fonts are based on the Bitstream Vera fonts and are distributed under a
permissive license (the Bitstream Vera license plus the DejaVu changes license),
which allows redistribution and embedding. The fonts and their full license text
are available from the DejaVu project.

Source: <https://dejavu-fonts.github.io/>
