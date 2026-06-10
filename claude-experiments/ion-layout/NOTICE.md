# Attribution

This project is a Graphviz layout plugin whose layout algorithm is a Rust port
of the layered-graph layout from **iongraph**, an interactive visualizer for the
SpiderMonkey Ion compiler backend.

- Upstream project: iongraph — https://github.com/mozilla-spidermonkey/iongraph
  (originally https://github.com/bvisness/iongraph-web)
- Original author: **Ben Visness**
- License: **Mozilla Public License 2.0** (see `LICENSE`)

The layout core in `src/lib.rs` (stratify / expand / position / stack / edge
routing) is derived from `generic-layout/essence.ts` and the accompanying
`essence-render.ts` in the upstream repository. The Graphviz C plugin glue
(`plugin/gvplugin_ion.c`, `include/ion_layout.h`) is new code written for this
integration and is also released under MPL-2.0 to keep the project uniformly
licensed.

As required by MPL-2.0, the original license text is preserved in `LICENSE` and
the per-file notice (Exhibit A) is attached to source files in this project.
