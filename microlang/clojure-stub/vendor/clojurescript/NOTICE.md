# ClojureScript — vendored source & license

This directory vendors **ClojureScript** core source for the purpose of porting
its persistent data structures into microlang's mini-Clojure frontend.

- **Upstream:** https://github.com/clojure/clojurescript
- **File:** `core.cljs` is `src/main/cljs/cljs/core.cljs` fetched verbatim from the
  `master` branch.
- **Copyright:** © Rich Hickey and the ClojureScript contributors.
- **License:** Eclipse Public License 1.0 (EPL-1.0). The full text is in
  `LICENSE` in this directory.

## License compliance

ClojureScript is licensed under EPL-1.0. Per EPL-1.0 §3, any distribution of the
Program or a Contribution must:
- retain the copyright notice and this license (done: `LICENSE` + this NOTICE);
- make the source code available (the vendored `core.cljs` is the source);
- not remove or alter the license/attribution.

Any file in the microlang tree that contains code **ported or adapted from**
ClojureScript carries a header naming ClojureScript as the source and EPL-1.0 as
the license of that ported code, keeping the derived portions identifiable and
under EPL as the license requires. microlang's own (non-ported) code is separate
and is not relicensed by this vendoring (EPL-1.0 "Separate Modules", §3).

The port targets a different host: ClojureScript's datatypes are written against
the JavaScript host (`js*`, `aget`/`aset` on JS arrays, `bit-shift-right-zero-fill`,
`^:mutable` caching-hash fields, ES6 iterators, Google Closure). The port keeps
the datatype structure and algorithms as close to verbatim as the microlang host
allows, swapping only the host primitives — see `DATATYPES.md`.
