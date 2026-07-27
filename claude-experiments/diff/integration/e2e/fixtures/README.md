# integration/e2e/fixtures — FIRST-PARTY applications

Everything in `apps/` is a pinned third-party application, copied verbatim from an
upstream git SHA and never edited. Everything in **this** directory is the opposite: an
application **written by diffpack**, checked into this repository.

A first-party fixture is a last resort, and each one has to justify itself. It exists only
where **no pinned upstream example exercises the behaviour at all**, and its corpus entry
must carry a `firstPartyReason` saying which examples were considered and what they leave
uncovered. `integration/e2e/lib/corpus-mdx.test.mjs` enforces that field.

What a first-party fixture still is:

- **A differential test, not a self-assertion.** It is built twice from one source tree —
  once by its own toolchain (`next build`), once by diffpack — served twice, driven by the
  same script in the same browser, and compared across the same eleven channels. The app's
  own toolchain is still the oracle. Nothing here compares diffpack against diffpack, and
  nothing here compares diffpack against an expectation a diffpack author wrote down.

What it is **not**:

- **Not third-party evidence.** It cannot show that an application nobody here wrote
  builds, because the person who wrote it knew what diffpack supports. That limitation is
  recorded in each materialized copy's `DIFFPACK_E2E_PROVENANCE.json`
  (`"origin": "first-party"`), so a reader of the results can always tell the two kinds
  apart.

## The fixtures

| id | what only this covers |
| --- | --- |
| `next-mdx-features` | MDX as it is actually written: a non-empty `mdx-components` override map, GFM (aligned table, strikethrough, task list, autolink literal, footnote), a component imported into MDX, a client component inside an MDX route, `export const` read by the route and by an importing module. Configures `remark-gfm` and nothing else, which is the branch diffpack answers with its own native Rust MDX compiler. |
| `next-pages-mdx-plugins` | The other MDX compiler: a `next.config` configuring remark/rehype plugins the native emitter cannot run, so every `.mdx` is compiled by the app's own `@mdx-js/mdx` (`src/mdx_runner.mjs`). Covers YAML frontmatter being stripped and exposed, on the pages router. |

Vercel's own MDX examples (`next-mdx`, `next-pages-mdx`) stay in the corpus and are still
the primary evidence; between them they use no GFM, no plugin, no frontmatter, and an
empty override map, which is exactly the hole these two fill.

## Editing one

`node integration/e2e/fetch.mjs <id>` re-copies the fixture into `apps/<id>/` on every run
(a third-party app is copied once and left alone), so an edit here is picked up without
`--force`. `node_modules` is preserved across copies, and `npm install` runs again only
when the dependency list changed.
