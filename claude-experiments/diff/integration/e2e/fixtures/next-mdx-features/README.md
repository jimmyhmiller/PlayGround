# next-mdx-features — a FIRST-PARTY diffpack fixture

Written by diffpack (see `../README.md`), not copied from anywhere. It is still built by
`next build` as well as by diffpack, and the two deployments are compared: `next build` is
the oracle here exactly as it is for the pinned third-party apps.

What it covers that no pinned example does:

- `mdx-components.js` with a **non-empty** override map (`h1`, `table`, `del`, `a`), each
  marking itself with a `data-testid` so the e2e probe actually observes it. Vercel's
  `next-mdx` ships `const components: MDXComponents = {}`.
- **GFM** via `createMDX({ options: { remarkPlugins: [remarkGfm] } })`: an aligned table,
  strikethrough, a task list, a bare `www.` autolink, and a footnote. `remark-gfm` alone
  is the one plugin configuration diffpack answers with its own native Rust compiler, so
  this fixture is what keeps that compiler honest against `@mdx-js/mdx`.
- A component **imported into MDX** and used as JSX, and a `"use client"` component used
  inside an MDX route (the client boundary crossed from a server-rendered MDX page).
- `export const` from an MDX file, read both inside the file (`{audience}`) and by the
  module that imports it (`app/page.jsx` reads `revision` from `app/intro.mdx`).
- An `.mdx` file used as a plain **component** rather than a route, alongside an `.mdx`
  file that **is** a route (`app/docs/page.mdx`).

Two things are deliberately absent:

- **No `export const metadata` from the MDX route.** `next build` rejects it — with
  `@next/mdx` the MDX module resolves its provider through
  `next-mdx-import-source-file`, and Next then reports "You are attempting to export
  `metadata` from a component marked with `use client`". An app cannot do it, so the
  fixture does not either.
- **No YAML frontmatter.** Without `remark-frontmatter` configured, `next build` renders
  `---` as a thematic break; frontmatter is covered by the sibling fixture
  `next-pages-mdx-plugins`, which configures the plugins that make it mean something.
