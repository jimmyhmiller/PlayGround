# next-pages-mdx-plugins — a FIRST-PARTY diffpack fixture

Written by diffpack (see `../README.md`), not copied from anywhere. Built by `next build`
as well as by diffpack, and compared: `next build` is the oracle.

This fixture exists for diffpack's **other** MDX compiler. Its `next.config.mjs`
configures `remark-frontmatter`, `remark-mdx-frontmatter`, `rehype-slug` and
`rehype-autolink-headings` — unified plugins are arbitrary JavaScript functions over an
mdast/hast, which a Rust emitter cannot run — so every `.mdx` file here is compiled by the
**app's own `@mdx-js/mdx`**, driven by `src/mdx_runner.mjs`. Nothing pinned in the corpus
configures an MDX plugin, so that path had no end-to-end coverage at all.

It also covers **YAML frontmatter**, which cannot be compared any other way: `next build`
has no opinion about frontmatter unless `remark-frontmatter` is configured (it renders
`---` as a thematic break), so an app that wants frontmatter stripped *and* readable
configures these two plugins, and that is the behaviour diffpack has to agree with. The
page reads its own frontmatter back (`# {frontmatter.title}`).

Plus, on the pages router: an `.mdx` file **as a page**, a component imported into it and
used as JSX, and an `export const` the page renders.
