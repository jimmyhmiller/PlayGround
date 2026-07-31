import createMDX from "@next/mdx";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeSlug from "rehype-slug";
import remarkFrontmatter from "remark-frontmatter";
import remarkMdxFrontmatter from "remark-mdx-frontmatter";

// Plugins diffpack's native Rust MDX compiler cannot run (a unified plugin is an arbitrary
// JS function over an mdast/hast), so every `.mdx` file here must be compiled by the APP's
// OWN `@mdx-js/mdx` — `src/mdx.rs::compile_with_app_pipeline` + `src/mdx_runner.mjs`. That
// branch had no e2e coverage at all until this fixture existed.
//
// `remark-frontmatter` + `remark-mdx-frontmatter` are also the only way YAML frontmatter is
// STRIPPED and EXPOSED by `next build` itself: without them, `next build` renders `---` as a
// thematic break, so an app whose author expects frontmatter must configure them, and this
// is what diffpack has to agree with.
const withMDX = createMDX({
  options: {
    remarkPlugins: [remarkFrontmatter, remarkMdxFrontmatter],
    rehypePlugins: [rehypeSlug, [rehypeAutolinkHeadings, { behavior: "append" }]],
  },
});

export default withMDX({
  pageExtensions: ["js", "jsx", "mdx"],
});
