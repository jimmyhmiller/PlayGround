import createMDX from "@next/mdx";
import remarkGfm from "remark-gfm";

// `remark-gfm` and nothing else: this is the configuration diffpack answers with its
// NATIVE Rust MDX compiler (`src/mdx.rs`, `Gfm::On`) rather than by shelling out to the
// app's own `@mdx-js/mdx`. The sibling fixture `next-pages-mdx-plugins` covers the other
// branch. Written as ESM because every remark plugin is ESM-only — a `next.config.js`
// doing `require("remark-gfm")` cannot load one.
const withMDX = createMDX({
  options: {
    remarkPlugins: [remarkGfm],
  },
});

export default withMDX({
  pageExtensions: ["js", "jsx", "md", "mdx"],
});
