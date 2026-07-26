import type { MDXComponents } from "mdx/types";

// Next's root override file: `useMDXComponents` returns a map of intrinsic MDX elements
// to custom implementations. Here we override `h1` (adding a marker attribute + class) and
// `code` (a marker class), leaving every other element to its intrinsic fallback.
const components: MDXComponents = {
  h1: (props) => <h1 data-mdx-override="h1" className="mdx-heading" {...props} />,
  code: (props) => <code data-mdx-override="code" {...props} />,
};

export function useMDXComponents(): MDXComponents {
  return components;
}
