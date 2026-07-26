// A static page whose `template.tsx` wraps it. Because the template folds into the
// document tree at build time, the prerendered HTML already contains the template
// wrapper around this content (proving build-time composition + SSR coverage).
export default function ConventionsDemo() {
  return <main id="conventions-page">conventions-demo page content</main>;
}
