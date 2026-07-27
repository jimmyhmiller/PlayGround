// The default pages-router `Document` used when the project has no `pages/_document`.
// Mirrors Next's built-in document skeleton.
import { Html, Head, Main, NextScript } from "./next-document.jsx";

export default function Document() {
  return (
    <Html>
      <Head />
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
