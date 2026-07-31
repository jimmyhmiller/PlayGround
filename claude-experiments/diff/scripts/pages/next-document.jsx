// `next/document` shim (pages-router). A custom `pages/_document` composes these to
// describe the outer HTML document. They read the render result out of
// `DocumentContext` (filled by the server render entry): the app HTML string, the
// <head> elements collected from `next/head`, the serialized `__NEXT_DATA__`, and
// the client bundle URL. Rendered once per request with `renderToStaticMarkup`.

import { Fragment, useContext } from "react";
import { DocumentContext } from "./pages-runtime.jsx";

// With built-in i18n configured, Next fills `<html lang>` with the request's locale
// unless the app's own `_document` set one explicitly.
export function Html({ children, ...props }) {
  const ctx = useContext(DocumentContext);
  const lang = props.lang !== undefined ? props.lang : ctx && ctx.locale ? ctx.locale : undefined;
  return (
    <html {...props} lang={lang}>
      {children}
    </html>
  );
}

export function Head({ children }) {
  const ctx = useContext(DocumentContext);
  const collected = ctx ? ctx.head : [];
  return (
    <head>
      {children}
      {collected.map((node, index) => (
        <Fragment key={index}>{node}</Fragment>
      ))}
    </head>
  );
}

// The mount point React hydrates on the client. Server-rendered app HTML is injected
// verbatim so hydration matches the client's `<App>` tree.
export function Main() {
  const ctx = useContext(DocumentContext);
  return (
    <div
      id="__next"
      dangerouslySetInnerHTML={{ __html: ctx ? ctx.appHtml : "" }}
    />
  );
}

export function NextScript() {
  const ctx = useContext(DocumentContext);
  if (!ctx) return null;
  return (
    <>
      <script
        id="__NEXT_DATA__"
        type="application/json"
        dangerouslySetInnerHTML={{ __html: ctx.nextDataJson }}
      />
      <script type="module" src={ctx.clientEntry} />
    </>
  );
}

// `_document` may `import Document from "next/document"` and extend it; expose a base.
export default class Document {
  static getInitialProps(ctx) {
    return ctx && ctx.defaultGetInitialProps ? ctx.defaultGetInitialProps() : {};
  }
}
