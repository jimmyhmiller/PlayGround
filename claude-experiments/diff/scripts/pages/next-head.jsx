// `next/head` shim (pages-router). Its children never render into the app tree
// (matching Next, so the hydrated `#__next` DOM has no <head> content and there is
// no hydration mismatch). Instead:
//   * On the server, it pushes its children into the HeadManagerContext collector
//     during render; the custom `_document`'s <Head> renders them into the document.
//   * On the client, an effect applies its children to `document.head` directly, so
//     a `router.push` that swaps the page updates <title>/<meta> live.

import { Children, isValidElement, useContext, useEffect } from "react";
import { HeadManagerContext } from "./pages-head-manager.jsx";

function textOf(node) {
  if (node == null || node === false) return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (isValidElement(node)) return textOf(node.props.children);
  return "";
}

// Apply a Head element list to `document.head`, replacing whatever a previous
// `next/head` render put there (tracked via a data attribute).
function applyHead(children) {
  const managed = document.head.querySelectorAll("[data-diffpack-head]");
  managed.forEach((node) => node.remove());
  Children.forEach(children, (child) => {
    if (!isValidElement(child)) return;
    const type = child.type;
    if (type === "title") {
      document.title = textOf(child.props.children);
      return;
    }
    if (typeof type !== "string") return;
    const el = document.createElement(type);
    for (const [key, value] of Object.entries(child.props || {})) {
      if (key === "children") {
        el.textContent = textOf(value);
      } else if (value != null && value !== false) {
        el.setAttribute(key, value === true ? "" : String(value));
      }
    }
    el.setAttribute("data-diffpack-head", "");
    document.head.appendChild(el);
  });
}

export default function Head({ children }) {
  const collector = useContext(HeadManagerContext);
  // Server render: collect the children so `_document` can render them.
  if (collector && typeof collector.push === "function") {
    collector.push(children);
  }
  // Client: effects never run during SSR, so this is a client-only sync.
  useEffect(() => {
    applyHead(children);
  });
  return null;
}
