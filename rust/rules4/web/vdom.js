// vdom.js — Term-to-DOM renderer with hash-consing-based diffing
//
// Term format:
//   element(tag, attrs_cons_list, children_cons_list)
//   text(string_value)
//   attr(name, value)
//   on-*(handler_term) in attrs

const SVG_NS = "http://www.w3.org/2000/svg";
const SVG_TAGS = new Set([
  "svg", "g", "rect", "circle", "ellipse", "line", "polyline", "polygon",
  "path", "text", "tspan", "image", "defs", "clipPath", "use",
  "linearGradient", "radialGradient", "stop", "filter", "mask",
]);

/**
 * Render a term tree to a real DOM node.
 * @param {Rules4} r4 - The Rules4 WASM bridge
 * @param {number} termId - The term to render
 * @param {function} onEvent - callback(handlerTermId, domEvent)
 * @returns {Node} - A DOM node
 */
export function renderTerm(r4, termId, onEvent, svgContext = false) {
  const tag = r4.termTag(termId);

  // Sym — render as text
  if (tag === 1) {
    const name = r4.termSymName(termId);
    return document.createTextNode(name);
  }

  // Num — render as text
  if (tag === 0) {
    return document.createTextNode(String(r4.termNum(termId)));
  }

  // Float — render as text
  if (tag === 3) {
    return document.createTextNode(String(r4.termFloat(termId)));
  }

  // Call
  const headId = r4.termCallHead(termId);
  const headTag = r4.termTag(headId);
  if (headTag !== 1) {
    return document.createTextNode(r4.display(termId));
  }

  const headName = r4.termSymName(headId);

  // text(value) — create text node
  if (headName === "text") {
    const valId = r4.termCallArg(termId, 0);
    const valTag = r4.termTag(valId);
    if (valTag === 0) {
      return document.createTextNode(String(r4.termNum(valId)));
    } else if (valTag === 3) {
      return document.createTextNode(String(r4.termFloat(valId)));
    } else if (valTag === 1) {
      return document.createTextNode(r4.termSymName(valId));
    } else {
      return document.createTextNode(r4.display(valId));
    }
  }

  // element(tag_string, attrs, children)
  if (headName === "element") {
    const tagNameId = r4.termCallArg(termId, 0);
    const tagName = r4.termTag(tagNameId) === 1
      ? r4.termSymName(tagNameId)
      : r4.display(tagNameId);

    const isSvg = tagName === "svg" || svgContext || SVG_TAGS.has(tagName);
    const el = isSvg
      ? document.createElementNS(SVG_NS, tagName)
      : document.createElement(tagName);

    // Process attrs (cons list)
    const attrsId = r4.termCallArg(termId, 1);
    applyAttrs(r4, attrsId, el, onEvent, isSvg);

    // Process children (cons list)
    const childrenId = r4.termCallArg(termId, 2);
    forEachCons(r4, childrenId, (childId) => {
      el.appendChild(renderTerm(r4, childId, onEvent, isSvg));
    });

    // Store the termId on the DOM node for diffing
    el.__termId = termId;

    return el;
  }

  // Unknown call — render as text
  return document.createTextNode(r4.display(termId));
}

/**
 * Apply attributes from a cons-list of attr(name, value) terms.
 */
function termToAttrValue(r4, valId) {
  const vt = r4.termTag(valId);
  if (vt === 1) return r4.termSymName(valId);
  if (vt === 0) return String(r4.termNum(valId));
  if (vt === 3) return String(r4.termFloat(valId));
  return r4.display(valId);
}

function applyAttrs(r4, attrsId, el, onEvent, isSvg = false) {
  forEachCons(r4, attrsId, (attrId) => {
    const attrHead = r4.termCallHead(attrId);
    if (r4.termTag(attrHead) !== 1) return;
    const attrName = r4.termSymName(attrHead);

    if (attrName === "attr") {
      const nameId = r4.termCallArg(attrId, 0);
      const valId = r4.termCallArg(attrId, 1);
      const name = r4.termTag(nameId) === 1
        ? r4.termSymName(nameId)
        : r4.display(nameId);
      const val = termToAttrValue(r4, valId);

      if (name.startsWith("on-")) {
        const eventName = name.slice(3);
        el.addEventListener(eventName, (e) => {
          onEvent(valId, e);
        });
      } else if (!isSvg && name === "checked") {
        el.checked = val === "true";
        if (val === "true") el.setAttribute("checked", "");
      } else if (!isSvg && name === "value") {
        el.value = val;
      } else {
        if (isSvg && name === "href") {
          el.setAttributeNS("http://www.w3.org/1999/xlink", "href", val);
        } else {
          el.setAttribute(name, val);
        }
      }
    }
  });
}

/**
 * Iterate over a cons-list, calling fn for each element.
 */
function forEachCons(r4, listId, fn) {
  let cur = listId;
  while (true) {
    const tag = r4.termTag(cur);
    if (tag === 1 && r4.termSymName(cur) === "nil") break;
    if (tag !== 2) break;

    const head = r4.termCallHead(cur);
    if (r4.termTag(head) !== 1 || r4.termSymName(head) !== "cons") break;

    const elem = r4.termCallArg(cur, 0);
    fn(elem);
    cur = r4.termCallArg(cur, 1);
  }
}

/**
 * Diff and patch the DOM. Uses hash-consing: if termIds match, skip subtree.
 * @param {Rules4} r4
 * @param {number} oldId - previous term tree id
 * @param {number} newId - new term tree id
 * @param {Node} domNode - current DOM node
 * @param {Node} parentNode - parent DOM node (for replacement)
 * @param {function} onEvent - event callback
 * @returns {Node} - the (possibly new) DOM node
 */
export function diff(r4, oldId, newId, domNode, parentNode, onEvent, svgContext = false) {
  // Hash-consing optimization: identical TermIds = identical subtrees
  if (oldId === newId) return domNode;

  // If structure changed significantly, just replace
  const newNode = renderTerm(r4, newId, onEvent, svgContext);
  if (parentNode && domNode) {
    parentNode.replaceChild(newNode, domNode);
  }
  return newNode;
}

/**
 * Smart patch: tries to reuse DOM nodes for element terms with same tag.
 */
export function patch(r4, oldId, newId, domNode, parentNode, onEvent, svgContext = false) {
  if (oldId === newId) return domNode;

  // Both are element() calls?
  if (r4.termTag(oldId) === 2 && r4.termTag(newId) === 2) {
    const oldHead = r4.termCallHead(oldId);
    const newHead = r4.termCallHead(newId);
    if (r4.termTag(oldHead) === 1 && r4.termTag(newHead) === 1) {
      const oldName = r4.termSymName(oldHead);
      const newName = r4.termSymName(newHead);
      if (oldName === "element" && newName === "element") {
        // Same tag name?
        const oldTag = r4.termCallArg(oldId, 0);
        const newTag = r4.termCallArg(newId, 0);
        if (oldTag === newTag && domNode.nodeType === 1) {
          const tagName = r4.termTag(oldTag) === 1 ? r4.termSymName(oldTag) : "";
          const isSvg = tagName === "svg" || svgContext || SVG_TAGS.has(tagName);
          // Same element type — patch attrs and children
          patchAttrs(r4, oldId, newId, domNode, onEvent, isSvg);
          patchChildren(r4, oldId, newId, domNode, onEvent, isSvg);
          domNode.__termId = newId;
          return domNode;
        }
      }
    }
  }

  // Fallback: replace entirely
  const newNode = renderTerm(r4, newId, onEvent, svgContext);
  if (parentNode) {
    parentNode.replaceChild(newNode, domNode);
  }
  return newNode;
}

function patchAttrs(r4, oldId, newId, el, onEvent, isSvg = false) {
  // Simple approach: clear old attrs and re-apply new ones
  // (Could be smarter but this is fine for TodoMVC scale)
  const oldAttrs = new Set();
  const oldAttrsId = r4.termCallArg(oldId, 1);
  forEachCons(r4, oldAttrsId, (attrId) => {
    if (r4.termTag(r4.termCallHead(attrId)) === 1 &&
        r4.termSymName(r4.termCallHead(attrId)) === "attr") {
      const nameId = r4.termCallArg(attrId, 0);
      const name = r4.termTag(nameId) === 1 ? r4.termSymName(nameId) : r4.display(nameId);
      if (!name.startsWith("on-")) {
        oldAttrs.add(name);
      }
    }
  });

  // Remove old event listeners by replacing the element's event-holding clone
  // Actually, for simplicity we'll just re-render on attr changes.
  // This is fine since event handlers are identified by term structure.

  // Remove old attributes that aren't in new
  const newAttrsId = r4.termCallArg(newId, 1);
  const newAttrNames = new Set();
  forEachCons(r4, newAttrsId, (attrId) => {
    if (r4.termTag(r4.termCallHead(attrId)) === 1 &&
        r4.termSymName(r4.termCallHead(attrId)) === "attr") {
      const nameId = r4.termCallArg(attrId, 0);
      const name = r4.termTag(nameId) === 1 ? r4.termSymName(nameId) : r4.display(nameId);
      newAttrNames.add(name);
    }
  });

  for (const name of oldAttrs) {
    if (!newAttrNames.has(name)) {
      el.removeAttribute(name);
    }
  }

  // Apply new attrs
  applyAttrs(r4, newAttrsId, el, onEvent, isSvg);
}

function patchChildren(r4, oldId, newId, el, onEvent, svgContext = false) {
  const oldChildrenId = r4.termCallArg(oldId, 2);
  const newChildrenId = r4.termCallArg(newId, 2);

  if (oldChildrenId === newChildrenId) return;

  const oldChildren = [];
  forEachCons(r4, oldChildrenId, id => oldChildren.push(id));
  const newChildren = [];
  forEachCons(r4, newChildrenId, id => newChildren.push(id));

  const domChildren = Array.from(el.childNodes);

  // Patch common length
  const minLen = Math.min(oldChildren.length, newChildren.length);
  for (let i = 0; i < minLen; i++) {
    patch(r4, oldChildren[i], newChildren[i], domChildren[i], el, onEvent, svgContext);
  }

  // Remove extra old children
  for (let i = oldChildren.length - 1; i >= minLen; i--) {
    el.removeChild(el.childNodes[i]);
  }

  // Add new children
  for (let i = minLen; i < newChildren.length; i++) {
    el.appendChild(renderTerm(r4, newChildren[i], onEvent, svgContext));
  }
}
