#!/usr/bin/env python3
"""Generate `crates/jsir-ast/src/schema_generated.rs` from the upstream JSIR
AST sources (the spec we mirror byte-for-byte).

Sources of truth (in `vendor/jsir-upstream/maldoca/js/ast/`):
  - ast.generated.h   -> class hierarchy + per-class private member TYPES (repr)
  - ast_to_json.generated.cc -> per-class Serialize()/SerializeFields():
        JSON key names, field ORDER, presence (required / maybe-null / maybe-
        undefined), array-ness, array holes, variants.

We emit a data-driven schema: each node/helper struct is a list of FieldSpec
{ key, presence, repr }. A single generic interpreter in `model.rs` serializes
to our nlohmann-faithful `Json` and parses back, so byte-exactness lives in one
place (already proven by the dumper) rather than in 87 bespoke impls.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AST_DIR = os.path.join(ROOT, "vendor/jsir-upstream/maldoca/js/ast")
HEADER = open(os.path.join(AST_DIR, "ast.generated.h")).read()
CC = open(os.path.join(AST_DIR, "ast_to_json.generated.cc")).read()

FIXED_HELPERS = {
    "JsPosition", "JsSourceLocation", "JsSymbolId",
    "JsDirectiveLiteralExtra", "JsRegExpLiteralExtra", "JsStringLiteralExtra",
    "JsNumericLiteralExtra", "JsBigIntLiteralExtra", "JsTemplateElementValue",
}

# ---------------------------------------------------------------------------
# Parse the header: per-class private member types (the field repr source).
# ---------------------------------------------------------------------------

def parse_classes():
    """Return {class_name: {'members': [(name, cpptype)], 'serialize_type': str|None}}."""
    classes = {}
    for m in re.finditer(r'\nclass (Js\w+)([^{]*)\{(.*?)\n\};', HEADER, re.S):
        name = m.group(1)
        body = m.group(3)
        # Private members live after the last "private:" label as `Type name_;`.
        members = []
        priv = body.rsplit("private:", 1)
        if len(priv) == 2:
            for line in priv[1].splitlines():
                line = line.strip()
                mm = re.match(r'(.+?)\s+(\w+_);$', line)
                if mm:
                    members.append((mm.group(2), mm.group(1).strip()))
        classes[name] = {"members": members}
    return classes

# ---------------------------------------------------------------------------
# Parse the serializer: per-class own field (key, presence, is_array, holes).
# ---------------------------------------------------------------------------

def serialize_fields_body(cls):
    m = re.search(r'void %s::SerializeFields\(std::ostream& os, bool &needs_comma\) const \{(.*?)\n\}' % cls, CC, re.S)
    return m.group(1) if m else None

def serialize_type_tag(cls):
    m = re.search(r'void %s::Serialize\(std::ostream& os\) const.*?\{(.*?)\n\}' % cls, CC, re.S)
    if not m:
        return None
    tag = re.search(r'"type":\\?"(\w+)\\?"', m.group(1))
    # The literal is os << "\"type\":\"BinaryExpression\"" -> in source: "\"type\":\"X\""
    tag = re.search(r'\\"type\\":\\"(\w+)\\"', m.group(1))
    return tag.group(1) if tag else None

def serialize_mro(cls):
    """Ordered list of `JsX::SerializeFields` calls inside Serialize() = field MRO."""
    m = re.search(r'void %s::Serialize\(std::ostream& os\) const.*?\{(.*?)\n\}' % cls, CC, re.S)
    if not m:
        return []
    return re.findall(r'(\w+)::SerializeFields\(os, needs_comma\)', m.group(1))

def parse_own_fields(cls):
    """Parse one class's SerializeFields into ordered [(key, presence, kind)].
    kind in {scalar, object, array, array_holes}; presence in
    {req, maybe_null, maybe_undef}. Uses char-level brace matching so that
    `} else {` (balanced braces) does not confuse block boundaries."""
    s = serialize_fields_body(cls)
    if s is None:
        return []
    L = len(s)

    def match_brace(i):
        # s[i] must be '{'; return index just AFTER the matching '}'.
        depth = 0
        while i < L:
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return L

    def find_key(text):
        mm = re.search(r'\\"(\w+)\\":', text)
        return mm.group(1) if mm else None

    fields = []
    pos = 0
    re_comma = re.compile(r'\s*MaybeAddComma\(os, needs_comma\);')
    re_if = re.compile(r'\s*if \((\w+)\.has_value\(\)\) \{')
    re_else = re.compile(r'\s*else \{')
    while pos < L:
        m = re_comma.match(s, pos)
        if m:
            pos = m.end()
            continue
        m = re_if.match(s, pos)
        if m:
            brace = s.index('{', pos)
            end_if = match_brace(brace)
            if_body = s[brace + 1:end_if - 1]
            key = find_key(if_body)
            m2 = re_else.match(s, end_if)
            if m2:
                brace2 = s.index('{', end_if)
                end_else = match_brace(brace2)
                else_body = s[brace2 + 1:end_else - 1]
                # else branch emits `"key": null` -> MAYBE_NULL.
                presence = "maybe_null" if '"null"' in else_body else "maybe_undef"
                pos = end_else
            else:
                presence = "maybe_undef"
                pos = end_if
            if key:
                fields.append((key, presence, classify_kind(if_body)))
            continue
        # Otherwise, a required field. Read the next key and a window to classify;
        # required arrays/switches span braces, so include up to the next guard
        # or end. Find this statement's extent by reading to the next top-level
        # MaybeAddComma / if / end.
        nxt = L
        for r in (re_comma, re_if):
            mm = r.search(s, pos + 1)
            if mm:
                nxt = min(nxt, mm.start())
        window = s[pos:nxt]
        key = find_key(window)
        if key:
            fields.append((key, "req", classify_kind(window)))
            pos = nxt
        else:
            pos += 1
    # Variant fields emit the same JSON key once per `switch` case; collapse
    # consecutive duplicate keys into a single field (keeping the first's
    # presence/kind, which classifies the variant as an object).
    deduped = []
    for f in fields:
        if deduped and deduped[-1][0] == f[0]:
            continue
        deduped.append(f)
    return deduped

def classify_kind(text):
    if 'os << "[";' in text or '<< "[";' in text or '"[";' in text:
        if "element.has_value()" in text or ".has_value()) {\n          switch" in text:
            return "array_holes"
        return "array"
    if ".dump()" in text:
        return "scalar"
    if "->Serialize(os)" in text or ".index()" in text:
        return "object"
    return "scalar"

# ---------------------------------------------------------------------------
# Repr from header member type.
# ---------------------------------------------------------------------------

def strip_optional(t):
    m = re.match(r'std::optional<(.*)>$', t)
    return m.group(1) if m else t

def inner_unique(t):
    m = re.match(r'std::unique_ptr<(Js\w+)>$', t)
    return m.group(1) if m else None

def repr_for_type(cpptype):
    """Map a (already optional/vector-unwrapped element) C++ type to a Rust repr token."""
    t = cpptype.strip()
    if t == "bool":
        return "Bool"
    if t == "int64_t":
        return "Int"
    if t == "double":
        return "Float"
    if t == "std::string":
        return "Str"
    if t.endswith("Operator"):  # the 5 operator enums serialize as strings
        return "Str"
    u = inner_unique(t)
    if u:
        if u in FIXED_HELPERS:
            return f'Extra("{u[2:]}")'
        return "Node"
    if t.startswith("std::variant<"):
        return "Node"  # polymorphic, dispatch by "type"
    raise SystemExit(f"unhandled element type: {cpptype!r}")

def repr_for_member(cpptype):
    """Full repr including List wrapping. Returns repr token string."""
    t = cpptype.strip()
    t = strip_optional(t)
    mvec = re.match(r'std::vector<(.*)>$', t)
    if mvec:
        elem = mvec.group(1).strip()
        elem_opt = strip_optional(elem)
        holes = elem_opt != elem
        elem_repr = repr_for_type(elem_opt)
        if holes:
            return f'ListHoles(&{elem_repr})'
        return f'List(&{elem_repr})'
    return repr_for_type(t)

# ---------------------------------------------------------------------------
# Build schemas.
# ---------------------------------------------------------------------------

def build():
    classes = parse_classes()
    # Determine node set (has "type" tag) and helper set.
    type_tags = {}
    for cls in classes:
        tag = serialize_type_tag(cls)
        if tag:
            type_tags[cls] = tag

    # Per-class own field schema: zip serializer keys/presence with header types.
    own_schema = {}  # cls -> [(key, presence, repr)]
    for cls in classes:
        ser = parse_own_fields(cls)
        members = classes[cls]["members"]
        if len(ser) != len(members):
            # Try to align by count; mismatch indicates a parse gap.
            # Helpful debug:
            print(f"// WARN {cls}: ser={len(ser)} members={len(members)} "
                  f"ser={[s[0] for s in ser]} mem={[m[0] for m in members]}",
                  file=sys.stderr)
        schema = []
        for idx, (key, presence, kind) in enumerate(ser):
            cpptype = members[idx][1] if idx < len(members) else "std::string"
            rep = repr_for_member(cpptype)
            schema.append((key, presence, rep))
        own_schema[cls] = schema

    # Node full schema = concat of own schemas along Serialize() MRO.
    node_schema = {}
    for cls, tag in type_tags.items():
        mro = serialize_mro(cls)
        full = []
        for ancestor in mro:
            full.extend(own_schema.get(ancestor, []))
        node_schema[tag] = full

    # Helper (fixed, no type) schema = own schema directly.
    helper_schema = {}
    for cls in FIXED_HELPERS:
        helper_schema[cls[2:]] = own_schema.get(cls, [])

    return type_tags, node_schema, helper_schema

# ---------------------------------------------------------------------------
# Emit Rust.
# ---------------------------------------------------------------------------

def emit(type_tags, node_schema, helper_schema):
    out = []
    out.append("// @generated by tools/gen_ast.py from vendor/jsir-upstream. Do not edit.")
    out.append("#![allow(clippy::all)]")
    out.append("use crate::model::{FieldSpec, Presence::*, Repr::*};")
    out.append("")

    tags = sorted(node_schema.keys())
    out.append("/// All node `type` tag strings (polymorphic nodes).")
    out.append("pub const NODE_TYPES: &[&str] = &[")
    for t in tags:
        out.append(f'    "{t}",')
    out.append("];")
    out.append("")

    def emit_schema_fn(fn_name, key, schema):
        out.append(f'        "{key}" => &[')
        for (k, presence, rep) in schema:
            pres = {"req": "Required", "maybe_null": "MaybeNull", "maybe_undef": "MaybeUndef"}[presence]
            out.append(f'            FieldSpec {{ key: "{k}", presence: {pres}, repr: {rep} }},')
        out.append("        ],")

    out.append("/// Field schema for a node `type` (includes base fields then specific).")
    out.append("pub fn node_schema(ty: &str) -> &'static [FieldSpec] {")
    out.append("    match ty {")
    for t in tags:
        emit_schema_fn("node_schema", t, node_schema[t])
    out.append('        _ => &[],')
    out.append("    }")
    out.append("}")
    out.append("")

    out.append("/// Field schema for a fixed helper struct (no `type` tag).")
    out.append("pub fn helper_schema(name: &str) -> &'static [FieldSpec] {")
    out.append("    match name {")
    for name in sorted(helper_schema.keys()):
        emit_schema_fn("helper_schema", name, helper_schema[name])
    out.append('        _ => panic!("unknown helper struct: {name}"),')
    out.append("    }")
    out.append("}")
    out.append("")
    return "\n".join(out) + "\n"

if __name__ == "__main__":
    type_tags, node_schema, helper_schema = build()
    rust = emit(type_tags, node_schema, helper_schema)
    dest = os.path.join(ROOT, "crates/jsir-ast/src/schema_generated.rs")
    open(dest, "w").write(rust)
    print(f"wrote {dest}: {len(node_schema)} nodes, {len(helper_schema)} helpers")
