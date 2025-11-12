#lang pyret

import parse-pyret as P
import file as F
import ast as A
import string-dict as SD
import json as J
import cmdline as C

# Helper to convert Option<Expr> - visits the AST node
fun visit-option-expr(visitor, opt):
  cases (Option) opt:
    | none => nothing
    | some(v) => v.visit(visitor)
  end
end

# Helper to convert Option<Loc> - uses torepr for Srcloc
fun visit-option-loc(opt):
  cases (Option) opt:
    | none => nothing
    | some(v) => torepr(v)
  end
end

# Generic helper - tries to handle both
fun visit-option(visitor, opt):
  cases (Option) opt:
    | none => nothing
    | some(v) =>
      if is-nothing(v):
        nothing
      else if is-string(v) or is-number(v) or is-boolean(v):
        v
      else:
        # Assume it's an AST node
        v.visit(visitor)
      end
  end
end

# Escape string for JSON - handle newlines and other control characters
# The JSON library doesn't properly escape control chars in some cases
fun escape-json-string(s :: String) -> String:
  # Manually iterate through string and escape special chars
  fun helper(chars):
    cases (List) chars:
      | empty => ""
      | link(c, rest) =>
        # Only escape the most common problematic characters
        escaped = if c == "\n":
          "\\n"
        else if c == "\r":
          "\\r"
        else if c == "\t":
          "\\t"
        else:
          c
        end
        escaped + helper(rest)
    end
  end
  helper(string-explode(s))
end

# Comprehensive JSON visitor that handles all 177 AST node types
# Using a custom visitor object instead of extending default-map-visitor
json-visitor = {
  method option(self, opt):
    visit-option(self, opt)
  end,

  # ===== Name variants (6 types) =====
  method s-underscore(self, l):
    [SD.string-dict: "type", "s-underscore"]
  end,

  method s-name(self, l, s):
    [SD.string-dict: "type", "s-name", "name", s]
  end,

  method s-global(self, s):
    [SD.string-dict: "type", "s-global", "name", s]
  end,

  method s-module-global(self, s):
    [SD.string-dict: "type", "s-module-global", "name", s]
  end,

  method s-type-global(self, s):
    [SD.string-dict: "type", "s-type-global", "name", s]
  end,

  method s-atom(self, base, serial):
    [SD.string-dict: "type", "s-atom", "base", base, "serial", serial]
  end,

  # ===== AppInfo and PrimAppInfo =====
  method app-info-c(self, l, is-reactive):
    [SD.string-dict: "type", "app-info-c", "is-reactive", is-reactive]
  end,

  method prim-app-info-c(self, l, is-reactive):
    [SD.string-dict: "type", "prim-app-info-c", "is-reactive", is-reactive]
  end,

  # ===== Use =====
  method s-use(self, l, name, mod):
    [SD.string-dict: "type", "s-use", "name", name.visit(self), "mod", mod.visit(self)]
  end,

  # ===== Program =====
  method s-program(self, l, _use, _provide, provided-types, provides, imports, body):
    [SD.string-dict:
      "type", "s-program",
      "use", visit-option(self, _use),
      "provide", _provide.visit(self),
      "provided-types", provided-types.visit(self),
      "provides", provides.map(_.visit(self)),
      "imports", imports.map(_.visit(self)),
      "body", body.visit(self)
    ]
  end,

  # ===== Import variants (5 types) =====
  method s-include(self, l, import-type):
    [SD.string-dict: "type", "s-include", "import-type", import-type.visit(self)]
  end,

  method s-include-from(self, l, mod, specs):
    [SD.string-dict:
      "type", "s-include-from",
      "mod", mod.map(_.visit(self)),
      "specs", specs.map(_.visit(self))
    ]
  end,

  method s-import(self, l, import-type, name):
    [SD.string-dict:
      "type", "s-import",
      "import-type", import-type.visit(self),
      "name", name.visit(self)
    ]
  end,

  method s-import-types(self, l, import-type, name, types):
    [SD.string-dict:
      "type", "s-import-types",
      "import-type", import-type.visit(self),
      "name", name.visit(self),
      "types", types.visit(self)
    ]
  end,

  method s-import-fields(self, l, fields, import-type):
    [SD.string-dict:
      "type", "s-import-fields",
      "fields", fields.map(_.visit(self)),
      "import-type", import-type.visit(self)
    ]
  end,

  # ===== IncludeSpec variants (4 types) =====
  method s-include-name(self, l, name-spec):
    [SD.string-dict: "type", "s-include-name", "name-spec", name-spec.visit(self)]
  end,

  method s-include-data(self, l, name-spec, hidden):
    [SD.string-dict: "type", "s-include-data", "name-spec", name-spec.visit(self), "hidden", hidden.map(_.visit(self))]
  end,

  method s-include-type(self, l, name-spec):
    [SD.string-dict: "type", "s-include-type", "name-spec", name-spec.visit(self)]
  end,

  method s-include-module(self, l, name-spec):
    [SD.string-dict: "type", "s-include-module", "name-spec", name-spec.visit(self)]
  end,

  # ===== ProvidedModule, ProvidedValue, ProvidedAlias, ProvidedDatatype =====
  method p-module(self, l, name, original):
    [SD.string-dict: "type", "p-module", "name", name, "original", original.visit(self)]
  end,

  method p-value(self, l, name, original):
    [SD.string-dict: "type", "p-value", "name", name, "original", original.visit(self)]
  end,

  method p-alias(self, l, name, original):
    [SD.string-dict: "type", "p-alias", "name", name, "original", original.visit(self)]
  end,

  method p-data(self, l, name, original):
    [SD.string-dict: "type", "p-data", "name", name, "original", original.visit(self)]
  end,

  # ===== Provide variants (3 types) =====
  method s-provide(self, l, block):
    [SD.string-dict: "type", "s-provide", "block", block.visit(self)]
  end,

  method s-provide-all(self, l):
    [SD.string-dict: "type", "s-provide-all"]
  end,

  method s-provide-none(self, l):
    [SD.string-dict: "type", "s-provide-none"]
  end,

  # ===== ProvideBlock =====
  method s-provide-block(self, l, path, specs):
    [SD.string-dict:
      "type", "s-provide-block",
      "path", path.map(_.visit(self)),
      "specs", specs.map(_.visit(self))
    ]
  end,

  # ===== ProvideSpec variants (4 types) =====
  method s-provide-name(self, l, name-spec):
    [SD.string-dict: "type", "s-provide-name", "name-spec", name-spec.visit(self)]
  end,

  method s-provide-data(self, l, name-spec, hidden):
    [SD.string-dict: "type", "s-provide-data", "name-spec", name-spec.visit(self), "hidden", hidden.map(_.visit(self))]
  end,

  method s-provide-type(self, l, name-spec):
    [SD.string-dict: "type", "s-provide-type", "name-spec", name-spec.visit(self)]
  end,

  method s-provide-module(self, l, name-spec):
    [SD.string-dict: "type", "s-provide-module", "name-spec", name-spec.visit(self)]
  end,

  # ===== NameSpec variants (4 types) =====
  method s-star(self, l, hidden):
    [SD.string-dict: "type", "s-star", "hidden", hidden.map(_.visit(self))]
  end,

  method s-module-ref(self, l, path, as-name):
    [SD.string-dict: "type", "s-module-ref", "path", path.map(_.visit(self)), "as-name", visit-option(self, as-name)]
  end,

  method s-remote-ref(self, l, uri, name, as-name):
    [SD.string-dict: "type", "s-remote-ref", "uri", uri, "name", name.visit(self), "as-name", as-name.visit(self)]
  end,

  method s-local-ref(self, l, name, as-name):
    [SD.string-dict: "type", "s-local-ref", "name", name.visit(self), "as-name", as-name.visit(self)]
  end,

  # ===== ProvideTypes variants (3 types) =====
  method s-provide-types(self, l, anns):
    [SD.string-dict: "type", "s-provide-types", "anns", anns.map(_.visit(self))]
  end,

  method s-provide-types-all(self, l):
    [SD.string-dict: "type", "s-provide-types-all"]
  end,

  method s-provide-types-none(self, l):
    [SD.string-dict: "type", "s-provide-types-none"]
  end,

  # ===== ImportType variants (2 types) =====
  method s-const-import(self, l, mod):
    [SD.string-dict: "type", "s-const-import", "mod", mod]
  end,

  method s-special-import(self, l, kind, args):
    [SD.string-dict: "type", "s-special-import", "kind", kind, "args", args]
  end,

  # ===== Hint =====
  method h-use-loc(self, l):
    [SD.string-dict: "type", "h-use-loc"]
  end,

  # ===== LetBind variants (2 types) =====
  method s-let-bind(self, l, b, value):
    [SD.string-dict:
      "type", "s-let-bind",
      "bind", b.visit(self),
      "value", value.visit(self)
    ]
  end,

  method s-var-bind(self, l, b, value):
    [SD.string-dict:
      "type", "s-var-bind",
      "bind", b.visit(self),
      "value", value.visit(self)
    ]
  end,

  # ===== LetrecBind =====
  method s-letrec-bind(self, l, b, value):
    [SD.string-dict:
      "type", "s-letrec-bind",
      "bind", b.visit(self),
      "value", value.visit(self)
    ]
  end,

  # ===== TypeLetBind variants (2 types) =====
  method s-type-bind(self, l, name, params, ann):
    [SD.string-dict:
      "type", "s-type-bind",
      "name", name.visit(self),
      "params", params.map(_.visit(self)),
      "ann", ann.visit(self)
    ]
  end,

  method s-newtype-bind(self, l, name, namet):
    [SD.string-dict:
      "type", "s-newtype-bind",
      "name", name.visit(self),
      "namet", namet.visit(self)
    ]
  end,

  # ===== DefinedModule =====
  method s-defined-module(self, name, prov, value):
    [SD.string-dict:
      "type", "s-defined-module",
      "name", name,
      "prov", prov.visit(self),
      "value", value.visit(self)
    ]
  end,

  # ===== DefinedValue variants (2 types) =====
  method s-defined-value(self, name, value):
    [SD.string-dict:
      "type", "s-defined-value",
      "name", name,
      "value", value.visit(self)
    ]
  end,

  method s-defined-var(self, name, id):
    [SD.string-dict:
      "type", "s-defined-var",
      "name", name,
      "id", id.visit(self)
    ]
  end,

  # ===== DefinedType =====
  method s-defined-type(self, name, typ):
    [SD.string-dict:
      "type", "s-defined-type",
      "name", name,
      "typ", typ.visit(self)
    ]
  end,

  # ===== Expr variants (80 types!) =====
  method s-module(self, l, answer, dms, dvs, dts, checks):
    [SD.string-dict:
      "type", "s-module",
      "answer", answer.visit(self),
      "dms", dms.map(_.visit(self)),
      "dvs", dvs.map(_.visit(self)),
      "dts", dts.map(_.visit(self)),
      "checks", checks.visit(self)
    ]
  end,

  method s-template(self, l):
    [SD.string-dict: "type", "s-template"]
  end,

  method s-type-let-expr(self, l, binds, body, blocky):
    [SD.string-dict:
      "type", "s-type-let-expr",
      "binds", binds.map(_.visit(self)),
      "body", body.visit(self),
      "blocky", blocky
    ]
  end,

  method s-let-expr(self, l, binds, body, blocky):
    [SD.string-dict:
      "type", "s-let-expr",
      "binds", binds.map(_.visit(self)),
      "body", body.visit(self),
      "blocky", blocky
    ]
  end,

  method s-letrec(self, l, binds, body, blocky):
    [SD.string-dict:
      "type", "s-letrec",
      "binds", binds.map(_.visit(self)),
      "body", body.visit(self),
      "blocky", blocky
    ]
  end,

  method s-hint-exp(self, l, hints, exp):
    [SD.string-dict:
      "type", "s-hint-exp",
      "hints", hints.map(_.visit(self)),
      "exp", exp.visit(self)
    ]
  end,

  method s-instantiate(self, l, expr, params):
    [SD.string-dict:
      "type", "s-instantiate",
      "expr", expr.visit(self),
      "params", params.map(_.visit(self))
    ]
  end,

  method s-block(self, l, stmts):
    [SD.string-dict:
      "type", "s-block",
      "stmts", stmts.map(_.visit(self))
    ]
  end,

  method s-user-block(self, l, body):
    [SD.string-dict:
      "type", "s-user-block",
      "body", body.visit(self)
    ]
  end,

  method s-fun(self, l, name, params, args, ann, doc, body, _check-loc, _check, blocky):
    [SD.string-dict:
      "type", "s-fun",
      "name", name,
      "params", params.map(_.visit(self)),
      "args", args.map(_.visit(self)),
      "ann", ann.visit(self),
      "doc", doc,
      "body", body.visit(self),
      "check-loc", visit-option-loc(_check-loc),
      "check", visit-option-expr(self, _check),
      "blocky", blocky
    ]
  end,

  method s-type(self, l, name, params, ann):
    [SD.string-dict:
      "type", "s-type",
      "name", name.visit(self),
      "params", params.map(_.visit(self)),
      "ann", ann.visit(self)
    ]
  end,

  method s-newtype(self, l, name, namet):
    [SD.string-dict:
      "type", "s-newtype",
      "name", name.visit(self),
      "namet", namet.visit(self)
    ]
  end,

  method s-var(self, l, name, value):
    [SD.string-dict:
      "type", "s-var",
      "name", name.visit(self),
      "value", value.visit(self)
    ]
  end,

  method s-rec(self, l, name, value):
    [SD.string-dict:
      "type", "s-rec",
      "name", name.visit(self),
      "value", value.visit(self)
    ]
  end,

  method s-let(self, l, name, value, keyword-val):
    [SD.string-dict:
      "type", "s-let",
      "name", name.visit(self),
      "value", value.visit(self),
      "keyword-val", keyword-val
    ]
  end,

  method s-ref(self, l, ann):
    [SD.string-dict:
      "type", "s-ref",
      "ann", visit-option(self, ann)
    ]
  end,

  method s-contract(self, l, name, params, ann):
    [SD.string-dict:
      "type", "s-contract",
      "name", name.visit(self),
      "params", params.map(_.visit(self)),
      "ann", ann.visit(self)
    ]
  end,

  method s-when(self, l, test, block, blocky):
    [SD.string-dict:
      "type", "s-when",
      "test", test.visit(self),
      "block", block.visit(self),
      "blocky", blocky
    ]
  end,

  method s-assign(self, l, id, value):
    [SD.string-dict:
      "type", "s-assign",
      "id", id.visit(self),
      "value", value.visit(self)
    ]
  end,

  method s-if-pipe(self, l, branches, blocky):
    [SD.string-dict:
      "type", "s-if-pipe",
      "branches", branches.map(_.visit(self)),
      "blocky", blocky
    ]
  end,

  method s-if-pipe-else(self, l, branches, _else, blocky):
    [SD.string-dict:
      "type", "s-if-pipe-else",
      "branches", branches.map(_.visit(self)),
      "else", _else.visit(self),
      "blocky", blocky
    ]
  end,

  method s-if(self, l, branches, blocky):
    [SD.string-dict:
      "type", "s-if",
      "branches", branches.map(_.visit(self)),
      "blocky", blocky
    ]
  end,

  method s-if-else(self, l, branches, _else, blocky):
    [SD.string-dict:
      "type", "s-if-else",
      "branches", branches.map(_.visit(self)),
      "else", _else.visit(self),
      "blocky", blocky
    ]
  end,

  method s-cases(self, l, typ, val, branches, blocky):
    [SD.string-dict:
      "type", "s-cases",
      "typ", typ.visit(self),
      "val", val.visit(self),
      "branches", branches.map(_.visit(self)),
      "blocky", blocky
    ]
  end,

  method s-cases-else(self, l, typ, val, branches, _else, blocky):
    [SD.string-dict:
      "type", "s-cases-else",
      "typ", typ.visit(self),
      "val", val.visit(self),
      "branches", branches.map(_.visit(self)),
      "else", _else.visit(self),
      "blocky", blocky
    ]
  end,

  method s-op(self, l, op-l, op, left, right):
    [SD.string-dict:
      "type", "s-op",
      "op-l", torepr(op-l),
      "op", op,
      "left", left.visit(self),
      "right", right.visit(self)
    ]
  end,

  method s-check-test(self, l, op, refinement, left, right, cause):
    [SD.string-dict:
      "type", "s-check-test",
      "op", op.visit(self),
      "refinement", visit-option(self, refinement),
      "left", left.visit(self),
      "right", visit-option(self, right),
      "cause", visit-option(self, cause)
    ]
  end,

  method s-check-expr(self, l, expr, ann):
    [SD.string-dict:
      "type", "s-check-expr",
      "expr", expr.visit(self),
      "ann", ann.visit(self)
    ]
  end,

  method s-paren(self, l, expr):
    [SD.string-dict:
      "type", "s-paren",
      "expr", expr.visit(self)
    ]
  end,

  method s-lam(self, l, name, params, args, ann, doc, body, _check-loc, _check, blocky):
    [SD.string-dict:
      "type", "s-lam",
      "name", name,
      "params", params.map(_.visit(self)),
      "args", args.map(_.visit(self)),
      "ann", ann.visit(self),
      "doc", doc,
      "body", body.visit(self),
      "check-loc", visit-option-loc(_check-loc),
      "check", visit-option-expr(self, _check),
      "blocky", blocky
    ]
  end,

  method s-method(self, l, name, params, args, ann, doc, body, _check-loc, _check, blocky):
    [SD.string-dict:
      "type", "s-method",
      "name", name,
      "params", params.map(_.visit(self)),
      "args", args.map(_.visit(self)),
      "ann", ann.visit(self),
      "doc", doc,
      "body", body.visit(self),
      "check-loc", visit-option-loc(_check-loc),
      "check", visit-option-expr(self, _check),
      "blocky", blocky
    ]
  end,

  method s-extend(self, l, supe, fields):
    [SD.string-dict:
      "type", "s-extend",
      "super", supe.visit(self),
      "fields", fields.map(_.visit(self))
    ]
  end,

  method s-update(self, l, supe, fields):
    [SD.string-dict:
      "type", "s-update",
      "super", supe.visit(self),
      "fields", fields.map(_.visit(self))
    ]
  end,

  method s-tuple(self, l, fields):
    [SD.string-dict:
      "type", "s-tuple",
      "fields", fields.map(_.visit(self))
    ]
  end,

  method s-tuple-get(self, l, tup, index, index-loc):
    [SD.string-dict:
      "type", "s-tuple-get",
      "tup", tup.visit(self),
      "index", index,
      "index-loc", torepr(index-loc)
    ]
  end,

  method s-obj(self, l, fields):
    [SD.string-dict:
      "type", "s-obj",
      "fields", fields.map(_.visit(self))
    ]
  end,

  method s-array(self, l, values):
    [SD.string-dict:
      "type", "s-array",
      "values", values.map(_.visit(self))
    ]
  end,

  method s-construct(self, l, modifier, constructor, values):
    [SD.string-dict:
      "type", "s-construct",
      "modifier", modifier.visit(self),
      "constructor", constructor.visit(self),
      "values", values.map(_.visit(self))
    ]
  end,

  method s-app(self, l, _fun, args):
    [SD.string-dict:
      "type", "s-app",
      "fun", _fun.visit(self),
      "args", args.map(_.visit(self))
    ]
  end,

  method s-app-enriched(self, l, _fun, args, app-info):
    [SD.string-dict:
      "type", "s-app-enriched",
      "fun", _fun.visit(self),
      "args", args.map(_.visit(self)),
      "app-info", app-info.visit(self)
    ]
  end,

  method s-prim-app(self, l, _fun, args, app-info):
    [SD.string-dict:
      "type", "s-prim-app",
      "fun", _fun,
      "args", args.map(_.visit(self)),
      "app-info", app-info.visit(self)
    ]
  end,

  method s-prim-val(self, l, name):
    [SD.string-dict:
      "type", "s-prim-val",
      "name", name
    ]
  end,

  method s-id(self, l, id):
    [SD.string-dict: "type", "s-id", "id", id.visit(self)]
  end,

  method s-id-var(self, l, id):
    [SD.string-dict: "type", "s-id-var", "id", id.visit(self)]
  end,

  method s-id-letrec(self, l, id, safe):
    [SD.string-dict: "type", "s-id-letrec", "id", id.visit(self), "safe", safe]
  end,

  method s-id-var-modref(self, l, id, uri, name):
    [SD.string-dict:
      "type", "s-id-var-modref",
      "id", id.visit(self),
      "uri", uri,
      "name", name
    ]
  end,

  method s-id-modref(self, l, id, uri, name):
    [SD.string-dict:
      "type", "s-id-modref",
      "id", id.visit(self),
      "uri", uri,
      "name", name
    ]
  end,

  method s-undefined(self, l):
    [SD.string-dict: "type", "s-undefined"]
  end,

  method s-srcloc(self, l, shadow loc):
    [SD.string-dict: "type", "s-srcloc", "loc", torepr(loc)]
  end,

  method s-num(self, l, n):
    [SD.string-dict: "type", "s-num", "value", torepr(n)]
  end,

  method s-frac(self, l, num, den):
    [SD.string-dict: "type", "s-frac", "num", torepr(num), "den", torepr(den)]
  end,

  method s-rfrac(self, l, num, den):
    [SD.string-dict: "type", "s-rfrac", "num", torepr(num), "den", torepr(den)]
  end,

  method s-bool(self, l, b):
    [SD.string-dict: "type", "s-bool", "value", b]
  end,

  method s-str(self, l, s):
    [SD.string-dict: "type", "s-str", "value", s]
  end,

  method s-dot(self, l, obj, field):
    [SD.string-dict:
      "type", "s-dot",
      "obj", obj.visit(self),
      "field", field
    ]
  end,

  method s-get-bang(self, l, obj, field):
    [SD.string-dict:
      "type", "s-get-bang",
      "obj", obj.visit(self),
      "field", field
    ]
  end,

  method s-bracket(self, l, obj, field):
    [SD.string-dict:
      "type", "s-bracket",
      "obj", obj.visit(self),
      "field", field.visit(self)
    ]
  end,

  method s-data(self, l, name, params, mixins, variants, shared-members, _check-loc, _check):
    [SD.string-dict:
      "type", "s-data",
      "name", name,
      "params", params.map(_.visit(self)),
      "mixins", mixins.map(_.visit(self)),
      "variants", variants.map(_.visit(self)),
      "shared-members", shared-members.map(_.visit(self)),
      "check-loc", visit-option-loc(_check-loc),
      "check", visit-option-expr(self, _check)
    ]
  end,

  method s-data-expr(self, l, name, namet, params, mixins, variants, shared-members, _check-loc, _check):
    [SD.string-dict:
      "type", "s-data-expr",
      "name", name,
      "namet", namet.visit(self),
      "params", params.map(_.visit(self)),
      "mixins", mixins.map(_.visit(self)),
      "variants", variants.map(_.visit(self)),
      "shared-members", shared-members.map(_.visit(self)),
      "check-loc", visit-option-loc(_check-loc),
      "check", visit-option-expr(self, _check)
    ]
  end,

  method s-for(self, l, iterator, bindings, ann, body, blocky):
    [SD.string-dict:
      "type", "s-for",
      "iterator", iterator.visit(self),
      "bindings", bindings.map(_.visit(self)),
      "ann", ann.visit(self),
      "body", body.visit(self),
      "blocky", blocky
    ]
  end,

  method s-check(self, l, name, body, keyword-check):
    [SD.string-dict:
      "type", "s-check",
      "name", visit-option(self, name),
      "body", body.visit(self),
      "keyword-check", keyword-check
    ]
  end,

  method s-reactor(self, l, fields):
    [SD.string-dict:
      "type", "s-reactor",
      "fields", fields.map(_.visit(self))
    ]
  end,

  method s-table-extend(self, l, column-binds, extensions):
    [SD.string-dict:
      "type", "s-table-extend",
      "column-binds", column-binds.visit(self),
      "extensions", extensions.map(_.visit(self))
    ]
  end,

  method s-table-update(self, l, column-binds, updates):
    [SD.string-dict:
      "type", "s-table-update",
      "column-binds", column-binds.visit(self),
      "updates", updates.map(_.visit(self))
    ]
  end,

  method s-table-select(self, l, columns, table):
    [SD.string-dict:
      "type", "s-table-select",
      "columns", columns.map(_.visit(self)),
      "table", table.visit(self)
    ]
  end,

  method s-table-order(self, l, table, ordering):
    [SD.string-dict:
      "type", "s-table-order",
      "table", table.visit(self),
      "ordering", ordering.map(_.visit(self))
    ]
  end,

  method s-table-filter(self, l, column-binds, predicate):
    [SD.string-dict:
      "type", "s-table-filter",
      "column-binds", column-binds.visit(self),
      "predicate", predicate.visit(self)
    ]
  end,

  method s-table-extract(self, l, column, table):
    [SD.string-dict:
      "type", "s-table-extract",
      "column", column.visit(self),
      "table", table.visit(self)
    ]
  end,

  method s-table(self, l, headers, rows):
    [SD.string-dict:
      "type", "s-table",
      "headers", headers.map(_.visit(self)),
      "rows", rows.map(_.visit(self))
    ]
  end,

  method s-load-table(self, l, headers, spec):
    [SD.string-dict:
      "type", "s-load-table",
      "headers", headers.map(_.visit(self)),
      "spec", spec.map(_.visit(self))
    ]
  end,

  method s-spy-block(self, l, message, contents):
    [SD.string-dict:
      "type", "s-spy-block",
      "message", visit-option(self, message),
      "contents", contents.map(_.visit(self))
    ]
  end,

  # ===== TableRow =====
  method s-table-row(self, l, elems):
    [SD.string-dict:
      "type", "s-table-row",
      "elems", elems.map(_.visit(self))
    ]
  end,

  # ===== SpyField =====
  method s-spy-expr(self, l, name, value, implicit-label):
    [SD.string-dict:
      "type", "s-spy-expr",
      "name", name,
      "value", value.visit(self),
      "implicit-label", implicit-label
    ]
  end,

  # ===== ConstructModifier variants (2 types) =====
  method s-construct-normal(self):
    [SD.string-dict: "type", "s-construct-normal"]
  end,

  method s-construct-lazy(self):
    [SD.string-dict: "type", "s-construct-lazy"]
  end,

  # ===== Bind variants (2 types) =====
  method s-bind(self, l, shadows, name, ann):
    [SD.string-dict:
      "type", "s-bind",
      "shadows", shadows,
      "name", name.visit(self),
      "ann", ann.visit(self)
    ]
  end,

  method s-tuple-bind(self, l, fields, as-name):
    [SD.string-dict:
      "type", "s-tuple-bind",
      "fields", fields.map(_.visit(self)),
      "as-name", visit-option(self, as-name)
    ]
  end,

  # ===== Member variants (3 types) =====
  method s-data-field(self, l, name, value):
    [SD.string-dict:
      "type", "s-data-field",
      "name", name,
      "value", value.visit(self)
    ]
  end,

  method s-mutable-field(self, l, name, ann, value):
    [SD.string-dict:
      "type", "s-mutable-field",
      "name", name,
      "ann", ann.visit(self),
      "value", value.visit(self)
    ]
  end,

  method s-method-field(self, l, name, params, args, ann, doc, body, _check-loc, _check, blocky):
    [SD.string-dict:
      "type", "s-method-field",
      "name", name,
      "params", params.map(_.visit(self)),
      "args", args.map(_.visit(self)),
      "ann", ann.visit(self),
      "doc", doc,
      "body", body.visit(self),
      "check-loc", visit-option-loc(_check-loc),
      "check", visit-option-expr(self, _check),
      "blocky", blocky
    ]
  end,

  # ===== FieldName =====
  method s-field-name(self, l, name, value):
    [SD.string-dict:
      "type", "s-field-name",
      "name", name,
      "value", value.visit(self)
    ]
  end,

  # ===== ForBind =====
  method s-for-bind(self, l, bind, value):
    [SD.string-dict:
      "type", "s-for-bind",
      "bind", bind.visit(self),
      "value", value.visit(self)
    ]
  end,

  # ===== ColumnBinds =====
  method s-column-binds(self, l, binds, table):
    [SD.string-dict:
      "type", "s-column-binds",
      "table", table.visit(self),
      "binds", binds.map(_.visit(self))
    ]
  end,

  # ===== ColumnSortOrder variants (2 types) =====
  method ASCENDING(self):
    [SD.string-dict: "type", "ASCENDING"]
  end,

  method DESCENDING(self):
    [SD.string-dict: "type", "DESCENDING"]
  end,

  # ===== ColumnSort =====
  method s-column-sort(self, l, column, direction):
    [SD.string-dict:
      "type", "s-column-sort",
      "column", column.visit(self),
      "direction", direction.visit(self)
    ]
  end,

  # ===== TableExtendField variants (2 types) =====
  method s-table-extend-field(self, l, name, value, ann):
    [SD.string-dict:
      "type", "s-table-extend-field",
      "name", name.visit(self),
      "value", value.visit(self),
      "ann", ann.visit(self)
    ]
  end,

  method s-table-extend-reducer(self, l, name, reducer, col, ann):
    [SD.string-dict:
      "type", "s-table-extend-reducer",
      "name", name,
      "reducer", reducer.visit(self),
      "col", col.visit(self),
      "ann", ann.visit(self)
    ]
  end,

  # ===== LoadTableSpec variants (2 types) =====
  method s-sanitize(self, l, name, sanitizer):
    [SD.string-dict:
      "type", "s-sanitize",
      "name", name.visit(self),
      "sanitizer", sanitizer.visit(self)
    ]
  end,

  method s-table-src(self, l, src):
    [SD.string-dict:
      "type", "s-table-src",
      "src", src.visit(self)
    ]
  end,

  # ===== VariantMemberType variants (2 types) =====
  method s-normal(self):
    [SD.string-dict: "type", "s-normal"]
  end,

  method s-mutable(self):
    [SD.string-dict: "type", "s-mutable"]
  end,

  # ===== VariantMember =====
  method s-variant-member(self, l, member-type, bind):
    [SD.string-dict:
      "type", "s-variant-member",
      "member-type", member-type.visit(self),
      "bind", bind.visit(self)
    ]
  end,

  # ===== Variant variants (2 types) =====
  method s-variant(self, l, constr-loc, name, members, with-members):
    [SD.string-dict:
      "type", "s-variant",
      "constr-loc", torepr(constr-loc),
      "name", name,
      "members", members.map(_.visit(self)),
      "with-members", with-members.map(_.visit(self))
    ]
  end,

  method s-singleton-variant(self, l, name, with-members):
    [SD.string-dict:
      "type", "s-singleton-variant",
      "name", name,
      "with-members", with-members.map(_.visit(self))
    ]
  end,

  # ===== IfBranch =====
  method s-if-branch(self, l, test, body):
    [SD.string-dict:
      "type", "s-if-branch",
      "test", test.visit(self),
      "body", body.visit(self)
    ]
  end,

  # ===== IfPipeBranch =====
  method s-if-pipe-branch(self, l, test, body):
    [SD.string-dict:
      "type", "s-if-pipe-branch",
      "test", test.visit(self),
      "body", body.visit(self)
    ]
  end,

  # ===== CasesBindType variants (2 types) =====
  method s-cases-bind-ref(self):
    [SD.string-dict: "type", "s-cases-bind-ref"]
  end,

  method s-cases-bind-normal(self):
    [SD.string-dict: "type", "s-cases-bind-normal"]
  end,

  # ===== CasesBind =====
  method s-cases-bind(self, l, field-type, bind):
    [SD.string-dict:
      "type", "s-cases-bind",
      "field-type", torepr(field-type),
      "bind", bind.visit(self)
    ]
  end,

  # ===== CasesBranch variants (2 types) =====
  method s-cases-branch(self, l, pat-loc, name, args, body):
    [SD.string-dict:
      "type", "s-cases-branch",
      "pat-loc", torepr(pat-loc),
      "name", name,
      "args", args.map(_.visit(self)),
      "body", body.visit(self)
    ]
  end,

  method s-singleton-cases-branch(self, l, pat-loc, name, body):
    [SD.string-dict:
      "type", "s-singleton-cases-branch",
      "pat-loc", torepr(pat-loc),
      "name", name,
      "body", body.visit(self)
    ]
  end,

  # ===== CheckOp variants (13 types) =====
  method s-op-is(self, l):
    [SD.string-dict: "type", "s-op-is"]
  end,

  method s-op-is-roughly(self, l):
    [SD.string-dict: "type", "s-op-is-roughly"]
  end,

  method s-op-is-not-roughly(self, l):
    [SD.string-dict: "type", "s-op-is-not-roughly"]
  end,

  method s-op-is-op(self, l, op):
    [SD.string-dict: "type", "s-op-is-op", "op", op]
  end,

  method s-op-is-not(self, l):
    [SD.string-dict: "type", "s-op-is-not"]
  end,

  method s-op-is-not-op(self, l, op):
    [SD.string-dict: "type", "s-op-is-not-op", "op", op]
  end,

  method s-op-satisfies(self, l):
    [SD.string-dict: "type", "s-op-satisfies"]
  end,

  method s-op-satisfies-not(self, l):
    [SD.string-dict: "type", "s-op-satisfies-not"]
  end,

  method s-op-raises(self, l):
    [SD.string-dict: "type", "s-op-raises"]
  end,

  method s-op-raises-other(self, l):
    [SD.string-dict: "type", "s-op-raises-other"]
  end,

  method s-op-raises-not(self, l):
    [SD.string-dict: "type", "s-op-raises-not"]
  end,

  method s-op-raises-satisfies(self, l):
    [SD.string-dict: "type", "s-op-raises-satisfies"]
  end,

  method s-op-raises-violates(self, l):
    [SD.string-dict: "type", "s-op-raises-violates"]
  end,

  # ===== Ann variants (13 types) =====
  method a-blank(self):
    [SD.string-dict: "type", "a-blank"]
  end,

  method a-any(self, l):
    [SD.string-dict: "type", "a-any"]
  end,

  method a-name(self, l, id):
    [SD.string-dict: "type", "a-name", "id", id.visit(self)]
  end,

  method a-type-var(self, l, id):
    [SD.string-dict: "type", "a-type-var", "id", id.visit(self)]
  end,

  method a-arrow(self, l, args, ret, use-parens):
    [SD.string-dict:
      "type", "a-arrow",
      "args", args.map(_.visit(self)),
      "ret", ret.visit(self),
      "use-parens", use-parens
    ]
  end,

  method a-arrow-argnames(self, l, args, ret, use-parens):
    [SD.string-dict:
      "type", "a-arrow-argnames",
      "args", args.map(_.visit(self)),
      "ret", ret.visit(self),
      "use-parens", use-parens
    ]
  end,

  method a-method(self, l, args, ret):
    [SD.string-dict:
      "type", "a-method",
      "args", args.map(_.visit(self)),
      "ret", ret.visit(self)
    ]
  end,

  method a-record(self, l, fields):
    [SD.string-dict:
      "type", "a-record",
      "fields", fields.map(_.visit(self))
    ]
  end,

  method a-tuple(self, l, fields):
    [SD.string-dict:
      "type", "a-tuple",
      "fields", fields.map(_.visit(self))
    ]
  end,

  method a-app(self, l, ann, args):
    [SD.string-dict:
      "type", "a-app",
      "ann", ann.visit(self),
      "args", args.map(_.visit(self))
    ]
  end,

  method a-pred(self, l, ann, exp):
    [SD.string-dict:
      "type", "a-pred",
      "ann", ann.visit(self),
      "exp", exp.visit(self)
    ]
  end,

  method a-dot(self, l, obj, field):
    [SD.string-dict:
      "type", "a-dot",
      "obj", obj.visit(self),
      "field", field
    ]
  end,

  method a-checked(self, checked, residual):
    [SD.string-dict:
      "type", "a-checked",
      "checked", checked.visit(self),
      "residual", residual.visit(self)
    ]
  end,

  # ===== AField =====
  method a-field(self, l, name, ann):
    [SD.string-dict:
      "type", "a-field",
      "name", name,
      "ann", ann.visit(self)
    ]
  end
}

# Main code: read file from command line and convert to JSON
cases (List<String>) C.other-args block:
  | empty => block:
    print("Usage: node ast-to-json.jarr <input.arr> [output.json]")
    print("Converts a Pyret file to its AST representation in JSON format")
    print("If output file is not specified, writes to /tmp/pyret-ast-output.json")
  end
  | link(filename, rest) => block:
    source = F.file-to-string(filename)
    parsed = P.surface-parse(source, filename)
    dict = parsed.visit(json-visitor)
    json = J.tojson(dict)
    output = json.serialize()

    # Determine output file
    output-path = cases (List<String>) rest block:
      | empty => "/tmp/pyret-ast-output.json"
      | link(outfile, _) => outfile
    end

    # Write to file
    output-file = F.output-file(output-path, false)
    output-file.display(output)
    output-file.close-file()

    print("JSON written to " + output-path)
  end
end

