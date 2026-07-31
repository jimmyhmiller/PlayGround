# KNOWN GAP: a CommonJS namespace's key set is STATIC in Node, and diffpack has no
# equivalent of `cjs-module-lexer`

Node builds a CommonJS module's ES namespace from the export names a *source
lexer* (`cjs-module-lexer`) finds — every `exports.<name> =` in the text, whether
or not it has run — and gives each the value it holds at the moment the namespace
is materialized. In this cycle that moment is BEFORE `exports.late = "late"` runs,
so Node prints `late:undefined` while still accepting the import.

diffpack has no static export-name set for a CommonJS module at runtime, so
`__import` cannot tell "a name this module will assign later" from "a name this
module never mentions". It resolves the ambiguity in the direction that keeps the
strict check honest: it reads through to the live `module.exports`, so a
late-assigned name reads its CURRENT value (`late:late` here) and a name that is
on neither the wrapper nor the exports is a hard `SyntaxError`, as in Node. The
opposite choice — a pure wrap-time snapshot — would make `import { typo }` from a
CommonJS module evaluate to `undefined` instead of throwing, which is the exact
silent wrongness `cjs-missing-named-throws` exists to prevent.

Closing this properly means detecting CommonJS export names statically at build
time (diffpack already parses every module) and building the namespace from that
set with wrap-time values. That is a feature, not a patch, and it is NOT done.

Scoreboard on this fixture: esbuild matches Node; diffpack and rolldown both print
`late:late`.
