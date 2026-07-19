# coil — Language Specification (surface + reader)

Status: draft, paired with `DESIGN.md` (rationale) and `KERNEL.md` (primitives).
This file defines the *surface*: how text reads into `Val`, the core forms, and
the desugaring rules that map everything to MLIR's generic operation form.

Everything here is either (a) a reader rule, (b) one of a small set of **special
forms** the evaluator knows intrinsically, or (c) **sugar** that a prelude macro
expands. Sugar is marked ⟨macro⟩ — it is *not* in the kernel (see `prelude.coil`).

---

## 1. Reader grammar

```
program    = form* ;
form       = atom | list | vector | map | reader-macro ;

list       = "(" form* ")" ;
vector     = "[" form* "]" ;
map        = "{" (form form)* "}" ;

atom       = number | string | bool | nil | keyword | symbol
           | type-lit | attr-lit | ssa-ref | block-ref ;

number     = int | float ;            ; 42, -7, 0x1F, 3.14, 1e9
string     = '"' char* '"' ;          ; supports \n \t \\ \" \0
bool       = "true" | "false" ;
nil        = "nil" ;
keyword    = ":" symbol-chars ;       ; :value :sym_name
symbol     = symbol-chars ;           ; foo, arith.addi, my/helper, +, ->

type-lit   = "!" balanced ;           ; !llvm.ptr  !llvm.struct<(i64,i64)>
attr-lit   = "#" balanced ;           ; #llvm.linkage<internal>  (# slt) via list form
ssa-ref    = "%" symbol-chars ;       ; %0, %acc   (optional; let-bindings preferred)
block-ref  = "^" symbol-chars ;       ; ^bb1
```

Reader notes:

- `!…` and `#…` consume a *balanced* token (matched `<>`, `()`, `[]`) so
  `!llvm.struct<(i64, i64)>` reads as one type literal. These become
  `(mlir/parse-type "…")` / `(mlir/parse-attr "…")` calls — the universal escape
  hatch for any dialect.
- A symbol containing `.` (e.g. `arith.addi`) is *not* special to the reader —
  it's an ordinary symbol. The `.` only gains meaning in **operator position**
  (head of a list), where the `op` sugar fires (§4).
- `;` line comments, `#_ form` datum comment, `#| … |#` block comment.
- `'x`→`(quote x)`, `` `x ``→`(quasiquote x)`, `~x`→`(unquote x)`,
  `~@x`→`(unquote-splicing x)`, `@sym`→`(symbol-ref sym)`.

---

## 2. Special forms (known to the evaluator)

These cannot be macros; they are the kernel's intrinsic forms. Full list and
semantics in `KERNEL.md §3`.

```
(quote x)            (quasiquote x) (unquote x) (unquote-splicing x)
(if c t e)           (do e…)        (let [b v …] e…)      (fn [p…] e…)
(def name v)         (defmacro name [p…] e…)
(eval-when phase e…) (require m …)  (require-for-syntax m …)
```

`let` here is the **kernel** let: it binds an evaluated value (which may be an
`MlirValue`, a `Type`, a number — anything) to a name in lexical scope. The
"SSA let" is the same form; binding an op-form's result just yields an
`MlirValue`. There is no separate binder.

---

## 3. The one op form

The single primitive that produces IR is `op` (kernel-level, see
`KERNEL.md §4`):

```clojure
(op NAME
    :operands   [v …]      ; MlirValue list             (default [])
    :results    [t …]      ; result Types; omit ⇒ infer (default :infer)
    :attrs      {k v …}    ; attribute map              (default {})
    :regions    [r …]      ; Region list                (default [])
    :successors [b …])     ; Block refs                 (default [])
;; ⇒ MlirOp, inserted at the current insertion point; its results are MlirValues
```

`NAME` is a string or symbol like `"arith.addi"`. `:results` semantics:

- omitted / `:infer` → use `InferTypeOpInterface` if the op has it; else 0 results.
- `[t …]` → exactly these result types (no inference).
- `:infer-or [t…]` → try inference, fall back to the given types.

This is the *whole* mapping. Everything below desugars to `op`.

---

## 4. Operator-position sugar ⟨macro⟩

When a list's head is a symbol containing `.`, it is an **op call**:

```
(d.op a b)                 ≡ (op "d.op" :operands [a b])
(d.op {attrs} a b)         ≡ (op "d.op" :attrs {attrs} :operands [a b])
(d.op {attrs} a b (region …) (region …))
                           ≡ (op "d.op" :attrs {attrs} :operands [a b]
                                        :regions [(region …) (region …)])
```

Disambiguation rules (resolved by the `op-call` reader/expander helper):

- A leading `{…}` map is `:attrs`.
- Trailing `(region …)` / `(do …)` forms are collected into `:regions`.
- `^name` arguments are collected into `:successors`.
- Everything else is an `:operand`.
- Result binding uses `let`/`def`, not a slot: `(let [s (arith.addi a b)] …)`.

No-dot heads (`if`, `let`, `defn`, `+`, user macros) are looked up as
macros/functions as usual; they are never op calls.

---

## 5. Types and attributes

Types and attributes evaluate to `MlirType` / `MlirAttr`.

```clojure
;; types
i1 i8 i16 i32 i64  index  f16 f32 f64  bf16        ; ⟨prelude consts⟩
(integer-type 7)            ; i7                    ⟨prelude fn⟩
(vector-type [4] f32)       ; vector<4xf32>         ⟨prelude fn⟩
(memref-type [? 3] f32)     ; memref<?x3xf32>       ⟨prelude fn⟩
(fn-type [i32 i32] [i32])   ; (i32,i32)->i32        ⟨prelude fn⟩
!llvm.ptr                   ; parse escape hatch    (reader)

;; attributes
(: 42 i32)                  ; typed int attr / constant (context-sensitive) ⟨macro⟩
(# slt)                     ; named attr "slt" for predicates                ⟨macro⟩
@printf                     ; (symbol-ref printf) → FlatSymbolRefAttr        (reader)
#llvm.linkage<internal>     ; parse escape hatch                              (reader)
{:sym_name "f" :value 3}    ; attr map: keywords→ident, values→attrs
```

`(: v t)` is context-sensitive sugar:

- in **operand** position → emits `(arith.constant {:value (int-attr v t)})` and
  yields its `MlirValue`.
- in **attribute** position (map value) → yields the `IntegerAttr` itself.

This single rule replaces `lispier`'s scattered constant/attr handling.

---

## 6. Blocks, regions, modules

```clojure
(region BODY…)                       ; ⟨macro⟩ region with a single entry block
(region (block ^entry [(: x i32)]    ; explicit multi-block region
          …)
        (block ^loop  [(: i i32)]
          …))

(block ^name [params…] BODY…)        ; params are typed; bound as MlirValues
```

- `region` with no explicit `block` wraps its body in one entry block that takes
  the surrounding op's expected args (used by `scf`/`func` bodies).
- A region's result is whatever its terminator yields; ⟨macro⟩ sugar may insert
  an implicit `scf.yield`/`func.return` of the last expression (opt-in per op,
  defined in the prelude — never in the kernel).
- The top-level program runs inside an implicit `builtin.module`, bound to
  `*module*`. `(module BODY…)` introduces a nested module.

---

## 7. Definitions that produce ops ⟨prelude macros⟩

```clojure
(defn name [(: p t) …] -> ret BODY…)   ; → func.func   (see prelude.coil)
(extern name (fn-type […] […]) :vararg b?)  ; → func.func {sym_visibility private} / llvm decl
(defstruct Name [field t] …)            ; → struct type + accessor macros
(defdialect d (defop …) …)              ; → irdl ops, registered now
(defpattern name :match … :rewrite …)   ; → pdl pattern, registered now
```

All of these are macros over `op`/kernel primitives. The kernel knows none of
them. This is the load-bearing claim of the design; `prelude.coil` discharges it.

---

## 8. Phases and modules

```clojure
(require "std/arith.coil")               ; runtime dep
(require-for-syntax "std/derive.coil")   ; available during macro expansion
(eval-when :compile  EXPR…)              ; run during meta-evaluation
(require-dialect arith func scf)         ; ⟨macro⟩ → mlir/load-dialect at compile time
```

Loading order per module: read → resolve `require`/`require-for-syntax`
(depth-first, compile-time deps first) → expand macros with the compile-time
environment populated → build IR. This is the principled version of the
`require` vs `require-macros` split that `lispier` never unified.

---

## 9. Worked desugaring

```clojure
(defn add [(: a i32) (: b i32)] -> i32
  (func.return (arith.addi a b)))
```

expands (eliding hygiene renames) to:

```clojure
(op "func.func"
    :attrs {:sym_name "add"
            :function_type (fn-type [i32 i32] [i32])
            :llvm.emit_c_interface true}
    :regions
    [(op-region
       (block ^entry [(: a i32) (: b i32)]
         (op "func.return"
             :operands [(op "arith.addi" :operands [a b])])))])
```

— a single tree of `op`/`block` kernel calls. `arith.addi`'s result type comes
from `InferTypeOpInterface` at build time, not from a language rule.
