# Grammar v0 (Draft)

This is the minimal grammar to start implementation. It is not yet fully specified; it is meant to be enough for the bootstrap compiler.

## 1. Lexical

### Whitespace and comments

- Whitespace: space, tab, newline are ignored except to separate tokens.
- Line comments: `// ...` to end of line.
- Block comments: `/* ... */` (no nesting for v0).

### Identifiers

```
ident = (alpha | '_') (alpha | digit | '_')*
alpha = [A-Za-z]
digit = [0-9]
```

### Keywords

```
module, use, pub, struct, enum, trait, impl, fn,
let, mut, if, else, while, match, return,
extern, repr, true, false
```

### Literals

- Integers: decimal only for v0. Example: `0`, `42`, `1_000`.
- Floats: `123.45`, `1.0`, `0.5`.
- Strings: double-quoted UTF-8, with `\n`, `\t`, `\"`, `\\` escapes.
- Chars: single-quoted ASCII byte, with `\n`, `\t`, `\r`, `\'`, `\\`, `\0`, `\xNN` escapes.
- Unit: `()` literal.

## 2. Operators and precedence

From highest to lowest:

1. Field access: `.`
2. Call: `()`
3. Unary: `-`, `!`
4. Multiplicative: `*`, `/`, `%`
5. Additive: `+`, `-`
6. Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
7. Logical AND: `&&`
8. Logical OR: `||`
9. Assignment: `=`

All binary operators are left-associative. Assignment is right-associative.

## 3. Top-level

```
module = module_decl? item*
module_decl = 'module' path ';'
path = ident ('::' ident)*

item = struct_decl | enum_decl | trait_decl | impl_decl | fn_decl | extern_decl
```

## 4. Declarations

```
struct_decl = vis? 'struct' ident type_params? '{' field_list? '}'
field_list = field (',' field)* ','?
field = ident ':' type

enum_decl = vis? 'enum' ident type_params? '{' variant_list? '}'
variant_list = variant (',' variant)* ','?
variant = ident | ident '(' type_list ')'

trait_decl = vis? 'trait' ident type_params? '{' trait_item* '}'
trait_item = 'fn' ident fn_sig ';'

impl_decl = 'impl' type_path type_params? '{' fn_decl* '}'

fn_decl = vis? 'fn' ident fn_sig block
fn_sig = '(' param_list? ')' '->' type
param_list = param (',' param)* ','?
param = ident ':' type

extern_decl = 'extern' 'fn' ident fn_sig extern_tail
extern_tail = ';' | '(' param_list? ','? '...' ')' '->' type ';'
```

Notes:

- `extern fn` may include varargs (`...`).
- `repr(C)` is an attribute form: `repr(C) struct ...`.
- `pub` visibility applies to top-level items only in v0.

## 5. Types

```
type = type_path type_args?
     | 'RawPointer' '<' type '>'
     | '(' type_list ')'

type_path = ident ('::' ident)*
type_args = '<' type_list '>'
type_list = type (',' type)* ','?
```

Built-in types:

```
I8 I16 I32 I64 U8 U16 U32 U64 F32 F64 Bool Unit
```

## 6. Expressions

```
expr = let_expr
     | if_expr
     | while_expr
     | match_expr
     | block
     | assign_expr

let_expr = 'let' 'mut'? ident (':' type)? '=' expr
assign_expr = logic_or ( '=' expr )?
logic_or = logic_and ( '||' logic_and )*
logic_and = equality ( '&&' equality )*
equality = comparison ( ('==' | '!=') comparison )*
comparison = additive ( ('<' | '<=' | '>' | '>=') additive )*
additive = multiplicative ( ('+' | '-') multiplicative )*
multiplicative = unary ( ('*' | '/' | '%') unary )*
unary = ('!' | '-') unary | call
call = primary ( call_suffix )*
call_suffix = '(' arg_list? ')'
arg_list = expr (',' expr)* ','?

primary = literal
        | path
        | '(' expr ')'
        | '(' expr ',' expr (',' expr)* ','? ')'
        | struct_lit

struct_lit = type_path '{' field_init_list? '}'
field_init_list = field_init (',' field_init)* ','?
field_init = ident ':' expr

literal = int | float | string | char | 'true' | 'false' | unit
```

## 7. Statements and blocks

```
block = '{' stmt* expr? '}'
stmt = expr ';' | 'return' expr? ';'
```
