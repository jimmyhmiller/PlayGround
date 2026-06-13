# funct Standard Library Reference

Every function below is available globally in any funct program (they come from
the prelude, installed at engine startup). Because of UFCS and pipes, every one
can be called three ways:

```text
len(xs)        xs.len()        xs |> len
push(xs, 1)    xs.push(1)      xs |> push(1)
```

The reference uses `name(args) -> ReturnType` signatures. Notation:

- `any` — any value.
- `Option` means a `Some(v)`/`None` variant; `Result` means `Ok(v)`/`Err(e)`.
- "**Faults**" means it aborts the run with an error (a contract violation),
  as opposed to returning an `Err`/`None`. See the
  [guide](guide.md#error-handling-result-option-and-) for fault vs. `Err`.
- Everything is immutable: functions that "add", "remove", or "set" return a
  **new** value and leave the input unchanged.

Contents:

- [Basics](#basics)
- [Conversions and parsing](#conversions-and-parsing)
- [Math](#math)
- [Strings](#strings)
- [Lists](#lists)
- [Records](#records)
- [Nested paths](#nested-paths)
- [Atoms](#atoms)
- [Higher-order list functions](#higher-order-list-functions)
- [JSON](#json)
- [Assertions](#assertions)

---

## Basics

### `str(v) -> Str`
The display form of any value. `str(42) + str(true)` → `"42true"`.

### `typeof(v) -> Str`
The type name: `"Unit"`, `"Bool"`, `"Int"`, `"Float"`, `"Str"`, `"List"`,
`"Tuple"`, `"Record"`, `"Variant(Tag)"`, `"Fn"`, `"NativeFn"`, `"Atom"`,
`"Range"`, or a registered host type's name. `typeof(1.5)` → `"Float"`.

### `print(...) -> Unit`
Print all arguments, space-separated, with no trailing newline.

### `println(...) -> Unit`
Like `print`, plus a trailing newline.

### `len(x) -> Int`
Element count of a `List` or `Tuple`, **character** count of a `Str`, or field
count of a `Record`. Faults on anything else. `len("héllo")` → `5`.

### `is_empty(x) -> Bool`
True if a `Str`, `List`, `Tuple`, or `Record` has length 0. Faults on anything
else.

---

## Conversions and parsing

### `to_int(x) -> Int`
`Int` unchanged; `Float` truncated toward zero. Faults if the float is NaN,
infinite, or out of `Int` range.

### `to_float(x) -> Float`
`Int` or `Float` to `Float`.

### `parse_int(s) -> Result<Int, Str>`
Parse a (trimmed) string as an integer. `parse_int("42")` → `Ok(42)`;
`parse_int("x")` → `Err("not an integer: x")`. Faults if the argument is not a
`Str`.

### `parse_float(s) -> Result<Float, Str>`
Parse a (trimmed) string as a float. `Ok(f)` or `Err("not a number: ...")`.

---

## Math

Trig/transcendental functions accept an `Int` or `Float` and return a `Float`.

### `sqrt(x) -> Float`
Square root. `(2.25).sqrt()` → `1.5`.

### `sin(x) -> Float` · `cos(x) -> Float` · `tan(x) -> Float`
Trigonometric functions (radians).

### `atan2(y, x) -> Float`
The angle (radians) of the vector `(x, y)`.

### `exp(x) -> Float` · `ln(x) -> Float` · `log10(x) -> Float`
`e^x`, natural log, base-10 log.

### `floor(x) -> Float` · `ceil(x) -> Float` · `round(x) -> Float`
Round down / up / to nearest. These return a `Float`; use `to_int` to get an
`Int`.

### `abs(x) -> Int | Float`
Absolute value, preserving the numeric type. Faults on integer overflow
(`abs` of the smallest `Int`) or a non-number.

### `min(a, b) -> Int | Float` · `max(a, b) -> Int | Float`
The smaller / larger of two numbers, returning the original value (so the
`Int`/`Float` type is preserved). Mixed `Int`/`Float` are compared numerically.
Faults on non-numbers.

### `clamp(x, lo, hi) -> Int | Float`
`min(max(x, lo), hi)`.

---

## Strings

### `contains(s, needle) -> Bool`
True if `Str` `s` contains substring `needle`. (Also works on lists — see
[Lists](#lists).)

### `starts_with(s, prefix) -> Bool` · `ends_with(s, suffix) -> Bool`
Prefix / suffix test.

### `index_of(s, needle) -> Option<Int>`
`Some(char_index)` of the first occurrence, else `None`. The index counts
**characters**, not bytes. Never returns `-1`. (Also works on lists.)

### `to_lower(s) -> Str` · `to_upper(s) -> Str`
Case conversion.

### `trim(s) -> Str`
Strip leading and trailing whitespace.

### `replace(s, from, to) -> Str`
Replace **all** occurrences of `from` with `to`.

### `split(s, sep) -> List<Str>`
Split on a non-empty separator string. `split("a,b,c", ",")` →
`["a", "b", "c"]`. Faults if `sep` is empty (use `chars` for characters).

### `chars(s) -> List<Str>`
The characters of `s`, each as a one-character string.

### `join(list, sep) -> Str`
Join a list of strings with a separator. Faults if any element is not a `Str`
(map with `str` first: `map(xs, str) |> join(",")`).

### `slice(s, start, count) -> Str`
A substring of up to `count` characters starting at character index `start`
(both non-negative `Int`s). Past the end just yields fewer characters. (Also
works on lists.)

---

## Lists

Lists are immutable; "mutating" functions return a new list.

### `push(list, v) -> List`
Append `v`. `push([1], 2)` → `[1, 2]`.

### `pop(list) -> List`
All but the last element. Faults if the list is empty.

### `first(list) -> Option` · `last(list) -> Option`
`Some(elem)` or `None` if empty.

### `rest(list) -> List`
All but the first element (empty list if already empty).

### `insert_at(list, i, v) -> List`
Insert `v` at index `i` (must be in `0..=len`). Faults if out of bounds.

### `remove_at(list, i) -> List`
Remove the element at index `i` (must be in `0..len`). Faults if out of bounds.

### `slice(list, start, count) -> List`
Up to `count` elements starting at index `start`. Past the end yields fewer.

### `contains(list, x) -> Bool`
True if some element equals `x` (structural equality).

### `index_of(list, x) -> Option<Int>`
`Some(i)` of the first element equal to `x`, else `None`.

### `sort(list) -> List`
Sorted ascending. Elements must be **all numbers or all strings**; faults on a
mix or on unorderable values.

### `sort_by(list, key_fn) -> List`
Sort by a computed key. `key_fn(elem)` must return all-numbers or all-strings.
`sort_by(people, p => p.age)`.

> For `map`, `filter`, `fold`, `sum`, `reverse`, `to_list`, see
> [Higher-order list functions](#higher-order-list-functions).

---

## Records

Records are immutable, string-keyed maps. Keys come back in sorted order.

### `has(record, key) -> Bool`
True if `record` has the string `key`.

### `get(coll, key) -> Option`
Look up a value: a `Str` key in a `Record`, or an `Int` index in a `List`/
`Tuple`. Returns `Some(v)` or `None` (a negative or out-of-range index is
`None`).

### `assoc(coll, key, v) -> coll`
Return a copy with `key` set to `v`: a `Str` key in a `Record` (adds or
replaces), or an `Int` index in a `List` (must be in bounds, else faults).

### `dissoc(record, key) -> Record`
Return a copy without `key`.

### `merge(a, b) -> Record`
Merge two records; keys in `b` win.

### `keys(record) -> List<Str>`
The keys, sorted.

### `values(record) -> List`
The values, ordered by sorted key.

### `entries(record) -> List<(Str, any)>`
`(key, value)` tuples, ordered by sorted key.

---

## Nested paths

A path is a `List` of keys: `Str`s for record fields, `Int`s for list/tuple
indices. e.g. `["ui", "panels", 0, "title"]`.

### `get_in(coll, path) -> Option`
Navigate the path and return `Some(v)`, or `None` if anything along the way is
missing or the wrong shape. Never faults on a missing path.

### `assoc_in(coll, path, v) -> coll`
Return a copy with the value at `path` set to `v`. Missing intermediate
**record** keys are created as empty records (Clojure-style); anything else
missing or mistyped faults loudly. The path must be non-empty.

### `update(coll, key, f) -> coll`
Apply `f` to the existing value at `key` and store the result. Faults if there
is no value at `key` (use `assoc` to add one). `update(scores, "ada", n => n + 1)`.

### `update_in(coll, path, f) -> coll`
Like `update`, but for a nested `path`. Faults if the path has no value (use
`assoc_in`). The path must be non-empty.

---

## Atoms

An atom is the only mutable cell in the language. See the
[guide](guide.md#atoms).

### `atom(v) -> Atom`
Create an atom holding `v`.

### `deref(a) -> any`
Read the current value. Also spelled `@a` and `a.value`.

### `swap!(a, f) -> any`
Set the atom to `f(old_value)` and return the new value. Fires watchers.
`swap!(counter, n => n + 1)`.

### `reset!(a, v) -> any`
Set the atom directly to `v` and return it. Fires watchers.

### `swap_in!(a, path, f) -> any`
Update the value at a nested `path` inside the atom by applying `f`, returning
the whole new value. Faults if the path has no value (use `reset_in!` to
create it). Fires watchers. `swap_in!(state, ["ui", "clicks"], n => n + 1)`.

### `reset_in!(a, path, v) -> any`
Set the value at a nested `path` inside the atom to `v` (creating intermediate
records like `assoc_in`), returning the whole new value. Fires watchers.

### `watch(a, key, f) -> Atom`
Register watcher `f` under a string `key`; after every successful mutation it
is called with `(old, new)`. Registering the same `key` again replaces the
previous watcher. `f` may itself mutate atoms. Returns the atom.

### `unwatch(a, key) -> Atom`
Remove the watcher registered under `key`. Returns the atom.

---

## Higher-order list functions

These are written in funct itself (in the prelude) and work on any iterable that
`for` accepts — lists, ranges, and strings — returning a list unless noted.

### `map(xs, f) -> List`
Apply `f` to each element. `map([1, 2, 3], x => x * x)` → `[1, 4, 9]`.

### `filter(xs, pred) -> List`
Keep elements where `pred(x)` is `true`. `filter([1, 2, 3, 4], x => x % 2 == 0)`
→ `[2, 4]`.

### `fold(xs, init, f) -> any`
Left fold: start from `init` and combine with `f(acc, x)` for each element.
`fold([1, 2, 3], 10, (a, b) => a + b)` → `16`.

### `sum(xs) -> any`
Sum of the elements (starting from `0`). `sum([1, 2, 3])` → `6`.

### `reverse(xs) -> List`
The elements in reverse order.

### `to_list(xs) -> List`
Materialize an iterable into a list — most often a range:
`to_list(1..4)` → `[1, 2, 3]`.

### `unwrap_or(x, default) -> any`
Collapse an `Option` or `Result` to a plain value: `Some(v)` and `Ok(v)` yield
`v`; `None` and `Err(_)` yield `default`.
`unwrap_or(parse_int(s), 0)`.

---

## JSON

### `json_parse(s) -> Result<any, Str>`
Parse a JSON string. On success returns `Ok(value)` using this mapping:

| JSON | funct |
|---|---|
| `null` | `Unit` |
| `true`/`false` | `Bool` |
| integer | `Int` |
| fractional / huge number | `Float` |
| string | `Str` |
| array | `List` |
| object | `Record` |

On invalid JSON returns `Err("invalid JSON: ...")`.

> A JSON object written as a string *literal* needs no escaping — interpolation
> is `${...}`, so a bare `{` is just a brace:
> `json_parse("{ \"a\": 1 }")` works directly.

### `json_stringify(v) -> Result<Str, Str>`
Serialize a value to a JSON string. Returns `Ok(json)`, or `Err(msg)` if the
value contains something with no JSON form (an `Atom`, `Fn`, or host `Native`) —
these fail loudly rather than being silently dropped.

---

## Assertions

For use in `#[test]` functions (see the [guide](guide.md#testing)). Each faults
on failure with a descriptive message; otherwise returns `Unit`.

### `assert(cond)` · `assert(cond, msg)`
Fault unless `cond` is `true`. `cond` must be a `Bool`.

### `assert_eq(left, right)` · `assert_eq(left, right, msg)`
Fault unless `left == right`. The failure message prints both values.

### `assert_ne(left, right)` · `assert_ne(left, right, msg)`
Fault unless `left != right`.

### `fail()` · `fail(msg)`
Unconditionally fault — for unreachable branches in a test.
