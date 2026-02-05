# Layout and ABI v0 (Draft)

This specifies the memory layout for language types and calling conventions for codegen.

## 1. Calling convention

- Language functions use the language ABI.
- v0 uses C calling convention for all generated functions to simplify AOT linkage.
- Every function takes an implicit `Thread*` as arg0 (runtime thread context).
- Extern functions do not take the implicit `Thread*`.

## 2. Primitive representations

- `I8/I16/I32/I64` -> signed integers of that width.
- `U8/U16/U32/U64` -> unsigned integers of that width.
- `F32/F64` -> IEEE-754.
- `Bool` -> `i1` in IR, ABI-lowered to `i8` when crossing C ABI boundaries.
- `Unit` -> no value / `void`.

## 3. Struct layout

- Default struct layout is compiler-defined and may change; only `repr(C)` is stable.
- `repr(C)` layout follows target ABI layout rules for C structs.

## 4. Enum layout (v0)

- Enums are represented as tagged unions.
- Tag is `I32`.
- Payload is a union of variant payloads.
- Layout is `{ tag: I32, payload: <max variant size, aligned> }`.

## 5. Pointers

- `RawPointer<T>` lowered to opaque pointer type (`ptr`).
- GC-managed references are also represented as `ptr`, but are tracked by the GC root map.

## 6. Aggregates across ABI boundaries

- `repr(C)` structs can be passed by value or by pointer depending on ABI.
- v0 will lower by-value `repr(C)` structs to the target ABI rules (LLVM handles this).

