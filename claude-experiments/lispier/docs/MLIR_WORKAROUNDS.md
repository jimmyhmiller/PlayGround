# MLIR Attribute Handling in Lispier

## Explicit MLIR Attribute Literals

Lispier supports explicit MLIR attribute syntax directly (without quotes). Spaces are automatically converted to commas for MLIR compatibility:

```lisp
;; Dense arrays (for operandSegmentSizes, etc.)
{:operandSegmentSizes array<i32: 0 1 1 1 1 1 1 0 0 0 0>}

;; Dense elements (for tensor constants)
{:value dense<1 2 3 4> : tensor<4xi32>}

;; Affine maps
{:map affine_map<(d0) -> (d0 + 1)>}
```

This syntax mirrors MLIR's native attribute syntax but uses spaces instead of commas (lisp style).

## How It Works

Any symbol containing `<` and ending with `>` in attribute position is treated as an MLIR literal:

1. Spaces are converted to commas: `array<i32: 0 1 1>` → `array<i32: 0, 1, 1>`
2. MLIR's parser attempts to parse it as an attribute
3. If parsing succeeds, the appropriate attribute type is created (DenseI32ArrayAttr, etc.)
4. If parsing fails (e.g., `memref<128xf32>`), it falls back to TypeAttribute

## Operations with Variadic Operands

Some MLIR operations have variadic or optional operands and need an `operandSegmentSizes` attribute:

```lisp
;; gpu.launch needs operandSegmentSizes
(gpu.launch {:operandSegmentSizes array<i32: 0 1 1 1 1 1 1 0 0 0 0>}
  grid_x grid_y grid_z block_x block_y block_z
  (region ...))

;; cf.cond_br needs operandSegmentSizes
(cf.cond_br {:operandSegmentSizes array<i32: 1 0 0>} condition ^true_block ^false_block)
```

## No Magic

Lispier doesn't try to automatically compute or infer any attributes. If an operation needs a specific attribute, you specify it explicitly. This keeps the behavior predictable and matches what you'd write in raw MLIR.
