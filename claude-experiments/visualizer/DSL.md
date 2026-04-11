# Visualizer DSL

A small S-expression language for describing animated 2D scenes that explain
code. Edit the code in the left pane; the canvas on the right re-renders on
every keystroke.

## Quick example

```lisp
(scene hello
  (defc radius 30 :min 10 :max 80 :step 1)
  (def cx 400)

  (circle :x (spring cx :stiffness 200 :damping 15)
          :y 300
          :r radius
          :fill "#5eace0")

  (on :click
    (set! cx (+ cx 50))))
```

Click the canvas (or press Space) to move the circle. Drag the `radius`
slider in the right-hand panel to resize it live.

## Top-level forms

### `(scene name body...)`

The root of a DSL file. The name is optional; body items are nodes,
`def`/`defc` declarations, or `on` handlers.

```lisp
(scene my-viz
  (def counter 0)
  (defc size 50)
  (rect :w size :h size :fill "#fff")
  (on :click (set! counter (+ counter 1))))
```

### `(def name value)`

Declares **mutable state**. State values are read reactively: springs bound
to state retarget automatically when the value changes via `set!`.

```lisp
(def op-index 0)
(def arrow-x 400)
(def entries (list))              ; empty list
(def ops (list "name" "age" "x")) ; list of strings
```

### `(defc name default :min M :max X :step S :category C)`

Declares a **tweakable constant** that appears as a slider in the control
panel on the right. The value can be adjusted live at runtime. Expressions
using a `defc` value are re-evaluated every frame.

```lisp
(defc bucket-width 70 :min 30 :max 120 :step 1 :category layout)
(defc stiffness 300 :min 50 :max 800 :step 10 :category springs)
```

`:min`, `:max`, `:step` default to sensible values based on the default.
`:category` groups sliders in the panel (default `"scene"`).

### `(on :click body...)`

Registers a click handler. The body runs when the user clicks the canvas,
presses Space / Enter, or clicks the **step** button.

```lisp
(on :click
  (when (< op-index 8)
    (set! op-index (+ op-index 1))))
```

## Nodes

Every node supports `:id`, `:opacity`, `:scale`. All positions are
**center-based** — `:x 100 :y 100` places the center of the shape at (100, 100).

### `(rect :x :y :w :h :radius :fill :opacity :scale :id)`

A rounded rectangle.

```lisp
(rect :x 400 :y 300
      :w 200 :h 100
      :radius 8
      :fill "#5eace0")
```

### `(circle :x :y :r :fill :opacity :scale :id)`

A circle.

```lisp
(circle :x 400 :y 300 :r 30 :fill (rgba 0.4 0.8 0.5 1))
```

### `(triangle :x :y :size :fill :opacity :id)`

An upward-pointing triangle.

```lisp
(triangle :x 400 :y 100 :size 10 :fill "#ffcc33")
```

### `(group body...)`

Groups child nodes. Opacity and scale cascade through groups.

```lisp
(group :id "buckets"
  (rect :x 100 :y 300 :w 70 :h 200 :fill "#111")
  (rect :x 200 :y 300 :w 70 :h 200 :fill "#111"))
```

## Control flow (compile-time)

### `(let name expr)`

Binds a local name in the current scope. If `expr` references a tweakable,
the binding is **deferred** — it re-evaluates whenever the tweakable changes.

```lisp
(each i (range 8)
  (let cx (+ 90 (* i (+ bucket-width bucket-gap))))
  (rect :x cx :y 300 :w bucket-width :h 200))
```

### `(each var (range N) body...)`

Loops `var` from `0` to `N-1`, expanding the body for each iteration. The
loop variable is available to all subsequent siblings in the body.

```lisp
(each i (range 8)
  (rect :x (* i 50) :y 100 :w 40 :h 40))
```

## Animations

Animations are **property wrappers** — you use them inside a shape's
property values to make that property animate.

### `(spring target :stiffness S :damping D :initial I)`

A spring-based value that tracks a target. If `target` references
state or tweakables, the spring retargets live.

```lisp
;; spring toward a state value
(rect :x (spring arrow-x :stiffness 300 :damping 18))

;; spring toward a derived expression — updates when tweakables change
(rect :x (spring (+ 100 (* i bucket-width))
                 :initial 400
                 :stiffness 300
                 :damping 18))
```

`:initial` sets the starting position (if different from the target).

### `(tween from to :duration D :easing E)`

A time-based tween that plays once when the node is created.

```lisp
(rect :scale (tween 0 1 :duration 0.4 :easing back-out))
```

**Easings**: `linear`, `quad-in`, `quad-out`, `quad-in-out`, `cubic-in`,
`cubic-out`, `cubic-in-out`, `elastic-in`, `elastic-out`, `elastic-in-out`,
`bounce-in`, `bounce-out`, `bounce-in-out`, `back-in`, `back-out`,
`back-in-out`.

## Colors

| Form                     | Example                         |
|--------------------------|---------------------------------|
| Hex string               | `"#5eace0"` or `"#5eace0ff"`    |
| `(rgb r g b)`            | `(rgb 0.4 0.8 0.5)`             |
| `(rgba r g b a)`         | `(rgba 1 0.8 0.2 0.9)`          |
| `(color-for idx)`        | `(color-for 3)` — preset palette |

## Expressions

**Arithmetic**: `+`, `-`, `*`, `/`, `%`
**Comparison**: `=`, `<`, `>`
**Conditional**: `(if cond then else)`
**Data**: `(hash key n)`, `(len list)`, `(nth list idx)`, `(count-where list field value)`

```lisp
(+ 90 (* i 50))
(if (< op-index 8) 1 0)
(hash key 8)
(count-where entries "bucket" 3)
```

## Handlers (runtime)

These forms run inside `on` handlers and sequences.

### `(set! name value)`

Mutates a `def` state value. Recorded in the undo journal.

```lisp
(set! op-index (+ op-index 1))
```

### `(when cond body...)`

Runs body forms only if `cond` is truthy.

```lisp
(when (< op-index (len ops))
  (sequence ...))
```

### `(sequence step...)`

Runs steps sequentially with optional waits. The canvas stays interactive
while the sequence runs, but only one sequence can be active at a time.

```lisp
(sequence
  (set! arrow-alpha 1)
  (wait 0.5)
  (set! arrow-alpha 0))
```

### `(wait seconds)`

Only valid inside a `sequence`. Pauses execution for the given duration.

```lisp
(wait 0.3)
```

### `(let name expr)` (inside a sequence)

Binds a local name in the sequence scope. Unlike compile-time `let`,
runtime `let` evaluates eagerly.

```lisp
(sequence
  (let key (nth ops op-index))
  (let bucket (hash key 8))
  (set! arrow-x (+ 90 (* bucket 86))))
```

### `(push! list-name (:field val ...))`

Appends an item to a state list. The item is a map built from keyword
arguments.

```lisp
(push! entries (:bucket 3 :slot 1))
```

### `(spawn! group-id (node-expr))`

Dynamically creates a node inside a named group. The node template can
reference sequence locals (like `slot`, `bucket`) and tweakables — they
become frozen / live as appropriate. Spawned nodes are undone when the
step is rewound.

```lisp
(spawn! "_spawned"
  (rect :x (spring (+ 90 (* bucket 86)) :initial 400)
        :y (spring cy :initial 80)
        :w 54 :h 36
        :fill (color-for op-index)))
```

The special group id `"_spawned"` is always available.

## Reactivity model

Every expression you write is categorized into one of three kinds:

1. **Constant** — pure arithmetic, literals, lexical-only. Evaluated once.
   ```lisp
   :x (+ 100 (* i 50))    ; i is a loop var, no tweakables
   ```

2. **Live** — expression contains one or more tweakables (directly or via
   a deferred `let`). Re-evaluated every frame.
   ```lisp
   :x (+ 100 (* i bucket-width))   ; bucket-width is a defc
   ```

3. **Animated** — wrapped in `(spring ...)` or `(tween ...)`. The wrapper
   produces a value that animates toward (or plays) the underlying target.
   Spring targets can themselves be live or state-bound.
   ```lisp
   :x (spring (+ 100 (* i bucket-width)) :stiffness 300 :damping 18)
   ```

State values (`def`) are read reactively when used as the **target of a
spring**. Bare references to state create a spring binding automatically.

## Undo

The runtime records every `set!`, `push!`, and `spawn!` inside a click
handler as one "step" in a journal. Press **left arrow** (or click the
**← back** button) to undo the most recent step.

Undo is general — it works for any DSL program. You don't need to write
per-visualization undo logic.

## Keyboard shortcuts

| Key                          | Action       |
|------------------------------|--------------|
| **→** / **Space** / **Enter** | Step forward (run click handler) |
| **←**                        | Step back (undo) |

Keys are ignored while you're typing in the code editor.
