# Project prism ("cube") — tilt / glass / reflection post-mortem

Status: **tilt + glass reverted.** Known-good baseline restored (see below).
This documents what was attempted, why each approach failed, and the
fundamental tensions, so the next attempt doesn't re-walk the same wall.

The "cube" is `crates/terminal-bevy/src/cube.rs` — the Compiz-style 3D
project overview. It is an **N-sided ring** (one vertical face per
switchable project), NOT a literal cube. That distinction is the root of
most of the trouble below.

## What we were trying to add

1. **Compiz-style tilt**: drag vertically to tilt the *shape itself* (so
   the reflection and geometry change with it), in addition to the
   existing horizontal-drag yaw spin.
2. **Glass faces**: give the face backboards an opacity so the prism reads
   as a translucent solid you can see into.
3. Keep the (already working) **per-project theming** and the **floor
   reflection**.

## Known-good baseline (the revert target)

This state was verified working and is what we reverted to:

- Prism rotates on **yaw only** (horizontal drag). No tilt/pitch.
- Faces are **opaque**, themed per project: each face backboard's
  `base_color` = that project's theme `bg` (`project_theme_bg`).
- **Static floor** at `y = -FACE_HEIGHT/2` with a static mirror reflection
  (`reflect_root`: translation `(0, 2*floor_y, 0)`, rotation = yaw, scale
  `(1,-1,1)`). The mirror is correct because a Y-rotation (yaw) commutes
  with the `scale.y = -1` flip.
- **Static camera** `near = 0.5`, `far = overview_dist + 340`.

No flicker, no lag, correct reflection. Everything below is what broke it.

## Tilt approaches tried (all failed)

1. **Camera elevation orbit** — pitch the *camera* on a sphere, keep the
   prism upright. Rejected: the user wants the *shape* to tilt, not the
   viewpoint.

2. **Shape tilt about its center** — `root.rotation = rot_x(pitch) *
   rot_y(yaw)`. The reflection mirror is `rot_x(-pitch) * rot_y(yaw)` with
   `scale.y=-1` (the y-flip conjugates the rotation: yaw unaffected, pitch
   negated). Geometrically correct, BUT: the ring's bottom edge sits *on*
   the floor when level, so any tilt dunks the front-bottom **through** the
   floor, and the (correct) opposite-tilting reflection of a wide ring
   crossing the floor plane reads as a broken double image. User: "flips on
   the reflection."

3. **Shape tilt about a floor pivot** — rotate about a point on the floor
   so the prism "stands up" like a cube on a surface
   (`root.translation = pivot - R*pivot`, `pivot = (0, floor_y, 0)`).
   For a **wide ring** (apothem is large for ~14 projects) the far faces
   still swing far below the pivot. User: "tilting into the ground."

4. **Dynamic floor tracking the lowest point** — each frame, compute the
   prism's lowest extent under the current tilt
   (`-(half_h*cos(pitch) + reach*sin(|pitch|))`, `reach` = vertex radius +
   float margin) and move the floor (and its mirror) to sit just under it,
   so the shape floats above and only ever *touches* the floor. Verified to
   keep the shape from penetrating in a forced-tilt screenshot, but in
   motion the user reported "reflection is all weird" and "weird lagging."

## Glass / opacity attempts (all unsatisfying)

- **Translucent backboards** (`AlphaMode::Blend`, alpha ~0.82) so you can
  see through the ring. **Order-Independent-Transparency problem**: the GPU
  sorts overlapping transparent faces by camera distance with an unstable
  tie-break. On a *symmetric ring* many faces are equidistant, so the sort
  order flips as you rotate/tilt and the composited brightness pulses
  ("flickering like lighting"), worst when looking *through* faces from the
  backside. Bevy has no OIT by default — this can't be cleanly fixed while
  keeping full see-through glass.

- **Back-face culling (front-only)** to halve the overlapping transparent
  layers. Dead end: the pane content floats on the **outside** of the ring
  and every face front points outward, so tilting to look *into* the ring
  shows only culled interiors → a **dark void**. (Screenshot confirmed.)

- **Opaque faces** — kills the flicker (opaque writes depth and just
  occludes) but loses the glass look the user explicitly wanted. This was
  shipped briefly as a "fix" and correctly rejected: it papers over the
  symptom instead of addressing it.

## Z-fighting (a *separate* issue from OIT)

Panes flashed/disappeared as the view moved. Two real causes:

1. **Near-plane-bound depth precision.** Bevy uses reverse-Z, where depth
   precision is dominated by the **near plane**. `near = 0.5` with the
   prism sitting ~20–40 units away leaves very few depth bits at that
   distance. Pushing the near plane out to hug the geometry
   (`near = dist_to_camera - prism_radius`, ~3–4.6 at overview) is ~9× the
   precision and is a genuinely good idea — but doing it by mutating
   `Projection` every frame is suspected in the "lagging" report.

2. **Pinned panes floated only `DEPTH_EPS` (0.004) apart** in the overview
   pose (the painter-order epsilon, tuned for the *dive endpoint* where the
   camera is close). At overview distance 0.004 can't resolve, so stacked
   pinned/background panes fought constantly. Bumping their overview
   separation to ~0.12 (while keeping the flat/dive pose tight) is the
   right move and should be kept *if* tilt is retried.

Even with both fixes the user reported continued flicker — which points
either at the unpinned 0.38 stacks fighting (depth-format dependent) or at
the OIT transparency flicker being conflated with z-fighting. Not resolved.

## Fundamental tensions (the actual lessons)

- **A wide RING + a floor reflection + tilt do not coexist cleanly.**
  Compiz works because it tilts a *compact cube*, not a wide ring. Any tilt
  sends the far faces of a wide ring far below the floor. If tilt is truly
  wanted, reconsider the overview *layout* (compact box / fewer faces /
  carousel) before reconsidering the math.
- **Translucent overlapping faces flicker without OIT.** Pick one: opaque,
  real OIT (weighted-blended), or single-sided faces with content on the
  interior.
- **Pane content floats OUTSIDE the ring**, so you can't "see into" the
  ring to view content — culling/looking-in gives a void.
- **Reverse-Z precision is near-plane-bound**; hug the geometry, but set it
  **once per overview open**, not every frame, to avoid churn/lag.
- **Temporal artifacts (flicker, lag) are invisible in screenshots.** They
  must be verified interactively with the user. Several "fixes" here were
  shipped on reasoning alone and were wrong.

## If tilt is retried

- Decide the layout question first: is the overview a wide ring or a
  compact shape? Tilt only makes sense for something compact.
- For the reflection: simplest is to **drop / fade the floor reflection
  while tilted** rather than try to mirror a tilted wide ring.
- Keep: near-plane hugging (set once on open), pinned-pane overview
  separation (~0.12), the board-recede intersection fix (recede the n-gon
  rigidly, never per-face-normal — that one was correct and is retained).
- Verify every step live with the user; do not trust screenshots for
  flicker/lag.
