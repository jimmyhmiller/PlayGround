# flame-bevy

Embed the [`flame-render`](../flame-render) flame-graph renderer inside a
Bevy 0.18 app, as either an embedded panel or full-window canvas.

## Why a separate crate?

`flame-render` is GPU-agnostic about *who* owns the wgpu device — it takes
references to a `wgpu::Device`/`Queue` and renders into a `TextureView`. So
in principle it could share Bevy's render device.

In practice, Bevy 0.18 pins `wgpu = 27` and `flame-render` pins `wgpu = 29`
(forced by `glyphon = 0.11`). The two `wgpu` Rust crates are version-
distinct types and cannot interop. This crate sidesteps the mismatch by
giving the flame graph its own private wgpu 29 stack and bridging via CPU
pixel readback. See the [workspace README](../../README.md) for the rationale
and the migration path once the versions converge.

## Quick start

```rust
use bevy::prelude::*;
use flame_bevy::{FlameGraph, FlameGraphInput, FlameGraphPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FlameGraphPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = images.add(FlameGraph::blank_image(1100, 700));
    commands.spawn(Camera2d);
    commands.spawn((
        FlameGraph::new(image.clone(), (1100, 700)),
        FlameGraphInput::default(),
        Sprite::from_image(image),
    ));
}
```

That's it. Drag-pan, scroll-wheel, hover, click, tab keys (1-5), `f` for
flame/icicle flip, `m` for merge mode, `a`/`0`/`Home`/`Esc` for fit-all,
`+`/`-` for zoom — all work the same as the standalone viewer.

## Loading a profile

Anything that produces a `flame_core::Profile` will do. The format crates
(`flame-format-chrome`, `flame-format-firefox`, etc.) are independently
selectable.

```rust
use std::sync::Arc;
use flame_bevy::flame_core::{ProfileBuilder, TraceSource};
use flame_format_chrome::ChromeSource;

fn load_chrome(mut q: Query<&mut FlameGraph>, path: &std::path::Path) {
    let bytes = std::fs::read(path).unwrap();
    let mut b = ProfileBuilder::new();
    ChromeSource.load(&bytes, &mut b).unwrap();
    let profile = Arc::new(b.finish());
    for mut flame in &mut q {
        flame.set_profile(profile.clone());
    }
}
```

## The two embed modes

### Embedded panel (recommended)

Spawn the `FlameGraph` entity, place its `image()` handle inside whatever
holds your scene: `Sprite`, `UiImage`, a 3D mesh material, anything that
accepts `Handle<Image>`.

If the panel is not at the window origin, set `FlameGraphInput::panel_origin`
to the panel's top-left position in window-pixel coordinates so cursor input
lands at the right place. The plugin handles `panel_size` automatically.

```rust
let mut input = FlameGraphInput::default();
input.panel_origin = Vec2::new(panel_x, panel_y); // top-left in window pixels
commands.spawn((FlameGraph::new(...), input, Sprite::from_image(...)));
```

See `examples/embed_panel.rs`.

### Full-window

Same setup, except the panel covers the whole window. Track window resizes
so the canvas resizes too:

```rust
fn fit_to_window(
    mut resized: MessageReader<bevy::window::WindowResized>,
    flames: Query<&FlameGraph>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(ev) = resized.read().last() else { return };
    for flame in &flames {
        let img = images.get_mut(&flame.image()).unwrap();
        img.texture_descriptor.size = Extent3d {
            width: ev.width as u32,
            height: ev.height as u32,
            depth_or_array_layers: 1,
        };
        img.data = Some(vec![0; (ev.width as u32 * ev.height as u32 * 4) as usize]);
    }
}
```

The plugin's `resize_image_to_panel` system detects the image-size change
and propagates it to the renderer. See `examples/embed_fullscreen.rs`.

## Driving the renderer manually

If you need state that the input bridge doesn't expose (sandwich view,
group key, sequence-lifeline picker, etc.) grab the renderer:

```rust
fn open_sandwich(mut q: Query<&mut FlameGraph>) {
    for mut flame in &mut q {
        let r = flame.renderer_mut();
        r.set_tab(flame_bevy::flame_render::MainTab::Sandwich);
        r.rebuild_instances();
        flame.mark_dirty(); // <- IMPORTANT: schedule a repaint
    }
}
```

Always call `mark_dirty()` after mutating via `renderer_mut()`, otherwise
the plugin won't schedule a redraw and your change won't appear on screen.

If you want to take over input entirely — for example to gate input on a
focused panel in your own UI — set `FlameGraphInput::enabled = false` and
drive everything via `renderer_mut()`.

## Performance notes

- **Idle is free.** The plugin only re-renders when `dirty` is set. Static
  profile sitting on screen costs no GPU work.
- **Readback cost.** Each redraw does a `copy_texture_to_buffer` + a
  synchronous `device.poll(Wait)` + a `Vec<u8>` copy into the `Image`. At
  1080p this is ~8 MB of bandwidth and typically <5 ms wall time on a
  modern integrated GPU. Larger canvases scale linearly.
- **Hover redraws.** Mouse motion that changes the hovered slice triggers a
  redraw. If you have a huge panel and want hover not to repaint, set
  `FlameGraphInput::enabled = false` and only call `mark_dirty()` on the
  events you care about.

## Public API

- `FlameGraphPlugin` — add to your `App`.
- `FlameGraph` (component) — owns the renderer and a `Handle<Image>`.
- `FlameGraphInput` (component) — input forwarding state per panel.
- Re-exports: `flame_core`, `flame_render`. Use these to construct profiles
  and reach the underlying renderer types (`MainTab`, `Direction`, etc.).
