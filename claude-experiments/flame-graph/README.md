# flame-graph

A GPU-accelerated flame graph viewer, split into composable library crates.

## Workspace layout

| Crate | Depends on | What it gives you |
| ----- | ---------- | ----------------- |
| `flame-core` | nothing graphical | Canonical in-memory trace model: `Profile`, `SliceTable`, `StringInterner`, `StackTable`, `ProfileBuilder`. Pure data. |
| `flame-format-folded` | `flame-core` | Linux `perf script`-style folded stack reader. |
| `flame-format-chrome` | `flame-core` | Chrome/Perfetto trace event JSON reader. |
| `flame-format-speedscope` | `flame-core` | speedscope.app trace reader. |
| `flame-format-firefox` | `flame-core` | Firefox profiler (`profiler.firefox.com`) reader. |
| `flame-format-otel` | `flame-core` | OpenTelemetry span JSON-lines reader. |
| `flame-live` | `flame-core` | Live-streaming protocol (postcard / NDJSON) for `samply`-style sources. |
| `flame-render` | `flame-core` + `wgpu` + `glyphon` | The actual GPU renderer. **Windowing-agnostic** — takes an external `wgpu::Device`, `wgpu::Queue`, `TextureFormat`, and renders into any `wgpu::TextureView` you hand it. |
| `flame-bevy` | `flame-render` + `bevy` | Bevy plugin that embeds the renderer as either a sprite/UI image or a full-window panel. See below. |
| `flame-viewer` | everything | The standalone winit binary. Reference integration for `flame-render`. |

## Pick-and-mix recipes

### "I just want to parse a trace"

```toml
[dependencies]
flame-core = { path = "crates/flame-core" }
flame-format-chrome = { path = "crates/flame-format-chrome" }
```

```rust
use flame_core::{ProfileBuilder, TraceSource};
use flame_format_chrome::ChromeSource;

let bytes = std::fs::read("trace.json")?;
let mut b = ProfileBuilder::new();
ChromeSource.load(&bytes, &mut b)?;
let profile = b.finish();
```

No GPU, no Bevy, no winit, no serde-json transitively unless your chosen
format crate pulls it in. Each format crate is independently selectable.

### "I want to embed the flame graph in my Bevy 0.18 app"

See `crates/flame-bevy/README.md` for the full guide. Short version:

```toml
[dependencies]
bevy = "0.18"
flame-bevy = { path = "crates/flame-bevy" }
```

```rust
use bevy::prelude::*;
use flame_bevy::{FlameGraph, FlameGraphInput, FlameGraphPlugin};

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = images.add(FlameGraph::blank_image(1100, 700));
    commands.spawn(Camera2d);
    commands.spawn((
        FlameGraph::new(image.clone(), (1100, 700)),
        FlameGraphInput::default(),
        Sprite::from_image(image),
    ));
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FlameGraphPlugin)
        .add_systems(Startup, setup)
        .run();
}
```

The `Handle<Image>` you create is what the flame graph paints into. Place it
in a `Sprite`, `UiImage`, or material — anywhere that accepts `Handle<Image>`.

### "I want to embed the flame graph in my own wgpu / winit app (not Bevy)"

Use `flame-render` directly. The renderer doesn't own a window or a
swapchain; you do.

```toml
[dependencies]
flame-render = { path = "crates/flame-render" }
wgpu = "29"           # required: must match flame-render's pin
glyphon = "0.11"      # only if you're combining text
```

```rust
// Once at startup, after you've built your wgpu Device/Queue/Surface:
let renderer = flame_render::Renderer::new(
    &device,
    &queue,
    surface_format,
    (window_width, window_height),
);
renderer.set_profile(Arc::new(profile));
renderer.rebuild_instances();

// On resize:
renderer.resize(new_w, new_h);
renderer.rebuild_instances();

// Each frame:
let frame = surface.get_current_texture()?;
let view = frame.texture.create_view(&Default::default());
renderer.render(&view);
frame.present();
```

For input you call `renderer.hit_test(x, y)`, `renderer.set_hover(...)`,
`renderer.viewport.pan_x_px(dx)`, `renderer.zoom_at(x, factor)`, etc. The
standalone viewer (`crates/flame-viewer/src/main.rs`) is the reference
implementation for input routing.

## Building & running

```sh
# Standalone viewer
cargo run --release -p flame-viewer -- path/to/trace.json

# Bevy panel example
cargo run --release -p flame-bevy --example embed_panel

# Bevy fullscreen example
cargo run --release -p flame-bevy --example embed_fullscreen
```

## Why is `flame-render` decoupled from windowing?

So it can be embedded. `flame-viewer` owns winit, the wgpu instance,
the surface, and the swapchain. `flame-bevy` owns its own wgpu instance and
renders to a CPU pixel buffer. Both go through the same `Renderer::new` and
`Renderer::render(&TextureView)` API. If you want to put the flame graph
inside an egui panel, an Iced container, or a Slint window, you write the
fourth integration in ~200 lines without changing `flame-render` at all.

## Why is `flame-bevy` so awkwardly decoupled (CPU pixel readback)?

`flame-render` pins `glyphon = "0.11"`, which requires `wgpu = "29"`. Bevy
0.18 ships with `wgpu = "27"`. There is no glyphon release that targets
wgpu 27. Until those align, `flame-bevy` keeps its own `wgpu` stack, renders
into an offscreen texture, reads the pixels back to CPU, and copies them
into a `bevy::Image`. Cost is one tightly-packed RGBA8 copy per redraw
(~8 MB at 1080p) plus a `device.poll` blocking wait — typically a few
milliseconds. The plugin only redraws on state changes, so a static profile
sitting on screen is free.

When Bevy and glyphon converge on the same wgpu version, the right move is
to delete the readback path in `flame-bevy/src/gpu.rs`, change the plugin to
hand Bevy's `RenderDevice`/`RenderQueue` straight into `Renderer::new`, and
render directly into the `bevy::Image`'s texture view. The public API of
`flame-bevy` (`FlameGraph`, `FlameGraphInput`, `FlameGraphPlugin`) stays the
same.
