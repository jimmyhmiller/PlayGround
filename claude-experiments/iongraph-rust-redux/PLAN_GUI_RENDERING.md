# GUI Rendering Implementation Plan

## Overview
Add interactive GUI rendering to IonGraph using Skia and winit, while maintaining existing SVG export functionality. The GUI will support pan/zoom, click interactions, and hot reload.

## Architecture Decision: LayoutProvider Approach

### Current Architecture
The codebase uses a trait-based `LayoutProvider` abstraction:
- `LayoutProvider` trait defines platform-agnostic rendering interface
- `PureSVGTextLayoutProvider` builds an in-memory SVG DOM tree
- `Graph` uses the layout provider to create visual elements
- SVG is exported by traversing the DOM tree

### Chosen Approach: Dual Implementation Strategy

**Keep both SVG and GUI rendering** via separate LayoutProvider implementations:

```
┌─────────────────────────────────────────────┐
│           Graph (layout logic)              │
│  - Block positioning                        │
│  - Arrow routing                            │
│  - Edge straightening                       │
└───────────────┬─────────────────────────────┘
                │ uses
                ▼
        LayoutProvider trait
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
 PureSVGText...    SkiaLayout...
 (existing)        (new)
        │                │
        │                └──► Immediate mode rendering
        └──► SVG export       to Skia canvas
```

**Why this approach:**
1. ✅ Preserves existing SVG export for batch processing
2. ✅ Clean separation of concerns
3. ✅ No changes to core Graph layout algorithms
4. ✅ Easy to switch between renderers
5. ✅ LayoutProvider trait already designed for this pattern

## Dependencies

Add to `Cargo.toml`:
```toml
[dependencies]
# Existing dependencies...
winit = "0.30"           # Window creation and event handling
skia-safe = "0.77"       # Skia 2D graphics
gl = "0.14"              # OpenGL bindings
glutin = "0.32"          # OpenGL context creation for winit
glutin-winit = "0.5"     # winit integration for glutin
raw-window-handle = "0.6" # Required for glutin-winit
notify = "6.0"           # File watching for hot reload

[features]
default = ["svg"]
svg = []                 # SVG rendering (existing)
gui = ["winit", "skia-safe", "gl", "glutin", "glutin-winit", "notify"]
```

## Implementation Steps

### Phase 1: Core GUI Infrastructure (Foundation)

#### 1.1 Create SkiaLayoutProvider
**File:** `src/skia_layout_provider.rs`

```rust
pub struct SkiaLayoutProvider {
    // Skia canvas for immediate-mode rendering
    surface: Option<Surface>,
    canvas: Canvas,

    // Layout state (for measurements)
    font: Font,
    paint: Paint,

    // Render commands (deferred execution)
    commands: Vec<RenderCommand>,
}

enum RenderCommand {
    DrawRect { x, y, width, height, color, stroke },
    DrawText { x, y, text, color, size },
    DrawPath { path, color, stroke_width },
    // etc.
}

impl LayoutProvider for SkiaLayoutProvider {
    type Element = SkiaElement;

    fn create_element(&mut self, tag: &str) -> Box<Self::Element> {
        // Return lightweight SkiaElement that records what to draw
    }

    fn calculate_block_size(...) -> Vec2 {
        // Use Skia's font metrics for accurate sizing
    }

    // Other trait methods...
}
```

**Key insight:** Unlike SVG which builds a DOM tree, Skia rendering is **immediate mode**. The LayoutProvider builds a list of render commands during the Graph::render() phase, then executes them when painting.

#### 1.2 Create SkiaElement
**File:** `src/skia_layout_provider.rs`

```rust
pub struct SkiaElement {
    node_type: String,
    bounds: Rect,
    classes: HashSet<String>,
    style: HashMap<String, String>,
    text_content: Option<String>,
    children: Vec<Rc<RefCell<SkiaElement>>>,

    // Skia-specific rendering data
    render_cmd: Option<RenderCommand>,
}

impl Element for SkiaElement {
    // Implement trait requirements
}
```

### Phase 2: Window and Rendering Loop

#### 2.1 Create Window Manager
**File:** `src/gui/window.rs`

```rust
use winit::{
    event_loop::{EventLoop, ControlFlow},
    window::WindowBuilder,
};
use glutin::{
    config::ConfigTemplateBuilder,
    context::{ContextAttributesBuilder, NotCurrentContext},
    display::GetGlDisplay,
    surface::{SurfaceAttributesBuilder, WindowSurface},
};

pub struct IonGraphWindow {
    window: Window,
    gl_surface: Surface<WindowSurface>,
    gl_context: PossiblyCurrentContext,
    skia_surface: skia_safe::Surface,

    // State
    graph_data: GraphData,
    view_state: ViewState,
}

pub struct ViewState {
    // Pan/Zoom
    offset: Vec2,
    scale: f64,

    // Interaction
    selected_block: Option<String>,
    hover_block: Option<String>,

    // Mouse state
    mouse_pos: Vec2,
    is_panning: bool,
    pan_start: Vec2,
}

impl IonGraphWindow {
    pub fn new(event_loop: &EventLoop<()>, ion_data: UniversalIR) -> Self {
        // 1. Create winit window
        // 2. Create glutin GL context
        // 3. Create Skia surface from GL context
        // 4. Initialize view state
    }

    pub fn render(&mut self) {
        // 1. Clear surface
        // 2. Apply transform (scale + offset)
        // 3. Create SkiaLayoutProvider with canvas
        // 4. Create Graph with provider
        // 5. Call graph.layout() and graph.render()
        // 6. Execute render commands
        // 7. Flush to screen
    }

    pub fn handle_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::MouseWheel { delta, .. } => {
                // Zoom
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Update hover state
                // Handle panning
            }
            WindowEvent::MouseInput { button, state, .. } => {
                // Click handling
                // Start/stop panning
            }
            WindowEvent::KeyboardInput { .. } => {
                // Keyboard shortcuts
            }
            _ => {}
        }
    }
}
```

#### 2.2 Create Main Event Loop
**File:** `src/bin/iongraph_viewer.rs`

```rust
use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};

fn main() {
    // Parse args (JSON path, function index, pass index)
    let args = parse_args();

    // Load Ion data
    let ion_data = load_ion_data(&args.json_path, args.func_idx, args.pass_idx);

    // Create event loop and window
    let event_loop = EventLoop::new().unwrap();
    let mut window = IonGraphWindow::new(&event_loop, ion_data);

    // Run event loop
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Wait);

        match event {
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::CloseRequested => {
                        elwt.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        window.render();
                    }
                    _ => {
                        window.handle_event(event);
                        window.request_redraw();
                    }
                }
            }
            Event::AboutToWait => {
                // Request redraw if needed
            }
            _ => {}
        }
    }).unwrap();
}
```

### Phase 3: Interactivity

#### 3.1 Pan & Zoom
**File:** `src/gui/viewport.rs`

```rust
pub struct Viewport {
    pub offset: Vec2,
    pub scale: f64,
    bounds: Rect,
}

impl Viewport {
    pub fn screen_to_world(&self, screen_pos: Vec2) -> Vec2 {
        Vec2 {
            x: (screen_pos.x - self.offset.x) / self.scale,
            y: (screen_pos.y - self.offset.y) / self.scale,
        }
    }

    pub fn world_to_screen(&self, world_pos: Vec2) -> Vec2 {
        Vec2 {
            x: world_pos.x * self.scale + self.offset.x,
            y: world_pos.y * self.scale + self.offset.y,
        }
    }

    pub fn zoom(&mut self, delta: f64, focal_point: Vec2) {
        // Zoom while keeping focal point stable
        let world_focal = self.screen_to_world(focal_point);
        self.scale = (self.scale * delta).clamp(MIN_ZOOM, MAX_ZOOM);
        let new_screen_focal = self.world_to_screen(world_focal);
        self.offset.x += focal_point.x - new_screen_focal.x;
        self.offset.y += focal_point.y - new_screen_focal.y;
    }

    pub fn pan(&mut self, delta: Vec2) {
        self.offset.x += delta.x;
        self.offset.y += delta.y;
    }

    pub fn apply_to_canvas(&self, canvas: &mut Canvas) {
        canvas.translate((self.offset.x as f32, self.offset.y as f32));
        canvas.scale((self.scale as f32, self.scale as f32));
    }
}
```

#### 3.2 Hit Testing (Click Detection)
**File:** `src/gui/interaction.rs`

```rust
pub struct HitTester {
    blocks: Vec<BlockHitBox>,
}

pub struct BlockHitBox {
    pub block_id: String,
    pub bounds: Rect,
}

impl HitTester {
    pub fn from_graph(graph: &Graph) -> Self {
        // Extract hit boxes from graph layout
    }

    pub fn find_block_at(&self, world_pos: Vec2) -> Option<&str> {
        for block in &self.blocks {
            if block.bounds.contains(world_pos) {
                return Some(&block.block_id);
            }
        }
        None
    }
}

pub struct InteractionHandler {
    hit_tester: HitTester,
    view_state: ViewState,
}

impl InteractionHandler {
    pub fn handle_click(&mut self, screen_pos: Vec2, viewport: &Viewport) {
        let world_pos = viewport.screen_to_world(screen_pos);
        if let Some(block_id) = self.hit_tester.find_block_at(world_pos) {
            self.view_state.selected_block = Some(block_id.to_string());
            // Could trigger detail panel, highlight connections, etc.
        }
    }

    pub fn handle_hover(&mut self, screen_pos: Vec2, viewport: &Viewport) {
        let world_pos = viewport.screen_to_world(screen_pos);
        self.view_state.hover_block =
            self.hit_tester.find_block_at(world_pos).map(|s| s.to_string());
    }
}
```

### Phase 4: Hot Reload

#### 4.1 File Watching
**File:** `src/gui/hot_reload.rs`

```rust
use notify::{Watcher, RecursiveMode, Event};
use std::sync::mpsc::channel;

pub struct HotReloader {
    watcher: RecommendedWatcher,
    file_path: PathBuf,
}

impl HotReloader {
    pub fn new(file_path: PathBuf, callback: impl Fn() + Send + 'static) -> Self {
        let (tx, rx) = channel();

        let mut watcher = notify::recommended_watcher(move |res: Result<Event, _>| {
            if let Ok(event) = res {
                if event.kind.is_modify() {
                    // File was modified, trigger reload
                    callback();
                }
            }
        }).unwrap();

        watcher.watch(&file_path, RecursiveMode::NonRecursive).unwrap();

        Self { watcher, file_path }
    }
}
```

Integration into event loop:
```rust
// In IonGraphWindow
pub fn reload_data(&mut self) {
    // Re-read JSON file
    // Re-parse
    // Re-create graph
    // Request redraw
}

// In main event loop
let (reload_tx, reload_rx) = channel();
let hot_reloader = HotReloader::new(args.json_path.clone(), move || {
    reload_tx.send(()).unwrap();
});

// In event loop
if reload_rx.try_recv().is_ok() {
    window.reload_data();
}
```

### Phase 5: Visual Enhancements

#### 5.1 Highlighting
**File:** `src/gui/highlighting.rs`

```rust
impl SkiaLayoutProvider {
    fn render_with_highlights(
        &mut self,
        selected: Option<&str>,
        hovered: Option<&str>,
    ) {
        // During render, apply different styles based on state
        for cmd in &self.commands {
            match cmd {
                RenderCommand::DrawBlock { id, .. } => {
                    let color = if Some(id.as_str()) == selected {
                        SELECTED_COLOR
                    } else if Some(id.as_str()) == hovered {
                        HOVER_COLOR
                    } else {
                        DEFAULT_COLOR
                    };
                    // Draw with appropriate color
                }
                // ... other commands
            }
        }
    }
}
```

#### 5.2 Connection Highlighting
When a block is selected, highlight its:
- Incoming arrows (predecessors)
- Outgoing arrows (successors)
- Related blocks (dim others)

```rust
fn highlight_connections(&mut self, selected_block_id: &str) {
    // Find block in graph
    // Get predecessors and successors
    // Mark arrows for highlighting
    // Dim unrelated blocks
}
```

### Phase 6: Refinements

#### 6.1 Performance Optimizations
1. **Caching:** Cache Skia render commands between frames (only rebuild on zoom/pan)
2. **Culling:** Only render blocks visible in viewport
3. **Level of Detail:** Simplify rendering at high zoom-out levels

#### 6.2 UI Polish
1. **Mini-map:** Small overview showing viewport position
2. **Zoom controls:** Buttons for zoom in/out/fit
3. **Keyboard shortcuts:**
   - `Space + drag`: Pan
   - `Scroll`: Zoom
   - `F`: Fit to window
   - `R`: Reset zoom
4. **Status bar:** Show current block info, zoom level

#### 6.3 Multi-pass Viewer
Add UI to switch between compilation passes:
```rust
pub struct PassSelector {
    current_pass: usize,
    total_passes: usize,
}

// Keyboard shortcuts:
// Left/Right arrows: Previous/Next pass
// Number keys: Jump to specific pass
```

## File Structure

```
src/
├── lib.rs                          (add gui module)
├── gui/
│   ├── mod.rs                      (gui module root)
│   ├── window.rs                   (IonGraphWindow)
│   ├── viewport.rs                 (Viewport, pan/zoom)
│   ├── interaction.rs              (HitTester, InteractionHandler)
│   ├── hot_reload.rs               (HotReloader)
│   └── highlighting.rs             (Visual effects)
├── skia_layout_provider.rs         (SkiaLayoutProvider + SkiaElement)
└── bin/
    └── iongraph_viewer.rs          (GUI binary)
```

## Testing Strategy

### Unit Tests
- Viewport coordinate transformations (screen ↔ world)
- Hit testing with various graph layouts
- Hot reload file change detection

### Integration Tests
- Full render pipeline (SVG vs Skia comparison)
- Interaction sequences (click, drag, zoom)

### Manual Testing
- Load various Ion JSON files
- Test pan/zoom smoothness
- Verify hot reload works
- Check cross-platform (Windows, macOS, Linux)

## Risks & Mitigations

### Risk: Skia build complexity
**Mitigation:** Use pre-built skia-safe binaries (default), document build requirements

### Risk: Performance with large graphs
**Mitigation:** Implement culling and caching early, profile with mega-complex.json

### Risk: Coordinate system confusion
**Mitigation:** Clear naming (world vs screen), comprehensive viewport tests

### Risk: Different rendering between SVG and Skia
**Mitigation:** Share font metrics constants, visual comparison tests

## Success Criteria

1. ✅ GUI window displays Ion graphs correctly
2. ✅ Smooth pan and zoom (60 FPS)
3. ✅ Click to select blocks, shows selection state
4. ✅ Hot reload updates graph when file changes
5. ✅ SVG export still works identically
6. ✅ Works on Windows, macOS, Linux
7. ✅ Handles large graphs (mega-complex.json) smoothly

## Timeline Estimates

**Note:** These are implementation complexity estimates, not time-based:

- **Phase 1 (Foundation):** High complexity - New LayoutProvider paradigm
- **Phase 2 (Window):** Medium complexity - Standard winit + glutin setup
- **Phase 3 (Interactivity):** Low complexity - Standard game dev patterns
- **Phase 4 (Hot Reload):** Low complexity - notify crate is straightforward
- **Phase 5 (Enhancements):** Low complexity - Visual polish
- **Phase 6 (Refinements):** Medium complexity - Performance tuning is iterative

**Total:** Substantial project, but well-scoped with clear phases.

## Open Questions

1. **Text rendering:** Should we match SVG text rendering exactly, or use Skia's better text shaping?
   - **Recommendation:** Use Skia's text rendering (better quality), but match sizing

2. **Styling:** CSS-like classes in SVG vs direct Skia drawing?
   - **Recommendation:** Map CSS classes to Skia Paint styles in SkiaLayoutProvider

3. **Export from GUI:** Should GUI window support exporting current view as PNG/SVG?
   - **Recommendation:** Yes, add `Cmd+E` to export (future enhancement)

## Future Enhancements (Out of Scope)

- Animation between compilation passes
- Dark mode theme
- Graph diffing (compare two passes visually)
- WebGPU backend (for web version via wasm)
- Touch gestures for tablets
