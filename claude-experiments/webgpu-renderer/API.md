# WebGPU Renderer API Reference

A comprehensive GPU-accelerated 2D rendering system inspired by GPUI, implemented using WebGPU.

## Table of Contents

1. [Core Primitives](#core-primitives)
2. [Geometry](#geometry)
3. [Scene Management](#scene-management)
4. [Rendering](#rendering)
5. [Text](#text)
6. [Hit Testing](#hit-testing)
7. [Advanced Features](#advanced-features)

## Core Primitives

### Quad

Rounded rectangles with borders and various background types.

```javascript
import { Quad, Background, Bounds, Point, Size, Corners, Edges, Hsla } from './core/primitives.js';

const quad = new Quad();
quad.bounds = new Bounds(new Point(100, 100), new Size(200, 150));
quad.background = Background.Solid(Hsla.rgb(0.9, 0.3, 0.5, 1));
quad.cornerRadii = Corners.uniform(12);
quad.borderWidths = Edges.uniform(2);
quad.borderColor = Hsla.rgb(0.2, 0.2, 0.2, 1);
quad.borderStyle = 0; // 0 = Solid, 1 = Dashed
```

**Background Types:**

- `Background.Solid(color)` - Solid color fill
- `Background.LinearGradient(angle, stops, colorSpace)` - Linear gradients
  - `colorSpace`: 0 = sRGB, 1 = Oklab
- `Background.RadialGradient(centerX, centerY, radius, stops, colorSpace)` - Radial gradients
  - `centerX`, `centerY`: 0-1 normalized position (0.5, 0.5 = center)
  - `radius`: 0-1+ normalized radius
  - `colorSpace`: 0 = sRGB, 1 = Oklab
- `Background.ConicGradient(centerX, centerY, angle, stops, colorSpace)` - Conic (angular/sweep) gradients
  - `centerX`, `centerY`: 0-1 normalized position (0.5, 0.5 = center)
  - `angle`: Starting angle in degrees (0 = right, 90 = down, etc.)
  - `colorSpace`: 0 = sRGB, 1 = Oklab
- `Background.Pattern(angle, color1, color2, spacing, patternType)` - Procedural patterns
  - `patternType`: 0 = stripes, 1 = dots, 2 = checkerboard, 3 = grid

**Example:**

```javascript
// Radial gradient from center
quad.background = Background.RadialGradient(
    0.5, 0.5, 0.7,  // center at (0.5, 0.5), radius 0.7
    [
        { color: Hsla.rgb(1.0, 0.9, 0.3, 1), position: 0 },
        { color: Hsla.rgb(0.9, 0.3, 0.5, 1), position: 1 }
    ],
    1  // Oklab color space
);
```

### Shadow

Analytically-rendered Gaussian blur shadows.

```javascript
import { Shadow } from './core/primitives.js';

const shadow = new Shadow();
shadow.bounds = new Bounds(new Point(100, 100), new Size(200, 150));
shadow.cornerRadii = Corners.uniform(12);
shadow.blurRadius = 10;
shadow.color = Hsla.black(0.3);
```

### Underline

Straight or wavy underlines.

```javascript
import { Underline } from './core/primitives.js';

const underline = new Underline();
underline.bounds = new Bounds(new Point(100, 200), new Size(150, 10));
underline.color = Hsla.rgb(0.2, 0.4, 0.8, 1);
underline.thickness = 3;
underline.wavy = 0; // 0 = straight, 1 = wavy
```

### MonochromeSprite

Single-channel textures with color tinting (glyphs, icons).

```javascript
import { MonochromeSprite } from './core/primitives.js';

const sprite = new MonochromeSprite();
sprite.bounds = new Bounds(new Point(100, 100), new Size(48, 48));
sprite.color = Hsla.rgb(0.9, 0.2, 0.3, 1);
sprite.tile = atlasTextile; // AtlasTile from texture atlas
```

### PolychromeSprite

Full-color RGBA textures (images, emojis).

```javascript
import { PolychromeSprite } from './core/primitives.js';

const sprite = new PolychromeSprite();
sprite.bounds = new Bounds(new Point(100, 100), new Size(80, 80));
sprite.tile = atlasTextile;
sprite.cornerRadii = Corners.uniform(10);
sprite.opacity = 0.8;
sprite.grayscale = false; // true to convert to grayscale
```

## Geometry

### Transform

2D affine transformations (translation, rotation, scale).

```javascript
import { Transform } from './core/geometry.js';

// Identity transform
const identity = Transform.identity();

// Translation
const translate = Transform.translation(50, 100);

// Rotation (radians)
const rotate = Transform.rotation(Math.PI / 4); // 45 degrees

// Scale
const scale = Transform.scale(2.0); // uniform
const scaleXY = Transform.scale(2.0, 0.5); // non-uniform

// Composition
const combined = Transform.translation(100, 100)
    .multiply(Transform.rotation(angle))
    .multiply(Transform.translation(-100, -100));

// Apply transform to primitives
quad.transform = combined;
```

### Bounds and Corners

```javascript
import { Bounds, Point, Size, Corners, Edges } from './core/geometry.js';

// Bounding rectangle
const bounds = new Bounds(
    new Point(x, y),    // origin
    new Size(width, height)
);

// Corner radii
const corners = Corners.uniform(10); // all corners
const corners = new Corners(
    topLeft, topRight,
    bottomRight, bottomLeft
);

// Border widths
const borders = Edges.uniform(2); // all edges
const borders = new Edges(top, right, bottom, left);
```

### Colors

```javascript
import { Hsla } from './core/geometry.js';

// Create colors
const color = Hsla.rgb(0.9, 0.3, 0.5, 1.0); // r, g, b, a (0-1)
const black = Hsla.black(0.5); // black with 50% alpha
const white = Hsla.white(1.0); // opaque white

// Colors in HSL space
const hsl = new Hsla(0.5, 0.8, 0.6, 1.0); // h, s, l, a
```

## Scene Management

### Scene

The scene manages all primitives and provides hierarchical clipping.

```javascript
import { Scene } from './core/primitives.js';

const scene = new Scene();

// Add primitives
scene.insertQuad(quad);
scene.insertShadow(shadow);
scene.insertUnderline(underline);
scene.insertMonochromeSprite(sprite);
scene.insertPolychromeSprite(sprite);

// Hierarchical clipping
scene.pushClip(new Bounds(new Point(100, 100), new Size(300, 200)));
// ... add primitives that will be clipped
scene.popClip();

// Nested clipping (intersects with parent)
scene.pushClip(bounds1);
scene.pushClip(bounds2); // intersection of bounds1 and bounds2
// ...
scene.popClip();
scene.popClip();

// Clear scene
scene.clear();
```

## Rendering

### WebGPURenderer

```javascript
import { WebGPURenderer } from './renderer/webgpu-renderer.js';

const canvas = document.getElementById('canvas');
const renderer = new WebGPURenderer(canvas);

// Initialize (async)
await renderer.initialize();

// Render scene
renderer.render(scene);

// Access atlases for texture upload
const monochromeTile = renderer.monochromeAtlas.getOrInsert(
    'circle',
    64, 64,
    imageData // Uint8Array
);

// Buffer pool statistics
const stats = renderer.bufferPool.getStats();
console.log(`Buffers: ${stats.total}, In use: ${stats.inUse}, Size: ${stats.totalSizeMB}MB`);

// Cleanup
renderer.destroy();
```

## Text

### TextRenderer

```javascript
import { TextRenderer } from './utils/text-renderer.js';

const textRenderer = new TextRenderer(renderer);

// Render text as monochrome sprites
const sprites = textRenderer.renderText(
    'Hello World',
    100, 100,              // x, y position
    Hsla.rgb(0.2, 0.2, 0.2, 1),  // color
    16,                    // font size
    'sans-serif',          // font family
    'bold'                 // font weight
);

// Add to scene
sprites.forEach(sprite => scene.insertMonochromeSprite(sprite));
```

### Text Layout

```javascript
import {
    TextMeasurer,
    getAlignedTextPosition,
    TextAlign,
    VerticalAlign,
    truncateText
} from './utils/text-layout.js';

const measurer = new TextMeasurer();

// Measure text
const measurement = measurer.measureText('Hello', 16, 'sans-serif');
console.log(measurement.width, measurement.height);

// Text wrapping
const lines = measurer.wrapText(
    'Long text that needs wrapping',
    200,  // max width
    14,   // font size
    'sans-serif'
);

// Multi-line measurement
const multiline = measurer.measureMultilineText(
    'Text\\nWith\\nLines',
    200,  // max width
    14    // font size
);

// Text alignment
const pos = getAlignedTextPosition(
    textWidth, textHeight,
    containerX, containerY,
    containerWidth, containerHeight,
    TextAlign.Center,
    VerticalAlign.Middle
);

// Truncation with ellipsis
const truncated = truncateText(
    'Very long text that should be truncated',
    150,  // max width
    12    // font size
);
```

## Hit Testing

### HitTester

```javascript
import { HitTester } from './utils/hit-tester.js';

const hitTester = new HitTester(scene);

// Test hit at screen coordinates
const result = hitTester.hitTest(mouseX, mouseY);

if (result) {
    console.log(`Hit ${result.type} at index ${result.index}`);
    const primitive = result.primitive;

    // result.type: 'quad', 'shadow', 'underline',
    //              'monochromeSprite', 'polychromeSprite'
}
```

Hit testing accounts for:
- Transform matrices
- Rounded corners (SDF-based)
- Z-order (tests in reverse, topmost first)

## Advanced Features

### Buffer Pooling

Automatically manages GPU buffer reuse to reduce allocation overhead.

```javascript
// Buffers are automatically pooled
renderer.render(scene); // acquires buffers from pool
// Next frame, buffers are automatically released and reused

// Get statistics
const stats = renderer.bufferPool.getStats();
```

### Texture Atlases

```javascript
import { Atlas } from './core/atlas.js';

// Monochrome atlas (single-channel)
const atlas = new Atlas(device, 'monochrome', 1024);

// Upload texture
const tile = atlas.getOrInsert('my-texture', width, height, imageData);

// Use tile with sprites
sprite.tile = tile;

// Clear atlas
atlas.clear();
```

### Procedural Textures

```javascript
import { ProceduralImages } from './utils/image-loader.js';

const imageLoader = new ImageLoader(renderer);

// Create procedural sprite
const sprite = await imageLoader.createProceduralSprite(
    64, 64,
    ProceduralImages.gradientCircle,
    x, y,
    { cornerRadii: Corners.uniform(8), opacity: 1.0 }
);

// Available generators:
// - ProceduralImages.gradientCircle
// - ProceduralImages.geometric
// - ProceduralImages.mandelbrot
```

### Camera (Viewport System)

```javascript
import { Camera } from './core/camera.js';

const camera = new Camera(canvas.width, canvas.height);

// Pan camera
camera.pan(deltaX, deltaY);

// Zoom at point
camera.zoomAt(1.1, mouseX, mouseY); // 10% zoom in

// Reset camera
camera.reset();

// Coordinate conversion
const world = camera.screenToWorld(screenX, screenY);
const screen = camera.worldToScreen(worldX, worldY);

// Get visible area
const bounds = camera.getVisibleBounds();
```

### Animation

```javascript
import { Animation, Easing, Spring, Oscillator, lerp } from './utils/animation.js';

// Simple animation with easing
const anim = new Animation(1000, Easing.easeInOutCubic);
anim.onUpdate = (t) => {
    quad.opacity = lerp(0, 1, t);
};
anim.onComplete = () => {
    console.log('Animation complete!');
};
anim.start();

// Loop animation
anim.setLoop(true).start();

// Yoyo animation (back and forth)
anim.setYoyo(true).start();

// Update in animation loop
function animate(time) {
    anim.update(time);
    requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

// Spring physics for smooth, natural motion
const spring = new Spring(170, 26); // stiffness, damping
spring.setValue(0);
spring.setTarget(100);

function updateSpring() {
    const dt = 1/60; // 60 FPS
    const value = spring.update(dt);
    quad.bounds.origin.x = value;

    if (!spring.isAtRest()) {
        requestAnimationFrame(updateSpring);
    }
}
updateSpring();

// Oscillators for periodic animations
const osc = new Oscillator(1.0, 50, 0); // frequency, amplitude, phase
const y = osc.sine(time);
quad.bounds.origin.y = 200 + y;

// Available easing functions:
// - Easing.linear
// - Easing.easeInQuad, easeOutQuad, easeInOutQuad
// - Easing.easeInCubic, easeOutCubic, easeInOutCubic
// - Easing.easeInQuart, easeOutQuart, easeInOutQuart
// - Easing.easeInQuint, easeOutQuint, easeInOutQuint
// - Easing.easeInSine, easeOutSine, easeInOutSine
// - Easing.easeInExpo, easeOutExpo, easeInOutExpo
// - Easing.easeInCirc, easeOutCirc, easeInOutCirc
// - Easing.easeInElastic, easeOutElastic, easeInOutElastic
// - Easing.easeInBack, easeOutBack, easeInOutBack
// - Easing.easeInBounce, easeOutBounce, easeInOutBounce

// Utility functions
const value = lerp(start, end, t); // Linear interpolation
const clamped = clamp(value, min, max); // Clamp to range
const mapped = map(value, inMin, inMax, outMin, outMax); // Map ranges
```

### Node.js Headless Rendering

```javascript
import { WebGPURenderer } from './renderer/webgpu-renderer.js';
import { exportImageData } from './platform/webgpu-platform.js';
import { Scene, Quad, Background } from './core/primitives.js';
import { Bounds, Point, Size, Hsla } from './core/geometry.js';

// Create renderer without canvas (for Node.js)
const renderer = new WebGPURenderer(null);
await renderer.initialize();

// Build scene
const scene = new Scene();
const quad = new Quad();
quad.bounds = new Bounds(new Point(100, 100), new Size(200, 150));
quad.background = Background.Solid(Hsla.rgb(0.9, 0.3, 0.5, 1));
scene.insertQuad(quad);

// Render to offscreen texture
const texture = renderer.render(scene, { width: 800, height: 600 });

// Export image data
const imageData = await exportImageData(renderer.device, texture, 800, 600);

// imageData is Uint8Array with RGBA pixels
// Can be saved to PNG, JPEG, etc. using libraries like pngjs

// Cleanup
texture.destroy();
renderer.destroy();
```

**Platform Detection:**

```javascript
import { isNode, getWebGPU, getPreferredCanvasFormat } from './platform/webgpu-platform.js';

if (isNode()) {
    console.log('Running in Node.js');
} else {
    console.log('Running in browser');
}

// Get WebGPU (works in both environments)
const gpu = await getWebGPU();

// Get preferred format (platform-aware)
const format = await getPreferredCanvasFormat(gpu);
```

**Note**: Image data from `exportImageData()` has GPU alignment (256-byte rows). See `examples/node-headless.js` for a complete example including PNG export.

## Performance Tips

1. **Use buffer pooling** - Automatically enabled, reduces allocation overhead
2. **Batch primitives** - Group similar primitives together in the scene
3. **Reuse textures** - Cache atlas tiles instead of re-uploading
4. **Minimize state changes** - Scene automatically batches by primitive type
5. **Use transforms** - More efficient than recreating primitives
6. **Monitor buffer stats** - Check `renderer.bufferPool.getStats()` for memory usage

## Platform Support

### Browser

Requires WebGPU support:
- Chrome 113+
- Edge 113+
- Firefox (behind flag)
- Safari Technology Preview

Check support: `navigator.gpu !== undefined`

### Node.js

Requires:
- Node.js 18+
- `webgpu` package (node-webgpu)
- Vulkan, Metal, or DirectX 12 support

Install: `npm install webgpu`

**Platform differences:**
- No canvas/DOM APIs
- Headless rendering only
- Offscreen textures
- Manual image export required
- Same primitive/scene API
