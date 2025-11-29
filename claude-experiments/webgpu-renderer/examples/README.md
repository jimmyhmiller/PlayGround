# Node.js Headless Rendering Example

This directory contains examples of using the WebGPU renderer in Node.js without a browser.

## Prerequisites

Install the required dependencies:

```bash
npm install webgpu pngjs
```

**Note**: The `webgpu` package (node-webgpu) requires:
- Node.js 18 or later
- A system with Vulkan, Metal, or DirectX 12 support
- On Linux: Vulkan drivers installed

## Running the Example

```bash
node examples/node-headless.js
```

This will:
1. Initialize a headless WebGPU renderer (no canvas)
2. Create a scene with gradients, shadows, and colored quads
3. Render the scene to an offscreen texture
4. Export the texture data
5. Save the result as `output.png`

## How It Works

### 1. Create Renderer Without Canvas

```javascript
import { WebGPURenderer } from '../src/renderer/webgpu-renderer.js';

const renderer = new WebGPURenderer(null); // null = no canvas
await renderer.initialize();
```

The renderer automatically detects it's running in Node.js and uses the `webgpu` package instead of the browser's `navigator.gpu`.

### 2. Build Scene

Create primitives as normal:

```javascript
import { Scene, Quad, Shadow, Background } from '../src/core/primitives.js';
import { Bounds, Point, Size, Hsla } from '../src/core/geometry.js';

const scene = new Scene();

const quad = new Quad();
quad.bounds = new Bounds(new Point(100, 100), new Size(200, 150));
quad.background = Background.Solid(Hsla.rgb(0.9, 0.3, 0.5, 1));
scene.insertQuad(quad);
```

### 3. Render to Offscreen Texture

```javascript
const texture = renderer.render(scene, { width: 800, height: 600 });
```

When there's no canvas, `render()` creates an offscreen texture and returns it.

### 4. Export Image Data

```javascript
import { exportImageData } from '../src/platform/webgpu-platform.js';

const imageData = await exportImageData(renderer.device, texture, 800, 600);
```

This copies the texture data from GPU to CPU memory.

### 5. Save as PNG

```javascript
import { PNG } from 'pngjs';
import fs from 'fs';

const png = new PNG({ width: 800, height: 600 });

// Copy data (accounting for GPU alignment)
const bytesPerRow = Math.ceil(800 * 4 / 256) * 256;
for (let y = 0; y < 600; y++) {
    for (let x = 0; x < 800; x++) {
        const srcOffset = y * bytesPerRow + x * 4;
        const dstOffset = (y * 800 + x) * 4;
        png.data[dstOffset + 0] = imageData[srcOffset + 0];
        png.data[dstOffset + 1] = imageData[srcOffset + 1];
        png.data[dstOffset + 2] = imageData[srcOffset + 2];
        png.data[dstOffset + 3] = imageData[srcOffset + 3];
    }
}

fs.writeFileSync('output.png', PNG.sync.write(png));
```

## Platform Abstraction

The renderer uses a platform abstraction layer (`src/platform/webgpu-platform.js`) that handles differences between browser and Node.js:

- **Browser**: Uses `navigator.gpu`, canvas context, preferred format
- **Node.js**: Uses `webgpu` package, offscreen rendering, default format

Your application code doesn't need to change - the same primitives, scenes, and rendering code work in both environments.

## Use Cases

Headless rendering is useful for:

- **Server-side rendering**: Generate UI screenshots or previews
- **Testing**: Automated visual regression testing
- **Image generation**: Batch processing, thumbnails, social media cards
- **CI/CD**: Generate images in build pipelines
- **CLI tools**: Command-line image generation utilities

## Limitations

Current limitations in Node.js mode:

- No text rendering (requires Canvas API for glyph generation)
- No image loading from URLs (requires browser fetch/Image APIs)
- Procedural textures require manual implementation

These features work in browser mode and could be adapted for Node.js with additional dependencies.

## Troubleshooting

### `webgpu` package not found

```bash
npm install webgpu
```

### Vulkan/graphics driver errors

Make sure your system has:
- **Linux**: Vulkan drivers (`vulkan-tools`, `mesa-vulkan-drivers`)
- **macOS**: Metal support (built-in on modern macOS)
- **Windows**: DirectX 12 or Vulkan drivers

### Image data is black/empty

Check that:
1. Scene has visible primitives
2. Primitive bounds are within render target (0,0 to width,height)
3. Colors have non-zero alpha
4. Device commands are submitted before reading texture

### Stride/alignment issues

GPU textures have alignment requirements (256-byte rows). The example code handles this by computing `bytesPerRow` and copying pixel-by-pixel.
