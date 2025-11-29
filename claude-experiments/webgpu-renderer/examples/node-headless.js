/**
 * Node.js headless rendering example
 *
 * This example demonstrates how to use the WebGPU renderer in Node.js
 * without a canvas, rendering to an offscreen texture and exporting
 * the result to an image file.
 *
 * Prerequisites:
 * - npm install webgpu
 * - npm install pngjs (for PNG export)
 */

import { WebGPURenderer } from '../src/renderer/webgpu-renderer.js';
import { Scene, Quad, Shadow, Background } from '../src/core/primitives.js';
import { Bounds, Point, Size, Corners, Edges, Hsla } from '../src/core/geometry.js';
import { exportImageData } from '../src/platform/webgpu-platform.js';
import { PNG } from 'pngjs';
import fs from 'fs';

async function main() {
    console.log('Initializing headless WebGPU renderer...');

    // Create renderer without canvas
    const renderer = new WebGPURenderer(null);
    await renderer.initialize();

    console.log('Renderer initialized successfully');

    // Create a scene
    const scene = new Scene();

    // Add a background quad with gradient
    const background = new Quad();
    background.bounds = new Bounds(new Point(0, 0), new Size(800, 600));
    background.background = Background.LinearGradient(
        135,
        [
            { offset: 0, color: Hsla.rgb(0.2, 0.3, 0.8, 1) },
            { offset: 1, color: Hsla.rgb(0.8, 0.3, 0.8, 1) }
        ],
        1 // Oklab color space
    );
    scene.insertQuad(background);

    // Add a card with shadow
    const shadow = new Shadow();
    shadow.bounds = new Bounds(new Point(250, 200), new Size(300, 200));
    shadow.cornerRadii = Corners.uniform(12);
    shadow.blurRadius = 20;
    shadow.color = Hsla.black(0.3);
    scene.insertShadow(shadow);

    const card = new Quad();
    card.bounds = new Bounds(new Point(250, 200), new Size(300, 200));
    card.background = Background.Solid(Hsla.white(1));
    card.cornerRadii = Corners.uniform(12);
    card.borderWidths = Edges.uniform(2);
    card.borderColor = Hsla.rgb(0.8, 0.8, 0.8, 1);
    scene.insertQuad(card);

    // Add title quad with gradient
    const titleQuad = new Quad();
    titleQuad.bounds = new Bounds(new Point(270, 220), new Size(260, 60));
    titleQuad.background = Background.LinearGradient(
        90,
        [
            { offset: 0, color: Hsla.rgb(0.9, 0.3, 0.5, 1) },
            { offset: 1, color: Hsla.rgb(0.5, 0.3, 0.9, 1) }
        ],
        0 // sRGB
    );
    titleQuad.cornerRadii = Corners.uniform(8);
    scene.insertQuad(titleQuad);

    // Add some colored quads
    const colors = [
        Hsla.rgb(1.0, 0.3, 0.3, 1), // Red
        Hsla.rgb(0.3, 1.0, 0.3, 1), // Green
        Hsla.rgb(0.3, 0.3, 1.0, 1), // Blue
        Hsla.rgb(1.0, 1.0, 0.3, 1), // Yellow
    ];

    for (let i = 0; i < 4; i++) {
        const quad = new Quad();
        quad.bounds = new Bounds(
            new Point(270 + i * 65, 300),
            new Size(50, 50)
        );
        quad.background = Background.Solid(colors[i]);
        quad.cornerRadii = Corners.uniform(8);
        quad.borderWidths = Edges.uniform(2);
        quad.borderColor = Hsla.white(0.5);
        scene.insertQuad(quad);
    }

    console.log('Rendering scene...');

    // Render to offscreen texture
    const texture = renderer.render(scene, { width: 800, height: 600 });

    console.log('Exporting image data...');

    // Export texture data
    const imageData = await exportImageData(renderer.device, texture, 800, 600);

    console.log('Writing PNG file...');

    // Create PNG
    const png = new PNG({ width: 800, height: 600 });

    // Copy data (need to handle stride)
    const bytesPerRow = Math.ceil(800 * 4 / 256) * 256;
    for (let y = 0; y < 600; y++) {
        for (let x = 0; x < 800; x++) {
            const srcOffset = y * bytesPerRow + x * 4;
            const dstOffset = (y * 800 + x) * 4;

            png.data[dstOffset + 0] = imageData[srcOffset + 0]; // R
            png.data[dstOffset + 1] = imageData[srcOffset + 1]; // G
            png.data[dstOffset + 2] = imageData[srcOffset + 2]; // B
            png.data[dstOffset + 3] = imageData[srcOffset + 3]; // A
        }
    }

    // Write to file
    const buffer = PNG.sync.write(png);
    fs.writeFileSync('output.png', buffer);

    console.log('Successfully wrote output.png');

    // Cleanup
    texture.destroy();
    renderer.destroy();

    console.log('Done!');
}

main().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
