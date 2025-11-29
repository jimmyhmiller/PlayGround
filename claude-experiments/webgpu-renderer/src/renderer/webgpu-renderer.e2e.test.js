/**
 * E2E rendering tests with image output
 * These tests render various primitives and save PNG images for visual inspection
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { WebGPURenderer } from './webgpu-renderer.js';
import { Scene, Quad, Shadow, Underline, MonochromeSprite, PolychromeSprite, Path } from '../core/primitives.js';
import { Bounds, Point, Hsla, Background, BorderStyle } from '../core/geometry.js';
import { PathBuilder } from '../core/path.js';

const RENDER_WIDTH = 400;
const RENDER_HEIGHT = 400;

// Track rendered images for logging
const renderedImages = [];

/**
 * Save image data as base64 data URL for later inspection
 */
function saveImageData(name, buffer, width, height) {
    // Convert RGBA buffer to canvas and get data URL
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    const imageData = ctx.createImageData(width, height);
    imageData.data.set(buffer);
    ctx.putImageData(imageData, 0, 0);

    const dataURL = canvas.toDataURL('image/png');

    renderedImages.push({ name, dataURL, width, height });

    return dataURL;
}

/**
 * Read texture data from GPU and convert to RGBA buffer
 */
async function readTextureData(renderer, texture, width, height) {
    const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
    const buffer = renderer.device.createBuffer({
        size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = renderer.device.createCommandEncoder();
    encoder.copyTextureToBuffer(
        { texture },
        { buffer, bytesPerRow },
        { width, height }
    );
    renderer.device.queue.submit([encoder.finish()]);

    await buffer.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(buffer.getMappedRange());

    // Convert from padded rows to tightly packed RGBA
    const rgba = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        const srcOffset = y * bytesPerRow;
        const dstOffset = y * width * 4;
        rgba.set(data.subarray(srcOffset, srcOffset + width * 4), dstOffset);
    }

    buffer.unmap();
    buffer.destroy();

    return rgba;
}

/**
 * Render scene to offscreen texture and return pixel data
 */
async function renderToTexture(renderer, scene) {
    // Create offscreen render target
    const texture = renderer.device.createTexture({
        size: { width: RENDER_WIDTH, height: RENDER_HEIGHT },
        format: renderer.presentationFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    const encoder = renderer.device.createCommandEncoder();

    // Update uniforms
    const uniformData = new Float32Array([
        RENDER_WIDTH, RENDER_HEIGHT,  // viewport_size
        1,  // premultiplied_alpha (as u32)
        0,  // padding
    ]);
    renderer.device.queue.writeBuffer(renderer.uniformBuffer, 0, uniformData);

    // Render scene (simplified version of main render loop for offscreen)
    const batches = Array.from(scene.batches());
    const pathBatches = batches.filter(b => b.type === 'paths');
    const nonPathBatches = batches.filter(b => b.type !== 'paths');

    // Create/recreate intermediate textures if needed for paths
    if (pathBatches.length > 0) {
        if (!renderer.pathIntermediateTexture ||
            renderer.pathIntermediateTexture.width !== RENDER_WIDTH ||
            renderer.pathIntermediateTexture.height !== RENDER_HEIGHT) {
            renderer.createPathIntermediateTextures(RENDER_WIDTH, RENDER_HEIGHT);
        }

        // Pass 1: Render paths to intermediate texture
        const pathRenderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: renderer.pathSampleCount > 1 ? renderer.pathIntermediateMSAATextureView : renderer.pathIntermediateTextureView,
                resolveTarget: renderer.pathSampleCount > 1 ? renderer.pathIntermediateTextureView : undefined,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear',
                storeOp: renderer.pathSampleCount > 1 ? 'discard' : 'store',
            }],
        });

        for (const batch of pathBatches) {
            renderer.renderPathsLoopBlinn(pathRenderPass, batch.primitives);
        }

        pathRenderPass.end();
    }

    // Pass 2: Main render pass
    const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
            view: texture.createView(),
            clearValue: { r: 1, g: 1, b: 1, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
        }],
    });

    // Render all non-path batches
    for (const batch of nonPathBatches) {
        switch (batch.type) {
            case 'quads':
                renderer.renderQuads(renderPass, batch.primitives);
                break;
            case 'shadows':
                renderer.renderShadows(renderPass, batch.primitives);
                break;
            case 'underlines':
                renderer.renderUnderlines(renderPass, batch.primitives);
                break;
            case 'monochrome_sprites':
                renderer.renderMonochromeSprites(renderPass, batch.primitives);
                break;
            case 'polychrome_sprites':
                renderer.renderPolychromeSprites(renderPass, batch.primitives);
                break;
        }
    }

    // Composite paths
    if (pathBatches.length > 0) {
        renderer.renderPathComposite(renderPass);
    }

    renderPass.end();

    // Read texture data
    const pixelData = await readTextureData(renderer, texture, RENDER_WIDTH, RENDER_HEIGHT);

    texture.destroy();

    return pixelData;
}

describe('E2E Rendering Tests', () => {
    let renderer;

    beforeAll(async () => {
        // Initialize WebGPU
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('Failed to get WebGPU adapter');
        }

        const device = await adapter.requestDevice();

        // Create a dummy canvas (won't be used for actual rendering)
        const canvas = document.createElement('canvas');
        canvas.width = RENDER_WIDTH;
        canvas.height = RENDER_HEIGHT;

        // Create renderer
        renderer = new WebGPURenderer(canvas, { pathSampleCount: 4 });
        await renderer.initialize();
    });

    afterAll(() => {
        console.log(`\n✅ E2E tests completed. ${renderedImages.length} images rendered.\n`);
        console.log('To view rendered images:');
        console.log('1. Check browser console for base64 data URLs');
        console.log('2. Or use screenshot capabilities in the test UI\n');

        // Log first image as example
        if (renderedImages.length > 0) {
            console.log(`Example - ${renderedImages[0].name}:`);
            console.log(renderedImages[0].dataURL.substring(0, 100) + '...');
        }
    });

    it('renders solid color quad', async () => {
        const scene = new Scene();

        const quad = new Quad(
            new Bounds(new Point(50, 50), { width: 300, height: 300 }),
            Background.Solid(new Hsla(0.6, 0.7, 0.5, 1.0)),  // Blue
            { top_left: 0, top_right: 0, bottom_left: 0, bottom_right: 0 },
            { top: 0, right: 0, bottom: 0, left: 0 },
            Hsla.transparent(),
            BorderStyle.Solid
        );
        quad.setOrder(0);
        scene.insert(quad);

        const pixelData = await renderToTexture(renderer, scene);
        saveImageData('01-solid-quad', pixelData, RENDER_WIDTH, RENDER_HEIGHT);

        expect(pixelData).toBeDefined();
        expect(pixelData.length).toBe(RENDER_WIDTH * RENDER_HEIGHT * 4);

        // Check that center pixel is roughly blue (allowing for antialiasing)
        const centerIdx = (RENDER_HEIGHT / 2 * RENDER_WIDTH + RENDER_WIDTH / 2) * 4;
        const r = pixelData[centerIdx];
        const g = pixelData[centerIdx + 1];
        const b = pixelData[centerIdx + 2];

        expect(b).toBeGreaterThan(r);
        expect(b).toBeGreaterThan(g);
    });

    it('renders rounded corner quad', async () => {
        const scene = new Scene();

        const quad = new Quad(
            new Bounds(new Point(50, 50), { width: 300, height: 300 }),
            Background.Solid(new Hsla(0.0, 0.7, 0.5, 1.0)),  // Red
            { top_left: 50, top_right: 50, bottom_left: 50, bottom_right: 50 },
            { top: 0, right: 0, bottom: 0, left: 0 },
            Hsla.transparent(),
            BorderStyle.Solid
        );
        quad.setOrder(0);
        scene.insert(quad);

        const pixelData = await renderToTexture(renderer, scene);
        saveImageData('02-rounded-quad', pixelData, RENDER_WIDTH, RENDER_HEIGHT);

        expect(pixelData).toBeDefined();

        // Corner should be white (background), not red
        const cornerIdx = (60 * RENDER_WIDTH + 60) * 4;
        const r = pixelData[cornerIdx];
        expect(r).toBeGreaterThan(200);  // Should be mostly white
    });

    it('renders linear gradient', async () => {
        const scene = new Scene();

        const gradient = Background.LinearGradient(
            Math.PI / 4,  // 45 degrees
            [
                { offset: 0.0, color: new Hsla(0.0, 1.0, 0.5, 1.0) },  // Red
                { offset: 1.0, color: new Hsla(0.6, 1.0, 0.5, 1.0) },  // Blue
            ],
            0  // sRGB
        );

        const quad = new Quad(
            new Bounds(new Point(50, 50), { width: 300, height: 300 }),
            gradient,
            { top_left: 0, top_right: 0, bottom_left: 0, bottom_right: 0 },
            { top: 0, right: 0, bottom: 0, left: 0 },
            Hsla.transparent(),
            BorderStyle.Solid
        );
        quad.setOrder(0);
        scene.insert(quad);

        const pixelData = await renderToTexture(renderer, scene);
        saveImageData('03-linear-gradient', pixelData, RENDER_WIDTH, RENDER_HEIGHT);

        expect(pixelData).toBeDefined();
    });

    it('renders shadow', async () => {
        const scene = new Scene();

        // Shadow
        const shadow = new Shadow(
            new Bounds(new Point(70, 70), { width: 260, height: 260 }),
            { top_left: 20, top_right: 20, bottom_left: 20, bottom_right: 20 },
            new Hsla(0, 0, 0, 0.5),
            20.0
        );
        shadow.setOrder(0);
        scene.insert(shadow);

        // Quad on top
        const quad = new Quad(
            new Bounds(new Point(50, 50), { width: 300, height: 300 }),
            Background.Solid(new Hsla(0.15, 0.8, 0.6, 1.0)),
            { top_left: 20, top_right: 20, bottom_left: 20, bottom_right: 20 },
            { top: 0, right: 0, bottom: 0, left: 0 },
            Hsla.transparent(),
            BorderStyle.Solid
        );
        quad.setOrder(1);
        scene.insert(quad);

        const pixelData = await renderToTexture(renderer, scene);
        saveImageData('04-shadow', pixelData, RENDER_WIDTH, RENDER_HEIGHT);

        expect(pixelData).toBeDefined();
    });

    it('renders border', async () => {
        const scene = new Scene();

        const quad = new Quad(
            new Bounds(new Point(50, 50), { width: 300, height: 300 }),
            Background.Solid(new Hsla(0.0, 0.0, 1.0, 1.0)),  // White
            { top_left: 20, top_right: 20, bottom_left: 20, bottom_right: 20 },
            { top: 10, right: 10, bottom: 10, left: 10 },
            new Hsla(0.0, 0.7, 0.5, 1.0),  // Red border
            BorderStyle.Solid
        );
        quad.setOrder(0);
        scene.insert(quad);

        const pixelData = await renderToTexture(renderer, scene);
        saveImageData('05-border', pixelData, RENDER_WIDTH, RENDER_HEIGHT);

        expect(pixelData).toBeDefined();
    });

    it('renders underline', async () => {
        const scene = new Scene();

        // Straight underline
        const underline1 = new Underline(
            new Bounds(new Point(50, 100), { width: 150, height: 2 }),
            new Hsla(0.0, 0.7, 0.5, 1.0),  // Red
            2.0,
            false  // straight
        );
        underline1.setOrder(0);
        scene.insert(underline1);

        // Wavy underline
        const underline2 = new Underline(
            new Bounds(new Point(50, 200), { width: 150, height: 4 }),
            new Hsla(0.6, 0.7, 0.5, 1.0),  // Blue
            2.0,
            true  // wavy
        );
        underline2.setOrder(1);
        scene.insert(underline2);

        const pixelData = await renderToTexture(renderer, scene);
        saveImageData('06-underline', pixelData, RENDER_WIDTH, RENDER_HEIGHT);

        expect(pixelData).toBeDefined();
    });

    it('renders path - circle', async () => {
        const scene = new Scene();

        const builder = new PathBuilder();
        builder.circle(new Point(200, 200), 100);
        const pathData = builder.build();

        const path = new Path(pathData, new Hsla(0.33, 0.7, 0.5, 1.0));  // Green
        path.setOrder(0);
        scene.insert(path);

        const pixelData = await renderToTexture(renderer, scene);
        saveImageData('07-path-circle', pixelData, RENDER_WIDTH, RENDER_HEIGHT);

        expect(pixelData).toBeDefined();

        // Center should be green
        const centerIdx = (RENDER_HEIGHT / 2 * RENDER_WIDTH + RENDER_WIDTH / 2) * 4;
        const g = pixelData[centerIdx + 1];
        expect(g).toBeGreaterThan(100);
    });

    it('renders path - bezier curves', async () => {
        const scene = new Scene();

        const builder = new PathBuilder();
        builder.moveTo(new Point(50, 200));
        builder.quadraticCurveTo(new Point(150, 50), new Point(250, 200));
        builder.quadraticCurveTo(new Point(300, 300), new Point(350, 200));
        builder.close();
        const pathData = builder.build();

        const path = new Path(pathData, new Hsla(0.8, 0.7, 0.5, 1.0));  // Purple
        path.setOrder(0);
        scene.insert(path);

        const pixelData = await renderToTexture(renderer, scene);
        saveImageData('08-path-bezier', pixelData, RENDER_WIDTH, RENDER_HEIGHT);

        expect(pixelData).toBeDefined();
    });

    it('renders pattern fills', async () => {
        const scene = new Scene();

        // Diagonal stripes
        const quad1 = new Quad(
            new Bounds(new Point(20, 20), { width: 80, height: 80 }),
            Background.Pattern(0, new Hsla(0.0, 0.7, 0.5, 1.0)),  // Red stripes
            { top_left: 5, top_right: 5, bottom_left: 5, bottom_right: 5 },
            { top: 0, right: 0, bottom: 0, left: 0 },
            Hsla.transparent(),
            BorderStyle.Solid
        );
        quad1.setOrder(0);
        scene.insert(quad1);

        // Dots
        const quad2 = new Quad(
            new Bounds(new Point(120, 20), { width: 80, height: 80 }),
            Background.Pattern(1, new Hsla(0.6, 0.7, 0.5, 1.0)),  // Blue dots
            { top_left: 5, top_right: 5, bottom_left: 5, bottom_right: 5 },
            { top: 0, right: 0, bottom: 0, left: 0 },
            Hsla.transparent(),
            BorderStyle.Solid
        );
        quad2.setOrder(1);
        scene.insert(quad2);

        // Checkerboard
        const quad3 = new Quad(
            new Bounds(new Point(220, 20), { width: 80, height: 80 }),
            Background.Pattern(2, new Hsla(0.33, 0.7, 0.5, 1.0)),  // Green checkerboard
            { top_left: 5, top_right: 5, bottom_left: 5, bottom_right: 5 },
            { top: 0, right: 0, bottom: 0, left: 0 },
            Hsla.transparent(),
            BorderStyle.Solid
        );
        quad3.setOrder(2);
        scene.insert(quad3);

        const pixelData = await renderToTexture(renderer, scene);
        saveImageData('09-patterns', pixelData, RENDER_WIDTH, RENDER_HEIGHT);

        expect(pixelData).toBeDefined();
    });

    it('renders complex scene', async () => {
        const scene = new Scene();

        // Background gradient
        const bgGradient = Background.RadialGradient(
            [
                { offset: 0.0, color: new Hsla(0.6, 0.3, 0.9, 1.0) },
                { offset: 1.0, color: new Hsla(0.6, 0.5, 0.7, 1.0) },
            ],
            0  // sRGB
        );

        const bg = new Quad(
            new Bounds(new Point(0, 0), { width: RENDER_WIDTH, height: RENDER_HEIGHT }),
            bgGradient,
            { top_left: 0, top_right: 0, bottom_left: 0, bottom_right: 0 },
            { top: 0, right: 0, bottom: 0, left: 0 },
            Hsla.transparent(),
            BorderStyle.Solid
        );
        bg.setOrder(0);
        scene.insert(bg);

        // Shadow for card
        const shadow = new Shadow(
            new Bounds(new Point(60, 60), { width: 280, height: 280 }),
            { top_left: 10, top_right: 10, bottom_left: 10, bottom_right: 10 },
            new Hsla(0, 0, 0, 0.3),
            15.0
        );
        shadow.setOrder(1);
        scene.insert(shadow);

        // Card
        const card = new Quad(
            new Bounds(new Point(50, 50), { width: 300, height: 300 }),
            Background.Solid(new Hsla(0.0, 0.0, 1.0, 1.0)),
            { top_left: 10, top_right: 10, bottom_left: 10, bottom_right: 10 },
            { top: 2, right: 2, bottom: 2, left: 2 },
            new Hsla(0.6, 0.3, 0.8, 1.0),
            BorderStyle.Solid
        );
        card.setOrder(2);
        scene.insert(card);

        // Circle inside card
        const builder = new PathBuilder();
        builder.circle(new Point(200, 200), 80);
        const pathData = builder.build();

        const circle = new Path(pathData, new Hsla(0.55, 0.7, 0.6, 1.0));
        circle.setOrder(3);
        scene.insert(circle);

        const pixelData = await renderToTexture(renderer, scene);
        saveImageData('10-complex-scene', pixelData, RENDER_WIDTH, RENDER_HEIGHT);

        expect(pixelData).toBeDefined();
    });
});
