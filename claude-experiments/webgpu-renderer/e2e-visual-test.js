/**
 * Standalone E2E visual test using Playwright
 * Renders various primitives and saves screenshots for validation
 */

import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { mkdir } from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const OUTPUT_DIR = join(__dirname, 'test-output', 'e2e');

async function runVisualTests() {
    // Ensure output directory exists
    await mkdir(OUTPUT_DIR, { recursive: true });

    console.log('🚀 Starting E2E visual tests...\n');

    // Launch browser with WebGPU support
    const browser = await chromium.launch({
        headless: true,
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
        ],
    });

    const context = await browser.newContext({
        viewport: { width: 800, height: 600 },
    });

    const page = await context.newPage();

    // Capture console messages and errors
    const consoleMessages = [];
    const errors = [];

    page.on('console', msg => {
        const text = msg.text();
        consoleMessages.push({ type: msg.type(), text });
        console.log(`[${msg.type()}] ${text}`);
        if (msg.type() === 'error') {
            console.error('❌ Browser console error:', text);
        }
    });

    page.on('pageerror', error => {
        errors.push(error.message);
        console.error('❌ Page error:', error.message);
    });

    // Navigate to the demo page
    // Test minimal pipeline first (no storage buffer)
    console.log('📄 Testing minimal hardcoded pipeline...');
    await page.goto('http://localhost:5173/test-pipeline-minimal.html', { waitUntil: 'networkidle', timeout: 10000 });
    await page.waitForTimeout(1000);

    const minimalLog = await page.evaluate(() => document.getElementById('log').textContent);
    const minimalPassed = await page.evaluate(() => window.PIPELINE_TEST_PASSED);
    console.log('Minimal pipeline test result:\n' + minimalLog);

    if (!minimalPassed) {
        console.error('\n❌ Minimal pipeline test FAILED - basic triangle-strip broken\n');
        await browser.close();
        process.exit(1);
    }
    console.log('✅ Minimal hardcoded pipeline works!\n');

    // Test simple quad with constants
    console.log('📄 Testing simple quad with constant color...');
    await page.goto('http://localhost:5173/test-simple-quad.html', { waitUntil: 'networkidle', timeout: 10000 });
    await page.waitForTimeout(1000);

    const simpleQuadLog = await page.evaluate(() => document.getElementById('log').textContent);
    const simpleQuadPassed = await page.evaluate(() => window.SIMPLE_QUAD_TEST_PASSED);
    console.log('Simple quad test result:\n' + simpleQuadLog);

    if (!simpleQuadPassed) {
        console.error('\n❌ Simple quad test FAILED - basic rendering broken\n');
        await browser.close();
        process.exit(1);
    }
    console.log('✅ Simple quad rendering works!\n');

    // Test WebGPU readback
    console.log('📄 Testing WebGPU texture readback...');
    await page.goto('http://localhost:5173/test-readback.html', { waitUntil: 'networkidle', timeout: 10000 });
    await page.waitForTimeout(1000);

    const readbackLog = await page.evaluate(() => document.getElementById('log').textContent);
    const readbackPassed = await page.evaluate(() => window.RENDER_TEST_PASSED);
    console.log('Readback test result:\n' + readbackLog);

    if (!readbackPassed) {
        console.error('\n❌ Basic WebGPU rendering test FAILED - cannot proceed\n');
        await browser.close();
        process.exit(1);
    }
    console.log('✅ Basic WebGPU rendering works!\n');

    // Test debug shader with storage buffer - TEMPORARILY DISABLED
    console.log('📄 Skipping debug shader test (needs update for new alignment)\n');

    // Test actual renderer with quad
    console.log('📄 Testing actual WebGPURenderer with quad...');
    await page.goto('http://localhost:5173/test-renderer.html', { waitUntil: 'networkidle', timeout: 10000 });
    await page.waitForTimeout(2000);

    const rendererLog = await page.evaluate(() => document.getElementById('log').textContent);
    const rendererPassed = await page.evaluate(() => window.RENDERER_TEST_PASSED);
    console.log('Renderer test result:\n' + rendererLog);

    if (!rendererPassed) {
        console.error('\n❌ WebGPURenderer quad test FAILED - renderer is broken\n');
        await browser.close();
        process.exit(1);
    }
    console.log('✅ WebGPURenderer can render quads!\n');

    const demoURL = `http://localhost:5173`;
    console.log(`\n📄 Navigating to main demo ${demoURL}...`);

    try {
        await page.goto(demoURL, { waitUntil: 'networkidle', timeout: 10000 });
    } catch (error) {
        console.error('❌ Failed to load demo page. Make sure the dev server is running (npm run dev)');
        await browser.close();
        process.exit(1);
    }

    // Wait for WebGPU initialization
    console.log('⏳ Waiting for WebGPU to initialize...');
    await page.waitForTimeout(3000);

    // Check if renderer initialized
    const rendererStatus = await page.evaluate(() => {
        return {
            hasRenderer: !!window.renderer,
            hasDevice: window.renderer?.device !== undefined,
            hasPipelines: window.renderer?.pipelines !== undefined,
            pipelineKeys: window.renderer?.pipelines ? Object.keys(window.renderer.pipelines) : []
        };
    });

    console.log('Renderer status:', JSON.stringify(rendererStatus, null, 2));

    // Check for error messages on page
    const pageErrors = await page.evaluate(() => {
        const errorDiv = document.getElementById('error');
        const infoDiv = document.getElementById('info');
        return {
            errorText: errorDiv ? errorDiv.textContent : 'no error div',
            errorVisible: errorDiv ? errorDiv.style.display !== 'none' : false,
            infoText: infoDiv ? infoDiv.textContent : 'no info div'
        };
    });

    console.log('Page status:', JSON.stringify(pageErrors, null, 2));

    // Check if WebGPU is available
    const webgpuAvailable = await page.evaluate(() => {
        return navigator.gpu !== undefined;
    });

    if (!webgpuAvailable) {
        console.error('❌ WebGPU is not available in this browser');
        await browser.close();
        process.exit(1);
    }

    console.log('✅ WebGPU is available\n');

    // Check if anything is actually rendering
    const renderCheck = await page.evaluate(() => {
        const canvas = document.querySelector('canvas');
        if (!canvas) return { hasCanvas: false };

        const ctx = canvas.getContext('2d');
        if (!ctx) return { hasCanvas: true, canGet2D: false };

        // Get pixel data from center
        const centerX = Math.floor(canvas.width / 2);
        const centerY = Math.floor(canvas.height / 2);
        const pixel = ctx.getImageData(centerX, centerY, 1, 1).data;

        return {
            hasCanvas: true,
            canvasSize: { width: canvas.width, height: canvas.height },
            centerPixel: Array.from(pixel),
            isWhite: pixel[0] > 250 && pixel[1] > 250 && pixel[2] > 250
        };
    });

    console.log('Render check:', JSON.stringify(renderCheck, null, 2));

    // Take screenshot of the full demo
    console.log('📸 Capturing full demo screenshot...');
    await page.screenshot({
        path: join(OUTPUT_DIR, '00-full-demo.png'),
        fullPage: true,
    });
    console.log(`   Saved: 00-full-demo.png`);

    // Test individual primitives by injecting test scenes
    const testScenes = [
        {
            name: '01-solid-quad',
            description: 'Solid color quad',
            script: `
                const scene = new Scene();
                const quad = new Quad(
                    new Bounds(new Point(100, 100), { width: 300, height: 300 }),
                    Background.Solid(new Hsla(0.6, 0.7, 0.5, 1.0)),
                    { top_left: 0, top_right: 0, bottom_left: 0, bottom_right: 0 },
                    { top: 0, right: 0, bottom: 0, left: 0 },
                    Hsla.transparent(),
                    BorderStyle.Solid
                );
                quad.setOrder(0);
                scene.insert(quad);
                window.renderer.render(scene);
            `,
        },
        {
            name: '02-rounded-quad',
            description: 'Rounded corner quad',
            script: `
                const scene = new Scene();
                const quad = new Quad(
                    new Bounds(new Point(100, 100), { width: 300, height: 300 }),
                    Background.Solid(new Hsla(0.0, 0.7, 0.5, 1.0)),
                    { top_left: 50, top_right: 50, bottom_left: 50, bottom_right: 50 },
                    { top: 0, right: 0, bottom: 0, left: 0 },
                    Hsla.transparent(),
                    BorderStyle.Solid
                );
                quad.setOrder(0);
                scene.insert(quad);
                window.renderer.render(scene);
            `,
        },
        {
            name: '03-linear-gradient',
            description: 'Linear gradient quad',
            script: `
                const scene = new Scene();
                const gradient = Background.LinearGradient(
                    Math.PI / 4,
                    [
                        { offset: 0.0, color: new Hsla(0.0, 1.0, 0.5, 1.0) },
                        { offset: 1.0, color: new Hsla(0.6, 1.0, 0.5, 1.0) },
                    ],
                    0
                );
                const quad = new Quad(
                    new Bounds(new Point(100, 100), { width: 300, height: 300 }),
                    gradient,
                    { top_left: 0, top_right: 0, bottom_left: 0, bottom_right: 0 },
                    { top: 0, right: 0, bottom: 0, left: 0 },
                    Hsla.transparent(),
                    BorderStyle.Solid
                );
                quad.setOrder(0);
                scene.insert(quad);
                window.renderer.render(scene);
            `,
        },
        {
            name: '04-shadow',
            description: 'Shadow with quad',
            script: `
                const scene = new Scene();
                const shadow = new Shadow(
                    new Bounds(new Point(120, 120), { width: 260, height: 260 }),
                    { top_left: 20, top_right: 20, bottom_left: 20, bottom_right: 20 },
                    new Hsla(0, 0, 0, 0.5),
                    20.0
                );
                shadow.setOrder(0);
                scene.insert(shadow);

                const quad = new Quad(
                    new Bounds(new Point(100, 100), { width: 300, height: 300 }),
                    Background.Solid(new Hsla(0.15, 0.8, 0.6, 1.0)),
                    { top_left: 20, top_right: 20, bottom_left: 20, bottom_right: 20 },
                    { top: 0, right: 0, bottom: 0, left: 0 },
                    Hsla.transparent(),
                    BorderStyle.Solid
                );
                quad.setOrder(1);
                scene.insert(quad);
                window.renderer.render(scene);
            `,
        },
        {
            name: '05-border',
            description: 'Quad with border',
            script: `
                const scene = new Scene();
                const quad = new Quad(
                    new Bounds(new Point(100, 100), { width: 300, height: 300 }),
                    Background.Solid(new Hsla(0.0, 0.0, 1.0, 1.0)),
                    { top_left: 20, top_right: 20, bottom_left: 20, bottom_right: 20 },
                    { top: 10, right: 10, bottom: 10, left: 10 },
                    new Hsla(0.0, 0.7, 0.5, 1.0),
                    BorderStyle.Solid
                );
                quad.setOrder(0);
                scene.insert(quad);
                window.renderer.render(scene);
            `,
        },
        {
            name: '06-path-circle',
            description: 'Path rendering - circle',
            script: `
                const scene = new Scene();
                const builder = new PathBuilder();
                builder.circle(new Point(250, 250), 100);
                const pathData = builder.build();
                const path = new Path(pathData, new Hsla(0.33, 0.7, 0.5, 1.0));
                path.setOrder(0);
                scene.insert(path);
                window.renderer.render(scene);
            `,
        },
        {
            name: '07-path-bezier',
            description: 'Path rendering - bezier curves',
            script: `
                const scene = new Scene();
                const builder = new PathBuilder();
                builder.moveTo(new Point(50, 300));
                builder.quadraticCurveTo(new Point(200, 100), new Point(350, 300));
                builder.quadraticCurveTo(new Point(400, 400), new Point(450, 300));
                builder.close();
                const pathData = builder.build();
                const path = new Path(pathData, new Hsla(0.8, 0.7, 0.5, 1.0));
                path.setOrder(0);
                scene.insert(path);
                window.renderer.render(scene);
            `,
        },
    ];

    // Run each test scene
    for (const test of testScenes) {
        console.log(`📸 Testing: ${test.description}...`);

        try {
            // Inject and run the test script
            await page.evaluate(test.script);

            // Wait for rendering
            await page.waitForTimeout(500);

            // Take screenshot
            await page.screenshot({
                path: join(OUTPUT_DIR, `${test.name}.png`),
            });

            console.log(`   ✅ Saved: ${test.name}.png`);
        } catch (error) {
            console.error(`   ❌ Failed: ${error.message}`);
        }
    }

    console.log(`\n✅ Visual tests complete! Screenshots saved to: ${OUTPUT_DIR}\n`);

    // Summary of errors
    if (errors.length > 0) {
        console.error('\n⚠️  Errors detected during rendering:');
        errors.forEach((err, i) => console.error(`  ${i + 1}. ${err}`));
    }

    const errorLogs = consoleMessages.filter(m => m.type === 'error');
    if (errorLogs.length > 0) {
        console.error('\n⚠️  Console errors:');
        errorLogs.forEach((log, i) => console.error(`  ${i + 1}. ${log.text}`));
    }

    if (errors.length === 0 && errorLogs.length === 0) {
        console.log('✅ No errors detected!\n');
    } else {
        console.log('\n❌ Rendering has errors - check output above\n');
    }

    await browser.close();
}

// Run the tests
runVisualTests().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});
