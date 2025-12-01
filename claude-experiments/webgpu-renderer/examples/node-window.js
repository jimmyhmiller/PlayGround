import sdl from '@kmamal/sdl';
import gpu from '@kmamal/gpu';
import { Scene, Quad, Shadow, Background } from '../src/core/primitives.js';
import { Bounds, Point, Size, Corners, Edges, Hsla } from '../src/core/geometry.js';
import { WebGPURenderer } from '../src/renderer/webgpu-renderer.js';

async function main() {
    console.log('Creating SDL window...');

    // Create SDL window with WebGPU support
    const window = sdl.video.createWindow({
        title: 'WebGPU Renderer - Native Window',
        width: 600,
        height: 900,
        webgpu: true,
    });

    console.log('Initializing WebGPU...');

    // Initialize @kmamal/gpu
    const instance = gpu.create([]);
    const adapter = await instance.requestAdapter();

    // Request device with required features
    const device = await adapter.requestDevice({
        requiredFeatures: [],
    });

    console.log('Creating window renderer...');

    // Connect GPU device to SDL window
    const windowRenderer = gpu.renderGPUDeviceToWindow({
        device,
        window,
    });

    console.log('Creating WebGPU renderer...');

    // Create renderer with external device
    const renderer = new WebGPURenderer(null, device);
    await renderer.initialize();

    // Set canvas size to match window
    renderer.canvasSize = new Size(window.width, window.height);

    console.log('Creating scene (skipping atlas generation for now)...');

    // Create scene
    const scene = new Scene();

    const fullMask = new Bounds(
        new Point(0, 0),
        new Size(window.width, window.height)
    );

    // Gradient quad 1
    const shadow1 = new Shadow();
    shadow1.bounds = new Bounds(new Point(25, 25), new Size(140, 90));
    shadow1.cornerRadii = Corners.uniform(12);
    shadow1.blurRadius = 6;
    shadow1.color = Hsla.black(0.3);
    shadow1.contentMask.bounds = fullMask;
    scene.insertShadow(shadow1);

    const gradQuad1 = new Quad();
    gradQuad1.bounds = new Bounds(new Point(20, 20), new Size(150, 100));
    gradQuad1.background = Background.Solid(Hsla.rgb(1.0, 0.0, 0.0, 1));  // Bright red solid color
    gradQuad1.cornerRadii = Corners.uniform(15);
    gradQuad1.borderWidths = Edges.uniform(3);
    gradQuad1.borderColor = Hsla.rgb(1.0, 1.0, 0.0, 1);  // Yellow border
    gradQuad1.contentMask.bounds = fullMask;
    scene.insertQuad(gradQuad1);

    // Gradient quad 2 (Oklab)
    const shadow2 = new Shadow();
    shadow2.bounds = new Bounds(new Point(195, 25), new Size(140, 90));
    shadow2.cornerRadii = Corners.uniform(15);
    shadow2.blurRadius = 8;
    shadow2.color = Hsla.black(0.25);
    shadow2.contentMask.bounds = fullMask;
    scene.insertShadow(shadow2);

    const gradQuad2 = new Quad();
    gradQuad2.bounds = new Bounds(new Point(190, 20), new Size(150, 100));
    gradQuad2.background = Background.LinearGradient(90, [
        { color: Hsla.rgb(0.95, 0.4, 0.3, 1), position: 0 },
        { color: Hsla.rgb(0.3, 0.9, 0.5, 1), position: 1 }
    ], 1); // Oklab
    gradQuad2.cornerRadii = Corners.uniform(18);
    gradQuad2.borderWidths = new Edges(4, 6, 4, 6);
    gradQuad2.borderColor = Hsla.white(0.9);
    gradQuad2.contentMask.bounds = fullMask;
    scene.insertQuad(gradQuad2);

    // Radial gradient
    const radial1 = new Quad();
    radial1.bounds = new Bounds(new Point(20, 140), new Size(150, 120));
    radial1.background = Background.RadialGradient(
        0.5, 0.5, 0.6,
        [
            { color: Hsla.rgb(1.0, 0.9, 0.3, 1), position: 0 },
            { color: Hsla.rgb(0.9, 0.3, 0.5, 1), position: 1 }
        ],
        0
    );
    radial1.cornerRadii = Corners.uniform(10);
    radial1.contentMask.bounds = fullMask;
    scene.insertQuad(radial1);

    // Conic gradient
    const conic1 = new Quad();
    conic1.bounds = new Bounds(new Point(190, 140), new Size(150, 120));
    conic1.background = Background.ConicGradient(
        0.5, 0.5, 0,
        [
            { color: Hsla.rgb(1.0, 0.3, 0.3, 1), position: 0 },
            { color: Hsla.rgb(0.3, 0.3, 1.0, 1), position: 1 }
        ],
        0
    );
    conic1.cornerRadii = Corners.uniform(10);
    conic1.contentMask.bounds = fullMask;
    scene.insertQuad(conic1);

    // Pattern backgrounds
    const pattern1 = new Quad();
    pattern1.bounds = new Bounds(new Point(20, 280), new Size(100, 80));
    pattern1.background = Background.Pattern(45,
        Hsla.rgb(0.2, 0.4, 0.8, 1),
        Hsla.rgb(0.8, 0.9, 0.95, 1),
        10);
    pattern1.cornerRadii = Corners.uniform(10);
    pattern1.contentMask.bounds = fullMask;
    scene.insertQuad(pattern1);

    const pattern2 = new Quad();
    pattern2.bounds = new Bounds(new Point(140, 280), new Size(100, 80));
    pattern2.background = Background.Pattern(0,
        Hsla.rgb(0.9, 0.3, 0.4, 1),
        Hsla.rgb(1.0, 0.9, 0.9, 1),
        8);
    pattern2.cornerRadii = Corners.uniform(10);
    pattern2.contentMask.bounds = fullMask;
    scene.insertQuad(pattern2);

    console.log('Starting render loop...');

    let frame = 0;
    let running = true;

    // Handle window close
    window.on('close', () => {
        console.log('Window closed');
        running = false;
    });

    // Handle window resize
    window.on('resize', () => {
        console.log('Window resized to', window.width, 'x', window.height);
        renderer.canvasSize = new Size(window.width, window.height);
        windowRenderer.resize();

        // Update full mask for all primitives
        fullMask.size = new Size(window.width, window.height);
    });

    // Render loop
    function render() {
        if (!running || window.destroyed) {
            console.log('Cleaning up...');
            device.destroy();
            gpu.destroy(instance);
            return;
        }

        frame++;

        // Animate gradients
        gradQuad1.background.gradientAngle = 45 + Math.sin(frame * 0.02) * 45;
        gradQuad2.background.gradientAngle = 90 + Math.sin(frame * 0.03) * 30;

        // Get current texture view from window renderer
        const textureView = windowRenderer.getCurrentTextureView();

        // Create command encoder
        const encoder = device.createCommandEncoder();

        // Begin render pass with window's texture view
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1 },  // White background
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        // Render scene using existing renderer
        renderer.renderToPass(renderPass, scene);

        renderPass.end();
        device.queue.submit([encoder.finish()]);

        // Present to window
        windowRenderer.swap();

        // Log every 60 frames
        if (frame % 60 === 0) {
            console.log(`Frame ${frame}`);
        }

        // Continue loop
        setTimeout(render, 0);
    }

    // Start render loop
    render();
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
