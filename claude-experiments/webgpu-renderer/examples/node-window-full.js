import sdl from '@kmamal/sdl';
import gpu from '@kmamal/gpu';
import { Scene, Quad, Shadow, Background, Underline, PolychromeSprite } from '../src/core/primitives.js';
import { Bounds, Point, Size, Corners, Edges, Hsla } from '../src/core/geometry.js';
import { WebGPURenderer } from '../src/renderer/webgpu-renderer.js';
import {
    generateGradientPattern,
    generateCheckerboard,
    generateCircularGradient
} from '../src/utils/texture-gen.js';

async function main() {
    console.log('Creating SDL window...');

    // Create SDL window with WebGPU support
    const window = sdl.video.createWindow({
        title: 'WebGPU Renderer - Native Window (Full Scene)',
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

    console.log('Creating full scene (skipping texture atlases due to Dawn limitations)...');

    // Create scene
    const scene = new Scene();

    const fullMask = new Bounds(
        new Point(0, 0),
        new Size(window.width, window.height)
    );

    // === ROW 1: GRADIENTS WITH BORDERS ===

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
    gradQuad1.background = Background.LinearGradient(45, [
        { color: Hsla.rgb(0.2, 0.5, 0.9, 1), position: 0 },
        { color: Hsla.rgb(0.8, 0.3, 0.6, 1), position: 1 }
    ], 0);
    gradQuad1.cornerRadii = Corners.uniform(15);
    gradQuad1.borderWidths = Edges.uniform(3);
    gradQuad1.borderColor = Hsla.rgb(0.1, 0.2, 0.4, 1);
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

    // === ROW 2: SOLID COLOR QUADS (replacing sprites) ===

    const solidQuad1 = new Quad();
    solidQuad1.bounds = new Bounds(new Point(40, 150), new Size(80, 80));
    solidQuad1.background = Background.Solid(Hsla.rgb(0.2, 0.6, 0.9, 1));
    solidQuad1.cornerRadii = Corners.uniform(10);
    solidQuad1.contentMask.bounds = fullMask;
    scene.insertQuad(solidQuad1);

    const solidQuad2 = new Quad();
    solidQuad2.bounds = new Bounds(new Point(140, 150), new Size(80, 80));
    solidQuad2.background = Background.Solid(Hsla.rgb(0.9, 0.6, 0.2, 1));
    solidQuad2.cornerRadii = Corners.uniform(10);
    solidQuad2.contentMask.bounds = fullMask;
    scene.insertQuad(solidQuad2);

    const solidQuad3 = new Quad();
    solidQuad3.bounds = new Bounds(new Point(240, 150), new Size(80, 80));
    solidQuad3.background = Background.Solid(Hsla.rgb(0.3, 0.9, 0.5, 1));
    solidQuad3.cornerRadii = Corners.uniform(10);
    solidQuad3.opacity = 0.5;
    solidQuad3.contentMask.bounds = fullMask;
    scene.insertQuad(solidQuad3);

    // === ROW 3: UNDERLINES ===

    const straightUnderline1 = new Underline();
    straightUnderline1.bounds = new Bounds(new Point(20, 250), new Size(150, 10));
    straightUnderline1.color = Hsla.rgb(0.2, 0.4, 0.8, 1);
    straightUnderline1.thickness = 3;
    straightUnderline1.wavy = 0;
    straightUnderline1.contentMask.bounds = fullMask;
    scene.insertUnderline(straightUnderline1);

    const wavyUnderline1 = new Underline();
    wavyUnderline1.bounds = new Bounds(new Point(190, 250), new Size(150, 20));
    wavyUnderline1.color = Hsla.rgb(0.3, 0.8, 0.5, 1);
    wavyUnderline1.thickness = 4;
    wavyUnderline1.wavy = 1;
    wavyUnderline1.contentMask.bounds = fullMask;
    scene.insertUnderline(wavyUnderline1);

    // === ROW 4: PATTERN BACKGROUNDS ===

    const pattern1 = new Quad();
    pattern1.bounds = new Bounds(new Point(20, 290), new Size(100, 80));
    pattern1.background = Background.Pattern(45,
        Hsla.rgb(0.2, 0.4, 0.8, 1),
        Hsla.rgb(0.8, 0.9, 0.95, 1),
        10);
    pattern1.cornerRadii = Corners.uniform(10);
    pattern1.contentMask.bounds = fullMask;
    scene.insertQuad(pattern1);

    const pattern2 = new Quad();
    pattern2.bounds = new Bounds(new Point(140, 290), new Size(100, 80));
    pattern2.background = Background.Pattern(0,
        Hsla.rgb(0.9, 0.3, 0.4, 1),
        Hsla.rgb(1.0, 0.9, 0.9, 1),
        8);
    pattern2.cornerRadii = Corners.uniform(10);
    pattern2.contentMask.bounds = fullMask;
    scene.insertQuad(pattern2);

    const pattern3 = new Quad();
    pattern3.bounds = new Bounds(new Point(260, 290), new Size(100, 80));
    pattern3.background = Background.Pattern(135,
        Hsla.rgb(0.3, 0.8, 0.5, 1),
        Hsla.rgb(0.9, 0.98, 0.95, 1),
        6,
        0);
    pattern3.cornerRadii = Corners.uniform(10);
    pattern3.borderWidths = Edges.uniform(2);
    pattern3.borderColor = Hsla.rgb(0.2, 0.6, 0.3, 1);
    pattern3.contentMask.bounds = fullMask;
    scene.insertQuad(pattern3);

    // === ROW 5: RADIAL GRADIENTS ===

    const radial1 = new Quad();
    radial1.bounds = new Bounds(new Point(20, 390), new Size(100, 80));
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

    const radial2 = new Quad();
    radial2.bounds = new Bounds(new Point(140, 390), new Size(100, 80));
    radial2.background = Background.RadialGradient(
        0.3, 0.3, 0.8,
        [
            { color: Hsla.rgb(0.3, 0.8, 0.9, 1), position: 0 },
            { color: Hsla.rgb(0.5, 0.3, 0.8, 1), position: 1 }
        ],
        1
    );
    radial2.cornerRadii = Corners.uniform(10);
    radial2.contentMask.bounds = fullMask;
    scene.insertQuad(radial2);

    const radial3 = new Quad();
    radial3.bounds = new Bounds(new Point(260, 390), new Size(100, 80));
    radial3.background = Background.RadialGradient(
        0.5, 0.5, 1.2,
        [
            { color: Hsla.rgb(0.9, 0.4, 0.3, 1), position: 0 },
            { color: Hsla.rgb(0.3, 0.4, 0.9, 1), position: 1 }
        ],
        1
    );
    radial3.cornerRadii = Corners.uniform(10);
    radial3.borderWidths = Edges.uniform(2);
    radial3.borderColor = Hsla.rgb(0.2, 0.2, 0.3, 1);
    radial3.contentMask.bounds = fullMask;
    scene.insertQuad(radial3);

    // === ROW 6: CONIC GRADIENTS ===

    const conic1 = new Quad();
    conic1.bounds = new Bounds(new Point(20, 490), new Size(100, 80));
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

    const conic2 = new Quad();
    conic2.bounds = new Bounds(new Point(140, 490), new Size(100, 80));
    conic2.background = Background.ConicGradient(
        0.5, 0.5, 90,
        [
            { color: Hsla.rgb(0.3, 0.9, 0.3, 1), position: 0 },
            { color: Hsla.rgb(0.9, 0.3, 0.9, 1), position: 1 }
        ],
        1
    );
    conic2.cornerRadii = Corners.uniform(10);
    conic2.borderWidths = Edges.uniform(1);
    conic2.borderColor = Hsla.rgb(0.2, 0.2, 0.2, 1);
    conic2.contentMask.bounds = fullMask;
    scene.insertQuad(conic2);

    const conic3 = new Quad();
    conic3.bounds = new Bounds(new Point(260, 490), new Size(100, 80));
    conic3.background = Background.ConicGradient(
        0.5, 0.5, 45,
        [
            { color: Hsla.rgb(0.9, 0.7, 0.2, 1), position: 0 },
            { color: Hsla.rgb(0.2, 0.7, 0.9, 1), position: 1 }
        ],
        1
    );
    conic3.cornerRadii = Corners.uniform(10);
    conic3.contentMask.bounds = fullMask;
    scene.insertQuad(conic3);

    // === ROW 7: CHECKERBOARD & DOT PATTERNS ===

    const checker1 = new Quad();
    checker1.bounds = new Bounds(new Point(20, 590), new Size(100, 80));
    checker1.background = Background.Pattern(
        0,
        Hsla.rgb(0.2, 0.2, 0.2, 1),
        Hsla.rgb(0.9, 0.9, 0.9, 1),
        20,
        2
    );
    checker1.cornerRadii = Corners.uniform(8);
    checker1.contentMask.bounds = fullMask;
    scene.insertQuad(checker1);

    const dotPattern1 = new Quad();
    dotPattern1.bounds = new Bounds(new Point(140, 590), new Size(100, 80));
    dotPattern1.background = Background.Pattern(0,
        Hsla.rgb(0.95, 0.95, 0.98, 1),
        Hsla.rgb(0.2, 0.4, 0.9, 1),
        12,
        1);
    dotPattern1.cornerRadii = Corners.uniform(5);
    dotPattern1.contentMask.bounds = fullMask;
    scene.insertQuad(dotPattern1);

    const grid1 = new Quad();
    grid1.bounds = new Bounds(new Point(260, 590), new Size(100, 80));
    grid1.background = Background.Pattern(
        0,
        Hsla.rgb(0.95, 0.95, 0.98, 1),
        Hsla.rgb(0.3, 0.4, 0.6, 1),
        15,
        3
    );
    grid1.cornerRadii = Corners.uniform(8);
    grid1.contentMask.bounds = fullMask;
    scene.insertQuad(grid1);

    // === ROW 8: OPACITY DEMONSTRATIONS ===

    const opacity1 = new Quad();
    opacity1.bounds = new Bounds(new Point(20, 690), new Size(100, 80));
    opacity1.background = Background.Solid(Hsla.rgb(0.9, 0.3, 0.3, 1));
    opacity1.cornerRadii = Corners.uniform(10);
    opacity1.opacity = 0.75;
    opacity1.contentMask.bounds = fullMask;
    scene.insertQuad(opacity1);

    const opacity2 = new Quad();
    opacity2.bounds = new Bounds(new Point(140, 690), new Size(100, 80));
    opacity2.background = Background.LinearGradient(
        90,
        [
            { color: Hsla.rgb(0.3, 0.9, 0.3, 1), position: 0 },
            { color: Hsla.rgb(0.3, 0.3, 0.9, 1), position: 1 }
        ],
        1
    );
    opacity2.cornerRadii = Corners.uniform(10);
    opacity2.borderWidths = Edges.uniform(2);
    opacity2.borderColor = Hsla.rgb(0.2, 0.2, 0.2, 1);
    opacity2.opacity = 0.5;
    opacity2.contentMask.bounds = fullMask;
    scene.insertQuad(opacity2);

    const opacity3 = new Quad();
    opacity3.bounds = new Bounds(new Point(260, 690), new Size(100, 80));
    opacity3.background = Background.Pattern(
        0,
        Hsla.rgb(0.2, 0.2, 0.2, 1),
        Hsla.rgb(0.9, 0.9, 0.9, 1),
        15,
        2
    );
    opacity3.cornerRadii = Corners.uniform(10);
    opacity3.opacity = 0.3;
    opacity3.contentMask.bounds = fullMask;
    scene.insertQuad(opacity3);

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
                clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1 },
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
        setTimeout(render, 16);  // ~60fps
    }

    // Start render loop
    render();
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
