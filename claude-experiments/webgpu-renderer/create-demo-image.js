import playwright from 'playwright';

async function createDemoImage() {
    const browser = await playwright.chromium.launch({
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--disable-vulkan-surface']
    });

    const page = await (await browser.newContext()).newPage();

    await page.goto('about:blank');

    const imageData = await page.evaluate(async () => {
        // Import the renderer modules
        const { WebGPURenderer } = await import('/src/renderer/webgpu-renderer.js');
        const { Scene, Quad, Background, Hsla, Bounds, Point, Size, Corners, Edges, Transform } = await import('/src/core/primitives.js');

        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.width = 800;
        canvas.height = 600;
        document.body.appendChild(canvas);

        // Initialize renderer
        const renderer = new WebGPURenderer(canvas);
        await renderer.initialize();

        // Create scene with multiple quads
        const scene = new Scene();

        // Red quad with rounded corners
        const quad1 = new Quad();
        quad1.bounds = new Bounds(new Point(50, 50), new Size(200, 150));
        quad1.background = Background.Solid(new Hsla(0, 1, 0.5, 1)); // Red
        quad1.cornerRadii = new Corners(20, 20, 20, 20);
        scene.insertQuad(quad1);

        // Blue quad with border
        const quad2 = new Quad();
        quad2.bounds = new Bounds(new Point(300, 100), new Size(180, 180));
        quad2.background = Background.Solid(new Hsla(0.6, 1, 0.5, 1)); // Blue
        quad2.borderColor = new Hsla(0, 0, 0, 1);
        quad2.borderWidths = new Edges(5, 5, 5, 5);
        scene.insertQuad(quad2);

        // Green quad with gradient (will render solid for now)
        const quad3 = new Quad();
        quad3.bounds = new Bounds(new Point(550, 50), new Size(200, 120));
        quad3.background = Background.Solid(new Hsla(0.33, 1, 0.5, 1)); // Green
        quad3.cornerRadii = new Corners(10, 10, 10, 10);
        scene.insertQuad(quad3);

        // Purple quad with high corner radius
        const quad4 = new Quad();
        quad4.bounds = new Bounds(new Point(150, 300), new Size(150, 150));
        quad4.background = Background.Solid(new Hsla(0.75, 1, 0.5, 1)); // Purple
        quad4.cornerRadii = new Corners(75, 75, 75, 75);
        scene.insertQuad(quad4);

        // Orange quad rotated (using identity transform for now)
        const quad5 = new Quad();
        quad5.bounds = new Bounds(new Point(400, 350), new Size(300, 100));
        quad5.background = Background.Solid(new Hsla(0.08, 1, 0.5, 1)); // Orange
        quad5.cornerRadii = new Corners(50, 50, 50, 50);
        scene.insertQuad(quad5);

        // Yellow small quad
        const quad6 = new Quad();
        quad6.bounds = new Bounds(new Point(100, 500), new Size(80, 60));
        quad6.background = Background.Solid(new Hsla(0.16, 1, 0.5, 1)); // Yellow
        scene.insertQuad(quad6);

        // Cyan quad
        const quad7 = new Quad();
        quad7.bounds = new Bounds(new Point(600, 400), new Size(120, 120));
        quad7.background = Background.Solid(new Hsla(0.5, 1, 0.5, 1)); // Cyan
        quad7.cornerRadii = new Corners(60, 60, 60, 60);
        scene.insertQuad(quad7);

        // Render the scene
        renderer.render(scene);

        // Get image data
        return canvas.toDataURL('image/png');
    });

    // Convert base64 to buffer and save
    const base64Data = imageData.replace(/^data:image\/png;base64,/, '');
    const buffer = Buffer.from(base64Data, 'base64');

    const fs = await import('fs');
    fs.writeFileSync('/tmp/webgpu-demo.png', buffer);

    console.log('✅ Demo image saved to /tmp/webgpu-demo.png');

    await browser.close();
}

createDemoImage().catch(console.error);
