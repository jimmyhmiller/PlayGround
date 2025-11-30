import commonShader from '../shaders/common.wgsl?raw';
import quadShader from '../shaders/quad.wgsl?raw';
import quadHardcodedShader from '../shaders/quad-hardcoded.wgsl?raw';
import shadowShader from '../shaders/shadow.wgsl?raw';
import underlineShader from '../shaders/underline.wgsl?raw';
import spriteShader from '../shaders/sprite.wgsl?raw';
import pathShader from '../shaders/path.wgsl?raw';
import surfaceShader from '../shaders/surface.wgsl?raw';
import pathRasterizationShader from '../shaders/path-rasterization.wgsl?raw';
import pathCompositeShader from '../shaders/path-composite.wgsl?raw';
import { Atlas } from '../core/atlas.js';
import { BufferPool } from '../utils/buffer-pool.js';
import { getWebGPU, getCanvasContext, getPreferredCanvasFormat, isNode } from '../platform/webgpu-platform.js';
import { PathTessellator } from '../utils/path-tessellator.js';

export class WebGPURenderer {
    constructor(canvas, externalDevice = null, options = {}) {
        this.canvas = canvas;
        this.externalDevice = externalDevice;
        this.device = null;
        this.context = null;
        this.pipelines = {};
        this.bindGroupLayouts = {};
        this.uniformBuffer = null;
        this.monochromeAtlas = null;
        this.polychromeAtlas = null;
        this.bufferPool = null;
        this.pathTessellator = null;
        this.ready = false;
        this.presentationFormat = null;
        this.canvasSize = null; // For external rendering without canvas

        // Path rendering settings (matches GPUI)
        this.pathSampleCount = options.pathSampleCount || 4; // MSAA sample count (4, 2, or 1)
        this.pathIntermediateTexture = null;
        this.pathIntermediateTextureView = null;
        this.pathIntermediateMSAATexture = null;
        this.pathIntermediateMSAATextureView = null;
    }

    async initialize() {
        // Use external device if provided, otherwise create one
        if (this.externalDevice) {
            this.device = this.externalDevice;
        } else {
            // Get WebGPU (browser or Node.js)
            const gpu = await getWebGPU();

            // Request adapter
            const adapter = await gpu.requestAdapter();
            if (!adapter) {
                throw new Error('Failed to get WebGPU adapter');
            }

            // Request device
            this.device = await adapter.requestDevice();
        }

        // Check supported MSAA sample counts and adjust if needed
        const supportedCounts = [4, 2, 1];
        for (const count of supportedCounts) {
            // WebGPU doesn't have a direct query for sample count support
            // but 4x MSAA is widely supported. We'll try to create textures and handle errors.
            if (count <= this.pathSampleCount) {
                this.pathSampleCount = count;
                break;
            }
        }

        // Get preferred format
        const gpu = await getWebGPU();
        this.presentationFormat = await getPreferredCanvasFormat(gpu);

        // Configure canvas context (if canvas exists)
        if (this.canvas) {
            this.context = getCanvasContext(this.canvas);
            this.context.configure({
                device: this.device,
                format: this.presentationFormat,
                alphaMode: 'premultiplied',
            });
        }

        // Create atlases
        this.monochromeAtlas = new Atlas(this.device, 'monochrome', 1024);
        this.polychromeAtlas = new Atlas(this.device, 'polychrome', 1024);

        // Create buffer pool
        this.bufferPool = new BufferPool(this.device);

        // Create path tessellator
        this.pathTessellator = new PathTessellator(0.5);

        // Create bind group layouts
        this.bindGroupLayouts.main = this.device.createBindGroupLayout({
            label: 'main bind group layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'read-only-storage' }
                }
            ]
        });

        this.bindGroupLayouts.sprite = this.device.createBindGroupLayout({
            label: 'sprite bind group layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    sampler: {}
                }
            ]
        });

        this.bindGroupLayouts.path = this.device.createBindGroupLayout({
            label: 'path bind group layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                }
            ]
        });

        this.bindGroupLayouts.surface = this.device.createBindGroupLayout({
            label: 'surface bind group layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {}
                }
            ]
        });

        this.bindGroupLayouts.pathRasterization = this.device.createBindGroupLayout({
            label: 'path rasterization bind group layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'read-only-storage' }
                }
            ]
        });

        this.bindGroupLayouts.pathComposite = this.device.createBindGroupLayout({
            label: 'path composite bind group layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {}
                }
            ]
        });

        // Create pipelines
        await this.createQuadPipeline();
        await this.createShadowPipeline();
        await this.createUnderlinePipeline();
        await this.createMonoSpritePipeline();
        await this.createPolySpritePipeline();
        await this.createPathPipeline();
        await this.createSurfacePipeline();
        await this.createPathRasterizationPipeline();
        await this.createPathCompositePipeline();

        // Create uniform buffers
        this.uniformBuffer = this.device.createBuffer({
            label: 'globals uniform buffer',
            size: 16, // vec2<f32> viewport_size (8) + u32 premultiplied_alpha (4) + u32 pad (4) = 16 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.gammaRatiosBuffer = this.device.createBuffer({
            label: 'gamma ratios uniform buffer',
            size: 16, // vec4<f32> = 16 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.enhancedContrastBuffer = this.device.createBuffer({
            label: 'enhanced contrast uniform buffer',
            size: 4, // f32 = 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Initialize gamma ratios (gamma = 1.8)
        const gammaData = new Float32Array([0.036725, -0.222775, 0.3661, -0.08085]);
        this.device.queue.writeBuffer(this.gammaRatiosBuffer, 0, gammaData);

        // Initialize enhanced contrast
        const contrastData = new Float32Array([1.0]);
        this.device.queue.writeBuffer(this.enhancedContrastBuffer, 0, contrastData);

        this.ready = true;
    }

    createBlendState() {
        return {
            color: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
            },
            alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
            },
        };
    }

    async createQuadPipeline() {
        // Use real quad shader with storage buffer access
        const shaderCode = commonShader + '\n' + quadShader;
        console.log('Using quad shader with storage buffer (62 floats, corrected alignment)');

        // DEBUG: Check if shader contains the hardcoded red return
        const fsMainIndex = shaderCode.indexOf('@fragment');
        const fsMainSnippet = shaderCode.substring(fsMainIndex, fsMainIndex + 200);
        console.log('Quad fragment shader snippet:', fsMainSnippet);

        const shaderModule = this.device.createShaderModule({
            label: 'quad shader',
            code: shaderCode,
        });

        // Check for shader compilation errors
        const compilationInfo = await shaderModule.getCompilationInfo();
        if (compilationInfo.messages.length > 0) {
            console.log('Quad shader compilation messages:');
            compilationInfo.messages.forEach(msg => {
                console.log(`  ${msg.type}: ${msg.message} at line ${msg.lineNum}`);
            });
        }

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'quad pipeline layout',
            bindGroupLayouts: [this.bindGroupLayouts.main]
        });

        this.pipelines.quad = this.device.createRenderPipeline({
            label: 'quad pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.presentationFormat,
                    blend: this.createBlendState(),
                }],
            },
            primitive: {
                topology: 'triangle-strip',
                // Don't set stripIndexFormat for non-indexed draws
            },
        });
        console.log('✓ Quad pipeline created successfully');
    }

    async createShadowPipeline() {
        const shaderCode = commonShader + '\n' + shadowShader;
        const shaderModule = this.device.createShaderModule({
            label: 'shadow shader',
            code: shaderCode,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'shadow pipeline layout',
            bindGroupLayouts: [this.bindGroupLayouts.main]
        });

        this.pipelines.shadow = this.device.createRenderPipeline({
            label: 'shadow pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.presentationFormat,
                    blend: this.createBlendState(),
                }],
            },
            primitive: {
                topology: 'triangle-strip',
                stripIndexFormat: 'uint32',
            },
        });
    }

    async createUnderlinePipeline() {
        const shaderCode = commonShader + '\n' + underlineShader;
        const shaderModule = this.device.createShaderModule({
            label: 'underline shader',
            code: shaderCode,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'underline pipeline layout',
            bindGroupLayouts: [this.bindGroupLayouts.main]
        });

        this.pipelines.underline = this.device.createRenderPipeline({
            label: 'underline pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.presentationFormat,
                    blend: this.createBlendState(),
                }],
            },
            primitive: {
                topology: 'triangle-strip',
                stripIndexFormat: 'uint32',
            },
        });
    }

    async createMonoSpritePipeline() {
        const shaderCode = commonShader + '\n' + spriteShader;
        const shaderModule = this.device.createShaderModule({
            label: 'mono sprite shader',
            code: shaderCode,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'mono sprite pipeline layout',
            bindGroupLayouts: [this.bindGroupLayouts.sprite]
        });

        this.pipelines.monoSprite = this.device.createRenderPipeline({
            label: 'mono sprite pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_mono',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_mono',
                targets: [{
                    format: this.presentationFormat,
                    blend: this.createBlendState(),
                }],
            },
            primitive: {
                topology: 'triangle-strip',
                stripIndexFormat: 'uint32',
            },
        });
    }

    async createPolySpritePipeline() {
        const shaderCode = commonShader + '\n' + spriteShader;
        const shaderModule = this.device.createShaderModule({
            label: 'poly sprite shader',
            code: shaderCode,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'poly sprite pipeline layout',
            bindGroupLayouts: [this.bindGroupLayouts.sprite]
        });

        this.pipelines.polySprite = this.device.createRenderPipeline({
            label: 'poly sprite pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_poly',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_poly',
                targets: [{
                    format: this.presentationFormat,
                    blend: this.createBlendState(),
                }],
            },
            primitive: {
                topology: 'triangle-strip',
                stripIndexFormat: 'uint32',
            },
        });
    }

    async createPathPipeline() {
        const shaderCode = commonShader + '\n' + pathShader;
        const shaderModule = this.device.createShaderModule({
            label: 'path shader',
            code: shaderCode,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'path pipeline layout',
            bindGroupLayouts: [this.bindGroupLayouts.path]
        });

        this.pipelines.path = this.device.createRenderPipeline({
            label: 'path pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 8,
                    attributes: [{
                        shaderLocation: 0,
                        offset: 0,
                        format: 'float32x2'
                    }]
                }]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.presentationFormat,
                    blend: this.createBlendState(),
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });
    }

    async createSurfacePipeline() {
        const shaderCode = commonShader + '\n' + surfaceShader;
        const shaderModule = this.device.createShaderModule({
            label: 'surface shader',
            code: shaderCode,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'surface pipeline layout',
            bindGroupLayouts: [this.bindGroupLayouts.surface]
        });

        this.pipelines.surface = this.device.createRenderPipeline({
            label: 'surface pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.presentationFormat,
                    blend: this.createBlendState(),
                }],
            },
            primitive: {
                topology: 'triangle-strip',
                stripIndexFormat: 'uint32',
            },
        });
    }

    async createPathRasterizationPipeline() {
        const shaderCode = commonShader + '\n' + pathRasterizationShader;
        const shaderModule = this.device.createShaderModule({
            label: 'path rasterization shader',
            code: shaderCode,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'path rasterization pipeline layout',
            bindGroupLayouts: [this.bindGroupLayouts.pathRasterization]
        });

        this.pipelines.pathRasterization = this.device.createRenderPipeline({
            label: 'path rasterization pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_path_rasterization',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_path_rasterization',
                targets: [{
                    format: this.presentationFormat,
                    blend: this.createBlendState(),
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
            multisample: {
                count: this.pathSampleCount,
            },
        });
    }

    async createPathCompositePipeline() {
        const shaderCode = commonShader + '\n' + pathCompositeShader;
        const shaderModule = this.device.createShaderModule({
            label: 'path composite shader',
            code: shaderCode,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'path composite pipeline layout',
            bindGroupLayouts: [this.bindGroupLayouts.pathComposite]
        });

        this.pipelines.pathComposite = this.device.createRenderPipeline({
            label: 'path composite pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.presentationFormat,
                    blend: this.createBlendState(),
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });
    }

    createPathIntermediateTextures(width, height) {
        // Destroy old textures if they exist
        if (this.pathIntermediateTexture) {
            this.pathIntermediateTexture.destroy();
        }
        if (this.pathIntermediateMSAATexture) {
            this.pathIntermediateMSAATexture.destroy();
        }

        // Create resolve target (non-MSAA)
        this.pathIntermediateTexture = this.device.createTexture({
            label: 'path intermediate texture',
            size: { width, height, depthOrArrayLayers: 1 },
            format: this.presentationFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            sampleCount: 1,
        });
        this.pathIntermediateTextureView = this.pathIntermediateTexture.createView();

        // Create MSAA texture if sample count > 1
        if (this.pathSampleCount > 1) {
            this.pathIntermediateMSAATexture = this.device.createTexture({
                label: 'path intermediate MSAA texture',
                size: { width, height, depthOrArrayLayers: 1 },
                format: this.presentationFormat,
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
                sampleCount: this.pathSampleCount,
            });
            this.pathIntermediateMSAATextureView = this.pathIntermediateMSAATexture.createView();
        }
    }

    render(scene, options = {}) {
        if (!this.ready) {
            console.warn('Renderer not ready');
            return;
        }

        // Release all buffers from previous frame for reuse
        this.bufferPool.releaseAll();

        scene.finish();

        const encoder = this.device.createCommandEncoder({
            label: 'render encoder'
        });

        // Determine render target dimensions
        const width = options.width || (this.canvas ? this.canvas.width : 800);
        const height = options.height || (this.canvas ? this.canvas.height : 600);

        // Create/recreate intermediate textures if needed
        if (!this.pathIntermediateTexture ||
            this.pathIntermediateTexture.width !== width ||
            this.pathIntermediateTexture.height !== height) {
            this.createPathIntermediateTextures(width, height);
        }

        // Update globals
        const globalsData = new Float32Array([
            width,
            height,
            1, // premultiplied_alpha
            0, // pad
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, globalsData);

        // Collect all batches and separate paths
        const batches = Array.from(scene.batches());
        const pathBatches = batches.filter(b => b.type === 'paths');
        const nonPathBatches = batches.filter(b => b.type !== 'paths');

        // PASS 1: Render paths to intermediate texture with MSAA (matches GPUI)
        if (pathBatches.length > 0) {
            const pathRenderPass = encoder.beginRenderPass({
                label: 'path rasterization pass',
                colorAttachments: [{
                    view: this.pathSampleCount > 1 ? this.pathIntermediateMSAATextureView : this.pathIntermediateTextureView,
                    resolveTarget: this.pathSampleCount > 1 ? this.pathIntermediateTextureView : undefined,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: this.pathSampleCount > 1 ? 'discard' : 'store',
                }],
            });

            // Render all path batches with Loop-Blinn
            for (const batch of pathBatches) {
                this.renderPathsLoopBlinn(pathRenderPass, batch.primitives);
            }

            pathRenderPass.end();
        }

        // Get texture view (canvas or offscreen)
        let textureView;
        let offscreenTexture = null;

        if (this.context) {
            // Render to canvas
            textureView = this.context.getCurrentTexture().createView();
        } else {
            // Render to offscreen texture (for Node.js)
            offscreenTexture = this.device.createTexture({
                size: { width, height },
                format: this.presentationFormat,
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
            });
            textureView = offscreenTexture.createView();
        }

        // PASS 2: Main render pass - render all non-path primitives + composite paths
        const renderPass = encoder.beginRenderPass({
            label: 'main render pass',
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 1, g: 1, b: 1, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        // Render non-path batches in order
        for (const batch of nonPathBatches) {
            switch (batch.type) {
                case 'shadows':
                    this.renderShadows(renderPass, batch.primitives);
                    break;
                case 'quads':
                    this.renderQuads(renderPass, batch.primitives);
                    break;
                case 'underlines':
                    this.renderUnderlines(renderPass, batch.primitives);
                    break;
                case 'monochromeSprites':
                    this.renderMonochromeSprites(renderPass, batch.primitives);
                    break;
                case 'polychromeSprites':
                    this.renderPolychromeSprites(renderPass, batch.primitives);
                    break;
                case 'surfaces':
                    this.renderSurfaces(renderPass, batch.primitives);
                    break;
            }
        }

        // Composite paths from intermediate texture (if any were rendered)
        if (pathBatches.length > 0) {
            this.renderPathComposite(renderPass);
        }

        renderPass.end();

        // Submit
        this.device.queue.submit([encoder.finish()]);

        // Return offscreen texture for headless rendering
        return offscreenTexture;
    }

    renderToPass(renderPass, scene) {
        if (!this.ready) {
            console.warn('Renderer not ready');
            return;
        }

        // Release all buffers from previous frame for reuse
        this.bufferPool.releaseAll();

        scene.finish();

        // Determine render target dimensions
        const width = this.canvasSize ? this.canvasSize.width : (this.canvas ? this.canvas.width : 800);
        const height = this.canvasSize ? this.canvasSize.height : (this.canvas ? this.canvas.height : 600);

        // Update globals
        const globalsData = new Float32Array([
            width,
            height,
            1, // premultiplied_alpha
            0, // pad
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, globalsData);

        // Collect all batches
        const batches = Array.from(scene.batches());

        // Render all batches in order
        for (const batch of batches) {
            switch (batch.type) {
                case 'shadows':
                    this.renderShadows(renderPass, batch.primitives);
                    break;
                case 'quads':
                    this.renderQuads(renderPass, batch.primitives);
                    break;
                case 'underlines':
                    this.renderUnderlines(renderPass, batch.primitives);
                    break;
                case 'monochromeSprites':
                    this.renderMonochromeSprites(renderPass, batch.primitives);
                    break;
                case 'polychromeSprites':
                    this.renderPolychromeSprites(renderPass, batch.primitives);
                    break;
                case 'surfaces':
                    this.renderSurfaces(renderPass, batch.primitives);
                    break;
                case 'paths':
                    // Note: Path rendering requires two-pass approach, so we skip it here
                    // TODO: Support paths in renderToPass by creating intermediate textures
                    console.warn('Path rendering not yet supported in renderToPass');
                    break;
            }
        }
    }

    createPrimitiveBuffer(primitives, floatsPerPrimitive) {
        const data = new Float32Array(primitives.length * floatsPerPrimitive);
        let offset = 0;

        for (const primitive of primitives) {
            const arr = primitive.toArray();
            data.set(arr, offset);
            offset += arr.length;
        }

        // Acquire buffer from pool
        const buffer = this.bufferPool.acquire(
            data.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );

        this.device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }

    renderShadows(renderPass, shadows) {
        if (shadows.length === 0) return;

        const buffer = this.createPrimitiveBuffer(shadows, 34); // was 32, added 2 for opacity

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts.main,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer } }
            ]
        });

        renderPass.setPipeline(this.pipelines.shadow);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(4, shadows.length, 0, 0);
    }

    renderQuads(renderPass, quads) {
        if (quads.length === 0) return;

        console.log(`renderQuads: ${quads.length} quads`);
        const quadData = quads[0].toArray();
        console.log(`  First quad data length:`, quadData.length);
        console.log(`  First quad data (first 20 floats):`, quadData.slice(0, 20));
        console.log(`  First quad bounds (floats 2-5):`, quadData.slice(2, 6));
        console.log(`  Expected bounds:`, quads[0].bounds.toArray());

        if (quads.length > 1) {
            const quad2Data = quads[1].toArray();
            console.log(`  Second quad (gradient) data length:`, quad2Data.length);
            console.log(`  Second quad background section (floats 12-40):`, quad2Data.slice(12, 40));
            console.log(`  Second quad tag (float 12):`, quad2Data[12], '(should be 1 for gradient)');
            console.log(`  Second quad colorSpace (float 13):`, quad2Data[13]);
            console.log(`  Second quad solid color (floats 14-17):`, quad2Data.slice(14, 18));
            console.log(`  Second quad gradientAngle (float 18):`, quad2Data[18]);
            console.log(`  Second quad color stop 0 (floats 22-26):`, quad2Data.slice(22, 27));
            console.log(`  Second quad color stop 1 (floats 30-34):`, quad2Data.slice(30, 35));
        }

        const buffer = this.createPrimitiveBuffer(quads, 64); // 64 floats = 256 bytes (32-byte aligned)

        console.log(`  Buffer size: ${buffer.size} bytes (expected ${64 * 4} bytes)`);
        console.log(`  Pipeline:`, this.pipelines.quad);
        console.log(`  Uniform buffer:`, this.uniformBuffer);

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts.main,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer } }
            ]
        });

        renderPass.setPipeline(this.pipelines.quad);
        renderPass.setBindGroup(0, bindGroup);
        console.log(`  Calling draw(4, ${quads.length}, 0, 0)`);
        renderPass.draw(4, quads.length, 0, 0);
    }

    renderUnderlines(renderPass, underlines) {
        if (underlines.length === 0) return;

        const buffer = this.createPrimitiveBuffer(underlines, 30); // was 28, added 2 for opacity

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts.main,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer } }
            ]
        });

        renderPass.setPipeline(this.pipelines.underline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(4, underlines.length, 0, 0);
    }

    renderMonochromeSprites(renderPass, sprites) {
        if (sprites.length === 0 || this.monochromeAtlas.textures.length === 0) return;

        const buffer = this.createPrimitiveBuffer(sprites, 36); // was 28, added 8 for transform

        // Use first atlas texture (simple implementation - should batch by texture)
        const atlasTexture = this.monochromeAtlas.textures[0];

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts.sprite,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.gammaRatiosBuffer } },
                { binding: 2, resource: { buffer: this.enhancedContrastBuffer } },
                { binding: 3, resource: { buffer } },
                { binding: 4, resource: atlasTexture.view },
                { binding: 5, resource: this.monochromeAtlas.getSampler() }
            ]
        });

        renderPass.setPipeline(this.pipelines.monoSprite);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(4, sprites.length, 0, 0);
    }

    renderPolychromeSprites(renderPass, sprites) {
        if (sprites.length === 0 || this.polychromeAtlas.textures.length === 0) return;

        const buffer = this.createPrimitiveBuffer(sprites, 34); // was 26, added 8 for transform

        // Use first atlas texture
        const atlasTexture = this.polychromeAtlas.textures[0];

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts.sprite,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.gammaRatiosBuffer } },
                { binding: 2, resource: { buffer: this.enhancedContrastBuffer } },
                { binding: 3, resource: { buffer } },
                { binding: 4, resource: atlasTexture.view },
                { binding: 5, resource: this.polychromeAtlas.getSampler() }
            ]
        });

        renderPass.setPipeline(this.pipelines.polySprite);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(4, sprites.length, 0, 0);
    }

    renderPaths(renderPass, paths) {
        if (paths.length === 0) return;

        renderPass.setPipeline(this.pipelines.path);

        for (const path of paths) {
            if (!path.segments || path.segments.length === 0) continue;

            // Tessellate path segments into polyline
            const points = this.pathTessellator.tessellate(path.segments);
            if (points.length < 2) continue;

            let vertices = [];
            let indices = [];

            if (path.filled) {
                // Generate fill mesh
                indices = this.pathTessellator.triangulate(points);
                vertices = points;
            } else if (path.stroked) {
                // Generate stroke mesh
                const strokeMesh = this.pathTessellator.generateStroke(
                    points,
                    path.strokeWidth,
                    false
                );
                vertices = strokeMesh.vertices;
                indices = strokeMesh.indices;
            }

            if (vertices.length === 0 || indices.length === 0) continue;

            // Create vertex buffer
            const vertexData = new Float32Array(vertices.length * 2);
            for (let i = 0; i < vertices.length; i++) {
                vertexData[i * 2] = vertices[i].x;
                vertexData[i * 2 + 1] = vertices[i].y;
            }

            const vertexBuffer = this.bufferPool.acquire(
                vertexData.byteLength,
                GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            );
            this.device.queue.writeBuffer(vertexBuffer, 0, vertexData);

            // Create index buffer
            const indexData = new Uint32Array(indices);
            const indexBuffer = this.bufferPool.acquire(
                indexData.byteLength,
                GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
            );
            this.device.queue.writeBuffer(indexBuffer, 0, indexData);

            // Create path uniforms buffer
            const transformArray = path.transform.toArray();

            // Build uniform data with proper layout
            // mat3x3 in std140: each column is vec3 (3 floats + 1 padding) = 16 bytes per column
            const uniformData = new Float32Array(24); // 96 bytes total
            let offset = 0;

            // Column 0 of transform matrix
            uniformData[offset++] = transformArray[0]; // a
            uniformData[offset++] = transformArray[2]; // c
            uniformData[offset++] = transformArray[4]; // tx
            offset++; // padding

            // Column 1 of transform matrix
            uniformData[offset++] = transformArray[1]; // b
            uniformData[offset++] = transformArray[3]; // d
            uniformData[offset++] = transformArray[5]; // ty
            offset++; // padding

            // Column 2 of transform matrix (identity for 2D)
            uniformData[offset++] = 0;
            uniformData[offset++] = 0;
            uniformData[offset++] = 1;
            offset++; // padding

            // fill_color - vec4<f32>
            const fillColor = path.fillColor.toArray();
            uniformData[offset++] = fillColor[0];
            uniformData[offset++] = fillColor[1];
            uniformData[offset++] = fillColor[2];
            uniformData[offset++] = fillColor[3];

            // stroke_color - vec4<f32>
            const strokeColor = path.strokeColor.toArray();
            uniformData[offset++] = strokeColor[0];
            uniformData[offset++] = strokeColor[1];
            uniformData[offset++] = strokeColor[2];
            uniformData[offset++] = strokeColor[3];

            // opacity (f32), filled (u32), stroked (u32), pad (u32)
            uniformData[offset++] = path.opacity;
            uniformData[offset++] = path.filled ? 1.0 : 0.0;
            uniformData[offset++] = path.stroked ? 1.0 : 0.0;
            uniformData[offset++] = 0;

            const pathUniformBuffer = this.bufferPool.acquire(
                uniformData.byteLength,
                GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            );
            this.device.queue.writeBuffer(pathUniformBuffer, 0, uniformData);

            // Create bind group
            const bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayouts.path,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: { buffer: pathUniformBuffer } }
                ]
            });

            // Draw
            renderPass.setBindGroup(0, bindGroup);
            renderPass.setVertexBuffer(0, vertexBuffer);
            renderPass.setIndexBuffer(indexBuffer, 'uint32');
            renderPass.drawIndexed(indices.length, 1, 0, 0, 0);
        }
    }

    renderPathsLoopBlinn(renderPass, paths) {
        if (paths.length === 0) return;

        renderPass.setPipeline(this.pipelines.pathRasterization);

        for (const path of paths) {
            if (!path.segments || path.segments.length === 0) continue;

            // Generate Loop-Blinn vertices
            const loopBlinnVertices = this.pathTessellator.generateLoopBlinnVertices(path.segments);
            if (loopBlinnVertices.length === 0) continue;

            // Pack vertices into buffer format
            // Each vertex: xy_position (vec2), st_position (vec2), fill_color (vec4), stroke_color (vec4), stroke_width (f32), bounds (vec4)
            const floatsPerVertex = 2 + 2 + 4 + 4 + 1 + 4; // 17 floats per vertex
            const vertexData = new Float32Array(loopBlinnVertices.length * floatsPerVertex);

            const fillColor = path.fillColor.toArray();
            const strokeColor = path.strokeColor.toArray();
            const bounds = path.bounds || { origin: { x: 0, y: 0 }, size: { width: 0, height: 0 } };

            for (let i = 0; i < loopBlinnVertices.length; i++) {
                const v = loopBlinnVertices[i];
                const offset = i * floatsPerVertex;

                // xy_position
                vertexData[offset + 0] = v.xy_position.x;
                vertexData[offset + 1] = v.xy_position.y;

                // st_position
                vertexData[offset + 2] = v.st_position.x;
                vertexData[offset + 3] = v.st_position.y;

                // fill_color
                vertexData[offset + 4] = fillColor[0];
                vertexData[offset + 5] = fillColor[1];
                vertexData[offset + 6] = fillColor[2];
                vertexData[offset + 7] = fillColor[3];

                // stroke_color
                vertexData[offset + 8] = strokeColor[0];
                vertexData[offset + 9] = strokeColor[1];
                vertexData[offset + 10] = strokeColor[2];
                vertexData[offset + 11] = strokeColor[3];

                // stroke_width
                vertexData[offset + 12] = path.strokeWidth || 1.0;

                // bounds
                vertexData[offset + 13] = bounds.origin.x;
                vertexData[offset + 14] = bounds.origin.y;
                vertexData[offset + 15] = bounds.size.width;
                vertexData[offset + 16] = bounds.size.height;
            }

            // Create storage buffer for vertices
            const vertexBuffer = this.bufferPool.acquire(
                vertexData.byteLength,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            );
            this.device.queue.writeBuffer(vertexBuffer, 0, vertexData);

            // Create bind group
            const bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayouts.pathRasterization,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: { buffer: vertexBuffer } }
                ]
            });

            // Draw
            renderPass.setBindGroup(0, bindGroup);
            renderPass.draw(loopBlinnVertices.length, 1, 0, 0);
        }
    }

    renderPathComposite(renderPass) {
        // Composite the path intermediate texture to the main render target
        // This is the second pass of two-pass path rendering (matches GPUI)

        const sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts.pathComposite,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.pathIntermediateTextureView },
                { binding: 2, resource: sampler }
            ]
        });

        renderPass.setPipeline(this.pipelines.pathComposite);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(3, 1, 0, 0); // Fullscreen triangle
    }

    renderSurfaces(renderPass, surfaces) {
        if (surfaces.length === 0) return;

        renderPass.setPipeline(this.pipelines.surface);

        for (const surface of surfaces) {
            if (!surface.texture) continue;

            // Create surface uniforms buffer
            const transformArray = surface.transform.toArray();

            // Build uniform data with proper layout
            // mat3x3: 3 columns × 16 bytes = 48 bytes
            // opacity: 4 bytes
            // grayscale: 4 bytes
            // corner_radii: 16 bytes
            // pad: 8 bytes
            // Total: 80 bytes = 20 floats
            const uniformData = new Float32Array(20);
            let offset = 0;

            // Column 0 of transform matrix
            uniformData[offset++] = transformArray[0]; // a
            uniformData[offset++] = transformArray[2]; // c
            uniformData[offset++] = transformArray[4]; // tx
            offset++; // padding

            // Column 1 of transform matrix
            uniformData[offset++] = transformArray[1]; // b
            uniformData[offset++] = transformArray[3]; // d
            uniformData[offset++] = transformArray[5]; // ty
            offset++; // padding

            // Column 2 of transform matrix (identity for 2D)
            uniformData[offset++] = 0;
            uniformData[offset++] = 0;
            uniformData[offset++] = 1;
            offset++; // padding

            // opacity, grayscale
            uniformData[offset++] = surface.opacity;
            uniformData[offset++] = surface.grayscale ? 1.0 : 0.0;
            offset += 2; // padding

            // corner_radii - vec4<f32>
            const cornerRadii = surface.cornerRadii.toArray();
            uniformData[offset++] = cornerRadii[0];
            uniformData[offset++] = cornerRadii[1];
            uniformData[offset++] = cornerRadii[2];
            uniformData[offset++] = cornerRadii[3];

            const surfaceUniformBuffer = this.bufferPool.acquire(
                uniformData.byteLength,
                GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            );
            this.device.queue.writeBuffer(surfaceUniformBuffer, 0, uniformData);

            // Create default sampler for surface
            const surfaceSampler = this.device.createSampler({
                magFilter: 'linear',
                minFilter: 'linear',
                mipmapFilter: 'linear',
                addressModeU: 'clamp-to-edge',
                addressModeV: 'clamp-to-edge',
            });

            // Create bind group
            const bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayouts.surface,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: { buffer: surfaceUniformBuffer } },
                    { binding: 2, resource: surface.texture.createView() },
                    { binding: 3, resource: surfaceSampler }
                ]
            });

            // Draw
            renderPass.setBindGroup(0, bindGroup);
            renderPass.draw(4, 1, 0, 0);
        }
    }

    destroy() {
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
        }
        if (this.monochromeAtlas) {
            this.monochromeAtlas.clear();
        }
        if (this.polychromeAtlas) {
            this.polychromeAtlas.clear();
        }
        if (this.bufferPool) {
            this.bufferPool.clear();
        }
        if (this.device) {
            this.device.destroy();
        }
    }
}
