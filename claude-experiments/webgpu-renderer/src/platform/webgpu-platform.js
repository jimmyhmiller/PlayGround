/**
 * Platform abstraction for WebGPU
 * Supports both browser WebGPU and node-webgpu
 */

/**
 * Detect if running in Node.js
 */
export function isNode() {
    return typeof process !== 'undefined' &&
           process.versions != null &&
           process.versions.node != null;
}

/**
 * Get WebGPU navigator object
 */
export async function getWebGPU() {
    if (isNode()) {
        // Node.js with node-webgpu
        // Use dynamic import with variable to prevent bundler from resolving it
        try {
            const moduleName = 'webgpu';
            const webgpu = await import(/* @vite-ignore */ moduleName);
            return webgpu.default || webgpu;
        } catch (error) {
            throw new Error('node-webgpu not found. Install it with: npm install webgpu');
        }
    } else {
        // Browser
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported in this browser');
        }
        return navigator.gpu;
    }
}

/**
 * Create a canvas for rendering
 */
export function createCanvas(width, height) {
    if (isNode()) {
        // Node.js - would need a headless canvas or offscreen rendering
        throw new Error('Canvas creation not yet supported in Node.js mode. Use browser for now.');
    } else {
        // Browser
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        return canvas;
    }
}

/**
 * Get canvas context
 */
export function getCanvasContext(canvas) {
    if (isNode()) {
        // Node.js - would need special handling
        throw new Error('Canvas context not yet supported in Node.js mode');
    } else {
        // Browser
        return canvas.getContext('webgpu');
    }
}

/**
 * Get preferred canvas format
 */
export async function getPreferredCanvasFormat(gpu) {
    if (isNode()) {
        // Node.js - typically BGRA8Unorm
        return 'bgra8unorm';
    } else {
        // Browser - use navigator method
        return gpu.getPreferredCanvasFormat();
    }
}

/**
 * Platform-specific initialization
 */
export async function initializePlatform() {
    const gpu = await getWebGPU();
    const adapter = await gpu.requestAdapter();

    if (!adapter) {
        throw new Error('Failed to get WebGPU adapter');
    }

    const device = await adapter.requestDevice();

    return {
        gpu,
        adapter,
        device,
        isNode: isNode()
    };
}

/**
 * Export image data (for headless rendering)
 */
export async function exportImageData(device, texture, width, height) {
    // Create a buffer to read the texture data
    const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
    const buffer = device.createBuffer({
        size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
        { texture },
        { buffer, bytesPerRow },
        { width, height }
    );
    device.queue.submit([encoder.finish()]);

    await buffer.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(buffer.getMappedRange());
    const copy = new Uint8Array(data);
    buffer.unmap();

    return copy;
}
