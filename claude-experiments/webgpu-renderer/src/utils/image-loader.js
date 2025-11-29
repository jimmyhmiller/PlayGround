/**
 * Image loading utilities for PolychromeSprite
 */

import { PolychromeSprite } from '../core/primitives.js';
import { Bounds, Point, Size, Corners } from '../core/geometry.js';

export class ImageLoader {
    constructor(renderer) {
        this.renderer = renderer;
        this.imageCache = new Map();
    }

    /**
     * Load an image from a URL and return a tile
     */
    async loadImage(url) {
        // Check cache first
        if (this.imageCache.has(url)) {
            return this.imageCache.get(url);
        }

        return new Promise((resolve, reject) => {
            const img = new Image();

            img.onload = () => {
                // Create canvas to extract pixel data
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');

                // Draw image
                ctx.drawImage(img, 0, 0);

                // Extract pixel data
                const imageData = ctx.getImageData(0, 0, img.width, img.height);
                const pixelData = new Uint8Array(imageData.data);

                // Upload to atlas
                const key = `image_${url}`;
                const tile = this.renderer.polychromeAtlas.getOrInsert(
                    key,
                    img.width,
                    img.height,
                    pixelData
                );

                // Cache the result
                this.imageCache.set(url, tile);
                resolve(tile);
            };

            img.onerror = () => {
                reject(new Error(`Failed to load image: ${url}`));
            };

            img.crossOrigin = 'anonymous'; // Enable CORS if needed
            img.src = url;
        });
    }

    /**
     * Create a PolychromeSprite from an image URL
     */
    async createSpriteFromImage(url, x, y, width = null, height = null, options = {}) {
        const tile = await this.loadImage(url);

        // Use image dimensions if not specified
        const actualWidth = width || tile.bounds.size.x;
        const actualHeight = height || tile.bounds.size.y;

        const sprite = new PolychromeSprite();
        sprite.bounds = new Bounds(
            new Point(x, y),
            new Size(actualWidth, actualHeight)
        );
        sprite.tile = tile;
        sprite.cornerRadii = options.cornerRadii || new Corners();
        sprite.opacity = options.opacity !== undefined ? options.opacity : 1.0;
        sprite.grayscale = options.grayscale || false;

        return sprite;
    }

    /**
     * Generate a data URL from canvas for procedural images
     */
    generateDataURL(width, height, drawFn) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        drawFn(ctx);

        return canvas.toDataURL();
    }

    /**
     * Create a sprite from a procedural drawing function
     */
    async createProceduralSprite(width, height, drawFn, x, y, options = {}) {
        const dataURL = this.generateDataURL(width, height, drawFn);
        return this.createSpriteFromImage(dataURL, x, y, width, height, options);
    }

    /**
     * Clear image cache
     */
    clearCache() {
        this.imageCache.clear();
    }
}

/**
 * Procedural image generators
 */
export const ProceduralImages = {
    /**
     * Generate a simple gradient circle
     */
    gradientCircle(ctx) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) / 2;

        const gradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, radius
        );
        gradient.addColorStop(0, '#ff6b9d');
        gradient.addColorStop(0.5, '#c06cd8');
        gradient.addColorStop(1, '#6b6bff');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
    },

    /**
     * Generate a colorful noise pattern
     */
    noise(ctx) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            data[i] = Math.random() * 255;     // R
            data[i + 1] = Math.random() * 255; // G
            data[i + 2] = Math.random() * 255; // B
            data[i + 3] = 255;                 // A
        }

        ctx.putImageData(imageData, 0, 0);
    },

    /**
     * Generate a mandelbrot fractal
     */
    mandelbrot(ctx) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;

        const maxIterations = 100;

        for (let py = 0; py < height; py++) {
            for (let px = 0; px < width; px++) {
                const x0 = (px / width) * 3.5 - 2.5;
                const y0 = (py / height) * 2 - 1;

                let x = 0;
                let y = 0;
                let iteration = 0;

                while (x * x + y * y <= 4 && iteration < maxIterations) {
                    const xtemp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = xtemp;
                    iteration++;
                }

                const idx = (py * width + px) * 4;
                const color = (iteration / maxIterations) * 255;
                data[idx] = color;
                data[idx + 1] = color * 0.8;
                data[idx + 2] = color * 0.6;
                data[idx + 3] = 255;
            }
        }

        ctx.putImageData(imageData, 0, 0);
    },

    /**
     * Generate a geometric pattern
     */
    geometric(ctx) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;

        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);

        const colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#a8dadc'];
        const size = 20;

        for (let y = 0; y < height; y += size) {
            for (let x = 0; x < width; x += size) {
                const colorIndex = ((x / size) + (y / size)) % colors.length;
                ctx.fillStyle = colors[colorIndex];

                ctx.beginPath();
                ctx.moveTo(x + size / 2, y);
                ctx.lineTo(x + size, y + size / 2);
                ctx.lineTo(x + size / 2, y + size);
                ctx.lineTo(x, y + size / 2);
                ctx.closePath();
                ctx.fill();
            }
        }
    }
};
