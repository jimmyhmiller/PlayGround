/**
 * Text rendering using Canvas API for glyph generation
 */

import { MonochromeSprite } from '../core/primitives.js';
import { Bounds, Point, Size, Hsla } from '../core/geometry.js';

export class TextRenderer {
    constructor(renderer) {
        this.renderer = renderer;
        this.glyphCache = new Map();
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }

    /**
     * Measure text dimensions
     */
    measureText(text, fontSize = 16, fontFamily = 'sans-serif') {
        this.ctx.font = `${fontSize}px ${fontFamily}`;
        const metrics = this.ctx.measureText(text);
        return {
            width: metrics.width,
            height: fontSize * 1.2, // Approximate line height
            ascent: metrics.actualBoundingBoxAscent || fontSize * 0.8,
            descent: metrics.actualBoundingBoxDescent || fontSize * 0.2
        };
    }

    /**
     * Generate a glyph texture for a single character
     */
    generateGlyph(char, fontSize = 16, fontFamily = 'sans-serif', fontWeight = 'normal') {
        const key = `${char}_${fontSize}_${fontFamily}_${fontWeight}`;

        if (this.glyphCache.has(key)) {
            return this.glyphCache.get(key);
        }

        // Set up canvas for this glyph
        this.ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;
        const metrics = this.ctx.measureText(char);

        // Calculate glyph dimensions with padding for antialiasing
        const padding = 4;
        const width = Math.ceil(metrics.width) + padding * 2;
        const height = Math.ceil(fontSize * 1.4) + padding * 2;
        const baseline = Math.ceil(fontSize * 1.1) + padding;

        // Resize canvas if needed
        if (this.canvas.width < width || this.canvas.height < height) {
            this.canvas.width = Math.max(width, 128);
            this.canvas.height = Math.max(height, 128);
        }

        // Clear and draw glyph
        this.ctx.clearRect(0, 0, width, height);
        this.ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;
        this.ctx.fillStyle = 'white';
        this.ctx.textBaseline = 'alphabetic';
        this.ctx.fillText(char, padding, baseline);

        // Extract pixel data
        const imageData = this.ctx.getImageData(0, 0, width, height);
        const glyphData = new Uint8Array(width * height);

        // Convert RGBA to single channel (use alpha channel)
        for (let i = 0; i < width * height; i++) {
            glyphData[i] = imageData.data[i * 4 + 3]; // Alpha channel
        }

        // Upload to atlas
        const tile = this.renderer.monochromeAtlas.getOrInsert(key, width, height, glyphData);

        const glyphInfo = {
            tile,
            width: metrics.width,
            height: fontSize,
            advance: metrics.width,
            bearingX: 0,
            bearingY: metrics.actualBoundingBoxAscent || fontSize * 0.8
        };

        this.glyphCache.set(key, glyphInfo);
        return glyphInfo;
    }

    /**
     * Render a text string and return array of MonochromeSprite primitives
     */
    renderText(text, x, y, color = Hsla.rgb(0, 0, 0, 1), fontSize = 16, fontFamily = 'sans-serif', fontWeight = 'normal') {
        const sprites = [];
        let currentX = x;

        for (let i = 0; i < text.length; i++) {
            const char = text[i];

            // Skip spaces (just advance position)
            if (char === ' ') {
                currentX += fontSize * 0.3;
                continue;
            }

            const glyph = this.generateGlyph(char, fontSize, fontFamily, fontWeight);

            const sprite = new MonochromeSprite();
            sprite.bounds = new Bounds(
                new Point(currentX, y - glyph.bearingY),
                new Size(glyph.tile.bounds.size.x, glyph.tile.bounds.size.y)
            );
            sprite.color = color;
            sprite.tile = glyph.tile;

            sprites.push(sprite);
            currentX += glyph.advance;
        }

        return sprites;
    }

    /**
     * Render multiline text
     */
    renderTextBlock(lines, x, y, color = Hsla.rgb(0, 0, 0, 1), fontSize = 16, fontFamily = 'sans-serif', fontWeight = 'normal', lineHeight = 1.4) {
        const sprites = [];
        let currentY = y;

        for (const line of lines) {
            const lineSprites = this.renderText(line, x, currentY, color, fontSize, fontFamily, fontWeight);
            sprites.push(...lineSprites);
            currentY += fontSize * lineHeight;
        }

        return sprites;
    }

    /**
     * Clear glyph cache
     */
    clearCache() {
        this.glyphCache.clear();
    }
}
