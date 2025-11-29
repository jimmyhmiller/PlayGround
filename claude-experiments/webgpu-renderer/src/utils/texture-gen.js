/**
 * Procedural texture generation for demos
 */

import { Bounds } from '../core/geometry.js';

/**
 * Generate a simple circle glyph (monochrome)
 */
export function generateCircleGlyph(size) {
    const data = new Uint8Array(size * size);
    const center = size / 2;
    const radius = size * 0.4;

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const dx = x - center;
            const dy = y - center;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const alpha = Math.max(0, Math.min(255, (radius - dist + 0.5) * 255));
            data[y * size + x] = alpha;
        }
    }

    return data;
}

/**
 * Generate a simple star glyph (monochrome)
 */
export function generateStarGlyph(size) {
    const data = new Uint8Array(size * size);
    const center = size / 2;

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const dx = x - center;
            const dy = y - center;
            const angle = Math.atan2(dy, dx);
            const dist = Math.sqrt(dx * dx + dy * dy);

            // 5-pointed star
            const points = 5;
            const starAngle = ((angle + Math.PI) / (2 * Math.PI)) * points;
            const modAngle = (starAngle % 1) * 2 - 1;
            const starRadius = center * 0.6 * (1 - Math.abs(modAngle) * 0.5);

            const alpha = Math.max(0, Math.min(255, (starRadius - dist + 0.5) * 255));
            data[y * size + x] = alpha;
        }
    }

    return data;
}

/**
 * Generate a heart glyph (monochrome)
 */
export function generateHeartGlyph(size) {
    const data = new Uint8Array(size * size);
    const center = size / 2;

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            // Normalize coordinates to [-1, 1]
            const nx = (x - center) / (size * 0.5);
            const ny = (center - y) / (size * 0.5); // Flip Y

            // Heart equation: (x^2 + y^2 - 1)^3 - x^2 * y^3 = 0
            const value = Math.pow(nx * nx + ny * ny - 1, 3) - nx * nx * ny * ny * ny;
            const alpha = value < 0 ? 255 : 0;

            // Add antialiasing
            const gradient = Math.abs(value) * 100;
            const smoothAlpha = Math.max(0, Math.min(255, 255 - gradient));

            data[y * size + x] = smoothAlpha;
        }
    }

    return data;
}

/**
 * Generate a gradient pattern (polychrome)
 */
export function generateGradientPattern(width, height) {
    const data = new Uint8Array(width * height * 4);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            const u = x / width;
            const v = y / height;

            // Colorful gradient
            data[i + 0] = Math.floor(255 * (1 - u)); // R
            data[i + 1] = Math.floor(255 * (1 - v)); // G
            data[i + 2] = Math.floor(255 * Math.sqrt(u * v)); // B
            data[i + 3] = 255; // A
        }
    }

    return data;
}

/**
 * Generate a checkerboard pattern (polychrome)
 */
export function generateCheckerboard(width, height, squareSize = 8) {
    const data = new Uint8Array(width * height * 4);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            const checkX = Math.floor(x / squareSize) % 2;
            const checkY = Math.floor(y / squareSize) % 2;
            const isWhite = (checkX + checkY) % 2 === 0;

            const value = isWhite ? 255 : 80;
            data[i + 0] = value;
            data[i + 1] = value;
            data[i + 2] = value;
            data[i + 3] = 255;
        }
    }

    return data;
}

/**
 * Generate a circular gradient (polychrome)
 */
export function generateCircularGradient(size) {
    const data = new Uint8Array(size * size * 4);
    const center = size / 2;

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const i = (y * size + x) * 4;
            const dx = x - center;
            const dy = y - center;
            const dist = Math.sqrt(dx * dx + dy * dy) / center;
            const angle = Math.atan2(dy, dx);

            // Rainbow gradient based on angle
            const hue = (angle + Math.PI) / (2 * Math.PI);
            const saturation = 1 - dist;
            const lightness = 0.5;

            const rgb = hslToRgb(hue, saturation, lightness);

            data[i + 0] = Math.floor(rgb[0] * 255);
            data[i + 1] = Math.floor(rgb[1] * 255);
            data[i + 2] = Math.floor(rgb[2] * 255);
            data[i + 3] = Math.floor((1 - dist) * 255); // Fade out at edges
        }
    }

    return data;
}

function hslToRgb(h, s, l) {
    let r, g, b;

    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };

        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }

    return [r, g, b];
}
