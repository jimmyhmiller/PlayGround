import { Bounds, Corners, Edges, Hsla, Point, Size, Gradient, Transform } from './geometry.js';

/**
 * Background types: Solid, LinearGradient, Pattern
 */
export class Background {
    static Solid(color) {
        return {
            tag: 0,
            colorSpace: 0,
            solid: color,
            gradientAngle: 0,
            colors: [
                { color: new Hsla(), percentage: 0 },
                { color: new Hsla(), percentage: 1 }
            ],
            pad: 0
        };
    }

    static LinearGradient(angle, stops, colorSpace = 0) {
        // Ensure we have exactly 2 stops
        const stop0 = stops[0] || { color: new Hsla(), position: 0 };
        const stop1 = stops[1] || stops[0] || { color: new Hsla(), position: 1 };

        return {
            tag: 1,
            colorSpace, // 0 = sRGB linear, 1 = Oklab
            solid: new Hsla(),
            gradientAngle: angle,
            colors: [
                {
                    color: stop0.color,
                    percentage: stop0.position !== undefined ? stop0.position : 0
                },
                {
                    color: stop1.color,
                    percentage: stop1.position !== undefined ? stop1.position : 1
                }
            ],
            pad: 0
        };
    }

    static Pattern(angle, color1, color2, spacing = 8.0, patternType = 0) {
        return {
            tag: 2,
            colorSpace: patternType, // Pattern type: 0 = diagonal stripes, 1 = dots
            solid: color1,
            gradientAngle: angle,
            colors: [
                {
                    color: color2,
                    percentage: spacing
                },
                {
                    color: new Hsla(),
                    percentage: 0
                }
            ],
            pad: 0
        };
    }

    static RadialGradient(centerX, centerY, radius, stops, colorSpace = 0) {
        // Ensure we have exactly 2 stops
        const stop0 = stops[0] || { color: new Hsla(), position: 0 };
        const stop1 = stops[1] || stops[0] || { color: new Hsla(), position: 1 };

        return {
            tag: 3,
            colorSpace, // 0 = sRGB linear, 1 = Oklab
            solid: new Hsla(centerX, centerY, radius, 0), // Store center and radius in solid color
            gradientAngle: 0,
            colors: [
                {
                    color: stop0.color,
                    percentage: stop0.position !== undefined ? stop0.position : 0
                },
                {
                    color: stop1.color,
                    percentage: stop1.position !== undefined ? stop1.position : 1
                }
            ],
            pad: 0
        };
    }

    static ConicGradient(centerX, centerY, angle, stops, colorSpace = 0) {
        // Ensure we have exactly 2 stops
        const stop0 = stops[0] || { color: new Hsla(), position: 0 };
        const stop1 = stops[1] || stops[0] || { color: new Hsla(), position: 1 };

        return {
            tag: 4,
            colorSpace, // 0 = sRGB linear, 1 = Oklab
            solid: new Hsla(centerX, centerY, 0, 0), // Store center in solid color (h, s components)
            gradientAngle: angle, // Starting angle in degrees
            colors: [
                {
                    color: stop0.color,
                    percentage: stop0.position !== undefined ? stop0.position : 0
                },
                {
                    color: stop1.color,
                    percentage: stop1.position !== undefined ? stop1.position : 1
                }
            ],
            pad: 0
        };
    }

    // Convert to flat array for GPU buffer (112 bytes total = 28 floats)
    static toArray(bg) {
        const arr = [];
        // tag, color_space (2 u32s = 8 bytes)
        arr.push(bg.tag, bg.colorSpace);
        // solid (Hsla struct, 4 floats = 16 bytes, offset 8 - no padding needed)
        arr.push(...bg.solid.toArray());
        // gradient_angle (1 float = 4 bytes, offset 24)
        arr.push(bg.gradientAngle);
        // Padding to align colors array to 16-byte boundary (offset 28 -> 32)
        arr.push(0); // 1 float = 4 bytes padding
        // colors array at offset 32: each LinearColorStop needs 32-byte stride
        for (let i = 0; i < 2; i++) {
            arr.push(...bg.colors[i].color.toArray()); // 4 floats (16 bytes)
            arr.push(bg.colors[i].percentage); // 1 float (4 bytes)
            arr.push(0, 0, 0); // 3 floats = 12 bytes padding for 32-byte stride
        }
        // pad (1 u32 = 4 bytes, offset 96)
        arr.push(bg.pad);
        // Padding to round struct size to 16-byte alignment (100 -> 112 bytes)
        arr.push(0, 0, 0); // 3 floats = 12 bytes padding
        return arr; // Total: 28 floats = 112 bytes
    }
}

export class ContentMask {
    constructor(bounds = new Bounds()) {
        this.bounds = bounds;
    }

    toArray() {
        return this.bounds.toArray();
    }
}

/**
 * Quad primitive - rectangle with rounded corners and borders
 */
export class Quad {
    constructor() {
        this.order = 0;
        this.borderStyle = 0; // 0 = Solid, 1 = Dashed
        this.bounds = new Bounds();
        this.contentMask = new ContentMask();
        this.background = Background.Solid(new Hsla(0, 0, 0.5, 1));
        this.borderColor = new Hsla(0, 0, 0, 1);
        this.cornerRadii = new Corners();
        this.borderWidths = new Edges();
        this.transform = Transform.identity();
        this.opacity = 1.0;
    }

    // Convert to flat Float32Array for GPU buffer (64 floats = 256 bytes)
    toArray() {
        const arr = [];

        // order, borderStyle (2 uints = 8 bytes, offset 0)
        arr.push(this.order, this.borderStyle);

        // bounds (4 floats = 16 bytes, offset 8)
        arr.push(...this.bounds.toArray());

        // contentMask (4 floats = 16 bytes, offset 24)
        arr.push(...this.contentMask.toArray());

        // Padding for Background alignment (offset 40 -> 48 for 16-byte alignment)
        arr.push(0, 0); // 2 floats = 8 bytes padding

        // background (28 floats = 112 bytes, offset 48)
        arr.push(...Background.toArray(this.background));

        // borderColor (4 floats = 16 bytes, offset 160)
        arr.push(...this.borderColor.toArray());

        // cornerRadii (4 floats = 16 bytes, offset 176)
        arr.push(...this.cornerRadii.toArray());

        // borderWidths (4 floats = 16 bytes, offset 192)
        arr.push(...this.borderWidths.toArray());

        // transform (8 floats = 32 bytes, offset 208)
        arr.push(...this.transform.toArray());

        // opacity + padding (2 floats = 8 bytes, offset 240)
        arr.push(this.opacity, 0);

        // Final padding to round struct to 32-byte alignment (from Transform field)
        // Struct size must be multiple of 32: 248 -> 256 bytes
        arr.push(0, 0);

        return arr; // Total: 64 floats = 256 bytes
    }

    static get SIZE() {
        return 256; // 64 floats
    }
}

/**
 * Shadow primitive
 */
export class Shadow {
    constructor() {
        this.order = 0;
        this.blurRadius = 0;
        this.bounds = new Bounds();
        this.cornerRadii = new Corners();
        this.contentMask = new ContentMask();
        this.color = new Hsla(0, 0, 0, 0.5);
        this.transform = Transform.identity();
        this.opacity = 1.0;
    }

    toArray() {
        return [
            this.order,
            this.blurRadius,
            ...this.bounds.toArray(),
            ...this.cornerRadii.toArray(),
            ...this.contentMask.toArray(),
            ...this.color.toArray(),
            ...this.transform.toArray(),
            this.opacity,
            0 // padding
        ];
    }

    static get SIZE() {
        return 136; // 34 floats
    }
}

/**
 * Underline primitive
 */
export class Underline {
    constructor() {
        this.order = 0;
        this.pad = 0;
        this.bounds = new Bounds();
        this.contentMask = new ContentMask();
        this.color = new Hsla(0, 0, 0, 1);
        this.thickness = 2.0;
        this.wavy = 0; // 0 = straight, 1 = wavy
        this.transform = Transform.identity();
        this.opacity = 1.0;
    }

    toArray() {
        return [
            this.order,
            this.pad,
            ...this.bounds.toArray(),
            ...this.contentMask.toArray(),
            ...this.color.toArray(),
            this.thickness,
            this.wavy,
            ...this.transform.toArray(),
            this.opacity,
            0 // padding
        ];
    }

    static get SIZE() {
        return 120; // 30 floats
    }
}

/**
 * MonochromeSprite - single-channel textures (glyphs, icons)
 */
export class MonochromeSprite {
    constructor() {
        this.order = 0;
        this.pad = 0;
        this.bounds = new Bounds();
        this.contentMask = new ContentMask();
        this.color = new Hsla(0, 0, 0, 1);
        this.tile = null; // AtlasTile
        this.transform = Transform.identity();
    }

    toArray() {
        if (!this.tile) {
            throw new Error('MonochromeSprite requires a tile');
        }
        return [
            this.order,
            this.pad,
            ...this.bounds.toArray(),
            ...this.contentMask.toArray(),
            ...this.color.toArray(),
            ...this.tile.toArray(),
            ...this.transform.toArray()
        ];
    }

    static get SIZE() {
        return 144; // 36 floats (was 28, added 8 for transform)
    }
}

/**
 * PolychromeSprite - full-color RGBA textures (images, emojis)
 */
export class PolychromeSprite {
    constructor() {
        this.order = 0;
        this.pad = 0;
        this.grayscale = false;
        this.opacity = 1.0;
        this.bounds = new Bounds();
        this.contentMask = new ContentMask();
        this.cornerRadii = new Corners();
        this.tile = null; // AtlasTile
        this.transform = Transform.identity();
    }

    toArray() {
        if (!this.tile) {
            throw new Error('PolychromeSprite requires a tile');
        }
        return [
            this.order,
            this.pad,
            this.grayscale ? 1 : 0,
            this.opacity,
            ...this.bounds.toArray(),
            ...this.contentMask.toArray(),
            ...this.cornerRadii.toArray(),
            ...this.tile.toArray(),
            ...this.transform.toArray()
        ];
    }

    static get SIZE() {
        return 136; // 34 floats (was 26, added 8 for transform)
    }
}

/**
 * Path primitive - vector paths
 * Rendered using GPU-accelerated tessellation
 */
export class Path {
    constructor(segments = []) {
        this.order = 0;
        this.segments = segments;
        this.fillColor = new Hsla(0, 0, 0, 1);
        this.strokeColor = new Hsla(0, 0, 0, 1);
        this.strokeWidth = 1.0;
        this.filled = true;
        this.stroked = false;
        this.contentMask = new ContentMask();
        this.transform = Transform.identity();
        this.opacity = 1.0;
    }

    static get SIZE() {
        return 0; // Paths use dynamic vertex buffers, not instance buffers
    }
}

/**
 * Surface primitive - external GPU textures
 * Used for rendering video frames, canvas elements, or other GPU textures
 */
export class Surface {
    constructor(texture = null) {
        this.order = 0;
        this.bounds = new Bounds();
        this.contentMask = new ContentMask();
        this.cornerRadii = new Corners();
        this.texture = texture; // GPUTexture or external texture
        this.opacity = 1.0;
        this.grayscale = false;
        this.transform = Transform.identity();
    }

    static get SIZE() {
        return 0; // Surfaces use custom rendering, not instance buffers
    }
}

/**
 * Scene - collection of primitives with draw ordering
 */
export class Scene {
    constructor() {
        this.quads = [];
        this.shadows = [];
        this.underlines = [];
        this.monochromeSprites = [];
        this.polychromeSprites = [];
        this.paths = [];
        this.surfaces = [];
        this.clipStack = [];
        this.currentClip = null;
        this.clear();
    }

    clear() {
        this.quads = [];
        this.shadows = [];
        this.underlines = [];
        this.monochromeSprites = [];
        this.polychromeSprites = [];
        this.paths = [];
        this.surfaces = [];
        this.clipStack = [];
        this.currentClip = null;
    }

    /**
     * Push a new clipping region onto the stack
     * If there's already a clip active, the new clip is intersected with it
     */
    pushClip(bounds) {
        if (this.currentClip) {
            // Intersect with existing clip
            this.currentClip = this.intersectBounds(this.currentClip, bounds);
        } else {
            this.currentClip = new Bounds(
                new Point(bounds.origin.x, bounds.origin.y),
                new Size(bounds.size.width, bounds.size.height)
            );
        }
        this.clipStack.push(this.currentClip);
    }

    /**
     * Pop the current clipping region from the stack
     */
    popClip() {
        this.clipStack.pop();
        this.currentClip = this.clipStack.length > 0
            ? this.clipStack[this.clipStack.length - 1]
            : null;
    }

    /**
     * Compute the intersection of two bounding rectangles
     */
    intersectBounds(b1, b2) {
        const x1 = Math.max(b1.origin.x, b2.origin.x);
        const y1 = Math.max(b1.origin.y, b2.origin.y);
        const x2 = Math.min(
            b1.origin.x + b1.size.width,
            b2.origin.x + b2.size.width
        );
        const y2 = Math.min(
            b1.origin.y + b1.size.height,
            b2.origin.y + b2.size.height
        );

        return new Bounds(
            new Point(x1, y1),
            new Size(Math.max(0, x2 - x1), Math.max(0, y2 - y1))
        );
    }

    insertQuad(quad) {
        quad.order = this.quads.length;
        if (this.currentClip) {
            quad.contentMask.bounds = new Bounds(
                new Point(this.currentClip.origin.x, this.currentClip.origin.y),
                new Size(this.currentClip.size.width, this.currentClip.size.height)
            );
        }
        this.quads.push(quad);
    }

    insertShadow(shadow) {
        shadow.order = this.shadows.length;
        if (this.currentClip) {
            shadow.contentMask.bounds = new Bounds(
                new Point(this.currentClip.origin.x, this.currentClip.origin.y),
                new Size(this.currentClip.size.width, this.currentClip.size.height)
            );
        }
        this.shadows.push(shadow);
    }

    insertUnderline(underline) {
        underline.order = this.underlines.length;
        if (this.currentClip) {
            underline.contentMask.bounds = new Bounds(
                new Point(this.currentClip.origin.x, this.currentClip.origin.y),
                new Size(this.currentClip.size.width, this.currentClip.size.height)
            );
        }
        this.underlines.push(underline);
    }

    insertMonochromeSprite(sprite) {
        sprite.order = this.monochromeSprites.length;
        if (this.currentClip) {
            sprite.contentMask.bounds = new Bounds(
                new Point(this.currentClip.origin.x, this.currentClip.origin.y),
                new Size(this.currentClip.size.width, this.currentClip.size.height)
            );
        }
        this.monochromeSprites.push(sprite);
    }

    insertPolychromeSprite(sprite) {
        sprite.order = this.polychromeSprites.length;
        if (this.currentClip) {
            sprite.contentMask.bounds = new Bounds(
                new Point(this.currentClip.origin.x, this.currentClip.origin.y),
                new Size(this.currentClip.size.width, this.currentClip.size.height)
            );
        }
        this.polychromeSprites.push(sprite);
    }

    insertPath(path) {
        path.order = this.paths.length;
        if (this.currentClip) {
            path.contentMask.bounds = new Bounds(
                new Point(this.currentClip.origin.x, this.currentClip.origin.y),
                new Size(this.currentClip.size.width, this.currentClip.size.height)
            );
        }
        this.paths.push(path);
    }

    insertSurface(surface) {
        surface.order = this.surfaces.length;
        if (this.currentClip) {
            surface.contentMask.bounds = new Bounds(
                new Point(this.currentClip.origin.x, this.currentClip.origin.y),
                new Size(this.currentClip.size.width, this.currentClip.size.height)
            );
        }
        this.surfaces.push(surface);
    }

    finish() {
        // Sort by draw order
        this.shadows.sort((a, b) => a.order - b.order);
        this.quads.sort((a, b) => a.order - b.order);
        this.underlines.sort((a, b) => a.order - b.order);
        this.monochromeSprites.sort((a, b) => a.order - b.order);
        this.polychromeSprites.sort((a, b) => a.order - b.order);
        this.paths.sort((a, b) => a.order - b.order);
        this.surfaces.sort((a, b) => a.order - b.order);
    }

    // Create batches for rendering
    *batches() {
        if (this.shadows.length > 0) {
            yield { type: 'shadows', primitives: this.shadows };
        }
        if (this.quads.length > 0) {
            yield { type: 'quads', primitives: this.quads };
        }
        if (this.underlines.length > 0) {
            yield { type: 'underlines', primitives: this.underlines };
        }
        if (this.monochromeSprites.length > 0) {
            yield { type: 'monochromeSprites', primitives: this.monochromeSprites };
        }
        if (this.polychromeSprites.length > 0) {
            yield { type: 'polychromeSprites', primitives: this.polychromeSprites };
        }
        if (this.paths.length > 0) {
            yield { type: 'paths', primitives: this.paths };
        }
        if (this.surfaces.length > 0) {
            yield { type: 'surfaces', primitives: this.surfaces };
        }
    }
}
