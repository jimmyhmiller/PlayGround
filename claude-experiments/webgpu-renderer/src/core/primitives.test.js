import { describe, it, expect } from 'vitest';
import {
    Quad,
    Shadow,
    Underline,
    MonochromeSprite,
    PolychromeSprite,
    Path,
    Surface,
    Background,
    Scene
} from './primitives.js';
import { Bounds, Point, Size, Corners, Edges, Hsla, Transform } from './geometry.js';
import { PathBuilder } from './path.js';

describe('Background', () => {
    it('should create solid background', () => {
        const color = Hsla.rgb(0.5, 0.6, 0.7, 1);
        const bg = Background.Solid(color);

        expect(bg.tag).toBe(0);
        expect(bg.solid).toBe(color);
    });

    it('should create linear gradient background', () => {
        const stops = [
            { color: Hsla.rgb(1, 0, 0, 1), position: 0 },
            { color: Hsla.rgb(0, 0, 1, 1), position: 1 }
        ];
        const bg = Background.LinearGradient(45, stops, 0);

        expect(bg.tag).toBe(1);
        expect(bg.gradientAngle).toBe(45);
        expect(bg.colorSpace).toBe(0);
        expect(bg.colors.length).toBe(2);
    });

    it('should create radial gradient background', () => {
        const stops = [
            { color: Hsla.white(), position: 0 },
            { color: Hsla.black(), position: 1 }
        ];
        const bg = Background.RadialGradient(100, 100, 50, stops);

        expect(bg.tag).toBe(3);
        expect(bg.colorSpace).toBe(0);
    });

    it('should create conic gradient background', () => {
        const stops = [
            { color: Hsla.rgb(1, 0, 0, 1), position: 0 },
            { color: Hsla.rgb(0, 0, 1, 1), position: 1 }
        ];
        const bg = Background.ConicGradient(100, 100, 0, stops);

        expect(bg.tag).toBe(4);
    });

    it('should convert background to array', () => {
        const color = Hsla.rgb(0.5, 0.5, 0.5, 1);
        const bg = Background.Solid(color);
        const array = Background.toArray(bg);

        expect(Array.isArray(array)).toBe(true);
        expect(array.length).toBeGreaterThan(0);
    });
});

describe('Quad', () => {
    it('should create quad with default values', () => {
        const quad = new Quad();

        expect(quad.order).toBe(0);
        expect(quad.bounds).toBeInstanceOf(Bounds);
        expect(quad.background).toBeDefined();
        expect(quad.opacity).toBe(1.0);
    });

    it('should set quad properties', () => {
        const quad = new Quad();
        quad.bounds = new Bounds(new Point(10, 20), new Size(100, 50));
        quad.opacity = 0.5;

        expect(quad.bounds.origin.x).toBe(10);
        expect(quad.bounds.origin.y).toBe(20);
        expect(quad.bounds.size.width).toBe(100);
        expect(quad.bounds.size.height).toBe(50);
        expect(quad.opacity).toBe(0.5);
    });

    it('should convert quad to array', () => {
        const quad = new Quad();
        const array = quad.toArray();

        expect(Array.isArray(array)).toBe(true);
        expect(array.length).toBe(50); // 2 + 4 + 4 + 29 + 4 + 4 + 4 - transform not included
    });
});

describe('Shadow', () => {
    it('should create shadow with default values', () => {
        const shadow = new Shadow();

        expect(shadow.order).toBe(0);
        expect(shadow.blurRadius).toBe(0);
        expect(shadow.bounds).toBeInstanceOf(Bounds);
        expect(shadow.opacity).toBe(1.0);
    });

    it('should set shadow blur radius', () => {
        const shadow = new Shadow();
        shadow.blurRadius = 10;

        expect(shadow.blurRadius).toBe(10);
    });

    it('should convert shadow to array', () => {
        const shadow = new Shadow();
        const array = shadow.toArray();

        expect(Array.isArray(array)).toBe(true);
        expect(array.length).toBe(28); // 2 + 4 + 4 + 4 + 4 + 8 + 2
    });
});

describe('Underline', () => {
    it('should create underline with default values', () => {
        const underline = new Underline();

        expect(underline.order).toBe(0);
        expect(underline.thickness).toBe(2.0);
        expect(underline.wavy).toBe(0);
        expect(underline.opacity).toBe(1.0);
    });

    it('should set underline as wavy', () => {
        const underline = new Underline();
        underline.wavy = 1;

        expect(underline.wavy).toBe(1);
    });

    it('should convert underline to array', () => {
        const underline = new Underline();
        const array = underline.toArray();

        expect(Array.isArray(array)).toBe(true);
        expect(array.length).toBe(26); // 2 + 4 + 4 + 4 + 2 + 8 + 2
    });
});

describe('MonochromeSprite', () => {
    it('should create monochrome sprite with default values', () => {
        const sprite = new MonochromeSprite();

        expect(sprite.order).toBe(0);
        expect(sprite.bounds).toBeInstanceOf(Bounds);
        expect(sprite.color).toBeInstanceOf(Hsla);
    });

    it('should require tile for conversion to array', () => {
        const sprite = new MonochromeSprite();

        expect(() => sprite.toArray()).toThrow('requires a tile');
    });
});

describe('PolychromeSprite', () => {
    it('should create polychrome sprite with default values', () => {
        const sprite = new PolychromeSprite();

        expect(sprite.order).toBe(0);
        expect(sprite.grayscale).toBe(false);
        expect(sprite.opacity).toBe(1.0);
    });

    it('should set grayscale mode', () => {
        const sprite = new PolychromeSprite();
        sprite.grayscale = true;

        expect(sprite.grayscale).toBe(true);
    });

    it('should require tile for conversion to array', () => {
        const sprite = new PolychromeSprite();

        expect(() => sprite.toArray()).toThrow('requires a tile');
    });
});

describe('Path', () => {
    it('should create path with default values', () => {
        const path = new Path();

        expect(path.order).toBe(0);
        expect(path.fillColor).toBeInstanceOf(Hsla);
        expect(path.strokeColor).toBeInstanceOf(Hsla);
        expect(path.strokeWidth).toBe(1.0);
        expect(path.filled).toBe(true);
        expect(path.stroked).toBe(false);
        expect(path.opacity).toBe(1.0);
    });

    it('should accept path segments', () => {
        const builder = new PathBuilder();
        builder.circle(50, 50, 25);
        const segments = builder.build();

        const path = new Path(segments);

        expect(path.segments).toBe(segments);
        expect(path.segments.length).toBeGreaterThan(0);
    });

    it('should set fill and stroke properties', () => {
        const path = new Path();
        path.fillColor = Hsla.rgb(1, 0, 0, 1);
        path.strokeColor = Hsla.rgb(0, 0, 1, 1);
        path.strokeWidth = 3;
        path.filled = false;
        path.stroked = true;

        expect(path.fillColor.a).toBe(1);
        expect(path.strokeWidth).toBe(3);
        expect(path.filled).toBe(false);
        expect(path.stroked).toBe(true);
    });
});

describe('Surface', () => {
    it('should create surface with default values', () => {
        const surface = new Surface();

        expect(surface.order).toBe(0);
        expect(surface.bounds).toBeInstanceOf(Bounds);
        expect(surface.opacity).toBe(1.0);
        expect(surface.grayscale).toBe(false);
        expect(surface.texture).toBe(null);
    });

    it('should accept texture', () => {
        const mockTexture = { type: 'mock-texture' };
        const surface = new Surface(mockTexture);

        expect(surface.texture).toBe(mockTexture);
    });

    it('should set surface properties', () => {
        const surface = new Surface();
        surface.opacity = 0.8;
        surface.grayscale = true;

        expect(surface.opacity).toBe(0.8);
        expect(surface.grayscale).toBe(true);
    });
});

describe('Scene', () => {
    it('should create empty scene', () => {
        const scene = new Scene();

        expect(scene.quads).toEqual([]);
        expect(scene.shadows).toEqual([]);
        expect(scene.underlines).toEqual([]);
        expect(scene.monochromeSprites).toEqual([]);
        expect(scene.polychromeSprites).toEqual([]);
        expect(scene.paths).toEqual([]);
        expect(scene.surfaces).toEqual([]);
    });

    it('should insert quad', () => {
        const scene = new Scene();
        const quad = new Quad();

        scene.insertQuad(quad);

        expect(scene.quads.length).toBe(1);
        expect(quad.order).toBe(0);
    });

    it('should insert shadow', () => {
        const scene = new Scene();
        const shadow = new Shadow();

        scene.insertShadow(shadow);

        expect(scene.shadows.length).toBe(1);
        expect(shadow.order).toBe(0);
    });

    it('should insert path', () => {
        const scene = new Scene();
        const path = new Path();

        scene.insertPath(path);

        expect(scene.paths.length).toBe(1);
        expect(path.order).toBe(0);
    });

    it('should insert surface', () => {
        const scene = new Scene();
        const surface = new Surface();

        scene.insertSurface(surface);

        expect(scene.surfaces.length).toBe(1);
        expect(surface.order).toBe(0);
    });

    it('should clear scene', () => {
        const scene = new Scene();
        scene.insertQuad(new Quad());
        scene.insertShadow(new Shadow());
        scene.insertPath(new Path());

        scene.clear();

        expect(scene.quads.length).toBe(0);
        expect(scene.shadows.length).toBe(0);
        expect(scene.paths.length).toBe(0);
    });

    it('should sort primitives on finish', () => {
        const scene = new Scene();

        const quad1 = new Quad();
        const quad2 = new Quad();
        const quad3 = new Quad();

        scene.insertQuad(quad1);
        scene.insertQuad(quad2);
        scene.insertQuad(quad3);

        // Manually mess with the order
        quad3.order = 0;
        quad2.order = 1;
        quad1.order = 2;

        scene.finish();

        expect(scene.quads[0]).toBe(quad3);
        expect(scene.quads[1]).toBe(quad2);
        expect(scene.quads[2]).toBe(quad1);
    });

    it('should generate batches', () => {
        const scene = new Scene();
        scene.insertQuad(new Quad());
        scene.insertShadow(new Shadow());
        scene.insertUnderline(new Underline());

        scene.finish();

        const batches = Array.from(scene.batches());

        expect(batches.length).toBeGreaterThan(0);
        expect(batches.some(b => b.type === 'quads')).toBe(true);
        expect(batches.some(b => b.type === 'shadows')).toBe(true);
        expect(batches.some(b => b.type === 'underlines')).toBe(true);
    });

    it('should apply clip mask to primitives', () => {
        const scene = new Scene();
        const clipBounds = new Bounds(new Point(10, 10), new Size(100, 100));

        scene.pushClip(clipBounds);

        const quad = new Quad();
        scene.insertQuad(quad);

        expect(quad.contentMask.bounds.origin.x).toBe(10);
        expect(quad.contentMask.bounds.origin.y).toBe(10);

        scene.popClip();

        const quad2 = new Quad();
        scene.insertQuad(quad2);

        // Second quad should not have clip mask applied
        expect(quad2.contentMask.bounds.origin.x).toBe(0);
        expect(quad2.contentMask.bounds.origin.y).toBe(0);
    });

    it('should intersect nested clips', () => {
        const scene = new Scene();
        const clip1 = new Bounds(new Point(0, 0), new Size(100, 100));
        const clip2 = new Bounds(new Point(50, 50), new Size(100, 100));

        scene.pushClip(clip1);
        scene.pushClip(clip2);

        const quad = new Quad();
        scene.insertQuad(quad);

        // Intersection should be (50, 50) to (100, 100)
        expect(quad.contentMask.bounds.origin.x).toBe(50);
        expect(quad.contentMask.bounds.origin.y).toBe(50);
        expect(quad.contentMask.bounds.size.width).toBe(50);
        expect(quad.contentMask.bounds.size.height).toBe(50);
    });
});
