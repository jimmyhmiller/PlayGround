import { describe, it, expect } from 'vitest';
import { Point, Size, Bounds, Corners, Edges, Hsla, Transform } from './geometry.js';

describe('Bounds', () => {
    it('should detect intersecting bounds', () => {
        const bounds1 = new Bounds(
            new Point(0, 0),
            new Size(5, 5)
        );
        const bounds2 = new Bounds(
            new Point(4, 4),
            new Size(5, 5)
        );
        const bounds3 = new Bounds(
            new Point(10, 10),
            new Size(5, 5)
        );

        // Test Case 1: Intersecting bounds
        expect(bounds1.intersects(bounds2)).toBe(true);

        // Test Case 2: Non-Intersecting bounds
        expect(bounds1.intersects(bounds3)).toBe(false);

        // Test Case 3: Bounds intersecting with themselves
        expect(bounds1.intersects(bounds1)).toBe(true);
    });

    it('should compute intersection of two bounds', () => {
        const bounds1 = new Bounds(
            new Point(0, 0),
            new Size(10, 10)
        );
        const bounds2 = new Bounds(
            new Point(5, 5),
            new Size(10, 10)
        );

        const intersection = bounds1.intersect(bounds2);

        expect(intersection.origin.x).toBe(5);
        expect(intersection.origin.y).toBe(5);
        expect(intersection.size.width).toBe(5);
        expect(intersection.size.height).toBe(5);
    });

    it('should check if bounds contains a point', () => {
        const bounds = new Bounds(
            new Point(10, 10),
            new Size(20, 20)
        );

        expect(bounds.contains(new Point(15, 15))).toBe(true);
        expect(bounds.contains(new Point(5, 5))).toBe(false);
        expect(bounds.contains(new Point(10, 10))).toBe(true); // Top-left corner
        expect(bounds.contains(new Point(30, 30))).toBe(true); // Bottom-right corner
        expect(bounds.contains(new Point(31, 31))).toBe(false); // Just outside
    });

    it('should check if bounds is empty', () => {
        const emptyBounds = new Bounds(
            new Point(0, 0),
            new Size(0, 0)
        );
        const nonEmptyBounds = new Bounds(
            new Point(0, 0),
            new Size(10, 10)
        );

        expect(emptyBounds.isEmpty()).toBe(true);
        expect(nonEmptyBounds.isEmpty()).toBe(false);
    });

    it('should convert to array correctly', () => {
        const bounds = new Bounds(
            new Point(10, 20),
            new Size(30, 40)
        );

        const array = bounds.toArray();
        expect(array).toEqual([10, 20, 30, 40]);
    });
});

describe('Point', () => {
    it('should create point with correct coordinates', () => {
        const point = new Point(10, 20);
        expect(point.x).toBe(10);
        expect(point.y).toBe(20);
    });

    it('should convert to array correctly', () => {
        const point = new Point(5, 15);
        expect(point.toArray()).toEqual([5, 15]);
    });
});

describe('Size', () => {
    it('should create size with correct dimensions', () => {
        const size = new Size(100, 200);
        expect(size.width).toBe(100);
        expect(size.height).toBe(200);
    });

    it('should convert to array correctly', () => {
        const size = new Size(50, 75);
        expect(size.toArray()).toEqual([50, 75]);
    });
});

describe('Corners', () => {
    it('should create uniform corners', () => {
        const corners = Corners.uniform(10);
        expect(corners.topLeft).toBe(10);
        expect(corners.topRight).toBe(10);
        expect(corners.bottomRight).toBe(10);
        expect(corners.bottomLeft).toBe(10);
    });

    it('should convert to array correctly', () => {
        const corners = new Corners(1, 2, 3, 4);
        expect(corners.toArray()).toEqual([1, 2, 3, 4]);
    });
});

describe('Edges', () => {
    it('should create uniform edges', () => {
        const edges = Edges.uniform(5);
        expect(edges.top).toBe(5);
        expect(edges.right).toBe(5);
        expect(edges.bottom).toBe(5);
        expect(edges.left).toBe(5);
    });

    it('should convert to array correctly', () => {
        const edges = new Edges(1, 2, 3, 4);
        expect(edges.toArray()).toEqual([1, 2, 3, 4]);
    });
});

describe('Hsla', () => {
    it('should create HSLA color', () => {
        const color = new Hsla(0.5, 0.6, 0.7, 0.8);
        expect(color.h).toBe(0.5);
        expect(color.s).toBe(0.6);
        expect(color.l).toBe(0.7);
        expect(color.a).toBe(0.8);
    });

    it('should create RGB color', () => {
        const color = Hsla.rgb(0.2, 0.4, 0.6, 0.8);
        expect(color.a).toBe(0.8);
        // RGB values should be stored
    });

    it('should create black color', () => {
        const black = Hsla.black(0.5);
        expect(black.a).toBe(0.5);
    });

    it('should create white color', () => {
        const white = Hsla.white(0.5);
        expect(white.a).toBe(0.5);
    });

    it('should convert to array correctly', () => {
        const color = new Hsla(0.1, 0.2, 0.3, 0.4);
        const array = color.toArray();
        expect(array).toHaveLength(4);
        expect(array[3]).toBe(0.4); // Alpha should be preserved
    });
});

describe('Transform', () => {
    it('should create identity transform', () => {
        const identity = Transform.identity();
        const array = identity.toArray();
        expect(array).toEqual([1, 0, 0, 1, 0, 0, 0, 0]); // Includes padding for GPU alignment
    });

    it('should create translation transform', () => {
        const translation = Transform.translation(10, 20);
        const array = translation.toArray();
        expect(array[4]).toBe(10); // tx
        expect(array[5]).toBe(20); // ty
    });

    it('should create rotation transform', () => {
        const rotation = Transform.rotation(Math.PI / 2); // 90 degrees
        const array = rotation.toArray();
        // cos(90°) ≈ 0, sin(90°) ≈ 1
        expect(Math.abs(array[0])).toBeLessThan(0.0001); // cos(90°) ≈ 0
        expect(Math.abs(array[1] - 1)).toBeLessThan(0.0001); // sin(90°) ≈ 1
        expect(Math.abs(array[2] + 1)).toBeLessThan(0.0001); // -sin(90°) ≈ -1
        expect(Math.abs(array[3])).toBeLessThan(0.0001); // cos(90°) ≈ 0
    });

    it('should create scale transform', () => {
        const scale = Transform.scale(2, 3);
        const array = scale.toArray();
        expect(array[0]).toBe(2); // sx
        expect(array[3]).toBe(3); // sy
    });

    it('should multiply transforms correctly', () => {
        const translation = Transform.translation(10, 20);
        const scale = Transform.scale(2);
        const combined = translation.multiply(scale);

        // Combined transform should have both translation and scale
        const array = combined.toArray();
        expect(array[0]).toBe(2); // Scale is applied
        expect(array[4]).toBe(10); // Translation is preserved
        expect(array[5]).toBe(20);
    });

    it('should convert to array correctly', () => {
        const transform = new Transform(1, 2, 3, 4, 5, 6);
        expect(transform.toArray()).toEqual([1, 2, 3, 4, 5, 6, 0, 0]); // Includes padding for GPU alignment
    });
});
