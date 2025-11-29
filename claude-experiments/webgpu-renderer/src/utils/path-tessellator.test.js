import { describe, it, expect } from 'vitest';
import { PathTessellator } from './path-tessellator.js';
import { PathBuilder, PathSegmentType } from '../core/path.js';
import { Point } from '../core/geometry.js';

describe('PathTessellator', () => {
    describe('Line segments', () => {
        it('should tessellate straight lines correctly', () => {
            const tessellator = new PathTessellator();
            const builder = new PathBuilder();
            builder.moveTo(0, 0)
                   .lineTo(10, 0)
                   .lineTo(10, 10);

            const segments = builder.build();
            const points = tessellator.tessellate(segments);

            expect(points.length).toBe(3);
            expect(points[0].x).toBe(0);
            expect(points[0].y).toBe(0);
            expect(points[1].x).toBe(10);
            expect(points[1].y).toBe(0);
            expect(points[2].x).toBe(10);
            expect(points[2].y).toBe(10);
        });
    });

    describe('Quadratic bezier curves', () => {
        it('should subdivide quadratic curves', () => {
            const tessellator = new PathTessellator(1.0);
            const p0 = new Point(0, 0);
            const p1 = new Point(5, 10);
            const p2 = new Point(10, 0);

            const points = tessellator.subdivideQuadratic(p0, p1, p2);

            // Should have at least start and end points
            expect(points.length).toBeGreaterThanOrEqual(2);
            // First point should be start
            expect(points[0].x).toBe(p0.x);
            expect(points[0].y).toBe(p0.y);
            // Last point should be end
            expect(points[points.length - 1].x).toBe(p2.x);
            expect(points[points.length - 1].y).toBe(p2.y);
        });

        it('should tessellate quadratic bezier segments', () => {
            const tessellator = new PathTessellator();
            const builder = new PathBuilder();
            builder.moveTo(0, 0)
                   .quadraticTo(5, 10, 10, 0);

            const segments = builder.build();
            const points = tessellator.tessellate(segments);

            expect(points.length).toBeGreaterThan(2); // Should subdivide the curve
            expect(points[0].x).toBe(0);
            expect(points[0].y).toBe(0);
            expect(points[points.length - 1].x).toBe(10);
            expect(points[points.length - 1].y).toBe(0);
        });
    });

    describe('Cubic bezier curves', () => {
        it('should subdivide cubic curves', () => {
            const tessellator = new PathTessellator(1.0);
            const p0 = new Point(0, 0);
            const p1 = new Point(3, 10);
            const p2 = new Point(7, 10);
            const p3 = new Point(10, 0);

            const points = tessellator.subdivideCubic(p0, p1, p2, p3);

            // Should have at least start and end points
            expect(points.length).toBeGreaterThanOrEqual(2);
            // First point should be start
            expect(points[0].x).toBe(p0.x);
            expect(points[0].y).toBe(p0.y);
            // Last point should be end
            expect(points[points.length - 1].x).toBe(p3.x);
            expect(points[points.length - 1].y).toBe(p3.y);
        });

        it('should tessellate cubic bezier segments', () => {
            const tessellator = new PathTessellator();
            const builder = new PathBuilder();
            builder.moveTo(0, 0)
                   .cubicTo(3, 10, 7, 10, 10, 0);

            const segments = builder.build();
            const points = tessellator.tessellate(segments);

            expect(points.length).toBeGreaterThan(2); // Should subdivide the curve
            expect(points[0].x).toBe(0);
            expect(points[0].y).toBe(0);
            expect(points[points.length - 1].x).toBe(10);
            expect(points[points.length - 1].y).toBe(0);
        });
    });

    describe('Triangulation', () => {
        it('should triangulate a simple square', () => {
            const tessellator = new PathTessellator();
            const points = [
                new Point(0, 0),
                new Point(10, 0),
                new Point(10, 10),
                new Point(0, 10)
            ];

            const indices = tessellator.triangulate(points);

            // A square should produce 2 triangles = 6 indices
            expect(indices.length).toBe(6);
            // All indices should be valid
            indices.forEach(index => {
                expect(index).toBeGreaterThanOrEqual(0);
                expect(index).toBeLessThan(points.length);
            });
        });

        it('should triangulate a triangle', () => {
            const tessellator = new PathTessellator();
            const points = [
                new Point(0, 0),
                new Point(10, 0),
                new Point(5, 10)
            ];

            const indices = tessellator.triangulate(points);

            // A triangle should produce 1 triangle = 3 indices
            expect(indices.length).toBe(3);
            expect(indices).toEqual([0, 1, 2]);
        });
    });

    describe('Stroke generation', () => {
        it('should generate stroke mesh for a line', () => {
            const tessellator = new PathTessellator();
            const points = [
                new Point(0, 0),
                new Point(10, 0)
            ];

            const mesh = tessellator.generateStroke(points, 2);

            // Should generate vertices for both sides of the line
            expect(mesh.vertices.length).toBeGreaterThan(0);
            expect(mesh.indices.length).toBeGreaterThan(0);

            // Vertices should come in pairs (left and right side of stroke)
            expect(mesh.vertices.length % 2).toBe(0);
        });

        it('should respect stroke width', () => {
            const tessellator = new PathTessellator();
            const points = [
                new Point(0, 0),
                new Point(10, 0)
            ];

            const mesh = tessellator.generateStroke(points, 4);

            // With a horizontal line, vertical offset should be strokeWidth/2
            // First vertex pair should be offset vertically
            expect(mesh.vertices.length).toBeGreaterThanOrEqual(4);

            const topVertex = mesh.vertices[0];
            const bottomVertex = mesh.vertices[1];

            // Difference in y should be approximately the stroke width
            expect(Math.abs(topVertex.y - bottomVertex.y)).toBeCloseTo(4, 0);
        });
    });

    describe('Closed paths', () => {
        it('should tessellate closed paths correctly', () => {
            const tessellator = new PathTessellator();
            const builder = new PathBuilder();
            builder.moveTo(0, 0)
                   .lineTo(10, 0)
                   .lineTo(10, 10)
                   .lineTo(0, 10)
                   .close();

            const segments = builder.build();
            const points = tessellator.tessellate(segments);

            // Should have 5 points (4 corners + close back to start)
            expect(points.length).toBe(5);
            // Last point should match first point
            expect(points[4].x).toBe(points[0].x);
            expect(points[4].y).toBe(points[0].y);
        });
    });

    describe('Tolerance', () => {
        it('should produce fewer points with higher tolerance', () => {
            const fineTessellator = new PathTessellator(0.1);
            const coarseTessellator = new PathTessellator(5.0);

            const builder = new PathBuilder();
            builder.moveTo(0, 0).cubicTo(30, 100, 70, 100, 100, 0);

            const segments = builder.build();
            const finePoints = fineTessellator.tessellate(segments);
            const coarsePoints = coarseTessellator.tessellate(segments);

            // Finer tolerance should produce more points
            expect(finePoints.length).toBeGreaterThan(coarsePoints.length);
        });
    });

    describe('Loop-Blinn vertex generation', () => {
        it('should generate vertices for quadratic curve', () => {
            const tessellator = new PathTessellator();
            const builder = new PathBuilder();
            builder.moveTo(0, 0).quadraticTo(50, 100, 100, 0);

            const segments = builder.build();
            const vertices = tessellator.generateLoopBlinnVertices(segments);

            // Should generate 3 vertices for the curve triangle
            expect(vertices.length).toBe(3);

            // Check Loop-Blinn coordinates
            expect(vertices[0].st_position.x).toBe(0.0);
            expect(vertices[0].st_position.y).toBe(0.0);

            expect(vertices[1].st_position.x).toBe(0.5);
            expect(vertices[1].st_position.y).toBe(0.0);

            expect(vertices[2].st_position.x).toBe(1.0);
            expect(vertices[2].st_position.y).toBe(1.0);
        });

        it('should generate vertices for straight line', () => {
            const tessellator = new PathTessellator();
            const builder = new PathBuilder();
            builder.moveTo(0, 0).lineTo(100, 100);

            const segments = builder.build();
            const vertices = tessellator.generateLoopBlinnVertices(segments);

            // Should generate 3 vertices for the line triangle
            expect(vertices.length).toBe(3);

            // All vertices should have st = (0, 1) for straight edges
            for (const v of vertices) {
                expect(v.st_position.x).toBe(0.0);
                expect(v.st_position.y).toBe(1.0);
            }
        });

        it('should generate vertices for closed path', () => {
            const tessellator = new PathTessellator();
            const builder = new PathBuilder();
            builder.moveTo(0, 0)
                   .lineTo(100, 0)
                   .lineTo(100, 100)
                   .lineTo(0, 100)
                   .close();

            const segments = builder.build();
            const vertices = tessellator.generateLoopBlinnVertices(segments);

            // Should generate vertices for all edges
            expect(vertices.length).toBeGreaterThan(0);
            expect(vertices.length % 3).toBe(0); // Should be multiple of 3 (triangles)
        });

        it('should have correct xy positions', () => {
            const tessellator = new PathTessellator();
            const builder = new PathBuilder();
            builder.moveTo(10, 20).quadraticTo(50, 60, 90, 20);

            const segments = builder.build();
            const vertices = tessellator.generateLoopBlinnVertices(segments);

            // Check positions match the path points
            expect(vertices[0].xy_position.x).toBe(10);
            expect(vertices[0].xy_position.y).toBe(20);

            expect(vertices[1].xy_position.x).toBe(50);
            expect(vertices[1].xy_position.y).toBe(60);

            expect(vertices[2].xy_position.x).toBe(90);
            expect(vertices[2].xy_position.y).toBe(20);
        });
    });
});
