/**
 * Path tessellation for GPU rendering
 * Converts bezier curves to line segments and generates triangle meshes
 */

import { PathSegmentType } from '../core/path.js';
import { Point } from '../core/geometry.js';

/**
 * Tessellate a path into line segments
 */
export class PathTessellator {
    constructor(tolerance = 0.5) {
        this.tolerance = tolerance; // Maximum distance from curve
    }

    /**
     * Tessellate path segments into polyline
     */
    tessellate(segments) {
        const points = [];
        let currentPoint = new Point(0, 0);

        for (const segment of segments) {
            switch (segment.type) {
                case PathSegmentType.MoveTo:
                    currentPoint = segment.points[0];
                    points.push(new Point(currentPoint.x, currentPoint.y));
                    break;

                case PathSegmentType.LineTo:
                    currentPoint = segment.points[0];
                    points.push(new Point(currentPoint.x, currentPoint.y));
                    break;

                case PathSegmentType.QuadraticTo: {
                    const control = segment.points[0];
                    const end = segment.points[1];
                    const subdivided = this.subdivideQuadratic(
                        currentPoint,
                        control,
                        end
                    );
                    points.push(...subdivided.slice(1)); // Skip first point (already in points)
                    currentPoint = end;
                    break;
                }

                case PathSegmentType.CubicTo: {
                    const control1 = segment.points[0];
                    const control2 = segment.points[1];
                    const end = segment.points[2];
                    const subdivided = this.subdivideCubic(
                        currentPoint,
                        control1,
                        control2,
                        end
                    );
                    points.push(...subdivided.slice(1));
                    currentPoint = end;
                    break;
                }

                case PathSegmentType.Close:
                    // Line back to start
                    if (points.length > 0) {
                        points.push(new Point(points[0].x, points[0].y));
                    }
                    break;
            }
        }

        return points;
    }

    /**
     * Subdivide quadratic bezier curve into line segments
     */
    subdivideQuadratic(p0, p1, p2, t = 0.5) {
        const points = [];
        this._subdivideQuadraticRecursive(p0, p1, p2, points);
        return points;
    }

    _subdivideQuadraticRecursive(p0, p1, p2, points) {
        // Check if curve is flat enough
        const dx = p2.x - p0.x;
        const dy = p2.y - p0.y;
        const d = Math.abs((p1.x - p0.x) * dy - (p1.y - p0.y) * dx);

        if (d * d <= this.tolerance * (dx * dx + dy * dy)) {
            if (points.length === 0 || points[points.length - 1].x !== p0.x || points[points.length - 1].y !== p0.y) {
                points.push(new Point(p0.x, p0.y));
            }
            points.push(new Point(p2.x, p2.y));
            return;
        }

        // Subdivide
        const p01 = new Point((p0.x + p1.x) / 2, (p0.y + p1.y) / 2);
        const p12 = new Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
        const p012 = new Point((p01.x + p12.x) / 2, (p01.y + p12.y) / 2);

        this._subdivideQuadraticRecursive(p0, p01, p012, points);
        this._subdivideQuadraticRecursive(p012, p12, p2, points);
    }

    /**
     * Subdivide cubic bezier curve into line segments
     */
    subdivideCubic(p0, p1, p2, p3) {
        const points = [];
        this._subdivideCubicRecursive(p0, p1, p2, p3, points);
        return points;
    }

    _subdivideCubicRecursive(p0, p1, p2, p3, points) {
        // Flatness test
        const ux = 3 * p1.x - 2 * p0.x - p3.x;
        const uy = 3 * p1.y - 2 * p0.y - p3.y;
        const vx = 3 * p2.x - 2 * p3.x - p0.x;
        const vy = 3 * p2.y - 2 * p3.y - p0.y;

        const maxDist = Math.max(ux * ux, vx * vx) + Math.max(uy * uy, vy * vy);

        if (maxDist <= this.tolerance * this.tolerance) {
            if (points.length === 0 || points[points.length - 1].x !== p0.x || points[points.length - 1].y !== p0.y) {
                points.push(new Point(p0.x, p0.y));
            }
            points.push(new Point(p3.x, p3.y));
            return;
        }

        // Subdivide using de Casteljau
        const p01 = new Point((p0.x + p1.x) / 2, (p0.y + p1.y) / 2);
        const p12 = new Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
        const p23 = new Point((p2.x + p3.x) / 2, (p2.y + p3.y) / 2);
        const p012 = new Point((p01.x + p12.x) / 2, (p01.y + p12.y) / 2);
        const p123 = new Point((p12.x + p23.x) / 2, (p12.y + p23.y) / 2);
        const p0123 = new Point((p012.x + p123.x) / 2, (p012.y + p123.y) / 2);

        this._subdivideCubicRecursive(p0, p01, p012, p0123, points);
        this._subdivideCubicRecursive(p0123, p123, p23, p3, points);
    }

    /**
     * Generate stroke mesh from polyline
     */
    generateStroke(points, strokeWidth, closed = false) {
        if (points.length < 2) return { vertices: [], indices: [] };

        const vertices = [];
        const indices = [];
        const halfWidth = strokeWidth / 2;

        // Generate vertices along the path with width
        for (let i = 0; i < points.length; i++) {
            const p = points[i];

            // Calculate tangent
            let tangent;
            if (i === 0 && !closed) {
                tangent = this._normalize({
                    x: points[1].x - points[0].x,
                    y: points[1].y - points[0].y
                });
            } else if (i === points.length - 1 && !closed) {
                tangent = this._normalize({
                    x: points[i].x - points[i - 1].x,
                    y: points[i].y - points[i - 1].y
                });
            } else {
                const prev = closed && i === 0 ? points[points.length - 2] : points[i - 1];
                const next = closed && i === points.length - 1 ? points[1] : points[i + 1];
                tangent = this._normalize({
                    x: next.x - prev.x,
                    y: next.y - prev.y
                });
            }

            // Normal perpendicular to tangent
            const normal = { x: -tangent.y, y: tangent.x };

            // Create two vertices (left and right of centerline)
            vertices.push({
                x: p.x + normal.x * halfWidth,
                y: p.y + normal.y * halfWidth
            });
            vertices.push({
                x: p.x - normal.x * halfWidth,
                y: p.y - normal.y * halfWidth
            });

            // Create triangles
            if (i > 0) {
                const base = (i - 1) * 2;
                indices.push(base, base + 1, base + 2);
                indices.push(base + 1, base + 3, base + 2);
            }
        }

        // Close the stroke if needed
        if (closed && points.length > 2) {
            const base = (points.length - 1) * 2;
            indices.push(base, base + 1, 0);
            indices.push(base + 1, 1, 0);
        }

        return { vertices, indices };
    }

    /**
     * Simple polygon triangulation (ear clipping)
     */
    triangulate(points) {
        if (points.length < 3) return [];

        const indices = [];
        const vertices = points.slice();
        const remaining = vertices.map((_, i) => i);

        while (remaining.length > 3) {
            let earFound = false;

            for (let i = 0; i < remaining.length; i++) {
                const i0 = remaining[i];
                const i1 = remaining[(i + 1) % remaining.length];
                const i2 = remaining[(i + 2) % remaining.length];

                const v0 = vertices[i0];
                const v1 = vertices[i1];
                const v2 = vertices[i2];

                if (this._isEar(v0, v1, v2, vertices, remaining)) {
                    indices.push(i0, i1, i2);
                    remaining.splice((i + 1) % remaining.length, 1);
                    earFound = true;
                    break;
                }
            }

            if (!earFound) break; // Prevent infinite loop on degenerate polygons
        }

        // Add final triangle
        if (remaining.length === 3) {
            indices.push(remaining[0], remaining[1], remaining[2]);
        }

        return indices;
    }

    _isEar(v0, v1, v2, vertices, remaining) {
        // Check if triangle is convex
        const cross = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
        if (cross <= 0) return false;

        // Check if any other vertex is inside the triangle
        for (const idx of remaining) {
            const v = vertices[idx];
            if (v === v0 || v === v1 || v === v2) continue;
            if (this._pointInTriangle(v, v0, v1, v2)) {
                return false;
            }
        }

        return true;
    }

    _pointInTriangle(p, a, b, c) {
        const sign = (p1, p2, p3) => {
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
        };

        const d1 = sign(p, a, b);
        const d2 = sign(p, b, c);
        const d3 = sign(p, c, a);

        const hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        const hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);

        return !(hasNeg && hasPos);
    }

    _normalize(v) {
        const len = Math.sqrt(v.x * v.x + v.y * v.y);
        if (len === 0) return { x: 0, y: 1 };
        return { x: v.x / len, y: v.y / len };
    }

    /**
     * Generate Loop-Blinn vertices for GPU curve rendering
     * Returns array of {xy_position, st_position} vertices for direct GPU rendering
     */
    generateLoopBlinnVertices(segments) {
        const vertices = [];
        let currentPoint = new Point(0, 0);
        let startPoint = new Point(0, 0);
        let contourCount = 0;

        for (const segment of segments) {
            switch (segment.type) {
                case PathSegmentType.MoveTo:
                    currentPoint = segment.points[0];
                    startPoint = new Point(currentPoint.x, currentPoint.y);
                    contourCount = 0;
                    break;

                case PathSegmentType.LineTo: {
                    const to = segment.points[0];

                    // Straight edge triangle with st = (0, 1) for all vertices
                    // This makes f = s² - 1, which is always negative (inside)
                    vertices.push({
                        xy_position: new Point(currentPoint.x, currentPoint.y),
                        st_position: new Point(0.0, 1.0)
                    });
                    vertices.push({
                        xy_position: new Point(to.x, to.y),
                        st_position: new Point(0.0, 1.0)
                    });
                    vertices.push({
                        xy_position: new Point(to.x, to.y),
                        st_position: new Point(0.0, 1.0)
                    });

                    currentPoint = to;
                    contourCount++;
                    break;
                }

                case PathSegmentType.QuadraticTo: {
                    const ctrl = segment.points[0];
                    const to = segment.points[1];

                    // If we have previous segments, add a fill triangle
                    if (contourCount > 0) {
                        vertices.push({
                            xy_position: new Point(startPoint.x, startPoint.y),
                            st_position: new Point(0.0, 1.0)
                        });
                        vertices.push({
                            xy_position: new Point(currentPoint.x, currentPoint.y),
                            st_position: new Point(0.0, 1.0)
                        });
                        vertices.push({
                            xy_position: new Point(to.x, to.y),
                            st_position: new Point(0.0, 1.0)
                        });
                    }

                    // Add curved triangle with Loop-Blinn coordinates
                    // P0 (start): st = (0.0, 0.0)
                    // P1 (control): st = (0.5, 0.0)
                    // P2 (end): st = (1.0, 1.0)
                    vertices.push({
                        xy_position: new Point(currentPoint.x, currentPoint.y),
                        st_position: new Point(0.0, 0.0)
                    });
                    vertices.push({
                        xy_position: new Point(ctrl.x, ctrl.y),
                        st_position: new Point(0.5, 0.0)
                    });
                    vertices.push({
                        xy_position: new Point(to.x, to.y),
                        st_position: new Point(1.0, 1.0)
                    });

                    currentPoint = to;
                    contourCount++;
                    break;
                }

                case PathSegmentType.CubicTo: {
                    // Cubic curves need to be converted to quadratics
                    // For now, subdivide into multiple quadratic segments
                    const ctrl1 = segment.points[0];
                    const ctrl2 = segment.points[1];
                    const to = segment.points[2];

                    // Simple cubic to quadratic conversion
                    // Use the midpoint of the two control points as quadratic control
                    const quadCtrl = new Point(
                        (ctrl1.x + ctrl2.x) / 2,
                        (ctrl1.y + ctrl2.y) / 2
                    );

                    if (contourCount > 0) {
                        vertices.push({
                            xy_position: new Point(startPoint.x, startPoint.y),
                            st_position: new Point(0.0, 1.0)
                        });
                        vertices.push({
                            xy_position: new Point(currentPoint.x, currentPoint.y),
                            st_position: new Point(0.0, 1.0)
                        });
                        vertices.push({
                            xy_position: new Point(to.x, to.y),
                            st_position: new Point(0.0, 1.0)
                        });
                    }

                    vertices.push({
                        xy_position: new Point(currentPoint.x, currentPoint.y),
                        st_position: new Point(0.0, 0.0)
                    });
                    vertices.push({
                        xy_position: quadCtrl,
                        st_position: new Point(0.5, 0.0)
                    });
                    vertices.push({
                        xy_position: new Point(to.x, to.y),
                        st_position: new Point(1.0, 1.0)
                    });

                    currentPoint = to;
                    contourCount++;
                    break;
                }

                case PathSegmentType.Close:
                    // Close the path back to start if needed
                    if (currentPoint.x !== startPoint.x || currentPoint.y !== startPoint.y) {
                        vertices.push({
                            xy_position: new Point(currentPoint.x, currentPoint.y),
                            st_position: new Point(0.0, 1.0)
                        });
                        vertices.push({
                            xy_position: new Point(startPoint.x, startPoint.y),
                            st_position: new Point(0.0, 1.0)
                        });
                        vertices.push({
                            xy_position: new Point(startPoint.x, startPoint.y),
                            st_position: new Point(0.0, 1.0)
                        });
                    }
                    contourCount = 0;
                    break;
            }
        }

        return vertices;
    }
}
