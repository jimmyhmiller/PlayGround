/**
 * Path primitive for vector graphics
 * Uses GPU-accelerated tessellation and rendering
 */

import { Bounds, Point, Hsla, Transform } from './geometry.js';

/**
 * Path segment types
 */
export const PathSegmentType = {
    MoveTo: 0,
    LineTo: 1,
    QuadraticTo: 2,
    CubicTo: 3,
    Close: 4
};

/**
 * Path segment
 */
export class PathSegment {
    constructor(type, points = []) {
        this.type = type;
        this.points = points; // Array of Point objects
    }

    static moveTo(x, y) {
        return new PathSegment(PathSegmentType.MoveTo, [new Point(x, y)]);
    }

    static lineTo(x, y) {
        return new PathSegment(PathSegmentType.LineTo, [new Point(x, y)]);
    }

    static quadraticTo(cx, cy, x, y) {
        return new PathSegment(PathSegmentType.QuadraticTo, [
            new Point(cx, cy),
            new Point(x, y)
        ]);
    }

    static cubicTo(cx1, cy1, cx2, cy2, x, y) {
        return new PathSegment(PathSegmentType.CubicTo, [
            new Point(cx1, cy1),
            new Point(cx2, cy2),
            new Point(x, y)
        ]);
    }

    static close() {
        return new PathSegment(PathSegmentType.Close, []);
    }
}

/**
 * Path builder for creating vector paths
 */
export class PathBuilder {
    constructor() {
        this.segments = [];
        this.currentPoint = new Point(0, 0);
        this.startPoint = new Point(0, 0);
    }

    moveTo(x, y) {
        this.segments.push(PathSegment.moveTo(x, y));
        this.currentPoint = new Point(x, y);
        this.startPoint = new Point(x, y);
        return this;
    }

    lineTo(x, y) {
        this.segments.push(PathSegment.lineTo(x, y));
        this.currentPoint = new Point(x, y);
        return this;
    }

    quadraticTo(cx, cy, x, y) {
        this.segments.push(PathSegment.quadraticTo(cx, cy, x, y));
        this.currentPoint = new Point(x, y);
        return this;
    }

    cubicTo(cx1, cy1, cx2, cy2, x, y) {
        this.segments.push(PathSegment.cubicTo(cx1, cy1, cx2, cy2, x, y));
        this.currentPoint = new Point(x, y);
        return this;
    }

    close() {
        this.segments.push(PathSegment.close());
        this.currentPoint = new Point(this.startPoint.x, this.startPoint.y);
        return this;
    }

    // Convenience methods
    rect(x, y, width, height) {
        return this.moveTo(x, y)
            .lineTo(x + width, y)
            .lineTo(x + width, y + height)
            .lineTo(x, y + height)
            .close();
    }

    circle(cx, cy, radius) {
        const k = 0.5522847498; // Approximation constant for circle
        const r = radius;
        const kr = k * r;

        return this.moveTo(cx, cy - r)
            .cubicTo(cx + kr, cy - r, cx + r, cy - kr, cx + r, cy)
            .cubicTo(cx + r, cy + kr, cx + kr, cy + r, cx, cy + r)
            .cubicTo(cx - kr, cy + r, cx - r, cy + kr, cx - r, cy)
            .cubicTo(cx - r, cy - kr, cx - kr, cy - r, cx, cy - r)
            .close();
    }

    ellipse(cx, cy, rx, ry) {
        const k = 0.5522847498;
        const krx = k * rx;
        const kry = k * ry;

        return this.moveTo(cx, cy - ry)
            .cubicTo(cx + krx, cy - ry, cx + rx, cy - kry, cx + rx, cy)
            .cubicTo(cx + rx, cy + kry, cx + krx, cy + ry, cx, cy + ry)
            .cubicTo(cx - krx, cy + ry, cx - rx, cy + kry, cx - rx, cy)
            .cubicTo(cx - rx, cy - kry, cx - krx, cy - ry, cx, cy - ry)
            .close();
    }

    roundedRect(x, y, width, height, radius) {
        const r = Math.min(radius, width / 2, height / 2);
        const k = 0.5522847498;
        const kr = k * r;

        return this.moveTo(x + r, y)
            .lineTo(x + width - r, y)
            .cubicTo(x + width - r + kr, y, x + width, y + r - kr, x + width, y + r)
            .lineTo(x + width, y + height - r)
            .cubicTo(x + width, y + height - r + kr, x + width - r + kr, y + height, x + width - r, y + height)
            .lineTo(x + r, y + height)
            .cubicTo(x + r - kr, y + height, x, y + height - r + kr, x, y + height - r)
            .lineTo(x, y + r)
            .cubicTo(x, y + r - kr, x + r - kr, y, x + r, y)
            .close();
    }

    arc(cx, cy, radius, startAngle, endAngle) {
        // Approximate arc with cubic bezier curves
        const segments = Math.ceil(Math.abs(endAngle - startAngle) / (Math.PI / 2));
        const angleStep = (endAngle - startAngle) / segments;
        const k = (4 / 3) * Math.tan(angleStep / 4);

        for (let i = 0; i < segments; i++) {
            const a1 = startAngle + i * angleStep;
            const a2 = a1 + angleStep;

            const x1 = cx + radius * Math.cos(a1);
            const y1 = cy + radius * Math.sin(a1);
            const x2 = cx + radius * Math.cos(a2);
            const y2 = cy + radius * Math.sin(a2);

            const cp1x = x1 - radius * k * Math.sin(a1);
            const cp1y = y1 + radius * k * Math.cos(a1);
            const cp2x = x2 + radius * k * Math.sin(a2);
            const cp2y = y2 - radius * k * Math.cos(a2);

            if (i === 0) {
                this.moveTo(x1, y1);
            }
            this.cubicTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
        }

        return this;
    }

    build() {
        return this.segments;
    }

    computeBounds() {
        if (this.segments.length === 0) {
            return new Bounds(new Point(0, 0), { width: 0, height: 0 });
        }

        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        for (const segment of this.segments) {
            for (const point of segment.points) {
                minX = Math.min(minX, point.x);
                minY = Math.min(minY, point.y);
                maxX = Math.max(maxX, point.x);
                maxY = Math.max(maxY, point.y);
            }
        }

        return new Bounds(
            new Point(minX, minY),
            { width: maxX - minX, height: maxY - minY }
        );
    }
}

/**
 * Parse SVG path data string
 */
export function parseSVGPath(pathData) {
    const builder = new PathBuilder();
    const commands = pathData.match(/[a-df-z][^a-df-z]*/gi);

    if (!commands) return builder;

    let currentX = 0, currentY = 0;
    let startX = 0, startY = 0;

    for (const cmd of commands) {
        const type = cmd[0];
        const args = cmd.slice(1).trim().split(/[\s,]+/).map(parseFloat).filter(n => !isNaN(n));

        let i = 0;
        const isRelative = type === type.toLowerCase();

        switch (type.toUpperCase()) {
            case 'M': // MoveTo
                while (i < args.length) {
                    const x = isRelative ? currentX + args[i] : args[i];
                    const y = isRelative ? currentY + args[i + 1] : args[i + 1];
                    builder.moveTo(x, y);
                    currentX = x;
                    currentY = y;
                    startX = x;
                    startY = y;
                    i += 2;
                }
                break;

            case 'L': // LineTo
                while (i < args.length) {
                    const x = isRelative ? currentX + args[i] : args[i];
                    const y = isRelative ? currentY + args[i + 1] : args[i + 1];
                    builder.lineTo(x, y);
                    currentX = x;
                    currentY = y;
                    i += 2;
                }
                break;

            case 'H': // Horizontal line
                while (i < args.length) {
                    const x = isRelative ? currentX + args[i] : args[i];
                    builder.lineTo(x, currentY);
                    currentX = x;
                    i++;
                }
                break;

            case 'V': // Vertical line
                while (i < args.length) {
                    const y = isRelative ? currentY + args[i] : args[i];
                    builder.lineTo(currentX, y);
                    currentY = y;
                    i++;
                }
                break;

            case 'Q': // Quadratic bezier
                while (i < args.length) {
                    const cx = isRelative ? currentX + args[i] : args[i];
                    const cy = isRelative ? currentY + args[i + 1] : args[i + 1];
                    const x = isRelative ? currentX + args[i + 2] : args[i + 2];
                    const y = isRelative ? currentY + args[i + 3] : args[i + 3];
                    builder.quadraticTo(cx, cy, x, y);
                    currentX = x;
                    currentY = y;
                    i += 4;
                }
                break;

            case 'C': // Cubic bezier
                while (i < args.length) {
                    const cx1 = isRelative ? currentX + args[i] : args[i];
                    const cy1 = isRelative ? currentY + args[i + 1] : args[i + 1];
                    const cx2 = isRelative ? currentX + args[i + 2] : args[i + 2];
                    const cy2 = isRelative ? currentY + args[i + 3] : args[i + 3];
                    const x = isRelative ? currentX + args[i + 4] : args[i + 4];
                    const y = isRelative ? currentY + args[i + 5] : args[i + 5];
                    builder.cubicTo(cx1, cy1, cx2, cy2, x, y);
                    currentX = x;
                    currentY = y;
                    i += 6;
                }
                break;

            case 'Z': // Close path
                builder.close();
                currentX = startX;
                currentY = startY;
                break;
        }
    }

    return builder;
}
