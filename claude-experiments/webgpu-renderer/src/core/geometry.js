/**
 * Core geometry data structures matching GPUI's types
 */

export class Point {
    constructor(x = 0, y = 0) {
        this.x = x;
        this.y = y;
    }

    scale(factor) {
        return new Point(this.x * factor, this.y * factor);
    }

    toArray() {
        return [this.x, this.y];
    }

    static zero() {
        return new Point(0, 0);
    }
}

export class Size {
    constructor(width = 0, height = 0) {
        this.width = width;
        this.height = height;
    }

    scale(factor) {
        return new Size(this.width * factor, this.height * factor);
    }

    toArray() {
        return [this.width, this.height];
    }
}

export class Bounds {
    constructor(origin = new Point(), size = new Size()) {
        this.origin = origin;
        this.size = size;
    }

    intersect(other) {
        const x1 = Math.max(this.origin.x, other.origin.x);
        const y1 = Math.max(this.origin.y, other.origin.y);
        const x2 = Math.min(
            this.origin.x + this.size.width,
            other.origin.x + other.size.width
        );
        const y2 = Math.min(
            this.origin.y + this.size.height,
            other.origin.y + other.size.height
        );

        if (x2 <= x1 || y2 <= y1) {
            return new Bounds(new Point(x1, y1), new Size(0, 0));
        }

        return new Bounds(new Point(x1, y1), new Size(x2 - x1, y2 - y1));
    }

    union(other) {
        const x1 = Math.min(this.origin.x, other.origin.x);
        const y1 = Math.min(this.origin.y, other.origin.y);
        const x2 = Math.max(
            this.origin.x + this.size.width,
            other.origin.x + other.size.width
        );
        const y2 = Math.max(
            this.origin.y + this.size.height,
            other.origin.y + other.size.height
        );

        return new Bounds(new Point(x1, y1), new Size(x2 - x1, y2 - y1));
    }

    isEmpty() {
        return this.size.width <= 0 || this.size.height <= 0;
    }

    intersects(other) {
        const x1Max = Math.max(this.origin.x, other.origin.x);
        const y1Max = Math.max(this.origin.y, other.origin.y);
        const x2Min = Math.min(
            this.origin.x + this.size.width,
            other.origin.x + other.size.width
        );
        const y2Min = Math.min(
            this.origin.y + this.size.height,
            other.origin.y + other.size.height
        );

        return x1Max < x2Min && y1Max < y2Min;
    }

    contains(point) {
        return (
            point.x >= this.origin.x &&
            point.x <= this.origin.x + this.size.width &&
            point.y >= this.origin.y &&
            point.y <= this.origin.y + this.size.height
        );
    }

    // Convert to flat array for GPU buffers
    toArray() {
        return [this.origin.x, this.origin.y, this.size.width, this.size.height];
    }
}

export class Corners {
    constructor(topLeft = 0, topRight = 0, bottomRight = 0, bottomLeft = 0) {
        this.topLeft = topLeft;
        this.topRight = topRight;
        this.bottomRight = bottomRight;
        this.bottomLeft = bottomLeft;
    }

    static uniform(radius) {
        return new Corners(radius, radius, radius, radius);
    }

    toArray() {
        return [this.topLeft, this.topRight, this.bottomRight, this.bottomLeft];
    }
}

export class Edges {
    constructor(top = 0, right = 0, bottom = 0, left = 0) {
        this.top = top;
        this.right = right;
        this.bottom = bottom;
        this.left = left;
    }

    static uniform(width) {
        return new Edges(width, width, width, width);
    }

    toArray() {
        return [this.top, this.right, this.bottom, this.left];
    }
}

export class Transform {
    constructor(m0 = 1, m1 = 0, m2 = 0, m3 = 1, m4 = 0, m5 = 0) {
        // 2D affine transform matrix in column-major order
        // [a c tx]   [m0 m2 m4]
        // [b d ty] = [m1 m3 m5]
        // [0 0 1 ]   [0  0  1 ]
        this.m0 = m0; // scale x
        this.m1 = m1; // shear y
        this.m2 = m2; // shear x
        this.m3 = m3; // scale y
        this.m4 = m4; // translate x
        this.m5 = m5; // translate y
    }

    static identity() {
        return new Transform();
    }

    static translation(x, y) {
        const t = new Transform();
        t.m4 = x;
        t.m5 = y;
        return t;
    }

    static rotation(radians) {
        const t = new Transform();
        const cos = Math.cos(radians);
        const sin = Math.sin(radians);
        t.m0 = cos;
        t.m1 = sin;
        t.m2 = -sin;
        t.m3 = cos;
        return t;
    }

    static scale(sx, sy = sx) {
        const t = new Transform();
        t.m0 = sx;
        t.m3 = sy;
        return t;
    }

    // Multiply this transform by another (this = this * other)
    multiply(other) {
        const result = new Transform();
        result.m0 = this.m0 * other.m0 + this.m2 * other.m1;
        result.m1 = this.m1 * other.m0 + this.m3 * other.m1;
        result.m2 = this.m0 * other.m2 + this.m2 * other.m3;
        result.m3 = this.m1 * other.m2 + this.m3 * other.m3;
        result.m4 = this.m0 * other.m4 + this.m2 * other.m5 + this.m4;
        result.m5 = this.m1 * other.m4 + this.m3 * other.m5 + this.m5;
        return result;
    }

    // Transform a point
    transformPoint(x, y) {
        return {
            x: this.m0 * x + this.m2 * y + this.m4,
            y: this.m1 * x + this.m3 * y + this.m5
        };
    }

    // For GPU buffer (8 floats with padding for alignment)
    toArray() {
        return [this.m0, this.m1, this.m2, this.m3, this.m4, this.m5, 0, 0];
    }
}

export class Hsla {
    constructor(h = 0, s = 0, l = 0, a = 1) {
        this.h = h; // [0, 1]
        this.s = s; // [0, 1]
        this.l = l; // [0, 1]
        this.a = a; // [0, 1]
    }

    static rgb(r, g, b, a = 1) {
        // Convert RGB to HSL
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const l = (max + min) / 2;

        let h = 0;
        let s = 0;

        if (max !== min) {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

            if (max === r) {
                h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
            } else if (max === g) {
                h = ((b - r) / d + 2) / 6;
            } else {
                h = ((r - g) / d + 4) / 6;
            }
        }

        return new Hsla(h, s, l, a);
    }

    static black(a = 1) {
        return new Hsla(0, 0, 0, a);
    }

    static white(a = 1) {
        return new Hsla(0, 0, 1, a);
    }

    toArray() {
        return [this.h, this.s, this.l, this.a];
    }
}

/**
 * Helper to create gradients
 */
export class Gradient {
    static linear(angle, stops, colorSpace = 0) {
        return {
            tag: 1,
            colorSpace, // 0 = linear sRGB, 1 = Oklab
            solid: new Hsla(),
            gradientAngle: angle,
            colors: stops.map((stop, i) => ({
                color: stop.color,
                percentage: stop.position !== undefined ? stop.position : i / (stops.length - 1)
            })),
            pad: 0
        };
    }

    static solid(color) {
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
}
