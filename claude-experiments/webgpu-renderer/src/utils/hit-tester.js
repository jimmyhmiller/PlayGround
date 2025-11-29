/**
 * Hit testing utilities for determining which primitive is at a given position
 */

export class HitTester {
    constructor(scene) {
        this.scene = scene;
    }

    /**
     * Find the topmost primitive at the given (x, y) position
     * Returns { type, primitive, index } or null if nothing is hit
     */
    hitTest(x, y) {
        // Test in reverse order (topmost first)
        // Order: polychromeSprites, monochromeSprites, underlines, quads, shadows

        // Test polychrome sprites
        for (let i = this.scene.polychromeSprites.length - 1; i >= 0; i--) {
            const sprite = this.scene.polychromeSprites[i];
            if (this.testPolychromeSprite(sprite, x, y)) {
                return { type: 'polychromeSprite', primitive: sprite, index: i };
            }
        }

        // Test monochrome sprites
        for (let i = this.scene.monochromeSprites.length - 1; i >= 0; i--) {
            const sprite = this.scene.monochromeSprites[i];
            if (this.testMonochromeSprite(sprite, x, y)) {
                return { type: 'monochromeSprite', primitive: sprite, index: i };
            }
        }

        // Test underlines
        for (let i = this.scene.underlines.length - 1; i >= 0; i--) {
            const underline = this.scene.underlines[i];
            if (this.testUnderline(underline, x, y)) {
                return { type: 'underline', primitive: underline, index: i };
            }
        }

        // Test quads
        for (let i = this.scene.quads.length - 1; i >= 0; i--) {
            const quad = this.scene.quads[i];
            if (this.testQuad(quad, x, y)) {
                return { type: 'quad', primitive: quad, index: i };
            }
        }

        // Test shadows (usually behind everything)
        for (let i = this.scene.shadows.length - 1; i >= 0; i--) {
            const shadow = this.scene.shadows[i];
            if (this.testShadow(shadow, x, y)) {
                return { type: 'shadow', primitive: shadow, index: i };
            }
        }

        return null;
    }

    /**
     * Test if a point is inside a quad
     */
    testQuad(quad, x, y) {
        // Transform point into quad's local space by applying inverse transform
        const localPoint = this.inverseTransformPoint(x, y, quad.transform);

        // Check if point is in bounds
        const bounds = quad.bounds;
        if (localPoint.x < bounds.origin.x || localPoint.x > bounds.origin.x + bounds.size.width ||
            localPoint.y < bounds.origin.y || localPoint.y > bounds.origin.y + bounds.size.height) {
            return false;
        }

        // Check against rounded corners using SDF
        const halfSize = {
            x: bounds.size.width / 2,
            y: bounds.size.height / 2
        };
        const center = {
            x: bounds.origin.x + halfSize.x,
            y: bounds.origin.y + halfSize.y
        };
        const centerToPoint = {
            x: localPoint.x - center.x,
            y: localPoint.y - center.y
        };

        // Pick corner radius based on quadrant
        let cornerRadius;
        if (centerToPoint.x < 0 && centerToPoint.y < 0) {
            cornerRadius = quad.cornerRadii.topLeft;
        } else if (centerToPoint.x >= 0 && centerToPoint.y < 0) {
            cornerRadius = quad.cornerRadii.topRight;
        } else if (centerToPoint.x >= 0 && centerToPoint.y >= 0) {
            cornerRadius = quad.cornerRadii.bottomRight;
        } else {
            cornerRadius = quad.cornerRadii.bottomLeft;
        }

        // Calculate SDF distance
        const cornerToPoint = {
            x: Math.abs(centerToPoint.x) - halfSize.x,
            y: Math.abs(centerToPoint.y) - halfSize.y
        };
        const cornerCenterToPoint = {
            x: cornerToPoint.x + cornerRadius,
            y: cornerToPoint.y + cornerRadius
        };

        let sdf;
        if (cornerRadius === 0) {
            sdf = Math.max(cornerCenterToPoint.x, cornerCenterToPoint.y);
        } else {
            const distance = Math.sqrt(
                Math.max(0, cornerCenterToPoint.x) ** 2 +
                Math.max(0, cornerCenterToPoint.y) ** 2
            ) + Math.min(0, Math.max(cornerCenterToPoint.x, cornerCenterToPoint.y));
            sdf = distance - cornerRadius;
        }

        // Point is inside if SDF <= 0
        return sdf <= 0;
    }

    /**
     * Test if a point is inside a shadow
     */
    testShadow(shadow, x, y) {
        // Shadows are just rounded rectangles, similar to quads
        const bounds = shadow.bounds;
        if (x < bounds.origin.x || x > bounds.origin.x + bounds.size.width ||
            y < bounds.origin.y || y > bounds.origin.y + bounds.size.height) {
            return false;
        }

        // Check against rounded corners (similar logic to quad)
        const halfSize = {
            x: bounds.size.width / 2,
            y: bounds.size.height / 2
        };
        const center = {
            x: bounds.origin.x + halfSize.x,
            y: bounds.origin.y + halfSize.y
        };
        const centerToPoint = {
            x: x - center.x,
            y: y - center.y
        };

        let cornerRadius;
        if (centerToPoint.x < 0 && centerToPoint.y < 0) {
            cornerRadius = shadow.cornerRadii.topLeft;
        } else if (centerToPoint.x >= 0 && centerToPoint.y < 0) {
            cornerRadius = shadow.cornerRadii.topRight;
        } else if (centerToPoint.x >= 0 && centerToPoint.y >= 0) {
            cornerRadius = shadow.cornerRadii.bottomRight;
        } else {
            cornerRadius = shadow.cornerRadii.bottomLeft;
        }

        const cornerToPoint = {
            x: Math.abs(centerToPoint.x) - halfSize.x,
            y: Math.abs(centerToPoint.y) - halfSize.y
        };
        const cornerCenterToPoint = {
            x: cornerToPoint.x + cornerRadius,
            y: cornerToPoint.y + cornerRadius
        };

        let sdf;
        if (cornerRadius === 0) {
            sdf = Math.max(cornerCenterToPoint.x, cornerCenterToPoint.y);
        } else {
            const distance = Math.sqrt(
                Math.max(0, cornerCenterToPoint.x) ** 2 +
                Math.max(0, cornerCenterToPoint.y) ** 2
            ) + Math.min(0, Math.max(cornerCenterToPoint.x, cornerCenterToPoint.y));
            sdf = distance - cornerRadius;
        }

        return sdf <= 0;
    }

    /**
     * Test if a point is inside an underline
     */
    testUnderline(underline, x, y) {
        const bounds = underline.bounds;
        return x >= bounds.origin.x && x <= bounds.origin.x + bounds.size.width &&
               y >= bounds.origin.y && y <= bounds.origin.y + bounds.size.height;
    }

    /**
     * Test if a point is inside a monochrome sprite
     */
    testMonochromeSprite(sprite, x, y) {
        const bounds = sprite.bounds;
        return x >= bounds.origin.x && x <= bounds.origin.x + bounds.size.width &&
               y >= bounds.origin.y && y <= bounds.origin.y + bounds.size.height;
    }

    /**
     * Test if a point is inside a polychrome sprite
     */
    testPolychromeSprite(sprite, x, y) {
        const bounds = sprite.bounds;
        if (x < bounds.origin.x || x > bounds.origin.x + bounds.size.width ||
            y < bounds.origin.y || y > bounds.origin.y + bounds.size.height) {
            return false;
        }

        // Check against rounded corners if they exist
        const hasRoundedCorners = sprite.cornerRadii.topLeft > 0 ||
                                 sprite.cornerRadii.topRight > 0 ||
                                 sprite.cornerRadii.bottomRight > 0 ||
                                 sprite.cornerRadii.bottomLeft > 0;

        if (!hasRoundedCorners) {
            return true;
        }

        // Similar corner check as quad
        const halfSize = {
            x: bounds.size.width / 2,
            y: bounds.size.height / 2
        };
        const center = {
            x: bounds.origin.x + halfSize.x,
            y: bounds.origin.y + halfSize.y
        };
        const centerToPoint = {
            x: x - center.x,
            y: y - center.y
        };

        let cornerRadius;
        if (centerToPoint.x < 0 && centerToPoint.y < 0) {
            cornerRadius = sprite.cornerRadii.topLeft;
        } else if (centerToPoint.x >= 0 && centerToPoint.y < 0) {
            cornerRadius = sprite.cornerRadii.topRight;
        } else if (centerToPoint.x >= 0 && centerToPoint.y >= 0) {
            cornerRadius = sprite.cornerRadii.bottomRight;
        } else {
            cornerRadius = sprite.cornerRadii.bottomLeft;
        }

        const cornerToPoint = {
            x: Math.abs(centerToPoint.x) - halfSize.x,
            y: Math.abs(centerToPoint.y) - halfSize.y
        };
        const cornerCenterToPoint = {
            x: cornerToPoint.x + cornerRadius,
            y: cornerToPoint.y + cornerRadius
        };

        let sdf;
        if (cornerRadius === 0) {
            sdf = Math.max(cornerCenterToPoint.x, cornerCenterToPoint.y);
        } else {
            const distance = Math.sqrt(
                Math.max(0, cornerCenterToPoint.x) ** 2 +
                Math.max(0, cornerCenterToPoint.y) ** 2
            ) + Math.min(0, Math.max(cornerCenterToPoint.x, cornerCenterToPoint.y));
            sdf = distance - cornerRadius;
        }

        return sdf <= 0;
    }

    /**
     * Apply inverse transform to a point (simplified for 2D affine transforms)
     */
    inverseTransformPoint(x, y, transform) {
        // Calculate determinant
        const det = transform.m0 * transform.m3 - transform.m1 * transform.m2;

        if (Math.abs(det) < 1e-10) {
            // Singular matrix, return original point
            return { x, y };
        }

        // Translate by inverse translation
        const tx = x - transform.m4;
        const ty = y - transform.m5;

        // Apply inverse of linear transform
        const invDet = 1.0 / det;
        return {
            x: invDet * (transform.m3 * tx - transform.m2 * ty),
            y: invDet * (-transform.m1 * tx + transform.m0 * ty)
        };
    }
}
