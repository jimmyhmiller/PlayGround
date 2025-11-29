import { Transform } from './geometry.js';

/**
 * Camera - manages viewport transforms for pan/zoom
 */
export class Camera {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.position = { x: 0, y: 0 };
        this.zoom = 1.0;
        this.minZoom = 0.1;
        this.maxZoom = 10.0;
    }

    /**
     * Pan the camera by a delta
     */
    pan(dx, dy) {
        this.position.x += dx / this.zoom;
        this.position.y += dy / this.zoom;
    }

    /**
     * Zoom the camera, centered on a point
     */
    zoomAt(zoomDelta, centerX, centerY) {
        const oldZoom = this.zoom;
        this.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoom * zoomDelta));

        // Adjust position to keep the zoom centered on the given point
        if (oldZoom !== this.zoom) {
            const worldX = (centerX - this.width / 2) / oldZoom + this.position.x;
            const worldY = (centerY - this.height / 2) / oldZoom + this.position.y;

            this.position.x = worldX - (centerX - this.width / 2) / this.zoom;
            this.position.y = worldY - (centerY - this.height / 2) / this.zoom;
        }
    }

    /**
     * Reset camera to default position and zoom
     */
    reset() {
        this.position.x = 0;
        this.position.y = 0;
        this.zoom = 1.0;
    }

    /**
     * Get the view transform matrix
     * This transforms world coordinates to screen coordinates
     */
    getViewTransform() {
        // First translate by half viewport (to center)
        // Then scale by zoom
        // Then translate by camera position (negative to move world opposite direction)
        const centerTransform = Transform.translation(this.width / 2, this.height / 2);
        const scaleTransform = Transform.scale(this.zoom);
        const positionTransform = Transform.translation(-this.position.x, -this.position.y);

        return centerTransform.multiply(scaleTransform).multiply(positionTransform);
    }

    /**
     * Transform a screen point to world coordinates
     */
    screenToWorld(screenX, screenY) {
        const worldX = (screenX - this.width / 2) / this.zoom + this.position.x;
        const worldY = (screenY - this.height / 2) / this.zoom + this.position.y;
        return { x: worldX, y: worldY };
    }

    /**
     * Transform a world point to screen coordinates
     */
    worldToScreen(worldX, worldY) {
        const screenX = (worldX - this.position.x) * this.zoom + this.width / 2;
        const screenY = (worldY - this.position.y) * this.zoom + this.height / 2;
        return { x: screenX, y: screenY };
    }

    /**
     * Get the world-space bounds visible in the viewport
     */
    getVisibleBounds() {
        const topLeft = this.screenToWorld(0, 0);
        const bottomRight = this.screenToWorld(this.width, this.height);

        return {
            x: topLeft.x,
            y: topLeft.y,
            width: bottomRight.x - topLeft.x,
            height: bottomRight.y - topLeft.y
        };
    }

    /**
     * Update viewport size (e.g., on window resize)
     */
    resize(width, height) {
        this.width = width;
        this.height = height;
    }
}
