/**
 * Texture atlas for efficient sprite rendering
 * Packs multiple small textures into larger atlas textures
 */

import { Bounds, Point, Size } from './geometry.js';

export class AtlasTextureId {
    constructor(index = 0, kind = 'monochrome') {
        this.index = index;
        this.kind = kind; // 'monochrome' or 'polychrome'
    }
}

export class AtlasTile {
    constructor(textureId, bounds) {
        this.textureId = textureId;
        this.tileId = 0;
        this.padding = 0;
        this.bounds = bounds; // Bounds within atlas texture
    }

    toArray() {
        return [
            this.textureId.index,
            this.textureId.kind === 'monochrome' ? 0 : 1,
            this.tileId,
            this.padding,
            ...this.bounds.toArray()
        ];
    }
}

/**
 * Simple atlas allocator using shelf packing
 */
class ShelfAllocator {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.shelves = [];
        this.currentY = 0;
    }

    allocate(width, height) {
        // Try existing shelves first
        for (const shelf of this.shelves) {
            const allocation = shelf.allocate(width, height);
            if (allocation) {
                return allocation;
            }
        }

        // Create new shelf if there's space
        if (this.currentY + height <= this.height) {
            const shelf = new Shelf(this.currentY, this.width, height);
            this.shelves.push(shelf);
            this.currentY += height;
            return shelf.allocate(width, height);
        }

        return null;
    }

    clear() {
        this.shelves = [];
        this.currentY = 0;
    }
}

class Shelf {
    constructor(y, width, height) {
        this.y = y;
        this.width = width;
        this.height = height;
        this.currentX = 0;
    }

    allocate(width, height) {
        if (height > this.height) {
            return null;
        }

        if (this.currentX + width <= this.width) {
            const x = this.currentX;
            this.currentX += width;
            return {
                x,
                y: this.y,
                width,
                height
            };
        }

        return null;
    }
}

export class Atlas {
    constructor(device, kind = 'monochrome', size = 1024) {
        this.device = device;
        this.kind = kind;
        this.size = size;
        this.textures = [];
        this.allocators = [];
        this.tilesByKey = new Map();
        this.nextTileId = 0;
    }

    getFormat() {
        // Use rgba8unorm for all atlases to avoid needing texture-component-swizzle feature
        return 'rgba8unorm';
    }

    getBytesPerPixel() {
        // Always use 4 bytes per pixel now
        return 4;
    }

    createTexture() {
        const texture = this.device.createTexture({
            label: `${this.kind} atlas`,
            size: [this.size, this.size, 1],
            format: this.getFormat(),
            usage: GPUTextureUsage.TEXTURE_BINDING |
                   GPUTextureUsage.COPY_DST,
        });

        const view = texture.createView();

        // Clear texture to transparent by writing zeros
        const bytesPerPixel = this.getBytesPerPixel();
        const clearData = new Uint8Array(this.size * this.size * bytesPerPixel);
        this.device.queue.writeTexture(
            { texture },
            clearData,
            {
                offset: 0,
                bytesPerRow: this.size * bytesPerPixel,
                rowsPerImage: this.size,
            },
            [this.size, this.size, 1]
        );

        const allocator = new ShelfAllocator(this.size, this.size);

        const index = this.textures.length;
        this.textures.push({ texture, view, allocator });
        this.allocators.push(allocator);

        return index;
    }

    getOrInsert(key, width, height, data) {
        // Check if already cached
        if (this.tilesByKey.has(key)) {
            return this.tilesByKey.get(key);
        }

        // Try to allocate in existing textures
        for (let i = 0; i < this.allocators.length; i++) {
            const allocation = this.allocators[i].allocate(width, height);
            if (allocation) {
                const tile = this.uploadTile(i, allocation, width, height, data);
                this.tilesByKey.set(key, tile);
                return tile;
            }
        }

        // Create new texture
        const index = this.createTexture();
        const allocation = this.allocators[index].allocate(width, height);
        if (!allocation) {
            throw new Error(`Failed to allocate ${width}x${height} in atlas`);
        }

        const tile = this.uploadTile(index, allocation, width, height, data);
        this.tilesByKey.set(key, tile);
        return tile;
    }

    uploadTile(textureIndex, allocation, width, height, data) {
        const texture = this.textures[textureIndex].texture;
        const bytesPerPixel = this.getBytesPerPixel();

        // Convert monochrome data (1 byte per pixel) to RGBA (4 bytes per pixel)
        let uploadData = data;
        if (this.kind === 'monochrome' && data.length === width * height) {
            // Data is 1 byte per pixel, need to expand to RGBA
            uploadData = new Uint8Array(width * height * 4);
            for (let i = 0; i < width * height; i++) {
                uploadData[i * 4 + 0] = 255;      // R
                uploadData[i * 4 + 1] = 255;      // G
                uploadData[i * 4 + 2] = 255;      // B
                uploadData[i * 4 + 3] = data[i];  // A (alpha from input)
            }
        }

        this.device.queue.writeTexture(
            {
                texture,
                origin: [allocation.x, allocation.y, 0],
            },
            uploadData,
            {
                offset: 0,
                bytesPerRow: width * bytesPerPixel,
                rowsPerImage: height,
            },
            [width, height, 1]
        );

        const bounds = new Bounds(
            new Point(allocation.x, allocation.y),
            new Size(width, height)
        );

        return new AtlasTile(
            new AtlasTextureId(textureIndex, this.kind),
            bounds
        );
    }

    getTexture(id) {
        return this.textures[id.index];
    }

    getSampler() {
        if (!this._sampler) {
            this._sampler = this.device.createSampler({
                magFilter: 'linear',
                minFilter: 'linear',
            });
        }
        return this._sampler;
    }

    clear() {
        for (const { texture } of this.textures) {
            texture.destroy();
        }
        this.textures = [];
        this.allocators = [];
        this.tilesByKey.clear();
    }
}
