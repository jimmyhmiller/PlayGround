/**
 * BufferPool - manages reusable GPU buffers to reduce allocation overhead
 */
export class BufferPool {
    constructor(device) {
        this.device = device;
        this.buffers = [];
    }

    /**
     * Acquire a buffer of at least the given size
     * Returns an existing available buffer if possible, otherwise creates a new one
     */
    acquire(size, usage) {
        // Find an available buffer of sufficient size with matching usage
        for (const entry of this.buffers) {
            if (!entry.inUse &&
                entry.buffer.size >= size &&
                entry.usage === usage) {
                entry.inUse = true;
                return entry.buffer;
            }
        }

        // No suitable buffer found, create a new one
        // Round up size to next power of 2 for better reuse
        const alignedSize = this.nextPowerOfTwo(size);

        const buffer = this.device.createBuffer({
            size: alignedSize,
            usage: usage,
        });

        this.buffers.push({
            buffer,
            size: alignedSize,
            usage,
            inUse: true
        });

        return buffer;
    }

    /**
     * Release a buffer back to the pool for reuse
     */
    release(buffer) {
        for (const entry of this.buffers) {
            if (entry.buffer === buffer) {
                entry.inUse = false;
                return;
            }
        }
    }

    /**
     * Release all buffers in the pool
     */
    releaseAll() {
        for (const entry of this.buffers) {
            entry.inUse = false;
        }
    }

    /**
     * Clear the pool and destroy all buffers
     */
    clear() {
        for (const entry of this.buffers) {
            entry.buffer.destroy();
        }
        this.buffers = [];
    }

    /**
     * Get statistics about the buffer pool
     */
    getStats() {
        const inUse = this.buffers.filter(e => e.inUse).length;
        const available = this.buffers.filter(e => !e.inUse).length;
        const totalSize = this.buffers.reduce((sum, e) => sum + e.size, 0);

        return {
            total: this.buffers.length,
            inUse,
            available,
            totalSize,
            totalSizeMB: (totalSize / (1024 * 1024)).toFixed(2)
        };
    }

    /**
     * Round up to next power of 2 for better buffer reuse
     */
    nextPowerOfTwo(n) {
        if (n <= 256) return 256; // Minimum buffer size
        let power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }
}
