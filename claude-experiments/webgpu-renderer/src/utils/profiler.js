/**
 * Performance profiling utilities
 */

export class Profiler {
    constructor() {
        this.metrics = new Map();
        this.frameMetrics = {
            frameTime: 0,
            fps: 0,
            drawCalls: 0,
            primitiveCount: 0,
            bufferCount: 0,
            bufferMemory: 0
        };
        this.history = [];
        this.maxHistory = 120; // 2 seconds at 60fps
    }

    /**
     * Start timing a labeled section
     */
    startSection(label) {
        if (!this.metrics.has(label)) {
            this.metrics.set(label, {
                startTime: 0,
                duration: 0,
                calls: 0,
                totalTime: 0
            });
        }
        const metric = this.metrics.get(label);
        metric.startTime = performance.now();
    }

    /**
     * End timing a labeled section
     */
    endSection(label) {
        const endTime = performance.now();
        const metric = this.metrics.get(label);
        if (metric) {
            metric.duration = endTime - metric.startTime;
            metric.totalTime += metric.duration;
            metric.calls++;
        }
    }

    /**
     * Record frame metrics
     */
    recordFrame(frameTime, fps, drawCalls, primitiveCount, bufferStats) {
        this.frameMetrics = {
            frameTime,
            fps,
            drawCalls,
            primitiveCount,
            bufferCount: bufferStats.total,
            bufferMemory: parseFloat(bufferStats.totalSizeMB)
        };

        // Add to history
        this.history.push({ ...this.frameMetrics, timestamp: performance.now() });
        if (this.history.length > this.maxHistory) {
            this.history.shift();
        }
    }

    /**
     * Get current metrics
     */
    getMetrics() {
        return {
            frame: this.frameMetrics,
            sections: Array.from(this.metrics.entries()).map(([label, data]) => ({
                label,
                duration: data.duration,
                averageTime: data.totalTime / data.calls,
                calls: data.calls
            }))
        };
    }

    /**
     * Get performance statistics over history
     */
    getStats() {
        if (this.history.length === 0) {
            return null;
        }

        const fpsSamples = this.history.map(h => h.fps);
        const frameSamples = this.history.map(h => h.frameTime);

        return {
            fps: {
                current: this.frameMetrics.fps,
                average: this.average(fpsSamples),
                min: Math.min(...fpsSamples),
                max: Math.max(...fpsSamples)
            },
            frameTime: {
                current: this.frameMetrics.frameTime,
                average: this.average(frameSamples),
                min: Math.min(...frameSamples),
                max: Math.max(...frameSamples)
            },
            primitives: {
                current: this.frameMetrics.primitiveCount,
                average: this.average(this.history.map(h => h.primitiveCount))
            },
            buffers: {
                count: this.frameMetrics.bufferCount,
                memory: this.frameMetrics.bufferMemory
            }
        };
    }

    /**
     * Calculate average of array
     */
    average(arr) {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    /**
     * Reset all metrics
     */
    reset() {
        this.metrics.clear();
        this.history = [];
    }

    /**
     * Get formatted report
     */
    getReport() {
        const stats = this.getStats();
        if (!stats) return 'No data';

        return `
Performance Report:
------------------
FPS: ${stats.fps.current.toFixed(1)} (avg: ${stats.fps.average.toFixed(1)}, min: ${stats.fps.min.toFixed(1)}, max: ${stats.fps.max.toFixed(1)})
Frame Time: ${stats.frameTime.current.toFixed(2)}ms (avg: ${stats.frameTime.average.toFixed(2)}ms)
Primitives: ${stats.primitives.current} (avg: ${stats.primitives.average.toFixed(0)})
Buffers: ${stats.buffers.count} (${stats.buffers.memory.toFixed(2)}MB)

Sections:
${this.getMetrics().sections.map(s =>
    `  ${s.label}: ${s.duration.toFixed(2)}ms (avg: ${s.averageTime.toFixed(2)}ms, calls: ${s.calls})`
).join('\n')}
        `.trim();
    }
}

/**
 * Frame time graph renderer
 */
export class FrameTimeGraph {
    constructor(width, height, maxSamples = 120) {
        this.width = width;
        this.height = height;
        this.maxSamples = maxSamples;
        this.samples = [];
        this.maxFrameTime = 16.67; // 60 FPS target
    }

    /**
     * Add a frame time sample
     */
    addSample(frameTime) {
        this.samples.push(frameTime);
        if (this.samples.length > this.maxSamples) {
            this.samples.shift();
        }
    }

    /**
     * Generate graph data as image
     */
    generateGraph() {
        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');

        // Background
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, this.width, this.height);

        // Grid lines
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;

        // Horizontal grid (time markers)
        const timeSteps = [16.67, 33.33]; // 60fps, 30fps
        timeSteps.forEach(time => {
            const y = this.height - (time / this.maxFrameTime) * this.height;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.width, y);
            ctx.stroke();
        });

        if (this.samples.length < 2) return canvas;

        // Draw graph line
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 2;
        ctx.beginPath();

        const xStep = this.width / this.maxSamples;
        this.samples.forEach((sample, i) => {
            const x = i * xStep;
            const normalizedValue = Math.min(sample / this.maxFrameTime, 1);
            const y = this.height - normalizedValue * this.height;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        // Fill area under curve
        ctx.lineTo(this.samples.length * xStep, this.height);
        ctx.lineTo(0, this.height);
        ctx.closePath();
        ctx.fillStyle = 'rgba(76, 175, 80, 0.2)';
        ctx.fill();

        return canvas;
    }

    /**
     * Get image data for upload to GPU
     */
    getImageData() {
        const canvas = this.generateGraph();
        const ctx = canvas.getContext('2d');
        return ctx.getImageData(0, 0, this.width, this.height);
    }
}

/**
 * Simple memory monitor
 */
export class MemoryMonitor {
    constructor() {
        this.samples = [];
        this.maxSamples = 60;
    }

    /**
     * Sample current memory usage
     */
    sample() {
        if (performance.memory) {
            this.samples.push({
                used: performance.memory.usedJSHeapSize / 1048576, // MB
                total: performance.memory.totalJSHeapSize / 1048576,
                limit: performance.memory.jsHeapSizeLimit / 1048576,
                timestamp: performance.now()
            });

            if (this.samples.length > this.maxSamples) {
                this.samples.shift();
            }
        }
    }

    /**
     * Get memory statistics
     */
    getStats() {
        if (this.samples.length === 0 || !performance.memory) {
            return null;
        }

        const latest = this.samples[this.samples.length - 1];
        const usedSamples = this.samples.map(s => s.used);

        return {
            used: latest.used,
            total: latest.total,
            limit: latest.limit,
            average: usedSamples.reduce((a, b) => a + b, 0) / usedSamples.length,
            min: Math.min(...usedSamples),
            max: Math.max(...usedSamples)
        };
    }
}
