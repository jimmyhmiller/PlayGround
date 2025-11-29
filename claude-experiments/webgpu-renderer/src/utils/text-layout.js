/**
 * Text layout and measurement utilities
 */

export class TextMeasurer {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.cache = new Map();
    }

    /**
     * Measure text dimensions
     */
    measureText(text, fontSize, fontFamily, fontWeight = 'normal') {
        const cacheKey = `${text}-${fontSize}-${fontFamily}-${fontWeight}`;

        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        this.ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;
        const metrics = this.ctx.measureText(text);

        const result = {
            width: metrics.width,
            height: fontSize * 1.2, // Approximation
            ascent: metrics.actualBoundingBoxAscent,
            descent: metrics.actualBoundingBoxDescent,
            actualWidth: metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight,
            actualHeight: metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent
        };

        this.cache.set(cacheKey, result);
        return result;
    }

    /**
     * Calculate line breaks for text that needs to wrap
     */
    wrapText(text, maxWidth, fontSize, fontFamily, fontWeight = 'normal') {
        this.ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;

        const words = text.split(' ');
        const lines = [];
        let currentLine = '';

        for (const word of words) {
            const testLine = currentLine + (currentLine ? ' ' : '') + word;
            const metrics = this.ctx.measureText(testLine);

            if (metrics.width > maxWidth && currentLine) {
                lines.push(currentLine);
                currentLine = word;
            } else {
                currentLine = testLine;
            }
        }

        if (currentLine) {
            lines.push(currentLine);
        }

        return lines;
    }

    /**
     * Calculate bounding box for multi-line text
     */
    measureMultilineText(text, maxWidth, fontSize, fontFamily, fontWeight = 'normal', lineHeight = 1.2) {
        const lines = this.wrapText(text, maxWidth, fontSize, fontFamily, fontWeight);
        const lineHeightPx = fontSize * lineHeight;

        let maxLineWidth = 0;
        for (const line of lines) {
            const measurement = this.measureText(line, fontSize, fontFamily, fontWeight);
            maxLineWidth = Math.max(maxLineWidth, measurement.width);
        }

        return {
            width: maxLineWidth,
            height: lines.length * lineHeightPx,
            lineCount: lines.length,
            lines
        };
    }

    /**
     * Clear the measurement cache
     */
    clearCache() {
        this.cache.clear();
    }
}

/**
 * Text alignment utilities
 */
export const TextAlign = {
    Left: 'left',
    Center: 'center',
    Right: 'right'
};

export const VerticalAlign = {
    Top: 'top',
    Middle: 'middle',
    Bottom: 'bottom'
};

/**
 * Calculate position for aligned text
 */
export function getAlignedTextPosition(
    textWidth,
    textHeight,
    containerX,
    containerY,
    containerWidth,
    containerHeight,
    horizontalAlign = TextAlign.Left,
    verticalAlign = VerticalAlign.Top
) {
    let x = containerX;
    let y = containerY;

    // Horizontal alignment
    switch (horizontalAlign) {
        case TextAlign.Center:
            x = containerX + (containerWidth - textWidth) / 2;
            break;
        case TextAlign.Right:
            x = containerX + containerWidth - textWidth;
            break;
        default: // Left
            x = containerX;
            break;
    }

    // Vertical alignment
    switch (verticalAlign) {
        case VerticalAlign.Middle:
            y = containerY + (containerHeight - textHeight) / 2;
            break;
        case VerticalAlign.Bottom:
            y = containerY + containerHeight - textHeight;
            break;
        default: // Top
            y = containerY;
            break;
    }

    return { x, y };
}

/**
 * Truncate text with ellipsis if it exceeds max width
 */
export function truncateText(text, maxWidth, fontSize, fontFamily, fontWeight = 'normal', ellipsis = '...') {
    const measurer = new TextMeasurer();
    const fullMeasurement = measurer.measureText(text, fontSize, fontFamily, fontWeight);

    if (fullMeasurement.width <= maxWidth) {
        return text;
    }

    const ellipsisMeasurement = measurer.measureText(ellipsis, fontSize, fontFamily, fontWeight);
    const availableWidth = maxWidth - ellipsisMeasurement.width;

    let truncated = text;
    while (truncated.length > 0) {
        const measurement = measurer.measureText(truncated, fontSize, fontFamily, fontWeight);
        if (measurement.width <= availableWidth) {
            return truncated + ellipsis;
        }
        truncated = truncated.slice(0, -1);
    }

    return ellipsis;
}
