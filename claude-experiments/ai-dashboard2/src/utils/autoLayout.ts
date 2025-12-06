import type { WidgetConfig } from '../types/dashboard';

export interface LayoutPosition {
  x: number;
  y: number;
}

export interface WidgetBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Get the bounds of a widget, normalizing width/height vs w/h properties
 */
function getWidgetBounds(widget: WidgetConfig): WidgetBounds {
  const x = widget.x ?? 0;
  const y = widget.y ?? 0;
  const width = widget.width ?? widget.w ?? 200;
  const height = widget.height ?? widget.h ?? 200;

  return { x, y, width, height };
}

/**
 * Check if two rectangles overlap
 */
function rectsOverlap(a: WidgetBounds, b: WidgetBounds): boolean {
  return !(
    a.x + a.width <= b.x ||
    b.x + b.width <= a.x ||
    a.y + a.height <= b.y ||
    b.y + b.height <= a.y
  );
}

/**
 * Check if a position is available (doesn't overlap with any existing widgets)
 */
function isPositionAvailable(
  widgets: WidgetConfig[],
  candidateBounds: WidgetBounds
): boolean {
  for (const widget of widgets) {
    const bounds = getWidgetBounds(widget);
    if (rectsOverlap(bounds, candidateBounds)) {
      return false;
    }
  }
  return true;
}

/**
 * Find the first available space in the grid for a widget.
 * Searches in reading order (top-to-bottom, left-to-right) with configurable step size.
 *
 * @param widgets - Existing widgets to check against
 * @param newWidget - Widget to place (needs width/height or w/h properties)
 * @param gridSize - Grid cell size in pixels (default: 16)
 * @param maxWidth - Maximum width to search (default: 2000)
 * @param maxHeight - Maximum height to search (default: 10000)
 * @returns Position where the widget can be placed without overlapping
 */
export function findFirstAvailableSpace(
  widgets: WidgetConfig[],
  newWidget: WidgetConfig,
  gridSize: number = 16,
  maxWidth: number = 2000,
  maxHeight: number = 10000
): LayoutPosition {
  const newBounds = getWidgetBounds(newWidget);

  // Calculate the farthest right and bottom edges of existing widgets
  let maxX = 0;
  let maxY = 0;
  for (const widget of widgets) {
    const bounds = getWidgetBounds(widget);
    maxX = Math.max(maxX, bounds.x + bounds.width);
    maxY = Math.max(maxY, bounds.y + bounds.height);
  }

  // Search grid in reading order (top-to-bottom, left-to-right)
  // Use grid-aligned positions for cleaner layout
  for (let y = 0; y < maxHeight; y += gridSize) {
    for (let x = 0; x < maxWidth; x += gridSize) {
      const candidateBounds: WidgetBounds = {
        x,
        y,
        width: newBounds.width,
        height: newBounds.height,
      };

      if (isPositionAvailable(widgets, candidateBounds)) {
        return { x, y };
      }
    }
  }

  // Fallback: place at bottom-right of existing content
  // This should rarely happen, but ensures we always find a position
  return {
    x: Math.max(0, maxX + gridSize),
    y: Math.max(0, maxY + gridSize),
  };
}

/**
 * Calculate whether a dragged widget overlaps with a target widget
 * Used for detecting drop zones during drag operations
 */
export function checkOverlap(
  draggedBounds: WidgetBounds,
  targetBounds: WidgetBounds,
  threshold: number = 0.5
): boolean {
  if (!rectsOverlap(draggedBounds, targetBounds)) {
    return false;
  }

  // Calculate overlap area
  const overlapX1 = Math.max(draggedBounds.x, targetBounds.x);
  const overlapY1 = Math.max(draggedBounds.y, targetBounds.y);
  const overlapX2 = Math.min(
    draggedBounds.x + draggedBounds.width,
    targetBounds.x + targetBounds.width
  );
  const overlapY2 = Math.min(
    draggedBounds.y + draggedBounds.height,
    targetBounds.y + targetBounds.height
  );

  const overlapArea = (overlapX2 - overlapX1) * (overlapY2 - overlapY1);
  const draggedArea = draggedBounds.width * draggedBounds.height;

  // Return true if overlap is greater than threshold (default 50%)
  return overlapArea / draggedArea >= threshold;
}

/**
 * Snap a position to the nearest grid cell
 */
export function snapToGrid(value: number, gridSize: number): number {
  return Math.round(value / gridSize) * gridSize;
}

/**
 * Get widget bounds from a DOM element
 */
export function getBoundsFromElement(element: HTMLElement): WidgetBounds {
  const rect = element.getBoundingClientRect();
  return {
    x: rect.left,
    y: rect.top,
    width: rect.width,
    height: rect.height,
  };
}
