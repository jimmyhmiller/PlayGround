/**
 * Global UI Module
 *
 * Exports the global UI rendering system.
 */

// Main renderer
export { GlobalUIRenderer } from './GlobalUIRenderer';

// Slot renderers (for custom slot implementations)
export { CornerSlot, BarSlot, PanelSlot } from './SlotRenderers';
