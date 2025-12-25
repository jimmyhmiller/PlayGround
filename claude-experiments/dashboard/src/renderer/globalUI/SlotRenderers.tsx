/**
 * Slot Renderers
 *
 * Built-in slot container components for common positioning patterns.
 * Each renderer handles the CSS positioning for its slot type.
 */

import { memo, type ReactNode, type ReactElement, type CSSProperties } from 'react';
import type { CornerPosition, BarEdge, PanelSide } from '../../types/globalUI';

// ========== Corner Slot ==========

interface CornerSlotProps {
  corner: CornerPosition;
  children: ReactNode;
  zIndex?: number;
}

const cornerPositionStyles: Record<CornerPosition, CSSProperties> = {
  'top-left': { top: 0, left: 0 },
  'top-right': { top: 0, right: 0 },
  'bottom-left': { bottom: 0, left: 0 },
  'bottom-right': { bottom: 0, right: 0 },
};

/**
 * Corner slot - positioned in one of the four screen corners
 */
export const CornerSlot = memo(function CornerSlot({
  corner,
  children,
  zIndex = 200,
}: CornerSlotProps): ReactElement {
  return (
    <div
      className={`global-ui-corner global-ui-${corner}`}
      style={{
        position: 'fixed',
        ...cornerPositionStyles[corner],
        zIndex,
        padding: 'var(--theme-spacing-md, 16px)',
        pointerEvents: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--theme-spacing-sm, 8px)',
      }}
    >
      {children}
    </div>
  );
});

// ========== Bar Slot ==========

interface BarSlotProps {
  edge: BarEdge;
  children: ReactNode;
  zIndex?: number;
}

/**
 * Bar slot - full-width bar at top or bottom edge
 */
export const BarSlot = memo(function BarSlot({
  edge,
  children,
  zIndex = 100,
}: BarSlotProps): ReactElement {
  const isTop = edge === 'top';

  return (
    <div
      className={`global-ui-bar global-ui-${edge}-bar`}
      style={{
        position: 'fixed',
        left: 0,
        right: 0,
        [isTop ? 'top' : 'bottom']: 0,
        zIndex,
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--theme-spacing-md, 16px)',
        padding: 'var(--theme-spacing-xs, 4px) var(--theme-spacing-md, 16px)',
        background: 'var(--theme-bg-secondary, #1a1a2e)',
        borderTop: isTop ? 'none' : '1px solid var(--theme-border-primary, #333)',
        borderBottom: isTop ? '1px solid var(--theme-border-primary, #333)' : 'none',
        pointerEvents: 'auto',
      }}
    >
      {children}
    </div>
  );
});

// ========== Panel Slot ==========

interface PanelSlotProps {
  side: PanelSide;
  width?: number;
  collapsible?: boolean;
  children: ReactNode;
  zIndex?: number;
}

/**
 * Panel slot - side panel on left or right edge
 */
export const PanelSlot = memo(function PanelSlot({
  side,
  width = 200,
  children,
  zIndex = 100,
}: PanelSlotProps): ReactElement {
  const isLeft = side === 'left';

  return (
    <div
      className={`global-ui-panel global-ui-${side}-panel`}
      style={{
        position: 'fixed',
        top: 0,
        bottom: 0,
        [isLeft ? 'left' : 'right']: 0,
        width,
        zIndex,
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--theme-spacing-sm, 8px)',
        padding: 'var(--theme-spacing-md, 16px)',
        background: 'var(--theme-bg-secondary, #1a1a2e)',
        borderLeft: isLeft ? 'none' : '1px solid var(--theme-border-primary, #333)',
        borderRight: isLeft ? '1px solid var(--theme-border-primary, #333)' : 'none',
        pointerEvents: 'auto',
        overflowY: 'auto',
      }}
    >
      {children}
    </div>
  );
});
