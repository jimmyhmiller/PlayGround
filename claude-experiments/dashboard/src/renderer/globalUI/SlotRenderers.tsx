/**
 * Slot Renderers
 *
 * Built-in slot container components for common positioning patterns.
 * Each renderer handles the CSS positioning for its slot type.
 */

import { memo, useState, useEffect, useCallback, type ReactNode, type ReactElement, type CSSProperties } from 'react';
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
  const [isPinned, setIsPinned] = useState(true);
  const [isHovered, setIsHovered] = useState(false);
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [contextMenuPos, setContextMenuPos] = useState({ x: 0, y: 0 });

  // Listen for panel pin toggle events using direct subscription
  useEffect(() => {
    const unsubscribe = window.eventAPI.subscribe(`globalUI.${side}Panel.toggle`, (event) => {
      const pinned = event.payload as boolean;
      setIsPinned(pinned);
      // Clear hover state when unpinning to ensure immediate collapse
      if (!pinned) {
        setIsHovered(false);
      }
    });
    return unsubscribe;
  }, [side]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    
    // Menu dimensions (approximate)
    const menuWidth = 150;
    const menuHeight = 40;
    
    // Get viewport dimensions
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Calculate position with boundary checks
    let x = e.clientX;
    let y = e.clientY;
    
    // Adjust if menu would overflow right edge
    if (x + menuWidth > viewportWidth) {
      x = viewportWidth - menuWidth - 10;
    }
    
    // Adjust if menu would overflow bottom edge
    if (y + menuHeight > viewportHeight) {
      y = viewportHeight - menuHeight - 10;
    }
    
    // Ensure menu doesn't go off left or top edges
    x = Math.max(10, x);
    y = Math.max(10, y);
    
    setContextMenuPos({ x, y });
    setShowContextMenu(true);
  }, []);

  const handleToggleSide = useCallback(() => {
    setShowContextMenu(false);
    const newSide = side === 'left' ? 'right' : 'left';
    window.eventAPI.emit('globalUI.panel.moveTo', newSide);
  }, [side]);

  // Close context menu on click outside
  useEffect(() => {
    if (!showContextMenu) return;
    const handleClick = () => setShowContextMenu(false);
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [showContextMenu]);

  const isVisible = isPinned || isHovered;
  const collapsedWidth = 8; // Thin bar when hidden
  const trafficLightPadding = 40; // Space for macOS traffic lights on both sides

  return (
    <>
      {/* Hover trigger area when hidden */}
      {!isPinned && (
        <div
          className={`global-ui-panel-trigger global-ui-${side}-panel-trigger`}
          style={{
            position: 'fixed',
            top: 0,
            bottom: 0,
            [isLeft ? 'left' : 'right']: 0,
            width: collapsedWidth,
            zIndex: zIndex - 1,
            pointerEvents: 'auto',
            background: 'var(--theme-bg-tertiary, #252538)',
            borderLeft: isLeft ? 'none' : '1px solid var(--theme-border-primary, #333)',
            borderRight: isLeft ? '1px solid var(--theme-border-primary, #333)' : 'none',
          }}
          onMouseEnter={() => setIsHovered(true)}
        />
      )}

      {/* Main panel */}
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
          paddingTop: trafficLightPadding > 0 ? `${trafficLightPadding}px` : 'var(--theme-spacing-md, 16px)',
          background: 'var(--theme-bg-secondary, #1a1a2e)',
          borderLeft: isLeft ? 'none' : '1px solid var(--theme-border-primary, #333)',
          borderRight: isLeft ? '1px solid var(--theme-border-primary, #333)' : 'none',
          pointerEvents: 'auto',
          overflowY: 'auto',
          transform: isVisible ? 'translateX(0)' : `translateX(${isLeft ? '-' : ''}100%)`,
          transition: 'transform 0.3s ease',
        }}
        onMouseEnter={() => !isPinned && setIsHovered(true)}
        onMouseLeave={() => !isPinned && setIsHovered(false)}
        onContextMenu={handleContextMenu}
      >
        {children}
      </div>

      {/* Context menu */}
      {showContextMenu && (
        <div
          style={{
            position: 'fixed',
            left: contextMenuPos.x,
            top: contextMenuPos.y,
            zIndex: zIndex + 10,
            background: 'var(--theme-bg-elevated, #252540)',
            border: '1px solid var(--theme-border-primary, #333)',
            borderRadius: '4px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            minWidth: 120,
            padding: '4px 0',
            pointerEvents: 'auto',
          }}
        >
          <button
            onClick={handleToggleSide}
            style={{
              width: '100%',
              padding: '6px 12px',
              background: 'none',
              border: 'none',
              color: 'var(--theme-text-primary, #e0e0e0)',
              textAlign: 'left',
              cursor: 'pointer',
              fontSize: '0.85em',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'var(--theme-bg-tertiary, #2a2a3e)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'none';
            }}
          >
            Tabs on {side === 'left' ? 'Right' : 'Left'}
          </button>
        </div>
      )}
    </>
  );
});
