import { memo, useState, useRef, useCallback, useEffect, ReactNode } from 'react';
import type { WindowUpdates } from '../../types/components';

export interface WindowProps {
  id: string;
  title: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  minWidth?: number;
  minHeight?: number;
  onClose?: (id: string) => void;
  onFocus?: (id: string) => void;
  onUpdate?: (id: string, updates: WindowUpdates) => void;
  zIndex?: number;
  isFocused?: boolean;
  children?: ReactNode;
}

interface Position {
  x: number;
  y: number;
}

interface Size {
  width: number;
  height: number;
}

/**
 * Draggable, resizable window component
 * Uses CSS theme variables for styling
 *
 * Supports two modes:
 * - Normal: Traditional window with title bar and chrome
 * - Chromeless: Bare widget style (like ai-dashboard2) - controlled by --theme-window-chromeless
 *
 * Implements optimistic updates pattern:
 * - Position/size from props (backend state)
 * - Local state for smooth dragging
 * - Debounced sync back to backend
 */
const Window = memo(function Window({
  id,
  title,
  x = 100,
  y = 100,
  width = 400,
  height = 300,
  minWidth = 200,
  minHeight = 150,
  onClose,
  onFocus,
  onUpdate,
  zIndex = 1,
  isFocused = false,
  children,
}: WindowProps) {
  // Local state for optimistic updates during drag/resize
  const [localPosition, setLocalPosition] = useState<Position>({ x, y });
  const [localSize, setLocalSize] = useState<Size>({ width, height });
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [isChromeless, setIsChromeless] = useState(false);

  const dragOffset = useRef({ x: 0, y: 0 });
  const windowRef = useRef<HTMLDivElement>(null);
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Check for chromeless mode from CSS variable
  useEffect(() => {
    const checkChromeless = () => {
      const value = getComputedStyle(document.documentElement)
        .getPropertyValue('--theme-window-chromeless')
        .trim();
      setIsChromeless(value === '1');
    };
    checkChromeless();
    // Re-check when theme might change
    const observer = new MutationObserver(checkChromeless);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'style'] });
    return () => observer.disconnect();
  }, []);

  // Use refs to track current values for use in event handlers
  const positionRef = useRef(localPosition);
  const sizeRef = useRef(localSize);

  useEffect(() => {
    positionRef.current = localPosition;
  }, [localPosition]);

  useEffect(() => {
    sizeRef.current = localSize;
  }, [localSize]);

  // Sync local state with props when not actively manipulating
  useEffect(() => {
    if (!isDragging) {
      setLocalPosition({ x, y });
    }
  }, [x, y, isDragging]);

  useEffect(() => {
    if (!isResizing) {
      setLocalSize({ width, height });
    }
  }, [width, height, isResizing]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.window-resize-handle')) return;

    onFocus?.(id);
    setIsDragging(true);

    const startPos = positionRef.current;
    dragOffset.current = {
      x: e.clientX - startPos.x,
      y: e.clientY - startPos.y,
    };

    let hasMoved = false;

    const handleMouseMove = (e: MouseEvent) => {
      hasMoved = true;
      const newPosition = {
        x: Math.max(0, e.clientX - dragOffset.current.x),
        y: Math.max(0, e.clientY - dragOffset.current.y),
      };
      setLocalPosition(newPosition);
      positionRef.current = newPosition;

      // Debounced update to backend
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
      updateTimeoutRef.current = setTimeout(() => {
        onUpdate?.(id, newPosition);
      }, 50);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
      // Only sync to backend if we actually moved
      if (hasMoved) {
        onUpdate?.(id, positionRef.current);
      }
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [id, onFocus, onUpdate]);

  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onFocus?.(id);
    setIsResizing(true);

    const startX = e.clientX;
    const startY = e.clientY;
    const startSize = sizeRef.current;

    let hasResized = false;

    const handleMouseMove = (e: MouseEvent) => {
      hasResized = true;
      const newSize = {
        width: Math.max(minWidth, startSize.width + (e.clientX - startX)),
        height: Math.max(minHeight, startSize.height + (e.clientY - startY)),
      };
      setLocalSize(newSize);
      sizeRef.current = newSize;

      // Debounced update to backend
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
      updateTimeoutRef.current = setTimeout(() => {
        onUpdate?.(id, newSize);
      }, 50);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
      // Only sync to backend if we actually resized
      if (hasResized) {
        onUpdate?.(id, sizeRef.current);
      }
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [id, minWidth, minHeight, onFocus, onUpdate]);

  const handleWindowClick = useCallback((e: React.MouseEvent) => {
    // Don't fire focus again if we clicked on title bar (already handled in mouseDown)
    if ((e.target as HTMLElement).closest('.window-titlebar')) return;
    onFocus?.(id);
  }, [id, onFocus]);

  const handleWindowMouseDown = useCallback((e: React.MouseEvent) => {
    // Ctrl + Option + Cmd click to close window
    if (e.ctrlKey && e.altKey && e.metaKey) {
      e.preventDefault();
      e.stopPropagation();
      onClose?.(id);
      return;
    }
  }, [id, onClose]);

  // Chromeless mode - bare widget style like ai-dashboard2
  if (isChromeless) {
    return (
      <div
        ref={windowRef}
        className="window window-chromeless"
        onClick={handleWindowClick}
        onMouseDown={handleWindowMouseDown}
        style={{
          position: 'absolute',
          left: localPosition.x,
          top: localPosition.y,
          width: localSize.width,
          height: localSize.height,
          zIndex,
          display: 'flex',
          flexDirection: 'column',
          background: 'var(--theme-window-bg)',
          borderRadius: 'var(--theme-window-radius)',
          boxShadow: isFocused
            ? 'var(--theme-window-shadow)'
            : 'var(--theme-window-shadow-inactive)',
          borderWidth: 'var(--theme-window-border-width)',
          borderStyle: 'var(--theme-window-border-style)',
          borderColor: 'var(--theme-window-border-color)',
          clipPath: 'var(--theme-window-clip-path)',
          overflow: 'hidden',
          userSelect: isDragging || isResizing ? 'none' : 'auto',
          transition: 'box-shadow var(--theme-transition-fast), opacity 0.2s',
        }}
      >
        {/* Drag handle - extends across top including padding area */}
        <div
          className="widget-drag-handle"
          onMouseDown={handleMouseDown}
          style={{
            padding: '15px 20px 0 20px',
          }}
        >
          <div
            className="widget-label"
            style={{
              fontSize: '0.65rem',
              textTransform: 'uppercase',
              color: 'var(--theme-text-muted)',
              marginBottom: '10px',
              display: 'flex',
              justifyContent: 'space-between',
              fontFamily: 'var(--theme-font-family)',
              letterSpacing: '1px',
            }}
          >
            <span>{title}</span>
          </div>
        </div>

        {/* Content */}
        <div
          className="window-content"
          style={{
            flex: 1,
            overflow: 'auto',
            color: 'var(--theme-text-primary)',
            fontFamily: 'var(--theme-font-family)',
            padding: '0 20px 15px 20px',
          }}
        >
          {children}
        </div>

        {/* Subtle resize handle for chromeless mode */}
        <div
          className="window-resize-handle"
          onMouseDown={handleResizeMouseDown}
          style={{
            position: 'absolute',
            right: 0,
            bottom: 0,
            width: 12,
            height: 12,
            cursor: 'se-resize',
            opacity: 0.3,
          }}
        />
      </div>
    );
  }

  // Normal mode - traditional window with chrome
  return (
    <div
      ref={windowRef}
      className="window"
      onClick={handleWindowClick}
      onMouseDown={handleWindowMouseDown}
      style={{
        position: 'absolute',
        left: localPosition.x,
        top: localPosition.y,
        width: localSize.width,
        height: localSize.height,
        zIndex,
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--theme-window-bg)',
        borderRadius: 'var(--theme-window-radius)',
        boxShadow: isFocused
          ? 'var(--theme-window-shadow)'
          : 'var(--theme-window-shadow-inactive)',
        borderWidth: 'var(--theme-window-border-width)',
        borderStyle: 'var(--theme-window-border-style)',
        borderColor: 'var(--theme-window-border-color)',
        clipPath: 'var(--theme-window-clip-path)',
        overflow: 'hidden',
        userSelect: isDragging || isResizing ? 'none' : 'auto',
        transition: 'box-shadow var(--theme-transition-fast)',
      }}
    >
      {/* Title bar */}
      <div
        className="window-titlebar"
        onMouseDown={handleMouseDown}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
          background: isFocused
            ? 'var(--theme-window-header-bg)'
            : 'var(--theme-window-header-inactive-bg)',
          borderBottom: '1px solid var(--theme-border-primary)',
          flexShrink: 0,
          transition: 'background var(--theme-transition-fast)',
        }}
      >
        <span
          className="window-title"
          style={{
            fontSize: 'var(--theme-font-size-md)',
            fontWeight: 'var(--theme-window-title-weight)',
            textTransform: 'var(--theme-window-title-transform)',
            color: isFocused
              ? 'var(--theme-text-primary)'
              : 'var(--theme-text-muted)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            letterSpacing: '0.5px',
          }}
        >
          {title}
        </span>
        <button
          className="window-close"
          onClick={(e) => {
            e.stopPropagation();
            onClose?.(id);
          }}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--theme-text-muted)',
            cursor: 'pointer',
            fontSize: '16px',
            padding: '0 4px',
            lineHeight: 1,
            transition: 'color var(--theme-transition-fast)',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--theme-accent-error)')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--theme-text-muted)')}
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div
        className="window-content"
        style={{
          flex: 1,
          overflow: 'auto',
          background: 'var(--theme-window-bg)',
        }}
      >
        {children}
      </div>

      {/* Resize handle */}
      <div
        className="window-resize-handle"
        onMouseDown={handleResizeMouseDown}
        style={{
          position: 'absolute',
          right: 0,
          bottom: 0,
          width: 16,
          height: 16,
          cursor: 'se-resize',
          background: 'linear-gradient(135deg, transparent 50%, var(--theme-border-secondary) 50%)',
          borderRadius: '0 0 var(--theme-radius-lg) 0',
        }}
      />
    </div>
  );
});

export default Window;
