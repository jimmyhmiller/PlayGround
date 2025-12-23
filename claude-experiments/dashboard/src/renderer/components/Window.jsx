import { memo, useState, useRef, useCallback, useEffect } from 'react';

/**
 * Draggable, resizable window component
 * Uses CSS theme variables for styling
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
}) {
  // Local state for optimistic updates during drag/resize
  const [localPosition, setLocalPosition] = useState({ x, y });
  const [localSize, setLocalSize] = useState({ width, height });
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);

  const dragOffset = useRef({ x: 0, y: 0 });
  const windowRef = useRef(null);
  const updateTimeoutRef = useRef(null);

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

  const handleMouseDown = useCallback((e) => {
    if (e.target.closest('.window-resize-handle')) return;

    onFocus?.(id);
    setIsDragging(true);

    const startPos = positionRef.current;
    dragOffset.current = {
      x: e.clientX - startPos.x,
      y: e.clientY - startPos.y,
    };

    let hasMoved = false;

    const handleMouseMove = (e) => {
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

  const handleResizeMouseDown = useCallback((e) => {
    e.stopPropagation();
    onFocus?.(id);
    setIsResizing(true);

    const startX = e.clientX;
    const startY = e.clientY;
    const startSize = sizeRef.current;

    let hasResized = false;

    const handleMouseMove = (e) => {
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

  const handleWindowClick = useCallback((e) => {
    // Don't fire focus again if we clicked on title bar (already handled in mouseDown)
    if (e.target.closest('.window-titlebar')) return;
    onFocus?.(id);
  }, [id, onFocus]);

  return (
    <div
      ref={windowRef}
      className="window"
      onClick={handleWindowClick}
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
        borderRadius: 'var(--theme-radius-lg)',
        boxShadow: isFocused
          ? 'var(--theme-window-shadow)'
          : 'var(--theme-window-shadow-inactive)',
        border: '1px solid var(--theme-window-border)',
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
            ? 'var(--theme-window-header)'
            : 'var(--theme-window-header-inactive)',
          borderBottom: '1px solid var(--theme-border-primary)',
          cursor: 'move',
          flexShrink: 0,
          transition: 'background var(--theme-transition-fast)',
        }}
      >
        <span
          className="window-title"
          style={{
            fontSize: 'var(--theme-font-size-md)',
            fontWeight: 500,
            color: isFocused
              ? 'var(--theme-text-primary)'
              : 'var(--theme-text-muted)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
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
          onMouseEnter={(e) => e.target.style.color = 'var(--theme-accent-error)'}
          onMouseLeave={(e) => e.target.style.color = 'var(--theme-text-muted)'}
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
