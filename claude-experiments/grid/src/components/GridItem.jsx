import React, { useEffect, useRef, useState, useId } from 'react';
import { useGrid } from './Grid';

// Helper function to measure intrinsic content size
const measureIntrinsicSize = (element) => {
  const contentWrapper = element?.querySelector('.grid-item-content');
  if (!contentWrapper) return null;

  // Create a temporary clone to measure actual content size
  const clone = contentWrapper.cloneNode(true);

  // Style the clone for measurement
  Object.assign(clone.style, {
    position: 'absolute',
    visibility: 'hidden',
    left: '-9999px',
    top: '-9999px',
    width: 'auto',
    height: 'auto',
    maxWidth: 'none',
    maxHeight: 'none',
    overflow: 'visible',
    display: 'block'
  });

  document.body.appendChild(clone);

  // Measure the clone's natural size
  const rect = clone.getBoundingClientRect();
  const intrinsicWidth = rect.width;
  const intrinsicHeight = rect.height;

  document.body.removeChild(clone);

  return { width: intrinsicWidth, height: intrinsicHeight };
};

export const GridItem = ({
  children,
  x: initialX = 0,
  y: initialY = 0,
  width: initialWidth,
  height: initialHeight,
  minWidth,
  minHeight,
  resizable = true,
  draggable = true,
  enforceContentSize = false,
  onDragStart,
  onDrag,
  onDragEnd,
  onResize,
  className = '',
  style = {}
}) => {
  const id = useId();
  const { cellSize, snapToGrid, snapSize, registerItem, unregisterItem, updateItem, getNextZIndex } = useGrid();
  const itemRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [position, setPosition] = useState({ x: initialX, y: initialY });
  const [size, setSize] = useState({
    width: initialWidth || cellSize,
    height: initialHeight || cellSize
  });
  const [intrinsicSize, setIntrinsicSize] = useState(null);
  const [zIndex, setZIndex] = useState(1);
  const dragStartPos = useRef({ x: 0, y: 0 });
  const resizeStartPos = useRef({ x: 0, y: 0 });
  const resizeStartSize = useRef({ width: 0, height: 0 });
  const isManualResizeRef = useRef(false);

  useEffect(() => {
    registerItem(id, { id, x: position.x, y: position.y, width: size.width, height: size.height });
    return () => unregisterItem(id);
  }, [id, registerItem, unregisterItem]);

  useEffect(() => {
    updateItem(id, { x: position.x, y: position.y, width: size.width, height: size.height });
  }, [id, position.x, position.y, size.width, size.height, updateItem]);

  // Measure intrinsic size at mount (only if enforceContentSize is enabled)
  useEffect(() => {
    if (enforceContentSize && itemRef.current) {
      const measured = measureIntrinsicSize(itemRef.current);
      if (measured) {
        setIntrinsicSize(measured);
      }
    } else if (!enforceContentSize) {
      setIntrinsicSize(null);
    }
  }, [enforceContentSize]);

  // Use ResizeObserver to detect when content changes (not manual resizes)
  useEffect(() => {
    if (!enforceContentSize) return;

    const contentElement = itemRef.current?.querySelector('.grid-item-content');
    if (!contentElement) return;

    const resizeObserver = new ResizeObserver(() => {
      // Only remeasure if this isn't a manual resize
      if (!isManualResizeRef.current) {
        const measured = measureIntrinsicSize(itemRef.current);
        if (measured) {
          setIntrinsicSize(measured);
        }
      }
    });

    resizeObserver.observe(contentElement);
    return () => resizeObserver.disconnect();
  }, [enforceContentSize]);

  const handleMouseDownDrag = (e) => {
    if (!draggable || isResizing) return;

    e.stopPropagation();
    setIsDragging(true);

    // Bring to front by getting next z-index from grid context
    setZIndex(getNextZIndex());

    dragStartPos.current = {
      x: e.clientX - position.x,
      y: e.clientY - position.y
    };

    if (onDragStart) {
      onDragStart({ x: position.x, y: position.y });
    }
  };

  const handleMouseDownResize = (e) => {
    if (!resizable) return;

    e.stopPropagation();
    setIsResizing(true);
    isManualResizeRef.current = true;

    resizeStartPos.current = { x: e.clientX, y: e.clientY };
    resizeStartSize.current = { width: size.width, height: size.height };
  };

  useEffect(() => {
    if (!isDragging && !isResizing) return;

    const handleMouseMove = (e) => {
      if (isDragging) {
        const newX = e.clientX - dragStartPos.current.x;
        const newY = e.clientY - dragStartPos.current.y;

        const snappedX = snapToGrid(newX, 'x');
        const snappedY = snapToGrid(newY, 'y');

        setPosition({ x: snappedX, y: snappedY });

        if (onDrag) {
          onDrag({ x: snappedX, y: snappedY });
        }
      }

      if (isResizing) {
        const deltaX = e.clientX - resizeStartPos.current.x;
        const deltaY = e.clientY - resizeStartPos.current.y;

        let newWidth = resizeStartSize.current.width + deltaX;
        let newHeight = resizeStartSize.current.height + deltaY;

        // Apply intrinsic size constraints FIRST (prevents resizing smaller than content)
        if (intrinsicSize) {
          newWidth = Math.max(newWidth, intrinsicSize.width);
          newHeight = Math.max(newHeight, intrinsicSize.height);
        }

        // Then apply user-specified minimum constraints
        if (minWidth !== undefined) {
          newWidth = Math.max(newWidth, minWidth);
        }
        if (minHeight !== undefined) {
          newHeight = Math.max(newHeight, minHeight);
        }

        // Finally, snap size to cellSize multiples (not cellSize + gap)
        newWidth = snapSize(newWidth, 'x');
        newHeight = snapSize(newHeight, 'y');

        setSize({ width: newWidth, height: newHeight });

        if (onResize) {
          onResize({ width: newWidth, height: newHeight });
        }
      }
    };

    const handleMouseUp = () => {
      if (isDragging) {
        setIsDragging(false);
        if (onDragEnd) {
          onDragEnd({ x: position.x, y: position.y });
        }
      }
      if (isResizing) {
        setIsResizing(false);
        // Clear the manual resize flag after a short delay
        setTimeout(() => {
          isManualResizeRef.current = false;
        }, 100);
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, isResizing, snapToGrid, position.x, position.y, minWidth, minHeight, intrinsicSize, onDrag, onDragEnd, onResize]);

  const itemStyle = {
    position: 'absolute',
    left: `${position.x}px`,
    top: `${position.y}px`,
    width: `${size.width}px`,
    height: `${size.height}px`,
    zIndex: zIndex,
    cursor: isDragging ? 'grabbing' : (draggable ? 'grab' : 'default'),
    userSelect: 'none',
    boxSizing: 'border-box',
    ...style
  };

  return (
    <div
      ref={itemRef}
      className={`grid-item ${className} ${isDragging ? 'dragging' : ''} ${isResizing ? 'resizing' : ''}`}
      style={itemStyle}
      onMouseDown={handleMouseDownDrag}
    >
      <div className="grid-item-content" style={{ width: '100%', height: '100%', overflow: 'hidden' }}>
        {children}
      </div>
      {resizable && (
        <div
          className="resize-handle"
          onMouseDown={handleMouseDownResize}
          style={{
            position: 'absolute',
            right: 0,
            bottom: 0,
            width: '12px',
            height: '12px',
            cursor: 'se-resize',
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '0 0 4px 0'
          }}
        />
      )}
    </div>
  );
};
