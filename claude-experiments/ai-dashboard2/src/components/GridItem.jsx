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
  dragMode = 'modifier', // 'full', 'handle', or 'modifier' (default: modifier key required)
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
  const [currentCursor, setCurrentCursor] = useState('grab');
  const resizeEdgeRef = useRef(null); // Which edge(s) are being resized
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
  const resizeStartPosition = useRef({ x: 0, y: 0 });
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

  // Detect which edge(s) the mouse is near
  const getResizeEdge = (e) => {
    if (!resizable || !itemRef.current) return null;

    const rect = itemRef.current.getBoundingClientRect();
    const edgeThreshold = 8; // pixels from edge to trigger resize

    const nearTop = e.clientY - rect.top < edgeThreshold;
    const nearBottom = rect.bottom - e.clientY < edgeThreshold;
    const nearLeft = e.clientX - rect.left < edgeThreshold;
    const nearRight = rect.right - e.clientX < edgeThreshold;

    // Corners (combinations)
    if (nearTop && nearLeft) return 'nw';
    if (nearTop && nearRight) return 'ne';
    if (nearBottom && nearLeft) return 'sw';
    if (nearBottom && nearRight) return 'se';

    // Edges
    if (nearTop) return 'n';
    if (nearBottom) return 's';
    if (nearLeft) return 'w';
    if (nearRight) return 'e';

    return null;
  };

  // Get cursor style for resize edge
  const getCursorForEdge = (edge, modifierPressed = false) => {
    if (!edge) {
      // Only show grab cursor if draggable and conditions are met
      if (!draggable) return 'default';
      if (dragMode === 'modifier' && !modifierPressed) return 'default';
      return 'grab';
    }

    const cursorMap = {
      'n': 'ns-resize',
      's': 'ns-resize',
      'e': 'ew-resize',
      'w': 'ew-resize',
      'ne': 'nesw-resize',
      'sw': 'nesw-resize',
      'nw': 'nwse-resize',
      'se': 'nwse-resize'
    };

    return cursorMap[edge] || 'default';
  };

  // Update cursor on mouse move
  const handleMouseMove = (e) => {
    if (isDragging || isResizing) return;

    const edge = getResizeEdge(e);
    const modifierPressed = e.metaKey || e.ctrlKey;
    const cursor = getCursorForEdge(edge, modifierPressed);
    setCurrentCursor(cursor);
  };

  const handleMouseDown = (e) => {
    const edge = getResizeEdge(e);

    if (edge) {
      // Start resizing
      if (!resizable) return;

      e.stopPropagation();
      setIsResizing(true);
      resizeEdgeRef.current = edge;
      isManualResizeRef.current = true;
      setCurrentCursor(getCursorForEdge(edge));

      resizeStartPos.current = { x: e.clientX, y: e.clientY };
      resizeStartSize.current = { width: size.width, height: size.height };
      resizeStartPosition.current = { x: position.x, y: position.y };
    } else {
      // Start dragging - check drag mode
      if (!draggable || isResizing) return;

      // Check if drag is allowed based on mode
      if (dragMode === 'modifier') {
        // Require Cmd (Mac) or Ctrl (Windows/Linux) key to be held
        if (!e.metaKey && !e.ctrlKey) return;
      } else if (dragMode === 'handle') {
        // Only drag from the drag handle (checked via class name)
        if (!e.target.classList.contains('drag-handle')) return;
      }
      // dragMode === 'full' allows dragging from anywhere (no extra check needed)

      e.stopPropagation();
      setIsDragging(true);
      setCurrentCursor('grabbing');

      // Bring to front by getting next z-index from grid context
      setZIndex(getNextZIndex());

      dragStartPos.current = {
        x: e.clientX - position.x,
        y: e.clientY - position.y
      };

      if (onDragStart) {
        onDragStart({ x: position.x, y: position.y });
      }
    }
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

      if (isResizing && resizeEdgeRef.current) {
        const deltaX = e.clientX - resizeStartPos.current.x;
        const deltaY = e.clientY - resizeStartPos.current.y;

        let newWidth = resizeStartSize.current.width;
        let newHeight = resizeStartSize.current.height;
        let newX = resizeStartPosition.current.x;
        let newY = resizeStartPosition.current.y;

        const edge = resizeEdgeRef.current;

        // Apply deltas based on which edge is being resized
        if (edge.includes('e')) {
          newWidth = resizeStartSize.current.width + deltaX;
        }
        if (edge.includes('w')) {
          newWidth = resizeStartSize.current.width - deltaX;
          newX = resizeStartPosition.current.x + deltaX;
        }
        if (edge.includes('s')) {
          newHeight = resizeStartSize.current.height + deltaY;
        }
        if (edge.includes('n')) {
          newHeight = resizeStartSize.current.height - deltaY;
          newY = resizeStartPosition.current.y + deltaY;
        }

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

        // Snap size to cellSize multiples
        const snappedWidth = snapSize(newWidth, 'x');
        const snappedHeight = snapSize(newHeight, 'y');

        // Adjust position for left/top edges based on size change
        if (edge.includes('w')) {
          newX = resizeStartPosition.current.x + (resizeStartSize.current.width - snappedWidth);
        }
        if (edge.includes('n')) {
          newY = resizeStartPosition.current.y + (resizeStartSize.current.height - snappedHeight);
        }

        // Snap position
        newX = snapToGrid(newX, 'x');
        newY = snapToGrid(newY, 'y');

        setSize({ width: snappedWidth, height: snappedHeight });
        setPosition({ x: newX, y: newY });

        if (onResize) {
          onResize({ width: snappedWidth, height: snappedHeight });
        }
        if (onDrag && (edge.includes('w') || edge.includes('n'))) {
          onDrag({ x: newX, y: newY });
        }
      }
    };

    const handleMouseUp = () => {
      if (isDragging) {
        setIsDragging(false);
        setCurrentCursor(draggable ? 'grab' : 'default');
        if (onDragEnd) {
          onDragEnd({ x: position.x, y: position.y });
        }
      }
      if (isResizing) {
        setIsResizing(false);
        resizeEdgeRef.current = null;
        setCurrentCursor(draggable ? 'grab' : 'default');
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
  }, [isDragging, isResizing, snapToGrid, position.x, position.y, minWidth, minHeight, intrinsicSize, draggable, onDrag, onDragEnd, onResize]);

  const itemStyle = {
    position: 'absolute',
    left: `${position.x}px`,
    top: `${position.y}px`,
    width: `${size.width}px`,
    height: `${size.height}px`,
    zIndex: zIndex,
    cursor: currentCursor,
    userSelect: (isDragging || isResizing) ? 'none' : 'auto',
    boxSizing: 'border-box',
    ...style
  };

  return (
    <div
      ref={itemRef}
      className={`grid-item ${className} ${isDragging ? 'dragging' : ''} ${isResizing ? 'resizing' : ''}`}
      style={itemStyle}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
    >
      <div className="grid-item-content" style={{ width: '100%', height: '100%', overflow: 'hidden', pointerEvents: isDragging || isResizing ? 'none' : 'auto' }}>
        {children}
      </div>
    </div>
  );
};
