import { useEffect, useRef, useState, useId, ReactNode, CSSProperties, MouseEvent as ReactMouseEvent } from 'react';
import { useGrid } from './Grid';

interface IntrinsicSize {
  width: number;
  height: number;
}

const measureIntrinsicSize = (element: HTMLElement | null): IntrinsicSize | null => {
  const contentWrapper = element?.querySelector('.grid-item-content');
  if (!contentWrapper) return null;

  const clone = contentWrapper.cloneNode(true) as HTMLElement;

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

  const rect = clone.getBoundingClientRect();
  const intrinsicWidth = rect.width;
  const intrinsicHeight = rect.height;

  document.body.removeChild(clone);

  return { width: intrinsicWidth, height: intrinsicHeight };
};

type ResizeEdge = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw' | null;
type DragMode = 'full' | 'handle' | 'modifier';

interface GridItemProps {
  children: ReactNode;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  minWidth?: number;
  minHeight?: number;
  resizable?: boolean;
  draggable?: boolean;
  dragMode?: DragMode;
  enforceContentSize?: boolean;
  onDragStart?: (pos: { x: number; y: number }) => void;
  onDrag?: (pos: { x: number; y: number }) => void;
  onDragEnd?: (pos: { x: number; y: number }) => void;
  onResize?: (size: { width: number; height: number }) => void;
  onDragOverNested?: (targetWidgetId: string) => void;
  onDragLeaveNested?: () => void;
  onDropIntoNested?: (targetWidgetId: string) => void;
  className?: string;
  style?: CSSProperties;
}

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
  dragMode = 'modifier',
  enforceContentSize = false,
  onDragStart,
  onDrag,
  onDragEnd,
  onResize,
  onDragOverNested,
  onDragLeaveNested,
  onDropIntoNested,
  className = '',
  style = {}
}: GridItemProps) => {
  const id = useId();
  const { cellSize, snapToGrid, snapSize, registerItem, unregisterItem, updateItem, getNextZIndex } = useGrid();
  const itemRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [currentCursor, setCurrentCursor] = useState('grab');
  const resizeEdgeRef = useRef<ResizeEdge>(null);
  const [position, setPosition] = useState({ x: initialX, y: initialY });
  const [size, setSize] = useState({
    width: initialWidth || cellSize,
    height: initialHeight || cellSize
  });
  const [intrinsicSize, setIntrinsicSize] = useState<IntrinsicSize | null>(null);
  const [zIndex, setZIndex] = useState(1);
  const dragStartPos = useRef({ x: 0, y: 0 });
  const resizeStartPos = useRef({ x: 0, y: 0 });
  const resizeStartSize = useRef({ width: 0, height: 0 });
  const resizeStartPosition = useRef({ x: 0, y: 0 });
  const isManualResizeRef = useRef(false);
  const [hoveredNestedId, setHoveredNestedId] = useState<string | null>(null);
  const lastHoveredNestedIdRef = useRef<string | null>(null);

  useEffect(() => {
    registerItem(id, { id, x: position.x, y: position.y, width: size.width, height: size.height });
    return () => unregisterItem(id);
  }, [id, registerItem, unregisterItem]);

  useEffect(() => {
    updateItem(id, { x: position.x, y: position.y, width: size.width, height: size.height });
  }, [id, position.x, position.y, size.width, size.height, updateItem]);

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

  useEffect(() => {
    if (!enforceContentSize) return;

    const contentElement = itemRef.current?.querySelector('.grid-item-content');
    if (!contentElement) return;

    const resizeObserver = new ResizeObserver(() => {
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

  const getResizeEdge = (e: ReactMouseEvent<HTMLDivElement>): ResizeEdge => {
    if (!resizable || !itemRef.current) return null;

    const rect = itemRef.current.getBoundingClientRect();
    const edgeThreshold = 8;

    const nearTop = e.clientY - rect.top < edgeThreshold;
    const nearBottom = rect.bottom - e.clientY < edgeThreshold;
    const nearLeft = e.clientX - rect.left < edgeThreshold;
    const nearRight = rect.right - e.clientX < edgeThreshold;

    if (nearTop && nearLeft) return 'nw';
    if (nearTop && nearRight) return 'ne';
    if (nearBottom && nearLeft) return 'sw';
    if (nearBottom && nearRight) return 'se';

    if (nearTop) return 'n';
    if (nearBottom) return 's';
    if (nearLeft) return 'w';
    if (nearRight) return 'e';

    return null;
  };

  const getCursorForEdge = (edge: ResizeEdge, modifierPressed = false): string => {
    if (!edge) {
      if (!draggable) return 'default';
      if (dragMode === 'modifier' && !modifierPressed) return 'default';
      return 'grab';
    }

    const cursorMap: Record<string, string> = {
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

  const handleMouseMove = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (isDragging || isResizing) return;

    const edge = getResizeEdge(e);
    const modifierPressed = e.metaKey || e.ctrlKey;
    const cursor = getCursorForEdge(edge, modifierPressed);
    setCurrentCursor(cursor);
  };

  const handleMouseDown = (e: ReactMouseEvent<HTMLDivElement>) => {
    const edge = getResizeEdge(e);

    if (edge) {
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
      if (!draggable || isResizing) return;

      if (dragMode === 'modifier') {
        if (!e.metaKey && !e.ctrlKey) return;
      } else if (dragMode === 'handle') {
        if (!(e.target as HTMLElement).classList.contains('drag-handle')) return;
      }

      e.stopPropagation();
      setIsDragging(true);
      setCurrentCursor('grabbing');

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

    const handleMouseMove = (e: globalThis.MouseEvent) => {
      if (isDragging) {
        const newX = e.clientX - dragStartPos.current.x;
        const newY = e.clientY - dragStartPos.current.y;

        const snappedX = snapToGrid(newX, 'x');
        const snappedY = snapToGrid(newY, 'y');

        setPosition({ x: snappedX, y: snappedY });

        // ONLY check for nested dashboard collision when BOTH Cmd/Ctrl AND Shift are held
        // This completely avoids any interference with normal drag operations
        const isNestingModeActive = e.shiftKey && (e.metaKey || e.ctrlKey);

        if (isNestingModeActive && (onDragOverNested || onDragLeaveNested)) {
          // Find nested dashboard widgets by checking for data-nested-dashboard attribute
          let foundNestedId: string | null = null;

          // Use the current mouse position directly for accurate detection
          const elementsAtPoint = document.elementsFromPoint(e.clientX, e.clientY);

          for (const element of elementsAtPoint) {
            const nestedDashboard = element.closest('[data-nested-dashboard="true"]');
            if (nestedDashboard && nestedDashboard !== itemRef.current?.closest('[data-nested-dashboard="true"]')) {
              const nestedId = nestedDashboard.getAttribute('data-widget-id');
              if (nestedId) {
                foundNestedId = nestedId;
                break;
              }
            }
          }

          // Handle enter/leave events
          if (foundNestedId !== lastHoveredNestedIdRef.current) {
            if (lastHoveredNestedIdRef.current && onDragLeaveNested) {
              onDragLeaveNested();
            }
            if (foundNestedId && onDragOverNested) {
              onDragOverNested(foundNestedId);
            }
            lastHoveredNestedIdRef.current = foundNestedId;
            setHoveredNestedId(foundNestedId);
          }
        } else if (lastHoveredNestedIdRef.current) {
          // Clear nested hover state if not in nesting mode
          if (onDragLeaveNested) {
            onDragLeaveNested();
          }
          lastHoveredNestedIdRef.current = null;
          setHoveredNestedId(null);
        }

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

        if (intrinsicSize) {
          newWidth = Math.max(newWidth, intrinsicSize.width);
          newHeight = Math.max(newHeight, intrinsicSize.height);
        }

        if (minWidth !== undefined) {
          newWidth = Math.max(newWidth, minWidth);
        }
        if (minHeight !== undefined) {
          newHeight = Math.max(newHeight, minHeight);
        }

        const snappedWidth = snapSize(newWidth, 'x');
        const snappedHeight = snapSize(newHeight, 'y');

        if (edge.includes('w')) {
          newX = resizeStartPosition.current.x + (resizeStartSize.current.width - snappedWidth);
        }
        if (edge.includes('n')) {
          newY = resizeStartPosition.current.y + (resizeStartSize.current.height - snappedHeight);
        }

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
        // Handle drop into nested dashboard
        if (lastHoveredNestedIdRef.current && onDropIntoNested) {
          onDropIntoNested(lastHoveredNestedIdRef.current);
        }

        setIsDragging(false);
        setCurrentCursor(draggable ? 'grab' : 'default');

        // Reset hover state
        if (lastHoveredNestedIdRef.current && onDragLeaveNested) {
          onDragLeaveNested();
        }
        lastHoveredNestedIdRef.current = null;
        setHoveredNestedId(null);

        if (onDragEnd) {
          onDragEnd({ x: position.x, y: position.y });
        }
      }
      if (isResizing) {
        setIsResizing(false);
        resizeEdgeRef.current = null;
        setCurrentCursor(draggable ? 'grab' : 'default');
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
  }, [isDragging, isResizing, snapToGrid, snapSize, position, minWidth, minHeight, intrinsicSize, draggable, onDrag, onDragEnd, onResize]);

  const itemStyle: CSSProperties = {
    position: 'absolute',
    left: `${position.x}px`,
    top: `${position.y}px`,
    width: `${size.width}px`,
    height: `${size.height}px`,
    zIndex: zIndex,
    cursor: currentCursor,
    userSelect: (isDragging || isResizing) ? 'none' : 'auto',
    boxSizing: 'border-box',
    transform: hoveredNestedId ? 'scale(0.6)' : 'scale(1)',
    transition: hoveredNestedId ? 'transform 200ms ease-in-out' : 'none',
    opacity: hoveredNestedId ? 0.8 : 1,
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
