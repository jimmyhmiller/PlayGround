import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';

const GridContext = createContext(null);

export const useGrid = () => {
  const context = useContext(GridContext);
  if (!context) {
    throw new Error('useGrid must be used within a Grid component');
  }
  return context;
};

export const Grid = ({
  children,
  cellSize = 16,
  gap,
  gapX,
  gapY,
  width = '100%',
  height = '100vh',
  onLayoutChange,
  showGrid = false,
  className = '',
  mode = 'single-pane'
}) => {
  // Support both gap (single value) and gapX/gapY (separate values)
  const actualGapX = gapX !== undefined ? gapX : (gap !== undefined ? gap : 8);
  const actualGapY = gapY !== undefined ? gapY : (gap !== undefined ? gap : 8);

  const [items, setItems] = useState(new Map());
  const maxZIndexRef = useRef(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [isAltKeyPressed, setIsAltKeyPressed] = useState(false);
  const panStartPos = useRef({ x: 0, y: 0 });
  const panStartOffset = useRef({ x: 0, y: 0 });

  const registerItem = useCallback((id, data) => {
    setItems(prev => {
      const next = new Map(prev);
      next.set(id, data);
      return next;
    });
  }, []);

  const unregisterItem = useCallback((id) => {
    setItems(prev => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);

  const updateItem = useCallback((id, data) => {
    setItems(prev => {
      const next = new Map(prev);
      const existing = next.get(id);
      next.set(id, { ...existing, ...data });

      if (onLayoutChange) {
        onLayoutChange(Array.from(next.values()));
      }

      return next;
    });
  }, [onLayoutChange]);

  const snapToGrid = useCallback((value, axis = 'x') => {
    const gapSize = axis === 'x' ? actualGapX : actualGapY;
    const totalCellSize = cellSize + gapSize;
    return Math.round(value / totalCellSize) * totalCellSize;
  }, [cellSize, actualGapX, actualGapY]);

  const snapSize = useCallback((value, axis = 'x') => {
    // Sizes should account for: n * cellSize + (n-1) * gap
    // Where n is the number of cells spanned
    const gapSize = axis === 'x' ? actualGapX : actualGapY;
    const totalCellSize = cellSize + gapSize;

    // Find how many cells this spans
    const numCells = Math.round(value / totalCellSize);

    // Size = (numCells * cellSize) + ((numCells - 1) * gap)
    // Which simplifies to: numCells * cellSize + numCells * gap - gap
    // Which is: numCells * (cellSize + gap) - gap
    if (numCells <= 0) return cellSize;
    return numCells * cellSize + (numCells - 1) * gapSize;
  }, [cellSize, actualGapX, actualGapY]);

  const getNextZIndex = useCallback(() => {
    maxZIndexRef.current += 1;
    return maxZIndexRef.current;
  }, []);

  // Calculate the bounding box of all items for scroll modes
  const calculateContentSize = useCallback(() => {
    if (items.size === 0) {
      return { width: 0, height: 0 };
    }

    let maxX = 0;
    let maxY = 0;

    items.forEach((item) => {
      const right = (item.x || 0) + (item.width || 0);
      const bottom = (item.y || 0) + (item.height || 0);
      maxX = Math.max(maxX, right);
      maxY = Math.max(maxY, bottom);
    });

    return { width: maxX, height: maxY };
  }, [items]);

  const contentSize = calculateContentSize();


  // Apply overflow styles based on mode
  const getOverflowStyles = () => {
    // Parse height to number if it's a string like "calc(100% - 80px)"
    const parseHeight = (h) => {
      if (typeof h === 'string' && h.includes('calc')) {
        // For calc expressions, we'll use the string as-is for fixed dimension
        return h;
      }
      return h;
    };

    const parsedHeight = parseHeight(height);

    switch (mode) {
      case 'vertical-scroll':
        return {
          overflowY: 'auto',
          overflowX: 'hidden',
          height: parsedHeight,
          minHeight: contentSize.height * 1.5 + 400 // 50% extra space + 400px padding
        };
      case 'horizontal-scroll':
        return {
          overflowX: 'auto',
          overflowY: 'hidden',
          height: parsedHeight, // Fixed height, no extra vertical space
          width: width,
          minWidth: contentSize.width * 1.5 + 400 // 50% extra space + 400px padding
        };
      case 'infinite-canvas':
        return {
          overflow: 'hidden', // No scrollbars, pan with drag
          cursor: isPanning ? 'grabbing' : (isAltKeyPressed ? 'grab' : 'default')
        };
      case 'single-pane':
      default:
        return {
          overflow: 'hidden'
        };
    }
  };

  const overflowStyles = getOverflowStyles();


  // For scroll modes, the inner container needs to be sized to the content
  const containerStyle = {
    position: 'relative',
    // Width: expand for horizontal scroll, 100% otherwise
    width: mode === 'horizontal-scroll' ? overflowStyles.minWidth : '100%',
    // Height: expand for vertical scroll, 100% for horizontal (fixed), auto for infinite canvas/single-pane
    height: mode === 'vertical-scroll' ? 'auto' : '100%',
    minWidth: mode === 'horizontal-scroll' ? overflowStyles.minWidth : undefined,
    minHeight: mode === 'vertical-scroll' ? overflowStyles.minHeight : undefined,
    // Apply pan transform for infinite canvas mode
    transform: mode === 'infinite-canvas' ? `translate(${panOffset.x}px, ${panOffset.y}px)` : 'none',
    transition: isPanning ? 'none' : 'transform 0.1s ease-out',
    backgroundImage: showGrid
      ? `
        linear-gradient(to right, rgba(0,0,0,0.05) ${cellSize}px, transparent ${cellSize}px),
        linear-gradient(to bottom, rgba(0,0,0,0.05) ${cellSize}px, transparent ${cellSize}px)
      `
      : 'none',
    backgroundSize: showGrid ? `${cellSize + actualGapX}px ${cellSize + actualGapY}px` : 'auto',
    backgroundPosition: showGrid ? `${panOffset.x}px ${panOffset.y}px` : '0 0',
  };

  const gridStyle = {
    position: 'relative',
    width: overflowStyles.width || width,
    height: overflowStyles.height || height,
    minWidth: overflowStyles.minWidth,
    minHeight: overflowStyles.minHeight,
    overflowX: overflowStyles.overflowX,
    overflowY: overflowStyles.overflowY,
    overflow: overflowStyles.overflow,
    cursor: overflowStyles.cursor,
  };

  // Pan handlers for infinite canvas mode
  const handleMouseDown = (e) => {
    if (mode !== 'infinite-canvas') return;

    // Only pan when holding Option/Alt key
    if (!e.altKey) {
      return;
    }

    e.preventDefault(); // Prevent text selection while panning
    setIsPanning(true);
    panStartPos.current = { x: e.clientX, y: e.clientY };
    panStartOffset.current = { ...panOffset };
  };

  useEffect(() => {
    if (!isPanning || mode !== 'infinite-canvas') return;

    const handleMouseMove = (e) => {
      const deltaX = e.clientX - panStartPos.current.x;
      const deltaY = e.clientY - panStartPos.current.y;

      setPanOffset({
        x: panStartOffset.current.x + deltaX,
        y: panStartOffset.current.y + deltaY
      });
    };

    const handleMouseUp = () => {
      setIsPanning(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isPanning, mode]);

  // Track Alt key state for cursor changes in infinite canvas mode
  useEffect(() => {
    if (mode !== 'infinite-canvas') {
      setIsAltKeyPressed(false);
      return;
    }

    const handleKeyDown = (e) => {
      if (e.altKey) {
        setIsAltKeyPressed(true);
      }
    };

    const handleKeyUp = (e) => {
      if (!e.altKey) {
        setIsAltKeyPressed(false);
      }
    };

    // Handle blur to reset state when window loses focus
    const handleBlur = () => {
      setIsAltKeyPressed(false);
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', handleBlur);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', handleBlur);
    };
  }, [mode]);

  return (
    <GridContext.Provider value={{
      cellSize,
      gapX: actualGapX,
      gapY: actualGapY,
      snapToGrid,
      snapSize,
      registerItem,
      unregisterItem,
      updateItem,
      items,
      getNextZIndex
    }}>
      <div className={`grid-container ${className}`} style={gridStyle} onMouseDown={handleMouseDown}>
        <div style={containerStyle}>
          {children}
        </div>
      </div>
    </GridContext.Provider>
  );
};
