import React, { createContext, useContext, useState, useCallback, useRef } from 'react';

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
  className = ''
}) => {
  // Support both gap (single value) and gapX/gapY (separate values)
  const actualGapX = gapX !== undefined ? gapX : (gap !== undefined ? gap : 8);
  const actualGapY = gapY !== undefined ? gapY : (gap !== undefined ? gap : 8);

  const [items, setItems] = useState(new Map());
  const maxZIndexRef = useRef(1);

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

  const gridStyle = {
    position: 'relative',
    width,
    height,
    backgroundImage: showGrid
      ? `
        linear-gradient(to right, rgba(0,0,0,0.05) ${cellSize}px, transparent ${cellSize}px),
        linear-gradient(to bottom, rgba(0,0,0,0.05) ${cellSize}px, transparent ${cellSize}px)
      `
      : 'none',
    backgroundSize: showGrid ? `${cellSize + actualGapX}px ${cellSize + actualGapY}px` : 'auto',
    backgroundPosition: showGrid ? '0px 0px' : '0 0',
  };

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
      <div className={`grid-container ${className}`} style={gridStyle}>
        {children}
      </div>
    </GridContext.Provider>
  );
};
