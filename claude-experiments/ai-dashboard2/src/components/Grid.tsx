import { createContext, useContext, useState, useCallback, useRef, useEffect, ReactNode, CSSProperties, MouseEvent } from 'react';

interface GridItem {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface GridContextValue {
  cellSize: number;
  gapX: number;
  gapY: number;
  snapToGrid: (value: number, axis?: 'x' | 'y') => number;
  snapSize: (value: number, axis?: 'x' | 'y') => number;
  registerItem: (id: string, data: GridItem) => void;
  unregisterItem: (id: string) => void;
  updateItem: (id: string, data: Partial<GridItem>) => void;
  items: Map<string, GridItem>;
  getNextZIndex: () => number;
}

const GridContext = createContext<GridContextValue | null>(null);

export const useGrid = (): GridContextValue => {
  const context = useContext(GridContext);
  if (!context) {
    throw new Error('useGrid must be used within a Grid component');
  }
  return context;
};

interface GridProps {
  children: ReactNode;
  cellSize?: number;
  gap?: number;
  gapX?: number;
  gapY?: number;
  width?: string | number;
  height?: string | number;
  onLayoutChange?: (items: GridItem[]) => void;
  showGrid?: boolean;
  className?: string;
  mode?: 'single-pane' | 'vertical-scroll' | 'horizontal-scroll' | 'infinite-canvas';
}

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
}: GridProps) => {
  // Support both gap (single value) and gapX/gapY (separate values)
  const actualGapX = gapX !== undefined ? gapX : (gap !== undefined ? gap : 8);
  const actualGapY = gapY !== undefined ? gapY : (gap !== undefined ? gap : 8);

  const [items, setItems] = useState(new Map<string, GridItem>());
  const maxZIndexRef = useRef(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [isAltKeyPressed, setIsAltKeyPressed] = useState(false);
  const panStartPos = useRef({ x: 0, y: 0 });
  const panStartOffset = useRef({ x: 0, y: 0 });

  const registerItem = useCallback((id: string, data: GridItem) => {
    setItems(prev => {
      const next = new Map(prev);
      next.set(id, data);
      return next;
    });
  }, []);

  const unregisterItem = useCallback((id: string) => {
    setItems(prev => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);

  const updateItem = useCallback((id: string, data: Partial<GridItem>) => {
    setItems(prev => {
      const next = new Map(prev);
      const existing = next.get(id);
      if (existing) {
        next.set(id, { ...existing, ...data });
      }

      if (onLayoutChange) {
        onLayoutChange(Array.from(next.values()));
      }

      return next;
    });
  }, [onLayoutChange]);

  const snapToGrid = useCallback((value: number, axis: 'x' | 'y' = 'x'): number => {
    const gapSize = axis === 'x' ? actualGapX : actualGapY;
    const totalCellSize = cellSize + gapSize;
    return Math.round(value / totalCellSize) * totalCellSize;
  }, [cellSize, actualGapX, actualGapY]);

  const snapSize = useCallback((value: number, axis: 'x' | 'y' = 'x'): number => {
    const gapSize = axis === 'x' ? actualGapX : actualGapY;
    const totalCellSize = cellSize + gapSize;

    const numCells = Math.round(value / totalCellSize);
    if (numCells <= 0) return cellSize;
    return numCells * cellSize + (numCells - 1) * gapSize;
  }, [cellSize, actualGapX, actualGapY]);

  const getNextZIndex = useCallback(() => {
    maxZIndexRef.current += 1;
    return maxZIndexRef.current;
  }, []);

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

  const getOverflowStyles = (): CSSProperties => {
    const parseHeight = (h: string | number) => {
      if (typeof h === 'string' && h.includes('calc')) {
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
          minHeight: contentSize.height * 1.5 + 400
        };
      case 'horizontal-scroll':
        return {
          overflowX: 'auto',
          overflowY: 'hidden',
          height: parsedHeight,
          width: width,
          minWidth: contentSize.width * 1.5 + 400
        };
      case 'infinite-canvas':
        return {
          overflow: 'hidden',
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

  const containerStyle: CSSProperties = {
    position: 'relative',
    width: mode === 'horizontal-scroll' ? overflowStyles.minWidth : '100%',
    height: mode === 'vertical-scroll' ? 'auto' : '100%',
    minWidth: mode === 'horizontal-scroll' ? overflowStyles.minWidth : undefined,
    minHeight: mode === 'vertical-scroll' ? overflowStyles.minHeight : undefined,
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

  const gridStyle: CSSProperties = {
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

  const handleMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    if (mode !== 'infinite-canvas') return;

    if (!e.altKey) {
      return;
    }

    e.preventDefault();
    setIsPanning(true);
    panStartPos.current = { x: e.clientX, y: e.clientY };
    panStartOffset.current = { ...panOffset };
  };

  useEffect(() => {
    if (!isPanning || mode !== 'infinite-canvas') return;

    const handleMouseMove = (e: globalThis.MouseEvent) => {
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

  useEffect(() => {
    if (mode !== 'infinite-canvas') {
      setIsAltKeyPressed(false);
      return;
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.altKey) {
        setIsAltKeyPressed(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (!e.altKey) {
        setIsAltKeyPressed(false);
      }
    };

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
