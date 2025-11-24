# Grid Component Library - Usage Guide

A comprehensive guide to integrating and using the grid component library in your React applications.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Component API](#component-api)
5. [Common Patterns](#common-patterns)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Copy Components to Your Project

```bash
# Copy the grid components to your project
cp -r src/components/Grid.jsx your-project/src/components/
cp -r src/components/GridItem.jsx your-project/src/components/
```

### Install Dependencies

The grid components require React 18+:

```bash
npm install react react-dom
```

---

## Quick Start

### Basic Example

```jsx
import React from 'react';
import { Grid, GridItem } from './components';

function MyDashboard() {
  return (
    <Grid
      cellSize={100}
      gapX={10}
      gapY={10}
      width="100%"
      height="600px"
      showGrid={true}
    >
      <GridItem
        x={0}
        y={0}
        width={100}
        height={100}
      >
        <div>Single Cell Item</div>
      </GridItem>

      <GridItem
        x={110}
        y={0}
        width={210}
        height={100}
      >
        <div>Two Cell Wide Item</div>
      </GridItem>
    </Grid>
  );
}
```

---

## Core Concepts

### Understanding the Grid System

#### Cell Size
The `cellSize` defines the base unit of your grid. All items will snap to multiples of this size.

```jsx
<Grid cellSize={50}>  // Each cell is 50x50 pixels
```

#### Gaps
Gaps are the spaces between grid cells:
- `gapX`: Horizontal spacing between cells
- `gapY`: Vertical spacing between cells
- `gap`: Shorthand for uniform spacing (sets both gapX and gapY)

```jsx
// Uniform gaps
<Grid cellSize={50} gap={10} />

// Different horizontal and vertical gaps
<Grid cellSize={50} gapX={12} gapY={8} />
```

#### Positioning
Items are positioned at grid boundaries:
- X positions: `0, (cellSize + gapX), 2 * (cellSize + gapX), ...`
- Y positions: `0, (cellSize + gapY), 2 * (cellSize + gapY), ...`

#### Sizing
Items spanning multiple cells include the gaps between those cells:
- **1 cell**: `cellSize`
- **2 cells**: `2 * cellSize + gapX`
- **3 cells**: `3 * cellSize + 2 * gapX`
- **n cells**: `n * cellSize + (n-1) * gap`

### Example Calculations

With `cellSize={100}`, `gapX={10}`, `gapY={10}`:

```jsx
// 1x1 item at position (0, 0)
<GridItem x={0} y={0} width={100} height={100} />

// 2x1 item at position (110, 0)
// x = 0 + (100 + 10) = 110
// width = 2 * 100 + 1 * 10 = 210
<GridItem x={110} y={0} width={210} height={100} />

// 2x2 item at position (0, 110)
// y = 0 + (100 + 10) = 110
// width = height = 2 * 100 + 1 * 10 = 210
<GridItem x={0} y={110} width={210} height={210} />
```

---

## Component API

### Grid Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `cellSize` | number | `16` | Size of each grid cell in pixels |
| `gap` | number | `8` | Gap between grid cells (both X and Y) |
| `gapX` | number | `gap \|\| 8` | Horizontal gap between grid cells |
| `gapY` | number | `gap \|\| 8` | Vertical gap between grid cells |
| `width` | string | `'100%'` | Width of the grid container |
| `height` | string | `'100vh'` | Height of the grid container |
| `showGrid` | boolean | `false` | Show visual grid lines |
| `onLayoutChange` | function | - | Callback when layout changes |
| `className` | string | `''` | Additional CSS class |
| `children` | ReactNode | - | GridItem components |

### GridItem Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `x` | number | `0` | Initial x position in pixels |
| `y` | number | `0` | Initial y position in pixels |
| `width` | number | `cellSize` | Initial width in pixels |
| `height` | number | `cellSize` | Initial height in pixels |
| `minWidth` | number | - | Minimum width constraint |
| `minHeight` | number | - | Minimum height constraint |
| `draggable` | boolean | `true` | Enable dragging |
| `resizable` | boolean | `true` | Enable resizing |
| `enforceContentSize` | boolean | `false` | Prevent resizing smaller than content |
| `onDragStart` | function | - | Called when drag starts |
| `onDrag` | function | - | Called during drag |
| `onDragEnd` | function | - | Called when drag ends |
| `onResize` | function | - | Called when resizing |
| `className` | string | `''` | Additional CSS class |
| `style` | object | `{}` | Additional inline styles |
| `children` | ReactNode | - | Content to display |

---

## Common Patterns

### 1. Dashboard Layout

```jsx
function Dashboard() {
  return (
    <Grid cellSize={100} gap={10} height="100vh">
      {/* Header - spans full width */}
      <GridItem x={0} y={0} width={320} height={100}>
        <Header />
      </GridItem>

      {/* Sidebar */}
      <GridItem x={0} y={110} width={100} height={320}>
        <Sidebar />
      </GridItem>

      {/* Main content */}
      <GridItem x={110} y={110} width={210} height={320}>
        <MainContent />
      </GridItem>
    </Grid>
  );
}
```

### 2. Card Grid

```jsx
function CardGrid({ cards }) {
  return (
    <Grid cellSize={16} gapX={8} gapY={8}>
      {cards.map((card, index) => (
        <GridItem
          key={card.id}
          x={card.x}
          y={card.y}
          width={card.width}
          height={card.height}
          enforceContentSize={true}
        >
          <Card {...card} />
        </GridItem>
      ))}
    </Grid>
  );
}
```

### 3. Responsive Grid

```jsx
function ResponsiveGrid() {
  const [cellSize, setCellSize] = useState(16);

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      if (width < 768) {
        setCellSize(8);
      } else if (width < 1200) {
        setCellSize(16);
      } else {
        setCellSize={24);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <Grid cellSize={cellSize} gap={cellSize / 2}>
      {/* Your grid items */}
    </Grid>
  );
}
```

### 4. Persisting Layout

```jsx
function PersistentGrid() {
  const [layout, setLayout] = useState(() => {
    const saved = localStorage.getItem('gridLayout');
    return saved ? JSON.parse(saved) : defaultLayout;
  });

  const handleLayoutChange = useCallback((newLayout) => {
    setLayout(newLayout);
    localStorage.setItem('gridLayout', JSON.stringify(newLayout));
  }, []);

  return (
    <Grid
      cellSize={16}
      gap={8}
      onLayoutChange={handleLayoutChange}
    >
      {layout.map(item => (
        <GridItem
          key={item.id}
          x={item.x}
          y={item.y}
          width={item.width}
          height={item.height}
        >
          <DynamicComponent type={item.type} data={item.data} />
        </GridItem>
      ))}
    </Grid>
  );
}
```

### 5. Controlled Grid Items

```jsx
function ControlledGrid() {
  const [items, setItems] = useState([
    { id: 1, x: 0, y: 0, width: 100, height: 100 }
  ]);

  const handleDragEnd = (id, newPosition) => {
    setItems(items.map(item =>
      item.id === id
        ? { ...item, ...newPosition }
        : item
    ));
  };

  const handleResize = (id, newSize) => {
    setItems(items.map(item =>
      item.id === id
        ? { ...item, ...newSize }
        : item
    ));
  };

  return (
    <Grid cellSize={50} gap={5}>
      {items.map(item => (
        <GridItem
          key={item.id}
          x={item.x}
          y={item.y}
          width={item.width}
          height={item.height}
          onDragEnd={pos => handleDragEnd(item.id, pos)}
          onResize={size => handleResize(item.id, size)}
        >
          Item {item.id}
        </GridItem>
      ))}
    </Grid>
  );
}
```

---

## Advanced Usage

### Custom Item Components

Create reusable grid item wrappers:

```jsx
// Widget.jsx
function Widget({ title, children, ...gridProps }) {
  return (
    <GridItem {...gridProps}>
      <div className="widget">
        <h3 className="widget-title">{title}</h3>
        <div className="widget-content">
          {children}
        </div>
      </div>
    </GridItem>
  );
}

// Usage
<Grid cellSize={100} gap={10}>
  <Widget
    title="Analytics"
    x={0}
    y={0}
    width={210}
    height={210}
  >
    <AnalyticsChart />
  </Widget>
</Grid>
```

### Dynamic Grid Items

Add and remove items dynamically:

```jsx
function DynamicGrid() {
  const [items, setItems] = useState([]);

  const addItem = () => {
    const newItem = {
      id: Date.now(),
      x: 0,
      y: 0,
      width: 100,
      height: 100,
      type: 'default'
    };
    setItems([...items, newItem]);
  };

  const removeItem = (id) => {
    setItems(items.filter(item => item.id !== id));
  };

  return (
    <>
      <button onClick={addItem}>Add Item</button>
      <Grid cellSize={50} gap={5}>
        {items.map(item => (
          <GridItem
            key={item.id}
            x={item.x}
            y={item.y}
            width={item.width}
            height={item.height}
          >
            <button onClick={() => removeItem(item.id)}>×</button>
            {item.type}
          </GridItem>
        ))}
      </Grid>
    </>
  );
}
```

### Collision Detection

Prevent items from overlapping:

```jsx
function CollisionGrid() {
  const [items, setItems] = useState([...]);

  const checkCollision = (item1, item2) => {
    return !(
      item1.x + item1.width <= item2.x ||
      item2.x + item2.width <= item1.x ||
      item1.y + item1.height <= item2.y ||
      item2.y + item2.height <= item1.y
    );
  };

  const handleDragEnd = (id, newPos) => {
    const draggedItem = items.find(i => i.id === id);
    const testItem = { ...draggedItem, ...newPos };

    const hasCollision = items.some(
      item => item.id !== id && checkCollision(testItem, item)
    );

    if (!hasCollision) {
      setItems(items.map(item =>
        item.id === id ? { ...item, ...newPos } : item
      ));
    }
  };

  return (
    <Grid cellSize={50} gap={5}>
      {items.map(item => (
        <GridItem
          key={item.id}
          {...item}
          onDragEnd={pos => handleDragEnd(item.id, pos)}
        >
          Item {item.id}
        </GridItem>
      ))}
    </Grid>
  );
}
```

### Context Integration

Access grid context in nested components:

```jsx
import { useGrid } from './components';

function GridAwareComponent() {
  const { cellSize, gapX, gapY, items } = useGrid();

  return (
    <div>
      Cell size: {cellSize}px
      Gaps: {gapX}px × {gapY}px
      Total items: {items.size}
    </div>
  );
}

// Use inside Grid
<Grid cellSize={50} gap={10}>
  <GridAwareComponent />
  <GridItem x={0} y={0} width={50} height={50}>
    Item 1
  </GridItem>
</Grid>
```

---

## Best Practices

### 1. Choose Appropriate Grid Sizes

**Coarse Grid (Large Cells)**
- Use for: Dashboards, high-level layouts, widget containers
- Cell size: 50-100px
- Gap: 8-16px

```jsx
<Grid cellSize={100} gap={10} />
```

**Fine Grid (Small Cells)**
- Use for: Detailed layouts, precise positioning
- Cell size: 8-24px
- Gap: 4-8px

```jsx
<Grid cellSize={16} gap={8} />
```

**Canvas Mode (Pixel-Perfect)**
- Use for: Design tools, free-form layouts
- Cell size: 1-2px
- Gap: 0-1px

```jsx
<Grid cellSize={1} gap={0} />
```

### 2. Performance Optimization

**Memoize Callbacks**
```jsx
const handleLayoutChange = useCallback((items) => {
  // Save layout
}, []);

const handleDragEnd = useCallback((id, pos) => {
  // Update position
}, []);
```

**Limit Grid Items**
- Keep the number of GridItems under 100 for optimal performance
- Use virtualization for larger grids

**Use Keys Properly**
```jsx
{items.map(item => (
  <GridItem key={item.id} {...item}>
    {/* Stable keys prevent unnecessary re-renders */}
  </GridItem>
))}
```

### 3. Accessibility

**Keyboard Support**
```jsx
<GridItem
  onKeyDown={(e) => {
    if (e.key === 'ArrowRight') {
      // Move item right
    }
  }}
  tabIndex={0}
  role="button"
  aria-label="Draggable item"
>
  Content
</GridItem>
```

**Screen Reader Support**
```jsx
<Grid aria-label="Dashboard grid layout">
  <GridItem aria-label="Analytics widget">
    <Analytics />
  </GridItem>
</Grid>
```

### 4. Styling

**Use CSS Modules or Styled Components**
```jsx
import styles from './MyGrid.module.css';

<GridItem className={styles.myItem}>
  Content
</GridItem>
```

**Avoid Inline Styles for Performance**
```jsx
// ❌ Avoid
<GridItem style={{ background: 'red', padding: '10px' }}>

// ✅ Better
<GridItem className="my-styled-item">
```

### 5. State Management

**Centralize Layout State**
```jsx
// Store layout in context or state management library
const LayoutContext = createContext();

function LayoutProvider({ children }) {
  const [layout, setLayout] = useState([]);

  return (
    <LayoutContext.Provider value={{ layout, setLayout }}>
      {children}
    </LayoutContext.Provider>
  );
}
```

---

## Troubleshooting

### Items Don't Snap to Grid

**Problem**: Items position anywhere instead of snapping to grid.

**Solution**: Ensure you're using the correct positioning:
```jsx
// ❌ Wrong - random positions
<GridItem x={15} y={23} />

// ✅ Correct - grid positions
<GridItem x={0} y={0} />
<GridItem x={110} y={0} />  // Next grid position with cellSize=100, gap=10
```

### Items Overlap Grid Cells

**Problem**: Items are too large for their grid cells.

**Solution**: Use the sizing formula `n * cellSize + (n-1) * gap`:
```jsx
// For 3 cells with cellSize=100, gap=10
// width = 3 * 100 + 2 * 10 = 320
<GridItem width={320} />
```

### Grid Lines Don't Match Items

**Problem**: Visual grid doesn't align with item positions.

**Solution**: Ensure you're passing the same gapX/gapY to the Grid:
```jsx
<Grid cellSize={100} gapX={10} gapY={10} showGrid={true}>
```

### Items Can Resize Too Small

**Problem**: Items resize smaller than their content.

**Solution**: Use `enforceContentSize` or set `minWidth`/`minHeight`:
```jsx
<GridItem
  enforceContentSize={true}
  // OR
  minWidth={100}
  minHeight={100}
/>
```

### Performance Issues with Many Items

**Solutions**:
1. Reduce the number of visible items
2. Disable `showGrid` for better performance
3. Use `React.memo` for GridItem content
4. Implement virtualization for large grids

```jsx
const MemoizedWidget = React.memo(Widget);

<GridItem>
  <MemoizedWidget data={data} />
</GridItem>
```

### Z-Index Issues

**Problem**: Items don't come to front when dragged.

**Solution**: The grid automatically manages z-index. If you're overriding with custom styles, remove z-index from your CSS.

---

## Examples Repository

Check out the `src/App.jsx` file for complete working examples of:
- Coarse grid layouts
- Fine-grained grids
- Canvas mode
- Live gap controls
- Multiple grid modes

## Support

For issues, questions, or contributions, please refer to the main README.md file.
