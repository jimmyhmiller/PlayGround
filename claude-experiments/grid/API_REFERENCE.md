# API Reference

Complete API documentation for the Grid Component Library.

## Components

### Grid

The main container component that manages the grid layout and provides context to GridItems.

#### Import

```jsx
import { Grid } from './components';
```

#### Props

##### `cellSize` (number, default: `16`)

Size of each grid cell in pixels. This is the base unit for positioning and sizing.

```jsx
<Grid cellSize={100} />
```

##### `gap` (number, default: `8`)

Uniform gap between grid cells (sets both gapX and gapY). Overridden by gapX/gapY if specified.

```jsx
<Grid gap={10} />
```

##### `gapX` (number, default: `gap || 8`)

Horizontal gap between grid cells in pixels. Overrides `gap` for horizontal spacing.

```jsx
<Grid gapX={12} />
```

##### `gapY` (number, default: `gap || 8`)

Vertical gap between grid cells in pixels. Overrides `gap` for vertical spacing.

```jsx
<Grid gapY={8} />
```

##### `width` (string, default: `'100%'`)

CSS width value for the grid container.

```jsx
<Grid width="1200px" />
<Grid width="100vw" />
```

##### `height` (string, default: `'100vh'`)

CSS height value for the grid container.

```jsx
<Grid height="800px" />
<Grid height="100%" />
```

##### `showGrid` (boolean, default: `false`)

Display visual grid lines showing cell boundaries and gaps.

```jsx
<Grid showGrid={true} />
```

##### `onLayoutChange` (function, optional)

Callback function called when the grid layout changes (items moved or resized).

**Signature:**
```typescript
(items: Array<{
  id: string,
  x: number,
  y: number,
  width: number,
  height: number
}>) => void
```

**Example:**
```jsx
const handleLayoutChange = (items) => {
  console.log('Layout updated:', items);
  localStorage.setItem('layout', JSON.stringify(items));
};

<Grid onLayoutChange={handleLayoutChange} />
```

##### `className` (string, default: `''`)

Additional CSS class name for the grid container.

```jsx
<Grid className="my-custom-grid" />
```

##### `children` (ReactNode, required)

GridItem components and other React elements.

```jsx
<Grid>
  <GridItem>...</GridItem>
  <GridItem>...</GridItem>
</Grid>
```

---

### GridItem

Individual draggable and resizable items within the grid.

#### Import

```jsx
import { GridItem } from './components';
```

#### Props

##### `x` (number, default: `0`)

Initial X position in pixels. Should be a multiple of `(cellSize + gapX)`.

```jsx
<GridItem x={0} />
<GridItem x={110} /> // For cellSize=100, gapX=10
```

##### `y` (number, default: `0`)

Initial Y position in pixels. Should be a multiple of `(cellSize + gapY)`.

```jsx
<GridItem y={0} />
<GridItem y={110} /> // For cellSize=100, gapY=10
```

##### `width` (number, default: `cellSize`)

Initial width in pixels. Should follow the formula: `n * cellSize + (n-1) * gapX`.

```jsx
<GridItem width={100} />     // 1 cell
<GridItem width={210} />     // 2 cells (2*100 + 1*10)
<GridItem width={320} />     // 3 cells (3*100 + 2*10)
```

##### `height` (number, default: `cellSize`)

Initial height in pixels. Should follow the formula: `n * cellSize + (n-1) * gapY`.

```jsx
<GridItem height={100} />    // 1 cell
<GridItem height={210} />    // 2 cells
```

##### `minWidth` (number, optional)

Minimum width constraint in pixels. Item cannot be resized below this width.

```jsx
<GridItem minWidth={100} />
```

##### `minHeight` (number, optional)

Minimum height constraint in pixels. Item cannot be resized below this height.

```jsx
<GridItem minHeight={100} />
```

##### `draggable` (boolean, default: `true`)

Enable or disable dragging for this item.

```jsx
<GridItem draggable={false} /> // Fixed position
```

##### `resizable` (boolean, default: `true`)

Enable or disable resizing for this item.

```jsx
<GridItem resizable={false} /> // Fixed size
```

##### `enforceContentSize` (boolean, default: `false`)

Prevent resizing smaller than the content's intrinsic size.

```jsx
<GridItem enforceContentSize={true}>
  <div>This content sets the minimum size</div>
</GridItem>
```

##### `onDragStart` (function, optional)

Callback called when dragging starts.

**Signature:**
```typescript
(position: { x: number, y: number }) => void
```

**Example:**
```jsx
<GridItem onDragStart={({ x, y }) => {
  console.log('Drag started at:', x, y);
}} />
```

##### `onDrag` (function, optional)

Callback called continuously during dragging.

**Signature:**
```typescript
(position: { x: number, y: number }) => void
```

**Example:**
```jsx
<GridItem onDrag={({ x, y }) => {
  console.log('Dragging to:', x, y);
}} />
```

##### `onDragEnd` (function, optional)

Callback called when dragging ends.

**Signature:**
```typescript
(position: { x: number, y: number }) => void
```

**Example:**
```jsx
<GridItem onDragEnd={({ x, y }) => {
  console.log('Dropped at:', x, y);
  savePosition(x, y);
}} />
```

##### `onResize` (function, optional)

Callback called during resizing.

**Signature:**
```typescript
(size: { width: number, height: number }) => void
```

**Example:**
```jsx
<GridItem onResize={({ width, height }) => {
  console.log('Resized to:', width, height);
}} />
```

##### `className` (string, default: `''`)

Additional CSS class name for the grid item.

```jsx
<GridItem className="my-widget" />
```

##### `style` (object, default: `{}`)

Additional inline styles for the grid item. Note: position, left, top, width, height are controlled by the component.

```jsx
<GridItem style={{ border: '2px solid blue' }} />
```

##### `children` (ReactNode, required)

Content to display in the grid item.

```jsx
<GridItem>
  <h3>My Widget</h3>
  <p>Content here</p>
</GridItem>
```

---

## Hooks

### useGrid

Access grid context from within Grid children.

#### Import

```jsx
import { useGrid } from './components';
```

#### Returns

```typescript
{
  cellSize: number,
  gapX: number,
  gapY: number,
  snapToGrid: (value: number, axis: 'x' | 'y') => number,
  snapSize: (value: number, axis: 'x' | 'y') => number,
  registerItem: (id: string, data: object) => void,
  unregisterItem: (id: string) => void,
  updateItem: (id: string, data: object) => void,
  items: Map<string, object>,
  getNextZIndex: () => number
}
```

#### Example

```jsx
function GridAwareComponent() {
  const { cellSize, gapX, gapY, items } = useGrid();

  return (
    <div>
      Grid Configuration:
      - Cell Size: {cellSize}px
      - Gap X: {gapX}px
      - Gap Y: {gapY}px
      - Total Items: {items.size}
    </div>
  );
}

// Must be used inside <Grid>
<Grid cellSize={50} gap={10}>
  <GridAwareComponent />
</Grid>
```

#### Context Properties

##### `cellSize` (number)

The current grid cell size.

##### `gapX` (number)

The current horizontal gap.

##### `gapY` (number)

The current vertical gap.

##### `snapToGrid(value, axis)` (function)

Snaps a position value to the nearest grid boundary.

```jsx
const { snapToGrid } = useGrid();
const snappedX = snapToGrid(155, 'x'); // Returns nearest grid X position
const snappedY = snapToGrid(87, 'y');  // Returns nearest grid Y position
```

##### `snapSize(value, axis)` (function)

Snaps a size value to valid multi-cell dimensions.

```jsx
const { snapSize } = useGrid();
const snappedWidth = snapSize(250, 'x');  // Returns valid width
const snappedHeight = snapSize(180, 'y'); // Returns valid height
```

##### `items` (Map)

Map of all registered grid items with their current state.

```jsx
const { items } = useGrid();
console.log('Number of items:', items.size);
```

##### `getNextZIndex()` (function)

Gets the next available z-index value.

```jsx
const { getNextZIndex } = useGrid();
const newZIndex = getNextZIndex(); // Used internally by GridItem
```

---

## Type Definitions

### GridProps

```typescript
interface GridProps {
  children: ReactNode;
  cellSize?: number;
  gap?: number;
  gapX?: number;
  gapY?: number;
  width?: string;
  height?: string;
  onLayoutChange?: (items: LayoutItem[]) => void;
  showGrid?: boolean;
  className?: string;
}
```

### GridItemProps

```typescript
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
  enforceContentSize?: boolean;
  onDragStart?: (position: Position) => void;
  onDrag?: (position: Position) => void;
  onDragEnd?: (position: Position) => void;
  onResize?: (size: Size) => void;
  className?: string;
  style?: CSSProperties;
}
```

### Position

```typescript
interface Position {
  x: number;
  y: number;
}
```

### Size

```typescript
interface Size {
  width: number;
  height: number;
}
```

### LayoutItem

```typescript
interface LayoutItem {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}
```

---

## Formulas

### Position Calculation

```
Grid Position (X) = n × (cellSize + gapX)
Grid Position (Y) = n × (cellSize + gapY)
```

Where `n` is the grid cell index (0, 1, 2, ...).

### Size Calculation

```
Width  = (numCells × cellSize) + ((numCells - 1) × gapX)
Height = (numCells × cellSize) + ((numCells - 1) × gapY)
```

Where `numCells` is the number of cells to span.

### Examples

With `cellSize=100`, `gapX=10`, `gapY=10`:

**Positions:**
- Cell 0: `x=0, y=0`
- Cell 1: `x=110, y=0` (0 + (100+10))
- Cell 2: `x=220, y=0` (0 + 2×(100+10))

**Sizes:**
- 1 cell: `100` (1×100 + 0×10)
- 2 cells: `210` (2×100 + 1×10)
- 3 cells: `320` (3×100 + 2×10)
- 4 cells: `430` (4×100 + 3×10)

---

## Events

### Drag Events

Events fire in this order during drag:
1. `onDragStart` - When mouse down on draggable item
2. `onDrag` - Continuously during mouse move
3. `onDragEnd` - When mouse up

All events receive the current position: `{ x, y }`

### Resize Events

`onResize` - Fires continuously during resize with current size: `{ width, height }`

### Layout Events

`onLayoutChange` - Fires on Grid after any item moves or resizes with all items: `[{ id, x, y, width, height }, ...]`

---

## CSS Classes

### Grid Container

```css
.grid-container {
  /* Your custom styles */
}
```

### Grid Item

```css
.grid-item {
  /* Base item styles */
}

.grid-item.dragging {
  /* Styles when dragging */
}

.grid-item.resizing {
  /* Styles when resizing */
}
```

### Resize Handle

```css
.resize-handle {
  /* Custom resize handle styles */
}
```

---

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support (14+)
- IE11: ❌ Not supported (requires ResizeObserver polyfill)

---

## Performance Notes

- Grid items use React Context for efficient updates
- Z-index management is automatic and optimized
- Dragging and resizing use native DOM events for best performance
- Content size measurement only occurs when `enforceContentSize={true}`
- Visual grid rendering uses CSS gradients (hardware accelerated)

---

## See Also

- [README.md](./README.md) - Overview and quick start
- [USAGE_GUIDE.md](./USAGE_GUIDE.md) - Comprehensive usage guide
- [EXAMPLES.md](./EXAMPLES.md) - Advanced examples
