# Grid Component Library

A flexible, React-based grid component library that allows draggable and resizable components with automatic grid snapping.

## Features

- **Flexible Grid Sizes**: From coarse grids (3x3) to fine-grained grids (16px) to pixel-perfect canvas mode
- **Drag and Drop**: Click and drag items to reposition them
- **Resizable Components**: Resize items by dragging the bottom-right corner
- **Grid Snapping**: Items automatically snap to grid boundaries
- **Separate X/Y Gaps**: Independent horizontal and vertical spacing
- **Content Size Enforcement**: Optional prevention of resizing smaller than content
- **Z-Index Management**: Dragged items automatically come to front
- **Layout Change Callbacks**: Get notified when the grid layout changes

## Documentation

📖 **[Complete Usage Guide](./USAGE_GUIDE.md)** - Comprehensive guide with patterns, best practices, and troubleshooting

## Quick Start

### Installation

```bash
npm install
```

### Running the Demo

```bash
npm run dev
```

Open http://localhost:5174 to see the interactive demo with live controls.

## Basic Usage

### Basic Example

```jsx
import { Grid, GridItem } from './components';

function App() {
  const handleLayoutChange = (items) => {
    console.log('Layout changed:', items);
  };

  return (
    <Grid
      cellSize={16}
      gap={8}
      width="100%"
      height="600px"
      showGrid={true}
      onLayoutChange={handleLayoutChange}
    >
      <GridItem
        x={0}
        y={0}
        width={96}
        height={72}
        draggable={true}
        resizable={true}
      >
        <div>Your content here</div>
      </GridItem>
    </Grid>
  );
}
```

## Grid Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `cellSize` | number | `16` | Size of each grid cell in pixels |
| `gap` | number | `8` | Gap between grid cells in pixels (both X and Y) |
| `gapX` | number | `gap \|\| 8` | Horizontal gap between grid cells (overrides `gap`) |
| `gapY` | number | `gap \|\| 8` | Vertical gap between grid cells (overrides `gap`) |
| `width` | string | `'100%'` | Width of the grid container |
| `height` | string | `'100vh'` | Height of the grid container |
| `showGrid` | boolean | `false` | Show visual grid lines |
| `onLayoutChange` | function | - | Callback when layout changes |
| `className` | string | `''` | Additional CSS class |

## GridItem Props

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
| `enforceContentSize` | boolean | `false` | Prevent resizing smaller than content's intrinsic size |
| `onDragStart` | function | - | Called when drag starts |
| `onDrag` | function | - | Called during drag |
| `onDragEnd` | function | - | Called when drag ends |
| `onResize` | function | - | Called when resizing |
| `className` | string | `''` | Additional CSS class |
| `style` | object | `{}` | Additional inline styles |

## Grid Modes

### Coarse Grid (3x3)
Perfect for dashboard layouts with larger components:
```jsx
<Grid cellSize={100} gap={10}>
```

### Fine-Grained Grid (16px)
Ideal for organized free-form layouts:
```jsx
<Grid cellSize={16} gap={8}>
```

### Canvas Mode
Pixel-perfect positioning with minimal snapping:
```jsx
<Grid cellSize={1} gap={0}>
```

### Different Horizontal and Vertical Gaps
Create rectangular grids with different spacing:
```jsx
<Grid cellSize={16} gapX={12} gapY={8}>
```

You can use either:
- `gap` - Sets both horizontal and vertical gaps to the same value
- `gapX` and `gapY` - Set horizontal and vertical gaps independently (overrides `gap`)

## Features in Detail

### Grid Snapping
Items automatically snap to grid boundaries when dragged or resized. The snap point is calculated based on `cellSize + gapX` for horizontal positioning and `cellSize + gapY` for vertical positioning.

### Content Size Enforcement (Optional)
When `enforceContentSize={true}` is set on a GridItem, the component cannot be resized smaller than its content's intrinsic size. The library measures the actual content dimensions and enforces them as minimum constraints during resize operations. This is disabled by default to allow maximum flexibility.

```jsx
<GridItem enforceContentSize={true}>
  <div>This content sets the minimum size</div>
</GridItem>
```

### Z-Index Management
Items automatically manage their stacking order. When an item is dragged, it comes to the front by receiving the highest z-index. This ensures a natural interaction where the item being moved always appears on top of others.

### Layout Change Notifications
The `onLayoutChange` callback provides an array of all items with their current positions and sizes:
```jsx
[
  { id, x, y, width, height },
  // ...
]
```

### Context API
The Grid component uses React Context to provide grid parameters to all GridItems. This ensures consistent snapping behavior and efficient updates.

## Quick Reference

### Calculating Item Positions

Items snap to grid boundaries:
```
X Position = n × (cellSize + gapX)
Y Position = n × (cellSize + gapY)
```

### Calculating Item Sizes

Items spanning multiple cells:
```
Width  = (numCells × cellSize) + ((numCells - 1) × gapX)
Height = (numCells × cellSize) + ((numCells - 1) × gapY)
```

**Examples** with `cellSize=100`, `gapX=10`, `gapY=10`:
- 1 cell: `100px`
- 2 cells: `210px` (2×100 + 1×10)
- 3 cells: `320px` (3×100 + 2×10)

### Common Grid Configurations

| Use Case | cellSize | gap | Description |
|----------|----------|-----|-------------|
| Dashboard | 100 | 10 | Large widgets |
| Card Layout | 16-24 | 8 | Medium cards |
| Pixel Canvas | 1-2 | 0-1 | Free-form design |

## Browser Support

Works in all modern browsers that support:
- ES6+
- React 18+
- ResizeObserver API

## Further Reading

- **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** - Complete usage guide with patterns and examples
- **[EXAMPLES.md](./EXAMPLES.md)** - Advanced implementation examples

## License

ISC
