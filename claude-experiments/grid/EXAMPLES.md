# Advanced Examples

## Dynamic Grid Items

Add and remove items dynamically:

```jsx
import { useState } from 'react';
import { Grid, GridItem } from './components';

function DynamicGrid() {
  const [items, setItems] = useState([
    { id: 1, x: 0, y: 0, width: 100, height: 100 }
  ]);

  const addItem = () => {
    const newItem = {
      id: Date.now(),
      x: Math.random() * 400,
      y: Math.random() * 400,
      width: 100,
      height: 100
    };
    setItems([...items, newItem]);
  };

  const removeItem = (id) => {
    setItems(items.filter(item => item.id !== id));
  };

  return (
    <>
      <button onClick={addItem}>Add Item</button>
      <Grid cellSize={20} gap={5}>
        {items.map(item => (
          <GridItem
            key={item.id}
            x={item.x}
            y={item.y}
            width={item.width}
            height={item.height}
          >
            <div>
              Item {item.id}
              <button onClick={() => removeItem(item.id)}>×</button>
            </div>
          </GridItem>
        ))}
      </Grid>
    </>
  );
}
```

## Persisting Layout

Save and restore grid layouts:

```jsx
import { useState, useEffect } from 'react';
import { Grid, GridItem } from './components';

function PersistentGrid() {
  const [layout, setLayout] = useState(() => {
    const saved = localStorage.getItem('gridLayout');
    return saved ? JSON.parse(saved) : [];
  });

  const handleLayoutChange = (newLayout) => {
    setLayout(newLayout);
    localStorage.setItem('gridLayout', JSON.stringify(newLayout));
  };

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
          {/* Your content */}
        </GridItem>
      ))}
    </Grid>
  );
}
```

## Constrained Dragging

Keep items within bounds:

```jsx
function ConstrainedGrid() {
  const [position, setPosition] = useState({ x: 0, y: 0 });

  const handleDrag = ({ x, y }) => {
    // Constrain to 500x500 area
    const constrainedX = Math.max(0, Math.min(x, 400));
    const constrainedY = Math.max(0, Math.min(y, 400));
    setPosition({ x: constrainedX, y: constrainedY });
  };

  return (
    <Grid cellSize={20} gap={5} width="500px" height="500px">
      <GridItem
        x={position.x}
        y={position.y}
        width={100}
        height={100}
        onDrag={handleDrag}
      >
        Constrained Item
      </GridItem>
    </Grid>
  );
}
```

## Custom Item Components

Create reusable grid item components:

```jsx
function Card({ title, content, ...gridProps }) {
  return (
    <GridItem {...gridProps}>
      <div style={{
        padding: '1rem',
        background: 'white',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        height: '100%'
      }}>
        <h3>{title}</h3>
        <p>{content}</p>
      </div>
    </GridItem>
  );
}

function Dashboard() {
  return (
    <Grid cellSize={16} gap={8}>
      <Card
        x={0}
        y={0}
        width={200}
        height={150}
        title="Analytics"
        content="View your stats"
      />
      <Card
        x={220}
        y={0}
        width={200}
        height={150}
        title="Users"
        content="Manage users"
      />
    </Grid>
  );
}
```

## Responsive Grid

Adapt grid size based on viewport:

```jsx
import { useState, useEffect } from 'react';

function ResponsiveGrid() {
  const [cellSize, setCellSize] = useState(16);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setCellSize(8);  // Smaller grid on mobile
      } else {
        setCellSize(16); // Larger grid on desktop
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <Grid cellSize={cellSize} gap={cellSize / 2}>
      {/* Your items */}
    </Grid>
  );
}
```

## Collision Detection

Prevent items from overlapping:

```jsx
function CollisionGrid() {
  const [items, setItems] = useState([
    { id: 1, x: 0, y: 0, width: 100, height: 100 },
    { id: 2, x: 150, y: 0, width: 100, height: 100 }
  ]);

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
    <Grid cellSize={20} gap={5}>
      {items.map(item => (
        <GridItem
          key={item.id}
          x={item.x}
          y={item.y}
          width={item.width}
          height={item.height}
          onDragEnd={pos => handleDragEnd(item.id, pos)}
        >
          Item {item.id}
        </GridItem>
      ))}
    </Grid>
  );
}
```

## Grid with Zoom

Add zoom functionality to the grid:

```jsx
import { useState } from 'react';

function ZoomableGrid() {
  const [zoom, setZoom] = useState(1);

  return (
    <div>
      <div>
        <button onClick={() => setZoom(z => Math.max(0.5, z - 0.1))}>-</button>
        <span>{Math.round(zoom * 100)}%</span>
        <button onClick={() => setZoom(z => Math.min(2, z + 0.1))}>+</button>
      </div>
      <div style={{ transform: `scale(${zoom})`, transformOrigin: 'top left' }}>
        <Grid cellSize={16} gap={8}>
          {/* Your items */}
        </Grid>
      </div>
    </div>
  );
}
```
