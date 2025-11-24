import React, { useState } from 'react';
import { Grid, GridItem } from './components';
import './App.css';

function App() {
  const [mode, setMode] = useState('coarse');
  const [showGrid, setShowGrid] = useState(true);
  const [enforceContentSize, setEnforceContentSize] = useState(false);
  const [gapX, setGapX] = useState(10);
  const [gapY, setGapY] = useState(10);

  const handleLayoutChange = (items) => {
    console.log('Layout changed:', items);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Grid Component Library</h1>
        <div className="controls">
          <button
            className={mode === 'coarse' ? 'active' : ''}
            onClick={() => setMode('coarse')}
          >
            3x3 Grid (Coarse)
          </button>
          <button
            className={mode === 'fine' ? 'active' : ''}
            onClick={() => setMode('fine')}
          >
            Fine-Grained Grid (16px)
          </button>
          <button
            className={mode === 'canvas' ? 'active' : ''}
            onClick={() => setMode('canvas')}
          >
            Canvas Mode (1px)
          </button>
          <label>
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => setShowGrid(e.target.checked)}
            />
            Show Grid
          </label>
          <label>
            <input
              type="checkbox"
              checked={enforceContentSize}
              onChange={(e) => setEnforceContentSize(e.target.checked)}
            />
            Enforce Content Size
          </label>
        </div>
        <div className="gap-controls">
          <div className="gap-control">
            <label>
              Gap X: <span className="gap-value">{gapX}px</span>
            </label>
            <input
              type="range"
              min="0"
              max="50"
              value={gapX}
              onChange={(e) => setGapX(Number(e.target.value))}
            />
          </div>
          <div className="gap-control">
            <label>
              Gap Y: <span className="gap-value">{gapY}px</span>
            </label>
            <input
              type="range"
              min="0"
              max="50"
              value={gapY}
              onChange={(e) => setGapY(Number(e.target.value))}
            />
          </div>
        </div>
      </header>

      <div className="grid-wrapper">
        {mode === 'coarse' && (
          <Grid
            cellSize={100}
            gapX={gapX}
            gapY={gapY}
            width="100%"
            height="600px"
            showGrid={showGrid}
            onLayoutChange={handleLayoutChange}
          >
            <GridItem
              x={0}
              y={0}
              width={100}
              height={100}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#ff6b6b' }}>
                <h3>Item 1</h3>
                <p>1x1 grid cell</p>
              </div>
            </GridItem>

            <GridItem
              x={110}
              y={0}
              width={320}
              height={100}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#4ecdc4' }}>
                <h3>Item 2</h3>
                <p>3x1 grid cells</p>
              </div>
            </GridItem>

            <GridItem
              x={0}
              y={110}
              width={210}
              height={210}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#45b7d1' }}>
                <h3>Item 3</h3>
                <p>2x2 grid cells</p>
              </div>
            </GridItem>

            <GridItem
              x={220}
              y={110}
              width={100}
              height={100}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#f9ca24' }}>
                <h3>Item 4</h3>
                <p>1x1 grid cell</p>
              </div>
            </GridItem>
          </Grid>
        )}

        {mode === 'fine' && (
          <Grid
            cellSize={16}
            gapX={gapX}
            gapY={gapY}
            width="100%"
            height="600px"
            showGrid={showGrid}
            onLayoutChange={handleLayoutChange}
          >
            <GridItem
              x={0}
              y={0}
              width={64}
              height={48}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#a29bfe' }}>
                <h4>Card 1</h4>
                <p>4x3 cells</p>
              </div>
            </GridItem>

            <GridItem
              x={74}
              y={0}
              width={96}
              height={80}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#fd79a8' }}>
                <h4>Card 2</h4>
                <p>6x5 cells</p>
              </div>
            </GridItem>

            <GridItem
              x={180}
              y={0}
              width={128}
              height={112}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#fdcb6e' }}>
                <h4>Card 3</h4>
                <p>8x7 cells</p>
                <p>Larger card</p>
              </div>
            </GridItem>

            <GridItem
              x={0}
              y={58}
              width={160}
              height={64}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#74b9ff' }}>
                <h4>Wide Card</h4>
                <p>10x4 cells</p>
              </div>
            </GridItem>
          </Grid>
        )}

        {mode === 'canvas' && (
          <Grid
            cellSize={1}
            gapX={gapX}
            gapY={gapY}
            width="100%"
            height="600px"
            showGrid={false}
            onLayoutChange={handleLayoutChange}
          >
            <GridItem
              x={50}
              y={50}
              width={150}
              height={100}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#6c5ce7' }}>
                <h4>Free Form 1</h4>
                <p>Pixel-perfect positioning</p>
              </div>
            </GridItem>

            <GridItem
              x={250}
              y={80}
              width={180}
              height={120}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#00b894' }}>
                <h4>Free Form 2</h4>
                <p>Position anywhere on the canvas</p>
              </div>
            </GridItem>

            <GridItem
              x={100}
              y={200}
              width={200}
              height={80}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#e17055' }}>
                <h4>Free Form 3</h4>
                <p>Like a design canvas</p>
              </div>
            </GridItem>

            <GridItem
              x={350}
              y={250}
              width={120}
              height={120}
              className="grid-item-demo"
              enforceContentSize={enforceContentSize}
            >
              <div className="item-content" style={{ background: '#d63031' }}>
                <h4>Square</h4>
                <p>Any size, any position</p>
              </div>
            </GridItem>
          </Grid>
        )}
      </div>

      <div className="info">
        <h3>Instructions</h3>
        <ul>
          <li>Click and drag any item to move it</li>
          <li>Drag the bottom-right corner to resize items</li>
          <li>Items will snap to the grid automatically</li>
          <li>Items respect their content's minimum size</li>
          <li>Toggle between different grid modes to see different behaviors</li>
        </ul>
      </div>
    </div>
  );
}

export default App;
