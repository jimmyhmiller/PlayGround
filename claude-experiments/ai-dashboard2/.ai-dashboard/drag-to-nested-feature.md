# Drag-to-Nested Dashboard Feature

## Overview

Enable dragging widgets from a parent dashboard and dropping them into nested dashboards with visual feedback and smooth animations.

## Feature Requirements

### Visual Feedback
- **Shrinking Animation**: When hovering over a nested dashboard with a dragged widget, the widget should shrink to indicate it can be dropped
- **Drop Zone Highlight**: Nested dashboard should show visual feedback (border glow, overlay text)
- **Smooth Transitions**: Animations should be configurable (instant, 100ms, 200ms, 400ms)

### Core Functionality
- **Drag Detection**: Detect when a dragged widget overlaps a nested dashboard widget
- **Widget Transfer**: Move widget from parent dashboard to nested dashboard's internal widgets array
- **Position Calculation**: Determine where the widget should land in the nested dashboard
- **State Preservation**: Maintain widget configuration, conversations, and data during transfer

## Implementation Architecture

### 1. Enhanced GridItem Component (`src/components/GridItem.tsx`)

**Current State:**
- Uses native mouse events for drag/drop
- Supports drag modifiers: `modifier` (Cmd/Ctrl), `handle`, `full`
- Grid snapping with `snapToGrid()` and `snapSize()`
- Z-index management for bringing items to front

**Required Changes:**
- Add collision detection during drag to identify overlapping nested dashboards
- Emit new callbacks:
  - `onDragOverNested(targetWidgetId: string)`
  - `onDragLeaveNested()`
  - `onDropIntoNested(targetWidgetId: string)`
- Apply scale transform when `isDraggingOverNested` state is true
- Track which nested dashboard is currently being hovered

### 2. NestedDashboard Drop Zone (`src/widgets/NestedDashboard.tsx`)

**Current State:**
- Renders full Dashboard with own Grid and Widgets
- Supports up to 5 levels of nesting
- Double-click to navigate "into" nested dashboard
- Independent layout settings per nested dashboard

**Required Changes:**
- Accept `isDropTarget: boolean` prop from parent
- Visual feedback when `isDropTarget === true`:
  ```tsx
  border: isDropTarget ? `2px solid ${theme.accent}` : '1px solid rgba(255,255,255,0.1)',
  boxShadow: isDropTarget ? `0 0 20px ${theme.accent}44` : 'none',
  animation: isDropTarget ? 'pulse 1s ease-in-out infinite' : 'none'
  ```
- Show overlay text: "Drop here to move widget"
- Handle drop event to trigger widget transfer

### 3. Widget Transfer Logic (`src/App.tsx`)

**Required Implementation:**
```typescript
const handleWidgetTransferToNested = (
  widgetId: string,
  fromDashboardId: string,
  toNestedWidgetId: string,
  dropPosition?: { x: number; y: number }
) => {
  // 1. Find and remove widget from source dashboard
  const sourceDashboard = dashboards.find(d => d.id === fromDashboardId);
  const widget = sourceDashboard.widgets.find(w => w.id === widgetId);
  const remainingWidgets = sourceDashboard.widgets.filter(w => w.id !== widgetId);

  // 2. Find target nested dashboard widget
  const nestedDashboardWidget = sourceDashboard.widgets.find(w => w.id === toNestedWidgetId);
  const nestedDashboard = nestedDashboardWidget.dashboard;

  // 3. Calculate position in nested dashboard
  const position = calculateDropPosition(widget, nestedDashboard, dropPosition);

  // 4. Add widget to nested dashboard
  nestedDashboard.widgets.push({
    ...widget,
    x: position.x,
    y: position.y
  });

  // 5. Update parent dashboard with both changes
  await window.dashboardAPI.updateWidget(fromDashboardId, toNestedWidgetId, {
    dashboard: nestedDashboard
  });
};
```

### 4. Position Calculation Strategies

**Option 1: Center Placement**
- Place widget at center of nested dashboard
- Simple and predictable

**Option 2: Proportional Scaling**
- If widget at 20% from left in parent, place at 20% from left in nested
- Maintains relative positioning

**Option 3: Mouse Position**
- Drop at mouse position relative to nested dashboard bounds
- Most intuitive for users

**Option 4: Auto-Layout**
- Find first available empty space
- Prevents overlapping

## Edge Cases to Handle

### Circular Reference Prevention
```typescript
function canDropIntoNested(widgetId: string, targetNestedId: string): boolean {
  // Prevent dropping a nested dashboard into itself
  if (widgetId === targetNestedId) return false;

  // Prevent circular nesting (A contains B contains C contains A)
  const widget = findWidget(widgetId);
  if (widget.type !== 'nested-dashboard') return true;

  return !containsWidget(widget.dashboard, targetNestedId);
}
```

### Max Nesting Depth
```typescript
function getNestedDepth(dashboard: Dashboard, currentDepth = 0): number {
  const maxDepth = 5; // Current limit
  if (currentDepth >= maxDepth) return currentDepth;

  const nestedDashboards = dashboard.widgets.filter(w => w.type === 'nested-dashboard');
  if (nestedDashboards.length === 0) return currentDepth;

  return Math.max(...nestedDashboards.map(w =>
    getNestedDepth(w.dashboard, currentDepth + 1)
  ));
}
```

### State Preservation
- Widget conversations (for chat widgets)
- Widget data (charts, stats, etc.)
- Widget-specific settings
- Undo/redo support

## CSS Animations

### Shrinking Effect
```css
.dragging-over-nested {
  transform: scale(0.6);
  transition: transform 200ms ease-in-out;
  opacity: 0.8;
}
```

### Drop Zone Pulse
```css
@keyframes pulse {
  0%, 100% {
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
  }
  50% {
    box-shadow: 0 0 30px rgba(0, 217, 255, 0.6);
  }
}
```

## Backend API Changes

### New IPC Method
```typescript
interface DashboardAPI {
  // Existing methods...
  moveWidgetToNestedDashboard: (
    widgetId: string,
    fromDashboardId: string,
    toNestedWidgetId: string,
    newPosition?: { x: number; y: number }
  ) => Promise<{ success: boolean; error?: string }>;
}
```

### File Update Strategy
- Atomic updates to prevent corruption
- Update parent dashboard JSON file
- Trigger file watcher reload
- Broadcast to all clients

## Testing Plan

1. **Basic Transfer**
   - Drag widget over nested dashboard
   - Verify shrinking animation
   - Verify drop zone highlight
   - Verify widget appears in nested dashboard

2. **Edge Cases**
   - Try to drop nested dashboard into itself (should fail)
   - Try to exceed max nesting depth (should fail)
   - Drop at max depth - 1 (should succeed)

3. **State Preservation**
   - Drag chat widget with conversation history
   - Verify messages preserved after transfer
   - Drag chart widget with data
   - Verify data preserved

4. **Performance**
   - Drag over multiple nested dashboards quickly
   - Verify no lag or jank
   - Test with 20+ widgets

## Future Enhancements

- **Drag between nested dashboards**: Currently only parent → nested, could support nested → nested
- **Drag out of nested dashboards**: Move widgets back to parent
- **Multi-select drag**: Drag multiple widgets at once
- **Snap to grid in nested**: Maintain grid alignment after drop
- **Visual preview**: Show ghost/preview of widget before dropping
- **Keyboard shortcuts**: Alt+Drop for copy instead of move
- **Confirmation dialog**: "Are you sure you want to move this widget?"

## Related Files

- `src/components/GridItem.tsx` - Drag handling
- `src/widgets/NestedDashboard.tsx` - Drop zone
- `src/App.tsx` - Widget transfer logic
- `src/types/dashboard.ts` - Type definitions
- `preload.ts` - IPC handlers
- `main.ts` - Backend transfer logic

## Resources

- Current drag implementation: Uses native mouse events, not HTML5 drag-and-drop
- Grid system: 16px base unit with configurable gap
- Z-index management: Global counter, increments on drag
- File watching: `fs.watch()` with debouncing
