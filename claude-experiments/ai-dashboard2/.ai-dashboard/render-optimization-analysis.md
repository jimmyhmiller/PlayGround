# Dashboard Render Optimization Analysis

## Problem Statement

Monaco editor widgets (and other complex widgets) flash/re-render when dragging ANY widget on the dashboard, causing poor user experience.

## Root Cause Analysis

### Current Architecture Flow

```
User drags widget
  ↓
GridItem onDrag fires (60fps during drag)
  ↓
Widget.handleDrag calls onResize
  ↓
App.handleWidgetResize
  ↓
IPC: updateWidgetDimensions
  ↓
main.ts: writes JSON file
  ↓
[Current: broadcastDashboards() or manual reload]
  ↓
App.setDashboards(newArray)
  ↓
Dashboard re-renders with new props
  ↓
ALL Widget components receive new config object references
  ↓
Monaco editors re-mount/reconcile state → FLASH
```

### Why React.memo Doesn't Work Here

Even with React.memo, the problem persists because:

1. **Object reference instability**: When we reload dashboards via `setDashboards(updatedDashboards)`, the entire array is new
2. **Config objects are new**: Every widget config object has a new reference, even if content is identical
3. **Memo comparison fails**: React.memo's comparison function sees `prevProps.config !== nextProps.config`
4. **Deep equality is expensive**: Using `JSON.stringify()` for comparison negates memo benefits

```javascript
// Even with memo, this happens:
const oldConfig = { id: "editor1", x: 100, y: 100, content: "..." }
const newConfig = { id: "editor1", x: 100, y: 100, content: "..." }

oldConfig === newConfig // false! New object reference
// → Memo comparison fails → Re-render happens
```

## Attempted Solutions

### Attempt 1: Remove broadcast during drag
- **What we did**: Commented out `broadcastDashboards()` in `update-widget-dimensions`
- **Result**: No flashing during drag, but positions don't save
- **Why it failed**: UI never learns about the updated state

### Attempt 2: Manual reload after drag ends
- **What we did**: Added `loadDashboards()` call after `updateWidgetDimensions()`
- **Result**: Positions save, but editors flash at end of drag
- **Why it failed**: New dashboard array = new config objects = all widgets re-render

### Attempt 3: React.memo with deep comparison
- **What we did**: Wrapped Widget in memo with JSON.stringify comparison
- **Result**: Still flashes
- **Why it failed**: Deep comparison is expensive AND still sees reference changes

## Potential Solutions

### Solution 1: Immutable State Management with Normalization ⭐ RECOMMENDED

**Concept**: Store widgets in a normalized structure where each widget has a stable reference until its actual data changes.

```javascript
// Instead of:
const dashboards = [
  {
    id: "dash1",
    widgets: [
      { id: "widget1", x: 10, y: 20 },  // New object every reload
      { id: "widget2", x: 30, y: 40 }   // New object every reload
    ]
  }
]

// Use:
const widgetCache = new Map([
  ["widget1", { id: "widget1", x: 10, y: 20 }],  // Stable reference
  ["widget2", { id: "widget2", x: 30, y: 40 }]   // Stable reference
])

// Only update the specific widget that changed
widgetCache.set("widget1", { id: "widget1", x: 15, y: 20 })
```

**Implementation**:
1. Create a widget cache in App.tsx using useMemo/useRef
2. When dashboards load, only update cache entries that actually changed
3. Pass stable widget references to Widget components
4. Use object identity for memo comparison

**Pros**:
- ✅ Surgical updates - only changed widgets re-render
- ✅ Fast comparison - can use `===` instead of deep equality
- ✅ Works with React.memo perfectly
- ✅ Scalable to large dashboards

**Cons**:
- ❌ More complex state management
- ❌ Need to handle cache invalidation
- ❌ Requires refactoring App.tsx dashboard state

**Estimated effort**: 4-6 hours

---

### Solution 2: Local UI State with Debounced Persistence

**Concept**: Keep drag position in local React state, only persist to file after user stops dragging (debounced).

```javascript
// In GridItem.tsx
const [localPosition, setLocalPosition] = useState({ x, y })

// During drag: update local state only
onDrag: (pos) => setLocalPosition(pos)

// On drag end: debounced persist
onDragEnd: debounce((pos) => {
  persistToFile(pos)  // Only writes to file
}, 500)
```

**Implementation**:
1. GridItem maintains local x/y state separate from props
2. During drag: update local state (smooth visual feedback)
3. On drag end: debounce IPC call (500ms)
4. Remove dashboard reload entirely
5. Rely on file watcher for external changes only

**Pros**:
- ✅ Zero re-renders during drag
- ✅ Smooth UX (local state updates are instant)
- ✅ Fewer file writes (debounced)
- ✅ Simpler than normalization

**Cons**:
- ❌ Potential state drift if file changes externally during drag
- ❌ Need to handle sync between local and persisted state
- ❌ Debouncing means changes aren't immediate

**Estimated effort**: 2-3 hours

---

### Solution 3: Virtual DOM Diffing with Immer

**Concept**: Use Immer to produce new state with structural sharing, so unchanged objects keep same references.

```javascript
import { produce } from 'immer'

const newDashboards = produce(dashboards, draft => {
  const widget = draft[0].widgets.find(w => w.id === 'widget1')
  widget.x = 15  // Only this widget gets new reference
})

// widget1: new reference
// widget2, widget3, etc: SAME references as before
```

**Implementation**:
1. Install Immer
2. Wrap all dashboard state updates in `produce()`
3. Only mutate the specific widget that changed
4. Other widgets retain same object identity

**Pros**:
- ✅ Minimal changes to existing code
- ✅ Automatic structural sharing
- ✅ Works with React.memo

**Cons**:
- ❌ Adds dependency
- ❌ Still requires careful update patterns
- ❌ Doesn't solve the broadcast problem (still need to reload)

**Estimated effort**: 1-2 hours

---

### Solution 4: Key-based Forced Stability

**Concept**: Use stable keys and prevent re-renders via shouldComponentUpdate at GridItem level.

```javascript
// In GridItem, prevent re-render if only siblings changed
const GridItem = React.memo(({ config, children }) => {
  // ...
}, (prev, next) => {
  // Only re-render if THIS widget's config changed
  return prev.config.id === next.config.id &&
         prev.config.x === next.config.x &&
         prev.config.y === next.config.y &&
         prev.config.width === next.config.width &&
         prev.config.height === next.config.height
})
```

**Implementation**:
1. Wrap GridItem in React.memo with shallow comparison
2. Only compare position/size properties
3. Ignore other prop changes during drag

**Pros**:
- ✅ Simple implementation
- ✅ No state management changes needed
- ✅ Works immediately

**Cons**:
- ❌ Brittle - easy to break if props change
- ❌ Still does deep comparison on every render
- ❌ Doesn't fix the root cause

**Estimated effort**: 30 minutes

---

### Solution 5: Separate Position State from Config State

**Concept**: Split widget config into two pieces: static config (content, type) and dynamic state (position, size).

```javascript
// Static config (never changes during drag)
const widgetConfigs = {
  editor1: { id: "editor1", type: "codeEditor", language: "js", content: "..." }
}

// Dynamic state (changes during drag)
const widgetPositions = {
  editor1: { x: 100, y: 100, width: 400, height: 300 }
}
```

**Implementation**:
1. Split dashboard JSON into two structures
2. Position updates only change position map
3. Config updates only change config map
4. Widget receives both as separate props

**Pros**:
- ✅ Clear separation of concerns
- ✅ Position updates don't trigger content re-renders
- ✅ Scales well

**Cons**:
- ❌ Major refactor of file format
- ❌ Breaking change to existing dashboards
- ❌ Need migration strategy

**Estimated effort**: 8-10 hours

---

## Comparison Matrix

| Solution | Effectiveness | Effort | Complexity | Breaking Changes |
|----------|--------------|---------|------------|------------------|
| 1. Normalization | ⭐⭐⭐⭐⭐ | 4-6h | Medium | No |
| 2. Local State + Debounce | ⭐⭐⭐⭐ | 2-3h | Low | No |
| 3. Immer | ⭐⭐⭐ | 1-2h | Low | No |
| 4. Key-based Stability | ⭐⭐ | 30m | Low | No |
| 5. Separate State | ⭐⭐⭐⭐⭐ | 8-10h | High | Yes |

## Recommended Approach

**Short-term (Quick Fix)**: Solution 2 - Local State with Debounced Persistence
- Gets you 90% of the way there
- Minimal code changes
- Can implement today

**Long-term (Proper Fix)**: Solution 1 - Normalized State Management
- Solves root cause
- Enables future features (undo/redo, collaborative editing)
- Clean architecture

## Implementation Plan - Solution 2 (Quick Fix)

### Step 1: Modify GridItem to use local state

```javascript
// src/components/GridItem.tsx
const GridItem = ({ x: propX, y: propY, onDragEnd, ... }) => {
  // Local state for position during drag
  const [localPosition, setLocalPosition] = useState({ x: propX, y: propY })

  // Sync props to local state when props change from external source
  useEffect(() => {
    setLocalPosition({ x: propX, y: propY })
  }, [propX, propY])

  // During drag: only update local state
  const handleMouseMove = (e) => {
    if (isDragging) {
      setLocalPosition({ x: newX, y: newY })
      // Don't call onDrag
    }
  }

  // On drag end: call onDragEnd with final position
  const handleMouseUp = () => {
    if (isDragging) {
      onDragEnd?.(localPosition)
    }
  }
}
```

### Step 2: Remove dashboard reload in App.tsx

```javascript
// src/App.tsx
const handleWidgetResize = async (dashboardId, widgetId, dimensions) => {
  await window.dashboardAPI.updateWidgetDimensions(dashboardId, widgetId, dimensions)

  // DON'T reload here - let file watcher handle it
  // Or debounce reload by 500ms
}
```

### Step 3: Add debounced reload (optional)

```javascript
// src/App.tsx
const debouncedReload = useMemo(
  () => debounce(async () => {
    const updated = await window.dashboardAPI.loadDashboards()
    setDashboards(updated)
  }, 500),
  []
)

const handleWidgetResize = async (...) => {
  await window.dashboardAPI.updateWidgetDimensions(...)
  debouncedReload()
}
```

## Testing Checklist

After implementing the fix, verify:

- [ ] Dragging a widget is smooth (no jank)
- [ ] Monaco editors don't flash during drag
- [ ] Monaco editors don't flash at end of drag
- [ ] Position is saved after drag completes
- [ ] Position persists after page reload
- [ ] Resizing works the same way
- [ ] Nested dashboards still work
- [ ] Multiple concurrent drags work
- [ ] External file changes are reflected
- [ ] Undo/redo (if implemented) still works

## Notes

- The core issue is **object reference instability** caused by reloading entire dashboard state
- React.memo alone can't solve this - we need stable object references
- Any solution must balance: performance, correctness, and code complexity
- File-based persistence adds complexity vs. pure React state management

## References

- React.memo documentation: https://react.dev/reference/react/memo
- Immer library: https://immerjs.github.io/immer/
- React rendering optimization: https://react.dev/learn/render-and-commit
