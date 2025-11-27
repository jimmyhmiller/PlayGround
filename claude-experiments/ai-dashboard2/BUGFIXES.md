# Critical Bug Fixes

## Issue 1: Black Screen / UI Crashes ❌→✅

### Problem
UI was going completely black due to uncaught JavaScript errors.

### Root Cause
**Missing `widgetKey` dependency in useEffect** (line 2309)
- The question listener useEffect was using `widgetKey` variable
- `widgetKey` wasn't in the dependency array or in scope
- This caused a `ReferenceError` that crashed the entire component
- No error boundaries were in place to catch it

### Fix Applied
1. ✅ Added `widgetKey` to dependency array (line 2331)
2. ✅ Added safety check for `window.claudeAPI.offQuestion` (line 2327)
3. ✅ Implemented **ErrorBoundary** component (lines 8-91)
4. ✅ Wrapped each widget with ErrorBoundary (line 2792)
5. ✅ Wrapped entire App with top-level ErrorBoundary (line 3880)

### Code Changes
```javascript
// Before (BROKEN):
useEffect(() => {
  // ... uses widgetKey but not in deps
}, [backend, currentConversationId]); // Missing widgetKey!

// After (FIXED):
useEffect(() => {
  // ... uses widgetKey
  return () => {
    if (window.claudeAPI && window.claudeAPI.offQuestion) {
      window.claudeAPI.offQuestion(questionHandler);
    }
  };
}, [backend, currentConversationId, widgetKey]); // ✅ Added widgetKey
```

---

## Issue 2: Dashboard Keeps Switching Back ❌→✅

### Problem
User keeps getting "booted off" their current dashboard back to the first one.

### Root Cause
**No persistence for active dashboard ID**
1. `activeId` state not persisted to localStorage
2. On dashboard updates (`onDashboardUpdate`), the logic could reset to first dashboard
3. Fallback logic `dashboards[0]` was too aggressive

### Issues Identified
```javascript
// PROBLEM 1: Not persisted
const [activeId, setActiveId] = useState(null); // Lost on reload!

// PROBLEM 2: Aggressive fallback
const activeDashboard = dashboards.find((d) => d.id === activeId) || dashboards[0];
// ↑ Falls back to first dashboard even when activeId is set!
```

### Fix Applied
1. ✅ **Persist activeId to localStorage** (lines 3771-3796)
2. ✅ **Validate activeId on load** (lines 3835-3841)
3. ✅ **Validate activeId on updates** (lines 3856-3869)
4. ✅ **Smart fallback logic** (lines 3929-3930)
5. ✅ **Added debug logging** to track dashboard switches

### Code Changes

**1. Load from localStorage:**
```javascript
const [activeId, setActiveId] = useState(() => {
  try {
    return localStorage.getItem('activeDashboardId') || null;
  } catch (e) {
    return null;
  }
});
```

**2. Save to localStorage:**
```javascript
useEffect(() => {
  if (activeId) {
    try {
      localStorage.setItem('activeDashboardId', activeId);
    } catch (e) {
      console.error('Failed to save active dashboard ID:', e);
    }
  }
}, [activeId]);
```

**3. Validate on load:**
```javascript
window.dashboardAPI.loadDashboards().then((loaded) => {
  setDashboards(loaded);

  // Validate that activeId exists in loaded dashboards
  if (activeId && !loaded.find(d => d.id === activeId)) {
    console.warn(`Stored dashboard ID ${activeId} not found`);
    setActiveId(loaded[0]?.id || null);
  } else if (!activeId && loaded.length > 0) {
    setActiveId(loaded[0].id);
  }
});
```

**4. Validate on updates:**
```javascript
window.dashboardAPI.onDashboardUpdate((updated) => {
  setDashboards(updated);
  setActiveId(prev => {
    // Keep activeId if it still exists
    if (prev && updated.find(d => d.id === prev)) {
      return prev; // ✅ Don't change!
    }
    // Only fallback if dashboard was deleted
    return updated[0]?.id || null;
  });
});
```

**5. Smart fallback:**
```javascript
// Only fallback to first dashboard if NO activeId selected
const activeDashboard = dashboards.find((d) => d.id === activeId) ||
  (!activeId && dashboards.length > 0 ? dashboards[0] : null);
```

---

## Testing the Fixes

### Test 1: Black Screen Fix
1. Open any dashboard
2. UI should render without black screen
3. If a widget errors, only that widget shows error UI
4. Other widgets continue working
5. Console shows error details

### Test 2: Dashboard Persistence
1. Select a specific dashboard (not the first one)
2. Refresh the page
3. ✅ Should stay on the same dashboard
4. Resize/modify a widget
5. ✅ Should stay on the same dashboard
6. Switch to another dashboard
7. Refresh again
8. ✅ Should stay on the newly selected dashboard

### Console Debugging
Look for these log messages:
- `[App] Dashboard update received, current activeId: <id>`
- `[App] Keeping activeId: <id>` ✅ Good
- `[App] Setting activeId to first dashboard: <id>` ⚠️ Check why
- `Stored dashboard ID <id> not found, selecting first dashboard` ⚠️ Dashboard was deleted

---

## Files Modified

1. **src/App.jsx**
   - Line 8-91: Added ErrorBoundary class component
   - Line 2331: Fixed widgetKey dependency
   - Line 2327-2329: Added safety checks
   - Line 2792-2808: Wrapped widgets in ErrorBoundary
   - Line 3771-3796: Added localStorage persistence
   - Line 3835-3841: Added validation on load
   - Line 3856-3869: Added validation on updates
   - Line 3929-3930: Fixed fallback logic
   - Line 3880-3883: Wrapped App in ErrorBoundary

2. **ERROR_BOUNDARY_DOCS.md** (new)
   - Documentation for error boundary system

3. **BUGFIXES.md** (this file)
   - Documentation of fixes applied

---

## Known Remaining Issues

### Minor Issues
- ⚠️ Error boundaries don't catch async errors (use try-catch)
- ⚠️ Error boundaries don't catch event handler errors
- ⚠️ Large bundle size warning (>500KB)

### Future Enhancements
- Add error reporting service (Sentry)
- Implement auto-retry for transient errors
- Add loading states during dashboard switches
- Persist activeId to backend instead of localStorage
- Add undo/redo for dashboard switching

---

## Prevention Checklist

To prevent similar issues in the future:

✅ **Always include error boundaries around:**
- New widgets/components
- User-facing features
- Third-party integrations

✅ **Always persist UI state when:**
- User makes a selection
- State affects navigation
- Loss would be frustrating

✅ **Always add to dependency arrays:**
- Any external variables used in useEffect
- Props and state referenced inside
- Functions from outer scope

✅ **Always validate state on updates:**
- Check IDs still exist after changes
- Don't blindly fallback
- Log unexpected state transitions

✅ **Always test error scenarios:**
- Component throws during render
- Async operations fail
- Network requests timeout
- Invalid data received
