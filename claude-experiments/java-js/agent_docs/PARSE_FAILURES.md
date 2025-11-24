# Parse Failures Analysis

## Current Status

**Total Parse Failures:** 13 (0.03% of test262 files with cache)
**Match Rate:** 92.10%

## Remaining Failures (13 total)

| Category | Count | Notes |
|----------|-------|-------|
| Expected '{' before class body | 12 | Top-level await edge cases in class extends |
| Dynamic import with 'in' keyword | 1 | `import(x in y)` edge case |

These are edge cases involving:
1. `class C extends (await x) {}` - await in class extends
2. `import(expr in obj, {})` - in operator in dynamic import options

## Fixed Issues

### Static Initialization Blocks (was 198 failures)
```javascript
class C { static { console.log(1); } }
```

### Import Attributes (was 85 failures)
```javascript
import x from './y.json' with { type: 'json' };
import('./y.json', { with: { type: 'json' } });
export * from './x.js' with { type: 'json' };
```

### Assignment Operators (was 90 failures)
```javascript
class C {
  [x ??= 1]() {}  // nullish coalescing assignment
  [x ||= 2]() {}  // logical or assignment
  [x &&= 3]() {}  // logical and assignment
}
```

### Empty Statements in Class Bodies
```javascript
class C { ; method() {} }
```

### Hashbang Comments
```javascript
#!/usr/bin/env node
console.log("Hello");
```

### Private Field `in` Expression
```javascript
class C {
  #field;
  check(obj) { return #field in obj; }
}
```

### String as Import/Export Binding
```javascript
export { x as "string-name" };
import { "string-name" as y } from './x.js';
export * as "All" from './x.js';
```

### Static as Field Name
```javascript
class C { static; }       // field named "static"
class C { static = 1; }   // field named "static" with value
```

### Keywords in Import Attributes
```javascript
import x from './y.json' with { if: 'value' };
```

## Progress Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Parse Failures | 394 | 13 | -381 (97% reduction) |
| Match Rate | 89.80% | 92.10% | +2.30% |
