# Modern JavaScript Test Data

The benchmarks now focus on modern ES6+ JavaScript features:

## Features Covered

### ES6+ (2015-2024)
- ✅ Arrow functions
- ✅ Template literals
- ✅ Destructuring
- ✅ Async/await
- ✅ Promises
- ✅ Classes
- ✅ Modules (import/export)
- ✅ Spread operator
- ✅ Optional chaining (?.)
- ✅ Nullish coalescing (??)
- ✅ Default parameters
- ✅ Rest parameters
- ✅ for...of loops

## Test Cases

### Small Function - Arrow Functions & Template Literals
```javascript
const add = (a, b) => a + b;
const greet = name => `Hello, ${name}!`;
```

### Small Class - ES6 Class Syntax
```javascript
class Calculator {
    constructor() {
        this.result = 0;
    }

    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}
```

### Medium Module - Async/Await & Destructuring
```javascript
class UserDataFetcher {
    constructor() {
        this.cache = new Map();
    }

    async fetchUserData(userId) {
        if (this.cache.has(userId)) {
            return this.cache.get(userId);
        }

        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) throw new Error('Failed to fetch user');

        const data = await response.json();
        this.cache.set(userId, data);
        return data;
    }

    async processUserBatch(userIds) {
        return Promise.all(userIds.map(id => this.fetchUserData(id)));
    }
}
```

### Large Module - Full ES6+ Features
- Async/await
- Classes with private fields
- Optional chaining
- Nullish coalescing
- Spread/rest operators
- Destructuring
- Template literals
- Arrow functions

## Why Modern Features?

1. **Real-world usage** - Modern JavaScript is what developers actually write
2. **Parser stress test** - Modern syntax is more complex to parse
3. **Future-proof** - Tests against current and upcoming language features
4. **Fair comparison** - Eliminates parsers that can't handle modern JS (like older Rhino)
