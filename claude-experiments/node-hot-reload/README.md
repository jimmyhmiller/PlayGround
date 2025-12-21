# Hot Reload

Expression-level hot reloading for JavaScript. Evaluate code directly in your running Node.js process from your editor, like CIDER/Calva for Clojure.

## Quick Start

```bash
# Install dependencies
npm install

# Build
npm run build

# Run your app with hot reloading
npx hot run ./example/index.js
```

## Editor Setup

### Sublime Text

1. Copy `editor-plugins/sublime/` contents to `~/Library/Application Support/Sublime Text/Packages/HotReload/`
2. Install websocket-client: `pip install websocket-client`
3. Create `.python-version` file in the package directory containing `3.8`

### Keybindings Cheatsheet (Sublime Text)

| Keybinding | Command | Description |
|------------|---------|-------------|
| `Ctrl+C, Ctrl+J` | Jack In | Start server for project & auto-connect |
| `Ctrl+C, Ctrl+L` | Connect | Connect to running server (with picker) |
| `Ctrl+C, Ctrl+C` | Eval Top Form | Evaluate function/class at cursor |
| `Ctrl+C, Alt+E` | Eval Last Sexp | Evaluate expression at/before cursor (CIDER-style) |
| `Ctrl+C, Ctrl+P` | Eval to Panel | Evaluate & show result in side panel |
| `Ctrl+C, Ctrl+K` | Eval Buffer | Evaluate entire file |
| `Ctrl+C, Ctrl+Q` | Clear Results | Clear inline result phantoms |

**Eval Last Sexp behavior:**
- Cursor after `)` `]` `}` → evals the complete expression
- Cursor after `;` → evals the statement
- Cursor inside parens → evals enclosing expression
- Cursor on identifier → evals the identifier (or call if followed by `(`)

### Commands (Command Palette)

- **Hot Reload: Jack In (Start Server)** - Find project entry point and start server
- **Hot Reload: Stop Server** - Stop the running server
- **Hot Reload: Connect** - Connect to a running server
- **Hot Reload: Disconnect** - Disconnect from server
- **Hot Reload: Eval Selection/At Point** - Evaluate selected code
- **Hot Reload: Eval Top Form (Defun)** - Evaluate enclosing function/class
- **Hot Reload: Eval Buffer** - Evaluate entire file
- **Hot Reload: Eval to Panel (Pretty Print)** - Evaluate and show in side panel
- **Hot Reload: Clear Results** - Remove inline result displays

## How It Works

1. **Transform**: Your ES module code is transformed to use a `__hot` runtime that tracks module exports
2. **Eval**: When you evaluate code from your editor, it runs in the context of the module with access to all bindings
3. **Reload**: File changes are detected and the module is hot-reloaded without restarting

## Project Structure

```
src/
  cli.ts        # CLI entry point
  transform.ts  # Babel transform for hot reloading
  runtime.ts    # Runtime that manages modules
  server.ts     # WebSocket server for editor communication

editor-plugins/
  sublime/      # Sublime Text plugin
  emacs/        # Emacs plugin (elisp)
  vscode/       # VS Code extension

example/        # Example project to test with
```

## Integration Guide

### Installing in Your Project

```bash
# From npm (when published)
npm install hot-reload

# Or link locally during development
cd /path/to/hot-reload
npm link
cd /path/to/your-project
npm link hot-reload
```

### Method 1: CLI (Recommended)

The simplest way to use hot-reload is via the CLI:

```bash
# Run your app with hot reloading enabled
npx hot run ./src/index.js
```

This automatically:
- Discovers all your project's modules
- Transforms them for hot reloading
- Starts the WebSocket server (default port 3456)
- Connects the runtime

### Method 2: Require Hook

Add hot-reload to your app's entry point:

```javascript
// At the very top of your entry file
require('hot-reload/register');

// Rest of your code
const app = require('./app');
```

Or use Node's `-r` flag:

```bash
node -r hot-reload/register ./src/index.js
```

You can customize the port via environment variable:

```bash
HOT_PORT=4000 node -r hot-reload/register ./src/index.js
```

### Method 3: Programmatic API

For full control, use the API directly:

```javascript
const { createRuntime, startServer, transform } = require('hot-reload');

// Create runtime
const runtime = createRuntime();
global.__hot = runtime;

// Start WebSocket server
const server = startServer({
  port: 3456,
  sourceRoot: process.cwd()
});

// Connect runtime to server
runtime.connect(3456);

// Transform and load your modules...
```

## API Reference

### `defonce(value)`

Preserves a value across hot reloads. Use for state that should survive code changes:

```javascript
const { defonce } = require('hot-reload/api');

// Cache persists across reloads
const cache = defonce(new Map());

// Counter keeps its value
let connectionCount = defonce(0);

// Server instance created only once
const server = defonce(createServer());
```

### `once(expression)`

Executes an expression only once, even if the module is reloaded. Use for side effects:

```javascript
const { once } = require('hot-reload/api');

// Only registers handler once
once(ipcMain.handle('get-data', async () => {
  return getData();
}));

// Only logs once
once(console.log('Module initialized'));

// Only adds listener once
once(process.on('exit', cleanup));
```

## Common Patterns

### Stateful Modules

```javascript
const { defonce } = require('hot-reload/api');

// State preserved across reloads
const users = defonce(new Map());
let requestCount = defonce(0);

// These functions can be hot-reloaded
export function addUser(id, data) {
  users.set(id, data);
  requestCount++;
}

export function getUser(id) {
  return users.get(id);
}
```

### Event Handlers (Electron/IPC)

```javascript
const { once, defonce } = require('hot-reload/api');
const { ipcMain } = require('electron');

// Register handlers once
once(ipcMain.handle('save-file', handleSave));
once(ipcMain.handle('load-file', handleLoad));

// Handler implementations can be reloaded
function handleSave(event, data) {
  // Your logic here - edit and reload!
}

function handleLoad(event, path) {
  // Your logic here - edit and reload!
}
```

### Express/HTTP Servers

```javascript
const { defonce, once } = require('hot-reload/api');
const express = require('express');

// Server created once
const app = defonce(express());
const server = defonce(null);

// Routes can be hot-reloaded using a router
export const router = express.Router();

router.get('/api/users', (req, res) => {
  // Edit this and reload!
  res.json({ users: [] });
});

// Mount router once
once(app.use(router));

// Start server once
once(() => {
  server = app.listen(3000);
  console.log('Server running on port 3000');
})();
```

### WebSocket Connections

```javascript
const { defonce, once } = require('hot-reload/api');
const WebSocket = require('ws');

// Server persists across reloads
const wss = defonce(new WebSocket.Server({ port: 8080 }));
const clients = defonce(new Set());

// Connection handler registered once
once(wss.on('connection', (ws) => {
  clients.add(ws);
  ws.on('close', () => clients.delete(ws));
  ws.on('message', handleMessage);
}));

// Message handler can be reloaded
function handleMessage(data) {
  // Edit this and reload!
  console.log('Received:', data);
}
```

## Editor Workflow

1. **Start your app** with hot-reload enabled
2. **Connect your editor** to the WebSocket server
3. **Edit code** in your editor
4. **Evaluate** with keyboard shortcuts:
   - Eval expression at cursor
   - Eval entire function/class
   - Eval entire buffer
5. **See results** inline in your editor

The running process updates immediately - no restart needed!

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOT_PORT` | `3456` | WebSocket server port |

## Troubleshooting

### Port already in use

The CLI automatically tries ports 3456-3460. Set a specific port:

```bash
HOT_PORT=4000 npx hot run ./src/index.js
```

### Module not found errors

Ensure your entry point uses relative paths:

```javascript
// Good
const utils = require('./utils');

// May not work
const utils = require('utils');
```

### State not preserved

Make sure you're using `defonce()`:

```javascript
// Won't preserve - recreated each reload
const cache = new Map();

// Will preserve
const cache = defonce(new Map());
```

### Side effects running multiple times

Wrap side effects with `once()`:

```javascript
// Runs every reload
console.log('loaded');

// Runs only once
once(console.log('loaded'));
```

## Comparison with nodemon/node --watch

| Feature | hot-reload | nodemon |
|---------|------------|---------|
| Restart process | No | Yes |
| Preserve state | Yes | No |
| Eval from editor | Yes | No |
| Speed | Instant | Seconds |
| Connection state | Preserved | Lost |

Hot-reload updates code in-place without restarting, making it ideal for:
- Long-running servers with active connections
- Stateful applications (databases, caches)
- Interactive development with immediate feedback

## Production Builds

The `once` and `defonce` functions are imported from `hot-reload/api` and work as identity functions at runtime. For production builds, you can strip these calls entirely.

### CLI Strip Command

```bash
# Strip a single file (output to stdout)
npx hot strip ./src/app.js

# Strip a single file to output file
npx hot strip ./src/app.js -o ./dist/app.js

# Strip an entire directory
npx hot strip -d ./src -o ./dist

# Custom file extensions
npx hot strip -d ./src -o ./dist -e .js,.mjs
```

### Programmatic API

```javascript
const { strip } = require('hot-reload/strip');

const code = `
import { once, defonce } from 'hot-reload/api';

const cache = defonce(new Map());
once(console.log('init'));
`;

const stripped = strip(code);
// Output:
// const cache = new Map();
// console.log('init');
```

### Babel Plugin

Add to your `babel.config.js` for automatic stripping during build:

```javascript
// babel.config.js
module.exports = {
  presets: ['@babel/preset-env'],
  plugins: [
    // Only strip in production
    process.env.NODE_ENV === 'production' && 'hot-reload/strip'
  ].filter(Boolean)
};
```

### Build Script Example

```json
{
  "scripts": {
    "dev": "hot run ./src/index.js",
    "build": "hot strip -d ./src -o ./dist && node ./dist/index.js",
    "start": "node ./dist/index.js"
  }
}
```

### Without Hot-Reload Installed

Since `once` and `defonce` are identity functions, your code works without the hot-reload package installed. You just need to provide your own stubs:

```javascript
// hot-reload-stub.js (for production without hot-reload installed)
export const once = (v) => v;
export const defonce = (v) => v;
```

Then alias `hot-reload/api` to your stub in your bundler:

```javascript
// webpack.config.js
module.exports = {
  resolve: {
    alias: {
      'hot-reload/api': './hot-reload-stub.js'
    }
  }
};
```

```javascript
// vite.config.js
export default {
  resolve: {
    alias: {
      'hot-reload/api': './hot-reload-stub.js'
    }
  }
};
```
