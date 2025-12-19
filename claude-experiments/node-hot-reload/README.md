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
