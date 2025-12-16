# Hot Reload Editor Plugins

Expression-level hot reloading for JavaScript, similar to CIDER/Calva for Clojure. Evaluate functions, variables, or expressions directly into a running Node.js process.

## Quick Start

1. Start your app with the hot reload server:
   ```bash
   hot run ./your-app.js
   ```

2. Connect your editor (instructions below)

3. Edit a function and evaluate it with `Ctrl+C Ctrl+C`

## Protocol

Editors communicate with the hot reload server via WebSocket (default port 3456).

### Messages

**Editor → Server: Identify**
```json
{
  "type": "identify",
  "clientType": "editor"
}
```

**Editor → Server: Eval Request**
```json
{
  "type": "eval-request",
  "moduleId": "src/utils.js",
  "expr": "function add(a, b) { return a + b; }",
  "requestId": "unique-id-123"
}
```

**Server → Editor: Eval Result**
```json
{
  "type": "eval-result",
  "requestId": "unique-id-123",
  "moduleId": "src/utils.js",
  "success": true,
  "value": "Defined in src/utils.js",
  "exprType": "declaration"
}
```

---

## Emacs

### Installation

1. Install the `websocket` package:
   ```
   M-x package-install RET websocket RET
   ```

2. Copy `hot-reload.el` to your load path or add to your config:
   ```elisp
   (load "/path/to/hot-reload.el")
   ```

3. Optionally add to your JavaScript mode hook:
   ```elisp
   (add-hook 'js-mode-hook #'hot-reload-mode)
   (add-hook 'typescript-mode-hook #'hot-reload-mode)
   ```

### Usage

1. Connect: `M-x hot-reload-connect`
2. Enter source root when prompted (usually your project root)

### Keybindings (with `hot-reload-mode` enabled)

| Key         | Command                   | Description                      |
|-------------|---------------------------|----------------------------------|
| `C-c C-c`   | `hot-reload-eval-defun`   | Evaluate function at point       |
| `C-x C-e`   | `hot-reload-eval-last-sexp` | Evaluate expression before point |
| `C-c C-r`   | `hot-reload-eval-region`  | Evaluate selected region         |
| `C-c C-k`   | `hot-reload-eval-buffer`  | Evaluate entire buffer           |
| `C-c C-q`   | `hot-reload-disconnect`   | Disconnect from server           |

### Configuration

```elisp
(setq hot-reload-port 3456)           ; WebSocket port
(setq hot-reload-host "localhost")    ; Server host
(setq hot-reload-show-result-in-buffer t)  ; Show in buffer vs minibuffer
```

---

## Sublime Text

### Installation

1. Install `websocket-client`:
   ```bash
   pip install websocket-client
   ```

2. Copy the `sublime/` folder to your Packages directory:
   - macOS: `~/Library/Application Support/Sublime Text/Packages/HotReload/`
   - Linux: `~/.config/sublime-text/Packages/HotReload/`
   - Windows: `%APPDATA%\Sublime Text\Packages\HotReload\`

3. Restart Sublime Text

### Usage

1. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run "Hot Reload: Connect"
3. Enter source root when prompted

### Keybindings

| Key         | Command                        | Description                |
|-------------|--------------------------------|----------------------------|
| `Ctrl+C Ctrl+C` | `hot_reload_eval_defun`    | Evaluate function at point |
| `Ctrl+X Ctrl+E` | `hot_reload_eval_selection`| Evaluate selection         |
| `Ctrl+C Ctrl+K` | `hot_reload_eval_buffer`   | Evaluate entire buffer     |

### Commands (via Command Palette)

- Hot Reload: Connect
- Hot Reload: Disconnect
- Hot Reload: Eval Selection
- Hot Reload: Eval Defun
- Hot Reload: Eval Buffer
- Hot Reload: Eval Expression...

---

## VS Code

### Installation

1. Navigate to the extension directory:
   ```bash
   cd editor-plugins/vscode
   npm install
   npm run compile
   ```

2. Copy to VS Code extensions:
   ```bash
   # macOS/Linux
   cp -r . ~/.vscode/extensions/hot-reload

   # Or create a symlink for development
   ln -s $(pwd) ~/.vscode/extensions/hot-reload
   ```

3. Reload VS Code

### Usage

1. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run "Hot Reload: Connect"
3. Enter source root when prompted

### Keybindings

| Key         | Command                    | Description                |
|-------------|----------------------------|----------------------------|
| `Ctrl+C Ctrl+C` | Eval Function/Class at Cursor | Evaluate function at point |
| `Ctrl+X Ctrl+E` | Eval Selection           | Evaluate selection         |
| `Ctrl+C Ctrl+K` | Eval Buffer              | Evaluate entire buffer     |

### Settings

```json
{
  "hotReload.port": 3456,
  "hotReload.host": "localhost",
  "hotReload.autoConnect": false
}
```

### Features

- Status bar indicator (click to connect)
- Inline result display for expressions
- Output panel for logs and results

---

## Writing Your Own Editor Plugin

Connect to `ws://localhost:3456` (or configured port) and implement:

1. **On connect**: Send identify message
2. **To evaluate**: Send eval-request with moduleId, expr, requestId
3. **Handle results**: Parse eval-result messages

The server transforms your expression and sends it to the running Node.js process. Results are sent back to your editor.

### Example (Python)

```python
import websocket
import json
import uuid

ws = websocket.create_connection("ws://localhost:3456")

# Identify as editor
ws.send(json.dumps({"type": "identify", "clientType": "editor"}))

# Evaluate expression
request_id = str(uuid.uuid4())
ws.send(json.dumps({
    "type": "eval-request",
    "moduleId": "src/app.js",
    "expr": "function hello() { return 'world'; }",
    "requestId": request_id
}))

# Wait for result
result = json.loads(ws.recv())
print(result)
# {"type": "eval-result", "success": true, "value": "Defined in src/app.js", ...}
```

### Example (Node.js)

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3456');

ws.on('open', () => {
  ws.send(JSON.stringify({ type: 'identify', clientType: 'editor' }));

  ws.send(JSON.stringify({
    type: 'eval-request',
    moduleId: 'src/app.js',
    expr: 'function hello() { return "world"; }',
    requestId: 'req-1'
  }));
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  if (msg.type === 'eval-result') {
    console.log(msg.success ? msg.value : msg.error);
  }
});
```
