"""
Hot Reload - Expression-level hot reloading for JavaScript in Sublime Text
"""

import sublime
import sublime_plugin
import json
import os
import re
import sys
import threading
import queue

# Add plugin directory to path for bundled websocket
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

# WebSocket client
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError as e:
    WEBSOCKET_AVAILABLE = False
    print("[hot-reload] websocket-client not available: " + str(e))


# Default ports to scan for servers
DEFAULT_PORTS = [3456, 3457, 3458, 3459, 3460]


class HotReloadConnection:
    """Manages the WebSocket connection to the hot reload server."""

    def __init__(self):
        self.ws = None
        self.connected = False
        self.source_root = None
        self.server_port = None
        self.pending_evals = {}
        self.request_counter = 0
        self.message_queue = queue.Queue()
        self._thread = None

    def connect(self, host="127.0.0.1", port=3456):
        """Connect to the hot reload server."""
        if self.connected:
            self.disconnect()

        self.server_port = port
        # Use 127.0.0.1 to avoid IPv6 issues
        if host == "localhost":
            host = "127.0.0.1"
        url = "ws://{}:{}".format(host, port)

        def on_open(ws):
            self.connected = True
            ws.send(json.dumps({
                "type": "identify",
                "clientType": "editor"
            }))

        def on_message(ws, message):
            try:
                msg = json.loads(message)
                self.message_queue.put(msg)
                sublime.set_timeout(self._process_messages, 0)
            except Exception:
                pass

        def on_error(ws, error):
            sublime.set_timeout(lambda: sublime.status_message("[hot] Error: " + str(error)), 0)

        def on_close(ws, close_status_code, close_msg):
            self.connected = False
            self.source_root = None
            sublime.set_timeout(lambda: sublime.status_message("[hot] Disconnected"), 0)

        self.ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        import socket
        self._thread = threading.Thread(
            target=lambda: self.ws.run_forever(sockopt=((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),)),
            daemon=True
        )
        self._thread.start()

    def disconnect(self):
        """Disconnect from the server."""
        if self.ws:
            self.ws.close()
            self.ws = None
        self.connected = False
        self.source_root = None

    def _process_messages(self):
        """Process queued messages on the main thread."""
        while not self.message_queue.empty():
            try:
                msg = self.message_queue.get_nowait()
                self._handle_message(msg)
            except queue.Empty:
                break

    def _handle_message(self, msg):
        """Handle a message from the server."""
        msg_type = msg.get("type")

        if msg_type == "server-info":
            # Server tells us its source root
            self.source_root = msg.get("sourceRoot", "")
            sublime.status_message("[hot] Connected to " + self.source_root)

        elif msg_type == "eval-result":
            request_id = msg.get("requestId")
            success = msg.get("success", False)
            value = msg.get("value")
            error = msg.get("error")
            module_id = msg.get("moduleId", "")

            if success:
                result_str = self._format_value(value)
                sublime.status_message("[hot] " + module_id + " => " + result_str)
                if len(result_str) > 80:
                    self._show_in_panel(result_str, module_id)
            else:
                sublime.status_message("[hot] Error: " + str(error))
                self._show_in_panel("Error: " + str(error), module_id)

            callback = self.pending_evals.pop(request_id, None)
            if callback:
                callback(msg)

    def _format_value(self, value):
        """Format a value for display."""
        if value is None:
            return "null"
        if value is True:
            return "true"
        if value is False:
            return "false"
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def _show_in_panel(self, result, module_id):
        """Show result in output panel."""
        window = sublime.active_window()
        panel = window.create_output_panel("hot_reload")
        panel.run_command("append", {"characters": "\n;; " + module_id + "\n" + result + "\n"})
        window.run_command("show_panel", {"panel": "output.hot_reload"})

    def _generate_request_id(self):
        """Generate a unique request ID."""
        self.request_counter += 1
        return "sublime-" + str(os.getpid()) + "-" + str(self.request_counter)

    def get_module_id(self, file_path):
        """Get the module ID for a file path."""
        if self.source_root and file_path:
            try:
                return os.path.relpath(file_path, self.source_root)
            except ValueError:
                pass
        return os.path.basename(file_path) if file_path else ""

    def eval_expr(self, expr, module_id, callback=None):
        """Evaluate an expression in the given module context."""
        if not self.connected or not self.ws:
            sublime.status_message("[hot] Not connected. Run Hot Reload: Connect first")
            return

        request_id = self._generate_request_id()

        if callback:
            self.pending_evals[request_id] = callback

        self.ws.send(json.dumps({
            "type": "eval-request",
            "moduleId": module_id,
            "expr": expr,
            "requestId": request_id
        }))

        sublime.status_message("[hot] Evaluating in " + module_id + "...")


# Global connection instance
_connection = HotReloadConnection()


def scan_for_servers(callback):
    """Scan default ports for running hot reload servers."""
    import socket

    found_servers = []

    def check_port(port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result == 0:
                return port
        except Exception:
            pass
        return None

    for port in DEFAULT_PORTS:
        if check_port(port):
            found_servers.append(port)

    sublime.set_timeout(lambda: callback(found_servers), 0)


def find_defun_bounds(view, point):
    """Find the bounds of the function/class definition containing point."""
    content = view.substr(sublime.Region(0, view.size()))

    patterns = [
        r'^(async\s+)?function\s+\w+\s*\([^)]*\)\s*\{',
        r'^class\s+\w+(\s+extends\s+\w+)?\s*\{',
        r'^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?(\([^)]*\)|[^=])\s*=>',
        r'^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?function',
        r'^export\s+(async\s+)?function\s+\w+',
        r'^export\s+class\s+\w+',
        r'^export\s+default\s+(async\s+)?function',
        r'^export\s+default\s+class',
    ]

    combined_pattern = '|'.join('(' + p + ')' for p in patterns)

    matches = []
    for match in re.finditer(combined_pattern, content, re.MULTILINE):
        matches.append(match.start())

    if not matches:
        line_region = view.line(point)
        return (line_region.begin(), line_region.end())

    current_start = 0
    for i, start in enumerate(matches):
        if start > point:
            break
        current_start = start

    brace_count = 0
    in_string = False
    string_char = None
    pos = current_start
    started = False

    while pos < len(content):
        char = content[pos]

        if char in '"\'`' and (pos == 0 or content[pos-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
        elif not in_string:
            if char == '{':
                brace_count += 1
                started = True
            elif char == '}':
                brace_count -= 1
                if started and brace_count == 0:
                    return (current_start, pos + 1)

        pos += 1

    return (current_start, view.line(current_start).end())


class HotReloadConnectCommand(sublime_plugin.WindowCommand):
    """Connect to the hot reload server."""

    def run(self):
        if not WEBSOCKET_AVAILABLE:
            sublime.error_message(
                "websocket-client not installed.\n\n"
                "Run: pip install websocket-client"
            )
            return

        sublime.status_message("[hot] Scanning for servers...")

        def on_servers_found(ports):
            if not ports:
                sublime.error_message(
                    "No hot reload servers found.\n\n"
                    "Start a server with: npx hot run ./your-app.js"
                )
                return

            if len(ports) == 1:
                # Only one server, connect directly
                _connection.connect(port=ports[0])
            else:
                # Multiple servers, let user pick
                items = ["localhost:" + str(p) for p in ports]

                def on_select(index):
                    if index >= 0:
                        _connection.connect(port=ports[index])

                self.window.show_quick_panel(items, on_select)

        # Run scan in background thread
        threading.Thread(
            target=lambda: scan_for_servers(on_servers_found),
            daemon=True
        ).start()


class HotReloadDisconnectCommand(sublime_plugin.WindowCommand):
    """Disconnect from the hot reload server."""

    def run(self):
        _connection.disconnect()
        sublime.status_message("[hot] Disconnected")


class HotReloadEvalSelectionCommand(sublime_plugin.TextCommand):
    """Evaluate the selected text."""

    def run(self, edit):
        for region in self.view.sel():
            if region.empty():
                region = self.view.word(region)

            expr = self.view.substr(region)
            if expr.strip():
                module_id = _connection.get_module_id(self.view.file_name())
                _connection.eval_expr(expr, module_id)


class HotReloadEvalDefunCommand(sublime_plugin.TextCommand):
    """Evaluate the function/class definition at point."""

    def run(self, edit):
        point = self.view.sel()[0].begin()
        start, end = find_defun_bounds(self.view, point)
        expr = self.view.substr(sublime.Region(start, end))

        if expr.strip():
            module_id = _connection.get_module_id(self.view.file_name())
            _connection.eval_expr(expr, module_id)


class HotReloadEvalBufferCommand(sublime_plugin.TextCommand):
    """Evaluate the entire buffer."""

    def run(self, edit):
        expr = self.view.substr(sublime.Region(0, self.view.size()))
        module_id = _connection.get_module_id(self.view.file_name())
        _connection.eval_expr(expr, module_id)


class HotReloadEvalExpressionCommand(sublime_plugin.WindowCommand):
    """Evaluate an expression entered by the user."""

    def run(self):
        view = self.window.active_view()
        module_id = _connection.get_module_id(view.file_name() if view else "")

        def on_done(expr):
            if expr.strip():
                _connection.eval_expr(expr, module_id)

        self.window.show_input_panel(
            "Eval (" + module_id + "):",
            "",
            on_done,
            None,
            None
        )
