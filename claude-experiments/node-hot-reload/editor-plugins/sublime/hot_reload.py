"""
Hot Reload - Expression-level hot reloading for JavaScript in Sublime Text

Clojure-style REPL experience with inline evaluation results.
"""

import sublime
import sublime_plugin
import json
import os
import re
import sys
import threading
import queue
import time
import html

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

# Common entry point patterns (in priority order)
ENTRY_PATTERNS = [
    "example/index.js",
    "example/main.js",
    "examples/index.js",
    "src/index.js",
    "src/main.js",
    "src/app.js",
    "index.js",
    "main.js",
    "app.js",
    "lib/index.js",
]

# Directories to skip when looking for entry in package.json
SKIP_DIRS = ["dist", "build", "out", "lib", "node_modules"]

# Phantom styling
PHANTOM_STYLE = """
<style>
    div.hot-result {
        padding: 4px 8px;
        margin: 4px 0;
        border-radius: 4px;
        font-family: monospace;
    }
    div.hot-success {
        background-color: color(var(--greenish) alpha(0.15));
        border-left: 3px solid color(var(--greenish) alpha(0.8));
    }
    div.hot-error {
        background-color: color(var(--redish) alpha(0.15));
        border-left: 3px solid color(var(--redish) alpha(0.8));
    }
    span.hot-arrow {
        color: color(var(--foreground) alpha(0.5));
    }
    span.hot-value {
        color: var(--foreground);
    }
    span.hot-time {
        color: color(var(--foreground) alpha(0.4));
        font-size: 0.9em;
    }
</style>
"""


class EvalResult:
    """Represents an evaluation result with its display state."""

    def __init__(self, view, region, request_id):
        self.view = view
        self.region = region
        self.request_id = request_id
        self.start_time = time.time()
        self.result = None
        self.success = None
        self.phantom_id = None
        self.phantom_set = None

    def set_result(self, success, value, error=None):
        self.success = success
        self.elapsed = time.time() - self.start_time
        if success:
            self.result = value
        else:
            self.result = error

    def format_value(self, value):
        """Format a value for display."""
        if value is None:
            return "nil"
        if value is True:
            return "true"
        if value is False:
            return "false"
        if isinstance(value, str):
            # Truncate long strings
            if len(value) > 200:
                return value[:200] + "..."
            return value
        if isinstance(value, (int, float)):
            return str(value)
        # JSON for objects/arrays
        try:
            formatted = json.dumps(value, indent=2)
            if len(formatted) > 500:
                return json.dumps(value)[:500] + "..."
            return formatted
        except:
            return str(value)

    def get_phantom_html(self):
        """Generate HTML for the inline phantom."""
        elapsed_ms = int(self.elapsed * 1000)
        time_str = "{}ms".format(elapsed_ms) if elapsed_ms < 1000 else "{:.1f}s".format(self.elapsed)

        if self.success:
            value_str = html.escape(self.format_value(self.result))
            return """
            {style}
            <div class="hot-result hot-success">
                <span class="hot-arrow">=&gt;</span>
                <span class="hot-value">{value}</span>
                <span class="hot-time"> ({time})</span>
            </div>
            """.format(style=PHANTOM_STYLE, value=value_str, time=time_str)
        else:
            error_str = html.escape(str(self.result) if self.result else "Unknown error")
            return """
            {style}
            <div class="hot-result hot-error">
                <span class="hot-arrow">!! </span>
                <span class="hot-value">{error}</span>
                <span class="hot-time"> ({time})</span>
            </div>
            """.format(style=PHANTOM_STYLE, error=error_str, time=time_str)

    def show_phantom(self):
        """Display the result as an inline phantom."""
        if not self.view.is_valid():
            return

        # Create phantom set if needed
        if self.phantom_set is None:
            self.phantom_set = sublime.PhantomSet(self.view, "hot_reload_" + self.request_id)

        # Position phantom at end of evaluated region
        point = self.region.end()
        line_region = self.view.line(point)

        phantom = sublime.Phantom(
            sublime.Region(line_region.end()),
            self.get_phantom_html(),
            sublime.LAYOUT_BLOCK
        )

        self.phantom_set.update([phantom])

    def clear_phantom(self):
        """Remove the phantom."""
        if self.phantom_set:
            self.phantom_set.update([])
            self.phantom_set = None


class HotReloadConnection:
    """Manages the WebSocket connection to the hot reload server."""

    def __init__(self):
        self.ws = None
        self.connected = False
        self.source_root = None
        self.server_port = None
        self.pending_evals = {}  # request_id -> EvalResult
        self.request_counter = 0
        self.message_queue = queue.Queue()
        self._thread = None

    def connect(self, host="127.0.0.1", port=3456):
        """Connect to the hot reload server."""
        if self.connected:
            self.disconnect()

        self.server_port = port
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
            self.source_root = msg.get("sourceRoot", "")
            sublime.status_message("[hot] Connected to " + self.source_root)

        elif msg_type == "eval-result":
            request_id = msg.get("requestId")
            success = msg.get("success", False)
            value = msg.get("value")
            error = msg.get("error")
            module_id = msg.get("moduleId", "")

            eval_result = self.pending_evals.get(request_id)
            if eval_result:
                eval_result.set_result(success, value, error)
                if getattr(eval_result, 'show_in_panel', False):
                    self._show_result_panel(eval_result, module_id)
                else:
                    eval_result.show_phantom()
                # Keep in pending for a while for clearing
                sublime.set_timeout(lambda: self._maybe_clear_result(request_id), 10000)

    def _maybe_clear_result(self, request_id):
        """Clear old results after timeout (can be interrupted by new eval)."""
        # Don't auto-clear for now - let user clear manually
        pass

    def _show_result_panel(self, eval_result, module_id):
        """Show result in a pretty-printed side panel."""
        window = sublime.active_window()

        # Format the value
        if eval_result.success:
            value = eval_result.result
            if isinstance(value, (dict, list)):
                try:
                    formatted = json.dumps(value, indent=2, sort_keys=True)
                except:
                    formatted = str(value)
            elif value is None:
                formatted = "null"
            elif value is True:
                formatted = "true"
            elif value is False:
                formatted = "false"
            else:
                formatted = str(value)
            status = "OK"
        else:
            formatted = str(eval_result.result)
            status = "ERROR"

        elapsed_ms = int(eval_result.elapsed * 1000)

        content = formatted

        # Find or create the result view
        result_view = None
        for view in window.views():
            if view.name() == "Hot Reload Result":
                result_view = view
                break

        if result_view is None:
            # Create new view in a side split
            result_view = window.new_file()
            result_view.set_name("Hot Reload Result")
            result_view.set_scratch(True)

            # Move to side group
            num_groups = window.num_groups()
            if num_groups == 1:
                # Create a 2-column layout
                window.set_layout({
                    "cols": [0.0, 0.6, 1.0],
                    "rows": [0.0, 1.0],
                    "cells": [[0, 0, 1, 1], [1, 0, 2, 1]]
                })
            # Move result view to the right group
            window.set_view_index(result_view, window.num_groups() - 1, 0)

        # Set syntax for highlighting
        result_view.assign_syntax("Packages/JavaScript/JSON.sublime-syntax")

        # Disable auto-indent to preserve formatting
        result_view.settings().set("auto_indent", False)
        result_view.settings().set("smart_indent", False)
        result_view.settings().set("indent_to_bracket", False)
        result_view.settings().set("tab_size", 2)

        # Update content - erase and insert to avoid indent issues
        result_view.set_read_only(False)
        result_view.run_command("select_all")
        result_view.run_command("right_delete")
        result_view.run_command("append", {"characters": content, "scroll_to_end": False})
        result_view.set_read_only(True)

        # Move cursor to start
        result_view.sel().clear()
        result_view.sel().add(sublime.Region(0, 0))

        # Update status
        result_view.set_status("hot_reload", "{} | {} | {}ms".format(module_id, status, elapsed_ms))

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

    def eval_expr(self, view, region, expr, show_in_panel=False):
        """Evaluate an expression and show result inline."""
        if not self.connected or not self.ws:
            sublime.status_message("[hot] Not connected. Run Hot Reload: Connect first")
            return None

        request_id = self._generate_request_id()
        module_id = self.get_module_id(view.file_name())

        # Create result tracker
        eval_result = EvalResult(view, region, request_id)
        eval_result.show_in_panel = show_in_panel
        self.pending_evals[request_id] = eval_result

        # Highlight the region being evaluated
        view.add_regions(
            "hot_eval_" + request_id,
            [region],
            "region.bluish",
            "",
            sublime.DRAW_NO_FILL
        )

        # Send eval request
        self.ws.send(json.dumps({
            "type": "eval-request",
            "moduleId": module_id,
            "expr": expr,
            "requestId": request_id
        }))

        sublime.status_message("[hot] Evaluating...")
        return eval_result

    def clear_results(self, view):
        """Clear all inline results for a view."""
        to_remove = []
        for request_id, eval_result in self.pending_evals.items():
            if eval_result.view.id() == view.id():
                eval_result.clear_phantom()
                view.erase_regions("hot_eval_" + request_id)
                to_remove.append(request_id)

        for request_id in to_remove:
            del self.pending_evals[request_id]


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


def find_project_root(start_path):
    """Find project root by looking for package.json or .git."""
    current = start_path
    while current and current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "package.json")):
            return current
        if os.path.exists(os.path.join(current, ".git")):
            return current
        current = os.path.dirname(current)
    return start_path


def find_entry_point(project_root):
    """Find the entry point for a project."""
    # First try common source patterns (prefer these over package.json main which is often dist/)
    for pattern in ENTRY_PATTERNS:
        full_path = os.path.join(project_root, pattern)
        if os.path.exists(full_path):
            return pattern

    # Check package.json for main field (but skip compiled output dirs)
    pkg_path = os.path.join(project_root, "package.json")
    if os.path.exists(pkg_path):
        try:
            with open(pkg_path, 'r') as f:
                pkg = json.load(f)
                if "main" in pkg:
                    main = pkg["main"]
                    # Skip if it's in a build output directory
                    first_dir = main.split("/")[0] if "/" in main else ""
                    if first_dir not in SKIP_DIRS:
                        main_path = os.path.join(project_root, main)
                        if os.path.exists(main_path):
                            return main
        except:
            pass

    return None


def find_available_port():
    """Find an available port for the server."""
    import socket
    for port in DEFAULT_PORTS:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result != 0:
                return port
        except:
            return port
    return DEFAULT_PORTS[0]


def find_top_level_form(view, point):
    """Find the bounds of the top-level form (function/class/variable) at point."""
    content = view.substr(sublime.Region(0, view.size()))

    # Patterns for top-level declarations
    patterns = [
        r'^(async\s+)?function\s+\w+\s*\([^)]*\)\s*\{',
        r'^class\s+\w+(\s+extends\s+\w+)?\s*\{',
        r'^(export\s+)?(const|let|var)\s+\w+\s*=',
        r'^export\s+(async\s+)?function\s+\w+',
        r'^export\s+class\s+\w+',
        r'^export\s+default\s+(async\s+)?function',
        r'^export\s+default\s+class',
    ]

    combined_pattern = '|'.join('(' + p + ')' for p in patterns)

    # Find all matches
    matches = []
    for match in re.finditer(combined_pattern, content, re.MULTILINE):
        matches.append(match.start())

    if not matches:
        # Fall back to current line
        line_region = view.line(point)
        return line_region

    # Find which definition contains our point
    current_start = 0
    for start in matches:
        if start > point:
            break
        current_start = start

    # Find the end by counting braces
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
                    return sublime.Region(current_start, pos + 1)

        pos += 1

    # Check for simple variable declarations without braces (arrow functions, values)
    # e.g., const x = 5; or const fn = () => expr;
    line_end = content.find('\n', current_start)
    if line_end == -1:
        line_end = len(content)

    # Look for semicolon or end of expression
    semicolon = content.find(';', current_start)
    if semicolon != -1 and semicolon < line_end + 100:
        return sublime.Region(current_start, semicolon + 1)

    return sublime.Region(current_start, view.line(current_start).end())


def find_matching_bracket_backward(view, close_pos):
    """Find the matching opening bracket for a closing bracket."""
    close_char = view.substr(close_pos)
    pairs = {')': '(', ']': '[', '}': '{'}
    open_char = pairs.get(close_char)
    if not open_char:
        return None

    count = 1
    pos = close_pos - 1
    in_string = False
    string_char = None

    while pos >= 0 and count > 0:
        char = view.substr(pos)

        # Handle strings (simplified - doesn't handle escapes perfectly)
        if char in '"\'`' and (pos == 0 or view.substr(pos - 1) != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
        elif not in_string:
            if char == close_char:
                count += 1
            elif char == open_char:
                count -= 1

        pos -= 1

    if count == 0:
        return pos + 1
    return None


def expand_to_identifier_start(view, paren_pos):
    """Expand backward from opening paren to include identifier (for function calls)."""
    pos = paren_pos - 1

    # Skip whitespace
    while pos >= 0 and view.substr(pos) in ' \t':
        pos -= 1

    # Check if there's an identifier before
    if pos >= 0 and (view.substr(pos).isalnum() or view.substr(pos) in '_$'):
        # Find start of identifier
        while pos > 0 and (view.substr(pos - 1).isalnum() or view.substr(pos - 1) in '_$'):
            pos -= 1
        return pos

    return paren_pos


def find_enclosing_expression(view, point):
    """Find the innermost balanced expression containing point."""
    # Search backward for unclosed opening brackets
    pos = point - 1
    depth = {'(': 0, '[': 0, '{': 0}
    pairs = {'(': ')', '[': ']', '{': '}'}
    in_string = False
    string_char = None

    while pos >= 0:
        char = view.substr(pos)

        if char in '"\'`' and (pos == 0 or view.substr(pos - 1) != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
        elif not in_string:
            if char in ')]}':
                for o, c in pairs.items():
                    if c == char:
                        depth[o] += 1
            elif char in '([{':
                depth[char] -= 1
                if depth[char] < 0:
                    # Found unclosed opener - now find its closer
                    close_char = pairs[char]
                    close_pos = find_matching_close_forward(view, pos)
                    if close_pos and close_pos >= point:
                        start = expand_to_identifier_start(view, pos)
                        return sublime.Region(start, close_pos + 1)
                    return None
        pos -= 1

    return None


def find_matching_close_forward(view, open_pos):
    """Find matching closing bracket going forward."""
    open_char = view.substr(open_pos)
    pairs = {'(': ')', '[': ']', '{': '}'}
    close_char = pairs.get(open_char)
    if not close_char:
        return None

    count = 1
    pos = open_pos + 1
    size = view.size()
    in_string = False
    string_char = None

    while pos < size and count > 0:
        char = view.substr(pos)

        if char in '"\'`' and view.substr(pos - 1) != '\\':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
        elif not in_string:
            if char == open_char:
                count += 1
            elif char == close_char:
                count -= 1

        pos += 1

    if count == 0:
        return pos - 1
    return None


def find_statement_before(view, point):
    """Find the statement ending at point (before semicolon)."""
    pos = point - 2  # Skip the semicolon itself

    # Skip whitespace
    while pos >= 0 and view.substr(pos) in ' \t\n':
        pos -= 1

    if pos < 0:
        return sublime.Region(0, point)

    # Find start of statement (look for previous semicolon, { or line start)
    start = pos
    while start > 0:
        char = view.substr(start - 1)
        if char in ';\n{':
            break
        start -= 1

    # Skip leading whitespace
    while start < pos and view.substr(start) in ' \t\n':
        start += 1

    return sublime.Region(start, point)


def find_expression_at_point(view, point):
    """Find expression to evaluate at point (CIDER-style eval last sexp)."""

    # Check character immediately before cursor
    if point > 0:
        char_before = view.substr(point - 1)

        # If after closing delimiter, find matching opener and eval the expression
        if char_before in ')]}':
            match_pos = find_matching_bracket_backward(view, point - 1)
            if match_pos is not None:
                start = expand_to_identifier_start(view, match_pos)
                return sublime.Region(start, point)

        # If after semicolon, eval the statement
        if char_before == ';':
            return find_statement_before(view, point)

    # Check if cursor is inside a balanced expression
    enclosing = find_enclosing_expression(view, point)
    if enclosing:
        return enclosing

    # Check if on identifier that's part of a call
    word_region = view.word(point)
    if not word_region.empty():
        word_end = word_region.end()
        if word_end < view.size() and view.substr(word_end) == '(':
            close_pos = find_matching_close_forward(view, word_end)
            if close_pos:
                return sublime.Region(word_region.begin(), close_pos + 1)

        # Just return the word/identifier
        return word_region

    # Fallback: current line
    return view.line(point)


# Commands

class HotReloadJackInCommand(sublime_plugin.WindowCommand):
    """Start a hot reload server for the current project (like CIDER jack-in)."""

    def run(self):
        view = self.window.active_view()
        if not view or not view.file_name():
            sublime.error_message("No file open. Open a file in your project first.")
            return

        # Find project root
        project_root = find_project_root(os.path.dirname(view.file_name()))

        # Find entry point
        entry = find_entry_point(project_root)

        if entry:
            self._start_server(project_root, entry)
        else:
            # Ask user for entry point
            self._ask_entry_point(project_root)

    def _ask_entry_point(self, project_root):
        """Prompt user to enter the entry point."""
        def on_done(entry):
            if entry:
                full_path = os.path.join(project_root, entry)
                if os.path.exists(full_path):
                    self._start_server(project_root, entry)
                else:
                    sublime.error_message("File not found: " + entry)

        self.window.show_input_panel(
            "Entry point (relative to project root):",
            "index.js",
            on_done,
            None,
            None
        )

    def _start_server(self, project_root, entry):
        """Start the hot reload server."""
        import subprocess

        port = find_available_port()
        entry_path = os.path.join(project_root, entry)

        sublime.status_message("[hot] Starting server for {} on port {}...".format(entry, port))

        # Create a terminal/panel to show server output
        panel = self.window.create_output_panel("hot_reload_server")
        panel.settings().set("word_wrap", False)
        self.window.run_command("show_panel", {"panel": "output.hot_reload_server"})

        def run_server():
            try:
                # Run npx hot run
                process = subprocess.Popen(
                    ["npx", "hot", "run", entry_path],
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # Store process for later cleanup
                global _server_process
                _server_process = process

                # Read output and display in panel
                for line in iter(process.stdout.readline, ''):
                    if line:
                        sublime.set_timeout(
                            lambda l=line: self._append_to_panel(panel, l), 0
                        )

                        # Auto-connect when server is ready
                        if "[hot] Server listening" in line or "Connected to dev server" in line:
                            sublime.set_timeout(
                                lambda: self._auto_connect(port), 500
                            )

            except FileNotFoundError:
                sublime.set_timeout(
                    lambda: sublime.error_message(
                        "Could not find 'npx'. Make sure Node.js is installed."
                    ), 0
                )
            except Exception as e:
                sublime.set_timeout(
                    lambda: self._append_to_panel(panel, "Error: " + str(e) + "\n"), 0
                )

        threading.Thread(target=run_server, daemon=True).start()

    def _append_to_panel(self, panel, text):
        panel.set_read_only(False)
        panel.run_command("append", {"characters": text, "scroll_to_end": True})
        panel.set_read_only(True)

    def _auto_connect(self, port):
        """Auto-connect to the server after it starts."""
        if not _connection.connected:
            _connection.connect(port=port)


# Global server process reference
_server_process = None


class HotReloadStopServerCommand(sublime_plugin.WindowCommand):
    """Stop the running hot reload server."""

    def run(self):
        global _server_process
        if _server_process:
            _server_process.terminate()
            _server_process = None
            _connection.disconnect()
            sublime.status_message("[hot] Server stopped")
        else:
            sublime.status_message("[hot] No server running")


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
                    "Start a server with: Hot Reload: Jack In\n"
                    "Or manually: npx hot run ./your-app.js"
                )
                return

            if len(ports) == 1:
                _connection.connect(port=ports[0])
            else:
                # Multiple servers - let user pick
                items = []
                for p in ports:
                    items.append(["localhost:{}".format(p), "Hot reload server on port {}".format(p)])

                def on_select(index):
                    if index >= 0:
                        _connection.connect(port=ports[index])

                self.window.show_quick_panel(items, on_select)

        threading.Thread(
            target=lambda: scan_for_servers(on_servers_found),
            daemon=True
        ).start()


class HotReloadDisconnectCommand(sublime_plugin.WindowCommand):
    """Disconnect from the hot reload server."""

    def run(self):
        _connection.disconnect()
        sublime.status_message("[hot] Disconnected")


class HotReloadEvalTopFormCommand(sublime_plugin.TextCommand):
    """Evaluate the top-level form at cursor (like cider-eval-defun-at-point)."""

    def run(self, edit):
        point = self.view.sel()[0].begin()
        region = find_top_level_form(self.view, point)
        expr = self.view.substr(region)

        if expr.strip():
            _connection.eval_expr(self.view, region, expr)


class HotReloadEvalSelectionCommand(sublime_plugin.TextCommand):
    """Evaluate the selected text."""

    def run(self, edit):
        for sel in self.view.sel():
            if sel.empty():
                # No selection - eval expression at point
                region = find_expression_at_point(self.view, sel.begin())
            else:
                region = sel

            expr = self.view.substr(region)
            if expr.strip():
                _connection.eval_expr(self.view, region, expr)


class HotReloadEvalBufferCommand(sublime_plugin.TextCommand):
    """Evaluate the entire buffer."""

    def run(self, edit):
        region = sublime.Region(0, self.view.size())
        expr = self.view.substr(region)
        _connection.eval_expr(self.view, region, expr)


class HotReloadClearResultsCommand(sublime_plugin.TextCommand):
    """Clear all inline evaluation results."""

    def run(self, edit):
        _connection.clear_results(self.view)
        sublime.status_message("[hot] Results cleared")


class HotReloadEvalToPanelCommand(sublime_plugin.TextCommand):
    """Evaluate selection/top-form and show result in pretty-printed panel."""

    def run(self, edit):
        sel = self.view.sel()[0]
        if sel.empty():
            # No selection - eval top form
            point = sel.begin()
            region = find_top_level_form(self.view, point)
        else:
            region = sel

        expr = self.view.substr(region)
        if expr.strip():
            _connection.eval_expr(self.view, region, expr, show_in_panel=True)


class HotReloadEvalExpressionCommand(sublime_plugin.WindowCommand):
    """Evaluate an expression entered by the user."""

    def run(self):
        view = self.window.active_view()
        module_id = _connection.get_module_id(view.file_name() if view else "")

        def on_done(expr):
            if expr.strip() and view:
                # Create a dummy region at cursor for result display
                point = view.sel()[0].begin() if view.sel() else 0
                region = sublime.Region(point, point)
                _connection.eval_expr(view, region, expr)

        self.window.show_input_panel(
            "Eval ({})".format(module_id),
            "",
            on_done,
            None,
            None
        )


# Event listener for clearing results on edit
class HotReloadEventListener(sublime_plugin.EventListener):
    """Clear stale results when the cursor moves or buffer is modified."""

    def on_selection_modified(self, view):
        # Clear results when cursor moves
        _connection.clear_results(view)

    def on_close(self, view):
        # Clean up any results for this view
        _connection.clear_results(view)
