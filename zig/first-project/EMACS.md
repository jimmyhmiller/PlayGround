# Using the Lisp REPL with Emacs

This REPL integrates with Emacs `inferior-lisp-mode` for interactive development.

## Setup

1. Build the REPL:
   ```bash
   zig build-exe src/repl.zig
   ```

2. In Emacs, load the configuration:
   ```
   M-x load-file RET /path/to/lisp-repl.el RET
   ```

   Or add to your `~/.emacs` or `~/.emacs.d/init.el`:
   ```elisp
   (load-file "/path/to/lisp-repl.el")
   ```

3. Start the REPL:
   ```
   M-x run-lisp RET
   ```

## Interactive Development Workflow

1. Open a `.lisp` file (e.g., `example.lisp`)
2. Start the REPL if not already running: `M-x run-lisp`
3. Evaluate code from your file:

### Key Bindings

| Key     | Command                        | Description                                      |
|---------|--------------------------------|--------------------------------------------------|
| C-x C-e | lisp-repl-eval-last-sexp-inline| Evaluate expression before point (inline result) |
| C-M-x   | lisp-repl-eval-defun-inline    | Evaluate current defun (inline result)           |
| C-c C-v | lisp-repl--remove-result-overlay| Clear inline result overlay                     |
| C-c C-r | lisp-eval-region               | Evaluate selected region                         |
| C-c C-l | lisp-load-file                 | Load entire file into REPL                       |
| C-c C-z | switch-to-lisp                 | Switch to REPL buffer                            |

### Example Session

1. Open `example.lisp`
2. Position cursor after `(+ 1 2)` and press `C-x C-e`
   - Result appears **inline** after the expression: `=> 3`
   - Result also appears in REPL buffer
3. Position cursor anywhere in the `square` function and press `C-M-x`
   - Function is defined in REPL
4. Position cursor after `(square 7)` and press `C-x C-e`
   - Inline result: `=> 49`
5. Press `C-c C-v` to clear the inline overlay if desired

The inline results appear in **green** (like CIDER) and automatically disappear after 10 seconds.

## Features

- **Inline evaluation (CIDER-style)**: Results appear inline in your source file with `=> result`
- **Multi-line expressions**: The REPL automatically handles expressions spanning multiple lines
- **Persistent definitions**: Functions and variables remain available across evaluations
- **Continuation prompt**: Shows `  ` when waiting for more input to complete an expression
- **Balance checking**: Automatically detects when parentheses are balanced
- **Auto-dismiss**: Inline results automatically disappear after 10 seconds

## REPL Commands

- `(exit)` - Exit the REPL
- Any Lisp expression - Evaluate and print result

## Tips

- The REPL maintains all definitions across evaluations
- You can define functions and use them in subsequent expressions
- Multi-line definitions work seamlessly from Emacs
- Use `C-c C-z` to switch between your file and the REPL buffer
