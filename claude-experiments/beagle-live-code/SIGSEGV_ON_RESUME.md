# SIGSEGV on main-resume after REPL error recovery

## Date
2026-03-27

## Summary
When the main thread (running a raylib game loop) throws an exception, gets suspended by the REPL recovery mechanism, and then `main-resume` is sent after fixing the function via `eval`, the Beagle process crashes with SIGSEGV.

## Steps to Reproduce

### 1. Start the raylib game via the agent's REPL wrapper

```
beag run -I /Users/jimmyhmiller/Documents/Code/beagle/examples /tmp/__beagle_repl_runner.bg
```

Where `__beagle_repl_runner.bg` wraps `raylib_game.bg` with the REPL server:

```beagle
namespace __repl_runner

use beagle.repl-main as repl-main
use raylib_game as target

fn main() {
    eval("namespace raylib_game")
    repl-main/run-with-repl("127.0.0.1", 7888, fn() {
        target/main()
    })
}
```

### 2. Connect to the REPL and eval a broken render function

Send to REPL (port 7888):
```json
{"op":"eval","id":"1","code":"fn render(game) {\n    BeginDrawing()\n    ClearBackground(BG)\n    for ball in game.balls {\n        let c = ball.nonexistent_field + 1\n        DrawRectangle(ball.x, ball.y, ball.size, ball.size, c)\n    }\n    EndDrawing()\n}"}
```

The game loop calls the redefined `render`, which throws:
```
Uncaught exception in main thread (caught by REPL recovery):
SystemError.FieldError { message: "Field 'nonexistent_field' does not exist on Ball", location: null }
Waiting for REPL resume or abort...
```

### 3. Fix the function via eval

Send to REPL:
```json
{"op":"eval","id":"2","code":"fn render(game) {\n    BeginDrawing()\n    ClearBackground(BG)\n    for ball in game.balls {\n        DrawRectangle(ball.x, ball.y, ball.size, ball.size, SKYBLUE)\n    }\n    EndDrawing()\n}"}
```

This succeeds (the function is redefined).

### 4. Resume the main thread

Send to REPL:
```json
{"op":"main-resume","id":"3"}
```

**Result**: Process exits with signal `SIGSEGV`.

## Expected Behavior
The main thread resumes, calls the now-fixed `render` function, and the game loop continues.

## Actual Behavior
The Beagle process crashes with SIGSEGV immediately upon receiving the `main-resume` op.

## Key Details

- The error itself (`FieldError`) was caught correctly by the REPL recovery mechanism
- The main thread was suspended successfully ("Waiting for REPL resume or abort...")
- The eval to fix the function succeeded on the REPL thread
- The crash only happens on the resume, not during any of the prior steps
- Raylib had initialized and was rendering frames successfully before the error was introduced

## Possible Areas to Investigate

- **GC during suspension**: While the main thread is suspended and the REPL thread runs an eval, the GC may collect objects referenced by the suspended main thread's stack/continuation
- **Continuation/stack restore**: The mechanism that captures and restores the main thread's execution state on resume may have a bug
- **Thread safety of function redefinition**: The eval redefines `render` on the REPL thread; when the main thread resumes, it may be reading a partially-written function pointer or closure
