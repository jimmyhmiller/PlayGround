# Claude Code Hang Analysis - December 22, 2025

## Summary
Claude Code hangs when running specific Bash commands that pipe to `cargo run -- show-ast /dev/stdin` in the lispier project.

## REPRODUCIBLE BUG - Minimal Reproduction

**MINIMAL REPRO** - This simple command causes Claude Code to hang:
```bash
echo "hello" | cat /dev/stdin
```

The original command that triggered investigation:
```bash
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/lispier && \
echo '(require-dialect arith)
(module
  ; Test nested angle brackets
  (artml.constant {:value custom<tensor<4xf32>>}))' | cargo run --quiet -- show-ast /dev/stdin 2>&1
```

## Root Cause Theory

**The issue is `/dev/stdin` specifically.**

1. `/dev/stdin` is a symlink to `/dev/fd/0` - the process's stdin file descriptor
2. Claude Code uses a **persistent shell** that stays alive across commands
3. When a child process opens `/dev/stdin`, it accesses the shell's stdin (the TTY)
4. Claude Code communicates with the shell via this same TTY
5. **DEADLOCK**: Claude Code waits for shell output, but the child process is now reading from the same TTY input

The command output completes, but the shell/Claude Code gets into a bad state afterward because file descriptor 0 was accessed in an unexpected way.

**Key observation**: The command itself completes successfully (output is returned), but Claude Code hangs afterward. The last output before hang showed:
```
Shell cwd was reset to /Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor2/process-test
```
This suggests shell state corruption - the cwd was reset to a completely unrelated directory.

## Original Incident Root Cause
Claude Code sent 3 Bash commands in parallel, but only 1 returned. The other 2 never got responses, leaving the main thread blocked waiting for results.

## Timeline (all times UTC)
- **21:22** - Claude session started
- **04:30:06** - Assistant sent text response
- **04:30:08** - Bash command 1 sent: `test <= inside angle brackets`
- **04:30:09** - Bash command 2 sent: `nested angle brackets`
- **04:30:10** - Bash command 3 sent: `< comparison inside brackets`
- **04:30:13** - Only command 1 returned a result
- **Commands 2 and 3 never returned**

## The Commands That Hung

### Command 2 (no response):
```bash
echo '(require-dialect arith)
(module
  ; Test nested angle brackets
  (arith.constant {:value custom<tensor<4xf32>>}))' | cargo run --quiet -- show-ast /dev/stdin 2>&1
```

### Command 3 (no response):
```bash
echo '(require-dialect arith)
(module
  ; Test standalone < comparison
  (arith.constant {:value expr<(a < b)>}))' | cargo run --quiet -- show-ast /dev/stdin 2>&1
```

## Process State When Hung

| Component | Status |
|-----------|--------|
| Main Thread | Blocked on `__read_nocancel` (blocking read syscall) |
| Bun Thread Pools (3) | All waiting on locks (`__ulock_wait2`) |
| HTTP Client Thread | Waiting for network events (`kevent64`) |
| API Connection | GONE - No active TCP connections to Anthropic API |
| Zombie Child | PID 11618 - spawned at 11:30 PM, never reaped |

## Technical Details

1. The process lost its API connection (no TCP sockets to 160.79.104.10)
2. A zombie child process exists (not properly reaped)
3. Main thread is stuck on a blocking read, likely waiting for subprocess output
4. All Bun thread pools are waiting on locks

## Likely Bug
When running parallel Bash commands that pipe to `cargo run`, if one or more subprocesses fail to return properly, Claude Code's event loop gets stuck waiting for results that never come. The zombie process suggests improper child process handling.

## To Reproduce
Run this command in Claude Code (will cause immediate hang):
```bash
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/lispier && \
echo '(require-dialect arith)
(module
  (arith.constant {:value custom<tensor<4xf32>>}))' | cargo run --quiet -- show-ast /dev/stdin 2>&1
```

## Potential Causes
1. **Shell state corruption** - The "Shell cwd was reset" message suggests Claude Code's persistent shell gets into a bad state
2. **Pipe handling issue** - Something about piping to `cargo run -- show-ast /dev/stdin` triggers the bug
3. **Child process reaping failure** - Zombie process suggests improper waitpid() handling
4. **Event loop deadlock** - Main thread blocks on read while holding locks needed by other threads

## Workaround
**Don't use `/dev/stdin` in piped commands.** Instead of:
```bash
echo "hello" | cat /dev/stdin   # HANGS
echo "hello" | myprogram /dev/stdin  # HANGS
```

Use:
```bash
echo "hello" | cat              # Works
echo "hello" | myprogram -      # Works (if program supports - for stdin)
myprogram <<< "hello"           # Works (here-string)
```

If the hang occurs, you must kill the Claude Code process.

## Environment
- macOS Darwin 25.0.0 (ARM64)
- Claude Code version 2.0.71
- Bun runtime (bundled in Claude Code binary)
