# Beagle VS Code Eval

Live evaluation of Beagle code via the socket REPL server.

## Usage

1. Start the Beagle REPL server: `cargo run -- resources/examples/repl_server.bg`
2. Open a `.bg` file in VS Code
3. Run "Beagle: Eval File" from the command palette

## Settings

- `beagle.repl.host` - REPL server host (default: `127.0.0.1`)
- `beagle.repl.port` - REPL server port (default: `7888`)
