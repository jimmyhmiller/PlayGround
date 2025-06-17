# Debugger Frontend Claude

A reusable library for building debuggers on top of LLDB, with a focus on custom language support.

## Overview

This project modernizes and modularizes the existing debug-frontend to create a reusable library for custom language debugging. The library provides:

- **Message Protocol Handling**: Binary serialization for debugging messages
- **Memory and Type Analysis**: Tagged pointer system for runtime type identification  
- **Source-to-Machine-Code Mapping**: Multi-level mapping from source → tokens → IR → machine code
- **Disassembly Analysis**: Instruction parsing and display formatting
- **Breakpoint Management**: Advanced breakpoint mapping between source and machine addresses
- **LLDB Client Wrapper**: High-level API for LLDB interactions

## Architecture

The project is organized as a workspace with:

- **`debugger-frontend-claude/`** - Core library with modular components
- **`cli-tools/`** - Command-line tools for testing library functionality
- **`egui-frontend/`** - Modern egui-based graphical frontend (TODO)

### Library Structure

```
src/
├── core/           # Core data types and message protocol
│   ├── types.rs    # BuiltInTypes, Value, Memory, Register types
│   ├── messages.rs # DebugMessage and MessageData definitions
│   └── serialization.rs # Binary serialization helpers
├── debugger/       # LLDB client and process management
│   ├── client.rs   # DebuggerClient wrapper
│   ├── process.rs  # ProcessState management
│   └── extensions.rs # LLDB trait extensions
├── analysis/       # Code analysis and mapping
│   ├── disassembly.rs # DisassemblyAnalyzer
│   ├── memory.rs   # MemoryInspector
│   ├── mapping.rs  # SourceMapper
│   └── breakpoints.rs # BreakpointMapper
└── display/        # Display formatting utilities
    ├── formatters.rs # Value and memory formatters
    └── layout.rs   # UI layout helpers
```

## Command-Line Tools

The CLI tools demonstrate library functionality and can be used for testing:

### `debugger-disasm`
Show disassembly for a function in a debugged program.

```bash
cargo run --bin debugger-disasm -- /path/to/program --function main --window 20 --addresses
```

### `debugger-memory`  
Inspect memory at a specific address.

```bash
cargo run --bin debugger-memory -- /path/to/program 0x12345678 --size 64 --types
cargo run --bin debugger-memory -- /path/to/program --stack --types
```

### `debugger-breakpoints`
Test breakpoint mapping between source and machine code.

```bash
cargo run --bin debugger-breakpoints -- /path/to/program --source main.bg:10
cargo run --bin debugger-breakpoints -- /path/to/program --address 0x100154000
cargo run --bin debugger-breakpoints -- /path/to/program --list
```

### `debugger-run`
Interactive debugging session with TUI-like interface.

```bash
cargo run --bin debugger-run -- /path/to/program --args arg1 arg2
```

## Building

```bash
# Build the library
cargo build

# Build CLI tools
cargo build -p cli-tools

# Run tests
cargo test

# Check all components
cargo check --workspace
```

## Dependencies

- **LLDB**: Requires LLDB development libraries
- **Rust**: 2021 edition with stable toolchain
- **Platform**: Currently supports macOS (Darwin), can be extended to Linux

## Usage Example

```rust
use debugger_frontend_claude::{
    DebuggerClient, ProcessState, DisassemblyAnalyzer, 
    MemoryInspector, SourceMapper
};

fn main() -> anyhow::Result<()> {
    // Create debugger client
    let mut client = DebuggerClient::new()?;
    
    // Set up target and breakpoints
    client.create_target("/path/to/program")?;
    client.set_breakpoint_by_name("main", "main")?;
    client.launch_process(vec![])?;
    
    // Initialize analysis components
    let mut state = ProcessState::new();
    let mut disasm = DisassemblyAnalyzer::new();
    let mut memory = MemoryInspector::new();
    let mapper = SourceMapper::new();
    
    // Update state from stopped process
    let process = client.get_process().unwrap();
    let target = client.get_target().unwrap();
    state.update_from_process(process, target)?;
    
    // Analyze and display
    println!("PC: 0x{:x}", state.pc);
    for instruction in &state.instructions {
        disasm.add_instruction(instruction.clone());
    }
    
    let instructions = disasm.get_instructions_around_pc(state.pc, 10);
    for inst in instructions {
        println!("{}", inst.to_string(true, false));
    }
    
    Ok(())
}
```

## Original vs New

### Original (debug-frontend)
- Monolithic Skia-based GUI application
- Tightly coupled UI and debugging logic  
- Hard-coded paths and configurations
- Limited reusability

### New (debugger-frontend-claude)
- Modular library with clean separation of concerns
- Multiple frontends: CLI tools + future egui GUI
- Configurable and extensible design
- Reusable across different projects
- Comprehensive testing infrastructure

## Roadmap

- [x] Core library implementation
- [x] CLI tools for testing
- [ ] egui-based graphical frontend  
- [ ] Comprehensive test suite
- [ ] Documentation and examples
- [ ] Cross-platform support (Linux, Windows)
- [ ] Performance optimizations
- [ ] Plugin system for custom language support

## Contributing

This project demonstrates a complete rewrite and modernization of a debugging frontend. The code is structured to be educational and reusable, showing best practices for:

- Rust library design and modularization
- LLDB integration and wrapper APIs
- Binary protocol handling and serialization
- Command-line tool development
- Test-driven development

See `CLAUDE.md` for detailed analysis and architectural decisions.