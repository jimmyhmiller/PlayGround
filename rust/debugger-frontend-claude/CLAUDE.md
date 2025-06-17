# Debugger Frontend Claude

This project is a rewrite and modernization of the existing debug-frontend, designed to create a reusable library for debugging custom languages built on top of LLDB.

## Original Code Analysis

### Architecture Overview
The original debug-frontend is a Skia-based GUI application that provides debugging capabilities for a custom language called "Beagle". Key components:

1. **LLDB Integration**: Uses lldb-rs bindings to control debugging sessions
2. **Message Protocol**: Receives structured debugging information via binary messages
3. **WebSocket Server**: Broadcasts debugging messages to external clients
4. **Custom UI**: Skia-based rendering for disassembly, registers, stack, and memory views
5. **Breakpoint Management**: Maps source locations to machine code addresses

### Key Data Structures

#### Core Message Types (`Data` enum):
- `ForeignFunction` / `BuiltinFunction` / `UserFunction`: Function definitions with pointers and metadata
- `Label`: Source-level labels mapped to machine addresses  
- `StackMap`: Stack frame analysis with local variable counts
- `Allocate`: Memory allocation tracking
- `Tokens`: Source file tokenization with line/column mappings
- `Ir`: Intermediate representation with token-to-IR mappings
- `Arm`: ARM assembly with IR-to-machine-code mappings
- `HeapSegmentPointer`: Heap memory segment tracking

#### Key Features:
- **Type System**: Custom tagged pointer system for runtime type identification
- **Memory Views**: Stack and heap inspection with type information
- **Source Mapping**: Multi-level mapping from source → tokens → IR → machine code
- **Breakpoint System**: Advanced breakpoint mapping using `BreakpointMapper`
- **Interactive Debugging**: Step-over, step-in, continue operations
- **Function Filtering**: Regex-based function breakpoint filtering

### Current Limitations:
- Tightly coupled UI and debugging logic
- Hard-coded paths and configurations
- Limited reusability across projects
- Skia dependency makes it heavy
- No command-line interface for testing individual features

## New Architecture Design

### Project Structure
```
debugger-frontend-claude/
├── src/lib.rs                    # Main library interface
├── src/core/                     # Core data structures and protocols
│   ├── mod.rs
│   ├── types.rs                  # BuiltInTypes, Value, Memory types
│   ├── messages.rs               # Message protocol definitions
│   └── serialization.rs          # Binary serialization helpers
├── src/debugger/                 # LLDB client and process management
│   ├── mod.rs
│   ├── client.rs                 # DebuggerClient wrapper
│   ├── process.rs                # Process control and state
│   └── extensions.rs             # LLDB trait extensions
├── src/analysis/                 # Code analysis and mapping
│   ├── mod.rs
│   ├── disassembly.rs           # Disassembly parsing and display
│   ├── memory.rs                # Memory inspection utilities
│   ├── mapping.rs               # Source-to-machine-code mapping
│   └── breakpoints.rs           # Breakpoint management
├── src/display/                  # Display formatting utilities
│   ├── mod.rs
│   ├── formatters.rs            # Value and memory formatters
│   └── layout.rs                # UI layout helpers
├── cli-tools/                   # Command-line testing tools
│   ├── src/main.rs              # CLI entry point
│   ├── src/disasm.rs            # Disassembly command
│   ├── src/memory.rs            # Memory inspection command
│   └── src/breakpoints.rs       # Breakpoint testing command
└── egui-frontend/               # Modern egui-based frontend
    ├── src/main.rs              # GUI entry point
    ├── src/app.rs               # Main application state
    ├── src/views/               # UI components
    │   ├── mod.rs
    │   ├── disassembly.rs
    │   ├── registers.rs
    │   ├── stack.rs
    │   ├── memory.rs
    │   └── controls.rs
    └── src/state.rs             # Application state management
```

### Library API Design

The library will expose high-level APIs for:

1. **DebuggerSession**: Main debugging session management
2. **MessageProcessor**: Handle incoming debugging messages
3. **SourceMapper**: Map between source and machine code
4. **MemoryInspector**: Analyze stack and heap contents
5. **DisassemblyAnalyzer**: Parse and format disassembly
6. **BreakpointManager**: Set and manage breakpoints

### Command-Line Tools

Individual CLI tools for testing each component:
- `debugger-disasm <program> [function]` - Show disassembly for a function
- `debugger-memory <program> <address>` - Inspect memory at address
- `debugger-breakpoints <program> <file:line>` - Test breakpoint mapping
- `debugger-run <program>` - Full debugging session

### Migration Strategy

1. **Phase 1**: Extract core types and message protocol
2. **Phase 2**: Create LLDB client wrapper with clean API
3. **Phase 3**: Build analysis modules (disassembly, memory, mapping)
4. **Phase 4**: Implement CLI tools for testing
5. **Phase 5**: Create egui-based frontend
6. **Phase 6**: Add comprehensive tests and documentation

## Implementation Notes

### Dependencies to Add:
- `lldb` and `lldb-sys` for debugger integration
- `bincode` for message serialization  
- `regex` for function filtering
- `clap` for CLI argument parsing
- `egui` and `eframe` for the new frontend
- `serde` for configuration management

### Key Design Principles:
- **Modularity**: Each component should be usable independently
- **Testability**: Clear interfaces enable unit testing
- **Configurability**: Avoid hard-coded paths and settings  
- **Performance**: Efficient memory usage and fast updates
- **Extensibility**: Easy to add new analysis capabilities

### Testing Strategy:
- Unit tests for each module
- Integration tests with mock LLDB sessions
- CLI tools serve as functional tests
- Property-based testing for message serialization