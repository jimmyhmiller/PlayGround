# ARM Code Generator - Generic Library

A Rust library that generates ARM assembly encoder functions in multiple programming languages. Parse ARM specifications once, generate type-safe encoder functions in Rust, C++, Python, or any language you implement.

## Features

- **799 ARM instructions** from official ARM specification
- **Multi-language support**: Rust, C++, Python generators included
- **Function-based API**: Direct `add(rd, rn, imm)` calls instead of enums
- **Snake case naming**: `AddAddsubImm` → `add_addsub_imm` 
- **Instruction discovery**: Find instructions by pattern or category
- **Flexible filtering**: Generate only the instructions you need
- **Zero-cost abstractions**: `constexpr` C++, inlined Rust
- **Type safety**: Proper Register and immediate parameter types

## Quick Start

```rust
use arm_codegen_generic::{ArmCodeGen, rust_function_generator::RustFunctionGenerator};

// One-time initialization (loads all ARM instructions)
let arm = ArmCodeGen::new()?;

// Generate Rust functions for specific instructions
let rust_code = arm.generate(
    RustFunctionGenerator,
    vec!["AddAddsubImm", "MovMovzImm"]
);

// Generated functions look like:
// pub fn add_addsub_imm(sf: i32, sh: i32, imm12: i32, rn: Register, rd: Register) -> u32
// pub fn mov_movz_imm(sf: i32, hw: i32, imm16: i32, rd: Register) -> u32
```

## Examples

### Basic Usage
```rust
use arm_codegen_generic::{ArmCodeGen, rust_function_generator::RustFunctionGenerator};

let arm = ArmCodeGen::new()?;
let code = arm.generate(RustFunctionGenerator, vec!["AddAddsubImm"]);
```

### Instruction Discovery
```rust
let arm = ArmCodeGen::new()?;

// Find all ADD-related instructions
let add_instructions = arm.find_instructions("add");
println!("Found {} ADD instructions", add_instructions.len());

// Get instruction details
let (name, title) = arm.instruction_info("AddAddsubImm").unwrap();
println!("{} - {}", name, title);
```

### Advanced Filtering
```rust
use arm_codegen_generic::{ArmCodeGen, InstructionFilter, cpp_function_generator::CppFunctionGenerator};

let arm = ArmCodeGen::new()?;

let filter = InstructionFilter::new()
    .allow(vec!["AddAddsubImm".to_string(), "MovMovzImm".to_string()])
    .block(vec!["Deprecated".to_string()]);

let cpp_code = arm.generate_filtered(CppFunctionGenerator, filter);
```

### Multiple Languages
```rust
// Generate the same instructions in different languages
let rust_code = arm.generate(RustFunctionGenerator, vec!["AddAddsubImm"]);
let cpp_code = arm.generate(CppFunctionGenerator, vec!["AddAddsubImm"]);
```

## Generated Code

### Rust Output
```rust
pub fn add_addsub_imm(sf: i32, sh: i32, imm12: i32, rn: Register, rd: Register) -> u32 {
    let mut result = 0b1001000100000000000000000000000;
    result |= (sf as u32) << 31;
    result |= (sh as u32) << 22;
    result |= (imm12 as u32) << 10;
    result |= (rn.encode() as u32) << 5;
    result |= (rd.encode() as u32) << 0;
    result
}
```

### C++ Output
```cpp
constexpr uint32_t add_addsub_imm(int32_t sf, int32_t sh, int32_t imm12, Register rn, Register rd) noexcept {
    uint32_t result = 0b1001000100000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(sh) << 22U;
    result |= static_cast<uint32_t>(imm12) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}
```

## API Reference

### Core Types
- `ArmCodeGen` - Main interface for generating code
- `CodeGenerator` - Trait for implementing new language generators
- `InstructionFilter` - For filtering which instructions to generate

### Built-in Generators
- `RustFunctionGenerator` - Generates Rust functions
- `CppFunctionGenerator` - Generates C++ constexpr functions  
- `PythonCodeGenerator` - Generates Python classes

### Main Methods
- `ArmCodeGen::new()` - Initialize with all ARM instructions
- `generate(generator, instruction_names)` - Generate specific instructions
- `generate_filtered(generator, filter)` - Generate with complex filtering
- `generate_all(generator)` - Generate all 799 instructions
- `find_instructions(pattern)` - Search for instructions by name/description
- `instruction_info(name)` - Get details about a specific instruction

## Extending

Create custom generators by implementing the `CodeGenerator` trait:

```rust
struct JavaScriptGenerator;

impl CodeGenerator for JavaScriptGenerator {
    fn generate_prefix(&self) -> String { /* ... */ }
    fn generate_registers(&self) -> String { /* ... */ }
    fn generate_instruction_enum(&self, instructions: &[Instruction]) -> String { /* ... */ }
    // ... other required methods
}
```

## Use Cases

- **JIT Compilers**: Generate only the instructions you need
- **Assemblers**: Full ARM instruction set support
- **Code Generators**: Type-safe instruction encoding
- **Embedded Systems**: Minimal code generation
- **Learning**: Explore ARM instruction encoding

## Requirements

- Rust 2021 edition or later
- No external dependencies at runtime (XML parsing happens at compile time)