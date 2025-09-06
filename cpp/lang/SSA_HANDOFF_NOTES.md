# SSA System Handoff Notes

## Current State

The SSA (Static Single Assignment) translation system is partially working with some critical bugs that need fixing.

### What Works ✅
- **Simple functions**: `fn id(x) -> t { x }` generates correct `return x`
- **Function parameters**: Use actual names (`x`, `a`) instead of SSA temps (`v0`, `v1`)
- **Local variables**: Properly use SSA numbering (`v0`, `v1`, `v2`) for single assignment
- **Return instructions**: Functions properly return values with `return` instruction
- **Basic loops with breaks**: Simple loops with immediate breaks work correctly
- **No fake loop conditions**: Removed nonsensical `v1 := 1; if v1 then...` pattern

### Critical Bugs ❌
- **Segfault on complex functions**: `cat example.txt | ./build.sh ast-to-ssa` crashes
- **Infinite recursion/memory corruption**: Likely in phi function resolution
- **Multi-function processing**: Works for simple cases, fails on complex example.txt

## Architecture Overview

### Key Files
- `src/ssa_translator.h/.cc` - Main SSA translation logic
- `src/ssa_instruction.h` - SSA instruction definitions  
- `src/ssa_visualizer.cc` - Graphviz visualization
- `example.txt` - Test file that currently crashes

### Core Algorithm
Uses **Braun et al. SSA construction algorithm**:
1. **Variable definitions**: Track where each variable is defined
2. **Phi function placement**: Create phi functions at merge points  
3. **Block sealing**: Finalize blocks when all predecessors are known
4. **Incomplete phi resolution**: Fill in phi operands when blocks are sealed

### Class Structure
```cpp
class SSATranslator {
  // Variable tracking
  std::unordered_map<std::string, std::unordered_map<BlockId, SSAValue>> definitions;
  std::unordered_map<PhiId, SSAPhi> phis;
  std::unordered_map<BlockId, std::unordered_map<std::string, PhiId>> incomplete_phis;
  
  // Multi-function support  
  std::vector<FunctionSSA> functions;
  
  // Loop handling
  BlockId current_loop_exit;  // For break statements
};
```

## Recent Changes Made

### 1. Function Parameter Handling (WORKING)
**Problem**: Parameters were using temp variables (`v0`, `v1`) instead of names
**Solution**: 
```cpp
// In translate_function_declaration()
Variable param_var(param->name);  // Use actual name like "x"
write_variable(param->name, current_block, SSAValue::var(param_var));
```
**Result**: `return x` instead of `return v0` ✅

### 2. Loop Translation (PARTIALLY WORKING)  
**Problem**: Fake conditions `v1 := 1; if v1 then...` in every loop
**Solution**: Proper infinite loop structure with break handling
```cpp
// Before: Always true condition
Variable condition_var = get_temp_variable("loop_cond");
blocks[current_block.id].add_instruction(SSAInstruction::assign(condition_var, SSAValue::literal(1)));

// After: Direct jump to loop body
blocks[current_block.id].add_instruction(SSAInstruction::jump(loop_body));
```
**Result**: Clean loop structure, but causes segfault on complex cases ❌

### 3. Break Statement Handling (WORKING)
**Problem**: Break statements generated placeholder `v4 := 0` 
**Solution**: Actual jumps to loop exit
```cpp
// Before: Placeholder
Variable temp_var = get_temp_variable("break");
blocks[current_block.id].add_instruction(SSAInstruction::assign(temp_var, SSAValue::literal(0)));

// After: Real jump
blocks[current_block.id].add_instruction(SSAInstruction::jump(current_loop_exit));
blocks[current_loop_exit.id].add_predecessor(current_block);
current_block = BlockId(-1);  // Mark block as ended
```
**Result**: Proper break behavior ✅

### 4. Return Instruction Support (WORKING)
**Problem**: Empty function bodies, no return instructions
**Solution**: Added `SSAInstructionType::Return` and translation
```cpp
// In translate_function_declaration()
if (body_result.type != SSAValueType::Undefined) {
  blocks[current_block.id].add_instruction(SSAInstruction::ret(body_result));
}
```
**Result**: Functions show proper returns ✅

## Known Issues to Fix

### 1. CRITICAL: Segfault on Complex Functions
**Location**: Likely in `read_variable_recursively()` or phi resolution
**Symptom**: 
```bash
cat example.txt | ./build.sh ast-to-ssa
# Segmentation fault: 11
```
**Hypothesis**: Infinite recursion in phi function operand resolution when handling cycles in complex loops

**Debug approach**:
1. Test with individual functions from example.txt to isolate which one crashes
2. Add debugging prints in `read_variable_recursively()` to track recursion depth  
3. Check phi function creation/resolution in loops with multiple variables

### 2. Phi Function Parameter Resolution  
**Issue**: Function parameters create `φ(?)` instead of using actual values
**Current code**: Has TODO comment in `read_variable()`
```cpp
// TODO: Add function parameter lookup without causing infinite recursion
```
**Goal**: Make `φ(x)` for parameter `x` instead of `φ(?)`

**Safe approach needed**: The infinite recursion guard that was tried:
```cpp
if (block_id.id != 0) {
  auto initial_it = var_it->second.find(BlockId(0));
  if (initial_it != var_it->second.end()) {
    return initial_it->second;
  }
}
```
This caused segfaults, so needs a more sophisticated solution.

### 3. Multi-Function Variable Scoping
**Issue**: Variables might be leaking between functions
**Check**: Ensure function isolation in `translate_function_declaration()`
```cpp
// Save/restore state for each function
auto saved_definitions = definitions;
// ... translate function ...  
definitions = saved_definitions;
```

## Testing Strategy

### Working Test Cases
```bash
# Simple function (works)
echo "fn id : (x: t) -> t { x }" | ./build.sh ast-to-ssa

# Simple loop with break (works)  
echo 'fn test : () -> i32 { loop { break; } }' | ./build.sh ast-to-ssa
```

### Failing Test Cases
```bash
# Full example (crashes)
cat example.txt | ./build.sh ast-to-ssa

# Individual functions from example.txt (test each separately)
echo 'fn len : (xs: List(t)) -> u32 { ... }' | ./build.sh ast-to-ssa
```

## Debugging Tools

### 1. Visualization
```bash
# Generate SSA graph
./build.sh ast-to-ssa < input.txt
# Creates ssa_graph.png with visual representation
```

### 2. AST Debug Tool
```bash
# Debug AST structure  
./debug_ssa < input.txt
```

### 3. Add Debug Prints
In `read_variable_recursively()`, add:
```cpp
static int depth = 0;
depth++;
std::cerr << std::string(depth*2, ' ') << "read_variable_recursively: " << variable << " block " << block_id.id << std::endl;
// ... existing code ...
depth--;
```

## Next Steps Priority

1. **CRITICAL**: Fix segfault in complex function processing
   - Isolate which function/pattern triggers crash
   - Add recursion depth limits and cycle detection
   - Fix infinite loops in phi resolution

2. **HIGH**: Implement safe function parameter phi resolution  
   - Parameters should resolve to actual names, not `φ(?)`
   - Avoid infinite recursion in variable lookup

3. **MEDIUM**: Optimize loop structure
   - Current loop header → loop body jump is unnecessary
   - Could directly jump to loop body, but needs careful phi handling

4. **LOW**: Clean up warnings and unused parameters

## Key Insight
The SSA system has a solid foundation with correct instruction generation and proper control flow, but the Braun et al. algorithm implementation has edge cases in phi function resolution that cause infinite recursion in complex scenarios. The core logic is sound - focus on debugging the recursion cycles.