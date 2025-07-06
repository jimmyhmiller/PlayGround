# CCSeva-Swift - Claude Usage Monitor for macOS

## Project Overview

CCSeva-Swift is a native macOS menu bar application that monitors Claude Code usage in real-time. It's a Swift port of the original CCSeva (Electron/React) application that provides users with detailed usage statistics, cost tracking, and notifications about their Claude usage patterns.

## Architecture

### Core Components

1. **CCSeva (Main App)** - SwiftUI-based menu bar application
2. **ClaudeUsageCore** - Shared library for usage parsing and calculations
3. **ClaudeUsageCLI** - Command-line interface for testing and debugging
4. **TestMenuBar** - Development target for menu bar testing

### Key Classes

- `CCUsageService.swift:17` - Main service class with `@ObservableObject` for SwiftUI binding
- `ClaudeUsageReader.swift:129` - Core file parsing and data processing logic
- `ContentView.swift` - SwiftUI interface for displaying usage statistics

## Current Performance Issues

### CPU-Intensive Operations

1. **Timer-Based Polling (Every 30 seconds)**
   - `CCUsageService.swift:34` - Uses `Timer.scheduledTimer` for continuous polling
   - `CCUsageService.swift:25` - 30-second refresh interval causes frequent CPU spikes

2. **Complete File Re-parsing**
   - `ClaudeUsageReader.swift:140` - `readUsageData()` rescans entire filesystem every refresh
   - `ClaudeUsageReader.swift:153-165` - Recursive directory traversal through all projects
   - `ClaudeUsageReader.swift:171-217` - Complete JSONL file parsing with no incremental updates

3. **Expensive Processing Operations**
   - `ClaudeUsageReader.swift:232-301` - Complex statistics calculations on every refresh
   - `ClaudeUsageReader.swift:394-508` - Heavy date calculations and timezone conversions
   - `ClaudeUsageReader.swift:177-214` - JSON decoding of entire file contents repeatedly

## Optimization Plan

### 1. File System Watcher Implementation
Replace timer-based polling with native file system monitoring:
- Use `DispatchSource.makeFileSystemObjectSource()` for directory watching
- Monitor `~/.claude/projects/` directory for file changes
- Only trigger processing when actual file modifications occur

### 2. Incremental Parsing with Caching
Implement smart caching to avoid re-processing unchanged files:
- Track file modification dates and sizes
- Cache parsed results by file hash/timestamp
- Only re-parse files that have actually changed
- Store intermediate results to avoid full recalculation

### 3. Background Processing Optimization
Move all I/O operations off the main thread:
- `CCUsageService.swift:76-87` already uses background queue, but can be optimized
- Implement proper async/await patterns throughout
- Use `Task.detached` for long-running operations

### 4. Data Structure Optimization
Optimize in-memory data structures:
- Pre-calculate frequently accessed aggregations
- Use efficient data structures for deduplication (Set vs repeated iterations)
- Cache expensive calculations like timezone conversions

## Data Flow

```
Timer (30s) → CCUsageService.fetchUsageData() → ClaudeUsageReader.generateUsageStats()
    ↓
FileSystem Scan → JSONL Parsing → Deduplication → Aggregation → UI Update
```

## File Processing Details

### Current Flow
1. `ClaudeUsageReader.swift:153` - Scan all project directories
2. `ClaudeUsageReader.swift:158-160` - Find all `.jsonl` files
3. `ClaudeUsageReader.swift:172-173` - Read entire file content as string
4. `ClaudeUsageReader.swift:182-214` - Parse every line as JSON
5. `ClaudeUsageReader.swift:194-205` - Deduplication using message+request ID hash
6. `ClaudeUsageReader.swift:232-301` - Statistical aggregation and calculations

### Optimization Targets
- Eliminate unnecessary file system scans
- Cache parsed data with invalidation on file changes
- Stream-parse large JSONL files instead of loading entire content
- Pre-compute expensive aggregations and store incrementally

## Dependencies

- **Foundation** - Core Swift framework for file I/O and data structures
- **SwiftUI** - UI framework for the menu bar interface
- **AppKit** - macOS-specific functionality for menu bar integration

## Testing Strategy

- Monitor CPU usage before and after optimizations using Activity Monitor
- Test with large JSONL files to ensure performance improvements scale
- Verify accuracy of incremental parsing vs. full re-parsing
- Test file watcher reliability across different file system events

## Implementation Priority

1. **High**: File system watcher to replace timer polling
2. **High**: Incremental parsing with modification date checking
3. **Medium**: Background processing optimizations
4. **Low**: Data structure and algorithm optimizations

The current implementation prioritizes correctness and simplicity over performance, which works for small datasets but causes CPU spikes with larger usage histories or frequent updates.