# LogViewerApp - Development Notes

## Project Overview
A macOS log viewer application with a timeline sidebar that shows event density over time, similar to Sublime Text's minimap but for temporal data.

## Architecture
The app now uses a **virtual/lazy loading architecture** inspired by HexFiend to handle gigabyte-sized log files instantly.

### Current Implementation (Virtual)
- SwiftUI-based macOS app with virtual scrolling
- **Zero-copy file access** - files are never fully loaded into memory
- **Lazy line indexing** - only indexes portions of the file as needed
- **On-demand parsing** - log entries are parsed only when displayed
- Timeline sidebar showing entire log compressed to window height
- Click-to-jump navigation on timeline
- Supports various log formats (JSON, timestamped, plain text)

### Legacy Implementation (Memory-based)
- Old implementation that loads entire file into memory
- Kept for reference in `LogStore`, `TimelineSidebar.swift`, `LogContentView.swift`

## Performance Analysis
See [HEXFIEND_ANALYSIS.md](HEXFIEND_ANALYSIS.md) for detailed analysis of how HexFiend handles gigabyte-sized files instantly and how we applied similar techniques.

## Key Files

### Virtual Architecture (NEW)
- `VirtualLogStore.swift` - Lazy loading log store with on-demand parsing
- `LogFileSlice.swift` - File reference abstraction without loading data
- `LogLineIndex.swift` - Efficient line indexing for fast navigation
- `VirtualLogContentView.swift` - Virtual scrolling list view
- `VirtualTimelineSidebar.swift` - Timeline with virtual store integration

### Core Components
- `LogViewerApp.swift` - Main app entry point with file opening
- `LogEntry.swift` - Data models and log parsing utilities

### Legacy Components (Memory-based)
- `TimelineSidebar.swift` - Original timeline (loads all data)
- `LogContentView.swift` - Original log view (loads all data)
- `ContentView.swift` - Original layout

### Development Tools
- `generate-logs.js` - Node script to generate test log files with various patterns

## Test Data
- `sample.log` - Small sample log file
- `large-sample.log` - Generated file with 100K+ entries showing various patterns:
  - Continuous activity sections
  - Bursts of events
  - Large gaps (quiet periods)
  - Error incidents
  - Heavy load periods

## Running the App
```bash
swift run
```
The app will show a welcome screen. Press Cmd+O or click "Open Log File" to load a log file.

**Performance**: Should now open any size log file instantly!

## Generating Test Data
```bash
node generate-logs.js
```
This creates `large-sample.log` with realistic patterns including gaps, bursts, and continuous activity.

## Architecture Benefits
1. **Instant file opening** - O(1) startup time regardless of file size
2. **Minimal memory usage** - Only visible content + small buffers in memory
3. **Responsive scrolling** - Virtual scrolling handles millions of lines
4. **Progressive indexing** - Line index builds incrementally as needed
5. **Cache efficiency** - Parsed entries are cached but can be evicted