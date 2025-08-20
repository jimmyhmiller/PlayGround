# LogViewerApp TODO

## High Priority Issues

### 🎨 Timeline Visualization (Critical)
- [ ] **Make timeline clearer and more visible**
  - Current timeline is hard to see and understand
  - Need better color contrast and visual design
  - Consider adding labels, grid lines, or other visual aids
  - Improve the event density visualization (current bars are too subtle)

### 🎯 Timeline Navigation (Critical)
- [ ] **Implement proper jump-to-line functionality**
  - Timeline clicks should scroll the log view to the corresponding line
  - Currently clicking timeline doesn't navigate properly to the exact line
  - Need to calculate precise line mapping from timeline position to log content
  - Ensure smooth scrolling animation when jumping

### 📊 Log Generation Improvements (High)
- [ ] **Create realistic event distributions in generator**
  - Current generator needs more varied, real-world patterns
  - Add different event density distributions throughout the day
  - Simulate realistic logging scenarios:
    - Morning startup bursts
    - Lunch time quiet periods  
    - End-of-day batch processing
    - Overnight maintenance windows
    - Error clusters during incidents
    - Gradual load increases/decreases

## Medium Priority Enhancements

### 🚀 Performance & UX
- [ ] **Add loading indicators**
  - Show progress when opening very large files
  - Display file size and line count info
  - Add memory usage statistics

- [ ] **Improve virtual scrolling edge cases**
  - Handle resizing better
  - Optimize for very wide log lines
  - Test with different screen sizes

### 🔍 Search & Filtering
- [ ] **Add search functionality**
  - Text search within log entries
  - Filter by log level (ERROR, WARN, INFO, DEBUG)
  - Date/time range filtering
  - Regular expression support

### 📝 Log Format Support
- [ ] **Expand log format support**
  - Better JSON log parsing
  - Syslog format support
  - Custom timestamp formats
  - Multi-line log entries (stack traces)

## Low Priority Features

### 🎛️ User Interface
- [ ] **Add preferences/settings**
  - Font size adjustment
  - Color theme customization
  - Timeline width adjustment
  - Default file associations

- [ ] **Keyboard shortcuts**
  - Quick navigation (Cmd+G for go-to-line)
  - Search shortcuts (Cmd+F)
  - Timeline navigation (arrow keys)

### 💾 File Handling
- [ ] **Recent files menu**
  - Remember recently opened log files
  - Quick access to common log locations

- [ ] **File watching**
  - Auto-refresh when log file changes
  - Tail mode for live log viewing

## Technical Debt

### 🏗️ Architecture
- [ ] **Refactor timeline sampling**
  - Current timeline sampling in VirtualTimelineSidebar could be more efficient
  - Consider caching timeline data for better performance

- [ ] **Error handling**
  - Better error messages for corrupted files
  - Graceful handling of binary files
  - Memory pressure handling

### 🧪 Testing
- [ ] **Add test files**
  - Create test suite with various log formats
  - Performance benchmarks with different file sizes
  - Edge case testing (empty files, malformed logs)

## Future Ideas

### 🔮 Advanced Features
- [ ] **Log analysis tools**
  - Error rate graphs
  - Event frequency analysis
  - Export filtered results

- [ ] **Collaboration features**
  - Share log snippets
  - Bookmark interesting lines
  - Add annotations

---

## Current Status
✅ **Virtual scrolling working** - Handles 94K+ entries efficiently  
✅ **HexFiend architecture ported** - Zero-copy file access implemented  
✅ **Basic timeline implemented** - Shows event density over time  

**Next up**: Timeline visualization improvements for better user experience