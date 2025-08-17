# HexFiend Architecture Analysis: How It Handles Gigabyte Files Instantly

After investigating HexFiend's source code, here's how they achieve instant loading of gigabyte-sized files:

## 1. Core Architecture: ByteSlice Abstraction
- Files are NOT loaded into memory entirely
- Instead, HexFiend uses `HFByteSlice` as an abstraction over data sources
- Two main slice types:
  - `HFFileByteSlice`: References a range in a file WITHOUT loading it
  - `HFSharedMemoryByteSlice`: For in-memory data (edits, pastes, etc.)

## 2. B+ Tree Data Structure (HFBTree)
- Uses a 10-way B+ tree to manage byte slices
- Each node can have up to 10 children
- Provides O(log₁₀ n) operations for insertion, deletion, and search
- The tree stores slices, not raw bytes

## 3. File Memory Mapping
- `HFFileReference` uses `F_NOCACHE` flag (non-caching IO)
- Files are accessed via file descriptors with on-demand reading
- `HFFileByteSlice` only stores:
  ```objc
  HFFileReference *fileReference;  // Shared file descriptor wrapper
  unsigned long long offset;       // Offset into file
  unsigned long long length;       // Length of this slice
  ```
- Actual bytes are read only when needed via:
  ```objc
  [fileReference readBytes:dst length:length from:offset]
  ```

## 4. Lazy Loading Strategy
- Opening a file creates a single `HFFileByteSlice` covering the entire file
- No data is actually read until displayed
- When displaying, only visible portions are loaded:
  ```objc
  - (void)copyBytes:(unsigned char *)dst range:(HFRange)range {
      [fileReference readBytes:dst length:range.length from:offset + range.location];
  }
  ```

## 5. Edit Optimization
- Edits don't modify the original file immediately
- New edits create `HFSharedMemoryByteSlice` objects
- The B+ tree maintains a mix of file slices and memory slices
- Example after editing:
  - Original: `[FileSlice: 0-1GB]`
  - After inserting "ABC" at position 500:
    - `[FileSlice: 0-500]`
    - `[MemorySlice: "ABC"]`
    - `[FileSlice: 500-1GB]`

## 6. Virtualized Display
- Text views only render visible lines
- Controller maintains `displayedLineRange`
- Views request data only for visible range
- Scrolling updates the range and fetches new data

## 7. Smart Coalescing
- Adjacent edits can be merged:
  ```objc
  - (HFByteSlice *)byteSliceByAppendingSlice:(HFByteSlice *)slice
  ```
- Typing consecutive characters extends existing memory slices
- Reduces tree fragmentation

## 8. Key Performance Tricks
- **Zero-copy architecture**: File data is never copied unless edited
- **Shared file references**: Multiple slices can reference same file
- **Range-based operations**: Everything works with ranges, not individual bytes
- **Cached lengths**: Tree nodes cache cumulative lengths for fast offset calculations

## 9. Memory Usage
For a 1GB file:
- Initial: ~100 bytes (one HFFileByteSlice object)
- After 1000 edits: ~100KB (tree nodes + edit slices)
- Visible data: ~100KB (only what's on screen)

## Why Our Log Viewer Is Slow

Our current implementation:
1. Loads entire file into memory (`String(contentsOf:)`)
2. Parses all lines immediately
3. Creates Swift objects for every log entry
4. Stores everything in arrays

For 119K log entries, we're creating 119K LogEntry objects plus parsing overhead!

## How to Apply HexFiend's Approach to Log Viewer

1. **Don't parse entire file** - only parse visible portions
2. **Use memory mapping** or streaming reads
3. **Virtual scrolling** - calculate total height but only render visible
4. **Index on demand** - build line offset index progressively
5. **Lazy parsing** - parse log entries only when displayed
6. **Chunk-based loading** - load file in chunks, parse chunks as needed

## Key Takeaways for Performance

### The Magic Formula
1. **Never load what you don't need to display**
2. **Use indirection** - reference data, don't copy it
3. **Defer work** - parse/process only when needed
4. **Cache strategically** - keep computed values, discard raw data
5. **Think in ranges** - work with offsets and lengths, not content

### Architecture Pattern
```
File on Disk → Slice References → B+ Tree → Controller → View (only visible)
     ↑                                           ↓
     └──────── Read on demand ←──────────────────┘
```

This architecture allows HexFiend to open files of any size instantly because the initial cost is O(1) - just creating a single slice reference. All expensive operations (reading, parsing) are deferred until absolutely necessary.