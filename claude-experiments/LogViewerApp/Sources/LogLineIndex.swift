// Line indexing system for efficient line-based navigation in large log files
// Inspired by HexFiend's B+ tree approach

import Foundation

/// Represents a line in the log file with its byte range
struct LogLineInfo {
    let lineNumber: Int
    let byteOffset: UInt64
    let byteLength: UInt64
    
    var byteRange: Range<UInt64> {
        byteOffset..<(byteOffset + byteLength)
    }
}

/// Manages an index of line positions in a log file for fast lookup
class LogLineIndex {
    private var lineOffsets: [UInt64] = []
    private let slice: LogSlice
    private var isFullyIndexed = false
    private let chunkSize: Int = 1024 * 1024 // 1MB chunks for indexing
    
    init(slice: LogSlice) {
        self.slice = slice
        // Start with just the beginning
        lineOffsets.append(0)
    }
    
    /// Build index up to the specified byte offset
    func indexUpTo(byteOffset: UInt64) {
        guard !isFullyIndexed else { return }
        
        let lastIndexedOffset = lineOffsets.last ?? 0
        if lastIndexedOffset >= byteOffset {
            return // Already indexed
        }
        
        var currentOffset = lastIndexedOffset
        
        while currentOffset < byteOffset && currentOffset < slice.length {
            let chunkEnd = min(currentOffset + UInt64(chunkSize), slice.length)
            guard let data = slice.data(in: currentOffset..<chunkEnd) else { break }
            
            // Find newlines in the chunk
            var searchOffset = 0
            while searchOffset < data.count {
                if let newlineRange = data.range(of: Data([0x0A]), // \n
                                                options: [],
                                                in: searchOffset..<data.count) {
                    let newlineOffset = currentOffset + UInt64(newlineRange.lowerBound) + 1
                    if newlineOffset > (lineOffsets.last ?? 0) {
                        lineOffsets.append(newlineOffset)
                    }
                    searchOffset = newlineRange.upperBound
                } else {
                    break
                }
            }
            
            currentOffset = chunkEnd
        }
        
        if currentOffset >= slice.length {
            isFullyIndexed = true
        }
    }
    
    /// Get the total number of lines (may trigger full indexing)
    func totalLines() -> Int {
        if !isFullyIndexed {
            indexUpTo(byteOffset: slice.length)
        }
        return lineOffsets.count
    }
    
    /// Get line info for a specific line number
    func lineInfo(at lineNumber: Int) -> LogLineInfo? {
        // Ensure we have indexed enough
        if lineNumber >= lineOffsets.count {
            indexUpTo(byteOffset: slice.length)
        }
        
        guard lineNumber < lineOffsets.count else { return nil }
        
        let startOffset = lineOffsets[lineNumber]
        let endOffset: UInt64
        
        if lineNumber + 1 < lineOffsets.count {
            endOffset = lineOffsets[lineNumber + 1]
        } else {
            // Need to find the end of this line
            if let data = slice.data(in: startOffset..<min(startOffset + 10000, slice.length)) {
                if let newlineRange = data.range(of: Data([0x0A])) {
                    endOffset = startOffset + UInt64(newlineRange.lowerBound) + 1
                } else {
                    endOffset = min(startOffset + UInt64(data.count), slice.length)
                }
            } else {
                endOffset = slice.length
            }
        }
        
        return LogLineInfo(lineNumber: lineNumber,
                          byteOffset: startOffset,
                          byteLength: endOffset - startOffset)
    }
    
    /// Find the line containing the given byte offset
    func lineContaining(byteOffset: UInt64) -> Int {
        indexUpTo(byteOffset: byteOffset + UInt64(chunkSize))
        
        // Binary search for the line
        var left = 0
        var right = lineOffsets.count - 1
        
        while left < right {
            let mid = (left + right + 1) / 2
            if lineOffsets[mid] <= byteOffset {
                left = mid
            } else {
                right = mid - 1
            }
        }
        
        return left
    }
    
    /// Get multiple lines efficiently
    func lines(in range: Range<Int>) -> [LogLineInfo] {
        var result: [LogLineInfo] = []
        result.reserveCapacity(range.count)
        
        for lineNum in range {
            if let info = lineInfo(at: lineNum) {
                result.append(info)
            }
        }
        
        return result
    }
}