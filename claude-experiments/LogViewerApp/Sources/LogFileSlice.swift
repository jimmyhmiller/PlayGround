// Inspired by HexFiend's architecture (https://github.com/HexFiend/HexFiend)
// Copyright (c) 2005-2016, Peter Ammon (BSD 2-Clause License)
// Swift adaptation for LogViewerApp

import Foundation

/// A reference to a file that can be read on-demand without loading into memory
class LogFileReference {
    private let fileHandle: FileHandle
    private let fileURL: URL
    private(set) var length: UInt64
    
    init?(url: URL) {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        self.fileURL = url
        self.fileHandle = handle
        
        // Get file size
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            self.length = attributes[.size] as? UInt64 ?? 0
        } catch {
            return nil
        }
    }
    
    func readData(offset: UInt64, length: Int) -> Data? {
        guard offset < self.length else { return nil }
        
        do {
            try fileHandle.seek(toOffset: offset)
            let data = fileHandle.readData(ofLength: length)
            return data
        } catch {
            return nil
        }
    }
    
    deinit {
        try? fileHandle.close()
    }
}

/// Protocol for different types of log data sources
protocol LogSlice {
    var length: UInt64 { get }
    func data(in range: Range<UInt64>) -> Data?
    func substring(in range: Range<UInt64>) -> String?
}

/// A slice that references data in a file without loading it
class LogFileSlice: LogSlice {
    private let fileRef: LogFileReference
    private let offset: UInt64
    let length: UInt64
    
    init(fileRef: LogFileReference, offset: UInt64 = 0, length: UInt64? = nil) {
        self.fileRef = fileRef
        self.offset = offset
        self.length = length ?? (fileRef.length - offset)
    }
    
    func data(in range: Range<UInt64>) -> Data? {
        guard range.lowerBound < length else { return nil }
        let adjustedEnd = min(range.upperBound, length)
        let adjustedRange = range.lowerBound..<adjustedEnd
        let readLength = Int(adjustedRange.upperBound - adjustedRange.lowerBound)
        
        return fileRef.readData(offset: offset + adjustedRange.lowerBound, length: readLength)
    }
    
    func substring(in range: Range<UInt64>) -> String? {
        guard let data = data(in: range) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

/// A slice that holds data in memory (for edits, filters, etc.)
class LogMemorySlice: LogSlice {
    private let data: Data
    let length: UInt64
    
    init(data: Data) {
        self.data = data
        self.length = UInt64(data.count)
    }
    
    init(string: String) {
        self.data = string.data(using: .utf8) ?? Data()
        self.length = UInt64(self.data.count)
    }
    
    func data(in range: Range<UInt64>) -> Data? {
        guard range.lowerBound < length else { return nil }
        let adjustedEnd = min(range.upperBound, length)
        let nsRange = NSRange(location: Int(range.lowerBound), 
                              length: Int(adjustedEnd - range.lowerBound))
        return data.subdata(in: Range(nsRange)!)
    }
    
    func substring(in range: Range<UInt64>) -> String? {
        guard let data = data(in: range) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}