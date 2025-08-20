#!/usr/bin/env swift

import Foundation

// Simple performance test script
func main() {
    // Generate a test file if needed
    let testFileURL = URL(fileURLWithPath: "large-test.log")
    
    if !FileManager.default.fileExists(atPath: testFileURL.path) {
        print("Generating large test file...")
        generateLargeLogFile(url: testFileURL)
    }
    
    print("Testing file loading performance...")
    print("File size: \(getFileSize(url: testFileURL)) MB")
    
    let start = CFAbsoluteTimeGetCurrent()
    
    // This would normally test VirtualLogStore initialization
    // For now, just test file reading
    guard let fileHandle = try? FileHandle(forReadingFrom: testFileURL) else {
        print("Failed to open test file")
        return
    }
    
    // Test reading first 1MB (what our optimization does)
    let sampleSize = 1024 * 1024
    let data = fileHandle.readData(ofLength: sampleSize)
    let newlineCount = data.reduce(0) { count, byte in
        count + (byte == 0x0A ? 1 : 0)
    }
    
    try? fileHandle.close()
    
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    print("Sample reading took: \(String(format: "%.3f", elapsed)) seconds")
    print("Found \(newlineCount) lines in first 1MB sample")
    
    if newlineCount > 0 {
        let avgLineLength = Double(sampleSize) / Double(newlineCount)
        let fileSize = getFileSize(url: testFileURL) * 1024 * 1024
        let estimatedLines = Int(fileSize / avgLineLength)
        print("Estimated total lines: \(estimatedLines)")
    }
}

func generateLargeLogFile(url: URL) {
    let entries = [
        "2024-01-01 12:00:00.000 [INFO] Application started successfully",
        "2024-01-01 12:00:01.123 [DEBUG] Processing user request #12345",
        "2024-01-01 12:00:02.456 [WARNING] High memory usage detected: 85%",
        "2024-01-01 12:00:03.789 [ERROR] Database connection failed: timeout",
        "2024-01-01 12:00:04.012 [INFO] Retrying database connection...",
    ]
    
    guard let output = OutputStream(url: url, append: false) else { return }
    output.open()
    defer { output.close() }
    
    // Generate 500K lines (approximately 50MB)
    for i in 0..<500_000 {
        let entry = entries[i % entries.count]
        let line = "\(entry)\n"
        let data = line.data(using: .utf8)!
        data.withUnsafeBytes { bytes in
            output.write(bytes.bindMemory(to: UInt8.self).baseAddress!, maxLength: data.count)
        }
        
        if i % 50_000 == 0 {
            print("Generated \(i) lines...")
        }
    }
    
    print("Generated 500K lines test file")
}

func getFileSize(url: URL) -> Double {
    do {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let size = attributes[.size] as? UInt64 ?? 0
        return Double(size) / (1024 * 1024) // Convert to MB
    } catch {
        return 0
    }
}

main()