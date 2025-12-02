//
//  DrawingManager.swift
//  PDFMarkup
//
//  Manages persistence of PencilKit drawings for PDFs
//

import Foundation
import PencilKit
import Combine

/// Represents all drawings for a single PDF document
struct PDFDrawings: Codable {
    let pdfHash: String
    var lastModified: Date
    var pages: [Int: Data]  // Page index -> PKDrawing.dataRepresentation()

    init(pdfHash: String) {
        self.pdfHash = pdfHash
        self.lastModified = Date()
        self.pages = [:]
    }

    /// Saves a drawing for a specific page
    mutating func setDrawing(_ drawing: PKDrawing, forPage page: Int) {
        pages[page] = drawing.dataRepresentation()
        lastModified = Date()
    }

    /// Retrieves a drawing for a specific page
    func getDrawing(forPage page: Int) -> PKDrawing? {
        guard let data = pages[page] else { return nil }
        return try? PKDrawing(data: data)
    }

    /// Removes a drawing for a specific page
    mutating func removeDrawing(forPage page: Int) {
        pages.removeValue(forKey: page)
        lastModified = Date()
    }

    /// Checks if any pages have drawings
    var hasDrawings: Bool {
        return !pages.isEmpty
    }
}

/// Manages loading, saving, and caching of PDF drawings
class DrawingManager: ObservableObject {
    static let shared = DrawingManager()

    private let cacheDir: URL
    private var cache: [String: PKDrawing] = [:]  // Simple dictionary cache
    private var saveWorkItems: [String: DispatchWorkItem] = [:]
    private let fileManager = FileManager.default
    private let maxCacheSize = 20  // Keep max 20 drawings in memory

    private init() {
        // Setup cache directory
        let cachesDir = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        self.cacheDir = cachesDir.appendingPathComponent("DrawingsCache")

        // Create directory if it doesn't exist
        try? fileManager.createDirectory(at: cacheDir, withIntermediateDirectories: true)
    }

    // MARK: - Public API

    /// Saves a drawing for a specific PDF and page (with debouncing)
    func scheduleSave(_ drawing: PKDrawing, pdfHash: String, page: Int) {
        let key = "\(pdfHash)-\(page)"

        // Cancel any pending save for this key
        saveWorkItems[key]?.cancel()

        // Schedule new save after 1 second debounce
        let workItem = DispatchWorkItem { [weak self] in
            self?.saveDrawing(drawing, pdfHash: pdfHash, page: page)
        }
        saveWorkItems[key] = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0, execute: workItem)
    }

    /// Immediately saves a drawing for a specific PDF and page
    func saveDrawing(_ drawing: PKDrawing, pdfHash: String, page: Int) {
        // Update cache
        let cacheKey = "\(pdfHash)-\(page)"
        cache[cacheKey] = drawing

        // Limit cache size
        if cache.count > maxCacheSize {
            // Remove random entries if cache is too large
            let keysToRemove = Array(cache.keys.prefix(cache.count - maxCacheSize))
            keysToRemove.forEach { cache.removeValue(forKey: $0) }
        }

        // Load or create PDFDrawings
        var pdfDrawings = loadDrawings(pdfHash: pdfHash) ?? PDFDrawings(pdfHash: pdfHash)
        pdfDrawings.setDrawing(drawing, forPage: page)

        // Save to disk
        saveToDisk(pdfDrawings)

        // Mark for S3 sync
        DrawingSyncManager.shared.markForSync(pdfHash: pdfHash)
    }

    /// Loads a specific drawing for a PDF page
    func loadDrawing(pdfHash: String, page: Int) -> PKDrawing? {
        let cacheKey = "\(pdfHash)-\(page)"

        // Check cache first
        if let cached = cache[cacheKey] {
            return cached
        }

        // Load from disk
        guard let pdfDrawings = loadDrawings(pdfHash: pdfHash),
              let drawing = pdfDrawings.getDrawing(forPage: page) else {
            return nil
        }

        // Update cache
        cache[cacheKey] = drawing

        // Limit cache size
        if cache.count > maxCacheSize {
            let keysToRemove = Array(cache.keys.prefix(cache.count - maxCacheSize))
            keysToRemove.forEach { cache.removeValue(forKey: $0) }
        }

        return drawing
    }

    /// Loads all drawings for a PDF
    func loadDrawings(pdfHash: String) -> PDFDrawings? {
        let fileURL = drawingsURL(for: pdfHash)

        guard fileManager.fileExists(atPath: fileURL.path) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            return try decoder.decode(PDFDrawings.self, from: data)
        } catch {
            print("Error loading drawings for \(pdfHash): \(error)")
            return nil
        }
    }

    /// Removes all drawings for a specific PDF
    func clearDrawings(pdfHash: String) {
        let fileURL = drawingsURL(for: pdfHash)
        try? fileManager.removeItem(at: fileURL)

        // Clear from cache
        if let pdfDrawings = loadDrawings(pdfHash: pdfHash) {
            for page in pdfDrawings.pages.keys {
                let cacheKey = "\(pdfHash)-\(page)"
                cache.removeValue(forKey: cacheKey)
            }
        }
    }

    /// Removes a drawing for a specific page
    func clearDrawing(pdfHash: String, page: Int) {
        var pdfDrawings = loadDrawings(pdfHash: pdfHash) ?? PDFDrawings(pdfHash: pdfHash)
        pdfDrawings.removeDrawing(forPage: page)

        if pdfDrawings.hasDrawings {
            saveToDisk(pdfDrawings)
        } else {
            // If no drawings left, delete the file
            let fileURL = drawingsURL(for: pdfHash)
            try? fileManager.removeItem(at: fileURL)
        }

        // Clear from cache
        let cacheKey = "\(pdfHash)-\(page)"
        cache.removeValue(forKey: cacheKey)
    }

    /// Returns all drawing files in cache
    func getAllDrawingFiles() -> [String] {
        guard let files = try? fileManager.contentsOfDirectory(atPath: cacheDir.path) else {
            return []
        }
        return files.filter { $0.hasSuffix(".drawings.json") }
    }

    // MARK: - Private Helpers

    private func drawingsURL(for pdfHash: String) -> URL {
        return cacheDir.appendingPathComponent("\(pdfHash).drawings.json")
    }

    private func saveToDisk(_ pdfDrawings: PDFDrawings) {
        let fileURL = drawingsURL(for: pdfDrawings.pdfHash)

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(pdfDrawings)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            print("Error saving drawings for \(pdfDrawings.pdfHash): \(error)")
        }
    }
}
