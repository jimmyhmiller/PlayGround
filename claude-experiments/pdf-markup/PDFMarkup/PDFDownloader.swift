import Foundation
import PDFKit

@MainActor
class PDFDownloader: ObservableObject {
    @Published var isDownloading = false
    @Published var downloadProgress: Double = 0
    @Published var error: String?

    private let cacheDirectory: URL

    init() {
        // Use app's cache directory
        let cachesURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        self.cacheDirectory = cachesURL.appendingPathComponent("PDFCache", isDirectory: true)

        // Create cache directory if it doesn't exist
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
    }

    func getCachedPath(for hash: String) -> URL {
        cacheDirectory.appendingPathComponent("\(hash).pdf")
    }

    func isCached(hash: String) -> Bool {
        FileManager.default.fileExists(atPath: getCachedPath(for: hash).path)
    }

    func downloadPDF(metadata: PDFMetadata) async throws -> PDFDocument {
        // Check cache first
        let cachedPath = getCachedPath(for: metadata.hash)
        if isCached(hash: metadata.hash),
           let document = PDFDocument(url: cachedPath) {
            return document
        }

        // Download from S3
        guard let s3URL = metadata.s3URL() else {
            throw PDFDownloadError.invalidURL
        }

        isDownloading = true
        downloadProgress = 0
        error = nil

        defer {
            isDownloading = false
            downloadProgress = 0
        }

        do {
            // Download the file
            let (tempURL, response) = try await URLSession.shared.download(from: s3URL)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw PDFDownloadError.downloadFailed
            }

            // Move to cache
            if FileManager.default.fileExists(atPath: cachedPath.path) {
                try? FileManager.default.removeItem(at: cachedPath)
            }
            try FileManager.default.moveItem(at: tempURL, to: cachedPath)

            // Load PDF
            guard let document = PDFDocument(url: cachedPath) else {
                throw PDFDownloadError.invalidPDF
            }

            return document
        } catch {
            self.error = error.localizedDescription
            throw error
        }
    }

    func clearCache() {
        try? FileManager.default.removeItem(at: cacheDirectory)
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
    }

    func getCacheSize() -> String {
        guard let enumerator = FileManager.default.enumerator(at: cacheDirectory, includingPropertiesForKeys: [.fileSizeKey]) else {
            return "0 MB"
        }

        var totalSize: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                totalSize += Int64(size)
            }
        }

        let mb = Double(totalSize) / 1_024 / 1_024
        return String(format: "%.1f MB", mb)
    }
}

enum PDFDownloadError: Error, LocalizedError {
    case invalidURL
    case downloadFailed
    case invalidPDF

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid S3 URL"
        case .downloadFailed:
            return "Failed to download PDF from S3"
        case .invalidPDF:
            return "Downloaded file is not a valid PDF"
        }
    }
}
