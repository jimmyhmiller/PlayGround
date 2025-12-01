import Foundation

struct PDFMetadata: Codable, Identifiable, Hashable {
    let hash: String
    let path: String
    let fileName: String
    let title: String?
    let author: String?
    let metadataFound: Bool
    let totalPages: Int
    let creator: String?
    let producer: String?
    let creationDate: String?
    let error: String?
    let processedAt: String
    let ocr_title: String?
    let ocr_author: String?

    // Use path as ID since it's unique, not hash (which can be same in different folders)
    var id: String { path }

    var displayTitle: String {
        ocr_title ?? title ?? fileName.replacingOccurrences(of: "_original.pdf", with: "")
    }

    var displayAuthor: String {
        ocr_author ?? author ?? "Unknown"
    }

    var folder: String {
        // Extract folder from S3 key via S3StateManager
        if let s3Key = S3StateManager.shared.s3Key(for: hash) {
            // Extract folder from "pdfs/folder/hash.pdf"
            let components = s3Key.components(separatedBy: "/")
            if components.count >= 2 {
                return components[1] // Return the folder part
            }
        }

        // Fallback: try to extract from path
        let components = path.components(separatedBy: "/")
        if let readingsIndex = components.lastIndex(where: { $0 == "readings" }),
           readingsIndex + 1 < components.count {
            return components[readingsIndex + 1]
        }
        return "uncategorized"
    }

    func s3URL(bucket: String = "jimmyhmiller-bucket") -> URL? {
        // Use S3StateManager to get the correct URL
        return S3StateManager.shared.s3URL(for: hash, bucket: bucket)
    }
}

struct PDFLibrary {
    let pdfs: [PDFMetadata]

    // Cache the grouped folders to avoid recalculating
    private let _folders: [String: [PDFMetadata]]
    private let _sortedFolders: [String]

    init(pdfs: [PDFMetadata]) {
        self.pdfs = pdfs
        // Pre-calculate folders once
        self._folders = Dictionary(grouping: pdfs, by: { $0.folder })
        self._sortedFolders = _folders.keys.sorted()
    }

    var folders: [String: [PDFMetadata]] {
        _folders
    }

    var sortedFolders: [String] {
        _sortedFolders
    }

    func search(query: String) -> [PDFMetadata] {
        guard !query.isEmpty else { return pdfs }

        let lowercased = query.lowercased()
        return pdfs.filter { pdf in
            pdf.displayTitle.lowercased().contains(lowercased) ||
            pdf.displayAuthor.lowercased().contains(lowercased) ||
            pdf.fileName.lowercased().contains(lowercased)
        }
    }

    func pdfs(in folder: String) -> [PDFMetadata] {
        _folders[folder] ?? []
    }
}
