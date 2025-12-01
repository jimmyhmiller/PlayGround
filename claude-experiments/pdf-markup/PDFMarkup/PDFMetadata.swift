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

    var id: String { hash }

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

    var folders: [String: [PDFMetadata]] {
        Dictionary(grouping: pdfs, by: { $0.folder })
    }

    var sortedFolders: [String] {
        folders.keys.sorted()
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
        folders[folder] ?? []
    }
}
