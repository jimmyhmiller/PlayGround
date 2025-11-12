import Foundation
import AppKit

@MainActor
class PDFManager: ObservableObject {
    @Published var pdfFiles: [URL] = []
    @Published var currentIndex: Int = 0
    @Published var workingDirectory: URL?

    var currentPDF: URL? {
        guard !pdfFiles.isEmpty, currentIndex < pdfFiles.count else { return nil }
        return pdfFiles[currentIndex]
    }

    var progress: String {
        guard !pdfFiles.isEmpty else { return "No PDFs" }
        return "\(currentIndex + 1) of \(pdfFiles.count)"
    }

    func selectDirectory() async {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select directory containing PDFs to categorize"

        let response = await panel.begin()

        if response == .OK, let url = panel.url {
            workingDirectory = url
            scanForPDFs(in: url)
        }
    }

    func scanForPDFs(in directory: URL) {
        let fileManager = FileManager.default

        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return }

        var foundPDFs: [URL] = []

        for case let fileURL as URL in enumerator {
            guard fileURL.pathExtension.lowercased() == "pdf" else { continue }

            // Skip if already in a subdirectory (already categorized)
            let relativePath = fileURL.path.replacingOccurrences(of: directory.path, with: "")
            let pathComponents = relativePath.split(separator: "/")
            if pathComponents.count == 1 {
                // Only include PDFs directly in the working directory
                foundPDFs.append(fileURL)
            }
        }

        pdfFiles = foundPDFs.sorted { $0.lastPathComponent < $1.lastPathComponent }
        currentIndex = 0
    }

    func movePDFToCategory(_ category: String) throws {
        guard let workingDir = workingDirectory,
              let currentFile = currentPDF else { return }

        let categoryURL = workingDir.appendingPathComponent(category)
        let fileManager = FileManager.default

        // Create category directory if it doesn't exist
        if !fileManager.fileExists(atPath: categoryURL.path) {
            try fileManager.createDirectory(at: categoryURL, withIntermediateDirectories: true)
        }

        // Move file to category
        let destination = categoryURL.appendingPathComponent(currentFile.lastPathComponent)
        try fileManager.moveItem(at: currentFile, to: destination)

        // Remove from list
        pdfFiles.remove(at: currentIndex)

        // Adjust index if needed
        if currentIndex >= pdfFiles.count && currentIndex > 0 {
            currentIndex = pdfFiles.count - 1
        }
    }

    func skipCurrent() {
        if currentIndex < pdfFiles.count - 1 {
            currentIndex += 1
        }
    }

    func goToPrevious() {
        if currentIndex > 0 {
            currentIndex -= 1
        }
    }
}
