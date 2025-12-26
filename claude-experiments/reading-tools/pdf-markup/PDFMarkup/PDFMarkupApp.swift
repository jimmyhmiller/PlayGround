import SwiftUI
import PDFKit
import CryptoKit

@main
struct PDFMarkupApp: App {
    @StateObject private var sharedPDFManager = SharedPDFManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(sharedPDFManager)
                .onOpenURL { url in
                    sharedPDFManager.handleIncomingPDF(url: url)
                }
        }
    }
}

/// Manages PDFs shared via the share sheet or Files app
class SharedPDFManager: ObservableObject {
    @Published var pendingPDF: SharedPDF?

    struct SharedPDF: Identifiable, Equatable {
        let id = UUID()
        let url: URL
        let document: PDFDocument
        let hash: String
        let fileName: String

        static func == (lhs: SharedPDF, rhs: SharedPDF) -> Bool {
            lhs.id == rhs.id
        }
    }

    func handleIncomingPDF(url: URL) {
        // Start accessing security-scoped resource
        let accessing = url.startAccessingSecurityScopedResource()
        defer {
            if accessing {
                url.stopAccessingSecurityScopedResource()
            }
        }

        // Copy to app's documents directory
        guard let document = PDFDocument(url: url) else {
            print("Failed to load PDF from \(url)")
            return
        }

        // Compute hash of the PDF data
        guard let data = document.dataRepresentation() else {
            print("Failed to get PDF data representation")
            return
        }

        let hash = SHA256.hash(data: data)
        let hashString = hash.compactMap { String(format: "%02x", $0) }.joined().prefix(16)

        let fileName = url.lastPathComponent
        let pageCount = document.pageCount

        // Save to local storage
        let fileManager = FileManager.default
        let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let sharedPDFsDir = documentsDir.appendingPathComponent("SharedPDFs")

        try? fileManager.createDirectory(at: sharedPDFsDir, withIntermediateDirectories: true)

        let destURL = sharedPDFsDir.appendingPathComponent("\(hashString)_\(fileName)")

        // Copy if not already exists
        if !fileManager.fileExists(atPath: destURL.path) {
            try? fileManager.copyItem(at: url, to: destURL)
        }

        let hashStr = String(hashString)

        DispatchQueue.main.async {
            self.pendingPDF = SharedPDF(
                url: destURL,
                document: document,
                hash: hashStr,
                fileName: fileName
            )
        }

        // Upload to S3 in background
        Task {
            do {
                try await DrawingSyncManager.shared.uploadSharedPDF(
                    data: data,
                    hash: hashStr,
                    fileName: fileName,
                    pageCount: pageCount
                )
            } catch {
                print("Failed to upload shared PDF to S3: \(error)")
            }
        }
    }

    func clearPending() {
        pendingPDF = nil
    }
}
