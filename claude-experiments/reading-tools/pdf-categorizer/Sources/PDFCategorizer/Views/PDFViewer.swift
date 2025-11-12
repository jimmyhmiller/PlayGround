import SwiftUI
import PDFKit

struct PDFViewer: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> PDFView {
        let pdfView = PDFView()
        pdfView.autoScales = true
        pdfView.displayMode = .singlePageContinuous
        pdfView.displayDirection = .vertical

        // Enable smooth scrolling
        pdfView.interpolationQuality = .high

        return pdfView
    }

    func updateNSView(_ pdfView: PDFView, context: Context) {
        // Only load if URL changed
        if pdfView.document?.documentURL != url {
            loadPDFAsync(into: pdfView, from: url)
        }
    }

    private func loadPDFAsync(into pdfView: PDFView, from url: URL) {
        Task.detached(priority: .userInitiated) {
            // Load PDF document in background
            guard let document = PDFDocument(url: url) else {
                await MainActor.run {
                    pdfView.document = nil
                }
                return
            }

            // Set document on main thread
            await MainActor.run {
                pdfView.document = document

                // Immediately display first page for speed
                if let firstPage = document.page(at: 0) {
                    pdfView.go(to: firstPage)
                }
            }

            // Prefetch remaining pages in background
            let pageCount = document.pageCount
            for i in 1..<min(pageCount, 10) {
                // Prefetch first 10 pages
                _ = document.page(at: i)
            }
        }
    }
}
