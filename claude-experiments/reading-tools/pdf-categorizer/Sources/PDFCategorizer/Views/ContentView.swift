import SwiftUI

struct ContentView: View {
    @StateObject private var pdfManager = PDFManager()
    @StateObject private var categoryManager = CategoryManager()
    @State private var errorMessage: String?

    var body: some View {
        Group {
            if pdfManager.workingDirectory == nil {
                welcomeView
            } else if pdfManager.pdfFiles.isEmpty {
                noPDFsView
            } else {
                mainView
            }
        }
        .frame(minWidth: 1000, minHeight: 700)
        .alert("Error", isPresented: .constant(errorMessage != nil)) {
            Button("OK") { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
    }

    private var welcomeView: some View {
        VStack(spacing: 20) {
            Text("PDF Categorizer")
                .font(.largeTitle)
                .bold()

            Text("Select a directory containing PDFs to categorize")
                .foregroundColor(.secondary)

            Button("Select Directory") {
                Task {
                    await pdfManager.selectDirectory()
                    categoryManager.setWorkingDirectory(pdfManager.workingDirectory)
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding()
    }

    private var noPDFsView: some View {
        VStack(spacing: 20) {
            Text("No PDFs Found")
                .font(.title)
                .bold()

            Text("No uncategorized PDFs in the selected directory")
                .foregroundColor(.secondary)

            Button("Select Different Directory") {
                Task {
                    await pdfManager.selectDirectory()
                    categoryManager.setWorkingDirectory(pdfManager.workingDirectory)
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    private var mainView: some View {
        HStack(spacing: 0) {
            // Left: PDF Viewer
            VStack(spacing: 0) {
                // Header
                HStack {
                    Text(pdfManager.currentPDF?.lastPathComponent ?? "")
                        .font(.headline)
                        .lineLimit(1)

                    Spacer()

                    Text(pdfManager.progress)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color(NSColor.controlBackgroundColor))

                // PDF Viewer
                if let pdfURL = pdfManager.currentPDF {
                    PDFViewer(url: pdfURL)
                } else {
                    Text("No PDF selected")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }

                // Navigation Controls
                HStack {
                    Button("Previous") {
                        pdfManager.goToPrevious()
                    }
                    .disabled(pdfManager.currentIndex == 0)

                    Spacer()

                    Button("Skip") {
                        pdfManager.skipCurrent()
                    }
                    .disabled(pdfManager.currentIndex >= pdfManager.pdfFiles.count - 1)
                }
                .padding()
                .background(Color(NSColor.controlBackgroundColor))
            }

            // Right: Category Sidebar
            CategorySidebar(
                categoryManager: categoryManager,
                onCategorySelected: { category in
                    do {
                        try pdfManager.movePDFToCategory(category)
                    } catch {
                        errorMessage = "Failed to move PDF: \(error.localizedDescription)"
                    }
                }
            )
            .frame(width: 250)
        }
    }
}
