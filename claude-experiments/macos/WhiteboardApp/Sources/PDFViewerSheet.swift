import SwiftUI
import Quartz

struct PDFViewerView: View {
    let pdfItem: PDFItem
    let onClose: () -> Void
    
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button(action: onClose) {
                    Label("Back to Canvas", systemImage: "arrow.left")
                }
                .buttonStyle(.bordered)
                .keyboardShortcut(.escape)
                
                Spacer()
                
                Text("PDF Viewer")
                    .font(.headline)
                
                Spacer()
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            PDFKitView(pdfDocument: pdfItem.pdfDocument)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

struct PDFKitView: NSViewRepresentable {
    let pdfDocument: PDFDocument
    
    func makeNSView(context: Context) -> PDFView {
        let pdfView = PDFView()
        pdfView.document = pdfDocument
        pdfView.autoScales = true
        pdfView.displaysPageBreaks = true
        pdfView.displayMode = .singlePageContinuous
        pdfView.backgroundColor = NSColor.windowBackgroundColor
        return pdfView
    }
    
    func updateNSView(_ nsView: PDFView, context: Context) {
        nsView.document = pdfDocument
    }
}