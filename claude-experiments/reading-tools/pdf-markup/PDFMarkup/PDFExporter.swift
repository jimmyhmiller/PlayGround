//
//  PDFExporter.swift
//  PDFMarkup
//
//  Exports PDFs with PencilKit drawings flattened onto pages
//

import Foundation
import PDFKit
import PencilKit

#if os(macOS)
import AppKit
#else
import UIKit
#endif

struct PDFExporter {

    static func exportWithMarkups(
        document: PDFDocument,
        pdfHash: String,
        outputURL: URL
    ) throws {
        let drawingManager = DrawingManager.shared
        let pdfData = NSMutableData()

        #if os(macOS)
        // macOS: Use CGContext directly
        var mediaBox = CGRect.zero
        if let firstPage = document.page(at: 0) {
            mediaBox = firstPage.bounds(for: .mediaBox)
        }

        guard let consumer = CGDataConsumer(data: pdfData as CFMutableData),
              let pdfContext = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
            throw NSError(domain: "PDFExporter", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create PDF context"])
        }

        for pageIndex in 0..<document.pageCount {
            guard let page = document.page(at: pageIndex) else { continue }
            let bounds = page.bounds(for: .mediaBox)

            pdfContext.beginPDFPage(nil)

            // Draw the original PDF page
            pdfContext.saveGState()
            page.draw(with: .mediaBox, to: pdfContext)
            pdfContext.restoreGState()

            // Draw any markups on top
            if let drawing = drawingManager.loadDrawing(pdfHash: pdfHash, page: pageIndex) {
                let image = drawing.image(from: bounds, scale: 2.0)
                if let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                    pdfContext.draw(cgImage, in: bounds)
                }
            }

            pdfContext.endPDFPage()
        }

        pdfContext.closePDF()
        #else
        // iOS: Use UIGraphics PDF context
        UIGraphicsBeginPDFContextToData(pdfData, .zero, nil)
        defer { UIGraphicsEndPDFContext() }

        for pageIndex in 0..<document.pageCount {
            guard let page = document.page(at: pageIndex) else { continue }
            let bounds = page.bounds(for: .mediaBox)

            UIGraphicsBeginPDFPageWithInfo(bounds, nil)
            guard let context = UIGraphicsGetCurrentContext() else { continue }

            context.saveGState()
            context.translateBy(x: 0, y: bounds.height)
            context.scaleBy(x: 1.0, y: -1.0)
            page.draw(with: .mediaBox, to: context)
            context.restoreGState()

            if let drawing = drawingManager.loadDrawing(pdfHash: pdfHash, page: pageIndex) {
                let image = drawing.image(from: bounds, scale: 2.0)
                image.draw(in: bounds)
            }
        }
        #endif

        try pdfData.write(to: outputURL, options: .atomic)
    }

    static func exportToTemporaryFile(
        document: PDFDocument,
        pdfHash: String
    ) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let filename = "exported-\(pdfHash).pdf"
        let outputURL = tempDir.appendingPathComponent(filename)

        try? FileManager.default.removeItem(at: outputURL)

        try exportWithMarkups(document: document, pdfHash: pdfHash, outputURL: outputURL)
        return outputURL
    }

    static func hasDrawings(pdfHash: String) -> Bool {
        let drawingManager = DrawingManager.shared
        if let drawings = drawingManager.loadDrawings(pdfHash: pdfHash) {
            return drawings.hasDrawings
        }
        return false
    }
}
