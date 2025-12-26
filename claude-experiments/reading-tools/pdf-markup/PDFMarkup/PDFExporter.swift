//
//  PDFExporter.swift
//  PDFMarkup
//
//  Exports PDFs with PencilKit drawings flattened onto pages
//

import Foundation
import PDFKit
import PencilKit
import UIKit

struct PDFExporter {

    /// Exports a PDF document with drawings flattened onto the pages
    /// - Parameters:
    ///   - document: The source PDF document
    ///   - pdfHash: The hash of the PDF to load drawings for
    ///   - outputURL: The destination URL for the exported PDF
    /// - Throws: Any errors that occur during PDF creation or file writing
    static func exportWithMarkups(
        document: PDFDocument,
        pdfHash: String,
        outputURL: URL
    ) throws {
        let drawingManager = DrawingManager.shared
        let pdfData = NSMutableData()

        // Begin PDF context
        UIGraphicsBeginPDFContextToData(pdfData, .zero, nil)
        defer { UIGraphicsEndPDFContext() }

        // Process each page
        for pageIndex in 0..<document.pageCount {
            guard let page = document.page(at: pageIndex) else { continue }
            let bounds = page.bounds(for: .mediaBox)

            // Start a new PDF page
            UIGraphicsBeginPDFPageWithInfo(bounds, nil)
            guard let context = UIGraphicsGetCurrentContext() else { continue }

            // Draw the original PDF page
            context.saveGState()
            context.translateBy(x: 0, y: bounds.height)
            context.scaleBy(x: 1.0, y: -1.0)
            page.draw(with: .mediaBox, to: context)
            context.restoreGState()

            // Draw any markups on top
            if let drawing = drawingManager.loadDrawing(pdfHash: pdfHash, page: pageIndex) {
                let image = drawing.image(from: bounds, scale: 2.0) // Use 2x scale for better quality
                image.draw(in: bounds)
            }
        }

        // Write to file
        try pdfData.write(to: outputURL, options: .atomic)
    }

    /// Convenience method to export with markups to a temporary file
    /// - Parameters:
    ///   - document: The source PDF document
    ///   - pdfHash: The hash of the PDF to load drawings for
    /// - Returns: URL of the exported PDF in the temporary directory
    /// - Throws: Any errors that occur during export
    static func exportToTemporaryFile(
        document: PDFDocument,
        pdfHash: String
    ) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let filename = "exported-\(pdfHash).pdf"
        let outputURL = tempDir.appendingPathComponent(filename)

        // Remove existing file if it exists
        try? FileManager.default.removeItem(at: outputURL)

        try exportWithMarkups(document: document, pdfHash: pdfHash, outputURL: outputURL)
        return outputURL
    }

    /// Checks if a PDF has any drawings
    /// - Parameter pdfHash: The hash of the PDF to check
    /// - Returns: True if the PDF has drawings, false otherwise
    static func hasDrawings(pdfHash: String) -> Bool {
        let drawingManager = DrawingManager.shared
        if let drawings = drawingManager.loadDrawings(pdfHash: pdfHash) {
            return drawings.hasDrawings
        }
        return false
    }
}
