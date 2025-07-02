import SwiftUI

struct ContentView: View {
    @State private var rectangles: [Rectangle] = []
    @State private var textBubbles: [TextBubble] = []
    @State private var pdfItems: [PDFItem] = []
    @State private var selectedColor: Color = .blue
    @State private var selectedRectangle: Rectangle?
    @State private var selectedTextBubble: TextBubble?
    @State private var selectedPDF: PDFItem?
    @State private var selectedTool: Tool = .select
    @State private var viewingPDF: PDFItem?
    @State private var pdfImages: [PDFImageService.PDFPageImage] = []
    @State private var isLoadingPDFImages = false
    @State private var pdfLoadingProgress: Double = 0.0
    
    let colors: [Color] = [
        .blue, .red, .green, .yellow, .orange, .purple, .pink, .gray, .black
    ]
    
    var body: some View {
        HStack(spacing: 0) {
            if viewingPDF != nil {
                if isLoadingPDFImages {
                    VStack {
                        ProgressView("Converting PDF to Images...", value: pdfLoadingProgress, total: 1.0)
                            .progressViewStyle(LinearProgressViewStyle())
                            .frame(width: 300)
                        Text("\(Int(pdfLoadingProgress * 100))% complete")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.white)
                } else {
                    ImageCanvasView(
                        rectangles: $rectangles,
                        textBubbles: $textBubbles,
                        selectedColor: $selectedColor,
                        selectedRectangle: $selectedRectangle,
                        selectedTextBubble: $selectedTextBubble,
                        selectedTool: $selectedTool,
                        pdfImages: pdfImages
                    )
                    .background(Color.white)
                }
            } else {
                CanvasView(
                    rectangles: $rectangles,
                    textBubbles: $textBubbles,
                    pdfItems: $pdfItems,
                    selectedColor: $selectedColor,
                    selectedRectangle: $selectedRectangle,
                    selectedTextBubble: $selectedTextBubble,
                    selectedPDF: $selectedPDF,
                    selectedTool: $selectedTool,
                    onPDFDoubleClick: { pdfItem in
                        convertPDFToImages(pdfItem)
                    }
                )
                .background(Color.white)
            }
            
            Divider()
            
            VStack(spacing: 0) {
                VStack(spacing: 16) {
                    Text("Tools")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    VStack(spacing: 8) {
                        ForEach(Tool.allCases, id: \.self) { tool in
                            ToolButton(
                                tool: tool,
                                isSelected: selectedTool == tool,
                                action: { selectedTool = tool }
                            )
                        }
                    }
                    
                    Divider()
                    
                    Text("Colors")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 8) {
                        ForEach(colors, id: \.self) { color in
                            ColorButton(
                                color: color,
                                isSelected: selectedColor == color,
                                action: { 
                                    selectedColor = color
                                    changeSelectedRectangleColor(to: color)
                                }
                            )
                        }
                    }
                    
                    Spacer()
                    
                    VStack(spacing: 8) {
                        if viewingPDF != nil {
                            Button("Back to Canvas") {
                                viewingPDF = nil
                                pdfImages = []
                                isLoadingPDFImages = false
                            }
                            .buttonStyle(.bordered)
                            .tint(.blue)
                        }
                        
                        if selectedRectangle != nil || selectedTextBubble != nil || selectedPDF != nil {
                            Button("Delete Selected") {
                                if let selected = selectedRectangle {
                                    rectangles.removeAll { $0.id == selected.id }
                                    selectedRectangle = nil
                                } else if let selected = selectedTextBubble {
                                    textBubbles.removeAll { $0.id == selected.id }
                                    selectedTextBubble = nil
                                } else if let selected = selectedPDF {
                                    pdfItems.removeAll { $0.id == selected.id }
                                    selectedPDF = nil
                                }
                            }
                            .buttonStyle(.bordered)
                            .tint(.red)
                        }
                        
                        Button("Clear All") {
                            rectangles.removeAll()
                            textBubbles.removeAll()
                            pdfItems.removeAll()
                            selectedRectangle = nil
                            selectedTextBubble = nil
                            selectedPDF = nil
                        }
                        .buttonStyle(.bordered)
                        
                        Divider()
                        
                        Text("Debug")
                            .font(.headline)
                            .fontWeight(.semibold)
                        
                        Button("Stress Test (1000)") {
                            stressTest()
                        }
                        .buttonStyle(.bordered)
                        .tint(.orange)
                        
                        if viewingPDF != nil {
                            Button("← Back to Canvas") {
                                viewingPDF = nil
                                pdfImages = []
                                isLoadingPDFImages = false
                            }
                            .buttonStyle(.bordered)
                            
                            Divider()
                        }
                        
                        VStack(spacing: 2) {
                            Text("Rectangles: \(rectangles.count)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("Text Bubbles: \(textBubbles.count)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            if viewingPDF == nil {
                                Text("PDFs: \(pdfItems.count)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            } else {
                                Text("PDF Pages: \(pdfImages.count)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
                .padding()
            }
            .frame(width: 200)
            .background(Color(NSColor.controlBackgroundColor))
        }
    }
    
    private func changeSelectedRectangleColor(to color: Color) {
        guard let selectedRect = selectedRectangle else { return }
        
        for i in 0..<rectangles.count {
            if rectangles[i].id == selectedRect.id {
                rectangles[i].color = color
                selectedRectangle = rectangles[i]
                break
            }
        }
    }
    
    private func convertPDFToImages(_ pdfItem: PDFItem) {
        isLoadingPDFImages = true
        pdfLoadingProgress = 0.0
        viewingPDF = pdfItem
        
        Task {
            let images = await PDFImageService.shared.convertPDFToImages(
                pdfItem.pdfDocument
            ) { progress in
                DispatchQueue.main.async {
                    pdfLoadingProgress = progress
                }
            }
            
            DispatchQueue.main.async {
                pdfImages = images
                isLoadingPDFImages = false
            }
        }
    }
    
    private func stressTest() {
        let canvasSize = CGSize(width: 800, height: 600)
        let minSize: CGFloat = 20
        let maxSize: CGFloat = 100
        
        for _ in 0..<1000 {
            let width = CGFloat.random(in: minSize...maxSize)
            let height = CGFloat.random(in: minSize...maxSize)
            let x = CGFloat.random(in: 0...(canvasSize.width - width))
            let y = CGFloat.random(in: 0...(canvasSize.height - height))
            let color = colors.randomElement() ?? .blue
            
            let newRectangle = Rectangle(
                origin: CGPoint(x: x, y: y),
                size: CGSize(width: width, height: height),
                color: color
            )
            rectangles.append(newRectangle)
        }
    }
    
}

struct ToolButton: View {
    let tool: Tool
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: tool.icon)
                    .frame(width: 16)
                Text(tool.name)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(isSelected ? Color.accentColor : Color.clear)
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(6)
            .contentShape(SwiftUI.Rectangle())  // Make entire button area clickable
        }
        .buttonStyle(.borderless)  // Better hit area than .plain
    }
}

struct ColorButton: View {
    let color: Color
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            RoundedRectangle(cornerRadius: 6)
                .fill(color)
                .frame(width: 32, height: 32)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(
                            isSelected ? Color.black : Color.gray.opacity(0.3),
                            lineWidth: isSelected ? 3 : 1
                        )
                )
        }
        .buttonStyle(.plain)
    }
}