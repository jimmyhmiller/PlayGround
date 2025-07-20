# Note Canvas TODO List

## ✅ Completed High Priority

### 1. ✅ Change panning to use scroll
- **COMPLETED**: Added right-click pan gesture and scroll wheel support for canvas navigation
- **Impact**: Better user experience, no conflicts with PDF scrolling

### 2. ✅ Make PDF be larger by default
- **COMPLETED**: Increased default PDF scale to minimum 1.5x, maximum 2.0x
- **Impact**: Much better readability and annotation experience

### 3. ✅ Allow zooming of PDF 
- **COMPLETED**: Added independent PDF zoom with pinch gestures (0.5x to 5.0x range)
- **Impact**: Essential zoom functionality for detailed annotation work

## Remaining High Priority

### 4. Change it so that PDF isn't a modal, but rather a page navigation
- **Current Issue**: PDF opens as overlay modal, blocking canvas interaction
- **Goal**: Integrate PDF as a navigable page/tab within the main interface
- **Impact**: Better workflow, ability to work with multiple documents
- **Status**: Major architectural change - requires significant UI restructuring

## Medium Priority (Improvements)

### 17. Make PDF view change with the window (responsive to window resizing)
- **Current Issue**: PDF view layout may not respond properly to window size changes
- **Goal**: Ensure PDF content and UI elements scale and reposition appropriately when window is resized
- **Impact**: Better user experience across different window sizes and screen configurations

## ✅ Completed Medium Priority

### 5. ✅ Multi-color highlight tools
- **COMPLETED**: Implemented 4 highlight colors with visual color buttons
- **Colors available**:
  - 🟡 Yellow (traditional highlight)
  - 🟢 Green (approved/good)
  - 🔴 Red (important/issues)  
  - 🔵 Blue (notes/references)
- **Features**: Color-coded buttons in toolbar, works with both PDFKit and image renderer
- **Impact**: Much better annotation organization and visual categorization

### 13. ✅ Move highlighters to floating panel on right
- **COMPLETED**: Created floating panel on right side with vertical layout
- **Features**: Semi-transparent background, subtle shadow, positioned to not interfere with content
- **Impact**: Better UI organization, more space in main toolbar

### 14. ✅ Make highlighter draw more square
- **COMPLETED**: Changed highlight line caps from round to square with miter joins
- **Technical**: Square line caps (.square) and miter line joins (.miter) for highlights
- **Impact**: More traditional highlighter appearance, sharper highlight edges

### 15. ✅ Fix highlighter panel button clicks
- **COMPLETED**: Fixed hitTest method to prioritize highlighter panel events
- **Technical**: Added panel bounds check at top of hitTest override
- **Impact**: Highlighter buttons now respond properly to clicks

### 16. ✅ Fix floating panel position
- **COMPLETED**: Repositioned panel to avoid scroll bar overlap
- **Technical**: Panel now positioned relative to PDF content area, not window bounds
- **Impact**: Panel no longer interferes with scroll bar functionality

## Low Priority (Advanced Features)

### 6. Research LAB colors for dynamic highlight colors
- **Goal**: Investigate LAB color space for perceptually uniform highlight colors
- **Benefits**: 
  - Better contrast ratios
  - More pleasing color combinations
  - Accessibility improvements
- **Research areas**:
  - LAB color space implementation in AppKit
  - Perceptual uniformity for highlight colors
  - Color harmony algorithms

### 7. Full text search of PDF
- **Goal**: Implement text search functionality within PDF documents
- **Requirements**:
  - Search highlighting
  - Navigation between results
  - Case-sensitive/insensitive options
- **Technical considerations**:
  - PDFKit text extraction capabilities
  - Search result visualization
  - Performance with large documents

## Implementation Notes

### Priority Order
1. **PDF Zoom** (Critical for usability)
2. **Larger default size** (Quick win)
3. **Page navigation instead of modal** (Architecture change)
4. **Scroll-based panning** (UX improvement)
5. **Multi-color highlight sidebar** (Feature enhancement)
6. **LAB color research** (Research task)
7. **Text search** (Advanced feature)

### Technical Considerations
- **PDF Zoom**: Need to modify ImagePDFRenderer to handle zoom independently from canvas
- **Page Navigation**: Major UI restructuring - consider tab-based or sidebar navigation
- **Color Tools**: Design sidebar UI, implement color picker/palette system
- **LAB Colors**: May require custom color conversion utilities
- **Text Search**: Leverage PDFKit's built-in text extraction capabilities

### Dependencies
- Items 1-4 are PDF viewer improvements that can be tackled in parallel
- Item 5 depends on UI design decisions from item 4
- Item 6 is research that can inform item 5
- Item 7 can be developed independently once PDF navigation is stable