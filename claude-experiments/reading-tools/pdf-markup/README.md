# PDF Markup App

A simple iPad/macOS app for highlighting PDFs using PencilKit, with an easy-to-use color picker.

## Features

- ✏️ Use PencilKit for smooth, natural highlighting
- 🎨 Quick color switching (Yellow, Green, Pink, Orange, Blue, Purple)
- 📄 Navigate through PDF pages
- 💻 Works on iPad and macOS (via Mac Catalyst)
- 🚫 No default PencilKit toolbar - just simple color buttons

## Quick Start

The Xcode project is ready to go with a sample PDF included! Just open it:

```bash
open PDFMarkup.xcodeproj
```

Or double-click `PDFMarkup.xcodeproj` in Finder.

The app will automatically load a sample PDF when you run it, so you can start highlighting immediately!

## Running the App

### On Mac (via Catalyst)
- Select "My Mac (Designed for iPad)" as the run destination in Xcode
- Click Run (⌘R)

### On iPad
- Connect your iPad
- Select your iPad as the run destination
- Click Run (⌘R)

### On iPad Simulator
- Select an iPad simulator (e.g., "iPad Pro 12.9-inch")
- Click Run (⌘R)

## Usage

1. **Start highlighting**: A sample PDF loads automatically when you launch the app
2. **Choose a color**: Tap any of the colored circles at the top to switch colors
3. **Highlight**: Draw on the PDF with Apple Pencil (or mouse/trackpad on Mac)
4. **Navigate**: Use the arrow buttons to move between pages
5. **Open your own PDF** (optional): Click the "Open PDF" button to load a different file

## Why Not `swift run`?

PencilKit's PKCanvasView is only available in proper iOS/iPadOS app bundles. It's not available for command-line Swift tools on macOS. Using Mac Catalyst gives you the full PencilKit experience on both iPad and Mac.

## Technical Details

- **Framework**: SwiftUI + PencilKit + PDFKit
- **Platforms**: iOS 17+, macOS 14+ (via Catalyst)
- **Drawing Policy**: `.anyInput` (works with Pencil, touch, and mouse)
- **Ink Type**: `.marker` (semi-transparent, good for highlighting)
