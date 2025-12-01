# PDF Library Sidebar Features

## Overview

The app now includes a full PDF library sidebar that:
- Loads your entire PDF collection from the JSON index
- Organizes PDFs by folder
- Provides search functionality across all metadata
- Downloads PDFs from S3 on-demand
- Caches downloaded PDFs locally

## Features

### 📚 PDF Library Sidebar

- **Folder Navigation**: Browse PDFs organized by category (ai, books, computer-philosophy, etc.)
- **Expandable Folders**: Click folder headers to expand/collapse
- **PDF Count**: Shows number of PDFs in each folder
- **Rich Metadata Display**:
  - Title (from OCR or metadata)
  - Author name
  - Page count

### 🔍 Search

- **Global Search**: Search ignores folder structure
- **Searches across**:
  - Titles (OCR title preferred)
  - Author names
  - File names
- **Real-time filtering**: Results update as you type
- **Clear button**: Quick reset

### ⬇️ Smart Downloads

- **S3 Integration**: Downloads PDFs directly from your S3 bucket
  - URL format: `https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/{folder}/{hash}.pdf`
- **Progress Indicator**: Shows "Downloading..." status in toolbar
- **Auto-caching**: Downloaded PDFs cached locally to avoid re-downloading
- **Cache Management**: PDFs stored in app's cache directory

### 💾 Local Caching

- **Cache Location**: `~/Library/Caches/PDFCache/`
- **File Naming**: PDFs cached by hash (e.g., `abc123...xyz.pdf`)
- **Smart Loading**: Checks cache before downloading
- **Cache Info**: Downloader tracks cache size (accessible via code)

## File Structure

```
PDFMarkup/
├── PDFMarkupApp.swift           # App entry point
├── ContentView.swift            # Main view with sidebar integration
├── PDFMetadata.swift            # Data models (PDFMetadata, PDFLibrary)
├── PDFDownloader.swift          # S3 download + caching logic
├── PDFLibrarySidebar.swift      # Sidebar UI component
└── sample.pdf                   # Bundled sample PDF
```

## How It Works

1. **On Launch**:
   - Loads `/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/reading-tools/pdf-indexer/pdf-index.json`
   - Parses metadata for all PDFs
   - Groups PDFs by folder (extracted from file path)

2. **Browsing**:
   - Folders show in alphabetical order
   - Click to expand and see PDFs
   - Click a PDF to select it

3. **Selecting a PDF**:
   - Checks local cache first
   - If not cached, downloads from S3
   - Shows download progress
   - Opens PDF in viewer when ready

4. **Search**:
   - Type in search box at top of sidebar
   - Shows flat list of matching PDFs
   - Click to load

## Configuration

### S3 Settings (in PDFMetadata.swift)

```swift
func s3URL(bucket: String = "jimmyhmiller-bucket",
           prefix: String = "pdfs") -> URL?
```

### Index Path (in ContentView.swift)

```swift
let indexPath = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/reading-tools/pdf-indexer/pdf-index.json"
```

## Usage

1. **Browse by Folder**:
   - Click folder name to expand
   - Click PDF name to load

2. **Search Across All PDFs**:
   - Type in search box
   - Click any result to load

3. **Open Local File** (still available):
   - Click "Open File" button in toolbar
   - Select any PDF from your computer

## Technical Details

- **NavigationSplitView**: Native iOS/macOS sidebar
- **Async/Await**: Modern concurrency for downloads
- **URLSession**: Standard HTTP downloads from S3
- **FileManager**: Local cache management
- **Codable**: JSON parsing for metadata
- **Published Properties**: Reactive UI updates

## Future Enhancements

Possible additions:
- Recently viewed PDFs
- Favorites/bookmarks
- Download all PDFs in folder
- Offline mode indicator
- Cache size limit/cleanup
- Sort options (by title, author, date, page count)
- Folder icons/colors
