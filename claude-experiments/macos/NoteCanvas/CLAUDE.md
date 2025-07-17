# Note Canvas Project

## Overview
A note-taking canvas app for macOS built with AppKit in Swift. The goal is to recreate a polished, highly animated interface similar to the reference design with support for various note types (text, images, PDFs, files, sticky notes) and folder groupings.

![Reference Design](./reference-design.png)

## Current Status

### ✅ Completed Foundation
- **Core Data Models**: Protocol-oriented design with `NoteItem` protocol, `TextNote`, `StickyNote`, `ImageNote`, and `Canvas` models
- **Base View System**: `BaseNoteView` with layer-based rendering (shadows, selection, rounded corners)
- **Visual Effects Library**: Reusable components for shadows, animations, and visual styling
- **Project Structure**: Proper Swift Package with library and executable targets
- **Build System**: App compiles and runs successfully

### ⚠️ Partially Implemented (Needs Testing/Fixing)
- **CanvasView**: Has mouse/keyboard handling code but interactions may not work properly
- **Note Views**: `TextNoteView`, `ImageNoteView`, `StickyNoteView` exist but dragging/editing may be broken
- **Selection System**: Visual feedback code exists but hit testing likely needs fixes
- **Pan/Zoom**: Code exists but viewport updates may not work correctly

### ❌ Not Yet Implemented
- Individual note dragging (mouse events not properly wired)
- Working pan/zoom with spacebar
- Functional multi-selection with drag rectangles
- Folder view with expand/collapse animations
- Resize handles and context menus
- Animation system for smooth transitions
- Layout engine for snap-to-grid
- Theme system for consistent styling

## Todo List

### High Priority
- [ ] **Fix note dragging** - Debug and fix individual note movement
- [ ] **Fix canvas pan/zoom** - Ensure spacebar panning and trackpad zoom work
- [ ] **Fix selection system** - Debug multi-select drag rectangles and hit testing
- [ ] **Build animation system** - Smooth transitions for note movement and selection
- [ ] **Write unit tests** - Test data model and business logic

### Medium Priority
- [ ] **Create FolderView** - iPhone-style app groups with expand/collapse
- [ ] **Build interaction system** - Resize handles, context menus, drag-and-drop
- [ ] **Write UI tests** - Test user interactions and animations

### Low Priority
- [ ] **Create layout engine** - Automatic arrangement and snap-to-grid
- [ ] **Create theme system** - Consistent colors, fonts, and spacing

## Architecture Notes

### Data Flow
- `Canvas` class manages note collection and state using Combine `@Published` properties
- `CanvasView` observes canvas changes and updates note views reactively
- Individual note views inherit from `BaseNoteView` and implement `NoteViewProtocol`

### Key Design Decisions
- Protocol-oriented design for extensibility
- Separation of data models (`NoteItem`) from views (`NoteViewProtocol`)
- Layer-based rendering for visual effects
- Combine for reactive updates
- Public API design for library/app separation

## Known Issues
1. **Mouse Event Handling**: Note dragging likely broken due to event handling conflicts
2. **Coordinate Systems**: Viewport transformations may not be applied correctly
3. **Hit Testing**: Multi-selection rectangle intersection logic needs verification
4. **Text Editing**: Text note editing may not work properly
5. **Performance**: No optimization for large numbers of notes

## Next Steps
1. **Debug interactive features** - Test and fix mouse handling, dragging, selection
2. **Implement missing core features** - Folders, animations, resize handles
3. **Polish and optimize** - Smooth animations, performance improvements
4. **Add tests** - Ensure reliability and prevent regressions

## Build Instructions
```bash
swift build
swift run NoteCanvasApp
```

## Project Structure
```
NoteCanvas/
├── Sources/
│   ├── NoteCanvas/          # Main library
│   │   ├── Models/          # Data models
│   │   │   ├── NoteProtocol.swift      # Core protocols
│   │   │   ├── TextNote.swift          # Text note model
│   │   │   ├── StickyNote.swift        # Sticky note model
│   │   │   ├── ImageNote.swift         # Image note model
│   │   │   ├── FolderNote.swift        # Folder model
│   │   │   └── Canvas.swift            # Main canvas state
│   │   ├── Views/           # UI components
│   │   │   ├── NoteViewProtocol.swift  # Base view protocol
│   │   │   ├── CanvasView.swift        # Main canvas view
│   │   │   ├── TextNoteView.swift      # Text note rendering
│   │   │   ├── StickyNoteView.swift    # Sticky note rendering
│   │   │   └── ImageNoteView.swift     # Image note rendering
│   │   └── Utilities/       # Helper components
│   │       └── VisualEffects.swift     # Reusable effects
│   └── NoteCanvasApp/       # Executable app
│       └── main.swift       # App entry point
├── Tests/                   # Test files (empty)
├── Package.swift           # Swift package manifest
├── CLAUDE.md              # This documentation
└── reference-design.png   # UI reference image
```

## Key Files to Understand

### Core Models (`Sources/NoteCanvas/Models/`)
- **`NoteProtocol.swift`**: Defines `NoteItem` protocol that all notes implement
- **`Canvas.swift`**: Main state management with `@Published` properties for reactive updates
- **`TextNote.swift`**: Text note with font, alignment, content properties
- **`StickyNote.swift`**: Sticky note with color variants and tape effect
- **`AnyNote`**: Type-erased wrapper for heterogeneous note collections

### Core Views (`Sources/NoteCanvas/Views/`)
- **`CanvasView.swift`**: Main canvas with pan/zoom/selection (⚠️ needs debugging)
- **`BaseNoteView.swift`**: Shared note view logic with layers and mouse handling
- **Individual note views**: Render specific note types with proper styling

## Critical Issues to Fix First

### 1. Mouse Event Handling (`CanvasView.swift:201-258`)
The canvas intercepts all mouse events, preventing individual notes from receiving them. Need to:
- Fix event routing between canvas and note views
- Ensure note dragging works alongside canvas panning
- Test hit testing logic for multi-selection

### 2. Coordinate System Issues
- Viewport offset updates may not properly transform `contentView`
- Zoom transforms need to be applied correctly
- Note positioning relative to canvas viewport needs verification

### 3. Note View Integration
- Note views exist but may not be properly wired to canvas delegates
- Text editing in `TextNoteView` needs testing
- Selection visual feedback may not trigger

## Design Patterns Used

### Protocol-Oriented Design
```swift
protocol NoteItem: Identifiable, Codable, Hashable {
    var position: CGPoint { get set }
    var size: CGSize { get set }
    // ... other common properties
}
```

### Reactive State Management
```swift
class Canvas: ObservableObject {
    @Published var notes: [AnyNote] = []
    @Published var selectedNotes: Set<UUID> = []
    // Canvas automatically updates views when these change
}
```

### Layer-Based Rendering
```swift
class BaseNoteView: NSView {
    var shadowLayer: CALayer!    // Drop shadow
    var contentLayer: CALayer!   // Main content
    var selectionLayer: CALayer! // Selection border
}
```

## Testing Strategy
1. **Manual Testing**: Run app and verify each interaction works
2. **Unit Tests**: Test data model logic and canvas state management
3. **Integration Tests**: Test view updates when canvas state changes
4. **UI Tests**: Test complete user workflows

## Architecture Dependencies
- **AppKit**: Core UI framework
- **Combine**: Reactive programming for data flow
- **CoreAnimation**: Layer-based animations and effects

## Reference Design Analysis
The target design shows:
- Various note types (text, images, sticky notes)
- Rounded corners and drop shadows
- Overlapping layout with depth
- Sticky notes with realistic tape effects
- Mixed content types (text documents, images, handwritten notes)
- Light background with subtle texture