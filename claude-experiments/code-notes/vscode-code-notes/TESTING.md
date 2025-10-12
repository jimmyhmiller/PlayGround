# Testing the Code Notes VS Code Extension

## Quick Start

### 1. Open the Extension in VS Code

```bash
cd vscode-code-notes
code .
```

### 2. Run the Extension

Press `F5` in VS Code to launch a new Extension Development Host window.

### 3. Open the Beagle Project

In the Extension Development Host window:
- File → Open Folder
- Navigate to `/Users/jimmyhmiller/Documents/Code/beagle`
- Open the folder

### 4. View Notes

The extension should automatically:
- Show blue circle indicators in the gutter for lines with notes
- Display notes when you hover over those lines

Try opening these files that have notes:
- `src/types.rs` (lines 34, 163)
- `src/main.rs` (lines 113, 538)
- `src/compiler.rs` (lines 36, 657)
- `src/gc/mod.rs` (line 26)
- `src/ir.rs` (line 52)
- `src/register_allocation/linear_scan.rs` (line 27)

### 5. Test Commands

- Open Command Palette (`Cmd+Shift+P`)
- Try these commands:
  - `Code Notes: Show All Notes in File`
  - `Code Notes: Refresh Notes`

## What to Look For

### Gutter Icons
- Blue circles should appear in the gutter next to lines with notes
- The circles should persist as you scroll
- They should update when you switch between files

### Hover
- Hover over a line with a gutter icon
- You should see a popup with:
  - Note author
  - Creation date
  - Note content
  - Metadata (if any)

### Note Details Panel
- Use "Show All Notes in File" command
- Select a note from the quick pick
- Should open a side panel with full note details
- Should navigate to the note location in the editor

## Known Issues

- The gutter icon uses an emoji which may not render perfectly
- Notes are only loaded when the extension activates (workspace open)
- File watching works but may have a slight delay

## Debug Output

Check the Debug Console in VS Code for logs:
- Extension activation messages
- Note loading information
- File watching updates

## Configuration

Test configuration changes:
1. Open Settings (`Cmd+,`)
2. Search for "Code Notes"
3. Try changing the gutter icon color
4. The decorations should update automatically
