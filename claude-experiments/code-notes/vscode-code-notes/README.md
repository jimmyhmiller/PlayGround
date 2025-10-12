# Code Notes VS Code Extension

Display layered annotations for code that persist across commits.

## Features

- **Gutter Icons**: Shows visual indicators on lines that have notes attached
- **Hover Support**: Hover over lines with notes to see the full note content
- **Quick Navigation**: Command to view all notes in the current file
- **Live Updates**: Automatically refreshes when notes are added or modified
- **Collection Support**: Works with note collections (personal, architecture, onboarding, etc.)

## Usage

1. Make sure you have notes created using the `code-notes` CLI tool
2. Open a project in VS Code that has notes
3. Lines with notes will show a blue circle indicator in the gutter
4. Hover over highlighted lines to see note content
5. Use "Code Notes: Show All Notes in File" command to see all notes

## Commands

- `Code Notes: Refresh Notes` - Manually refresh notes from disk
- `Code Notes: Show All Notes in File` - Show all notes in a quick pick menu

## Configuration

- `codeNotes.storageDir` - Directory where code notes are stored (default: `~/.code-notes`)
- `codeNotes.gutterIconColor` - Color for the gutter icon (default: `#4A9EFF`)

## Requirements

The `code-notes` CLI tool must be installed and you must have created notes for your project.

## Extension Development

To run the extension in development mode:

1. `npm install`
2. `npm run compile`
3. Press F5 to open a new VS Code window with the extension loaded
