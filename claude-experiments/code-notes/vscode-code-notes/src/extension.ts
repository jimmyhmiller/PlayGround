import * as vscode from 'vscode';
import * as path from 'path';
import { NotesReader } from './notesReader';
import { DecorationProvider } from './decorationProvider';
import { NotesHoverProvider } from './hoverProvider';
import { Note } from './types';

let notesReader: NotesReader;
let decorationProvider: DecorationProvider;
let hoverProvider: NotesHoverProvider;
let fileWatcher: any = null;

export function activate(context: vscode.ExtensionContext) {
    console.log('Code Notes extension is now active');

    // Get configuration
    const config = vscode.workspace.getConfiguration('codeNotes');
    const storageDir = config.get<string>('storageDir', '~/.code-notes');
    const gutterIconColor = config.get<string>('gutterIconColor', '#4A9EFF');

    // Initialize components
    notesReader = new NotesReader(storageDir);
    decorationProvider = new DecorationProvider(gutterIconColor);
    hoverProvider = new NotesHoverProvider();

    // Register hover provider for all languages
    context.subscriptions.push(
        vscode.languages.registerHoverProvider(
            { scheme: 'file', pattern: '**/*' },
            hoverProvider
        )
    );

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('code-notes.refreshNotes', () => {
            refreshAllNotes();
            vscode.window.showInformationMessage('Code Notes refreshed');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('code-notes.showNotes', () => {
            showNotesForCurrentFile();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('code-notes.showNoteAtCursor', () => {
            showNoteAtCursor();
        })
    );

    // Update decorations when the active editor changes
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(editor => {
            if (editor) {
                updateDecorationsForEditor(editor);
            }
        })
    );

    // Update decorations when the document changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(event => {
            const editor = vscode.window.activeTextEditor;
            if (editor && event.document === editor.document) {
                // Debounce updates
                setTimeout(() => updateDecorationsForEditor(editor), 500);
            }
        })
    );

    // Update decorations when configuration changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('codeNotes')) {
                const newConfig = vscode.workspace.getConfiguration('codeNotes');
                const newColor = newConfig.get<string>('gutterIconColor', '#4A9EFF');

                // Recreate decoration provider with new color
                decorationProvider.dispose();
                decorationProvider = new DecorationProvider(newColor);

                refreshAllNotes();
            }
        })
    );

    // Initial update for the active editor
    if (vscode.window.activeTextEditor) {
        updateDecorationsForEditor(vscode.window.activeTextEditor);
        setupFileWatcher();
    }

    context.subscriptions.push({
        dispose: () => {
            if (fileWatcher) {
                fileWatcher.close();
            }
            decorationProvider.dispose();
        }
    });
}

/**
 * Set up a file watcher for the notes directory
 */
function setupFileWatcher(): void {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
        return;
    }

    const projectPath = workspaceFolders[0].uri.fsPath;

    if (fileWatcher) {
        fileWatcher.close();
    }

    fileWatcher = notesReader.watchNotes(projectPath, () => {
        console.log('Notes changed, refreshing...');
        refreshAllNotes();
    });
}

/**
 * Update decorations for a specific editor
 */
function updateDecorationsForEditor(editor: vscode.TextEditor): void {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
        return;
    }

    const projectPath = workspaceFolders[0].uri.fsPath;
    const filePath = editor.document.uri.fsPath;

    // Get relative path from workspace
    const relativePath = path.relative(projectPath, filePath);

    console.log(`Looking for notes in project: ${projectPath}`);
    console.log(`File path: ${filePath}`);
    console.log(`Relative path: ${relativePath}`);

    // Load notes for this file
    const notes = notesReader.getNotesForFile(projectPath, relativePath);

    console.log(`Found ${notes.length} notes for ${relativePath}`);

    // Update hover provider cache
    hoverProvider.updateNotes(filePath, notes);

    // Update decorations
    decorationProvider.updateDecorations(editor, notes);

    console.log(`Updated decorations for ${relativePath}: ${notes.length} notes`);
}

/**
 * Refresh all notes in all visible editors
 */
function refreshAllNotes(): void {
    vscode.window.visibleTextEditors.forEach(editor => {
        updateDecorationsForEditor(editor);
    });
}

/**
 * Show note at the current cursor position
 */
function showNoteAtCursor(): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
        vscode.window.showWarningMessage('No workspace folder open');
        return;
    }

    const projectPath = workspaceFolders[0].uri.fsPath;
    const filePath = editor.document.uri.fsPath;
    const relativePath = path.relative(projectPath, filePath);
    const cursorLine = editor.selection.active.line;

    const notes = notesReader.getNotesForFile(projectPath, relativePath);

    // Find notes at the current cursor line
    const notesAtCursor = notes.filter(note => {
        const noteLine = note.anchor.primary.line_number; // Already 0-based from tree-sitter
        return noteLine === cursorLine;
    });

    if (notesAtCursor.length === 0) {
        vscode.window.showInformationMessage('No notes at cursor position');
        return;
    }

    if (notesAtCursor.length === 1) {
        // Show the note directly
        openNotePanel(notesAtCursor[0]);
    } else {
        // Multiple notes, show quick pick
        const items = notesAtCursor.map(note => ({
            label: `${note.author}: ${note.content.substring(0, 50)}${note.content.length > 50 ? '...' : ''}`,
            description: `Created: ${new Date(note.created_at * 1000).toLocaleString()}`,
            note: note
        }));

        vscode.window.showQuickPick(items, {
            placeHolder: 'Multiple notes at this location'
        }).then(selected => {
            if (selected) {
                openNotePanel(selected.note);
            }
        });
    }
}

/**
 * Open a note in a webview panel
 */
function openNotePanel(note: Note): void {
    const panel = vscode.window.createWebviewPanel(
        'codeNote',
        `Note by ${note.author}`,
        vscode.ViewColumn.Beside,
        { enableScripts: false }
    );
    panel.webview.html = createNoteWebview(note);
}

/**
 * Show all notes for the current file in a quick pick
 */
function showNotesForCurrentFile(): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
        vscode.window.showWarningMessage('No workspace folder open');
        return;
    }

    const projectPath = workspaceFolders[0].uri.fsPath;
    const filePath = editor.document.uri.fsPath;
    const relativePath = path.relative(projectPath, filePath);

    console.log(`ShowNotes - Project: ${projectPath}`);
    console.log(`ShowNotes - Relative: ${relativePath}`);

    const notes = notesReader.getNotesForFile(projectPath, relativePath);

    console.log(`ShowNotes - Found ${notes.length} notes`);

    if (notes.length === 0) {
        vscode.window.showInformationMessage('No notes found for this file');
        return;
    }

    // Create quick pick items
    const items = notes.map(note => {
        if (!note || !note.anchor || !note.anchor.primary) {
            console.error('Invalid note structure:', note);
            return null;
        }
        return {
            label: `Line ${note.anchor.primary.line_number}: ${note.author}`,
            description: note.content.substring(0, 100) + (note.content.length > 100 ? '...' : ''),
            detail: `Created: ${new Date(note.created_at * 1000).toLocaleString()}`,
            note: note
        };
    }).filter(item => item !== null) as Array<{label: string, description: string, detail: string, note: Note}>;

    vscode.window.showQuickPick(items, {
        placeHolder: 'Select a note to view',
        matchOnDescription: true,
        matchOnDetail: true
    }).then(selected => {
        if (selected) {
            // Navigate to the note location
            const line = selected.note.anchor.primary.line_number; // Already 0-based from tree-sitter
            const position = new vscode.Position(line, selected.note.anchor.primary.column);
            editor.selection = new vscode.Selection(position, position);
            editor.revealRange(
                new vscode.Range(position, position),
                vscode.TextEditorRevealType.InCenter
            );

            // Show the note content
            openNotePanel(selected.note);
        }
    });
}

/**
 * Create HTML content for note webview
 */
function createNoteWebview(note: Note): string {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Note</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: var(--vscode-textLink-foreground);
            border-bottom: 1px solid var(--vscode-panel-border);
            padding-bottom: 10px;
        }
        .metadata {
            background: var(--vscode-editor-background);
            padding: 10px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .content {
            margin: 20px 0;
            white-space: pre-wrap;
        }
        .label {
            font-weight: bold;
            color: var(--vscode-textLink-foreground);
        }
    </style>
</head>
<body>
    <h1>📝 Code Note</h1>

    <div class="metadata">
        <p><span class="label">Author:</span> ${note.author}</p>
        <p><span class="label">Created:</span> ${new Date(note.created_at * 1000).toLocaleString()}</p>
        <p><span class="label">Updated:</span> ${new Date(note.updated_at * 1000).toLocaleString()}</p>
        <p><span class="label">File:</span> ${note.anchor.primary.file_path}</p>
        <p><span class="label">Line:</span> ${note.anchor.primary.line_number}</p>
        <p><span class="label">Node Kind:</span> ${note.anchor.primary.node_kind}</p>
        ${Object.keys(note.metadata).length > 0 ? `
            <p><span class="label">Metadata:</span></p>
            <pre>${JSON.stringify(note.metadata, null, 2)}</pre>
        ` : ''}
    </div>

    <div class="content">
        <h2>Content</h2>
        <p>${note.content}</p>
    </div>
</body>
</html>
    `;
}

export function deactivate() {
    if (fileWatcher) {
        fileWatcher.close();
    }
    if (decorationProvider) {
        decorationProvider.dispose();
    }
}
