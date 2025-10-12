import * as vscode from 'vscode';
import { Note } from './types';

export class DecorationProvider {
    private decorationType: vscode.TextEditorDecorationType;

    constructor(iconColor: string = '#4A9EFF') {
        // Create a decoration type for the gutter icon
        this.decorationType = vscode.window.createTextEditorDecorationType({
            gutterIconPath: this.createGutterIcon(iconColor),
            gutterIconSize: 'contain',
            isWholeLine: false,
            backgroundColor: new vscode.ThemeColor('editor.wordHighlightBackground'),
            overviewRulerColor: iconColor,
            overviewRulerLane: vscode.OverviewRulerLane.Right
        });
    }

    /**
     * Create an SVG icon for the gutter
     */
    private createGutterIcon(color: string): vscode.Uri {
        // Create a simple blue dot
        const svg = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
                <circle cx="8" cy="8" r="4" fill="${color}"/>
            </svg>
        `.trim();

        const dataUri = `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
        return vscode.Uri.parse(dataUri);
    }

    /**
     * Update decorations for a text editor
     */
    public updateDecorations(editor: vscode.TextEditor, notes: Note[]): void {
        if (!editor || !editor.document) {
            return;
        }

        const decorations: vscode.DecorationOptions[] = [];

        for (const note of notes) {
            try {
                const line = note.anchor.primary.line_number; // Tree-sitter uses 0-based line numbers
                const column = note.anchor.primary.column;

                // Validate line number
                if (line < 0 || line >= editor.document.lineCount) {
                    console.warn(`Note line ${note.anchor.primary.line_number} is out of range for document with ${editor.document.lineCount} lines`);
                    continue;
                }

                // Get the actual line to validate column
                const lineText = editor.document.lineAt(line).text;
                const safeColumn = Math.min(column, lineText.length);
                const endColumn = Math.min(column + note.anchor.primary.node_text.length, lineText.length);

                const range = new vscode.Range(
                    new vscode.Position(line, safeColumn),
                    new vscode.Position(line, endColumn)
                );

                const decoration: vscode.DecorationOptions = {
                    range,
                    hoverMessage: this.createHoverMessage(note)
                };

                decorations.push(decoration);
            } catch (error) {
                console.error('Error creating decoration for note:', note, error);
            }
        }

        editor.setDecorations(this.decorationType, decorations);
    }

    /**
     * Create a formatted hover message for a note
     */
    private createHoverMessage(note: Note): vscode.MarkdownString {
        const markdown = new vscode.MarkdownString();
        markdown.isTrusted = true;
        markdown.supportHtml = true;

        markdown.appendMarkdown(`### 📝 Code Note\n\n`);
        markdown.appendMarkdown(`**Author:** ${note.author}\n\n`);
        markdown.appendMarkdown(`**Created:** ${new Date(note.created_at * 1000).toLocaleString()}\n\n`);

        if (note.metadata && Object.keys(note.metadata).length > 0) {
            markdown.appendMarkdown(`**Metadata:** \`${JSON.stringify(note.metadata)}\`\n\n`);
        }

        markdown.appendMarkdown(`---\n\n`);
        markdown.appendMarkdown(note.content);

        return markdown;
    }

    /**
     * Clear all decorations
     */
    public clearDecorations(editor: vscode.TextEditor): void {
        if (editor) {
            editor.setDecorations(this.decorationType, []);
        }
    }

    /**
     * Dispose of the decoration type
     */
    public dispose(): void {
        this.decorationType.dispose();
    }
}
