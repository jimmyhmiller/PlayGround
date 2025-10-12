import * as vscode from 'vscode';
import { Note } from './types';

export class NotesHoverProvider implements vscode.HoverProvider {
    private notesCache: Map<string, Note[]> = new Map();

    /**
     * Update the notes cache for a file
     */
    public updateNotes(filePath: string, notes: Note[]): void {
        this.notesCache.set(filePath, notes);
    }

    /**
     * Provide hover information for a position in a document
     */
    public provideHover(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): vscode.ProviderResult<vscode.Hover> {
        const notes = this.notesCache.get(document.uri.fsPath);
        if (!notes || notes.length === 0) {
            return null;
        }

        // Find notes that match the current position
        const matchingNotes = notes.filter(note => {
            const line = note.anchor.primary.line_number; // Already 0-based from tree-sitter
            return line === position.line;
        });

        if (matchingNotes.length === 0) {
            return null;
        }

        // Create hover content
        const contents: vscode.MarkdownString[] = matchingNotes.map(note =>
            this.createHoverContent(note)
        );

        return new vscode.Hover(contents);
    }

    /**
     * Create formatted hover content for a note
     */
    private createHoverContent(note: Note): vscode.MarkdownString {
        const markdown = new vscode.MarkdownString();
        markdown.isTrusted = true;
        markdown.supportHtml = true;

        markdown.appendMarkdown(`### 📝 Note by ${note.author}\n\n`);
        markdown.appendMarkdown(note.content);
        markdown.appendMarkdown(`\n\n---\n\n`);
        markdown.appendMarkdown(`*Created: ${new Date(note.created_at * 1000).toLocaleString()}*`);

        if (note.metadata && Object.keys(note.metadata).length > 0) {
            markdown.appendMarkdown(`\n\n**Tags/Metadata:**\n\n`);
            for (const [key, value] of Object.entries(note.metadata)) {
                markdown.appendMarkdown(`- **${key}:** ${JSON.stringify(value)}\n`);
            }
        }

        return markdown;
    }

    /**
     * Clear the notes cache
     */
    public clearCache(): void {
        this.notesCache.clear();
    }
}
