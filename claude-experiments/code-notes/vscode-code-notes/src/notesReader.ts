import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { Note, ProjectIndex, FileNotes } from './types';

export class NotesReader {
    private storageDir: string;
    private projectIndex: ProjectIndex | null = null;

    constructor(storageDir: string = '~/.code-notes') {
        // Expand tilde to home directory
        this.storageDir = storageDir.replace(/^~/, os.homedir());
    }

    /**
     * Load the project index to find where notes are stored
     */
    private loadProjectIndex(): ProjectIndex {
        if (this.projectIndex) {
            return this.projectIndex;
        }

        const indexPath = path.join(this.storageDir, 'project_index.json');
        try {
            const content = fs.readFileSync(indexPath, 'utf-8');
            this.projectIndex = JSON.parse(content);
            console.log(`Loaded project index with ${this.projectIndex!.length} entries`);
            return this.projectIndex!;
        } catch (error) {
            console.error('Failed to load project index:', error);
            return [];
        }
    }

    /**
     * Find the storage directory for a given project path
     */
    private findProjectStorage(projectPath: string): string | null {
        const index = this.loadProjectIndex();

        // Normalize the project path
        const normalizedPath = path.resolve(projectPath);
        console.log(`Looking for project: ${normalizedPath}`);

        // Look for exact match or parent directory match
        for (const entry of index) {
            console.log(`Checking entry: ${entry.canonical_path} -> ${entry.storage_dir}`);
            if (normalizedPath === entry.canonical_path ||
                normalizedPath.startsWith(entry.canonical_path + path.sep)) {
                const storagePath = path.join(this.storageDir, entry.storage_dir);
                console.log(`Found match! Storage path: ${storagePath}`);
                return storagePath;
            }
        }

        // Try just the project name as fallback
        const projectName = path.basename(normalizedPath);
        const fallbackPath = path.join(this.storageDir, projectName);
        if (fs.existsSync(fallbackPath)) {
            return fallbackPath;
        }

        return null;
    }

    /**
     * Load all notes for a given project
     */
    public loadNotesForProject(projectPath: string): Note[] {
        const storagePath = this.findProjectStorage(projectPath);
        if (!storagePath) {
            console.log(`No notes storage found for project: ${projectPath}`);
            return [];
        }

        const notesDir = path.join(storagePath, 'notes');
        if (!fs.existsSync(notesDir)) {
            console.log(`Notes directory not found: ${notesDir}`);
            return [];
        }

        const notes: Note[] = [];
        try {
            const files = fs.readdirSync(notesDir);
            for (const file of files) {
                if (file.endsWith('.json')) {
                    const notePath = path.join(notesDir, file);
                    try {
                        const content = fs.readFileSync(notePath, 'utf-8');
                        const note = JSON.parse(content) as Note;
                        notes.push(note);
                    } catch (error) {
                        console.error(`Failed to load note ${file}:`, error);
                    }
                }
            }
        } catch (error) {
            console.error(`Failed to read notes directory: ${notesDir}`, error);
        }

        return notes;
    }

    /**
     * Group notes by file path for easier lookup
     */
    public groupNotesByFile(notes: Note[]): FileNotes {
        const fileNotes: FileNotes = {};

        for (const note of notes) {
            const filePath = note.anchor.primary.file_path;
            if (!fileNotes[filePath]) {
                fileNotes[filePath] = [];
            }
            fileNotes[filePath].push(note);
        }

        return fileNotes;
    }

    /**
     * Get notes for a specific file
     */
    public getNotesForFile(projectPath: string, filePath: string): Note[] {
        const allNotes = this.loadNotesForProject(projectPath);
        return allNotes.filter(note => {
            const noteFilePath = note.anchor.primary.file_path;
            // Handle both absolute and relative paths
            return noteFilePath === filePath ||
                   noteFilePath.endsWith(filePath) ||
                   filePath.endsWith(noteFilePath);
        });
    }

    /**
     * Watch for changes to notes
     */
    public watchNotes(projectPath: string, callback: () => void): fs.FSWatcher | null {
        const storagePath = this.findProjectStorage(projectPath);
        if (!storagePath) {
            return null;
        }

        const notesDir = path.join(storagePath, 'notes');
        if (!fs.existsSync(notesDir)) {
            return null;
        }

        try {
            return fs.watch(notesDir, { recursive: false }, (eventType, filename) => {
                if (filename && filename.endsWith('.json')) {
                    callback();
                }
            });
        } catch (error) {
            console.error('Failed to watch notes directory:', error);
            return null;
        }
    }
}
