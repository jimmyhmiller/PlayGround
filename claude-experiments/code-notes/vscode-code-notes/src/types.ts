// Types for code notes

export interface CodeAnchor {
    file_path: string;
    line_number: number;
    column: number;
    node_kind: string;
    node_text: string;
    ast_path: Array<[string, number]>;
    context: string[];
}

export interface NoteAnchor {
    primary: CodeAnchor;
    alternatives: CodeAnchor[];
}

export interface Note {
    id: string;
    content: string;
    author: string;
    created_at: number;  // Unix timestamp in seconds
    updated_at: number;  // Unix timestamp in seconds
    anchor: NoteAnchor;
    git_commit: string;
    metadata: Record<string, any>;
}

export interface NoteCollection {
    name: string;
    description: string;
    notes: string[]; // Note IDs
}

export interface ProjectIndexEntry {
    canonical_path: string;
    storage_dir: string;
}

export type ProjectIndex = ProjectIndexEntry[];

export interface FileNotes {
    [filePath: string]: Note[];
}
