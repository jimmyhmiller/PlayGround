use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::anchor::NoteAnchor;

/// A note attached to code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    /// Unique identifier for this note
    pub id: Uuid,

    /// The actual note content (markdown supported)
    pub content: String,

    /// Who created this note (can be a person, AI agent, etc.)
    pub author: String,

    /// When this note was created (Unix timestamp)
    pub created_at: i64,

    /// When this note was last updated
    pub updated_at: i64,

    /// Where this note is anchored in the code
    pub anchor: NoteAnchor,

    /// Open-ended metadata for extensions
    /// Examples: tags, priority, trail_id, audience, etc.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Whether this note is orphaned (can't find its anchor anymore)
    pub is_orphaned: bool,
}

impl Note {
    pub fn new(content: String, author: String, anchor: NoteAnchor) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        Self {
            id: Uuid::new_v4(),
            content,
            author,
            created_at: now,
            updated_at: now,
            anchor,
            metadata: HashMap::new(),
            is_orphaned: false,
        }
    }

    /// Add or update metadata
    pub fn set_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
        self.update_timestamp();
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Update the note content
    pub fn update_content(&mut self, content: String) {
        self.content = content;
        self.update_timestamp();
    }

    /// Mark as orphaned
    pub fn mark_orphaned(&mut self) {
        self.is_orphaned = true;
        self.update_timestamp();
    }

    fn update_timestamp(&mut self) {
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
    }
}

/// A collection of notes with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteCollection {
    /// Name of this collection
    pub name: String,

    /// Description of this collection
    pub description: Option<String>,

    /// All notes in this collection
    pub notes: Vec<Note>,

    /// Metadata about the collection
    pub metadata: HashMap<String, serde_json::Value>,
}

impl NoteCollection {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            notes: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_note(&mut self, note: Note) {
        self.notes.push(note);
    }

    pub fn remove_note(&mut self, note_id: Uuid) -> Option<Note> {
        if let Some(pos) = self.notes.iter().position(|n| n.id == note_id) {
            Some(self.notes.remove(pos))
        } else {
            None
        }
    }

    pub fn get_note(&self, note_id: Uuid) -> Option<&Note> {
        self.notes.iter().find(|n| n.id == note_id)
    }

    pub fn get_note_mut(&mut self, note_id: Uuid) -> Option<&mut Note> {
        self.notes.iter_mut().find(|n| n.id == note_id)
    }

    /// Get all notes for a specific file
    pub fn notes_for_file(&self, file_path: &str) -> Vec<&Note> {
        self.notes
            .iter()
            .filter(|n| n.anchor.primary.file_path == file_path)
            .collect()
    }

    /// Get all orphaned notes
    pub fn orphaned_notes(&self) -> Vec<&Note> {
        self.notes.iter().filter(|n| n.is_orphaned).collect()
    }
}
