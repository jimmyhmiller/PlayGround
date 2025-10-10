use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::models::{Note, NoteCollection};

const GLOBAL_NOTES_DIR: &str = ".code-notes";
const COLLECTIONS_FILE: &str = "collections.json";
const NOTES_SUBDIR: &str = "notes";
const PROJECT_INDEX_FILE: &str = "project_index.json";

/// Metadata about a project stored in the global index
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ProjectMetadata {
    /// Canonical path to the project
    canonical_path: String,
    /// Directory name used for this project (unique)
    storage_dir: String,
}

/// File-based storage for notes in ~/.code-notes/
pub struct NoteStorage {
    /// Path to this project's storage directory
    project_path: PathBuf,
    /// Canonical path to the repository root
    repo_canonical_path: String,
}

impl NoteStorage {
    /// Create a new storage instance for the given repository
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Result<Self> {
        let repo_path = repo_root.as_ref();
        let repo_canonical = repo_path.canonicalize()?;
        let repo_canonical_str = repo_canonical.to_string_lossy().to_string();

        // Get or create the global notes directory
        let global_notes_dir = Self::global_notes_dir()?;

        // Get or create unique project directory
        let storage_dir = Self::get_or_create_project_dir(&global_notes_dir, &repo_canonical_str)?;
        let project_path = global_notes_dir.join(&storage_dir);

        Ok(Self {
            project_path,
            repo_canonical_path: repo_canonical_str,
        })
    }

    /// Get the global ~/.code-notes directory
    fn global_notes_dir() -> Result<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| anyhow!("Could not determine home directory"))?;

        Ok(PathBuf::from(home).join(GLOBAL_NOTES_DIR))
    }

    /// Get or create a unique project directory with smart conflict resolution
    fn get_or_create_project_dir(global_dir: &Path, canonical_path: &str) -> Result<String> {
        // Load or create project index
        let index_path = global_dir.join(PROJECT_INDEX_FILE);
        let mut projects: Vec<ProjectMetadata> = if index_path.exists() {
            let content = fs::read_to_string(&index_path)?;
            serde_json::from_str(&content)?
        } else {
            Vec::new()
        };

        // Check if this project already has a directory
        if let Some(metadata) = projects.iter().find(|p| p.canonical_path == canonical_path) {
            return Ok(metadata.storage_dir.clone());
        }

        // Generate a unique directory name
        let path = Path::new(canonical_path);
        let project_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown-project");

        let storage_dir = Self::find_unique_dir_name(project_name, canonical_path, &projects);

        // Add to index
        projects.push(ProjectMetadata {
            canonical_path: canonical_path.to_string(),
            storage_dir: storage_dir.clone(),
        });

        // Save index
        fs::create_dir_all(global_dir)?;
        let json = serde_json::to_string_pretty(&projects)?;
        fs::write(index_path, json)?;

        Ok(storage_dir)
    }

    /// Find a unique directory name, adding parent directories only if needed
    fn find_unique_dir_name(base_name: &str, canonical_path: &str, existing_projects: &[ProjectMetadata]) -> String {
        // Start with just the project name
        let existing_dirs: Vec<&str> = existing_projects.iter().map(|p| p.storage_dir.as_str()).collect();

        if !existing_dirs.contains(&base_name) {
            return base_name.to_string();
        }

        // If there's a conflict, start adding parent directories
        let path = Path::new(canonical_path);
        let mut components: Vec<&str> = path
            .components()
            .filter_map(|c| {
                if let std::path::Component::Normal(s) = c {
                    s.to_str()
                } else {
                    None
                }
            })
            .collect();

        components.reverse(); // Start from the project name and work backwards

        // Try adding parent directories one at a time
        for i in 2..=components.len() {
            let parts: Vec<&str> = components.iter().take(i).copied().collect();
            let mut parts_reversed = parts.clone();
            parts_reversed.reverse();
            let candidate = parts_reversed.join("_");

            if !existing_dirs.contains(&candidate.as_str()) {
                return candidate;
            }
        }

        // If we still have conflicts, add a hash suffix
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        canonical_path.hash(&mut hasher);
        let hash = hasher.finish();

        format!("{}_{:x}", base_name, hash)
    }

    /// Initialize storage (create directories if needed)
    pub fn initialize(&self) -> Result<()> {
        fs::create_dir_all(&self.project_path)?;
        fs::create_dir_all(self.project_path.join(NOTES_SUBDIR))?;

        // Create empty collections file if it doesn't exist
        let collections_path = self.project_path.join(COLLECTIONS_FILE);
        if !collections_path.exists() {
            fs::write(collections_path, "[]")?;
        }

        Ok(())
    }

    /// Check if storage is initialized
    pub fn is_initialized(&self) -> bool {
        self.project_path.exists() && self.project_path.join(COLLECTIONS_FILE).exists()
    }

    /// Save a note collection
    pub fn save_collection(&self, collection: &NoteCollection) -> Result<()> {
        // Save individual notes
        for note in &collection.notes {
            self.save_note(note)?;
        }

        // Update collections index
        let mut collections = self.load_all_collections()?;

        // Remove existing collection with same name
        collections.retain(|c| c.name != collection.name);

        // Add this collection (without notes, as they're stored separately)
        let mut collection_meta = collection.clone();
        collection_meta.notes = Vec::new();
        collections.push(collection_meta);

        let collections_path = self.project_path.join(COLLECTIONS_FILE);
        let json = serde_json::to_string_pretty(&collections)?;
        fs::write(collections_path, json)?;

        Ok(())
    }

    /// Load a note collection by name
    pub fn load_collection(&self, name: &str) -> Result<NoteCollection> {
        let collections = self.load_all_collections()?;
        let mut collection = collections
            .into_iter()
            .find(|c| c.name == name)
            .ok_or_else(|| anyhow!("Collection not found: {}", name))?;

        // Load all notes for this collection
        let notes_dir = self.project_path.join(NOTES_SUBDIR);
        for entry in fs::read_dir(notes_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(note) = self.load_note_from_path(&entry.path()) {
                    // Filter notes that belong to this collection
                    // For now, we include all notes. In the future, we could add collection metadata to notes
                    collection.add_note(note);
                }
            }
        }

        Ok(collection)
    }

    /// Load all collection metadata (without notes)
    pub fn load_all_collections(&self) -> Result<Vec<NoteCollection>> {
        let collections_path = self.project_path.join(COLLECTIONS_FILE);

        if !collections_path.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(collections_path)?;
        let collections: Vec<NoteCollection> = serde_json::from_str(&content)?;
        Ok(collections)
    }

    /// List all collection names
    pub fn list_collections(&self) -> Result<Vec<String>> {
        let collections = self.load_all_collections()?;
        Ok(collections.into_iter().map(|c| c.name).collect())
    }

    /// Delete a collection
    pub fn delete_collection(&self, name: &str) -> Result<()> {
        let mut collections = self.load_all_collections()?;
        collections.retain(|c| c.name != name);

        let collections_path = self.project_path.join(COLLECTIONS_FILE);
        let json = serde_json::to_string_pretty(&collections)?;
        fs::write(collections_path, json)?;

        // Note: This doesn't delete the individual note files
        // They can be cleaned up with a separate garbage collection operation

        Ok(())
    }

    /// Save a single note
    fn save_note(&self, note: &Note) -> Result<()> {
        let note_path = self.project_path.join(NOTES_SUBDIR).join(format!("{}.json", note.id));
        let json = serde_json::to_string_pretty(note)?;
        fs::write(note_path, json)?;
        Ok(())
    }

    /// Load a single note by ID
    pub fn load_note(&self, id: Uuid) -> Result<Note> {
        let note_path = self.project_path.join(NOTES_SUBDIR).join(format!("{}.json", id));
        self.load_note_from_path(&note_path)
    }

    /// Load a note from a specific path
    fn load_note_from_path(&self, path: &Path) -> Result<Note> {
        let content = fs::read_to_string(path)?;
        let note: Note = serde_json::from_str(&content)?;
        Ok(note)
    }

    /// Load all notes
    pub fn load_all_notes(&self) -> Result<Vec<Note>> {
        let notes_dir = self.project_path.join(NOTES_SUBDIR);
        let mut notes = Vec::new();

        if !notes_dir.exists() {
            return Ok(notes);
        }

        for entry in fs::read_dir(notes_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(note) = self.load_note_from_path(&entry.path()) {
                    notes.push(note);
                }
            }
        }

        Ok(notes)
    }

    /// Export a collection to a bundle file
    pub fn export_collection(&self, name: &str, output_path: &Path) -> Result<()> {
        let collection = self.load_collection(name)?;
        let json = serde_json::to_string_pretty(&collection)?;
        fs::write(output_path, json)?;
        Ok(())
    }

    /// Import a collection from a bundle file
    pub fn import_collection(&self, bundle_path: &Path) -> Result<String> {
        let content = fs::read_to_string(bundle_path)?;
        let collection: NoteCollection = serde_json::from_str(&content)?;
        let name = collection.name.clone();
        self.save_collection(&collection)?;
        Ok(name)
    }

    /// Get the storage root path for this project
    pub fn root_path(&self) -> &Path {
        &self.project_path
    }

    /// Get the canonical repository path
    pub fn repo_path(&self) -> &str {
        &self.repo_canonical_path
    }
}
