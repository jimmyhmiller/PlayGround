use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

use crate::git::{GitRepo, NoteMigrator};
use crate::models::{Note, NoteCollection};
use crate::parsers::{AnchorBuilder, CodeParser, LanguageRegistry, LanguageInstaller, GrammarSource, find_node_at_position};
use crate::storage::NoteStorage;
use super::lang_commands::*;

#[derive(Parser)]
#[command(name = "code-notes")]
#[command(about = "A layered annotation system for code that persists across commits", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Initialize code-notes in the current repository
    Init,

    /// Add a new note to code
    Add {
        /// File path
        #[arg(short, long)]
        file: PathBuf,

        /// Line number (1-indexed)
        #[arg(short, long)]
        line: usize,

        /// Column number (0-indexed)
        #[arg(long, default_value = "0")]
        column: usize,

        /// Note content
        #[arg(short, long)]
        content: String,

        /// Author name
        #[arg(short, long)]
        author: String,

        /// Collection name (default: "default")
        #[arg(long, default_value = "default")]
        collection: String,
    },

    /// List notes for a file
    List {
        /// File path (optional, lists all notes if not provided)
        file: Option<PathBuf>,

        /// Collection name (optional, lists from all collections if not provided)
        #[arg(long)]
        collection: Option<String>,

        /// Include deleted notes in the list
        #[arg(long, default_value = "false")]
        include_deleted: bool,
    },

    /// View a specific note
    View {
        /// Note ID
        id: String,
    },

    /// Update a note
    Update {
        /// Note ID
        id: String,

        /// New content
        #[arg(short, long)]
        content: String,
    },

    /// Delete a note (soft delete - marks as deleted)
    Delete {
        /// Note ID
        id: String,

        /// Collection name
        #[arg(long, default_value = "default")]
        collection: String,
    },

    /// Permanently delete a note (hard delete - removes from disk)
    HardDelete {
        /// Note ID
        id: String,

        /// Collection name
        #[arg(long, default_value = "default")]
        collection: String,
    },

    /// Migrate notes to the current commit
    Migrate {
        /// Collection name (default: "default")
        #[arg(long, default_value = "default")]
        collection: String,
    },

    /// List all collections
    Collections,

    /// Create a new collection
    CreateCollection {
        /// Collection name
        name: String,

        /// Description
        #[arg(short, long)]
        description: Option<String>,
    },

    /// Export a collection to a file
    Export {
        /// Collection name
        collection: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Import a collection from a file
    Import {
        /// Bundle file path
        bundle: PathBuf,
    },

    /// Show orphaned notes
    Orphaned {
        /// Collection name (default: "default")
        #[arg(long, default_value = "default")]
        collection: String,
    },

    /// Language management commands
    #[command(subcommand)]
    Lang(LangCommands),

    /// Inject notes as inline comments into source files
    Inject {
        /// File path (optional, injects all files if not provided)
        file: Option<PathBuf>,

        /// Collection name (optional, injects all collections if not provided)
        #[arg(long)]
        collection: Option<String>,

        /// Output file (default: overwrites input file)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Remove all inline note comments from source files
    RemoveInline {
        /// File path (optional, removes from all files if not provided)
        file: Option<PathBuf>,

        /// Output file (default: overwrites input file)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Extract notes from inline comments in source files
    Extract {
        /// File path (optional, extracts from all files if not provided)
        file: Option<PathBuf>,

        /// Collection name (optional, searches all collections if not provided)
        #[arg(long)]
        collection: Option<String>,
    },

    /// Capture @note: markers in code and convert them to proper notes
    Capture {
        /// File path
        file: PathBuf,

        /// Author name
        #[arg(short, long)]
        author: String,

        /// Collection name (default: "default")
        #[arg(long, default_value = "default")]
        collection: String,
    },
}

#[derive(Subcommand)]
pub enum LangCommands {
    /// Install a tree-sitter language grammar
    Install {
        /// Language name (e.g., "rust", "python", "go")
        language: String,
    },

    /// Uninstall a language grammar
    Uninstall {
        /// Language name
        language: String,
    },

    /// List installed language grammars
    ListInstalled,

    /// List all available language grammars
    ListAvailable,

    /// Show information about a language
    Info {
        /// Language name
        language: String,
    },
}

pub fn execute_command(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Init => cmd_init(),
        Commands::Add {
            file,
            line,
            column,
            content,
            author,
            collection,
        } => cmd_add(file, line, column, content, author, collection),
        Commands::List { file, collection, include_deleted } => cmd_list(file, collection, include_deleted),
        Commands::View { id } => cmd_view(id),
        Commands::Update { id, content } => cmd_update(id, content),
        Commands::Delete { id, collection } => cmd_delete(id, collection),
        Commands::HardDelete { id, collection } => cmd_hard_delete(id, collection),
        Commands::Migrate { collection } => cmd_migrate(collection),
        Commands::Collections => cmd_collections(),
        Commands::CreateCollection { name, description } => cmd_create_collection(name, description),
        Commands::Export { collection, output } => cmd_export(collection, output),
        Commands::Import { bundle } => cmd_import(bundle),
        Commands::Orphaned { collection } => cmd_orphaned(collection),
        Commands::Lang(lang_cmd) => execute_lang_command(lang_cmd),
        Commands::Inject { file, collection, output } => cmd_inject(file, collection, output),
        Commands::RemoveInline { file, output } => cmd_remove_inline(file, output),
        Commands::Extract { file, collection } => cmd_extract(file, collection),
        Commands::Capture { file, author, collection } => cmd_capture(file, author, collection),
    }
}

fn execute_lang_command(cmd: LangCommands) -> Result<()> {
    match cmd {
        LangCommands::Install { language } => cmd_lang_install(language),
        LangCommands::Uninstall { language } => cmd_lang_uninstall(language),
        LangCommands::ListInstalled => cmd_lang_list_installed(),
        LangCommands::ListAvailable => cmd_lang_list_available(),
        LangCommands::Info { language } => cmd_lang_info(language),
    }
}

/// Resolve a file path by checking if it exists
/// Tries multiple strategies:
/// 1. If file exists relative to current directory, use that
/// 2. If file exists relative to repo root, use that
fn resolve_file_path(repo: &GitRepo, file_path: &Path) -> Result<PathBuf> {
    let root = repo.root_path()?;
    let current_dir = std::env::current_dir()?;

    // Strategy 1: Try relative to current directory
    let abs_path = if file_path.is_absolute() {
        file_path.to_path_buf()
    } else {
        current_dir.join(file_path)
    };

    if abs_path.exists() {
        return Ok(abs_path);
    }

    // Strategy 2: Try relative to repo root
    let repo_relative = root.join(file_path);
    if repo_relative.exists() {
        return Ok(repo_relative);
    }

    Err(anyhow!("File '{}' not found", file_path.display()))
}

/// Auto-install a language if it's not already installed
/// Returns true if installation was attempted (whether successful or not)
fn auto_install_language(extension: &str, registry: &LanguageRegistry) -> bool {
    // Try to get language name from extension via registry first
    if let Ok(lang_name) = registry.language_from_extension(extension) {
        // Check if already installed
        if registry.is_installed(&lang_name).unwrap_or(false) {
            return false; // Already installed, no action needed
        }

        // Check if this language is available for installation
        if GrammarSource::find_by_name(&lang_name).is_some() {
            println!("Language '{}' not installed. Installing automatically...", lang_name);

            if let Ok(mut installer) = LanguageInstaller::new() {
                match installer.install(&lang_name) {
                    Ok(_) => {
                        println!("✓ Successfully installed {}", lang_name);
                        return true;
                    }
                    Err(e) => {
                        eprintln!("⚠ Failed to auto-install {}: {}", lang_name, e);
                        eprintln!("  You can install it manually with: code-notes lang install {}", lang_name);
                        return true;
                    }
                }
            }
        }
    }

    false
}

fn cmd_init() -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    if storage.is_initialized() {
        println!("Code-notes is already initialized for this repository");
        println!("Repository: {}", storage.repo_path());
        println!("Storage location: {}", storage.root_path().display());
        return Ok(());
    }

    storage.initialize()?;
    println!("Initialized code-notes");
    println!("Repository: {}", storage.repo_path());
    println!("Storage location: {}", storage.root_path().display());
    Ok(())
}

fn cmd_add(
    file: PathBuf,
    line: usize,
    column: usize,
    content: String,
    author: String,
    collection_name: String,
) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    if !storage.is_initialized() {
        return Err(anyhow!("Code-notes not initialized. Run 'code-notes init' first"));
    }

    // Resolve the file path
    let resolved_file = resolve_file_path(&repo, &file)?;

    // Read file content
    let file_content = std::fs::read_to_string(&resolved_file)?;

    // Detect language
    let extension = resolved_file
        .extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("Could not determine file extension"))?;

    // Create language registry
    let mut registry = LanguageRegistry::new()?;
    registry.initialize()?;

    // Auto-install language if needed
    if auto_install_language(extension, &registry) {
        // Re-initialize registry to pick up the newly installed language
        registry = LanguageRegistry::new()?;
        registry.initialize()?;
    }

    // Create parser
    let mut parser = CodeParser::from_extension(extension, &mut registry)?;
    let tree = parser.parse(&file_content)?;

    // Find node at position (convert 1-indexed line to 0-indexed)
    let node = find_node_at_position(&tree, &file_content, line - 1, column)
        .ok_or_else(|| anyhow!("No code found at {}:{}", line, column))?;

    // Get relative path
    let rel_path = repo.relative_path(&resolved_file)?;

    // Build anchor
    let builder = AnchorBuilder::new(&file_content, rel_path);
    let commit_hash = repo.current_commit_hash()?;
    let anchor = builder.build_note_anchor(node, commit_hash)?;

    // Create note
    let note = Note::new(content.clone(), author, anchor, collection_name.clone());

    // Load or create collection
    let mut collection = storage
        .load_collection(&collection_name)
        .unwrap_or_else(|_| NoteCollection::new(collection_name.clone()));

    collection.add_note(note.clone());
    storage.save_collection(&collection)?;

    println!("Added note {} to collection '{}'", note.id, collection_name);
    println!("Content: {}", content);
    Ok(())
}

fn cmd_list(file: Option<PathBuf>, collection_name: Option<String>, include_deleted: bool) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    // Collect all notes from specified collection or all collections
    let mut all_notes = Vec::new();

    if let Some(name) = collection_name {
        // Single collection
        let collection = storage.load_collection(&name)?;
        all_notes.extend(collection.notes);
    } else {
        // All collections
        let collection_names = storage.list_collections()?;
        for name in collection_names {
            if let Ok(collection) = storage.load_collection(&name) {
                all_notes.extend(collection.notes);
            }
        }
    }

    // Filter out deleted notes unless include_deleted is true
    if !include_deleted {
        all_notes.retain(|note| !note.deleted);
    }

    // Filter by file if specified
    let notes: Vec<&Note> = if let Some(file_path) = file {
        // Match against the file paths stored in notes
        // Try exact match first, then match on filename
        let file_str = file_path.to_str().unwrap_or("");
        all_notes.iter().filter(|n| {
            let note_path = &n.anchor.primary.file_path;
            // Exact match
            if note_path == file_str {
                return true;
            }
            // Match on filename
            if let Some(filename) = file_path.file_name().and_then(|f| f.to_str())
                && note_path.ends_with(filename) {
                    return true;
                }
            false
        }).collect()
    } else {
        all_notes.iter().collect()
    };

    if notes.is_empty() {
        println!("No notes found");
        return Ok(());
    }

    println!("Found {} note(s):", notes.len());
    for note in notes {
        println!("\nID: {}", note.id);
        println!("File: {}:{}", note.anchor.primary.file_path, note.anchor.primary.line_number + 1);
        println!("Author: {}", note.author);
        println!("Content: {}", note.content);
        if note.deleted {
            println!("🗑️  DELETED");
        }
        if note.is_orphaned {
            println!("⚠️  ORPHANED");
        }
    }

    Ok(())
}

fn cmd_view(id: String) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let note_id = id.parse()?;
    let note = storage.load_note(note_id)?;

    println!("Note ID: {}", note.id);
    println!("File: {}:{}", note.anchor.primary.file_path, note.anchor.primary.line_number + 1);
    println!("Author: {}", note.author);
    println!("Created: {}", note.created_at);
    println!("Updated: {}", note.updated_at);
    println!("Commit: {}", note.anchor.commit_hash);
    println!("\nContent:\n{}", note.content);

    if note.is_orphaned {
        println!("\n⚠️  This note is ORPHANED");
    }

    if !note.anchor.migration_history.is_empty() {
        println!("\nMigration History:");
        for record in &note.anchor.migration_history {
            println!("  {} -> {}: {}",
                &record.from_commit[..8],
                &record.to_commit[..8],
                if record.success { "✓" } else { "✗" }
            );
        }
    }

    Ok(())
}

fn cmd_update(id: String, content: String) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let note_id = id.parse()?;
    let mut note = storage.load_note(note_id)?;

    note.update_content(content);

    // Load the collection this note belongs to
    let collection_name = note.collection.clone();
    let mut collection = storage.load_collection(&collection_name)?;

    // Update the note in the collection
    if let Some(existing) = collection.get_note_mut(note.id) {
        *existing = note.clone();
        storage.save_collection(&collection)?;
        println!("Updated note {}", note.id);
        return Ok(());
    }

    Err(anyhow!("Note not found in collection '{}'", collection_name))
}

fn cmd_delete(id: String, collection_name: String) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let note_id = id.parse()?;
    let mut collection = storage.load_collection(&collection_name)?;

    // Soft delete: mark the note as deleted instead of removing it
    if let Some(note) = collection.get_note_mut(note_id) {
        note.mark_deleted();
        storage.save_collection(&collection)?;
        println!("Deleted note {} (soft delete)", note_id);
        Ok(())
    } else {
        Err(anyhow!("Note not found in collection"))
    }
}

fn cmd_hard_delete(id: String, collection_name: String) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let note_id = id.parse()?;
    let mut collection = storage.load_collection(&collection_name)?;

    // Hard delete: permanently remove the note
    collection.remove_note(note_id)
        .ok_or_else(|| anyhow!("Note not found in collection"))?;

    // Delete the note file from disk
    storage.delete_note_file(note_id)?;

    storage.save_collection(&collection)?;
    println!("Permanently deleted note {}", note_id);
    Ok(())
}

fn cmd_migrate(collection_name: String) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let mut collection = storage.load_collection(&collection_name)?;
    let migrator = NoteMigrator::new(repo);

    println!("Migrating {} notes...", collection.notes.len());
    let report = migrator.migrate_collection(&mut collection.notes)?;

    storage.save_collection(&collection)?;

    println!("\nMigration complete:");
    println!("  Total: {}", report.total);
    println!("  Successful: {}", report.successful);
    println!("  Failed: {}", report.failed);
    println!("  Success rate: {:.1}%", report.success_rate() * 100.0);

    Ok(())
}

fn cmd_collections() -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let collections = storage.list_collections()?;

    if collections.is_empty() {
        println!("No collections found");
        return Ok(());
    }

    println!("Collections:");
    for name in collections {
        println!("  - {}", name);
    }

    Ok(())
}

fn cmd_create_collection(name: String, description: Option<String>) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let mut collection = NoteCollection::new(name.clone());
    collection.description = description;

    storage.save_collection(&collection)?;
    println!("Created collection '{}'", name);
    Ok(())
}

fn cmd_export(collection_name: String, output: PathBuf) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    storage.export_collection(&collection_name, &output)?;
    println!("Exported collection '{}' to {}", collection_name, output.display());
    Ok(())
}

fn cmd_import(bundle: PathBuf) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let name = storage.import_collection(&bundle)?;
    println!("Imported collection '{}'", name);
    Ok(())
}

fn cmd_orphaned(collection_name: String) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let collection = storage.load_collection(&collection_name)?;
    let orphaned = collection.orphaned_notes();

    if orphaned.is_empty() {
        println!("No orphaned notes found");
        return Ok(());
    }

    println!("Orphaned notes in collection '{}':", collection_name);
    for note in orphaned {
        println!("\nID: {}", note.id);
        println!("File: {}:{}", note.anchor.primary.file_path, note.anchor.primary.line_number + 1);
        println!("Content: {}", note.content);
    }

    Ok(())
}

fn cmd_inject(file: Option<PathBuf>, collection_name: Option<String>, output: Option<PathBuf>) -> Result<()> {
    use crate::inline;
    use std::collections::HashMap;

    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    // Determine which collections to process
    let collection_names = if let Some(name) = collection_name {
        vec![name]
    } else {
        storage.list_collections()?
    };

    // Collect all notes from selected collections
    let mut notes_by_file: HashMap<String, Vec<crate::models::Note>> = HashMap::new();

    for coll_name in &collection_names {
        if let Ok(collection) = storage.load_collection(coll_name) {
            for note in collection.notes {
                if !note.deleted {
                    let file_path = note.anchor.primary.file_path.clone();
                    notes_by_file.entry(file_path).or_insert_with(Vec::new).push(note);
                }
            }
        }
    }

    if notes_by_file.is_empty() {
        println!("No notes found to inject");
        return Ok(());
    }

    // If specific file requested, filter to just that file
    let files_to_process: Vec<String> = if let Some(file_path) = file {
        let resolved_file = resolve_file_path(&repo, &file_path)?;
        let rel_path = repo.relative_path(&resolved_file)?;

        // Find notes matching this file
        let matching_keys: Vec<_> = notes_by_file.keys()
            .filter(|k| *k == &rel_path || k.ends_with(&rel_path))
            .cloned()
            .collect();

        if matching_keys.is_empty() {
            println!("No notes found for file {}", file_path.display());
            return Ok(());
        }
        matching_keys
    } else {
        notes_by_file.keys().cloned().collect()
    };

    // Process each file
    let mut files_processed = 0;
    let mut notes_injected = 0;

    for file_key in &files_to_process {
        if let Some(file_notes) = notes_by_file.get(file_key) {
            // Resolve the file path (it's stored as relative path)
            let file_full_path = root.join(file_key);

            if !file_full_path.exists() {
                eprintln!("Warning: File {} not found, skipping", file_key);
                continue;
            }

            match inline::inject_notes(&file_full_path, file_notes) {
                Ok(injected_content) => {
                    let output_path = if let Some(ref out) = output {
                        out.clone()
                    } else {
                        file_full_path.clone()
                    };

                    std::fs::write(&output_path, injected_content)?;
                    println!("✓ Injected {} notes into {}", file_notes.len(), file_key);
                    files_processed += 1;
                    notes_injected += file_notes.len();
                }
                Err(e) => {
                    eprintln!("✗ Failed to inject notes into {}: {}", file_key, e);
                }
            }
        }
    }

    println!("\nSummary: Injected {} notes across {} files", notes_injected, files_processed);
    Ok(())
}

fn cmd_remove_inline(file: Option<PathBuf>, output: Option<PathBuf>) -> Result<()> {
    use crate::inline;
    use std::collections::HashSet;

    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    // Determine which files to process
    let files_to_process: Vec<PathBuf> = if let Some(file_path) = file {
        vec![resolve_file_path(&repo, &file_path)?]
    } else {
        // Get unique files from all notes across all collections
        let mut file_set = HashSet::new();
        let collection_names = storage.list_collections()?;

        for coll_name in &collection_names {
            if let Ok(collection) = storage.load_collection(coll_name) {
                for note in collection.notes {
                    file_set.insert(note.anchor.primary.file_path.clone());
                }
            }
        }

        // Convert to full paths
        file_set.iter()
            .map(|rel_path| root.join(rel_path))
            .filter(|p| p.exists())
            .collect()
    };

    if files_to_process.is_empty() {
        println!("No files with notes found");
        return Ok(());
    }

    let mut files_processed = 0;

    for file_path in &files_to_process {
        match inline::remove_notes(file_path) {
            Ok(cleaned_content) => {
                let output_path = if let Some(ref out) = output {
                    out.clone()
                } else {
                    file_path.clone()
                };

                // Only write and report if notes were actually removed
                let original = std::fs::read_to_string(file_path)?;
                if original != cleaned_content {
                    std::fs::write(&output_path, cleaned_content)?;
                    let rel_path = file_path.strip_prefix(&root).unwrap_or(file_path);
                    println!("✓ Removed inline notes from {}", rel_path.display());
                    files_processed += 1;
                }
            }
            Err(e) => {
                let rel_path = file_path.strip_prefix(&root).unwrap_or(file_path);
                eprintln!("✗ Failed to process {}: {}", rel_path.display(), e);
            }
        }
    }

    if files_processed > 0 {
        println!("\nSummary: Removed inline notes from {} files", files_processed);
    } else {
        println!("No inline notes found to remove");
    }
    Ok(())
}

fn cmd_extract(file: Option<PathBuf>, collection_name: Option<String>) -> Result<()> {
    use crate::inline;
    use std::collections::HashSet;

    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    // Determine which files to process
    let files_to_process: Vec<PathBuf> = if let Some(file_path) = file {
        vec![resolve_file_path(&repo, &file_path)?]
    } else {
        // Get unique files from notes in specified collections (or all collections)
        let mut file_set = HashSet::new();
        let collection_names = if let Some(name) = &collection_name {
            vec![name.clone()]
        } else {
            storage.list_collections()?
        };

        for coll_name in &collection_names {
            if let Ok(collection) = storage.load_collection(coll_name) {
                for note in collection.notes {
                    file_set.insert(note.anchor.primary.file_path.clone());
                }
            }
        }

        // Convert to full paths
        file_set.iter()
            .map(|rel_path| root.join(rel_path))
            .filter(|p| p.exists())
            .collect()
    };

    if files_to_process.is_empty() {
        println!("No files with notes found");
        return Ok(());
    }

    // Determine which collections to search for extraction
    let search_collections = if let Some(name) = collection_name {
        Some(vec![name])
    } else {
        None
    };

    let mut total_notes = 0;
    let mut files_with_notes = 0;

    for file_path in &files_to_process {
        match inline::extract_notes(file_path, &storage, search_collections.clone()) {
            Ok(notes) => {
                if !notes.is_empty() {
                    let rel_path = file_path.strip_prefix(&root).unwrap_or(file_path);
                    println!("\n=== {} ===", rel_path.display());
                    println!("Found {} note(s):", notes.len());

                    for note in &notes {
                        println!("\nID: {}", note.id);
                        println!("Collection: {}", note.collection);
                        println!("Line: {}", note.anchor.primary.line_number + 1);
                        println!("Author: {}", note.author);
                        println!("Content: {}", note.content);
                        if note.deleted {
                            println!("🗑️  DELETED");
                        }
                        if note.is_orphaned {
                            println!("⚠️  ORPHANED");
                        }
                    }

                    total_notes += notes.len();
                    files_with_notes += 1;
                }
            }
            Err(e) => {
                let rel_path = file_path.strip_prefix(&root).unwrap_or(file_path);
                eprintln!("✗ Failed to extract from {}: {}", rel_path.display(), e);
            }
        }
    }

    if total_notes > 0 {
        println!("\nSummary: Extracted {} notes from {} files", total_notes, files_with_notes);
    } else {
        println!("No inline notes found to extract");
    }
    Ok(())
}

fn cmd_capture(file: PathBuf, author: String, collection_name: String) -> Result<()> {
    use crate::inline;

    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    if !storage.is_initialized() {
        return Err(anyhow!("Code-notes not initialized. Run 'code-notes init' first"));
    }

    // Resolve the file path
    let resolved_file = resolve_file_path(&repo, &file)?;

    // Scan for @note: markers
    let captured_notes = inline::scan_for_note_markers(&resolved_file)?;

    if captured_notes.is_empty() {
        println!("No @note: markers found in {}", file.display());
        return Ok(());
    }

    println!("Found {} @note: marker(s) in {}", captured_notes.len(), file.display());

    // Read file content for tree-sitter parsing
    let file_content = std::fs::read_to_string(&resolved_file)?;

    // Detect language
    let extension = resolved_file
        .extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("Could not determine file extension"))?;

    // Create language registry
    let mut registry = LanguageRegistry::new()?;
    registry.initialize()?;

    // Auto-install language if needed
    if auto_install_language(extension, &registry) {
        registry = LanguageRegistry::new()?;
        registry.initialize()?;
    }

    // Create parser
    let mut parser = CodeParser::from_extension(extension, &mut registry)?;
    let tree = parser.parse(&file_content)?;

    // Get relative path
    let rel_path = repo.relative_path(&resolved_file)?;
    let commit_hash = repo.current_commit_hash()?;

    // Load or create collection
    let mut collection = storage
        .load_collection(&collection_name)
        .unwrap_or_else(|_| NoteCollection::new(collection_name.clone()));

    // Create notes for each captured marker
    let mut notes_with_ids = Vec::new();

    for captured in &captured_notes {
        // Find the actual code line (line after the @note: comment)
        let code_line = captured.line_number;

        // Find the first non-comment line after the note
        let lines: Vec<&str> = file_content.lines().collect();
        let mut target_line = code_line;

        // Skip past all comment lines to find the actual code
        let comment_prefix = inline::comment_prefix_for_extension(extension)?;
        while target_line < lines.len() {
            let line = lines[target_line].trim();
            if !line.starts_with(comment_prefix) && !line.is_empty() {
                break;
            }
            target_line += 1;
        }

        if target_line >= lines.len() {
            eprintln!("Warning: Could not find code for note at line {}", captured.line_number + 1);
            continue;
        }

        // Find node at the code line (0-indexed for tree-sitter)
        let node = find_node_at_position(&tree, &file_content, target_line, 0)
            .ok_or_else(|| anyhow!("No code found at line {}", target_line + 1))?;

        // Build anchor
        let builder = AnchorBuilder::new(&file_content, rel_path.clone());
        let anchor = builder.build_note_anchor(node, commit_hash.clone())?;

        // Create note with new UUID
        let note = Note::new(captured.content.clone(), author.clone(), anchor, collection_name.clone());
        let note_id = note.id;

        // Add to collection
        collection.add_note(note);

        // Store for file replacement
        notes_with_ids.push((captured.clone(), note_id));

        println!("✓ Created note {} at line {}", note_id, captured.line_number + 1);
    }

    // Save collection
    storage.save_collection(&collection)?;

    // Replace @note: markers with full format
    let updated_content = inline::replace_note_markers_with_full_notes(&resolved_file, &notes_with_ids)?;
    std::fs::write(&resolved_file, updated_content)?;

    println!("\n✓ Captured {} notes and saved to collection '{}'", notes_with_ids.len(), collection_name);
    println!("✓ Updated {} with full note markers", file.display());

    Ok(())
}
