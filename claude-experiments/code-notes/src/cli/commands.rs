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
        /// File path (injects all notes for this file)
        file: PathBuf,

        /// Collection name (default: "default")
        #[arg(long, default_value = "default")]
        collection: String,

        /// Output file (default: overwrites input file)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Remove all inline note comments from a source file
    RemoveInline {
        /// File path
        file: PathBuf,

        /// Output file (default: overwrites input file)
        #[arg(short, long)]
        output: Option<PathBuf>,
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
