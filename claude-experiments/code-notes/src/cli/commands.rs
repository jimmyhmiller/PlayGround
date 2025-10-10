use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::git::{GitRepo, NoteMigrator};
use crate::models::{Note, NoteCollection};
use crate::parsers::{AnchorBuilder, CodeParser, SupportedLanguage, find_node_at_position};
use crate::storage::NoteStorage;

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
        #[arg(short, long, default_value = "0")]
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

        /// Collection name (default: "default")
        #[arg(long, default_value = "default")]
        collection: String,
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

    /// Delete a note
    Delete {
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
        Commands::List { file, collection } => cmd_list(file, collection),
        Commands::View { id } => cmd_view(id),
        Commands::Update { id, content } => cmd_update(id, content),
        Commands::Delete { id, collection } => cmd_delete(id, collection),
        Commands::Migrate { collection } => cmd_migrate(collection),
        Commands::Collections => cmd_collections(),
        Commands::CreateCollection { name, description } => cmd_create_collection(name, description),
        Commands::Export { collection, output } => cmd_export(collection, output),
        Commands::Import { bundle } => cmd_import(bundle),
        Commands::Orphaned { collection } => cmd_orphaned(collection),
    }
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

    // Read file content
    let file_content = std::fs::read_to_string(&file)?;

    // Detect language
    let extension = file
        .extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("Could not determine file extension"))?;

    let language = SupportedLanguage::from_extension(extension)
        .ok_or_else(|| anyhow!("Unsupported language: {}", extension))?;

    // Parse file
    let mut parser = CodeParser::new(language)?;
    let tree = parser.parse(&file_content)?;

    // Find node at position (convert 1-indexed line to 0-indexed)
    let node = find_node_at_position(&tree, &file_content, line - 1, column)
        .ok_or_else(|| anyhow!("No code found at {}:{}", line, column))?;

    // Get relative path
    let rel_path = repo.relative_path(&file)?;

    // Build anchor
    let builder = AnchorBuilder::new(&file_content, rel_path);
    let commit_hash = repo.current_commit_hash()?;
    let anchor = builder.build_note_anchor(node, commit_hash)?;

    // Create note
    let note = Note::new(content.clone(), author, anchor);

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

fn cmd_list(file: Option<PathBuf>, collection_name: String) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let collection = storage.load_collection(&collection_name)?;

    let notes = if let Some(file_path) = file {
        let rel_path = repo.relative_path(&file_path)?;
        collection.notes_for_file(&rel_path)
    } else {
        collection.notes.iter().collect()
    };

    if notes.is_empty() {
        println!("No notes found");
        return Ok(());
    }

    println!("Notes in collection '{}':", collection_name);
    for note in notes {
        println!("\nID: {}", note.id);
        println!("File: {}:{}", note.anchor.primary.file_path, note.anchor.primary.line_number + 1);
        println!("Author: {}", note.author);
        println!("Content: {}", note.content);
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

    // Save by loading the collection and re-saving
    let collections = storage.load_all_collections()?;
    for mut collection in collections {
        if let Some(existing) = collection.get_note_mut(note.id) {
            *existing = note.clone();
            storage.save_collection(&collection)?;
            println!("Updated note {}", note.id);
            return Ok(());
        }
    }

    Err(anyhow!("Note not found in any collection"))
}

fn cmd_delete(id: String, collection_name: String) -> Result<()> {
    let repo = GitRepo::discover(".")?;
    let root = repo.root_path()?;
    let storage = NoteStorage::new(&root)?;

    let note_id = id.parse()?;
    let mut collection = storage.load_collection(&collection_name)?;

    collection.remove_note(note_id)
        .ok_or_else(|| anyhow!("Note not found in collection"))?;

    storage.save_collection(&collection)?;
    println!("Deleted note {}", note_id);
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
