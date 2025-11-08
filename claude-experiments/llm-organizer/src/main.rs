use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod analyzer;
mod config;
mod db;
mod fuse;
mod llm;
mod vfs_cli;
mod view;
mod watcher;

use config::Config;
use db::Database;
use llm::{LLMClient, CachedLLMClient};
use view::ViewEngine;
use analyzer::{analyze_file, DynamicAnalyzer};
use watcher::{FileWatcher, FileEvent, Debouncer};
use vfs_cli::{VirtualPath, VirtualFilesystem};

#[derive(Parser)]
#[command(name = "llm-organizer")]
#[command(about = "LLM-powered file organization with virtual filesystem", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to configuration file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the filesystem watcher and analyzer
    Watch,

    /// Mount the FUSE filesystem
    Mount {
        /// Override mount point from config
        #[arg(short, long)]
        mountpoint: Option<PathBuf>,
    },

    /// Create a new view
    CreateView {
        /// Name of the view
        name: String,

        /// Natural language query for the view
        query: String,
    },

    /// List all views
    ListViews,

    /// Analyze a specific file
    Analyze {
        /// Path to the file to analyze
        path: PathBuf,
    },

    /// Analyze all files in a directory recursively
    AnalyzeDir {
        /// Path to the directory to analyze
        path: PathBuf,

        /// Maximum number of files to process (default: no limit)
        #[arg(short, long)]
        limit: Option<usize>,
    },

    /// Run both watcher and mount filesystem
    Run,

    /// Browse the virtual filesystem (interactive mode)
    Browse {
        /// Starting path (e.g., /views, /all)
        #[arg(default_value = "/")]
        path: String,
    },

    /// List directory contents at a virtual path
    Ls {
        /// Virtual path (e.g., /views, /views/work, /all)
        #[arg(default_value = "/")]
        path: String,
    },

    /// Display file contents from a virtual path
    Cat {
        /// Virtual file path (e.g., /views/work/doc.pdf)
        path: String,
    },

    /// Show file information for a virtual path
    Stat {
        /// Virtual path
        path: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(if cli.verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();

    // Load configuration
    let config = if let Some(config_path) = cli.config {
        Config::from_file(config_path)?
    } else {
        Config::load_default()?
    };

    log::info!("Loaded configuration");
    log::debug!("Config: {:?}", config);

    // Initialize database
    let db = Database::new(&config.database.path)?;
    log::info!("Database initialized at {}", config.database.path.display());

    // Initialize LLM client
    let llm_client = LLMClient::new(
        config.llm.endpoint.clone(),
        config.llm.model.clone(),
        config.llm.max_tokens,
        config.llm.temperature,
        config.llm.timeout_secs,
    );

    let cached_llm = CachedLLMClient::new(llm_client, config.organization.cache_ttl_secs);
    log::info!("LLM client initialized for endpoint: {}", config.llm.endpoint);

    match cli.command {
        Commands::Watch => {
            run_watcher(config, db, cached_llm).await?;
        }
        Commands::Mount { mountpoint } => {
            let mount_point = mountpoint.unwrap_or(config.filesystem.mount_point.clone());
            run_mount(config, db, cached_llm, mount_point).await?;
        }
        Commands::CreateView { name, query } => {
            let view_engine = ViewEngine::new(db, cached_llm);
            let view_id = view_engine.create_view(&name, &query).await?;
            log::info!("Created view '{}' with ID {}", name, view_id);
        }
        Commands::ListViews => {
            let view_engine = ViewEngine::new(db, cached_llm);
            let views = view_engine.get_all_views()?;

            if views.is_empty() {
                println!("No views found");
            } else {
                println!("Available views:");
                for view in views {
                    println!("  - {} (query: {})", view.name, view.query_prompt);
                }
            }
        }
        Commands::Analyze { path } => {
            analyze_single_file(&path, &db, &cached_llm, &config.organization.prompt).await?;
        }
        Commands::AnalyzeDir { path, limit } => {
            analyze_directory(&path, &db, &cached_llm, &config, limit).await?;
        }
        Commands::Run => {
            // Run both watcher and mount in parallel
            let config_clone = config.clone();
            let db_clone = Database::new(&config.database.path)?;
            let cached_llm_clone = cached_llm.clone();

            let mount_point = config.filesystem.mount_point.clone();

            tokio::select! {
                result = run_watcher(config_clone, db_clone, cached_llm_clone) => {
                    result?;
                }
                result = run_mount(config, db, cached_llm, mount_point) => {
                    result?;
                }
            }
        }
        Commands::Browse { path } => {
            run_browse(&db, &path, &cached_llm, &config).await?;
        }
        Commands::Ls { path } => {
            run_ls(&db, &path)?;
        }
        Commands::Cat { path } => {
            run_cat(&db, &path)?;
        }
        Commands::Stat { path } => {
            run_stat(&db, &path)?;
        }
    }

    Ok(())
}

async fn run_watcher(config: Config, db: Database, llm: CachedLLMClient) -> Result<()> {
    log::info!("Starting file watcher...");

    let watch_dirs: Vec<PathBuf> = config.filesystem.watch_dirs.clone();
    let mut watcher = FileWatcher::new(&watch_dirs)?;
    let mut debouncer = Debouncer::new(2);

    loop {
        if let Some(event) = watcher.next_event().await {
            match event {
                FileEvent::Created(path) | FileEvent::Modified(path) => {
                    // Check if we should ignore this file
                    if watcher.should_ignore(&path, &config.filesystem.ignore_extensions) {
                        continue;
                    }

                    // Debounce rapid changes
                    if !debouncer.should_process(&path) {
                        continue;
                    }

                    log::info!("Processing file: {}", path.display());

                    if let Err(e) = analyze_and_store(&path, &db, &llm, &config.organization.prompt).await {
                        log::error!("Failed to analyze {}: {}", path.display(), e);
                    }
                }
                FileEvent::Removed(path) => {
                    log::info!("File removed: {}", path.display());
                    if let Err(e) = db.delete_file(&path.to_string_lossy()) {
                        log::error!("Failed to remove file from database: {}", e);
                    }
                }
            }

            // Periodically clean up old debouncer entries
            debouncer.cleanup_old_entries();
        }
    }
}

async fn run_mount(config: Config, db: Database, llm: CachedLLMClient, mount_point: PathBuf) -> Result<()> {
    log::info!("Mounting filesystem at {}", mount_point.display());

    // Create mount point if it doesn't exist
    if !mount_point.exists() {
        std::fs::create_dir_all(&mount_point)?;
    }

    let view_engine = ViewEngine::new(db.clone(), llm);
    let fs = fuse::OrganizedFS::new(db, view_engine)?;

    // This will block until unmounted
    fs.mount(mount_point)?;

    Ok(())
}

async fn analyze_and_store(
    path: &std::path::Path,
    db: &Database,
    llm: &CachedLLMClient,
    organization_prompt: &str,
) -> Result<()> {
    // Analyze the file
    let analysis = analyze_file(path).await?;

    // Insert file record
    let file_id = db.insert_file(
        &analysis.path,
        &analysis.content_hash,
        analysis.size_bytes,
        analysis.modified_time,
        Some(&analysis.file_type),
        Some(&analysis.content.text),
    )?;

    // Skip LLM analysis for empty files
    if analysis.content.text.is_empty() {
        log::debug!("Skipping LLM analysis for empty/binary file: {}", path.display());
        return Ok(());
    }

    // Generate summary
    let summary = llm.summarize_file(&analysis.content.text, &analysis.file_type).await?;
    log::debug!("Summary: {}", summary);

    // Extract tags
    let tags = llm.extract_tags(&analysis.content.text, &summary).await
        .unwrap_or_default();
    log::debug!("Tags: {:?}", tags);

    // Extract categories
    let categories = llm.extract_categories(&analysis.content.text, &summary, organization_prompt).await
        .unwrap_or_default();
    log::debug!("Categories: {:?}", categories);

    // Extract entities
    let entities = llm.extract_entities(&analysis.content.text).await
        .unwrap_or(serde_json::json!({}));
    log::debug!("Entities: {:?}", entities);

    // Store metadata
    db.insert_metadata(file_id, Some(&summary), &tags, &categories, &entities)?;

    log::info!("Successfully analyzed and stored metadata for {}", path.display());

    Ok(())
}

async fn analyze_single_file(
    path: &std::path::Path,
    db: &Database,
    llm: &CachedLLMClient,
    organization_prompt: &str,
) -> Result<()> {
    println!("Analyzing file: {}", path.display());

    analyze_and_store(path, db, llm, organization_prompt).await?;

    // Print results
    if let Some(file) = db.get_file_by_path(&path.to_string_lossy())? {
        if let Some(metadata) = db.get_metadata(file.id)? {
            println!("\nAnalysis Results:");
            println!("================");
            println!("File Type: {}", file.file_type.unwrap_or_default());
            println!("Size: {} bytes", file.size_bytes);
            println!("\nSummary:");
            println!("{}", metadata.llm_summary.unwrap_or_default());
            println!("\nTags: {}", metadata.tags.join(", "));
            println!("Categories: {}", metadata.categories.join(", "));
            println!("\nEntities:");
            println!("{}", serde_json::to_string_pretty(&metadata.entities)?);
        }
    }

    Ok(())
}

async fn run_browse(db: &Database, start_path: &str, llm: &CachedLLMClient, config: &Config) -> Result<()> {
    use std::io::{self, Write};

    let vfs = VirtualFilesystem::new(db.clone());
    let mut current_path = start_path.to_string();

    // Print welcome banner
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║         📁 LLM File Organizer - Virtual Shell          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("\n💡 Type 'help' for available commands, 'exit' to quit\n");

    loop {
        // Pretty prompt with color-like characters
        print!("\x1b[36m{}\x1b[0m \x1b[32m❯\x1b[0m ", current_path);
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let cmd = parts[0];

        match cmd {
            "exit" | "quit" | "q" => {
                println!("Goodbye!");
                break;
            }
            "pwd" => {
                println!("{}", current_path);
            }
            "ls" => {
                let path_to_list = if parts.len() > 1 {
                    resolve_path(&current_path, parts[1])
                } else {
                    current_path.clone()
                };

                if let Err(e) = list_directory(&vfs, &path_to_list) {
                    eprintln!("Error: {}", e);
                }
            }
            "cd" => {
                if parts.len() < 2 {
                    eprintln!("Usage: cd <path>");
                    continue;
                }
                let new_path = resolve_path(&current_path, parts[1]);

                // Verify the path exists and is a directory
                match VirtualPath::parse(&new_path) {
                    Ok(vpath) => {
                        match vfs.stat(&vpath) {
                            Ok(vfs_cli::VirtualStat::Directory) => {
                                current_path = new_path;
                            }
                            Ok(vfs_cli::VirtualStat::File { .. }) => {
                                eprintln!("Error: Not a directory");
                            }
                            Err(e) => {
                                eprintln!("Error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {}", e);
                    }
                }
            }
            "cat" => {
                if parts.len() < 2 {
                    eprintln!("Usage: cat <file>");
                    continue;
                }
                let file_path = resolve_path(&current_path, parts[1]);
                if let Err(e) = cat_file(&vfs, &file_path) {
                    eprintln!("Error: {}", e);
                }
            }
            "stat" => {
                if parts.len() < 2 {
                    eprintln!("Usage: stat <path>");
                    continue;
                }
                let path = resolve_path(&current_path, parts[1]);
                if let Err(e) = show_stat(&vfs, &path) {
                    eprintln!("Error: {}", e);
                }
            }
            "help" => {
                println!("\n╭─────────────────────────────────────────────────────╮");
                println!("│            📖 Available Commands                    │");
                println!("├─────────────────────────────────────────────────────┤");
                println!("│ \x1b[36mNavigation:\x1b[0m                                      │");
                println!("│ \x1b[33mls\x1b[0m [path]        List directory contents          │");
                println!("│ \x1b[33mcd\x1b[0m <path>        Change directory                  │");
                println!("│ \x1b[33mpwd\x1b[0m              Print current directory            │");
                println!("│                                                     │");
                println!("│ \x1b[36mFile Operations:\x1b[0m                                 │");
                println!("│ \x1b[33mcat\x1b[0m <file>       Display file contents             │");
                println!("│ \x1b[33mstat\x1b[0m <path>      Show file/directory info           │");
                println!("│ \x1b[33minfo\x1b[0m <file>      Show LLM-extracted metadata        │");
                println!("│                                                     │");
                println!("│ \x1b[36mOrganization:\x1b[0m                                     │");
                println!("│ \x1b[33mviews\x1b[0m            List all views                    │");
                println!("│ \x1b[33mcreate-view\x1b[0m      Create new semantic view          │");
                println!("│                                                     │");
                println!("│ \x1b[36mAnalysis:\x1b[0m                                         │");
                println!("│ \x1b[33madd\x1b[0m <file>       Analyze and add a file            │");
                println!("│ \x1b[33madd-dir\x1b[0m <path>   Analyze and add directory         │");
                println!("│                                                     │");
                println!("│ \x1b[36mOther:\x1b[0m                                            │");
                println!("│ \x1b[33mclear\x1b[0m            Clear screen                       │");
                println!("│ \x1b[33mhelp\x1b[0m             Show this help                     │");
                println!("│ \x1b[33mexit\x1b[0m             Exit shell (or quit, q)            │");
                println!("╰─────────────────────────────────────────────────────╯\n");
            }
            "clear" => {
                print!("\x1b[2J\x1b[H");
                io::stdout().flush()?;
            }
            "info" => {
                if parts.len() < 2 {
                    eprintln!("Usage: info <file>");
                    continue;
                }
                let file_path = resolve_path(&current_path, parts[1]);
                if let Err(e) = show_file_info(&vfs, &file_path, db) {
                    eprintln!("Error: {}", e);
                }
            }
            "add" => {
                if parts.len() < 2 {
                    eprintln!("Usage: add <file>");
                    continue;
                }
                let file_path = std::path::PathBuf::from(parts[1]);

                println!("\x1b[36m🔍 Analyzing\x1b[0m {}...", file_path.display());
                match analyze_and_store(&file_path, db, llm, &config.organization.prompt).await {
                    Ok(_) => {
                        println!("\x1b[32m✓ Successfully added!\x1b[0m File is now available in /all");
                    }
                    Err(e) => {
                        eprintln!("\x1b[31m✗ Failed:\x1b[0m {}", e);
                    }
                }
            }
            "add-dir" => {
                if parts.len() < 2 {
                    eprintln!("Usage: add-dir <directory> [--limit N]");
                    continue;
                }
                let dir_path = std::path::PathBuf::from(parts[1]);

                // Check for --limit flag
                let limit = if parts.len() > 2 && parts[2] == "--limit" && parts.len() > 3 {
                    parts[3].parse::<usize>().ok()
                } else {
                    None
                };

                println!("\x1b[36m🔍 Analyzing directory\x1b[0m {}...", dir_path.display());
                match analyze_directory(&dir_path, db, llm, config, limit).await {
                    Ok(_) => {
                        println!("\x1b[32m✓ Directory analysis complete!\x1b[0m");
                    }
                    Err(e) => {
                        eprintln!("\x1b[31m✗ Failed:\x1b[0m {}", e);
                    }
                }
            }
            "views" => {
                let view_engine = ViewEngine::new(db.clone(), llm.clone());
                match view_engine.get_all_views() {
                    Ok(views) => {
                        if views.is_empty() {
                            println!("\n  \x1b[90mNo views created yet.\x1b[0m");
                            println!("  Use '\x1b[33mcreate-view\x1b[0m' to create one!\n");
                        } else {
                            println!("\n\x1b[1m📁 Available Views:\x1b[0m");
                            for view in views {
                                println!("  \x1b[36m{}\x1b[0m", view.name);
                                println!("    \x1b[90m{}\x1b[0m", view.query_prompt);
                            }
                            println!();
                        }
                    }
                    Err(e) => {
                        eprintln!("\x1b[31m✗ Failed to list views:\x1b[0m {}", e);
                    }
                }
            }
            "create-view" => {
                println!("\n\x1b[1m📋 Create New View\x1b[0m");
                println!("\x1b[90mThis will use the LLM to organize your files based on a query.\x1b[0m\n");

                print!("View name: ");
                io::stdout().flush()?;
                let mut view_name = String::new();
                io::stdin().read_line(&mut view_name)?;
                let view_name = view_name.trim();

                if view_name.is_empty() {
                    eprintln!("\x1b[31m✗ View name cannot be empty\x1b[0m");
                    continue;
                }

                print!("Query (e.g., 'Work documents from 2024', 'Python code files'): ");
                io::stdout().flush()?;
                let mut query = String::new();
                io::stdin().read_line(&mut query)?;
                let query = query.trim();

                if query.is_empty() {
                    eprintln!("\x1b[31m✗ Query cannot be empty\x1b[0m");
                    continue;
                }

                println!("\n\x1b[36m🔮 Creating view...\x1b[0m");
                let view_engine = ViewEngine::new(db.clone(), llm.clone());
                match view_engine.create_view(view_name, query).await {
                    Ok(view_id) => {
                        println!("\x1b[32m✓ View '{}' created successfully!\x1b[0m", view_name);
                        println!("  View ID: {}", view_id);
                        println!("  Access it at: \x1b[36m/views/{}\x1b[0m\n", view_name);
                    }
                    Err(e) => {
                        eprintln!("\x1b[31m✗ Failed to create view:\x1b[0m {}", e);
                    }
                }
            }
            _ => {
                eprintln!("\x1b[31m✗\x1b[0m Unknown command: \x1b[33m{}\x1b[0m", cmd);
                eprintln!("  Type '\x1b[33mhelp\x1b[0m' for available commands.");
            }
        }
    }

    Ok(())
}

fn run_ls(db: &Database, path: &str) -> Result<()> {
    let vfs = VirtualFilesystem::new(db.clone());
    list_directory(&vfs, path)
}

fn run_cat(db: &Database, path: &str) -> Result<()> {
    let vfs = VirtualFilesystem::new(db.clone());
    cat_file(&vfs, path)
}

fn run_stat(db: &Database, path: &str) -> Result<()> {
    let vfs = VirtualFilesystem::new(db.clone());
    show_stat(&vfs, path)
}

fn list_directory(vfs: &VirtualFilesystem, path: &str) -> Result<()> {
    let vpath = VirtualPath::parse(path)?;
    let entries = vfs.list(&vpath)?;

    if entries.is_empty() {
        println!("  \x1b[90m(empty)\x1b[0m");
        return Ok(());
    }

    // Separate directories and files
    let mut dirs = Vec::new();
    let mut files = Vec::new();

    for entry in entries {
        if entry.is_dir() {
            dirs.push(entry);
        } else {
            files.push(entry);
        }
    }

    // List directories first
    for entry in dirs {
        println!("  \x1b[34m📁 {}/\x1b[0m", entry.name());
    }

    // Then files
    for entry in files {
        match entry {
            vfs_cli::VirtualEntry::File { name, size, .. } => {
                let size_str = format_size(size);
                println!("  \x1b[37m📄 {}\x1b[0m \x1b[90m({})\x1b[0m", name, size_str);
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn cat_file(vfs: &VirtualFilesystem, path: &str) -> Result<()> {
    let vpath = VirtualPath::parse(path)?;
    let real_path = vfs.get_real_path(&vpath)?;

    let contents = std::fs::read_to_string(&real_path)?;
    println!("{}", contents);

    Ok(())
}

fn show_stat(vfs: &VirtualFilesystem, path: &str) -> Result<()> {
    let vpath = VirtualPath::parse(path)?;
    let stat = vfs.stat(&vpath)?;

    match stat {
        vfs_cli::VirtualStat::Directory => {
            println!("\n  \x1b[34m📁 Directory\x1b[0m");
            println!("  Path: \x1b[36m{}\x1b[0m", path);
        }
        vfs_cli::VirtualStat::File { size, real_path } => {
            println!("\n  \x1b[37m📄 File\x1b[0m");
            println!("  Virtual path: \x1b[36m{}\x1b[0m", path);
            println!("  Real path: \x1b[90m{}\x1b[0m", real_path.display());
            println!("  Size: \x1b[33m{}\x1b[0m", format_size(size));
        }
    }
    println!();

    Ok(())
}

fn show_file_info(vfs: &VirtualFilesystem, path: &str, db: &Database) -> Result<()> {
    use anyhow::Context;

    let vpath = VirtualPath::parse(path)?;
    let real_path = vfs.get_real_path(&vpath)?;

    // Get file record from database
    let file = db.get_file_by_path(&real_path.to_string_lossy())
        .context("Failed to query database")?
        .ok_or_else(|| anyhow::anyhow!("File not found in database"))?;

    // Get metadata
    let metadata = db.get_metadata(file.id)
        .context("Failed to query metadata")?
        .ok_or_else(|| anyhow::anyhow!("No metadata found for this file"))?;

    // Display info with nice formatting
    println!("\n╭─────────────────────────────────────────────────────────────╮");
    println!("│  \x1b[1m📄 {}\x1b[0m", file.path.split('/').last().unwrap_or(&file.path));
    println!("├─────────────────────────────────────────────────────────────┤");

    if let Some(ref summary) = metadata.llm_summary {
        println!("│ \x1b[1m📝 Summary:\x1b[0m");
        for line in textwrap::wrap(summary, 58) {
            println!("│   {}", line);
        }
        println!("│");
    }

    if !metadata.tags.is_empty() {
        println!("│ \x1b[1m🏷️  Tags:\x1b[0m");
        println!("│   \x1b[33m{}\x1b[0m", metadata.tags.join(", "));
        println!("│");
    }

    if !metadata.categories.is_empty() {
        println!("│ \x1b[1m📂 Categories:\x1b[0m");
        println!("│   \x1b[36m{}\x1b[0m", metadata.categories.join(" › "));
        println!("│");
    }

    // Show entities if they exist
    if let Some(obj) = metadata.entities.as_object() {
        let mut has_entities = false;

        for (key, value) in obj {
            if let Some(arr) = value.as_array() {
                if !arr.is_empty() {
                    if !has_entities {
                        println!("│ \x1b[1m👤 Entities:\x1b[0m");
                        has_entities = true;
                    }
                    let items: Vec<String> = arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    if !items.is_empty() {
                        println!("│   \x1b[90m{}:\x1b[0m {}", key, items.join(", "));
                    }
                }
            }
        }
        if has_entities {
            println!("│");
        }
    }

    println!("│ \x1b[90mSize: {} • Type: {}\x1b[0m",
             format_size(file.size_bytes as u64),
             file.file_type.as_deref().unwrap_or("unknown"));
    println!("╰─────────────────────────────────────────────────────────────╯\n");

    Ok(())
}

fn resolve_path(current: &str, target: &str) -> String {
    if target.starts_with('/') {
        // Absolute path
        target.to_string()
    } else if target == ".." {
        // Go up one level
        let parts: Vec<&str> = current.trim_end_matches('/').split('/').collect();
        if parts.len() <= 1 {
            "/".to_string()
        } else {
            parts[..parts.len() - 1].join("/")
        }
    } else if target == "." {
        // Current directory
        current.to_string()
    } else {
        // Relative path
        let base = current.trim_end_matches('/');
        if base.is_empty() || base == "/" {
            format!("/{}", target)
        } else {
            format!("{}/{}", base, target)
        }
    }
}

async fn analyze_directory(
    dir_path: &std::path::Path,
    db: &Database,
    llm: &CachedLLMClient,
    config: &Config,
    limit: Option<usize>,
) -> Result<()> {
    use std::fs;

    if !dir_path.is_dir() {
        anyhow::bail!("Path is not a directory: {}", dir_path.display());
    }

    println!("Analyzing directory: {}", dir_path.display());
    println!("Scanning for files...");

    // Collect all files recursively
    let mut files_to_process = Vec::new();
    collect_files_recursive(dir_path, &mut files_to_process, &config.filesystem.ignore_extensions)?;

    let total_files = files_to_process.len();
    let files_to_process = if let Some(limit) = limit {
        files_to_process.into_iter().take(limit).collect()
    } else {
        files_to_process
    };

    let count = files_to_process.len();
    println!("Found {} files to analyze (total: {})", count, total_files);

    let mut processed = 0;
    let mut succeeded = 0;
    let mut failed = 0;

    for file_path in files_to_process {
        processed += 1;

        // Show progress every 10 files or for small batches show each
        if count <= 10 || processed % 10 == 0 || processed == count {
            println!("[{}/{}] Processing files...", processed, count);
        }

        match analyze_and_store(&file_path, db, llm, &config.organization.prompt).await {
            Ok(_) => {
                succeeded += 1;
            }
            Err(e) => {
                failed += 1;
                eprintln!("  ✗ Failed ({}): {}", file_path.display(), e);
            }
        }
    }

    println!("\n=== Analysis Complete ===");
    println!("Total processed: {}", processed);
    println!("Succeeded: {}", succeeded);
    println!("Failed: {}", failed);

    Ok(())
}

fn collect_files_recursive(
    dir: &std::path::Path,
    files: &mut Vec<PathBuf>,
    ignore_extensions: &[String],
) -> Result<()> {
    use std::fs;

    if !dir.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recurse into subdirectories
            collect_files_recursive(&path, files, ignore_extensions)?;
        } else if path.is_file() {
            // Check if we should ignore this file
            let should_ignore = if let Some(ext) = path.extension() {
                let ext_str = format!(".{}", ext.to_string_lossy());
                ignore_extensions.contains(&ext_str)
            } else {
                false
            };

            if !should_ignore {
                files.push(path);
            }
        }
    }

    Ok(())
}
