use anyhow::{anyhow, Result};
use std::fs;
use std::path::Path;
use std::process::Command;

use crate::parsers::language_registry::{GrammarSource, LanguageMetadata, LanguageRegistry};

/// Handles downloading and installing tree-sitter grammars
pub struct LanguageInstaller {
    registry: LanguageRegistry,
}

impl LanguageInstaller {
    /// Create a new language installer
    pub fn new() -> Result<Self> {
        let registry = LanguageRegistry::new()?;
        registry.initialize()?;
        Ok(Self { registry })
    }

    /// Install a language grammar by name
    pub fn install(&mut self, name: &str) -> Result<()> {
        // Check if already installed
        if self.registry.is_installed(name)? {
            println!("Language '{}' is already installed", name);
            return Ok(());
        }

        // Find the grammar source
        let source = GrammarSource::find_by_name(name)
            .ok_or_else(|| anyhow!("Unknown language '{}'. Use 'code-notes lang list-available' to see available languages", name))?;

        println!("Installing {}...", name);
        println!("Repository: {}", source.repo);

        // Create language directory
        let lang_dir = self.registry.language_dir(name);
        fs::create_dir_all(&lang_dir)?;

        // Clone the repository
        self.clone_grammar_repo(&source.repo, &lang_dir)?;

        // Build the grammar
        self.build_grammar(&lang_dir, name)?;

        // Register the language
        let metadata = LanguageMetadata {
            name: name.to_string(),
            version: "latest".to_string(), // Could fetch actual version from git
            extensions: source.extensions.clone(),
            source: source.repo.clone(),
            installed_at: chrono::Utc::now().to_rfc3339(),
        };

        self.registry.register_language(metadata)?;

        println!("✓ Successfully installed {}", name);
        Ok(())
    }

    /// Uninstall a language grammar
    pub fn uninstall(&self, name: &str) -> Result<()> {
        if !self.registry.is_installed(name)? {
            return Err(anyhow!("Language '{}' is not installed", name));
        }

        self.registry.unregister_language(name)?;
        println!("✓ Successfully uninstalled {}", name);
        Ok(())
    }

    /// Clone a grammar repository
    fn clone_grammar_repo(&self, repo: &str, dest: &Path) -> Result<()> {
        let url = format!("https://github.com/{}.git", repo);

        println!("Cloning {}...", url);

        let output = Command::new("git")
            .args(["clone", "--depth", "1", &url, dest.to_str().unwrap()])
            .output()?;

        if !output.status.success() {
            return Err(anyhow!(
                "Failed to clone repository: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(())
    }

    /// Build a tree-sitter grammar
    fn build_grammar(&self, grammar_dir: &Path, name: &str) -> Result<()> {
        println!("Building grammar...");

        // Some grammars have subdirectories (e.g., typescript has /typescript and /tsx)
        let src_dir = if grammar_dir.join("src").exists() {
            grammar_dir.to_path_buf()
        } else if grammar_dir.join(name).join("src").exists() {
            grammar_dir.join(name)
        } else {
            return Err(anyhow!("Could not find grammar source directory"));
        };

        // Check if we need to generate the parser
        if src_dir.join("grammar.js").exists() && !src_dir.join("src").join("parser.c").exists() {
            println!("Generating parser from grammar.js...");
            let output = Command::new("tree-sitter")
                .args(["generate"])
                .current_dir(&src_dir)
                .output()?;

            if !output.status.success() {
                return Err(anyhow!(
                    "Failed to generate parser: {}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        }

        // Build the shared library using tree-sitter CLI
        let output = Command::new("tree-sitter")
            .args(["build", "--output", &format!("libtree-sitter-{}", name)])
            .current_dir(&src_dir)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!(
                "Failed to build grammar: {}\n\nMake sure you have tree-sitter CLI installed: npm install -g tree-sitter-cli",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(())
    }

    /// List all available grammars
    pub fn list_available(&self) -> Vec<GrammarSource> {
        GrammarSource::known_grammars()
    }

    /// List installed grammars
    pub fn list_installed(&self) -> Result<Vec<LanguageMetadata>> {
        self.registry.list_installed()
    }
}

// Add chrono for timestamps
