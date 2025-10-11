use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tree_sitter::Language;
use tree_sitter_loader::{Loader, CompileConfig};

const GRAMMARS_DIR: &str = "grammars";
const REGISTRY_FILE: &str = "registry.json";

/// Metadata about an installed language grammar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageMetadata {
    /// Language name (e.g., "rust", "python")
    pub name: String,
    /// Version of the grammar
    pub version: String,
    /// File extensions this language handles
    pub extensions: Vec<String>,
    /// Source URL or repository
    pub source: String,
    /// Installation timestamp
    pub installed_at: String,
}

/// Registry of installed tree-sitter language grammars
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RegistryData {
    languages: HashMap<String, LanguageMetadata>,
}

/// Manages dynamic tree-sitter language loading
pub struct LanguageRegistry {
    grammars_dir: PathBuf,
    registry_path: PathBuf,
    loader: Loader,
    /// In-memory cache of loaded languages
    loaded_languages: HashMap<String, Language>,
}

impl LanguageRegistry {
    /// Create a new language registry
    pub fn new() -> Result<Self> {
        let grammars_dir = Self::grammars_dir()?;
        let registry_path = grammars_dir.join(REGISTRY_FILE);

        // Initialize loader with our grammars directory
        let mut loader = Loader::new()?;
        loader.configure_highlights(&[]);

        Ok(Self {
            grammars_dir,
            registry_path,
            loader,
            loaded_languages: HashMap::new(),
        })
    }

    /// Get the global grammars directory
    fn grammars_dir() -> Result<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| anyhow!("Could not determine home directory"))?;

        Ok(PathBuf::from(home).join(".code-notes").join(GRAMMARS_DIR))
    }

    /// Initialize the registry (create directories if needed)
    pub fn initialize(&self) -> Result<()> {
        fs::create_dir_all(&self.grammars_dir)?;

        // Create empty registry if it doesn't exist
        if !self.registry_path.exists() {
            let empty_registry = RegistryData::default();
            self.save_registry(&empty_registry)?;
        }

        Ok(())
    }

    /// Load the registry data
    fn load_registry(&self) -> Result<RegistryData> {
        if !self.registry_path.exists() {
            return Ok(RegistryData::default());
        }

        let content = fs::read_to_string(&self.registry_path)?;
        let registry: RegistryData = serde_json::from_str(&content)?;
        Ok(registry)
    }

    /// Save the registry data
    fn save_registry(&self, registry: &RegistryData) -> Result<()> {
        let json = serde_json::to_string_pretty(registry)?;
        fs::write(&self.registry_path, json)?;
        Ok(())
    }

    /// Get a language by name, loading it if necessary
    pub fn get_language(&mut self, name: &str) -> Result<Language> {
        // Check cache first
        if let Some(lang) = self.loaded_languages.get(name) {
            return Ok(lang.clone());
        }

        // Check if language is installed
        let registry = self.load_registry()?;
        if !registry.languages.contains_key(name) {
            return Err(anyhow!(
                "Language '{}' is not installed. Run: code-notes lang install {}",
                name,
                name
            ));
        }

        // Load the language using tree-sitter-loader
        let lang_dir = self.grammars_dir.join(name);

        // Note: tree-sitter builds libraries without extension on macOS/Unix,
        // with .dll on Windows
        let lib_name = format!("libtree-sitter-{}", name);
        let output_path = lang_dir.join(&lib_name);
        let flags: &[&str] = &[];

        // Create compile config for the language
        let config = CompileConfig {
            src_path: &lang_dir,
            header_paths: vec![],
            parser_path: lang_dir.join("src").join("parser.c"),
            scanner_path: None,
            external_files: None,
            output_path: Some(output_path),
            flags,
            name: name.to_string(),
        };

        let language = self.loader.load_language_at_path(config)?;

        // Cache it
        self.loaded_languages.insert(name.to_string(), language.clone());

        Ok(language)
    }

    /// Get language name from file extension
    pub fn language_from_extension(&self, ext: &str) -> Result<String> {
        let registry = self.load_registry()?;

        // Check installed languages first
        for (name, metadata) in &registry.languages {
            if metadata.extensions.contains(&ext.to_string()) {
                return Ok(name.clone());
            }
        }

        // Fall back to built-in language mappings
        match ext {
            "rs" => Ok("rust".to_string()),
            "py" | "pyw" => Ok("python".to_string()),
            "js" | "jsx" | "mjs" => Ok("javascript".to_string()),
            "ts" | "tsx" => Ok("typescript".to_string()),
            _ => Err(anyhow!("No language found for extension '.{}'", ext))
        }
    }

    /// Check if a language is installed
    pub fn is_installed(&self, name: &str) -> Result<bool> {
        let registry = self.load_registry()?;
        Ok(registry.languages.contains_key(name))
    }

    /// List all installed languages
    pub fn list_installed(&self) -> Result<Vec<LanguageMetadata>> {
        let registry = self.load_registry()?;
        Ok(registry.languages.values().cloned().collect())
    }

    /// Register a language in the registry
    pub fn register_language(&self, metadata: LanguageMetadata) -> Result<()> {
        let mut registry = self.load_registry()?;
        registry.languages.insert(metadata.name.clone(), metadata);
        self.save_registry(&registry)?;
        Ok(())
    }

    /// Unregister a language from the registry
    pub fn unregister_language(&self, name: &str) -> Result<()> {
        let mut registry = self.load_registry()?;
        registry.languages.remove(name);
        self.save_registry(&registry)?;

        // Remove the language directory
        let lang_dir = self.grammars_dir.join(name);
        if lang_dir.exists() {
            fs::remove_dir_all(lang_dir)?;
        }

        Ok(())
    }

    /// Get the directory for a specific language
    pub fn language_dir(&self, name: &str) -> PathBuf {
        self.grammars_dir.join(name)
    }

    /// Get metadata for a language
    pub fn get_metadata(&self, name: &str) -> Result<LanguageMetadata> {
        let registry = self.load_registry()?;
        registry
            .languages
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Language '{}' not found", name))
    }
}

/// Known tree-sitter grammar repositories
pub struct GrammarSource {
    pub name: String,
    pub repo: String,
    pub extensions: Vec<String>,
}

impl GrammarSource {
    /// Get the list of well-known grammars
    pub fn known_grammars() -> Vec<Self> {
        vec![
            Self {
                name: "rust".to_string(),
                repo: "tree-sitter/tree-sitter-rust".to_string(),
                extensions: vec!["rs".to_string()],
            },
            Self {
                name: "python".to_string(),
                repo: "tree-sitter/tree-sitter-python".to_string(),
                extensions: vec!["py".to_string(), "pyw".to_string()],
            },
            Self {
                name: "javascript".to_string(),
                repo: "tree-sitter/tree-sitter-javascript".to_string(),
                extensions: vec!["js".to_string(), "jsx".to_string(), "mjs".to_string()],
            },
            Self {
                name: "typescript".to_string(),
                repo: "tree-sitter/tree-sitter-typescript".to_string(),
                extensions: vec!["ts".to_string(), "tsx".to_string()],
            },
            Self {
                name: "go".to_string(),
                repo: "tree-sitter/tree-sitter-go".to_string(),
                extensions: vec!["go".to_string()],
            },
            Self {
                name: "java".to_string(),
                repo: "tree-sitter/tree-sitter-java".to_string(),
                extensions: vec!["java".to_string()],
            },
            Self {
                name: "c".to_string(),
                repo: "tree-sitter/tree-sitter-c".to_string(),
                extensions: vec!["c".to_string(), "h".to_string()],
            },
            Self {
                name: "cpp".to_string(),
                repo: "tree-sitter/tree-sitter-cpp".to_string(),
                extensions: vec!["cpp".to_string(), "cc".to_string(), "hpp".to_string(), "hh".to_string()],
            },
            Self {
                name: "ruby".to_string(),
                repo: "tree-sitter/tree-sitter-ruby".to_string(),
                extensions: vec!["rb".to_string()],
            },
            Self {
                name: "php".to_string(),
                repo: "tree-sitter/tree-sitter-php".to_string(),
                extensions: vec!["php".to_string()],
            },
            Self {
                name: "swift".to_string(),
                repo: "tree-sitter/tree-sitter-swift".to_string(),
                extensions: vec!["swift".to_string()],
            },
            Self {
                name: "kotlin".to_string(),
                repo: "tree-sitter/tree-sitter-kotlin".to_string(),
                extensions: vec!["kt".to_string(), "kts".to_string()],
            },
            Self {
                name: "scala".to_string(),
                repo: "tree-sitter/tree-sitter-scala".to_string(),
                extensions: vec!["scala".to_string()],
            },
            Self {
                name: "haskell".to_string(),
                repo: "tree-sitter/tree-sitter-haskell".to_string(),
                extensions: vec!["hs".to_string()],
            },
            Self {
                name: "ocaml".to_string(),
                repo: "tree-sitter/tree-sitter-ocaml".to_string(),
                extensions: vec!["ml".to_string(), "mli".to_string()],
            },
            Self {
                name: "elixir".to_string(),
                repo: "elixir-lang/tree-sitter-elixir".to_string(),
                extensions: vec!["ex".to_string(), "exs".to_string()],
            },
            Self {
                name: "erlang".to_string(),
                repo: "WhatsApp/tree-sitter-erlang".to_string(),
                extensions: vec!["erl".to_string(), "hrl".to_string()],
            },
            Self {
                name: "clojure".to_string(),
                repo: "sogaiu/tree-sitter-clojure".to_string(),
                extensions: vec!["clj".to_string(), "cljs".to_string(), "cljc".to_string()],
            },
            Self {
                name: "racket".to_string(),
                repo: "6cdh/tree-sitter-racket".to_string(),
                extensions: vec!["rkt".to_string()],
            },
            Self {
                name: "lua".to_string(),
                repo: "tree-sitter-grammars/tree-sitter-lua".to_string(),
                extensions: vec!["lua".to_string()],
            },
            Self {
                name: "zig".to_string(),
                repo: "maxxnino/tree-sitter-zig".to_string(),
                extensions: vec!["zig".to_string()],
            },
            Self {
                name: "bash".to_string(),
                repo: "tree-sitter/tree-sitter-bash".to_string(),
                extensions: vec!["sh".to_string(), "bash".to_string()],
            },
            Self {
                name: "css".to_string(),
                repo: "tree-sitter/tree-sitter-css".to_string(),
                extensions: vec!["css".to_string()],
            },
            Self {
                name: "html".to_string(),
                repo: "tree-sitter/tree-sitter-html".to_string(),
                extensions: vec!["html".to_string(), "htm".to_string()],
            },
            Self {
                name: "json".to_string(),
                repo: "tree-sitter/tree-sitter-json".to_string(),
                extensions: vec!["json".to_string()],
            },
            Self {
                name: "yaml".to_string(),
                repo: "tree-sitter-grammars/tree-sitter-yaml".to_string(),
                extensions: vec!["yaml".to_string(), "yml".to_string()],
            },
            Self {
                name: "toml".to_string(),
                repo: "tree-sitter-grammars/tree-sitter-toml".to_string(),
                extensions: vec!["toml".to_string()],
            },
            Self {
                name: "markdown".to_string(),
                repo: "tree-sitter-grammars/tree-sitter-markdown".to_string(),
                extensions: vec!["md".to_string(), "markdown".to_string()],
            },
        ]
    }

    /// Find a grammar by name
    pub fn find_by_name(name: &str) -> Option<Self> {
        Self::known_grammars()
            .into_iter()
            .find(|g| g.name == name)
    }
}
