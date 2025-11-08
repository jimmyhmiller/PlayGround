use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use crate::llm::CachedLLMClient;
use crate::db::Database;

pub struct DynamicAnalyzer {
    llm: CachedLLMClient,
    db: Database,
    analyzer_dir: PathBuf,
}

impl DynamicAnalyzer {
    pub fn new(llm: CachedLLMClient, db: Database, analyzer_dir: PathBuf) -> Result<Self> {
        // Create analyzer directory if it doesn't exist
        std::fs::create_dir_all(&analyzer_dir)
            .context("Failed to create analyzer directory")?;

        Ok(Self {
            llm,
            db,
            analyzer_dir,
        })
    }

    /// Generate and save an analyzer for a specific file type
    pub async fn generate_analyzer(&self, file_type: &str, sample_path: &Path) -> Result<PathBuf> {
        log::info!("Generating analyzer for file type: {}", file_type);

        // Check if we already have an analyzer for this type
        if let Ok(Some(existing)) = self.db.get_analyzer(file_type) {
            log::info!("Using existing analyzer: {}", existing.script_path);
            return Ok(PathBuf::from(existing.script_path));
        }

        // Read sample bytes
        let sample_bytes = std::fs::read(sample_path)
            .context("Failed to read sample file")?;

        // Ask LLM to generate analyzer code
        let code = self.llm.generate_analyzer(file_type, &sample_bytes[..256.min(sample_bytes.len())])
            .await
            .context("Failed to generate analyzer code from LLM")?;

        log::debug!("Generated analyzer code:\n{}", code);

        // Determine the language and script path
        let (language, extension) = if code.contains("fn main()") || code.starts_with("use ") {
            ("rust", "rs")
        } else if code.contains("def ") || code.contains("import ") {
            ("python", "py")
        } else if code.contains("#!/bin/bash") {
            ("shell", "sh")
        } else {
            ("rust", "rs") // Default to Rust
        };

        // Save the script
        let safe_type = file_type.replace('/', "_");
        let script_name = format!("analyzer_{}.{}", safe_type, extension);
        let script_path = self.analyzer_dir.join(&script_name);

        std::fs::write(&script_path, &code)
            .context("Failed to write analyzer script")?;

        // Make executable if shell script
        if language == "shell" {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = std::fs::metadata(&script_path)?.permissions();
                perms.set_mode(0o755);
                std::fs::set_permissions(&script_path, perms)?;
            }
        }

        // Compile if Rust
        let final_script_path = if language == "rust" {
            self.compile_rust_analyzer(&script_path).await?
        } else {
            script_path.clone()
        };

        // Register in database
        self.db.register_analyzer(
            file_type,
            final_script_path.to_str().unwrap(),
            language,
            Some(&format!("Auto-generated analyzer for {}", file_type)),
        )?;

        log::info!("Successfully generated and registered analyzer: {}", final_script_path.display());

        Ok(final_script_path)
    }

    /// Compile Rust analyzer to executable
    async fn compile_rust_analyzer(&self, source_path: &Path) -> Result<PathBuf> {
        log::info!("Compiling Rust analyzer: {}", source_path.display());

        let output_path = source_path.with_extension("");

        let output = Command::new("rustc")
            .arg(source_path)
            .arg("-o")
            .arg(&output_path)
            .arg("-C")
            .arg("opt-level=2")
            .output()
            .context("Failed to execute rustc")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to compile analyzer:\n{}", stderr);
        }

        Ok(output_path)
    }

    /// Execute an analyzer on a file
    pub async fn run_analyzer(&self, file_type: &str, file_path: &Path) -> Result<serde_json::Value> {
        let analyzer = self.db.get_analyzer(file_type)?
            .context("No analyzer found for this file type")?;

        log::info!("Running analyzer {} on {}", analyzer.script_path, file_path.display());

        let output = match analyzer.language.as_str() {
            "rust" => {
                // Execute compiled binary
                Command::new(&analyzer.script_path)
                    .arg(file_path)
                    .output()
                    .context("Failed to execute Rust analyzer")?
            }
            "python" => {
                Command::new("python3")
                    .arg(&analyzer.script_path)
                    .arg(file_path)
                    .output()
                    .context("Failed to execute Python analyzer")?
            }
            "shell" => {
                Command::new("sh")
                    .arg(&analyzer.script_path)
                    .arg(file_path)
                    .output()
                    .context("Failed to execute shell analyzer")?
            }
            _ => anyhow::bail!("Unsupported analyzer language: {}", analyzer.language),
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Analyzer failed:\n{}", stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let metadata: serde_json::Value = serde_json::from_str(&stdout)
            .context("Failed to parse analyzer output as JSON")?;

        Ok(metadata)
    }
}
