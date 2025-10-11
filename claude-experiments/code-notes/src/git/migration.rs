use anyhow::{anyhow, Result};
use std::path::Path;

use crate::git::GitRepo;
use crate::models::{MigrationRecord, Note, NoteAnchor};
use crate::parsers::{AnchorBuilder, AnchorMatcher, CodeParser, LanguageRegistry, LanguageInstaller, GrammarSource};

/// Threshold for considering a match valid (0.0 to 1.0)
const MATCH_CONFIDENCE_THRESHOLD: f64 = 0.7;

/// Handles migration of notes across git commits
pub struct NoteMigrator {
    repo: GitRepo,
}

impl NoteMigrator {
    pub fn new(repo: GitRepo) -> Self {
        Self { repo }
    }

    /// Migrate a note from its original commit to the current commit
    pub fn migrate_note(&self, note: &mut Note) -> Result<()> {
        let current_commit = self.repo.current_commit_hash()?;

        // If already at current commit, nothing to do
        if note.anchor.commit_hash == current_commit {
            note.anchor.is_valid = true;
            return Ok(());
        }

        // Get commits between note's commit and current
        let commits = self
            .repo
            .commits_between(&note.anchor.commit_hash, &current_commit)?;

        // If no intermediate commits, try direct migration
        if commits.is_empty() {
            return self.migrate_note_direct(note, &current_commit);
        }

        // Try to migrate through each commit
        let mut current_anchor = note.anchor.clone();

        for commit in commits.iter().rev() {
            // Check if file changed in this commit
            let file_changed = self.repo.file_changed_between(
                &current_anchor.commit_hash,
                commit,
                &current_anchor.primary.file_path,
            )?;

            if !file_changed {
                // File didn't change, just update commit hash
                current_anchor.commit_hash = commit.clone();
                continue;
            }

            // File changed, need to find new anchor
            match self.migrate_anchor_to_commit(&current_anchor, commit) {
                Ok((new_anchor, confidence)) => {
                    let record = MigrationRecord {
                        from_commit: current_anchor.commit_hash.clone(),
                        to_commit: commit.clone(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs() as i64,
                        success: true,
                        confidence: Some(confidence),
                    };

                    current_anchor = new_anchor;
                    current_anchor.migration_history.push(record);
                }
                Err(_) => {
                    // Migration failed
                    let record = MigrationRecord {
                        from_commit: current_anchor.commit_hash.clone(),
                        to_commit: commit.clone(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs() as i64,
                        success: false,
                        confidence: None,
                    };

                    note.anchor.migration_history.push(record);
                    note.mark_orphaned();
                    return Err(anyhow!("Failed to migrate note to commit {}", commit));
                }
            }
        }

        // Successfully migrated through all commits
        note.anchor = current_anchor;
        note.anchor.is_valid = true;
        Ok(())
    }

    /// Attempt direct migration to target commit
    fn migrate_note_direct(&self, note: &mut Note, target_commit: &str) -> Result<()> {
        match self.migrate_anchor_to_commit(&note.anchor, target_commit) {
            Ok((new_anchor, confidence)) => {
                let record = MigrationRecord {
                    from_commit: note.anchor.commit_hash.clone(),
                    to_commit: target_commit.to_string(),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as i64,
                    success: true,
                    confidence: Some(confidence),
                };

                note.anchor = new_anchor;
                note.anchor.migration_history.push(record);
                note.anchor.is_valid = true;
                Ok(())
            }
            Err(e) => {
                let record = MigrationRecord {
                    from_commit: note.anchor.commit_hash.clone(),
                    to_commit: target_commit.to_string(),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as i64,
                    success: false,
                    confidence: None,
                };

                note.anchor.migration_history.push(record);
                note.mark_orphaned();
                Err(e)
            }
        }
    }

    /// Migrate an anchor to a specific commit
    fn migrate_anchor_to_commit(
        &self,
        anchor: &NoteAnchor,
        target_commit: &str,
    ) -> Result<(NoteAnchor, f64)> {
        // Get file content at target commit
        let file_content = self
            .repo
            .file_content_at_commit(target_commit, &anchor.primary.file_path)?;

        // Detect language
        let extension = Path::new(&anchor.primary.file_path)
            .extension()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Could not determine file extension"))?;

        // Create language registry
        let mut registry = LanguageRegistry::new()?;
        registry.initialize()?;

        // Auto-install language if needed
        Self::auto_install_language_if_needed(extension, &registry)?;

        // Re-initialize registry to pick up newly installed language
        registry = LanguageRegistry::new()?;
        registry.initialize()?;

        // Create parser
        let mut parser = CodeParser::from_extension(extension, &mut registry)?;
        let tree = parser.parse(&file_content)?;

        // Try to find matching node
        let matcher = AnchorMatcher::new(&file_content, &tree);
        let (matched_node, confidence) = matcher
            .find_match(&anchor.primary)
            .ok_or_else(|| anyhow!("Could not find matching node"))?;

        if confidence < MATCH_CONFIDENCE_THRESHOLD {
            return Err(anyhow!(
                "Match confidence too low: {} < {}",
                confidence,
                MATCH_CONFIDENCE_THRESHOLD
            ));
        }

        // Build new anchor
        let builder = AnchorBuilder::new(&file_content, anchor.primary.file_path.clone());
        let new_anchor = builder.build_note_anchor(matched_node, target_commit.to_string())?;

        Ok((new_anchor, confidence))
    }

    /// Migrate all notes in a collection
    pub fn migrate_collection(&self, notes: &mut [Note]) -> Result<MigrationReport> {
        let mut report = MigrationReport::default();

        for note in notes.iter_mut() {
            match self.migrate_note(note) {
                Ok(_) => report.successful += 1,
                Err(_) => report.failed += 1,
            }
            report.total += 1;
        }

        Ok(report)
    }

    /// Auto-install a language if it's not already installed (for migration)
    fn auto_install_language_if_needed(extension: &str, registry: &LanguageRegistry) -> Result<()> {
        // Try to get language name from extension via registry first
        if let Ok(lang_name) = registry.language_from_extension(extension) {
            // Check if already installed
            if registry.is_installed(&lang_name).unwrap_or(false) {
                return Ok(()); // Already installed
            }

            // Check if this language is available for installation
            if GrammarSource::find_by_name(&lang_name).is_some() {
                eprintln!("Language '{}' not installed. Installing automatically for migration...", lang_name);

                if let Ok(mut installer) = LanguageInstaller::new() {
                    match installer.install(&lang_name) {
                        Ok(_) => {
                            eprintln!("✓ Successfully installed {}", lang_name);
                        }
                        Err(e) => {
                            eprintln!("⚠ Failed to auto-install {}: {}", lang_name, e);
                            eprintln!("  You can install it manually with: code-notes lang install {}", lang_name);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct MigrationReport {
    pub total: usize,
    pub successful: usize,
    pub failed: usize,
}

impl MigrationReport {
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.successful as f64 / self.total as f64
    }
}
