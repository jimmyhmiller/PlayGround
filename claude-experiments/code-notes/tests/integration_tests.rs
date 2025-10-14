use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

#[derive(Debug, Deserialize)]
struct Scenario {
    name: String,
    description: String,
    steps: Vec<Step>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "action")]
enum Step {
    #[serde(rename = "create_file")]
    CreateFile { file: String, content: String },

    #[serde(rename = "modify_file")]
    ModifyFile { file: String, content: String },

    #[serde(rename = "delete_file")]
    DeleteFile { file: String },

    #[serde(rename = "git_commit")]
    GitCommit { message: String },

    #[serde(rename = "add_note")]
    AddNote {
        file: String,
        line: usize,
        column: usize,
        content: String,
        author: String,
        collection: String,
        store_id_as: Option<String>,
    },

    #[serde(rename = "update_note")]
    UpdateNote {
        note_id_var: String,
        content: String,
    },

    #[serde(rename = "delete_note")]
    DeleteNote {
        note_id_var: String,
        collection: String,
    },

    #[serde(rename = "view_note")]
    ViewNote {
        note_id_var: String,
    },

    #[serde(rename = "create_collection")]
    CreateCollection {
        name: String,
        description: Option<String>,
    },

    #[serde(rename = "export_collection")]
    ExportCollection {
        collection: String,
        output: String,
    },

    #[serde(rename = "import_collection")]
    ImportCollection {
        bundle: String,
    },

    #[serde(rename = "list_collections")]
    ListCollections,

    #[serde(rename = "list_orphaned")]
    ListOrphaned {
        collection: String,
    },

    #[serde(rename = "migrate_notes")]
    MigrateNotes { collection: String },

    #[serde(rename = "expect_note")]
    ExpectNote {
        content: String,
        file: Option<String>,
        line: Option<usize>,
        is_orphaned: bool,
        collection: Option<String>,
    },

    #[serde(rename = "expect_note_not_exists")]
    ExpectNoteNotExists {
        content: String,
        collection: Option<String>,
    },

    #[serde(rename = "expect_note_count")]
    ExpectNoteCount {
        count: usize,
        collection: Option<String>,
    },

    #[serde(rename = "expect_collection_exists")]
    ExpectCollectionExists {
        name: String,
    },

    #[serde(rename = "expect_migration_success")]
    ExpectMigrationSuccess {
        total: usize,
        successful: usize,
        failed: usize,
    },

    #[serde(rename = "inject_notes")]
    InjectNotes {
        file: String,
        collection: String,
    },

    #[serde(rename = "remove_inline_notes")]
    RemoveInlineNotes {
        file: String,
    },

    #[serde(rename = "extract_notes")]
    ExtractNotes {
        file: String,
        collection: Option<String>,
    },

    #[serde(rename = "expect_file_contains")]
    ExpectFileContains {
        file: String,
        content: String,
    },

    #[serde(rename = "expect_file_not_contains")]
    ExpectFileNotContains {
        file: String,
        content: String,
    },

    #[serde(rename = "expect_extracted_note_count")]
    ExpectExtractedNoteCount {
        count: usize,
    },

    #[serde(rename = "read_file")]
    ReadFile {
        file: String,
        store_as: String,
    },

    #[serde(rename = "capture_notes")]
    CaptureNotes {
        file: String,
        author: String,
        collection: String,
        metadata: Option<String>,
    },

    #[serde(rename = "expect_metadata")]
    ExpectMetadata {
        content: String,
        metadata_key: String,
        metadata_value: String,
        collection: Option<String>,
    },

    #[serde(rename = "expect_metadata_not_exists")]
    ExpectMetadataNotExists {
        content: String,
        metadata_key: String,
        collection: Option<String>,
    },
}

struct TestContext {
    #[allow(dead_code)]
    temp_dir: TempDir,
    repo_path: PathBuf,
    binary_path: PathBuf,
    note_ids: HashMap<String, String>,
    last_extracted_output: String,
    stored_files: HashMap<String, String>,
}

impl TestContext {
    fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let repo_path = temp_dir.path().to_path_buf();

        // Find the code-notes binary
        let binary_path = std::env::current_exe()?
            .parent()
            .ok_or_else(|| anyhow!("Could not find parent dir"))?
            .parent()
            .ok_or_else(|| anyhow!("Could not find parent dir"))?
            .join("code-notes");

        // Initialize git repo
        Command::new("git")
            .args(&["init"])
            .current_dir(&repo_path)
            .output()?;

        // Configure git
        Command::new("git")
            .args(&["config", "user.name", "Test User"])
            .current_dir(&repo_path)
            .output()?;

        Command::new("git")
            .args(&["config", "user.email", "test@example.com"])
            .current_dir(&repo_path)
            .output()?;

        // Initialize code-notes
        Command::new(&binary_path)
            .args(&["init"])
            .current_dir(&repo_path)
            .output()?;

        Ok(TestContext {
            temp_dir,
            repo_path,
            binary_path,
            note_ids: HashMap::new(),
            last_extracted_output: String::new(),
            stored_files: HashMap::new(),
        })
    }

    fn create_file(&self, file: &str, content: &str) -> Result<()> {
        let file_path = self.repo_path.join(file);

        // Create parent directories if needed
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(file_path, content)?;
        Ok(())
    }

    fn modify_file(&self, file: &str, content: &str) -> Result<()> {
        self.create_file(file, content)
    }

    fn delete_file(&self, file: &str) -> Result<()> {
        let file_path = self.repo_path.join(file);
        std::fs::remove_file(file_path)?;
        Ok(())
    }

    fn git_commit(&self, message: &str) -> Result<()> {
        // Add all files
        Command::new("git")
            .args(&["add", "-A"])
            .current_dir(&self.repo_path)
            .output()?;

        // Commit
        let output = Command::new("git")
            .args(&["commit", "-m", message])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Git commit failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(())
    }

    fn add_note(&mut self, file: &str, line: usize, column: usize, content: &str, author: &str, collection: &str, store_id_as: Option<&str>) -> Result<()> {
        let output = Command::new(&self.binary_path)
            .args(&[
                "add",
                "--file", file,
                "--line", &line.to_string(),
                "--column", &column.to_string(),
                "--content", content,
                "--author", author,
                "--collection", collection,
            ])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Add note failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        // Extract note ID from output if requested
        if let Some(var_name) = store_id_as {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Output format: "Added note <uuid> to collection '<name>'"
            if let Some(id_start) = stdout.find("Added note ") {
                let rest = &stdout[id_start + 11..];
                if let Some(id_end) = rest.find(" to collection") {
                    let note_id = rest[..id_end].to_string();
                    self.note_ids.insert(var_name.to_string(), note_id);
                }
            }
        }

        Ok(())
    }

    fn migrate_notes(&self, collection: &str) -> Result<String> {
        let output = Command::new(&self.binary_path)
            .args(&["migrate", "--collection", collection])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Migration failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn list_notes(&self, collection: Option<&str>) -> Result<String> {
        let mut args = vec!["list"];
        if let Some(coll) = collection {
            args.push("--collection");
            args.push(coll);
        }
        // Don't include deleted notes by default
        // This matches the default behavior of the list command

        let output = Command::new(&self.binary_path)
            .args(&args)
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("List notes failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn expect_note(&self, content: &str, file: Option<&str>, line: Option<usize>, is_orphaned: bool, collection: Option<&str>) -> Result<()> {
        let output = self.list_notes(collection)?;

        // Check that the note content exists
        if !output.contains(content) {
            return Err(anyhow!("Note with content '{}' not found in output:\n{}", content, output));
        }

        // Check file if specified
        if let Some(expected_file) = file {
            let lines: Vec<&str> = output.lines().collect();
            let mut found_content = false;

            for (i, line) in lines.iter().enumerate() {
                if line.contains(&format!("Content: {}", content)) {
                    found_content = true;

                    // Look backwards for the File: line
                    for j in (0..i).rev() {
                        if lines[j].starts_with("File:") {
                            if !lines[j].contains(expected_file) {
                                return Err(anyhow!("Note content '{}' found but file doesn't match. Expected '{}', got '{}'", content, expected_file, lines[j]));
                            }
                            break;
                        }
                    }
                    break;
                }
            }

            if !found_content {
                return Err(anyhow!("Could not verify file for note '{}'", content));
            }
        }

        // Check line number if specified
        if let Some(expected_line) = line {
            let lines: Vec<&str> = output.lines().collect();
            let mut found_content = false;

            for (i, line_text) in lines.iter().enumerate() {
                if line_text.contains(&format!("Content: {}", content)) {
                    found_content = true;

                    // Look backwards for the File: line which includes line number
                    for j in (0..i).rev() {
                        if lines[j].starts_with("File:") {
                            let line_str = format!(":{}", expected_line);
                            if !lines[j].contains(&line_str) {
                                return Err(anyhow!("Note content '{}' found but line number doesn't match. Expected '{}', line: {}", content, expected_line, lines[j]));
                            }
                            break;
                        }
                    }
                    break;
                }
            }

            if !found_content {
                return Err(anyhow!("Could not verify line number for note '{}'", content));
            }
        }

        // Check orphaned status
        let lines: Vec<&str> = output.lines().collect();
        let mut found_content = false;

        for (i, line) in lines.iter().enumerate() {
            if line.contains(&format!("Content: {}", content)) {
                found_content = true;

                // Check if next few lines contain ORPHANED marker
                let has_orphaned_marker = lines.iter().skip(i).take(3).any(|l| l.contains("ORPHANED"));

                if is_orphaned && !has_orphaned_marker {
                    return Err(anyhow!("Note '{}' should be orphaned but isn't", content));
                }
                if !is_orphaned && has_orphaned_marker {
                    return Err(anyhow!("Note '{}' should not be orphaned but is", content));
                }
                break;
            }
        }

        if !found_content {
            return Err(anyhow!("Could not verify orphaned status for note '{}'", content));
        }

        Ok(())
    }

    fn expect_migration_success(&self, migration_output: &str, total: usize, successful: usize, failed: usize) -> Result<()> {
        // Parse migration output
        if !migration_output.contains(&format!("Total: {}", total)) {
            return Err(anyhow!("Expected total: {}, output:\n{}", total, migration_output));
        }
        if !migration_output.contains(&format!("Successful: {}", successful)) {
            return Err(anyhow!("Expected successful: {}, output:\n{}", successful, migration_output));
        }
        if !migration_output.contains(&format!("Failed: {}", failed)) {
            return Err(anyhow!("Expected failed: {}, output:\n{}", failed, migration_output));
        }
        Ok(())
    }

    fn update_note(&self, note_id_var: &str, content: &str) -> Result<()> {
        let note_id = self.note_ids.get(note_id_var)
            .ok_or_else(|| anyhow!("Note ID variable '{}' not found", note_id_var))?;

        let output = Command::new(&self.binary_path)
            .args(&["update", note_id, "--content", content])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Update note failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(())
    }

    fn delete_note(&self, note_id_var: &str, collection: &str) -> Result<()> {
        let note_id = self.note_ids.get(note_id_var)
            .ok_or_else(|| anyhow!("Note ID variable '{}' not found", note_id_var))?;

        // Use hard-delete for tests to match expected behavior of complete removal
        let output = Command::new(&self.binary_path)
            .args(&["hard-delete", note_id, "--collection", collection])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Delete note failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(())
    }

    fn view_note(&self, note_id_var: &str) -> Result<String> {
        let note_id = self.note_ids.get(note_id_var)
            .ok_or_else(|| anyhow!("Note ID variable '{}' not found", note_id_var))?;

        let output = Command::new(&self.binary_path)
            .args(&["view", note_id])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("View note failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn create_collection(&self, name: &str, description: Option<&str>) -> Result<()> {
        let mut args = vec!["create-collection", name];
        if let Some(desc) = description {
            args.push("--description");
            args.push(desc);
        }

        let output = Command::new(&self.binary_path)
            .args(&args)
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Create collection failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(())
    }

    fn export_collection(&self, collection: &str, output_file: &str) -> Result<()> {
        let output_path = self.repo_path.join(output_file);
        let output = Command::new(&self.binary_path)
            .args(&["export", collection, "--output", &output_path.to_string_lossy()])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Export collection failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(())
    }

    fn import_collection(&self, bundle_file: &str) -> Result<()> {
        let bundle_path = self.repo_path.join(bundle_file);
        let output = Command::new(&self.binary_path)
            .args(&["import", &bundle_path.to_string_lossy()])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Import collection failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(())
    }

    fn list_collections(&self) -> Result<String> {
        let output = Command::new(&self.binary_path)
            .args(&["collections"])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("List collections failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn list_orphaned(&self, collection: &str) -> Result<String> {
        let output = Command::new(&self.binary_path)
            .args(&["orphaned", "--collection", collection])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("List orphaned failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn expect_note_not_exists(&self, content: &str, collection: Option<&str>) -> Result<()> {
        let output = self.list_notes(collection)?;

        if output.contains(content) {
            return Err(anyhow!("Note with content '{}' should not exist but was found", content));
        }

        Ok(())
    }

    fn expect_note_count(&self, expected_count: usize, collection: Option<&str>) -> Result<()> {
        let output = self.list_notes(collection)?;

        // Count occurrences of "ID: " which appears once per note
        let actual_count = output.matches("ID: ").count();

        if actual_count != expected_count {
            return Err(anyhow!("Expected {} notes, found {}", expected_count, actual_count));
        }

        Ok(())
    }

    fn expect_collection_exists(&self, name: &str) -> Result<()> {
        let output = self.list_collections()?;

        if !output.contains(name) {
            return Err(anyhow!("Collection '{}' not found in:\n{}", name, output));
        }

        Ok(())
    }

    fn inject_notes(&self, file: &str, collection: &str) -> Result<()> {
        let output = Command::new(&self.binary_path)
            .args(&["inject", file, "--collection", collection])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Inject notes failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(())
    }

    fn remove_inline_notes(&self, file: &str) -> Result<()> {
        let output = Command::new(&self.binary_path)
            .args(&["remove-inline", file])
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Remove inline notes failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(())
    }

    fn extract_notes(&mut self, file: &str, collection: Option<&str>) -> Result<()> {
        let mut args = vec!["extract", file];
        if let Some(coll) = collection {
            args.push("--collection");
            args.push(coll);
        }

        let output = Command::new(&self.binary_path)
            .args(&args)
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Extract notes failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        self.last_extracted_output = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(())
    }

    fn expect_file_contains(&self, file: &str, content: &str) -> Result<()> {
        let file_path = self.repo_path.join(file);
        let file_content = std::fs::read_to_string(&file_path)
            .map_err(|e| anyhow!("Could not read file {}: {}", file, e))?;

        if !file_content.contains(content) {
            return Err(anyhow!(
                "File '{}' does not contain expected content '{}'\nFile contents:\n{}",
                file, content, file_content
            ));
        }

        Ok(())
    }

    fn expect_file_not_contains(&self, file: &str, content: &str) -> Result<()> {
        let file_path = self.repo_path.join(file);
        let file_content = std::fs::read_to_string(&file_path)
            .map_err(|e| anyhow!("Could not read file {}: {}", file, e))?;

        if file_content.contains(content) {
            return Err(anyhow!(
                "File '{}' should not contain '{}' but it does\nFile contents:\n{}",
                file, content, file_content
            ));
        }

        Ok(())
    }

    fn expect_extracted_note_count(&self, expected_count: usize) -> Result<()> {
        // Count occurrences of "ID: " in extracted output
        let actual_count = self.last_extracted_output.matches("ID: ").count();

        if actual_count != expected_count {
            return Err(anyhow!(
                "Expected {} extracted notes, found {}\nExtracted output:\n{}",
                expected_count, actual_count, self.last_extracted_output
            ));
        }

        Ok(())
    }

    fn read_file(&mut self, file: &str, store_as: &str) -> Result<()> {
        let file_path = self.repo_path.join(file);
        let content = std::fs::read_to_string(&file_path)
            .map_err(|e| anyhow!("Could not read file {}: {}", file, e))?;

        self.stored_files.insert(store_as.to_string(), content);
        Ok(())
    }

    fn capture_notes(&self, file: &str, author: &str, collection: &str, metadata: Option<&str>) -> Result<()> {
        let mut args = vec!["capture", file, "--author", author, "--collection", collection];

        if let Some(meta) = metadata {
            args.push("--metadata");
            args.push(meta);
        }

        let output = Command::new(&self.binary_path)
            .args(&args)
            .current_dir(&self.repo_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("Capture notes failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        Ok(())
    }

    fn expect_metadata(&self, content: &str, metadata_key: &str, metadata_value: &str, collection: Option<&str>) -> Result<()> {
        let output = self.list_notes(collection)?;

        // Find the note with matching content
        let lines: Vec<&str> = output.lines().collect();
        let mut found_note = false;

        for (i, line) in lines.iter().enumerate() {
            if line.contains(&format!("Content: {}", content)) {
                found_note = true;

                // Look for the metadata section
                let mut metadata_found = false;
                for j in (i + 1)..(i + 50).min(lines.len()) {
                    // Check if we hit the next note or end of this note's section
                    if lines[j].starts_with("ID: ") && j != i + 1 {
                        break;
                    }

                    // Look for metadata
                    if lines[j].contains(&format!("\"{}\": {}", metadata_key, metadata_value))
                        || lines[j].contains(&format!("\"{}\": \"{}\"", metadata_key, metadata_value))
                        || lines[j].contains(&format!("\"{}\":\"{}\"", metadata_key, metadata_value))
                        || lines[j].contains(&format!("\"{}\":{}", metadata_key, metadata_value))
                    {
                        metadata_found = true;
                        break;
                    }
                }

                if !metadata_found {
                    return Err(anyhow!(
                        "Note with content '{}' found but metadata '{}': '{}' not found\nOutput:\n{}",
                        content, metadata_key, metadata_value, output
                    ));
                }
                break;
            }
        }

        if !found_note {
            return Err(anyhow!("Note with content '{}' not found", content));
        }

        Ok(())
    }

    fn expect_metadata_not_exists(&self, content: &str, metadata_key: &str, collection: Option<&str>) -> Result<()> {
        let output = self.list_notes(collection)?;

        // Find the note with matching content
        let lines: Vec<&str> = output.lines().collect();
        let mut found_note = false;

        for (i, line) in lines.iter().enumerate() {
            if line.contains(&format!("Content: {}", content)) {
                found_note = true;

                // Look for the metadata section
                for j in (i + 1)..(i + 50).min(lines.len()) {
                    // Check if we hit the next note
                    if lines[j].starts_with("ID: ") && j != i + 1 {
                        break;
                    }

                    // Look for the metadata key
                    if lines[j].contains(&format!("\"{}\":", metadata_key)) {
                        return Err(anyhow!(
                            "Note with content '{}' should not have metadata key '{}' but it does\nOutput:\n{}",
                            content, metadata_key, output
                        ));
                    }
                }
                break;
            }
        }

        if !found_note {
            return Err(anyhow!("Note with content '{}' not found", content));
        }

        Ok(())
    }
}

fn run_scenario(scenario_path: &Path) -> Result<()> {
    let scenario_content = std::fs::read_to_string(scenario_path)?;
    let scenario: Scenario = serde_json::from_str(&scenario_content)?;

    println!("\n=== Running scenario: {} ===", scenario.name);
    println!("Description: {}", scenario.description);

    let mut ctx = TestContext::new()?;
    let mut last_migration_output = String::new();

    for (i, step) in scenario.steps.iter().enumerate() {
        println!("  Step {}: {:?}", i + 1, step);

        match step {
            Step::CreateFile { file, content } => {
                ctx.create_file(file, content)?;
            }
            Step::ModifyFile { file, content } => {
                ctx.modify_file(file, content)?;
            }
            Step::DeleteFile { file } => {
                ctx.delete_file(file)?;
            }
            Step::GitCommit { message } => {
                ctx.git_commit(message)?;
            }
            Step::AddNote { file, line, column, content, author, collection, store_id_as } => {
                ctx.add_note(file, *line, *column, content, author, collection, store_id_as.as_deref())?;
            }
            Step::UpdateNote { note_id_var, content } => {
                ctx.update_note(note_id_var, content)?;
            }
            Step::DeleteNote { note_id_var, collection } => {
                ctx.delete_note(note_id_var, collection)?;
            }
            Step::ViewNote { note_id_var } => {
                let _output = ctx.view_note(note_id_var)?;
            }
            Step::CreateCollection { name, description } => {
                ctx.create_collection(name, description.as_deref())?;
            }
            Step::ExportCollection { collection, output } => {
                ctx.export_collection(collection, output)?;
            }
            Step::ImportCollection { bundle } => {
                ctx.import_collection(bundle)?;
            }
            Step::ListCollections => {
                let _output = ctx.list_collections()?;
            }
            Step::ListOrphaned { collection } => {
                let _output = ctx.list_orphaned(collection)?;
            }
            Step::MigrateNotes { collection } => {
                last_migration_output = ctx.migrate_notes(collection)?;
            }
            Step::ExpectNote { content, file, line, is_orphaned, collection } => {
                ctx.expect_note(content, file.as_deref(), *line, *is_orphaned, collection.as_deref())?;
            }
            Step::ExpectNoteNotExists { content, collection } => {
                ctx.expect_note_not_exists(content, collection.as_deref())?;
            }
            Step::ExpectNoteCount { count, collection } => {
                ctx.expect_note_count(*count, collection.as_deref())?;
            }
            Step::ExpectCollectionExists { name } => {
                ctx.expect_collection_exists(name)?;
            }
            Step::ExpectMigrationSuccess { total, successful, failed } => {
                ctx.expect_migration_success(&last_migration_output, *total, *successful, *failed)?;
            }
            Step::InjectNotes { file, collection } => {
                ctx.inject_notes(file, collection)?;
            }
            Step::RemoveInlineNotes { file } => {
                ctx.remove_inline_notes(file)?;
            }
            Step::ExtractNotes { file, collection } => {
                ctx.extract_notes(file, collection.as_deref())?;
            }
            Step::ExpectFileContains { file, content } => {
                ctx.expect_file_contains(file, content)?;
            }
            Step::ExpectFileNotContains { file, content } => {
                ctx.expect_file_not_contains(file, content)?;
            }
            Step::ExpectExtractedNoteCount { count } => {
                ctx.expect_extracted_note_count(*count)?;
            }
            Step::ReadFile { file, store_as } => {
                ctx.read_file(file, store_as)?;
            }
            Step::CaptureNotes { file, author, collection, metadata } => {
                ctx.capture_notes(file, author, collection, metadata.as_deref())?;
            }
            Step::ExpectMetadata { content, metadata_key, metadata_value, collection } => {
                ctx.expect_metadata(content, metadata_key, metadata_value, collection.as_deref())?;
            }
            Step::ExpectMetadataNotExists { content, metadata_key, collection } => {
                ctx.expect_metadata_not_exists(content, metadata_key, collection.as_deref())?;
            }
        }
    }

    println!("  ✓ Scenario passed!");
    Ok(())
}

#[test]
fn test_all_scenarios() {
    let scenarios_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/scenarios");

    let mut scenario_files: Vec<_> = std::fs::read_dir(&scenarios_dir)
        .expect("Could not read scenarios directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension().and_then(|s| s.to_str()) == Some("json")
        })
        .map(|entry| entry.path())
        .collect();

    scenario_files.sort();

    if scenario_files.is_empty() {
        panic!("No scenario files found in {:?}", scenarios_dir);
    }

    let mut passed = 0;
    let mut failed = 0;

    for scenario_file in &scenario_files {
        match run_scenario(scenario_file) {
            Ok(_) => passed += 1,
            Err(e) => {
                eprintln!("\n❌ Scenario failed: {:?}", scenario_file);
                eprintln!("Error: {}", e);
                failed += 1;
            }
        }
    }

    println!("\n=== Test Summary ===");
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);

    if failed > 0 {
        panic!("{} scenario(s) failed", failed);
    }
}
