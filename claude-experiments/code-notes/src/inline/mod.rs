use anyhow::{anyhow, Result};
use std::fs;
use std::path::Path;
use uuid::Uuid;

use crate::models::Note;
use crate::storage::NoteStorage;

/// Get the single-line comment prefix for a language based on file extension
pub fn comment_prefix_for_extension(ext: &str) -> Result<&'static str> {
    match ext {
        "rs" | "js" | "jsx" | "ts" | "tsx" | "java" | "c" | "cpp" | "cc" | "h" | "hpp" | "go" => Ok("//"),
        "py" | "rb" | "sh" => Ok("#"),
        _ => Err(anyhow!("Unsupported file extension: {}", ext)),
    }
}

/// Inject notes into a source file as inline comments
/// Inserts comments at the line specified in each note's anchor
pub fn inject_notes(file_path: &Path, notes: &[Note]) -> Result<String> {
    let content = fs::read_to_string(file_path)?;
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| anyhow!("File has no extension"))?;
    let comment_prefix = comment_prefix_for_extension(ext)?;

    let mut lines: Vec<String> = content.lines().map(String::from).collect();

    // Sort notes by line number in reverse order so we can insert from bottom to top
    // This prevents line number shifting issues
    let mut sorted_notes = notes.to_vec();
    sorted_notes.sort_by(|a, b| b.anchor.primary.line_number.cmp(&a.anchor.primary.line_number));

    for note in sorted_notes {
        let line_num = note.anchor.primary.line_number;

        // Validate line number
        if line_num >= lines.len() {
            eprintln!(
                "Warning: Note {} references line {} but file only has {} lines",
                note.id,
                line_num,
                lines.len()
            );
            continue;
        }

        // Get the indentation of the target line
        let target_line = &lines[line_num];
        let indent = target_line
            .chars()
            .take_while(|c| c.is_whitespace())
            .collect::<String>();

        // Create the note comment block
        let mut comment_lines = Vec::new();
        comment_lines.push(format!("{}{} @code-note[{}]", indent, comment_prefix, note.id));

        for content_line in note.content.lines() {
            comment_lines.push(format!("{}{} {}", indent, comment_prefix, content_line));
        }

        comment_lines.push(format!("{}{} @end-code-note", indent, comment_prefix));

        // Insert the comment block before the target line
        for (i, comment_line) in comment_lines.iter().enumerate() {
            lines.insert(line_num + i, comment_line.clone());
        }
    }

    Ok(lines.join("\n") + "\n")
}

/// Extract notes from inline comments in a source file
/// Returns a vector of notes found in the file
///
/// This function searches through the specified collections (or all collections if None)
/// to find notes matching the IDs embedded in the inline comments.
pub fn extract_notes(file_path: &Path, storage: &NoteStorage, collection_names: Option<Vec<String>>) -> Result<Vec<Note>> {
    let content = fs::read_to_string(file_path)?;
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| anyhow!("File has no extension"))?;
    let comment_prefix = comment_prefix_for_extension(ext)?;

    let mut notes = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        // Check for start marker
        if line.starts_with(comment_prefix) && line.contains("@code-note[") {
            if let Some(note) = extract_single_note(&lines, i, comment_prefix, storage, collection_names.as_ref())? {
                notes.push(note);
            }
        }

        i += 1;
    }

    Ok(notes)
}

/// Extract a single note starting from the given line index
fn extract_single_note(
    lines: &[&str],
    start_idx: usize,
    comment_prefix: &str,
    storage: &NoteStorage,
    collection_names: Option<&Vec<String>>,
) -> Result<Option<Note>> {
    let start_line = lines[start_idx].trim();

    // Extract UUID from the start marker
    let uuid_start = start_line
        .find("@code-note[")
        .ok_or_else(|| anyhow!("Invalid note start marker"))?
        + "@code-note[".len();
    let uuid_end = start_line[uuid_start..]
        .find(']')
        .ok_or_else(|| anyhow!("Invalid note start marker - missing ]"))?
        + uuid_start;
    let note_id_str = &start_line[uuid_start..uuid_end];

    // Parse the UUID
    let note_id = Uuid::parse_str(note_id_str)
        .map_err(|e| anyhow!("Invalid UUID '{}': {}", note_id_str, e))?;

    // Collect content lines until we find the end marker (for validation)
    let mut i = start_idx + 1;
    let mut found_end = false;

    while i < lines.len() {
        let line = lines[i].trim();

        if line.starts_with(comment_prefix) && line.contains("@end-code-note") {
            found_end = true;
            break;
        }

        if !line.starts_with(comment_prefix) {
            // Non-comment line means the note block ended unexpectedly
            break;
        }

        i += 1;
    }

    if !found_end {
        return Err(anyhow!(
            "Note {} is missing end marker @end-code-note",
            note_id
        ));
    }

    // Load the note from storage
    // Try to load directly first (fastest path)
    if let Ok(note) = storage.load_note(note_id) {
        return Ok(Some(note));
    }

    // If direct load fails, search through collections
    let collections_to_search = if let Some(names) = collection_names {
        names.clone()
    } else {
        // Search all collections
        storage.list_collections()?
    };

    for collection_name in collections_to_search {
        if let Ok(collection) = storage.load_collection(&collection_name) {
            if let Some(note) = collection.get_note(note_id) {
                return Ok(Some(note.clone()));
            }
        }
    }

    // Note not found in storage
    eprintln!("Warning: Note {} found in inline comments but not in storage", note_id);
    Ok(None)
}

/// Remove all inline note comments from a source file
pub fn remove_notes(file_path: &Path) -> Result<String> {
    let content = fs::read_to_string(file_path)?;
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| anyhow!("File has no extension"))?;
    let comment_prefix = comment_prefix_for_extension(ext)?;

    let lines: Vec<&str> = content.lines().collect();
    let mut output_lines = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        // Check for start marker
        if line.starts_with(comment_prefix) && line.contains("@code-note[") {
            // Skip until we find the end marker
            i += 1;
            while i < lines.len() {
                let inner_line = lines[i].trim();
                if inner_line.starts_with(comment_prefix) && inner_line.contains("@end-code-note") {
                    break;
                }
                i += 1;
            }
            i += 1; // Skip the end marker line
            continue;
        }

        output_lines.push(lines[i]);
        i += 1;
    }

    Ok(output_lines.join("\n") + "\n")
}

/// Captured note information from @note: markers
#[derive(Debug, Clone)]
pub struct CapturedNote {
    pub line_number: usize,
    pub content: String,
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// Scan a file for @note: markers and extract their content
pub fn scan_for_note_markers(file_path: &Path) -> Result<Vec<CapturedNote>> {
    let content = fs::read_to_string(file_path)?;
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| anyhow!("File has no extension"))?;
    let comment_prefix = comment_prefix_for_extension(ext)?;

    let mut captured_notes = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        // Check for @note: marker
        if line.starts_with(comment_prefix) && line.contains("@note:") {
            // Extract content after @note:
            let marker_start = line.find("@note:").unwrap() + "@note:".len();
            let first_line_content = line[marker_start..].trim();

            let mut note_content = vec![first_line_content.to_string()];
            let mut metadata_json: Option<String> = None;
            let start_line = i;

            // Check for continuation lines (same indentation + comment prefix, no @note:)
            i += 1;
            while i < lines.len() {
                let next_line = lines[i].trim();

                // Stop if we hit a non-comment line or another @note: marker
                if !next_line.starts_with(comment_prefix) || next_line.contains("@note:") {
                    break;
                }

                // Extract content after comment prefix
                let content_after_prefix = next_line[comment_prefix.len()..].trim();

                // Stop if this looks like code or another marker
                if content_after_prefix.is_empty()
                    || content_after_prefix.starts_with("@code-note")
                    || content_after_prefix.starts_with("@end-code-note") {
                    break;
                }

                // Check for @meta: marker
                if content_after_prefix.starts_with("@meta:") {
                    let meta_start = content_after_prefix.find("@meta:").unwrap() + "@meta:".len();
                    metadata_json = Some(content_after_prefix[meta_start..].trim().to_string());
                    i += 1;
                    continue;
                }

                note_content.push(content_after_prefix.to_string());
                i += 1;
            }

            // Parse metadata if present
            let metadata = if let Some(json_str) = metadata_json {
                match serde_json::from_str(&json_str) {
                    Ok(meta) => Some(meta),
                    Err(e) => {
                        eprintln!("Warning: Invalid metadata JSON at line {}: {}", start_line + 1, e);
                        None
                    }
                }
            } else {
                None
            };

            captured_notes.push(CapturedNote {
                line_number: start_line,
                content: note_content.join("\n"),
                metadata,
            });

            continue;
        }

        i += 1;
    }

    Ok(captured_notes)
}

/// Replace @note: markers with full @code-note format
pub fn replace_note_markers_with_full_notes(
    file_path: &Path,
    notes_with_ids: &[(CapturedNote, uuid::Uuid)],
) -> Result<String> {
    let content = fs::read_to_string(file_path)?;
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| anyhow!("File has no extension"))?;
    let comment_prefix = comment_prefix_for_extension(ext)?;

    let mut lines: Vec<String> = content.lines().map(String::from).collect();

    // Sort in reverse order so we can modify from bottom to top
    let mut sorted_notes = notes_with_ids.to_vec();
    sorted_notes.sort_by(|a, b| b.0.line_number.cmp(&a.0.line_number));

    for (captured, uuid) in sorted_notes {
        let line_num = captured.line_number;

        if line_num >= lines.len() {
            continue;
        }

        // Get the indentation of the line
        let target_line = &lines[line_num];
        let indent = target_line
            .chars()
            .take_while(|c| c.is_whitespace())
            .collect::<String>();

        // Find how many lines belong to this note
        let mut lines_to_replace = 1;
        for i in (line_num + 1)..lines.len() {
            let next_line = lines[i].trim();
            if !next_line.starts_with(comment_prefix) || next_line.contains("@note:") {
                break;
            }
            let content_after_prefix = next_line[comment_prefix.len()..].trim();
            if content_after_prefix.is_empty()
                || content_after_prefix.starts_with("@code-note")
                || content_after_prefix.starts_with("@end-code-note") {
                break;
            }
            lines_to_replace += 1;
        }

        // Remove the old @note: lines
        for _ in 0..lines_to_replace {
            lines.remove(line_num);
        }

        // Insert the new formatted note
        let mut new_lines = Vec::new();
        new_lines.push(format!("{}{} @code-note[{}]", indent, comment_prefix, uuid));

        for content_line in captured.content.lines() {
            new_lines.push(format!("{}{} {}", indent, comment_prefix, content_line));
        }

        new_lines.push(format!("{}{} @end-code-note", indent, comment_prefix));

        // Insert in reverse order
        for new_line in new_lines.into_iter().rev() {
            lines.insert(line_num, new_line);
        }
    }

    Ok(lines.join("\n") + "\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CodeAnchor, Note, NoteAnchor};
    use std::collections::HashMap;
    use std::io::Write;

    fn create_test_note(id: &str, content: &str, line: usize) -> Note {
        use uuid::Uuid;
        Note {
            id: Uuid::parse_str(id).unwrap_or_else(|_| Uuid::new_v4()),
            content: content.to_string(),
            author: "test".to_string(),
            created_at: 0,
            updated_at: 0,
            anchor: NoteAnchor {
                primary: CodeAnchor {
                    file_path: "test.rs".to_string(),
                    line_number: line,
                    column: 0,
                    node_kind: "identifier".to_string(),
                    node_text: "test".to_string(),
                    ast_path: vec![],
                    context: vec![],
                    semantic_id: None,
                    normalized_text: "test".to_string(),
                },
                alternatives: vec![],
                commit_hash: "abc123".to_string(),
                is_valid: true,
                migration_history: vec![],
            },
            collection: "test".to_string(),
            metadata: HashMap::new(),
            is_orphaned: false,
            deleted: false,
        }
    }

    #[test]
    fn test_comment_prefix_for_extension() {
        assert_eq!(comment_prefix_for_extension("rs").unwrap(), "//");
        assert_eq!(comment_prefix_for_extension("py").unwrap(), "#");
        assert_eq!(comment_prefix_for_extension("js").unwrap(), "//");
        assert!(comment_prefix_for_extension("unknown").is_err());
    }

    #[test]
    fn test_inject_notes_rust() -> Result<()> {
        let mut file = tempfile::Builder::new().suffix(".rs").tempfile()?;
        let original_content = "fn main() {\n    println!(\"Hello\");\n}\n";
        write!(file, "{}", original_content)?;

        let test_uuid = "12345678-1234-1234-1234-123456789abc";
        let note = create_test_note(test_uuid, "This is a test note", 0);
        let result = inject_notes(file.path(), &[note])?;

        assert!(result.contains(&format!("@code-note[{}]", test_uuid)));
        assert!(result.contains("This is a test note"));
        assert!(result.contains("@end-code-note"));
        assert!(result.contains("fn main()"));

        Ok(())
    }

    #[test]
    fn test_inject_multiline_note() -> Result<()> {
        let mut file = tempfile::Builder::new().suffix(".rs").tempfile()?;
        write!(file, "fn test() {{\n}}\n")?;

        let note = create_test_note(
            "12345678-1234-1234-1234-123456789abc",
            "Line 1\nLine 2\nLine 3",
            0,
        );
        let result = inject_notes(file.path(), &[note])?;

        assert!(result.contains("// Line 1"));
        assert!(result.contains("// Line 2"));
        assert!(result.contains("// Line 3"));

        Ok(())
    }

    #[test]
    fn test_inject_preserves_indentation() -> Result<()> {
        let mut file = tempfile::Builder::new().suffix(".rs").tempfile()?;
        write!(file, "fn main() {{\n    let x = 42;\n}}\n")?;

        let test_uuid = "12345678-1234-1234-1234-123456789abc";
        let note = create_test_note(test_uuid, "Indented note", 1);
        let result = inject_notes(file.path(), &[note])?;

        // The note should be indented to match the target line
        assert!(result.contains(&format!("    // @code-note[{}]", test_uuid)));
        assert!(result.contains("    // Indented note"));

        Ok(())
    }

    #[test]
    fn test_remove_notes() -> Result<()> {
        let mut file = tempfile::Builder::new().suffix(".rs").tempfile()?;
        let content = "fn main() {\n// @code-note[test-123]\n// This is a note\n// @end-code-note\n    println!(\"Hello\");\n}\n";
        write!(file, "{}", content)?;

        let result = remove_notes(file.path())?;

        assert!(!result.contains("@code-note"));
        assert!(!result.contains("@end-code-note"));
        assert!(!result.contains("This is a note"));
        assert!(result.contains("fn main()"));
        assert!(result.contains("println!"));

        Ok(())
    }

    #[test]
    fn test_remove_multiple_notes() -> Result<()> {
        let mut file = tempfile::Builder::new().suffix(".rs").tempfile()?;
        let content = "// @code-note[note1]\n// Note 1\n// @end-code-note\nfn test() {}\n// @code-note[note2]\n// Note 2\n// @end-code-note\nfn main() {}\n";
        write!(file, "{}", content)?;

        let result = remove_notes(file.path())?;

        assert!(!result.contains("@code-note"));
        assert!(!result.contains("Note 1"));
        assert!(!result.contains("Note 2"));
        assert!(result.contains("fn test()"));
        assert!(result.contains("fn main()"));

        Ok(())
    }

    #[test]
    fn test_inject_multiple_notes() -> Result<()> {
        let mut file = tempfile::Builder::new().suffix(".rs").tempfile()?;
        write!(file, "fn test1() {{}}\nfn test2() {{}}\nfn test3() {{}}\n")?;

        let uuid1 = "11111111-1111-1111-1111-111111111111";
        let uuid2 = "22222222-2222-2222-2222-222222222222";
        let uuid3 = "33333333-3333-3333-3333-333333333333";

        let notes = vec![
            create_test_note(uuid1, "First note", 0),
            create_test_note(uuid2, "Second note", 1),
            create_test_note(uuid3, "Third note", 2),
        ];

        let result = inject_notes(file.path(), &notes)?;

        assert!(result.contains(&format!("@code-note[{}]", uuid1)));
        assert!(result.contains(&format!("@code-note[{}]", uuid2)));
        assert!(result.contains(&format!("@code-note[{}]", uuid3)));
        assert!(result.contains("First note"));
        assert!(result.contains("Second note"));
        assert!(result.contains("Third note"));

        Ok(())
    }

    #[test]
    fn test_roundtrip_inject_remove() -> Result<()> {
        let mut file = tempfile::Builder::new().suffix(".rs").tempfile()?;
        let original = "fn main() {\n    println!(\"test\");\n}\n";
        write!(file, "{}", original)?;

        let note = create_test_note("12345678-1234-1234-1234-123456789abc", "Test note", 0);
        let injected = inject_notes(file.path(), &[note])?;

        // Write injected content back
        fs::write(file.path(), &injected)?;

        // Now remove notes
        let removed = remove_notes(file.path())?;

        // Should be back to original (minus potential whitespace differences)
        assert_eq!(removed.trim(), original.trim());

        Ok(())
    }
}
