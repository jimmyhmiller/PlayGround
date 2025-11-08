use anyhow::{Context, Result};
use std::path::Path;
use sha2::{Sha256, Digest};

mod extractors;
pub mod dynamic;

pub use extractors::{extract_text, FileContent};
pub use dynamic::DynamicAnalyzer;

/// Analyze a file and extract its content and metadata
pub async fn analyze_file(path: &Path) -> Result<FileAnalysis> {
    let metadata = std::fs::metadata(path)
        .context("Failed to read file metadata")?;

    let file_bytes = std::fs::read(path)
        .context("Failed to read file")?;

    // Compute content hash
    let mut hasher = Sha256::new();
    hasher.update(&file_bytes);
    let content_hash = format!("{:x}", hasher.finalize());

    // Detect file type
    let file_type = detect_file_type(&file_bytes, path);

    // Extract text content
    let content = extract_text(path, &file_bytes, &file_type).await?;

    let modified_time = metadata.modified()
        .context("Failed to get modification time")?
        .duration_since(std::time::UNIX_EPOCH)
        .context("Failed to convert modification time")?
        .as_secs() as i64;

    Ok(FileAnalysis {
        path: path.to_string_lossy().to_string(),
        content_hash,
        size_bytes: metadata.len() as i64,
        modified_time,
        file_type,
        content,
    })
}

#[derive(Debug, Clone)]
pub struct FileAnalysis {
    pub path: String,
    pub content_hash: String,
    pub size_bytes: i64,
    pub modified_time: i64,
    pub file_type: String,
    pub content: FileContent,
}

/// Detect file type using infer library and file extension fallback
fn detect_file_type(bytes: &[u8], path: &Path) -> String {
    // Try infer first (magic bytes)
    if let Some(kind) = infer::get(bytes) {
        return kind.mime_type().to_string();
    }

    // Fall back to extension-based detection
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext.to_lowercase().as_str() {
            "txt" => "text/plain".to_string(),
            "md" => "text/markdown".to_string(),
            "json" => "application/json".to_string(),
            "xml" => "application/xml".to_string(),
            "html" | "htm" => "text/html".to_string(),
            "csv" => "text/csv".to_string(),
            "rs" => "text/x-rust".to_string(),
            "py" => "text/x-python".to_string(),
            "js" => "text/javascript".to_string(),
            "ts" => "text/typescript".to_string(),
            "java" => "text/x-java".to_string(),
            "c" => "text/x-c".to_string(),
            "cpp" | "cc" | "cxx" => "text/x-c++".to_string(),
            "h" | "hpp" => "text/x-c++-hdr".to_string(),
            "go" => "text/x-go".to_string(),
            _ => format!("application/x-{}", ext),
        }
    } else {
        "application/octet-stream".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_file_type() {
        // Test PDF magic bytes
        let pdf_bytes = b"%PDF-1.4\n";
        let path = Path::new("test.pdf");
        assert_eq!(detect_file_type(pdf_bytes, path), "application/pdf");

        // Test extension fallback
        let unknown_bytes = b"some random content";
        let txt_path = Path::new("test.txt");
        assert_eq!(detect_file_type(unknown_bytes, txt_path), "text/plain");
    }
}
