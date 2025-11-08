use anyhow::{Context, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FileContent {
    pub text: String,
    pub metadata: serde_json::Value,
}

impl FileContent {
    pub fn new(text: String) -> Self {
        Self {
            text,
            metadata: serde_json::json!({}),
        }
    }

    pub fn with_metadata(text: String, metadata: serde_json::Value) -> Self {
        Self { text, metadata }
    }
}

/// Extract text content from a file based on its type
pub async fn extract_text(path: &Path, bytes: &[u8], file_type: &str) -> Result<FileContent> {
    match file_type {
        // PDF files
        "application/pdf" => extract_pdf(bytes),

        // Word documents
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            extract_docx(bytes)
        }

        // Text-based formats
        t if t.starts_with("text/") => extract_text_file(bytes),

        // JSON
        "application/json" => extract_json(bytes),

        // Fallback: try to read as UTF-8 text
        _ => {
            if let Ok(text) = std::str::from_utf8(bytes) {
                Ok(FileContent::new(text.to_string()))
            } else {
                // Binary file, return empty content
                Ok(FileContent::new(String::new()))
            }
        }
    }
}

/// Extract text from PDF using pdf-extract
fn extract_pdf(bytes: &[u8]) -> Result<FileContent> {
    let text = pdf_extract::extract_text_from_mem(bytes)
        .context("Failed to extract text from PDF")?;

    // Try to extract basic metadata
    let metadata = serde_json::json!({
        "extracted_by": "pdf-extract"
    });

    Ok(FileContent::with_metadata(text, metadata))
}

/// Extract text from DOCX using docx-rs
fn extract_docx(bytes: &[u8]) -> Result<FileContent> {
    let docx = docx_rs::read_docx(bytes)
        .context("Failed to parse DOCX file")?;

    // Extract all text from paragraphs
    let mut text = String::new();

    for child in &docx.document.children {
        if let docx_rs::DocumentChild::Paragraph(para) = child {
            for child in &para.children {
                if let docx_rs::ParagraphChild::Run(run) = child {
                    for child in &run.children {
                        if let docx_rs::RunChild::Text(t) = child {
                            text.push_str(&t.text);
                        }
                    }
                }
            }
            text.push('\n');
        }
    }

    let metadata = serde_json::json!({
        "extracted_by": "docx-rs",
        "paragraph_count": docx.document.children.len()
    });

    Ok(FileContent::with_metadata(text, metadata))
}

/// Extract text from plain text files
fn extract_text_file(bytes: &[u8]) -> Result<FileContent> {
    let text = std::str::from_utf8(bytes)
        .context("Failed to decode text file as UTF-8")?
        .to_string();

    Ok(FileContent::new(text))
}

/// Extract and format JSON files
fn extract_json(bytes: &[u8]) -> Result<FileContent> {
    let text = std::str::from_utf8(bytes)
        .context("Failed to decode JSON file as UTF-8")?;

    // Try to parse and pretty-print JSON
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
        let pretty = serde_json::to_string_pretty(&json)
            .unwrap_or_else(|_| text.to_string());
        Ok(FileContent::new(pretty))
    } else {
        Ok(FileContent::new(text.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_extract_text_file() {
        let content = b"Hello, world!\nThis is a test.";
        let result = extract_text_file(content).unwrap();
        assert_eq!(result.text, "Hello, world!\nThis is a test.");
    }

    #[tokio::test]
    async fn test_extract_json() {
        let content = br#"{"key": "value", "number": 42}"#;
        let result = extract_json(content).unwrap();
        assert!(result.text.contains("key"));
        assert!(result.text.contains("value"));
    }
}
