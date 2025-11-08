use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};
use chrono::Utc;
use serde::{Deserialize, Serialize};

const SCHEMA_SQL: &str = include_str!("schema.sql");

#[derive(Debug, Clone)]
pub struct Database {
    path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub id: i64,
    pub path: String,
    pub content_hash: String,
    pub size_bytes: i64,
    pub modified_time: i64,
    pub file_type: Option<String>,
    pub content_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub id: i64,
    pub file_id: i64,
    pub llm_summary: Option<String>,
    pub tags: Vec<String>,
    pub categories: Vec<String>,
    pub entities: serde_json::Value,
    pub analyzed_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct View {
    pub id: i64,
    pub name: String,
    pub query_prompt: String,
    pub sql_query: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Analyzer {
    pub id: i64,
    pub file_type: String,
    pub script_path: String,
    pub language: String,
    pub description: Option<String>,
}

impl Database {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create database directory")?;
        }

        let db = Self { path };
        db.init()?;
        Ok(db)
    }

    fn connect(&self) -> Result<Connection> {
        let conn = Connection::open(&self.path)
            .context("Failed to open database connection")?;

        // Enable WAL mode for better concurrent access
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        Ok(conn)
    }

    fn init(&self) -> Result<()> {
        let conn = self.connect()?;
        conn.execute_batch(SCHEMA_SQL)
            .context("Failed to initialize database schema")?;
        Ok(())
    }

    // File operations
    pub fn insert_file(&self, path: &str, hash: &str, size: i64, modified: i64, file_type: Option<&str>, content: Option<&str>) -> Result<i64> {
        let conn = self.connect()?;

        conn.execute(
            "INSERT INTO files (path, content_hash, size_bytes, modified_time, file_type, content_text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(path) DO UPDATE SET
                content_hash = ?2,
                size_bytes = ?3,
                modified_time = ?4,
                file_type = ?5,
                content_text = ?6,
                updated_at = strftime('%s', 'now')",
            params![path, hash, size, modified, file_type, content],
        )?;

        // Get the actual file ID (works for both INSERT and UPDATE)
        let file_id: i64 = conn.query_row(
            "SELECT id FROM files WHERE path = ?1",
            params![path],
            |row| row.get(0)
        )?;

        Ok(file_id)
    }

    pub fn get_file_by_path(&self, path: &str) -> Result<Option<FileRecord>> {
        let conn = self.connect()?;

        let mut stmt = conn.prepare(
            "SELECT id, path, content_hash, size_bytes, modified_time, file_type, content_text
             FROM files WHERE path = ?1"
        )?;

        let mut rows = stmt.query(params![path])?;

        if let Some(row) = rows.next()? {
            Ok(Some(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                size_bytes: row.get(3)?,
                modified_time: row.get(4)?,
                file_type: row.get(5)?,
                content_text: row.get(6)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn get_all_files(&self) -> Result<Vec<FileRecord>> {
        let conn = self.connect()?;

        let mut stmt = conn.prepare(
            "SELECT id, path, content_hash, size_bytes, modified_time, file_type, content_text
             FROM files ORDER BY modified_time DESC"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                size_bytes: row.get(3)?,
                modified_time: row.get(4)?,
                file_type: row.get(5)?,
                content_text: row.get(6)?,
            })
        })?;

        rows.collect::<Result<Vec<_>, _>>().context("Failed to fetch files")
    }

    pub fn delete_file(&self, path: &str) -> Result<()> {
        let conn = self.connect()?;
        conn.execute("DELETE FROM files WHERE path = ?1", params![path])?;
        Ok(())
    }

    // Metadata operations
    pub fn insert_metadata(&self, file_id: i64, summary: Option<&str>, tags: &[String], categories: &[String], entities: &serde_json::Value) -> Result<i64> {
        let conn = self.connect()?;

        let tags_json = serde_json::to_string(tags)?;
        let categories_json = serde_json::to_string(categories)?;
        let entities_json = serde_json::to_string(entities)?;

        // Delete existing metadata for this file
        conn.execute("DELETE FROM metadata WHERE file_id = ?1", params![file_id])?;

        conn.execute(
            "INSERT INTO metadata (file_id, llm_summary, tags, categories, entities)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![file_id, summary, tags_json, categories_json, entities_json],
        )?;

        Ok(conn.last_insert_rowid())
    }

    pub fn get_metadata(&self, file_id: i64) -> Result<Option<FileMetadata>> {
        let conn = self.connect()?;

        let mut stmt = conn.prepare(
            "SELECT id, file_id, llm_summary, tags, categories, entities, analyzed_at
             FROM metadata WHERE file_id = ?1"
        )?;

        let mut rows = stmt.query(params![file_id])?;

        if let Some(row) = rows.next()? {
            let tags_json: String = row.get(3)?;
            let categories_json: String = row.get(4)?;
            let entities_json: String = row.get(5)?;

            Ok(Some(FileMetadata {
                id: row.get(0)?,
                file_id: row.get(1)?,
                llm_summary: row.get(2)?,
                tags: serde_json::from_str(&tags_json)?,
                categories: serde_json::from_str(&categories_json)?,
                entities: serde_json::from_str(&entities_json)?,
                analyzed_at: row.get(6)?,
            }))
        } else {
            Ok(None)
        }
    }

    // View operations
    pub fn create_view(&self, name: &str, prompt: &str, sql_query: Option<&str>) -> Result<i64> {
        let conn = self.connect()?;

        conn.execute(
            "INSERT INTO views (name, query_prompt, sql_query)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(name) DO UPDATE SET
                query_prompt = ?2,
                sql_query = ?3,
                updated_at = strftime('%s', 'now')",
            params![name, prompt, sql_query],
        )?;

        Ok(conn.last_insert_rowid())
    }

    pub fn get_view(&self, name: &str) -> Result<Option<View>> {
        let conn = self.connect()?;

        let mut stmt = conn.prepare(
            "SELECT id, name, query_prompt, sql_query FROM views WHERE name = ?1"
        )?;

        let mut rows = stmt.query(params![name])?;

        if let Some(row) = rows.next()? {
            Ok(Some(View {
                id: row.get(0)?,
                name: row.get(1)?,
                query_prompt: row.get(2)?,
                sql_query: row.get(3)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn get_all_views(&self) -> Result<Vec<View>> {
        let conn = self.connect()?;

        let mut stmt = conn.prepare(
            "SELECT id, name, query_prompt, sql_query FROM views ORDER BY name"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(View {
                id: row.get(0)?,
                name: row.get(1)?,
                query_prompt: row.get(2)?,
                sql_query: row.get(3)?,
            })
        })?;

        rows.collect::<Result<Vec<_>, _>>().context("Failed to fetch views")
    }

    pub fn add_file_to_view(&self, view_id: i64, file_id: i64, relevance: f32) -> Result<()> {
        let conn = self.connect()?;

        conn.execute(
            "INSERT INTO view_files (view_id, file_id, relevance_score)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(view_id, file_id) DO UPDATE SET relevance_score = ?3",
            params![view_id, file_id, relevance],
        )?;

        Ok(())
    }

    pub fn get_view_files(&self, view_id: i64) -> Result<Vec<FileRecord>> {
        let conn = self.connect()?;

        let mut stmt = conn.prepare(
            "SELECT f.id, f.path, f.content_hash, f.size_bytes, f.modified_time, f.file_type, f.content_text
             FROM files f
             JOIN view_files vf ON f.id = vf.file_id
             WHERE vf.view_id = ?1
             ORDER BY vf.relevance_score DESC, f.modified_time DESC"
        )?;

        let rows = stmt.query_map(params![view_id], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                size_bytes: row.get(3)?,
                modified_time: row.get(4)?,
                file_type: row.get(5)?,
                content_text: row.get(6)?,
            })
        })?;

        rows.collect::<Result<Vec<_>, _>>().context("Failed to fetch view files")
    }

    pub fn clear_view_files(&self, view_id: i64) -> Result<()> {
        let conn = self.connect()?;
        conn.execute("DELETE FROM view_files WHERE view_id = ?1", params![view_id])?;
        Ok(())
    }

    // Analyzer operations
    pub fn register_analyzer(&self, file_type: &str, script_path: &str, language: &str, description: Option<&str>) -> Result<i64> {
        let conn = self.connect()?;

        conn.execute(
            "INSERT INTO analyzers (file_type, script_path, language, description)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(file_type) DO UPDATE SET
                script_path = ?2,
                language = ?3,
                description = ?4",
            params![file_type, script_path, language, description],
        )?;

        Ok(conn.last_insert_rowid())
    }

    pub fn get_analyzer(&self, file_type: &str) -> Result<Option<Analyzer>> {
        let conn = self.connect()?;

        let mut stmt = conn.prepare(
            "SELECT id, file_type, script_path, language, description
             FROM analyzers WHERE file_type = ?1"
        )?;

        let mut rows = stmt.query(params![file_type])?;

        if let Some(row) = rows.next()? {
            Ok(Some(Analyzer {
                id: row.get(0)?,
                file_type: row.get(1)?,
                script_path: row.get(2)?,
                language: row.get(3)?,
                description: row.get(4)?,
            }))
        } else {
            Ok(None)
        }
    }

    // LLM cache operations
    pub fn cache_llm_response(&self, prompt: &str, prompt_hash: &str, response: &str) -> Result<()> {
        let conn = self.connect()?;

        conn.execute(
            "INSERT INTO llm_cache (prompt_hash, prompt, response)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(prompt_hash) DO UPDATE SET
                accessed_at = strftime('%s', 'now'),
                access_count = access_count + 1",
            params![prompt_hash, prompt, response],
        )?;

        Ok(())
    }

    pub fn get_cached_response(&self, prompt_hash: &str) -> Result<Option<String>> {
        let conn = self.connect()?;

        let mut stmt = conn.prepare(
            "SELECT response FROM llm_cache WHERE prompt_hash = ?1"
        )?;

        let mut rows = stmt.query(params![prompt_hash])?;

        if let Some(row) = rows.next()? {
            // Update access time
            conn.execute(
                "UPDATE llm_cache SET accessed_at = strftime('%s', 'now'), access_count = access_count + 1
                 WHERE prompt_hash = ?1",
                params![prompt_hash],
            )?;

            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub fn clean_old_cache(&self, max_age_secs: i64) -> Result<usize> {
        let conn = self.connect()?;
        let cutoff = Utc::now().timestamp() - max_age_secs;

        let deleted = conn.execute(
            "DELETE FROM llm_cache WHERE accessed_at < ?1",
            params![cutoff],
        )?;

        Ok(deleted)
    }
}
