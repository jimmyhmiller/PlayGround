use anyhow::{Context, Result};
use crate::db::{Database, FileRecord};
use crate::llm::CachedLLMClient;

pub struct ViewEngine {
    db: Database,
    llm: CachedLLMClient,
}

impl ViewEngine {
    pub fn new(db: Database, llm: CachedLLMClient) -> Self {
        Self { db, llm }
    }

    /// Create a new view from a natural language query
    pub async fn create_view(&self, name: &str, query: &str) -> Result<i64> {
        log::info!("Creating view '{}' from query: {}", name, query);

        // Ask LLM to generate SQL filter
        let sql_where = self.llm.generate_sql_filter(query).await
            .context("Failed to generate SQL filter from query")?;

        log::debug!("Generated SQL WHERE clause: {}", sql_where);

        // Create the view in database
        let view_id = self.db.create_view(name, query, Some(&sql_where))
            .context("Failed to create view in database")?;

        // Populate the view with matching files
        self.refresh_view(view_id).await?;

        Ok(view_id)
    }

    /// Refresh a view by re-evaluating its query
    pub async fn refresh_view(&self, view_id: i64) -> Result<()> {
        let view = self.db.get_all_views()?
            .into_iter()
            .find(|v| v.id == view_id)
            .context("View not found")?;

        log::info!("Refreshing view: {}", view.name);

        // Clear existing files in the view
        self.db.clear_view_files(view_id)?;

        // Get matching files
        let files = if let Some(sql_query) = &view.sql_query {
            self.query_files_with_sql(sql_query)?
        } else {
            // Fallback: use LLM to filter files
            self.query_files_with_llm(&view.query_prompt).await?
        };

        let file_count = files.len();

        // Add files to view
        for (file, relevance) in files {
            self.db.add_file_to_view(view_id, file.id, relevance)?;
        }

        log::info!("View '{}' refreshed with {} files", view.name, file_count);

        Ok(())
    }

    /// Query files using SQL WHERE clause
    fn query_files_with_sql(&self, where_clause: &str) -> Result<Vec<(FileRecord, f32)>> {
        // This is a simplified version - in production you'd want to use
        // rusqlite's query builder or a safe SQL construction method
        let all_files = self.db.get_all_files()?;

        // For now, we'll use a simple approach and filter in Rust
        // In a production system, you'd execute the SQL directly
        let filtered: Vec<_> = all_files.into_iter()
            .filter(|f| self.matches_simple_filter(f, where_clause))
            .map(|f| (f, 1.0))
            .collect();

        Ok(filtered)
    }

    /// Simple filter matching (placeholder - real implementation would use SQL)
    fn matches_simple_filter(&self, _file: &FileRecord, _where_clause: &str) -> bool {
        // This is a placeholder - in reality, we should execute the SQL
        // For now, just return true to include all files
        true
    }

    /// Query files using LLM to determine matches
    async fn query_files_with_llm(&self, query: &str) -> Result<Vec<(FileRecord, f32)>> {
        let all_files = self.db.get_all_files()?;
        let mut results = Vec::new();

        for file in all_files {
            // Get metadata for the file
            if let Ok(Some(metadata)) = self.db.get_metadata(file.id) {
                let summary = metadata.llm_summary.unwrap_or_default();
                let tags = metadata.tags.join(", ");
                let categories = metadata.categories.join(", ");

                // Ask LLM if this file matches the query
                let prompt = format!(
                    "Query: \"{}\"\n\nFile: {}\nSummary: {}\nTags: {}\nCategories: {}\n\n\
                     Does this file match the query? Answer with a JSON object: {{\"match\": true/false, \"relevance\": 0.0-1.0}}",
                    query, file.path, summary, tags, categories
                );

                if let Ok(response) = self.llm.complete(&prompt).await {
                    if let Ok(result) = serde_json::from_str::<serde_json::Value>(&response) {
                        if result.get("match").and_then(|v| v.as_bool()).unwrap_or(false) {
                            let relevance = result.get("relevance")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.5) as f32;
                            results.push((file, relevance));
                        }
                    }
                }
            }
        }

        // Sort by relevance
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Get all files in a view
    pub fn get_view_files(&self, view_id: i64) -> Result<Vec<FileRecord>> {
        self.db.get_view_files(view_id)
    }

    /// Get all views
    pub fn get_all_views(&self) -> Result<Vec<crate::db::View>> {
        self.db.get_all_views()
    }
}
