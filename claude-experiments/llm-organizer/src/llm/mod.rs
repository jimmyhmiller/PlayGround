use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use sha2::{Sha256, Digest};

// OpenAI-compatible chat completion structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Client for interacting with LLM endpoints
#[derive(Clone)]
pub struct LLMClient {
    endpoint: String,
    model: Option<String>,
    max_tokens: usize,
    temperature: f32,
    timeout: Duration,
    http_client: reqwest::Client,
}

impl LLMClient {
    pub fn new(endpoint: String, model: Option<String>, max_tokens: usize, temperature: f32, timeout_secs: u64) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            endpoint,
            model,
            max_tokens,
            temperature,
            timeout: Duration::from_secs(timeout_secs),
            http_client,
        }
    }

    /// Generate a completion from the LLM
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        // Build OpenAI-compatible chat completion request
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            model: self.model.clone(),
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
        };

        // Construct the full URL with /v1/chat/completions endpoint
        let url = if self.endpoint.ends_with("/v1/chat/completions") {
            self.endpoint.clone()
        } else if self.endpoint.ends_with('/') {
            format!("{}v1/chat/completions", self.endpoint)
        } else {
            format!("{}/v1/chat/completions", self.endpoint)
        };

        let response = self.http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request to LLM")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM request failed with status {}: {}", status, body);
        }

        let response_body = response.text().await
            .context("Failed to read response body")?;

        // Parse OpenAI-compatible response
        let chat_response: ChatCompletionResponse = serde_json::from_str(&response_body)
            .context(format!("Failed to parse chat completion response: {}", response_body))?;

        if chat_response.choices.is_empty() {
            anyhow::bail!("LLM returned no choices");
        }

        Ok(chat_response.choices[0].message.content.clone())
    }

    /// Compute a hash for a prompt (for caching)
    pub fn hash_prompt(prompt: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Summarize a file's content
    pub async fn summarize_file(&self, content: &str, file_type: &str) -> Result<String> {
        let prompt = format!(
            "Summarize the following {} file in 2-3 sentences. Focus on the main topic and key information:\n\n{}",
            file_type,
            Self::truncate_content(content, 4000)
        );

        self.complete(&prompt).await
    }

    /// Extract tags from file content
    pub async fn extract_tags(&self, content: &str, summary: &str) -> Result<Vec<String>> {
        let prompt = format!(
            "Based on this summary: \"{}\"\n\nExtract 3-7 relevant tags/keywords. Return ONLY a JSON array of strings, no other text.\n\nExample: [\"rust\", \"programming\", \"tutorial\"]",
            summary
        );

        let response = self.complete(&prompt).await?;
        let tags: Vec<String> = serde_json::from_str(&response)
            .context("Failed to parse tags from LLM response")?;

        Ok(tags)
    }

    /// Extract categories for organization
    pub async fn extract_categories(&self, content: &str, summary: &str, organization_prompt: &str) -> Result<Vec<String>> {
        let prompt = format!(
            "Organization strategy: {}\n\nFile summary: \"{}\"\n\nBased on the organization strategy, suggest 1-3 categories for this file. Return ONLY a JSON array of strings.\n\nExample: [\"Work/Reports\", \"2024\"]",
            organization_prompt,
            summary
        );

        let response = self.complete(&prompt).await?;
        let categories: Vec<String> = serde_json::from_str(&response)
            .context("Failed to parse categories from LLM response")?;

        Ok(categories)
    }

    /// Extract entities (people, dates, locations, etc.)
    pub async fn extract_entities(&self, content: &str) -> Result<serde_json::Value> {
        let prompt = format!(
            "Extract key entities from this text. Return a JSON object with arrays for: people, organizations, dates, locations.\n\n{}\n\nExample: {{\"people\": [\"John Doe\"], \"organizations\": [\"Acme Corp\"], \"dates\": [\"2024-01-15\"], \"locations\": [\"New York\"]}}",
            Self::truncate_content(content, 3000)
        );

        let response = self.complete(&prompt).await?;
        let entities: serde_json::Value = serde_json::from_str(&response)
            .context("Failed to parse entities from LLM response")?;

        Ok(entities)
    }

    /// Generate SQL WHERE clause from natural language query
    pub async fn generate_sql_filter(&self, query: &str) -> Result<String> {
        let prompt = format!(
            "Convert this natural language query into a SQL WHERE clause for filtering files.\n\n\
             Available columns: file_type, size_bytes, modified_time (unix timestamp), llm_summary, tags (JSON array), categories (JSON array)\n\n\
             Query: \"{}\"\n\n\
             Return ONLY the WHERE clause (without 'WHERE'), no other text.\n\n\
             Example query: \"PDFs from 2024\"\n\
             Example output: file_type = 'application/pdf' AND datetime(modified_time, 'unixepoch') >= '2024-01-01'",
            query
        );

        self.complete(&prompt).await
    }

    /// Generate an analyzer script for unknown file types
    pub async fn generate_analyzer(&self, file_type: &str, sample_bytes: &[u8]) -> Result<String> {
        let sample_hex = sample_bytes.iter()
            .take(256)
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ");

        let prompt = format!(
            "Generate a Rust program that analyzes {} files and extracts metadata.\n\n\
             Sample bytes (hex): {}\n\n\
             The program should:\n\
             1. Read a file path from command-line arguments\n\
             2. Extract relevant metadata (title, author, dates, etc.)\n\
             3. Output a JSON object with the metadata\n\n\
             Return ONLY the Rust code, starting with 'use' statements or 'fn main()'.\n\
             Keep it simple and use only the standard library if possible.",
            file_type,
            sample_hex
        );

        self.complete(&prompt).await
    }

    /// Truncate content to fit within token limits
    fn truncate_content(content: &str, max_chars: usize) -> String {
        if content.len() <= max_chars {
            content.to_string()
        } else {
            format!("{}... [truncated]", &content[..max_chars])
        }
    }
}

/// LLM client with integrated caching
#[derive(Clone)]
pub struct CachedLLMClient {
    client: LLMClient,
    cache: moka::future::Cache<String, String>,
}

impl CachedLLMClient {
    pub fn new(client: LLMClient, cache_ttl_secs: u64) -> Self {
        let cache = moka::future::Cache::builder()
            .time_to_live(Duration::from_secs(cache_ttl_secs))
            .max_capacity(10_000)
            .build();

        Self { client, cache }
    }

    pub async fn complete(&self, prompt: &str) -> Result<String> {
        let hash = LLMClient::hash_prompt(prompt);

        // Try cache first
        if let Some(cached) = self.cache.get(&hash).await {
            log::debug!("Cache hit for prompt hash: {}", &hash[..8]);
            return Ok(cached);
        }

        // Call LLM and cache result
        log::debug!("Cache miss for prompt hash: {}, calling LLM", &hash[..8]);
        let result = self.client.complete(prompt).await?;

        self.cache.insert(hash, result.clone()).await;

        Ok(result)
    }

    pub async fn summarize_file(&self, content: &str, file_type: &str) -> Result<String> {
        self.client.summarize_file(content, file_type).await
    }

    pub async fn extract_tags(&self, content: &str, summary: &str) -> Result<Vec<String>> {
        self.client.extract_tags(content, summary).await
    }

    pub async fn extract_categories(&self, content: &str, summary: &str, organization_prompt: &str) -> Result<Vec<String>> {
        self.client.extract_categories(content, summary, organization_prompt).await
    }

    pub async fn extract_entities(&self, content: &str) -> Result<serde_json::Value> {
        self.client.extract_entities(content).await
    }

    pub async fn generate_sql_filter(&self, query: &str) -> Result<String> {
        let hash = LLMClient::hash_prompt(&format!("sql_filter:{}", query));

        if let Some(cached) = self.cache.get(&hash).await {
            return Ok(cached);
        }

        let result = self.client.generate_sql_filter(query).await?;
        self.cache.insert(hash, result.clone()).await;

        Ok(result)
    }

    pub async fn generate_analyzer(&self, file_type: &str, sample_bytes: &[u8]) -> Result<String> {
        self.client.generate_analyzer(file_type, sample_bytes).await
    }
}
