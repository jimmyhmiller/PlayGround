-- Files table: basic file information
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_time INTEGER NOT NULL,
    file_type TEXT,
    content_text TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type);

-- Metadata table: LLM-generated analysis
CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    llm_summary TEXT,
    tags TEXT, -- JSON array of strings
    categories TEXT, -- JSON array of strings
    entities TEXT, -- JSON object with extracted entities (people, dates, etc.)
    analyzed_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metadata_file_id ON metadata(file_id);

-- Views table: virtual directories based on queries
CREATE TABLE IF NOT EXISTS views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    query_prompt TEXT NOT NULL,
    sql_query TEXT, -- Generated SQL query or NULL for LLM-based filtering
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_views_name ON views(name);

-- View files: mapping between views and files with relevance scores
CREATE TABLE IF NOT EXISTS view_files (
    view_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    relevance_score REAL DEFAULT 1.0,
    added_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    PRIMARY KEY (view_id, file_id),
    FOREIGN KEY (view_id) REFERENCES views(id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_view_files_view ON view_files(view_id);
CREATE INDEX IF NOT EXISTS idx_view_files_file ON view_files(file_id);

-- Analyzers table: generated scripts for unknown file types
CREATE TABLE IF NOT EXISTS analyzers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_type TEXT NOT NULL UNIQUE,
    script_path TEXT NOT NULL,
    language TEXT NOT NULL, -- 'rust', 'python', 'shell', etc.
    description TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    last_used_at INTEGER
);

CREATE INDEX IF NOT EXISTS idx_analyzers_type ON analyzers(file_type);

-- LLM response cache table
CREATE TABLE IF NOT EXISTS llm_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_hash TEXT NOT NULL UNIQUE,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    accessed_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    access_count INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_llm_cache_hash ON llm_cache(prompt_hash);
CREATE INDEX IF NOT EXISTS idx_llm_cache_accessed ON llm_cache(accessed_at);
