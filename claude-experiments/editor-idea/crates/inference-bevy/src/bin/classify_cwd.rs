//! Smoke-test the cwd-default classifier against the configured LLM
//! endpoint. Reads (`project_name`, `cwd`) from argv and prints the
//! parsed classification (or the error) to stdout.
//!
//! Example:
//!   cargo run -p inference_bevy --bin infer-classify-cwd -- editor-idea /Users/me/code/editor-idea
//!
//! Honours the same env vars as the in-process classifier:
//!   LLM_BASE_URL  (default https://api.deepseek.com/v1)
//!   LLM_API_KEY   (fallback DEEPSEEK_KEY, OPENAI_API_KEY)
//!   LLM_MODEL     (default deepseek-chat)

use inference_bevy::{classifiers, llm::LlmConfig};

fn main() {
    let mut args = std::env::args().skip(1);
    let project = args.next().unwrap_or_else(|| {
        eprintln!("usage: infer-classify-cwd <project_name> <cwd>");
        std::process::exit(2);
    });
    let cwd = args.next().unwrap_or_else(|| {
        eprintln!("usage: infer-classify-cwd <project_name> <cwd>");
        std::process::exit(2);
    });
    let cfg = match LlmConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config error: {e}");
            std::process::exit(3);
        }
    };
    eprintln!(
        "[infer] base_url={} model={} key=<{}-char>",
        cfg.base_url,
        cfg.model,
        cfg.api_key.len()
    );
    match classifiers::classify_default_cwd(&cfg, &project, &cwd) {
        Ok(c) => {
            println!(
                "good_default={} confidence={:.2} reason={}",
                c.good_default, c.confidence, c.reason
            );
        }
        Err(e) => {
            eprintln!("classifier error: {e}");
            std::process::exit(4);
        }
    }
}
