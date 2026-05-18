//! Interactive chat REPL for the constrained-language agent loop.
//!
//! The runtime, manifest, WASM bodies, and adapters are set up once. Each
//! line the user types becomes one turn:
//!   - the first turn enqueues `GoalReceived` (which records the session
//!     and kicks off the first LLM call);
//!   - every subsequent turn enqueues `UserMessage` against the same
//!     session_id (which appends to the conversation and kicks off another
//!     LLM call).
//! After enqueueing, we `run_to_quiescence`; tools may run, the LLM may be
//! called multiple times, and finally `NotifyUser` prints the assistant's
//! answer. Then we prompt again.
//!
//! Quits on EOF (Ctrl-D), empty line "exit" / "quit", or Ctrl-C.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use serde_json::json;

use host_adapters::{AnthropicLlmAdapter, SimpleToolsAdapter, StdoutNotifyAdapter};
use ir::load_manifest_file;
use runtime::wasm::load_handler_body;
use runtime::{InboundEvent, Runtime};

#[derive(Parser)]
#[command(name = "agent-demo")]
#[command(about = "Interactive chat loop for the constrained-language agent")]
struct Cli {
    /// Optional override for the manifest path.
    #[arg(long)]
    manifest: Option<PathBuf>,

    /// Optional override for the directory containing the WASM component files.
    #[arg(long)]
    components_dir: Option<PathBuf>,

    /// Use a scripted LLM (no network, no API key). Each user turn becomes a
    /// `needs_tool calc <input>` then a `final` echoing the tool result.
    #[arg(long)]
    fake_llm: bool,

    /// Run one turn from this string and exit (non-interactive). Useful for
    /// tests / smoke-checking.
    #[arg(long)]
    once: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("agent-demo: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = repo_root();
    let manifest_path = cli
        .manifest
        .unwrap_or_else(|| repo_root.join("wasm-samples/agent_loop/manifest.toml"));
    let components_dir = cli
        .components_dir
        .unwrap_or_else(|| repo_root.join("wasm-samples/agent_loop"));

    let manifest = load_manifest_file(&manifest_path)?;
    let mut rt = Runtime::new(manifest.clone())?;

    for handler in [
        "start_goal",
        "on_user_message",
        "on_llm_returned",
        "on_tool_returned",
    ] {
        let component_path = components_dir
            .join(handler)
            .join(format!("{handler}.component.wasm"));
        let mut body = load_handler_body(&manifest, handler, &component_path)?;
        rt.bodies.register(handler, move |ctx| body(ctx));
    }

    if cli.fake_llm {
        rt.adapters
            .register("CallLlm", Box::new(FakeLlmAdapter::default()));
    } else {
        rt.adapters
            .register("CallLlm", Box::new(AnthropicLlmAdapter::from_env()?));
    }
    rt.adapters
        .register("ExecuteTool", Box::new(SimpleToolsAdapter::new()));
    rt.adapters
        .register("NotifyUser", Box::new(StdoutNotifyAdapter::new()));

    let session_id = format!("s-{}", std::process::id());
    let mut first_turn = true;

    let enqueue_turn = |rt: &mut Runtime, text: &str, first: bool| -> Result<(), Box<dyn std::error::Error>> {
        if first {
            rt.enqueue(InboundEvent::new(
                "GoalReceived",
                json!({ "goal_id": session_id, "prompt": text }),
            ))?;
        } else {
            rt.enqueue(InboundEvent::new(
                "UserMessage",
                json!({ "session_id": session_id, "text": text }),
            ))?;
        }
        rt.run_to_quiescence()?;
        Ok(())
    };

    if let Some(line) = cli.once {
        enqueue_turn(&mut rt, &line, true)?;
        return Ok(());
    }

    eprintln!("agent-demo chat. session_id={session_id}. Ctrl-D to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut lines = stdin.lock().lines();
    loop {
        write!(stdout, "> ")?;
        stdout.flush()?;

        let line = match lines.next() {
            Some(Ok(l)) => l,
            Some(Err(e)) => return Err(e.into()),
            None => {
                eprintln!("\n(eof)");
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "exit" || trimmed == "quit" {
            break;
        }

        enqueue_turn(&mut rt, trimmed, first_turn)?;
        first_turn = false;
    }

    Ok(())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Stateful fake LLM. On every call, looks at the most recent user (or tool)
/// message and either:
///   - asks for `calc` with that text as the args (first time for a turn), or
///   - returns a `final` answer echoing the most recent tool result.
///
/// Approximation: the body's `on_tool_returned` appends a "tool" message, so
/// we treat seeing a tool message as the trigger to finalize.
#[derive(Default)]
struct FakeLlmAdapter {}

impl runtime::effect::Adapter for FakeLlmAdapter {
    fn fulfill(
        &mut self,
        request: runtime::Value,
        _emit_id: u64,
    ) -> runtime::effect::AdapterResult {
        let messages = request
            .get("messages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let last = messages.last().cloned().unwrap_or_else(|| json!({}));
        let role = last.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let text = last.get("text").and_then(|v| v.as_str()).unwrap_or("");

        let resp = match role {
            "tool" => json!({
                "kind": "final",
                "text": format!("(fake-llm) result: {text}"),
                "tool": "",
                "args": "",
            }),
            _ => {
                // Either the original user goal or a follow-up user message:
                // route it to calc as a math expression. If it's not a valid
                // expression, simple_tools will return Failed and the agent
                // will surface the error.
                json!({
                    "kind": "needs_tool",
                    "text": "let me compute that",
                    "tool": "calc",
                    "args": text,
                })
            }
        };
        runtime::effect::AdapterResult::Ok(resp)
    }
}

