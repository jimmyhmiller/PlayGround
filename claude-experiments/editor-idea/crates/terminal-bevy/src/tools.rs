//! DeepSeek tool surface. The "tools" exposed to the model are exactly
//! the app's IPC actions ([`crate::ipc::IpcRequest`]) — so a tool call is
//! executed by [`crate::ipc::dispatch_local`], driving the app over the
//! same socket the `tb*` CLIs use. This module declares which actions are
//! exposed, their safety class, and builds the system prompt describing
//! them. Keeping the catalog here (rather than scattered) means the
//! prompt and the executor never drift.

use serde::Deserialize;

/// The model's planned response.
#[derive(Deserialize, Debug, Default)]
pub struct ToolPlan {
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub calls: Vec<ToolCall>,
}

/// One planned tool invocation.
#[derive(Deserialize, Debug, Clone)]
pub struct ToolCall {
    pub tool: String,
    #[serde(default)]
    pub args: serde_json::Value,
    #[serde(default)]
    pub reason: String,
}

/// Validate + convert a tool call into an [`IpcRequest`](crate::ipc::IpcRequest)
/// by folding `{"action": tool}` into its args and round-tripping through
/// serde — which rejects unknown tools and malformed args.
pub fn to_ipc_request(call: &ToolCall) -> Result<crate::ipc::IpcRequest, String> {
    let mut obj = match &call.args {
        serde_json::Value::Object(m) => m.clone(),
        serde_json::Value::Null => serde_json::Map::new(),
        _ => return Err("args must be a JSON object".into()),
    };
    obj.insert(
        "action".into(),
        serde_json::Value::String(call.tool.clone()),
    );
    serde_json::from_value(serde_json::Value::Object(obj)).map_err(|e| e.to_string())
}

/// Whether a tool runs immediately or needs explicit user confirmation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Risk {
    /// Additive / reversible — run as soon as the model proposes it.
    Safe,
    /// Destroys or overwrites state — show the user and require Enter.
    Risky,
}

/// One exposed tool. `name` matches the IPC action tag (snake_case), so
/// `{"action": name, ...args}` deserializes into an `IpcRequest`.
pub struct ToolSpec {
    pub name: &'static str,
    pub risk: Risk,
    pub description: &'static str,
    /// JSON shape of `args`, shown to the model.
    pub args_hint: &'static str,
}

pub const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        name: "send_inbox",
        risk: Risk::Safe,
        description: "Leave a message in a project's inbox (a notification the user reads later).",
        args_hint: r#"{"project":"<name>|active","sender":"deepseek","subject":"<short>","body":"<text>"}"#,
    },
    ToolSpec {
        name: "spawn_widget",
        risk: Risk::Safe,
        description: "Open a new pane running a shell command (a small terminal-like widget).",
        args_hint: r#"{"command":"<shell cmd>","title":"<short>","project":"<name>|active"}"#,
    },
    ToolSpec {
        name: "open_file",
        risk: Risk::Safe,
        description: "Open a file in an editor pane.",
        args_hint: r#"{"path":"<absolute path>","project":"<name>|active"}"#,
    },
    ToolSpec {
        name: "suggest_pane",
        risk: Risk::Safe,
        description: "Park a suggested pane in the drawer for the user to pull down later, instead of spawning it now. Good for 'maybe run this' ideas.",
        args_hint: r#"{"command":"<shell cmd>","title":"<short>","reason":"<why>","project":"<name>|active"}"#,
    },
    ToolSpec {
        name: "widget_message",
        risk: Risk::Safe,
        description: "Publish a message on the widget message bus to a project's widgets.",
        args_hint: r#"{"project":"<name>|active","topic":"<topic>","payload":{},"retain":false}"#,
    },
    ToolSpec {
        name: "toggle_cube",
        risk: Risk::Safe,
        description: "Toggle the 3D project-overview cube.",
        args_hint: r#"{}"#,
    },
    ToolSpec {
        name: "set_project_default_cwd",
        risk: Risk::Risky,
        description: "Overwrite a project's remembered default working directory.",
        args_hint: r#"{"project":"<name>|active","cwd":"<absolute path>"}"#,
    },
    ToolSpec {
        name: "close_project_panes",
        risk: Risk::Risky,
        description: "Close (despawn) panes in a project, optionally only a given kind. Destructive.",
        args_hint: r#"{"project":"<name>|active","kind":"<pane kind>|null"}"#,
    },
];

pub fn spec_of(name: &str) -> Option<&'static ToolSpec> {
    TOOLS.iter().find(|t| t.name == name)
}

/// Build the system prompt: role + tool catalog + output contract.
pub fn system_prompt() -> String {
    let mut s = String::new();
    s.push_str(
        "You are an action planner embedded in a developer's canvas app \
         (floating panes: terminals, editors, widgets, an inbox, a project \
         cube). The user types a natural-language request. Respond ONLY with \
         a JSON object of the form:\n\
         {\"message\":\"<one short sentence to the user>\",\"calls\":[{\"tool\":\"<name>\",\"args\":{...},\"reason\":\"<short>\"}]}\n\n\
         Use the fewest calls that satisfy the request. If the request is \
         purely conversational or no tool fits, return an empty calls array \
         and put your answer in message. Never invent tools or args. \
         Available tools:\n\n",
    );
    for t in TOOLS {
        let tag = match t.risk {
            Risk::Safe => "safe",
            Risk::Risky => "needs-confirmation",
        };
        s.push_str(&format!(
            "- {} ({}): {}\n  args: {}\n",
            t.name, tag, t.description, t.args_hint
        ));
    }
    s
}
