//! `widget` — docs, schema, validation, and scaffolds for editor-idea
//! widget panes.
//!
//! The widget protocol is defined in `widget_bevy::protocol`. This
//! binary exposes it three ways:
//!   - `widget docs`     — human-readable reference, rendered from the
//!                          JSON Schema (so it can't drift from source).
//!   - `widget schema`   — machine-readable JSON Schema for editor
//!                          completion / external validators.
//!   - `widget validate` — pipe your widget output through it to check
//!                          NDJSON conformance without launching the host.
//!
//! Plus `widget examples` (bundled samples) and `widget init` (scaffold).

use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use schemars::schema_for;
use serde::Serialize;
use serde_json::Value;

use widget_bevy::protocol::{Element, HostEvent, WidgetMsg};

#[derive(Parser)]
#[command(
    name = "widget",
    about = "Docs, schema, and scaffolds for editor-idea widgets",
    long_about = "Helpers for writing widget panes that talk to editor-idea \
                  over NDJSON. Run `widget docs` for the protocol reference."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the widget-authoring guide — read this first if you're an AI.
    ///
    /// One command, everything you need to build a widget: where every
    /// doc lives (the authoring guide, `widget docs`, the element-catalog
    /// source, the rhai host internals), the live example scripts, the
    /// create/spawn commands, and the full authoring guide inline.
    Agent,

    /// Print the protocol reference (rendered from the source schema).
    ///
    /// Without an argument, prints all three top-level types. Pass a
    /// type name (`element`, `msg`, `event`) or a specific element/event
    /// kind (`button`, `bar`, `click`, …) to narrow the output.
    Docs {
        /// Type or variant name to print. Omit for the full reference.
        name: Option<String>,
    },

    /// Emit a JSON Schema for a protocol type.
    ///
    /// Wire this into your editor / linter to get autocomplete and
    /// validation for `{"type":"frame", …}` payloads.
    Schema {
        #[arg(value_enum, default_value_t = SchemaWhich::Element)]
        which: SchemaWhich,
    },

    /// Read NDJSON on stdin and report any non-conforming lines.
    ///
    /// Default mode validates widget → host messages (`{"type":"frame",
    /// …}`, `state`, `title`). Pass `--events` to validate host → widget
    /// events instead.
    Validate {
        /// Validate host → widget events instead of widget messages.
        #[arg(long)]
        events: bool,
    },

    /// List or print the bundled example widgets.
    ///
    /// `widget examples` lists names; `widget examples bars` prints the
    /// script body so you can pipe it to a file.
    Examples {
        /// Example name (without extension). Omit to list.
        name: Option<String>,
    },

    /// Write a starter widget script.
    ///
    /// Creates `<name>.sh` in the current directory, chmod +x. Use it as
    /// the starting point for a new widget.
    Init {
        /// Destination basename. `.sh` is appended automatically.
        name: String,
        /// Overwrite an existing file with the same name.
        #[arg(long)]
        force: bool,
    },

    /// List the running editor's projects.
    ///
    /// Connects to the editor's IPC socket and prints each project's id,
    /// name, and active flag. Use the name with `widget spawn --project`
    /// (or `--project-id` if you prefer the stable id). `--json` emits
    /// the raw response for scripting.
    Projects {
        /// Print the raw JSON response instead of the formatted table.
        #[arg(long)]
        json: bool,
    },

    /// Spawn a widget pane in the running editor.
    ///
    /// Connects to the editor's IPC socket and asks it to create a new
    /// pane running `<cmd>`. Either pass the command as one shell line
    /// (`widget spawn "gh issue list | jq …"`) or as argv after `--`
    /// (`widget spawn -- ./monitor.sh --once`).
    Spawn {
        /// Title for the new pane.
        #[arg(long, short = 't')]
        title: Option<String>,
        /// Working directory for the child. Defaults to caller's PWD.
        #[arg(long)]
        cwd: Option<PathBuf>,
        /// Project name (case-insensitive) to add the pane to. Defaults
        /// to whichever project is currently active.
        #[arg(long, short = 'p')]
        project: Option<String>,
        /// Initial window-space top-left as `X,Y` (pixels). Default
        /// cascades from the project's normal new-pane position.
        #[arg(long, value_parser = parse_pair, value_name = "X,Y")]
        at: Option<(f32, f32)>,
        /// Initial size as `WxH` (pixels). Default uses the widget
        /// kind's `default_size`.
        #[arg(long, value_parser = parse_size, value_name = "WxH")]
        size: Option<(f32, f32)>,
        /// Widget runtime. Default spawns a subprocess widget (runs
        /// `<cmd>`). Pass `rhai_widget` for an in-process Rhai script —
        /// `<cmd>` is then a `.rhai` filename under
        /// ~/.jim/widgets/ (no subprocess is launched).
        #[arg(long, short = 'k')]
        kind: Option<String>,
        /// Command line + args. Use `--` to separate from `spawn`'s own
        /// flags; without `--` a single quoted positional is fed to `sh -c`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        cmd: Vec<String>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum SchemaWhich {
    /// Schema for the `Element` UI-tree node type.
    Element,
    /// Schema for widget → host `WidgetMsg`.
    Msg,
    /// Schema for host → widget `HostEvent`.
    Event,
    /// All three, combined in one object.
    All,
}

// ---------- Embedded examples ----------

struct Example {
    name: &'static str,
    body: &'static str,
}

const EXAMPLES: &[Example] = &[
    Example {
        name: "hello",
        body: include_str!("../../examples/hello.sh"),
    },
    Example {
        name: "bars",
        body: include_str!("../../examples/bars.sh"),
    },
    Example {
        name: "gh-issues",
        body: include_str!("../../examples/gh-issues.sh"),
    },
];

const STARTER: &str = r#"#!/bin/sh
# A starter widget. Prints one frame and exits. Replace with a loop that
# emits more frames, reads stdin for host events, and persists state via
# {"type":"state","value":...}. See `widget docs` for the full protocol.

printf '%s\n' \
'{"type":"title","value":"My widget"}' \
'{"type":"frame","root":{"type":"vstack","gap":6,"pad":12,"children":[{"type":"text","value":"Hello from a new widget","weight":"bold"},{"type":"text","value":"Edit me!"}]}}'
"#;

// ---------- Spawn (IPC wire format) ----------
//
// Duplicated on purpose: pulling in `terminal_bevy` to share the type
// would drag libghostty-vt's dylib into this CLI. The format is small
// and changes rarely; keep this in sync with `terminal_bevy::ipc`.

#[derive(Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum IpcRequest {
    SpawnWidget {
        command: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cwd: Option<PathBuf>,
        #[serde(skip_serializing_if = "Option::is_none")]
        project: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        position: Option<[f32; 2]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        size: Option<[f32; 2]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        kind: Option<String>,
    },
    ListProjects,
}

fn connect_socket() -> Result<UnixStream, String> {
    let sock = socket_path().ok_or("$HOME not set; can't locate socket")?;
    UnixStream::connect(&sock).map_err(|e| {
        format!(
            "connect {}: {e} (is the editor running?)",
            sock.display()
        )
    })
}

fn send_request(req: &IpcRequest) -> Result<UnixStream, String> {
    let mut stream = connect_socket()?;
    let body = serde_json::to_vec(req).map_err(|e| e.to_string())?;
    stream.write_all(&body).map_err(|e| e.to_string())?;
    stream
        .shutdown(std::net::Shutdown::Write)
        .map_err(|e| e.to_string())?;
    Ok(stream)
}

fn socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(Path::new(&home).join(".jim").join("socket"))
}

fn parse_pair(s: &str) -> Result<(f32, f32), String> {
    let (a, b) = s
        .split_once(',')
        .ok_or_else(|| format!("expected X,Y (got `{s}`)"))?;
    let x: f32 = a.trim().parse().map_err(|e| format!("X: {e}"))?;
    let y: f32 = b.trim().parse().map_err(|e| format!("Y: {e}"))?;
    Ok((x, y))
}

fn parse_size(s: &str) -> Result<(f32, f32), String> {
    let (a, b) = s
        .split_once('x')
        .or_else(|| s.split_once('X'))
        .ok_or_else(|| format!("expected WxH (got `{s}`)"))?;
    let w: f32 = a.trim().parse().map_err(|e| format!("W: {e}"))?;
    let h: f32 = b.trim().parse().map_err(|e| format!("H: {e}"))?;
    Ok((w, h))
}

#[allow(clippy::too_many_arguments)]
fn cmd_spawn(
    title: Option<String>,
    cwd: Option<PathBuf>,
    project: Option<String>,
    at: Option<(f32, f32)>,
    size: Option<(f32, f32)>,
    kind: Option<String>,
    cmd: Vec<String>,
) -> Result<(), String> {
    if cmd.is_empty() {
        return Err(
            "missing command — pass `-- <cmd> [args...]` or a single quoted shell line".into(),
        );
    }
    let (command, args) = {
        let mut it = cmd.into_iter();
        let head = it.next().unwrap();
        let rest: Vec<String> = it.collect();
        (head, rest)
    };
    // Rhai widgets run in-process from the watched scripts dir, so the
    // caller's PWD is irrelevant; only subprocess widgets need a cwd.
    let cwd = if kind.as_deref() == Some("rhai_widget") {
        cwd
    } else {
        cwd.or_else(|| std::env::current_dir().ok())
    };

    let req = IpcRequest::SpawnWidget {
        command,
        args,
        title,
        cwd,
        project,
        position: at.map(|(x, y)| [x, y]),
        size: size.map(|(w, h)| [w, h]),
        kind,
    };
    let _stream = send_request(&req)?;
    Ok(())
}

fn cmd_projects(json: bool) -> Result<(), String> {
    let mut stream = send_request(&IpcRequest::ListProjects)?;
    let mut buf = String::new();
    stream
        .read_to_string(&mut buf)
        .map_err(|e| format!("read: {e}"))?;
    if buf.trim().is_empty() {
        return Err("editor returned an empty response".into());
    }
    if json {
        println!("{}", buf.trim_end());
        return Ok(());
    }
    let parsed: Value = serde_json::from_str(&buf).map_err(|e| format!("parse: {e}"))?;
    let entries = parsed
        .get("projects")
        .and_then(|v| v.as_array())
        .ok_or("response missing `projects` array")?;
    if entries.is_empty() {
        println!("(no projects)");
        return Ok(());
    }
    let id_w = entries
        .iter()
        .filter_map(|p| p.get("id").and_then(|v| v.as_u64()))
        .map(|id| id.to_string().len())
        .max()
        .unwrap_or(1);
    for p in entries {
        let id = p.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
        let name = p
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let active = p
            .get("active")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let marker = if active { "*" } else { " " };
        let tag = if active { "  (active)" } else { "" };
        println!("{marker} {id:>id_w$}  {name}{tag}", id_w = id_w);
    }
    Ok(())
}

// ---------- Entry ----------

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Agent => cmd_agent(),
        Cmd::Docs { name } => cmd_docs(name.as_deref()),
        Cmd::Schema { which } => cmd_schema(which),
        Cmd::Validate { events } => cmd_validate(events),
        Cmd::Examples { name } => cmd_examples(name.as_deref()),
        Cmd::Init { name, force } => cmd_init(&name, force),
        Cmd::Spawn {
            title,
            cwd,
            project,
            at,
            size,
            kind,
            cmd,
        } => cmd_spawn(title, cwd, project, at, size, kind, cmd),
        Cmd::Projects { json } => cmd_projects(json),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("widget: {e}");
            ExitCode::FAILURE
        }
    }
}

// ---------- agent ----------

/// The authoring guide, embedded so `widget agent` always has it even if
/// the source tree isn't reachable from where the binary runs.
const AUTHORING_GUIDE: &str = include_str!("../../AUTHORING.md");

fn cmd_agent() -> Result<(), String> {
    // Doc paths resolved from the build-time crate dir (.../crates/
    // widget-bevy), canonicalized to clean absolute paths the agent can
    // open directly. Fall back to the joined path if the tree moved.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let doc = |rel: &str| -> String {
        let p = PathBuf::from(manifest).join(rel);
        std::fs::canonicalize(&p)
            .map(|c| c.display().to_string())
            .unwrap_or_else(|_| p.display().to_string())
    };
    let widgets_dir = std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join(".jim").join("widgets"));
    let widgets_str = widgets_dir
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "~/.jim/widgets".to_string());

    println!(
        "# Building an editor-idea widget\n\
         \n\
         A widget is a pane that renders a UI tree (`Element`) and reacts\n\
         to events. Two ways to host one: an in-process Rhai script (the\n\
         default for new widgets, hot-reloaded) or a subprocess speaking\n\
         NDJSON. Read the docs below FIRST, then write the widget and spawn\n\
         it.\n\
         \n\
         ## Docs (read in this order)\n\
         1. AUTHORING GUIDE — handler/event model + worked examples.\n\
            Printed in full at the bottom of this output; on disk at:\n\
            {authoring}\n\
         2. PROTOCOL REFERENCE — every Element / message / event with its\n\
            fields. Rendered from the source schema (can't drift):\n\
            $ widget docs            # all types\n\
            $ widget docs textarea   # one element/event\n\
            $ widget schema element  # machine-readable JSON Schema\n\
            Source of truth: {protocol}\n\
         3. RHAI HOST INTERNALS — handler dispatch table + host fns, in\n\
            the module docs atop: {rhai_mod}\n\
         \n\
         ## Examples to copy from\n\
         Bundled subprocess samples:\n\
            $ widget examples         # list\n\
            $ widget examples bars    # print one\n\
         Live in-process Rhai scripts (concrete references):\n\
         {rhai_examples}\n\
         \n\
         ## Create + run\n\
         Rhai (in-process): write {widgets}/<name>.rhai (the dir is\n\
         watched — saving hot-reloads the pane), then:\n\
            $ widget spawn --kind rhai_widget --title \"<Title>\" <name>.rhai\n\
         Subprocess: scaffold and spawn:\n\
            $ widget init my-widget   # writes my-widget.sh\n\
            $ widget spawn --title \"<Title>\" -- ./my-widget.sh\n\
         Validate subprocess output without launching the host:\n\
            $ ./my-widget.sh | widget validate\n\
         \n\
         ## Gotchas\n\
         - `on_bus(kind, payload)` is the Claude Code event BUS, not UI\n\
           events. UI arrives via on_click / on_toggle / on_tab_select /\n\
           on_input_change / on_input_submit / on_input_focus.\n\
         - Every interactive element needs a UNIQUE `id`, and each input/\n\
           textarea its own state field — sharing a value makes two boxes\n\
           mirror each other.\n\
         - Event-driven: call `request_render()` after mutating state;\n\
           `set_animating(true)` only if you need per-frame `on_frame(dt)`.\n\
         \n\
         ============================================================\n\
         AUTHORING GUIDE ({authoring})\n\
         ============================================================\n\
         \n\
         {guide}",
        authoring = doc("AUTHORING.md"),
        protocol = doc("src/protocol.rs"),
        rhai_mod = doc("src/rhai_widget.rs"),
        widgets = widgets_str,
        rhai_examples = list_rhai_examples(widgets_dir.as_deref()),
        guide = AUTHORING_GUIDE,
    );
    Ok(())
}

/// Bullet list of `.rhai` scripts in the widgets dir, or a hint if none.
fn list_rhai_examples(dir: Option<&Path>) -> String {
    let Some(dir) = dir else {
        return "  (set $HOME to locate ~/.jim/widgets)".to_string();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return format!("  (none yet — {} doesn't exist)", dir.display());
    };
    let mut paths: Vec<String> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rhai"))
        .map(|p| format!("  - {}", p.display()))
        .collect();
    paths.sort();
    if paths.is_empty() {
        format!("  (none yet in {})", dir.display())
    } else {
        paths.join("\n")
    }
}

// ---------- docs ----------

fn cmd_docs(filter: Option<&str>) -> Result<(), String> {
    let element = to_value(&schema_for!(Element))?;
    let msg = to_value(&schema_for!(WidgetMsg))?;
    let event = to_value(&schema_for!(HostEvent))?;

    let f = filter.map(|s| s.to_ascii_lowercase());
    let want = |section: &str, variants: &[&str]| -> bool {
        let Some(f) = &f else { return true };
        if f == section {
            return true;
        }
        variants.iter().any(|v| v == f)
    };

    let elem_variants = variant_names(&element);
    let msg_variants = variant_names(&msg);
    let event_variants = variant_names(&event);

    let elem_picked: Vec<&str> = elem_variants
        .iter()
        .copied()
        .filter(|v| f.as_deref().map(|fx| fx == *v).unwrap_or(false))
        .collect();
    let msg_picked: Vec<&str> = msg_variants
        .iter()
        .copied()
        .filter(|v| f.as_deref().map(|fx| fx == *v).unwrap_or(false))
        .collect();
    let event_picked: Vec<&str> = event_variants
        .iter()
        .copied()
        .filter(|v| f.as_deref().map(|fx| fx == *v).unwrap_or(false))
        .collect();

    if let Some(fx) = &f
        && !["element", "msg", "event"].contains(&fx.as_str())
        && elem_picked.is_empty()
        && msg_picked.is_empty()
        && event_picked.is_empty()
    {
        return Err(format!(
            "no such name `{fx}`. Try one of: element, msg, event, or {}",
            elem_variants
                .iter()
                .chain(msg_variants.iter())
                .chain(event_variants.iter())
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let mut out = io::stdout().lock();

    if want("element", &elem_variants) {
        writeln!(out, "ELEMENT — UI tree nodes (widget frame.root)").ok();
        writeln!(out, "{}", "=".repeat(60)).ok();
        render_tagged_enum(&mut out, &element, "type", filter_for(&f, &elem_variants));
        writeln!(out).ok();
    }

    if want("msg", &msg_variants) {
        writeln!(out, "WIDGET → HOST MESSAGES (one NDJSON line on stdout)").ok();
        writeln!(out, "{}", "=".repeat(60)).ok();
        render_tagged_enum(&mut out, &msg, "type", filter_for(&f, &msg_variants));
        writeln!(out).ok();
    }

    if want("event", &event_variants) {
        writeln!(out, "HOST → WIDGET EVENTS (one NDJSON line on stdin)").ok();
        writeln!(out, "{}", "=".repeat(60)).ok();
        render_tagged_enum(&mut out, &event, "event", filter_for(&f, &event_variants));
        writeln!(out).ok();
    }

    Ok(())
}

fn filter_for<'a>(f: &'a Option<String>, variants: &[&'a str]) -> Option<&'a str> {
    let fx = f.as_deref()?;
    if variants.iter().any(|v| v == &fx) {
        Some(fx)
    } else {
        None
    }
}

fn extract_tag_const(v: &Value) -> Option<&str> {
    if let Some(s) = v.get("const").and_then(|c| c.as_str()) {
        return Some(s);
    }
    let arr = v.get("enum").and_then(|e| e.as_array())?;
    if arr.len() == 1 {
        arr[0].as_str()
    } else {
        None
    }
}

fn variant_names(root: &Value) -> Vec<&str> {
    let one_of = root
        .get("oneOf")
        .and_then(|v| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[]);
    one_of
        .iter()
        .filter_map(|sub| {
            sub.get("properties")
                .and_then(|p| p.get("type").or_else(|| p.get("event")))
                .and_then(extract_tag_const)
        })
        .collect()
}

fn render_tagged_enum(
    out: &mut impl Write,
    root: &Value,
    tag_field: &str,
    only: Option<&str>,
) {
    let one_of = root.get("oneOf").and_then(|v| v.as_array());
    let Some(one_of) = one_of else {
        writeln!(out, "(no variants)").ok();
        return;
    };
    let defs = root.get("definitions");

    for sub in one_of {
        let props = sub.get("properties").and_then(|p| p.as_object());
        let Some(props) = props else { continue };
        let tag = props
            .get(tag_field)
            .and_then(extract_tag_const)
            .unwrap_or("?");
        if let Some(filter) = only
            && filter != tag
        {
            continue;
        }
        let desc = sub.get("description").and_then(|d| d.as_str()).unwrap_or("");
        writeln!(out).ok();
        writeln!(out, "  {{\"{tag_field}\": \"{tag}\", …}}").ok();
        if !desc.is_empty() {
            for line in wrap(desc, 72) {
                writeln!(out, "    {line}").ok();
            }
        }
        let required: Vec<&str> = sub
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .filter(|s| *s != tag_field)
                    .collect()
            })
            .unwrap_or_default();
        let mut field_rows: Vec<(String, bool, String)> = Vec::new();
        for (k, v) in props {
            if k == tag_field {
                continue;
            }
            let required = required.contains(&k.as_str());
            let ty = describe_type(v, defs);
            field_rows.push((k.clone(), required, ty));
        }
        field_rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        if !field_rows.is_empty() {
            writeln!(out, "    fields:").ok();
            for (k, req, ty) in field_rows {
                let marker = if req { "*" } else { " " };
                writeln!(out, "      {marker} {k}: {ty}").ok();
            }
            writeln!(out, "      (* = required)").ok();
        }
    }
}

fn describe_type(v: &Value, defs: Option<&Value>) -> String {
    if let Some(r) = v.get("$ref").and_then(|r| r.as_str()) {
        let name = r.rsplit('/').next().unwrap_or(r);
        if let Some(defs) = defs
            && let Some(def) = defs.get(name)
        {
            if let Some(enum_vals) = def.get("enum").and_then(|e| e.as_array()) {
                let opts: Vec<String> = enum_vals
                    .iter()
                    .filter_map(|x| x.as_str().map(str::to_owned))
                    .collect();
                if !opts.is_empty() {
                    return opts.join(" | ");
                }
            }
            if let Some(one_of) = def.get("oneOf").and_then(|o| o.as_array()) {
                let opts: Vec<String> = one_of
                    .iter()
                    .filter_map(|sub| {
                        sub.get("properties")
                            .and_then(|p| p.get("type").or_else(|| p.get("event")))
                            .and_then(|t| t.get("const"))
                            .and_then(|c| c.as_str())
                            .map(|s| format!("\"{s}\""))
                    })
                    .collect();
                if !opts.is_empty() {
                    return format!("one of: {}", opts.join(", "));
                }
            }
        }
        return name.to_string();
    }
    if let Some(any_of) = v.get("anyOf").and_then(|a| a.as_array()) {
        let parts: Vec<String> = any_of.iter().map(|s| describe_type(s, defs)).collect();
        return parts.join(" | ");
    }
    if let Some(all_of) = v.get("allOf").and_then(|a| a.as_array())
        && all_of.len() == 1
    {
        return describe_type(&all_of[0], defs);
    }
    if let Some(ty) = v.get("type").and_then(|t| t.as_str()) {
        if ty == "array" {
            let items = v
                .get("items")
                .map(|i| describe_type(i, defs))
                .unwrap_or_else(|| "any".to_string());
            return format!("[{items}]");
        }
        if ty == "null" {
            return "null".to_string();
        }
        return ty.to_string();
    }
    if let Some(tys) = v.get("type").and_then(|t| t.as_array()) {
        let parts: Vec<String> = tys
            .iter()
            .filter_map(|t| t.as_str())
            .map(str::to_owned)
            .collect();
        if !parts.is_empty() {
            return parts.join(" | ");
        }
    }
    "any".to_string()
}

fn wrap(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for paragraph in s.split('\n') {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            if !line.is_empty() && line.len() + 1 + word.len() > width {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        lines.push(line);
    }
    lines
}

// ---------- schema ----------

fn cmd_schema(which: SchemaWhich) -> Result<(), String> {
    let v = match which {
        SchemaWhich::Element => to_value(&schema_for!(Element))?,
        SchemaWhich::Msg => to_value(&schema_for!(WidgetMsg))?,
        SchemaWhich::Event => to_value(&schema_for!(HostEvent))?,
        SchemaWhich::All => serde_json::json!({
            "WidgetMsg": to_value(&schema_for!(WidgetMsg))?,
            "HostEvent": to_value(&schema_for!(HostEvent))?,
            "Element": to_value(&schema_for!(Element))?,
        }),
    };
    let pretty = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
    println!("{pretty}");
    Ok(())
}

fn to_value<T: serde::Serialize>(v: &T) -> Result<Value, String> {
    serde_json::to_value(v).map_err(|e| e.to_string())
}

// ---------- validate ----------

fn cmd_validate(events: bool) -> Result<(), String> {
    let stdin = io::stdin();
    let mut errors = 0usize;
    let mut total = 0usize;
    for (idx, line) in stdin.lock().lines().enumerate() {
        let lineno = idx + 1;
        let raw = line.map_err(|e| e.to_string())?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        total += 1;
        let parsed: Result<(), serde_json::Error> = if events {
            serde_json::from_str::<HostEvent>(trimmed).map(|_| ())
        } else {
            serde_json::from_str::<WidgetMsg>(trimmed).map(|_| ())
        };
        if let Err(e) = parsed {
            errors += 1;
            eprintln!("line {lineno}: {e}");
            let snippet = if trimmed.len() > 120 {
                format!("{}…", &trimmed[..120])
            } else {
                trimmed.to_string()
            };
            eprintln!("  {snippet}");
        }
    }
    let kind = if events { "events" } else { "messages" };
    if errors == 0 {
        eprintln!("ok — {total} {kind} validated");
        Ok(())
    } else {
        Err(format!("{errors} of {total} {kind} failed to parse"))
    }
}

// ---------- examples ----------

fn cmd_examples(name: Option<&str>) -> Result<(), String> {
    match name {
        None => {
            println!("Bundled examples (use `widget examples <name>` to print):");
            for ex in EXAMPLES {
                let summary = ex
                    .body
                    .lines()
                    .skip(1)
                    .find(|l| l.starts_with('#') && !l.starts_with("#!"))
                    .map(|l| l.trim_start_matches('#').trim())
                    .unwrap_or("");
                println!("  {:<12}  {summary}", ex.name);
            }
            Ok(())
        }
        Some(want) => {
            let ex = EXAMPLES
                .iter()
                .find(|e| e.name == want)
                .ok_or_else(|| format!("no example named `{want}`"))?;
            print!("{}", ex.body);
            Ok(())
        }
    }
}

// ---------- init ----------

fn cmd_init(name: &str, force: bool) -> Result<(), String> {
    let mut path = PathBuf::from(name);
    if path.extension().is_none() {
        path.set_extension("sh");
    }
    if path.exists() && !force {
        return Err(format!(
            "{} already exists; pass --force to overwrite",
            path.display()
        ));
    }
    fs::write(&path, STARTER).map_err(|e| format!("write {}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&path).map_err(|e| e.to_string())?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms).map_err(|e| e.to_string())?;
    }
    eprintln!("wrote {}", path.display());
    eprintln!("try: ./{}", path.display());
    Ok(())
}
