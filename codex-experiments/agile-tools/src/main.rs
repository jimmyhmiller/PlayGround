use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use clap::{Parser, Subcommand};
use chrono::{SecondsFormat, Utc};
use dirs_next::home_dir;
use ed25519_dalek::{Signer, SigningKey};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use ulid::Ulid;

const DEFAULT_PROJECTS_ROOT: &str = "~/.scope/projects";
const DEFAULT_ID_PREFIX: &str = "SC";
const DEFAULT_MAX_ATTEMPTS: u32 = 100;

const ADJECTIVES: &[&str] = &[
    "brisk", "silent", "calm", "rapid", "lucid", "steady", "bright", "sharp", "clear",
    "bold", "swift", "gentle", "mild", "eager", "keen", "solid", "fresh", "dry", "cool",
];

const ANIMALS: &[&str] = &[
    "otter", "falcon", "panda", "tiger", "wren", "lynx", "orca", "koala", "badger",
    "heron", "ram", "fox", "yak", "eel", "dolphin", "wolf", "raven", "moose",
];

#[derive(Parser)]
#[command(name = "scope", version, about = "Scope issue tracking CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Issues {
        #[command(subcommand)]
        command: IssuesCommand,
    },
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
    Skills {
        #[command(subcommand)]
        command: SkillsCommand,
    },
}

#[derive(Subcommand)]
enum SkillsCommand {
    Install {
        #[arg(long)]
        codex: bool,
        #[arg(long)]
        claude: bool,
        #[arg(long)]
        force: bool,
    },
}

#[derive(Subcommand)]
enum AuthCommand {
    Keys {
        #[command(subcommand)]
        command: KeysCommand,
    },
    Signup {
        #[arg(long)]
        remote: String,
    },
    Whoami,
}

#[derive(Subcommand)]
enum KeysCommand {
    Generate {
        #[arg(long)]
        force: bool,
    },
    Show,
}

#[derive(Subcommand)]
enum IssuesCommand {
    Init {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        sync_engine: Option<String>,
    },
    Create {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        title: String,
        #[arg(long)]
        priority: Option<String>,
        #[arg(long)]
        assignee: Option<String>,
        #[arg(long = "label")]
        labels: Vec<String>,
        #[arg(long, conflicts_with = "body_file")]
        body: Option<String>,
        #[arg(long)]
        body_file: Option<PathBuf>,
    },
    Update {
        #[arg(long)]
        project: Option<String>,
        id: String,
        #[arg(long)]
        title: Option<String>,
        #[arg(long)]
        status: Option<String>,
        #[arg(long)]
        priority: Option<String>,
        #[arg(long)]
        assignee: Option<String>,
        #[arg(long)]
        clear_assignee: bool,
        #[arg(long = "add-label")]
        add_labels: Vec<String>,
        #[arg(long = "remove-label")]
        remove_labels: Vec<String>,
    },
    Edit {
        #[arg(long)]
        project: Option<String>,
        id: String,
        #[arg(long)]
        editor: Option<String>,
    },
    Close {
        #[arg(long)]
        project: Option<String>,
        id: String,
    },
    Reopen {
        #[arg(long)]
        project: Option<String>,
        id: String,
        #[arg(long)]
        status: Option<String>,
    },
    Delete {
        #[arg(long)]
        project: Option<String>,
        id: String,
        #[arg(long)]
        force: bool,
    },
    Restore {
        #[arg(long)]
        project: Option<String>,
        id: String,
    },
    List {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        status: Option<String>,
        #[arg(long)]
        assignee: Option<String>,
        #[arg(long = "label")]
        labels: Vec<String>,
        #[arg(long)]
        priority: Option<String>,
        #[arg(long)]
        query: Option<String>,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        json: bool,
    },
    Show {
        #[arg(long)]
        project: Option<String>,
        id: String,
        #[arg(long)]
        json: bool,
    },
    Rebuild {
        #[arg(long)]
        project: Option<String>,
    },
    Conflicts {
        #[arg(long)]
        project: Option<String>,
        #[command(subcommand)]
        command: ConflictsCommand,
    },
    Comments {
        #[command(subcommand)]
        command: CommentsCommand,
    },
    Sync {
        #[arg(long)]
        project: Option<String>,
        #[command(subcommand)]
        command: SyncCommand,
    },
    Projects {
        #[arg(long)]
        json: bool,
    },
    Project {
        #[command(subcommand)]
        command: ProjectCommand,
    },
    Index {
        #[arg(long)]
        project: Option<String>,
        #[command(subcommand)]
        command: IndexCommand,
    },
}

#[derive(Subcommand)]
enum IndexCommand {
    Rebuild,
    Status,
    Verify,
}

#[derive(Subcommand)]
enum ProjectCommand {
    Show {
        #[arg(long)]
        project: Option<String>,
    },
    ConfigSet {
        #[arg(long)]
        project: Option<String>,
        key: String,
        value: String,
    },
    Links {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Link {
        #[arg(long)]
        project: String,
        #[arg(long)]
        path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ConflictsCommand {
    List,
    Show {
        id: String,
    },
    Resolve {
        id: String,
        #[arg(long)]
        keep: String,
    },
}

#[derive(Subcommand)]
enum CommentsCommand {
    List {
        #[arg(long)]
        project: Option<String>,
        id: String,
        #[arg(long)]
        json: bool,
    },
    Add {
        #[arg(long)]
        project: Option<String>,
        id: String,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        body_file: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum SyncCommand {
    Pull,
    Push,
    Status,
}

#[derive(Debug, Deserialize, Default)]
struct GlobalConfig {
    default_project: Option<String>,
    editor: Option<String>,
    projects: Option<HashMap<String, String>>,
    paths: Option<PathsConfig>,
    ids: Option<IdsConfig>,
    events: Option<EventsConfig>,
    sync: Option<SyncConfig>,
}

#[derive(Debug, Deserialize, Default)]
struct PathsConfig {
    projects_root: Option<String>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct IdsConfig {
    prefix: Option<String>,
    pattern: Option<String>,
    wordlist_adjectives: Option<String>,
    wordlist_animals: Option<String>,
    max_attempts: Option<u32>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct EventsConfig {
    conflict_window_seconds: Option<i64>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct SyncConfig {
    engine: Option<String>,
    remote: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct ProjectConfig {
    name: Option<String>,
    paths: Option<ProjectPathsConfig>,
    ids: Option<IdsConfig>,
    events: Option<EventsConfig>,
    sync: Option<SyncConfig>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct ProjectPathsConfig {
    root: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct IssueFrontmatter {
    id: String,
    title: String,
    status: String,
    priority: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    assignee: Option<String>,
    #[serde(default)]
    labels: Vec<String>,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, Serialize)]
struct Event {
    id: String,
    #[serde(rename = "type")]
    event_type: String,
    issue: String,
    author: String,
    ts: String,
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct EventRecord {
    #[allow(dead_code)]
    id: String,
    #[serde(rename = "type")]
    event_type: String,
    issue: String,
    author: String,
    ts: String,
    data: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexFile {
    format_version: u32,
    generated_at: String,
    project: String,
    issues: Vec<IndexIssue>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConflictFile {
    format_version: u32,
    generated_at: String,
    conflicts: Vec<ConflictRecord>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ConflictRecord {
    id: String,
    issue: String,
    field: String,
    local_event: String,
    remote_event: String,
    local_value: serde_json::Value,
    remote_value: serde_json::Value,
    detected_at: String,
    state: String,
}

#[derive(Debug, Default, Clone)]
struct IssueState {
    id: String,
    title: String,
    status: String,
    priority: String,
    assignee: Option<String>,
    labels: Vec<String>,
    body: String,
    created_at: String,
    updated_at: String,
    comments: Vec<IssueComment>,
}

#[derive(Debug, Clone)]
struct IssueComment {
    author: String,
    ts: String,
    body: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexIssue {
    id: String,
    title: String,
    status: String,
    priority: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    assignee: Option<String>,
    labels: Vec<String>,
    created_at: String,
    updated_at: String,
    path: String,
    events_path: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Issues { command } => match command {
            IssuesCommand::Init { project, sync_engine } => {
                let project = resolve_init_project(project)?;
                issues_init(&project, sync_engine.as_deref())?;
            }
            IssuesCommand::Create {
                project,
                title,
                priority,
                assignee,
                labels,
                body,
                body_file,
            } => {
                let project = resolve_project(project)?;
                issues_create(&project, &title, priority, assignee, labels, body, body_file)?;
            }
            IssuesCommand::Update {
                project,
                id,
                title,
                status,
                priority,
                assignee,
                clear_assignee,
                add_labels,
                remove_labels,
            } => {
                let project = resolve_project(project)?;
                let id = resolve_issue_id(&project, &id)?;
                issues_update(
                    &project,
                    &id,
                    title,
                    status,
                    priority,
                    assignee,
                    clear_assignee,
                    add_labels,
                    remove_labels,
                )?;
            }
            IssuesCommand::Edit { project, id, editor } => {
                let project = resolve_project(project)?;
                let id = resolve_issue_id(&project, &id)?;
                issues_edit(&project, &id, editor)?;
            }
            IssuesCommand::Close { project, id } => {
                let project = resolve_project(project)?;
                let id = resolve_issue_id(&project, &id)?;
                issues_close(&project, &id)?;
            }
            IssuesCommand::Reopen { project, id, status } => {
                let project = resolve_project(project)?;
                let id = resolve_issue_id(&project, &id)?;
                issues_reopen(&project, &id, status)?;
            }
            IssuesCommand::Delete { project, id, force } => {
                let project = resolve_project(project)?;
                let id = resolve_issue_id(&project, &id)?;
                issues_delete(&project, &id, force)?;
            }
            IssuesCommand::Restore { project, id } => {
                let project = resolve_project(project)?;
                let id = resolve_issue_id(&project, &id)?;
                issues_restore(&project, &id)?;
            }
            IssuesCommand::List {
                project,
                status,
                assignee,
                labels,
                priority,
                query,
                limit,
                json,
            } => {
                let project = resolve_project(project)?;
                issues_list(&project, status, assignee, labels, priority, query, limit, json)?;
            }
            IssuesCommand::Show { project, id, json } => {
                let project = resolve_project(project)?;
                let id = resolve_issue_id(&project, &id)?;
                issues_show(&project, &id, json)?;
            }
            IssuesCommand::Rebuild { project } => {
                let project = resolve_project(project)?;
                issues_rebuild(&project)?;
            }
            IssuesCommand::Conflicts { project, command } => {
                let project = resolve_project(project)?;
                match command {
                    ConflictsCommand::List => issues_conflicts_list(&project)?,
                    ConflictsCommand::Show { id } => {
                        let id = resolve_issue_id(&project, &id)?;
                        issues_conflicts_show(&project, &id)?
                    }
                    ConflictsCommand::Resolve { id, keep } => {
                        let id = resolve_issue_id(&project, &id)?;
                        issues_conflicts_resolve(&project, &id, &keep)?
                    }
                }
            }
            IssuesCommand::Comments { command } => match command {
                CommentsCommand::List { project, id, json } => {
                    let project = resolve_project(project)?;
                    let id = resolve_issue_id(&project, &id)?;
                    issues_comments_list(&project, &id, json)?;
                }
                CommentsCommand::Add {
                    project,
                    id,
                    body,
                    body_file,
                } => {
                    let project = resolve_project(project)?;
                    let id = resolve_issue_id(&project, &id)?;
                    issues_comments_add(&project, &id, body, body_file)?;
                }
            },
            IssuesCommand::Sync { project, command } => {
                let project = resolve_project(project)?;
                match command {
                    SyncCommand::Pull => issues_sync_pull(&project)?,
                    SyncCommand::Push => issues_sync_push(&project)?,
                    SyncCommand::Status => issues_sync_status(&project)?,
                }
            }
            IssuesCommand::Projects { json } => {
                issues_projects(json)?;
            }
            IssuesCommand::Project { command } => match command {
                ProjectCommand::Show { project } => {
                    let project = resolve_project(project)?;
                    issues_project_show(&project)?;
                }
                ProjectCommand::ConfigSet { project, key, value } => {
                    let project = resolve_project(project)?;
                    issues_project_config_set(&project, &key, &value)?;
                }
                ProjectCommand::Links { project, json } => {
                    let project = project.or_else(|| read_global_config().ok().and_then(|c| c.default_project));
                    issues_project_links_list(project.as_deref(), json)?;
                }
                ProjectCommand::Link { project, path } => {
                    issues_project_link(&project, path)?;
                }
            },
            IssuesCommand::Index { project, command } => {
                let project = resolve_project(project)?;
                match command {
                    IndexCommand::Rebuild => issues_index_rebuild(&project)?,
                    IndexCommand::Status => issues_index_status(&project)?,
                    IndexCommand::Verify => issues_index_verify(&project)?,
                }
            }
        },
        Command::Auth { command } => match command {
            AuthCommand::Keys { command } => match command {
                KeysCommand::Generate { force } => auth_keys_generate(force)?,
                KeysCommand::Show => auth_keys_show()?,
            },
            AuthCommand::Signup { remote } => auth_signup(&remote)?,
            AuthCommand::Whoami => auth_whoami()?,
        },
        Command::Skills { command } => match command {
            SkillsCommand::Install {
                codex,
                claude,
                force,
            } => {
                skills_install(codex, claude, force)?;
            }
        },
    }

    Ok(())
}

fn issues_init(project: &str, _sync_engine: Option<&str>) -> anyhow::Result<()> {
    let projects_root = projects_root()?;
    let project_root = projects_root.join(project);
    let project_existed = project_root.exists();
    fs::create_dir_all(project_root.join("issues"))?;
    fs::create_dir_all(project_root.join("events"))?;
    fs::create_dir_all(project_root.join("index"))?;

    let project_toml = project_root.join("project.toml");
    if !project_toml.exists() {
        let content = format!("name = \"{}\"\n", project);
        fs::write(project_toml, content)?;
    }

    let index_path = project_root.join("index").join("issues.json");
    if !index_path.exists() {
        let index = IndexFile {
            format_version: 1,
            generated_at: now_ts(),
            project: project.to_string(),
            issues: Vec::new(),
        };
        write_index(&index_path, &index)?;
    }

    let global_config = global_config_path()?;
    let mut created_global = false;
    if !global_config.exists() {
        let content = format!("default_project = \"{}\"\n", project);
        fs::create_dir_all(global_config.parent().unwrap())?;
        fs::write(&global_config, content)?;
        created_global = true;
    }

    if project_existed {
        println!("Project {} already exists; ensured required directories.", project);
    } else {
        println!("Initialized project {}", project);
        println!("Undo (if you just created this project): rm -rf {}", project_root.display());
    }
    if created_global {
        println!("Undo (if you want to remove the created config): rm -f {}", global_config.display());
    }

    Ok(())
}

fn resolve_init_project(project: Option<String>) -> anyhow::Result<String> {
    if let Some(project) = project {
        return Ok(project);
    }

    let cwd = std::env::current_dir()?;
    let name = cwd
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("cannot derive project name from cwd"))?
        .to_string();

    println!("No project specified. Using current directory name: {}", name);
    Ok(name)
}

fn issues_create(
    project: &str,
    title: &str,
    priority: Option<String>,
    assignee: Option<String>,
    labels: Vec<String>,
    body: Option<String>,
    body_file: Option<PathBuf>,
) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let ids_config = resolved_ids_config(project)?;
    let id = generate_issue_id(&project_root, &ids_config)?;

    let now = now_ts();
    let frontmatter = IssueFrontmatter {
        id: id.clone(),
        title: title.to_string(),
        status: "todo".to_string(),
        priority: priority.unwrap_or_else(|| "p2".to_string()),
        assignee,
        labels,
        created_at: now.clone(),
        updated_at: now.clone(),
    };

    let body = if let Some(body) = body {
        body
    } else if let Some(path) = body_file {
        fs::read_to_string(path)?
    } else {
        "## Summary\n\n## Acceptance Criteria\n".to_string()
    };

    let issue_path = project_root.join("issues").join(format!("{}.md", id));
    write_issue(&issue_path, &frontmatter, &body)?;

    let events_path = project_root.join("events").join(format!("{}.jsonl", id));
    append_event(
        &events_path,
        "issue.create",
        &id,
        serde_json::json!({
            "title": frontmatter.title,
            "status": frontmatter.status,
            "priority": frontmatter.priority,
            "assignee": frontmatter.assignee,
            "labels": frontmatter.labels,
            "body": body,
        }),
    )?;

    let mut index = read_index_or_empty(project, &project_root)?;
    index.issues.retain(|i| i.id != id);
    index.issues.push(IndexIssue {
        id: id.clone(),
        title: frontmatter.title,
        status: frontmatter.status,
        priority: frontmatter.priority,
        assignee: frontmatter.assignee,
        labels: frontmatter.labels,
        created_at: frontmatter.created_at,
        updated_at: frontmatter.updated_at,
        path: format!("issues/{}.md", id),
        events_path: format!("events/{}.jsonl", id),
    });
    index.generated_at = now_ts();
    write_index(&project_root.join("index").join("issues.json"), &index)?;

    println!("{}", id);
    Ok(())
}

fn issues_update(
    project: &str,
    id: &str,
    title: Option<String>,
    status: Option<String>,
    priority: Option<String>,
    assignee: Option<String>,
    clear_assignee: bool,
    add_labels: Vec<String>,
    remove_labels: Vec<String>,
) -> anyhow::Result<()> {
    if title.is_none()
        && status.is_none()
        && priority.is_none()
        && assignee.is_none()
        && !clear_assignee
        && add_labels.is_empty()
        && remove_labels.is_empty()
    {
        return Err(anyhow::anyhow!("no updates provided"));
    }

    let project_root = project_root(project)?;
    let issue_path = project_root.join("issues").join(format!("{}.md", id));
    let content = fs::read_to_string(&issue_path)?;
    let (mut frontmatter, body) = parse_frontmatter(&content)?;

    if let Some(title) = title.clone() {
        frontmatter.title = title;
    }
    if let Some(status) = status.clone() {
        frontmatter.status = status;
    }
    if let Some(priority) = priority.clone() {
        frontmatter.priority = priority;
    }
    if clear_assignee {
        frontmatter.assignee = None;
    } else if let Some(assignee) = assignee.clone() {
        frontmatter.assignee = Some(assignee);
    }

    if !add_labels.is_empty() || !remove_labels.is_empty() {
        let mut label_set: HashSet<String> = frontmatter.labels.into_iter().collect();
        for l in add_labels.iter() {
            label_set.insert(l.to_string());
        }
        for l in remove_labels.iter() {
            label_set.remove(l);
        }
        let mut labels: Vec<String> = label_set.into_iter().collect();
        labels.sort();
        frontmatter.labels = labels;
    }

    frontmatter.updated_at = now_ts();
    write_issue(&issue_path, &frontmatter, &body)?;

    let events_path = project_root.join("events").join(format!("{}.jsonl", id));
    let mut data = serde_json::Map::new();
    if let Some(title) = title {
        data.insert("title".to_string(), serde_json::Value::String(title));
    }
    if let Some(status) = status {
        data.insert("status".to_string(), serde_json::Value::String(status));
    }
    if let Some(priority) = priority {
        data.insert("priority".to_string(), serde_json::Value::String(priority));
    }
    if clear_assignee {
        data.insert("assignee".to_string(), serde_json::Value::Null);
    } else if let Some(assignee) = assignee {
        data.insert(
            "assignee".to_string(),
            serde_json::Value::String(assignee),
        );
    }
    if !add_labels.is_empty() || !remove_labels.is_empty() {
        let labels = frontmatter
            .labels
            .iter()
            .map(|l| serde_json::Value::String(l.clone()))
            .collect::<Vec<_>>();
        data.insert("labels".to_string(), serde_json::Value::Array(labels));
    }

    append_event(&events_path, "issue.update", id, serde_json::Value::Object(data))?;
    upsert_index_from_issue(project, &project_root, &frontmatter)?;
    Ok(())
}

fn issues_edit(project: &str, id: &str, editor: Option<String>) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let issue_path = project_root.join("issues").join(format!("{}.md", id));
    let editor_cmd = editor.or_else(|| resolved_editor());
    let editor_cmd = editor_cmd.ok_or_else(|| anyhow::anyhow!("editor not configured"))?;

    let status = std::process::Command::new(&editor_cmd)
        .arg(&issue_path)
        .status()?;
    if !status.success() {
        return Err(anyhow::anyhow!("editor exited with error"));
    }

    let content = fs::read_to_string(&issue_path)?;
    let (mut frontmatter, body) = parse_frontmatter(&content)?;
    frontmatter.updated_at = now_ts();
    write_issue(&issue_path, &frontmatter, &body)?;

    let events_path = project_root.join("events").join(format!("{}.jsonl", id));
    append_event(
        &events_path,
        "issue.update",
        id,
        serde_json::json!({ "body": body }),
    )?;
    upsert_index_from_issue(project, &project_root, &frontmatter)?;
    Ok(())
}

fn issues_close(project: &str, id: &str) -> anyhow::Result<()> {
    issues_set_status(project, id, "done", "issue.close")
}

fn issues_reopen(project: &str, id: &str, status: Option<String>) -> anyhow::Result<()> {
    let status = status.unwrap_or_else(|| "todo".to_string());
    issues_set_status(project, id, &status, "issue.reopen")
}

fn issues_delete(project: &str, id: &str, force: bool) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let issue_path = project_root.join("issues").join(format!("{}.md", id));
    let events_path = project_root.join("events").join(format!("{}.jsonl", id));

    if force {
        if issue_path.exists() {
            fs::remove_file(&issue_path)?;
        }
        if events_path.exists() {
            fs::remove_file(&events_path)?;
        }
        let mut index = read_index_or_empty(project, &project_root)?;
        index.issues.retain(|i| i.id != id);
        index.generated_at = now_ts();
        write_index(&project_root.join("index").join("issues.json"), &index)?;
        println!("Deleted {} (hard).", id);
        println!("No undo available (used --force).");
        return Ok(());
    }

    if !issue_path.exists() {
        return Err(anyhow::anyhow!("issue not found: {}", id));
    }

    let (trash_issue_path, trash_events_path) = issue_trash_paths(&project_root, id);
    if let Some(parent) = trash_issue_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = trash_events_path.parent() {
        fs::create_dir_all(parent)?;
    }

    if trash_issue_path.exists() {
        fs::remove_file(&trash_issue_path)?;
    }
    fs::rename(&issue_path, &trash_issue_path)?;

    if events_path.exists() {
        append_event(&events_path, "issue.delete", id, serde_json::json!({}))?;
        if trash_events_path.exists() {
            fs::remove_file(&trash_events_path)?;
        }
        fs::rename(&events_path, &trash_events_path)?;
    }

    let mut index = read_index_or_empty(project, &project_root)?;
    index.issues.retain(|i| i.id != id);
    index.generated_at = now_ts();
    write_index(&project_root.join("index").join("issues.json"), &index)?;
    println!("Deleted {}.", id);
    println!("Undo: scope issues restore --project {} {}", project, id);
    Ok(())
}

fn issues_restore(project: &str, id: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let issue_path = project_root.join("issues").join(format!("{}.md", id));
    let events_path = project_root.join("events").join(format!("{}.jsonl", id));
    let (trash_issue_path, trash_events_path) = issue_trash_paths(&project_root, id);

    if issue_path.exists() {
        return Err(anyhow::anyhow!("issue already exists: {}", id));
    }
    if !trash_issue_path.exists() {
        return Err(anyhow::anyhow!("no trashed issue found for {}", id));
    }

    if let Some(parent) = issue_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = events_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::rename(&trash_issue_path, &issue_path)?;
    if trash_events_path.exists() {
        if events_path.exists() {
            return Err(anyhow::anyhow!("events already exist for {}", id));
        }
        fs::rename(&trash_events_path, &events_path)?;
    }

    let content = fs::read_to_string(&issue_path)?;
    let (frontmatter, _) = parse_frontmatter(&content)?;
    upsert_index_from_issue(project, &project_root, &frontmatter)?;
    println!("Restored {}.", id);
    Ok(())
}

fn issue_trash_paths(project_root: &Path, id: &str) -> (PathBuf, PathBuf) {
    let trash_root = project_root.join("trash");
    let issue_path = trash_root.join("issues").join(format!("{}.md", id));
    let events_path = trash_root.join("events").join(format!("{}.jsonl", id));
    (issue_path, events_path)
}

fn issues_list(
    project: &str,
    status: Option<String>,
    assignee: Option<String>,
    labels: Vec<String>,
    priority: Option<String>,
    query: Option<String>,
    limit: Option<usize>,
    json: bool,
) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let index = read_index(&project_root)?;
    let mut issues: Vec<IndexIssue> = index
        .issues
        .into_iter()
        .filter(|i| status.as_ref().map_or(true, |s| &i.status == s))
        .filter(|i| assignee.as_ref().map_or(true, |a| i.assignee.as_deref() == Some(a.as_str())))
        .filter(|i| priority.as_ref().map_or(true, |p| &i.priority == p))
        .filter(|i| {
            labels.iter().all(|l| i.labels.iter().any(|il| il == l))
        })
        .filter(|i| {
            query.as_ref().map_or(true, |q| {
                let q = q.to_lowercase();
                i.title.to_lowercase().contains(&q) || i.id.to_lowercase().contains(&q)
            })
        })
        .collect();

    issues.sort_by(|a, b| a.updated_at.cmp(&b.updated_at));
    if let Some(limit) = limit {
        issues.truncate(limit);
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&issues)?);
        return Ok(());
    }

    let mut out = String::new();
    out.push_str("ID\tSTATUS\tTITLE\n");
    for issue in issues {
        out.push_str(&format!("{}\t{}\t{}\n", issue.id, issue.status, issue.title));
    }
    print!("{}", out);
    Ok(())
}

fn resolve_issue_id(project: &str, id: &str) -> anyhow::Result<String> {
    let project_root = project_root(project)?;
    let index = read_index(&project_root)?;
    // Exact match first
    if index.issues.iter().any(|i| i.id == id) {
        return Ok(id.to_string());
    }
    // Substring match
    let matches: Vec<&IndexIssue> = index
        .issues
        .iter()
        .filter(|i| i.id.contains(id))
        .collect();
    match matches.len() {
        0 => Err(anyhow::anyhow!("no issue found matching '{}'", id)),
        1 => Ok(matches[0].id.clone()),
        _ => {
            let mut msg = format!(
                "ambiguous ID '{}' matches {} issues:\n",
                id,
                matches.len()
            );
            for m in &matches {
                msg.push_str(&format!("  {} - {}\n", m.id, m.title));
            }
            Err(anyhow::anyhow!(msg))
        }
    }
}

fn issues_show(project: &str, id: &str, json: bool) -> anyhow::Result<()> {
    let id = resolve_issue_id(project, id)?;
    let project_root = project_root(project)?;
    let issue_path = project_root.join("issues").join(format!("{}.md", id));
    let content = fs::read_to_string(&issue_path)
        .map_err(|_| anyhow::anyhow!("issue file not found for '{}'", id))?;
    let (frontmatter, body) = parse_frontmatter(&content)?;

    if json {
        let value = serde_json::json!({
            "frontmatter": frontmatter,
            "body": body,
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }

    print!("{}", content);
    Ok(())
}

fn issues_index_rebuild(project: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let issues_dir = project_root.join("issues");
    let mut issues = Vec::new();

    if issues_dir.exists() {
        for entry in fs::read_dir(issues_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let content = fs::read_to_string(&path)?;
            let (frontmatter, _) = parse_frontmatter(&content)?;
            let id = frontmatter.id.clone();
            issues.push(IndexIssue {
                id: id.clone(),
                title: frontmatter.title,
                status: frontmatter.status,
                priority: frontmatter.priority,
                assignee: frontmatter.assignee,
                labels: frontmatter.labels,
                created_at: frontmatter.created_at,
                updated_at: frontmatter.updated_at,
                path: format!("issues/{}.md", id),
                events_path: format!("events/{}.jsonl", id),
            });
        }
    }

    let index = IndexFile {
        format_version: 1,
        generated_at: now_ts(),
        project: project.to_string(),
        issues,
    };
    write_index(&project_root.join("index").join("issues.json"), &index)?;
    Ok(())
}

fn issues_rebuild(project: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let events_dir = project_root.join("events");
    let issues_dir = project_root.join("issues");
    fs::create_dir_all(&issues_dir)?;

    let conflict_window = resolved_conflict_window_seconds(project)?;
    let mut new_conflicts: Vec<ConflictRecord> = Vec::new();

    if events_dir.exists() {
        for entry in fs::read_dir(&events_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let state = rebuild_issue_from_events(&path)?;
            if let Some(state) = state {
                let conflicts = detect_conflicts_from_events(&path, conflict_window)?;
                new_conflicts.extend(conflicts);
                let issue_path = issues_dir.join(format!("{}.md", state.id));
                let (frontmatter, body) = issue_state_to_snapshot(&state);
                write_issue(&issue_path, &frontmatter, &body)?;
                upsert_index_from_issue(project, &project_root, &frontmatter)?;
            }
        }
    }

    let mut conflicts_file = read_conflicts_or_empty(&project_root)?;
    merge_conflicts(&mut conflicts_file, new_conflicts);
    write_conflicts(&project_root, &conflicts_file)?;

    issues_index_rebuild(project)?;
    Ok(())
}

fn issues_conflicts_list(project: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let conflicts = read_conflicts_or_empty(&project_root)?;
    for conflict in conflicts.conflicts.iter().filter(|c| c.state == "unresolved") {
        println!(
            "{}\t{}\t{}\t{}",
            conflict.id, conflict.issue, conflict.field, conflict.state
        );
    }
    Ok(())
}

fn issues_conflicts_show(project: &str, id: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let conflicts = read_conflicts_or_empty(&project_root)?;
    if let Some(conflict) = conflicts.conflicts.iter().find(|c| c.id == id) {
        println!("{}", serde_json::to_string_pretty(conflict)?);
        return Ok(());
    }
    Err(anyhow::anyhow!("conflict not found"))
}

fn issues_conflicts_resolve(project: &str, id: &str, keep: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let mut conflicts = read_conflicts_or_empty(&project_root)?;
    let Some(conflict) = conflicts.conflicts.iter_mut().find(|c| c.id == id) else {
        return Err(anyhow::anyhow!("conflict not found"));
    };

    let chosen = match keep {
        "local" => conflict.local_value.clone(),
        "remote" => conflict.remote_value.clone(),
        _ => return Err(anyhow::anyhow!("keep must be 'local' or 'remote'")),
    };

    apply_conflict_resolution(
        project,
        &project_root,
        conflict,
        chosen,
        keep.to_string(),
    )?;
    conflict.state = "resolved".to_string();
    conflicts.generated_at = now_ts();
    write_conflicts(&project_root, &conflicts)?;
    Ok(())
}

fn issues_comments_add(
    project: &str,
    id: &str,
    body: Option<String>,
    body_file: Option<PathBuf>,
) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let body = if let Some(body) = body {
        body
    } else if let Some(path) = body_file {
        fs::read_to_string(path)?
    } else {
        return Err(anyhow::anyhow!("comment body required"));
    };

    let events_path = project_root.join("events").join(format!("{}.jsonl", id));
    append_event(
        &events_path,
        "issue.comment",
        id,
        serde_json::json!({ "body": body }),
    )?;

    if let Some(state) = rebuild_issue_from_events(&events_path)? {
        let (frontmatter, body) = issue_state_to_snapshot(&state);
        let issue_path = project_root.join("issues").join(format!("{}.md", state.id));
        write_issue(&issue_path, &frontmatter, &body)?;
        upsert_index_from_issue(project, &project_root, &frontmatter)?;
    }
    Ok(())
}

fn issues_comments_list(project: &str, id: &str, json: bool) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let events_path = project_root.join("events").join(format!("{}.jsonl", id));
    if !events_path.exists() {
        return Err(anyhow::anyhow!("issue events not found"));
    }
    let content = fs::read_to_string(&events_path)?;
    let mut comments = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let event: EventRecord = serde_json::from_str(line)?;
        if event.event_type == "issue.comment" {
            if let Some(body) = event.data.get("body").and_then(|v| v.as_str()) {
                comments.push(serde_json::json!({
                    "author": event.author,
                    "ts": event.ts,
                    "body": body,
                }));
            }
        }
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&comments)?);
        return Ok(());
    }

    for c in comments {
        println!(
            "{}\t{}\t{}",
            c.get("ts").and_then(|v| v.as_str()).unwrap_or(""),
            c.get("author").and_then(|v| v.as_str()).unwrap_or(""),
            c.get("body").and_then(|v| v.as_str()).unwrap_or("")
        );
    }
    Ok(())
}

fn issues_sync_pull(project: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let sync = resolved_sync_config(project)?;
    match sync.engine.as_deref().unwrap_or("git") {
        "git" => {
            run_git(&project_root, &["pull"])?;
            issues_rebuild(project)?;
            Ok(())
        }
        "noop" => Ok(()),
        "service" => {
            let remote = sync
                .remote
                .ok_or_else(|| anyhow::anyhow!("sync.remote is required for service engine"))?;
            service_sync_pull(project, &project_root, &remote)
        }
        other => Err(anyhow::anyhow!("sync engine not implemented: {}", other)),
    }
}

fn issues_sync_push(project: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let sync = resolved_sync_config(project)?;
    match sync.engine.as_deref().unwrap_or("git") {
        "git" => run_git(&project_root, &["push"]),
        "noop" => Ok(()),
        "service" => {
            let remote = sync
                .remote
                .ok_or_else(|| anyhow::anyhow!("sync.remote is required for service engine"))?;
            service_sync_push(project, &project_root, &remote)
        }
        other => Err(anyhow::anyhow!("sync engine not implemented: {}", other)),
    }
}

fn issues_sync_status(project: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let sync = resolved_sync_config(project)?;
    match sync.engine.as_deref().unwrap_or("git") {
        "git" => run_git(&project_root, &["status", "-sb"]),
        "noop" => Ok(()),
        "service" => {
            let remote = sync
                .remote
                .ok_or_else(|| anyhow::anyhow!("sync.remote is required for service engine"))?;
            service_sync_status(project, &project_root, &remote)
        }
        other => Err(anyhow::anyhow!("sync engine not implemented: {}", other)),
    }
}

fn issues_index_status(project: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let index_path = project_root.join("index").join("issues.json");
    if !index_path.exists() {
        println!("missing index");
        return Ok(());
    }
    let index = read_index(&project_root)?;
    println!("issues: {}", index.issues.len());
    println!("generated_at: {}", index.generated_at);
    Ok(())
}

fn issues_index_verify(project: &str) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let index = read_index(&project_root)?;
    let mut missing = Vec::new();
    for issue in &index.issues {
        let path = project_root.join(&issue.path);
        if !path.exists() {
            missing.push(issue.id.clone());
        }
    }
    if missing.is_empty() {
        println!("ok");
    } else {
        println!("missing: {}", missing.join(", "));
    }
    Ok(())
}

fn issues_projects(json: bool) -> anyhow::Result<()> {
    let root = projects_root()?;
    if !root.exists() {
        return Ok(());
    }
    let global = read_global_config()?;
    let default_project = global.default_project.unwrap_or_default();
    let mut names = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            names.push(name);
        }
    }
    names.sort();

    if json {
        let items: Vec<_> = names
            .iter()
            .map(|name| {
                serde_json::json!({
                    "name": name,
                    "default": name == &default_project,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&items)?);
        return Ok(());
    }

    for name in names {
        if name == default_project {
            println!("{} *", name);
        } else {
            println!("{}", name);
        }
    }
    Ok(())
}

fn issues_project_show(project: &str) -> anyhow::Result<()> {
    let root = project_root(project)?;
    println!("project: {}", project);
    println!("root: {}", root.display());
    let project_toml = root.join("project.toml");
    if project_toml.exists() {
        let content = fs::read_to_string(project_toml)?;
        println!("project.toml:\n{}", content.trim());
    }
    Ok(())
}

fn issues_project_config_set(project: &str, key: &str, value: &str) -> anyhow::Result<()> {
    let root = project_root(project)?;
    let path = root.join("project.toml");
    let mut doc = if path.exists() {
        let content = fs::read_to_string(&path)?;
        toml::from_str::<toml::Value>(&content)?
    } else {
        toml::Value::Table(toml::map::Map::new())
    };
    set_toml_key(&mut doc, key, toml::Value::String(value.to_string()));
    fs::write(path, toml::to_string_pretty(&doc)?)?;
    Ok(())
}

fn issues_project_links_list(project: Option<&str>, json: bool) -> anyhow::Result<()> {
    let global = read_global_config()?;
    let map = global.projects.unwrap_or_default();
    let mut entries: Vec<(String, String)> = map
        .into_iter()
        .filter(|(_, p)| project.map_or(true, |proj| p == proj))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    if json {
        let items: Vec<_> = entries
            .iter()
            .map(|(path, proj)| serde_json::json!({ "path": path, "project": proj }))
            .collect();
        println!("{}", serde_json::to_string_pretty(&items)?);
        return Ok(());
    }

    for (path, proj) in entries {
        println!("{}\t{}", path, proj);
    }
    Ok(())
}

fn issues_project_link(project: &str, path: Option<PathBuf>) -> anyhow::Result<()> {
    let path = path.unwrap_or(std::env::current_dir()?);
    let path = fs::canonicalize(path)?;
    let mut global_value = read_global_config_value()?;
    set_toml_table_entry(
        &mut global_value,
        "projects",
        &path.to_string_lossy(),
        toml::Value::String(project.to_string()),
    );
    write_global_config_value(&global_value)?;
    Ok(())
}

fn skills_install(codex: bool, claude: bool, force: bool) -> anyhow::Result<()> {
    let codex_dir = codex_skills_dir()?;
    let claude_dir = claude_skills_dir()?;

    let install_codex = codex || (!codex && !claude);
    let install_claude = claude || (!codex && !claude);

    if install_codex {
        write_skill_install(&codex_dir, force)?;
    }
    if install_claude {
        write_skill_install(&claude_dir, force)?;
    }
    Ok(())
}

fn codex_skills_dir() -> anyhow::Result<PathBuf> {
    let home = home_dir().ok_or_else(|| anyhow::anyhow!("cannot resolve home dir"))?;
    Ok(home.join(".codex").join("skills"))
}

fn claude_skills_dir() -> anyhow::Result<PathBuf> {
    let home = home_dir().ok_or_else(|| anyhow::anyhow!("cannot resolve home dir"))?;
    Ok(home.join(".claude").join("skills"))
}

fn write_skill_install(dest_dir: &Path, force: bool) -> anyhow::Result<()> {
    let dest = dest_dir.join("scope-cli");
    if dest.exists() {
        if !force {
            return Ok(());
        }
        fs::remove_dir_all(&dest)?;
    }
    fs::create_dir_all(dest_dir)?;
    write_embedded_skill(&dest)?;
    println!("Installed skill to {}", dest.display());
    Ok(())
}

fn write_embedded_skill(dest: &Path) -> anyhow::Result<()> {
    if dest.exists() {
        fs::remove_dir_all(dest)?;
    }
    fs::create_dir_all(dest.join("agents"))?;
    fs::create_dir_all(dest.join("references"))?;

    fs::write(dest.join("SKILL.md"), embedded_skill_md())?;
    fs::write(dest.join("agents").join("openai.yaml"), embedded_openai_yaml())?;
    fs::write(dest.join("references").join("cli.md"), embedded_cli_md())?;
    fs::write(dest.join("references").join("specs.md"), embedded_specs_md())?;
    Ok(())
}


fn embedded_skill_md() -> &'static str {
    r#"---
name: scope-cli
description: Use the scope CLI for developer-first issue tracking in this repo. Trigger when the user asks to create/update/list/show issues, manage comments, conflicts, indexes, project mapping, or scope config files, or when demonstrating or scripting the scope CLI.
---

# Scope CLI

## Overview

Use the `scope` CLI to manage local-first issues stored as markdown snapshots plus JSONL event logs. Prefer CLI commands over manual file edits.

## Quick Start

1. Initialize a project:
   - `scope issues init --project <name>`
2. Create an issue:
   - `scope issues create --project <name> --title "..." [--priority p1] [--assignee ...] [--label ...] [--body "..."] [--body-file ./issue.md]`
3. List and show:
   - `scope issues list --project <name>`
   - `scope issues show --project <name> <id>`

## Core Tasks

### Update and workflow

- Update fields: `scope issues update <id> --status in_progress --add-label planning`
- Close/reopen: `scope issues close <id>` / `scope issues reopen <id>`
- Delete/restore (no prompts): `scope issues delete <id>` / `scope issues restore <id>` (`--force` is permanent)
- Edit body: `scope issues edit <id>`
  - Avoid stdin/heredoc bodies (ex: `/dev/stdin`, `<<EOF`) in agent shells; prefer `--body`, a real file path, or `scope issues edit`

### Comments

- Add: `scope issues comments add <id> --body "..."` or `--body-file ...`
- List: `scope issues comments list <id> [--json]`

### Index and rebuild

- `scope issues index status|rebuild|verify`
- `scope issues rebuild` to regenerate snapshots + index from events

### Projects and mappings

- `scope issues projects [--json]`
- `scope issues project show --project <name>`
- Map multiple working copies:
  - `scope issues project link --project <name> --path /path/to/repo`
  - `scope issues project links --project <name> [--json]`

### Conflicts

- `scope issues conflicts list`
- `scope issues conflicts show <id>`
- `scope issues conflicts resolve <id> --keep local|remote`

## References

Read these when you need the detailed spec or schema:

- `references/cli.md` for command surface and file layout
- `references/specs.md` for links to the full design docs
"#
}

fn embedded_openai_yaml() -> &'static str {
    r#"display_name: Scope CLI
short_description: Developer-first issue tracking via scope CLI
default_prompt: Help with the scope CLI: create/update/list/show issues, comments, conflicts, indexes, projects, and local config. Prefer CLI commands, avoid stdin/heredoc bodies; use --body or a real file path, and reference the spec docs in this repo when needed.
"#
}

fn embedded_cli_md() -> &'static str {
    r#"# Scope CLI Reference (Summary)

## Storage

- Projects root: `~/.scope/projects/<subdir>/`
- Issues: `issues/<id>.md`
- Events: `events/<id>.jsonl`
- Index: `index/issues.json`
- Conflicts: `index/conflicts.json`
- Project config: `project.toml`
- Global config: `~/.scope/config.toml`

## Common Commands

```bash
scope issues init --project <name>
scope issues create --project <name> --title "..." [--body "..."] [--body-file ./issue.md]
scope issues list --project <name> [--json]
scope issues show --project <name> <id>
scope issues update <id> [--status ...] [--priority ...] [--add-label ...]
scope issues close <id>
scope issues reopen <id> [--status ...]
scope issues delete <id> [--force]
scope issues restore <id>

scope issues comments add <id> --body "..."
scope issues comments list <id> [--json]

scope issues index status|rebuild|verify
scope issues rebuild

scope issues projects [--json]
scope issues project show --project <name>
scope issues project config set --project <name> <key> <value>
scope issues project link --project <name> --path /path/to/repo
scope issues project links --project <name> [--json]

scope issues conflicts list
scope issues conflicts show <conflict_id>
scope issues conflicts resolve <conflict_id> --keep local|remote
```

Note: avoid stdin/heredoc bodies (ex: `/dev/stdin`, `<<EOF`) in agent shells; prefer `--body`, a real file path, or `scope issues edit`. `scope issues delete` is a soft delete by default and prints an undo command; use `--force` for permanent removal.
"#
}

fn embedded_specs_md() -> &'static str {
    r#"# Scope Specs (Pointers)

Use these repo docs for authoritative behavior and schema details:

- `docs/cli-foundation.md`
- `docs/config-schema.md`
- `docs/event-schema.md`
- `docs/index-schema.md`
- `docs/conflict-model.md`
"#
}

fn issues_set_status(
    project: &str,
    id: &str,
    status: &str,
    event_type: &str,
) -> anyhow::Result<()> {
    let project_root = project_root(project)?;
    let issue_path = project_root.join("issues").join(format!("{}.md", id));
    let content = fs::read_to_string(&issue_path)?;
    let (mut frontmatter, body) = parse_frontmatter(&content)?;
    frontmatter.status = status.to_string();
    frontmatter.updated_at = now_ts();
    write_issue(&issue_path, &frontmatter, &body)?;

    let events_path = project_root.join("events").join(format!("{}.jsonl", id));
    append_event(
        &events_path,
        event_type,
        id,
        serde_json::json!({ "status": status }),
    )?;
    upsert_index_from_issue(project, &project_root, &frontmatter)?;
    Ok(())
}

fn resolve_project(project: Option<String>) -> anyhow::Result<String> {
    if let Some(project) = project {
        return Ok(project);
    }
    let global = read_global_config()?;
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(project) = resolve_project_from_path(&global, &cwd) {
            return Ok(project);
        }
    }
    if let Some(default_project) = global.default_project {
        return Ok(default_project);
    }
    Err(anyhow::anyhow!("project is required (use --project)"))
}

fn project_root(project: &str) -> anyhow::Result<PathBuf> {
    let projects_root = projects_root()?;
    let project_root = projects_root.join(project);
    if !project_root.exists() {
        return Err(anyhow::anyhow!(
            "project not initialized (run `scope issues init --project {}`)",
            project
        ));
    }
    Ok(project_root)
}

fn projects_root() -> anyhow::Result<PathBuf> {
    let global = read_global_config()?;
    let root = global
        .paths
        .and_then(|p| p.projects_root)
        .unwrap_or_else(|| DEFAULT_PROJECTS_ROOT.to_string());
    Ok(expand_tilde(&root))
}

fn resolved_ids_config(project: &str) -> anyhow::Result<IdsConfig> {
    let global = read_global_config()?;
    let project_config = read_project_config(project)?;
    let mut ids = global.ids.unwrap_or_default();
    if let Some(project_ids) = project_config.and_then(|p| p.ids) {
        if project_ids.prefix.is_some() {
            ids.prefix = project_ids.prefix;
        }
        if project_ids.pattern.is_some() {
            ids.pattern = project_ids.pattern;
        }
        if project_ids.wordlist_adjectives.is_some() {
            ids.wordlist_adjectives = project_ids.wordlist_adjectives;
        }
        if project_ids.wordlist_animals.is_some() {
            ids.wordlist_animals = project_ids.wordlist_animals;
        }
        if project_ids.max_attempts.is_some() {
            ids.max_attempts = project_ids.max_attempts;
        }
    }
    Ok(ids)
}

fn generate_issue_id(project_root: &Path, ids: &IdsConfig) -> anyhow::Result<String> {
    let prefix = ids
        .prefix
        .clone()
        .unwrap_or_else(|| DEFAULT_ID_PREFIX.to_string());
    let max_attempts = ids.max_attempts.unwrap_or(DEFAULT_MAX_ATTEMPTS);

    let adjectives = load_wordlist(ids.wordlist_adjectives.as_deref(), ADJECTIVES)?;
    let animals = load_wordlist(ids.wordlist_animals.as_deref(), ANIMALS)?;

    let mut rng = rand::thread_rng();
    let mut tried = HashSet::new();

    for _ in 0..max_attempts {
        let adj1 = adjectives.choose(&mut rng).ok_or_else(|| {
            anyhow::anyhow!("adjective wordlist is empty")
        })?;
        let adj2 = adjectives.choose(&mut rng).ok_or_else(|| {
            anyhow::anyhow!("adjective wordlist is empty")
        })?;
        let animal = animals.choose(&mut rng).ok_or_else(|| {
            anyhow::anyhow!("animal wordlist is empty")
        })?;

        let slug = format!("{}-{}-{}", adj1, adj2, animal);
        if tried.contains(&slug) {
            continue;
        }
        tried.insert(slug.clone());

        let id = format!("{}:{}", prefix, slug);
        let issue_path = project_root.join("issues").join(format!("{}.md", id));
        if !issue_path.exists() {
            return Ok(id);
        }
    }

    Err(anyhow::anyhow!("failed to generate unique issue id"))
}

fn load_wordlist(path: Option<&str>, fallback: &[&str]) -> anyhow::Result<Vec<String>> {
    if let Some(path) = path {
        let path = expand_tilde(path);
        if path.exists() {
            let content = fs::read_to_string(path)?;
            let list: Vec<String> = content
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            if !list.is_empty() {
                return Ok(list);
            }
        }
    }
    Ok(fallback.iter().map(|s| s.to_string()).collect())
}

fn write_issue(path: &Path, frontmatter: &IssueFrontmatter, body: &str) -> anyhow::Result<()> {
    let yaml = serde_yaml::to_string(frontmatter)?;
    let mut content = String::new();
    content.push_str("---\n");
    content.push_str(&yaml);
    content.push_str("---\n\n");
    content.push_str(body);
    fs::write(path, content)?;
    Ok(())
}

fn parse_frontmatter(content: &str) -> anyhow::Result<(IssueFrontmatter, String)> {
    let mut lines = content.lines();
    let first = lines.next().ok_or_else(|| anyhow::anyhow!("empty issue file"))?;
    if first.trim() != "---" {
        return Err(anyhow::anyhow!("missing frontmatter"));
    }

    let mut yaml_lines = Vec::new();
    for line in &mut lines {
        if line.trim() == "---" {
            break;
        }
        yaml_lines.push(line);
    }

    let yaml = yaml_lines.join("\n");
    let frontmatter: IssueFrontmatter = serde_yaml::from_str(&yaml)?;
    let body = lines.collect::<Vec<_>>().join("\n");
    Ok((frontmatter, body))
}

fn append_event(
    events_path: &Path,
    event_type: &str,
    issue_id: &str,
    data: serde_json::Value,
) -> anyhow::Result<()> {
    let event = Event {
        id: format!("evt_{}", Ulid::new()),
        event_type: event_type.to_string(),
        issue: issue_id.to_string(),
        author: current_user(),
        ts: now_ts(),
        data,
    };
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(events_path)?;
    let line = serde_json::to_string(&event)?;
    writeln!(file, "{}", line)?;
    Ok(())
}

fn apply_event(state: &mut IssueState, event: &EventRecord) {
    if state.id.is_empty() {
        state.id = event.issue.clone();
    }

    match event.event_type.as_str() {
        "issue.create" => {
            apply_field_updates(state, &event.data);
            if state.created_at.is_empty() {
                state.created_at = event.ts.clone();
            }
        }
        "issue.update" => {
            apply_field_updates(state, &event.data);
        }
        "issue.close" | "issue.reopen" => {
            if let Some(status) = event.data.get("status").and_then(|v| v.as_str()) {
                state.status = status.to_string();
            }
        }
        "issue.comment" => {
            if let Some(body) = event.data.get("body").and_then(|v| v.as_str()) {
                state.comments.push(IssueComment {
                    author: event.author.clone(),
                    ts: event.ts.clone(),
                    body: body.to_string(),
                });
            }
        }
        _ => {}
    }

    state.updated_at = event.ts.clone();

    if state.status.is_empty() {
        state.status = "todo".to_string();
    }
    if state.priority.is_empty() {
        state.priority = "p2".to_string();
    }
}

fn apply_field_updates(state: &mut IssueState, data: &serde_json::Value) {
    if let Some(title) = data.get("title").and_then(|v| v.as_str()) {
        state.title = title.to_string();
    }
    if let Some(status) = data.get("status").and_then(|v| v.as_str()) {
        state.status = status.to_string();
    }
    if let Some(priority) = data.get("priority").and_then(|v| v.as_str()) {
        state.priority = priority.to_string();
    }
    if let Some(assignee) = data.get("assignee").and_then(|v| v.as_str()) {
        state.assignee = Some(assignee.to_string());
    }
    if data.get("assignee").map_or(false, |v| v.is_null()) {
        state.assignee = None;
    }
    if let Some(labels) = data.get("labels").and_then(|v| v.as_array()) {
        state.labels = labels
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
    }
    if data.get("labels").map_or(false, |v| v.is_null()) {
        state.labels = Vec::new();
    }
    if let Some(body) = data.get("body").and_then(|v| v.as_str()) {
        state.body = body.to_string();
    }
}

fn resolved_conflict_window_seconds(project: &str) -> anyhow::Result<i64> {
    let global = read_global_config()?;
    let project_config = read_project_config(project)?;
    if let Some(project_events) = project_config.and_then(|p| p.events) {
        if let Some(seconds) = project_events.conflict_window_seconds {
            return Ok(seconds);
        }
    }
    if let Some(global_events) = global.events {
        if let Some(seconds) = global_events.conflict_window_seconds {
            return Ok(seconds);
        }
    }
    Ok(300)
}

fn resolved_sync_config(project: &str) -> anyhow::Result<SyncConfig> {
    let global = read_global_config()?;
    let project_config = read_project_config(project)?;
    let mut sync = global.sync.unwrap_or_default();
    if let Some(project_sync) = project_config.and_then(|p| p.sync) {
        if project_sync.engine.is_some() {
            sync.engine = project_sync.engine;
        }
        if project_sync.remote.is_some() {
            sync.remote = project_sync.remote;
        }
    }
    Ok(sync)
}

fn detect_conflicts_from_events(
    events_path: &Path,
    conflict_window_seconds: i64,
) -> anyhow::Result<Vec<ConflictRecord>> {
    let content = fs::read_to_string(events_path)?;
    let mut conflicts = Vec::new();
    let mut last_by_field: HashMap<String, (String, String, serde_json::Value, i64)> = HashMap::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let event: EventRecord = serde_json::from_str(line)?;
        let ts = parse_ts_secs(&event.ts)?;
        let mut fields = Vec::new();

        match event.event_type.as_str() {
            "issue.create" | "issue.update" => {
                if let Some(obj) = event.data.as_object() {
                    for (k, v) in obj {
                        fields.push((k.clone(), v.clone()));
                    }
                }
            }
            "issue.close" | "issue.reopen" => {
                if let Some(status) = event.data.get("status") {
                    fields.push(("status".to_string(), status.clone()));
                }
            }
            _ => {}
        }

        for (field, value) in fields {
            if let Some((prev_event, prev_author, prev_value, prev_ts)) =
                last_by_field.get(&field)
            {
                let delta = (ts - *prev_ts).abs();
                if delta <= conflict_window_seconds && prev_author != &event.author {
                    conflicts.push(ConflictRecord {
                        id: format!("conf_{}", Ulid::new()),
                        issue: event.issue.clone(),
                        field: field.clone(),
                        local_event: prev_event.clone(),
                        remote_event: event.id.clone(),
                        local_value: prev_value.clone(),
                        remote_value: value.clone(),
                        detected_at: now_ts(),
                        state: "unresolved".to_string(),
                    });
                }
            }
            last_by_field.insert(
                field,
                (event.id.clone(), event.author.clone(), value, ts),
            );
        }
    }

    Ok(conflicts)
}

fn parse_ts_secs(ts: &str) -> anyhow::Result<i64> {
    let dt = chrono::DateTime::parse_from_rfc3339(ts)?;
    Ok(dt.timestamp())
}

fn read_conflicts_or_empty(project_root: &Path) -> anyhow::Result<ConflictFile> {
    let path = project_root.join("index").join("conflicts.json");
    if !path.exists() {
        return Ok(ConflictFile {
            format_version: 1,
            generated_at: now_ts(),
            conflicts: Vec::new(),
        });
    }
    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

fn write_conflicts(project_root: &Path, conflicts: &ConflictFile) -> anyhow::Result<()> {
    let path = project_root.join("index").join("conflicts.json");
    fs::write(path, serde_json::to_string_pretty(conflicts)?)?;
    Ok(())
}

fn merge_conflicts(conflicts: &mut ConflictFile, new_conflicts: Vec<ConflictRecord>) {
    let mut existing_keys = HashSet::new();
    for c in &conflicts.conflicts {
        existing_keys.insert(conflict_key(c));
    }
    for c in new_conflicts {
        let key = conflict_key(&c);
        if !existing_keys.contains(&key) {
            conflicts.conflicts.push(c);
            existing_keys.insert(key);
        }
    }
    conflicts.generated_at = now_ts();
}

fn conflict_key(conflict: &ConflictRecord) -> String {
    format!(
        "{}|{}|{}|{}",
        conflict.issue, conflict.field, conflict.local_event, conflict.remote_event
    )
}

fn apply_conflict_resolution(
    project: &str,
    project_root: &Path,
    conflict: &ConflictRecord,
    chosen: serde_json::Value,
    resolution: String,
) -> anyhow::Result<()> {
    let issue_path = project_root
        .join("issues")
        .join(format!("{}.md", conflict.issue));
    let content = fs::read_to_string(&issue_path)?;
    let (mut frontmatter, mut body) = parse_frontmatter(&content)?;

    match conflict.field.as_str() {
        "title" => {
            if let Some(v) = chosen.as_str() {
                frontmatter.title = v.to_string();
            }
        }
        "status" => {
            if let Some(v) = chosen.as_str() {
                frontmatter.status = v.to_string();
            }
        }
        "priority" => {
            if let Some(v) = chosen.as_str() {
                frontmatter.priority = v.to_string();
            }
        }
        "assignee" => {
            if chosen.is_null() {
                frontmatter.assignee = None;
            } else if let Some(v) = chosen.as_str() {
                frontmatter.assignee = Some(v.to_string());
            }
        }
        "labels" => {
            if let Some(arr) = chosen.as_array() {
                frontmatter.labels = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
            }
        }
        "body" => {
            if let Some(v) = chosen.as_str() {
                body = v.to_string();
            }
        }
        _ => {}
    }

    frontmatter.updated_at = now_ts();
    write_issue(&issue_path, &frontmatter, &body)?;

    let events_path = project_root
        .join("events")
        .join(format!("{}.jsonl", conflict.issue));
    append_event(
        &events_path,
        "issue.update",
        &conflict.issue,
        serde_json::json!({ conflict.field.clone(): chosen }),
    )?;
    append_event(
        &events_path,
        "issue.resolve_conflict",
        &conflict.issue,
        serde_json::json!({
            "conflict_id": conflict.id,
            "resolution": resolution,
        }),
    )?;

    upsert_index_from_issue(project, project_root, &frontmatter)?;
    Ok(())
}

fn run_git(project_root: &Path, args: &[&str]) -> anyhow::Result<()> {
    let status = std::process::Command::new("git")
        .arg("-C")
        .arg(project_root)
        .args(args)
        .status()?;
    if !status.success() {
        return Err(anyhow::anyhow!("git command failed"));
    }
    Ok(())
}

fn issue_state_to_snapshot(state: &IssueState) -> (IssueFrontmatter, String) {
    let frontmatter = IssueFrontmatter {
        id: state.id.clone(),
        title: state.title.clone(),
        status: state.status.clone(),
        priority: state.priority.clone(),
        assignee: state.assignee.clone(),
        labels: state.labels.clone(),
        created_at: state.created_at.clone(),
        updated_at: state.updated_at.clone(),
    };

    let mut body = if state.body.is_empty() {
        "## Summary\n\n## Acceptance Criteria\n".to_string()
    } else {
        state.body.clone()
    };

    if !state.comments.is_empty() {
        body.push_str("\n\n## Comments\n");
        for comment in &state.comments {
            body.push_str(&format!(
                "- {} @{}: {}\n",
                comment.ts, comment.author, comment.body
            ));
        }
    }

    (frontmatter, body)
}

fn rebuild_issue_from_events(events_path: &Path) -> anyhow::Result<Option<IssueState>> {
    let content = fs::read_to_string(events_path)?;
    let mut state = IssueState::default();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let event: EventRecord = serde_json::from_str(line)?;
        apply_event(&mut state, &event);
    }
    if state.id.is_empty() || state.title.is_empty() {
        return Ok(None);
    }
    Ok(Some(state))
}

fn current_user() -> String {
    std::env::var("USER").unwrap_or_else(|_| "unknown".to_string())
}

fn read_index(project_root: &Path) -> anyhow::Result<IndexFile> {
    let index_path = project_root.join("index").join("issues.json");
    let content = fs::read_to_string(index_path)?;
    Ok(serde_json::from_str(&content)?)
}

fn read_index_or_empty(project: &str, project_root: &Path) -> anyhow::Result<IndexFile> {
    let index_path = project_root.join("index").join("issues.json");
    if index_path.exists() {
        return read_index(project_root);
    }
    Ok(IndexFile {
        format_version: 1,
        generated_at: now_ts(),
        project: project.to_string(),
        issues: Vec::new(),
    })
}

fn upsert_index_from_issue(
    project: &str,
    project_root: &Path,
    frontmatter: &IssueFrontmatter,
) -> anyhow::Result<()> {
    let mut index = read_index_or_empty(project, project_root)?;
    index.issues.retain(|i| i.id != frontmatter.id);
    index.issues.push(IndexIssue {
        id: frontmatter.id.clone(),
        title: frontmatter.title.clone(),
        status: frontmatter.status.clone(),
        priority: frontmatter.priority.clone(),
        assignee: frontmatter.assignee.clone(),
        labels: frontmatter.labels.clone(),
        created_at: frontmatter.created_at.clone(),
        updated_at: frontmatter.updated_at.clone(),
        path: format!("issues/{}.md", frontmatter.id),
        events_path: format!("events/{}.jsonl", frontmatter.id),
    });
    index.generated_at = now_ts();
    write_index(&project_root.join("index").join("issues.json"), &index)?;
    Ok(())
}

fn write_index(path: &Path, index: &IndexFile) -> anyhow::Result<()> {
    let content = serde_json::to_string_pretty(index)?;
    fs::write(path, content)?;
    Ok(())
}

fn now_ts() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from("/"));
    }
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(stripped);
        }
    }
    PathBuf::from(path)
}

fn global_config_path() -> anyhow::Result<PathBuf> {
    let home = home_dir().ok_or_else(|| anyhow::anyhow!("cannot resolve home dir"))?;
    Ok(home.join(".scope").join("config.toml"))
}

fn read_global_config() -> anyhow::Result<GlobalConfig> {
    let path = global_config_path()?;
    if !path.exists() {
        return Ok(GlobalConfig::default());
    }
    let content = fs::read_to_string(path)?;
    Ok(toml::from_str(&content)?)
}

fn read_global_config_value() -> anyhow::Result<toml::Value> {
    let path = global_config_path()?;
    if !path.exists() {
        return Ok(toml::Value::Table(toml::map::Map::new()));
    }
    let content = fs::read_to_string(path)?;
    Ok(toml::from_str(&content)?)
}

fn write_global_config_value(value: &toml::Value) -> anyhow::Result<()> {
    let path = global_config_path()?;
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, toml::to_string_pretty(value)?)?;
    Ok(())
}

fn resolve_project_from_path(global: &GlobalConfig, cwd: &Path) -> Option<String> {
    let map = global.projects.as_ref()?;
    let mut current = cwd.to_path_buf();
    loop {
        let key = current.to_string_lossy().to_string();
        if let Some(project) = map.get(&key) {
            return Some(project.clone());
        }
        if !current.pop() {
            break;
        }
    }
    None
}

fn set_toml_key(root: &mut toml::Value, key: &str, value: toml::Value) {
    let parts: Vec<&str> = key.split('.').collect();
    let mut current = root;
    for (i, part) in parts.iter().enumerate() {
        let is_last = i == parts.len() - 1;
        if is_last {
            if let toml::Value::Table(table) = current {
                table.insert(part.to_string(), value);
            }
            return;
        }

        if let toml::Value::Table(table) = current {
            if !table.contains_key(*part) {
                table.insert(part.to_string(), toml::Value::Table(toml::map::Map::new()));
            }
            current = table.get_mut(*part).unwrap();
        }
    }
}

fn set_toml_table_entry(
    root: &mut toml::Value,
    table_name: &str,
    key: &str,
    value: toml::Value,
) {
    if let toml::Value::Table(table) = root {
        let entry = table
            .entry(table_name.to_string())
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
        if let toml::Value::Table(inner) = entry {
            inner.insert(key.to_string(), value);
        }
    }
}

fn resolved_editor() -> Option<String> {
    if let Ok(editor) = std::env::var("EDITOR") {
        if !editor.trim().is_empty() {
            return Some(editor);
        }
    }
    if let Ok(global) = read_global_config() {
        if let Some(editor) = global.editor {
            return Some(editor);
        }
    }
    None
}

fn read_project_config(project: &str) -> anyhow::Result<Option<ProjectConfig>> {
    let projects_root = projects_root()?;
    let path = projects_root.join(project).join("project.toml");
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path)?;
    Ok(Some(toml::from_str(&content)?))
}

// ── Auth & key management ──────────────────────────────────────────────

fn keys_dir() -> anyhow::Result<PathBuf> {
    let home = home_dir().ok_or_else(|| anyhow::anyhow!("cannot resolve home dir"))?;
    Ok(home.join(".scope").join("keys"))
}

fn auth_keys_generate(force: bool) -> anyhow::Result<()> {
    let dir = keys_dir()?;
    let priv_path = dir.join("private.key");
    let pub_path = dir.join("public.key");

    if priv_path.exists() && !force {
        return Err(anyhow::anyhow!(
            "keys already exist at {}. Use --force to overwrite.",
            dir.display()
        ));
    }

    fs::create_dir_all(&dir)?;

    let mut csprng = rand::rngs::OsRng;
    let signing_key = SigningKey::generate(&mut csprng);
    let verifying_key = signing_key.verifying_key();

    let priv_b64 = BASE64.encode(signing_key.to_bytes());
    let pub_b64 = BASE64.encode(verifying_key.to_bytes());

    fs::write(&priv_path, &priv_b64)?;
    fs::write(&pub_path, &pub_b64)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&priv_path, fs::Permissions::from_mode(0o600))?;
    }

    println!("Generated Ed25519 keypair");
    println!("  Private key: {}", priv_path.display());
    println!("  Public key:  {}", pub_path.display());
    println!("\nYour public key:");
    println!("  {}", pub_b64);
    Ok(())
}

fn auth_keys_show() -> anyhow::Result<()> {
    let pub_b64 = load_public_key_b64()?;
    println!("{}", pub_b64);
    Ok(())
}

fn auth_whoami() -> anyhow::Result<()> {
    let pub_b64 = load_public_key_b64()?;
    let pub_bytes = BASE64.decode(&pub_b64)?;
    let account_id = derive_account_id(&pub_bytes);
    println!("Account ID: {}", account_id);
    println!("Public key: {}", pub_b64);
    Ok(())
}

fn auth_signup(remote: &str) -> anyhow::Result<()> {
    let pub_b64 = load_public_key_b64()?;
    let signing_key = load_signing_key()?;
    let pub_bytes = BASE64.decode(&pub_b64)?;
    let account_id = derive_account_id(&pub_bytes);

    let body = serde_json::json!({
        "public_key": pub_b64,
        "display_name": current_user(),
    });
    let body_str = serde_json::to_string(&body)?;

    let url = format!("{}/auth/signup", remote.trim_end_matches('/'));
    let (timestamp, pubkey_header, signature) = sign_request("POST", "/auth/signup", &body_str, &signing_key)?;

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("X-Scope-PublicKey", &pubkey_header)
        .header("X-Scope-Timestamp", &timestamp)
        .header("X-Scope-Signature", &signature)
        .body(body_str)
        .send()?;

    let status = resp.status();
    let resp_body: serde_json::Value = resp.json()?;

    if status.is_success() {
        println!("Signed up successfully");
        println!("Account ID: {}", account_id);
    } else {
        let msg = resp_body
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        return Err(anyhow::anyhow!("signup failed ({}): {}", status, msg));
    }

    Ok(())
}

fn load_public_key_b64() -> anyhow::Result<String> {
    let dir = keys_dir()?;
    let pub_path = dir.join("public.key");
    if !pub_path.exists() {
        return Err(anyhow::anyhow!(
            "no keys found. Run `scope auth keys generate` first."
        ));
    }
    let content = fs::read_to_string(pub_path)?;
    Ok(content.trim().to_string())
}

fn load_signing_key() -> anyhow::Result<SigningKey> {
    let dir = keys_dir()?;
    let priv_path = dir.join("private.key");
    if !priv_path.exists() {
        return Err(anyhow::anyhow!(
            "no private key found. Run `scope auth keys generate` first."
        ));
    }
    let content = fs::read_to_string(priv_path)?;
    let bytes = BASE64.decode(content.trim())?;
    let key_bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid private key length"))?;
    Ok(SigningKey::from_bytes(&key_bytes))
}

fn derive_account_id(public_key_bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(public_key_bytes);
    let hash = hasher.finalize();
    let hex = hex_encode(&hash);
    format!("acct_{}", &hex[..16])
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn sign_request(
    method: &str,
    path: &str,
    body: &str,
    signing_key: &SigningKey,
) -> anyhow::Result<(String, String, String)> {
    let timestamp = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
    let body_hash = {
        let mut hasher = Sha256::new();
        hasher.update(body.as_bytes());
        hex_encode(&hasher.finalize())
    };
    let payload = format!("{}:{}:{}:{}", method, path, timestamp, body_hash);
    let signature = signing_key.sign(payload.as_bytes());
    let pubkey_b64 = BASE64.encode(signing_key.verifying_key().to_bytes());
    let sig_b64 = BASE64.encode(signature.to_bytes());
    Ok((timestamp, pubkey_b64, sig_b64))
}

// ── Sync state ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Default)]
struct SyncState {
    last_pushed_event: Option<String>,
    last_pulled_batch: Option<String>,
    last_sync_at: Option<String>,
}

fn sync_state_path(project: &str) -> anyhow::Result<PathBuf> {
    let root = project_root(project)?;
    Ok(root.join("sync.json"))
}

fn read_sync_state(project: &str) -> anyhow::Result<SyncState> {
    let path = sync_state_path(project)?;
    if !path.exists() {
        return Ok(SyncState::default());
    }
    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

fn write_sync_state(project: &str, state: &SyncState) -> anyhow::Result<()> {
    let path = sync_state_path(project)?;
    fs::write(path, serde_json::to_string_pretty(state)?)?;
    Ok(())
}

// ── Collect events since cursor ────────────────────────────────────────

fn collect_events_since(
    project_root: &Path,
    cursor: Option<&str>,
) -> anyhow::Result<Vec<serde_json::Value>> {
    let events_dir = project_root.join("events");
    if !events_dir.exists() {
        return Ok(Vec::new());
    }
    let mut all_events = Vec::new();
    for entry in fs::read_dir(&events_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let content = fs::read_to_string(&path)?;
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let event: serde_json::Value = serde_json::from_str(line)?;
            if let Some(cursor) = cursor {
                if let Some(id) = event.get("id").and_then(|v| v.as_str()) {
                    if id <= cursor {
                        continue;
                    }
                }
            }
            all_events.push(event);
        }
    }
    all_events.sort_by(|a, b| {
        let a_id = a.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let b_id = b.get("id").and_then(|v| v.as_str()).unwrap_or("");
        a_id.cmp(b_id)
    });
    Ok(all_events)
}

// ── Signed API request helper ──────────────────────────────────────────

fn signed_api_request(
    client: &reqwest::blocking::Client,
    method: &str,
    remote: &str,
    path: &str,
    body_str: &str,
    signing_key: &SigningKey,
) -> anyhow::Result<serde_json::Value> {
    let url = format!("{}{}", remote.trim_end_matches('/'), path);
    let (timestamp, pubkey, signature) = sign_request(method, path, body_str, signing_key)?;

    let resp = match method {
        "GET" => client
            .get(&url)
            .header("X-Scope-PublicKey", &pubkey)
            .header("X-Scope-Timestamp", &timestamp)
            .header("X-Scope-Signature", &signature)
            .send()?,
        _ => client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-Scope-PublicKey", &pubkey)
            .header("X-Scope-Timestamp", &timestamp)
            .header("X-Scope-Signature", &signature)
            .body(body_str.to_string())
            .send()?,
    };

    let status = resp.status();
    let resp_body: serde_json::Value = resp.json()?;

    if !status.is_success() {
        let msg = resp_body
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        return Err(anyhow::anyhow!("request failed ({}): {}", status, msg));
    }

    Ok(resp_body)
}

// ── Service sync engine ────────────────────────────────────────────────

fn service_sync_push(project: &str, project_root: &Path, remote: &str) -> anyhow::Result<()> {
    let signing_key = load_signing_key()?;
    let sync_state = read_sync_state(project)?;

    let events = collect_events_since(project_root, sync_state.last_pushed_event.as_deref())?;
    if events.is_empty() {
        println!("Nothing to push (no new events).");
        return Ok(());
    }

    let client = reqwest::blocking::Client::new();

    // 1. Get presigned upload URL from server
    let body = serde_json::json!({ "project": project });
    let resp = signed_api_request(
        &client, "POST", remote, "/sync/push",
        &serde_json::to_string(&body)?, &signing_key,
    )?;

    let upload_url = resp
        .get("upload_url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("server did not return upload_url"))?;
    let batch_id = resp
        .get("batch_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // 2. Build batch JSONL content
    let batch_content: String = events
        .iter()
        .map(|e| serde_json::to_string(e).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    // 3. Upload batch directly to storage
    let upload_resp = client
        .put(upload_url)
        .header("Content-Type", "application/x-ndjson")
        .body(batch_content)
        .send()?;

    if !upload_resp.status().is_success() {
        return Err(anyhow::anyhow!("batch upload failed ({})", upload_resp.status()));
    }

    // 4. Track the last event we pushed
    let last_event_id = events
        .last()
        .and_then(|e| e.get("id").and_then(|v| v.as_str()))
        .map(|s| s.to_string());

    let mut state = read_sync_state(project)?;
    state.last_pushed_event = last_event_id;
    state.last_sync_at = Some(now_ts());
    write_sync_state(project, &state)?;

    println!("Pushed {} events in batch {}.", events.len(), batch_id);
    Ok(())
}

fn service_sync_pull(project: &str, project_root: &Path, remote: &str) -> anyhow::Result<()> {
    let signing_key = load_signing_key()?;
    let sync_state = read_sync_state(project)?;

    let client = reqwest::blocking::Client::new();

    // 1. Get batch download URLs from server
    let body = serde_json::json!({
        "project": project,
        "since_batch": sync_state.last_pulled_batch,
    });
    let resp = signed_api_request(
        &client, "POST", remote, "/sync/pull",
        &serde_json::to_string(&body)?, &signing_key,
    )?;

    let batches = resp
        .get("batches")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    if batches.is_empty() {
        println!("Already up to date.");
        return Ok(());
    }

    // 2. Download each batch and apply events
    let mut total_events = 0usize;
    let mut latest_batch: Option<String> = sync_state.last_pulled_batch.clone();

    for batch in &batches {
        let batch_id = batch
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("batch missing id"))?;
        let download_url = batch
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("batch missing url"))?;

        let batch_resp = client.get(download_url).send()?;
        if !batch_resp.status().is_success() {
            return Err(anyhow::anyhow!("failed to download batch {} ({})", batch_id, batch_resp.status()));
        }
        let batch_content = batch_resp.text()?;

        // Parse events and append to local JSONL files, deduplicating
        for line in batch_content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let event: serde_json::Value = serde_json::from_str(line)?;
            let issue_id = event
                .get("issue")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("event missing issue field"))?;

            let events_path = project_root
                .join("events")
                .join(format!("{}.jsonl", issue_id));
            fs::create_dir_all(events_path.parent().unwrap())?;

            let existing_ids = if events_path.exists() {
                let content = fs::read_to_string(&events_path)?;
                content
                    .lines()
                    .filter_map(|l| {
                        serde_json::from_str::<serde_json::Value>(l)
                            .ok()
                            .and_then(|v| v.get("id").and_then(|id| id.as_str().map(|s| s.to_string())))
                    })
                    .collect::<HashSet<_>>()
            } else {
                HashSet::new()
            };

            let event_id = event.get("id").and_then(|v| v.as_str()).unwrap_or("");
            if !existing_ids.contains(event_id) {
                let mut file = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&events_path)?;
                writeln!(file, "{}", serde_json::to_string(&event)?)?;
                total_events += 1;
            }
        }

        match &latest_batch {
            Some(b) if batch_id > b.as_str() => latest_batch = Some(batch_id.to_string()),
            None => latest_batch = Some(batch_id.to_string()),
            _ => {}
        }
    }

    if total_events > 0 {
        issues_rebuild(project)?;
    }

    let mut state = read_sync_state(project)?;
    state.last_pulled_batch = latest_batch;
    state.last_sync_at = Some(now_ts());
    write_sync_state(project, &state)?;

    println!("Pulled {} batches ({} new events).", batches.len(), total_events);
    Ok(())
}

fn service_sync_status(project: &str, _project_root: &Path, remote: &str) -> anyhow::Result<()> {
    let signing_key = load_signing_key()?;
    let sync_state = read_sync_state(project)?;

    let client = reqwest::blocking::Client::new();
    let path_str = format!("/sync/status?project={}", project);
    let resp = signed_api_request(&client, "GET", remote, &path_str, "", &signing_key)?;

    println!("Remote sync status for project '{}':", project);
    if let Some(total) = resp.get("total_batches").and_then(|v| v.as_u64()) {
        println!("  Remote batches: {}", total);
    }
    if let Some(latest) = resp.get("latest_batch").and_then(|v| v.as_str()) {
        println!("  Latest batch:   {}", latest);
    }

    println!("\nLocal sync state:");
    println!(
        "  Last pushed event: {}",
        sync_state.last_pushed_event.as_deref().unwrap_or("(none)")
    );
    println!(
        "  Last pulled batch: {}",
        sync_state.last_pulled_batch.as_deref().unwrap_or("(none)")
    );
    println!(
        "  Last sync at:      {}",
        sync_state.last_sync_at.as_deref().unwrap_or("(never)")
    );

    Ok(())
}
