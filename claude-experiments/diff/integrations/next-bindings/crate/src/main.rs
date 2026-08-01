//! Process boundary used by the experimental Next custom-binding adapter.
//!
//! Next loads a JavaScript module as its native Turbopack binding. That module
//! keeps Next's actual SWC addon for transforms and invokes this executable for
//! Diffpack-owned project operations. A line-delimited JSON protocol keeps the
//! boundary inspectable while its shape is still evolving.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use notify::{RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};

const PROTOCOL_VERSION: u32 = 1;

fn forward_stdout(
    child: &mut Child,
) -> Result<std::thread::JoinHandle<Result<(), String>>, String> {
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Diffpack child stdout was not piped".to_string())?;
    Ok(std::thread::spawn(move || {
        std::io::copy(&mut stdout, &mut std::io::stderr())
            .map(|_| ())
            .map_err(|error| format!("cannot forward Diffpack stdout: {error}"))
    }))
}

#[derive(Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case")]
enum Request {
    BuildProduction {
        protocol_version: u32,
        project_root: PathBuf,
        output_dir: PathBuf,
        next_config_output: Option<String>,
        #[serde(default)]
        development: bool,
    },
    WatchDevelopment {
        protocol_version: u32,
        project_root: PathBuf,
        output_dir: PathBuf,
        next_config_output: Option<String>,
        poll_interval_ms: Option<u64>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct Response {
    protocol_version: u32,
    ok: bool,
    artifact_root: Option<PathBuf>,
    error: Option<String>,
    routes: Vec<RouteResponse>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RouteResponse {
    pathname: String,
    original_name: String,
    kind: String,
}

fn main() {
    let result = run();
    let response = match result {
        Ok(Some(response)) => response,
        Ok(None) => return,
        Err(error) => Response {
            protocol_version: PROTOCOL_VERSION,
            ok: false,
            artifact_root: None,
            error: Some(error),
            routes: Vec::new(),
        },
    };
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &response).expect("serialize bridge response");
    writeln!(stdout).expect("terminate bridge response");
    if !response.ok {
        std::process::exit(1);
    }
}

fn run() -> Result<Option<Response>, String> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("cannot read bridge request: {error}"))?;
    let request: Request =
        serde_json::from_str(&input).map_err(|error| format!("invalid bridge request: {error}"))?;
    match request {
        Request::WatchDevelopment {
            protocol_version,
            project_root,
            output_dir,
            next_config_output,
            poll_interval_ms,
        } => {
            check_protocol(protocol_version)?;
            watch_development(
                &project_root,
                &output_dir,
                next_config_output.as_deref(),
                poll_interval_ms,
            )?;
            Ok(None)
        }
        Request::BuildProduction {
            protocol_version,
            project_root,
            output_dir,
            next_config_output,
            development,
        } => {
            check_protocol(protocol_version)?;
            let binary = std::env::var_os("DIFFPACK_BINARY")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("diffpack"));
            // Native Next consumes Diffpack's browser assets and the SSR graph's
            // reference ids, then owns every server route artifact under `.next`.
            // Do not render Diffpack's independent standalone RSC/SSR server and
            // prerender output: none of it is reachable from `next start`.
            // The client publishes its reference manifest atomically before it emits
            // browser chunks. Start SSR reference discovery at that exact dependency
            // boundary so the two independent tails overlap.
            let client_manifest = project_root
                .join(".diffpack-output")
                .join(diffpack_next::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
            let _ = std::fs::remove_file(&client_manifest);
            let mut client_command = Command::new(&binary);
            client_command
                .arg("build-app")
                .arg(&project_root)
                .arg("client");
            if development {
                client_command.arg("--development");
            }
            let mut client = client_command
                .env("DIFFPACK_NATIVE_NEXT_OUTPUT", &output_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|error| format!("cannot run {}: {error}", binary.display()))?;
            let client_log = forward_stdout(&mut client)?;
            loop {
                if client_manifest.is_file() {
                    break;
                }
                match client.try_wait() {
                    Ok(Some(status)) if !status.success() => {
                        return Err(format!(
                            "Diffpack native Next client preparation failed ({status})"
                        ));
                    }
                    Ok(Some(_)) => break,
                    Ok(None) => std::thread::sleep(std::time::Duration::from_millis(5)),
                    Err(error) => return Err(format!("cannot wait for Diffpack client: {error}")),
                }
            }
            let mut ssr_command = Command::new(&binary);
            ssr_command
                .arg("build-app")
                .arg(&project_root)
                .arg("ssr")
                .arg("--references-only");
            if development {
                ssr_command.arg("--development");
            }
            let mut ssr = ssr_command
                .env("DIFFPACK_NATIVE_NEXT_OUTPUT", &output_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|error| format!("cannot run {}: {error}", binary.display()))?;
            let ssr_log = forward_stdout(&mut ssr)?;
            let client_status = client
                .wait()
                .map_err(|error| format!("cannot wait for Diffpack client: {error}"))?;
            let ssr_status = ssr
                .wait()
                .map_err(|error| format!("cannot wait for Diffpack SSR references: {error}"))?;
            client_log
                .join()
                .map_err(|_| "Diffpack client log thread panicked".to_string())??;
            ssr_log
                .join()
                .map_err(|_| "Diffpack SSR log thread panicked".to_string())??;
            if !client_status.success() {
                return Err(format!(
                    "Diffpack native Next client preparation failed ({client_status})"
                ));
            }
            if !ssr_status.success() {
                return Err(format!(
                    "Diffpack native Next SSR reference preparation failed ({ssr_status})"
                ));
            }
            let routes = diffpack_next::artifacts::discover_app_routes(&project_root)?;
            let standalone_root = project_root.join(".diffpack-output");
            if development {
                diffpack_next::native::compile_app_entries_development(
                    &project_root,
                    &output_dir,
                    next_config_output.as_deref(),
                )?;
            } else {
                diffpack_next::native::compile_app_entries(
                    &project_root,
                    &output_dir,
                    next_config_output.as_deref(),
                )?;
            }
            diffpack_next::artifacts::NativeNextOutput {
                dist_dir: &output_dir,
                standalone_root: &standalone_root,
            }
            .write_route_manifests(&routes)?;
            let routes = routes
                .into_iter()
                .map(|route| RouteResponse {
                    pathname: route.pathname,
                    original_name: route.original_name,
                    kind: match route.kind {
                        diffpack_next::artifacts::NextRouteArtifactKind::AppPage
                        | diffpack_next::artifacts::NextRouteArtifactKind::ImplicitAppPage => {
                            "app-page"
                        }
                        diffpack_next::artifacts::NextRouteArtifactKind::AppRoute => "app-route",
                        diffpack_next::artifacts::NextRouteArtifactKind::PagesPage => "pages-page",
                        diffpack_next::artifacts::NextRouteArtifactKind::PagesApi => "pages-api",
                    }
                    .to_string(),
                })
                .collect();
            Ok(Some(Response {
                protocol_version: PROTOCOL_VERSION,
                ok: true,
                artifact_root: Some(project_root.join(".diffpack-output")),
                error: None,
                routes,
            }))
        }
    }
}

fn check_protocol(protocol_version: u32) -> Result<(), String> {
    if protocol_version == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(format!(
            "unsupported protocol version {protocol_version}; expected {PROTOCOL_VERSION}"
        ))
    }
}

fn production_request(
    project_root: &PathBuf,
    output_dir: &PathBuf,
    next_config_output: Option<&str>,
) -> Result<Response, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("cannot locate Diffpack Next bridge: {error}"))?;
    let mut child = Command::new(&executable)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", executable.display()))?;
    let request = serde_json::json!({
        "operation": "build-production",
        "protocol_version": PROTOCOL_VERSION,
        "project_root": project_root,
        "output_dir": output_dir,
        "next_config_output": next_config_output,
        "development": true,
    });
    serde_json::to_writer(
        child
            .stdin
            .as_mut()
            .ok_or_else(|| "bridge stdin was not piped".to_string())?,
        &request,
    )
    .map_err(|error| format!("cannot encode development build request: {error}"))?;
    drop(child.stdin.take());
    let output = child
        .wait_with_output()
        .map_err(|error| format!("cannot wait for development build: {error}"))?;
    serde_json::from_slice(&output.stdout).map_err(|error| {
        format!(
            "development build returned invalid JSON: {error}: {}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

fn write_event(response: &Response) -> Result<(), String> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, response)
        .map_err(|error| format!("cannot serialize development event: {error}"))?;
    writeln!(stdout).map_err(|error| format!("cannot write development event: {error}"))?;
    stdout
        .flush()
        .map_err(|error| format!("cannot flush development event: {error}"))
}

fn watch_development(
    project_root: &PathBuf,
    output_dir: &PathBuf,
    next_config_output: Option<&str>,
    _poll_interval_ms: Option<u64>,
) -> Result<(), String> {
    write_event(&production_request(
        project_root,
        output_dir,
        next_config_output,
    )?)?;

    let (send, receive) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |event| {
        let _ = send.send(event);
    })
    .map_err(|error| format!("cannot create development watcher: {error}"))?;
    watcher
        .watch(project_root, RecursiveMode::Recursive)
        .map_err(|error| format!("cannot watch {}: {error}", project_root.display()))?;

    while let Ok(event) = receive.recv() {
        let event = event.map_err(|error| format!("development watcher failed: {error}"))?;
        let relevant = event.paths.iter().any(|path| {
            path.starts_with(project_root)
                && !path.starts_with(output_dir)
                && !path.starts_with(project_root.join(".diffpack-next"))
                && !path.starts_with(project_root.join(".diffpack-output"))
                && !path
                    .components()
                    .any(|part| part.as_os_str() == "node_modules")
                && matches!(
                    path.extension().and_then(|value| value.to_str()),
                    Some(
                        "js" | "jsx"
                            | "ts"
                            | "tsx"
                            | "mjs"
                            | "cjs"
                            | "json"
                            | "css"
                            | "scss"
                            | "sass"
                            | "less"
                            | "md"
                            | "mdx"
                    )
                )
        });
        if !relevant {
            continue;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        while receive.try_recv().is_ok() {}
        let response = production_request(project_root, output_dir, next_config_output)
            .unwrap_or_else(|error| Response {
                protocol_version: PROTOCOL_VERSION,
                ok: false,
                artifact_root: None,
                error: Some(error),
                routes: Vec::new(),
            });
        write_event(&response)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_version_is_explicit() {
        assert_eq!(PROTOCOL_VERSION, 1);
        let request = serde_json::from_str::<Request>(
            r#"{"operation":"build-production","protocol_version":1,"project_root":"/app","output_dir":"/app/.next"}"#,
        )
        .unwrap();
        assert!(matches!(request, Request::BuildProduction { .. }));
    }
}
