//! Compiling a Tailwind entry with the APP's OWN installed Tailwind.
//!
//! diffpack's native Tailwind engine ([`crate::tailwind`]) owns the sheets it can
//! serve completely — no Node process, no `node_modules` read, which is the whole
//! performance story. But a Tailwind plugin is an arbitrary JavaScript function that
//! registers utilities, variants and base rules at runtime; `@tailwindcss/typography`,
//! `tailwind-scrollbar`, `@tailwindcss/forms` and `daisyui` are just the popular head
//! of an unbounded tail. Reimplementing them in Rust would never finish and would
//! drift from every release.
//!
//! So when [`crate::tailwind::native_gap`] finds something in an entry the native
//! engine does not implement, the WHOLE entry compile is handed to the app's own
//! Tailwind — the same shape already shipped for MDX plugins ([`crate::mdx`]),
//! PostCSS, Less/Stylus and Vue/Svelte SFCs: a Node side-process running the app's
//! own tool. This is not a plugin framework and must not become one; it is one
//! delegation of one compile.
//!
//! What crosses the boundary is deliberately small. diffpack still splices the
//! entry's `@import`s, rewrites its `url()`s to content-hashed asset URLs,
//! absolutizes its `@source` globs, and scans the class candidates — so both engines
//! see the same input and produce a sheet with the same asset references. Only the
//! CSS-to-CSS compile itself is delegated.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// The node script that drives the app's own Tailwind compiler.
const RUNNER: &str = include_str!("tailwind_runner.mjs");

/// A stylesheet produced by the app's own Tailwind.
#[derive(Debug)]
pub struct DelegatedSheet {
    /// The compiled stylesheet.
    pub css: String,
    /// Which package compiled it (`@tailwindcss/node` or `tailwindcss`).
    pub engine: String,
    /// That package's installed version.
    pub version: String,
}

/// Compiles `css` (the already-spliced entry text of `entry`) against `candidates`
/// with the app's own Tailwind.
///
/// `gap` is the reason the native engine could not serve the sheet; it is carried
/// into every error message so a failure here always says what made diffpack leave
/// the native path.
/// `cancel` lets a DEFERRED compile be abandoned: the dev loop recompiles the sheet
/// after an edit, and if the developer types again while the app's Tailwind is running
/// this compile's result is already stale. The child is then killed rather than waited
/// out, and `None` is returned. Production builds pass
/// [`crate::bundler::EmitCancel::never`] and always run to completion.
pub fn compile(
    entry: &Path,
    css: &str,
    candidates: &BTreeSet<String>,
    gap: &crate::tailwind::NativeGap,
    cancel: crate::bundler::EmitCancel<'_>,
) -> Result<Option<DelegatedSheet>, String> {
    let _stage = crate::build_profile::stage("css/tailwind-delegate");
    let context = format!(
        "Tailwind {}: {gap}, so the sheet is compiled with the app's own tailwindcss",
        entry.display()
    );
    let request = serde_json::json!({
        "css": css,
        "candidates": candidates.iter().collect::<Vec<_>>(),
    })
    .to_string();

    // The runner lives in a temp file; it resolves the app's Tailwind from `entry`,
    // never from its own location (see its header). `current_dir` is the entry's
    // directory so a plugin that reads relative paths behaves as under `next build`.
    let loader = std::env::temp_dir().join("diffpack-tailwind-runner.mjs");
    std::fs::write(&loader, RUNNER)
        .map_err(|error| format!("{context} — cannot write {}: {error}", loader.display()))?;

    let mut child = Command::new("node")
        .arg("--no-warnings")
        .arg(&loader)
        .arg(entry)
        .current_dir(entry.parent().unwrap_or_else(|| Path::new(".")))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("{context} — cannot run node: {error}"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| format!("{context} — node stdin unavailable"))?
        .write_all(request.as_bytes())
        .map_err(|error| format!("{context} — cannot write to node: {error}"))?;
    // The compiled sheet is far bigger than a pipe buffer, so BOTH streams are drained
    // by their own threads while this one polls: a reader-less poll loop would fill the
    // stdout pipe, block the compiler inside its own write, and never see it exit.
    // Draining on threads keeps the cancel signal answerable within milliseconds
    // instead of at the end of a compile whose result nobody wants any more.
    let drain = |stream: Option<std::process::ChildStdout>| {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut stream) = stream {
                use std::io::Read;
                let _ = stream.read_to_end(&mut bytes);
            }
            let _ = sender.send(bytes);
        });
        receiver
    };
    let stdout_rx = drain(child.stdout.take());
    let stderr_rx = {
        let stream = child.stderr.take();
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut stream) = stream {
                use std::io::Read;
                let _ = stream.read_to_end(&mut bytes);
            }
            let _ = sender.send(bytes);
        });
        receiver
    };
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(error) => return Err(format!("{context} — cannot wait for node: {error}")),
        }
        if cancel.cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(None);
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    };
    let stdout_bytes = stdout_rx.recv().unwrap_or_default();
    let stderr_bytes = stderr_rx.recv().unwrap_or_default();
    if !status.success() {
        return Err(format!(
            "{context} — that compiler failed:\n{}",
            String::from_utf8_lossy(&stderr_bytes).trim()
        ));
    }
    let parsed: serde_json::Value = serde_json::from_slice(&stdout_bytes).map_err(|error| {
        format!(
            "{context} — unreadable compiler output ({error}): {}",
            String::from_utf8_lossy(&stdout_bytes).trim()
        )
    })?;
    let sheet = parsed
        .get("css")
        .and_then(|value| value.as_str())
        .ok_or_else(|| format!("{context} — compiler output has no `css` field"))?;
    Ok(Some(DelegatedSheet {
        css: sheet.to_string(),
        engine: parsed
            .get("engine")
            .and_then(|value| value.as_str())
            .unwrap_or("tailwindcss")
            .to_string(),
        version: parsed
            .get("version")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tailwind::{NativeGap, native_gap};
    use std::fs;
    use std::path::{Path, PathBuf};

    fn node_available() -> bool {
        Command::new("node").arg("--version").output().is_ok()
    }

    /// A real Tailwind v4 install to delegate to. The corpus apps carry one; it is
    /// never vendored into this repo, so a checkout that has not fetched them yet
    /// has nothing to test against.
    fn corpus_node_modules(required: &str) -> Option<PathBuf> {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let apps = fs::read_dir(repo.join("integration/e2e/apps")).ok()?;
        for app in apps.flatten() {
            let modules = app.path().join("node_modules");
            if modules.join(required).join("package.json").is_file() {
                return Some(modules);
            }
        }
        None
    }

    #[cfg(unix)]
    fn link(from: &Path, to: &Path) {
        std::os::unix::fs::symlink(from, to).unwrap();
    }
    #[cfg(windows)]
    fn link(from: &Path, to: &Path) {
        std::os::windows::fs::symlink_dir(from, to).unwrap();
    }

    /// An entry that loads a plugin written for this test. `@plugin` is the gap,
    /// and the utility the plugin registers is one no built-in and no `@utility`
    /// provides — so its presence in the output proves the app's own compiler ran.
    fn plugin_entry(dir: &Path) -> (PathBuf, &'static str) {
        fs::write(
            dir.join("plugin.js"),
            "module.exports = function ({ addUtilities }) {\n\
             \x20 addUtilities({ '.diffpack-probe': { 'caret-color': 'rebeccapurple' } });\n\
             };\n",
        )
        .unwrap();
        let css = "@import 'tailwindcss';\n@plugin './plugin.js';\n";
        let entry = dir.join("entry.css");
        fs::write(&entry, css).unwrap();
        (entry, css)
    }

    fn probe_candidates() -> BTreeSet<String> {
        ["diffpack-probe".to_string(), "flex".to_string()].into_iter().collect()
    }

    /// The preferred engine: `@tailwindcss/node`, the adapter `@tailwindcss/postcss`,
    /// `@tailwindcss/vite` and `@tailwindcss/cli` all drive.
    #[test]
    fn delegates_through_the_apps_tailwindcss_node_adapter() {
        if !node_available() {
            return;
        }
        let Some(modules) = corpus_node_modules("@tailwindcss/node") else {
            eprintln!("skipped: no corpus app has @tailwindcss/node installed");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        link(&modules, &dir.path().join("node_modules"));
        let (entry, css) = plugin_entry(dir.path());

        let gap = native_gap(css, None).expect("a `@plugin` is a native gap");
        assert_eq!(gap, NativeGap::Plugin("./plugin.js".to_string()));

        let sheet = compile(&entry, css, &probe_candidates(), &gap, crate::bundler::EmitCancel::never())
            .unwrap()
            .expect("an uncancellable compile always produces a sheet");
        assert_eq!(sheet.engine, "@tailwindcss/node");
        assert!(sheet.version.starts_with('4'), "version {:?}", sheet.version);
        assert!(
            sheet.css.contains("rebeccapurple"),
            "the plugin's utility is missing from the delegated sheet"
        );
        // The rest of the sheet is a real Tailwind compile, not just the plugin.
        assert!(sheet.css.contains("display: flex") || sheet.css.contains("display:flex"));
    }

    /// The fallback engine: an app that installs `tailwindcss` but none of the
    /// packages that bundle `@tailwindcss/node`. The runner then supplies the
    /// module/stylesheet loaders itself.
    #[test]
    fn delegates_through_core_tailwindcss_when_the_adapter_is_absent() {
        if !node_available() {
            return;
        }
        let Some(modules) = corpus_node_modules("tailwindcss") else {
            eprintln!("skipped: no corpus app has tailwindcss installed");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        // A node_modules holding ONLY tailwindcss: `@tailwindcss/node` resolves
        // nowhere from here, so the core path is the one under test.
        let local = dir.path().join("node_modules");
        fs::create_dir_all(&local).unwrap();
        link(&modules.join("tailwindcss"), &local.join("tailwindcss"));
        let (entry, css) = plugin_entry(dir.path());

        let gap = native_gap(css, None).unwrap();
        let sheet = compile(&entry, css, &probe_candidates(), &gap, crate::bundler::EmitCancel::never())
            .unwrap()
            .expect("an uncancellable compile always produces a sheet");
        assert_eq!(sheet.engine, "tailwindcss");
        assert!(sheet.version.starts_with('4'), "version {:?}", sheet.version);
        assert!(sheet.css.contains("rebeccapurple"));
        assert!(sheet.css.contains("display: flex") || sheet.css.contains("display:flex"));
    }

    /// No usable Tailwind and a sheet that needs one: a hard error naming the entry
    /// and the package. Never a silently under-compiled sheet.
    #[test]
    fn a_missing_tailwindcss_is_a_hard_error_naming_the_entry() {
        if !node_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let (entry, css) = plugin_entry(dir.path());
        let gap = native_gap(css, None).unwrap();
        let error = compile(&entry, css, &probe_candidates(), &gap, crate::bundler::EmitCancel::never())
            .unwrap_err();
        assert!(error.contains(&entry.display().to_string()), "{error}");
        assert!(error.contains("tailwindcss"), "{error}");
        // And it says WHY the native engine was left.
        assert!(error.contains("@plugin"), "{error}");
    }
}
