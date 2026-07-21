//! Vite's `.env` file convention, loaded natively.
//!
//! Vite loads `.env`, `.env.local`, `.env.<mode>`, and `.env.<mode>.local` from
//! the project root, in that order, with later files overriding earlier ones and
//! the real process environment overriding every file. Only `VITE_`-prefixed
//! variables are exposed to client code through `import.meta.env`. This module
//! reimplements that convention without dotenv: a plain key/value parse with the
//! quoting rules code in the wild actually relies on.
//!
//! Deliberately unsupported (hard errors, never silently-wrong values):
//! variable expansion (`$VAR` / `${VAR}` outside single quotes) and multiline
//! double-quoted values. Both are dotenv-expand features a project may lean on;
//! loading such a file and producing an unexpanded literal would be a
//! wrong-valued build, so the parse refuses and names the file and key instead.

use std::path::Path;

/// Loads the Vite env-file stack for `mode` from `root` and returns every
/// `VITE_`-prefixed variable as `(name, value)`, files overridden by later
/// files and everything overridden by the real process environment.
pub fn load_vite_env(root: &Path, mode: &str) -> Result<Vec<(String, String)>, String> {
    let mut merged: Vec<(String, String)> = Vec::new();
    let mut set = |name: String, value: String| {
        match merged.iter_mut().find(|(existing, _)| *existing == name) {
            Some(entry) => entry.1 = value,
            None => merged.push((name, value)),
        }
    };
    for file_name in [
        ".env".to_string(),
        ".env.local".to_string(),
        format!(".env.{mode}"),
        format!(".env.{mode}.local"),
    ] {
        let path = root.join(&file_name);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (name, value) in parse(&text, &path.display().to_string())? {
            set(name, value);
        }
    }
    // The process environment wins over every file, exactly as in Vite.
    for (name, value) in std::env::vars() {
        if name.starts_with("VITE_") {
            set(name, value);
        }
    }
    Ok(merged
        .into_iter()
        .filter(|(name, _)| name.starts_with("VITE_"))
        .collect())
}

/// Parses one env file's text into `(name, value)` pairs, in file order.
/// `origin` names the file in errors.
fn parse(text: &str, origin: &str) -> Result<Vec<(String, String)>, String> {
    let mut variables = Vec::new();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // dotenv accepts an optional `export ` prefix so a file doubles as a
        // shell script.
        let line = line.strip_prefix("export ").map(str::trim).unwrap_or(line);
        let Some((name, rest)) = line.split_once('=') else {
            return Err(format!(
                "{origin}:{}: not a KEY=value line: `{raw_line}`",
                line_index + 1
            ));
        };
        let name = name.trim();
        if name.is_empty()
            || !name
                .chars()
                .enumerate()
                .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        {
            return Err(format!(
                "{origin}:{}: invalid variable name `{name}`",
                line_index + 1
            ));
        }
        let value = parse_value(rest.trim(), origin, line_index + 1, name)?;
        variables.push((name.to_string(), value));
    }
    Ok(variables)
}

fn parse_value(raw: &str, origin: &str, line: usize, name: &str) -> Result<String, String> {
    if let Some(inner) = quoted(raw, '\'') {
        // Single quotes are fully literal — `$` included.
        return Ok(inner.to_string());
    }
    let (value, expandable) = match quoted(raw, '"') {
        // Double quotes: `\n` escapes expand (the one escape dotenv processes),
        // and the value is still subject to `$` expansion.
        Some(inner) => (inner.replace("\\n", "\n"), inner.to_string()),
        None => {
            // Unquoted: an ` #` starts a trailing comment.
            let value = match raw.find(" #") {
                Some(index) => raw[..index].trim_end(),
                None => raw,
            };
            (value.to_string(), value.to_string())
        }
    };
    // dotenv-expand would substitute `$VAR`/`${VAR}` here. Producing the
    // unexpanded literal instead would be a silently wrong value, so refuse.
    let mut chars = expandable.char_indices().peekable();
    while let Some((_, c)) = chars.next() {
        if c == '\\' {
            chars.next();
            continue;
        }
        if c == '$'
            && let Some(&(_, next)) = chars.peek()
            && (next == '{' || next == '_' || next.is_ascii_alphabetic())
        {
            return Err(format!(
                "{origin}:{line}: `{name}` uses variable expansion (`$...`), which \
                 diffpack does not support; inline the value or single-quote it \
                 to keep the `$` literal"
            ));
        }
    }
    Ok(value)
}

/// The inside of `raw` when it is wholly wrapped in `quote`, else `None`.
fn quoted(raw: &str, quote: char) -> Option<&str> {
    let inner = raw.strip_prefix(quote)?.strip_suffix(quote)?;
    Some(inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, contents: &str) {
        std::fs::write(dir.join(name), contents).unwrap();
    }

    #[test]
    fn later_files_override_earlier_and_only_vite_vars_are_exposed() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), ".env", "VITE_A=base\nVITE_B=base\nSECRET=hidden\n");
        write(dir.path(), ".env.local", "VITE_B=local\n");
        write(dir.path(), ".env.production", "VITE_A=production\n");
        let vars = load_vite_env(dir.path(), "production").unwrap();
        // `contains` rather than exact equality: sibling tests may set their own
        // VITE_* process variables concurrently, and the process env is merged in.
        assert!(vars.contains(&("VITE_A".to_string(), "production".to_string())), "{vars:?}");
        assert!(vars.contains(&("VITE_B".to_string(), "local".to_string())), "{vars:?}");
        assert!(!vars.iter().any(|(name, _)| name == "SECRET"), "{vars:?}");
    }

    #[test]
    fn mode_local_wins_over_mode_and_other_modes_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), ".env.production", "VITE_X=production\n");
        write(dir.path(), ".env.production.local", "VITE_X=production-local\n");
        write(dir.path(), ".env.development", "VITE_X=development\n");
        let vars = load_vite_env(dir.path(), "production").unwrap();
        assert!(
            vars.contains(&("VITE_X".to_string(), "production-local".to_string())),
            "{vars:?}"
        );
    }

    #[test]
    fn quoting_comments_and_export_prefix() {
        let parsed = parse(
            "# comment\nexport VITE_A=plain value\nVITE_B=\"line\\nbreak\"\nVITE_C='$literal'\nVITE_D=inline # trailing note\n\n",
            "test.env",
        )
        .unwrap();
        assert_eq!(
            parsed,
            vec![
                ("VITE_A".to_string(), "plain value".to_string()),
                ("VITE_B".to_string(), "line\nbreak".to_string()),
                ("VITE_C".to_string(), "$literal".to_string()),
                ("VITE_D".to_string(), "inline".to_string()),
            ]
        );
    }

    #[test]
    fn expansion_is_a_hard_error_naming_the_file_and_key() {
        let error = parse("VITE_URL=$BASE/api\n", "test.env").unwrap_err();
        assert!(error.contains("test.env:1"), "{error}");
        assert!(error.contains("VITE_URL"), "{error}");
        assert!(error.contains("expansion"), "{error}");
        let error = parse("VITE_URL=\"${BASE}/api\"\n", "test.env").unwrap_err();
        assert!(error.contains("expansion"), "{error}");
    }

    #[test]
    fn malformed_lines_are_hard_errors() {
        assert!(parse("JUSTAWORD\n", "test.env").unwrap_err().contains("KEY=value"));
        assert!(parse("2BAD=x\n", "test.env").unwrap_err().contains("invalid variable name"));
    }

    #[test]
    fn process_environment_wins_over_files() {
        let dir = tempfile::tempdir().unwrap();
        // Use a name unique to this test to avoid cross-test env pollution.
        write(dir.path(), ".env", "VITE_ENV_FILE_TEST_PRECEDENCE=from-file\n");
        // SAFETY: test-only; the name is unique to this test so no other thread
        // reads it concurrently.
        unsafe { std::env::set_var("VITE_ENV_FILE_TEST_PRECEDENCE", "from-process") };
        let vars = load_vite_env(dir.path(), "production").unwrap();
        unsafe { std::env::remove_var("VITE_ENV_FILE_TEST_PRECEDENCE") };
        assert!(vars.contains(&(
            "VITE_ENV_FILE_TEST_PRECEDENCE".to_string(),
            "from-process".to_string()
        )));
    }
}
