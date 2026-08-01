//! Framework-neutral Node resolution policy shared by the default resolver and
//! integrations that preflight project imports.

/// Node built-in module names without the `node:` prefix.
pub const NODE_BUILTINS: &[&str] = &[
    "assert",
    "async_hooks",
    "buffer",
    "child_process",
    "cluster",
    "console",
    "constants",
    "crypto",
    "dgram",
    "diagnostics_channel",
    "dns",
    "domain",
    "events",
    "fs",
    "http",
    "http2",
    "https",
    "inspector",
    "module",
    "net",
    "os",
    "path",
    "perf_hooks",
    "process",
    "punycode",
    "querystring",
    "readline",
    "repl",
    "stream",
    "string_decoder",
    "sys",
    "timers",
    "tls",
    "trace_events",
    "tty",
    "url",
    "util",
    "v8",
    "vm",
    "wasi",
    "worker_threads",
    "zlib",
];

/// Whether a specifier names a Node built-in, either through `node:` or its
/// conventional bare name.
pub fn is_node_builtin(specifier: &str) -> bool {
    if let Some(builtin) = specifier.strip_prefix("node:") {
        return !builtin.is_empty();
    }
    let root = specifier.split('/').next().unwrap_or(specifier);
    NODE_BUILTINS.contains(&root)
}

/// Whether the default loader leaves a specifier for the Node runtime.
pub fn is_external_specifier(specifier: &str) -> bool {
    is_node_builtin(specifier)
}

/// The package owned by a bare package specifier.
pub fn bare_package_name(specifier: &str) -> Option<String> {
    if specifier.starts_with('.') || specifier.starts_with('/') || specifier.starts_with('#') {
        return None;
    }
    let mut segments = specifier.split('/');
    let first = segments.next().filter(|segment| !segment.is_empty())?;
    match first.strip_prefix('@') {
        Some(scope) if !scope.is_empty() => {
            let second = segments.next().filter(|segment| !segment.is_empty())?;
            Some(format!("{first}/{second}"))
        }
        _ => Some(first.to_string()),
    }
}

const RESOURCE_SCHEMES: &[&str] = &["http", "https", "data", "file", "blob"];

/// Returns the runtime scheme of a host-provided module such as
/// `cloudflare:sockets`, excluding URLs and ordinary paths/packages.
pub fn host_provided_scheme(specifier: &str) -> Option<&str> {
    let (scheme, rest) = specifier.split_once(':')?;
    if scheme.len() < 2 || rest.is_empty() || rest.starts_with('/') {
        return None;
    }
    if !scheme.chars().next()?.is_ascii_alphabetic()
        || !scheme.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
        })
    {
        return None;
    }
    (!RESOURCE_SCHEMES.contains(&scheme)).then_some(scheme)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_builtins_are_external_in_both_spellings() {
        for specifier in ["node:stream", "node:fs/promises", "fs", "path/posix"] {
            assert!(is_external_specifier(specifier), "{specifier}");
        }
        assert!(!is_external_specifier("react"));
        assert!(!is_external_specifier("./local"));
        assert!(!is_external_specifier("node:"));
    }

    #[test]
    fn classifies_package_names_and_host_schemes() {
        assert_eq!(
            bare_package_name("@scope/pkg/sub"),
            Some("@scope/pkg".into())
        );
        assert_eq!(bare_package_name("pkg/sub"), Some("pkg".into()));
        assert_eq!(bare_package_name("./local"), None);
        assert_eq!(
            host_provided_scheme("cloudflare:sockets"),
            Some("cloudflare")
        );
        assert_eq!(host_provided_scheme("https://example.com/x"), None);
        assert_eq!(host_provided_scheme("C:/x"), None);
    }
}
