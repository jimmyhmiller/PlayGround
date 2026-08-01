//! Query-aware module identity.
//!
//! A bundler module id is not a bare path. Host tools may attach a query
//! (`module.js?raw`) and, occasionally, a fragment. The query is
//! part of the module's identity: `module.js` and `module.js?raw` are two distinct
//! modules that load differently. Treating the whole string as a literal path
//! makes any path-based loader receive an invalid path.
//!
//! [`ResourceId`] splits a specifier/id into `(path, query, fragment)` at the
//! resolution boundary and round-trips it back losslessly, so the path (and only
//! the path) is passed to a loader while the query remains in graph identity.

/// A parsed module identity: a resource path plus an optional loader query and
/// fragment.
///
/// Parsing splits on the first `?` (query) and the first `#` (fragment), with
/// the fragment binding *after* the query. A `#` that appears before any `?`
/// is a fragment with no query (the later `?` becomes part of the fragment
/// text). An empty query string (`foo?`) is a present-but-empty query and is
/// deliberately distinct from an absent query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceId {
    pub path: String,
    pub query: Option<String>,
    pub fragment: Option<String>,
}

impl ResourceId {
    /// Splits a specifier/id into `(path, query, fragment)`.
    ///
    /// Never panics and never allocates beyond the component strings.
    pub fn parse(input: &str) -> Self {
        let question = input.find('?');
        // A leading `#` is a Node subpath-import specifier (`#tanstack-router-entry`),
        // not a URL fragment; a fragment requires content before it. So only a
        // `#` past position 0 (and, when a query is present, past the `?`) opens a
        // fragment.
        let hash = input.find('#').filter(|&index| index > 0);
        match (question, hash) {
            // `path?query#fragment`: query runs from `?` to the first `#`.
            (Some(q), Some(h)) if q < h => Self {
                path: input[..q].to_string(),
                query: Some(input[q + 1..h].to_string()),
                fragment: Some(input[h + 1..].to_string()),
            },
            // `path#fragment` where the `#` precedes any `?`: fragment-only.
            // Any `?` after the `#` is part of the fragment text.
            (_, Some(h)) => Self {
                path: input[..h].to_string(),
                query: None,
                fragment: Some(input[h + 1..].to_string()),
            },
            // `path?query` with no fragment.
            (Some(q), None) => Self {
                path: input[..q].to_string(),
                query: Some(input[q + 1..].to_string()),
                fragment: None,
            },
            // Bare path.
            (None, None) => Self {
                path: input.to_string(),
                query: None,
                fragment: None,
            },
        }
    }

    /// Reconstructs the `path?query#fragment` string byte-for-byte, only
    /// re-emitting the `?`/`#` separators for components that are present.
    pub fn to_id(&self) -> String {
        let mut id = self.path.clone();
        if let Some(query) = &self.query {
            id.push('?');
            id.push_str(query);
        }
        if let Some(fragment) = &self.fragment {
            id.push('#');
            id.push_str(fragment);
        }
        id
    }

    /// Whether the query carries `flag` as one of its `&`-separated tokens (a
    /// bare flag, e.g. the `inline` in `?worker&inline`). The token must be a
    /// standalone key, not a value: `?worker&inline` matches `inline`,
    /// `?name=inline` does not.
    pub fn query_has_flag(&self, flag: &str) -> bool {
        let Some(query) = self.query.as_deref() else {
            return false;
        };
        query
            .split('&')
            .any(|token| token.split('=').next() == Some(flag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trips(input: &str) -> ResourceId {
        let parsed = ResourceId::parse(input);
        assert_eq!(parsed.to_id(), input, "to_id must reconstruct {input:?}");
        parsed
    }

    #[test]
    fn parses_a_bare_path() {
        let parsed = round_trips("src/styles/app.css");
        assert_eq!(parsed.path, "src/styles/app.css");
        assert_eq!(parsed.query, None);
        assert_eq!(parsed.fragment, None);
    }

    #[test]
    fn parses_a_path_with_query() {
        let parsed = round_trips("src/styles/app.css?url");
        assert_eq!(parsed.path, "src/styles/app.css");
        assert_eq!(parsed.query.as_deref(), Some("url"));
        assert_eq!(parsed.fragment, None);
    }

    #[test]
    fn parses_a_path_with_query_and_fragment() {
        let parsed = round_trips("a.tsx?view=component#anchor");
        assert_eq!(parsed.path, "a.tsx");
        assert_eq!(parsed.query.as_deref(), Some("view=component"));
        assert_eq!(parsed.fragment.as_deref(), Some("anchor"));
    }

    #[test]
    fn parses_a_fragment_before_any_query() {
        // The `#` binds first: the trailing `?b` is part of the fragment, not a
        // query.
        let parsed = round_trips("mod.js#a?b");
        assert_eq!(parsed.path, "mod.js");
        assert_eq!(parsed.query, None);
        assert_eq!(parsed.fragment.as_deref(), Some("a?b"));
    }

    #[test]
    fn a_leading_hash_is_a_subpath_import_not_a_fragment() {
        let parsed = round_trips("#tanstack-router-entry");
        assert_eq!(parsed.path, "#tanstack-router-entry");
        assert_eq!(parsed.query, None);
        assert_eq!(parsed.fragment, None);

        // A leading-hash specifier can still carry a query.
        let queried = round_trips("#entry?url");
        assert_eq!(queried.path, "#entry");
        assert_eq!(queried.query.as_deref(), Some("url"));
        assert_eq!(queried.fragment, None);
    }

    #[test]
    fn parses_a_fragment_only_path() {
        let parsed = round_trips("mod.js#section");
        assert_eq!(parsed.path, "mod.js");
        assert_eq!(parsed.query, None);
        assert_eq!(parsed.fragment.as_deref(), Some("section"));
    }

    #[test]
    fn distinguishes_an_empty_query_from_an_absent_query() {
        let empty = round_trips("foo?");
        assert_eq!(empty.path, "foo");
        assert_eq!(empty.query.as_deref(), Some(""));
        assert_eq!(empty.fragment, None);

        let absent = round_trips("foo");
        assert_eq!(absent.query, None);
        assert_ne!(empty, absent);
    }

    #[test]
    fn preserves_a_query_value_containing_equals() {
        let parsed = round_trips("route.tsx?view=component");
        assert_eq!(parsed.query.as_deref(), Some("view=component"));
    }

    #[test]
    fn query_has_flag_matches_only_standalone_bare_flags() {
        let worker_inline = ResourceId::parse("w.js?worker&inline");
        assert!(worker_inline.query_has_flag("worker"));
        assert!(worker_inline.query_has_flag("inline"));
        assert!(!worker_inline.query_has_flag("url"));

        // A value, not a bare flag, must not match.
        let named = ResourceId::parse("w.js?name=inline");
        assert!(!named.query_has_flag("inline"));
        assert!(named.query_has_flag("name"));

        assert!(!ResourceId::parse("w.js").query_has_flag("inline"));
    }
}
