//! Query-aware module identity.
//!
//! A bundler module id is not a bare filesystem path. Tools like Vite and
//! TanStack Start attach a loader query (`app.css?url`, `logo.svg?raw`,
//! `route.tsx?tsr-split=component`) and, occasionally, a fragment. The query is
//! part of the module's identity: `app.css` and `app.css?url` are two distinct
//! modules that load differently. Treating the whole string as a literal path
//! makes `fs::read_to_string` crash on the `?` the moment such an id reaches the
//! load frontier.
//!
//! [`ResourceId`] splits a specifier/id into `(path, query, fragment)` at the
//! resolution boundary and round-trips it back losslessly, so the path (and only
//! the path) touches the filesystem while the query is carried through the graph
//! for the loader to interpret later.

/// A parsed module identity: a resource path plus an optional loader query and
/// fragment.
///
/// Parsing splits on the first `?` (query) and the first `#` (fragment), with
/// the fragment binding *after* the query. A `#` that appears before any `?`
/// is a fragment with no query (the later `?` becomes part of the fragment
/// text). An empty query string (`foo?`) is a present-but-empty query and is
/// deliberately distinct from an absent query, matching Vite semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceId {
    pub path: String,
    pub query: Option<String>,
    pub fragment: Option<String>,
}

/// A recognized loader query. These are recognized-but-unimplemented: the point
/// of naming them is to produce a specific, actionable error instead of a
/// misleading file-not-found crash, not to implement the loader here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoaderKind {
    /// `?url` — emit the asset and import its public URL.
    Url,
    /// `?raw` — import the file's contents as a string.
    Raw,
    /// `?tsr-split=…` — TanStack Router server/client route splitting.
    TsrSplit,
}

impl LoaderKind {
    /// The canonical query token for this loader, without the leading `?`.
    pub fn token(self) -> &'static str {
        match self {
            LoaderKind::Url => "url",
            LoaderKind::Raw => "raw",
            LoaderKind::TsrSplit => "tsr-split",
        }
    }
}

impl ResourceId {
    /// Splits a specifier/id into `(path, query, fragment)`.
    ///
    /// Never panics and never allocates beyond the component strings.
    pub fn parse(input: &str) -> Self {
        let question = input.find('?');
        let hash = input.find('#');
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

    /// Classifies this id's query into a recognized [`LoaderKind`], if any.
    ///
    /// Returns `None` when there is no query or the query key is not one we
    /// recognize.
    pub fn loader_kind(&self) -> Option<LoaderKind> {
        let query = self.query.as_deref()?;
        // A query may be `key`, `key=value`, or `key&other`; the loader is
        // identified by the first key.
        let key = query.split(['=', '&']).next().unwrap_or(query);
        match key {
            "url" => Some(LoaderKind::Url),
            "raw" => Some(LoaderKind::Raw),
            "tsr-split" => Some(LoaderKind::TsrSplit),
            _ => None,
        }
    }

    /// Builds the specific error for a query-bearing id whose loader is not yet
    /// implemented.
    ///
    /// Callers invoke this only when a query is present; a recognized loader is
    /// named by its canonical token, an unrecognized query is quoted verbatim.
    /// Both name the requested resource path. This is deliberately never a
    /// silent `null`/empty fallthrough and never the misleading
    /// `cannot read …: No such file or directory` crash.
    pub fn unimplemented_loader_error(&self) -> String {
        match self.query.as_deref() {
            None => format!(
                "unimplemented_loader_error called on a query-less id ({})",
                self.path
            ),
            Some(query) => match self.loader_kind() {
                Some(kind) => format!(
                    "loader `?{}` is not yet implemented (requested for {})",
                    kind.token(),
                    self.path
                ),
                None => format!(
                    "unrecognized loader query `?{}` (requested for {})",
                    query, self.path
                ),
            },
        }
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
        let parsed = round_trips("a.tsx?tsr-split=component#anchor");
        assert_eq!(parsed.path, "a.tsx");
        assert_eq!(parsed.query.as_deref(), Some("tsr-split=component"));
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
        let parsed = round_trips("route.tsx?tsr-split=component");
        assert_eq!(parsed.query.as_deref(), Some("tsr-split=component"));
    }

    #[test]
    fn classifies_recognized_loaders() {
        assert_eq!(
            ResourceId::parse("a.css?url").loader_kind(),
            Some(LoaderKind::Url)
        );
        assert_eq!(
            ResourceId::parse("a.svg?raw").loader_kind(),
            Some(LoaderKind::Raw)
        );
        assert_eq!(
            ResourceId::parse("a.tsx?tsr-split=component").loader_kind(),
            Some(LoaderKind::TsrSplit)
        );
        assert_eq!(ResourceId::parse("a.css").loader_kind(), None);
        assert_eq!(ResourceId::parse("a.css?weird").loader_kind(), None);
    }

    #[test]
    fn names_the_loader_and_resource_in_the_unimplemented_error() {
        let error = ResourceId::parse("src/styles/app.css?url").unimplemented_loader_error();
        assert_eq!(
            error,
            "loader `?url` is not yet implemented (requested for src/styles/app.css)"
        );
        // It must not be the misleading filesystem crash message.
        assert!(!error.contains("No such file or directory"));
        assert!(!error.contains("cannot read"));
    }

    #[test]
    fn names_an_unrecognized_query_specifically() {
        let error = ResourceId::parse("data.bin?mystery").unimplemented_loader_error();
        assert_eq!(
            error,
            "unrecognized loader query `?mystery` (requested for data.bin)"
        );
        assert!(!error.contains("No such file or directory"));
    }
}
