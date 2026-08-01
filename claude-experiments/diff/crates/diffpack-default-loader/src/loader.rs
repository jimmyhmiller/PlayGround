//! Built-in query-loader classification and diagnostics.

use diffpack_core::ResourceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoaderKind {
    Url,
    Raw,
    TsrSplit,
    CssMedia,
    Worker,
    Inline,
    WasmInit,
    PublicUrl,
}

impl LoaderKind {
    pub fn token(self) -> &'static str {
        match self {
            Self::Url => "url",
            Self::Raw => "raw",
            Self::TsrSplit => "tsr-split",
            Self::CssMedia => "media",
            Self::Worker => "worker",
            Self::Inline => "inline",
            Self::WasmInit => "init",
            Self::PublicUrl => "public-url",
        }
    }
}

pub fn kind(resource: &ResourceId) -> Option<LoaderKind> {
    let query = resource.query.as_deref()?;
    match query.split(['=', '&']).next().unwrap_or(query) {
        "url" => Some(LoaderKind::Url),
        "raw" => Some(LoaderKind::Raw),
        "tsr-split" => Some(LoaderKind::TsrSplit),
        "media" => Some(LoaderKind::CssMedia),
        "worker" | "sharedworker" => Some(LoaderKind::Worker),
        "inline" => Some(LoaderKind::Inline),
        "init" => Some(LoaderKind::WasmInit),
        "public-url" => Some(LoaderKind::PublicUrl),
        _ => None,
    }
}

pub fn unimplemented_error(resource: &ResourceId) -> String {
    match resource.query.as_deref() {
        None => format!(
            "unimplemented loader error requested for query-less id ({})",
            resource.path
        ),
        Some(query) => match kind(resource) {
            Some(loader) => format!(
                "loader `?{}` is not yet implemented (requested for {})",
                loader.token(),
                resource.path
            ),
            None => format!(
                "unrecognized loader query `?{query}` (requested for {})",
                resource.path
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_builtin_and_integration_queries() {
        assert_eq!(kind(&ResourceId::parse("a.css?url")), Some(LoaderKind::Url));
        assert_eq!(
            kind(&ResourceId::parse("route.tsx?tsr-split=component")),
            Some(LoaderKind::TsrSplit)
        );
        assert_eq!(kind(&ResourceId::parse("a.css?unknown")), None);
    }
}
