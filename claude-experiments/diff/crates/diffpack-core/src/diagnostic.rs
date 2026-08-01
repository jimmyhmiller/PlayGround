//! Structured diagnostics shared by graph construction and emission.

use std::path::{Path, PathBuf};

use crate::transform::TransformDiagnostic;

/// Attributes compiler diagnostics to a module path while preserving severity.
pub fn from_transform(path: &Path, diagnostics: &[TransformDiagnostic]) -> Vec<Diagnostic> {
    diagnostics
        .iter()
        .map(|diagnostic| Diagnostic {
            kind: DiagnosticKind::Source {
                fatal: diagnostic.fatal,
            },
            message: format!("{}: {}", path.display(), diagnostic.message),
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticKind {
    UnresolvedImport {
        specifier: String,
        importer: PathBuf,
    },
    NodeBuiltinInBrowser {
        specifier: String,
        importer: PathBuf,
    },
    Source {
        fatal: bool,
    },
    SideEffectsGlob,
    OptionalDependencyMissing {
        specifier: String,
        importer: PathBuf,
    },
    HostProvidedModule {
        specifier: String,
        importer: PathBuf,
    },
    SpecifierResolvesTwoWays {
        specifier: String,
        importer: PathBuf,
    },
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub kind: DiagnosticKind,
    pub message: String,
}

impl Diagnostic {
    pub fn is_fatal(&self) -> bool {
        match &self.kind {
            DiagnosticKind::UnresolvedImport { .. }
            | DiagnosticKind::NodeBuiltinInBrowser { .. }
            | DiagnosticKind::SpecifierResolvesTwoWays { .. } => true,
            DiagnosticKind::Source { fatal } => *fatal,
            DiagnosticKind::SideEffectsGlob
            | DiagnosticKind::OptionalDependencyMissing { .. }
            | DiagnosticKind::HostProvidedModule { .. } => false,
        }
    }
}

pub fn partition_diagnostics(
    diagnostics: &[Diagnostic],
    context: &str,
) -> Result<Vec<String>, String> {
    let (fatal, warnings): (Vec<_>, Vec<_>) = diagnostics
        .iter()
        .partition(|diagnostic| diagnostic.is_fatal());
    if fatal.is_empty() {
        return Ok(warnings
            .into_iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect());
    }
    let dangling = fatal.iter().any(|diagnostic| {
        matches!(
            diagnostic.kind,
            DiagnosticKind::UnresolvedImport { .. } | DiagnosticKind::NodeBuiltinInBrowser { .. }
        )
    });
    let unparsed = fatal
        .iter()
        .any(|diagnostic| matches!(diagnostic.kind, DiagnosticKind::Source { .. }));
    let consequence = match (dangling, unparsed) {
        (true, true) => {
            "An artifact missing code diffpack could not compile, with dangling references to \
             the rest, would crash at runtime"
        }
        (true, false) => "An artifact with dangling references would crash at runtime",
        (false, _) => "The emitted code would not match the source",
    };
    let mut message = format!(
        "{context}: {} fatal build diagnostic(s). {consequence}, so no output was written.",
        fatal.len()
    );
    for diagnostic in fatal {
        message.push_str("\n\n  ");
        message.push_str(&diagnostic.message.replace('\n', "\n  "));
    }
    Err(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warnings_are_returned_without_failing_the_build() {
        let diagnostic = Diagnostic {
            kind: DiagnosticKind::SideEffectsGlob,
            message: "kept conservatively".into(),
        };
        assert_eq!(
            partition_diagnostics(&[diagnostic], "build").unwrap(),
            ["kept conservatively"]
        );
    }

    #[test]
    fn every_fatal_diagnostic_is_reported() {
        let diagnostics = ["first", "second"].map(|message| Diagnostic {
            kind: DiagnosticKind::Source { fatal: true },
            message: message.into(),
        });
        let error = partition_diagnostics(&diagnostics, "client build").unwrap_err();
        assert!(error.contains("2 fatal build diagnostic(s)"));
        assert!(error.contains("first"));
        assert!(error.contains("second"));
    }
}
