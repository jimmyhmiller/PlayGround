use anyhow::{Context, Result};
use globset::{Glob, GlobBuilder, GlobSet, GlobSetBuilder};

use super::filetypes::DEFAULT_FILETYPES;

#[derive(Debug)]
pub struct FileFilter {
    allow_all_filetypes: bool,
    filetype_matcher: GlobSet,
    only_matcher: Option<GlobSet>,
    ignore_matcher: Option<GlobSet>,
}

impl FileFilter {
    pub fn new(
        allow_all_filetypes: bool,
        only_patterns: &[String],
        ignore_patterns: &[String],
    ) -> Result<Self> {
        let only_patterns = normalize_only_patterns(only_patterns, ignore_patterns);
        let only_matcher = build_globset_from_owned(&only_patterns)?;
        let ignore_matcher = build_globset_from_owned(ignore_patterns)?;
        let filetype_matcher = build_globset_from_static(DEFAULT_FILETYPES)?;

        Ok(Self {
            allow_all_filetypes,
            filetype_matcher,
            only_matcher,
            ignore_matcher,
        })
    }

    pub fn matches(&self, path: &str, file_name: &str) -> bool {
        if let Some(only) = &self.only_matcher {
            if !only.is_match(path) {
                return false;
            }
        }
        if let Some(ignore) = &self.ignore_matcher {
            if ignore.is_match(path) {
                return false;
            }
        }
        if self.allow_all_filetypes {
            return true;
        }
        self.filetype_matcher.is_match(file_name)
    }
}

fn normalize_only_patterns(only: &[String], ignore: &[String]) -> Vec<String> {
    if only.is_empty() && !ignore.is_empty() {
        return vec!["**".to_string()];
    }
    only.to_vec()
}

fn build_glob(pattern: &str) -> Result<Glob> {
    GlobBuilder::new(pattern)
        .literal_separator(false)
        .backslash_escape(true)
        .build()
        .with_context(|| format!("Invalid glob pattern: {pattern}"))
}

fn build_globset_from_owned(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(build_glob(pattern)?);
    }
    Ok(Some(builder.build()?))
}

fn build_globset_from_static(patterns: &[&'static str]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(build_glob(pattern)?);
    }
    builder.build().context("Failed to build filetype matcher")
}
