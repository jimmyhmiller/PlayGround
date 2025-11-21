//! Compatibility layer for migrating from git2 to gix
//!
//! This module provides wrappers and helper functions to ease the transition

use anyhow::{Context, Result, anyhow};
use gix::{ObjectId, Repository};
use std::path::Path;

/// Open a git repository at the given path
pub fn open_repository(path: &Path) -> Result<Repository> {
    gix::open(path)
        .with_context(|| format!("Failed to open repository at {}", path.display()))
}

/// Convert ObjectId to hex string (like git2's Oid::to_string())
pub fn oid_to_string(oid: &ObjectId) -> String {
    oid.to_hex().to_string()
}

/// Get commit time in seconds since epoch
pub fn commit_time_seconds(repo: &Repository, commit_id: ObjectId) -> Result<i64> {
    let commit = repo.find_object(commit_id)?
        .try_into_commit()?;
    Ok(commit.time()?.seconds)
}

/// Helper to get a reference by name
pub fn find_reference<'a>(repo: &'a Repository, name: &str) -> Result<gix::Reference<'a>> {
    repo.find_reference(name)
        .with_context(|| format!("Failed to find reference {}", name))
}
