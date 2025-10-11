use anyhow::{anyhow, Result};
use git2::Repository;
use std::path::{Path, PathBuf};

/// Wrapper around git repository operations
pub struct GitRepo {
    repo: Repository,
}

impl GitRepo {
    /// Open a git repository at the given path
    #[allow(dead_code)]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let repo = Repository::open(path)?;
        Ok(Self { repo })
    }

    /// Discover and open the repository containing the given path
    pub fn discover<P: AsRef<Path>>(path: P) -> Result<Self> {
        let repo = Repository::discover(path)?;
        Ok(Self { repo })
    }

    /// Get the current HEAD commit hash
    pub fn current_commit_hash(&self) -> Result<String> {
        let head = self.repo.head()?;
        let commit = head.peel_to_commit()?;
        Ok(commit.id().to_string())
    }

    /// Get the repository root path
    pub fn root_path(&self) -> Result<PathBuf> {
        self.repo
            .workdir()
            .ok_or_else(|| anyhow!("Repository has no working directory"))
            .map(|p| p.to_path_buf())
    }

    /// Get a relative path from the repository root
    pub fn relative_path<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let root = self.root_path()?;
        let abs_path = path.as_ref().canonicalize()?;
        let rel_path = abs_path
            .strip_prefix(&root)
            .map_err(|_| anyhow!("Path is not within repository"))?;
        Ok(rel_path.to_string_lossy().to_string())
    }

    /// Get file content at a specific commit
    pub fn file_content_at_commit(&self, commit_hash: &str, file_path: &str) -> Result<String> {
        let oid = git2::Oid::from_str(commit_hash)?;
        let commit = self.repo.find_commit(oid)?;
        let tree = commit.tree()?;

        let entry = tree
            .get_path(Path::new(file_path))
            .map_err(|_| anyhow!("File not found in commit: {}", file_path))?;

        let object = entry.to_object(&self.repo)?;
        let blob = object
            .as_blob()
            .ok_or_else(|| anyhow!("Object is not a blob"))?;

        let content = std::str::from_utf8(blob.content())
            .map_err(|_| anyhow!("File content is not valid UTF-8"))?;

        Ok(content.to_string())
    }

    /// Get list of commits between two commits
    pub fn commits_between(&self, from: &str, to: &str) -> Result<Vec<String>> {
        let from_oid = git2::Oid::from_str(from)?;
        let to_oid = git2::Oid::from_str(to)?;

        let mut revwalk = self.repo.revwalk()?;
        revwalk.push(to_oid)?;
        revwalk.hide(from_oid)?;

        let mut commits = Vec::new();
        for oid in revwalk {
            let oid = oid?;
            commits.push(oid.to_string());
        }

        Ok(commits)
    }

    /// Check if a file changed between two commits
    pub fn file_changed_between(&self, from: &str, to: &str, file_path: &str) -> Result<bool> {
        let from_oid = git2::Oid::from_str(from)?;
        let to_oid = git2::Oid::from_str(to)?;

        let from_commit = self.repo.find_commit(from_oid)?;
        let to_commit = self.repo.find_commit(to_oid)?;

        let from_tree = from_commit.tree()?;
        let to_tree = to_commit.tree()?;

        let diff = self
            .repo
            .diff_tree_to_tree(Some(&from_tree), Some(&to_tree), None)?;

        let path = Path::new(file_path);
        for delta in diff.deltas() {
            // Check both old and new file paths to catch modifications, deletions, and additions
            let old_path_matches = delta.old_file().path() == Some(path);
            let new_path_matches = delta.new_file().path() == Some(path);

            if old_path_matches || new_path_matches {
                return Ok(true);
            }
        }

        Ok(false)
    }
}
