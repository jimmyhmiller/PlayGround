use crate::db::Database;
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

/// Represents a parsed virtual filesystem path
#[derive(Debug, Clone, PartialEq)]
pub enum VirtualPath {
    /// Root directory
    Root,
    /// /views directory
    ViewsDir,
    /// /views/<view_name>
    ViewDir(String),
    /// /views/<view_name>/<file_path>
    ViewFile { view_name: String, file_path: PathBuf },
    /// /all directory
    AllDir,
    /// /all/<file_path>
    AllFile(PathBuf),
}

impl VirtualPath {
    /// Parse a virtual path string
    pub fn parse(path: &str) -> Result<Self> {
        let path = path.trim_start_matches('/');

        if path.is_empty() || path == "." {
            return Ok(VirtualPath::Root);
        }

        let parts: Vec<&str> = path.split('/').collect();

        match parts[0] {
            "views" => {
                if parts.len() == 1 {
                    Ok(VirtualPath::ViewsDir)
                } else if parts.len() == 2 {
                    Ok(VirtualPath::ViewDir(parts[1].to_string()))
                } else {
                    // parts.len() > 2
                    let view_name = parts[1].to_string();
                    let file_path: PathBuf = parts[2..].iter().collect();
                    Ok(VirtualPath::ViewFile { view_name, file_path })
                }
            }
            "all" => {
                if parts.len() == 1 {
                    Ok(VirtualPath::AllDir)
                } else {
                    let file_path: PathBuf = parts[1..].iter().collect();
                    Ok(VirtualPath::AllFile(file_path))
                }
            }
            _ => Err(anyhow!("Invalid virtual path: must start with /views or /all")),
        }
    }
}

/// Entry in a virtual directory listing
#[derive(Debug, Clone)]
pub enum VirtualEntry {
    Directory(String),
    File { name: String, size: u64, real_path: PathBuf },
}

impl VirtualEntry {
    pub fn name(&self) -> &str {
        match self {
            VirtualEntry::Directory(name) => name,
            VirtualEntry::File { name, .. } => name,
        }
    }

    pub fn is_dir(&self) -> bool {
        matches!(self, VirtualEntry::Directory(_))
    }
}

/// Virtual filesystem navigator
pub struct VirtualFilesystem {
    db: Database,
}

impl VirtualFilesystem {
    pub fn new(db: Database) -> Self {
        Self { db }
    }

    /// List entries at a virtual path
    pub fn list(&self, vpath: &VirtualPath) -> Result<Vec<VirtualEntry>> {
        match vpath {
            VirtualPath::Root => {
                Ok(vec![
                    VirtualEntry::Directory("views".to_string()),
                    VirtualEntry::Directory("all".to_string()),
                ])
            }
            VirtualPath::ViewsDir => {
                let views = self.db.get_all_views()?;
                Ok(views
                    .into_iter()
                    .map(|v| VirtualEntry::Directory(v.name))
                    .collect())
            }
            VirtualPath::ViewDir(view_name) => {
                // Find the view ID first
                let views = self.db.get_all_views()?;
                let view = views.iter().find(|v| &v.name == view_name)
                    .ok_or_else(|| anyhow!("View not found: {}", view_name))?;

                let files = self.db.get_view_files(view.id)?;
                Ok(files
                    .into_iter()
                    .map(|file| {
                        let name = Path::new(&file.path)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        VirtualEntry::File {
                            name,
                            size: file.size_bytes as u64,
                            real_path: PathBuf::from(&file.path),
                        }
                    })
                    .collect())
            }
            VirtualPath::AllDir => {
                let files = self.db.get_all_files()?;
                Ok(files
                    .into_iter()
                    .map(|file| {
                        let name = Path::new(&file.path)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        VirtualEntry::File {
                            name,
                            size: file.size_bytes as u64,
                            real_path: PathBuf::from(&file.path),
                        }
                    })
                    .collect())
            }
            VirtualPath::ViewFile { view_name, file_path } => {
                // Can't list a file
                Err(anyhow!("Not a directory: /views/{}/{}", view_name, file_path.display()))
            }
            VirtualPath::AllFile(file_path) => {
                // Can't list a file
                Err(anyhow!("Not a directory: /all/{}", file_path.display()))
            }
        }
    }

    /// Get the real filesystem path for a virtual file path
    pub fn get_real_path(&self, vpath: &VirtualPath) -> Result<PathBuf> {
        match vpath {
            VirtualPath::ViewFile { view_name, file_path } => {
                // Find the view ID first
                let views = self.db.get_all_views()?;
                let view = views.iter().find(|v| &v.name == view_name)
                    .ok_or_else(|| anyhow!("View not found: {}", view_name))?;

                let files = self.db.get_view_files(view.id)?;
                let filename = file_path
                    .file_name()
                    .ok_or_else(|| anyhow!("Invalid file path"))?
                    .to_string_lossy();

                for file in files {
                    if Path::new(&file.path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        == filename
                    {
                        return Ok(PathBuf::from(file.path));
                    }
                }
                Err(anyhow!("File not found in view: {}", filename))
            }
            VirtualPath::AllFile(file_path) => {
                let files = self.db.get_all_files()?;
                let filename = file_path
                    .file_name()
                    .ok_or_else(|| anyhow!("Invalid file path"))?
                    .to_string_lossy();

                for file in files {
                    if Path::new(&file.path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        == filename
                    {
                        return Ok(PathBuf::from(file.path));
                    }
                }
                Err(anyhow!("File not found: {}", filename))
            }
            _ => Err(anyhow!("Not a file path")),
        }
    }

    /// Get file metadata for a virtual path
    pub fn stat(&self, vpath: &VirtualPath) -> Result<VirtualStat> {
        match vpath {
            VirtualPath::Root | VirtualPath::ViewsDir | VirtualPath::AllDir => {
                Ok(VirtualStat::Directory)
            }
            VirtualPath::ViewDir(view_name) => {
                // Check if view exists
                let views = self.db.get_all_views()?;
                if views.iter().any(|v| &v.name == view_name) {
                    Ok(VirtualStat::Directory)
                } else {
                    Err(anyhow!("View not found: {}", view_name))
                }
            }
            VirtualPath::ViewFile { .. } | VirtualPath::AllFile(_) => {
                let real_path = self.get_real_path(vpath)?;
                let metadata = std::fs::metadata(&real_path)?;
                Ok(VirtualStat::File {
                    size: metadata.len(),
                    real_path,
                })
            }
        }
    }
}

/// File statistics for a virtual path
#[derive(Debug)]
pub enum VirtualStat {
    Directory,
    File { size: u64, real_path: PathBuf },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_root() {
        assert_eq!(VirtualPath::parse("/").unwrap(), VirtualPath::Root);
        assert_eq!(VirtualPath::parse("").unwrap(), VirtualPath::Root);
        assert_eq!(VirtualPath::parse(".").unwrap(), VirtualPath::Root);
    }

    #[test]
    fn test_parse_views() {
        assert_eq!(VirtualPath::parse("/views").unwrap(), VirtualPath::ViewsDir);
        assert_eq!(
            VirtualPath::parse("/views/work").unwrap(),
            VirtualPath::ViewDir("work".to_string())
        );
        assert_eq!(
            VirtualPath::parse("/views/work/doc.pdf").unwrap(),
            VirtualPath::ViewFile {
                view_name: "work".to_string(),
                file_path: PathBuf::from("doc.pdf")
            }
        );
    }

    #[test]
    fn test_parse_all() {
        assert_eq!(VirtualPath::parse("/all").unwrap(), VirtualPath::AllDir);
        assert_eq!(
            VirtualPath::parse("/all/test.txt").unwrap(),
            VirtualPath::AllFile(PathBuf::from("test.txt"))
        );
    }

    #[test]
    fn test_parse_invalid() {
        assert!(VirtualPath::parse("/invalid").is_err());
        assert!(VirtualPath::parse("/foo/bar").is_err());
    }
}
