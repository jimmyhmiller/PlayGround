use anyhow::Result;
use fuser::{
    FileAttr, FileType, Filesystem, MountOption, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry,
    Request,
};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::db::Database;
use crate::view::ViewEngine;

const TTL: Duration = Duration::from_secs(1);
const ROOT_INODE: u64 = 1;
const VIEWS_DIR_INODE: u64 = 2;
const ALL_DIR_INODE: u64 = 3;
const FIRST_VIEW_INODE: u64 = 1000;
const FIRST_FILE_INODE: u64 = 1_000_000;

pub struct OrganizedFS {
    db: Database,
    view_engine: ViewEngine,
    inode_map: HashMap<u64, InodeEntry>,
    path_to_inode: HashMap<PathBuf, u64>,
    next_inode: u64,
}

#[derive(Debug, Clone)]
enum InodeEntry {
    Root,
    ViewsDir,
    AllDir,
    View { id: i64, name: String },
    File { id: i64, real_path: PathBuf },
}

impl OrganizedFS {
    pub fn new(db: Database, view_engine: ViewEngine) -> Result<Self> {
        let mut fs = Self {
            db,
            view_engine,
            inode_map: HashMap::new(),
            path_to_inode: HashMap::new(),
            next_inode: FIRST_VIEW_INODE,
        };

        // Initialize root structure
        fs.inode_map.insert(ROOT_INODE, InodeEntry::Root);
        fs.inode_map.insert(VIEWS_DIR_INODE, InodeEntry::ViewsDir);
        fs.inode_map.insert(ALL_DIR_INODE, InodeEntry::AllDir);

        Ok(fs)
    }

    pub fn mount<P: AsRef<Path>>(self, mountpoint: P) -> Result<()> {
        let options = vec![
            MountOption::RO,
            MountOption::FSName("llm-organizer".to_string()),
        ];

        log::info!("Mounting filesystem at: {}", mountpoint.as_ref().display());

        fuser::mount2(self, mountpoint, &options)?;

        Ok(())
    }

    fn get_or_create_inode(&mut self, entry: InodeEntry) -> u64 {
        // Check if we already have this entry
        for (&inode, existing_entry) in &self.inode_map {
            if Self::entries_match(&entry, existing_entry) {
                return inode;
            }
        }

        // Create new inode
        let inode = self.next_inode;
        self.next_inode += 1;
        self.inode_map.insert(inode, entry);
        inode
    }

    fn entries_match(a: &InodeEntry, b: &InodeEntry) -> bool {
        match (a, b) {
            (InodeEntry::Root, InodeEntry::Root) => true,
            (InodeEntry::ViewsDir, InodeEntry::ViewsDir) => true,
            (InodeEntry::AllDir, InodeEntry::AllDir) => true,
            (InodeEntry::View { id: id1, .. }, InodeEntry::View { id: id2, .. }) => id1 == id2,
            (InodeEntry::File { id: id1, .. }, InodeEntry::File { id: id2, .. }) => id1 == id2,
            _ => false,
        }
    }

    fn get_file_attr(&self, inode: u64, entry: &InodeEntry) -> Option<FileAttr> {
        let now = SystemTime::now();

        match entry {
            InodeEntry::Root | InodeEntry::ViewsDir | InodeEntry::AllDir | InodeEntry::View { .. } => {
                Some(FileAttr {
                    ino: inode,
                    size: 0,
                    blocks: 0,
                    atime: now,
                    mtime: now,
                    ctime: now,
                    crtime: now,
                    kind: FileType::Directory,
                    perm: 0o755,
                    nlink: 2,
                    uid: 501,
                    gid: 20,
                    rdev: 0,
                    blksize: 512,
                    flags: 0,
                })
            }
            InodeEntry::File { real_path, .. } => {
                if let Ok(metadata) = std::fs::metadata(real_path) {
                    let mtime = metadata.modified().unwrap_or(now);
                    let atime = metadata.accessed().unwrap_or(now);

                    Some(FileAttr {
                        ino: inode,
                        size: metadata.len(),
                        blocks: (metadata.len() + 511) / 512,
                        atime,
                        mtime,
                        ctime: mtime,
                        crtime: mtime,
                        kind: FileType::RegularFile,
                        perm: 0o444,
                        nlink: 1,
                        uid: 501,
                        gid: 20,
                        rdev: 0,
                        blksize: 512,
                        flags: 0,
                    })
                } else {
                    None
                }
            }
        }
    }
}

impl Filesystem for OrganizedFS {
    fn lookup(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEntry) {
        log::debug!("lookup: parent={}, name={:?}", parent, name);

        let name_str = match name.to_str() {
            Some(n) => n,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let parent_entry = match self.inode_map.get(&parent) {
            Some(e) => e.clone(),
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        match parent_entry {
            InodeEntry::Root => {
                // Root contains "views" and "all" directories
                let inode = match name_str {
                    "views" => VIEWS_DIR_INODE,
                    "all" => ALL_DIR_INODE,
                    _ => {
                        reply.error(libc::ENOENT);
                        return;
                    }
                };

                if let Some(entry) = self.inode_map.get(&inode) {
                    if let Some(attr) = self.get_file_attr(inode, entry) {
                        reply.entry(&TTL, &attr, 0);
                        return;
                    }
                }
                reply.error(libc::ENOENT);
            }
            InodeEntry::ViewsDir => {
                // Views directory contains view subdirectories
                if let Ok(views) = self.view_engine.get_all_views() {
                    for view in views {
                        if view.name == name_str {
                            let entry = InodeEntry::View {
                                id: view.id,
                                name: view.name.clone(),
                            };
                            let inode = self.get_or_create_inode(entry.clone());

                            if let Some(attr) = self.get_file_attr(inode, &entry) {
                                reply.entry(&TTL, &attr, 0);
                                return;
                            }
                        }
                    }
                }
                reply.error(libc::ENOENT);
            }
            InodeEntry::View { id, .. } => {
                // View directory contains files
                let files = self.view_engine.get_view_files(id).unwrap_or_default();

                for file in files {
                    let file_name = Path::new(&file.path).file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");

                    if file_name == name_str {
                        let entry = InodeEntry::File {
                            id: file.id,
                            real_path: PathBuf::from(&file.path),
                        };
                        let inode = self.get_or_create_inode(entry.clone());

                        if let Some(attr) = self.get_file_attr(inode, &entry) {
                            reply.entry(&TTL, &attr, 0);
                            return;
                        }
                    }
                }
                reply.error(libc::ENOENT);
            }
            InodeEntry::AllDir => {
                // All directory contains files
                let files = self.db.get_all_files().unwrap_or_default();

                for file in files {
                    let file_name = Path::new(&file.path).file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");

                    if file_name == name_str {
                        let entry = InodeEntry::File {
                            id: file.id,
                            real_path: PathBuf::from(&file.path),
                        };
                        let inode = self.get_or_create_inode(entry.clone());

                        if let Some(attr) = self.get_file_attr(inode, &entry) {
                            reply.entry(&TTL, &attr, 0);
                            return;
                        }
                    }
                }
                reply.error(libc::ENOENT);
            }
            InodeEntry::File { .. } => {
                reply.error(libc::ENOTDIR);
            }
        }
    }

    fn getattr(&mut self, _req: &Request, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        log::debug!("getattr: ino={}", ino);

        if let Some(entry) = self.inode_map.get(&ino).cloned() {
            if let Some(attr) = self.get_file_attr(ino, &entry) {
                reply.attr(&TTL, &attr);
                return;
            }
        }

        reply.error(libc::ENOENT);
    }

    fn read(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        log::debug!("read: ino={}, offset={}, size={}", ino, offset, size);

        if let Some(InodeEntry::File { real_path, .. }) = self.inode_map.get(&ino) {
            match std::fs::read(real_path) {
                Ok(data) => {
                    let start = offset as usize;
                    let end = std::cmp::min(start + size as usize, data.len());

                    if start < data.len() {
                        reply.data(&data[start..end]);
                    } else {
                        reply.data(&[]);
                    }
                }
                Err(e) => {
                    log::error!("Failed to read file: {}", e);
                    reply.error(libc::EIO);
                }
            }
        } else {
            reply.error(libc::EISDIR);
        }
    }

    fn readdir(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        log::debug!("readdir: ino={}, offset={}", ino, offset);

        let entry = match self.inode_map.get(&ino).cloned() {
            Some(e) => e,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let mut entries: Vec<(u64, FileType, String)> = vec![
            (ino, FileType::Directory, ".".to_string()),
            (ino, FileType::Directory, "..".to_string()),
        ];

        match entry {
            InodeEntry::Root => {
                entries.push((VIEWS_DIR_INODE, FileType::Directory, "views".to_string()));
                entries.push((ALL_DIR_INODE, FileType::Directory, "all".to_string()));
            }
            InodeEntry::ViewsDir => {
                if let Ok(views) = self.view_engine.get_all_views() {
                    for view in views {
                        let entry = InodeEntry::View {
                            id: view.id,
                            name: view.name.clone(),
                        };
                        let inode = self.get_or_create_inode(entry);
                        entries.push((inode, FileType::Directory, view.name));
                    }
                }
            }
            InodeEntry::View { id, .. } => {
                if let Ok(files) = self.view_engine.get_view_files(id) {
                    for file in files {
                        let file_name = Path::new(&file.path).file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("")
                            .to_string();

                        let entry = InodeEntry::File {
                            id: file.id,
                            real_path: PathBuf::from(&file.path),
                        };
                        let inode = self.get_or_create_inode(entry);
                        entries.push((inode, FileType::RegularFile, file_name));
                    }
                }
            }
            InodeEntry::AllDir => {
                if let Ok(files) = self.db.get_all_files() {
                    for file in files {
                        let file_name = Path::new(&file.path).file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("")
                            .to_string();

                        let entry = InodeEntry::File {
                            id: file.id,
                            real_path: PathBuf::from(&file.path),
                        };
                        let inode = self.get_or_create_inode(entry);
                        entries.push((inode, FileType::RegularFile, file_name));
                    }
                }
            }
            InodeEntry::File { .. } => {
                reply.error(libc::ENOTDIR);
                return;
            }
        }

        for (i, entry) in entries.iter().enumerate().skip(offset as usize) {
            if reply.add(entry.0, (i + 1) as i64, entry.1, &entry.2) {
                break;
            }
        }

        reply.ok();
    }
}
