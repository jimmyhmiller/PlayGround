//! `Coil.toml` — a small build manifest, so a project's entry point, output name,
//! target, and (especially) its linker needs live in one file instead of a wall of
//! `--link-flag`s on every invocation. A thin front-end over the existing compiler:
//! `coil build` / `coil run` with no file argument read `./Coil.toml`, expand it
//! into the flags the CLI already accepts, and drive the normal pipeline.
//!
//! ```toml
//! [package]
//! name  = "chip8-gui"
//! entry = "src/main.coil"      # default: src/main.coil
//!
//! [build]
//! # out    = "chip8-gui"       # default: package name
//! # target = "<triple>"        # default: host
//! # debug  = false
//!
//! [link]
//! frameworks = ["AppKit", "CoreGraphics"]   # -> -framework X
//! libs       = ["objc"]                      # -> -lX
//! flags      = []                            # raw passthrough escape hatch
//! ```
//!
//! Deliberately tiny: no registry, no lockfile, no dependency resolution (Coil's
//! `import` already resolves files by relative path). Those are clean later layers.

use serde::Deserialize;
use std::path::Path;

pub const MANIFEST: &str = "Coil.toml";

#[derive(Deserialize, Debug)]
pub struct Manifest {
    pub package: Package,
    #[serde(default)]
    pub build: Build,
    #[serde(default)]
    pub link: Link,
}

#[derive(Deserialize, Debug)]
pub struct Package {
    pub name: String,
    pub entry: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
pub struct Build {
    pub out: Option<String>,
    pub target: Option<String>,
    #[serde(default)]
    pub debug: bool,
}

#[derive(Deserialize, Debug, Default)]
pub struct Link {
    #[serde(default)]
    pub frameworks: Vec<String>,
    #[serde(default)]
    pub libs: Vec<String>,
    #[serde(default)]
    pub flags: Vec<String>,
}

impl Manifest {
    /// Read + parse `./Coil.toml` (or the given path).
    pub fn load(path: &Path) -> Result<Manifest, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The entry source file (default `src/main.coil`).
    pub fn entry(&self) -> String {
        self.package.entry.clone().unwrap_or_else(|| "src/main.coil".to_string())
    }

    /// The output executable name (default: the package name).
    pub fn out(&self) -> String {
        self.build.out.clone().unwrap_or_else(|| self.package.name.clone())
    }

    /// The `[link]` block expanded into `cc` arguments, in the same form the CLI's
    /// `--link-flag` produces: `frameworks` → `-framework X`, `libs` → `-lX`, then
    /// any raw `flags` verbatim.
    pub fn link_flags(&self) -> Vec<String> {
        let mut out = Vec::new();
        for fw in &self.link.frameworks {
            out.push("-framework".to_string());
            out.push(fw.clone());
        }
        for lib in &self.link.libs {
            out.push(format!("-l{lib}"));
        }
        out.extend(self.link.flags.iter().cloned());
        out
    }
}
