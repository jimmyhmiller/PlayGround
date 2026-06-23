//! `memscope-symbols` — turns the type-erased allocation sites captured by
//! `memscope-core` into concrete, JVM-style type information by reading the
//! running binary's own DWARF.
//!
//! Pipeline (see `project-rust-memory-allocator`):
//! 1. Symbolicate each captured return address (via the `backtrace` crate,
//!    which transparently handles ASLR) into a mangled linkage name + a
//!    demangled display name + file/line.
//! 2. Join the mangled linkage name to the binary's DWARF index, recovering the
//!    monomorphized function's concrete `DW_TAG_template_type_parameter`s.
//! 3. Run the allocation-shape recognizer over the frame chain to pick the
//!    semantic container (`Box`/`Vec`/`Rc`/…) and its element type.
//!
//! The result is folded back into the [`Snapshot`]/[`SiteInfo`] wire types so a
//! consumer sees `Vec<u64>`, `Box<Widget>`, etc. — not raw addresses.

mod dwarf;
mod load;
mod recognizer;

use std::collections::HashMap;
use std::error::Error;
use std::os::raw::c_void;

use memscope_proto::{Frame, SiteInfo, Snapshot, TypeId, TypeInfo};

pub use dwarf::{DwarfIndex, FnTypeInfo};
pub use load::{dwarf_bytes_for, dwarf_bytes_for_current_exe};

type DynErr = Box<dyn Error + Send + Sync>;

/// Resolves allocation sites to concrete types using a binary's DWARF.
pub struct TypeOracle {
    index: DwarfIndex,
}

/// A symbolicated sub-frame (one IP can expand into several inlined frames).
struct SymFrame {
    ip: u64,
    mangled: Option<String>,
    demangled: Option<String>,
    file: Option<String>,
    line: Option<u32>,
    inlined: bool,
}

impl TypeOracle {
    /// Build an oracle from the current process's binary + DWARF.
    pub fn for_current_process() -> Result<Self, DynErr> {
        let bytes = load::dwarf_bytes_for_current_exe()?;
        let index = dwarf::build_index(&bytes)?;
        Ok(TypeOracle { index })
    }

    /// Build an oracle from a specific binary path (for posthoc exploration of a
    /// heap dump produced by another process).
    pub fn from_binary(path: &std::path::Path) -> Result<Self, DynErr> {
        let bytes = load::dwarf_bytes_for(path)?;
        let index = dwarf::build_index(&bytes)?;
        Ok(TypeOracle { index })
    }

    /// Number of monomorphized functions indexed (diagnostic).
    pub fn indexed_functions(&self) -> usize {
        self.index.len()
    }

    /// Symbolicate one return address into its (possibly inlined) sub-frames.
    fn symbolicate(ip: u64) -> Vec<SymFrame> {
        let mut frames = Vec::new();
        backtrace::resolve(ip as *mut c_void, |sym| {
            let (mangled, demangled) = match sym.name() {
                Some(n) => (n.as_str().map(|s| s.to_string()), Some(format!("{n}"))),
                None => (None, None),
            };
            frames.push(SymFrame {
                ip,
                mangled,
                demangled,
                file: sym.filename().map(|p| p.display().to_string()),
                line: sym.lineno(),
                inlined: false,
            });
        });
        // backtrace yields innermost (most-inlined) first; mark all but the last
        // physical frame as inlined.
        let n = frames.len();
        for (i, f) in frames.iter_mut().enumerate() {
            f.inlined = i + 1 < n;
        }
        frames
    }

    /// Resolve a raw site (list of return addresses) into display frames plus
    /// the recovered (element type, shape). Used both for live streaming and
    /// snapshot enrichment.
    pub fn resolve_site_ips(
        &self,
        ips: &[u64],
    ) -> (Vec<Frame>, Option<String>, Option<memscope_proto::AllocShape>) {
        // Expand every ip into sub-frames once; reuse for display + recognizer.
        let mut sym_frames: Vec<SymFrame> = Vec::new();
        for &ip in ips {
            let subs = Self::symbolicate(ip);
            if subs.is_empty() {
                sym_frames.push(SymFrame {
                    ip,
                    mangled: None,
                    demangled: None,
                    file: None,
                    line: None,
                    inlined: false,
                });
            } else {
                sym_frames.extend(subs);
            }
        }

        // Recognizer input: demangled name + the joined DWARF type info.
        let infos: Vec<Option<&FnTypeInfo>> = sym_frames
            .iter()
            .map(|f| f.mangled.as_deref().and_then(|m| self.index.lookup(m)))
            .collect();
        let rframes: Vec<recognizer::ResolvedFrame> = sym_frames
            .iter()
            .zip(infos.iter())
            .map(|(f, info)| recognizer::ResolvedFrame {
                fn_name: f.demangled.as_deref(),
                info: *info,
            })
            .collect();
        let recognized = recognizer::recognize(&rframes);

        let display: Vec<Frame> = sym_frames
            .iter()
            .map(|f| Frame {
                ip: f.ip,
                function: f.demangled.clone(),
                file: f.file.clone(),
                line: f.line,
                inlined: f.inlined,
            })
            .collect();

        (display, recognized.element_type, recognized.shape)
    }

    /// Enrich a snapshot in place: fill every site's frames with symbol/line
    /// info, recover its type + shape, and build the snapshot's type table.
    pub fn resolve_snapshot(&self, snap: &mut Snapshot) {
        let mut type_ids: HashMap<String, u32> = HashMap::new();
        let mut types: Vec<TypeInfo> = Vec::new();

        for site in &mut snap.sites {
            let ips: Vec<u64> = site.frames.iter().map(|f| f.ip).collect();
            let (frames, elem, shape) = self.resolve_site_ips(&ips);
            site.frames = frames;
            site.shape = shape;
            site.ty = match elem {
                Some(name) => {
                    let id = *type_ids.entry(name.clone()).or_insert_with(|| {
                        let id = types.len() as u32;
                        types.push(TypeInfo {
                            id,
                            name,
                            size: None,
                        });
                        id
                    });
                    TypeId(id)
                }
                None => TypeId::UNKNOWN,
            };
        }

        snap.types = types;
    }

    /// A readable per-site label without the full frame list.
    pub fn site_label(&self, site: &SiteInfo) -> String {
        let ips: Vec<u64> = site.frames.iter().map(|f| f.ip).collect();
        let (_frames, elem, shape) = self.resolve_site_ips(&ips);
        match (shape, elem) {
            (Some(shape), Some(ty)) => format!("{shape:?}<{ty}>"),
            (Some(shape), None) => format!("{shape:?}<?>"),
            (None, Some(ty)) => ty,
            (None, None) => "<unknown>".to_string(),
        }
    }
}
