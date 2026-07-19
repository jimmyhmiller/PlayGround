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

mod addr_resolve;
mod dwarf;
mod load;
mod recognizer;

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::os::raw::c_void;

use memscope_proto::{Frame, SiteInfo, Snapshot, TypeId, TypeInfo};

pub use dwarf::{DwarfIndex, FnTypeInfo, LayoutIndex, MemReader, PtrField};
pub use addr_resolve::resolve as resolve_raw_sites_targeted;
pub use load::{
    current_image_path, current_image_slide, dwarf_bytes_for, dwarf_bytes_for_current_exe,
};

type DynErr = Box<dyn Error + Send + Sync>;

/// Frames that belong to our own allocation-capture path (the global allocator
/// hook + the unwinder), which prefix every captured site and carry no signal.
fn is_capture_noise(name: &str) -> bool {
    name.starts_with("backtrace::")
        || name.contains("memscope_core::unwind")
        || name.contains("memscope_core::recorder")
        || name.contains("memscope_core::capture")
        || name.contains("as memscope_core::unwind::Unwind")
        || name.contains("as core::alloc::global::GlobalAlloc>::alloc")
        // Innermost-first ordering guarantees the leading run is capture
        // internals, so trimming a leading bare call_once (the capture closure
        // bridge) never eats a user frame (those are outermost = last).
        || name.contains("core::ops::function::FnOnce::call_once")
}

/// Turn a chain of symbolicated sub-frames into display frames + recovered
/// (element type, shape). Shared by the in-process (`backtrace`) path and the
/// read-time (`addr2line`) path — both produce [`SymFrame`]s, only the source of
/// the symbols differs.
fn finish_frames(
    sym_frames: &[SymFrame],
    index: &DwarfIndex,
) -> (Vec<Frame>, Option<String>, Option<memscope_proto::AllocShape>) {
    // Recognizer input: demangled name + the joined DWARF type info.
    let infos: Vec<Option<&FnTypeInfo>> = sym_frames
        .iter()
        .map(|f| f.mangled.as_deref().and_then(|m| index.lookup(m)))
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

    // Drop the leading capture-machinery frames (the allocator hook + unwinder)
    // so a site reads as "where the program allocated", not "how we observed it".
    let start = sym_frames
        .iter()
        .position(|f| {
            f.demangled
                .as_deref()
                .map(|n| !is_capture_noise(n))
                .unwrap_or(true)
        })
        .unwrap_or(0);

    let display: Vec<Frame> = sym_frames[start..]
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

/// One site resolved at read time: display frames + recovered type/shape.
pub struct SiteResolution {
    pub frames: Vec<Frame>,
    pub element_type: Option<String>,
    pub shape: Option<memscope_proto::AllocShape>,
}

/// Two-phase, constant-memory site resolver.
///
/// [`build`](SiteResolver::build) symbolicates each **unique** return address
/// once (bounded by the binary's debug info + the number of distinct IPs, *not*
/// the number of sites). [`resolve_site`](SiteResolver::resolve_site) then
/// assembles any site's frames + recovered type on demand from that shared
/// cache. This lets a caller resolve a recording with a million deeply-nested
/// sites (built from a few thousand IPs) while keeping only the compact result
/// it needs per site — instead of materializing every site's full stack at once.
pub struct SiteResolver {
    ip_frames: HashMap<u64, Vec<SymFrame>>,
    index: DwarfIndex,
}

impl SiteResolver {
    /// Symbolicate the given distinct return addresses once.
    pub fn build(exe: &std::path::Path, slide: u64, unique_ips: &[u64]) -> Result<Self, DynErr> {
        let (ip_frames, index) = addr_resolve::resolve_unique(exe, slide, unique_ips)?;
        Ok(SiteResolver { ip_frames, index })
    }

    /// Number of distinct IPs symbolicated (diagnostic).
    pub fn ip_count(&self) -> usize {
        self.ip_frames.len()
    }

    /// Assemble one site's frames + recovered (element type, shape) from the
    /// pre-resolved per-IP cache. Cheap; allocates only this site's frames.
    pub fn resolve_site(&self, ips: &[u64]) -> SiteResolution {
        let mut sym_frames: Vec<SymFrame> = Vec::new();
        for ip in ips {
            if let Some(fs) = self.ip_frames.get(ip) {
                sym_frames.extend(fs.iter().cloned());
            }
        }
        let (frames, element_type, shape) = finish_frames(&sym_frames, &self.index);
        SiteResolution { frames, element_type, shape }
    }
}

/// Resolve raw allocation sites (interned id -> captured runtime return
/// addresses) against a binary's dSYM — **off the traced process entirely**.
///
/// This is the default entry point. It currently delegates to the addr2line
/// implementation; the constant-memory walker (see `addr_resolve`) will replace
/// the body once it passes the differential tests. The addr2line version is
/// retained as [`resolve_raw_sites_addr2line`] to serve as the test oracle.
pub fn resolve_raw_sites(
    exe: &std::path::Path,
    slide: u64,
    sites: &[(u32, Vec<u64>)],
) -> Result<HashMap<u32, SiteResolution>, DynErr> {
    addr_resolve::resolve(exe, slide, sites)
}

/// addr2line-based resolver (the known-correct reference). Symbolicates each
/// recorded runtime address by mapping it to a static address (`ip - slide`) and
/// querying `addr2line`, then recovers types via the DWARF index. Correct but
/// memory-heavy (retains whole-function inline trees) — kept as the differential
/// test oracle for the constant-memory replacement.
pub fn resolve_raw_sites_addr2line(
    exe: &std::path::Path,
    slide: u64,
    sites: &[(u32, Vec<u64>)],
) -> Result<HashMap<u32, SiteResolution>, DynErr> {
    use std::borrow::Cow;

    use object::{Object, ObjectSection};

    let mmap = load::dwarf_mmap_for(exe)?;
    let bytes: &[u8] = &mmap;

    // Pass 1 — addresses → frames. Symbolicate each *distinct* return address
    // once: the recorded sites are 64-deep stacks that overlap heavily (shared
    // allocator glue + runtime frames), so the same ip recurs across thousands
    // of sites; resolving per unique ip instead of per site-frame collapses
    // millions of `find_frames` calls into tens of thousands.
    //
    // The `addr2line::Context` and the `dwarf::build` type index are each a full
    // pass over this binary's (large) DWARF. Scoping the context here drops it —
    // and its lazily-materialized per-unit caches — before the type index is
    // built in pass 2, so the two never coexist in memory.
    let ip_frames: HashMap<u64, Vec<SymFrame>> = {
        // The distinct addresses to resolve. The recorded sites are 64-deep
        // stacks that overlap heavily, so resolving per unique ip (not per
        // site-frame) collapses ~6M frame refs into ~10k lookups.
        let mut uniq: Vec<u64> = {
            let mut set: HashSet<u64> = HashSet::new();
            for (_, ips) in sites {
                set.extend(ips.iter().copied());
            }
            set.into_iter().collect()
        };
        uniq.sort_unstable();

        let object = object::File::parse(&*bytes)?;
        let endian = if object.is_little_endian() {
            gimli::RunTimeEndian::Little
        } else {
            gimli::RunTimeEndian::Big
        };
        let load_section = |id: gimli::SectionId| -> Result<Cow<[u8]>, gimli::Error> {
            match object.section_by_name(id.name()) {
                Some(ref section) => Ok(section
                    .uncompressed_data()
                    .unwrap_or(Cow::Borrowed(&[][..]))),
                None => Ok(Cow::Borrowed(&[][..])),
            }
        };
        let dwarf_sections = gimli::DwarfSections::load(&load_section)?;

        let dwarf = dwarf_sections.borrow(|section| gimli::EndianSlice::new(&section[..], endian));
        let ctx = addr2line::Context::from_dwarf(dwarf)?;
        let mut cache: HashMap<u64, Vec<SymFrame>> = HashMap::with_capacity(uniq.len());
        for &ip in &uniq {
            cache.insert(ip, symbolicate_addr(&ctx, ip, slide));
        }
        cache
    };

    // Pass 2 — frames → types. Join mangled names to the DWARF type index to
    // recover concrete `T`s (the recognizer turns the frame chain into a shape +
    // element type).
    let (index, _layout) = dwarf::build(bytes)?;
    let mut out = HashMap::with_capacity(sites.len());
    let mut sym_frames: Vec<SymFrame> = Vec::new();
    for (id, ips) in sites {
        sym_frames.clear();
        for ip in ips {
            if let Some(frames) = ip_frames.get(ip) {
                sym_frames.extend(frames.iter().cloned());
            }
        }
        let (frames, element_type, shape) = finish_frames(&sym_frames, &index);
        out.insert(
            *id,
            SiteResolution {
                frames,
                element_type,
                shape,
            },
        );
    }
    Ok(out)
}

/// Symbolicate one runtime return address against `ctx`'s DWARF: map it back to
/// a static address (`ip - slide`, minus one byte to land in the call), expand
/// inlined frames, and return them innermost-first (the real function last).
fn symbolicate_addr<R: gimli::Reader>(
    ctx: &addr2line::Context<R>,
    ip: u64,
    slide: u64,
) -> Vec<SymFrame> {
    let probe = ip.wrapping_sub(slide).wrapping_sub(1);
    let mut frames: Vec<SymFrame> = Vec::new();
    if let Ok(mut iter) = ctx.find_frames(probe).skip_all_loads() {
        while let Ok(Some(frame)) = iter.next() {
            let mangled = frame
                .function
                .as_ref()
                .and_then(|f| f.raw_name().ok().map(|c| c.into_owned()));
            let demangled = frame
                .function
                .as_ref()
                .and_then(|f| f.demangle().ok().map(|c| c.into_owned()));
            let (file, line) = match &frame.location {
                Some(l) => (l.file.map(|s| s.to_string()), l.line),
                None => (None, None),
            };
            frames.push(SymFrame {
                ip,
                mangled,
                demangled,
                file,
                line,
                inlined: false,
            });
        }
        if let Some(last) = frames.len().checked_sub(1) {
            for f in &mut frames[..last] {
                f.inlined = true;
            }
        }
    }
    if frames.is_empty() {
        frames.push(SymFrame {
            ip,
            mangled: None,
            demangled: None,
            file: None,
            line: None,
            inlined: false,
        });
    }
    frames
}

/// Resolves allocation sites to concrete types using a binary's DWARF.
pub struct TypeOracle {
    index: DwarfIndex,
    layout: LayoutIndex,
}

/// A symbolicated sub-frame (one IP can expand into several inlined frames).
#[derive(Clone)]
pub(crate) struct SymFrame {
    pub(crate) ip: u64,
    pub(crate) mangled: Option<String>,
    pub(crate) demangled: Option<String>,
    pub(crate) file: Option<String>,
    pub(crate) line: Option<u32>,
    pub(crate) inlined: bool,
}

impl TypeOracle {
    /// Build an oracle from the current process's binary + DWARF.
    pub fn for_current_process() -> Result<Self, DynErr> {
        let bytes = load::dwarf_bytes_for_current_exe()?;
        let (index, layout) = dwarf::build(&bytes)?;
        Ok(TypeOracle { index, layout })
    }

    /// Build an oracle from a specific binary path (for posthoc exploration of a
    /// heap dump produced by another process).
    pub fn from_binary(path: &std::path::Path) -> Result<Self, DynErr> {
        let bytes = load::dwarf_bytes_for(path)?;
        let (index, layout) = dwarf::build(&bytes)?;
        Ok(TypeOracle { index, layout })
    }

    /// Number of monomorphized functions indexed (diagnostic).
    pub fn indexed_functions(&self) -> usize {
        self.index.len()
    }

    /// The type-layout index (field offsets, pointer fields) for heap-graph
    /// reconstruction.
    pub fn layout(&self) -> &LayoutIndex {
        &self.layout
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

        finish_frames(&sym_frames, &self.index)
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
