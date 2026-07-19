//! Constant-memory address symbolication.
//!
//! See `CONSTANT_MEMORY_SYMBOLICATION_PLAN.md`. The invariant: peak memory is
//! bounded by the binary's debug info (one parsed unit + a compact function
//! table) and the per-probe inline depth — **independent of how many addresses
//! are resolved**, and never materializing a function's whole inline tree.
//!
//! 1. mmap the dSYM; build a sorted `addr → function` table once (a few MB).
//! 2. Sort probes; sweep in address order holding **at most one parsed unit +
//!    its line index** at a time.
//! 3. Per probe: binary-search the table, walk only the inline-subroutine path
//!    covering the probe (names + call sites + template params), then drop.
//!
//! Output matches the addr2line oracle: a `Vec<SymFrame>` per ip (innermost
//! first), plus per-site `(frames, element_type, shape)` via `finish_frames`.

use std::collections::{HashMap, HashSet};

use gimli::{AttributeValue, EndianSlice, RunTimeEndian, UnitOffset};

use crate::dwarf::{DwarfIndex, FnTypeInfo};
use crate::{DynErr, SiteResolution, SymFrame};

type Reader<'a> = EndianSlice<'a, RunTimeEndian>;
type Die<'a, 'u> = gimli::DebuggingInformationEntry<'a, 'u, Reader<'a>>;
type Header<'a> = gimli::UnitHeader<Reader<'a>>;

/// One function's `[low, high)` PC range, its unit, and its subprogram DIE.
struct FuncEntry {
    low: u64,
    high: u64,
    unit_idx: usize,
    die: UnitOffset,
}

/// One frame collected during the walk (outermost-first); `call_*` is the call
/// site of this frame's inlined child, used as this frame's displayed location.
struct Frame {
    mangled: Option<String>,
    demangled: Option<String>,
    call_file: Option<String>,
    call_line: Option<u32>,
}

/// Resolve raw sites with bounded memory. Drop-in for `resolve_raw_sites`.
///
/// NOTE: this materializes full frames for *every* site. For recordings with
/// huge numbers of sites built from few unique IPs (e.g. a million 58-deep
/// stacks from 25k IPs), prefer [`crate::SiteResolver`], which resolves the
/// unique IPs once and lets the caller keep only a compact per-site result.
pub fn resolve(
    exe: &std::path::Path,
    slide: u64,
    sites: &[(u32, Vec<u64>)],
) -> Result<HashMap<u32, SiteResolution>, DynErr> {
    // Phase 1: resolve each distinct IP once (bounded).
    let unique: Vec<u64> = {
        let mut set: HashSet<u64> = HashSet::new();
        for (_, ips) in sites {
            set.extend(ips.iter().copied());
        }
        set.into_iter().collect()
    };
    let (ip_frames, index) = resolve_unique(exe, slide, &unique)?;

    // Phase 2: assemble per-site frames (the part that scales with site count).
    let mut out = HashMap::with_capacity(sites.len());
    let mut sym_frames: Vec<SymFrame> = Vec::new();
    for (id, ips) in sites {
        sym_frames.clear();
        for ip in ips {
            if let Some(fs) = ip_frames.get(ip) {
                sym_frames.extend(fs.iter().cloned());
            }
        }
        let (frames, element_type, shape) = crate::finish_frames(&sym_frames, &index);
        out.insert(*id, SiteResolution { frames, element_type, shape });
    }
    Ok(out)
}

/// Phase 1 of resolution: symbolicate each distinct return address exactly once,
/// holding at most one parsed unit + its line index at a time. Returns the
/// per-IP frames and the (small) type index for the touched functions. Memory is
/// bounded by the binary's debug info + the number of *unique* IPs — independent
/// of how many sites reference them.
pub(crate) fn resolve_unique(
    exe: &std::path::Path,
    slide: u64,
    unique_ips: &[u64],
) -> Result<(HashMap<u64, Vec<SymFrame>>, DwarfIndex), DynErr> {
    use std::borrow::Cow;

    use object::{Object, ObjectSection};

    let mmap = crate::load::dwarf_mmap_for(exe)?;
    let object = object::File::parse(&*mmap)?;
    let endian = if object.is_little_endian() {
        RunTimeEndian::Little
    } else {
        RunTimeEndian::Big
    };
    let load_section = |id: gimli::SectionId| -> Result<Cow<[u8]>, gimli::Error> {
        match object.section_by_name(id.name()) {
            Some(ref s) => Ok(s.uncompressed_data().unwrap_or(Cow::Borrowed(&[][..]))),
            None => Ok(Cow::Borrowed(&[][..])),
        }
    };
    let dwarf_sections = gimli::DwarfSections::load(&load_section)?;
    let dwarf = dwarf_sections.borrow(|s| EndianSlice::new(&s[..], endian));

    // Unit headers (cheap; borrow the mmap), indexed so the function table can
    // name a unit without holding it parsed.
    let mut headers = Vec::new();
    {
        let mut it = dwarf.units();
        while let Some(h) = it.next()? {
            headers.push(h);
        }
    }

    // Pass A — address→function table. Parse each unit transiently (dropped per
    // iteration); record subprogram ranges only.
    let mut funcs: Vec<FuncEntry> = Vec::new();
    for (unit_idx, header) in headers.iter().enumerate() {
        let Ok(unit) = dwarf.unit(*header) else { continue };
        let mut entries = unit.entries();
        while let Ok(Some((_, entry))) = entries.next_dfs() {
            if entry.tag() == gimli::DW_TAG_subprogram {
                if let Some((low, high)) = entry_pc(&dwarf, &unit, entry) {
                    funcs.push(FuncEntry { low, high, unit_idx, die: entry.offset() });
                }
            }
        }
    }
    funcs.sort_unstable_by_key(|f| f.low);

    // Probes (static address, original ip), sorted by address for the sweep.
    let mut probes: Vec<(u64, u64)> = unique_ips
        .iter()
        .map(|&ip| (ip.wrapping_sub(slide).wrapping_sub(1), ip))
        .collect();
    probes.sort_unstable_by_key(|p| p.0);

    let mut ip_frames: HashMap<u64, Vec<SymFrame>> = HashMap::with_capacity(probes.len());
    // Type index for ONLY the functions we resolve (not all of them).
    let mut index = DwarfIndex { by_linkage: HashMap::new() };

    // At most one parsed unit + its line index live at a time.
    let mut cur_idx: Option<usize> = None;
    let mut cur_unit: Option<gimli::Unit<Reader>> = None;
    let mut cur_lines: Option<LineIndex> = None;

    for &(probe, ip) in &probes {
        let fi = funcs.partition_point(|f| f.low <= probe);
        if fi == 0 || probe >= funcs[fi - 1].high {
            ip_frames.insert(ip, vec![unknown_frame(ip)]);
            continue;
        }
        let unit_idx = funcs[fi - 1].unit_idx;
        let die = funcs[fi - 1].die;

        if cur_idx != Some(unit_idx) {
            match dwarf.unit(headers[unit_idx]) {
                Ok(u) => {
                    cur_lines = Some(LineIndex::build(&dwarf, &u));
                    cur_unit = Some(u);
                    cur_idx = Some(unit_idx);
                }
                Err(_) => {
                    ip_frames.insert(ip, vec![unknown_frame(ip)]);
                    continue;
                }
            }
        }
        let unit = cur_unit.as_ref().unwrap();
        let lines = cur_lines.as_ref().unwrap();
        let frames = walk_function(&dwarf, &headers, unit, die, probe, ip, lines, &mut index);
        ip_frames.insert(ip, if frames.is_empty() { vec![unknown_frame(ip)] } else { frames });
    }
    drop(cur_unit);
    drop(cur_lines);

    Ok((ip_frames, index))
}

fn unknown_frame(ip: u64) -> SymFrame {
    SymFrame { ip, mangled: None, demangled: None, file: None, line: None, inlined: false }
}

/// Walk one subprogram subtree following only the inline path covering `probe`.
fn walk_function(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    headers: &[Header<'_>],
    unit: &gimli::Unit<Reader<'_>>,
    func_die: UnitOffset,
    probe: u64,
    ip: u64,
    lines: &LineIndex,
    index: &mut DwarfIndex,
) -> Vec<SymFrame> {
    let Ok(mut tree) = unit.entries_tree(Some(func_die)) else {
        return Vec::new();
    };
    let Ok(root) = tree.root() else { return Vec::new() };

    let mut chain: Vec<Frame> = Vec::new();
    // Real (outermost) function: name + its own template params.
    let (mangled, demangled) = func_name_and_params(dwarf, headers, unit, root.entry(), index);
    chain.push(Frame { mangled, demangled, call_file: None, call_line: None });

    descend(dwarf, headers, unit, root, probe, index, &mut chain);

    // Locations: innermost = line program at the probe; outer frame i's location
    // is the call site of its inlined child (frame i+1).
    let n = chain.len();
    let (innermost_file, innermost_line) = lines.lookup(probe);
    let mut frames: Vec<SymFrame> = Vec::with_capacity(n);
    for (i, t) in chain.iter().enumerate() {
        let (file, line) = if i + 1 < n {
            (chain[i + 1].call_file.clone(), chain[i + 1].call_line)
        } else {
            (innermost_file.clone(), innermost_line)
        };
        frames.push(SymFrame {
            ip,
            mangled: t.mangled.clone(),
            demangled: t.demangled.clone(),
            file,
            line,
            inlined: i + 1 < n,
        });
    }
    frames.reverse(); // innermost-first
    frames
}

/// Among `node`'s children, descend the single chain of inlined_subroutines /
/// lexical_blocks whose range covers `probe`.
fn descend(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    headers: &[Header<'_>],
    unit: &gimli::Unit<Reader<'_>>,
    node: gimli::EntriesTreeNode<'_, '_, '_, Reader<'_>>,
    probe: u64,
    index: &mut DwarfIndex,
    chain: &mut Vec<Frame>,
) {
    let mut children = node.children();
    while let Ok(Some(child)) = children.next() {
        let tag = child.entry().tag();
        if tag != gimli::DW_TAG_inlined_subroutine && tag != gimli::DW_TAG_lexical_block {
            continue;
        }
        if !die_covers(dwarf, unit, child.entry(), probe) {
            continue;
        }
        if tag == gimli::DW_TAG_inlined_subroutine {
            let (mangled, demangled) = func_name_and_params(dwarf, headers, unit, child.entry(), index);
            let call_file = call_file(dwarf, unit, child.entry());
            let call_line = call_line(child.entry());
            chain.push(Frame { mangled, demangled, call_file, call_line });
        }
        descend(dwarf, headers, unit, child, probe, index, chain);
        // The covering child is unique on the path; stop scanning siblings.
        return;
    }
}

/// Resolve a subprogram/inlined DIE's (mangled, demangled) name AND record its
/// template params in `index` for type recovery. For inlined subroutines the
/// name + params come from the `DW_AT_abstract_origin` definition DIE.
fn func_name_and_params(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    headers: &[Header<'_>],
    unit: &gimli::Unit<Reader<'_>>,
    entry: &Die,
    index: &mut DwarfIndex,
) -> (Option<String>, Option<String>) {
    // Resolve the name-bearing definition, following `abstract_origin` /
    // `specification` chains (same-unit or cross-unit) until a name + template
    // params are found — addr2line does the same; one hop isn't enough.
    let (mangled, concrete_name, params) = resolve_def(dwarf, headers, unit, entry.offset(), 8);

    // Alternate (`{:#}`) form omits the `::h<hash>` / crate `[hash]`
    // disambiguators — matching addr2line's `demangle()`, so downstream
    // capture-noise trimming and the shape recognizer (which match on clean
    // paths like `alloc::sync::Arc`) work identically.
    let demangled = mangled
        .as_deref()
        .map(|m| format!("{:#}", rustc_demangle::demangle(m)));

    if let Some(m) = &mangled {
        if !index.by_linkage.contains_key(m) {
            index
                .by_linkage
                .insert(m.clone(), FnTypeInfo { concrete_name, template_params: params });
        }
    }
    (mangled, demangled)
}

/// Read (mangled, concrete_name, template_params) for the DIE at `off`,
/// following `DW_AT_abstract_origin` / `DW_AT_specification` (same- or
/// cross-unit) until name + params are filled. Bounded by `depth`.
fn resolve_def(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    headers: &[Header<'_>],
    unit: &gimli::Unit<Reader<'_>>,
    off: UnitOffset,
    depth: u32,
) -> (Option<String>, Option<String>, Vec<(String, String)>) {
    let Ok(die) = unit.entry(off) else {
        return (None, None, Vec::new());
    };
    let mut mangled = attr_str(dwarf, unit, &die, gimli::DW_AT_linkage_name)
        .or_else(|| attr_str(dwarf, unit, &die, gimli::DW_AT_name));
    let mut concrete_name = attr_str(dwarf, unit, &die, gimli::DW_AT_name);
    let mut params = template_params(dwarf, unit, off);

    if depth > 0 && (mangled.is_none() || params.is_empty()) {
        // Follow the definition chain (abstract instance / out-of-line decl).
        let link = die
            .attr_value(gimli::DW_AT_abstract_origin)
            .ok()
            .flatten()
            .or_else(|| die.attr_value(gimli::DW_AT_specification).ok().flatten());
        let next = match link {
            Some(AttributeValue::UnitRef(o)) => Some((None, o)),
            Some(AttributeValue::DebugInfoRef(g)) => unit_for_offset(dwarf, headers, g)
                .and_then(|fu| g.to_unit_offset(&fu.header).map(|o| (Some(fu), o))),
            _ => None,
        };
        if let Some((foreign, o)) = next {
            let (m, c, p) = match &foreign {
                Some(fu) => resolve_def(dwarf, headers, fu, o, depth - 1),
                None => resolve_def(dwarf, headers, unit, o, depth - 1),
            };
            mangled = mangled.or(m);
            concrete_name = concrete_name.or(c);
            if params.is_empty() {
                params = p;
            }
        }
    }
    (mangled, concrete_name, params)
}

/// Parse (transiently) the unit whose `.debug_info` range contains `goff`.
/// `headers` are in ascending `.debug_info` order, so the containing unit is the
/// one with the greatest start offset ≤ `goff`.
fn unit_for_offset<'a>(
    dwarf: &gimli::Dwarf<Reader<'a>>,
    headers: &[Header<'a>],
    goff: gimli::DebugInfoOffset<usize>,
) -> Option<gimli::Unit<Reader<'a>>> {
    let mut best: Option<&Header<'a>> = None;
    for h in headers {
        match h.offset().as_debug_info_offset() {
            Some(dio) if dio.0 <= goff.0 => best = Some(h),
            _ => break,
        }
    }
    best.and_then(|h| dwarf.unit(*h).ok())
}

/// `DW_TAG_template_type_parameter` children of the DIE at `off`.
fn template_params(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    unit: &gimli::Unit<Reader<'_>>,
    off: UnitOffset,
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if let Ok(mut tree) = unit.entries_tree(Some(off)) {
        if let Ok(root) = tree.root() {
            let mut children = root.children();
            while let Ok(Some(child)) = children.next() {
                if child.entry().tag() == gimli::DW_TAG_template_type_parameter {
                    let e = child.entry();
                    if let Some(pn) = attr_str(dwarf, unit, e, gimli::DW_AT_name) {
                        // Reuse dwarf.rs's type-name logic (handles `[T]`/`&T`/`*T`)
                        // so recovered element types match the addr2line path.
                        if let Some(tn) = crate::dwarf::attr_type_name(dwarf, unit, e) {
                            out.push((pn, tn));
                        }
                    }
                }
            }
        }
    }
    out
}

fn die_covers(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    unit: &gimli::Unit<Reader<'_>>,
    entry: &Die,
    probe: u64,
) -> bool {
    let has = entry.attr_value(gimli::DW_AT_low_pc).ok().flatten().is_some()
        || entry.attr_value(gimli::DW_AT_ranges).ok().flatten().is_some();
    if !has {
        return true; // no range → transparent
    }
    if let Ok(mut ranges) = dwarf.die_ranges(unit, entry) {
        while let Ok(Some(r)) = ranges.next() {
            if probe >= r.begin && probe < r.end {
                return true;
            }
        }
    }
    false
}

fn entry_pc(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    unit: &gimli::Unit<Reader<'_>>,
    entry: &Die,
) -> Option<(u64, u64)> {
    if let Some(low) = entry.attr_value(gimli::DW_AT_low_pc).ok().flatten() {
        let low = match low {
            AttributeValue::Addr(a) => a,
            v => dwarf.attr_address(unit, v).ok().flatten()?,
        };
        let high = match entry.attr_value(gimli::DW_AT_high_pc).ok().flatten() {
            Some(AttributeValue::Addr(a)) => a,
            Some(AttributeValue::Udata(off)) => low + off,
            _ => low + 1,
        };
        if high > low {
            return Some((low, high));
        }
    }
    if entry.attr_value(gimli::DW_AT_ranges).ok().flatten().is_some() {
        let mut min = u64::MAX;
        let mut max = 0u64;
        if let Ok(mut ranges) = dwarf.die_ranges(unit, entry) {
            while let Ok(Some(r)) = ranges.next() {
                min = min.min(r.begin);
                max = max.max(r.end);
            }
        }
        if max > min {
            return Some((min, max));
        }
    }
    None
}

fn call_file(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    unit: &gimli::Unit<Reader<'_>>,
    entry: &Die,
) -> Option<String> {
    let idx = match entry.attr_value(gimli::DW_AT_call_file).ok().flatten()? {
        AttributeValue::FileIndex(i) => i,
        AttributeValue::Udata(i) => i,
        _ => return None,
    };
    file_name(dwarf, unit, idx)
}

fn call_line(entry: &Die) -> Option<u32> {
    match entry.attr_value(gimli::DW_AT_call_line).ok().flatten()? {
        AttributeValue::Udata(l) => Some(l as u32),
        _ => None,
    }
}

fn attr_str(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    unit: &gimli::Unit<Reader<'_>>,
    entry: &Die,
    attr: gimli::DwAt,
) -> Option<String> {
    let v = entry.attr_value(attr).ok().flatten()?;
    dwarf.attr_string(unit, v).ok().map(|s| s.to_string_lossy().into_owned())
}

/// Resolve a line-program file index to a full path, matching addr2line: join
/// `comp_dir` + the file's directory + name, skipping the prefix when a
/// component is already absolute.
fn file_name(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    unit: &gimli::Unit<Reader<'_>>,
    idx: u64,
) -> Option<String> {
    let header = unit.line_program.as_ref()?.header();
    let file = header.file(idx)?;
    let name = dwarf
        .attr_string(unit, file.path_name())
        .ok()
        .map(|s| s.to_string_lossy().into_owned())?;
    if name.starts_with('/') {
        return Some(name);
    }
    let dir = file
        .directory(header)
        .and_then(|d| dwarf.attr_string(unit, d).ok())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let comp_dir = unit
        .comp_dir
        .as_ref()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    // Absolute directory wins; otherwise prefix the unit's compilation dir.
    let path = if dir.starts_with('/') || comp_dir.is_empty() {
        join_path(&dir, &name)
    } else {
        join_path(&join_path(&comp_dir, &dir), &name)
    };
    Some(path)
}

fn join_path(base: &str, rest: &str) -> String {
    if base.is_empty() {
        rest.to_string()
    } else if base.ends_with('/') {
        format!("{base}{rest}")
    } else {
        format!("{base}/{rest}")
    }
}

/// Compact, sorted `(address → file, line)` index for one unit's line program.
/// Built once per unit, dropped when the unit changes — so only one unit's line
/// rows are ever resident.
struct LineIndex {
    rows: Vec<(u64, Option<String>, Option<u32>)>,
}

impl LineIndex {
    fn build(dwarf: &gimli::Dwarf<Reader<'_>>, unit: &gimli::Unit<Reader<'_>>) -> LineIndex {
        let mut rows: Vec<(u64, Option<String>, Option<u32>)> = Vec::new();
        if let Some(program) = unit.line_program.clone() {
            let mut state = program.rows();
            while let Ok(Some((_, row))) = state.next_row() {
                if row.end_sequence() {
                    // Boundary marker so a probe past the last real row isn't
                    // attributed to it.
                    rows.push((row.address(), None, None));
                } else {
                    let file = file_name_from_row(dwarf, unit, row);
                    let line = row.line().map(|l| l.get() as u32);
                    rows.push((row.address(), file, line));
                }
            }
        }
        rows.sort_by_key(|r| r.0);
        LineIndex { rows }
    }

    /// File/line for `probe`: the row with the greatest address ≤ probe.
    fn lookup(&self, probe: u64) -> (Option<String>, Option<u32>) {
        let i = self.rows.partition_point(|r| r.0 <= probe);
        if i == 0 {
            return (None, None);
        }
        let (_, f, l) = &self.rows[i - 1];
        (f.clone(), *l)
    }
}

fn file_name_from_row(
    dwarf: &gimli::Dwarf<Reader<'_>>,
    unit: &gimli::Unit<Reader<'_>>,
    row: &gimli::LineRow,
) -> Option<String> {
    file_name(dwarf, unit, row.file_index())
}
