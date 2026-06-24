//! Reads the binary's DWARF into two indexes, using `entries_tree` recursion
//! (the traversal proven to visit every DIE):
//!
//! 1. [`DwarfIndex`] — mangled linkage name -> concrete monomorphized type info
//!    (the M2 join: `DW_AT_linkage_name` + `DW_TAG_template_type_parameter`).
//! 2. [`LayoutIndex`] — every type's byte size + the byte offsets of its
//!    pointer-typed fields, found by recursively flattening inline members down
//!    to the underlying `DW_TAG_pointer_type`.
//!
//! Template-parameter type names are resolved inline (follow `DW_AT_type` ->
//! `DW_AT_name`). Layout member/pointer targets are stored as global
//! `.debug_info` offsets and resolved after the whole binary is indexed (they
//! may be forward references), with a parallel name -> offset map so a recovered
//! type name (`serve::Session`) finds its layout.

use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;

use gimli::{AttributeValue, EndianSlice, RunTimeEndian};
use object::{Object, ObjectSection};

type DynErr = Box<dyn Error + Send + Sync>;
type Reader<'a> = EndianSlice<'a, RunTimeEndian>;
type Die<'a, 'u> = gimli::DebuggingInformationEntry<'a, 'u, Reader<'a>>;

/// Concrete type facts for one monomorphized function.
#[derive(Clone, Debug, Default)]
pub struct FnTypeInfo {
    /// `DW_AT_name`, e.g. `with_capacity_in<u64, alloc::alloc::Global>`.
    pub concrete_name: Option<String>,
    /// `DW_TAG_template_type_parameter` children, e.g. `[("T","u64"),("A","alloc::alloc::Global")]`.
    pub template_params: Vec<(String, String)>,
}

impl FnTypeInfo {
    /// The element/payload type — the parameter named `T` if present, else the
    /// first parameter whose type isn't an allocator.
    pub fn element_type(&self) -> Option<&str> {
        if let Some((_, ty)) = self.template_params.iter().find(|(n, _)| n == "T") {
            return Some(ty.as_str());
        }
        self.template_params
            .iter()
            .find(|(_, ty)| !ty.contains("Global") && !ty.contains("Allocator"))
            .map(|(_, ty)| ty.as_str())
    }
}

/// mangled linkage name -> concrete type info.
pub struct DwarfIndex {
    pub by_linkage: HashMap<String, FnTypeInfo>,
}

impl DwarfIndex {
    /// Look up a (possibly platform-decorated) mangled name, tolerating a
    /// leading-underscore mismatch between Mach-O symbol tables and DWARF.
    pub fn lookup(&self, mangled: &str) -> Option<&FnTypeInfo> {
        if let Some(v) = self.by_linkage.get(mangled) {
            return Some(v);
        }
        if let Some(stripped) = mangled.strip_prefix('_') {
            if let Some(v) = self.by_linkage.get(stripped) {
                return Some(v);
            }
        }
        let mut with = String::with_capacity(mangled.len() + 1);
        with.push('_');
        with.push_str(mangled);
        self.by_linkage.get(&with)
    }

    pub fn len(&self) -> usize {
        self.by_linkage.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_linkage.is_empty()
    }
}

// --- type layout index -------------------------------------------------------

#[derive(Clone, Debug)]
struct RawType {
    byte_size: u64,
    byte_align: Option<u64>,
    kind: RawKind,
}

#[derive(Clone, Debug)]
enum RawKind {
    Pointer { pointee: Option<usize> },
    Aggregate { members: Vec<RawMember> },
    /// A Rust data enum (a struct with a `DW_TAG_variant_part`). Walked
    /// discriminant-aware so only the live variant's fields are followed.
    Enum {
        /// Byte offset of the discriminant within the enum.
        discr_offset: u64,
        /// Type id of the discriminant (for its byte size).
        discr_type_id: Option<usize>,
        /// Members common to all variants (outside the variant_part) — rare.
        common: Vec<RawMember>,
        variants: Vec<RawVariant>,
    },
    Array { elem: Option<usize>, count: u64 },
    Scalar,
}

#[derive(Clone, Debug)]
struct RawVariant {
    /// The discriminant value selecting this variant; `None` marks the default
    /// (untagged) variant used when no explicit value matches (niche `Some`).
    discr_value: Option<u64>,
    members: Vec<RawMember>,
}

#[derive(Clone, Debug)]
struct RawMember {
    offset: u64,
    type_id: Option<usize>,
    in_variant: bool,
}

/// Reads `size` (1/2/4/8) bytes of unsigned integer / pointer from process
/// memory at `addr`. The layout walker only asks for addresses provably inside a
/// live allocation, so an in-process reader can dereference directly.
pub trait MemReader {
    fn read_uint(&self, addr: u64, size: u64) -> Option<u64>;
}

/// A flattened pointer-typed field of a type: a byte offset at which a pointer
/// lives, plus the (best-effort) pointee type name.
#[derive(Clone, Debug)]
pub struct PtrField {
    pub offset: u64,
    pub pointee: Option<String>,
    pub in_variant: bool,
}

/// Per-type layout info for the heap-graph walker.
pub struct LayoutIndex {
    by_id: HashMap<usize, RawType>,
    id_to_name: HashMap<usize, String>,
    name_to_id: HashMap<String, usize>,
}

impl LayoutIndex {
    pub fn type_count(&self) -> usize {
        self.by_id.len()
    }

    pub fn size_of(&self, name: &str) -> Option<u64> {
        let id = *self.name_to_id.get(name)?;
        self.by_id.get(&id).map(|t| t.byte_size)
    }

    /// `DW_AT_alignment` of a named type, if recorded.
    pub fn align_of(&self, name: &str) -> Option<u64> {
        let id = *self.name_to_id.get(name)?;
        self.by_id.get(&id).and_then(|t| t.byte_align)
    }

    /// All pointer fields (relative byte offsets) of a named type, found by
    /// statically flattening inline members. Over-approximates enums (includes
    /// every variant's pointers). For sound, discriminant-aware walking use
    /// [`LayoutIndex::collect_pointer_offsets`].
    pub fn pointer_fields(&self, name: &str) -> Option<Vec<PtrField>> {
        let id = *self.name_to_id.get(name)?;
        let mut acc = Vec::new();
        self.flatten(id, 0, false, 0, &mut acc);
        Some(acc)
    }

    /// Sound, reader-aware pointer collection for an instance of `name` located
    /// at `inst_addr`: reads enum discriminants from memory and follows only the
    /// live variant's fields. Returns offsets relative to `inst_addr`.
    pub fn collect_pointer_offsets(
        &self,
        name: &str,
        inst_addr: u64,
        reader: &dyn MemReader,
    ) -> Vec<u64> {
        let mut acc = Vec::new();
        if let Some(&id) = self.name_to_id.get(name) {
            self.collect(id, 0, inst_addr, reader, 0, &mut acc);
        }
        acc
    }

    fn size_of_id(&self, id: usize) -> u64 {
        self.by_id.get(&id).map(|t| t.byte_size).unwrap_or(0)
    }

    fn collect(
        &self,
        id: usize,
        base: u64,
        inst_addr: u64,
        reader: &dyn MemReader,
        depth: u32,
        acc: &mut Vec<u64>,
    ) {
        if depth > 32 {
            return;
        }
        let Some(rt) = self.by_id.get(&id) else {
            return;
        };
        match &rt.kind {
            RawKind::Pointer { .. } => acc.push(base),
            RawKind::Aggregate { members } => {
                for m in members {
                    if let Some(t) = m.type_id {
                        self.collect(t, base + m.offset, inst_addr, reader, depth + 1, acc);
                    }
                }
            }
            RawKind::Enum {
                discr_offset,
                discr_type_id,
                common,
                variants,
            } => {
                for m in common {
                    if let Some(t) = m.type_id {
                        self.collect(t, base + m.offset, inst_addr, reader, depth + 1, acc);
                    }
                }
                let dsize = discr_type_id.map(|d| self.size_of_id(d)).unwrap_or(8).clamp(1, 8);
                let discr = reader.read_uint(inst_addr + base + discr_offset, dsize);
                // Pick the variant whose discr_value matches; else the default
                // (no discr_value) variant. If neither, walk nothing (sound).
                let active = discr
                    .and_then(|d| variants.iter().find(|v| v.discr_value == Some(d)))
                    .or_else(|| variants.iter().find(|v| v.discr_value.is_none()));
                if let Some(v) = active {
                    for m in &v.members {
                        if let Some(t) = m.type_id {
                            self.collect(t, base + m.offset, inst_addr, reader, depth + 1, acc);
                        }
                    }
                }
            }
            RawKind::Array { elem, count } => {
                if let Some(e) = elem {
                    let esz = self.size_of_id(*e);
                    if esz > 0 {
                        for i in 0..(*count).min(4096) {
                            self.collect(*e, base + i * esz, inst_addr, reader, depth + 1, acc);
                        }
                    }
                }
            }
            RawKind::Scalar => {}
        }
    }

    fn flatten(&self, id: usize, base: u64, in_variant: bool, depth: u32, acc: &mut Vec<PtrField>) {
        if depth > 32 {
            return;
        }
        let Some(rt) = self.by_id.get(&id) else {
            return;
        };
        match &rt.kind {
            RawKind::Pointer { pointee } => acc.push(PtrField {
                offset: base,
                pointee: pointee.and_then(|p| self.id_to_name.get(&p).cloned()),
                in_variant,
            }),
            RawKind::Aggregate { members } => {
                for m in members {
                    if let Some(t) = m.type_id {
                        self.flatten(t, base + m.offset, in_variant || m.in_variant, depth + 1, acc);
                    }
                }
            }
            RawKind::Enum {
                common, variants, ..
            } => {
                for m in common {
                    if let Some(t) = m.type_id {
                        self.flatten(t, base + m.offset, in_variant, depth + 1, acc);
                    }
                }
                for v in variants {
                    for m in &v.members {
                        if let Some(t) = m.type_id {
                            self.flatten(t, base + m.offset, true, depth + 1, acc);
                        }
                    }
                }
            }
            RawKind::Array { elem, count } => {
                if let Some(e) = elem {
                    let esz = self.size_of_id(*e);
                    if esz > 0 {
                        for i in 0..(*count).min(4096) {
                            self.flatten(*e, base + i * esz, in_variant, depth + 1, acc);
                        }
                    }
                }
            }
            RawKind::Scalar => {}
        }
    }
}

/// Mutable indexes threaded through the recursive walk.
struct Idx {
    by_linkage: HashMap<String, FnTypeInfo>,
    by_id: HashMap<usize, RawType>,
    id_to_name: HashMap<usize, String>,
    name_to_id: HashMap<String, usize>,
}

/// Parse `obj_data` (an ELF with embedded DWARF, or a dSYM Mach-O) and build
/// both indexes.
pub fn build(obj_data: &[u8]) -> Result<(DwarfIndex, LayoutIndex), DynErr> {
    let object = object::File::parse(obj_data)?;
    let endian = if object.is_little_endian() {
        RunTimeEndian::Little
    } else {
        RunTimeEndian::Big
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
    let dwarf = dwarf_sections.borrow(|section| EndianSlice::new(&section[..], endian));

    let mut idx = Idx {
        by_linkage: HashMap::new(),
        by_id: HashMap::new(),
        id_to_name: HashMap::new(),
        name_to_id: HashMap::new(),
    };

    let mut units = dwarf.units();
    while let Some(header) = units.next()? {
        let unit = dwarf.unit(header)?;
        let mut tree = unit.entries_tree(None)?;
        let root = tree.root()?;
        walk(&dwarf, &unit, root, &mut idx)?;
    }

    Ok((
        DwarfIndex {
            by_linkage: idx.by_linkage,
        },
        LayoutIndex {
            by_id: idx.by_id,
            id_to_name: idx.id_to_name,
            name_to_id: idx.name_to_id,
        },
    ))
}

fn global_offset(entry: &Die, unit: &gimli::Unit<Reader>) -> Option<usize> {
    entry.offset().to_debug_info_offset(&unit.header).map(|o| o.0)
}

fn walk<'a>(
    dwarf: &gimli::Dwarf<Reader<'a>>,
    unit: &gimli::Unit<Reader<'a>>,
    node: gimli::EntriesTreeNode<'a, '_, '_, Reader<'a>>,
    idx: &mut Idx,
) -> Result<(), DynErr> {
    let entry = node.entry();
    let tag = entry.tag();
    let global = global_offset(entry, unit);
    let name = attr_string(dwarf, unit, entry, gimli::DW_AT_name);
    let linkage = if tag == gimli::DW_TAG_subprogram {
        attr_string(dwarf, unit, entry, gimli::DW_AT_linkage_name)
    } else {
        None
    };
    let byte_size = attr_udata(entry, gimli::DW_AT_byte_size);
    let byte_align = attr_udata(entry, gimli::DW_AT_alignment);
    let elem_ref = if tag == gimli::DW_TAG_array_type {
        attr_ref_global(entry, gimli::DW_AT_type, unit)
    } else {
        None
    };
    let pointee = if matches!(
        tag,
        gimli::DW_TAG_pointer_type | gimli::DW_TAG_reference_type | gimli::DW_TAG_rvalue_reference_type
    ) {
        attr_ref_global(entry, gimli::DW_AT_type, unit)
    } else {
        None
    };
    // entry borrow ends here.

    // Register the type skeleton + names.
    if let Some(off) = global {
        match tag {
            gimli::DW_TAG_pointer_type
            | gimli::DW_TAG_reference_type
            | gimli::DW_TAG_rvalue_reference_type => {
                idx.by_id.insert(
                    off,
                    RawType {
                        byte_size: byte_size.unwrap_or(8),
                        byte_align,
                        kind: RawKind::Pointer { pointee },
                    },
                );
                register_name(idx, off, &name);
            }
            gimli::DW_TAG_structure_type | gimli::DW_TAG_union_type => {
                idx.by_id.insert(
                    off,
                    RawType {
                        byte_size: byte_size.unwrap_or(0),
                        byte_align,
                        kind: RawKind::Aggregate {
                            members: Vec::new(),
                        },
                    },
                );
                register_name(idx, off, &name);
            }
            gimli::DW_TAG_array_type => {
                idx.by_id.insert(
                    off,
                    RawType {
                        byte_size: byte_size.unwrap_or(0),
                        byte_align,
                        kind: RawKind::Array {
                            elem: elem_ref,
                            count: 0,
                        },
                    },
                );
            }
            gimli::DW_TAG_base_type | gimli::DW_TAG_enumeration_type => {
                idx.by_id.insert(
                    off,
                    RawType {
                        byte_size: byte_size.unwrap_or(0),
                        byte_align,
                        kind: RawKind::Scalar,
                    },
                );
                register_name(idx, off, &name);
            }
            _ => {}
        }
    }

    let is_struct = matches!(
        tag,
        gimli::DW_TAG_structure_type | gimli::DW_TAG_union_type
    );
    let is_array = tag == gimli::DW_TAG_array_type;
    let is_subprogram = tag == gimli::DW_TAG_subprogram;

    let mut template_params: Vec<(String, String)> = Vec::new();

    let mut children = node.children();
    while let Some(child) = children.next()? {
        // Pull out what we need from the child before recursing (which moves it).
        let ctag = child.entry().tag();

        if is_subprogram && ctag == gimli::DW_TAG_template_type_parameter {
            let ce = child.entry();
            if let Some(pn) = attr_string(dwarf, unit, ce, gimli::DW_AT_name) {
                if let Some(tn) = attr_type_name(dwarf, unit, ce) {
                    template_params.push((pn, tn));
                }
            }
        } else if is_struct && ctag == gimli::DW_TAG_member {
            if let Some(off) = global {
                let ce = child.entry();
                let m = RawMember {
                    offset: attr_udata(ce, gimli::DW_AT_data_member_location).unwrap_or(0),
                    type_id: attr_ref_global(ce, gimli::DW_AT_type, unit),
                    in_variant: false,
                };
                push_member(idx, off, m);
            }
        } else if is_struct && ctag == gimli::DW_TAG_variant_part {
            if let Some(off) = global {
                if let Some((discr_offset, discr_type_id, variants)) =
                    parse_variant_part(unit, &child)
                {
                    // Convert this aggregate into an enum, preserving any
                    // members declared outside the variant_part as `common`.
                    let common = match idx.by_id.get(&off).map(|t| &t.kind) {
                        Some(RawKind::Aggregate { members }) => members.clone(),
                        Some(RawKind::Enum { common, .. }) => common.clone(),
                        _ => Vec::new(),
                    };
                    if let Some(rt) = idx.by_id.get_mut(&off) {
                        rt.kind = RawKind::Enum {
                            discr_offset,
                            discr_type_id,
                            common,
                            variants,
                        };
                    }
                }
            }
            // also recurse so nested types inside variants get indexed
        } else if is_array && ctag == gimli::DW_TAG_subrange_type {
            if let Some(off) = global {
                let ce = child.entry();
                let count = attr_udata(ce, gimli::DW_AT_count)
                    .or_else(|| attr_udata(ce, gimli::DW_AT_upper_bound).map(|u| u + 1));
                if let Some(c) = count {
                    if let Some(rt) = idx.by_id.get_mut(&off) {
                        if let RawKind::Array { count: c0, .. } = &mut rt.kind {
                            *c0 = c;
                        }
                    }
                }
            }
        }

        walk(dwarf, unit, child, idx)?;
    }

    if let Some(linkage) = linkage {
        let info = FnTypeInfo {
            concrete_name: name,
            template_params,
        };
        idx.by_linkage
            .entry(linkage)
            .and_modify(|e: &mut FnTypeInfo| {
                if e.template_params.is_empty() && !info.template_params.is_empty() {
                    *e = info.clone();
                }
            })
            .or_insert(info);
    }

    Ok(())
}

fn register_name(idx: &mut Idx, off: usize, name: &Option<String>) {
    if let Some(n) = name {
        idx.id_to_name.insert(off, n.clone());
        idx.name_to_id.entry(n.clone()).or_insert(off);
    }
}

fn push_member(idx: &mut Idx, struct_off: usize, m: RawMember) {
    if let Some(rt) = idx.by_id.get_mut(&struct_off) {
        if let RawKind::Aggregate { members } = &mut rt.kind {
            members.push(m);
        }
    }
}

/// Parse an enum `variant_part` subtree into its discriminant location + the
/// per-variant member sets. Reads via a fresh `entries_tree` rooted at the
/// variant_part so the caller's child cursor is undisturbed.
///
/// Structure (Rust): the variant_part has `DW_AT_discr` -> a member (the tag,
/// at some offset, of some int type). Each `DW_TAG_variant` child carries an
/// optional `DW_AT_discr_value` and the variant's payload member(s). A variant
/// with no `DW_AT_discr_value` is the default (niche `Some`).
fn parse_variant_part(
    unit: &gimli::Unit<Reader>,
    vp_node: &gimli::EntriesTreeNode<Reader>,
) -> Option<(u64, Option<usize>, Vec<RawVariant>)> {
    let discr_ref = attr_ref_global(vp_node.entry(), gimli::DW_AT_discr, unit);

    let vp_off = vp_node.entry().offset();
    let mut tree = unit.entries_tree(Some(vp_off)).ok()?;
    let root = tree.root().ok()?;

    let mut discr_offset: u64 = 0;
    let mut discr_type_id: Option<usize> = None;
    let mut variants: Vec<RawVariant> = Vec::new();

    let mut children = root.children();
    while let Ok(Some(child)) = children.next() {
        let ctag = child.entry().tag();
        if ctag == gimli::DW_TAG_member {
            // The discriminant member (matched by DW_AT_discr; fall back to the
            // first member seen).
            let ce = child.entry();
            let g = global_offset(ce, unit);
            let off = attr_udata(ce, gimli::DW_AT_data_member_location).unwrap_or(0);
            let ty = attr_ref_global(ce, gimli::DW_AT_type, unit);
            if discr_ref.is_some() && g == discr_ref {
                discr_offset = off;
                discr_type_id = ty;
            } else if discr_ref.is_none() && discr_type_id.is_none() {
                discr_offset = off;
                discr_type_id = ty;
            }
        } else if ctag == gimli::DW_TAG_variant {
            let discr_value = attr_udata(child.entry(), gimli::DW_AT_discr_value);
            let mut members = Vec::new();
            let mut vc = child.children();
            while let Ok(Some(m)) = vc.next() {
                if m.entry().tag() == gimli::DW_TAG_member {
                    let me = m.entry();
                    members.push(RawMember {
                        offset: attr_udata(me, gimli::DW_AT_data_member_location).unwrap_or(0),
                        type_id: attr_ref_global(me, gimli::DW_AT_type, unit),
                        in_variant: true,
                    });
                }
            }
            variants.push(RawVariant {
                discr_value,
                members,
            });
        }
    }

    if variants.is_empty() {
        return None;
    }
    Some((discr_offset, discr_type_id, variants))
}

// --- attribute helpers -------------------------------------------------------

fn attr_string<'a>(
    dwarf: &gimli::Dwarf<Reader<'a>>,
    unit: &gimli::Unit<Reader<'a>>,
    entry: &Die<'a, '_>,
    attr: gimli::DwAt,
) -> Option<String> {
    let value = entry.attr_value(attr).ok().flatten()?;
    let reader = dwarf.attr_string(unit, value).ok()?;
    Some(reader.to_string_lossy().into_owned())
}

fn attr_udata(entry: &Die, attr: gimli::DwAt) -> Option<u64> {
    match entry.attr_value(attr).ok().flatten()? {
        AttributeValue::Udata(u) => Some(u),
        AttributeValue::Data1(u) => Some(u as u64),
        AttributeValue::Data2(u) => Some(u as u64),
        AttributeValue::Data4(u) => Some(u as u64),
        AttributeValue::Data8(u) => Some(u),
        AttributeValue::Sdata(i) => Some(i as u64),
        _ => None,
    }
}

fn attr_ref_global(entry: &Die, attr: gimli::DwAt, unit: &gimli::Unit<Reader>) -> Option<usize> {
    match entry.attr_value(attr).ok().flatten()? {
        AttributeValue::UnitRef(uoff) => uoff.to_debug_info_offset(&unit.header).map(|o| o.0),
        AttributeValue::DebugInfoRef(o) => Some(o.0),
        _ => None,
    }
}

/// Resolve `DW_AT_type` on `entry` to a human type name (the referenced DIE's
/// `DW_AT_name`), reconstructing pointer/ref/array shapes that have no name.
fn attr_type_name<'a>(
    dwarf: &gimli::Dwarf<Reader<'a>>,
    unit: &gimli::Unit<Reader<'a>>,
    entry: &Die<'a, '_>,
) -> Option<String> {
    let value = entry.attr_value(gimli::DW_AT_type).ok().flatten()?;
    let offset = match value {
        AttributeValue::UnitRef(o) => o,
        _ => return None,
    };
    type_name_at(dwarf, unit, offset, 0)
}

fn type_name_at<'a>(
    dwarf: &gimli::Dwarf<Reader<'a>>,
    unit: &gimli::Unit<Reader<'a>>,
    offset: gimli::UnitOffset,
    depth: u32,
) -> Option<String> {
    if depth > 8 {
        return None;
    }
    let die = unit.entry(offset).ok()?;
    if let Some(name) = attr_string(dwarf, unit, &die, gimli::DW_AT_name) {
        return Some(name);
    }
    let inner = || -> Option<String> {
        let v = die.attr_value(gimli::DW_AT_type).ok().flatten()?;
        let o = match v {
            AttributeValue::UnitRef(o) => o,
            _ => return None,
        };
        type_name_at(dwarf, unit, o, depth + 1)
    };
    match die.tag() {
        gimli::DW_TAG_pointer_type => Some(format!("*{}", inner().unwrap_or_else(|| "?".into()))),
        gimli::DW_TAG_reference_type => Some(format!("&{}", inner().unwrap_or_else(|| "?".into()))),
        gimli::DW_TAG_array_type => Some(format!("[{}]", inner().unwrap_or_else(|| "?".into()))),
        gimli::DW_TAG_const_type => inner(),
        _ => None,
    }
}
