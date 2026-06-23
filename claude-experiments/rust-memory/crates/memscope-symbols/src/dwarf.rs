//! Reads the binary's DWARF into two indexes via a single flat DFS:
//!
//! 1. [`DwarfIndex`] — mangled linkage name -> concrete monomorphized type info
//!    (the M2 join: `DW_AT_linkage_name` + `DW_TAG_template_type_parameter`).
//! 2. [`LayoutIndex`] — every type's byte size + the byte offsets of its
//!    pointer-typed fields, found by recursively flattening inline members
//!    (struct / array / `RawVec` / `Unique` / `NonNull`) down to the underlying
//!    `DW_TAG_pointer_type`. This is what lets the heap-graph walker turn an
//!    allocation's bytes into outgoing reference edges.
//!
//! Types are keyed by their global `.debug_info` offset; named types also get a
//! name -> offset entry so a recovered type name (`serve::Session`) resolves to
//! its layout.

use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;

use gimli::{AttributeValue, EndianSlice, RunTimeEndian};
use object::{Object, ObjectSection};

type DynErr = Box<dyn Error + Send + Sync>;

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

/// A raw type DIE, keyed by `.debug_info` offset.
#[derive(Clone, Debug)]
struct RawType {
    byte_size: u64,
    kind: RawKind,
}

#[derive(Clone, Debug)]
enum RawKind {
    /// `DW_TAG_pointer_type` / reference — an outgoing edge candidate.
    Pointer { pointee: Option<usize> },
    /// struct / union (Rust data enums are structs with a `variant_part`).
    Aggregate { members: Vec<RawMember> },
    Array { elem: Option<usize>, count: u64 },
    /// Base types, C-like enums — no pointers inside.
    Scalar,
}

#[derive(Clone, Debug)]
struct RawMember {
    offset: u64,
    type_id: Option<usize>,
    /// True if the member lives inside a `variant_part` (an enum variant), so a
    /// pointer there is only valid when that variant is active.
    in_variant: bool,
}

/// A flattened pointer-typed field: a byte offset within an instance of the
/// owning type at which a pointer lives, plus the (best-effort) pointee type.
#[derive(Clone, Debug)]
pub struct PtrField {
    pub offset: u64,
    pub pointee: Option<String>,
    pub in_variant: bool,
}

/// Per-type layout information for the heap-graph walker.
pub struct LayoutIndex {
    by_id: HashMap<usize, RawType>,
    id_to_name: HashMap<usize, String>,
    name_to_id: HashMap<String, usize>,
}

impl LayoutIndex {
    pub fn type_count(&self) -> usize {
        self.by_id.len()
    }

    /// Byte size of a named type, if known.
    pub fn size_of(&self, name: &str) -> Option<u64> {
        let id = *self.name_to_id.get(name)?;
        self.by_id.get(&id).map(|t| t.byte_size)
    }

    /// All pointer fields (relative byte offsets) of a named type, found by
    /// recursively flattening inline members. Returns an empty vec for leaf /
    /// pointerless types, and `None` if the type isn't in the index.
    pub fn pointer_fields(&self, name: &str) -> Option<Vec<PtrField>> {
        let id = *self.name_to_id.get(name)?;
        let mut acc = Vec::new();
        self.flatten(id, 0, false, 0, &mut acc);
        Some(acc)
    }

    fn size_of_id(&self, id: usize) -> u64 {
        self.by_id.get(&id).map(|t| t.byte_size).unwrap_or(0)
    }

    fn flatten(&self, id: usize, base: u64, in_variant: bool, depth: u32, acc: &mut Vec<PtrField>) {
        if depth > 32 {
            return;
        }
        let Some(rt) = self.by_id.get(&id) else {
            return;
        };
        match &rt.kind {
            RawKind::Pointer { pointee } => {
                acc.push(PtrField {
                    offset: base,
                    pointee: pointee.and_then(|p| self.id_to_name.get(&p).cloned()),
                    in_variant,
                });
            }
            RawKind::Aggregate { members } => {
                for m in members {
                    if let Some(t) = m.type_id {
                        self.flatten(t, base + m.offset, in_variant || m.in_variant, depth + 1, acc);
                    }
                }
            }
            RawKind::Array { elem, count } => {
                if let Some(e) = elem {
                    let esz = self.size_of_id(*e);
                    if esz > 0 {
                        // Cap inline-array expansion so a giant `[T; N]` can't blow up.
                        let n = (*count).min(4096);
                        for i in 0..n {
                            self.flatten(*e, base + i * esz, in_variant, depth + 1, acc);
                        }
                    }
                }
            }
            RawKind::Scalar => {}
        }
    }
}

/// A currently-open DIE while walking (paired with its depth on the stack).
#[derive(Clone, Copy)]
enum Role {
    Type(usize), // global offset key into by_id
    Sub(usize),  // index into the pending-subs vec
    VariantPart, // marks an enum-variant subtree
    Other,
}

/// Parse `obj_data` (an ELF with embedded DWARF, or a dSYM Mach-O) and build
/// both the linkage index and the layout index in one pass.
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

    // Subprograms collected with template params as (name, type_id); resolved to
    // names after the full pass (forward references).
    struct PendingSub {
        linkage: Option<String>,
        concrete_name: Option<String>,
        params: Vec<(String, Option<usize>)>,
    }
    let mut subs: Vec<PendingSub> = Vec::new();

    let mut by_id: HashMap<usize, RawType> = HashMap::new();
    let mut id_to_name: HashMap<usize, String> = HashMap::new();
    let mut name_to_id: HashMap<String, usize> = HashMap::new();

    // Stack of currently-open DIEs while walking (depth, role) — see `Role`.
    let mut units = dwarf.units();
    while let Some(header) = units.next()? {
        let unit = dwarf.unit(header)?;
        let mut stack: Vec<(isize, Role)> = Vec::new();
        let mut depth: isize = 0;
        let mut entries = unit.entries();
        while let Some((delta, entry)) = entries.next_dfs()? {
            depth += delta;
            // Close DIEs that have ended.
            while let Some(&(d, _)) = stack.last() {
                if d >= depth {
                    stack.pop();
                } else {
                    break;
                }
            }

            let global = entry
                .offset()
                .to_debug_info_offset(&unit.header)
                .map(|o| o.0);

            let tag = entry.tag();
            match tag {
                gimli::DW_TAG_subprogram => {
                    let linkage = attr_string(&dwarf, &unit, entry, gimli::DW_AT_linkage_name);
                    let concrete_name = attr_string(&dwarf, &unit, entry, gimli::DW_AT_name);
                    subs.push(PendingSub {
                        linkage,
                        concrete_name,
                        params: Vec::new(),
                    });
                    stack.push((depth, Role::Sub(subs.len() - 1)));
                }

                gimli::DW_TAG_template_type_parameter => {
                    if let Some(name) = attr_string(&dwarf, &unit, entry, gimli::DW_AT_name) {
                        let tid = attr_ref_global(entry, gimli::DW_AT_type, &unit);
                        if let Some(i) = nearest_sub(&stack) {
                            subs[i].params.push((name, tid));
                        }
                    }
                    stack.push((depth, Role::Other));
                }

                gimli::DW_TAG_pointer_type
                | gimli::DW_TAG_reference_type
                | gimli::DW_TAG_rvalue_reference_type => {
                    if let Some(off) = global {
                        let pointee = attr_ref_global(entry, gimli::DW_AT_type, &unit);
                        by_id.insert(
                            off,
                            RawType {
                                byte_size: attr_udata(entry, gimli::DW_AT_byte_size).unwrap_or(8),
                                kind: RawKind::Pointer { pointee },
                            },
                        );
                        if let Some(name) = attr_string(&dwarf, &unit, entry, gimli::DW_AT_name) {
                            id_to_name.insert(off, name.clone());
                            name_to_id.entry(name).or_insert(off);
                        }
                        stack.push((depth, Role::Type(off)));
                    } else {
                        stack.push((depth, Role::Other));
                    }
                }

                gimli::DW_TAG_structure_type | gimli::DW_TAG_union_type => {
                    if let Some(off) = global {
                        by_id.insert(
                            off,
                            RawType {
                                byte_size: attr_udata(entry, gimli::DW_AT_byte_size).unwrap_or(0),
                                kind: RawKind::Aggregate {
                                    members: Vec::new(),
                                },
                            },
                        );
                        if let Some(name) = attr_string(&dwarf, &unit, entry, gimli::DW_AT_name) {
                            id_to_name.insert(off, name.clone());
                            name_to_id.entry(name).or_insert(off);
                        }
                        stack.push((depth, Role::Type(off)));
                    } else {
                        stack.push((depth, Role::Other));
                    }
                }

                gimli::DW_TAG_array_type => {
                    if let Some(off) = global {
                        let elem = attr_ref_global(entry, gimli::DW_AT_type, &unit);
                        by_id.insert(
                            off,
                            RawType {
                                byte_size: attr_udata(entry, gimli::DW_AT_byte_size).unwrap_or(0),
                                kind: RawKind::Array { elem, count: 0 },
                            },
                        );
                        stack.push((depth, Role::Type(off)));
                    } else {
                        stack.push((depth, Role::Other));
                    }
                }

                gimli::DW_TAG_base_type | gimli::DW_TAG_enumeration_type => {
                    if let Some(off) = global {
                        by_id.insert(
                            off,
                            RawType {
                                byte_size: attr_udata(entry, gimli::DW_AT_byte_size).unwrap_or(0),
                                kind: RawKind::Scalar,
                            },
                        );
                        if let Some(name) = attr_string(&dwarf, &unit, entry, gimli::DW_AT_name) {
                            id_to_name.insert(off, name.clone());
                            name_to_id.entry(name).or_insert(off);
                        }
                        stack.push((depth, Role::Type(off)));
                    } else {
                        stack.push((depth, Role::Other));
                    }
                }

                gimli::DW_TAG_member => {
                    // Attach to the nearest enclosing aggregate type.
                    let (type_off, in_variant) = nearest_type(&stack);
                    if let Some(off) = type_off {
                        let m = RawMember {
                            offset: attr_udata(entry, gimli::DW_AT_data_member_location)
                                .unwrap_or(0),
                            type_id: attr_ref_global(entry, gimli::DW_AT_type, &unit),
                            in_variant,
                        };
                        if let Some(rt) = by_id.get_mut(&off) {
                            if let RawKind::Aggregate { members } = &mut rt.kind {
                                members.push(m);
                            }
                        }
                    }
                    stack.push((depth, Role::Other));
                }

                gimli::DW_TAG_subrange_type => {
                    // Array length: count, or upper_bound + 1.
                    let count = attr_udata(entry, gimli::DW_AT_count).or_else(|| {
                        attr_udata(entry, gimli::DW_AT_upper_bound).map(|u| u + 1)
                    });
                    if let (Some(c), Some(Role::Type(off))) = (count, nearest_type_role(&stack)) {
                        if let Some(rt) = by_id.get_mut(&off) {
                            if let RawKind::Array { count: c0, .. } = &mut rt.kind {
                                *c0 = c;
                            }
                        }
                    }
                    stack.push((depth, Role::Other));
                }

                gimli::DW_TAG_variant_part | gimli::DW_TAG_variant => {
                    stack.push((depth, Role::VariantPart));
                }

                _ => {
                    stack.push((depth, Role::Other));
                }
            }
        }
    }

    // Finalize the linkage index, resolving template-param type ids to names.
    let mut by_linkage = HashMap::new();
    for s in subs {
        let Some(linkage) = s.linkage else { continue };
        let template_params = s
            .params
            .into_iter()
            .map(|(pname, tid)| {
                let tname = tid
                    .and_then(|i| id_to_name.get(&i).cloned())
                    .unwrap_or_else(|| "?".to_string());
                (pname, tname)
            })
            .collect();
        let info = FnTypeInfo {
            concrete_name: s.concrete_name,
            template_params,
        };
        by_linkage
            .entry(linkage)
            .and_modify(|e: &mut FnTypeInfo| {
                if e.template_params.is_empty() && !info.template_params.is_empty() {
                    *e = info.clone();
                }
            })
            .or_insert(info);
    }

    Ok((
        DwarfIndex { by_linkage },
        LayoutIndex {
            by_id,
            id_to_name,
            name_to_id,
        },
    ))
}

// --- stack helpers -----------------------------------------------------------

/// Index (into the pending-subs vec) of the nearest enclosing subprogram.
fn nearest_sub(stack: &[(isize, Role)]) -> Option<usize> {
    stack.iter().rev().find_map(|(_, r)| match r {
        Role::Sub(i) => Some(*i),
        _ => None,
    })
}

/// Nearest enclosing aggregate type offset + whether we're inside a variant.
fn nearest_type(stack: &[(isize, Role)]) -> (Option<usize>, bool) {
    let mut in_variant = false;
    for (_, r) in stack.iter().rev() {
        match r {
            Role::VariantPart => in_variant = true,
            Role::Type(off) => return (Some(*off), in_variant),
            _ => {}
        }
    }
    (None, in_variant)
}

fn nearest_type_role(stack: &[(isize, Role)]) -> Option<Role> {
    stack
        .iter()
        .rev()
        .find_map(|(_, r)| matches!(r, Role::Type(_)).then_some(*r))
}

// --- attribute helpers -------------------------------------------------------

fn attr_string<'a>(
    dwarf: &gimli::Dwarf<EndianSlice<'a, RunTimeEndian>>,
    unit: &gimli::Unit<EndianSlice<'a, RunTimeEndian>>,
    entry: &gimli::DebuggingInformationEntry<EndianSlice<'a, RunTimeEndian>>,
    attr: gimli::DwAt,
) -> Option<String> {
    let value = entry.attr_value(attr).ok().flatten()?;
    let reader = dwarf.attr_string(unit, value).ok()?;
    Some(reader.to_string_lossy().into_owned())
}

fn attr_udata<'a>(
    entry: &gimli::DebuggingInformationEntry<EndianSlice<'a, RunTimeEndian>>,
    attr: gimli::DwAt,
) -> Option<u64> {
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

fn attr_ref_global<'a>(
    entry: &gimli::DebuggingInformationEntry<EndianSlice<'a, RunTimeEndian>>,
    attr: gimli::DwAt,
    unit: &gimli::Unit<EndianSlice<'a, RunTimeEndian>>,
) -> Option<usize> {
    match entry.attr_value(attr).ok().flatten()? {
        AttributeValue::UnitRef(uoff) => uoff.to_debug_info_offset(&unit.header).map(|o| o.0),
        AttributeValue::DebugInfoRef(o) => Some(o.0),
        _ => None,
    }
}
