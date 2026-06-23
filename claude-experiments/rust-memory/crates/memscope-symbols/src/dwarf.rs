//! Builds the join table the type oracle relies on: a map from a function's
//! mangled linkage name to its *concrete* monomorphized type info, read out of
//! the binary's DWARF.
//!
//! The key fact (validated by the spike — see `project-rust-memory-allocator`):
//! a monomorphized generic like `Vec::<u64>::with_capacity_in` has a generic
//! `DW_AT_linkage_name` (`..Vec$LT$T$C$A$GT$..`) but a *concrete* `DW_AT_name`
//! (`with_capacity_in<u64, alloc::alloc::Global>`) and concrete
//! `DW_TAG_template_type_parameter` children (`T -> u64`). The linkage name is
//! exactly what a runtime backtrace resolves to, so we join on it with no
//! address arithmetic.

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

/// The whole index plus a sorted address map for symbolication fallback.
pub struct DwarfIndex {
    /// mangled linkage name -> concrete type info.
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

/// Parse `obj_data` (an ELF with embedded DWARF, or a dSYM Mach-O) and build the
/// linkage-name -> concrete-type index.
pub fn build_index(obj_data: &[u8]) -> Result<DwarfIndex, DynErr> {
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
    // No type annotation on `section`: let `borrow`'s HRTB signature drive the
    // lifetime, then deref the `Cow` to `&[u8]` via slicing.
    let dwarf = dwarf_sections.borrow(|section| EndianSlice::new(&section[..], endian));

    let mut by_linkage: HashMap<String, FnTypeInfo> = HashMap::new();

    let mut units = dwarf.units();
    while let Some(header) = units.next()? {
        let unit = dwarf.unit(header)?;
        let mut tree = unit.entries_tree(None)?;
        let root = tree.root()?;
        walk(&dwarf, &unit, root, &mut by_linkage)?;
    }

    Ok(DwarfIndex { by_linkage })
}

fn walk<'a>(
    dwarf: &gimli::Dwarf<EndianSlice<'a, RunTimeEndian>>,
    unit: &gimli::Unit<EndianSlice<'a, RunTimeEndian>>,
    node: gimli::EntriesTreeNode<EndianSlice<'a, RunTimeEndian>>,
    index: &mut HashMap<String, FnTypeInfo>,
) -> Result<(), DynErr> {
    let entry = node.entry();
    let is_subprogram = entry.tag() == gimli::DW_TAG_subprogram;

    let (linkage, concrete_name) = if is_subprogram {
        (
            attr_string(dwarf, unit, entry, gimli::DW_AT_linkage_name),
            attr_string(dwarf, unit, entry, gimli::DW_AT_name),
        )
    } else {
        (None, None)
    };

    let mut template_params: Vec<(String, String)> = Vec::new();

    let mut children = node.children();
    while let Some(child) = children.next()? {
        if is_subprogram {
            let ce = child.entry();
            if ce.tag() == gimli::DW_TAG_template_type_parameter {
                let pname =
                    attr_string(dwarf, unit, ce, gimli::DW_AT_name).unwrap_or_else(|| "?".into());
                if let Some(tname) = attr_type_name(dwarf, unit, ce) {
                    template_params.push((pname, tname));
                }
            }
            // (the borrow of `child` via `ce` ends here, before we move `child`)
        }
        walk(dwarf, unit, child, index)?;
    }

    if let Some(linkage) = linkage {
        // Keep the first/most-specific; don't clobber an entry that already has
        // template params with a barer one.
        let info = FnTypeInfo {
            concrete_name,
            template_params,
        };
        index
            .entry(linkage)
            .and_modify(|existing| {
                if existing.template_params.is_empty() && !info.template_params.is_empty() {
                    *existing = info.clone();
                }
            })
            .or_insert(info);
    }

    Ok(())
}

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

/// Resolve `DW_AT_type` on `entry` to a human type name (the referenced DIE's
/// `DW_AT_name`). Handles the common same-unit reference; reconstructs a few
/// shapes (pointer/ref/slice) when the target has no name of its own.
fn attr_type_name<'a>(
    dwarf: &gimli::Dwarf<EndianSlice<'a, RunTimeEndian>>,
    unit: &gimli::Unit<EndianSlice<'a, RunTimeEndian>>,
    entry: &gimli::DebuggingInformationEntry<EndianSlice<'a, RunTimeEndian>>,
) -> Option<String> {
    let value = entry.attr_value(gimli::DW_AT_type).ok().flatten()?;
    let offset = match value {
        AttributeValue::UnitRef(o) => o,
        _ => return None,
    };
    type_name_at(dwarf, unit, offset, 0)
}

fn type_name_at<'a>(
    dwarf: &gimli::Dwarf<EndianSlice<'a, RunTimeEndian>>,
    unit: &gimli::Unit<EndianSlice<'a, RunTimeEndian>>,
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
    // No direct name: reconstruct common derived types from their target.
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
