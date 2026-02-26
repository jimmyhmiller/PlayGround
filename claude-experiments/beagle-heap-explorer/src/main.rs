use clap::{Parser, Subcommand};
use colored::*;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

// ── Pointer tagging ──────────────────────────────────────────────────────────

const TAG_MASK: u64 = 0b111;
const TAG_INT: u64 = 0;
const TAG_FLOAT: u64 = 1;
const TAG_STRING: u64 = 2;
const TAG_BOOL: u64 = 3;
const TAG_FUNPTR: u64 = 4;
const TAG_CLOSURE: u64 = 5;
const TAG_HEAP: u64 = 6;
const TAG_NULL: u64 = 7;

fn tag_name(tag: u64) -> &'static str {
    match tag {
        TAG_INT => "int",
        TAG_FLOAT => "float",
        TAG_STRING => "string",
        TAG_BOOL => "bool",
        TAG_FUNPTR => "funptr",
        TAG_CLOSURE => "closure",
        TAG_HEAP => "heap",
        TAG_NULL => "null",
        _ => "unknown",
    }
}

fn untag(val: u64) -> u64 {
    // Beagle uses shift-based tagging: tagged = (payload << 3) | tag
    // So to get the payload (address for pointers): shift right by 3
    val >> 3
}

fn tag_of(val: u64) -> u64 {
    val & TAG_MASK
}

fn is_heap_pointer(val: u64) -> bool {
    let tag = tag_of(val);
    matches!(tag, TAG_FLOAT | TAG_STRING | TAG_CLOSURE | TAG_HEAP) && untag(val) != 0
}

fn is_in_young(val: u64, young_base: u64, young_offset: u64) -> bool {
    let addr = untag(val);
    addr >= young_base && addr < young_base + young_offset
}

fn is_in_old(val: u64, old_base: u64, old_highmark: u64) -> bool {
    let addr = untag(val);
    addr >= old_base && addr < old_base + old_highmark
}

// ── Header parsing ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ObjectHeader {
    raw: u64,
    type_id: u8,
    type_data: u32,
    size: u64,
    opaque: bool,
    #[allow(dead_code)]
    marked: bool,
    #[allow(dead_code)]
    large: bool,
    forwarding: bool,
    #[allow(dead_code)]
    type_flags: u8,
    header_size: u64,
}

impl ObjectHeader {
    fn parse(raw: u64, large_size_word: Option<u64>) -> Self {
        let type_id = (raw & 0x7F) as u8;
        let type_data = ((raw >> 7) & 0xFFFFFFFF) as u32;
        let size_field = (raw >> 39) & 0xFFFF;
        let opaque = (raw >> 55) & 1 != 0;
        let marked = (raw >> 56) & 1 != 0;
        let large = (raw >> 57) & 1 != 0;
        let forwarding = (raw >> 58) & 1 != 0;
        let type_flags = ((raw >> 59) & 0b111) as u8;

        let (size, header_size) = if large {
            (large_size_word.unwrap_or(size_field), 16)
        } else {
            (size_field, 8)
        };

        Self {
            raw,
            type_id,
            type_data,
            size,
            opaque,
            marked,
            large,
            forwarding,
            type_flags,
            header_size,
        }
    }

    fn type_name(&self) -> &'static str {
        match self.type_id {
            0 => "struct",
            1 => "string",
            2 => "array",
            3 => "closure_env",
            _ => "unknown",
        }
    }

    fn total_size(&self) -> u64 {
        let raw_size = self.header_size + self.size * 8;
        (raw_size + 7) & !7 // round up to 8-byte alignment
    }
}

impl fmt::Display for ObjectHeader {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}(id={}, data=0x{:x}, fields={}, opaque={}, fwd={})",
            self.type_name(),
            self.type_id,
            self.type_data,
            self.size,
            self.opaque,
            self.forwarding,
        )
    }
}

// ── LZ4 decompression ───────────────────────────────────────────────────────

fn decompress_lz4(path: &Path) -> Vec<u8> {
    let data =
        fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    if data.len() < 4 {
        return vec![];
    }
    let uncompressed_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if uncompressed_size == 0 {
        return vec![];
    }
    lz4_flex::decompress(&data[4..], uncompressed_size)
        .unwrap_or_else(|e| panic!("Failed to decompress {}: {}", path.display(), e))
}

fn read_u64_at(data: &[u8], offset: usize) -> Option<u64> {
    if offset + 8 > data.len() {
        return None;
    }
    Some(u64::from_le_bytes(
        data[offset..offset + 8].try_into().unwrap(),
    ))
}

// ── Manifest parsing ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct StackInfo {
    index: usize,
    base: u64,
    fp: u64,
    #[allow(dead_code)]
    gc_ret: u64,
    uncompressed_size: usize,
}

#[derive(Debug, Clone)]
struct SavedFrameInfo {
    index: usize,
    sp: u64,
    #[allow(dead_code)]
    fp: u64,
    #[allow(dead_code)]
    ret: u64,
    uncompressed_size: usize,
}

#[derive(Debug, Clone)]
struct Snapshot {
    gc_count: u32,
    label: String,
    young_base: u64,
    young_offset: u64,
    old_base: u64,
    old_highmark: u64,
    #[allow(dead_code)]
    timestamp_ns: u64,
    stacks: Vec<StackInfo>,
    saved_frames: Vec<SavedFrameInfo>,
}

fn parse_hex(s: &str) -> u64 {
    let s = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(s, 16).unwrap_or_else(|e| panic!("Bad hex '{}': {}", s, e))
}

fn parse_manifest(path: &Path) -> Vec<Snapshot> {
    let text = fs::read_to_string(path).expect("Failed to read manifest");
    let mut snapshots: Vec<Snapshot> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        if line.starts_with("gc_count=") {
            let mut gc_count = 0u32;
            let mut label = String::new();
            let mut young_base = 0u64;
            let mut young_offset = 0u64;
            let mut old_base = 0u64;
            let mut old_highmark = 0u64;
            let mut timestamp_ns = 0u64;

            for part in line.split_whitespace() {
                if let Some((k, v)) = part.split_once('=') {
                    match k {
                        "gc_count" => gc_count = v.parse().unwrap(),
                        "label" => label = v.to_string(),
                        "young_base" => young_base = parse_hex(v),
                        "young_offset" => young_offset = v.parse().unwrap(),
                        "old_base" => old_base = parse_hex(v),
                        "old_highmark" => old_highmark = v.parse().unwrap(),
                        "timestamp_ns" => timestamp_ns = v.parse().unwrap(),
                        _ => {}
                    }
                }
            }
            snapshots.push(Snapshot {
                gc_count,
                label,
                young_base,
                young_offset,
                old_base,
                old_highmark,
                timestamp_ns,
                stacks: Vec::new(),
                saved_frames: Vec::new(),
            });
        } else if line.starts_with("stack ") && !line.starts_with("stack_pointer") {
            let snap = snapshots.last_mut().unwrap();
            let colon_pos = line.find(':').unwrap();
            let idx_str = &line["stack ".len()..colon_pos];
            let index: usize = idx_str.trim().parse().unwrap();

            let mut base = 0u64;
            let mut fp = 0u64;
            let mut gc_ret = 0u64;
            let mut uncompressed_size = 0usize;

            for part in line[colon_pos + 1..].split_whitespace() {
                if let Some((k, v)) = part.split_once('=') {
                    match k {
                        "base" => base = parse_hex(v),
                        "fp" => fp = parse_hex(v),
                        "gc_ret" => gc_ret = parse_hex(v),
                        "uncompressed_size" => uncompressed_size = v.parse().unwrap(),
                        _ => {}
                    }
                }
            }
            snap.stacks.push(StackInfo {
                index,
                base,
                fp,
                gc_ret,
                uncompressed_size,
            });
        } else if line.starts_with("saved_frame ") {
            let snap = snapshots.last_mut().unwrap();
            let colon_pos = line.find(':').unwrap();
            let idx_str = &line["saved_frame ".len()..colon_pos];
            let index: usize = idx_str.trim().parse().unwrap();

            let mut sp = 0u64;
            let mut fp = 0u64;
            let mut ret = 0u64;
            let mut uncompressed_size = 0usize;

            for part in line[colon_pos + 1..].split_whitespace() {
                if let Some((k, v)) = part.split_once('=') {
                    match k {
                        "sp" => sp = parse_hex(v),
                        "fp" => fp = parse_hex(v),
                        "ret" => ret = parse_hex(v),
                        "uncompressed_size" => uncompressed_size = v.parse().unwrap(),
                        _ => {}
                    }
                }
            }
            snap.saved_frames.push(SavedFrameInfo {
                index,
                sp,
                fp,
                ret,
                uncompressed_size,
            });
        }
    }

    snapshots
}

// ── Heap object iteration ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct HeapObject {
    offset: u64,
    header: ObjectHeader,
    fields: Vec<u64>,
}

fn iterate_heap_objects(data: &[u8]) -> Vec<HeapObject> {
    let mut objects = Vec::new();
    let mut pos = 0usize;

    while pos + 8 <= data.len() {
        let raw_header = read_u64_at(data, pos).unwrap();
        if raw_header == 0 {
            pos += 8;
            continue;
        }

        let forwarding = (raw_header >> 58) & 1 != 0;
        if forwarding {
            objects.push(HeapObject {
                offset: pos as u64,
                header: ObjectHeader {
                    raw: raw_header,
                    type_id: 0,
                    type_data: 0,
                    size: 0,
                    opaque: false,
                    marked: false,
                    large: false,
                    forwarding: true,
                    type_flags: 0,
                    header_size: 8,
                },
                fields: vec![],
            });
            pos += 8;
            continue;
        }

        let large = (raw_header >> 57) & 1 != 0;
        let large_size = if large && pos + 16 <= data.len() {
            Some(read_u64_at(data, pos + 8).unwrap())
        } else {
            None
        };

        let header = ObjectHeader::parse(raw_header, large_size);
        let total = header.total_size() as usize;

        if pos + total > data.len() {
            break;
        }

        let field_start = pos + header.header_size as usize;
        let mut fields = Vec::new();
        for i in 0..header.size as usize {
            if let Some(val) = read_u64_at(data, field_start + i * 8) {
                fields.push(val);
            }
        }

        objects.push(HeapObject {
            offset: pos as u64,
            header,
            fields,
        });

        pos += total;
    }

    objects
}

// ── Stack scanning ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SlotInfo {
    source: String,
    address: u64,
    #[allow(dead_code)]
    offset: usize,
    value: u64,
    tag: u64,
    untagged: u64,
}

fn scan_for_young_pointers(
    data: &[u8],
    _base: u64,
    fp: u64,
    young_base: u64,
    young_offset: u64,
    source_label: &str,
) -> Vec<SlotInfo> {
    let mut results = Vec::new();
    let num_slots = data.len() / 8;
    for i in 0..num_slots {
        if let Some(val) = read_u64_at(data, i * 8) {
            if is_heap_pointer(val) && is_in_young(val, young_base, young_offset) {
                let vaddr = fp + (i as u64) * 8;
                results.push(SlotInfo {
                    source: source_label.to_string(),
                    address: vaddr,
                    offset: i * 8,
                    value: val,
                    tag: tag_of(val),
                    untagged: untag(val),
                });
            }
        }
    }
    results
}

fn scan_saved_frame_for_young_pointers(
    data: &[u8],
    sp: u64,
    young_base: u64,
    young_offset: u64,
    source_label: &str,
) -> Vec<SlotInfo> {
    let mut results = Vec::new();
    let num_slots = data.len() / 8;
    for i in 0..num_slots {
        if let Some(val) = read_u64_at(data, i * 8) {
            if is_heap_pointer(val) && is_in_young(val, young_base, young_offset) {
                let vaddr = sp + (i as u64) * 8;
                results.push(SlotInfo {
                    source: source_label.to_string(),
                    address: vaddr,
                    offset: i * 8,
                    value: val,
                    tag: tag_of(val),
                    untagged: untag(val),
                });
            }
        }
    }
    results
}

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "beagle-heap-explorer")]
#[command(about = "Explore Beagle GC diagnostic snapshots")]
struct Cli {
    /// Path to the snapshot directory (containing manifest.txt)
    #[arg(global = true, default_value = "/tmp/beagle_gc_snapshots/20260225_012545")]
    dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show a summary of all snapshots
    Summary,

    /// Scan stacks for young-gen pointers and diff before/after
    Stale,

    /// Dump heap objects from a young or old gen snapshot
    Heap {
        /// Which file to dump (e.g. "gc_000_before_young")
        file: String,
        /// Show only objects containing young-gen pointers in their fields
        #[arg(long)]
        young_refs: bool,
        /// Limit number of objects shown
        #[arg(long, default_value = "50")]
        limit: usize,
    },

    /// Look up a specific address across all snapshots
    Lookup {
        /// Address to search for (hex, e.g. 0x3001abf30)
        #[arg(value_parser = parse_hex_arg)]
        address: u64,
    },

    /// Dump raw stack slots for a specific stack
    Stack {
        /// File stem, e.g. "gc_001_before_stack_2"
        file: String,
        /// Only show heap pointers
        #[arg(long)]
        heap_only: bool,
    },

    /// Show object at a specific offset within a heap file
    Object {
        /// Heap file stem, e.g. "gc_000_before_young"
        file: String,
        /// Byte offset of the object
        #[arg(value_parser = parse_hex_arg)]
        offset: u64,
    },

    /// Trace a specific tagged pointer: find it in stacks, heap fields, saved frames
    Trace {
        /// Tagged pointer value to trace (hex)
        #[arg(value_parser = parse_hex_arg)]
        value: u64,
    },

    /// Deep diagnosis: characterize where stale pointers live
    Diagnose,

    /// Stackmap-aware analysis: walk frames and classify stale slots as live or dead
    Stackmap,
}

fn parse_hex_arg(s: &str) -> Result<u64, String> {
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    u64::from_str_radix(s, 16).map_err(|e| format!("Invalid hex: {}", e))
}

fn main() {
    let cli = Cli::parse();
    let dir = &cli.dir;
    let manifest_path = dir.join("manifest.txt");
    let snapshots = parse_manifest(&manifest_path);

    match cli.command {
        Commands::Summary => cmd_summary(&snapshots),
        Commands::Stale => cmd_stale(dir, &snapshots),
        Commands::Heap {
            file,
            young_refs,
            limit,
        } => cmd_heap(dir, &snapshots, &file, young_refs, limit),
        Commands::Lookup { address } => cmd_lookup(dir, &snapshots, address),
        Commands::Stack { file, heap_only } => cmd_stack(dir, &snapshots, &file, heap_only),
        Commands::Object { file, offset } => cmd_object(dir, &file, offset),
        Commands::Trace { value } => cmd_trace(dir, &snapshots, value),
        Commands::Diagnose => cmd_diagnose(dir, &snapshots),
        Commands::Stackmap => cmd_stackmap(dir, &snapshots),
    }
}

// ── Commands ─────────────────────────────────────────────────────────────────

fn cmd_summary(snapshots: &[Snapshot]) {
    println!("{}", "=== Beagle GC Snapshot Summary ===".bold());
    println!();
    for snap in snapshots {
        let label_color = if snap.label == "before" {
            snap.label.yellow()
        } else {
            snap.label.green()
        };
        println!(
            "gc_count={} label={} young=[0x{:x}..+{}] old=[0x{:x}..+{}]",
            snap.gc_count.to_string().bold(),
            label_color,
            snap.young_base,
            snap.young_offset,
            snap.old_base,
            snap.old_highmark,
        );
        let active_stacks = snap
            .stacks
            .iter()
            .filter(|s| s.uncompressed_size > 0)
            .count();
        println!(
            "  {} active stacks, {} saved frames",
            active_stacks,
            snap.saved_frames.len()
        );
    }
}

fn cmd_stale(dir: &Path, snapshots: &[Snapshot]) {
    println!("{}", "=== Stale Young-Gen Pointer Analysis ===".bold());
    println!();

    let trigger_tagged = 0x1800d5f986u64;
    let trigger_untagged = 0x3001abf30u64;
    println!(
        "Trigger: stale pointer {} (untagged {})",
        format!("0x{:x}", trigger_tagged).red().bold(),
        format!("0x{:x}", trigger_untagged).red(),
    );
    println!();

    // Group snapshots by gc_count
    let mut by_gc: BTreeMap<u32, Vec<&Snapshot>> = BTreeMap::new();
    for s in snapshots {
        by_gc.entry(s.gc_count).or_default().push(s);
    }

    for (gc_count, snaps) in &by_gc {
        println!("{}", format!("── GC cycle {} ──", gc_count).bold());

        let before = snaps.iter().find(|s| s.label == "before");
        let after = snaps.iter().find(|s| s.label == "after");

        if let Some(before) = before {
            println!(
                "  Young gen: [0x{:x} .. 0x{:x}] ({} bytes)",
                before.young_base,
                before.young_base + before.young_offset,
                before.young_offset,
            );

            // Scan before stacks
            let mut before_young_slots: Vec<SlotInfo> = Vec::new();

            for si in &before.stacks {
                if si.uncompressed_size == 0 {
                    continue;
                }
                let filename =
                    format!("gc_{:03}_{}_stack_{}.lz4", gc_count, before.label, si.index);
                let path = dir.join(&filename);
                if !path.exists() {
                    continue;
                }
                let data = decompress_lz4(&path);
                let slots = scan_for_young_pointers(
                    &data,
                    si.base,
                    si.fp,
                    before.young_base,
                    before.young_offset,
                    &format!("stack_{}", si.index),
                );
                before_young_slots.extend(slots);
            }

            for sf in &before.saved_frames {
                if sf.uncompressed_size == 0 {
                    continue;
                }
                let filename = format!(
                    "gc_{:03}_{}_saved_frame_{}.lz4",
                    gc_count, before.label, sf.index
                );
                let path = dir.join(&filename);
                if !path.exists() {
                    continue;
                }
                let data = decompress_lz4(&path);
                let slots = scan_saved_frame_for_young_pointers(
                    &data,
                    sf.sp,
                    before.young_base,
                    before.young_offset,
                    &format!("saved_frame_{}", sf.index),
                );
                before_young_slots.extend(slots);
            }

            println!(
                "  {} found {} young-gen pointers in stacks/saved_frames",
                "BEFORE:".yellow(),
                before_young_slots.len(),
            );

            if let Some(after) = after {
                let mut stale_count = 0;

                // For each before slot, read the same virtual address from the after stack
                for slot in &before_young_slots {
                    // Find the after stack that contains this virtual address
                    let after_val = find_slot_in_snapshot(dir, after, &slot.source, slot.address);
                    if let Some(after_val) = after_val {
                        let after_still_young = is_heap_pointer(after_val)
                            && is_in_young(after_val, before.young_base, before.young_offset);
                        if after_still_young {
                            stale_count += 1;
                            println!(
                                "    {} {} @ 0x{:x}: before=0x{:x} ({} -> 0x{:x}), after=0x{:x} ({} -> 0x{:x}) {}",
                                "STALE!".red().bold(),
                                slot.source,
                                slot.address,
                                slot.value,
                                tag_name(slot.tag),
                                slot.untagged,
                                after_val,
                                tag_name(tag_of(after_val)),
                                untag(after_val),
                                "← NOT UPDATED".red(),
                            );
                        }
                    }
                }

                if stale_count == 0 {
                    println!(
                        "    {} All young-gen pointers were properly updated",
                        "OK".green().bold(),
                    );
                } else {
                    println!(
                        "    {} {} stale pointers found!",
                        "BUG:".red().bold(),
                        stale_count,
                    );
                }
            }
        }
        println!();
    }

    // Search for the trigger pointer in gc_count=1 before
    println!("{}", "── Trigger Pointer Search ──".bold());
    if let Some(snap) = snapshots
        .iter()
        .find(|s| s.gc_count == 1 && s.label == "before")
    {
        println!(
            "Searching gc_count=1 before for trigger value 0x{:x} (untagged 0x{:x})",
            trigger_tagged, trigger_untagged
        );

        search_snapshot_for_value(dir, snap, trigger_tagged, trigger_untagged);
    }
}

fn find_slot_in_snapshot(dir: &Path, snap: &Snapshot, source: &str, vaddr: u64) -> Option<u64> {
    if source.starts_with("stack_") {
        let idx: usize = source.strip_prefix("stack_").unwrap().parse().ok()?;
        let si = snap.stacks.iter().find(|s| s.index == idx)?;
        if si.uncompressed_size == 0 {
            return None;
        }
        let filename = format!(
            "gc_{:03}_{}_stack_{}.lz4",
            snap.gc_count, snap.label, si.index
        );
        let path = dir.join(&filename);
        if !path.exists() {
            return None;
        }
        let data = decompress_lz4(&path);
        // vaddr = fp + slot_index * 8
        if vaddr < si.fp {
            return None;
        }
        let byte_offset = (vaddr - si.fp) as usize;
        read_u64_at(&data, byte_offset)
    } else if source.starts_with("saved_frame_") {
        let idx: usize = source.strip_prefix("saved_frame_").unwrap().parse().ok()?;
        let sf = snap.saved_frames.iter().find(|s| s.index == idx)?;
        if sf.uncompressed_size == 0 {
            return None;
        }
        let filename = format!(
            "gc_{:03}_{}_saved_frame_{}.lz4",
            snap.gc_count, snap.label, sf.index
        );
        let path = dir.join(&filename);
        if !path.exists() {
            return None;
        }
        let data = decompress_lz4(&path);
        if vaddr < sf.sp {
            return None;
        }
        let byte_offset = (vaddr - sf.sp) as usize;
        read_u64_at(&data, byte_offset)
    } else {
        None
    }
}

fn search_snapshot_for_value(dir: &Path, snap: &Snapshot, tagged: u64, untagged: u64) {
    for si in &snap.stacks {
        if si.uncompressed_size == 0 {
            continue;
        }
        let filename = format!(
            "gc_{:03}_{}_stack_{}.lz4",
            snap.gc_count, snap.label, si.index
        );
        let path = dir.join(&filename);
        if !path.exists() {
            continue;
        }
        let data = decompress_lz4(&path);
        let num_slots = data.len() / 8;
        for i in 0..num_slots {
            if let Some(val) = read_u64_at(&data, i * 8) {
                if val == tagged || untag(val) == untagged {
                    let vaddr = si.fp + (i as u64) * 8;
                    println!(
                        "  {} stack_{} @ vaddr 0x{:x} (slot {}): value=0x{:x} tag={} untagged=0x{:x}",
                        "FOUND".red().bold(),
                        si.index,
                        vaddr,
                        i,
                        val,
                        tag_name(tag_of(val)),
                        untag(val),
                    );
                }
            }
        }
    }

    for sf in &snap.saved_frames {
        if sf.uncompressed_size == 0 {
            continue;
        }
        let filename = format!(
            "gc_{:03}_{}_saved_frame_{}.lz4",
            snap.gc_count, snap.label, sf.index
        );
        let path = dir.join(&filename);
        if !path.exists() {
            continue;
        }
        let data = decompress_lz4(&path);
        let num_slots = data.len() / 8;
        for i in 0..num_slots {
            if let Some(val) = read_u64_at(&data, i * 8) {
                if val == tagged || untag(val) == untagged {
                    let vaddr = sf.sp + (i as u64) * 8;
                    println!(
                        "  {} saved_frame_{} @ vaddr 0x{:x} (slot {}): value=0x{:x} tag={} untagged=0x{:x}",
                        "FOUND".red().bold(),
                        sf.index,
                        vaddr,
                        i,
                        val,
                        tag_name(tag_of(val)),
                        untag(val),
                    );
                }
            }
        }
    }

    // Search heap
    for heap_gen in &["young", "old"] {
        let filename = format!("gc_{:03}_{}_{}.lz4", snap.gc_count, snap.label, heap_gen);
        let path = dir.join(&filename);
        if !path.exists() {
            continue;
        }
        let data = decompress_lz4(&path);
        let objects = iterate_heap_objects(&data);
        for obj in &objects {
            if obj.header.forwarding || obj.header.opaque {
                continue;
            }
            for (fi, &field_val) in obj.fields.iter().enumerate() {
                if field_val == tagged || untag(field_val) == untagged {
                    let heap_base = if *heap_gen == "young" {
                        snap.young_base
                    } else {
                        snap.old_base
                    };
                    println!(
                        "  {} {} heap obj @ offset 0x{:x} (addr 0x{:x}), field[{}]: 0x{:016x} -- {}",
                        "FOUND".red().bold(),
                        heap_gen,
                        obj.offset,
                        heap_base + obj.offset,
                        fi,
                        field_val,
                        obj.header,
                    );
                }
            }
        }
    }
}

fn cmd_heap(dir: &Path, snapshots: &[Snapshot], file: &str, young_refs: bool, limit: usize) {
    let path = dir.join(format!("{}.lz4", file));
    let data = decompress_lz4(&path);
    println!("{} ({} bytes decompressed)", file.bold(), data.len());

    let (young_base, young_offset) = find_young_bounds(snapshots, file);

    let objects = iterate_heap_objects(&data);
    println!("Total objects: {}", objects.len());
    println!();

    let mut shown = 0;
    for obj in &objects {
        if young_refs {
            let has_young = obj
                .fields
                .iter()
                .any(|&v| is_heap_pointer(v) && is_in_young(v, young_base, young_offset));
            if !has_young {
                continue;
            }
        }

        print_object(obj, young_base, young_offset);
        shown += 1;
        if shown >= limit {
            println!(
                "... (showing {}/{}, use --limit to see more)",
                shown,
                objects.len()
            );
            break;
        }
    }
}

fn cmd_lookup(dir: &Path, snapshots: &[Snapshot], address: u64) {
    println!("{} 0x{:x}", "Looking up address".bold(), address);
    println!();

    for snap in snapshots {
        let prefix = format!("gc_{:03}_{}", snap.gc_count, snap.label);
        let label = format!("gc_count={} {}", snap.gc_count, snap.label);

        // Check if it's in young gen range
        if address >= snap.young_base && address < snap.young_base + snap.young_offset {
            let offset = address - snap.young_base;
            println!(
                "  {} {} - address is in young gen at offset 0x{:x}",
                "HIT".green().bold(),
                label,
                offset,
            );

            let young_path = dir.join(format!("{}_young.lz4", prefix));
            if young_path.exists() {
                let data = decompress_lz4(&young_path);
                let objects = iterate_heap_objects(&data);
                for obj in &objects {
                    let obj_end = obj.offset + obj.header.total_size();
                    if offset >= obj.offset && offset < obj_end {
                        print_object_detail(obj, snap);
                        break;
                    }
                }
            }
        }

        // Check if in old gen
        if address >= snap.old_base && address < snap.old_base + snap.old_highmark {
            let offset = address - snap.old_base;
            println!(
                "  {} {} - address is in old gen at offset 0x{:x}",
                "HIT".green().bold(),
                label,
                offset,
            );

            let old_path = dir.join(format!("{}_old.lz4", prefix));
            if old_path.exists() {
                let data = decompress_lz4(&old_path);
                let objects = iterate_heap_objects(&data);
                for obj in &objects {
                    let obj_end = obj.offset + obj.header.total_size();
                    if offset >= obj.offset && offset < obj_end {
                        print_object_detail(obj, snap);
                        break;
                    }
                }
            }
        }

        // Search stacks
        for si in &snap.stacks {
            if si.uncompressed_size == 0 {
                continue;
            }
            let filename = format!("{}_stack_{}.lz4", prefix, si.index);
            let path = dir.join(&filename);
            if !path.exists() {
                continue;
            }
            let data = decompress_lz4(&path);
            let num_slots = data.len() / 8;
            for i in 0..num_slots {
                if let Some(val) = read_u64_at(&data, i * 8) {
                    if untag(val) == address || val == address {
                        let vaddr = si.fp + (i as u64) * 8;
                        println!(
                            "  {} {} stack_{} slot {} (vaddr 0x{:x}): 0x{:016x} ({} -> 0x{:x})",
                            "REF".cyan().bold(),
                            label,
                            si.index,
                            i,
                            vaddr,
                            val,
                            tag_name(tag_of(val)),
                            untag(val),
                        );
                    }
                }
            }
        }

        // Search saved frames
        for sf in &snap.saved_frames {
            if sf.uncompressed_size == 0 {
                continue;
            }
            let filename = format!("{}_saved_frame_{}.lz4", prefix, sf.index);
            let path = dir.join(&filename);
            if !path.exists() {
                continue;
            }
            let data = decompress_lz4(&path);
            let num_slots = data.len() / 8;
            for i in 0..num_slots {
                if let Some(val) = read_u64_at(&data, i * 8) {
                    if untag(val) == address || val == address {
                        let vaddr = sf.sp + (i as u64) * 8;
                        println!(
                            "  {} {} saved_frame_{} slot {} (vaddr 0x{:x}): 0x{:016x} ({} -> 0x{:x})",
                            "REF".cyan().bold(),
                            label,
                            sf.index,
                            i,
                            vaddr,
                            val,
                            tag_name(tag_of(val)),
                            untag(val),
                        );
                    }
                }
            }
        }
    }
}

fn cmd_stack(dir: &Path, snapshots: &[Snapshot], file: &str, heap_only: bool) {
    let path = dir.join(format!("{}.lz4", file));
    let data = decompress_lz4(&path);

    let (young_base, young_offset) = find_young_bounds(snapshots, file);
    let (old_base, old_highmark) = find_old_bounds(snapshots, file);
    let (fp, base) = find_stack_fp_base(snapshots, file);

    println!(
        "{} ({} bytes, {} slots)",
        file.bold(),
        data.len(),
        data.len() / 8,
    );
    if fp != 0 {
        println!("  fp=0x{:x} base=0x{:x}", fp, base);
    }
    println!();

    let num_slots = data.len() / 8;
    for i in 0..num_slots {
        if let Some(val) = read_u64_at(&data, i * 8) {
            let tag = tag_of(val);
            let ut = untag(val);
            let is_hp = is_heap_pointer(val);

            if heap_only && !is_hp {
                continue;
            }

            let vaddr = if fp != 0 {
                fp + (i as u64) * 8
            } else {
                i as u64 * 8
            };

            let mut markers = Vec::new();
            if is_hp && is_in_young(val, young_base, young_offset) {
                markers.push("YOUNG".yellow().to_string());
            }
            if is_hp && is_in_old(val, old_base, old_highmark) {
                markers.push("OLD".green().to_string());
            }

            let marker_str = if markers.is_empty() {
                String::new()
            } else {
                format!(" <- {}", markers.join(", "))
            };

            println!(
                "  [{:4}] vaddr 0x{:x}: 0x{:016x} ({} -> 0x{:x}){}",
                i,
                vaddr,
                val,
                tag_name(tag),
                ut,
                marker_str
            );
        }
    }
}

fn cmd_object(dir: &Path, file: &str, offset: u64) {
    let path = dir.join(format!("{}.lz4", file));
    let data = decompress_lz4(&path);

    let off = offset as usize;
    if off + 8 > data.len() {
        println!(
            "Offset 0x{:x} is beyond file bounds ({})",
            offset,
            data.len()
        );
        return;
    }

    let raw_header = read_u64_at(&data, off).unwrap();
    let forwarding = (raw_header >> 58) & 1 != 0;

    if forwarding {
        let fwd_addr = raw_header & !TAG_MASK;
        println!(
            "Object at offset 0x{:x}: {} (forwarding pointer to 0x{:x})",
            offset,
            "FORWARDED".yellow().bold(),
            fwd_addr,
        );
        return;
    }

    let large = (raw_header >> 57) & 1 != 0;
    let large_size = if large && off + 16 <= data.len() {
        Some(read_u64_at(&data, off + 8).unwrap())
    } else {
        None
    };

    let header = ObjectHeader::parse(raw_header, large_size);
    println!(
        "Object at offset 0x{:x}: raw_header=0x{:016x}",
        offset, raw_header
    );
    println!("  {}", header);
    println!("  total_size={} bytes", header.total_size());
    println!();

    let field_start = off + header.header_size as usize;
    for i in 0..header.size as usize {
        if let Some(val) = read_u64_at(&data, field_start + i * 8) {
            let tag = tag_of(val);
            let ut = untag(val);
            if header.opaque {
                println!("  field[{}]: 0x{:016x} (raw data)", i, val);
            } else {
                println!(
                    "  field[{}]: 0x{:016x} ({} -> 0x{:x})",
                    i,
                    val,
                    tag_name(tag),
                    ut
                );
            }
        }
    }
}

fn cmd_trace(dir: &Path, snapshots: &[Snapshot], value: u64) {
    let addr = untag(value);
    println!(
        "{} value 0x{:x} (untagged 0x{:x}, tag={})",
        "Tracing".bold(),
        value,
        addr,
        tag_name(tag_of(value)),
    );
    println!();

    for snap in snapshots {
        let prefix = format!("gc_{:03}_{}", snap.gc_count, snap.label);
        let label = format!("gc_count={} {}", snap.gc_count, snap.label);

        // Search stacks
        for si in &snap.stacks {
            if si.uncompressed_size == 0 {
                continue;
            }
            let filename = format!("{}_stack_{}.lz4", prefix, si.index);
            let path = dir.join(&filename);
            if !path.exists() {
                continue;
            }
            let data = decompress_lz4(&path);
            let num_slots = data.len() / 8;
            for i in 0..num_slots {
                if let Some(val) = read_u64_at(&data, i * 8) {
                    if val == value || (is_heap_pointer(val) && untag(val) == addr) {
                        let vaddr = si.fp + (i as u64) * 8;
                        println!(
                            "  {} {} stack_{} slot {} (vaddr 0x{:x}): 0x{:016x} ({}){}",
                            "STACK".cyan().bold(),
                            label,
                            si.index,
                            i,
                            vaddr,
                            val,
                            tag_name(tag_of(val)),
                            if val == value {
                                ""
                            } else {
                                " (same addr, diff tag)"
                            },
                        );
                    }
                }
            }
        }

        // Search saved frames
        for sf in &snap.saved_frames {
            if sf.uncompressed_size == 0 {
                continue;
            }
            let filename = format!("{}_saved_frame_{}.lz4", prefix, sf.index);
            let path = dir.join(&filename);
            if !path.exists() {
                continue;
            }
            let data = decompress_lz4(&path);
            let num_slots = data.len() / 8;
            for i in 0..num_slots {
                if let Some(val) = read_u64_at(&data, i * 8) {
                    if val == value || (is_heap_pointer(val) && untag(val) == addr) {
                        let vaddr = sf.sp + (i as u64) * 8;
                        println!(
                            "  {} {} saved_frame_{} slot {} (vaddr 0x{:x}): 0x{:016x} ({})",
                            "SAVED".magenta().bold(),
                            label,
                            sf.index,
                            i,
                            vaddr,
                            val,
                            tag_name(tag_of(val)),
                        );
                    }
                }
            }
        }

        // Search heap object fields
        for heap_gen in &["young", "old"] {
            let filename = format!("{}_{}.lz4", prefix, heap_gen);
            let path = dir.join(&filename);
            if !path.exists() {
                continue;
            }
            let data = decompress_lz4(&path);
            let objects = iterate_heap_objects(&data);
            for obj in &objects {
                if obj.header.forwarding || obj.header.opaque {
                    continue;
                }
                for (fi, &field_val) in obj.fields.iter().enumerate() {
                    if field_val == value
                        || (is_heap_pointer(field_val) && untag(field_val) == addr)
                    {
                        let heap_base = if *heap_gen == "young" {
                            snap.young_base
                        } else {
                            snap.old_base
                        };
                        println!(
                            "  {} {} {} obj @ 0x{:x} (heap addr 0x{:x}), field[{}]: 0x{:016x} ({}) -- obj is {}",
                            "HEAP".yellow().bold(),
                            label,
                            heap_gen,
                            obj.offset,
                            heap_base + obj.offset,
                            fi,
                            field_val,
                            tag_name(tag_of(field_val)),
                            obj.header,
                        );
                    }
                }
            }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn find_young_bounds(snapshots: &[Snapshot], file: &str) -> (u64, u64) {
    let parts: Vec<&str> = file.split('_').collect();
    if parts.len() >= 3 {
        if let Ok(gc) = parts[1].parse::<u32>() {
            let label = parts[2];
            if let Some(snap) = snapshots
                .iter()
                .find(|s| s.gc_count == gc && s.label == label)
            {
                return (snap.young_base, snap.young_offset);
            }
        }
    }
    if let Some(snap) = snapshots.last() {
        (snap.young_base, snap.young_offset)
    } else {
        (0, 0)
    }
}

fn find_old_bounds(snapshots: &[Snapshot], file: &str) -> (u64, u64) {
    let parts: Vec<&str> = file.split('_').collect();
    if parts.len() >= 3 {
        if let Ok(gc) = parts[1].parse::<u32>() {
            let label = parts[2];
            if let Some(snap) = snapshots
                .iter()
                .find(|s| s.gc_count == gc && s.label == label)
            {
                return (snap.old_base, snap.old_highmark);
            }
        }
    }
    if let Some(snap) = snapshots.last() {
        (snap.old_base, snap.old_highmark)
    } else {
        (0, 0)
    }
}

fn find_stack_fp_base(snapshots: &[Snapshot], file: &str) -> (u64, u64) {
    let parts: Vec<&str> = file.split('_').collect();
    if parts.len() >= 5 {
        if let Ok(gc) = parts[1].parse::<u32>() {
            let label = parts[2];
            if parts[3] == "stack" {
                if let Ok(idx) = parts[4].parse::<usize>() {
                    if let Some(snap) = snapshots
                        .iter()
                        .find(|s| s.gc_count == gc && s.label == label)
                    {
                        if let Some(si) = snap.stacks.iter().find(|s| s.index == idx) {
                            return (si.fp, si.base);
                        }
                    }
                }
            }
        }
    }
    (0, 0)
}

fn print_object(obj: &HeapObject, young_base: u64, young_offset: u64) {
    if obj.header.forwarding {
        let fwd_addr = obj.header.raw & !TAG_MASK;
        println!(
            "  0x{:06x}: {} -> 0x{:x}",
            obj.offset,
            "FWD".yellow(),
            fwd_addr,
        );
        return;
    }

    println!(
        "  0x{:06x}: {} (raw=0x{:016x})",
        obj.offset, obj.header, obj.header.raw,
    );

    for (i, &field) in obj.fields.iter().enumerate() {
        let tag = tag_of(field);
        let ut = untag(field);
        let mut markers = Vec::new();
        if is_heap_pointer(field) && is_in_young(field, young_base, young_offset) {
            markers.push("YOUNG".yellow().to_string());
        }
        let marker_str = if markers.is_empty() {
            String::new()
        } else {
            format!(" <- {}", markers.join(", "))
        };

        if obj.header.opaque {
            println!("    [{}]: 0x{:016x} (raw){}", i, field, marker_str);
        } else {
            println!(
                "    [{}]: 0x{:016x} ({} -> 0x{:x}){}",
                i,
                field,
                tag_name(tag),
                ut,
                marker_str,
            );
        }
    }
}

// ── Stackmap ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct StackMapEntry {
    function_name: String,
    num_locals: usize,
    current_stack_size: usize,
}

impl StackMapEntry {
    fn active_slots(&self) -> usize {
        self.num_locals + self.current_stack_size
    }
}

fn parse_stackmap(path: &Path) -> HashMap<u64, StackMapEntry> {
    let mut map = HashMap::new();
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return map,
    };
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            if let Ok(addr) = u64::from_str_radix(
                parts[0].strip_prefix("0x").unwrap_or(parts[0]),
                16,
            ) {
                let function_name = parts[1].to_string();
                let num_locals: usize = parts[2].parse().unwrap_or(0);
                let current_stack_size: usize = parts[3].parse().unwrap_or(0);
                map.insert(
                    addr,
                    StackMapEntry {
                        function_name,
                        num_locals,
                        current_stack_size,
                    },
                );
            }
        }
    }
    map
}

/// Walk frames on a stack and return (frame_fp, ret_addr, active_slots_range)
/// active_slots_range is the set of virtual addresses that are "live" per the stackmap.
fn walk_frames_with_stackmap(
    data: &[u8],
    fp: u64,
    base: u64,
    gc_ret: u64,
    stackmap: &HashMap<u64, StackMapEntry>,
) -> Vec<(u64, u64, String, usize, Vec<u64>)> {
    // Returns: (frame_fp, ret_addr, function_name, active_slots, live_vaddrs)
    //
    // Correct frame walk algorithm:
    //   - Innermost frame at fp: use gc_ret to determine active slots
    //   - For each frame, read saved_fp=[fp+0] and ret_addr=[fp+8]
    //   - ret_addr is in the CALLER's code (at saved_fp)
    //   - stackmap[ret_addr] gives the CALLER's active slot count
    //   - The caller's live locals are at [saved_fp-8] through [saved_fp-8*active]
    let mut frames = Vec::new();

    // 1. Innermost frame: use gc_ret for active slots
    let (inner_name, inner_active) = if let Some(entry) = stackmap.get(&gc_ret) {
        (entry.function_name.clone(), entry.active_slots())
    } else {
        ("(unknown)".to_string(), 0)
    };

    let inner_live: Vec<u64> = (1..=inner_active as u64)
        .map(|i| fp.wrapping_sub(i * 8))
        .filter(|&addr| addr >= fp.wrapping_sub(inner_active as u64 * 8))
        .collect();

    frames.push((fp, gc_ret, format!("{} [INNERMOST]", inner_name), inner_active, inner_live));

    // 2. Walk the frame chain
    let mut current_fp = fp;
    loop {
        let offset = (current_fp - fp) as usize;
        let saved_fp = match read_u64_at(data, offset) {
            Some(v) => v,
            None => break,
        };
        let ret_addr = match read_u64_at(data, offset + 8) {
            Some(v) => v,
            None => break,
        };

        if saved_fp == 0 || saved_fp <= current_fp || saved_fp > base {
            // Record the last frame but no further walk
            let (func_name, _active) = if let Some(entry) = stackmap.get(&ret_addr) {
                (entry.function_name.clone(), entry.active_slots())
            } else {
                ("(unknown)".to_string(), 0)
            };
            frames.push((saved_fp, ret_addr, func_name, 0, vec![]));
            break;
        }

        // ret_addr is in the CALLER's code (the function at saved_fp).
        // stackmap[ret_addr] gives the caller's active slot count.
        let (func_name, active) = if let Some(entry) = stackmap.get(&ret_addr) {
            (entry.function_name.clone(), entry.active_slots())
        } else {
            ("(unknown)".to_string(), 0)
        };

        // The caller's live locals are at [saved_fp - 8] through [saved_fp - 8*active]
        let live_addrs: Vec<u64> = (1..=active as u64)
            .map(|i| saved_fp.wrapping_sub(i * 8))
            .filter(|&addr| addr >= fp) // don't go below the data start
            .collect();

        frames.push((saved_fp, ret_addr, func_name, active, live_addrs));

        current_fp = saved_fp;
    }

    frames
}

fn cmd_stackmap(dir: &Path, snapshots: &[Snapshot]) {
    println!("{}", "=== Stackmap-Aware Stale Pointer Analysis ===".bold());
    println!();

    let stackmap_path = dir.join("stack_map.txt");
    let stackmap = parse_stackmap(&stackmap_path);
    println!("Loaded {} stackmap entries", stackmap.len());

    let before = snapshots
        .iter()
        .find(|s| s.gc_count == 0 && s.label == "before")
        .expect("no gc_count=0 before");
    let after = snapshots
        .iter()
        .find(|s| s.gc_count == 0 && s.label == "after")
        .expect("no gc_count=0 after");

    // For each stack, walk frames, classify stale pointers as live or dead
    let mut total_stale_live = 0;
    let mut total_stale_dead = 0;

    // Deduplicate by physical stack — only analyze the deepest FP per base
    let mut deepest_by_base: BTreeMap<u64, &StackInfo> = BTreeMap::new();
    for si in &before.stacks {
        if si.uncompressed_size == 0 || si.fp == 0 {
            continue;
        }
        let entry = deepest_by_base.entry(si.base).or_insert(si);
        if si.fp < entry.fp {
            *entry = si;
        }
    }

    for (_base, si) in &deepest_by_base {
        let before_file = format!("gc_000_before_stack_{}.lz4", si.index);
        let after_file = format!("gc_000_after_stack_{}.lz4", si.index);
        let before_path = dir.join(&before_file);
        let after_path = dir.join(&after_file);
        if !before_path.exists() || !after_path.exists() {
            continue;
        }
        let before_data = decompress_lz4(&before_path);
        let after_data = decompress_lz4(&after_path);

        // Walk frames using stackmap (gc_ret determines innermost frame's active slots)
        let frames = walk_frames_with_stackmap(&before_data, si.fp, si.base, si.gc_ret, &stackmap);

        // Build set of all "live" virtual addresses across all frames
        let mut live_set: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for (_ffp, _ret, _name, _active, live_addrs) in &frames {
            for &addr in live_addrs {
                live_set.insert(addr);
            }
        }

        // Now find stale pointers and classify
        let num_slots = before_data.len() / 8;
        let mut stack_stale_live = Vec::new();
        let mut stack_stale_dead = Vec::new();

        for i in 0..num_slots {
            let bval = read_u64_at(&before_data, i * 8).unwrap_or(0);
            let aval = read_u64_at(&after_data, i * 8).unwrap_or(0);
            if is_heap_pointer(bval)
                && is_in_young(bval, before.young_base, before.young_offset)
                && bval == aval
            {
                let vaddr = si.fp + (i as u64) * 8;
                let is_live = live_set.contains(&vaddr);
                if is_live {
                    stack_stale_live.push((i, vaddr, bval));
                } else {
                    stack_stale_dead.push((i, vaddr, bval));
                }
            }
        }

        if stack_stale_live.is_empty() && stack_stale_dead.is_empty() {
            continue;
        }

        println!(
            "stack_{} (base=0x{:x} fp=0x{:x} gc_ret=0x{:x}):",
            si.index, si.base, si.fp, si.gc_ret,
        );
        println!(
            "  Frames found: {} (live addrs: {})",
            frames.len(),
            live_set.len(),
        );

        // Show frame info
        for (ffp, ret, name, active, _) in &frames {
            println!(
                "    fp=0x{:x} ret=0x{:x} {} (active_slots={})",
                ffp, ret, name, active,
            );
        }

        println!(
            "  {} stale in LIVE slots (stackmap says these should be scanned!)",
            stack_stale_live.len().to_string().red().bold(),
        );
        for (slot, vaddr, val) in &stack_stale_live {
            // Find which frame this belongs to
            let frame_info = frames.iter().find(|(ffp, _, _, active, _)| {
                let lo = ffp.wrapping_sub(*active as u64 * 8);
                *vaddr >= lo && *vaddr < *ffp
            });
            let frame_label = if let Some((ffp, _ret, name, _, _)) = frame_info {
                format!("in {} (fp=0x{:x})", name, ffp)
            } else {
                // Check innermost
                if let Some(entry) = stackmap.get(&si.gc_ret) {
                    let lo = si.fp.wrapping_sub(entry.active_slots() as u64 * 8);
                    if *vaddr >= lo && *vaddr < si.fp {
                        format!("in {} (innermost)", entry.function_name)
                    } else {
                        "frame unknown".to_string()
                    }
                } else {
                    "frame unknown".to_string()
                }
            };
            println!(
                "    slot {} vaddr 0x{:x}: 0x{:x} ({} -> 0x{:x}) -- {}",
                slot,
                vaddr,
                val,
                tag_name(tag_of(*val)),
                untag(*val),
                frame_label,
            );
        }

        println!(
            "  {} stale in DEAD slots (outside stackmap live range)",
            stack_stale_dead.len(),
        );

        total_stale_live += stack_stale_live.len();
        total_stale_dead += stack_stale_dead.len();
    }

    // Also check saved frames
    println!();
    println!("{}", "── Saved frames ──".bold());
    for sf in &before.saved_frames {
        if sf.uncompressed_size == 0 {
            continue;
        }
        let before_file = format!("gc_000_before_saved_frame_{}.lz4", sf.index);
        let after_file = format!("gc_000_after_saved_frame_{}.lz4", sf.index);
        let before_path = dir.join(&before_file);
        let after_path = dir.join(&after_file);
        if !before_path.exists() || !after_path.exists() {
            continue;
        }
        let before_data = decompress_lz4(&before_path);
        let after_data = decompress_lz4(&after_path);

        // Saved frame: data from sp to fp. The ret addr tells us the function.
        let active = if let Some(entry) = stackmap.get(&sf.ret) {
            println!(
                "saved_frame_{}: sp=0x{:x} fp=0x{:x} ret=0x{:x} {} (active_slots={})",
                sf.index, sf.sp, sf.fp, sf.ret, entry.function_name, entry.active_slots(),
            );
            entry.active_slots()
        } else {
            println!(
                "saved_frame_{}: sp=0x{:x} fp=0x{:x} ret=0x{:x} (NOT IN STACKMAP)",
                sf.index, sf.sp, sf.fp, sf.ret,
            );
            0
        };

        // Saved frame slots: the data is from sp to fp, so slots are at
        // sp, sp+8, sp+16, ..., fp-8
        // But from the stackmap perspective, live slots are fp-8, fp-16, ..., fp-8*active
        let num_slots = before_data.len() / 8;
        let mut stale_live = 0;
        let mut stale_dead = 0;
        for i in 0..num_slots {
            let bval = read_u64_at(&before_data, i * 8).unwrap_or(0);
            let aval = read_u64_at(&after_data, i * 8).unwrap_or(0);
            if is_heap_pointer(bval)
                && is_in_young(bval, before.young_base, before.young_offset)
                && bval == aval
            {
                let vaddr = sf.sp + (i as u64) * 8;
                // Is it in the live range? fp-8*active <= vaddr < fp
                let live_lo = sf.fp.wrapping_sub(active as u64 * 8);
                let is_live = vaddr >= live_lo && vaddr < sf.fp;
                if is_live {
                    stale_live += 1;
                    println!(
                        "  {} LIVE slot vaddr 0x{:x}: 0x{:x} ({} -> 0x{:x})",
                        "STALE".red().bold(),
                        vaddr,
                        bval,
                        tag_name(tag_of(bval)),
                        untag(bval),
                    );
                } else {
                    stale_dead += 1;
                }
            }
        }
        println!(
            "  {} stale LIVE, {} stale DEAD",
            stale_live, stale_dead,
        );
        total_stale_live += stale_live;
        total_stale_dead += stale_dead;
    }

    println!();
    println!("{}", "── Summary ──".bold());
    println!(
        "  Total stale in LIVE slots: {} {}",
        total_stale_live,
        if total_stale_live > 0 {
            "(BUG: GC should have updated these!)".red().bold().to_string()
        } else {
            "(all stale pointers are in dead slots - stackmap working correctly)".green().to_string()
        },
    );
    println!("  Total stale in DEAD slots: {} (expected - outside stackmap)", total_stale_dead);
}

fn cmd_diagnose(dir: &Path, snapshots: &[Snapshot]) {
    println!("{}", "=== Deep Diagnosis of GC Cycle 0 Stale Pointers ===".bold());
    println!();

    let before = snapshots
        .iter()
        .find(|s| s.gc_count == 0 && s.label == "before")
        .expect("no gc_count=0 before");
    let after = snapshots
        .iter()
        .find(|s| s.gc_count == 0 && s.label == "after")
        .expect("no gc_count=0 after");

    // ── 1. Group stacks by physical stack (base address) ──
    println!("{}", "── Physical stack mapping ──".bold());
    let mut phys_stacks: BTreeMap<u64, Vec<&StackInfo>> = BTreeMap::new();
    for si in &before.stacks {
        phys_stacks.entry(si.base).or_default().push(si);
    }
    for (base, stacks) in &phys_stacks {
        let indices: Vec<String> = stacks.iter().map(|s| format!("{}", s.index)).collect();
        let active: Vec<&&StackInfo> = stacks.iter().filter(|s| s.uncompressed_size > 0).collect();
        println!(
            "  base 0x{:x}: stacks [{}] ({} active)",
            base,
            indices.join(", "),
            active.len(),
        );
        for si in &active {
            println!(
                "    stack_{}: fp=0x{:x} size={} (range 0x{:x}..0x{:x})",
                si.index,
                si.fp,
                si.uncompressed_size,
                si.fp,
                si.base,
            );
        }
    }
    println!();

    // ── 2. Per-stack analysis: how many young ptrs, how many updated, how many stale ──
    println!("{}", "── Per-stack young pointer update analysis ──".bold());

    // Collect unique stale vaddrs
    let mut all_stale_vaddrs: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut stale_by_phys: BTreeMap<u64, Vec<(u64, u64)>> = BTreeMap::new(); // base -> [(vaddr, value)]

    for si in &before.stacks {
        if si.uncompressed_size == 0 {
            continue;
        }
        let before_file = format!("gc_000_before_stack_{}.lz4", si.index);
        let after_file = format!("gc_000_after_stack_{}.lz4", si.index);
        let before_path = dir.join(&before_file);
        let after_path = dir.join(&after_file);
        if !before_path.exists() || !after_path.exists() {
            continue;
        }
        let before_data = decompress_lz4(&before_path);
        let after_data = decompress_lz4(&after_path);

        let num_slots = before_data.len() / 8;
        let mut young_count = 0;
        let mut updated_count = 0;
        let mut stale_count = 0;
        let mut stale_slots: Vec<(usize, u64, u64)> = Vec::new(); // (slot, before, after)

        for i in 0..num_slots {
            let bval = read_u64_at(&before_data, i * 8).unwrap_or(0);
            let aval = read_u64_at(&after_data, i * 8).unwrap_or(0);

            if is_heap_pointer(bval)
                && is_in_young(bval, before.young_base, before.young_offset)
            {
                young_count += 1;
                if bval != aval {
                    updated_count += 1;
                } else {
                    stale_count += 1;
                    let vaddr = si.fp + (i as u64) * 8;
                    stale_slots.push((i, bval, aval));
                    all_stale_vaddrs.insert(vaddr);
                    stale_by_phys
                        .entry(si.base)
                        .or_default()
                        .push((vaddr, bval));
                }
            }
        }

        if young_count > 0 {
            let status = if stale_count == 0 {
                "OK".green().bold().to_string()
            } else {
                format!("{} stale", stale_count).red().bold().to_string()
            };
            println!(
                "  stack_{} (base=0x{:x} fp=0x{:x}): {} young ptrs, {} updated, {} [{}]",
                si.index, si.base, si.fp, young_count, updated_count, status,
                if stale_count > 0 && updated_count > 0 {
                    "PARTIAL"
                } else if stale_count > 0 {
                    "NONE UPDATED"
                } else {
                    "ALL UPDATED"
                }
            );

            // Show frame boundary analysis for stale slots
            if stale_count > 0 && stale_count <= 5 {
                for (slot, bval, _aval) in &stale_slots {
                    let vaddr = si.fp + (*slot as u64) * 8;
                    println!(
                        "    slot {}: vaddr 0x{:x} = fp+0x{:x}, val=0x{:x} ({} -> 0x{:x})",
                        slot,
                        vaddr,
                        vaddr - si.fp,
                        bval,
                        tag_name(tag_of(*bval)),
                        untag(*bval),
                    );
                }
            }
        }
    }

    // saved frames
    for sf in &before.saved_frames {
        if sf.uncompressed_size == 0 {
            continue;
        }
        let before_file = format!("gc_000_before_saved_frame_{}.lz4", sf.index);
        let after_file = format!("gc_000_after_saved_frame_{}.lz4", sf.index);
        let before_path = dir.join(&before_file);
        let after_path = dir.join(&after_file);
        if !before_path.exists() || !after_path.exists() {
            continue;
        }
        let before_data = decompress_lz4(&before_path);
        let after_data = decompress_lz4(&after_path);

        let num_slots = before_data.len() / 8;
        let mut young_count = 0;
        let mut updated_count = 0;
        let mut stale_count = 0;

        for i in 0..num_slots {
            let bval = read_u64_at(&before_data, i * 8).unwrap_or(0);
            let aval = read_u64_at(&after_data, i * 8).unwrap_or(0);
            if is_heap_pointer(bval)
                && is_in_young(bval, before.young_base, before.young_offset)
            {
                young_count += 1;
                if bval != aval {
                    updated_count += 1;
                } else {
                    stale_count += 1;
                }
            }
        }
        if young_count > 0 {
            let status = if stale_count == 0 {
                "OK".green().bold().to_string()
            } else {
                format!("{} stale", stale_count).red().bold().to_string()
            };
            println!(
                "  saved_frame_{} (sp=0x{:x} fp=0x{:x}): {} young ptrs, {} updated, {} [{}]",
                sf.index, sf.sp, sf.fp, young_count, updated_count, status,
                if stale_count > 0 && updated_count > 0 {
                    "PARTIAL"
                } else if stale_count > 0 {
                    "NONE UPDATED"
                } else {
                    "ALL UPDATED"
                }
            );
        }
    }

    println!();
    println!("  Total unique stale virtual addresses: {}", all_stale_vaddrs.len());
    println!();

    // ── 3. Deduplicate by physical address ──
    println!("{}", "── Stale pointers by physical stack ──".bold());
    for (base, slots) in &stale_by_phys {
        let mut unique: BTreeMap<u64, u64> = BTreeMap::new();
        for (vaddr, val) in slots {
            unique.insert(*vaddr, *val);
        }
        println!(
            "  Physical stack 0x{:x}: {} unique stale vaddrs",
            base,
            unique.len(),
        );

        // Show address range
        if let (Some(lo), Some(hi)) = (unique.keys().next(), unique.keys().next_back()) {
            println!("    Range: 0x{:x} .. 0x{:x}", lo, hi);

            // Find the innermost (lowest FP) and outermost (highest FP) stacks for this base
            let stacks_on_base: Vec<&&StackInfo> = phys_stacks
                .get(base)
                .unwrap()
                .iter()
                .filter(|s| s.uncompressed_size > 0)
                .collect();
            let lowest_fp = stacks_on_base.iter().map(|s| s.fp).min().unwrap_or(0);
            let highest_fp = stacks_on_base.iter().map(|s| s.fp).max().unwrap_or(0);
            println!(
                "    Stack FP range: 0x{:x} (deepest) .. 0x{:x} (shallowest) .. 0x{:x} (base)",
                lowest_fp, highest_fp, base,
            );

            // Are stale addrs above or below the shallowest FP?
            let above_all_fps = unique.keys().filter(|&&v| v > highest_fp).count();
            let between_fps = unique
                .keys()
                .filter(|&&v| v >= lowest_fp && v <= highest_fp)
                .count();
            let below_all_fps = unique.keys().filter(|&&v| v < lowest_fp).count();
            println!(
                "    Stale locations: {} above shallowest FP, {} between FPs, {} below deepest FP",
                above_all_fps, between_fps, below_all_fps,
            );
        }
    }

    println!();

    // ── 4. Check old gen for refs to stale addresses ──
    println!("{}", "── Old gen references to stale young-gen addresses ──".bold());
    let old_after_path = dir.join("gc_000_after_old.lz4");
    if old_after_path.exists() {
        let data = decompress_lz4(&old_after_path);
        let objects = iterate_heap_objects(&data);
        let mut old_to_young_count = 0;
        for obj in &objects {
            if obj.header.forwarding || obj.header.opaque {
                continue;
            }
            for (_fi, &field_val) in obj.fields.iter().enumerate() {
                if is_heap_pointer(field_val)
                    && is_in_young(field_val, before.young_base, before.young_offset)
                {
                    old_to_young_count += 1;
                }
            }
        }
        println!(
            "  Old gen (after GC 0) has {} objects, {} fields pointing to young gen",
            objects.len(),
            old_to_young_count,
        );
    }

    // ── 5. Check what the stale young-gen addresses point to ──
    println!();
    println!("{}", "── What are the stale objects? ──".bold());
    let young_before_path = dir.join("gc_000_before_young.lz4");
    if young_before_path.exists() {
        let data = decompress_lz4(&young_before_path);

        // Get unique stale target addresses
        let mut stale_targets: BTreeMap<u64, usize> = BTreeMap::new(); // untagged addr -> count
        for slots in stale_by_phys.values() {
            for (_vaddr, val) in slots {
                *stale_targets.entry(untag(*val)).or_insert(0) += 1;
            }
        }

        let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
        for (addr, ref_count) in &stale_targets {
            if *addr < before.young_base {
                continue;
            }
            let offset = addr - before.young_base;
            if offset as usize + 8 > data.len() {
                continue;
            }
            let raw = read_u64_at(&data, offset as usize).unwrap_or(0);
            let fwd = (raw >> 58) & 1 != 0;
            if fwd {
                *by_type.entry("FORWARDED (object was copied to old gen)".to_string()).or_insert(0) += 1;
                // Show a few
                if *ref_count > 5 {
                    let fwd_target = raw >> 3; // forwarding pointer uses same shift scheme
                    println!(
                        "    0x{:x}: FORWARDED -> 0x{:x} ({} stack refs)",
                        addr, fwd_target, ref_count,
                    );
                }
            } else {
                let header = ObjectHeader::parse(raw, None);
                let key = format!(
                    "{}(type_data=0x{:x}, fields={}, opaque={})",
                    header.type_name(),
                    header.type_data,
                    header.size,
                    header.opaque,
                );
                *by_type.entry(key.clone()).or_insert(0) += 1;
            }
        }

        println!("  {} unique target addresses:", stale_targets.len());
        for (type_desc, count) in &by_type {
            println!("    {}: {}", count, type_desc);
        }
    }

    println!();

    // ── 6. Check which properly-updated pointers look like ──
    println!("{}", "── Properly updated pointers (sample) ──".bold());
    // Find a stack that has BOTH stale and updated pointers
    for si in &before.stacks {
        if si.uncompressed_size == 0 {
            continue;
        }
        let before_file = format!("gc_000_before_stack_{}.lz4", si.index);
        let after_file = format!("gc_000_after_stack_{}.lz4", si.index);
        let before_path = dir.join(&before_file);
        let after_path = dir.join(&after_file);
        if !before_path.exists() || !after_path.exists() {
            continue;
        }
        let before_data = decompress_lz4(&before_path);
        let after_data = decompress_lz4(&after_path);

        let num_slots = before_data.len() / 8;
        let mut updated_samples: Vec<(usize, u64, u64)> = Vec::new();
        let mut stale_samples: Vec<(usize, u64, u64)> = Vec::new();

        for i in 0..num_slots {
            let bval = read_u64_at(&before_data, i * 8).unwrap_or(0);
            let aval = read_u64_at(&after_data, i * 8).unwrap_or(0);
            if is_heap_pointer(bval)
                && is_in_young(bval, before.young_base, before.young_offset)
            {
                if bval != aval {
                    if updated_samples.len() < 5 {
                        updated_samples.push((i, bval, aval));
                    }
                } else if stale_samples.len() < 5 {
                    stale_samples.push((i, bval, aval));
                }
            }
        }

        if !updated_samples.is_empty() && !stale_samples.is_empty() {
            println!("  stack_{} has both updated and stale pointers:", si.index);
            println!("    Updated (before -> after):");
            for (slot, bval, aval) in &updated_samples {
                let vaddr = si.fp + (*slot as u64) * 8;
                println!(
                    "      slot {} (vaddr 0x{:x} = fp+0x{:x}): 0x{:x} ({} -> 0x{:x}) => 0x{:x} ({} -> 0x{:x})",
                    slot,
                    vaddr,
                    vaddr - si.fp,
                    bval,
                    tag_name(tag_of(*bval)),
                    untag(*bval),
                    aval,
                    tag_name(tag_of(*aval)),
                    untag(*aval),
                );
            }
            println!("    Stale (unchanged):");
            for (slot, bval, _aval) in &stale_samples {
                let vaddr = si.fp + (*slot as u64) * 8;
                println!(
                    "      slot {} (vaddr 0x{:x} = fp+0x{:x}): 0x{:x} ({} -> 0x{:x})",
                    slot,
                    vaddr,
                    vaddr - si.fp,
                    bval,
                    tag_name(tag_of(*bval)),
                    untag(*bval),
                );
            }
            break; // just show one example stack
        }
    }
}

fn print_object_detail(obj: &HeapObject, snap: &Snapshot) {
    println!("    Object at offset 0x{:x}: {}", obj.offset, obj.header);
    for (i, &field) in obj.fields.iter().enumerate() {
        let tag = tag_of(field);
        let ut = untag(field);
        let marker = if is_heap_pointer(field)
            && is_in_young(field, snap.young_base, snap.young_offset)
        {
            " <- YOUNG".yellow().to_string()
        } else {
            String::new()
        };
        println!(
            "      field[{}]: 0x{:016x} ({} -> 0x{:x}){}",
            i,
            field,
            tag_name(tag),
            ut,
            marker
        );
    }
}
