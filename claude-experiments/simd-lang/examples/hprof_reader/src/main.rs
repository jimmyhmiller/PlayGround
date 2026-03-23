use memmap2::Mmap;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::time::Instant;

#[link(name = "hprof_parser", kind = "static")]
extern "C" {
    fn build_directory(
        buf_alloc: *const u8, buf_align: *const u8, buf_offset: i64, buf_size: i64, buf_stride: i64,
        buf_len: f64,
        class_ids_alloc: *mut i64, class_ids_align: *mut i64, class_ids_offset: i64, class_ids_size: i64, class_ids_stride: i64,
        offsets_alloc: *mut i64, offsets_align: *mut i64, offsets_offset: i64, offsets_size: i64, offsets_stride: i64,
        lengths_alloc: *mut i64, lengths_align: *mut i64, lengths_offset: i64, lengths_size: i64, lengths_stride: i64,
        strides_alloc: *mut i64, strides_align: *mut i64, strides_offset: i64, strides_size: i64, strides_stride: i64,
        out_alloc: *mut i64, out_align: *mut i64, out_offset: i64, out_size: i64, out_stride: i64,
    );

    fn find_and_count(
        buf_alloc: *const u8, buf_align: *const u8, buf_offset: i64, buf_size: i64, buf_stride: i64,
        buf_len: f64, name_w1: f64, name_w2: f64, name_len: f64,
        out_alloc: *mut i64, out_align: *mut i64, out_offset: i64, out_size: i64, out_stride: i64,
    );
}

fn memref(ptr: *mut i64, len: usize) -> (*mut i64, *mut i64, i64, i64, i64) {
    (ptr, ptr, 0, len as i64, 1)
}

fn read_u32_be(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3]])
}

fn read_u64_be(buf: &[u8], off: usize) -> u64 {
    u64::from_be_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3], buf[off+4], buf[off+5], buf[off+6], buf[off+7]])
}

/// Find class_object_id for a class name by walking UTF8 + LOAD_CLASS records.
/// Single pass: find the matching UTF8 string_id, then find the LOAD_CLASS with that string_id.
fn find_class_id(buf: &[u8], target: &str) -> Option<u64> {
    let id_size = read_u32_be(buf, 19) as usize;
    let target_bytes = target.as_bytes();
    let mut target_sid: Option<u64> = None;
    let mut offset = 31;
    while offset + 9 <= buf.len() {
        let tag = buf[offset];
        let body_len = read_u32_be(buf, offset + 5) as usize;
        match tag {
            0x01 => {
                // UTF8: check if text matches target
                let text_len = body_len.saturating_sub(id_size);
                if text_len == target_bytes.len() {
                    let text_start = offset + 9 + id_size;
                    if text_start + text_len <= buf.len() && &buf[text_start..text_start+text_len] == target_bytes {
                        let sid = if id_size == 8 { read_u64_be(buf, offset+9) } else { read_u32_be(buf, offset+9) as u64 };
                        target_sid = Some(sid);
                    }
                }
            }
            0x02 if target_sid.is_some() => {
                // LOAD_CLASS: check if class_name_string_id matches
                let nsid = if id_size == 8 { read_u64_be(buf, offset+9+4+id_size+4) } else { read_u32_be(buf, offset+9+4+id_size+4) as u64 };
                if Some(nsid) == target_sid {
                    let cid = if id_size == 8 { read_u64_be(buf, offset+13) } else { read_u32_be(buf, offset+13) as u64 };
                    return Some(cid);
                }
            }
            _ => {}
        }
        offset += 9 + body_len;
    }
    None
}

fn encode_name(name: &str) -> (u64, u64) {
    let b = name.as_bytes();
    let mut w1 = 0u64;
    let mut w2 = 0u64;
    for i in 0..8.min(b.len()) { w1 |= (b[i] as u64) << (56 - i*8); }
    for i in 0..8.min(b.len().saturating_sub(8)) { w2 |= (b[8+i] as u64) << (56 - i*8); }
    (w1, w2)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).unwrap_or_else(|| { eprintln!("Usage: hprof-reader <path> [class]"); std::process::exit(1); });
    let class_name = args.get(2).map(|s| s.as_str()).unwrap_or("java/lang/String");

    let file = File::open(path).expect("failed to open file");
    let mmap = unsafe { Mmap::map(&file).expect("failed to mmap") };
    let len = mmap.len();
    println!("File: {} ({:.2} GB)", path, len as f64 / 1e9);

    // --- Approach 1: Single-pass find_and_count ---
    let (w1, w2) = encode_name(class_name);
    let mut out1 = vec![0i64; 2];
    let t1 = Instant::now();
    unsafe {
        find_and_count(
            mmap.as_ptr(), mmap.as_ptr(), 0, len as i64, 1,
            f64::from_bits(len as u64), f64::from_bits(w1), f64::from_bits(w2),
            f64::from_bits(class_name.len() as u64),
            out1.as_mut_ptr(), out1.as_mut_ptr(), 0, 2, 1,
        );
    }
    let d1 = t1.elapsed();
    println!("\n=== Single-pass (SIMD) ===");
    println!("{}: {} instances ({} segments) in {:.1} ms",
        class_name, out1[0], out1[1], d1.as_secs_f64() * 1000.0);

    // --- Approach 2: Build directory once, then O(1) lookups ---
    let max_segs = 100_000usize;
    let mut dir_cids = vec![0i64; max_segs];
    let mut dir_offs = vec![0i64; max_segs];
    let mut dir_lens = vec![0i64; max_segs];
    let mut dir_strs = vec![0i64; max_segs];
    let mut dir_out = vec![0i64; 1];

    let t2 = Instant::now();
    unsafe {
        let (ca, cb, co, cs, ct) = memref(dir_cids.as_mut_ptr(), max_segs);
        let (oa, ob, oo, os, ot) = memref(dir_offs.as_mut_ptr(), max_segs);
        let (la, lb, lo, ls, lt) = memref(dir_lens.as_mut_ptr(), max_segs);
        let (sa, sb, so, ss, st) = memref(dir_strs.as_mut_ptr(), max_segs);
        let (da, db, d_o, ds, dt) = memref(dir_out.as_mut_ptr(), 1);
        build_directory(
            mmap.as_ptr(), mmap.as_ptr(), 0, len as i64, 1,
            f64::from_bits(len as u64),
            ca, cb, co, cs, ct,
            oa, ob, oo, os, ot,
            la, lb, lo, ls, lt,
            sa, sb, so, ss, st,
            da, db, d_o, ds, dt,
        );
    }
    let d2_build = t2.elapsed();
    let num_segs = dir_out[0] as usize;

    // Build HashMap index: class_id -> list of (offset, length, stride)
    let t3 = Instant::now();
    let mut index: HashMap<i64, Vec<(i64, i64, i64)>> = HashMap::new();
    for i in 0..num_segs {
        index.entry(dir_cids[i])
            .or_default()
            .push((dir_offs[i], dir_lens[i], dir_strs[i]));
    }
    let d2_index = t3.elapsed();

    // Find class_id (Rust string matching)
    let t4 = Instant::now();
    let class_id = find_class_id(&mmap, class_name).unwrap_or(0);
    let d2_lookup_class = t4.elapsed();

    // O(1) lookup
    let t5 = Instant::now();
    let (inst_count, seg_count) = if let Some(entries) = index.get(&(class_id as i64)) {
        let count: i64 = entries.iter().map(|(_, l, s)| l / s).sum();
        (count, entries.len())
    } else {
        (0, 0)
    };
    let d2_query = t5.elapsed();

    println!("\n=== Directory approach ===");
    println!("Build directory (SIMD):  {:.1} ms  ({} instance segments)", d2_build.as_secs_f64() * 1000.0, num_segs);
    println!("Build HashMap (Rust):    {:.3} ms", d2_index.as_secs_f64() * 1000.0);
    println!("Find class_id (Rust):    {:.1} ms  (0x{:016x})", d2_lookup_class.as_secs_f64() * 1000.0, class_id);
    println!("Query (Rust):            {:.3} ms", d2_query.as_secs_f64() * 1000.0);
    println!("{}: {} instances ({} segments)",
        class_name, inst_count, seg_count);
    println!("Total:                   {:.1} ms",
        (d2_build + d2_index + d2_lookup_class + d2_query).as_secs_f64() * 1000.0);
    println!("  (after build: query any class in {:.3} ms)",
        (d2_lookup_class + d2_query).as_secs_f64() * 1000.0);
}
