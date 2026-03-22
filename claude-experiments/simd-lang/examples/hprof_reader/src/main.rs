use memmap2::Mmap;
use std::env;
use std::fs::File;
use std::time::Instant;

#[link(name = "hprof_parser", kind = "static")]
extern "C" {
    // vector<1xi64> params passed as f64 (NEON D register on aarch64)
    fn find_and_count(
        buf_alloc: *const u8, buf_align: *const u8, buf_offset: i64, buf_size: i64, buf_stride: i64,
        buf_len: f64,
        name_w1: f64,
        name_w2: f64,
        name_len: f64,
        out_alloc: *mut i64, out_align: *mut i64, out_offset: i64, out_size: i64, out_stride: i64,
    );
}

/// Encode a class name as two big-endian u64 words (first 8 bytes, second 8 bytes).
fn encode_name(name: &str) -> (u64, u64) {
    let bytes = name.as_bytes();
    let mut w1 = 0u64;
    let mut w2 = 0u64;
    for i in 0..8.min(bytes.len()) {
        w1 |= (bytes[i] as u64) << (56 - i * 8);
    }
    for i in 0..8.min(bytes.len().saturating_sub(8)) {
        w2 |= (bytes[8 + i] as u64) << (56 - i * 8);
    }
    (w1, w2)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: hprof-reader <path.hprof> [class_name]");
        std::process::exit(1);
    });
    let class_name = args.get(2).map(|s| s.as_str()).unwrap_or("java/lang/String");

    let file = File::open(path).expect("failed to open file");
    let mmap = unsafe { Mmap::map(&file).expect("failed to mmap") };
    let len = mmap.len();

    let (w1, w2) = encode_name(class_name);

    println!("File: {} ({:.2} GB)", path, len as f64 / 1e9);
    println!("Class: {} (w1=0x{:016x}, w2=0x{:016x}, len={})", class_name, w1, w2, class_name.len());

    let mut out = vec![0i64; 2];

    let start = Instant::now();
    unsafe {
        find_and_count(
            mmap.as_ptr(), mmap.as_ptr(), 0, len as i64, 1,
            f64::from_bits(len as u64),
            f64::from_bits(w1),
            f64::from_bits(w2),
            f64::from_bits(class_name.len() as u64),
            out.as_mut_ptr(), out.as_mut_ptr(), 0, 2, 1,
        );
    }
    let elapsed = start.elapsed();

    println!("Instances: {}", out[0]);
    println!("Segments:  {}", out[1]);
    println!("Time:      {:.1} ms ({:.2} GB/s)",
        elapsed.as_secs_f64() * 1000.0,
        len as f64 / elapsed.as_secs_f64() / 1e9);
}
