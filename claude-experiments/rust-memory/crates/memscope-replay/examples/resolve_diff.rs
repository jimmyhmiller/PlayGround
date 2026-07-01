//! Differential correctness check (plan §4.A): resolve every site of a recording
//! with BOTH the addr2line oracle and the constant-memory resolver, and report
//! any divergence in recovered type/shape or frames.
//!
//!   cargo run --release --example resolve_diff -- <recording.mscope> [max_sites]
//!
//! Acceptance: 0 mismatches on (shape, element_type) and boundary name+loc.

use memscope_replay::read_recording_raw;
use memscope_symbols::{resolve_raw_sites_addr2line, resolve_raw_sites_targeted};

fn main() {
    let mut args = std::env::args().skip(1);
    let file = args.next().expect("usage: resolve_diff <recording> [max_sites]");
    let max: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    let rec = read_recording_raw(&file).expect("read recording");
    let mut sites: Vec<(u32, Vec<u64>)> =
        rec.raw_sites.iter().map(|(k, v)| (*k, v.clone())).collect();
    sites.sort_by_key(|(k, _)| *k);
    sites.truncate(max);
    let exe = std::path::Path::new(&rec.exe);
    eprintln!("exe: {}\nslide: {:#x}\nsites: {}", rec.exe, rec.slide, sites.len());

    eprintln!("[oracle] addr2line resolving…");
    let a = resolve_raw_sites_addr2line(exe, rec.slide, &sites).expect("oracle");
    eprintln!("[new] targeted resolving…");
    let b = resolve_raw_sites_targeted(exe, rec.slide, &sites).expect("targeted");

    let (mut total, mut shape_mm, mut type_mm, mut bound_mm, mut frames_mm) = (0, 0, 0, 0, 0);
    let mut shown = 0;
    let frame_sig = |f: &memscope_proto::Frame| {
        (f.function.clone().unwrap_or_default(), f.file.clone().unwrap_or_default(), f.line.unwrap_or(0))
    };
    let boundary = |frames: &[memscope_proto::Frame]| {
        frames.iter().find(|f| {
            !memscope_replay::is_std_frame(&memscope_replay::clean_frame(
                f.function.as_deref().unwrap_or(""),
            ))
        }).map(frame_sig)
    };

    for (id, ra) in &a {
        total += 1;
        let Some(rb) = b.get(id) else {
            frames_mm += 1;
            continue;
        };
        let shape_bad = ra.shape != rb.shape;
        let type_bad = ra.element_type != rb.element_type;
        let bound_bad = boundary(&ra.frames) != boundary(&rb.frames);
        let frames_bad = ra.frames.iter().map(frame_sig).collect::<Vec<_>>()
            != rb.frames.iter().map(frame_sig).collect::<Vec<_>>();
        shape_mm += shape_bad as usize;
        type_mm += type_bad as usize;
        bound_mm += bound_bad as usize;
        frames_mm += frames_bad as usize;

        if (shape_bad || type_bad || bound_bad) && shown < 4 {
            shown += 1;
            let dump = |label: &str, fr: &[memscope_proto::Frame]| {
                eprintln!("  {label} frames ({}):", fr.len());
                for f in fr {
                    eprintln!(
                        "    {}{} ({}:{})",
                        if f.inlined { "[inl] " } else { "      " },
                        f.function.as_deref().unwrap_or("<none>"),
                        f.file.as_deref().unwrap_or(""),
                        f.line.unwrap_or(0)
                    );
                }
            };
            eprintln!("\nMISMATCH site {id}: oracle shape={:?} ty={:?} | new shape={:?} ty={:?}",
                ra.shape, ra.element_type, rb.shape, rb.element_type);
            dump("oracle", &ra.frames);
            dump("new   ", &rb.frames);
        }
    }

    println!("\n=== resolve_diff ===");
    println!("sites compared:       {total}");
    println!("shape mismatches:     {shape_mm}");
    println!("element_type mism.:   {type_mm}");
    println!("boundary mismatches:  {bound_mm}");
    println!("full-frame mismatches:{frames_mm}  ({:.2}% of sites)", 100.0 * frames_mm as f64 / total.max(1) as f64);
    let critical = shape_mm + type_mm + bound_mm;
    println!("\nCRITICAL (shape+type+boundary): {critical}  -> {}", if critical == 0 { "PASS" } else { "FAIL" });
}
