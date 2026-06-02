use ai_lang::typecheck::decode_scheme;
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let mut bad = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) != Some("type") { continue; }
        let bytes = std::fs::read(&p).unwrap();
        match decode_scheme(&bytes) {
            Ok(_) => {}
            Err(err) => { bad += 1; println!("BAD ({} bytes) {:?}: {}", bytes.len(), p.file_name().unwrap(), err); }
        }
    }
    println!("done, {} bad", bad);
}
