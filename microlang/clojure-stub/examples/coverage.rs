// Coverage check: for every clojure.* public var (from all_vars.txt), ask our
// runtime whether the qualified symbol is bound. One program, one runtime load.
use microlang::{LowBitModel, Runtime};
use microlang::code::TreeWalk;
fn main() {
    let vars = std::fs::read_to_string("/tmp/all_vars.txt").unwrap();
    let names: Vec<&str> = vars.lines().filter_map(|l| l.split('\t').next()).filter(|s| !s.is_empty()).collect();
    // build one program: for each name, print "name|true/false"
    let mut prog = String::from("(apply str (map (fn [n] (str n \"|\" (%global-bound? (symbol n)) \"\\n\")) (list ");
    for n in &names { prog.push_str(&format!("\"{n}\" ")); }
    prog.push_str(")))");
    let mut rt = Runtime::<LowBitModel>::new();
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = clojure_stub::run(&mut rt, &TreeWalk, &prog);
        clojure_stub::clj_str(&rt, r)
    })).unwrap_or_else(|e| format!("PANIC:{}", e.downcast_ref::<&str>().map(|s|s.to_string()).or_else(||e.downcast_ref::<String>().cloned()).unwrap_or("?".into())));
    // out is a quoted string; strip quotes
    let out = out.trim_matches('"').replace("\\n", "\n");
    let (mut have, mut miss) = (0, 0);
    let mut missing = Vec::new();
    for line in out.lines() {
        if let Some((name, b)) = line.split_once('|') {
            if b == "true" { have += 1; } else { miss += 1; missing.push(name.to_string()); }
        }
    }
    println!("COVERAGE: {}/{} bound ({} missing)", have, have+miss, miss);
    std::fs::write("/tmp/missing_vars.txt", missing.join("\n")).ok();
}
