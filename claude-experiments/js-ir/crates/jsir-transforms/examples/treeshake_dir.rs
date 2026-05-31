use std::collections::BTreeMap;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = std::path::PathBuf::from(&args[1]);
    let entry = args[2].clone();
    let outdir = args.get(3).cloned();
    let mut sources: BTreeMap<String, String> = BTreeMap::new();
    fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut BTreeMap<String,String>) {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() { walk(&p, root, out); }
            else if p.extension().map_or(false,|x|x=="js") {
                let rel = p.strip_prefix(root).unwrap().to_string_lossy().to_string();
                if let Ok(s) = std::fs::read_to_string(&p) { out.insert(rel, s); }
            }
        }
    }
    walk(&root, &root, &mut sources);
    let r = jsir_transforms::tree_shake(&sources, &entry).unwrap();
    eprintln!("reachable={} dropped={}", r.stats.modules_reachable, r.stats.modules_dropped);
    if let Some(od) = outdir {
        for (k,v) in &r.modules {
            let p = std::path::Path::new(&od).join(k);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, v).unwrap();
        }
        eprintln!("wrote {} modules to {od}", r.modules.len());
    }
}
