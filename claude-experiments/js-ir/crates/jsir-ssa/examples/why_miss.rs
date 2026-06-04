use std::collections::BTreeMap;
use std::fs;
fn main() {
    let list = std::env::args().nth(1).unwrap();
    let mut buckets: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
    for raw in fs::read_to_string(&list).unwrap().lines() {
        let name = raw.split_whitespace().next().unwrap_or("").trim();
        if name.is_empty() { continue; }
        let src = match fs::read_to_string(format!("oracle/fixtures/{name}")) { Ok(s)=>s, Err(_)=>continue };
        let ir = match jsir_swc::source_to_ir(&src) { Ok(i)=>i, Err(_)=>{ bump(&mut buckets,"source_to_ir err",name); continue } };
        let mut cfg = match jsir_ssa::lower::lower_function(&ir) { Ok(c)=>c, Err(e)=>{ bump(&mut buckets,&format!("lower: {}",head(&e)),name); continue } };
        jsir_ssa::ssa::construct(&mut cfg);
        jsir_ssa::constfold::fold_constants(&mut cfg);
        let r = jsir_ssa::aliasing_ranges::analyze(&cfg);
        let infos = jsir_ssa::scopes::analyze(&cfg, &r);
        match jsir_ssa::memoize_plan::memoize_inplace(&cfg, &infos, &r, &ir) {
            Ok(_) => bump(&mut buckets, "OK (structure differs?)", name),
            Err(e) => bump(&mut buckets, &format!("inplace: {}", head(&e)), name),
        }
    }
    let mut rows: Vec<_> = buckets.iter().collect();
    rows.sort_by(|a,b| b.1.0.cmp(&a.1.0));
    for (k,(n,ex)) in rows { println!("{n:4}  {k}\n       e.g. {}", ex.join(", ")); }
}
fn head(e:&str)->String { e.split([':','(']).take(2).collect::<Vec<_>>().join(":").chars().take(64).collect() }
fn bump(b:&mut BTreeMap<String,(usize,Vec<String>)>, k:&str, name:&str){ let e=b.entry(k.to_string()).or_default(); e.0+=1; if e.1.len()<3 {e.1.push(name.to_string());} }
