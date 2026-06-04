use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::Reader;
const CORE: &str = "/Users/jimmyhmiller/Documents/Code/open-source/clojure/src/clj/clojure/core.clj";
fn load() -> Session {
    std::panic::set_hook(Box::new(|i| { eprintln!("[HOOK] {i}"); }));
    let src = std::fs::read_to_string(CORE).unwrap();
    let mut s = Session::new(); let mut bp = 0usize;
    loop {
        let sl = &src[bp..];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { let mut rr = Reader::new(sl); let f = rr.read(); (f, rr.byte_pos()) }));
        let (form, after) = match r { Ok((Ok(Some(f)), a)) => (f, a), _ => break };
        bp += after;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { s.eval_form(form); }));
    }
    s
}
#[test] #[ignore]
fn lazy() {
    let mut s = load();
    for src in ["(pr-str (map inc [1 2 3]))", "(into [] (map inc [1 2 3]))", "(vec (map inc [1 2 3]))", "(doall (map inc [1 2 3]))", "(reduce + (map inc [1 2 3 4]))", "(count (filter even? (range 10)))", "(apply str (map str [1 2 3]))"] {
        eprintln!("=== TRY {src}");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| clojure_jvm::runtime::pr_str_bits(s.eval_str(src))));
        eprintln!("=== {src} => {:?}", r);
    }
}
