use cf_runtime::hooks::{self, Observer};

fn main() {
    let observer = Observer::new();
    hooks::register(observer.clone());

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        // Spawn 5 tasks that yield/wake themselves a few times.
        let mut handles = vec![];
        for i in 0..5 {
            handles.push(tokio::spawn(async move {
                for _ in 0..3 {
                    tokio::task::yield_now().await;
                }
                i
            }));
        }
        for h in handles { let _ = h.await; }
    });

    let log = observer.log.snapshot();
    println!("=== {} events ===", log.len());
    for e in log.iter().take(60) {
        println!("seq={:>3} task={:?} worker={:?} {:?}",
            e.seq, e.task, e.worker, e.kind);
    }
}
