//! Verifies the patched-tokio hooks fire and emit events into a
//! cf-runtime Observer. Lives in its own crate to avoid the
//! cf-runtime ↔ tokio dev-dep cycle that would compile cf-runtime twice
//! and split the static observer into two distinct instances.

#[cfg(test)]
mod tests {
    use cf_runtime::hooks::{self, Observer};
    use std::time::Duration;

    #[test]
    fn hooks_fire_on_tokio_runtime() {
        let observer = Observer::new();
        hooks::register(observer.clone());

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            let h1 = tokio::spawn(async {
                tokio::time::sleep(Duration::from_millis(20)).await;
                42
            });
            let h2 = tokio::spawn(async { "done" });
            let _ = h1.await;
            let _ = h2.await;
        });

        let snap = observer.registry.snapshot();
        let log = observer.log.snapshot();

        eprintln!("registry has {} task(s):", snap.len());
        for t in &snap {
            eprintln!("  #{} state={:?} polls={}", t.id.0, t.state, t.poll_count);
        }
        eprintln!("log has {} event(s):", log.len());
        for e in log.iter().take(20) {
            eprintln!("  seq={} {:?}", e.seq, e.kind);
        }

        assert!(snap.len() >= 2);
        assert!(log.iter().any(|e| matches!(e.kind, cf_runtime::EventKind::Spawned { .. })));
        assert!(log.iter().any(|e| matches!(e.kind, cf_runtime::EventKind::PollStart)));
    }
}
