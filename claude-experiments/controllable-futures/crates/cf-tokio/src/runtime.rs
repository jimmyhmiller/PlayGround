//! Mirrors `tokio::runtime`. Mini-redis only constructs a `Runtime` via the
//! `#[tokio::main]` macro path, which we redirect to our own. We expose
//! `Runtime` and `Builder` for completeness.

use std::future::Future;

pub struct Runtime {
    inner: cf_runtime::Runtime,
}

impl Runtime {
    pub fn new() -> std::io::Result<Self> {
        let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
        Ok(Self {
            inner: cf_runtime::Runtime::new(n),
        })
    }

    pub fn block_on<F: Future>(&self, fut: F) -> F::Output {
        self.inner.block_on(fut)
    }

    pub fn handle(&self) -> Handle {
        Handle {
            inner: self.inner.handle(),
        }
    }
}

#[derive(Clone)]
pub struct Handle {
    inner: cf_runtime::RuntimeHandle,
}

impl Handle {
    pub fn current() -> Handle {
        Handle {
            inner: cf_runtime::current(),
        }
    }

    /// Spawn a future onto this handle's runtime.
    pub fn spawn<F>(&self, fut: F) -> cf_runtime::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let loc = std::panic::Location::caller();
        self.inner.spawn(format!("spawn@{}:{}", loc.file(), loc.line()), fut)
    }

    /// Run a future to completion synchronously, blocking the calling
    /// thread. Used by code that holds a Handle and needs to bridge
    /// async→sync.
    pub fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        let _g = cf_runtime::runtime::enter(self.inner.clone());
        futures_lite::future::block_on(fut)
    }

    /// Install this handle as the current runtime on the calling thread.
    /// Returned guard removes it on drop.
    pub fn enter(&self) -> EnterGuard {
        EnterGuard {
            _inner: cf_runtime::runtime::enter(self.inner.clone()),
        }
    }
}

pub struct EnterGuard {
    _inner: cf_runtime::runtime::EnterGuard,
}

pub struct Builder {
    workers: usize,
}

impl Builder {
    pub fn new_multi_thread() -> Self {
        Self {
            workers: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(2),
        }
    }

    pub fn worker_threads(mut self, n: usize) -> Self {
        self.workers = n;
        self
    }

    pub fn enable_all(self) -> Self {
        self
    }

    pub fn enable_io(self) -> Self {
        self
    }

    pub fn enable_time(self) -> Self {
        self
    }

    pub fn build(self) -> std::io::Result<Runtime> {
        Ok(Runtime {
            inner: cf_runtime::Runtime::new(self.workers),
        })
    }
}
