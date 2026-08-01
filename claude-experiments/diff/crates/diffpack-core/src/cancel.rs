//! Cooperative cancellation for long-running graph and loader work.

#[derive(Copy, Clone)]
pub struct CancelToken<'a>(Option<&'a (dyn Fn() -> bool + Send + Sync)>);

impl<'a> CancelToken<'a> {
    pub fn never() -> Self {
        Self(None)
    }

    pub fn when(signal: &'a (dyn Fn() -> bool + Send + Sync)) -> Self {
        Self(Some(signal))
    }

    pub fn cancelled(&self) -> bool {
        self.0.is_some_and(|signal| signal())
    }
}
