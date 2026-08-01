//! Demand-driven route compilation state for Next development servers.

use std::collections::BTreeSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Lazy route compilation for the Next dev server: what is compiled right now, what has
/// been asked for, and where the orchestrator serving it is listening.
///
/// A cold start used to compile every route of the app into three whole-app graphs before
/// answering anything — 7.7 s on cal.com, for a browser that then asks for one route.
/// This is the state that lets it compile on demand instead: the proxy thread matches an
/// incoming request against the app's full pattern table (a directory walk, free) and
/// BLOCKS it until the graphs contain that route, while the build thread widens the scope
/// and rebuilds. A request is never answered 404 because its route has not been compiled
/// yet — it waits, which is the one behaviour that makes laziness invisible.
pub struct LazyRoutes {
    /// Every matchable pattern of the app, endpoints before pages (the orchestrator's
    /// precedence). Fixed for the process; a route file added later is a structural
    /// change, which rebuilds everything and re-derives the scope.
    patterns: Vec<crate::next_adapter::RoutePattern>,
    state: Mutex<LazyState>,
    /// Signalled when a build lands, when a want is registered, or on failure.
    changed: std::sync::Condvar,
}

/// The mutable half of [`LazyRoutes`].
#[derive(Default)]
struct LazyState {
    /// The scope the live graphs were built from, or `None` before the first build has
    /// landed (there is no orchestrator yet, so even a request matching no route waits).
    /// This is the ONE source of truth for "can this request be served now": asking the
    /// scope itself, rather than tracking a parallel set of compiled paths, is what keeps
    /// the answer right for endpoints, which a scope can compile wholesale.
    scope: Option<crate::next_adapter::RouteScope>,
    /// Paths asked for and not yet compiled, drained by the build thread.
    wanted: BTreeSet<String>,
    /// Requests currently being served, and when the last one finished. The fill waits for
    /// both to say "nothing is happening": a 6-second whole-app compile running alongside
    /// the very first render steals the cores that render needs, and measurably did —
    /// cal.com's first document went from 1.4s to 2.5s when the two overlapped.
    in_flight: usize,
    last_activity: Option<Instant>,
    /// Set when a request arrived that no pattern matched: there is nothing specific to
    /// compile, so the only correct answer is "compile everything and let the app 404 it
    /// properly".
    wanted_everything: bool,
    /// The orchestrator's loopback port, 0 until the first build boots it.
    node_port: u16,
    /// Set while the emitted output is being swapped under the orchestrator: for those few
    /// milliseconds there is no server to forward to, so requests wait rather than see a
    /// half-swapped tree.
    swapping: bool,
    /// The last build failure, so a waiter fails loudly instead of hanging forever.
    failure: Option<String>,
}

/// How long a request waits for its route to compile before giving up. Generous: it
/// covers a cold cal.com build of the whole app, and the alternative to waiting is a
/// wrong answer.
const LAZY_WAIT_TIMEOUT: Duration = Duration::from_secs(300);

impl LazyRoutes {
    pub fn new(patterns: Vec<crate::next_adapter::RoutePattern>) -> Self {
        LazyRoutes {
            patterns,
            state: Mutex::new(LazyState::default()),
            changed: std::sync::Condvar::new(),
        }
    }

    /// The pattern that will serve `request_path`, if any.
    pub fn match_path(&self, request_path: &str) -> Option<&crate::next_adapter::RoutePattern> {
        self.patterns
            .iter()
            .find(|pattern| pattern.matches(request_path))
    }

    /// Block until `request_path` can be served, and return the orchestrator's port.
    ///
    /// Returns `Ok(None)` when this dev server is not compiling lazily at all (every
    /// route is already built), so the caller uses its own port and pays nothing.
    pub fn ensure(&self, request_path: &str) -> Result<Option<u16>, String> {
        let matched = self
            .match_path(request_path)
            .map(|pattern| (pattern.url_path.clone(), pattern.kind));
        let mut state = self.state.lock().expect("lazy routes mutex");
        loop {
            if let Some(failure) = &state.failure {
                return Err(failure.clone());
            }
            let ready = match (&state.scope, &matched) {
                (Some(scope), Some((url_path, kind))) => scope.includes(url_path, *kind),
                // Nothing matched: the app itself owns the answer (a 404, a rewrite, a
                // static file), so a running orchestrator is all that is needed.
                (Some(_), None) => true,
                (None, _) => false,
            };
            if ready && !state.swapping {
                // Count this as activity BEFORE the caller takes its in-flight guard. The
                // fill starts as soon as the server looks idle, and the instant after a
                // build lands is exactly when a released request is about to render but has
                // not registered itself yet — without this the fill starts underneath it.
                state.last_activity = Some(Instant::now());
                return Ok(Some(state.node_port));
            }
            // Register the want and wake the build thread. Re-registered on every loop
            // turn, which is harmless (a set) and covers a build that landed without
            // covering this route.
            match &matched {
                Some((url_path, _)) => {
                    state.wanted.insert(url_path.clone());
                }
                None => state.wanted_everything = true,
            }
            self.changed.notify_all();
            let (next, timeout) = self
                .changed
                .wait_timeout(state, LAZY_WAIT_TIMEOUT)
                .expect("lazy routes condvar");
            state = next;
            if timeout.timed_out() {
                return Err(format!(
                    "timed out after {}s waiting for {request_path} to compile",
                    LAZY_WAIT_TIMEOUT.as_secs(),
                ));
            }
        }
    }

    /// Wait until there is something to build, and return the paths asked for. `None`
    /// means "build everything" (an unmatched request, or an explicit fill).
    pub fn take_wanted(&self) -> Option<BTreeSet<String>> {
        let mut state = self.state.lock().expect("lazy routes mutex");
        while state.wanted.is_empty() && !state.wanted_everything {
            state = self.changed.wait(state).expect("lazy routes condvar");
        }
        if state.wanted_everything {
            state.wanted_everything = false;
            state.wanted.clear();
            return None;
        }
        Some(std::mem::take(&mut state.wanted))
    }

    /// Mark a request as being served, so the fill knows the server is busy. The returned
    /// guard decrements on drop, including when the connection handler bails out early.
    pub fn serving_request(&self) -> InFlightGuard<'_> {
        let mut state = self.state.lock().expect("lazy routes mutex");
        state.in_flight += 1;
        state.last_activity = Some(Instant::now());
        InFlightGuard { lazy: self }
    }

    /// Block until nothing has been served for `quiet`, so a background build does not
    /// compete with a render the visitor is waiting on. Gives up after `budget` so a page
    /// that polls forever cannot postpone the fill indefinitely.
    ///
    /// Returns AT ONCE if anything is already waiting for a wider build, and that is not an
    /// optimization — it is what stops a deadlock. An in-flight render can itself depend on
    /// a route this build does not have: cal.com's server components call the app's own API
    /// over HTTP, so the document render sits in flight waiting for an endpoint, while a
    /// fill that waits for the server to go idle waits for that same render. Demand that is
    /// already blocked outranks idleness.
    pub fn wait_for_quiet(&self, quiet: Duration, budget: Duration) {
        let deadline = Instant::now() + budget;
        let mut state = self.state.lock().expect("lazy routes mutex");
        loop {
            if !state.wanted.is_empty() || state.wanted_everything {
                return;
            }
            let idle_for = match (state.in_flight, state.last_activity) {
                (0, Some(last)) => last.elapsed(),
                (0, None) => quiet,
                _ => Duration::ZERO,
            };
            if idle_for >= quiet || Instant::now() >= deadline {
                return;
            }
            // Poll rather than wait on the condvar: the interesting event is a request
            // FINISHING, which the guard signals, plus the passage of time.
            let (next, _) = self
                .changed
                .wait_timeout(state, (quiet - idle_for).min(Duration::from_millis(50)))
                .expect("lazy routes condvar");
            state = next;
        }
    }

    /// Hold every request: the emitted output is about to be replaced under the running
    /// orchestrator. Released by the next [`landed`](LazyRoutes::landed).
    pub fn begin_swap(&self) {
        let mut state = self.state.lock().expect("lazy routes mutex");
        state.swapping = true;
    }

    /// Publish a landed build: the scope it compiled and the orchestrator port serving it.
    pub fn landed(&self, scope: &crate::next_adapter::RouteScope, node_port: u16) {
        let mut state = self.state.lock().expect("lazy routes mutex");
        state.swapping = false;
        let complete = *scope == crate::next_adapter::RouteScope::All;
        state.scope = Some(scope.clone());
        state.node_port = node_port;
        // A landing releases whoever was waiting, and they are about to render. Counting
        // the landing itself as activity is what makes the idle wait race-free: the
        // released request does not have to win a lock against the thread that is deciding
        // whether the server is idle.
        state.last_activity = Some(Instant::now());
        state.failure = None;
        if complete {
            state.wanted.clear();
            state.wanted_everything = false;
        }
        self.changed.notify_all();
    }

    /// Publish a build failure, releasing every waiter with the error rather than
    /// letting them hang on a build that will never land.
    pub fn failed(&self, error: &str) {
        let mut state = self.state.lock().expect("lazy routes mutex");
        state.swapping = false;
        state.failure = Some(error.to_string());
        self.changed.notify_all();
    }

    /// Whether anything is still uncompiled (so the proxy keeps consulting this).
    pub fn incomplete(&self) -> bool {
        self.state.lock().expect("lazy routes mutex").scope
            != Some(crate::next_adapter::RouteScope::All)
    }

    pub fn partition_wanted(&self, wanted: BTreeSet<String>) -> (Vec<String>, Vec<String>) {
        wanted.into_iter().partition(|path| {
            self.patterns
                .iter()
                .find(|pattern| pattern.url_path == *path)
                .is_none_or(|pattern| pattern.kind == crate::next_adapter::PatternKind::Page)
        })
    }

    pub fn wait_for_first_wants(
        &self,
        grace: Duration,
        coalesce: Duration,
    ) -> Option<BTreeSet<String>> {
        {
            let mut state = self.state.lock().expect("lazy routes mutex");
            let deadline = Instant::now() + grace;
            while state.wanted.is_empty() && !state.wanted_everything {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return None;
                }
                let (next, _) = self
                    .changed
                    .wait_timeout(state, remaining)
                    .expect("lazy routes condvar");
                state = next;
            }
        }
        std::thread::sleep(coalesce);
        self.take_wanted()
    }
}

impl diffpack_web::node_proxy::RouteGate for LazyRoutes {
    fn incomplete(&self) -> bool {
        self.incomplete()
    }

    fn ensure(&self, path: &str) -> Result<Option<u16>, String> {
        self.ensure(path)
    }

    fn serving_request(&self) -> Box<dyn diffpack_web::node_proxy::RequestGuard + '_> {
        Box::new(self.serving_request())
    }
}

/// Decrements [`LazyState::in_flight`] when a request handler returns, however it returns.
pub struct InFlightGuard<'a> {
    lazy: &'a LazyRoutes,
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        let mut state = self.lazy.state.lock().expect("lazy routes mutex");
        state.in_flight = state.in_flight.saturating_sub(1);
        state.last_activity = Some(Instant::now());
        self.lazy.changed.notify_all();
    }
}

#[cfg(test)]
mod lazy_route_tests {
    use super::*;
    use crate::next_adapter::{PatternKind, RoutePattern, RouteScope};
    use std::sync::Arc;

    fn lazy() -> LazyRoutes {
        // Endpoints before pages, the orchestrator's own precedence — and the ordering that
        // matters here, since cal.com's `/api/**` endpoints would otherwise be swallowed by
        // a root catch-all page.
        LazyRoutes::new(vec![
            RoutePattern::parse("/api/auth/[...nextauth]", PatternKind::Endpoint),
            RoutePattern::parse("/api/trpc/[trpc]", PatternKind::Endpoint),
            RoutePattern::parse("/auth/login", PatternKind::Page),
            RoutePattern::parse("/[user]", PatternKind::Page),
        ])
    }

    #[test]
    fn a_request_matches_the_pattern_that_will_serve_it_endpoints_first() {
        let lazy = lazy();
        assert_eq!(
            lazy.match_path("/api/trpc/viewer.me")
                .map(|p| p.url_path.as_str()),
            Some("/api/trpc/[trpc]"),
        );
        assert_eq!(
            lazy.match_path("/auth/login").map(|p| p.url_path.as_str()),
            Some("/auth/login"),
        );
        // A path only the catch-all page can serve.
        assert_eq!(
            lazy.match_path("/jimmy").map(|p| p.url_path.as_str()),
            Some("/[user]")
        );
        // Nothing matches a path with too many segments for any pattern.
        assert_eq!(
            lazy.match_path("/a/b/c/d").map(|p| p.url_path.as_str()),
            None
        );
    }

    /// Regression: readiness has to be decided by asking the SCOPE, not by looking a path up
    /// in a set of compiled routes. A scope that compiles every endpoint knows
    /// `/api/trpc/[trpc]` is ready without ever naming it, and an earlier version — which
    /// recorded only the compiled PAGES — left every endpoint request blocked until the whole
    /// app had been compiled, turning a 5s first response into a 13s one.
    #[test]
    fn an_endpoint_request_is_ready_when_the_scope_compiles_all_endpoints() {
        let lazy = lazy();
        lazy.landed(&RouteScope::pages(["/auth/login".to_string()]), 4242);
        assert_eq!(lazy.ensure("/api/trpc/viewer.me"), Ok(Some(4242)));
        assert_eq!(lazy.ensure("/auth/login"), Ok(Some(4242)));
        // A path no pattern matches needs a running orchestrator and nothing else: the app
        // owns that answer (its own 404, a rewrite, a static file).
        assert_eq!(lazy.ensure("/a/b/c/d"), Ok(Some(4242)));
    }

    #[test]
    fn a_request_for_an_uncompiled_page_waits_and_is_released_by_the_build_that_covers_it() {
        let lazy = Arc::new(lazy());
        lazy.landed(&RouteScope::pages(["/auth/login".to_string()]), 4242);
        // `/[user]` is not compiled: this blocks, and registers the want that tells the
        // build thread what to compile.
        let waiter = {
            let lazy = Arc::clone(&lazy);
            std::thread::spawn(move || lazy.ensure("/jimmy"))
        };
        // The build thread sees the want (blocking until there is one) and lands a scope
        // covering it.
        assert_eq!(
            lazy.take_wanted(),
            Some(["/[user]".to_string()].into_iter().collect())
        );
        lazy.landed(&RouteScope::All, 4243);
        assert_eq!(waiter.join().expect("waiter thread"), Ok(Some(4243)));
        // Everything is compiled now, so the proxy stops consulting this at all.
        assert!(!lazy.incomplete());
    }

    #[test]
    fn a_build_failure_releases_waiters_with_the_error_instead_of_hanging_them() {
        let lazy = Arc::new(lazy());
        lazy.landed(&RouteScope::pages(["/auth/login".to_string()]), 4242);
        let waiter = {
            let lazy = Arc::clone(&lazy);
            std::thread::spawn(move || lazy.ensure("/jimmy"))
        };
        assert!(lazy.take_wanted().is_some());
        lazy.failed("the ssr graph did not compile");
        assert_eq!(
            waiter.join().expect("waiter thread"),
            Err("the ssr graph did not compile".to_string()),
        );
    }

    /// The fill must not start while a request is already blocked on a wider build: an
    /// in-flight render can itself depend on a route this build does not have (cal.com's
    /// server components call the app's own API over HTTP), so waiting for the server to go
    /// idle would wait for a render that is waiting for the fill.
    #[test]
    fn waiting_demand_outranks_idleness_so_the_fill_cannot_deadlock_behind_a_render() {
        let lazy = lazy();
        lazy.landed(&RouteScope::pages(["/auth/login".to_string()]), 4242);
        {
            let mut state = lazy.state.lock().expect("lazy routes mutex");
            state.in_flight = 1;
            state.wanted.insert("/[user]".to_string());
        }
        let started = Instant::now();
        lazy.wait_for_quiet(Duration::from_secs(30), Duration::from_secs(30));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "a blocked want must make the fill start at once, waited {:?}",
            started.elapsed(),
        );
    }

    #[test]
    fn the_fill_waits_for_an_in_flight_render_when_nothing_is_blocked() {
        let lazy = lazy();
        lazy.landed(&RouteScope::pages(["/auth/login".to_string()]), 4242);
        let guard = lazy.serving_request();
        let started = Instant::now();
        // Nothing is waiting for a build, so the request in flight holds the fill off — up
        // to the budget, which is what stops a page that polls forever from postponing it.
        lazy.wait_for_quiet(Duration::from_secs(30), Duration::from_millis(150));
        assert!(started.elapsed() >= Duration::from_millis(100));
        drop(guard);
    }
}
