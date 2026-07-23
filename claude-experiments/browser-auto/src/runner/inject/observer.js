// bat's transient watcher: notifies the runner (via the __batMutationTick
// binding) whenever the DOM mutates, coalesced to at most one call per
// animation frame. The runner then runs one-shot visibility checks on armed
// `expect appear` / `expect gone` watchers. We use our own observer because
// Playwright's injected waitFor poller can miss sub-200ms transients; a
// MutationObserver provably sees every insertion.
(() => {
  if (window.__batObserverInstalled) return;
  window.__batObserverInstalled = true;
  // monotonically increasing count of mutation batches — settlement reads it
  // page-side (same evaluate) to detect "the DOM changed during this window"
  window.__batMutationCount = 0;

  let scheduled = false;
  const notify = () => {
    window.__batMutationCount++;
    if (scheduled) return;
    scheduled = true;
    let fired = false;
    const fire = () => {
      if (fired) return;
      fired = true;
      scheduled = false;
      if (typeof window.__batMutationTick === "function") {
        window.__batMutationTick("");
      }
    };
    // coalesce per frame; fall back to a macrotask if rAF is starved/faked
    requestAnimationFrame(fire);
    setTimeout(fire, 32);
  };

  const start = () => {
    new MutationObserver(notify).observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      characterData: true,
    });
    notify();
  };

  if (document.documentElement) start();
  else addEventListener("DOMContentLoaded", start);
})();
