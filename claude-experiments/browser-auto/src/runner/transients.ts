import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { BrowserContext, Locator } from "playwright";

/**
 * Event-driven transient watchers for `expect appear` / `expect gone`.
 *
 * Why not locator.waitFor armed before the action? Empirically, Playwright's
 * injected poller can miss elements that live <200ms (see scripts/toast-debug.ts:
 * five independent waitFor watchers all missed a 150ms toast that a page-side
 * MutationObserver recorded 15/15 times). One-shot evaluations never miss.
 *
 * So: an injected MutationObserver (src/runner/inject/observer.js) pings the
 * __batMutationTick binding at most once per frame; on every ping we run
 * one-shot isVisible() checks against the armed watchers. Waiting is driven by
 * DOM events — never by a poll interval the DSL could feel.
 */

const OBSERVER_PATH = join(dirname(fileURLToPath(import.meta.url)), "inject", "observer.js");

interface Watcher {
  kind: "appear" | "gone";
  locator: Locator;
  satisfied: boolean;
  checking: boolean;
  waiters: Array<() => void>;
}

export class TransientHub {
  private watchers = new Set<Watcher>();

  /** Install the observer + binding on a fresh context (before any page loads). */
  static async install(context: BrowserContext): Promise<TransientHub> {
    const hub = new TransientHub();
    await context.exposeBinding("__batMutationTick", () => {
      void hub.tick();
    });
    await context.addInitScript({ path: OBSERVER_PATH });
    return hub;
  }

  /** Arm before the action. Runs an immediate first check. */
  async arm(kind: "appear" | "gone", locator: Locator): Promise<Watcher> {
    const w: Watcher = { kind, locator, satisfied: false, checking: false, waiters: [] };
    this.watchers.add(w);
    await this.check(w);
    return w;
  }

  /** One binding ping = one round of one-shot checks. */
  async tick(): Promise<void> {
    const waiters = this.tickWaiters;
    this.tickWaiters = [];
    for (const resolve of waiters) resolve();
    await Promise.all([...this.watchers].map((w) => this.check(w)));
  }

  private async check(w: Watcher): Promise<void> {
    if (w.satisfied || w.checking) return;
    w.checking = true;
    try {
      const visible = await w.locator.first().isVisible();
      const hit = w.kind === "appear" ? visible : !visible;
      if (hit) {
        w.satisfied = true;
        for (const resolve of w.waiters) resolve();
        w.waiters = [];
      }
    } catch {
      // page navigating mid-check; the next tick re-checks
    } finally {
      w.checking = false;
    }
  }

  /** Await satisfaction until the deadline; returns the final verdict. */
  async waitFor(w: Watcher, deadlineMs: number): Promise<boolean> {
    await this.check(w);
    if (w.satisfied) return true;
    if (deadlineMs <= 0) return false;
    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, deadlineMs);
      w.waiters.push(() => {
        clearTimeout(timer);
        resolve();
      });
    });
    // final one-shot in case the last mutation raced the deadline
    await this.check(w);
    return w.satisfied;
  }

  /** Step boundary: drop this step's watchers. */
  clear(): void {
    this.watchers.clear();
  }

  private tickWaiters: Array<() => void> = [];

  /** Resolves on the next DOM-mutation tick, or after maxWaitMs (runner
   * physics — frameworks like React commit transitions with no preceding
   * network or frame signal, so effect evaluation re-checks on mutations). */
  waitForNextTick(maxWaitMs: number): Promise<void> {
    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve(), maxWaitMs);
      this.tickWaiters.push(() => {
        clearTimeout(timer);
        resolve();
      });
    });
  }
}
