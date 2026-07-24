import type { BrowserContext, Dialog, Download, Frame, Page } from "playwright";
import type { Effect } from "../dsl/ir.js";
import { matchPath, pathOf } from "./patterns.js";
import type { SettleMatcher } from "./settle.js";

/**
 * First-class support for the interactions beyond a single quiet page:
 * tabs/popups, native dialogs, downloads. Same rules as everything else in
 * bat — declared as effects, armed BEFORE the action, event-driven waits,
 * explainable outcomes.
 */

export interface DialogRecord {
  dialogType: string;
  message: string;
  response: "accept" | "dismiss";
  declared: boolean;
}

interface DialogHandler {
  effect: Effect & { type: "dialog" };
  message: string;
  response: "accept" | "dismiss";
  text: string | undefined;
  matched: DialogRecord | null;
  resolve: () => void;
  promise: Promise<void>;
}

/** Routes native dialogs to the step's declared responses. Undeclared dialogs
 * are dismissed and fail the step (unless `allow dialogs`). */
export class DialogRouter {
  private handlers: DialogHandler[] = [];
  /** dialogs observed this step (for traces) */
  records: DialogRecord[] = [];
  /** failures produced by undeclared dialogs */
  unmodeled: string[] = [];

  constructor(private allowDialogs: boolean) {}

  attach(page: Page): void {
    page.on("dialog", (dialog) => void this.route(dialog));
  }

  private async route(dialog: Dialog): Promise<void> {
    const handler = this.handlers.find((h) => !h.matched && dialog.message().includes(h.message));
    if (handler) {
      const record: DialogRecord = {
        dialogType: dialog.type(),
        message: dialog.message(),
        response: handler.response,
        declared: true,
      };
      handler.matched = record;
      this.records.push(record);
      if (handler.response === "accept") await dialog.accept(handler.text).catch(() => {});
      else await dialog.dismiss().catch(() => {});
      handler.resolve();
      return;
    }
    this.records.push({ dialogType: dialog.type(), message: dialog.message(), response: "dismiss", declared: false });
    if (!this.allowDialogs) {
      this.unmodeled.push(
        `the page opened a ${dialog.type()} dialog (${JSON.stringify(dialog.message().slice(0, 120))}) with no declared response — ` +
          `add 'expect dialog "${dialog.message().slice(0, 40)}" accept' (or dismiss) to this step, or 'allow dialogs' to the flow. It was dismissed.`,
      );
    }
    await dialog.dismiss().catch(() => {});
  }

  /** arm a declared response for this step; returns a settle-gating matcher */
  arm(effect: Effect & { type: "dialog" }, message: string, text: string | undefined): SettleMatcher & { handler: DialogHandler } {
    let resolve!: () => void;
    const promise = new Promise<void>((r) => {
      resolve = r;
    });
    const handler: DialogHandler = { effect, message, response: effect.response, text, matched: null, resolve, promise };
    this.handlers.push(handler);
    return {
      handler,
      result: {
        get matched() {
          return handler.matched;
        },
      },
      matchedAndFinished: promise,
      describe: () =>
        `declared 'expect dialog "${message}" ${effect.response}' but no matching dialog appeared`,
    };
  }

  stepBoundary(): void {
    this.handlers = [];
    this.records = [];
    this.unmodeled = [];
  }
}

/** Armed BEFORE the action: matches a page (new or existing) whose url matches. */
export class TabMatcher implements SettleMatcher {
  result: { matched: string | null } = { matched: null };
  matchedAndFinished: Promise<void>;
  private disposers: Array<() => void> = [];

  constructor(private pattern: string, context: BrowserContext) {
    let resolveMatched!: () => void;
    this.matchedAndFinished = new Promise((resolve) => {
      resolveMatched = resolve;
    });

    const tryMatch = (url: string): void => {
      if (this.result.matched) return;
      if (!url || url === "about:blank") return;
      const { path, query } = pathOf(url);
      if (!matchPath(this.pattern, path, query)) return;
      this.result.matched = url;
      resolveMatched();
    };

    const watchPage = (page: Page): void => {
      tryMatch(page.url());
      const onNav = (frame: Frame): void => {
        if (frame.parentFrame() === null) tryMatch(frame.url());
      };
      page.on("framenavigated", onNav);
      this.disposers.push(() => page.off("framenavigated", onNav));
    };

    for (const page of context.pages()) watchPage(page);
    const onPage = (page: Page): void => watchPage(page);
    context.on("page", onPage);
    this.disposers.push(() => context.off("page", onPage));
  }

  dispose(): void {
    for (const d of this.disposers) d();
  }

  describe(): string {
    return `declared 'expect tab ${this.pattern}' but no tab with a matching url opened`;
  }
}

export interface DownloadRecord {
  filename: string;
  savedAs: string | null;
}

/** Collects downloads across all pages; matchers gate settlement. */
export class DownloadWatcher {
  records: DownloadRecord[] = [];
  private listeners = new Set<(d: DownloadRecord, download: Download) => void>();
  /** persist hook (set by the runner when persisting) */
  onDownload: ((download: Download, filename: string) => Promise<string | null>) | null = null;

  attach(page: Page): void {
    page.on("download", (download) => void this.handle(download));
  }

  private async handle(download: Download): Promise<void> {
    const filename = download.suggestedFilename();
    const savedAs = this.onDownload ? await this.onDownload(download, filename) : null;
    const record: DownloadRecord = { filename, savedAs };
    this.records.push(record);
    for (const l of this.listeners) l(record, download);
  }

  arm(nameContains: string): SettleMatcher & { dispose(): void } {
    const result: { matched: DownloadRecord | null } = { matched: null };
    let resolveMatched!: () => void;
    const matchedAndFinished = new Promise<void>((resolve) => {
      resolveMatched = resolve;
    });
    const check = (d: DownloadRecord): void => {
      if (result.matched) return;
      if (!d.filename.includes(nameContains)) return;
      result.matched = d;
      resolveMatched();
    };
    for (const d of this.records) check(d);
    const listener = (d: DownloadRecord): void => check(d);
    this.listeners.add(listener);
    return {
      result,
      matchedAndFinished,
      describe: () => `declared 'expect download "${nameContains}"' but no matching download started`,
      dispose: () => this.listeners.delete(listener),
    };
  }

  stepBoundary(): void {
    this.records = [];
  }
}

/** Event-driven wait for an open tab matching the pattern (for `switch tab`). */
export async function waitForTab(context: BrowserContext, pattern: string, deadlineMs: number): Promise<Page> {
  const matches = (page: Page): boolean => {
    const url = page.url();
    if (!url || url === "about:blank") return false;
    const { path, query } = pathOf(url);
    return matchPath(pattern, path, query);
  };

  const deadline = Date.now() + deadlineMs;
  for (;;) {
    const found = context.pages().find(matches);
    if (found) return found;
    const remaining = deadline - Date.now();
    if (remaining <= 0) {
      const open = context.pages().map((p) => pathOf(p.url() || "about:blank").path).join(", ");
      throw new Error(
        `switch tab ${pattern}: no open tab matches (open tabs: ${open || "none"}). ` +
          `Tabs must be opened by an earlier action — assert it with 'expect tab ${pattern}'.`,
      );
    }
    // wake on any page/navigation event, or a short pulse (runner physics)
    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, Math.min(250, remaining));
      const onEvent = (): void => {
        clearTimeout(timer);
        cleanup();
        resolve();
      };
      const pages = context.pages();
      const cleanup = (): void => {
        context.off("page", onEvent);
        for (const p of pages) p.off("framenavigated", onEvent as never);
      };
      context.on("page", onEvent);
      for (const p of pages) p.on("framenavigated", onEvent as never);
    });
  }
}
