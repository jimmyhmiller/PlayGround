import type { FrameLocator, Locator, Page } from "playwright";
import type { Target } from "../dsl/ir.js";
import { formatTarget } from "../dsl/ir.js";

export class TargetError extends Error {
  constructor(message: string, public ariaSnapshot?: string, public target?: Target) {
    super(message);
    this.name = "TargetError";
  }
}

export type Captures = Map<string, string>;

export function interpolate(s: string, captures: Captures): string {
  return s.replace(/\$([A-Za-z_][\w-]*)/g, (whole, name: string) => {
    const v = captures.get(name);
    if (v === undefined) {
      // parser guarantees defined-before-use; reaching here is a runner bug
      throw new TargetError(`internal: $${name} has no captured value (parser should have rejected this)`);
    }
    return v;
  });
}

const ROLE_SET = new Set([
  "button", "link", "heading", "textbox", "checkbox", "radio", "combobox",
  "option", "row", "cell", "table", "list", "listitem", "region", "dialog",
  "alert", "status", "tab", "tabpanel", "menu", "menuitem", "img", "banner",
  "navigation", "main", "article", "form", "group",
]);

/** Roles that do NOT compute an accessible name from their contents (per ARIA).
 * For these, `listitem "Blue Widget"` means "the listitem containing that text"
 * (or one labelled with it) — so we union the aria-name match with a content filter. */
const NO_NAME_FROM_CONTENT = new Set([
  "list", "listitem", "table", "region", "dialog", "alert", "status", "menu",
  "tabpanel", "article", "form", "group", "main", "banner", "navigation",
]);

function buildScope(page: Page, within: Target | undefined, captures: Captures, exact: boolean): Page | Locator | FrameLocator {
  if (!within) return page;
  if (within.kind === "frame") {
    const outer = buildScope(page, within.within, captures, exact);
    const name = interpolate(within.name ?? "", captures);
    // match an iframe by name, title, or src substring
    const esc = name.replace(/"/g, '\\"');
    return outer.frameLocator(`iframe[name="${esc}"], iframe[title="${esc}"], iframe[src*="${esc}"]`);
  }
  return buildLocator(page, within, captures, exact);
}

export function buildLocator(page: Page, target: Target, captures: Captures, exact = false): Locator {
  if (target.kind === "frame") {
    throw new TargetError(`"frame" is a scope, not a target — write '<target> in frame "${target.name ?? ""}"'`, undefined, target);
  }
  const scope: Page | Locator | FrameLocator = buildScope(page, target.within, captures, exact);
  const name = target.name !== undefined ? interpolate(target.name, captures) : undefined;
  switch (target.kind) {
    case "text":
      return scope.getByText(name!, { exact });
    case "field":
      return scope.getByLabel(name!, { exact });
    case "placeholder":
      return scope.getByPlaceholder(name!, { exact });
    case "testid":
      return scope.getByTestId(name!);
    default: {
      if (!ROLE_SET.has(target.kind)) {
        throw new TargetError(`internal: unhandled target kind "${target.kind}"`);
      }
      const role = target.kind as Parameters<Page["getByRole"]>[0];
      if (name === undefined) return scope.getByRole(role);
      if (NO_NAME_FROM_CONTENT.has(target.kind)) {
        const content = exact
          ? scope.getByRole(role).filter({ hasText: new RegExp(`^\\s*${escapeRegex(name)}\\s*$`) })
          : scope.getByRole(role).filter({ hasText: name });
        return scope.getByRole(role, { name, exact }).or(content);
      }
      return scope.getByRole(role, { name, exact });
    }
  }
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Resolve a target to exactly one element, or fail with an actionable story.
 * - waits (event-driven, bounded by the step budget) for at least one match;
 * - two or more matches at act time is a HARD error listing every match —
 *   bat never picks "the first one".
 */
export async function resolveUnique(
  page: Page,
  target: Target,
  captures: Captures,
  budgetMs: number,
): Promise<Locator> {
  const locator = buildLocator(page, target, captures);
  try {
    await locator.first().waitFor({ state: "visible", timeout: budgetMs });
  } catch {
    const snapshot = await ariaSnapshotSafe(page);
    const frames = page.frames().length - 1;
    const scopesFrame = (function has(t: Target | undefined): boolean {
      return !!t && (t.kind === "frame" || has(t.within));
    })(target);
    throw new TargetError(
      `no visible match for ${formatTarget(target)} (page: ${page.url()})\n` +
        (frames > 0 && !scopesFrame
          ? `note: the page contains ${frames} iframe(s); content inside an iframe is only reachable via '<target> in frame "<name|title|src>"'.\n`
          : "") +
        `The page's semantic tree at failure:\n${indent(snapshot)}`,
      snapshot,
      target,
    );
  }
  const count = await locator.count();
  if (count > 1) {
    // Disambiguation rule: a UNIQUE exact-name match wins over substring
    // matches ("Search" beats "search-results"). Anything else is still a
    // hard error — bat never picks the first one.
    const exactLoc = buildLocator(page, target, captures, true);
    const exactCount = await exactLoc.count();
    if (exactCount === 1) return exactLoc;
    const listing = await describeMatches(locator, count);
    throw new TargetError(
      `${formatTarget(target)} is ambiguous: ${count} elements match — bat never picks the first one.\n` +
        `Matches:\n${listing}\n` +
        (exactCount > 1
          ? `(${exactCount} of them match the name exactly, so exact matching cannot disambiguate either.)\n`
          : "") +
        `Scope the target (e.g. 'in <container>') or use a testid.`,
      undefined,
      target,
    );
  }
  return locator;
}

async function describeMatches(locator: Locator, count: number): Promise<string> {
  const lines: string[] = [];
  for (let i = 0; i < Math.min(count, 10); i++) {
    const el = locator.nth(i);
    const [text, tid] = await Promise.all([
      el.innerText().catch(() => ""),
      el.getAttribute("data-testid").catch(() => null),
    ]);
    lines.push(`  ${i + 1}. "${(text || "").slice(0, 80).replace(/\n/g, " ")}"${tid ? ` (testid: ${tid})` : ""}`);
  }
  if (count > 10) lines.push(`  … and ${count - 10} more`);
  return lines.join("\n");
}

export async function ariaSnapshotSafe(page: Page): Promise<string> {
  try {
    return await page.locator("body").ariaSnapshot();
  } catch (e) {
    return `(aria snapshot unavailable: ${e instanceof Error ? e.message : String(e)})`;
  }
}

function indent(s: string): string {
  return s
    .split("\n")
    .map((l) => `    ${l}`)
    .join("\n");
}
