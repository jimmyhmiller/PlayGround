/**
 * The IR a flow file parses to. Steps are flat, serializable records —
 * this is what makes traces, checkpoints, and atomic replay possible.
 * There is deliberately no way to represent a duration anywhere in here.
 */

export const ROLE_KINDS = [
  "button", "link", "heading", "textbox", "checkbox", "radio", "combobox",
  "option", "row", "cell", "table", "list", "listitem", "region", "dialog",
  "alert", "status", "tab", "tabpanel", "menu", "menuitem", "img", "banner",
  "navigation", "main", "article", "form", "group",
] as const;

export const SPECIAL_KINDS = ["text", "field", "placeholder", "testid"] as const;

export type TargetKind = (typeof ROLE_KINDS)[number] | (typeof SPECIAL_KINDS)[number];

export const ALL_KINDS: readonly string[] = [...ROLE_KINDS, ...SPECIAL_KINDS];

export interface Target {
  kind: TargetKind;
  /** Accessible name / text / label / testid. May contain $var. Optional for role kinds. */
  name?: string;
  /** Scope: this target is resolved within `within`. */
  within?: Target;
}

export type Action =
  | { type: "go"; path: string }
  | { type: "click" | "dblclick" | "hover"; target: Target }
  | { type: "fill"; target: Target; value: string }
  | { type: "select"; target: Target; value: string }
  | { type: "check" | "uncheck"; target: Target }
  | { type: "press"; key: string; target?: Target }
  | { type: "upload"; target: Target; file: string };

export type Effect =
  /** NOTE: a `visible` effect whose target's HEAD kind is `text` is not
   * surface-representable: `expect text "x"` belongs to the `text` effect,
   * which has equivalent meaning. Construct that shape only via the IR. */
  | { type: "visible"; target: Target }
  | { type: "absent"; target: Target }
  /** armed BEFORE the action — catches transients (toasts) */
  | { type: "appear"; target: Target }
  | { type: "gone"; target: Target }
  | { type: "text"; target?: Target; value: string; exact: boolean }
  | { type: "value"; target: Target; value: string }
  | { type: "checked" | "unchecked"; target: Target }
  | { type: "enabled" | "disabled"; target: Target }
  | { type: "selected"; target: Target; value: string }
  | { type: "count"; kind: TargetKind; name?: string; n: number; within?: Target }
  | { type: "url"; path: string }
  /** bodyContains: substring of the request body — how you pin a GraphQL
   * operation (`containing "mutation CreateInvoice"`) or a payload field */
  | { type: "request"; method: string; pathPattern: string; status: "ok" | number; bodyContains?: string }
  /** websocket frame matcher, armed before the action */
  | { type: "ws"; dir: "sent" | "received"; text: string; pathPattern?: string }
  | { type: "let"; name: string; from: Target };

export type Given =
  | { type: "seed"; name: string }
  | { type: "patch"; entity: string; key: string; field: string; value: unknown }
  | { type: "user"; key: string }
  | { type: "clock"; iso: string }
  | { type: "stub"; method: string; pathPattern: string; status: number; body?: unknown }
  | { type: "allow"; what: "console-errors" | "dialogs" };

export interface Step {
  action: Action;
  effects: Effect[];
  /** 1-based line number of the action in the source file */
  line: number;
  /** verbatim source of the action line (for traces/reports) */
  source: string;
}

export interface Flow {
  name: string;
  file: string;
  givens: Given[];
  steps: Step[];
}

/** Actions whose effects block MUST be non-empty. In bat, that is all of them:
 * an unobserved action is exactly where races hide. */
export function requiresEffects(_action: Action): boolean {
  return true;
}

export function formatTarget(t: Target): string {
  const base = t.name !== undefined ? `${t.kind} "${t.name}"` : t.kind;
  return t.within ? `${base} in ${formatTarget(t.within)}` : base;
}

export function formatAction(a: Action): string {
  switch (a.type) {
    case "go": return `go ${a.path}`;
    case "click": case "dblclick": case "hover": return `${a.type} ${formatTarget(a.target)}`;
    case "fill": return `fill ${formatTarget(a.target)} "${a.value}"`;
    case "select": return `select ${formatTarget(a.target)} "${a.value}"`;
    case "check": case "uncheck": return `${a.type} ${formatTarget(a.target)}`;
    case "press": return a.target ? `press "${a.key}" in ${formatTarget(a.target)}` : `press "${a.key}"`;
    case "upload": return `upload ${formatTarget(a.target)} "${a.file}"`;
  }
}

export function formatEffect(e: Effect): string {
  switch (e.type) {
    case "visible": return `expect ${formatTarget(e.target)}`;
    case "absent": return `expect no ${formatTarget(e.target)}`;
    case "appear": return `expect appear ${formatTarget(e.target)}`;
    case "gone": return `expect gone ${formatTarget(e.target)}`;
    case "text": return `expect ${e.exact ? "exact " : ""}text "${e.value}"${e.target ? ` in ${formatTarget(e.target)}` : ""}`;
    case "value": return `expect value "${e.value}" in ${formatTarget(e.target)}`;
    case "checked": return `expect checked ${formatTarget(e.target)}`;
    case "unchecked": return `expect unchecked ${formatTarget(e.target)}`;
    case "enabled": return `expect enabled ${formatTarget(e.target)}`;
    case "disabled": return `expect disabled ${formatTarget(e.target)}`;
    case "selected": return `expect selected "${e.value}" in ${formatTarget(e.target)}`;
    case "count": return `expect count ${e.kind}${e.name ? ` "${e.name}"` : ""} ${e.n}${e.within ? ` in ${formatTarget(e.within)}` : ""}`;
    case "url": return `expect url ${e.path}`;
    case "request":
      return `expect request ${e.method} ${e.pathPattern} ${e.status}${e.bodyContains !== undefined ? ` containing "${e.bodyContains}"` : ""}`;
    case "ws":
      return `expect ws ${e.dir} "${e.text}"${e.pathPattern !== undefined ? ` on ${e.pathPattern}` : ""}`;
    case "let": return `let ${e.name} = text in ${formatTarget(e.from)}`;
  }
}
